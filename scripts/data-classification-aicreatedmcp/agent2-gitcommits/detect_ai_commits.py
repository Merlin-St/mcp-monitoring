#!/usr/bin/env python3
"""
Agent 2: Detect AI-created MCP servers by mining git commit messages,
PR metadata, contributor info, and repository config files.

Uses FULL commit pagination (up to 10,000 commits) and a BINARY
classification system (ai_authored = yes/no) based on four criteria:
1. Co-Authored-By lines referencing AI tools
2. AI configuration files present in the repo
3. Bot contributors (AI-specific, not dependabot/renovate/snyk)
4. >=1 AI tool handle mentions in commits/PRs

Usage:
    python detect_ai_commits.py                    # Process all 500 servers
    python detect_ai_commits.py --limit 50         # Process first 50
    python detect_ai_commits.py --resume            # Resume from checkpoint
    python detect_ai_commits.py --batch-size 25     # Smaller batches
"""

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiohttp

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DATA_INPUT = PROJECT_ROOT / "data" / "external-aicreatedmcp" / "data_unified_filtered_subset.json"
DATA_OUTPUT_DIR = PROJECT_ROOT / "data" / "external-aicreatedmcp" / "agent2-gitcommits"
LOG_DIR = PROJECT_ROOT / "logs"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "agent2_detect_ai_commits.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("agent2")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_COMMITS_PER_REPO = 10_000
COMMITS_PER_PAGE = 100
MAX_COMMIT_PAGES = MAX_COMMITS_PER_REPO // COMMITS_PER_PAGE  # 100 pages

# ---------------------------------------------------------------------------
# AI Tool Handle Patterns (for criterion 5: mentions in commit/PR text)
# These are @-handle style references plus tool names in context
# ---------------------------------------------------------------------------
AI_HANDLE_PATTERNS: dict[str, list[str]] = {
    "claude": [
        r"@claude\b",
        r"@anthropic\b",
        r"\bclaude[\s-]?code\b",
        r"\bclaude[\s-]?sonnet\b",
        r"\bclaude[\s-]?opus\b",
        r"\bclaude[\s-]?haiku\b",
    ],
    "copilot": [
        r"@copilot\b",
        r"@github[\s-]?copilot\b",
        r"\bgithub[\s-]?copilot\b",
    ],
    "chatgpt": [
        r"@chatgpt\b",
        r"@openai\b",
        r"\bchatgpt\b",
        r"\bgpt[\s-]?4o?\b",
    ],
    "cursor": [
        r"@cursor\b",
        r"\bcursor[\s-]?ai\b",
    ],
    "devin": [
        r"@devin\b",
        r"\bdevin[\s-]?ai\b",
    ],
    "codex": [
        r"@codex\b",
        r"\bopenai[\s-]?codex\b",
    ],
    "aider": [
        r"@aider\b",
        r"\baider\b",
    ],
    "cline": [
        r"@cline\b",
        r"\bcline\b",
    ],
    "windsurf": [
        r"@windsurf\b",
        r"\bwindsurf\b",
    ],
    "gemini": [
        r"@gemini\b",
        r"\bgemini[\s-]?code\b",
    ],
    "roo": [
        r"@roo\b",
        r"\broo[\s-]?code\b",
    ],
    "augment": [
        r"@augment\b",
        r"\baugment[\s-]?code\b",
    ],
    "sourcegraph_cody": [
        r"@cody\b",
        r"\bsourcegraph[\s-]?cody\b",
    ],
    "replit": [
        r"@replit\b",
        r"\breplit[\s-]?agent\b",
    ],
    "continue_dev": [
        r"\bcontinue\.dev\b",
    ],
    "v0": [
        r"\bv0\.dev\b",
    ],
    "bolt": [
        r"\bbolt\.new\b",
        r"\bbolt\.diy\b",
    ],
    "lovable": [
        r"\blovable\.dev\b",
    ],
}

# Co-Authored-By patterns (case-insensitive) -- criterion 1
CO_AUTHOR_AI_PATTERNS: dict[str, list[str]] = {
    "claude": [
        r"co-authored-by:.*claude",
        r"co-authored-by:.*anthropic",
        r"co-authored-by:.*noreply@anthropic\.com",
    ],
    "copilot": [
        r"co-authored-by:.*copilot",
        r"co-authored-by:.*github-copilot",
        r"co-authored-by:.*noreply@github\.com.*copilot",
    ],
    "chatgpt": [
        r"co-authored-by:.*chatgpt",
        r"co-authored-by:.*openai",
    ],
    "devin": [
        r"co-authored-by:.*devin",
    ],
    "codex": [
        r"co-authored-by:.*codex",
    ],
    "aider": [
        r"co-authored-by:.*aider",
    ],
    "cline": [
        r"co-authored-by:.*cline",
        r"co-authored-by:.*claude[\s-]?dev",
    ],
    "roo": [
        r"co-authored-by:.*roo[\s-]?code",
    ],
    "augment": [
        r"co-authored-by:.*augment",
    ],
    "continue_dev": [
        r"co-authored-by:.*continue",
    ],
    "gemini": [
        r"co-authored-by:.*gemini",
        r"co-authored-by:.*google[\s-]?ai",
    ],
    "windsurf": [
        r"co-authored-by:.*windsurf",
    ],
}

# Config files that indicate AI tool usage -- criterion 2
AI_CONFIG_FILES: dict[str, list[str]] = {
    "claude": [
        "CLAUDE.md",
        ".claude",
        ".claude/settings.json",
        ".claude/settings.local.json",
    ],
    "cursor": [
        ".cursor",
        ".cursorrules",
        ".cursor/rules",
        ".cursorignore",
    ],
    "copilot": [
        ".github/copilot-instructions.md",
    ],
    "aider": [
        ".aider.conf.yml",
        ".aider",
        ".aiderignore",
    ],
    "codeium": [
        ".codeium",
    ],
    "windsurf": [
        ".windsurfrules",
    ],
    "cline": [
        ".clinerules",
        ".cline",
    ],
    "roo": [
        ".roo",
        ".roorules",
        ".roomodes",
    ],
    "codex": [
        "AGENTS.md",
        "codex.md",
    ],
    "augment": [
        ".augment",
        ".augment-guidelines",
    ],
    "continue_dev": [
        ".continue",
        ".continuerules",
    ],
}

# Bot accounts -- criterion 3 (AI-specific only; dependabot/renovate/snyk excluded)
AI_BOT_ACCOUNTS: dict[str, list[str]] = {
    "devin": ["devin-ai-integration", "devin-ai-integration[bot]"],
    "copilot": ["copilot", "copilot[bot]", "github-copilot[bot]"],
    "claude": ["claude[bot]", "anthropic-ai[bot]"],
}

# Non-AI bots to exclude from ai_authored classification
NON_AI_BOTS: set[str] = {
    "dependabot", "dependabot[bot]",
    "renovate", "renovate[bot]",
    "snyk-bot", "snyk[bot]",
    "greenkeeper[bot]",
    "allcontributors[bot]",
}

# Compile regex patterns
COMPILED_COAUTHOR_PATTERNS: dict[str, list[re.Pattern]] = {
    tool: [re.compile(p, re.IGNORECASE) for p in patterns]
    for tool, patterns in CO_AUTHOR_AI_PATTERNS.items()
}

COMPILED_HANDLE_PATTERNS: dict[str, list[re.Pattern]] = {
    tool: [re.compile(p, re.IGNORECASE) for p in patterns]
    for tool, patterns in AI_HANDLE_PATTERNS.items()
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class EvidenceItem:
    sha: str
    snippet: str
    evidence_type: str  # "co_author", "ai_handle_mention", "bot_contributor"
    tool: str = ""
    date: str = ""


@dataclass
class ServerResult:
    id: str
    name: str
    github_url: str
    # Binary classification
    ai_authored: str = "no"  # "yes" or "no"
    ai_authored_reasons: list = field(default_factory=list)
    likely_ai_agent: str = "none"
    # Evidence details
    total_commits_scanned: int = 0
    co_author_count: int = 0
    ai_config_files_found: list = field(default_factory=list)
    bot_contributors: list = field(default_factory=list)
    multiline_commit_ratio: float = 0.0
    ai_mention_count: int = 0
    ai_mention_details: dict = field(default_factory=dict)
    commit_evidence: list = field(default_factory=list)
    # Internal tracking
    tool_scores: dict = field(default_factory=dict)
    error: str = ""
    processed_at: str = ""


# ---------------------------------------------------------------------------
# GitHub API client with rate limiting
# ---------------------------------------------------------------------------
class GitHubAPIClient:
    """Async GitHub API client with rate-limit handling."""

    BASE_URL = "https://api.github.com"

    def __init__(self, token: str, max_concurrent: int = 10):
        self.token = token
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.rate_remaining = 5000
        self.rate_reset_time = 0
        self.request_count = 0
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"token {self.token}",
                    "Accept": "application/vnd.github.v3+json",
                    "User-Agent": "mcp-monitoring-agent2",
                },
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def check_rate_limit(self):
        """Check current rate limit and wait if needed."""
        session = await self._get_session()
        try:
            async with session.get(f"{self.BASE_URL}/rate_limit") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    core = data.get("resources", {}).get("core", {})
                    self.rate_remaining = core.get("remaining", 5000)
                    self.rate_reset_time = core.get("reset", 0)
                    logger.info(
                        "Rate limit: %d/%d remaining, resets in %ds",
                        self.rate_remaining,
                        core.get("limit", 5000),
                        max(self.rate_reset_time - int(time.time()), 0),
                    )
                    if self.rate_remaining < 100:
                        wait_sec = max(self.rate_reset_time - time.time(), 0) + 2
                        logger.warning(
                            "Rate limit low (%d remaining). Waiting %.0fs...",
                            self.rate_remaining,
                            wait_sec,
                        )
                        await asyncio.sleep(wait_sec)
                        async with session.get(f"{self.BASE_URL}/rate_limit") as resp2:
                            if resp2.status == 200:
                                data2 = await resp2.json()
                                core2 = data2.get("resources", {}).get("core", {})
                                self.rate_remaining = core2.get("remaining", 5000)
                                self.rate_reset_time = core2.get("reset", 0)
                                logger.info("After wait: %d remaining", self.rate_remaining)
        except Exception as exc:
            logger.warning("Could not check rate limit: %s", exc)

    async def _handle_rate_limit(self, response: aiohttp.ClientResponse):
        """Update rate-limit tracking from response headers."""
        remaining = response.headers.get("X-RateLimit-Remaining")
        reset_ts = response.headers.get("X-RateLimit-Reset")
        if remaining is not None:
            self.rate_remaining = int(remaining)
        if reset_ts is not None:
            self.rate_reset_time = int(reset_ts)

        if self.rate_remaining < 50:
            wait_sec = max(self.rate_reset_time - time.time(), 0) + 2
            logger.warning(
                "Rate limit low (%d remaining). Sleeping %.0fs.",
                self.rate_remaining,
                wait_sec,
            )
            await asyncio.sleep(wait_sec)

    async def get(self, path: str, params: dict | None = None) -> tuple[int, dict | list | None]:
        """Make a GET request. Returns (status_code, json_body)."""
        async with self.semaphore:
            session = await self._get_session()
            url = f"{self.BASE_URL}{path}" if path.startswith("/") else path
            try:
                async with session.get(url, params=params) as resp:
                    self.request_count += 1
                    await self._handle_rate_limit(resp)
                    if resp.status == 200:
                        body = await resp.json()
                        return resp.status, body
                    elif resp.status == 403:
                        body_text = await resp.text()
                        if "rate limit" in body_text.lower():
                            wait_sec = max(self.rate_reset_time - time.time(), 0) + 5
                            logger.warning("Rate limited. Sleeping %.0fs.", wait_sec)
                            await asyncio.sleep(wait_sec)
                            async with session.get(url, params=params) as retry_resp:
                                self.request_count += 1
                                await self._handle_rate_limit(retry_resp)
                                if retry_resp.status == 200:
                                    return retry_resp.status, await retry_resp.json()
                                return retry_resp.status, None
                        return resp.status, None
                    elif resp.status in (404, 409, 422):
                        return resp.status, None
                    else:
                        logger.debug("Unexpected status %d for %s", resp.status, path)
                        return resp.status, None
            except asyncio.TimeoutError:
                logger.debug("Timeout for %s", path)
                return 0, None
            except Exception as e:
                logger.debug("Request error for %s: %s", path, str(e))
                return 0, None


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def extract_owner_repo(github_url: str) -> tuple[str, str] | None:
    """Extract (owner, repo) from a GitHub URL."""
    if not github_url:
        return None
    url = github_url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    patterns = [
        r"github\.com/([^/]+)/([^/]+?)(?:\.git)?$",
        r"github\.com/([^/]+)/([^/]+)$",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1), m.group(2)
    return None


def detect_coauthor_ai(text: str) -> dict[str, int]:
    """Search for AI-related Co-Authored-By lines. Returns {tool: count}."""
    if not text:
        return {}
    results: dict[str, int] = {}
    for tool, patterns in COMPILED_COAUTHOR_PATTERNS.items():
        for pattern in patterns:
            matches = pattern.findall(text)
            if matches:
                results[tool] = results.get(tool, 0) + len(matches)
    return results


def count_ai_handle_mentions(text: str) -> dict[str, int]:
    """Count AI tool handle/name mentions in text. Returns {tool: count}."""
    if not text:
        return {}
    results: dict[str, int] = {}
    for tool, patterns in COMPILED_HANDLE_PATTERNS.items():
        for pattern in patterns:
            matches = pattern.findall(text)
            if matches:
                results[tool] = results.get(tool, 0) + len(matches)
    return results


def is_multiline_commit(message: str) -> bool:
    """Check if a commit message is multiline (>1 non-empty line)."""
    lines = [ln.strip() for ln in message.strip().split("\n") if ln.strip()]
    return len(lines) > 1


# ---------------------------------------------------------------------------
# Core analysis per repository
# ---------------------------------------------------------------------------

async def fetch_all_commits(
    client: GitHubAPIClient,
    owner: str,
    repo: str,
) -> list[dict]:
    """Fetch ALL commits with full pagination (up to 10,000)."""
    all_commits = []
    for page in range(1, MAX_COMMIT_PAGES + 1):
        status, commits_data = await client.get(
            f"/repos/{owner}/{repo}/commits",
            params={"per_page": COMMITS_PER_PAGE, "page": page},
        )
        if status != 200 or not isinstance(commits_data, list) or len(commits_data) == 0:
            break
        all_commits.extend(commits_data)
        if len(commits_data) < COMMITS_PER_PAGE:
            # Last page -- no more commits
            break
    return all_commits


async def analyze_repo(
    client: GitHubAPIClient,
    server: dict,
    max_prs: int = 30,
) -> ServerResult:
    """Analyze a single repository for AI tool evidence using binary classification."""

    github_url = server.get("github_url", "")
    result = ServerResult(
        id=server.get("id", ""),
        name=server.get("name", ""),
        github_url=github_url,
        processed_at=datetime.now(timezone.utc).isoformat(),
    )

    parsed = extract_owner_repo(github_url)
    if not parsed:
        result.error = "Could not parse GitHub URL"
        return result

    owner, repo = parsed
    tool_scores: dict[str, int] = {}  # track which tool has most evidence

    # -------------------------------------------------------------------
    # 1. Fetch ALL commits (paginated, up to 10,000)
    # -------------------------------------------------------------------
    commits = await fetch_all_commits(client, owner, repo)
    result.total_commits_scanned = len(commits)

    multiline_count = 0
    co_author_count = 0
    ai_mention_details: dict[str, int] = {}
    evidence_items: list[dict] = []

    for commit_obj in commits:
        commit_info = commit_obj.get("commit", {})
        message = commit_info.get("message", "")
        author_login = (commit_obj.get("author") or {}).get("login", "")
        committer_login = (commit_obj.get("committer") or {}).get("login", "")
        sha = commit_obj.get("sha", "")[:8]
        date = commit_info.get("author", {}).get("date", "")

        # Track multiline commits
        if is_multiline_commit(message):
            multiline_count += 1

        # Criterion 1: Co-Authored-By lines
        coauthor_hits = detect_coauthor_ai(message)
        for tool, count in coauthor_hits.items():
            co_author_count += count
            tool_scores[tool] = tool_scores.get(tool, 0) + count * 3
            evidence_items.append(asdict(EvidenceItem(
                sha=sha,
                snippet=message[:200],
                evidence_type="co_author",
                tool=tool,
                date=date,
            )))

        # Criterion 5: AI handle mentions in commits
        handle_hits = count_ai_handle_mentions(message)
        for tool, count in handle_hits.items():
            ai_mention_details[tool] = ai_mention_details.get(tool, 0) + count
            tool_scores[tool] = tool_scores.get(tool, 0) + count
            # Only add evidence for first few to keep output concise
            if len(evidence_items) < 200:
                evidence_items.append(asdict(EvidenceItem(
                    sha=sha,
                    snippet=message[:150],
                    evidence_type="ai_handle_mention",
                    tool=tool,
                    date=date,
                )))

        # Criterion 3: Bot contributors from commits
        for bot_tool, bot_names in AI_BOT_ACCOUNTS.items():
            bot_names_lower = [b.lower() for b in bot_names]
            if author_login.lower() in bot_names_lower or committer_login.lower() in bot_names_lower:
                bot_name = author_login if author_login.lower() in bot_names_lower else committer_login
                if bot_name not in result.bot_contributors:
                    result.bot_contributors.append(bot_name)
                    tool_scores[bot_tool] = tool_scores.get(bot_tool, 0) + 5

    # Compute multiline ratio
    if result.total_commits_scanned > 0:
        result.multiline_commit_ratio = round(multiline_count / result.total_commits_scanned, 4)

    result.co_author_count = co_author_count

    # -------------------------------------------------------------------
    # 2. Fetch recent PRs (for AI handle mentions and bot PR authors)
    # -------------------------------------------------------------------
    status, prs_data = await client.get(
        f"/repos/{owner}/{repo}/pulls",
        params={"state": "all", "per_page": min(max_prs, 100), "sort": "updated"},
    )
    prs = prs_data if status == 200 and isinstance(prs_data, list) else []

    for pr in prs:
        pr_title = pr.get("title", "")
        pr_body = pr.get("body", "") or ""
        pr_author = (pr.get("user") or {}).get("login", "")
        pr_text = f"{pr_title} {pr_body}"

        # Criterion 1: Co-Authored-By in PR body
        coauthor_hits = detect_coauthor_ai(pr_body)
        for tool, count in coauthor_hits.items():
            co_author_count += count
            result.co_author_count += count
            tool_scores[tool] = tool_scores.get(tool, 0) + count * 3

        # Criterion 5: AI handle mentions in PRs
        handle_hits = count_ai_handle_mentions(pr_text)
        for tool, count in handle_hits.items():
            ai_mention_details[tool] = ai_mention_details.get(tool, 0) + count
            tool_scores[tool] = tool_scores.get(tool, 0) + count

        # Criterion 3: Bot PR authors
        for bot_tool, bot_names in AI_BOT_ACCOUNTS.items():
            if pr_author.lower() in [b.lower() for b in bot_names]:
                if pr_author not in result.bot_contributors:
                    result.bot_contributors.append(pr_author)
                    tool_scores[bot_tool] = tool_scores.get(bot_tool, 0) + 5

    result.ai_mention_details = ai_mention_details
    result.ai_mention_count = sum(ai_mention_details.values())

    # -------------------------------------------------------------------
    # 3. Check for AI config files in the repo (criterion 2)
    # -------------------------------------------------------------------
    status, tree_data = await client.get(
        f"/repos/{owner}/{repo}/git/trees/HEAD",
        params={"recursive": "1"},
    )
    tree_items = []
    if status == 200 and isinstance(tree_data, dict):
        tree_items = tree_data.get("tree", [])

    # Build set of all file paths
    repo_paths = set()
    for item in tree_items:
        path = item.get("path", "")
        repo_paths.add(path)
        parts = path.split("/")
        for i in range(1, len(parts)):
            repo_paths.add("/".join(parts[:i]))

    for tool, config_paths in AI_CONFIG_FILES.items():
        for config_path in config_paths:
            if config_path in repo_paths:
                result.ai_config_files_found.append(config_path)
                tool_scores[tool] = tool_scores.get(tool, 0) + 10

    # -------------------------------------------------------------------
    # 4. Apply binary classification criteria
    # -------------------------------------------------------------------
    reasons = []

    # Criterion 1: >=1 Co-Authored-By line referencing an AI tool
    if result.co_author_count >= 1:
        reasons.append("co_authored_by")

    # Criterion 2: AI configuration files present
    if len(result.ai_config_files_found) > 0:
        reasons.append("config_files")

    # Criterion 3: Bot contributors (AI-specific only)
    if len(result.bot_contributors) > 0:
        reasons.append("bot_contributors")

    # Criterion 4 (DROPPED): multiline commits — tracked but not used for classification
    # Multiline ratio is still recorded in result.multiline_commit_ratio for reference

    # Criterion 4: >=1 AI tool handle mention total
    if result.ai_mention_count >= 1:
        reasons.append("ai_handle_mentions")

    result.ai_authored = "yes" if len(reasons) > 0 else "no"
    result.ai_authored_reasons = reasons

    # Determine likely AI agent from tool scores
    if tool_scores:
        result.likely_ai_agent = max(tool_scores, key=tool_scores.get)
    else:
        result.likely_ai_agent = "none"

    result.tool_scores = tool_scores

    # Trim evidence to keep output manageable
    result.commit_evidence = evidence_items[:100]

    return result


# ---------------------------------------------------------------------------
# Batch processing with checkpointing
# ---------------------------------------------------------------------------

async def process_batch(
    client: GitHubAPIClient,
    servers: list[dict],
    batch_idx: int,
    max_prs: int,
) -> list[ServerResult]:
    """Process a batch of servers concurrently."""
    tasks = [
        analyze_repo(client, server, max_prs=max_prs)
        for server in servers
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    processed = []
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            logger.error(
                "Error processing %s: %s",
                servers[i].get("id", "unknown"),
                str(res),
            )
            processed.append(ServerResult(
                id=servers[i].get("id", ""),
                name=servers[i].get("name", ""),
                github_url=servers[i].get("github_url", ""),
                error=str(res),
                processed_at=datetime.now(timezone.utc).isoformat(),
            ))
        else:
            processed.append(res)
    return processed


def save_checkpoint(results: list[dict], checkpoint_path: Path):
    """Save intermediate results."""
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Checkpoint saved: %d results -> %s", len(results), checkpoint_path)


def load_checkpoint(checkpoint_path: Path) -> list[dict]:
    """Load previously saved results."""
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def generate_summary(all_results: list[dict]) -> dict:
    """Generate aggregate statistics for binary classification."""
    total = len(all_results)
    if total == 0:
        return {"total_servers": 0}

    errors = sum(1 for r in all_results if r.get("error"))
    successfully_processed = total - errors

    # Binary classification counts
    ai_yes = sum(1 for r in all_results if r.get("ai_authored") == "yes" and not r.get("error"))
    ai_no = sum(1 for r in all_results if r.get("ai_authored") == "no" and not r.get("error"))

    # Reason breakdown
    reason_counts: dict[str, int] = {}
    for r in all_results:
        for reason in r.get("ai_authored_reasons", []):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    # Tool distribution (among ai_authored=yes)
    tool_counts: dict[str, int] = {}
    for r in all_results:
        if r.get("ai_authored") == "yes":
            agent = r.get("likely_ai_agent", "none")
            if agent and agent != "none":
                tool_counts[agent] = tool_counts.get(agent, 0) + 1

    # Config file prevalence
    config_counts: dict[str, int] = {}
    for r in all_results:
        for cfg in r.get("ai_config_files_found", []):
            config_counts[cfg] = config_counts.get(cfg, 0) + 1

    # Commit stats
    total_commits = sum(r.get("total_commits_scanned", 0) for r in all_results if not r.get("error"))
    repos_over_100 = sum(
        1 for r in all_results
        if not r.get("error") and r.get("total_commits_scanned", 0) > 100
    )

    # Co-author stats
    with_coauthor = sum(1 for r in all_results if r.get("co_author_count", 0) > 0)

    # Multiline ratio stats
    multiline_ratios = [
        r["multiline_commit_ratio"] for r in all_results
        if not r.get("error") and r.get("total_commits_scanned", 0) > 0
    ]
    avg_multiline = sum(multiline_ratios) / len(multiline_ratios) if multiline_ratios else 0

    # AI mention stats
    with_mentions = sum(1 for r in all_results if r.get("ai_mention_count", 0) >= 1)
    total_mentions = sum(r.get("ai_mention_count", 0) for r in all_results)

    return {
        "total_servers": total,
        "successfully_processed": successfully_processed,
        "errors": errors,
        "ai_authored_yes": ai_yes,
        "ai_authored_yes_pct": round(ai_yes / successfully_processed * 100, 1) if successfully_processed else 0,
        "ai_authored_no": ai_no,
        "ai_authored_no_pct": round(ai_no / successfully_processed * 100, 1) if successfully_processed else 0,
        "reason_breakdown": dict(sorted(reason_counts.items(), key=lambda x: -x[1])),
        "likely_ai_tool_distribution": dict(sorted(tool_counts.items(), key=lambda x: -x[1])),
        "ai_config_file_prevalence": dict(sorted(config_counts.items(), key=lambda x: -x[1])),
        "commit_statistics": {
            "total_commits_scanned": total_commits,
            "repos_with_over_100_commits": repos_over_100,
            "average_multiline_commit_ratio": round(avg_multiline, 4),
        },
        "co_author_statistics": {
            "repos_with_ai_coauthors": with_coauthor,
        },
        "ai_mention_statistics": {
            "repos_with_gte1_mentions": with_mentions,
            "total_mentions_across_all_repos": total_mentions,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_github_token() -> str:
    """Get GitHub token from env or gh CLI."""
    token = os.environ.get("GH_TOKEN", "")
    if token:
        return token
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        return token
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return ""


async def main():
    parser = argparse.ArgumentParser(
        description="Detect AI-created MCP servers via git history mining (binary classification)"
    )
    parser.add_argument("--limit", type=int, default=0, help="Max servers to process (0=all)")
    parser.add_argument("--batch-size", type=int, default=10, help="Servers per concurrent batch")
    parser.add_argument("--max-prs", type=int, default=30, help="Max PRs to fetch per repo")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent API requests")
    args = parser.parse_args()

    # Get token
    token = get_github_token()
    if not token:
        logger.error("No GitHub token found. Set GH_TOKEN or configure gh CLI.")
        sys.exit(1)
    logger.info("GitHub token found (length=%d)", len(token))

    # Load input data
    logger.info("Loading input data from %s", DATA_INPUT)
    with open(DATA_INPUT, "r", encoding="utf-8") as f:
        servers = json.load(f)
    logger.info("Loaded %d servers", len(servers))

    # Filter to servers with github_url
    servers = [s for s in servers if s.get("github_url")]
    logger.info("Servers with github_url: %d", len(servers))

    # Apply limit
    if args.limit > 0:
        servers = servers[:args.limit]
        logger.info("Limited to %d servers", len(servers))

    # Check for checkpoint
    checkpoint_path = DATA_OUTPUT_DIR / "checkpoint_results.json"
    all_results: list[dict] = []
    processed_ids: set[str] = set()

    if args.resume and checkpoint_path.exists():
        all_results = load_checkpoint(checkpoint_path)
        processed_ids = {r["id"] for r in all_results}
        logger.info("Resumed from checkpoint: %d already processed", len(all_results))
        servers = [s for s in servers if s.get("id") not in processed_ids]
        logger.info("Remaining to process: %d", len(servers))

    # Process in batches
    client = GitHubAPIClient(token, max_concurrent=args.concurrency)

    # Check rate limit before starting
    await client.check_rate_limit()

    total_batches = (len(servers) + args.batch_size - 1) // args.batch_size
    start_time = time.time()

    try:
        for batch_idx in range(total_batches):
            batch_start = batch_idx * args.batch_size
            batch_end = min(batch_start + args.batch_size, len(servers))
            batch = servers[batch_start:batch_end]

            logger.info(
                "Processing batch %d/%d (servers %d-%d, API requests: %d, rate remaining: %d)",
                batch_idx + 1,
                total_batches,
                batch_start + 1,
                batch_end,
                client.request_count,
                client.rate_remaining,
            )

            batch_results = await process_batch(
                client, batch, batch_idx,
                max_prs=args.max_prs,
            )

            for r in batch_results:
                all_results.append(asdict(r) if isinstance(r, ServerResult) else r)

            # Checkpoint every 5 batches
            if (batch_idx + 1) % 5 == 0:
                save_checkpoint(all_results, checkpoint_path)

            # Brief pause between batches
            if batch_idx < total_batches - 1:
                await asyncio.sleep(1.0)

            # Re-check rate limit every 10 batches
            if (batch_idx + 1) % 10 == 0:
                await client.check_rate_limit()

    except KeyboardInterrupt:
        logger.warning("Interrupted. Saving checkpoint...")
        save_checkpoint(all_results, checkpoint_path)
    finally:
        await client.close()

    elapsed = time.time() - start_time
    logger.info(
        "Processing complete: %d servers in %.1fs (%d API requests)",
        len(all_results),
        elapsed,
        client.request_count,
    )

    # Save final results
    output_path = DATA_OUTPUT_DIR / "ai_commit_evidence_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Results saved to %s", output_path)

    # Generate and save summary
    summary = generate_summary(all_results)
    summary_path = DATA_OUTPUT_DIR / "ai_commit_evidence_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s", summary_path)

    # Print summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("Total servers processed: %d", summary["total_servers"])
    logger.info("Errors: %d", summary["errors"])
    logger.info(
        "AI authored (yes): %d (%.1f%%)",
        summary["ai_authored_yes"],
        summary["ai_authored_yes_pct"],
    )
    logger.info(
        "AI authored (no): %d (%.1f%%)",
        summary["ai_authored_no"],
        summary["ai_authored_no_pct"],
    )
    logger.info("Reason breakdown: %s", json.dumps(summary["reason_breakdown"], indent=2))
    logger.info("Tool distribution: %s", json.dumps(summary["likely_ai_tool_distribution"], indent=2))
    logger.info("Config file prevalence: %s", json.dumps(summary["ai_config_file_prevalence"], indent=2))
    logger.info(
        "Commits scanned: %d total, %d repos with >100 commits",
        summary["commit_statistics"]["total_commits_scanned"],
        summary["commit_statistics"]["repos_with_over_100_commits"],
    )

    # Clean up checkpoint on successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Checkpoint removed (run completed successfully)")


if __name__ == "__main__":
    asyncio.run(main())

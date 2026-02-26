#!/usr/bin/env python3
"""
Detect AI-created MCP servers by mining git commit messages,
PR metadata, contributor info, and repository config files.

Uses FULL commit pagination (up to 10,000 commits) and a BINARY
classification system (ai_authored = yes/no) based on four criteria:
1. Co-Authored-By lines referencing AI tools
2. AI configuration files present in the repo
3. Bot contributors (AI-specific, not dependabot/renovate/snyk)
4. >=1 AI tool handle mentions in commits/PRs

Reads from data/initial/data_unified_filtered.json and outputs to
data/internal-cl/aicreated_results.json and aicreated_summary.json.

Usage:
    python detect_ai_created.py                              # Process all servers
    python detect_ai_created.py --limit 50                   # Process first 50
    python detect_ai_created.py --resume                     # Resume from checkpoint
    python detect_ai_created.py --batch-size 25              # Smaller batches
    python detect_ai_created.py --created-after 2025-10-01   # Only recent servers
    python detect_ai_created.py --append-to data/internal-cl/aicreated_results.json
    python detect_ai_created.py --backfill-dates             # Backfill missing dates
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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import aiohttp

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_INPUT = PROJECT_ROOT / "data" / "initial" / "data_unified_filtered.json"
DATA_OUTPUT_DIR = PROJECT_ROOT / "data" / "internal-cl"
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
        logging.FileHandler(LOG_DIR / "detect_ai_created.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("detect_ai_created")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_COMMITS_PER_REPO = 10_000
COMMITS_PER_PAGE = 100
MAX_COMMIT_PAGES = MAX_COMMITS_PER_REPO // COMMITS_PER_PAGE  # 100 pages

# ---------------------------------------------------------------------------
# AI Tool Handle Patterns (for criterion 4: mentions in commit/PR text)
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
    # Percentage breakdown of likely creators (sums to 100)
    likely_creators_details: dict = field(default_factory=dict)
    # Internal tracking
    tool_scores: dict = field(default_factory=dict)
    # First-month analysis (within 30 days of created_at)
    ai_authored_first_month: str = ""  # "yes", "no", or "" (no created_at)
    ai_authored_first_month_reasons: list = field(default_factory=list)
    first_month_co_author_count: int = 0
    first_month_bot_contributors: list = field(default_factory=list)
    first_month_ai_mention_count: int = 0
    first_month_ai_mention_details: dict = field(default_factory=dict)
    first_month_ai_config_files_found: list = field(default_factory=list)
    first_month_tool_scores: dict = field(default_factory=dict)
    first_month_likely_ai_agent: str = "none"
    first_month_commits_scanned: int = 0
    date_first_ai_evidence: str = ""  # ISO datetime of earliest commit_evidence entry
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
                    "User-Agent": "mcp-monitoring-ai-detection",
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

    async def get_oldest_commit_date_for_path(
        self, owner: str, repo: str, file_path: str
    ) -> str | None:
        """Get the date of the oldest commit that introduced/modified a file.

        Uses the commits API with ``path`` filter, paginating to find the
        earliest commit that touches ``file_path``.  Returns an ISO-8601
        date string or None.
        """
        oldest_date: str | None = None
        page = 1
        while True:
            status, data = await self.get(
                f"/repos/{owner}/{repo}/commits",
                params={"path": file_path, "per_page": "100", "page": str(page)},
            )
            if status != 200 or not isinstance(data, list) or not data:
                break
            # Commits come newest-first; the last item on the last page is oldest
            last_commit = data[-1]
            oldest_date = (
                last_commit.get("commit", {}).get("author", {}).get("date", "")
            )
            if len(data) < 100:
                break  # reached last page
            page += 1
            if page > 20:  # safety cap
                break
        return oldest_date if oldest_date else None


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


def parse_datetime_safe(date_str: str) -> datetime | None:
    """Parse an ISO 8601 date string into a timezone-aware datetime, or None."""
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


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
            break
    return all_commits


async def analyze_repo(
    client: GitHubAPIClient,
    server: dict,
    max_prs: int = 30,
) -> ServerResult:
    """Analyze a single repository for AI tool evidence using binary classification.

    Computes both full-history and first-month (within 30 days of created_at)
    AI detection metrics in a single pass over the commit/PR data.
    """

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
    tool_scores: dict[str, int] = {}

    # First-month cutoff
    created_dt = parse_datetime_safe(server.get("created_at", ""))
    first_month_cutoff = created_dt + timedelta(days=30) if created_dt else None

    # First-month accumulators
    fm_tool_scores: dict[str, int] = {}
    fm_co_author_count = 0
    fm_ai_mention_details: dict[str, int] = {}
    fm_bot_contributors: list[str] = []
    fm_commits_scanned = 0
    # Track newest first-month commit SHA for tree fetch
    fm_newest_commit_sha: str | None = None

    # -------------------------------------------------------------------
    # 1. Fetch ALL commits (paginated, up to 10,000)
    # -------------------------------------------------------------------
    commits = await fetch_all_commits(client, owner, repo)
    result.total_commits_scanned = len(commits)

    co_author_count = 0
    ai_mention_details: dict[str, int] = {}
    evidence_items: list[dict] = []

    for commit_obj in commits:
        commit_info = commit_obj.get("commit", {})
        message = commit_info.get("message", "")
        author_login = (commit_obj.get("author") or {}).get("login", "")
        committer_login = (commit_obj.get("committer") or {}).get("login", "")
        sha = commit_obj.get("sha", "")[:8]
        full_sha = commit_obj.get("sha", "")
        date = commit_info.get("author", {}).get("date", "")

        # Determine if commit is within first month
        in_first_month = False
        if first_month_cutoff and date:
            commit_dt = parse_datetime_safe(date)
            if commit_dt and commit_dt <= first_month_cutoff:
                in_first_month = True
                fm_commits_scanned += 1
                # Track newest first-month commit (API returns newest first)
                if fm_newest_commit_sha is None:
                    fm_newest_commit_sha = full_sha

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
            if in_first_month:
                fm_co_author_count += count
                fm_tool_scores[tool] = fm_tool_scores.get(tool, 0) + count * 3

        # Criterion 4: AI handle mentions in commits
        handle_hits = count_ai_handle_mentions(message)
        for tool, count in handle_hits.items():
            ai_mention_details[tool] = ai_mention_details.get(tool, 0) + count
            tool_scores[tool] = tool_scores.get(tool, 0) + count
            if len(evidence_items) < 200:
                evidence_items.append(asdict(EvidenceItem(
                    sha=sha,
                    snippet=message[:150],
                    evidence_type="ai_handle_mention",
                    tool=tool,
                    date=date,
                )))
            if in_first_month:
                fm_ai_mention_details[tool] = fm_ai_mention_details.get(tool, 0) + count
                fm_tool_scores[tool] = fm_tool_scores.get(tool, 0) + count

        # Criterion 3: Bot contributors from commits
        for bot_tool, bot_names in AI_BOT_ACCOUNTS.items():
            bot_names_lower = [b.lower() for b in bot_names]
            if author_login.lower() in bot_names_lower or committer_login.lower() in bot_names_lower:
                bot_name = author_login if author_login.lower() in bot_names_lower else committer_login
                if bot_name not in result.bot_contributors:
                    result.bot_contributors.append(bot_name)
                    tool_scores[bot_tool] = tool_scores.get(bot_tool, 0) + 5
                    if len(evidence_items) < 200:
                        evidence_items.append(asdict(EvidenceItem(
                            sha=sha,
                            snippet=f"bot commit by {bot_name}",
                            evidence_type="bot_contributor",
                            tool=bot_tool,
                            date=date,
                        )))
                if in_first_month and bot_name not in fm_bot_contributors:
                    fm_bot_contributors.append(bot_name)
                    fm_tool_scores[bot_tool] = fm_tool_scores.get(bot_tool, 0) + 5

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
        pr_created = pr.get("created_at", "")

        # Determine if PR is within first month
        pr_in_first_month = False
        if first_month_cutoff and pr_created:
            pr_dt = parse_datetime_safe(pr_created)
            if pr_dt and pr_dt <= first_month_cutoff:
                pr_in_first_month = True

        # Criterion 1: Co-Authored-By in PR body
        coauthor_hits = detect_coauthor_ai(pr_body)
        for tool, count in coauthor_hits.items():
            co_author_count += count
            result.co_author_count += count
            tool_scores[tool] = tool_scores.get(tool, 0) + count * 3
            if len(evidence_items) < 200:
                evidence_items.append(asdict(EvidenceItem(
                    sha=f"PR#{pr.get('number', '')}",
                    snippet=pr_body[:200],
                    evidence_type="co_author",
                    tool=tool,
                    date=pr_created,
                )))
            if pr_in_first_month:
                fm_co_author_count += count
                fm_tool_scores[tool] = fm_tool_scores.get(tool, 0) + count * 3

        # Criterion 4: AI handle mentions in PRs
        handle_hits = count_ai_handle_mentions(pr_text)
        for tool, count in handle_hits.items():
            ai_mention_details[tool] = ai_mention_details.get(tool, 0) + count
            tool_scores[tool] = tool_scores.get(tool, 0) + count
            if len(evidence_items) < 200:
                evidence_items.append(asdict(EvidenceItem(
                    sha=f"PR#{pr.get('number', '')}",
                    snippet=pr_text[:150],
                    evidence_type="ai_handle_mention",
                    tool=tool,
                    date=pr_created,
                )))
            if pr_in_first_month:
                fm_ai_mention_details[tool] = fm_ai_mention_details.get(tool, 0) + count
                fm_tool_scores[tool] = fm_tool_scores.get(tool, 0) + count

        # Criterion 3: Bot PR authors
        for bot_tool, bot_names in AI_BOT_ACCOUNTS.items():
            if pr_author.lower() in [b.lower() for b in bot_names]:
                if pr_author not in result.bot_contributors:
                    result.bot_contributors.append(pr_author)
                    tool_scores[bot_tool] = tool_scores.get(bot_tool, 0) + 5
                    if len(evidence_items) < 200:
                        evidence_items.append(asdict(EvidenceItem(
                            sha=f"PR#{pr.get('number', '')}",
                            snippet=f"PR by {pr_author}",
                            evidence_type="bot_contributor",
                            tool=bot_tool,
                            date=pr_created,
                        )))
                if pr_in_first_month and pr_author not in fm_bot_contributors:
                    fm_bot_contributors.append(pr_author)
                    fm_tool_scores[bot_tool] = fm_tool_scores.get(bot_tool, 0) + 5

    result.ai_mention_details = ai_mention_details
    result.ai_mention_count = sum(ai_mention_details.values())

    # -------------------------------------------------------------------
    # 3. Check for AI config files in the repo (criterion 2)
    # -------------------------------------------------------------------
    # Full-history: check current HEAD tree
    status, tree_data = await client.get(
        f"/repos/{owner}/{repo}/git/trees/HEAD",
        params={"recursive": "1"},
    )
    tree_items = []
    if status == 200 and isinstance(tree_data, dict):
        tree_items = tree_data.get("tree", [])

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

    # Look up when each config file was first committed (for date evidence)
    for config_path in result.ai_config_files_found:
        cfg_tool = ""
        for t, paths in AI_CONFIG_FILES.items():
            if config_path in paths:
                cfg_tool = t
                break
        date_str = await client.get_oldest_commit_date_for_path(
            owner, repo, config_path
        )
        if date_str and len(evidence_items) < 200:
            evidence_items.append(asdict(EvidenceItem(
                sha="config_file",
                snippet=f"Config file: {config_path}",
                evidence_type="config_file",
                tool=cfg_tool,
                date=date_str,
            )))

    # First-month: check tree at newest first-month commit
    if fm_newest_commit_sha:
        fm_status, fm_tree_data = await client.get(
            f"/repos/{owner}/{repo}/git/trees/{fm_newest_commit_sha}",
            params={"recursive": "1"},
        )
        fm_tree_items = []
        if fm_status == 200 and isinstance(fm_tree_data, dict):
            fm_tree_items = fm_tree_data.get("tree", [])

        fm_repo_paths = set()
        for item in fm_tree_items:
            path = item.get("path", "")
            fm_repo_paths.add(path)
            parts = path.split("/")
            for i in range(1, len(parts)):
                fm_repo_paths.add("/".join(parts[:i]))

        for tool, config_paths in AI_CONFIG_FILES.items():
            for config_path in config_paths:
                if config_path in fm_repo_paths:
                    result.first_month_ai_config_files_found.append(config_path)
                    fm_tool_scores[tool] = fm_tool_scores.get(tool, 0) + 10

    # -------------------------------------------------------------------
    # 4. Apply binary classification criteria (full history)
    # -------------------------------------------------------------------
    reasons = []

    if result.co_author_count >= 1:
        reasons.append("co_authored_by")
    if len(result.ai_config_files_found) > 0:
        reasons.append("config_files")
    if len(result.bot_contributors) > 0:
        reasons.append("bot_contributors")
    if result.ai_mention_count >= 1:
        reasons.append("ai_handle_mentions")

    result.ai_authored = "yes" if len(reasons) > 0 else "no"
    result.ai_authored_reasons = reasons

    if tool_scores:
        result.likely_ai_agent = max(tool_scores, key=tool_scores.get)
    else:
        result.likely_ai_agent = "none"

    result.tool_scores = tool_scores

    # -------------------------------------------------------------------
    # 4b. Apply binary classification criteria (first month)
    # -------------------------------------------------------------------
    result.first_month_co_author_count = fm_co_author_count
    result.first_month_bot_contributors = fm_bot_contributors
    result.first_month_ai_mention_details = fm_ai_mention_details
    result.first_month_ai_mention_count = sum(fm_ai_mention_details.values())
    result.first_month_tool_scores = fm_tool_scores
    result.first_month_commits_scanned = fm_commits_scanned

    if first_month_cutoff is None:
        # No created_at available — cannot determine first month
        result.ai_authored_first_month = ""
    else:
        fm_reasons = []
        if fm_co_author_count >= 1:
            fm_reasons.append("co_authored_by")
        if len(result.first_month_ai_config_files_found) > 0:
            fm_reasons.append("config_files")
        if len(fm_bot_contributors) > 0:
            fm_reasons.append("bot_contributors")
        if result.first_month_ai_mention_count >= 1:
            fm_reasons.append("ai_handle_mentions")

        result.ai_authored_first_month = "yes" if len(fm_reasons) > 0 else "no"
        result.ai_authored_first_month_reasons = fm_reasons

    if fm_tool_scores:
        result.first_month_likely_ai_agent = max(fm_tool_scores, key=fm_tool_scores.get)
    else:
        result.first_month_likely_ai_agent = "none"

    # -------------------------------------------------------------------
    # 5. Compute likely_creators_details (percentage breakdown)
    # -------------------------------------------------------------------
    result.likely_creators_details = _compute_creators_details(result)

    # Trim evidence to keep output manageable
    result.commit_evidence = evidence_items[:100]

    # Compute earliest AI evidence date from commit_evidence
    if result.ai_authored == "yes" and evidence_items:
        evidence_dates = []
        for ev in evidence_items:
            dt = parse_datetime_safe(ev.get("date", ""))
            if dt:
                evidence_dates.append(dt)
        if evidence_dates:
            earliest = min(evidence_dates)
            result.date_first_ai_evidence = earliest.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    return result


def _compute_creators_details(result: ServerResult) -> dict:
    """
    Compute a percentage breakdown of likely creators (AI tools + human).
    Returns a dict like {"claude": 50, "copilot": 15, "human": 35} summing to 100.

    AI share is estimated from evidence density:
    - co_author_count / total_commits gives the strongest signal (direct attribution)
    - bot_contributors count as direct AI commits
    - config_files indicate deliberate tool setup (minimum floor)
    - ai_handle_mentions are weaker indirect evidence

    For ai_authored="no", returns {"human": 100}.
    """
    if result.ai_authored == "no" or not result.tool_scores:
        return {"human": 100}

    total_commits = max(result.total_commits_scanned, 1)

    # Direct evidence: co-authored commits + bot-authored commits (estimate 5 each)
    direct_evidence = result.co_author_count + len(result.bot_contributors) * 5
    evidence_ratio = min(direct_evidence / total_commits, 1.0)

    # AI share from direct commit evidence
    ai_share = evidence_ratio * 100

    # Minimum floors based on how many criteria triggered
    criteria_count = len(result.ai_authored_reasons)
    min_floor = {1: 10, 2: 25, 3: 40, 4: 55}.get(criteria_count, 10)
    ai_share = max(ai_share, min_floor)

    # Modest cap for weak-only evidence (just a few handle mentions, nothing else)
    if (criteria_count == 1
            and "ai_handle_mentions" in result.ai_authored_reasons
            and result.ai_mention_count <= 3):
        ai_share = min(ai_share, 15)

    ai_share = min(round(ai_share), 100)

    # Distribute AI share among tools proportionally via tool_scores
    total_tool_score = sum(result.tool_scores.values())
    creators = {}
    allocated = 0
    sorted_tools = sorted(result.tool_scores.items(), key=lambda x: -x[1])
    for tool, score in sorted_tools:
        tool_pct = round(ai_share * (score / total_tool_score))
        if tool_pct > 0:
            creators[tool] = tool_pct
            allocated += tool_pct

    # Remainder goes to human (can be 0 for fully bot-authored repos)
    human_share = 100 - allocated
    if human_share > 0:
        creators["human"] = human_share

    return creators


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

    ai_yes = sum(1 for r in all_results if r.get("ai_authored") == "yes" and not r.get("error"))
    ai_no = sum(1 for r in all_results if r.get("ai_authored") == "no" and not r.get("error"))

    reason_counts: dict[str, int] = {}
    for r in all_results:
        for reason in r.get("ai_authored_reasons", []):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    tool_counts: dict[str, int] = {}
    for r in all_results:
        if r.get("ai_authored") == "yes":
            agent = r.get("likely_ai_agent", "none")
            if agent and agent != "none":
                tool_counts[agent] = tool_counts.get(agent, 0) + 1

    config_counts: dict[str, int] = {}
    for r in all_results:
        for cfg in r.get("ai_config_files_found", []):
            config_counts[cfg] = config_counts.get(cfg, 0) + 1

    total_commits = sum(r.get("total_commits_scanned", 0) for r in all_results if not r.get("error"))
    repos_over_100 = sum(
        1 for r in all_results
        if not r.get("error") and r.get("total_commits_scanned", 0) > 100
    )

    with_coauthor = sum(1 for r in all_results if r.get("co_author_count", 0) > 0)

    with_mentions = sum(1 for r in all_results if r.get("ai_mention_count", 0) >= 1)
    total_mentions = sum(r.get("ai_mention_count", 0) for r in all_results)

    # First-month statistics
    fm_eligible = [r for r in all_results if not r.get("error") and r.get("ai_authored_first_month") != ""]
    fm_total = len(fm_eligible)
    fm_yes = sum(1 for r in fm_eligible if r.get("ai_authored_first_month") == "yes")
    fm_no = sum(1 for r in fm_eligible if r.get("ai_authored_first_month") == "no")

    fm_reason_counts: dict[str, int] = {}
    for r in fm_eligible:
        for reason in r.get("ai_authored_first_month_reasons", []):
            fm_reason_counts[reason] = fm_reason_counts.get(reason, 0) + 1

    fm_tool_counts: dict[str, int] = {}
    for r in fm_eligible:
        if r.get("ai_authored_first_month") == "yes":
            agent = r.get("first_month_likely_ai_agent", "none")
            if agent and agent != "none":
                fm_tool_counts[agent] = fm_tool_counts.get(agent, 0) + 1

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
        },
        "co_author_statistics": {
            "repos_with_ai_coauthors": with_coauthor,
        },
        "ai_mention_statistics": {
            "repos_with_gte1_mentions": with_mentions,
            "total_mentions_across_all_repos": total_mentions,
        },
        "first_month": {
            "eligible_servers": fm_total,
            "ai_authored_first_month_yes": fm_yes,
            "ai_authored_first_month_yes_pct": round(fm_yes / fm_total * 100, 1) if fm_total else 0,
            "ai_authored_first_month_no": fm_no,
            "ai_authored_first_month_no_pct": round(fm_no / fm_total * 100, 1) if fm_total else 0,
            "reason_breakdown": dict(sorted(fm_reason_counts.items(), key=lambda x: -x[1])),
            "likely_ai_tool_distribution": dict(sorted(fm_tool_counts.items(), key=lambda x: -x[1])),
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


async def backfill_dates(args):
    """Backfill date_first_ai_evidence for servers that are missing it.

    Two phases:
    1. Re-derive dates from existing commit_evidence entries (no API calls).
    2. For config-file-only servers still missing dates, look up the oldest
       commit that introduced each config file via the GitHub commits API.

    Updates aicreated_results.json, aicreated_summary.json, and
    data_unified_filtered.json in place.
    """
    token = get_github_token()
    if not token:
        logger.error("No GitHub token found. Set GH_TOKEN or configure gh CLI.")
        sys.exit(1)

    results_path = DATA_OUTPUT_DIR / "aicreated_results.json"
    if not results_path.exists():
        logger.error("Results file not found: %s", results_path)
        sys.exit(1)

    with open(results_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    ai_yes = [r for r in all_results if r.get("ai_authored") == "yes" and not r.get("error")]
    missing_before = sum(1 for r in ai_yes if not r.get("date_first_ai_evidence"))
    logger.info(
        "Backfill: %d ai_authored=yes servers, %d missing date_first_ai_evidence",
        len(ai_yes), missing_before,
    )

    # ------------------------------------------------------------------
    # Phase 1: re-derive from existing commit_evidence (free, no API)
    # ------------------------------------------------------------------
    phase1_fixed = 0
    for r in ai_yes:
        if r.get("date_first_ai_evidence"):
            continue
        evidence = r.get("commit_evidence", [])
        if not evidence:
            continue
        dates = []
        for ev in evidence:
            dt = parse_datetime_safe(ev.get("date", ""))
            if dt:
                dates.append(dt)
        if dates:
            earliest = min(dates)
            r["date_first_ai_evidence"] = earliest.strftime("%Y-%m-%dT%H:%M:%S+00:00")
            phase1_fixed += 1

    logger.info("Phase 1 (re-derive from existing evidence): fixed %d", phase1_fixed)

    # ------------------------------------------------------------------
    # Phase 2: config file date lookup via GitHub commits API
    # ------------------------------------------------------------------
    still_missing = [
        r for r in ai_yes if not r.get("date_first_ai_evidence")
    ]
    logger.info("Phase 2: %d servers still need config file date lookup", len(still_missing))

    if still_missing:
        concurrency = getattr(args, "concurrency", 5)
        client = GitHubAPIClient(token, max_concurrent=concurrency)
        await client.check_rate_limit()

        phase2_fixed = 0
        for i, r in enumerate(still_missing):
            parsed = extract_owner_repo(r.get("github_url", ""))
            if not parsed:
                continue
            owner, repo = parsed
            config_files = r.get("ai_config_files_found", [])
            earliest_date: datetime | None = None
            earliest_cfg = ""

            for cfg in config_files:
                date_str = await client.get_oldest_commit_date_for_path(
                    owner, repo, cfg
                )
                if date_str:
                    dt = parse_datetime_safe(date_str)
                    if dt and (earliest_date is None or dt < earliest_date):
                        earliest_date = dt
                        earliest_cfg = cfg

            if earliest_date:
                date_iso = earliest_date.strftime("%Y-%m-%dT%H:%M:%S+00:00")
                r["date_first_ai_evidence"] = date_iso
                # Append a config_file evidence item
                cfg_tool = ""
                for t, paths in AI_CONFIG_FILES.items():
                    if earliest_cfg in paths:
                        cfg_tool = t
                        break
                if not r.get("commit_evidence"):
                    r["commit_evidence"] = []
                r["commit_evidence"].append(asdict(EvidenceItem(
                    sha="config_file",
                    snippet=f"Config file: {earliest_cfg}",
                    evidence_type="config_file",
                    tool=cfg_tool,
                    date=date_iso,
                )))
                phase2_fixed += 1

            if (i + 1) % 100 == 0:
                logger.info(
                    "Phase 2 progress: %d/%d processed, %d fixed, API calls: %d",
                    i + 1, len(still_missing), phase2_fixed, client.request_count,
                )

        await client.close()
        logger.info(
            "Phase 2 (config file lookup): fixed %d (%d API calls)",
            phase2_fixed, client.request_count,
        )
    else:
        phase2_fixed = 0

    # ------------------------------------------------------------------
    # Phase 3: re-fetch commits/PRs for servers still missing dates
    # (these have PR-based or bot evidence but no config files to date)
    # ------------------------------------------------------------------
    phase3_missing = [
        r for r in all_results
        if r.get("ai_authored") == "yes" and not r.get("error") and not r.get("date_first_ai_evidence")
    ]
    logger.info("Phase 3: %d servers need commit/PR re-fetch for dates", len(phase3_missing))

    phase3_fixed = 0
    if phase3_missing:
        concurrency = getattr(args, "concurrency", 5)
        client3 = GitHubAPIClient(token, max_concurrent=concurrency)
        await client3.check_rate_limit()

        for i, r in enumerate(phase3_missing):
            parsed = extract_owner_repo(r.get("github_url", ""))
            if not parsed:
                continue
            owner, repo = parsed
            reasons = r.get("ai_authored_reasons", [])
            earliest_date: datetime | None = None

            # Scan commits for co-author / mention / bot evidence dates
            if any(reason in reasons for reason in ("co_authored_by", "ai_handle_mentions", "bot_contributors")):
                status, commits_page = await client3.get(
                    f"/repos/{owner}/{repo}/commits",
                    params={"per_page": "100", "page": "1"},
                )
                if status == 200 and isinstance(commits_page, list):
                    for commit_obj in commits_page:
                        commit_info = commit_obj.get("commit", {})
                        message = commit_info.get("message", "")
                        author_login = (commit_obj.get("author") or {}).get("login", "")
                        committer_login = (commit_obj.get("committer") or {}).get("login", "")
                        date = commit_info.get("author", {}).get("date", "")
                        if not date:
                            continue

                        has_evidence = False
                        # Check co-author
                        if detect_coauthor_ai(message):
                            has_evidence = True
                        # Check mentions
                        if count_ai_handle_mentions(message):
                            has_evidence = True
                        # Check bot authors
                        for bot_tool, bot_names in AI_BOT_ACCOUNTS.items():
                            bot_lower = [b.lower() for b in bot_names]
                            if author_login.lower() in bot_lower or committer_login.lower() in bot_lower:
                                has_evidence = True

                        if has_evidence:
                            dt = parse_datetime_safe(date)
                            if dt and (earliest_date is None or dt < earliest_date):
                                earliest_date = dt

            # Scan PRs for co-author / mention / bot evidence dates
            status, prs_data = await client3.get(
                f"/repos/{owner}/{repo}/pulls",
                params={"state": "all", "per_page": "30", "sort": "created", "direction": "asc"},
            )
            if status == 200 and isinstance(prs_data, list):
                for pr in prs_data:
                    pr_title = pr.get("title", "")
                    pr_body = pr.get("body", "") or ""
                    pr_author = (pr.get("user") or {}).get("login", "")
                    pr_text = f"{pr_title} {pr_body}"
                    pr_created = pr.get("created_at", "")
                    if not pr_created:
                        continue

                    has_evidence = False
                    if detect_coauthor_ai(pr_body):
                        has_evidence = True
                    if count_ai_handle_mentions(pr_text):
                        has_evidence = True
                    for bot_tool, bot_names in AI_BOT_ACCOUNTS.items():
                        if pr_author.lower() in [b.lower() for b in bot_names]:
                            has_evidence = True

                    if has_evidence:
                        dt = parse_datetime_safe(pr_created)
                        if dt and (earliest_date is None or dt < earliest_date):
                            earliest_date = dt

            if earliest_date:
                date_iso = earliest_date.strftime("%Y-%m-%dT%H:%M:%S+00:00")
                r["date_first_ai_evidence"] = date_iso
                if not r.get("commit_evidence"):
                    r["commit_evidence"] = []
                r["commit_evidence"].append(asdict(EvidenceItem(
                    sha="backfill",
                    snippet="Re-derived from commit/PR scan",
                    evidence_type="backfill_scan",
                    tool=r.get("likely_ai_agent", ""),
                    date=date_iso,
                )))
                phase3_fixed += 1

            if (i + 1) % 100 == 0:
                logger.info(
                    "Phase 3 progress: %d/%d processed, %d fixed, API calls: %d",
                    i + 1, len(phase3_missing), phase3_fixed, client3.request_count,
                )

        await client3.close()
        logger.info(
            "Phase 3 (commit/PR re-fetch): fixed %d (%d API calls)",
            phase3_fixed, client3.request_count,
        )

    total_fixed = phase1_fixed + phase2_fixed + phase3_fixed
    final_missing = sum(
        1 for r in all_results
        if r.get("ai_authored") == "yes" and not r.get("error") and not r.get("date_first_ai_evidence")
    )
    logger.info(
        "Backfill complete: %d dates added (phase1=%d, phase2=%d, phase3=%d), %d still missing",
        total_fixed, phase1_fixed, phase2_fixed, phase3_fixed, final_missing,
    )

    # Save updated results
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Updated %s", results_path)

    # Re-generate summary
    summary = generate_summary(all_results)
    summary_path = DATA_OUTPUT_DIR / "aicreated_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Updated %s", summary_path)

    # Merge into data_unified_filtered.json
    merge_into_unified_filtered(all_results)
    logger.info("Backfill done.")


async def main():
    parser = argparse.ArgumentParser(
        description="Detect AI-created MCP servers via git history mining"
    )
    parser.add_argument("--limit", type=int, default=0, help="Max servers to process (0=all)")
    parser.add_argument("--batch-size", type=int, default=10, help="Servers per concurrent batch")
    parser.add_argument("--max-prs", type=int, default=30, help="Max PRs to fetch per repo")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent API requests")
    parser.add_argument("--created-after", type=str, default=None, help="Only servers created after this date (YYYY-MM-DD)")
    parser.add_argument("--created-before", type=str, default=None, help="Only servers created before this date (YYYY-MM-DD)")
    parser.add_argument("--append-to", type=str, default=None, help="Append results to existing file (dedup by id)")
    parser.add_argument(
        "--backfill-dates", action="store_true",
        help="Backfill date_first_ai_evidence for servers missing it "
             "(config file commit lookup + re-derive from existing evidence)",
    )
    args = parser.parse_args()

    if args.backfill_dates:
        await backfill_dates(args)
        return

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

    # Filter by creation date if specified
    if args.created_after or args.created_before:
        before_filter = len(servers)
        filtered = []
        for s in servers:
            created = s.get("created_at", "")
            if not created:
                continue
            date_str = created[:10]  # YYYY-MM-DD
            if args.created_after and date_str < args.created_after:
                continue
            if args.created_before and date_str > args.created_before:
                continue
            filtered.append(s)
        servers = filtered
        logger.info(
            "Date filtered: %d -> %d servers (after=%s, before=%s)",
            before_filter, len(servers), args.created_after, args.created_before,
        )

    # Filter to servers with github_url
    servers = [s for s in servers if s.get("github_url")]
    logger.info("Servers with github_url: %d", len(servers))

    # Apply limit
    if args.limit > 0:
        servers = servers[:args.limit]
        logger.info("Limited to %d servers", len(servers))

    # Check for checkpoint
    checkpoint_path = DATA_OUTPUT_DIR / "aicreated_checkpoint.json"
    all_results: list[dict] = []
    processed_ids: set[str] = set()

    if args.resume and checkpoint_path.exists():
        all_results = load_checkpoint(checkpoint_path)
        processed_ids = {r["id"] for r in all_results}
        logger.info("Resumed from checkpoint: %d already processed", len(all_results))
        servers = [s for s in servers if s.get("id") not in processed_ids]
        logger.info("Remaining to process: %d", len(servers))

    # If appending, also skip already-processed IDs from existing file
    if args.append_to:
        append_path = Path(args.append_to)
        if append_path.exists():
            with open(append_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing_ids = {r["id"] for r in existing}
            before = len(servers)
            servers = [s for s in servers if s.get("id") not in existing_ids]
            logger.info("Append mode: skipping %d already-classified servers, %d remaining", before - len(servers), len(servers))

    if not servers:
        logger.info("No servers to process. Done.")
        return

    # Process in batches
    client = GitHubAPIClient(token, max_concurrent=args.concurrency)
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

    # If appending, merge with existing
    if args.append_to:
        append_path = Path(args.append_to)
        if append_path.exists():
            with open(append_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing_by_id = {r["id"]: r for r in existing}
            for r in all_results:
                existing_by_id[r["id"]] = r
            all_results = list(existing_by_id.values())
            logger.info("Merged with existing: %d total results", len(all_results))

    # Save final results
    output_path = DATA_OUTPUT_DIR / "aicreated_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Results saved to %s", output_path)

    # Generate and save summary
    summary = generate_summary(all_results)
    summary_path = DATA_OUTPUT_DIR / "aicreated_summary.json"
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
    fm = summary.get("first_month", {})
    logger.info(
        "First-month AI authored (yes): %d/%d (%.1f%%)",
        fm.get("ai_authored_first_month_yes", 0),
        fm.get("eligible_servers", 0),
        fm.get("ai_authored_first_month_yes_pct", 0),
    )
    logger.info("First-month reason breakdown: %s", json.dumps(fm.get("reason_breakdown", {}), indent=2))
    logger.info("First-month tool distribution: %s", json.dumps(fm.get("likely_ai_tool_distribution", {}), indent=2))

    # Merge AI-created fields into data_unified_filtered.json
    merge_into_unified_filtered(all_results)

    # Clean up checkpoint on successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Checkpoint removed (run completed successfully)")


def merge_into_unified_filtered(all_results: list[dict]) -> None:
    """
    Merge ai_authored, ai_authored_reasons, likely_ai_agent, and
    likely_creators_details into data/initial/data_unified_filtered.json
    so downstream scripts (clservers_4_datamatch, cltools_datamatch) can
    pick them up via the standard metadata lookup.
    """
    if not DATA_INPUT.exists():
        logger.warning("Cannot merge: %s not found", DATA_INPUT)
        return

    logger.info("Merging AI-created fields into %s ...", DATA_INPUT)

    # Build lookup: id -> fields to merge
    ai_lookup: dict[str, dict] = {}
    for r in all_results:
        if r.get("error"):
            continue
        ai_lookup[r["id"]] = {
            "ai_authored": r.get("ai_authored", ""),
            "ai_authored_reasons": r.get("ai_authored_reasons", []),
            "likely_ai_agent": r.get("likely_ai_agent", ""),
            "likely_creators_details": r.get("likely_creators_details", {}),
            "ai_authored_first_month": r.get("ai_authored_first_month", ""),
            "ai_authored_first_month_reasons": r.get("ai_authored_first_month_reasons", []),
            "first_month_likely_ai_agent": r.get("first_month_likely_ai_agent", "none"),
            "date_first_ai_evidence": r.get("date_first_ai_evidence", ""),
        }

    logger.info("AI-created lookup: %d entries (excluding errors)", len(ai_lookup))

    # Load, update, save
    with open(DATA_INPUT, "r", encoding="utf-8") as f:
        unified = json.load(f)

    matched = 0
    for server in unified:
        server_id = server.get("id", "")
        if server_id in ai_lookup:
            server.update(ai_lookup[server_id])
            matched += 1

    logger.info(
        "Merged AI-created fields: %d/%d servers matched (%.1f%%)",
        matched, len(unified),
        matched / len(unified) * 100 if unified else 0,
    )

    with open(DATA_INPUT, "w", encoding="utf-8") as f:
        json.dump(unified, f, indent=2, ensure_ascii=False)

    logger.info("Updated %s with AI-created fields", DATA_INPUT)


if __name__ == "__main__":
    asyncio.run(main())

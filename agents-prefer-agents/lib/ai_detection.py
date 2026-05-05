"""Event-level AI-authorship detection.

Ported from ``scripts/data-classification-aicreatedmcp/detect_ai_created.py``.
The MCP pipeline classifies a *repository*; this module classifies a single
GitHub *event* (commit, PR body, review body, review comment, issue comment).

Three-way taxonomy (per CLAUDE/paper convention):
- **AI-bot**: actor login is on our AI-bot allowlist (excludes maintenance
  bots like Dependabot/Renovate/CI).
- **AI-powered**: actor login is human, but the event payload carries a
  ``Co-Authored-By:`` trailer naming an AI tool.
- **human**: none of the above (may include silent AI support that left no
  trailer or bot login).

``AI-bot + AI-powered`` together are referred to as **AI authored** in the
paper. Handle/name mentions in free text (e.g. ``@claude please fix``) are
not used to flag an event as AI — a human typing ``@claude`` is not
themselves an AI contributor.

This module is intentionally dependency-free beyond ``re`` so it can be
re-used inside async pipelines without adding to the import graph.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# AI Tool Handle Patterns (criterion 4: mentions in commit/PR/review text)
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
        r"\bcopilot[\s-]?coding[\s-]?agent\b",
        r"\bcopilot[\s-]?swe[\s-]?agent\b",
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
        r"\bcursor[\s-]?agent\b",
        r"\bcursor[\s-]?background[\s-]?agent\b",
    ],
    "devin": [
        r"@devin\b",
        r"\bdevin[\s-]?ai\b",
    ],
    "codex": [
        r"@codex\b",
        r"\bopenai[\s-]?codex\b",
        r"\bcodex[\s-]?cli\b",
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
        r"\bgemini[\s-]?cli\b",
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
    "jules": [
        r"\bjules[\s-]?ai\b",
        r"@jules\b",
    ],
    "openhands": [
        r"\bopenhands\b",
    ],
    "codegen_sh": [
        r"@codegen-sh\b",
        r"\bcodegen\.sh\b",
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
        r"co-authored-by:.*copilot-swe-agent",
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
    "cursor": [
        r"co-authored-by:.*cursor",
    ],
    "jules": [
        r"co-authored-by:.*jules",
    ],
    "openhands": [
        r"co-authored-by:.*openhands",
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
    "copilot": [
        "copilot",
        "copilot[bot]",
        "github-copilot[bot]",
        "copilot-swe-agent",
        "copilot-swe-agent[bot]",
        "copilot-pull-request-reviewer",
        "copilot-pull-request-reviewer[bot]",
    ],
    "claude": ["claude[bot]", "claude-bot", "anthropic-ai[bot]"],
    "cursor": ["cursor-agent", "cursor[bot]", "cursoragent[bot]"],
    "jules": ["jules-ai[bot]", "jules[bot]"],
    "codegen_sh": ["codegen-sh[bot]"],
    "openhands": ["openhands-agent", "openhands-agent[bot]"],
    "coderabbit": ["coderabbitai", "coderabbitai[bot]"],
    "cubic_ai": ["cubic-dev-ai", "cubic-dev-ai[bot]"],
    "gemini": ["gemini-code-assist", "gemini-code-assist[bot]"],
    "greptile": ["greptile-apps", "greptile-apps[bot]", "greptileai", "greptileai[bot]"],
    "sweep": ["sweep-ai[bot]", "sweep-nightly[bot]"],
    "qodo": ["qodo-merge[bot]", "qodo-merge-pro[bot]", "codiumai-pr-agent[bot]"],
    "ellipsis": ["ellipsis-dev[bot]", "ellipsis[bot]"],
}

# Non-AI bots to exclude from ai_authored classification
NON_AI_BOTS: set[str] = {
    "dependabot",
    "dependabot[bot]",
    "renovate",
    "renovate[bot]",
    "snyk-bot",
    "snyk[bot]",
    "greenkeeper[bot]",
    "allcontributors[bot]",
    "pre-commit-ci[bot]",
    "stale[bot]",
    "codecov[bot]",
    "codecov-commenter",
    "github-actions[bot]",
    "mergify[bot]",
    "semantic-release-bot",
    "release-please[bot]",
    "imgbot[bot]",
    "deepsource-autofix[bot]",
}

# Compile patterns once.
COMPILED_COAUTHOR_PATTERNS: dict[str, list[re.Pattern]] = {
    tool: [re.compile(p, re.IGNORECASE) for p in patterns]
    for tool, patterns in CO_AUTHOR_AI_PATTERNS.items()
}
COMPILED_HANDLE_PATTERNS: dict[str, list[re.Pattern]] = {
    tool: [re.compile(p, re.IGNORECASE) for p in patterns]
    for tool, patterns in AI_HANDLE_PATTERNS.items()
}

# Flat lookup: login-lower -> (tool, type). Type is "ai_bot" or "non_ai_bot".
_BOT_LOOKUP: dict[str, tuple[str, str]] = {}
for tool, names in AI_BOT_ACCOUNTS.items():
    for n in names:
        _BOT_LOOKUP[n.lower()] = (tool, "ai_bot")
for n in NON_AI_BOTS:
    _BOT_LOOKUP[n.lower()] = ("", "non_ai_bot")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@dataclass
class ActorClassification:
    """Result of classifying one event (commit / PR body / review / comment).

    ``actor_type`` values:
      - ``AI-bot``: login is on the AI-bot allowlist.
      - ``AI-powered``: login is human, but payload carries a ``Co-Authored-By``
        trailer naming an AI tool.
      - ``human``: neither of the above.
      - ``non_ai_bot``: maintenance bot (Dependabot/Renovate/CI/etc.) — kept
        as a separate label so downstream code can drop these events from the
        AI-vs-human comparison rather than misclassifying them as human.
    """

    actor_type: str  # "AI-bot" | "AI-powered" | "human" | "non_ai_bot"
    ai_family: str = "none"  # claude|copilot|devin|cursor|aider|cline|...|none
    confidence: str = "none"  # high | none (no medium/low after dropping handle-mention path)
    reasons: list[str] = field(default_factory=list)
    # For auditing. Each tool -> hit count.
    coauthor_hits: dict[str, int] = field(default_factory=dict)
    handle_hits: dict[str, int] = field(default_factory=dict)
    # If the login is a known bot, keep the exact family we matched.
    bot_login_family: str = ""


def detect_coauthor_ai(text: str) -> dict[str, int]:
    """Return {tool: hit_count} for Co-Authored-By trailers in text."""
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
    """Return {tool: hit_count} for AI tool handle/name mentions in text."""
    if not text:
        return {}
    results: dict[str, int] = {}
    for tool, patterns in COMPILED_HANDLE_PATTERNS.items():
        for pattern in patterns:
            matches = pattern.findall(text)
            if matches:
                results[tool] = results.get(tool, 0) + len(matches)
    return results


def classify_login(login: str) -> tuple[str, str]:
    """Classify an actor login. Returns (actor_type, ai_family).

    Possible returns:
    - ("AI-bot", family)       -- known AI bot account
    - ("non_ai_bot", "")       -- dependabot / renovate / CI bots we don't care about
    - ("human", "")            -- any other account
    """
    if not login:
        return ("human", "")
    key = login.lower().strip()
    if key in _BOT_LOOKUP:
        fam, t = _BOT_LOOKUP[key]
        return (("AI-bot" if t == "ai_bot" else "non_ai_bot"), fam)
    # Heuristic: any login ending in "[bot]" that isn't on the allow/deny list
    # is treated as a non-AI bot. This prevents unknown CI/maintenance bots
    # from being classified as humans.
    if key.endswith("[bot]"):
        return ("non_ai_bot", "")
    return ("human", "")


def classify_event(
    actor_login: str,
    event_text: str = "",
) -> ActorClassification:
    """Classify a single GitHub event.

    Parameters
    ----------
    actor_login : str
        The ``login`` field of the event's author / reviewer / commenter.
    event_text : str
        Concatenated text of the event payload (commit message, PR body,
        review body, comment body). Used to look for co-author trailers and
        handle mentions.

    Returns
    -------
    ActorClassification
    """
    login_type, login_family = classify_login(actor_login)
    coauthor_hits = detect_coauthor_ai(event_text)
    handle_hits = count_ai_handle_mentions(event_text)
    reasons: list[str] = []
    ai_family = "none"

    if login_type == "non_ai_bot":
        # CI / dependency bot event — explicitly excluded from AI analysis.
        return ActorClassification(
            actor_type="non_ai_bot",
            ai_family="none",
            confidence="none",
            reasons=["non_ai_bot_login"],
            coauthor_hits=coauthor_hits,
            handle_hits=handle_hits,
            bot_login_family="",
        )

    if login_type == "AI-bot":
        reasons.append("bot_account")
        ai_family = login_family or "unknown-ai"
        confidence = "high"
        # Bot account supersedes everything else.
        return ActorClassification(
            actor_type="AI-bot",
            ai_family=ai_family,
            confidence=confidence,
            reasons=reasons,
            coauthor_hits=coauthor_hits,
            handle_hits=handle_hits,
            bot_login_family=login_family,
        )

    # login is "human" — decide whether to upgrade to AI-powered.
    if coauthor_hits:
        reasons.append("co_authored_by")
        ai_family = max(coauthor_hits, key=coauthor_hits.get)
        return ActorClassification(
            actor_type="AI-powered",
            ai_family=ai_family,
            confidence="high",
            reasons=reasons,
            coauthor_hits=coauthor_hits,
            handle_hits=handle_hits,
        )

    # Handle/name mentions in free text (e.g. "@claude please fix") are NOT
    # treated as evidence that the actor is AI: a human typing @claude is
    # not themselves an AI contributor. We retain the hit dict for auditing.
    return ActorClassification(
        actor_type="human",
        ai_family="none",
        confidence="none",
        reasons=[],
        coauthor_hits={},
        handle_hits=handle_hits,
    )


def classify_config_files(repo_paths: set[str]) -> dict[str, list[str]]:
    """Given the set of paths in a repo's tree, return which AI tools have
    config-file evidence. Used at PR level via the HEAD tree of the PR's
    target repo. Returns {tool: [paths_found]}.
    """
    found: dict[str, list[str]] = {}
    for tool, paths in AI_CONFIG_FILES.items():
        hits = [p for p in paths if p in repo_paths]
        if hits:
            found[tool] = hits
    return found


__all__ = [
    "ActorClassification",
    "AI_HANDLE_PATTERNS",
    "CO_AUTHOR_AI_PATTERNS",
    "AI_CONFIG_FILES",
    "AI_BOT_ACCOUNTS",
    "NON_AI_BOTS",
    "detect_coauthor_ai",
    "count_ai_handle_mentions",
    "classify_login",
    "classify_event",
    "classify_config_files",
]

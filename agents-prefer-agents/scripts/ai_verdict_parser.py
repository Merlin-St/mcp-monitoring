"""AI-bot review verdict parser.

Per-bot regex catalogue that extracts approve/reject verdicts from the
structured-format headers each bot writes in its `COMMENTED` review bodies
(no sentiment classification). The pattern set was hand-audited against the
v2 corpus (see `paper/test_ai_verdict_regex.py` for the auditable test
harness and `99_causalvalidity.md` §7 for the full catalogue, sample yields,
and precision notes).

Public API:

    BOT_PATTERNS   -- per-bot ordered (kind, regex, description) list
    AI_BOTS_SUBS   -- substring allowlist for AI-bot logins
    classify(login, body) -> 'approve' | 'reject' | None
    parse_ai_opinions(prs_dir, repos) -> pandas.DataFrame
        with columns ['repo','number','t','kind','bot'].

The classifier strips <details>...</details> blocks for Claude bodies before
matching to avoid the trailing-LGTM regex hitting quoted text inside Claude's
"Extended reasoning" appendix.

This module is intentionally a near-verbatim copy of the patterns in
`paper/test_ai_verdict_regex.py` so the pipeline matches the audited spec.
Do not modify regexes here without re-running the test harness and updating
the precision audit.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

AI_BOTS_SUBS = [
    "cubic-dev-ai", "cubic-ai", "coderabbitai",
    "copilot-pull-request-reviewer", "gemini-code-assist", "greptile-apps",
    "sourcery-ai", "codiumai-pr-agent", "pr-agent", "sweep-ai",
    "devin-ai-integration", "claude", "cursor", "codegen-sh",
    "copilot-swe-agent",
]

BOT_PATTERNS: dict[str, list[tuple[str, re.Pattern, str]]] = {
    "cubic-dev-ai": [
        ("approve", re.compile(r"(?im)^\s*\*\*No issues found\b[^*]*\*\*"),
            'Cubic header "**No issues found** across N file(s)"'),
        ("reject", re.compile(r"(?im)^\s*\*\*([1-9]\d*) issues? found\b[^*]*\*\*"),
            'Cubic header "**N issues found** across N file(s)"'),
    ],
    "cubic-ai": [
        ("approve", re.compile(r"(?im)^\s*\*\*No issues found\b[^*]*\*\*"),
            "Cubic alt-name: same"),
        ("reject", re.compile(r"(?im)^\s*\*\*([1-9]\d*) issues? found\b[^*]*\*\*"),
            "Cubic alt-name: same"),
    ],
    "coderabbitai": [
        ("approve", re.compile(r"(?im)^\s*\*\*Actionable comments posted:\s*0\s*\*\*"),
            'CodeRabbit "**Actionable comments posted: 0**"'),
        ("reject", re.compile(r"(?im)^\s*\*\*Actionable comments posted:\s*([1-9]\d*)\s*\*\*"),
            'CodeRabbit "**Actionable comments posted: N**" (N>=1)'),
    ],
    "copilot-pull-request-reviewer": [
        ("approve", re.compile(r"(?im)Copilot reviewed[^\n]*and generated no new comments"),
            'Copilot "...generated no new comments"'),
        ("reject", re.compile(r"(?im)Copilot reviewed[^\n]*and generated ([1-9]\d*) comments?"),
            'Copilot "...generated N comments" (N>=1)'),
    ],
    "greptile-apps": [
        ("approve", re.compile(r"(?im)<sub>\s*\d+\s*files?\s*reviewed,\s*no comments?\s*</sub>"),
            'Greptile "<sub>N files reviewed, no comments</sub>"'),
        ("reject", re.compile(r"(?im)<sub>\s*\d+\s*files?\s*reviewed,\s*([1-9]\d*)\s*comments?\s*</sub>"),
            'Greptile "<sub>N files reviewed, M comments</sub>" (M>=1)'),
    ],
    "cursor": [
        ("approve", re.compile(r"(?im)^\s*(###\s*)?✅\s*Bugbot reviewed your changes and found no (bugs|new issues)!?"),
            'Cursor "✅ Bugbot reviewed your changes and found no bugs!" (or "no new issues!")'),
        ("reject", re.compile(r"(?im)Cursor (Bugbot|Bug-Bot) has reviewed your changes and found ([1-9]\d*) potential issues?"),
            'Cursor Bugbot "found N potential issues" (N>=1)'),
    ],
    "sourcery-ai": [
        ("approve",
            re.compile(r"(?im)^Hey(\s+(there|@\S+))?\s*-\s*I\'?ve reviewed your changes and they look great"),
            'Sourcery "Hey [there|@user] - I\'ve reviewed your changes and they look great"'),
        ("reject",
            re.compile(r"(?im)^Hey(\s+(there|@\S+))?\s*-\s*I\'?ve "
                       r"(reviewed your changes\s*-\s*here\'?s some feedback"
                       r"|reviewed your changes and found some issues"
                       r"|left some"
                       r"|found \d+)"),
            "Sourcery critique intros"),
    ],
    "claude": [
        ("approve", re.compile(r"\ALGTM\b", re.IGNORECASE),
            'Claude "LGTM..." anchored to absolute start of body'),
        ("approve", re.compile(
            r"(?im)\b(all\s+(prior|previous|previously)|prior|previous)\s+"
            r"(feedback|concerns?|issues?|review\s+concerns?|flagged\s+issues?"
            r"|identified\s+bugs?|review\s+rounds?)\b"
            r"[^\n]*?[—\-,.]?\s*LGTM\.?\s*$"),
            'Claude "...prior feedback addressed - LGTM." trailing-verdict approval'),
        ("approve", re.compile(
            r"(?im)^[\s>*\-#]*\*{0,2}\s*(Recommendation|Verdict|Conclusion|Decision)\s*\*{0,2}"
            r"\s*[:\-]\s*(\*{0,2}\s*)?(✅\s*)?(Approve|LGTM)\b"),
            'Claude verdict line "(Recommendation|Verdict|...): (Approve|LGTM)"'),
        ("approve", re.compile(
            r"(?im)^[\s>*\-]*(\*\*|✅\s*)\s*(Approve|LGTM)\s*(\*\*)?\s*[!.]?\s*$"),
            'Claude standalone bold/emoji verdict line'),
    ],
    # gemini-code-assist intentionally omitted: free-prose, no stable header.
}

_DETAILS_RE = re.compile(r"<details>.*?</details>", re.DOTALL)


def _preprocess_for_match(login: str, body: str) -> str:
    """Strip <details>...</details> for Claude bodies."""
    if "claude" in login:
        return _DETAILS_RE.sub("", body)
    return body


def classify(login: str, body: str) -> str | None:
    """Return 'approve', 'reject', or None for an AI-bot review body."""
    body_match = _preprocess_for_match(login, body)
    for sub, patlist in BOT_PATTERNS.items():
        if sub in login:
            for kind, cre, _desc in patlist:
                if cre.search(body_match):
                    return kind
    return None


def parse_ai_opinions(prs_dir: Path, repos: set[str] | None = None) -> pd.DataFrame:
    """Scan `prs_dir/*.jsonl` and return one row per AI-bot opinion event.

    Columns: repo, number, t (datetime), kind ∈ {approve_native, reject_native,
    approve_parsed, reject_parsed}, bot.

    If `repos` is given, only files matching those repos (after the
    `owner__name` -> `owner/name` convention) are scanned.
    """
    rows = []
    for fp in sorted(Path(prs_dir).glob("*.jsonl")):
        if repos is not None:
            repo_guess = fp.stem.replace("__", "/")
            if repo_guess not in repos:
                continue
        with open(fp) as f:
            for line in f:
                d = json.loads(line)
                repo, num = d["repo"], d["number"]
                for r in d.get("reviews", []):
                    login = (r.get("author_login") or "").lower()
                    state = r.get("state") or ""
                    body = (r.get("body") or "").strip()
                    ts = r.get("submitted_at")
                    if not any(s in login for s in AI_BOTS_SUBS):
                        continue
                    kind = None
                    if state == "APPROVED":
                        kind = "approve_native"
                    elif state == "CHANGES_REQUESTED":
                        kind = "reject_native"
                    elif state == "COMMENTED" and body:
                        v = classify(login, body)
                        if v == "approve":
                            kind = "approve_parsed"
                        elif v == "reject":
                            kind = "reject_parsed"
                    if kind:
                        rows.append((repo, num, ts, kind, login))
    df = pd.DataFrame(rows, columns=["repo", "number", "t", "kind", "bot"])
    if not df.empty:
        df["t"] = pd.to_datetime(df.t)
    return df

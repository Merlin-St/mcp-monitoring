#!/usr/bin/env python3
"""
test_ai_verdict_regex.py — exploratory parser for AI-bot review verdicts.

NOT part of the pipeline. Standalone, replicable. Run with:
    source /home/ubuntu/mcp-monitoring/.venv/bin/activate
    python paper/test_ai_verdict_regex.py [--samples N]

Inputs:  data/prs/*.jsonl  (raw GraphQL fetches; one PR per line)
Outputs: stdout summary (per-bot match counts, sample matches/unmatches)

The patterns parse a bot's own structured-format verdict line. They are
*not* sentiment classifiers — each pattern targets a fixed header that the
bot itself writes in a stable template.

Bot-by-bot logic. First match wins per review.
"""
from __future__ import annotations
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRS_DIR = ROOT / "data" / "prs"

# AI-bot allowlist substrings (matches the existing pipeline's classifier).
AI_BOTS_SUBS = [
    "cubic-dev-ai", "cubic-ai", "coderabbitai",
    "copilot-pull-request-reviewer", "gemini-code-assist", "greptile-apps",
    "sourcery-ai", "codiumai-pr-agent", "pr-agent", "sweep-ai",
    "devin-ai-integration", "claude", "cursor", "codegen-sh",
    "copilot-swe-agent",
]

# Per-bot ordered patterns. First match wins.
# Each entry: (verdict_kind, compiled_regex, description).
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
        # Cursor Bugbot DOES post "no bugs" reviews, just with a different prefix:
        # leading green-check emoji and an exclamation-marked verdict.
        ("approve", re.compile(r"(?im)^\s*(###\s*)?✅\s*Bugbot reviewed your changes and found no (bugs|new issues)!?"),
            'Cursor Bugbot "✅ Bugbot reviewed your changes and found no bugs!" (or "no new issues!"). '
            'Optional "### " heading prefix; emoji is required.'),
        # Reject: standard Bugbot one-liner. The prefix "<!-- BUGBOT_REVIEW -->"
        # often appears on a previous line; the (?im) flag finds the verdict
        # anywhere in the body. Status-only messages ("This PR is being reviewed",
        # "billing cycle", "<details open>" Bug-detail blocks) do NOT match because
        # they lack the verdict sentence.
        ("reject", re.compile(r"(?im)Cursor (Bugbot|Bug-Bot) has reviewed your changes and found ([1-9]\d*) potential issues?"),
            'Cursor Bugbot "found N potential issues" (N>=1).'),
    ],
    "sourcery-ai": [
        # Sourcery has multiple greeting variants: "Hey -", "Hey there -", and
        # "Hey @username -". All three lead to either an approval ("...look great!")
        # or a critique. The (there|@\S+) optional group covers all three.
        ("approve",
            re.compile(r"(?im)^Hey(\s+(there|@\S+))?\s*-\s*I\'?ve reviewed your changes and they look great"),
            'Sourcery: "Hey [there|@user] - I\'ve reviewed your changes and they look great" (LGTM-equivalent).'),
        ("reject",
            re.compile(r"(?im)^Hey(\s+(there|@\S+))?\s*-\s*I\'?ve "
                       r"(reviewed your changes\s*-\s*here\'?s some feedback"
                       r"|reviewed your changes and found some issues"
                       r"|left some"
                       r"|found \d+)"),
            'Sourcery: critique intros. Covers {here\'s some feedback, '
            'found some issues that need to be addressed, left some high level '
            'feedback, found N issues}. Excludes "Sorry... rate limit" and '
            '"We\'ve reviewed... using the Sourcery rules engine" (auto-lint, '
            'not an opinion).'),
    ],
    "claude": [
        # Approve at start of body: e.g., "LGTM — straightforward fix..."
        ("approve", re.compile(r"\ALGTM\b", re.IGNORECASE),
            'Claude "LGTM..." anchored to start of body. Excludes setup messages ("## Claude Code Review").'),
        # Approve at end of body: e.g., "All previous feedback addressed — LGTM."
        # Requires an "addressed/resolved" phrase about prior feedback, then LGTM at end of line.
        # In a hand-audit of 16 corpus matches this had 100% precision; a sanity check on
        # other bots' bodies produced zero matches, so the FP risk is negligible.
        # NOTE: classify() strips <details>...</details> blocks before applying this
        # pattern, since Claude's "Extended reasoning" appendix can contain "LGTM" in
        # quoted text.
        ("approve", re.compile(
            r"(?im)\b(all\s+(prior|previous|previously)|prior|previous)\s+"
            r"(feedback|concerns?|issues?|review\s+concerns?|flagged\s+issues?"
            r"|identified\s+bugs?|review\s+rounds?)\b"
            r"[^\n]*?[—\-,.]?\s*LGTM\.?\s*$"),
            'Claude "...prior feedback addressed - LGTM." trailing-verdict approval. '
            'Recovers ~16 reviews where LGTM comes at the end after a continuation phrase. '
            'Strip <details> blocks before matching.'),
        # No reject pattern: Claude's critique reviews are free-prose without a stable
        # structured header. ~150 unclassified bodies split roughly 60% "human-defer"
        # ("looks correct BUT human should sign off") / 25% trailing-LGTM-not-matched-here
        # / 15% clear reject. Auto-classifying any of these would either pollute the
        # reject cell with deference signals or require an LLM classifier.
    ],
    # gemini-code-assist intentionally omitted: format is free-prose, no stable
    # structured verdict line. ~7.5k events unclassified; would require an
    # LLM classifier (not done here).
}


_DETAILS_RE = re.compile(r"<details>.*?</details>", re.DOTALL)


def _preprocess_for_match(login: str, body: str) -> str:
    """Strip <details>...</details> blocks for Claude bodies so the trailing-LGTM
    pattern matches the visible verdict line, not text inside the
    'Extended reasoning' appendix.
    """
    if "claude" in login:
        return _DETAILS_RE.sub("", body)
    return body


def classify(login: str, body: str) -> str | None:
    """Return 'approve', 'reject', or None."""
    body_match = _preprocess_for_match(login, body)
    for sub, patlist in BOT_PATTERNS.items():
        if sub in login:
            for kind, cre, _desc in patlist:
                if cre.search(body_match):
                    return kind
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prs-dir", default=str(PRS_DIR))
    ap.add_argument("--samples", type=int, default=4,
                    help="Per-pattern sample matches to print")
    ap.add_argument("--unmatched-samples", type=int, default=3,
                    help="Per-bot unmatched samples to print")
    args = ap.parse_args()

    files = sorted(Path(args.prs_dir).glob("*.jsonl"))
    print(f"# AI-bot review verdict regex tester")
    print(f"Scanning {len(files)} repo JSONL files in {args.prs_dir}")

    counts = Counter()
    per_pattern = defaultdict(lambda: {"matches": 0, "samples": []})
    per_bot = Counter()
    unmatched_samples = defaultdict(list)

    for fp in files:
        with open(fp) as f:
            for line in f:
                d = json.loads(line)
                for r in d.get("reviews", []):
                    login = (r.get("author_login") or "").lower()
                    state = r.get("state") or ""
                    body = r.get("body") or ""
                    if not any(s in login for s in AI_BOTS_SUBS):
                        continue
                    if state != "COMMENTED" or not body.strip():
                        continue
                    counts["total"] += 1
                    matched = False
                    for sub, patlist in BOT_PATTERNS.items():
                        if sub not in login:
                            continue
                        for kind, cre, desc in patlist:
                            if cre.search(body):
                                pkey = f"{sub}:{kind}"
                                per_pattern[pkey]["matches"] += 1
                                if len(per_pattern[pkey]["samples"]) < args.samples:
                                    per_pattern[pkey]["samples"].append(
                                        body[:250].replace("\n", " / "))
                                counts[kind] += 1
                                per_bot[(login, kind)] += 1
                                matched = True
                                break
                        if matched:
                            break
                    if not matched:
                        per_bot[(login, "unmatched")] += 1
                        if len(unmatched_samples[login]) < args.unmatched_samples:
                            unmatched_samples[login].append(
                                body[:250].replace("\n", " / "))

    tot = counts["total"]
    print(f"\n## Totals")
    print(f"  AI-bot COMMENTED reviews with body: {tot}")
    print(f"  approve: {counts['approve']}  ({100*counts['approve']/max(tot,1):.1f}%)")
    print(f"  reject:  {counts['reject']}   ({100*counts['reject']/max(tot,1):.1f}%)")
    unm = tot - counts['approve'] - counts['reject']
    print(f"  unmatched: {unm}             ({100*unm/max(tot,1):.1f}%)")

    print(f"\n## Per-bot breakdown")
    bots = sorted({k[0] for k in per_bot})
    for b in bots:
        a = per_bot.get((b, "approve"), 0)
        r_ = per_bot.get((b, "reject"), 0)
        u = per_bot.get((b, "unmatched"), 0)
        t = a + r_ + u
        print(f"  {b:35s} approve={a:6d}  reject={r_:6d}  unmatched={u:6d}  (total={t})")

    print(f"\n## Sample matches per pattern")
    for sub, patlist in BOT_PATTERNS.items():
        for kind, _cre, desc in patlist:
            pkey = f"{sub}:{kind}"
            s = per_pattern[pkey]
            print(f"\n[{pkey}] ({s['matches']} matches) — {desc}")
            for ex in s["samples"]:
                print(f"  {ex[:240]}")

    print(f"\n## Unmatched samples per bot")
    for b in sorted(unmatched_samples):
        if per_bot.get((b, "unmatched"), 0) == 0:
            continue
        print(f"\n[{b}] {per_bot.get((b,'unmatched'), 0)} unmatched")
        for ex in unmatched_samples[b]:
            print(f"  {ex}")


if __name__ == "__main__":
    main()

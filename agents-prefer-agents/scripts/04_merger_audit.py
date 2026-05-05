"""Merger-detection audit (99_instruction.md §4.3).

Question: can we reliably identify when an AI agent merged a PR?

Approach, in order — stop at the first clean answer:
  Step 1. ``merged_by.login`` + ``merged_by.type`` on every merged PR in our
          already-fetched data. What fraction are AI bots (on the allowlist)?
          What fraction are ``Bot`` but NOT on our allowlist?
  Step 2. Cross-check timeline ``MergedEvent.actor`` vs ``merged_by``.
          If they disagree, which one is more informative?
  Step 3. (Not done here — would require additional API calls.)

Output: ``results/merger_detection_audit.md``.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from importlib import import_module

utils = import_module("99_utils")
get_logger = utils.get_logger
DATA_DIR = utils.DATA_DIR
RESULTS_DIR = utils.RESULTS_DIR

from lib.ai_detection import (  # noqa: E402
    classify_login,
    AI_BOT_ACCOUNTS,
    NON_AI_BOTS,
)

logger = get_logger("04_merger_audit")


def main():
    prs_path = DATA_DIR / "prs"
    all_merged: list[dict] = []
    for p in sorted(prs_path.glob("*.jsonl")):
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                pr = json.loads(line)
                if not pr.get("merged", False):
                    continue
                all_merged.append(pr)
    if not all_merged:
        logger.error("No merged PRs found. Run 02_fetch_prs.py first.")
        sys.exit(1)

    logger.info("Loaded %d merged PRs across %s.", len(all_merged), prs_path)

    # Optionally subsample for the headline audit (§4.3 says 1000 is enough).
    sample = all_merged if len(all_merged) <= 1000 else random.sample(all_merged, 1000)
    logger.info("Auditing %d merged PRs (sampled from %d).", len(sample), len(all_merged))

    # --- STEP 1: merged_by field ---
    mb_type_counts: Counter = Counter()
    mb_ai_family: Counter = Counter()
    mb_unknown_bot_logins: Counter = Counter()
    mb_human_count = 0
    mb_missing_count = 0
    for pr in sample:
        login = pr.get("merged_by_login", "")
        api_type = pr.get("merged_by_type", "") or ""
        mb_type_counts[api_type] += 1
        if not login:
            mb_missing_count += 1
            continue
        actor_type, family = classify_login(login)
        if actor_type == "AI-bot":
            mb_ai_family[family or "unknown-ai"] += 1
        elif actor_type == "non_ai_bot":
            mb_ai_family["_non_ai_bot"] += 1
        elif api_type == "Bot":
            mb_ai_family["_unclassified_bot"] += 1
            mb_unknown_bot_logins[login] += 1
        else:
            mb_human_count += 1

    # --- STEP 2: cross-check timeline MergedEvent actor ---
    agree_count = 0
    disagree_count = 0
    tl_missing_count = 0
    discrepancies: list[dict] = []
    for pr in sample:
        mb_login = pr.get("merged_by_login", "") or ""
        tl_login = ""
        for ev in pr.get("timeline_events", []):
            if ev.get("type") == "MergedEvent":
                tl_login = ev.get("actor_login", "") or ""
                break
        if not tl_login:
            tl_missing_count += 1
            continue
        if tl_login.lower() == mb_login.lower():
            agree_count += 1
        else:
            disagree_count += 1
            if len(discrepancies) < 40:
                discrepancies.append(
                    {
                        "repo": pr.get("repo"),
                        "number": pr.get("number"),
                        "merged_by": mb_login,
                        "merged_by_type": pr.get("merged_by_type", ""),
                        "timeline_actor": tl_login,
                    }
                )

    # --- STEP 3: what share of *AI-authored* PRs were merged by an AI bot? ---
    # Quick: PR is AI-authored if author_login is on allowlist OR any commit has AI trailer.
    from lib.ai_detection import detect_coauthor_ai  # local import

    ai_authored_total = 0
    ai_authored_merged_by_ai = 0
    ai_authored_merged_by_human = 0
    ai_authored_merged_by_unknown_bot = 0
    for pr in sample:
        al = pr.get("author_login", "") or ""
        author_bot_type, _ = classify_login(al)
        has_trailer = any(detect_coauthor_ai(c.get("message", "") or "") for c in pr.get("commits", []))
        if author_bot_type == "AI-bot" or has_trailer:
            ai_authored_total += 1
            mb_login = pr.get("merged_by_login", "") or ""
            mb_api_type = pr.get("merged_by_type", "") or ""
            mb_cls, _fam = classify_login(mb_login)
            if mb_cls == "AI-bot":
                ai_authored_merged_by_ai += 1
            elif mb_api_type == "Bot":
                ai_authored_merged_by_unknown_bot += 1
            else:
                ai_authored_merged_by_human += 1

    # --- Write audit report ---
    out_path = RESULTS_DIR / "merger_detection_audit.md"
    pct = lambda n, d: f"{(100*n/max(d,1)):.1f}%"

    total_bots_mb = mb_ai_family.total()
    total_api_bot = mb_type_counts.get("Bot", 0)
    pct_allowlist_of_bot_api = 0 if total_api_bot == 0 else 100 * (
        sum(v for k, v in mb_ai_family.items() if k and not k.startswith("_"))
    ) / total_api_bot

    lines = [
        "# Merger-detection audit",
        "",
        "Generated by `scripts/04_merger_audit.py`. Source: every merged PR in",
        "`data/prs/*.jsonl`, optionally sub-sampled to 1000.",
        "",
        f"- Total merged PRs in data: **{len(all_merged)}**",
        f"- Audit sample size: **{len(sample)}**",
        "",
        "## Step 1 — `merged_by.login` and `merged_by.type`",
        "",
        f"- `merged_by` missing: **{mb_missing_count}** ({pct(mb_missing_count, len(sample))})",
        "- `merged_by.type` distribution:",
    ]
    for t, n in mb_type_counts.most_common():
        lines.append(f"  - `{t or '(empty)'}`: {n} ({pct(n, len(sample))})")
    lines += [
        "",
        "- When `merged_by` is on our AI bot allowlist, family breakdown:",
    ]
    for fam, n in mb_ai_family.most_common():
        lines.append(f"  - `{fam}`: {n}")
    lines += [
        "",
        f"- Share of `type=Bot` mergers that are on our allowlist: **{pct_allowlist_of_bot_api:.1f}%**",
        f"- Unclassified bot merger logins (top 20):",
    ]
    for lg, n in mb_unknown_bot_logins.most_common(20):
        lines.append(f"  - `{lg}`: {n}")

    lines += [
        "",
        "## Step 2 — `merged_by` vs timeline `MergedEvent.actor`",
        "",
        f"- Agree: **{agree_count}** ({pct(agree_count, len(sample) - tl_missing_count)})",
        f"- Disagree: **{disagree_count}** ({pct(disagree_count, len(sample) - tl_missing_count)})",
        f"- Timeline MergedEvent missing: **{tl_missing_count}** ({pct(tl_missing_count, len(sample))})",
        "",
        "Sample disagreements:" if discrepancies else "",
    ]
    for d in discrepancies[:20]:
        lines.append(
            f"  - `{d['repo']}#{d['number']}`: merged_by=`{d['merged_by']}` (type=`{d['merged_by_type']}`), timeline=`{d['timeline_actor']}`"
        )

    lines += [
        "",
        "## Step 3 — Who merges AI-authored PRs?",
        "",
        f"- AI-authored PRs in sample: **{ai_authored_total}**",
        f"- …merged by an AI bot (allowlist): **{ai_authored_merged_by_ai}** ({pct(ai_authored_merged_by_ai, ai_authored_total)})",
        f"- …merged by an unclassified bot: **{ai_authored_merged_by_unknown_bot}** ({pct(ai_authored_merged_by_unknown_bot, ai_authored_total)})",
        f"- …merged by a human: **{ai_authored_merged_by_human}** ({pct(ai_authored_merged_by_human, ai_authored_total)})",
        "",
        "## Decision",
        "",
        "Fill in by hand after reading the numbers above.",
        "",
        "Suggested rule (per 99_instruction.md §4.3):",
        "",
        "- If `merged_by.type == 'Bot'` AND allowlist-share ≥ 80%: **accept `merged_by` as-is**.",
        "- Else if timeline agreement ≥ 95%: **accept `merged_by` but corroborate with timeline**.",
        "- Else: **fall back to 'AI reviewer who last approved' as the reviewer proxy** for the headline figure. Keep merger-type as a secondary column.",
        "",
        "---",
        "",
        f"_Allowlist ({sum(len(v) for v in AI_BOT_ACCOUNTS.values())} entries across "
        f"{len(AI_BOT_ACCOUNTS)} families):_",
        f"`{'`, `'.join(sorted(x for v in AI_BOT_ACCOUNTS.values() for x in v))}`",
        "",
        f"_Non-AI bot exclusions ({len(NON_AI_BOTS)}):_",
        f"`{'`, `'.join(sorted(NON_AI_BOTS))}`",
    ]
    out_path.write_text("\n".join(lines))
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()

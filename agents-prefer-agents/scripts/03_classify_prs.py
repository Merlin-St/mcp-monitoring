"""Classify each PR event using lib/ai_detection.

Reads data/prs/*.jsonl produced by 02_fetch_prs.py; emits:
  - data/pr_events.parquet : one row per event (commit, review, review_comment,
    issue_comment, merge, close, ready_for_review, head_ref_force_pushed, etc.)
  - data/pr_summary.parquet: one row per PR with author_type / reviewer_type /
    merger_type chosen per 99_instruction.md §5.2.

Decision rules (from 99_instruction.md §5.2):
- author_type ∈ {human, AI}: AI if PR author account is AI-bot, OR any PR
  commit has a Co-Authored-By trailer to an AI tool, OR the PR author login
  is AI-bot. (Both human-assisted-by-AI and AI-bot collapse to AI for the
  headline; AI-assisted tracked separately for robustness runs.)
- reviewer_type ∈ {human, AI}: dominant actor on the approving review;
  fallback to most-active reviewer by event count; skip "no reviews".
- merger_type ∈ {human, AI, none}: from merged_by + timeline MergedEvent
  actor; "none" if not merged.

Confidence: ``high`` if bot account or co-author trailer; ``low`` if handle-
mention-only. Primary analysis uses high-only.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

try:
    import orjson  # type: ignore[import-not-found]
    _loads = orjson.loads  # accepts bytes or str
except ImportError:
    orjson = None  # type: ignore
    def _loads(s):  # type: ignore[misc]
        if isinstance(s, bytes):
            s = s.decode("utf-8")
        return json.loads(s)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from importlib import import_module

utils = import_module("99_utils")
get_logger = utils.get_logger
parse_datetime_safe = utils.parse_datetime_safe
iso_week = utils.iso_week
DATA_DIR = utils.DATA_DIR
RESULTS_DIR = utils.RESULTS_DIR

from lib.ai_detection import classify_event, classify_login  # noqa: E402

PRS_DIR = DATA_DIR / "prs"

logger = get_logger("03_classify_prs")


def classify_role(logins_and_texts: list[tuple[str, str]]) -> tuple[str, str, str]:
    """Given list of (login, text) for events contributed by one role
    (reviewers, mergers, ...), return (type, family, confidence).
    - type in {"AI", "human", "none"}
    - family: best-guess
    - confidence: "high" if any high-confidence AI hit; "low" if only low-conf;
                  "none" for human.

    Primary classification (for the headline figure) tags as AI ONLY when there
    is at least one high-confidence AI signal: a bot account on our allowlist,
    OR a Co-Authored-By trailer. Handle-mention-only (low confidence) events
    stay as "human" — a human who writes "@claude fix this" in a review body
    is not themselves an AI reviewer.
    """
    if not logins_and_texts:
        return ("none", "none", "none")
    classes = [classify_event(lg, tx) for lg, tx in logins_and_texts]
    # Non-AI bots excluded (dependabot etc.)
    real = [c for c in classes if c.actor_type != "non_ai_bot"]
    if not real:
        return ("none", "none", "none")
    # Count HIGH-confidence AI signals only (bot account OR co-author trailer).
    ai_high = [c for c in real if c.actor_type in ("AI-bot", "AI-assisted") and c.confidence == "high"]
    ai_low = [c for c in real if c.actor_type in ("AI-bot", "AI-assisted") and c.confidence != "high"]
    human_count = sum(1 for c in real if c.actor_type == "human") + len(ai_low)
    if ai_high and len(ai_high) >= max(human_count, 1) - (human_count // 2):
        # Tie-breaking: AI wins when ai_high is >= half of non-high events (§5.2
        # conservative — biases *against* finding AI-AI self-preference).
        fam_counts = Counter(c.ai_family for c in ai_high if c.ai_family != "none")
        family = fam_counts.most_common(1)[0][0] if fam_counts else "unknown-ai"
        return ("AI", family, "high")
    if ai_high:
        # Some AI presence but dominated by humans.
        return ("human", "none", "none")
    return ("human", "none", "none")


def classify_pr(pr: dict) -> dict:
    """Return a per-PR summary row."""
    # --- AUTHOR ---
    author_login = pr.get("author_login", "")
    # Aggregate all commits' messages for trailer scan.
    commit_text = "\n".join(c.get("message", "") for c in pr.get("commits", []))
    body_text = pr.get("body", "") or ""
    author_cls = classify_event(author_login, body_text + "\n" + commit_text)

    # If any commit is authored by an AI bot, treat the PR as AI-authored.
    commit_authors = []
    for c in pr.get("commits", []):
        login = c.get("author_login", "") or c.get("committer_login", "")
        commit_authors.append((login, c.get("message", "") or ""))
    commit_classes = [classify_event(l, t) for l, t in commit_authors]
    any_bot_commit = any(cc.actor_type == "AI-bot" for cc in commit_classes)
    any_coauthor = any(cc.coauthor_hits for cc in commit_classes) or bool(author_cls.coauthor_hits)

    if author_cls.actor_type == "AI-bot" or any_bot_commit:
        author_type = "AI"
        author_family = author_cls.ai_family if author_cls.ai_family != "none" else next(
            (cc.ai_family for cc in commit_classes if cc.actor_type == "AI-bot"), "unknown-ai"
        )
        author_conf = "high"
    elif any_coauthor:
        author_type = "AI"
        fam_counts = Counter()
        for cc in commit_classes + [author_cls]:
            for k, v in cc.coauthor_hits.items():
                fam_counts[k] += v
        author_family = fam_counts.most_common(1)[0][0] if fam_counts else "unknown-ai"
        author_conf = "high"
    else:
        # No high-confidence AI authorship.
        if author_cls.actor_type == "non_ai_bot":
            author_type = "non_ai_bot"
        else:
            author_type = "human"
        author_family = "none"
        author_conf = "none"

    # --- REVIEWERS ---
    reviewer_events: list[tuple[str, str]] = []
    approving_events: list[tuple[str, str]] = []
    for r in pr.get("reviews", []):
        lg = r.get("author_login", "")
        tx = r.get("body", "") or ""
        reviewer_events.append((lg, tx))
        if (r.get("state") or "").upper() == "APPROVED":
            approving_events.append((lg, tx))
    # Primary: approving reviewers; fallback: all reviewers.
    role_events = approving_events if approving_events else reviewer_events
    reviewer_type, reviewer_family, reviewer_conf = classify_role(role_events)

    # --- MERGER ---
    merger_login = pr.get("merged_by_login", "")
    merger_type_raw = pr.get("merged_by_type", "") or ""
    merger_cls = classify_event(merger_login, "")
    # Also inspect timeline MergedEvent actor as corroboration.
    tl_merger = ""
    for ev in pr.get("timeline_events", []):
        if ev.get("type") == "MergedEvent":
            tl_merger = ev.get("actor_login", "")
            break
    merged = bool(pr.get("merged", False))
    if not merged:
        merger_type = "none"
        merger_family = "none"
    else:
        # Prefer merged_by; if merged_by is a Bot (type) and merger_login is on
        # our AI bot list, call it AI.
        if merger_cls.actor_type == "AI-bot":
            merger_type = "AI"
            merger_family = merger_cls.ai_family
        elif merger_type_raw == "Bot":
            # Unknown bot (not on allowlist); treat as non-AI bot.
            merger_type = "non_ai_bot"
            merger_family = "none"
        else:
            merger_type = "human"
            merger_family = "none"

    # --- DERIVED TIMING ---
    opened = parse_datetime_safe(pr.get("created_at", ""))
    merged_at = parse_datetime_safe(pr.get("merged_at", ""))
    closed_at = parse_datetime_safe(pr.get("closed_at", ""))
    days_to_merge = None
    if merged_at and opened:
        days_to_merge = (merged_at - opened).total_seconds() / 86400.0
    within_30d = bool(merged_at and opened and days_to_merge is not None and days_to_merge <= 30.0)
    opened_week = iso_week(opened) if opened else ""

    # --- DERIVED COUNTS ---
    ai_bot_events = sum(
        1 for c in commit_classes if c.actor_type == "AI-bot"
    )
    coauthor_events = sum(1 for c in commit_classes if c.coauthor_hits)

    return {
        "repo": pr.get("repo", ""),
        "number": pr.get("number", 0),
        "title_len": len(pr.get("title", "")),
        "body_len": len(pr.get("body", "") or ""),
        "author_login": author_login,
        "author_type": author_type,
        "author_family": author_family,
        "author_confidence": author_conf,
        "reviewer_type": reviewer_type,
        "reviewer_family": reviewer_family,
        "reviewer_confidence": reviewer_conf,
        "num_approving_reviews": len(approving_events),
        "num_reviews_total": len(reviewer_events),
        "merger_login": merger_login,
        "merger_api_type": merger_type_raw,
        "merger_type": merger_type,
        "merger_family": merger_family,
        "timeline_merger_login": tl_merger,
        "created_at": pr.get("created_at", ""),
        "updated_at": pr.get("updated_at", ""),
        "closed_at": pr.get("closed_at", ""),
        "merged_at": pr.get("merged_at", ""),
        "merged": merged,
        "merged_within_30d": within_30d,
        "days_to_merge": days_to_merge,
        "opened_week": opened_week,
        "additions": pr.get("additions", 0),
        "deletions": pr.get("deletions", 0),
        "changed_files": pr.get("changed_files", 0),
        "commits_in_pr": len(pr.get("commits", [])),
        "ai_bot_commits_in_pr": ai_bot_events,
        "coauthored_commits_in_pr": coauthor_events,
        "issue_comment_count": len(pr.get("issue_comments", [])),
        "review_comment_count": len(pr.get("review_comments", [])),
    }


def flatten_events(pr: dict) -> list[dict]:
    """Emit one row per attributed event for chain-length analysis (§5.1)."""
    rows: list[dict] = []
    base = {"repo": pr.get("repo", ""), "number": pr.get("number", 0)}

    # PR open event
    opened_cls = classify_event(pr.get("author_login", ""), pr.get("body", "") or "")
    rows.append({**base, "event_type": "pr_open", "actor_login": pr.get("author_login", ""),
                 "actor_type": opened_cls.actor_type, "ai_family": opened_cls.ai_family,
                 "confidence": opened_cls.confidence, "timestamp": pr.get("created_at", ""),
                 "payload_len": len(pr.get("body", "") or "")})

    for c in pr.get("commits", []):
        lg = c.get("author_login", "") or c.get("committer_login", "")
        cls = classify_event(lg, c.get("message", "") or "")
        rows.append({**base, "event_type": "commit", "actor_login": lg,
                     "actor_type": cls.actor_type, "ai_family": cls.ai_family,
                     "confidence": cls.confidence, "timestamp": c.get("authored_at", ""),
                     "payload_len": len(c.get("message", "") or "")})

    for r in pr.get("reviews", []):
        cls = classify_event(r.get("author_login", ""), r.get("body", "") or "")
        rows.append({**base, "event_type": f"review_{(r.get('state') or '').lower()}",
                     "actor_login": r.get("author_login", ""),
                     "actor_type": cls.actor_type, "ai_family": cls.ai_family,
                     "confidence": cls.confidence, "timestamp": r.get("submitted_at", ""),
                     "payload_len": len(r.get("body", "") or "")})

    for c in pr.get("review_comments", []):
        cls = classify_event(c.get("author_login", ""), c.get("body", "") or "")
        rows.append({**base, "event_type": "review_comment", "actor_login": c.get("author_login", ""),
                     "actor_type": cls.actor_type, "ai_family": cls.ai_family,
                     "confidence": cls.confidence, "timestamp": c.get("created_at", ""),
                     "payload_len": len(c.get("body", "") or "")})

    for c in pr.get("issue_comments", []):
        cls = classify_event(c.get("author_login", ""), c.get("body", "") or "")
        rows.append({**base, "event_type": "issue_comment", "actor_login": c.get("author_login", ""),
                     "actor_type": cls.actor_type, "ai_family": cls.ai_family,
                     "confidence": cls.confidence, "timestamp": c.get("created_at", ""),
                     "payload_len": len(c.get("body", "") or "")})

    for ev in pr.get("timeline_events", []):
        lg = ev.get("actor_login", "")
        cls = classify_event(lg, "")
        rows.append({**base, "event_type": f"tl_{ev.get('type','')}", "actor_login": lg,
                     "actor_type": cls.actor_type, "ai_family": cls.ai_family,
                     "confidence": cls.confidence, "timestamp": ev.get("created_at", ""),
                     "payload_len": 0})

    # Sort by timestamp (chain-length analysis needs this).
    def _ts(r):
        dt = parse_datetime_safe(r.get("timestamp") or "")
        return dt or parse_datetime_safe("1970-01-01T00:00:00Z")
    rows.sort(key=_ts)
    for i, r in enumerate(rows):
        r["event_idx"] = i
    return rows


def _process_jsonl(path_str: str) -> tuple[list[dict], list[dict], int]:
    """Worker: parse one JSONL file, return (summaries, events, n_prs).

    Defined at module level so it's picklable for multiprocessing.Pool. Each
    worker has its own copy of classify_pr / flatten_events / classify_event
    via module import (fork-inherited on Linux; re-imported on spawn).
    """
    summaries: list[dict] = []
    events: list[dict] = []
    n = 0
    with open(path_str, "rb") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            pr = _loads(raw)
            summaries.append(classify_pr(pr))
            events.extend(flatten_events(pr))
            n += 1
    return summaries, events, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-repos", type=int, default=0)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of worker processes (default: cpu_count-1).",
    )
    args = parser.parse_args()

    files = sorted(PRS_DIR.glob("*.jsonl"))
    if args.limit_repos:
        files = files[: args.limit_repos]
    if not files:
        logger.error("No data/prs/*.jsonl files found. Run 02_fetch_prs.py first.")
        sys.exit(1)
    logger.info("Found %d PR jsonl files. Using %d workers.", len(files), args.workers)

    summaries: list[dict] = []
    events: list[dict] = []
    totals = Counter()
    t0 = time.time()

    paths = [str(p) for p in files]
    if args.workers <= 1:
        # Serial path (useful for debugging + tests).
        for i, ps in enumerate(paths):
            s, e, n = _process_jsonl(ps)
            summaries.extend(s)
            events.extend(e)
            totals["prs"] += n
            totals["repos"] += 1
            if (i + 1) % 25 == 0:
                logger.info("[%d/%d] %s  prs_so_far=%d events_so_far=%d",
                            i + 1, len(files), Path(ps).name, totals["prs"], len(events))
    else:
        with mp.Pool(processes=args.workers) as pool:
            for i, (s, e, n) in enumerate(pool.imap_unordered(_process_jsonl, paths, chunksize=4)):
                summaries.extend(s)
                events.extend(e)
                totals["prs"] += n
                totals["repos"] += 1
                if (i + 1) % 50 == 0:
                    logger.info("[%d/%d] prs_so_far=%d events_so_far=%d  (%.0f files/min)",
                                i + 1, len(files), totals["prs"], len(events),
                                (i + 1) / max((time.time() - t0) / 60, 1e-3))

    summary_df = pd.DataFrame(summaries)
    events_df = pd.DataFrame(events)
    logger.info("Writing pr_summary.parquet (%d rows) and pr_events.parquet (%d rows)...",
                len(summary_df), len(events_df))
    summary_df.to_parquet(DATA_DIR / "pr_summary.parquet", index=False)
    events_df.to_parquet(DATA_DIR / "pr_events.parquet", index=False)

    # Quick sanity stats.
    stats = {
        "repos": int(totals["repos"]),
        "prs": int(totals["prs"]),
        "events": int(len(events_df)),
        "author_type_counts": summary_df["author_type"].value_counts().to_dict(),
        "reviewer_type_counts": summary_df["reviewer_type"].value_counts().to_dict(),
        "merger_type_counts": summary_df["merger_type"].value_counts().to_dict(),
        "merged_counts": summary_df["merged"].value_counts().to_dict(),
        "generated_at": utils.now_iso(),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(RESULTS_DIR / "phase3_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info("=== PHASE 3 DONE ===")
    logger.info(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()

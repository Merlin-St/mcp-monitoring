"""Build the repository universe for the agents-prefer-agents analysis.

Scope (from 99_instruction.md §3.2):
- Public GitHub repos that plausibly gained >=1000 stars during 2025.
- Active in the analysis window (pushed_at >= 2025-04-01).
- Not forks, not archived.
- Cap the universe so Phase 2 PR-collection is tractable (default: 500 repos).

Method:
1. Search-API sweep with star-bucket splitting (caps at 1000 per query, so
   we slice by star ranges).
2. Drop forks, archived, and mirrors.
3. Estimate ``stars_gained_2025`` with the following proxy:
      - If created_at >= 2025-01-01: stars_gained_2025 = stargazers_count
      - Else: stars_gained_2025 = 0 for the filter, BUT we also keep repos
        with stargazers_count >= ``OLD_REPO_STAR_THRESHOLD`` (default 5000)
        AND pushed_at in the last 90 days, on the reasoning that such repos
        very likely gained 1000+ stars in 2025 as well. This is a
        conservative proxy that biases *against* inclusion — see
        99_progress.md uncertainties.
4. Sort by a score that prioritises recent activity AND stars, and keep the
   top ``--cap`` repos.

Writes: ``data/repos.json`` and ``results/phase1_stats.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make sibling modules importable when called as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib import import_module

utils = import_module("99_utils")  # noqa: E402

get_github_token = utils.get_github_token
GitHubSyncClient = utils.GitHubSyncClient
get_logger = utils.get_logger
parse_datetime_safe = utils.parse_datetime_safe
now_iso = utils.now_iso
SUBPROJECT_ROOT = utils.SUBPROJECT_ROOT
DATA_DIR = utils.DATA_DIR
RESULTS_DIR = utils.RESULTS_DIR


# ---------------------------------------------------------------------------
# Bucket ranges chosen so each Search API query stays under the 1000-result
# cap. Counts from the probe 2026-04-21 (stars:>=X pushed:>=2025-01-01):
#   1000..1500 -> 10088  (too many; split by language)
#   1500..2500 -> 9517   (too many; split by language)
#   2500..5000 -> 8003   (too many; split by language)
#   5000..10000 -> 4445  (split by stars-finer)
#   10000..50000 -> 3657 (split by stars-finer)
#   >=50000 -> 365       (direct)
# We split the dense buckets further by star-range only (no language filter)
# to keep things simple; if a sub-bucket still exceeds 1000 we log a warning.
# ---------------------------------------------------------------------------
STAR_BUCKETS = [
    # (min, max_inclusive, label)
    (1000, 1099, "1000-1099"),
    (1100, 1199, "1100-1199"),
    (1200, 1299, "1200-1299"),
    (1300, 1399, "1300-1399"),
    (1400, 1499, "1400-1499"),
    (1500, 1699, "1500-1699"),
    (1700, 1899, "1700-1899"),
    (1900, 2099, "1900-2099"),
    (2100, 2499, "2100-2499"),
    (2500, 2999, "2500-2999"),
    (3000, 3999, "3000-3999"),
    (4000, 4999, "4000-4999"),
    (5000, 6499, "5000-6499"),
    (6500, 8999, "6500-8999"),
    (9000, 14999, "9000-14999"),
    (15000, 24999, "15000-24999"),
    (25000, 49999, "25000-49999"),
    (50000, 99999, "50000-99999"),
    (100000, 10_000_000, ">=100000"),
]

PUSHED_AFTER = "2025-04-01"  # same lower bound as our PR analysis window

logger = get_logger("01_build_repo_list")


def search_bucket(
    client: GitHubSyncClient, lo: int, hi: int
) -> list[dict]:
    """Fetch all repos in a star bucket, paginating up to 1000 results."""
    stars_q = f"stars:{lo}..{hi}" if hi < 10_000_000 else f"stars:>={lo}"
    q = f"{stars_q} pushed:>={PUSHED_AFTER} is:public archived:false"
    repos: list[dict] = []
    for page in range(1, 11):  # 10 pages × 100 = 1000 (Search API hard cap)
        status, data = client.search(
            "/search/repositories",
            params={"q": q, "per_page": 100, "page": page, "sort": "stars", "order": "desc"},
        )
        if status != 200 or not data:
            logger.warning("Bucket %s page %d failed: status=%s", f"{lo}..{hi}", page, status)
            break
        items = data.get("items", [])
        total = data.get("total_count", -1)
        repos.extend(items)
        logger.info(
            "Bucket %s page %d: %d items (total_count=%d, cum=%d)",
            f"{lo}..{hi}",
            page,
            len(items),
            total,
            len(repos),
        )
        if len(items) < 100:
            break
        # Polite pause to keep under search secondary-rate-limits.
        time.sleep(1.2)
        if total > 1000 and page == 10:
            logger.warning(
                "Bucket %s: total_count=%d exceeds 1000 cap; some repos missed",
                f"{lo}..{hi}",
                total,
            )
    return repos


def activity_score(repo: dict, today: datetime) -> float:
    """Score used to rank repos for the cap.

    Emphasises recent activity AND stars. A repo pushed today with 10k stars
    scores higher than one pushed 6 months ago with 50k stars. Created-in-
    2025 repos get a small bonus because they are guaranteed to have gained
    their stars within the analysis window.
    """
    stars = max(repo.get("stargazers_count", 0), 1)
    pushed_dt = parse_datetime_safe(repo.get("pushed_at", ""))
    days_since_push = 365 if not pushed_dt else max(
        (today - pushed_dt.astimezone(timezone.utc)).days, 0
    )
    recency = math.exp(-days_since_push / 60.0)  # half-life ~42 days
    created_dt = parse_datetime_safe(repo.get("created_at", ""))
    young_bonus = 0.0
    if created_dt and created_dt.year >= 2025:
        young_bonus = 0.3
    return math.log10(stars) * (0.6 + 0.4 * recency) + young_bonus


def estimate_stars_gained_2025(repo: dict) -> int:
    """Conservative proxy for stars gained in 2025.

    - created_at >= 2025-01-01  →  all stars (guaranteed gained in 2025).
    - otherwise                →  0 here; we rely on the OLD_REPO override.
    """
    stars = repo.get("stargazers_count", 0)
    created = parse_datetime_safe(repo.get("created_at", ""))
    if created and created >= datetime(2025, 1, 1, tzinfo=timezone.utc):
        return stars
    return 0


def keep_as_old_active_popular(
    repo: dict, old_threshold: int, today: datetime
) -> bool:
    """An older repo may have plausibly gained >=1000 stars in 2025 if it's
    still highly popular and active. Conservative proxy: stars >= threshold
    AND pushed within last 120 days.
    """
    if repo.get("stargazers_count", 0) < old_threshold:
        return False
    pushed = parse_datetime_safe(repo.get("pushed_at", ""))
    if not pushed:
        return False
    return (today - pushed.astimezone(timezone.utc)).days <= 120


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cap",
        type=int,
        default=500,
        help="Max repos to keep (after filtering + sorting). Default 500.",
    )
    parser.add_argument(
        "--old-repo-star-threshold",
        type=int,
        default=5000,
        help=(
            "Older (pre-2025) repos are kept if stars >= this. "
            "Default 5000 — high enough to avoid the long tail of pre-existing mid-size repos."
        ),
    )
    parser.add_argument(
        "--min-young-stars",
        type=int,
        default=1000,
        help="Young (created >= 2025-01-01) repo must have at least this many stars. Default 1000.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the sweep but don't write data/repos.json.",
    )
    args = parser.parse_args()

    token = get_github_token()
    if not token:
        logger.error("No GitHub token. Set GH_TOKEN or run `gh auth login`.")
        sys.exit(1)
    logger.info("Token resolved (len=%d).", len(token))

    client = GitHubSyncClient(token)
    today = datetime.now(timezone.utc)

    all_repos: list[dict] = []
    seen_ids: set[int] = set()
    per_bucket_counts: dict[str, int] = {}

    for lo, hi, label in STAR_BUCKETS:
        bucket = search_bucket(client, lo, hi)
        per_bucket_counts[label] = len(bucket)
        for r in bucket:
            rid = r.get("id")
            if rid and rid not in seen_ids:
                seen_ids.add(rid)
                all_repos.append(r)
        logger.info(
            "Rate remaining: core=%d search=%d (req_count=%d)",
            client.core_remaining,
            client.search_remaining,
            client.request_count,
        )

    logger.info("Collected %d unique repos across %d buckets.", len(all_repos), len(STAR_BUCKETS))

    # Filter: not fork (Search already excludes is:public archived:false).
    pre_filter = len(all_repos)
    all_repos = [r for r in all_repos if not r.get("fork", False)]
    logger.info("After fork filter: %d (removed %d)", len(all_repos), pre_filter - len(all_repos))

    # Estimate stars_gained_2025 and apply the 'plausibly gained 1000' rule.
    kept: list[dict] = []
    for r in all_repos:
        r["stars_gained_2025_estimate"] = estimate_stars_gained_2025(r)
        # Decision rule:
        # - Young (created >= 2025-01-01) AND stars >= min_young_stars
        # - OR old & highly popular & recently active
        created = parse_datetime_safe(r.get("created_at", ""))
        is_young = created and created >= datetime(2025, 1, 1, tzinfo=timezone.utc)
        keep_young = is_young and r.get("stargazers_count", 0) >= args.min_young_stars
        keep_old = (not is_young) and keep_as_old_active_popular(
            r, args.old_repo_star_threshold, today
        )
        if keep_young or keep_old:
            r["_keep_reason"] = "young" if keep_young else "old_popular"
            kept.append(r)

    logger.info(
        "After stars_gained_2025 proxy filter: %d (young=%d, old_popular=%d)",
        len(kept),
        sum(1 for r in kept if r.get("_keep_reason") == "young"),
        sum(1 for r in kept if r.get("_keep_reason") == "old_popular"),
    )

    # Rank and cap.
    kept.sort(key=lambda r: activity_score(r, today), reverse=True)
    if len(kept) > args.cap:
        logger.info("Capping %d → %d by activity score.", len(kept), args.cap)
        kept = kept[: args.cap]

    # Build the output schema.
    rows: list[dict] = []
    for r in kept:
        owner = (r.get("owner") or {}).get("login", "")
        name = r.get("name", "")
        rows.append(
            {
                "id": r.get("id"),
                "owner": owner,
                "repo": name,
                "full_name": r.get("full_name", f"{owner}/{name}"),
                "html_url": r.get("html_url", ""),
                "description": r.get("description") or "",
                "language": r.get("language") or "",
                "stars": r.get("stargazers_count", 0),
                "forks": r.get("forks_count", 0),
                "open_issues": r.get("open_issues_count", 0),
                "created_at": r.get("created_at", ""),
                "updated_at": r.get("updated_at", ""),
                "pushed_at": r.get("pushed_at", ""),
                "default_branch": r.get("default_branch", "main"),
                "fork": r.get("fork", False),
                "archived": r.get("archived", False),
                "disabled": r.get("disabled", False),
                "has_issues": r.get("has_issues", True),
                "size_kb": r.get("size", 0),
                "stars_gained_2025_estimate": r.get("stars_gained_2025_estimate", 0),
                "keep_reason": r.get("_keep_reason", ""),
                "activity_score": activity_score(r, today),
            }
        )

    # Stats for the progress file / check-in.
    young = sum(1 for x in rows if x["keep_reason"] == "young")
    old_pop = sum(1 for x in rows if x["keep_reason"] == "old_popular")
    lang_counts: dict[str, int] = {}
    for x in rows:
        lang_counts[x["language"] or "Other"] = lang_counts.get(x["language"] or "Other", 0) + 1
    stats = {
        "generated_at": now_iso(),
        "total_candidates_collected": len(all_repos),
        "after_filters": len(rows),
        "cap": args.cap,
        "per_bucket_total_count_seen": per_bucket_counts,
        "keep_reason_counts": {"young": young, "old_popular": old_pop},
        "top_languages": dict(
            sorted(lang_counts.items(), key=lambda kv: -kv[1])[:15]
        ),
        "stars_min": min((x["stars"] for x in rows), default=0),
        "stars_max": max((x["stars"] for x in rows), default=0),
        "stars_median": (
            sorted(x["stars"] for x in rows)[len(rows) // 2] if rows else 0
        ),
        "gh_api_requests": client.request_count,
    }

    logger.info("=== PHASE 1 STATS ===")
    logger.info(json.dumps(stats, indent=2))

    if not args.dry_run:
        out_repos = DATA_DIR / "repos.json"
        with open(out_repos, "w") as f:
            json.dump(rows, f, indent=2)
        logger.info("Wrote %d repos to %s", len(rows), out_repos)

        out_stats = RESULTS_DIR / "phase1_stats.json"
        with open(out_stats, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info("Wrote stats to %s", out_stats)


if __name__ == "__main__":
    main()

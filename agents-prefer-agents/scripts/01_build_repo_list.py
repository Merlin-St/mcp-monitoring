"""Build the repository universe (v2) from OpenSSF criticality scores.

This is the v2 selection pipeline. It replaces the ad-hoc star-bucket sweep
in ``scripts/old_01_build_repo_list.py``. v1 artefacts are preserved in
``data/old_phase1/``.

## Method (replication-friendly)

1. Read the cached OpenSSF criticality_score CSV (downloaded by
   ``scripts/01a_download_criticality.py``; default snapshot **2025.07.25**).
   The score is the OpenSSF "default_score" — a weighted arithmetic mean over
   eight signals (created_since, updated_since, contributor_count, org_count,
   commit_frequency, recent_release_count, updated_issues_count, comment
   frequency, github_mention_count, plus deps.dev dependent_count if the
   variant is ``all_w_depsdev.csv``). See:
       https://github.com/ossf/criticality_score#algorithm
       https://opensource.googleblog.com/2020/12/finding-critical-open-source-projects.html

2. Keep rows where:
   - ``repo.url`` starts with ``https://github.com/``
   - ``default_score`` is non-empty and parses as a float in (0, 1]

3. Sort by ``default_score`` descending. Tie-break by ``repo.url``
   alphabetically for stable runs.

4. Take the top ``--candidate-cap`` (default 15{,}000) candidates. We
   over-sample because the enrichment step in (5) drops roughly 10–25%:
   forks, archived, deleted/private since the snapshot, and repos with no
   activity in the analysis window.

5. Enrich each candidate via ``GET /repos/{owner}/{repo}`` (async, bounded
   concurrency 8). Drop:
   - HTTP 404 (deleted, renamed-to-private, etc.)
   - ``fork == true``
   - ``archived == true``
   - ``disabled == true``
   - ``pushed_at`` strictly before ``--window-start`` (default 2025-04-01)

6. From the survivors, keep the top ``--final-cap`` (default 10{,}000) by
   criticality score. Write ``data/repos.json``.

## Output schema (compatible with v1; the only fields downstream uses are
   ``owner``, ``repo``, ``full_name``, ``stars``)

    {
      "id": int,                              # GitHub numeric repo id
      "owner": str,
      "repo": str,
      "full_name": "owner/repo",
      "html_url": str,
      "description": str | null,
      "language": str | null,
      "stars": int,
      "forks": int,
      "open_issues": int,
      "created_at": str,
      "updated_at": str,
      "pushed_at": str,
      "default_branch": str,
      "fork": false,
      "archived": false,
      "disabled": false,
      "has_issues": bool,
      "size_kb": int,
      "criticality_score": float,             # OSSF default_score in (0,1]
      "criticality_rank": int,                # 1 = most critical
      "criticality_snapshot_date": "YYYY.MM.DD",
      "criticality_signals": {                # raw OSSF signal columns
        "contributor_count": int,
        "org_count": int,
        "commit_frequency": float,
        "recent_release_count": int,
        "updated_issues_count": int,
        "closed_issues_count": int,
        "issue_comment_frequency": float,
        "github_mention_count": int,
        "depsdev_dependent_count": int | null
      },
      "enrichment_at": "ISO-UTC timestamp"
    }

## Stats

Writes ``results/phase1_stats.json`` with row counts at every filtering step.

## Run

    # First, ensure the CSV is cached:
    python scripts/01a_download_criticality.py

    # Then build the universe:
    python scripts/01_build_repo_list.py --final-cap 10000

    # Smoke test:
    python scripts/01_build_repo_list.py --final-cap 50 --candidate-cap 75

Runtime: approximately 5–15 min for 15{,}000 candidates (rate-limited GH API,
8-way concurrency, with the standard 5{,}000/hr core quota).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
utils = import_module("99_utils")

GitHubAsyncClient = utils.GitHubAsyncClient
get_github_token = utils.get_github_token
get_logger = utils.get_logger
extract_owner_repo = utils.extract_owner_repo
parse_datetime_safe = utils.parse_datetime_safe
now_iso = utils.now_iso
DATA_DIR = utils.DATA_DIR
RESULTS_DIR = utils.RESULTS_DIR

CRITICALITY_DIR = DATA_DIR / "criticality"


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------
def _resolve_csv(snapshot_date: str, variant: str) -> Path:
    name = f"ossf-criticality-{snapshot_date}-{variant}"
    p = CRITICALITY_DIR / name
    if not p.exists():
        raise FileNotFoundError(
            f"Cached criticality CSV not found at {p}. "
            f"Run scripts/01a_download_criticality.py --snapshot-prefix {snapshot_date}/<HHMMSS> "
            f"--variant {variant}"
        )
    return p


def _parse_float(s: str) -> Optional[float]:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _parse_int(s: str) -> Optional[int]:
    if s is None or s == "":
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def read_candidates(csv_path: Path, candidate_cap: int, log) -> list[dict]:
    """Stream the OSSF CSV, keep GitHub-URL rows with a numeric default_score,
    return the top ``candidate_cap`` sorted by score desc."""
    csv.field_size_limit(sys.maxsize)
    log.info("Streaming %s ...", csv_path)
    rows = []
    skipped_no_score = 0
    skipped_non_github = 0
    total = 0
    with open(csv_path, newline="") as fh:
        rdr = csv.DictReader(fh)
        for total, r in enumerate(rdr, 1):
            score = _parse_float(r.get("default_score", ""))
            url = r.get("repo.url", "")
            if score is None:
                skipped_no_score += 1
                continue
            if not url.startswith("https://github.com/"):
                skipped_non_github += 1
                continue
            rows.append((score, url, r))
            if total % 100_000 == 0:
                log.info("  ... %d rows scanned, %d kept so far", total, len(rows))
    log.info(
        "Scan complete. total=%d  with_score=%d  github=%d  no_score=%d  non_github=%d",
        total,
        len(rows) + skipped_no_score,
        len(rows),
        skipped_no_score,
        skipped_non_github,
    )
    # Sort by score desc, then URL asc for stable tie-breaks.
    rows.sort(key=lambda x: (-x[0], x[1]))
    keep = rows[:candidate_cap]
    log.info(
        "Top-%d candidate range: max=%.5f  min=%.5f",
        len(keep),
        keep[0][0],
        keep[-1][0],
    )
    out = []
    for rank, (score, url, r) in enumerate(keep, 1):
        out.append(
            {
                "criticality_score": score,
                "criticality_rank": rank,
                "url": url,
                "csv_row": r,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Enrichment via GitHub REST
# ---------------------------------------------------------------------------
async def _enrich_one(client: GitHubAsyncClient, candidate: dict) -> dict:
    parsed = extract_owner_repo(candidate["url"])
    if not parsed:
        return {"_status": "bad_url", **candidate}
    owner, repo = parsed
    status, data = await client.get(f"/repos/{owner}/{repo}")
    if status != 200 or data is None:
        return {"_status": f"http_{status}", **candidate}
    return {"_status": "ok", "_data": data, **candidate}


async def _enrich_all(candidates: list[dict], log) -> list[dict]:
    token = get_github_token()
    if not token:
        log.error("No GitHub token available (set GH_TOKEN or `gh auth login`).")
        sys.exit(2)
    client = GitHubAsyncClient(token, max_concurrent=8)
    out: list[dict] = []
    t0 = time.time()
    try:
        # Schedule and await in chunks so progress logs are useful.
        chunk = 200
        for i in range(0, len(candidates), chunk):
            batch = candidates[i : i + chunk]
            results = await asyncio.gather(
                *[_enrich_one(client, c) for c in batch], return_exceptions=False
            )
            out.extend(results)
            elapsed = time.time() - t0
            log.info(
                "Enrichment progress: %d/%d  rate=%.1f rps  rl_remaining=%d",
                len(out),
                len(candidates),
                len(out) / max(elapsed, 1e-3),
                client.core_remaining,
            )
    finally:
        await client.close()
    return out


# ---------------------------------------------------------------------------
# Filtering + schema mapping
# ---------------------------------------------------------------------------
def _row_to_repo(enriched: dict, snapshot_date: str) -> Optional[dict]:
    if enriched.get("_status") != "ok":
        return None
    d = enriched["_data"]
    csv_row = enriched["csv_row"]
    return {
        "id": d.get("id"),
        "owner": d["owner"]["login"],
        "repo": d["name"],
        "full_name": d["full_name"],
        "html_url": d.get("html_url", enriched["url"]),
        "description": d.get("description"),
        "language": d.get("language"),
        "stars": d.get("stargazers_count", 0),
        "forks": d.get("forks_count", 0),
        "open_issues": d.get("open_issues_count", 0),
        "created_at": d.get("created_at"),
        "updated_at": d.get("updated_at"),
        "pushed_at": d.get("pushed_at"),
        "default_branch": d.get("default_branch", "main"),
        "fork": bool(d.get("fork", False)),
        "archived": bool(d.get("archived", False)),
        "disabled": bool(d.get("disabled", False)),
        "has_issues": bool(d.get("has_issues", True)),
        "size_kb": d.get("size", 0),
        "criticality_score": enriched["criticality_score"],
        "criticality_rank": enriched["criticality_rank"],
        "criticality_snapshot_date": snapshot_date,
        "criticality_signals": {
            "contributor_count": _parse_int(csv_row.get("legacy.contributor_count", "")),
            "org_count": _parse_int(csv_row.get("legacy.org_count", "")),
            "commit_frequency": _parse_float(csv_row.get("legacy.commit_frequency", "")),
            "recent_release_count": _parse_int(csv_row.get("legacy.recent_release_count", "")),
            "updated_issues_count": _parse_int(csv_row.get("legacy.updated_issues_count", "")),
            "closed_issues_count": _parse_int(csv_row.get("legacy.closed_issues_count", "")),
            "issue_comment_frequency": _parse_float(csv_row.get("legacy.issue_comment_frequency", "")),
            "github_mention_count": _parse_int(csv_row.get("legacy.github_mention_count", "")),
            "depsdev_dependent_count": _parse_int(csv_row.get("depsdev.dependent_count", "")),
        },
        "enrichment_at": now_iso(),
    }


def _activity_filter(repo: dict, window_start: datetime) -> tuple[bool, str]:
    if repo.get("fork"):
        return False, "fork"
    if repo.get("archived"):
        return False, "archived"
    if repo.get("disabled"):
        return False, "disabled"
    pushed_at = parse_datetime_safe(repo.get("pushed_at", ""))
    if pushed_at is None:
        return False, "no_pushed_at"
    if pushed_at < window_start:
        return False, "stale"
    return True, "ok"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot-date", default="2025.07.25")
    ap.add_argument("--variant", default="all.csv", choices=["all.csv", "all_w_depsdev.csv"])
    ap.add_argument(
        "--candidate-cap",
        type=int,
        default=15000,
        help="Pre-enrichment cap by criticality score. Over-sample to absorb drops.",
    )
    ap.add_argument(
        "--final-cap",
        type=int,
        default=10000,
        help="Final repo count after enrichment + filters.",
    )
    ap.add_argument(
        "--window-start",
        default="2025-04-01",
        help="Drop repos with pushed_at strictly before this date.",
    )
    ap.add_argument(
        "--out",
        default=str(DATA_DIR / "repos.json"),
        help="Output path for the repo list.",
    )
    ap.add_argument(
        "--stats-out",
        default=str(RESULTS_DIR / "phase1_stats.json"),
        help="Output path for filtering stats.",
    )
    args = ap.parse_args()

    log = get_logger("01_build_repo_list")
    csv_path = _resolve_csv(args.snapshot_date, args.variant)
    window_start = datetime.fromisoformat(args.window_start).replace(tzinfo=timezone.utc)

    if args.candidate_cap < args.final_cap:
        log.warning(
            "candidate-cap (%d) < final-cap (%d); raising candidate-cap to %d.",
            args.candidate_cap,
            args.final_cap,
            int(args.final_cap * 1.5),
        )
        args.candidate_cap = int(args.final_cap * 1.5)

    log.info("=== Phase 1 v2 (criticality-based) ===")
    log.info("snapshot=%s variant=%s candidate-cap=%d final-cap=%d window_start=%s",
             args.snapshot_date, args.variant, args.candidate_cap, args.final_cap, args.window_start)

    # Step 1: Read CSV, top candidates by score.
    candidates = read_candidates(csv_path, args.candidate_cap, log)

    # Step 2: Enrich via GitHub REST.
    log.info("Enriching %d candidates via GitHub /repos endpoint...", len(candidates))
    enriched = asyncio.run(_enrich_all(candidates, log))

    status_counts: dict[str, int] = {}
    for e in enriched:
        status_counts[e.get("_status", "unknown")] = status_counts.get(e.get("_status", "unknown"), 0) + 1
    log.info("Enrichment status counts: %s", status_counts)

    # Step 3: Map to schema + apply activity filter.
    repos = []
    drop_reasons: dict[str, int] = {}
    for e in enriched:
        r = _row_to_repo(e, args.snapshot_date)
        if r is None:
            drop_reasons["non_ok_status"] = drop_reasons.get("non_ok_status", 0) + 1
            continue
        keep, reason = _activity_filter(r, window_start)
        if not keep:
            drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
            continue
        repos.append(r)
    log.info("After activity filter: %d kept; drops=%s", len(repos), drop_reasons)

    # Step 4: Final cap by criticality score (lowest rank = most critical).
    repos.sort(key=lambda x: x["criticality_rank"])
    repos = repos[: args.final_cap]
    log.info("Final repo count: %d", len(repos))
    if repos:
        log.info(
            "Score range: max=%.5f  min=%.5f",
            repos[0]["criticality_score"],
            repos[-1]["criticality_score"],
        )
        log.info(
            "Star range: min=%d  median=%d  max=%d",
            min(r["stars"] for r in repos),
            sorted(r["stars"] for r in repos)[len(repos) // 2],
            max(r["stars"] for r in repos),
        )

    # Write outputs.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(repos, indent=2) + "\n")
    log.info("Wrote %d repos to %s", len(repos), out_path)

    stats = {
        "method": "ossf_criticality_score_v2",
        "snapshot_date": args.snapshot_date,
        "variant": args.variant,
        "csv_path": str(csv_path),
        "candidate_cap": args.candidate_cap,
        "final_cap": args.final_cap,
        "window_start": args.window_start,
        "candidates_loaded": len(candidates),
        "enrichment_status_counts": status_counts,
        "drop_reasons": drop_reasons,
        "final_repo_count": len(repos),
        "score_max": repos[0]["criticality_score"] if repos else None,
        "score_min": repos[-1]["criticality_score"] if repos else None,
        "star_min": min((r["stars"] for r in repos), default=None),
        "star_median": (sorted(r["stars"] for r in repos)[len(repos) // 2] if repos else None),
        "star_max": max((r["stars"] for r in repos), default=None),
        "language_top10": _top_n([r.get("language") for r in repos], 10),
        "generated_at": now_iso(),
    }
    Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats_out).write_text(json.dumps(stats, indent=2) + "\n")
    log.info("Wrote stats to %s", args.stats_out)

    return 0


def _top_n(items, n):
    counts: dict[Any, int] = {}
    for it in items:
        counts[it] = counts.get(it, 0) + 1
    return sorted(counts.items(), key=lambda x: -x[1])[:n]


if __name__ == "__main__":
    raise SystemExit(main())

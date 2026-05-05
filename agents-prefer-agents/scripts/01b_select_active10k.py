"""Select the top 10,000 PR-active critical repositories.

Reads the extended ``data/repos.json`` (after Phase 1 with --candidate-cap 14000)
and ``data/prs/_repo_pr_counts.json`` (after Phase 2). Filters to repos with
``total_in_window > 0`` (PR-active) and keeps the top 10,000 by
``criticality_rank``.

Outputs:
  data/repos.preselect.json  -- snapshot of the extended pre-truncation list
  data/repos.json            -- replaced with the top-10k PR-active subset
  data/prs_excluded/         -- moved JSONL files for repos not in the top-10k
                                (so 03_classify_prs.py's glob('*.jsonl') sees
                                only the selected universe)

Also injects ``pr_active_repo_count`` and ``pr_inactive_repo_count`` into
``results/phase1_stats.json`` so 08_fill_paper.py can fill the placeholders.

Run after both Phase 1 and Phase 2 finish for the extended set.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

utils = import_module("99_utils")
DATA_DIR = utils.DATA_DIR
RESULTS_DIR = utils.RESULTS_DIR
get_logger = utils.get_logger

logger = get_logger("01b_select_active10k")

REPOS_JSON = DATA_DIR / "repos.json"
REPOS_PRESELECT = DATA_DIR / "repos.preselect.json"
PRS_DIR = DATA_DIR / "prs"
COUNTS_FILE = PRS_DIR / "_repo_pr_counts.json"
EXCLUDED_DIR = DATA_DIR / "prs_excluded"
PHASE1_STATS = RESULTS_DIR / "phase1_stats.json"

TARGET_ACTIVE = 10000


def main() -> int:
    if not REPOS_JSON.exists():
        logger.error("repos.json missing at %s", REPOS_JSON)
        return 1
    if not COUNTS_FILE.exists():
        logger.error("_repo_pr_counts.json missing at %s -- run Phase 2 first", COUNTS_FILE)
        return 1

    repos = json.loads(REPOS_JSON.read_text())
    counts = json.loads(COUNTS_FILE.read_text())
    logger.info("Loaded %d repos and %d count entries", len(repos), len(counts))

    # Snapshot the extended list before truncation.
    if not REPOS_PRESELECT.exists():
        REPOS_PRESELECT.write_text(json.dumps(repos, indent=2) + "\n")
        logger.info("Snapshotted extended list to %s", REPOS_PRESELECT)
    else:
        logger.info("Preselect snapshot already exists at %s; not overwriting", REPOS_PRESELECT)

    # PR-active filter: total_in_window > 0 in the counts file.
    def is_active(repo: dict) -> bool:
        full = f"{repo['owner']}/{repo['repo']}"
        return int(counts.get(full, {}).get("total_in_window", 0)) > 0

    active = [r for r in repos if is_active(r)]
    inactive = [r for r in repos if not is_active(r)]
    logger.info("PR-active: %d ; PR-inactive: %d ; total: %d",
                len(active), len(inactive), len(repos))

    if len(active) < TARGET_ACTIVE:
        logger.error(
            "Only %d PR-active repos in the extended list -- not enough to reach the "
            "target of %d. Re-run Phase 1 with a larger --candidate-cap.",
            len(active), TARGET_ACTIVE,
        )
        return 2

    # Sort active repos by criticality_rank ascending and take top N.
    active.sort(key=lambda r: r.get("criticality_rank", 1_000_000))
    selected = active[:TARGET_ACTIVE]
    excluded_repos = active[TARGET_ACTIVE:] + inactive

    excluded_set = {f"{r['owner']}/{r['repo']}" for r in excluded_repos}
    logger.info("Selected %d ; excluded %d (active overflow %d + inactive %d)",
                len(selected), len(excluded_set),
                len(active) - TARGET_ACTIVE, len(inactive))

    # Write the new repos.json (top-10k PR-active).
    REPOS_JSON.write_text(json.dumps(selected, indent=2) + "\n")
    logger.info("Wrote %d repos to %s", len(selected), REPOS_JSON)

    # Move excluded repos' JSONL files to data/prs_excluded/ so Phase 3's glob
    # only sees the selected universe.
    EXCLUDED_DIR.mkdir(exist_ok=True)
    moved = 0
    not_found = 0
    for full in excluded_set:
        owner, name = full.split("/", 1)
        src = PRS_DIR / f"{owner}__{name}.jsonl"
        if src.exists():
            shutil.move(str(src), EXCLUDED_DIR / src.name)
            moved += 1
        else:
            not_found += 1
    logger.info("Moved %d excluded JSONLs to %s (not found: %d)",
                moved, EXCLUDED_DIR, not_found)

    # Inject PR-active counts into phase1_stats.json so the paper fills correctly.
    if PHASE1_STATS.exists():
        stats = json.loads(PHASE1_STATS.read_text())
        stats["pr_active_repo_count"] = len(selected)
        stats["pr_inactive_repo_count"] = len(repos) - len(selected)
        stats["final_repo_count"] = len(selected)
        stats["selection_method"] = "top_10000_pr_active_by_criticality_rank"
        stats["preselect_count"] = len(repos)
        PHASE1_STATS.write_text(json.dumps(stats, indent=2) + "\n")
        logger.info("Updated %s with pr_active_repo_count=%d", PHASE1_STATS, len(selected))

    # Sanity: confirm criticality range of the selected set.
    crits = [r.get("criticality_score", 0) for r in selected]
    logger.info("Selected criticality range: max=%.5f min=%.5f", max(crits), min(crits))
    return 0


if __name__ == "__main__":
    sys.exit(main())

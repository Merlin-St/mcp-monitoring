"""Compute per-PR longest AI→AI chain and weekly aggregates (§5.1).

Primary definition (99_instruction.md §5.1):
- Within one PR, order attributed events by timestamp.
- An AI→AI chain is a maximal contiguous subsequence of events classified
  ``AI-bot`` (``AI-powered`` excluded from the primary definition because
  chains are about successive bot turns, not authorship attribution).
- Per-PR metric: longest AI→AI chain length.
- Per-week aggregate: mean, median, p95 over PRs open in the week.

Robustness run (toggle via --include-ai-powered): treat ``AI-powered`` as
also satisfying "AI" in the chain (this is the "AI authored" rollup).

Outputs:
  - data/chains.parquet   one row per PR (pr_key, longest_chain, n_events, ...)
  - results/chain_stats.json  weekly and overall summaries
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

utils = import_module("99_utils")
get_logger = utils.get_logger
DATA_DIR = utils.DATA_DIR
RESULTS_DIR = utils.RESULTS_DIR

logger = get_logger("05_compute_chains")


def longest_chain(types: list[str], ai_set: set[str]) -> int:
    """Longest contiguous run of types in ai_set."""
    best = cur = 0
    for t in types:
        if t in ai_set:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def longest_chain_per_group(
    is_ai: np.ndarray, group_change: np.ndarray
) -> np.ndarray:
    """Vectorised: longest contiguous True run in ``is_ai`` per group, where
    group boundaries are marked by True in ``group_change``.

    Both arrays are 1-D, length N. Returns a 1-D array of length n_groups
    (= number of True entries in group_change), in group order.

    Equivalent to (but ~10-20× faster than) iterating the groupby Python loop
    in the previous implementation. The single Python loop here operates on
    numpy ndarrays, so each iteration is cheap.
    """
    n = is_ai.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)
    # group_change[0] must be True (first row starts the first group).
    n_groups = int(group_change.sum())
    out = np.empty(n_groups, dtype=np.int64)
    g = -1
    cur = 0
    best = 0
    for i in range(n):
        if group_change[i]:
            if g >= 0:
                out[g] = best
            g += 1
            cur = 0
            best = 0
        if is_ai[i]:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    if g >= 0:
        out[g] = best
    return out


def main():
    parser = argparse.ArgumentParser()
    # Older invocations may pass --include-ai-assisted; accept both names.
    parser.add_argument(
        "--include-ai-powered",
        "--include-ai-assisted",
        dest="include_ai_powered",
        action="store_true",
    )
    args = parser.parse_args()

    events_path = DATA_DIR / "pr_events.parquet"
    summary_path = DATA_DIR / "pr_summary.parquet"
    if not events_path.exists() or not summary_path.exists():
        logger.error("Need pr_events.parquet and pr_summary.parquet. Run 03_classify_prs.py first.")
        sys.exit(1)

    events = pd.read_parquet(events_path)
    summary = pd.read_parquet(summary_path)

    ai_set: set[str] = {"AI-bot"}
    if args.include_ai_powered:
        ai_set.add("AI-powered")

    # Primary analysis also requires confidence=high for the ai-set events
    # (filter low-confidence handle-mention-only).
    events["_is_ai_high"] = (
        (events["actor_type"].isin(ai_set)) & (events["confidence"] == "high")
    )
    events["_ai_type_high"] = np.where(events["_is_ai_high"], "AI-bot", events["actor_type"])

    # Sort within PR (already sorted but re-assert).
    events = events.sort_values(["repo", "number", "event_idx"]).reset_index(drop=True)

    # ----- Vectorised per-PR aggregation -----
    # All per-group counts via groupby.agg + sum (avoids slicing each group).
    is_ai_bot = events["actor_type"].eq("AI-bot")
    is_high = events["confidence"].eq("high")
    is_ai_powered = events["actor_type"].eq("AI-powered")
    is_human = events["actor_type"].eq("human")
    is_non_ai_bot = events["actor_type"].eq("non_ai_bot")
    counts = pd.DataFrame({
        "repo": events["repo"],
        "number": events["number"],
        "n_events": 1,
        "n_ai_bot_high": (is_ai_bot & is_high).astype(np.int64),
        "n_ai_powered": is_ai_powered.astype(np.int64),
        "n_human": is_human.astype(np.int64),
        "n_non_ai_bot": is_non_ai_bot.astype(np.int64),
    }).groupby(["repo", "number"], sort=False, as_index=False).sum()

    # ----- Vectorised longest-chain (numpy single pass) -----
    # group_change[i] = True iff row i starts a new (repo, number) group, given
    # the events DataFrame is already sorted by (repo, number, event_idx).
    repo_arr = events["repo"].to_numpy()
    num_arr = events["number"].to_numpy()
    group_change = np.empty(len(events), dtype=bool)
    group_change[0] = True
    if len(events) > 1:
        group_change[1:] = (repo_arr[1:] != repo_arr[:-1]) | (num_arr[1:] != num_arr[:-1])

    is_ai_high_arr = events["_is_ai_high"].to_numpy()
    longest_primary = longest_chain_per_group(is_ai_high_arr, group_change)

    # Loose chain: AI-bot OR AI-powered (= "AI authored" rollup).
    is_ai_loose_arr = events["actor_type"].isin({"AI-bot", "AI-powered"}).to_numpy()
    longest_loose = longest_chain_per_group(is_ai_loose_arr, group_change)

    # The order of groups in `counts` (groupby sort=False) matches the order
    # in which group_change went True, so we can attach the chain arrays directly.
    counts["longest_chain_primary"] = longest_primary
    counts["longest_chain_loose"] = longest_loose
    chains = counts
    # Join opened_week from summary.
    chains = chains.merge(
        summary[["repo", "number", "opened_week", "merged", "merged_within_30d",
                 "author_type", "reviewer_type"]],
        on=["repo", "number"],
        how="left",
    )
    chains.to_parquet(DATA_DIR / "chains.parquet", index=False)
    logger.info("Wrote %d rows to data/chains.parquet", len(chains))

    # Weekly aggregates.
    weekly = (
        chains.groupby("opened_week")
        .agg(
            n_prs=("number", "count"),
            chain_mean=("longest_chain_primary", "mean"),
            chain_median=("longest_chain_primary", "median"),
            chain_p95=("longest_chain_primary", lambda s: float(np.percentile(s, 95))),
            chain_max=("longest_chain_primary", "max"),
            share_chain_ge2=("longest_chain_primary", lambda s: float((s >= 2).mean())),
            share_chain_ge5=("longest_chain_primary", lambda s: float((s >= 5).mean())),
        )
        .reset_index()
        .sort_values("opened_week")
    )
    weekly.to_csv(RESULTS_DIR / "chain_stats_weekly.csv", index=False)

    # Quarterly distribution (for Figure 2).
    chains["opened_dt"] = pd.to_datetime(chains["opened_week"].str[:4] + "-01-01", errors="coerce")
    # Use opened_week to get quarter
    def week_to_quarter(w):
        try:
            y, wk = w.split("-W")
            week_n = int(wk)
            q = (week_n - 1) // 13 + 1
            return f"{y}-Q{min(q,4)}"
        except Exception:
            return ""
    chains["quarter"] = chains["opened_week"].apply(week_to_quarter)
    quarterly = (
        chains.groupby("quarter")["longest_chain_primary"]
        .agg(["count", "mean", "median",
              lambda s: float(np.percentile(s, 95)),
              "max"])
        .rename(columns={"<lambda_0>": "p95"})
        .reset_index()
        .sort_values("quarter")
    )
    quarterly.to_csv(RESULTS_DIR / "chain_stats_quarterly.csv", index=False)

    summary_stats = {
        "total_prs_with_events": int(len(chains)),
        "prs_with_any_ai_event": int((chains["n_ai_bot_high"] > 0).sum()),
        "prs_with_chain_ge2": int((chains["longest_chain_primary"] >= 2).sum()),
        "prs_with_chain_ge5": int((chains["longest_chain_primary"] >= 5).sum()),
        "chain_overall_mean": float(chains["longest_chain_primary"].mean()),
        "chain_overall_p95": float(np.percentile(chains["longest_chain_primary"], 95)),
        "chain_overall_max": int(chains["longest_chain_primary"].max()),
        "include_ai_powered": bool(args.include_ai_powered),
        "generated_at": utils.now_iso(),
    }
    with open(RESULTS_DIR / "chain_stats.json", "w") as f:
        json.dump(summary_stats, f, indent=2)
    logger.info("=== CHAINS DONE ===")
    logger.info(json.dumps(summary_stats, indent=2))


if __name__ == "__main__":
    main()

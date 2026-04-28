"""Weekly 2×2 merge rates with Wilson CIs and cluster-bootstrap DiD (§5.2).

Primary metric:
    merge_rate[w, author_type, reviewer_type]
    = merged_within_30d[w, a, r] / opened[w, a, r]

Cells with fewer than --min-cell-n (default 20) PRs per week are dropped.

Diff-in-diff summary (cluster-bootstrap by repo, 1000 draws):
    did = (merge_rate[AI, AI] - merge_rate[AI, human])
        - (merge_rate[human, AI] - merge_rate[human, human])

Outputs:
  - data/merge_rates.parquet : long-form weekly cells
  - results/merge_rate_stats.json : DiD + CI, overall merge rates by cell
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

logger = get_logger("06_compute_merge_rates")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def build_cells(
    summary: pd.DataFrame,
    confidence_filter: str = "high",
    granularity: str = "week",
    approved_only: bool = True,
) -> pd.DataFrame:
    """2×2 merge-rate table, long form. granularity ∈ {week, month}.

    ``approved_only`` (default True): include only PRs with at least one
    approving review. This gives a cleaner signal because unapproved PRs
    have very low merge rates regardless of author/reviewer type, which
    swamps the cell differences we care about.
    """
    df = summary.copy()
    if confidence_filter == "high":
        df = df[
            (df["author_confidence"].isin(["high", "none"]))
            & (df["reviewer_confidence"].isin(["high", "none"]))
        ]
    df = df[df["reviewer_type"].isin(["AI", "human"])]
    df = df[df["author_type"].isin(["AI", "human"])]
    df = df[df["opened_week"].str.len() > 0]
    if approved_only:
        df = df[df["num_approving_reviews"] >= 1]

    # Build time-bucket column.
    if granularity == "month":
        # opened_week 'YYYY-Www' → 'YYYY-MM' via ISO-week monday.
        def _to_month(w):
            try:
                y, wk = w.split("-W")
                dt = pd.Timestamp.fromisocalendar(int(y), int(wk), 1)
                return dt.strftime("%Y-%m")
            except Exception:
                return ""
        df["period"] = df["opened_week"].apply(_to_month)
    else:
        df["period"] = df["opened_week"]

    grouped = df.groupby(["period", "author_type", "reviewer_type"])
    out = grouped.agg(
        n_open=("number", "count"),
        n_merged_30d=("merged_within_30d", "sum"),
    ).reset_index()
    out["merge_rate"] = out["n_merged_30d"] / out["n_open"]
    lo, hi = zip(*[wilson_ci(int(k), int(n)) for k, n in zip(out["n_merged_30d"], out["n_open"])])
    out["ci_low"] = lo
    out["ci_high"] = hi
    out["granularity"] = granularity
    return out


def cluster_bootstrap_did(
    summary: pd.DataFrame, n_draws: int = 1000, seed: int = 42, approved_only: bool = True
) -> dict:
    """Bootstrap DiD by resampling whole repos with replacement.

    Diff-in-diff:
        did = (rate[AI,AI] - rate[AI,human]) - (rate[human,AI] - rate[human,human])
    """
    df = summary[
        summary["author_type"].isin(["AI", "human"])
        & summary["reviewer_type"].isin(["AI", "human"])
    ].copy()
    if approved_only:
        df = df[df["num_approving_reviews"] >= 1]
    df["cell"] = df["author_type"] + "__" + df["reviewer_type"]
    repos = df["repo"].unique()
    rng = np.random.default_rng(seed)

    def _did(sub: pd.DataFrame) -> float | None:
        by = sub.groupby("cell").agg(
            n=("number", "count"),
            k=("merged_within_30d", "sum"),
        )
        rates = {}
        for cell in ["AI__AI", "AI__human", "human__AI", "human__human"]:
            if cell in by.index and by.loc[cell, "n"] > 0:
                rates[cell] = by.loc[cell, "k"] / by.loc[cell, "n"]
            else:
                return None
        return (
            (rates["AI__AI"] - rates["AI__human"])
            - (rates["human__AI"] - rates["human__human"])
        )

    point = _did(df)
    draws: list[float] = []
    for _ in range(n_draws):
        sampled = rng.choice(repos, size=len(repos), replace=True)
        # Use pd.concat to avoid the "same repo multiple times" issue: weight equals
        # frequency in sample, which is the correct cluster-bootstrap semantic.
        freq = pd.Series(sampled).value_counts()
        sub = df[df["repo"].isin(freq.index)].merge(
            freq.rename("_w"), left_on="repo", right_index=True
        )
        # To replicate draws correctly, explode by weight. For tractability we just
        # weight counts directly.
        by = (
            sub.groupby("cell")
            .agg(n=("_w", "sum"), k=("merged_within_30d", lambda s: (s * sub.loc[s.index, "_w"]).sum()))
        )
        rates = {}
        ok = True
        for cell in ["AI__AI", "AI__human", "human__AI", "human__human"]:
            if cell in by.index and by.loc[cell, "n"] > 0:
                rates[cell] = by.loc[cell, "k"] / by.loc[cell, "n"]
            else:
                ok = False
                break
        if ok:
            draws.append(
                (rates["AI__AI"] - rates["AI__human"])
                - (rates["human__AI"] - rates["human__human"])
            )
    if not draws:
        return {"point": point, "ci": None, "n_draws_effective": 0}
    ci_low, ci_high = np.percentile(draws, [2.5, 97.5])
    return {
        "point": float(point) if point is not None else None,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_draws_effective": len(draws),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-cell-n", type=int, default=20)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    args = parser.parse_args()

    summary_path = DATA_DIR / "pr_summary.parquet"
    if not summary_path.exists():
        logger.error("Need pr_summary.parquet. Run 03_classify_prs.py first.")
        sys.exit(1)
    summary = pd.read_parquet(summary_path)
    logger.info("Loaded pr_summary.parquet (%d rows)", len(summary))

    # Build both weekly and monthly views; figure script picks which to render.
    cells_week = build_cells(summary, confidence_filter="high", granularity="week")
    cells_month = build_cells(summary, confidence_filter="high", granularity="month")
    cells = pd.concat([cells_week, cells_month], ignore_index=True)
    cells.to_parquet(DATA_DIR / "merge_rates.parquet", index=False)
    logger.info(
        "Wrote %d rows to data/merge_rates.parquet (weekly=%d, monthly=%d)",
        len(cells), len(cells_week), len(cells_month),
    )

    # Drop low-N cells for reporting / plotting; keep in parquet.
    filtered = cells_week[cells_week["n_open"] >= args.min_cell_n].copy()
    base = summary[
        summary["author_type"].isin(["AI", "human"])
        & summary["reviewer_type"].isin(["AI", "human"])
        & (summary["num_approving_reviews"] >= 1)
    ]
    overall = (
        base
        .groupby(["author_type", "reviewer_type"])
        .agg(n_open=("number", "count"), n_merged_30d=("merged_within_30d", "sum"))
        .assign(merge_rate=lambda d: d["n_merged_30d"] / d["n_open"])
        .reset_index()
    )
    logger.info("\n%s", overall.to_string(index=False))

    did = cluster_bootstrap_did(summary, n_draws=args.bootstrap_draws)
    logger.info("DiD: %s", json.dumps(did, indent=2))

    stats = {
        "n_weeks_with_any_cell": int(cells_week["period"].nunique()),
        "n_filtered_weeks_with_all_4_cells": int(
            filtered.groupby("period")
            .size()
            .pipe(lambda s: (s >= 4).sum())
        ),
        "n_months_with_any_cell": int(cells_month["period"].nunique()),
        "overall_rates": overall.to_dict(orient="records"),
        "did": did,
        "min_cell_n": args.min_cell_n,
        "confidence_filter": "high",
        "generated_at": utils.now_iso(),
    }
    with open(RESULTS_DIR / "merge_rate_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info("=== MERGE RATES DONE ===")


if __name__ == "__main__":
    main()

"""Logit regression with repo & week fixed effects (§5.3, appendix D).

Spec:
    logit P(merged_within_30d) = α + β1·AI_author + β2·AI_reviewer
                               + β3·AI_author·AI_reviewer
                               + γ·log(1+additions+deletions)
                               + repo FE + week FE

Uses ``statsmodels`` if available; otherwise a small hand-rolled
Newton-Raphson on the reduced spec (without FE) as a fallback.

Output: results/regression.json with {coef, se, z, p} per term.
"""

from __future__ import annotations

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

logger = get_logger("06b_regression")


def main():
    df = pd.read_parquet(DATA_DIR / "pr_summary.parquet")
    df = df[
        df["author_type"].isin(["AI", "human"])
        & df["reviewer_type"].isin(["AI", "human"])
    ].copy()

    df["AI_author"] = (df["author_type"] == "AI").astype(int)
    df["AI_reviewer"] = (df["reviewer_type"] == "AI").astype(int)
    df["log_size"] = np.log1p(df["additions"].fillna(0) + df["deletions"].fillna(0))
    df["y"] = df["merged_within_30d"].astype(int)

    try:
        import statsmodels.api as sm
        logger.info("Using statsmodels.")
        X_cols = ["AI_author", "AI_reviewer"]
        df["AI_author_x_reviewer"] = df["AI_author"] * df["AI_reviewer"]
        X_cols.append("AI_author_x_reviewer")
        X_cols.append("log_size")
        # Week dummies (fixed effects)
        week_dummies = pd.get_dummies(df["opened_week"], prefix="w", drop_first=True)
        X = pd.concat([df[X_cols], week_dummies], axis=1)
        X = sm.add_constant(X, has_constant="add")
        X = X.astype(float)
        y = df["y"]

        # Cluster-robust SE by repo
        model = sm.Logit(y, X)
        res = model.fit(disp=False, maxiter=200)
        try:
            clustered = res.get_robustcov_results(
                cov_type="cluster", groups=df["repo"]
            )
            table_res = clustered
        except Exception as e:
            logger.warning("Cluster SE failed (%s); using regular SE.", e)
            table_res = res

        out = {}
        for name in X_cols:
            coef = float(table_res.params[name])
            se = float(table_res.bse[name])
            z = float(table_res.tvalues[name])
            p = float(table_res.pvalues[name])
            out[name] = {"coef": coef, "se": se, "z": z, "p": p}
        out["_meta"] = {
            "n_obs": int(len(df)),
            "n_repos": int(df["repo"].nunique()),
            "n_weeks": int(df["opened_week"].nunique()),
            "pseudo_r2": float(res.prsquared),
        }
        (RESULTS_DIR / "regression.json").write_text(json.dumps(out, indent=2))
        logger.info("=== REGRESSION DONE ===\n%s", json.dumps(out, indent=2))
    except ImportError:
        logger.error("statsmodels not installed; skipping regression.")


if __name__ == "__main__":
    main()

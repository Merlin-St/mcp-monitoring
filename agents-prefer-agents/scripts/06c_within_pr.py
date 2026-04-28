"""Within-PR AI-AI bias test (v2 design).

This implements the §8.7 main DiD and the §8.4 anchoring robustness check
from `99_causalvalidity.md`. The estimand is

    rbt^A - rbt^H = [P(AI app | AI-auth) - P(H app | AI-auth)]
                  - [P(AI app | H-auth)  - P(H app | H-auth)]

on the **stable doubly-engaged** cohort: PRs that received both an AI-bot
opinion (native APPROVED/CHANGES_REQUESTED OR a regex-parsed verdict in a
COMMENTED body — see `ai_verdict_parser.py`) AND a human explicit review,
with no commits pushed strictly between the AI's first opinion timestamp and
the human's first explicit-review timestamp.

Outputs `results/within_pr_stats.json` with four top-level blocks:

    did:              cells, brackets, DiD point estimate, logit interaction
    anchoring:        Stratum A (no AI reviewer) vs Stratum B (AI-first) gap
                      decomposition; cross-stratum DiD (anchoring effect)
    ai_only:          context stat — % of AI-bot-reviewed PRs that have no
                      human explicit review (motivation for conditioning on
                      human review)
    ai_opinion_pool:  native vs parsed event counts; unique-PR coverage

Compatibility shim: a `legacy_within_pr` block reproduces the old
`conditional_on_ai_approval` schema for any consumer still expecting it.
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
utils = import_module("99_utils")
get_logger = utils.get_logger
DATA_DIR = utils.DATA_DIR
RESULTS_DIR = utils.RESULTS_DIR
SUBPROJECT_ROOT = utils.SUBPROJECT_ROOT

from ai_verdict_parser import parse_ai_opinions

logger = get_logger("06c_within_pr")


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _two_prop_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan"), float("nan")
    z = (p1 - p2) / se
    return float(z), float(2 * (1 - norm.cdf(abs(z))))


def _cell(numer: set, denom: set) -> tuple[int, int, float]:
    n = len(denom)
    k = len(denom & numer)
    return k, n, (k / n if n else 0.0)


def main():
    ev = pd.read_parquet(DATA_DIR / "pr_events.parquet")
    summ = pd.read_parquet(DATA_DIR / "pr_summary.parquet")
    ev["t"] = pd.to_datetime(ev.timestamp)
    summ["created_t"] = pd.to_datetime(summ.created_at)

    v2_repos = set(summ.repo)
    logger.info(
        "v2 universe: %d PRs across %d repos; %d events", len(summ), len(v2_repos), len(ev)
    )

    # ---------- 1. AI opinion table (native + parsed) ----------
    prs_dir = SUBPROJECT_ROOT / "data" / "prs"
    ai_op = parse_ai_opinions(prs_dir, repos=v2_repos)
    n_native_app = int((ai_op.kind == "approve_native").sum())
    n_native_rej = int((ai_op.kind == "reject_native").sum())
    n_parsed_app = int((ai_op.kind == "approve_parsed").sum())
    n_parsed_rej = int((ai_op.kind == "reject_parsed").sum())
    n_unique_op_prs = int(ai_op.groupby(["repo", "number"]).ngroups)
    logger.info(
        "AI opinion events: native=%d/%d  parsed=%d/%d  unique-PRs=%d",
        n_native_app, n_native_rej, n_parsed_app, n_parsed_rej, n_unique_op_prs,
    )

    ai_app_pr = set(map(tuple, ai_op[ai_op.kind.str.startswith("approve")][["repo", "number"]].drop_duplicates().to_numpy()))
    ai_rej_pr = set(map(tuple, ai_op[ai_op.kind.str.startswith("reject")][["repo", "number"]].drop_duplicates().to_numpy()))
    ai_opin_pr = ai_app_pr | ai_rej_pr
    ai_op_first = ai_op.groupby(["repo", "number"]).t.min()

    # ---------- 2. Human/AI review structure ----------
    hu_engage = ev[
        (ev.actor_type == "human")
        & ev.event_type.isin([
            "review_commented", "review_approved", "review_changes_requested",
            "review_comment", "issue_comment",
        ])
    ]
    hu_engage_first = hu_engage.groupby(["repo", "number"]).t.min()

    hu_expl = ev[
        (ev.actor_type == "human")
        & ev.event_type.isin(["review_approved", "review_changes_requested"])
    ]
    hu_expl_pr = set(map(tuple, hu_expl[["repo", "number"]].drop_duplicates().to_numpy()))
    hu_app_pr = set(map(tuple, hu_expl[hu_expl.event_type == "review_approved"][["repo", "number"]].drop_duplicates().to_numpy()))
    hu_expl_first = hu_expl.groupby(["repo", "number"]).t.min()

    ai_any = ev[
        (ev.actor_type == "AI-bot")
        & ev.event_type.isin(["review_commented", "review_approved", "review_changes_requested"])
    ]
    ai_any_set = set(map(tuple, ai_any[["repo", "number"]].drop_duplicates().to_numpy()))

    commits = ev[ev.event_type == "commit"][["repo", "number", "t"]]
    commits_by_pr = commits.groupby(["repo", "number"]).t.apply(list).to_dict()
    created = summ.set_index(["repo", "number"])["created_t"]

    def has_commit_between(key, lo, hi):
        if pd.isna(lo) or pd.isna(hi):
            return None
        if lo == hi:
            return False
        a, b = (lo, hi) if lo <= hi else (hi, lo)
        for c in commits_by_pr.get(key, []):
            if a < c < b:
                return True
        return False

    summ_idx = summ.set_index(["repo", "number"])

    # ---------- 3. §8.7 DiD on stable doubly-engaged cohort ----------
    both_set_all = ai_opin_pr & hu_expl_pr
    stable = {
        k for k in both_set_all
        if has_commit_between(k, ai_op_first.get(k), hu_expl_first.get(k)) is False
    }
    sub_stable = summ_idx.loc[list(stable)]
    stable_AI = set(sub_stable[sub_stable.author_type == "AI"].index)
    stable_H = set(sub_stable[sub_stable.author_type == "human"].index)

    AI_AI_k, AI_AI_n, AI_AI_r = _cell(ai_app_pr, stable_AI)
    H_AI_k,  H_AI_n,  H_AI_r  = _cell(hu_app_pr, stable_AI)
    AI_H_k,  AI_H_n,  AI_H_r  = _cell(ai_app_pr, stable_H)
    H_H_k,   H_H_n,   H_H_r   = _cell(hu_app_pr, stable_H)

    ai_AI_lo, ai_AI_hi = _wilson(AI_AI_k, AI_AI_n)
    h_AI_lo,  h_AI_hi  = _wilson(H_AI_k,  H_AI_n)
    ai_H_lo,  ai_H_hi  = _wilson(AI_H_k,  AI_H_n)
    h_H_lo,   h_H_hi   = _wilson(H_H_k,   H_H_n)

    bracket1_pp = (AI_AI_r - H_AI_r) * 100  # AI-vs-H approval on AI-authored
    bracket2_pp = (AI_H_r - H_H_r) * 100    # AI-vs-H approval on H-authored
    did_pp = bracket1_pp - bracket2_pp

    ai_gap_pp = (AI_AI_r - AI_H_r) * 100
    hu_gap_pp = (H_AI_r - H_H_r) * 100

    # Logit DiD with HC0 SEs on long-format data
    rows = []
    for k in stable:
        if k in stable_AI:
            auth = 1
        elif k in stable_H:
            auth = 0
        else:
            continue
        rows.append((1, auth, 1, 1 if k in ai_app_pr else 0))  # AI reviewer
        rows.append((1, auth, 0, 1 if k in hu_app_pr else 0))  # human reviewer
    df = pd.DataFrame(rows, columns=["const", "author_AI", "reviewer_AI", "approved"])
    df["interaction"] = df.author_AI * df.reviewer_AI
    X = df[["const", "reviewer_AI", "author_AI", "interaction"]].astype(float)
    m = sm.Logit(df.approved.astype(float), X).fit(disp=0, cov_type="HC0")

    # Asymmetric exclusion stat (drop rate of full -> stable cohort by author)
    sub_full = summ_idx.loc[list(both_set_all)]
    full_AI = set(sub_full[sub_full.author_type == "AI"].index)
    full_H = set(sub_full[sub_full.author_type == "human"].index)
    drop_AI = len(full_AI) - len(stable_AI)
    drop_H = len(full_H) - len(stable_H)

    did_block = {
        "n": int(len(stable)),
        "n_AI_auth": int(len(stable_AI)),
        "n_H_auth": int(len(stable_H)),
        "cells": {
            "AI_x_AI": {"k": AI_AI_k, "n": AI_AI_n, "rate": AI_AI_r,
                        "ci_lo": ai_AI_lo, "ci_hi": ai_AI_hi},
            "H_x_AI":  {"k": H_AI_k,  "n": H_AI_n,  "rate": H_AI_r,
                        "ci_lo": h_AI_lo,  "ci_hi": h_AI_hi},
            "AI_x_H":  {"k": AI_H_k,  "n": AI_H_n,  "rate": AI_H_r,
                        "ci_lo": ai_H_lo,  "ci_hi": ai_H_hi},
            "H_x_H":   {"k": H_H_k,   "n": H_H_n,   "rate": H_H_r,
                        "ci_lo": h_H_lo,   "ci_hi": h_H_hi},
        },
        "bracket1_pp": float(bracket1_pp),
        "bracket2_pp": float(bracket2_pp),
        "did_pp": float(did_pp),
        "ai_gap_pp": float(ai_gap_pp),
        "hu_gap_pp": float(hu_gap_pp),
        "logit_delta": float(m.params["interaction"]),
        "logit_se": float(m.bse["interaction"]),
        "logit_p": float(m.pvalues["interaction"]),
        "logit_reviewer_AI": float(m.params["reviewer_AI"]),
        "logit_author_AI": float(m.params["author_AI"]),
        "n_full": int(len(both_set_all)),
        "n_full_AI_auth": int(len(full_AI)),
        "n_full_H_auth": int(len(full_H)),
        "drop_AI": int(drop_AI),
        "drop_H": int(drop_H),
        "drop_pct": float(100 * (drop_AI + drop_H) / max(len(both_set_all), 1)),
        "drop_AI_pct": float(100 * drop_AI / max(len(full_AI), 1)),
        "drop_H_pct": float(100 * drop_H / max(len(full_H), 1)),
    }
    logger.info(
        "DiD: bracket1=%+.2fpp bracket2=%+.2fpp did=%+.2fpp delta=%+.4f p=%.4g (n=%d, AI=%d, H=%d)",
        bracket1_pp, bracket2_pp, did_pp, did_block["logit_delta"], did_block["logit_p"],
        did_block["n"], did_block["n_AI_auth"], did_block["n_H_auth"],
    )

    # ---------- 4. §8.4 anchoring robustness ----------
    # Stratum A: no AI-bot review event AT ALL + ≥1 human explicit review.
    # No no-commits filter applied — matches §8.2/§8.4 in 99_causalvalidity.md.
    A_base = (set(summ_idx.index) - ai_any_set) & hu_expl_pr
    A_sub = summ_idx.loc[list(A_base)]
    A_AI = set(A_sub[A_sub.author_type == "AI"].index)
    A_H = set(A_sub[A_sub.author_type == "human"].index)
    A_AI_k = len(A_AI & hu_app_pr); A_H_k = len(A_H & hu_app_pr)
    A_gap_pp = (A_AI_k / max(len(A_AI), 1) - A_H_k / max(len(A_H), 1)) * 100
    zA, pA = _two_prop_z(A_AI_k, len(A_AI), A_H_k, len(A_H))

    # Stratum B: AI bot's first comment/review event preceded any human
    # engagement + ≥1 human explicit review.
    both_kt = ai_op_first.to_frame("ai").join(hu_engage_first.to_frame("hu"), how="left")
    B_candidates = set(map(tuple, both_kt[both_kt.hu.notna() & (both_kt.ai < both_kt.hu)].reset_index()[["repo", "number"]].to_numpy())) & hu_expl_pr
    B_sub = summ_idx.loc[list(B_candidates)]
    B_AI = set(B_sub[B_sub.author_type == "AI"].index)
    B_H = set(B_sub[B_sub.author_type == "human"].index)
    B_AI_k = len(B_AI & hu_app_pr); B_H_k = len(B_H & hu_app_pr)
    B_gap_pp = (B_AI_k / max(len(B_AI), 1) - B_H_k / max(len(B_H), 1)) * 100
    zB, pB = _two_prop_z(B_AI_k, len(B_AI), B_H_k, len(B_H))

    cross_did_pp = B_gap_pp - A_gap_pp

    # Analytical SE on the cross-stratum DiD (rate-difference scale): two
    # independent gaps, each a difference of two binomial proportions, summed
    # in quadrature.
    def _gap_var(k1, n1, k2, n2):
        if n1 == 0 or n2 == 0:
            return float("nan")
        p1, p2 = k1 / n1, k2 / n2
        return p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2

    A_var = _gap_var(A_AI_k, len(A_AI), A_H_k, len(A_H))
    B_var = _gap_var(B_AI_k, len(B_AI), B_H_k, len(B_H))
    cross_se_pp = float(100 * np.sqrt(A_var + B_var)) if not (np.isnan(A_var) or np.isnan(B_var)) else float("nan")
    cross_ci_lo_pp = cross_did_pp - 1.96 * cross_se_pp
    cross_ci_hi_pp = cross_did_pp + 1.96 * cross_se_pp

    A_AI_lo, A_AI_hi = _wilson(A_AI_k, len(A_AI))
    A_H_lo, A_H_hi = _wilson(A_H_k, len(A_H))
    B_AI_lo, B_AI_hi = _wilson(B_AI_k, len(B_AI))
    B_H_lo, B_H_hi = _wilson(B_H_k, len(B_H))

    anchoring_block = {
        "A_n": int(len(A_AI) + len(A_H)),
        "A_AI": {"k": int(A_AI_k), "n": int(len(A_AI)),
                 "rate": A_AI_k / max(len(A_AI), 1),
                 "ci_lo": A_AI_lo, "ci_hi": A_AI_hi},
        "A_H":  {"k": int(A_H_k),  "n": int(len(A_H)),
                 "rate": A_H_k / max(len(A_H), 1),
                 "ci_lo": A_H_lo, "ci_hi": A_H_hi},
        "A_gap_pp": float(A_gap_pp), "A_z": float(zA), "A_p": float(pA),
        "B_n": int(len(B_AI) + len(B_H)),
        "B_AI": {"k": int(B_AI_k), "n": int(len(B_AI)),
                 "rate": B_AI_k / max(len(B_AI), 1),
                 "ci_lo": B_AI_lo, "ci_hi": B_AI_hi},
        "B_H":  {"k": int(B_H_k),  "n": int(len(B_H)),
                 "rate": B_H_k / max(len(B_H), 1),
                 "ci_lo": B_H_lo, "ci_hi": B_H_hi},
        "B_gap_pp": float(B_gap_pp), "B_z": float(zB), "B_p": float(pB),
        "cross_stratum_did_pp": float(cross_did_pp),
        "cross_stratum_did_se_pp": cross_se_pp,
        "cross_stratum_did_ci_lo_pp": float(cross_ci_lo_pp),
        "cross_stratum_did_ci_hi_pp": float(cross_ci_hi_pp),
    }
    logger.info(
        "Anchoring: A_gap=%+.2fpp B_gap=%+.2fpp cross_DiD=%+.2fpp",
        A_gap_pp, B_gap_pp, cross_did_pp,
    )

    # ---------- 5. AI-only review context ----------
    ai_only_no_expl = ai_any_set - hu_expl_pr
    ai_only_no_engage = ai_any_set - set(hu_engage_first.index)
    ai_only_block = {
        "ai_reviewed_n": int(len(ai_any_set)),
        "no_human_explicit_n": int(len(ai_only_no_expl)),
        "no_human_explicit_pct": float(100 * len(ai_only_no_expl) / max(len(ai_any_set), 1)),
        "no_human_engagement_n": int(len(ai_only_no_engage)),
        "no_human_engagement_pct": float(100 * len(ai_only_no_engage) / max(len(ai_any_set), 1)),
    }

    # ---------- 6. AI opinion pool summary ----------
    pool_block = {
        "n_native_approve": n_native_app,
        "n_native_reject": n_native_rej,
        "n_parsed_approve": n_parsed_app,
        "n_parsed_reject": n_parsed_rej,
        "n_unique_prs": n_unique_op_prs,
        "n_native_unique_prs": int(
            ai_op[ai_op.kind.isin(["approve_native", "reject_native"])]
            .groupby(["repo", "number"]).ngroups
        ),
    }

    # ---------- 7. Backward-compat: legacy schema (old 282-cohort) ----------
    # Keep so any consumer still expecting `conditional_on_ai_approval` works.
    r = ev[
        ev.event_type.str.startswith("review_") & ~ev.event_type.eq("review_comment")
    ].copy()
    r["is_human"] = r.actor_type == "human"
    r["is_ai_bot_high"] = (r.actor_type == "AI-bot") & (r.confidence == "high")
    r["is_approved"] = r.event_type == "review_approved"
    pr = r.groupby(["repo", "number"]).agg(
        ai_approved=("is_approved", lambda s: ((r.loc[s.index, "is_ai_bot_high"]) & s).any()),
        human_approved=("is_approved", lambda s: ((r.loc[s.index, "is_human"]) & s).any()),
    ).reset_index()
    j = pr.merge(summ[["repo", "number", "author_type"]], on=["repo", "number"], how="left")
    j = j[j.author_type.isin(["AI", "human"])].copy()
    cond_ai = j[j.ai_approved]
    cond_ai_ai = cond_ai[cond_ai.author_type == "AI"]
    cond_ai_hu = cond_ai[cond_ai.author_type == "human"]

    out = {
        "did": did_block,
        "anchoring": anchoring_block,
        "ai_only": ai_only_block,
        "ai_opinion_pool": pool_block,
        "legacy_within_pr": {
            "conditional_on_ai_approval": {
                "n_total": int(len(cond_ai)),
                "ai_authored": {"n": int(len(cond_ai_ai)),
                                "human_co_approved": int(cond_ai_ai.human_approved.sum())},
                "human_authored": {"n": int(len(cond_ai_hu)),
                                   "human_co_approved": int(cond_ai_hu.human_approved.sum())},
            },
        },
        "generated_at": utils.now_iso(),
    }

    (RESULTS_DIR / "within_pr_stats.json").write_text(json.dumps(out, indent=2))
    logger.info("=== WITHIN-PR DiD DONE ===")


if __name__ == "__main__":
    main()

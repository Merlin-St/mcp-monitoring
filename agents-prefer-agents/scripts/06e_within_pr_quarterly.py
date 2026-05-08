"""Quarterly RQ3 DiDs (companion to 06c_within_pr.py and 06d_within_family_claude.py).

For each PR-creation quarter in the analysis window we recompute:

    DiD-1  (cross-family / "AI vs. human" reviewer):
            on the stable doubly-engaged cohort (Any AI & human),
            $\\widehat{r_{bt}^A - r_{bt}^H}$ — same construction as 06c.

    DiD-2  (within-family / "AI vs. AI-authored code" — Claude reviewer
            on Claude- vs human-authored PRs minus the human-reviewer
            counterpart) — same construction as 06d.

For each DiD we report point estimates and 95% CIs (analytical, two
independent gaps summed in quadrature on the rate-difference scale; this
matches the anchoring-cross-stratum CI in 06c). Cell rates also carry
Wilson 95% CIs.

Output: results/within_pr_quarterly.json
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
utils = import_module("99_utils")
get_logger = utils.get_logger
DATA_DIR = utils.DATA_DIR
RESULTS_DIR = utils.RESULTS_DIR
SUBPROJECT_ROOT = utils.SUBPROJECT_ROOT

from ai_verdict_parser import parse_ai_opinions

logger = get_logger("06e_within_pr_quarterly")

CLAUDE_BOT_LOGIN_SUBS = ("claude", "anthropic-ai", "anthropic-code-agent")


def _is_claude_login(login: str) -> bool:
    lg = (login or "").lower()
    return any(s in lg for s in CLAUDE_BOT_LOGIN_SUBS)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _gap_var(k1: int, n1: int, k2: int, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return float("nan")
    p1, p2 = k1 / n1, k2 / n2
    return p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2


def _did_with_ci(
    *, AA_k: int, AA_n: int, HA_k: int, HA_n: int,
    AH_k: int, AH_n: int, HH_k: int, HH_n: int,
) -> dict:
    """Standard 2x2 DiD on rate-difference scale + analytical 95% CI.

    Cells (reviewer x author):
        AA = AI/treated reviewer × AI/treated author
        HA = reference reviewer × AI/treated author
        AH = AI/treated reviewer × reference author
        HH = reference reviewer × reference author

    bracket1 = rate(AA) - rate(HA)   (gap on AI-/treated-author side)
    bracket2 = rate(AH) - rate(HH)   (gap on H-/reference-author side)
    DiD      = bracket1 - bracket2
    """

    def rate(k, n): return (k / n) if n else float("nan")

    b1 = (rate(AA_k, AA_n) - rate(HA_k, HA_n)) * 100
    b2 = (rate(AH_k, AH_n) - rate(HH_k, HH_n)) * 100
    did = b1 - b2

    var_b1 = _gap_var(AA_k, AA_n, HA_k, HA_n)
    var_b2 = _gap_var(AH_k, AH_n, HH_k, HH_n)
    if np.isnan(var_b1) or np.isnan(var_b2):
        se = float("nan")
        lo = hi = float("nan")
        b1_lo = b1_hi = b2_lo = b2_hi = float("nan")
    else:
        se = float(100 * np.sqrt(var_b1 + var_b2))
        lo = float(did - 1.96 * se)
        hi = float(did + 1.96 * se)
        se_b1 = 100 * np.sqrt(var_b1)
        se_b2 = 100 * np.sqrt(var_b2)
        b1_lo = float(b1 - 1.96 * se_b1)
        b1_hi = float(b1 + 1.96 * se_b1)
        b2_lo = float(b2 - 1.96 * se_b2)
        b2_hi = float(b2 + 1.96 * se_b2)

    cells = {}
    for name, k, n in [("AA", AA_k, AA_n), ("HA", HA_k, HA_n),
                       ("AH", AH_k, AH_n), ("HH", HH_k, HH_n)]:
        ci_lo, ci_hi = _wilson(k, n)
        cells[name] = {
            "k": int(k), "n": int(n),
            "rate": float(rate(k, n)) if n else float("nan"),
            "ci_lo": ci_lo, "ci_hi": ci_hi,
        }

    return {
        "cells": cells,
        "bracket1_pp": float(b1),
        "bracket1_ci_lo_pp": b1_lo,
        "bracket1_ci_hi_pp": b1_hi,
        "bracket2_pp": float(b2),
        "bracket2_ci_lo_pp": b2_lo,
        "bracket2_ci_hi_pp": b2_hi,
        "did_pp": float(did),
        "did_se_pp": se,
        "did_ci_lo_pp": lo,
        "did_ci_hi_pp": hi,
    }


def _quarter_of(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return ""
    y = ts.year
    q = (ts.month - 1) // 3 + 1
    return f"{y}-Q{q}"


def main():
    ev = pd.read_parquet(DATA_DIR / "pr_events.parquet")
    summ = pd.read_parquet(DATA_DIR / "pr_summary.parquet")
    ev["t"] = pd.to_datetime(ev.timestamp)
    summ["created_t"] = pd.to_datetime(summ.created_at)
    summ["quarter"] = summ.created_t.apply(_quarter_of)

    v2_repos = set(summ.repo)
    logger.info("Universe: %d PRs across %d repos.", len(summ), len(v2_repos))

    # ---------- AI opinion table ----------
    prs_dir = SUBPROJECT_ROOT / "data" / "prs"
    ai_op = parse_ai_opinions(prs_dir, repos=v2_repos)
    ai_op["bot_l"] = ai_op.bot.str.lower()
    ai_op["is_claude"] = ai_op.bot_l.apply(_is_claude_login)
    ai_op["approve"] = ai_op.kind.str.startswith("approve")
    # "Native" = APPROVED / CHANGES_REQUESTED review states only (no
    # regex-parsed verdict lines from COMMENTED review bodies). Used for the
    # symmetric-with-human-side robustness pool.
    ai_op["is_native"] = ai_op.kind.str.endswith("_native")

    ai_app_pr = set(map(tuple, ai_op[ai_op.approve][["repo", "number"]].drop_duplicates().to_numpy()))
    ai_opin_pr = set(map(tuple, ai_op[["repo", "number"]].drop_duplicates().to_numpy()))
    ai_op_first = ai_op.groupby(["repo", "number"]).t.min()

    ai_op_native = ai_op[ai_op.is_native]
    ai_app_pr_n = set(map(tuple, ai_op_native[ai_op_native.approve][["repo", "number"]].drop_duplicates().to_numpy()))
    ai_opin_pr_n = set(map(tuple, ai_op_native[["repo", "number"]].drop_duplicates().to_numpy()))
    ai_op_first_n = ai_op_native.groupby(["repo", "number"]).t.min()

    claude_op = ai_op[ai_op.is_claude]
    claude_app_pr = set(map(tuple, claude_op[claude_op.approve][["repo", "number"]].drop_duplicates().to_numpy()))
    claude_op_pr = set(map(tuple, claude_op[["repo", "number"]].drop_duplicates().to_numpy()))
    claude_first_t = claude_op.groupby(["repo", "number"]).t.min()

    claude_op_native = claude_op[claude_op.is_native]
    claude_app_pr_n = set(map(tuple, claude_op_native[claude_op_native.approve][["repo", "number"]].drop_duplicates().to_numpy()))
    claude_op_pr_n = set(map(tuple, claude_op_native[["repo", "number"]].drop_duplicates().to_numpy()))
    claude_first_t_n = claude_op_native.groupby(["repo", "number"]).t.min()

    # ---------- Human reviews ----------
    hu_engage = ev[
        (ev.actor_type == "human")
        & ev.event_type.isin([
            "review_commented", "review_approved", "review_changes_requested",
            "review_comment", "issue_comment",
        ])
    ]
    hu_engage_first = hu_engage.groupby(["repo", "number"]).t.min()  # noqa: F841

    hu_expl = ev[
        (ev.actor_type == "human")
        & ev.event_type.isin(["review_approved", "review_changes_requested"])
    ]
    hu_expl_pr = set(map(tuple, hu_expl[["repo", "number"]].drop_duplicates().to_numpy()))
    hu_app_pr = set(map(tuple, hu_expl[hu_expl.event_type == "review_approved"][["repo", "number"]].drop_duplicates().to_numpy()))
    hu_expl_first = hu_expl.groupby(["repo", "number"]).t.min()

    # ---------- No-commits-between filter (same as 06c/06d) ----------
    commits = ev[ev.event_type == "commit"][["repo", "number", "t"]]
    commits_by_pr = commits.groupby(["repo", "number"]).t.apply(list).to_dict()

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
    quarter_lookup = summ_idx["quarter"].to_dict()

    is_claude_auth = (summ_idx.author_type == "AI") & (summ_idx.author_family == "claude")
    is_human_auth = (summ_idx.author_type == "human")
    claude_auth_pr = set(summ_idx[is_claude_auth].index)
    human_auth_pr = set(summ_idx[is_human_auth].index)

    # ============================================================
    # DiD-1 (cross-family): stable doubly-engaged cohort,
    # AI reviewer (any) vs human reviewer, AI author vs H author.
    # ============================================================
    cf_full = ai_opin_pr & hu_expl_pr
    cf_stable = {
        k for k in cf_full
        if has_commit_between(k, ai_op_first.get(k), hu_expl_first.get(k)) is False
    }
    cf_sub = summ_idx.loc[list(cf_stable)]
    cf_AI = set(cf_sub[cf_sub.author_type == "AI"].index)
    cf_H = set(cf_sub[cf_sub.author_type == "human"].index)

    # Native-only variant of the cross-family cohort (AI side restricted to
    # native APPROVED / CHANGES_REQUESTED states; human side already only
    # counts native explicit reviews, so this is the symmetric-pool check).
    cf_full_n = ai_opin_pr_n & hu_expl_pr
    cf_stable_n = {
        k for k in cf_full_n
        if has_commit_between(k, ai_op_first_n.get(k), hu_expl_first.get(k)) is False
    }
    cf_sub_n = summ_idx.loc[list(cf_stable_n)]
    cf_AI_n = set(cf_sub_n[cf_sub_n.author_type == "AI"].index)
    cf_H_n = set(cf_sub_n[cf_sub_n.author_type == "human"].index)

    # ============================================================
    # DiD-2 (within-family Claude): cohort with Claude opinion AND
    # human explicit review; stable filter on Claude-vs-human review;
    # author restricted to Claude or human.
    # ============================================================
    wf_full = claude_op_pr & hu_expl_pr
    wf_stable_all = {
        k for k in wf_full
        if has_commit_between(k, claude_first_t.get(k), hu_expl_first.get(k)) is False
    }
    wf_stable = wf_stable_all & (claude_auth_pr | human_auth_pr)
    wf_C = wf_stable & claude_auth_pr
    wf_H = wf_stable & human_auth_pr

    # Native-only Claude pool.
    wf_full_n = claude_op_pr_n & hu_expl_pr
    wf_stable_all_n = {
        k for k in wf_full_n
        if has_commit_between(k, claude_first_t_n.get(k), hu_expl_first.get(k)) is False
    }
    wf_stable_n = wf_stable_all_n & (claude_auth_pr | human_auth_pr)
    wf_C_n = wf_stable_n & claude_auth_pr
    wf_H_n = wf_stable_n & human_auth_pr

    # ---------- Bucket by quarter ----------
    quarters = sorted({quarter_lookup.get(k, "") for k in (cf_stable | wf_stable) if quarter_lookup.get(k)})
    logger.info("Quarters present: %s", quarters)

    cross_family_quarters = []
    within_family_quarters = []
    cross_family_native_quarters = []
    within_family_native_quarters = []

    def _cf_block(q: str, AI_set: set, H_set: set,
                  ai_app: set, hu_app: set) -> dict:
        AI_q = {k for k in AI_set if quarter_lookup.get(k) == q}
        H_q  = {k for k in H_set  if quarter_lookup.get(k) == q}
        AA_n, HA_n = len(AI_q), len(AI_q)
        AH_n, HH_n = len(H_q),  len(H_q)
        AA_k = len(AI_q & ai_app)
        HA_k = len(AI_q & hu_app)
        AH_k = len(H_q  & ai_app)
        HH_k = len(H_q  & hu_app)
        b = _did_with_ci(
            AA_k=AA_k, AA_n=AA_n, HA_k=HA_k, HA_n=HA_n,
            AH_k=AH_k, AH_n=AH_n, HH_k=HH_k, HH_n=HH_n,
        )
        b.update({
            "quarter": q,
            "n": int(AA_n + AH_n),
            "n_AI_auth": int(AA_n),
            "n_H_auth": int(AH_n),
        })
        return b

    def _wf_block(q: str, C_set: set, H_set: set,
                  c_app: set, hu_app: set) -> dict:
        C_q = {k for k in C_set if quarter_lookup.get(k) == q}
        H_q = {k for k in H_set if quarter_lookup.get(k) == q}
        CC_n, HC_n = len(C_q), len(C_q)
        CH_n, HH2_n = len(H_q), len(H_q)
        CC_k = len(C_q & c_app)
        HC_k = len(C_q & hu_app)
        CH_k = len(H_q & c_app)
        HH2_k = len(H_q & hu_app)
        b = _did_with_ci(
            AA_k=CC_k, AA_n=CC_n, HA_k=HC_k, HA_n=HC_n,
            AH_k=CH_k, AH_n=CH_n, HH_k=HH2_k, HH_n=HH2_n,
        )
        b.update({
            "quarter": q,
            "n": int(CC_n + CH_n),
            "n_C_auth": int(CC_n),
            "n_H_auth": int(CH_n),
        })
        return b

    for q in quarters:
        # Cross-family — full pool (native + parsed)
        cf = _cf_block(q, cf_AI, cf_H, ai_app_pr, hu_app_pr)
        cross_family_quarters.append(cf)
        logger.info(
            "[%s] CF (full):    n=%d (AI=%d, H=%d) DiD=%+.2fpp [%+.2f, %+.2f]",
            q, cf["n"], cf["n_AI_auth"], cf["n_H_auth"],
            cf["did_pp"], cf["did_ci_lo_pp"], cf["did_ci_hi_pp"],
        )
        # Cross-family — native-only AI pool
        cfn = _cf_block(q, cf_AI_n, cf_H_n, ai_app_pr_n, hu_app_pr)
        cross_family_native_quarters.append(cfn)
        logger.info(
            "[%s] CF (native):  n=%d (AI=%d, H=%d) DiD=%+.2fpp [%+.2f, %+.2f]",
            q, cfn["n"], cfn["n_AI_auth"], cfn["n_H_auth"],
            cfn["did_pp"], cfn["did_ci_lo_pp"], cfn["did_ci_hi_pp"],
        )

        # Within-family — full Claude pool
        wf = _wf_block(q, wf_C, wf_H, claude_app_pr, hu_app_pr)
        within_family_quarters.append(wf)
        logger.info(
            "[%s] WF (full):    n=%d (Claude=%d, H=%d) DiD=%+.2fpp [%+.2f, %+.2f]",
            q, wf["n"], wf["n_C_auth"], wf["n_H_auth"],
            wf["did_pp"], wf["did_ci_lo_pp"], wf["did_ci_hi_pp"],
        )
        # Within-family — native-only Claude pool
        wfn = _wf_block(q, wf_C_n, wf_H_n, claude_app_pr_n, hu_app_pr)
        within_family_native_quarters.append(wfn)
        logger.info(
            "[%s] WF (native):  n=%d (Claude=%d, H=%d) DiD=%+.2fpp [%+.2f, %+.2f]",
            q, wfn["n"], wfn["n_C_auth"], wfn["n_H_auth"],
            wfn["did_pp"], wfn["did_ci_lo_pp"], wfn["did_ci_hi_pp"],
        )

    # ---------- Pooled-across-quarters native-only DiD (for figure) ----------
    # Same 2x2 cells as 06c's `did` block, but computed on the native-only AI
    # opinion pool (APPROVED / CHANGES_REQUESTED states only — symmetric with
    # the human side).
    AA_n_p, HA_n_p = len(cf_AI_n), len(cf_AI_n)
    AH_n_p, HH_n_p = len(cf_H_n),  len(cf_H_n)
    AA_k_p = len(cf_AI_n & ai_app_pr_n)
    HA_k_p = len(cf_AI_n & hu_app_pr)
    AH_k_p = len(cf_H_n  & ai_app_pr_n)
    HH_k_p = len(cf_H_n  & hu_app_pr)
    cf_native_pooled = _did_with_ci(
        AA_k=AA_k_p, AA_n=AA_n_p, HA_k=HA_k_p, HA_n=HA_n_p,
        AH_k=AH_k_p, AH_n=AH_n_p, HH_k=HH_k_p, HH_n=HH_n_p,
    )
    cf_native_pooled.update({
        "n": int(AA_n_p + AH_n_p),
        "n_AI_auth": int(AA_n_p),
        "n_H_auth": int(AH_n_p),
    })
    logger.info(
        "Pooled CF (native): n=%d (AI=%d, H=%d) DiD=%+.2fpp [%+.2f, %+.2f]",
        cf_native_pooled["n"], cf_native_pooled["n_AI_auth"],
        cf_native_pooled["n_H_auth"], cf_native_pooled["did_pp"],
        cf_native_pooled["did_ci_lo_pp"], cf_native_pooled["did_ci_hi_pp"],
    )

    out = {
        "cross_family_by_quarter": cross_family_quarters,
        "within_family_by_quarter": within_family_quarters,
        "cross_family_native_by_quarter": cross_family_native_quarters,
        "within_family_native_by_quarter": within_family_native_quarters,
        "cross_family_native_pooled": cf_native_pooled,
        "generated_at": utils.now_iso(),
    }
    (RESULTS_DIR / "within_pr_quarterly.json").write_text(json.dumps(out, indent=2))
    logger.info("=== QUARTERLY DiD DONE ===")


if __name__ == "__main__":
    main()

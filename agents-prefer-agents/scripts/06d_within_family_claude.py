"""Within-family AI-AI bias test: does Claude prefer Claude-authored code?

This is the within-family sibling to 06c_within_pr.py. We re-use the §8.7
stable doubly-engaged cohort definition and stratify by *family*:

    author_family   ∈ {claude, human}
    reviewer_family ∈ {claude, human}

We compute one DiD:

    DiD_CvsH  = [Claude_rev gap on Claude-auth vs H-auth]
              - [human_rev gap on the same author split]
        > 0  ⇒  Claude prefers Claude-authored more than humans do.

The Claude opinion pool combines native APPROVED/CHANGES_REQUESTED states with
verdict-line regex parses on Claude review BODIES *and* Claude PR-thread
issue_comments — see ai_verdict_parser.py for the per-bot regex catalogue.

Outputs results/within_family_stats.json with one top-level block:

    claude_vs_human:   cohort, cells, brackets, DiD, logit interaction
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
utils = import_module("99_utils")
get_logger = utils.get_logger
DATA_DIR = utils.DATA_DIR
RESULTS_DIR = utils.RESULTS_DIR
SUBPROJECT_ROOT = utils.SUBPROJECT_ROOT

from ai_verdict_parser import parse_ai_opinions

logger = get_logger("06d_within_family_claude")

# What login substrings count as "Claude bot" for this analysis. These are the
# logins the AI opinion table emits for Claude (see ai_verdict_parser.py and
# lib/ai_detection.py allowlist). All comparisons are .lower().
CLAUDE_BOT_LOGIN_SUBS = ("claude", "anthropic-ai", "anthropic-code-agent")


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _cell(numer: set, denom: set) -> tuple[int, int, float]:
    n = len(denom)
    k = len(denom & numer)
    return k, n, (k / n if n else 0.0)


def _is_claude_login(login: str) -> bool:
    lg = (login or "").lower()
    return any(s in lg for s in CLAUDE_BOT_LOGIN_SUBS)


def _logit_did(rows: list[tuple], label: str) -> dict:
    """Fit Pr(approve) ~ const + reviewer_treat + author_treat + interaction.

    `rows` is a list of (reviewer_treat, author_treat, approved) tuples — each
    PR contributes one row per reviewer type that engaged on it.
    """
    if not rows:
        return {"logit_delta": float("nan"), "logit_se": float("nan"),
                "logit_p": float("nan"), "logit_reviewer_treat": float("nan"),
                "logit_author_treat": float("nan"), "n_long": 0}
    df = pd.DataFrame(rows, columns=["reviewer_treat", "author_treat", "approved"])
    df["const"] = 1.0
    df["interaction"] = df.reviewer_treat * df.author_treat
    X = df[["const", "reviewer_treat", "author_treat", "interaction"]].astype(float)
    try:
        m = sm.Logit(df.approved.astype(float), X).fit(disp=0, cov_type="HC0")
    except Exception as exc:
        logger.warning("logit fit failed for %s: %s", label, exc)
        return {"logit_delta": float("nan"), "logit_se": float("nan"),
                "logit_p": float("nan"), "logit_reviewer_treat": float("nan"),
                "logit_author_treat": float("nan"), "n_long": int(len(df))}
    return {
        "logit_delta": float(m.params["interaction"]),
        "logit_se": float(m.bse["interaction"]),
        "logit_p": float(m.pvalues["interaction"]),
        "logit_reviewer_treat": float(m.params["reviewer_treat"]),
        "logit_author_treat": float(m.params["author_treat"]),
        "n_long": int(len(df)),
    }


def _did_block(name: str, *,
               C_C_k: int, C_C_n: int,
               C_X_k: int, C_X_n: int,
               R_C_k: int, R_C_n: int,
               R_X_k: int, R_X_n: int,
               logit: dict) -> dict:
    """Build a DiD result block.

    C_* = Claude reviewer cells; R_* = reference reviewer cells (other-AI or
    human). _C = Claude-authored; _X = comparison-author (non-Claude-AI or
    human).
    """
    def rate(k, n): return (k / n) if n else 0.0

    cells = {
        "C_rev_x_C_auth": {"k": C_C_k, "n": C_C_n, "rate": rate(C_C_k, C_C_n),
                           "ci_lo": _wilson(C_C_k, C_C_n)[0],
                           "ci_hi": _wilson(C_C_k, C_C_n)[1]},
        "C_rev_x_X_auth": {"k": C_X_k, "n": C_X_n, "rate": rate(C_X_k, C_X_n),
                           "ci_lo": _wilson(C_X_k, C_X_n)[0],
                           "ci_hi": _wilson(C_X_k, C_X_n)[1]},
        "R_rev_x_C_auth": {"k": R_C_k, "n": R_C_n, "rate": rate(R_C_k, R_C_n),
                           "ci_lo": _wilson(R_C_k, R_C_n)[0],
                           "ci_hi": _wilson(R_C_k, R_C_n)[1]},
        "R_rev_x_X_auth": {"k": R_X_k, "n": R_X_n, "rate": rate(R_X_k, R_X_n),
                           "ci_lo": _wilson(R_X_k, R_X_n)[0],
                           "ci_hi": _wilson(R_X_k, R_X_n)[1]},
    }
    claude_gap_pp = (rate(C_C_k, C_C_n) - rate(C_X_k, C_X_n)) * 100
    ref_gap_pp = (rate(R_C_k, R_C_n) - rate(R_X_k, R_X_n)) * 100
    did_pp = claude_gap_pp - ref_gap_pp
    block = {
        "name": name,
        "cells": cells,
        "claude_gap_pp": float(claude_gap_pp),
        "ref_gap_pp": float(ref_gap_pp),
        "did_pp": float(did_pp),
    }
    block.update(logit)
    return block


def main():
    ev = pd.read_parquet(DATA_DIR / "pr_events.parquet")
    summ = pd.read_parquet(DATA_DIR / "pr_summary.parquet")
    ev["t"] = pd.to_datetime(ev.timestamp)
    summ["created_t"] = pd.to_datetime(summ.created_at)

    v2_repos = set(summ.repo)
    logger.info("Universe: %d PRs across %d repos.", len(summ), len(v2_repos))

    # ---------- 1. AI opinion table (native + parsed across all bots) ----------
    prs_dir = SUBPROJECT_ROOT / "data" / "prs"
    ai_op = parse_ai_opinions(prs_dir, repos=v2_repos)
    if ai_op.empty:
        logger.error("No AI opinion events parsed; aborting RQ3b.")
        return
    ai_op["bot_l"] = ai_op.bot.str.lower()
    ai_op["is_claude"] = ai_op.bot_l.apply(_is_claude_login)
    ai_op["approve"] = ai_op.kind.str.startswith("approve")

    # Per-PR Claude opinion
    claude_op = ai_op[ai_op.is_claude]
    claude_app_pr = set(map(tuple, claude_op[claude_op.approve][["repo", "number"]].drop_duplicates().to_numpy()))
    claude_op_pr = set(map(tuple, claude_op[["repo", "number"]].drop_duplicates().to_numpy()))
    claude_first_t = claude_op.groupby(["repo", "number"]).t.min()

    logger.info(
        "Opinion pool: Claude PRs=%d (approve=%d)",
        len(claude_op_pr), len(claude_app_pr),
    )

    # ---------- 2. Human explicit reviews (same as 06c) ----------
    hu_expl = ev[(ev.actor_type == "human")
                 & ev.event_type.isin(["review_approved", "review_changes_requested"])]
    hu_expl_pr = set(map(tuple, hu_expl[["repo", "number"]].drop_duplicates().to_numpy()))
    hu_app_pr = set(map(tuple, hu_expl[hu_expl.event_type == "review_approved"][["repo", "number"]].drop_duplicates().to_numpy()))
    hu_expl_first = hu_expl.groupby(["repo", "number"]).t.min()

    # ---------- 3. Stable cohort filter (no commits between AI verdict & human review) ----------
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
    # Restrict author identification to current allowlist + family attribution
    # (set by 03_classify_prs.py). For "claude" we match author_family literally.
    is_claude_auth = (summ_idx.author_type == "AI") & (summ_idx.author_family == "claude")
    is_human_auth = (summ_idx.author_type == "human")
    claude_auth_pr = set(summ_idx[is_claude_auth].index)
    human_auth_pr = set(summ_idx[is_human_auth].index)
    logger.info(
        "Author universe: Claude=%d, Human=%d",
        len(claude_auth_pr), len(human_auth_pr),
    )

    # ===================================================================
    # DiD: Claude reviewer vs HUMAN reviewer
    # Universe: PRs with Claude opinion AND human explicit review (same stable
    # filter on Claude verdict ↔ human review). Author restricted to Claude vs
    # human.
    # ===================================================================
    cohort2 = claude_op_pr & hu_expl_pr
    cohort2_stable_all = {
        k for k in cohort2
        if has_commit_between(k, claude_first_t.get(k), hu_expl_first.get(k)) is False
    }
    # Restrict the within-family cohort to PRs whose author is unambiguously
    # Claude or human; PRs authored by other AI families don't fit either side
    # of the Claude-vs-human comparison and are excluded from this analysis
    # (the cross-family DiD in 06c keeps them).
    cohort2_stable = cohort2_stable_all & (claude_auth_pr | human_auth_pr)
    n_other_auth_excl = len(cohort2_stable_all) - len(cohort2_stable)
    logger.info(
        "DiD cohort (Claude+HumanExpl, stable): n=%d "
        "(excluded %d non-Claude-non-human-authored PRs from within-family analysis)",
        len(cohort2_stable), n_other_auth_excl,
    )
    cohort2_C = cohort2_stable & claude_auth_pr
    cohort2_H = cohort2_stable & human_auth_pr
    C_C_k2, C_C_n2, _ = _cell(claude_app_pr, cohort2_C)
    C_H_k2, C_H_n2, _ = _cell(claude_app_pr, cohort2_H)
    H_C_k2, H_C_n2, _ = _cell(hu_app_pr, cohort2_C)
    H_H_k2, H_H_n2, _ = _cell(hu_app_pr, cohort2_H)

    rows2 = []
    for k in cohort2_stable:
        if k in cohort2_C:
            auth = 1
        elif k in cohort2_H:
            auth = 0
        else:
            continue
        rows2.append((1, auth, 1 if k in claude_app_pr else 0))   # Claude reviewer
        rows2.append((0, auth, 1 if k in hu_app_pr else 0))       # Human reviewer
    logit2 = _logit_did(rows2, "DiD-CvsH")
    block2 = _did_block(
        "claude_vs_human",
        C_C_k=C_C_k2, C_C_n=C_C_n2, C_X_k=C_H_k2, C_X_n=C_H_n2,
        R_C_k=H_C_k2, R_C_n=H_C_n2, R_X_k=H_H_k2, R_X_n=H_H_n2,
        logit=logit2,
    )
    block2.update({
        "cohort_n": int(len(cohort2_stable)),
        "n_C_auth": int(len(cohort2_C)),
        "n_H_auth": int(len(cohort2_H)),
        "n_other_auth_excluded": int(n_other_auth_excl),
        "cohort_n_pre_author_filter": int(len(cohort2_stable_all)),
    })
    logger.info(
        "DiD (Claude rev vs Human rev on Claude- vs Human-author): "
        "claude_gap=%+.2fpp ref_gap=%+.2fpp DiD=%+.2fpp delta=%+.4f p=%.4g",
        block2["claude_gap_pp"], block2["ref_gap_pp"], block2["did_pp"],
        block2["logit_delta"], block2["logit_p"],
    )

    out = {
        "claude_vs_human": block2,
        "claude_opinion_pool": {
            "n_unique_claude_prs": int(len(claude_op_pr)),
            "n_claude_approve_prs": int(len(claude_app_pr)),
            "n_claude_event_kinds": ai_op[ai_op.is_claude].kind.value_counts().to_dict(),
        },
        "generated_at": utils.now_iso(),
    }

    (RESULTS_DIR / "within_family_stats.json").write_text(json.dumps(out, indent=2, default=str))
    logger.info("=== WITHIN-FAMILY DiD DONE ===")


if __name__ == "__main__":
    main()

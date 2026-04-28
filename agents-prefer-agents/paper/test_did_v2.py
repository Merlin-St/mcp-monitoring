#!/usr/bin/env python3
"""
test_did_v2.py — apply the AI-verdict regex parser (test_ai_verdict_regex.py)
to the v2 dataset and compute the DiD statistics for the alternative RQ3
designs (Stratum A, Stratum B, doubly-explicit-plus).

NOT part of the pipeline. Standalone, reproducible. Run with:
    source /home/ubuntu/mcp-monitoring/.venv/bin/activate
    python paper/test_did_v2.py

Inputs:  data/pr_events.parquet, data/pr_summary.parquet, data/prs/*.jsonl
Outputs: stdout summary; intermediate /tmp/v2_ai_opinions.parquet

Notes:
- v2 is the criticality-based reselection (1,219 repos, 487k PRs).
- AI opinions = native APPROVED/CHANGES_REQUESTED + regex-parsed verdicts
  from COMMENTED bodies (per test_ai_verdict_regex.BOT_PATTERNS).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

PAPER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_DIR))
from test_ai_verdict_regex import AI_BOTS_SUBS, classify  # noqa: E402

ROOT = PAPER_DIR.parent
DATA = ROOT / "data"
PRS_DIR = DATA / "prs"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fmt(k: int, n: int) -> str:
    if n == 0:
        return f"NA (n=0)"
    lo, hi = wilson(k, n)
    return f"{100*k/n:.2f}% [{100*lo:.2f}, {100*hi:.2f}] (n={n}, k={k})"


def two_prop_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else float("nan")
    return z, 2 * (1 - norm.cdf(abs(z)))


def build_ai_opinions(v2_repos: set[str]) -> pd.DataFrame:
    rows = []
    files_seen = 0
    for fp in sorted(PRS_DIR.glob("*.jsonl")):
        repo_guess = fp.stem.replace("__", "/")
        if repo_guess not in v2_repos:
            continue
        files_seen += 1
        with open(fp) as f:
            for line in f:
                d = json.loads(line)
                repo, num = d["repo"], d["number"]
                for r in d.get("reviews", []):
                    login = (r.get("author_login") or "").lower()
                    state = r.get("state") or ""
                    body = (r.get("body") or "").strip()
                    ts = r.get("submitted_at")
                    if not any(s in login for s in AI_BOTS_SUBS):
                        continue
                    kind = None
                    if state == "APPROVED":
                        kind = "approve_native"
                    elif state == "CHANGES_REQUESTED":
                        kind = "reject_native"
                    elif state == "COMMENTED" and body:
                        v = classify(login, body)
                        if v == "approve":
                            kind = "approve_parsed"
                        elif v == "reject":
                            kind = "reject_parsed"
                    if kind:
                        rows.append((repo, num, ts, kind, login))
    print(f"# Scanned {files_seen} jsonl files (v2 repos only)")
    df = pd.DataFrame(rows, columns=["repo", "number", "t", "kind", "bot"])
    df["t"] = pd.to_datetime(df.t)
    return df


def main() -> None:
    print("=== test_did_v2.py — DiD on v2 with native + parsed AI opinions ===\n")
    summ = pd.read_parquet(DATA / "pr_summary.parquet")
    ev = pd.read_parquet(DATA / "pr_events.parquet")
    ev["t"] = pd.to_datetime(ev.timestamp)

    v2_repos = set(summ.repo)
    print(f"v2 dataset: {len(summ):,} PRs, {len(v2_repos):,} repos, {len(ev):,} events")
    print(f"v2 author types:\n{summ.author_type.value_counts().to_string()}\n")

    # 1) Build AI opinion table
    cache = Path("/tmp/v2_ai_opinions.parquet")
    if cache.exists():
        print(f"Loading cached AI opinions from {cache}")
        ai_op = pd.read_parquet(cache)
    else:
        ai_op = build_ai_opinions(v2_repos)
        ai_op.to_parquet(cache)
        print(f"Cached to {cache}")

    print(f"\nAI opinion events (native + parsed): {len(ai_op):,}")
    print(ai_op.kind.value_counts().to_string())
    n_unique = ai_op.groupby(["repo", "number"]).ngroups
    print(f"Unique PRs with at least one AI opinion: {n_unique:,}")

    # Per-PR AI verdict (any approve event takes precedence over reject)
    ai_app_pr = set(map(tuple, ai_op[ai_op.kind.str.startswith("approve")][["repo","number"]].drop_duplicates().to_numpy()))
    ai_rej_pr = set(map(tuple, ai_op[ai_op.kind.str.startswith("reject")][["repo","number"]].drop_duplicates().to_numpy()))
    ai_opin_pr = ai_app_pr | ai_rej_pr

    # First AI opinion timestamp
    ai_op_first = ai_op.groupby(["repo", "number"]).t.min()

    # Human engagement (any comment/review)
    hu_engage = ev[(ev.actor_type == "human") & ev.event_type.isin(
        ["review_commented","review_approved","review_changes_requested","review_comment","issue_comment"]
    )]
    hu_engage_first = hu_engage.groupby(["repo", "number"]).t.min()

    # Human explicit review state
    hu_expl = ev[(ev.actor_type == "human") & ev.event_type.isin(
        ["review_approved", "review_changes_requested"]
    )]
    hu_expl_pr = set(map(tuple, hu_expl[["repo","number"]].drop_duplicates().to_numpy()))
    hu_app_pr = set(map(tuple, hu_expl[hu_expl.event_type == "review_approved"][["repo","number"]].drop_duplicates().to_numpy()))

    # AI-bot any review event (for Stratum A exclusion)
    ai_any = ev[(ev.actor_type == "AI-bot") & ev.event_type.isin(
        ["review_commented","review_approved","review_changes_requested"]
    )]
    ai_any_set = set(map(tuple, ai_any[["repo","number"]].drop_duplicates().to_numpy()))

    summ_idx = summ.set_index(["repo", "number"])

    # =========================================================================
    # STRATUM A: no AI-bot reviewer, with ≥1 explicit human review
    # =========================================================================
    A_keys = set(summ_idx.index) - ai_any_set
    A_keys = A_keys & hu_expl_pr
    A_sub = summ_idx.loc[list(A_keys)]
    A_AI = set(A_sub[A_sub.author_type == "AI"].index)
    A_H = set(A_sub[A_sub.author_type == "human"].index)
    A_AI_k = len(A_AI & hu_app_pr)
    A_H_k = len(A_H & hu_app_pr)
    A_gap_pp = (A_AI_k / max(len(A_AI), 1) - A_H_k / max(len(A_H), 1)) * 100
    zA, pA = two_prop_z(A_AI_k, len(A_AI), A_H_k, len(A_H))

    print(f"\n=== STRATUM A (no AI-bot reviewer + human explicit) ===")
    print(f"  Total: n={len(A_AI) + len(A_H):,}")
    print(f"  AI-author human-approve: {fmt(A_AI_k, len(A_AI))}")
    print(f"  H-author  human-approve: {fmt(A_H_k,  len(A_H))}")
    print(f"  Gap (AI − H): {A_gap_pp:+.2f} pp; z={zA:.3f}, p={pA:.4f}")

    # =========================================================================
    # STRATUM B: AI clear opinion before any human engagement, + ≥1 human explicit
    # =========================================================================
    both = ai_op_first.to_frame("ai_op").join(hu_engage_first.to_frame("hu_first"), how="left")
    mask = both.hu_first.notna() & (both.ai_op < both.hu_first)
    B_candidates = set(map(tuple, both[mask].reset_index()[["repo","number"]].to_numpy()))
    B_keys = B_candidates & hu_expl_pr
    B_sub = summ_idx.loc[list(B_keys)]
    B_AI = set(B_sub[B_sub.author_type == "AI"].index)
    B_H = set(B_sub[B_sub.author_type == "human"].index)
    B_AI_k = len(B_AI & hu_app_pr)
    B_H_k = len(B_H & hu_app_pr)
    B_gap_pp = (B_AI_k / max(len(B_AI), 1) - B_H_k / max(len(B_H), 1)) * 100
    zB, pB = two_prop_z(B_AI_k, len(B_AI), B_H_k, len(B_H))

    print(f"\n=== STRATUM B (AI clear opinion before human + human explicit) ===")
    print(f"  Total: n={len(B_AI) + len(B_H):,}")
    print(f"  AI-author human-approve: {fmt(B_AI_k, len(B_AI))}")
    print(f"  H-author  human-approve: {fmt(B_H_k,  len(B_H))}")
    print(f"  Gap (AI − H): {B_gap_pp:+.2f} pp; z={zB:.3f}, p={pB:.4f}")

    DiD_anchor = B_gap_pp - A_gap_pp
    print(f"\n=== Cross-stratum DiD (B − A) = anchoring effect ===")
    print(f"  {DiD_anchor:+.2f} pp")

    # =========================================================================
    # OPTION B-PLUS: doubly-explicit (AI opinion AND human explicit) DiD
    #
    # Computed two ways:
    #   (a) all doubly-engaged PRs
    #   (b) STABLE cohort: only PRs with no commits between the AI's first
    #       opinion and the human's first explicit review. Without this filter,
    #       AI may be reviewing version V1 and the human may be reviewing V2 —
    #       they're not evaluating the same code. The stable cohort restricts
    #       to PRs where both reviewers saw the same commit graph.
    # =========================================================================
    hu_expl_first = hu_expl.groupby(["repo", "number"]).t.min()
    commits = ev[ev.event_type == "commit"][["repo", "number", "t"]]
    commits_by_pr = commits.groupby(["repo", "number"]).t.apply(list).to_dict()

    def changed_between(key):
        tai = ai_op_first.get(key)
        thu = hu_expl_first.get(key)
        if pd.isna(tai) or pd.isna(thu):
            return None
        lo, hi = (tai, thu) if tai <= thu else (thu, tai)
        for c in commits_by_pr.get(key, []):
            if lo < c < hi:
                return True
        return False

    both_set_all = ai_opin_pr & hu_expl_pr
    sub = summ_idx.loc[list(both_set_all)]
    both_AI = set(sub[sub.author_type == "AI"].index)
    both_H = set(sub[sub.author_type == "human"].index)

    def report_did(label: str, cohort: set):
        sub = summ_idx.loc[list(cohort)]
        AI = set(sub[sub.author_type == "AI"].index)
        H = set(sub[sub.author_type == "human"].index)

        def cell(rev_app, denom):
            inter = denom & cohort
            return len(inter & rev_app), len(inter)

        c_AI_AI = cell(ai_app_pr, AI)
        c_AI_H = cell(ai_app_pr, H)
        c_H_AI = cell(hu_app_pr, AI)
        c_H_H = cell(hu_app_pr, H)

        print(f"\n=== {label} ===")
        print(f"  Total: n={len(AI) + len(H):,} (AI-author {len(AI):,}, H-author {len(H):,})")
        print(f"  AI reviewer × AI author:    {fmt(*c_AI_AI)}")
        print(f"  AI reviewer × H author:     {fmt(*c_AI_H)}")
        print(f"  Human reviewer × AI author: {fmt(*c_H_AI)}")
        print(f"  Human reviewer × H author:  {fmt(*c_H_H)}")
        ai_gap = (c_AI_AI[0] / c_AI_AI[1] - c_AI_H[0] / c_AI_H[1]) * 100
        hu_gap = (c_H_AI[0] / c_H_AI[1] - c_H_H[0] / c_H_H[1]) * 100
        DiD = ai_gap - hu_gap
        print(f"  AI-side gap:    {ai_gap:+.2f} pp")
        print(f"  Human-side gap: {hu_gap:+.2f} pp")
        print(f"  DiD = AI-gap − Human-gap: {DiD:+.2f} pp")

        rows = []
        for k in cohort:
            if k in AI:
                auth = 1
            elif k in H:
                auth = 0
            else:
                continue
            rows.append((1, auth, 1, 1 if k in ai_app_pr else 0))
            rows.append((1, auth, 0, 1 if k in hu_app_pr else 0))
        df = pd.DataFrame(rows, columns=["const", "author_AI", "reviewer_AI", "approved"])
        df["interaction"] = df.author_AI * df.reviewer_AI
        X = df[["const", "reviewer_AI", "author_AI", "interaction"]].astype(float)
        m = sm.Logit(df.approved.astype(float), X).fit(disp=0, cov_type="HC0")
        print(f"  Logit DiD: delta={m.params['interaction']:.4f}, "
              f"SE={m.bse['interaction']:.4f}, p={m.pvalues['interaction']:.4g}")

    # (a) Full doubly-engaged cohort (does NOT control for code-version drift)
    report_did("OPTION B-PLUS (a): all doubly-engaged PRs", both_set_all)

    # (b) STABLE cohort: no commits between AI's first verdict and human's first review
    stable = {k for k in both_set_all if changed_between(k) is False}
    excluded = len(both_set_all) - len(stable)
    sub_excl = summ_idx.loc[list(both_set_all - stable)]
    excl_AI = (sub_excl.author_type == "AI").sum()
    excl_H = (sub_excl.author_type == "human").sum()
    print(f"\n  [Filter] Excluding PRs with commits between AI verdict and human review:")
    print(f"           {excluded:,} excluded (AI-author {excl_AI:,} / H-author {excl_H:,})")
    print(f"           Exclusion rate: AI={100*excl_AI/len(both_AI):.1f}%, "
          f"H={100*excl_H/len(both_H):.1f}%")

    report_did("OPTION B-PLUS (b): STABLE cohort (no commits between AI and human reviews)",
               stable)

    # =========================================================================
    # SECTION 8.7: Clean DiD aligned with the paper's rbt^A - rbt^H framework
    #
    # Same number as §8.5(b), reframed as the "AI prefers AI more than humans
    # do" minus "AI prefers humans less than humans do" decomposition. Maps
    # directly onto the paper's verbal description of $\rbt^A - \rbt^H$.
    # =========================================================================
    print("\n=== SECTION 8.7: Clean DiD aligned with paper's rbt^A - rbt^H framework ===")
    sub_stable = summ_idx.loc[list(stable)]
    stable_AI = set(sub_stable[sub_stable.author_type == "AI"].index)
    stable_H = set(sub_stable[sub_stable.author_type == "human"].index)
    print(f"  Cohort: §8.5(b) stable doubly-engaged. n={len(stable):,} "
          f"(AI-author {len(stable_AI):,}, H-author {len(stable_H):,})")

    def rate(rev_app, denom):
        return len(denom & rev_app), len(denom)

    AI_AI_k, AI_AI_n = rate(ai_app_pr, stable_AI)
    H_AI_k,  H_AI_n  = rate(hu_app_pr, stable_AI)
    AI_H_k,  AI_H_n  = rate(ai_app_pr, stable_H)
    H_H_k,   H_H_n   = rate(hu_app_pr, stable_H)

    # Bracket 1: "AI prefers AI more than humans do" on AI-authored
    b1 = (AI_AI_k/AI_AI_n - H_AI_k/H_AI_n) * 100
    # Bracket 2: "AI prefers humans less than humans do" on H-authored
    b2 = (AI_H_k/AI_H_n - H_H_k/H_H_n) * 100
    DiD = b1 - b2

    print(f"\n  Bracket 1 (AI-vs-Human approval on AI-authored PRs):")
    print(f"    P(AI app | AI-auth) - P(H app | AI-auth) "
          f"= {100*AI_AI_k/AI_AI_n:.2f}% - {100*H_AI_k/H_AI_n:.2f}% = {b1:+.2f} pp")
    print(f"    [if positive: AI prefers AI more than humans do]")
    print(f"  Bracket 2 (AI-vs-Human approval on H-authored PRs):")
    print(f"    P(AI app | H-auth) - P(H app | H-auth) "
          f"= {100*AI_H_k/AI_H_n:.2f}% - {100*H_H_k/H_H_n:.2f}% = {b2:+.2f} pp")
    print(f"    [if negative: AI prefers humans less than humans do]")
    print(f"\n  Estimate of rbt^A - rbt^H:  bracket1 - bracket2 = {DiD:+.2f} pp")
    print(f"  [positive => AI-AI bias; negative => anti-self-preference]")

    # =========================================================================
    # AI-only review (no human) — context stat
    # =========================================================================
    ai_only_no_expl = ai_any_set - hu_expl_pr
    ai_only_no_engage = ai_any_set - set(hu_engage_first.index)
    print(f"\n=== AI-only-reviewed PRs (context for why we condition on human review) ===")
    print(f"  AI-bot reviewed:                      {len(ai_any_set):,}")
    print(f"  ...without any human EXPLICIT review: {len(ai_only_no_expl):,}  ({100*len(ai_only_no_expl)/len(ai_any_set):.1f}%)")
    print(f"  ...without ANY human engagement:      {len(ai_only_no_engage):,}  ({100*len(ai_only_no_engage)/len(ai_any_set):.1f}%)")


if __name__ == "__main__":
    main()

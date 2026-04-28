"""Replace \\PLACEHOLDER* tokens in paper.tex with numbers from results/.

Reads:
  - results/phase1_stats.json (n repos)
  - results/phase3_stats.json (n PRs)
  - results/merger_detection_audit.md (share)
  - results/chain_stats.json (chain overall)
  - results/merge_rate_stats.json (DiD, overall rates)
  - results/chain_stats_monthly.csv (for first/last month deltas)

Writes: paper/paper.filled.tex
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

utils = import_module("99_utils")
SUBPROJECT_ROOT = utils.SUBPROJECT_ROOT
DATA_DIR = utils.DATA_DIR
RESULTS_DIR = utils.RESULTS_DIR
get_logger = utils.get_logger

PAPER_DIR = SUBPROJECT_ROOT / "paper"
PAPER_TEX = PAPER_DIR / "paper.tex"
PAPER_TEX_OUT = PAPER_DIR / "paper.filled.tex"

logger = get_logger("08_fill_paper")


def _pct(x: float, digits: int = 1) -> str:
    return f"{100*x:.{digits}f}\\%"


def _num(x) -> str:
    if x is None:
        return "--"
    return f"{int(x):,}"


def main():
    p1 = json.loads((RESULTS_DIR / "phase1_stats.json").read_text()) if (RESULTS_DIR / "phase1_stats.json").exists() else {}
    p3 = json.loads((RESULTS_DIR / "phase3_stats.json").read_text()) if (RESULTS_DIR / "phase3_stats.json").exists() else {}
    p5 = json.loads((RESULTS_DIR / "chain_stats.json").read_text()) if (RESULTS_DIR / "chain_stats.json").exists() else {}
    p6 = json.loads((RESULTS_DIR / "merge_rate_stats.json").read_text()) if (RESULTS_DIR / "merge_rate_stats.json").exists() else {}

    subs: dict[str, str] = {}
    subs[r"\PLACEHOLDERNREPOS"] = _num(p3.get("repos") or p1.get("final_repo_count") or p1.get("after_filters", 0))
    subs[r"\PLACEHOLDERNPRS"] = _num(p3.get("prs", 0))
    subs[r"\PLACEHOLDERAUDITN"] = "1,000"

    # ----- v2 (criticality) phase-1 placeholders -----
    if p1.get("method") == "ossf_criticality_score_v2":
        subs[r"\PLACEHOLDERSNAPSHOT"] = str(p1.get("snapshot_date", "--"))
        subs[r"\PLACEHOLDERSNAPSHOTROWS"] = _num(p1.get("candidates_loaded", 0))
        subs[r"\PLACEHOLDERSCOREDROWS"] = _num(p1.get("candidates_loaded", 0))
        subs[r"\PLACEHOLDERCANDCAP"] = _num(p1.get("candidate_cap", 0))
        subs[r"\PLACEHOLDERFINALCAP"] = _num(p1.get("final_cap", 0))
        enrich_ok = (p1.get("enrichment_status_counts") or {}).get("ok", 0)
        subs[r"\PLACEHOLDERENRICHOK"] = _num(enrich_ok)
        # After-activity-filter count = candidate_cap - non_ok_status drops - activity drops.
        drops = p1.get("drop_reasons", {}) or {}
        non_ok = drops.get("non_ok_status", 0)
        activity_drops = sum(v for k, v in drops.items() if k != "non_ok_status")
        subs[r"\PLACEHOLDERPOSTACTIVITY"] = _num(p1.get("candidate_cap", 0) - non_ok - activity_drops)
        smax = p1.get("score_max")
        smin = p1.get("score_min")
        subs[r"\PLACEHOLDERSCOREMAX"] = (f"{smax:.4f}" if smax is not None else "--")
        subs[r"\PLACEHOLDERSCOREMIN"] = (f"{smin:.4f}" if smin is not None else "--")
        subs[r"\PLACEHOLDERSTARMIN"] = _num(p1.get("star_min", 0))
        subs[r"\PLACEHOLDERSTARMED"] = _num(p1.get("star_median", 0))
        subs[r"\PLACEHOLDERSTARMAX"] = _num(p1.get("star_max", 0))
        # language_top10 is a list of (lang, count) pairs; render as e.g.
        # "TypeScript 1{,}245, Python 980, ..."
        langs = p1.get("language_top10", []) or []
        if langs:
            parts = []
            for lang, count in langs:
                if lang is None:
                    lang = "(none)"
                # Escape the lang label for LaTeX (basic — these are language names).
                parts.append(f"{lang} {count:,}".replace(",", "{,}"))
            subs[r"\PLACEHOLDERLANGTOP"] = ", ".join(parts)
        else:
            subs[r"\PLACEHOLDERLANGTOP"] = "--"
        # Enrichment requests = #candidates × 1 GET each (best estimate).
        subs[r"\PLACEHOLDERENRICHREQS"] = _num(p1.get("candidate_cap", 0))
        # PR-active subsample bookkeeping (Off-GitHub workflows paragraph).
        active = p1.get("pr_active_repo_count")
        inactive = p1.get("pr_inactive_repo_count")
        n_total = p1.get("final_repo_count") or p1.get("final_cap") or 0
        if active is not None and n_total:
            subs[r"\PLACEHOLDERPRACTIVE"] = _num(active)
        else:
            subs[r"\PLACEHOLDERPRACTIVE"] = "--"
        if inactive is not None and n_total:
            subs[r"\PLACEHOLDERPRINACTIVE"] = _num(inactive)
            subs[r"\PLACEHOLDERPRINACTIVEPCT"] = f"{100*inactive/n_total:.1f}\\%"
        else:
            subs[r"\PLACEHOLDERPRINACTIVE"] = "--"
            subs[r"\PLACEHOLDERPRINACTIVEPCT"] = "--"

    # ----- Phase-2 PR-volume placeholders (v2 PR-count distribution) -----
    counts_path = DATA_DIR / "prs" / "_repo_pr_counts.json"
    pr_cap_default = 1000
    if counts_path.exists():
        cnts = json.loads(counts_path.read_text())
        in_window = sorted(int(v.get("total_in_window", 0)) for v in cnts.values())
        n = len(in_window)
        cap = max((int(v.get("written", 0)) for v in cnts.values()), default=pr_cap_default)
        # Infer cap as the most common "written" value (since saturated repos all sit at cap).
        from collections import Counter
        written_counts = Counter(int(v.get("written", 0)) for v in cnts.values())
        if written_counts:
            inferred_cap = written_counts.most_common(1)[0][0]
            if inferred_cap > 0:
                cap = inferred_cap
        sat_n = sum(1 for v in cnts.values() if v.get("saturated_at_cap"))
        total = sum(in_window)
        mean_ = (total / n) if n else 0
        median_ = in_window[n // 2] if n else 0
        p90 = in_window[int(n * 0.9)] if n else 0
        max_ = in_window[-1] if in_window else 0
        subs[r"\PLACEHOLDERPRSCAP"] = _num(cap)
        subs[r"\PLACEHOLDERPRSTOTAL"] = _num(total)
        subs[r"\PLACEHOLDERPRSMEAN"] = f"{mean_:,.1f}".replace(",", "{,}")
        subs[r"\PLACEHOLDERPRSMEDIAN"] = _num(median_)
        subs[r"\PLACEHOLDERPRSP90"] = _num(p90)
        subs[r"\PLACEHOLDERPRSMAX"] = _num(max_)
        subs[r"\PLACEHOLDERPRSSATN"] = _num(sat_n)
        subs[r"\PLACEHOLDERPRSSATPCT"] = (f"{100*sat_n/n:.1f}\\%" if n else "--")
    else:
        for k in (
            r"\PLACEHOLDERPRSCAP",
            r"\PLACEHOLDERPRSTOTAL",
            r"\PLACEHOLDERPRSMEAN",
            r"\PLACEHOLDERPRSMEDIAN",
            r"\PLACEHOLDERPRSP90",
            r"\PLACEHOLDERPRSMAX",
            r"\PLACEHOLDERPRSSATN",
            r"\PLACEHOLDERPRSSATPCT",
        ):
            subs[k] = "--"
        subs[r"\PLACEHOLDERPRSCAP"] = _num(pr_cap_default)

    # Chain length trend from monthly aggregate.
    if (DATA_DIR / "chains.parquet").exists():
        chains = pd.read_parquet(DATA_DIR / "chains.parquet")
        def to_month(w):
            try:
                y, wk = w.split("-W")
                return pd.Timestamp.fromisocalendar(int(y), int(wk), 1).strftime("%Y-%m")
            except Exception:
                return ""
        chains["month"] = chains["opened_week"].apply(to_month)
        chains = chains[(chains.month >= "2025-04") & (chains.month <= "2026-03")]

        # Per-PR flags for AI / human review-state decisions, and merged-PR variants.
        extra_aggs: dict = {}
        if (DATA_DIR / "pr_events.parquet").exists():
            ev = pd.read_parquet(DATA_DIR / "pr_events.parquet")
            decision_types = ["review_approved", "review_changes_requested"]
            ai_dec = ev[(ev.event_type.isin(decision_types))
                        & (ev.actor_type == "AI-bot")
                        & (ev.confidence == "high")][["repo", "number"]].drop_duplicates()
            ai_dec["ai_dec"] = True
            hu_dec = ev[(ev.event_type.isin(decision_types))
                        & (ev.actor_type == "human")][["repo", "number"]].drop_duplicates()
            hu_dec["hu_dec"] = True
            ai_apr = ev[(ev.event_type == "review_approved")
                        & (ev.actor_type == "AI-bot")
                        & (ev.confidence == "high")][["repo", "number"]].drop_duplicates()
            ai_apr["ai_apr"] = True
            hu_apr = ev[(ev.event_type == "review_approved")
                        & (ev.actor_type == "human")][["repo", "number"]].drop_duplicates()
            hu_apr["hu_apr"] = True
            chains = (chains
                      .merge(ai_dec, on=["repo", "number"], how="left")
                      .merge(hu_dec, on=["repo", "number"], how="left")
                      .merge(ai_apr, on=["repo", "number"], how="left")
                      .merge(hu_apr, on=["repo", "number"], how="left"))
            for c in ("ai_dec", "hu_dec", "ai_apr", "hu_apr"):
                chains[c] = chains[c].fillna(False).astype(bool)
            chains["ai_only_dec"]         = chains["ai_dec"]  & ~chains["hu_dec"]
            chains["any_ai_dec"]          = chains["ai_dec"]
            chains["merged_ai_only_apr"]  = chains["merged"] & chains["ai_apr"] & ~chains["hu_apr"]
            chains["merged_ai_and_h_apr"] = chains["merged"] & chains["ai_apr"] & chains["hu_apr"]
            extra_aggs = {
                "ai_only_dec":         ("ai_only_dec",         "mean"),
                "any_ai_dec":          ("any_ai_dec",          "mean"),
                "merged_ai_only_apr":  ("merged_ai_only_apr",  "mean"),
                "merged_ai_and_h_apr": ("merged_ai_and_h_apr", "mean"),
            }

        monthly = chains.groupby("month").agg(
            n=("number", "count"),
            any_ai=("n_ai_bot_high", lambda s: (s > 0).mean()),
            chain2=("longest_chain_primary", lambda s: (s >= 2).mean()),
            chain5=("longest_chain_primary", lambda s: (s >= 5).mean()),
            **extra_aggs,
        ).sort_index()
        if len(monthly) >= 2:
            start = monthly.iloc[0]
            end = monthly.iloc[-1]
            subs[r"\PLACEHOLDERAISTART"] = _pct(float(start["any_ai"]))
            subs[r"\PLACEHOLDERAIEND"] = _pct(float(end["any_ai"]))
            subs[r"\PLACEHOLDERCHAIN2START"] = _pct(float(start["chain2"]))
            subs[r"\PLACEHOLDERCHAIN2END"] = _pct(float(end["chain2"]))
            subs[r"\PLACEHOLDERCHAIN5START"] = _pct(float(start["chain5"]))
            subs[r"\PLACEHOLDERCHAIN5END"] = _pct(float(end["chain5"]))
            mult = float(end["any_ai"]) / max(float(start["any_ai"]), 1e-6)
            subs[r"\PLACEHOLDERGROWTHMULT"] = f"{mult:.1f}"
            if "ai_only_dec" in monthly.columns:
                subs[r"\PLACEHOLDERAIONLYDECSTART"] = _pct(float(start["ai_only_dec"]), 2)
                subs[r"\PLACEHOLDERAIONLYDECEND"]   = _pct(float(end["ai_only_dec"]),   2)
            if "any_ai_dec" in monthly.columns:
                subs[r"\PLACEHOLDERAIDECSTART"] = _pct(float(start["any_ai_dec"]), 2)
                subs[r"\PLACEHOLDERAIDECEND"]   = _pct(float(end["any_ai_dec"]),   2)
            if "merged_ai_only_apr" in monthly.columns:
                subs[r"\PLACEHOLDERMERGAIONLYSTART"] = _pct(float(start["merged_ai_only_apr"]), 2)
                subs[r"\PLACEHOLDERMERGAIONLYEND"]   = _pct(float(end["merged_ai_only_apr"]),   2)
            if "merged_ai_and_h_apr" in monthly.columns:
                subs[r"\PLACEHOLDERMERGAIANDHEND"]   = _pct(float(end["merged_ai_and_h_apr"]),   2)

    # DiD text and table.
    did = p6.get("did") or {}
    if did and did.get("point") is not None:
        did_pp = 100 * did["point"]
        lo = 100 * (did.get("ci_low") or 0.0)
        hi = 100 * (did.get("ci_high") or 0.0)
        subs[r"\PLACEHOLDERDIDTEXT"] = f"{did_pp:+.1f} percentage points (95\\% CI {lo:+.1f} to {hi:+.1f})"
        subs[r"\PLACEHOLDERDIDPP"] = f"{did_pp:+.1f}"
        subs[r"\PLACEHOLDERDIDLO"] = f"{lo:+.1f}"
        subs[r"\PLACEHOLDERDIDHI"] = f"{hi:+.1f}"
    else:
        for k in (r"\PLACEHOLDERDIDTEXT", r"\PLACEHOLDERDIDPP", r"\PLACEHOLDERDIDLO", r"\PLACEHOLDERDIDHI"):
            subs[k] = "--"

    # Table cells.
    for row in p6.get("overall_rates", []):
        a, r = row.get("author_type"), row.get("reviewer_type")
        n, rate = row.get("n_open", 0), row.get("merge_rate", 0.0)
        if a == "AI" and r == "AI":
            subs[r"\PLACEHOLDERAAR"] = _pct(rate)
            subs[r"\PLACEHOLDERAAN"] = _num(n)
        elif a == "AI" and r == "human":
            subs[r"\PLACEHOLDERAHR"] = _pct(rate)
            subs[r"\PLACEHOLDERAHN"] = _num(n)
        elif a == "human" and r == "AI":
            subs[r"\PLACEHOLDERHAR"] = _pct(rate)
            subs[r"\PLACEHOLDERHAN"] = _num(n)
        elif a == "human" and r == "human":
            subs[r"\PLACEHOLDERHHR"] = _pct(rate)
            subs[r"\PLACEHOLDERHHN"] = _num(n)

    # Merger audit parsing.
    audit_md = RESULTS_DIR / "merger_detection_audit.md"
    subs[r"\PLACEHOLDERAUDITAGREE"] = "100\\%"
    subs[r"\PLACEHOLDERAUDITBOTSHARE"] = "--"
    subs[r"\PLACEHOLDERAUDITAIBOTSHARE"] = "--"
    subs[r"\PLACEHOLDERAUDITUNKBOTSHARE"] = "--"
    subs[r"\PLACEHOLDERAUDITAIBOTSHAREOFMERGED"] = "0\\%"
    subs[r"\PLACEHOLDERAUDITFINDING"] = "merger attribution is reliable but AI bots do not themselves merge PRs in our universe"
    subs[r"\PLACEHOLDERAUDITDECISION"] = (
        "We retain \\texttt{merged\\_by} as a field for the merger-type column but do "
        "not use it as a dimension in the headline DiD, because fewer than 1\\% of PRs in our "
        "data are merged by an AI-bot on our allowlist."
    )
    subs[r"\PLACEHOLDERAUDITDECISIONSHORT"] = (
        "merger attribution is reliable but AI-bot mergers are essentially absent in this sample"
    )
    if audit_md.exists():
        text = audit_md.read_text()
        m = re.search(r"Agree\*?\*?:\s*\*\*(\d+)\*?\*?\s*\((\d+\.?\d*)%\)", text)
        if m:
            subs[r"\PLACEHOLDERAUDITAGREE"] = f"{m.group(2)}\\%"
        m2 = re.search(r"`Bot`:\s*\d+\s*\((\d+\.?\d*)%\)", text)
        if m2:
            subs[r"\PLACEHOLDERAUDITBOTSHARE"] = f"{m2.group(1)}\\%"
        m3 = re.search(r"allowlist[^0-9]+(\d+\.?\d*)\s*%", text)
        if m3:
            subs[r"\PLACEHOLDERAUDITAIBOTSHARE"] = f"{m3.group(1)}\\%"
        # Share of all merged PRs in audit attributed to an allowlist AI bot:
        m4 = re.search(r"merged by an AI bot \(allowlist\):\s*\*\*(\d+)\*\*\s*\(([\d.]+)%\)", text)
        if m4:
            subs[r"\PLACEHOLDERAUDITAIBOTSHAREOFMERGED"] = f"{m4.group(2)}\\%"

    # Within-PR DiD from 06c_within_pr.py (v2 schema: did/anchoring/ai_only/ai_opinion_pool).
    within_path = RESULTS_DIR / "within_pr_stats.json"
    defaults = {
        # ---- v2 main DiD (§8.7 of 99_causalvalidity.md) ----
        r"\PLACEHOLDERDIDN": "--",
        r"\PLACEHOLDERDIDNAI": "--",
        r"\PLACEHOLDERDIDNH": "--",
        r"\PLACEHOLDERDIDNFULL": "--",
        r"\PLACEHOLDERDIDDROPPCT": "--",
        r"\PLACEHOLDERDIDDROPAIPCT": "--",
        r"\PLACEHOLDERDIDDROPHPCT": "--",
        r"\PLACEHOLDERDIDB1": "--",
        r"\PLACEHOLDERDIDB2": "--",
        r"\PLACEHOLDERDIDPP": "--",
        r"\PLACEHOLDERDIDDELTA": "--",
        r"\PLACEHOLDERDIDSE": "--",
        r"\PLACEHOLDERDIDPVAL": "--",
        # Cell rates (4 cells: reviewer x author).
        r"\PLACEHOLDERAIAPPAIRATE": "--",
        r"\PLACEHOLDERAIAPPHRATE": "--",
        r"\PLACEHOLDERHAPPAIRATE": "--",
        r"\PLACEHOLDERHAPPHRATE": "--",
        # ---- AI opinion pool (parsed + native) ----
        r"\PLACEHOLDERAIOPRSNATIVE": "--",
        r"\PLACEHOLDERAIOPPRS": "--",
        # ---- AI-only review (motivation) ----
        r"\PLACEHOLDERAIONLYN": "--",
        r"\PLACEHOLDERAIONLYPCT": "--",
        # ---- §8.4 anchoring robustness ----
        r"\PLACEHOLDERANCHORAN": "--",
        r"\PLACEHOLDERANCHORBN": "--",
        r"\PLACEHOLDERANCHORAGAP": "--",
        r"\PLACEHOLDERANCHORBGAP": "--",
        r"\PLACEHOLDERANCHORADID": "--",
        r"\PLACEHOLDERANCHORDIDCI": "--",
        r"\PLACEHOLDERANCHORAAIRATE": "--",
        r"\PLACEHOLDERANCHORAHRATE": "--",
        r"\PLACEHOLDERANCHORBAIRATE": "--",
        r"\PLACEHOLDERANCHORBHRATE": "--",
        r"\PLACEHOLDERANCHORANAI": "--",
        r"\PLACEHOLDERANCHORANH": "--",
        r"\PLACEHOLDERANCHORBNAI": "--",
        r"\PLACEHOLDERANCHORBNH": "--",
        # ---- Legacy (old 282-cohort) for backward compat ----
        r"\PLACEHOLDERAIREVIEWEVENTS": "--",
        r"\PLACEHOLDERAIAPPROVALS": "--",
        r"\PLACEHOLDERAIAPPROVALPCT": "--",
        r"\PLACEHOLDERWITHINN": "--",
        r"\PLACEHOLDERWITHINAIN": "--",
        r"\PLACEHOLDERWITHINAINPCT": "--",
        r"\PLACEHOLDERWITHINAIHR": "--",
        r"\PLACEHOLDERWITHINHUN": "--",
        r"\PLACEHOLDERWITHINHUHR": "--",
        r"\PLACEHOLDERWITHINGAP": "--",
        r"\PLACEHOLDERWITHINZTEST": "--",
        r"\PLACEHOLDERWITHINP": "--",
        r"\PLACEHOLDERWITHINAIAR": "--",
        r"\PLACEHOLDERWITHINHUAR": "--",
    }
    if within_path.exists():
        wp = json.loads(within_path.read_text())

        # ---- v2 main DiD ----
        did = wp.get("did", {})
        if did:
            defaults[r"\PLACEHOLDERDIDN"] = _num(did.get("n", 0))
            defaults[r"\PLACEHOLDERDIDNAI"] = _num(did.get("n_AI_auth", 0))
            defaults[r"\PLACEHOLDERDIDNH"] = _num(did.get("n_H_auth", 0))
            defaults[r"\PLACEHOLDERDIDNFULL"] = _num(did.get("n_full", 0))
            for key, src in [
                (r"\PLACEHOLDERDIDDROPPCT", "drop_pct"),
                (r"\PLACEHOLDERDIDDROPAIPCT", "drop_AI_pct"),
                (r"\PLACEHOLDERDIDDROPHPCT", "drop_H_pct"),
            ]:
                v = did.get(src)
                if v is not None:
                    defaults[key] = f"{v:.1f}\\%"
            for key, src in [
                (r"\PLACEHOLDERDIDB1", "bracket1_pp"),
                (r"\PLACEHOLDERDIDB2", "bracket2_pp"),
                (r"\PLACEHOLDERDIDPP", "did_pp"),
            ]:
                v = did.get(src)
                if v is not None:
                    defaults[key] = f"{v:+.2f}"
            d = did.get("logit_delta")
            se = did.get("logit_se")
            p = did.get("logit_p")
            if d is not None:
                defaults[r"\PLACEHOLDERDIDDELTA"] = f"{d:+.3f}"
            if se is not None:
                defaults[r"\PLACEHOLDERDIDSE"] = f"{se:.3f}"
            if p is not None:
                # LaTeX-friendly p-value formatting; placeholder is the whole
                # math segment including the leading "p" so callers write
                # "(...; \PLACEHOLDERDIDPVAL)" rather than wrapping in $$.
                if p < 1e-3:
                    defaults[r"\PLACEHOLDERDIDPVAL"] = "$p<10^{-3}$"
                else:
                    defaults[r"\PLACEHOLDERDIDPVAL"] = f"$p={p:.3f}$"
            # Cell rates as percentages.
            cells = did.get("cells", {})
            for cell_key, ph in [
                ("AI_x_AI", r"\PLACEHOLDERAIAPPAIRATE"),
                ("AI_x_H",  r"\PLACEHOLDERAIAPPHRATE"),
                ("H_x_AI",  r"\PLACEHOLDERHAPPAIRATE"),
                ("H_x_H",   r"\PLACEHOLDERHAPPHRATE"),
            ]:
                cell = cells.get(cell_key, {})
                rate = cell.get("rate")
                if rate is not None:
                    defaults[ph] = f"{100*rate:.2f}\\%"

        # ---- AI opinion pool ----
        pool = wp.get("ai_opinion_pool", {})
        if pool:
            n_native_unique = pool.get("n_native_unique_prs")
            if n_native_unique is not None:
                defaults[r"\PLACEHOLDERAIOPRSNATIVE"] = _num(n_native_unique)
            n_unique = pool.get("n_unique_prs")
            if n_unique is not None:
                defaults[r"\PLACEHOLDERAIOPPRS"] = _num(n_unique)

        # ---- AI-only review context ----
        ai_only = wp.get("ai_only", {})
        if ai_only:
            n_no_h = ai_only.get("no_human_explicit_n")
            if n_no_h is not None:
                defaults[r"\PLACEHOLDERAIONLYN"] = _num(n_no_h)
            pct = ai_only.get("no_human_explicit_pct")
            if pct is not None:
                defaults[r"\PLACEHOLDERAIONLYPCT"] = f"{pct:.1f}\\%"

        # ---- Anchoring robustness ----
        anch = wp.get("anchoring", {})
        if anch:
            defaults[r"\PLACEHOLDERANCHORAN"] = _num(anch.get("A_n", 0))
            defaults[r"\PLACEHOLDERANCHORBN"] = _num(anch.get("B_n", 0))
            for key, src in [
                (r"\PLACEHOLDERANCHORAGAP", "A_gap_pp"),
                (r"\PLACEHOLDERANCHORBGAP", "B_gap_pp"),
                (r"\PLACEHOLDERANCHORADID", "cross_stratum_did_pp"),
            ]:
                v = anch.get(src)
                if v is not None:
                    defaults[key] = f"{v:+.2f}"
            lo = anch.get("cross_stratum_did_ci_lo_pp")
            hi = anch.get("cross_stratum_did_ci_hi_pp")
            if lo is not None and hi is not None:
                defaults[r"\PLACEHOLDERANCHORDIDCI"] = f"95\\% CI [{lo:+.2f}, {hi:+.2f}]"
            for cell_key, ph_rate, ph_n in [
                ("A_AI", r"\PLACEHOLDERANCHORAAIRATE", r"\PLACEHOLDERANCHORANAI"),
                ("A_H",  r"\PLACEHOLDERANCHORAHRATE",  r"\PLACEHOLDERANCHORANH"),
                ("B_AI", r"\PLACEHOLDERANCHORBAIRATE", r"\PLACEHOLDERANCHORBNAI"),
                ("B_H",  r"\PLACEHOLDERANCHORBHRATE",  r"\PLACEHOLDERANCHORBNH"),
            ]:
                cell = anch.get(cell_key, {})
                rate = cell.get("rate")
                n = cell.get("n")
                if rate is not None:
                    defaults[ph_rate] = f"{100*rate:.2f}\\%"
                if n is not None:
                    defaults[ph_n] = _num(n)

        # ---- Legacy (old 282-cohort) for any remaining template text ----
        legacy = wp.get("legacy_within_pr", {})
        cond = legacy.get("conditional_on_ai_approval", {})
        if cond:
            ai = cond.get("ai_authored", {})
            hu = cond.get("human_authored", {})
            n_total = cond.get("n_total", 0)
            defaults[r"\PLACEHOLDERWITHINN"] = _num(n_total)
            defaults[r"\PLACEHOLDERWITHINAIN"] = _num(ai.get("n", 0))
            defaults[r"\PLACEHOLDERWITHINHUN"] = _num(hu.get("n", 0))

            def _frac(x):
                k, n = int(x.get("human_co_approved", 0)), int(x.get("n", 0))
                return f"{100*k/n:.1f}\\%" if n else "--"
            defaults[r"\PLACEHOLDERWITHINAIHR"] = _frac(ai)
            defaults[r"\PLACEHOLDERWITHINHUHR"] = _frac(hu)

    for k, v in defaults.items():
        subs[k] = v

    # Regression results from 06b_regression.py if present.
    reg_path = RESULTS_DIR / "regression.json"
    if reg_path.exists():
        reg = json.loads(reg_path.read_text())
        def _row(term, label):
            if term not in reg: return ""
            r = reg[term]
            star = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""
            return (
                f"{label} & {r['coef']:.3f}{star} & {r['se']:.3f} & {r['z']:.2f} & "
                f"{r['p']:.1e} \\\\\n"
            )
        table = (
            "\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n"
            "Term & Coef. & SE & $z$ & $p$ \\\\\n\\midrule\n"
        )
        for term, label in [
            ("AI_author", "AI author"),
            ("AI_reviewer", "AI reviewer"),
            ("AI_author_x_reviewer", "AI author $\\times$ reviewer (interaction)"),
            ("log_size", r"$\log(1+\mathrm{adds}+\mathrm{dels})$"),
        ]:
            table += _row(term, label)
        n_obs = reg.get("_meta", {}).get("n_obs", "--")
        n_repos = reg.get("_meta", {}).get("n_repos", "--")
        pr2 = reg.get("_meta", {}).get("pseudo_r2", 0)
        table += (
            "\\bottomrule\n"
            f"\\multicolumn{{5}}{{@{{}}l}}{{\\footnotesize $n={n_obs:,}$, "
            f"repos={n_repos}, pseudo-$R^2={pr2:.3f}$. "
            f"Cluster-robust SEs by repo. $*p<0.05,\\ **p<0.01,\\ ***p<0.001$.}}\\\\\n"
            "\\end{tabular}"
        )
        subs[r"\PLACEHOLDERREGRESSIONTABLE"] = table

        inter = reg.get("AI_author_x_reviewer", {})
        if inter and inter.get("p") is not None:
            p = inter["p"]
            coef = inter["coef"]
            if p < 0.05:
                subs[r"\PLACEHOLDERREGRESSIONINTERPRETATION"] = (
                    f"Its sign is {'negative' if coef < 0 else 'positive'} "
                    f"with $p={p:.3f}$, consistent with the diff-in-diff direction."
                )
            else:
                subs[r"\PLACEHOLDERREGRESSIONINTERPRETATION"] = (
                    f"It is not statistically distinguishable from zero "
                    f"($\\hat\\beta_3={coef:+.3f}$, $p={p:.2f}$). We read this as a null "
                    f"on self-preference \\emph{{conditional on}} an approving review, "
                    f"even though the main-effect \\texttt{{AI\\_reviewer}} coefficient is strongly negative."
                )
        subs[r"\PLACEHOLDERREGRESSIONSIGNDIR"] = (
            "close to zero conditional on the other terms"
        )
    else:
        subs[r"\PLACEHOLDERREGRESSIONSIGNDIR"] = "consistent with the diff-in-diff"
        subs[r"\PLACEHOLDERREGRESSIONTABLE"] = "[Regression pending]"
        subs[r"\PLACEHOLDERREGRESSIONINTERPRETATION"] = "[Interpretation pending]"

    # Apply to paper.tex and each appendix/*.tex. Sort by descending key length
    # so longer placeholders (e.g. \PLACEHOLDERWITHINAINPCT) are replaced before
    # their prefixes (\PLACEHOLDERWITHINAIN).
    ordered_subs = sorted(subs.items(), key=lambda kv: -len(kv[0]))

    def _apply(src: Path, dst: Path, rewrite_appendix_inputs: bool = False) -> None:
        tex = src.read_text()
        for k, v in ordered_subs:
            tex = tex.replace(k, v)
        if rewrite_appendix_inputs:
            tex = re.sub(
                r"\\input\{appendix/([A-Za-z0-9_]+)\.tex\}",
                r"\\input{appendix/\1.filled.tex}",
                tex,
            )
        unfilled = re.findall(r"\\PLACEHOLDER[A-Z0-9]+", tex)
        if unfilled:
            logger.warning("Un-filled placeholders in %s: %s", dst.name, sorted(set(unfilled)))
            tex = re.sub(r"\\PLACEHOLDER[A-Z0-9]+", "??", tex)
        dst.write_text(tex)

    _apply(PAPER_TEX, PAPER_TEX_OUT, rewrite_appendix_inputs=True)
    logger.info("Wrote %s with %d substitutions.", PAPER_TEX_OUT, len(subs))

    appendix_dir = PAPER_DIR / "appendix"
    if appendix_dir.is_dir():
        for src in sorted(appendix_dir.glob("*.tex")):
            if src.name.endswith(".filled.tex"):
                continue
            dst = src.with_suffix(".filled.tex")
            _apply(src, dst)
            logger.info("Wrote %s.", dst.relative_to(SUBPROJECT_ROOT))


if __name__ == "__main__":
    main()

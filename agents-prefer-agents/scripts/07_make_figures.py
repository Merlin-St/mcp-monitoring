"""Render figures for the paper.

Figure 1 (headline): weekly merge rate, 2×2 author × reviewer, with Wilson
95% CIs. Tool-launch reference lines.

Figure 2: AI→AI chain length distribution, by quarter (boxplot).

Optional Figure 3 (--all): share of PRs with any AI involvement over time.

Outputs to ``paper/figures/figure1_merge_rates.pdf`` etc., 300 dpi, serif,
sized for two-column ICML.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

utils = import_module("99_utils")
get_logger = utils.get_logger
DATA_DIR = utils.DATA_DIR
RESULTS_DIR = utils.RESULTS_DIR
PAPER_FIG_DIR = utils.SUBPROJECT_ROOT / "paper" / "figures"
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

logger = get_logger("07_make_figures")

# ICML two-column paper: text width ~6.5 in, column width ~3.2 in.
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,  # TrueType (workshop requirement commonly)
    "ps.fonttype": 42,
})

# Tool-launch reference dates (to mark on Figure 1, per §1.5). Dates are
# approximate public availability.
LAUNCH_MARKS = [
    ("Claude Code GA",            "2025-02-24"),
    ("GitHub Copilot coding agent", "2025-05-19"),
    ("Devin public",              "2024-12-11"),  # before window, but kept for context
    ("Cursor Background Agents",  "2025-05-09"),
    ("Jules GA",                  "2025-08-06"),
    ("Codex cloud",               "2025-05-16"),
]


CELL_STYLE = {
    ("AI", "AI"):       {"color": "#d62728", "marker": "o", "label": "AI author × AI reviewer"},
    ("AI", "human"):    {"color": "#ff7f0e", "marker": "s", "label": "AI author × human reviewer"},
    ("human", "AI"):    {"color": "#1f77b4", "marker": "^", "label": "human author × AI reviewer"},
    ("human", "human"): {"color": "#2ca02c", "marker": "D", "label": "human × human (baseline)"},
}


def period_to_date(w: str) -> datetime | None:
    """'YYYY-Www' (ISO week) or 'YYYY-MM' → first day of that period."""
    try:
        if "W" in w:
            y, wk = w.split("-W")
            return datetime.fromisocalendar(int(y), int(wk), 1)
        y, mm = w.split("-")
        return datetime(int(y), int(mm), 1)
    except Exception:
        return None


def week_to_date(w: str) -> datetime | None:  # backward-compat wrapper
    return period_to_date(w)


def figure1_headline(granularity: str = "month"):
    """HEADLINE figure: monthly AI-agent participation on multiple dimensions.

    Series plotted (only the lines directly answering an RQ):
      (1) share of PRs with any AI-agent event (RQ 1),
      (2) share of PRs with AI-AI comment chain >= 5 (RQ 1),
      (3) share of PRs with explicit AI review (RQ 2).
    First and last month values are annotated on every line.
    """
    import pandas as pd
    chains = pd.read_parquet(DATA_DIR / "chains.parquet")
    events = pd.read_parquet(DATA_DIR / "pr_events.parquet")

    def to_period(w: str):
        try:
            y, wk = w.split("-W")
            if granularity == "month":
                return pd.Timestamp.fromisocalendar(int(y), int(wk), 1).strftime("%Y-%m")
            return w
        except Exception:
            return ""

    chains["period"] = chains["opened_week"].apply(to_period)
    chains = chains[chains.period != ""]

    # Per-PR flags from the events table.
    ai_decision = events[
        (events["event_type"].isin(["review_approved", "review_changes_requested"]))
        & (events["actor_type"] == "AI-bot")
        & (events["confidence"] == "high")
    ][["repo", "number"]].drop_duplicates()
    ai_decision["has_ai_decision"] = True

    any_approval = events[events["event_type"] == "review_approved"][
        ["repo", "number"]].drop_duplicates()
    any_approval["has_any_approval"] = True

    ai_approval = events[
        (events["event_type"] == "review_approved")
        & (events["actor_type"] == "AI-bot")
        & (events["confidence"] == "high")
    ][["repo", "number"]].drop_duplicates()
    ai_approval["has_ai_approval"] = True

    human_approval = events[
        (events["event_type"] == "review_approved")
        & (events["actor_type"] == "human")
    ][["repo", "number"]].drop_duplicates()
    human_approval["has_human_approval"] = True

    df = (chains
          .merge(ai_decision,    on=["repo", "number"], how="left")
          .merge(any_approval,   on=["repo", "number"], how="left")
          .merge(ai_approval,    on=["repo", "number"], how="left")
          .merge(human_approval, on=["repo", "number"], how="left"))
    for c in ("has_ai_decision", "has_any_approval", "has_ai_approval", "has_human_approval"):
        df[c] = df[c].fillna(False).astype(bool)
    df["author_is_ai"] = (df["author_type"] == "AI")

    # "AI authored" rollup = AI-bot events (high confidence) + AI-powered events.
    # The chains.parquet column may be named ``n_ai_powered`` (current) or
    # ``n_ai_assisted`` (legacy); accept either so the figure renders against
    # both snapshots.
    n_ai_powered_col = (
        "n_ai_powered" if "n_ai_powered" in df.columns
        else "n_ai_assisted" if "n_ai_assisted" in df.columns
        else None
    )
    if n_ai_powered_col is None:
        df["_n_ai_authored_evt"] = df["n_ai_bot_high"]
    else:
        df["_n_ai_authored_evt"] = df["n_ai_bot_high"] + df[n_ai_powered_col]

    def _conditional_share(mask_col: str):
        def _agg(g):
            m = g[mask_col]
            n = int(m.sum())
            if n == 0:
                return np.nan
            return float(g.loc[m, "author_is_ai"].sum()) / n
        return _agg

    def _merged_share(g, *, need_ai_approval: bool, need_human_approval: bool | None):
        merged = g[g["merged"].astype(bool)]
        if len(merged) == 0:
            return np.nan
        m = merged["has_ai_approval"] == need_ai_approval
        if need_human_approval is not None:
            m &= merged["has_human_approval"] == need_human_approval
        return float(m.sum()) / len(merged)

    grp = df.groupby("period").apply(lambda g: pd.Series({
        "n_prs": len(g),
        "share_any_ai": (g["_n_ai_authored_evt"] > 0).mean(),
        "share_chain_ge5": (g["longest_chain_primary"] >= 5).mean(),
        "share_ai_decision": g["has_ai_decision"].mean(),
        "share_approved_by_ai_author": _conditional_share("has_any_approval")(g),
        "share_ai_approved_by_ai_author": _conditional_share("has_ai_approval")(g),
        "share_merged_ai_only":
            _merged_share(g, need_ai_approval=True,  need_human_approval=False),
        "share_merged_ai_and_human":
            _merged_share(g, need_ai_approval=True,  need_human_approval=True),
    })).reset_index()
    grp["date"] = grp["period"].apply(period_to_date)
    grp = grp.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    SERIES = [
        ("share_any_ai",                   "#d62728", "o",
         "share of PRs with any AI authored event"),
        ("share_chain_ge5",                "#8c564b", "^",
         r"share of PRs with AI bot--AI bot comment chain $\geq 5$"),
        ("share_ai_decision",              "#9467bd", "v",
         "share of PRs with explicit AI bot review"),
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for col, color, marker, label in SERIES:
        ax.plot(grp["date"], grp[col], color=color, lw=1.3, marker=marker, ms=3.5, label=label)

    # Annotate first and last non-NaN value on every line.
    def _fmt(v):
        if pd.isna(v):
            return ""
        pct = 100 * v
        # Match paper-text precision: sub-1% values are reported to 2 decimals,
        # larger values to 1 decimal.
        return f"{pct:.2f}%" if pct < 1 else f"{pct:.1f}%"
    for col, color, _, _ in SERIES:
        vals = grp[col]
        nz = vals.dropna()
        if nz.empty:
            continue
        first_i = nz.index[0]
        last_i = nz.index[-1]
        ax.annotate(_fmt(vals.iloc[first_i]),
                    (grp["date"].iloc[first_i], vals.iloc[first_i]),
                    textcoords="offset points", xytext=(-4, 5),
                    ha="right", va="bottom", fontsize=5.5, color=color)
        ax.annotate(_fmt(vals.iloc[last_i]),
                    (grp["date"].iloc[last_i], vals.iloc[last_i]),
                    textcoords="offset points", xytext=(4, 5),
                    ha="left", va="bottom", fontsize=5.5, color=color)

    # Tool launch reference lines.
    for name, d in LAUNCH_MARKS:
        dt = datetime.strptime(d, "%Y-%m-%d")
        if dt < datetime(2025, 4, 1) or dt > datetime(2026, 3, 31):
            continue
        ax.axvline(dt, color="grey", lw=0.5, ls=":", alpha=0.6)
        ax.text(dt, 0.98, name, rotation=90, va="top", ha="right",
                fontsize=5, color="grey", alpha=0.8)

    series_max = max((grp[c].max() or 0) for c, *_ in SERIES)
    ax.set_ylim(0, max(0.35, series_max * 1.15))
    ax.set_xlim(datetime(2025, 4, 1), datetime(2026, 4, 1))
    ax.set_xlabel(f"PR opened ({granularity})")
    ax.set_ylabel("Share of PRs")
    ax.set_title(r"AI-agent participation in GitHub PRs, monthly — Apr 2025 to Mar 2026")
    ax.legend(loc="upper left", frameon=False, fontsize=6.5)
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    fig.autofmt_xdate()
    fig.tight_layout()
    out = PAPER_FIG_DIR / "figure1_ai_participation.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Wrote %s", out)


def figure1(min_cell_n: int, granularity: str = "month"):
    cells_all = pd.read_parquet(DATA_DIR / "merge_rates.parquet")
    cells = cells_all[cells_all["granularity"] == granularity].copy()
    # Drop thin cells.
    cells = cells[cells["n_open"] >= min_cell_n].copy()
    cells["date"] = cells["period"].apply(period_to_date)
    cells = cells.dropna(subset=["date"])
    cells = cells.sort_values("date")

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for (a, r), style in CELL_STYLE.items():
        sub = cells[(cells["author_type"] == a) & (cells["reviewer_type"] == r)]
        if sub.empty:
            continue
        ax.plot(sub["date"], sub["merge_rate"], color=style["color"], lw=1.3,
                marker=style["marker"], ms=4, label=style["label"], alpha=0.95)
        ax.fill_between(sub["date"], sub["ci_low"], sub["ci_high"],
                        color=style["color"], alpha=0.15, lw=0)

    # Tool launch reference lines (within window only).
    for name, d in LAUNCH_MARKS:
        dt = datetime.strptime(d, "%Y-%m-%d")
        if dt < datetime(2025, 4, 1) or dt > datetime(2026, 3, 31):
            continue
        ax.axvline(dt, color="grey", lw=0.5, ls=":", alpha=0.6)
        ax.text(dt, 0.98, name, rotation=90, va="top", ha="right",
                fontsize=5, color="grey", alpha=0.8)

    ax.set_ylim(0, 1.02)
    ax.set_xlim(datetime(2025, 4, 1), datetime(2026, 4, 1))
    ax.set_xlabel(f"PR opened ({granularity})")
    ax.set_ylabel("Share merged within 30 days (Wilson 95% CI)")
    ax.set_title(f"PR merge rate by author $\\times$ reviewer type, {granularity}ly — Apr 2025 to Mar 2026")
    ax.legend(loc="lower left", frameon=False, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    out = PAPER_FIG_DIR / "figure1_merge_rates.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Wrote %s", out)


def _did_ci_pp(panels, z: float = 1.96) -> tuple[float, float]:
    """Normal-approximation 95% CI for the DiD on the probability scale.

    DiD = (p_treat_A − p_ref_A) − (p_treat_B − p_ref_B), so the variance is
    the sum of the four cell variances p(1−p)/n. Returned values are in pp.
    """
    se_sq = 0.0
    for p in panels:
        for cell in (p["ref_cell"], p["treat_cell"]):
            r, n = cell["rate"], cell.get("n") or cell.get("n_open") or 0
            if n > 0:
                se_sq += r * (1 - r) / n
    se_pp = (se_sq ** 0.5) * 100.0
    did_pp = panels[0]["gap_pp"] - panels[1]["gap_pp"]
    return did_pp - z * se_pp, did_pp + z * se_pp


def _render_two_panel_didfig(
    *,
    out_path,
    panels,
    cohort_label: str,
    n_total: int,
    did_pp: float,
    delta: float,           # accepted but unused in current rendering
    pval: float | None,     # accepted but unused in current rendering
    treat_color: str = "black",   # legacy kw — ignored; figure is B&W
    ref_color: str = "black",     # legacy kw — ignored; figure is B&W
    treat_xtick: str = "AI\nreviewer",
    ref_xtick: str = "Human\nreviewer",
    did_xlabel: str = r"$\Delta^A - \Delta^H$",
):
    """Black-and-white two-panel renderer for the within-PR DiD figures.

    Layout: [panel_A | panel_B | DiD] with the y-axis dropped between
    panels (only the leftmost panel keeps tick labels and left spine).
    Each side panel shows the two reviewer rates as black points with
    Wilson 95% CIs and the bracket (gap) as a small white-fill bar with
    black edge. The rightmost column shows the DiD as a solid black bar
    with the value and 95% CI annotated below.
    """
    from matplotlib.ticker import PercentFormatter

    del treat_color, ref_color  # B&W only

    def _direction_word(gap_pp: float) -> str:
        return "less" if gap_pp < 0 else "more"

    for p in panels:
        p["_dir"] = _direction_word(p["gap_pp"])

    fig = plt.figure(figsize=(7.4, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[4, 4, 1.2], wspace=0.12)
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1], sharey=axA)
    axD = fig.add_subplot(gs[2], sharey=axA)

    bar_w = 0.45
    x_ref, x_treat, x_gap = 0, 1, 2

    for ax, p in zip([axA, axB], panels):
        ax.set_title(p["title_template"].format(direction=p["_dir"]), fontsize=8.8)
        ref = p["ref_cell"]
        treat = p["treat_cell"]

        # Points with CI lines for the two reviewer rates (black only).
        # When the two dots are close in y, stagger the inline rate label
        # vertically (one above, one below) to avoid overlap.
        rates_close = abs(ref["rate"] - treat["rate"]) < 0.06
        for xi, cell, is_lower in [
            (x_ref, ref, ref["rate"] <= treat["rate"]),
            (x_treat, treat, treat["rate"] < ref["rate"]),
        ]:
            ax.errorbar(
                xi, cell["rate"],
                yerr=[[cell["rate"] - cell["ci_lo"]], [cell["ci_hi"] - cell["rate"]]],
                fmt="o", color="black", ecolor="black", capsize=4, lw=1.0, ms=6,
                mec="black", mfc="black", mew=0.6,
            )
            if rates_close:
                # Stagger above/below the dot.
                dy = -0.025 if is_lower else 0.025
                va = "top" if is_lower else "bottom"
            else:
                dy, va = 0, "center"
            ax.text(xi + 0.12, cell["rate"] + dy, f"{100*cell['rate']:.1f}%",
                    ha="left", va=va, fontsize=8, color="black",
                    fontweight="bold")

        # Small bar showing the bracket (gap): white fill, black edge.
        gap_top = max(ref["rate"], treat["rate"])
        gap_bot = min(ref["rate"], treat["rate"])
        gap_height = gap_top - gap_bot
        ax.bar(x_gap, gap_height, bottom=gap_bot, width=bar_w,
               color="white", edgecolor="black", lw=0.8)
        # Place gap label inside the bar if it's tall enough; otherwise above
        # the bar to avoid cramped/overflowing text.
        if gap_height >= 0.06:
            ax.text(x_gap, (gap_top + gap_bot) / 2,
                    f"{p['gap_pp']:+.1f}\npp",
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color="black")
        else:
            ax.text(x_gap, gap_top + 0.025,
                    f"{p['gap_pp']:+.1f} pp",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                    color="black")

        ax.set_xticks([x_ref, x_treat, x_gap])
        ax.set_xticklabels([ref_xtick, treat_xtick, r"$\Delta$"], fontsize=8.5)
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_xlabel(p["cohort_sublabel"], fontsize=8.5, labelpad=8)
        ax.grid(axis="y", lw=0.3, alpha=0.3, color="black")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axA.set_ylabel("Approval rate (Wilson 95% CI)", fontsize=8.8)

    # Drop the y-axis on panel B and the DiD panel: hide tick labels and
    # left spine.
    for ax in (axB, axD):
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis="y", left=False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Right: DiD bar positioned at the height of the gap. The two AI-reviewer
    # rates (= bottoms of bracket A and bracket B) define the bar's vertical
    # range; the dashed horizontal lines from the side panels indicate the
    # alignment.
    ci_lo, ci_hi = _did_ci_pp(panels)
    a_A = panels[0]["treat_cell"]["rate"]
    a_B = panels[1]["treat_cell"]["rate"]
    bar_bot = min(a_A, a_B)
    bar_top = max(a_A, a_B)

    # Dashed horizontal guide lines across the side panels.
    for ax in (axA, axB):
        ax.axhline(a_A, ls=(0, (4, 3)), color="black", lw=0.6, alpha=0.55,
                   zorder=0)
        ax.axhline(a_B, ls=(0, (4, 3)), color="black", lw=0.6, alpha=0.55,
                   zorder=0)

    # In the DiD column, draw the dashed lines only from the left edge up to
    # the DiD bar's left edge so they don't cut through the bar.
    bar_w = 0.5
    bar_left_x = 0 - bar_w / 2
    xlim_lo, xlim_hi = -0.6, 0.6
    xmax_dash = (bar_left_x - xlim_lo) / (xlim_hi - xlim_lo)
    axD.axhline(a_A, xmin=0, xmax=xmax_dash, ls=(0, (4, 3)),
                color="black", lw=0.6, alpha=0.55)
    axD.axhline(a_B, xmin=0, xmax=xmax_dash, ls=(0, (4, 3)),
                color="black", lw=0.6, alpha=0.55)

    axD.bar(0, bar_top - bar_bot, bottom=bar_bot, width=bar_w,
            color="black", edgecolor="black", lw=0.5)
    # The signed DiD value goes in the xlabel below the bar (set later).
    # Inside the bar, only show the magnitude when there is room.
    if (bar_top - bar_bot) >= 0.06:
        axD.text(0, (bar_bot + bar_top) / 2, f"{did_pp:+.2f}\npp",
                 ha="center", va="center", fontsize=8.5, fontweight="bold",
                 color="white")
    axD.set_xticks([0])
    axD.set_xticklabels([did_xlabel], fontsize=9)
    axD.set_xlim(xlim_lo, xlim_hi)
    axD.set_xlabel(
        f"{did_pp:+.2f} pp\n[95% CI: {ci_lo:+.2f}, {ci_hi:+.2f}]",
        fontsize=8, labelpad=8,
    )
    axD.grid(axis="y", lw=0.3, alpha=0.3, color="black")

    fig.suptitle(
        f"{cohort_label} (n={n_total:,})",
        fontsize=9, y=1.02,
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def figure2_withinpr():
    """Within-PR AI-AI-bias DiD on the dual-review cohort."""
    import json as _json
    wp_path = RESULTS_DIR / "within_pr_stats.json"
    if not wp_path.exists():
        logger.warning("No within_pr_stats.json; skipping figure 2.")
        return
    wp = _json.loads(wp_path.read_text())
    did = wp.get("did")
    if not did:
        logger.warning("No 'did' block in within_pr_stats.json; skipping figure 2.")
        return
    cells = did["cells"]
    n_AI = did["n_AI_auth"]; n_H = did["n_H_auth"]
    panels = [
        {
            "title_template": "AI prefers AI {direction} than humans do",
            "cohort_sublabel": f"(AI-authored, n={n_AI:,})",
            "ref_cell":   cells["H_x_AI"],
            "treat_cell": cells["AI_x_AI"],
            "gap_pp":     did["bracket1_pp"],
        },
        {
            "title_template": "AI prefers humans {direction} than humans do",
            "cohort_sublabel": f"(human-authored, n={n_H:,})",
            "ref_cell":   cells["H_x_H"],
            "treat_cell": cells["AI_x_H"],
            "gap_pp":     did["bracket2_pp"],
        },
    ]
    _render_two_panel_didfig(
        out_path=PAPER_FIG_DIR / "figure2_withinpr.pdf",
        panels=panels,
        cohort_label="Dual-review cohort (Any AI & human)",
        n_total=did["n"],
        did_pp=did["did_pp"],
        delta=did["logit_delta"],
        pval=did["logit_p"],
    )


def figure2_withinfamily_claude():
    """Within-family DiD: Claude reviewer vs Human reviewer, Claude vs human author."""
    import json as _json
    wf_path = RESULTS_DIR / "within_family_stats.json"
    if not wf_path.exists():
        logger.warning("No within_family_stats.json; skipping Claude-family figure.")
        return
    wf = _json.loads(wf_path.read_text())
    cv = wf.get("claude_vs_human")
    if not cv:
        logger.warning("No 'claude_vs_human' block in within_family_stats.json; skipping.")
        return
    cells = cv["cells"]
    n_C = cv["n_C_auth"]; n_H = cv["n_H_auth"]
    # bracket1 = Claude-reviewer rate − Human-reviewer rate on Claude-authored
    bracket1 = (cells["C_rev_x_C_auth"]["rate"] - cells["R_rev_x_C_auth"]["rate"]) * 100
    bracket2 = (cells["C_rev_x_X_auth"]["rate"] - cells["R_rev_x_X_auth"]["rate"]) * 100
    panels = [
        {
            "title_template": "Claude prefers Claude {direction} than humans do",
            "cohort_sublabel": f"(Claude-authored, n={n_C:,})",
            "ref_cell":   cells["R_rev_x_C_auth"],
            "treat_cell": cells["C_rev_x_C_auth"],
            "gap_pp":     bracket1,
        },
        {
            "title_template": "Claude prefers humans {direction} than humans do",
            "cohort_sublabel": f"(human-authored, n={n_H:,})",
            "ref_cell":   cells["R_rev_x_X_auth"],
            "treat_cell": cells["C_rev_x_X_auth"],
            "gap_pp":     bracket2,
        },
    ]
    _render_two_panel_didfig(
        out_path=PAPER_FIG_DIR / "figure2b_withinfamily_claude.pdf",
        panels=panels,
        cohort_label="Dual-review cohort (Claude & human)",
        n_total=cv["cohort_n"],
        did_pp=cv["did_pp"],
        delta=cv["logit_delta"],
        pval=cv["logit_p"],
        treat_xtick="Claude\nreviewer",
        ref_xtick="Human\nreviewer",
        did_xlabel=r"$\Delta^{A,\mathrm{Claude}} - \Delta^H$",
    )


def figure2():
    """Quarterly stacked bars of chain-length buckets.

    Most PRs have no AI events at all (chain=0), so a standard boxplot is
    uninformative (all boxes at 0). Instead, bucket PRs by chain length and
    show the bucket mix per quarter as a stacked bar. This reveals how the
    tail is growing: darker-red slices (long chains) expand over time.
    """
    chains = pd.read_parquet(DATA_DIR / "chains.parquet")
    chains = chains[chains["opened_week"].str.len() > 0]

    def week_to_quarter(w):
        try:
            y, wk = w.split("-W")
            q = (int(wk) - 1) // 13 + 1
            return f"{y}-Q{min(q, 4)}"
        except Exception:
            return None

    chains["quarter"] = chains["opened_week"].apply(week_to_quarter)
    chains = chains.dropna(subset=["quarter"])
    # Restrict to in-window quarters (drop the partial 2025-Q1 artefact and 2026-Q2).
    in_window = [q for q in sorted(chains["quarter"].unique()) if q >= "2025-Q2" and q <= "2026-Q1"]
    chains = chains[chains["quarter"].isin(in_window)]

    # Bucketise.
    def bucket(v):
        if v == 0:
            return "0 (no AI event)"
        if v == 1:
            return "1"
        if v <= 2:
            return "2"
        if v <= 4:
            return "3–4"
        if v <= 7:
            return "5–7"
        return "8+"

    chains["bucket"] = chains["longest_chain_primary"].apply(bucket)
    bucket_order = ["0 (no AI event)", "1", "2", "3–4", "5–7", "8+"]
    # Proportions per quarter.
    pivot = (
        chains.groupby(["quarter", "bucket"]).size()
        .unstack(fill_value=0)
        .reindex(columns=bucket_order, fill_value=0)
    )
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0)

    # Drop the "0" bucket from the figure — dominates the bar and washes out the tail.
    # Show only the PRs with *some* AI activity, as a share of all PRs.
    tail = pivot_pct.drop(columns=["0 (no AI event)"])

    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    colors = ["#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"]
    bottom = np.zeros(len(tail))
    for col, color in zip(tail.columns, colors):
        vals = tail[col].values
        ax.bar(range(len(tail)), vals, bottom=bottom, color=color,
               edgecolor="white", lw=0.4, label=col)
        bottom = bottom + vals

    ax.set_xticks(range(len(tail)))
    ax.set_xticklabels(tail.index, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Share of PRs by chain-length bucket")
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title("Growth of AI$\\to$AI chain-length tail, by quarter", fontsize=8.5)
    ax.legend(title="chain length", loc="upper left", frameon=False,
              fontsize=6.5, title_fontsize=6.5, ncol=2)
    ax.set_ylim(0, tail.values.sum(axis=1).max() * 1.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = PAPER_FIG_DIR / "figure2_chain_length.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Wrote %s", out)


def figure3():
    """Share of PRs with any AI involvement over time. Descriptive."""
    summary = pd.read_parquet(DATA_DIR / "pr_summary.parquet")
    summary = summary[summary["opened_week"].str.len() > 0].copy()
    summary["any_ai"] = (
        (summary["author_type"] == "AI") | (summary["reviewer_type"] == "AI")
    )
    weekly = (
        summary.groupby("opened_week")["any_ai"].mean().reset_index()
    )
    weekly["date"] = weekly["opened_week"].apply(week_to_date)
    weekly = weekly.dropna(subset=["date"]).sort_values("date")
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    ax.plot(weekly["date"], weekly["any_ai"], color="#d62728", lw=1.2)
    ax.set_ylabel("Share of PRs with any AI involvement")
    ax.set_title("Any-AI PR share, weekly")
    ax.set_ylim(0, max(0.01, weekly["any_ai"].max() * 1.1))
    fig.autofmt_xdate()
    fig.tight_layout()
    out = PAPER_FIG_DIR / "figure3_any_ai_share.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Wrote %s", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-cell-n", type=int, default=20)
    parser.add_argument("--granularity", choices=["week", "month"], default="month")
    parser.add_argument("--all", action="store_true", help="also render figure 3")
    args = parser.parse_args()

    if not (DATA_DIR / "chains.parquet").exists():
        logger.error("Run 05_compute_chains.py first.")
        sys.exit(1)

    figure1_headline(granularity=args.granularity)
    figure2_withinpr()
    figure2_withinfamily_claude()
    figure2()
    logger.info("=== FIGURES DONE ===")


if __name__ == "__main__":
    main()

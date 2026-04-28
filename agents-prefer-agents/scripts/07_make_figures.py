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
      (2) share with AI-AI comment chain >= 5 (RQ 1),
      (3) share of PRs with an AI agent approval/rejection decision (RQ 2),
      (4) share of merged PRs with AI approval AND no human approval (RQ 2),
      (5) share of merged PRs with AI approval AND a human approval (RQ 2).
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
        "share_any_ai": (g["n_ai_bot_high"] > 0).mean(),
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
         "share of PRs with any AI-agent event"),
        ("share_chain_ge5",                "#8c564b", "^",
         r"share with AI--AI comment chain $\geq 5$"),
        ("share_ai_decision",              "#9467bd", "v",
         "share with AI-agent approval/rejection decision"),
        ("share_merged_ai_only",           "#e377c2", "P",
         "share of merged PRs: AI approval, no human approval"),
        ("share_merged_ai_and_human",      "#17becf", "X",
         "share of merged PRs: AI approval AND human approval"),
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for col, color, marker, label in SERIES:
        ax.plot(grp["date"], grp[col], color=color, lw=1.3, marker=marker, ms=3.5, label=label)

    # Annotate first and last non-NaN value on every line.
    def _fmt(v):
        if pd.isna(v):
            return ""
        return f"{100 * v:.1f}%"
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


def figure2_withinpr():
    """Within-PR AI-AI-bias test, strict definition.

    Restricts to the small set of PRs where an AI bot actually issued an
    APPROVED review (not a comment). For each author type, plots the human
    co-approval rate on those PRs. The null prediction is: human co-approval
    is the same for AI-authored and human-authored AI-approved PRs.
    """
    import json as _json
    wp_path = RESULTS_DIR / "within_pr_stats.json"
    if not wp_path.exists():
        logger.warning("No within_pr_stats.json; skipping figure 2.")
        return
    wp = _json.loads(wp_path.read_text())
    cond = wp["conditional_on_ai_approval"]
    ai = cond["ai_authored"]
    hu = cond["human_authored"]
    n_total = cond["n_total"]

    def wilson(k, n, z=1.96):
        if n <= 0:
            return (0.0, 0.0)
        phat = k / n
        d = 1 + z * z / n
        c = (phat + z * z / (2 * n)) / d
        h = (z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / d
        return (max(0.0, c - h), min(1.0, c + h))

    ai_lo, ai_hi = wilson(ai["human_co_approved"], ai["n"])
    hu_lo, hu_hi = wilson(hu["human_co_approved"], hu["n"])

    fig, ax = plt.subplots(figsize=(3.3, 3.1))

    colors = ["#d62728", "#1f77b4"]
    labels = [
        "AI-authored\n(n={:,})".format(ai["n"]),
        "human-authored\n(n={:,})".format(hu["n"]),
    ]
    rates = [ai["human_co_approved_rate"], hu["human_co_approved_rate"]]
    cis = [(ai_lo, ai_hi), (hu_lo, hu_hi)]

    x = [0, 1]
    ax.bar(x, rates, color=colors, alpha=0.85, width=0.6,
           edgecolor="black", lw=0.5)
    for xi, rate, (lo, hi) in zip(x, rates, cis):
        ax.errorbar(xi, rate, yerr=[[rate - lo], [hi - rate]],
                    fmt="none", color="black", capsize=4, lw=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlim(-0.7, 1.7)
    ax.set_ylim(0, 0.80)
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Share with HUMAN co-approval (Wilson 95% CI)")
    # Pull z-test stats if available.
    ztest = cond.get("ztest", {})
    zval = ztest.get("z")
    pval = ztest.get("p_two_sided")
    p_line = (
        f"two-prop. $z={zval:.2f}$, $p={pval:.2f}$"
        if zval is not None and pval is not None else ""
    )
    ax.set_title(
        "Human co-approval on {:,} AI-approved PRs\n{}".format(n_total, p_line),
        fontsize=8.5,
    )
    ax.grid(axis="y", lw=0.3, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Base-rate annotation for context (top of plot).
    brate = wp.get("corpus_base_rate_ai_authored", 0.074)
    ax.text(0.5, 0.78,
            f"base rate of AI-authored in corpus: {brate*100:.1f}%\n"
            f"share of AI-approved PRs that are AI-authored: "
            f"{100*ai['n']/max(n_total,1):.1f}%",
            ha="center", va="top", fontsize=6.5, color="#555555",
            style="italic", transform=ax.transData)

    fig.tight_layout()
    out = PAPER_FIG_DIR / "figure2_withinpr.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Wrote %s", out)


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
    figure2()
    logger.info("=== FIGURES DONE ===")


if __name__ == "__main__":
    main()

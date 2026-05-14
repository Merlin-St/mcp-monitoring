"""
Plot daily Claude Code commit activity on GitHub.

Reads data/internal-cl/gh-claude-activity.json (produced by gh-claude-activity.py)
and writes a PNG with the daily Claude commit count over time. If the records
contain `total_github_commits` (from --with-totals), a second panel shows the
share of all GitHub commits that include a Claude trailer.

Usage:
  python gh-claude-activity-plot.py
  python gh-claude-activity-plot.py --rolling 7    # 7-day rolling mean overlay
  python gh-claude-activity-plot.py --output other.png
"""

import argparse
import json
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_INPUT = "data/internal-cl/gh-claude-activity.json"
DEFAULT_OUTPUT = "output-validation/gh-claude-activity.png"
TOTAL_KEY = "total_github_commits"


def load(input_path):
    with open(input_path) as f:
        df = pd.DataFrame(json.load(f))
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def plot(df, output_path, rolling):
    has_totals = TOTAL_KEY in df.columns and df[TOTAL_KEY].notna().any()
    nrows = 2 if has_totals else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 4.5 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    ax = axes[0]
    ax.plot(df["date"], df["total"], lw=1.0, color="#1f77b4", label="Daily Claude commits")
    if rolling and len(df) >= rolling:
        roll = df["total"].rolling(rolling, min_periods=1).mean()
        ax.plot(df["date"], roll, lw=2.0, color="#d62728", label=f"{rolling}-day rolling mean")
    ax.set_ylabel("Claude-attributed commits")
    ax.set_title("GitHub commits attributed to Claude (Co-authored-by trailer)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    if has_totals:
        ax2 = axes[1]
        share = df["total"] / df[TOTAL_KEY] * 100
        ax2.plot(df["date"], share, lw=1.0, color="#2ca02c", label="% of all GitHub commits")
        if rolling and len(df) >= rolling:
            ax2.plot(df["date"], share.rolling(rolling, min_periods=1).mean(),
                     lw=2.0, color="#d62728", label=f"{rolling}-day rolling mean")
        ax2.set_ylabel("Share of all GitHub commits (%)")
        ax2.set_title("Claude-attributed commits as % of all daily GitHub commits")
        ax2.legend(loc="upper left")
        ax2.grid(alpha=0.3)

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved figure to {output_path}")
    print(f"Rows: {len(df)}  range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Total Claude commits in window: {int(df['total'].sum())}")
    if has_totals:
        valid = df.dropna(subset=[TOTAL_KEY])
        if len(valid):
            overall = valid["total"].sum() / valid[TOTAL_KEY].sum() * 100
            print(f"Overall share (sum/sum): {overall:.4f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--rolling", type=int, default=7, help="Rolling-mean window in days (0 to disable)")
    args = parser.parse_args()

    df = load(args.input)
    plot(df, args.output, args.rolling)


if __name__ == "__main__":
    main()

"""Hand-label a sample of events for validation of ai_detection.

Per 99_instruction.md §4.4: sample 100 PRs (stratified), hand-label, compare
precision & recall to classifier. If AI-bot-author precision <0.9, investigate.

This script draws the sample and writes a CSV the user can fill in; it then
computes precision/recall given a filled CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from importlib import import_module

utils = import_module("99_utils")
get_logger = utils.get_logger
DATA_DIR = utils.DATA_DIR
RESULTS_DIR = utils.RESULTS_DIR

from lib.ai_detection import classify_event, classify_login  # noqa: E402

logger = get_logger("99_validate_detection")

HAND_LABEL_CSV = DATA_DIR / "hand_labels.csv"


def draw_sample(seed: int = 42, per_stratum: int = 25) -> list[dict]:
    """Draw 4×per_stratum stratified PRs (author-human, author-AI-bot,
    author-AI-assisted, approving-review) from all PRs."""
    prs_path = DATA_DIR / "prs"
    rng = random.Random(seed)
    strata: dict[str, list[dict]] = {
        "author_AI_bot": [],
        "author_AI_assisted": [],
        "author_human": [],
        "review_AI": [],
    }
    for p in sorted(prs_path.glob("*.jsonl")):
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                pr = json.loads(line)
                author_login = pr.get("author_login", "") or ""
                author_cls = classify_event(author_login, pr.get("body", "") or "")
                commit_msgs = "\n".join(c.get("message", "") or "" for c in pr.get("commits", []))
                any_bot_commit = any(
                    classify_login(c.get("author_login", "") or c.get("committer_login", ""))[0] == "AI-bot"
                    for c in pr.get("commits", [])
                )
                coauthor_trailer = "co-authored-by" in commit_msgs.lower() and (
                    "claude" in commit_msgs.lower()
                    or "copilot" in commit_msgs.lower()
                    or "devin" in commit_msgs.lower()
                )
                if author_cls.actor_type == "AI-bot" or any_bot_commit:
                    strata["author_AI_bot"].append(pr)
                elif coauthor_trailer:
                    strata["author_AI_assisted"].append(pr)
                else:
                    strata["author_human"].append(pr)
                for r in pr.get("reviews", []):
                    if (r.get("state") or "").upper() == "APPROVED":
                        rev_cls = classify_event(r.get("author_login", ""), r.get("body", "") or "")
                        if rev_cls.actor_type in ("AI-bot", "AI-assisted"):
                            strata["review_AI"].append({**pr, "_sample_reason": "review"})
                        break

    sample = []
    for name, items in strata.items():
        rng.shuffle(items)
        for it in items[:per_stratum]:
            sample.append({
                "stratum": name,
                "repo": it.get("repo", ""),
                "number": it.get("number", 0),
                "author_login": it.get("author_login", ""),
                "pred_author_type": classify_event(it.get("author_login", ""), it.get("body", "") or "").actor_type,
                "pred_author_family": classify_event(it.get("author_login", ""), it.get("body", "") or "").ai_family,
                "pr_url": f"https://github.com/{it.get('repo','')}/pull/{it.get('number',0)}",
                "hand_label_author_type": "",  # fill me: human / AI-bot / AI-assisted / non_ai_bot
                "notes": "",
            })
    return sample


def write_sample(sample: list[dict], out_path: Path):
    fields = list(sample[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sample)
    logger.info("Wrote %d rows to %s", len(sample), out_path)


def score(labels_path: Path):
    df = pd.read_csv(labels_path)
    df = df[df["hand_label_author_type"].notna() & (df["hand_label_author_type"] != "")]
    if df.empty:
        logger.error("No hand labels filled in yet.")
        return
    y_true = df["hand_label_author_type"]
    y_pred = df["pred_author_type"]
    total = len(df)
    accuracy = float((y_true == y_pred).mean())
    # Per-class precision/recall
    classes = sorted(set(y_true).union(y_pred))
    report = {"total_labelled": total, "accuracy": accuracy, "per_class": {}}
    for c in classes:
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        report["per_class"][c] = {
            "precision": round(p, 3),
            "recall": round(r, 3),
            "tp": tp, "fp": fp, "fn": fn,
        }
    (RESULTS_DIR / "validation.json").write_text(json.dumps(report, indent=2))
    logger.info("Validation report:\n%s", json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("draw", help="Draw a stratified sample for hand-labelling.")
    sub.add_parser("score", help="Score hand-labelled CSV against predictions.")
    args = parser.parse_args()
    if args.cmd == "draw":
        sample = draw_sample()
        write_sample(sample, HAND_LABEL_CSV)
    elif args.cmd == "score":
        score(HAND_LABEL_CSV)


if __name__ == "__main__":
    main()

"""Within-PR AI-AI-bias test (strict definition).

"AI reviewer" means an AI bot account (high confidence) that issued an
APPROVED review event — not a commented review, not a change-request, not a
dismissal. "Human reviewer" means a human account that issued an APPROVED
review event.

We compute the following bidirectional co-approval comparison:

  Conditional on an AI APPROVAL:
    - For AI-authored PRs:  what fraction also received a HUMAN approval?
    - For human-authored PRs: same question.
  Conditional on a HUMAN APPROVAL:
    - For AI-authored PRs:  what fraction also received an AI approval?
    - For human-authored PRs: same question.

The AI-AI-bias test is whether, conditional on an AI approval, the co-approval
rate by humans differs between author types (indicating humans agree more or
less with AI endorsements of AI code than of human code). Symmetric for the
human-approval conditional.

Additional descriptive stats: number of AI-bot review events by state and
actor account, base rate of AI-authored PRs, and the 3,347-PR "any-review"
context filter (reported honestly, not as the headline number).

Writes: results/within_pr_stats.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

utils = import_module("99_utils")
get_logger = utils.get_logger
DATA_DIR = utils.DATA_DIR
RESULTS_DIR = utils.RESULTS_DIR

logger = get_logger("06c_within_pr")


def _rate(k: int, n: int) -> float:
    return k / n if n else 0.0


def main():
    ev = pd.read_parquet(DATA_DIR / "pr_events.parquet")
    sm = pd.read_parquet(DATA_DIR / "pr_summary.parquet")

    # Review-STATE events (approved / commented / changes_requested / dismissed).
    r = ev[
        ev["event_type"].str.startswith("review_")
        & ~ev["event_type"].eq("review_comment")
    ].copy()
    r["is_human"] = r["actor_type"] == "human"
    r["is_ai_bot_high"] = (r["actor_type"] == "AI-bot") & (r["confidence"] == "high")
    r["is_approved"] = r["event_type"] == "review_approved"

    pr = r.groupby(["repo", "number"]).agg(
        human_reviewed=("is_human", "any"),
        ai_reviewed=("is_ai_bot_high", "any"),
        human_approved=("is_approved", lambda s: ((r.loc[s.index, "is_human"]) & s).any()),
        ai_approved=("is_approved", lambda s: ((r.loc[s.index, "is_ai_bot_high"]) & s).any()),
    ).reset_index()

    j = pr.merge(
        sm[["repo", "number", "author_type", "author_confidence"]],
        on=["repo", "number"],
        how="left",
    )
    j = j[j.author_type.isin(["AI", "human"])].copy()
    j.loc[(j.author_type == "AI") & (j.author_confidence != "high"), "author_type"] = "human"

    # Descriptive: AI review-event breakdown by state.
    ai_review_events = r[r.is_ai_bot_high]
    state_counts = ai_review_events["event_type"].value_counts().to_dict()

    # --- Primary: conditional on AI approval, how often did humans co-approve? ---
    cond_ai = j[j.ai_approved]
    cond_ai_ai = cond_ai[cond_ai.author_type == "AI"]
    cond_ai_hu = cond_ai[cond_ai.author_type == "human"]

    # Two-proportion z-test (no continuity correction) for the conditional-on-AI
    # approval comparison, pooled standard error.
    def _ztest(k1, n1, k2, n2):
        from math import sqrt, erf
        if n1 == 0 or n2 == 0:
            return {"z": None, "p_two_sided": None, "gap_pp": None}
        p1, p2 = k1 / n1, k2 / n2
        pool = (k1 + k2) / (n1 + n2)
        se = sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2)) if pool not in (0, 1) else 0
        z = (p1 - p2) / se if se > 0 else 0.0
        p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
        return {"z": float(z), "p_two_sided": float(p), "gap_pp": float(100 * (p2 - p1))}

    z_ai = _ztest(
        int(cond_ai_ai["human_approved"].sum()), int(len(cond_ai_ai)),
        int(cond_ai_hu["human_approved"].sum()), int(len(cond_ai_hu)),
    )

    # --- Symmetric: conditional on human approval, how often did AI co-approve? ---
    cond_hu = j[j.human_approved]
    cond_hu_ai = cond_hu[cond_hu.author_type == "AI"]
    cond_hu_hu = cond_hu[cond_hu.author_type == "human"]

    # --- Context: base rate of AI-authored PRs in the 500-repo corpus. ---
    base_rate_ai = (sm.author_type == "AI").mean()

    # --- Context: the broader "both-types-engaged" filter that was originally
    #              reported. Kept for transparency; not the headline. ---
    both_engaged = j[j.human_reviewed & j.ai_reviewed]

    out = {
        "ai_review_event_states": state_counts,
        "ai_approval_total": int(state_counts.get("review_approved", 0)),
        "corpus_base_rate_ai_authored": float(base_rate_ai),
        "conditional_on_ai_approval": {
            "n_total": int(len(cond_ai)),
            "ai_authored": {
                "n": int(len(cond_ai_ai)),
                "human_co_approved": int(cond_ai_ai.human_approved.sum()),
                "human_co_approved_rate": _rate(int(cond_ai_ai.human_approved.sum()), len(cond_ai_ai)),
            },
            "human_authored": {
                "n": int(len(cond_ai_hu)),
                "human_co_approved": int(cond_ai_hu.human_approved.sum()),
                "human_co_approved_rate": _rate(int(cond_ai_hu.human_approved.sum()), len(cond_ai_hu)),
            },
            "ztest": z_ai,
        },
        "conditional_on_human_approval": {
            "n_total": int(len(cond_hu)),
            "ai_authored": {
                "n": int(len(cond_hu_ai)),
                "ai_co_approved": int(cond_hu_ai.ai_approved.sum()),
                "ai_co_approved_rate": _rate(int(cond_hu_ai.ai_approved.sum()), len(cond_hu_ai)),
            },
            "human_authored": {
                "n": int(len(cond_hu_hu)),
                "ai_co_approved": int(cond_hu_hu.ai_approved.sum()),
                "ai_co_approved_rate": _rate(int(cond_hu_hu.ai_approved.sum()), len(cond_hu_hu)),
            },
        },
        "context_both_engaged_any_state": {
            "n_total": int(len(both_engaged)),
            "ai_authored": int((both_engaged.author_type == "AI").sum()),
            "human_authored": int((both_engaged.author_type == "human").sum()),
        },
        "generated_at": utils.now_iso(),
    }

    (RESULTS_DIR / "within_pr_stats.json").write_text(json.dumps(out, indent=2))
    logger.info("=== WITHIN-PR (strict) DONE ===")
    logger.info(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

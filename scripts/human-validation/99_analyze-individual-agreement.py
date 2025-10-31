#!/usr/bin/env python3
"""
Analyze individual participant agreement with LLM classifications.

Generates detailed reports showing which participants are most/least aligned with the LLM.
"""

import json
import sys
from pathlib import Path


def load_results(path: Path) -> dict:
    """Load validation results from JSON file."""
    with open(path) as f:
        return json.load(f)


def analyze_participant_agreement(results: dict) -> None:
    """Analyze and report individual participant agreement with LLM."""
    print("=" * 100)
    print("INDIVIDUAL PARTICIPANT AGREEMENT ANALYSIS")
    print("Original LLM Model vs. Each Human Validator")
    print("=" * 100)
    print()

    # Extract participant data
    participants = []
    for p_id, p_data in results["by_participant"].items():
        participants.append(
            {
                "id": p_id,
                "overall_kappa": p_data["overall_kappa"],
                "n_servers": p_data["n_servers_rated"],
                "questions": p_data["agreement_by_question"],
            }
        )

    # Sort by overall kappa (descending)
    participants.sort(key=lambda x: x["overall_kappa"], reverse=True)

    # Overall ranking
    print("OVERALL RANKING (by mean Kappa across all questions)")
    print("-" * 100)
    print(f"{'Rank':<6} {'Participant ID':<28} {'Overall κ':<12} {'Servers':<10} {'Interpretation':<30}")
    print("-" * 100)

    kappa_interpretation = [
        (0.81, "Almost perfect"),
        (0.61, "Substantial"),
        (0.41, "Moderate"),
        (0.21, "Fair"),
        (0.00, "Slight"),
        (-1.00, "Poor (worse than chance)"),
    ]

    for rank, p in enumerate(participants, 1):
        kappa = p["overall_kappa"]
        interp = next(desc for threshold, desc in kappa_interpretation if kappa >= threshold)
        print(f"{rank:<6} {p['id']:<28} {kappa:<12.3f} {p['n_servers']:<10} {interp:<30}")

    print()
    print()

    # Per-question breakdown
    questions = [
        ("func_main", "Main functionality level"),
        ("func_sub", "Sub-functionality classification"),
        ("onet_l1", "O*NET Level 1 category"),
        ("q3", "Industry generality"),
        ("q4", "Environment generality"),
        ("q5", "Payment autonomy"),
    ]

    print("DETAILED PER-QUESTION BREAKDOWN")
    print("=" * 100)
    print()

    for q_id, q_name in questions:
        print(f"Question: {q_name} ({q_id})")
        print("-" * 100)
        print(
            f"{'Rank':<6} {'Participant ID':<28} {'Kappa':<12} {'Agreement %':<15} {'N':<8} {'Interpretation':<20}"
        )
        print("-" * 100)

        # Sort participants by kappa for this question
        q_data = []
        for p in participants:
            if q_id in p["questions"]:
                q_info = p["questions"][q_id]
                q_data.append(
                    {
                        "id": p["id"],
                        "kappa": q_info["kappa"],
                        "agreement": q_info["agreement_pct"],
                        "n": q_info["n_responses"],
                    }
                )

        q_data.sort(key=lambda x: x["kappa"], reverse=True)

        for rank, entry in enumerate(q_data, 1):
            kappa = entry["kappa"]
            interp = next(desc for threshold, desc in kappa_interpretation if kappa >= threshold)
            print(
                f"{rank:<6} {entry['id']:<28} {kappa:<12.3f} {entry['agreement']:<14.1%} {entry['n']:<8} {interp:<20}"
            )

        print()
        print()

    # Summary statistics
    print("=" * 100)
    print("KEY FINDINGS")
    print("=" * 100)
    print()

    best = participants[0]
    worst = participants[-1]

    print(f"🏆 MOST ALIGNED WITH LLM:")
    print(f"   Participant: {best['id']}")
    print(f"   Overall Kappa: {best['overall_kappa']:.3f}")
    print(f"   Interpretation: {next(desc for threshold, desc in kappa_interpretation if best['overall_kappa'] >= threshold)}")
    print()

    # Find their best questions
    best_questions = sorted(best["questions"].items(), key=lambda x: x[1]["kappa"], reverse=True)
    print(f"   Best performing questions:")
    for q_id, q_data in best_questions[:3]:
        q_name = next(name for qid, name in questions if qid == q_id)
        print(f"      • {q_name}: κ={q_data['kappa']:.3f}, {q_data['agreement_pct']:.1%} agreement")
    print()

    print(f"❌ LEAST ALIGNED WITH LLM:")
    print(f"   Participant: {worst['id']}")
    print(f"   Overall Kappa: {worst['overall_kappa']:.3f}")
    print(
        f"   Interpretation: {next(desc for threshold, desc in kappa_interpretation if worst['overall_kappa'] >= threshold)}"
    )
    print()

    # Find their worst questions
    worst_questions = sorted(worst["questions"].items(), key=lambda x: x[1]["kappa"])
    print(f"   Worst performing questions:")
    for q_id, q_data in worst_questions[:3]:
        q_name = next(name for qid, name in questions if qid == q_id)
        print(f"      • {q_name}: κ={q_data['kappa']:.3f}, {q_data['agreement_pct']:.1%} agreement")
    print()

    # Range analysis
    kappa_range = best["overall_kappa"] - worst["overall_kappa"]
    print(f"📊 VARIABILITY ANALYSIS:")
    print(f"   Range of overall Kappa: {kappa_range:.3f}")
    print(
        f"   This indicates {'substantial' if kappa_range > 0.3 else 'moderate' if kappa_range > 0.2 else 'slight'} variability in how closely"
    )
    print(f"   different participants align with the LLM's classifications.")
    print()

    # Question-specific variability
    print("   Kappa range by question:")
    for q_id, q_name in questions:
        kappas = [p["questions"][q_id]["kappa"] for p in participants if q_id in p["questions"]]
        q_range = max(kappas) - min(kappas)
        max_p = max([p for p in participants if q_id in p["questions"]], key=lambda x: x["questions"][q_id]["kappa"])
        min_p = min([p for p in participants if q_id in p["questions"]], key=lambda x: x["questions"][q_id]["kappa"])
        print(f"      • {q_name}: {q_range:.3f} (range: {min(kappas):.3f} to {max(kappas):.3f})")

    print()

    # Consistency analysis
    print("=" * 100)
    print("CONSISTENCY PATTERNS")
    print("=" * 100)
    print()

    # Find participants who are consistently good or consistently poor
    consistent_high = []
    consistent_low = []

    for p in participants:
        kappas = [q_data["kappa"] for q_data in p["questions"].values()]
        avg_kappa = sum(kappas) / len(kappas)
        std_kappa = (sum((k - avg_kappa) ** 2 for k in kappas) / len(kappas)) ** 0.5

        if avg_kappa > 0.4 and std_kappa < 0.15:
            consistent_high.append((p["id"], avg_kappa, std_kappa))
        elif avg_kappa < 0.3 and std_kappa < 0.15:
            consistent_low.append((p["id"], avg_kappa, std_kappa))

    if consistent_high:
        print("✅ CONSISTENTLY HIGH AGREEMENT (high mean, low variability):")
        for p_id, avg, std in consistent_high:
            print(f"   • {p_id}: mean κ={avg:.3f}, std={std:.3f}")
        print()

    if consistent_low:
        print("⚠️  CONSISTENTLY LOW AGREEMENT (low mean, low variability):")
        for p_id, avg, std in consistent_low:
            print(f"   • {p_id}: mean κ={avg:.3f}, std={std:.3f}")
        print()

    # Variable performers
    variable = []
    for p in participants:
        kappas = [q_data["kappa"] for q_data in p["questions"].values()]
        avg_kappa = sum(kappas) / len(kappas)
        std_kappa = (sum((k - avg_kappa) ** 2 for k in kappas) / len(kappas)) ** 0.5

        if std_kappa > 0.2:
            variable.append((p["id"], avg_kappa, std_kappa, min(kappas), max(kappas)))

    if variable:
        print("📊 VARIABLE PERFORMERS (high standard deviation across questions):")
        for p_id, avg, std, min_k, max_k in variable:
            print(f"   • {p_id}: mean κ={avg:.3f}, std={std:.3f}, range=[{min_k:.3f}, {max_k:.3f}]")
        print()


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze individual participant agreement with LLM")
    parser.add_argument(
        "--excluded",
        action="store_true",
        help="Use excluded dataset (without participant 6638e8aa3d1f38846080806a)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent

    if args.excluded:
        filename = "human-validation-servers-scores-excluded.json"
    else:
        filename = "human-validation-servers-scores.json"

    orig_path = project_root / "output-validation" / "cl-validation" / filename

    if not orig_path.exists():
        print(f"Error: Validation results not found at {orig_path}", file=sys.stderr)
        sys.exit(1)

    results = load_results(orig_path)
    analyze_participant_agreement(results)


if __name__ == "__main__":
    main()

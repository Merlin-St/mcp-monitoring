#!/usr/bin/env python3
"""
Compare all validation scenarios: original vs alternative, with and without excluded participant.

Shows the impact of excluding participant 6638e8aa3d1f38846080806a on agreement metrics.
"""

import json
import sys
from pathlib import Path


def load_results(path: Path) -> dict:
    """Load validation results from JSON file."""
    with open(path) as f:
        return json.load(f)


def compare_all_scenarios() -> None:
    """Generate comprehensive comparison across all scenarios."""
    project_root = Path(__file__).parent.parent.parent
    validation_dir = project_root / "output-validation" / "cl-validation"

    # Load all four scenarios
    scenarios = {
        "Original (All)": validation_dir / "human-validation-servers-scores.json",
        "Original (Excluded)": validation_dir / "human-validation-servers-scores-excluded.json",
        "Alternative (All)": validation_dir / "human-validation-servers-scores-alternative.json",
        "Alternative (Excluded)": validation_dir / "human-validation-servers-scores-alternative-excluded.json",
    }

    results = {}
    for name, path in scenarios.items():
        if not path.exists():
            print(f"Warning: {name} not found at {path}", file=sys.stderr)
            continue
        results[name] = load_results(path)

    if len(results) < 4:
        print("Error: Not all validation result files found", file=sys.stderr)
        sys.exit(1)

    print("=" * 120)
    print("COMPREHENSIVE VALIDATION COMPARISON")
    print("Impact of excluding participant 6638e8aa3d1f38846080806a (lowest agreement)")
    print("=" * 120)
    print()

    # Overall summary comparison
    print("OVERALL SUMMARY")
    print("-" * 120)
    print(f"{'Metric':<50} {'Original (All)':<15} {'Original (Excl)':<15} {'Alternative (All)':<15} {'Alternative (Excl)':<15}")
    print("-" * 120)

    # Number of participants
    print(
        f"{'Number of Participants':<50} "
        f"{results['Original (All)']['summary']['n_participants']:<15} "
        f"{results['Original (Excluded)']['summary']['n_participants']:<15} "
        f"{results['Alternative (All)']['summary']['n_participants']:<15} "
        f"{results['Alternative (Excluded)']['summary']['n_participants']:<15}"
    )

    # Overall agreement
    orig_all = results["Original (All)"]["summary"]["overall_agreement_with_llm"]
    orig_excl = results["Original (Excluded)"]["summary"]["overall_agreement_with_llm"]
    alt_all = results["Alternative (All)"]["summary"]["overall_agreement_with_llm"]
    alt_excl = results["Alternative (Excluded)"]["summary"]["overall_agreement_with_llm"]

    print(
        f"{'Overall Agreement with LLM':<50} "
        f"{orig_all:<14.1%} "
        f"{orig_excl:<14.1%} "
        f"{alt_all:<14.1%} "
        f"{alt_excl:<14.1%}"
    )

    # Improvement from exclusion
    orig_improve = orig_excl - orig_all
    alt_improve = alt_excl - alt_all

    print(
        f"{'  → Improvement from exclusion':<50} "
        f"{'':<15} "
        f"{orig_improve:<+14.1%} "
        f"{'':<15} "
        f"{alt_improve:<+14.1%}"
    )

    # Inter-rater reliability
    irr_orig_all = results["Original (All)"]["summary"]["overall_inter_rater_reliability"]
    irr_orig_excl = results["Original (Excluded)"]["summary"]["overall_inter_rater_reliability"]
    irr_alt_all = results["Alternative (All)"]["summary"]["overall_inter_rater_reliability"]
    irr_alt_excl = results["Alternative (Excluded)"]["summary"]["overall_inter_rater_reliability"]

    print(
        f"{'Overall Inter-Rater Reliability (κ)':<50} "
        f"{irr_orig_all:<15.3f} "
        f"{irr_orig_excl:<15.3f} "
        f"{irr_alt_all:<15.3f} "
        f"{irr_alt_excl:<15.3f}"
    )

    # Improvement from exclusion
    irr_orig_improve = irr_orig_excl - irr_orig_all
    irr_alt_improve = irr_alt_excl - irr_alt_all

    print(
        f"{'  → Improvement from exclusion':<50} "
        f"{'':<15} "
        f"{irr_orig_improve:<+15.3f} "
        f"{'':<15} "
        f"{irr_alt_improve:<+15.3f}"
    )

    print()
    print()

    # Per-question comparison
    questions = [
        ("func_main", "Main functionality"),
        ("func_sub", "Sub-functionality"),
        ("onet_l1", "O*NET category"),
        ("q3", "Industry generality"),
        ("q4", "Environment generality"),
        ("q5", "Payment autonomy"),
    ]

    print("PER-QUESTION BREAKDOWN")
    print("=" * 120)
    print()

    for q_id, q_name in questions:
        print(f"Question: {q_name} ({q_id})")
        print("-" * 120)
        print(f"{'Metric':<50} {'Original (All)':<15} {'Original (Excl)':<15} {'Alternative (All)':<15} {'Alternative (Excl)':<15}")
        print("-" * 120)

        # Mean Kappa
        orig_all_k = results["Original (All)"]["by_question"][q_id]["mean_kappa"]
        orig_excl_k = results["Original (Excluded)"]["by_question"][q_id]["mean_kappa"]
        alt_all_k = results["Alternative (All)"]["by_question"][q_id]["mean_kappa"]
        alt_excl_k = results["Alternative (Excluded)"]["by_question"][q_id]["mean_kappa"]

        print(
            f"{'Mean Kappa (vs LLM)':<50} "
            f"{orig_all_k:<15.3f} "
            f"{orig_excl_k:<15.3f} "
            f"{alt_all_k:<15.3f} "
            f"{alt_excl_k:<15.3f}"
        )

        # Improvement
        orig_k_improve = orig_excl_k - orig_all_k
        alt_k_improve = alt_excl_k - alt_all_k

        print(
            f"{'  → Improvement from exclusion':<50} "
            f"{'':<15} "
            f"{orig_k_improve:<+15.3f} "
            f"{'':<15} "
            f"{alt_k_improve:<+15.3f}"
        )

        # Agreement %
        orig_all_a = results["Original (All)"]["by_question"][q_id]["agreement_pct"]
        orig_excl_a = results["Original (Excluded)"]["by_question"][q_id]["agreement_pct"]
        alt_all_a = results["Alternative (All)"]["by_question"][q_id]["agreement_pct"]
        alt_excl_a = results["Alternative (Excluded)"]["by_question"][q_id]["agreement_pct"]

        print(
            f"{'Agreement %':<50} "
            f"{orig_all_a:<14.1%} "
            f"{orig_excl_a:<14.1%} "
            f"{alt_all_a:<14.1%} "
            f"{alt_excl_a:<14.1%}"
        )

        # Improvement
        orig_a_improve = orig_excl_a - orig_all_a
        alt_a_improve = alt_excl_a - alt_all_a

        print(
            f"{'  → Improvement from exclusion':<50} "
            f"{'':<15} "
            f"{orig_a_improve:<+14.1%} "
            f"{'':<15} "
            f"{alt_a_improve:<+14.1%}"
        )

        # Fleiss Kappa
        orig_all_f = results["Original (All)"]["by_question"][q_id]["fleiss_kappa"]
        orig_excl_f = results["Original (Excluded)"]["by_question"][q_id]["fleiss_kappa"]
        alt_all_f = results["Alternative (All)"]["by_question"][q_id]["fleiss_kappa"]
        alt_excl_f = results["Alternative (Excluded)"]["by_question"][q_id]["fleiss_kappa"]

        if all(k is not None for k in [orig_all_f, orig_excl_f, alt_all_f, alt_excl_f]):
            print(
                f"{'Fleiss Kappa (inter-rater)':<50} "
                f"{orig_all_f:<15.3f} "
                f"{orig_excl_f:<15.3f} "
                f"{alt_all_f:<15.3f} "
                f"{alt_excl_f:<15.3f}"
            )

            # Improvement
            orig_f_improve = orig_excl_f - orig_all_f
            alt_f_improve = alt_excl_f - alt_all_f

            print(
                f"{'  → Improvement from exclusion':<50} "
                f"{'':<15} "
                f"{orig_f_improve:<+15.3f} "
                f"{'':<15} "
                f"{alt_f_improve:<+15.3f}"
            )

        print()

    # Key findings
    print("=" * 120)
    print("KEY FINDINGS")
    print("=" * 120)
    print()

    print("📊 IMPACT OF EXCLUDING PARTICIPANT 6638e8aa3d1f38846080806a:")
    print()

    print(f"   Overall Agreement:")
    print(f"      • Original model:    {orig_all:.1%} → {orig_excl:.1%} (improvement: {orig_improve:+.1%})")
    print(f"      • Alternative model: {alt_all:.1%} → {alt_excl:.1%} (improvement: {alt_improve:+.1%})")
    print()

    print(f"   Inter-Rater Reliability:")
    print(f"      • Original model:    κ={irr_orig_all:.3f} → κ={irr_orig_excl:.3f} (improvement: {irr_orig_improve:+.3f})")
    print(f"      • Alternative model: κ={irr_alt_all:.3f} → κ={irr_alt_excl:.3f} (improvement: {irr_alt_improve:+.3f})")
    print()

    # Find questions with biggest improvement
    improvements = []
    for q_id, q_name in questions:
        orig_k_improve = results["Original (Excluded)"]["by_question"][q_id]["mean_kappa"] - results["Original (All)"]["by_question"][q_id]["mean_kappa"]
        improvements.append((q_name, q_id, orig_k_improve))

    improvements.sort(key=lambda x: x[2], reverse=True)

    print("   Questions with largest improvement (Original model):")
    for q_name, q_id, improve in improvements[:3]:
        print(f"      • {q_name}: κ improvement of {improve:+.3f}")
    print()

    # Interpretation
    kappa_ranges = [
        (0.41, 1.00, "Moderate or better"),
        (0.21, 0.40, "Fair"),
        (0.00, 0.20, "Slight"),
        (-1.00, 0.00, "Poor"),
    ]

    print("   Agreement Level Changes:")
    for threshold_low, threshold_high, desc in kappa_ranges:
        if threshold_low <= irr_orig_all <= threshold_high:
            print(f"      • Original (All):      κ={irr_orig_all:.3f} - {desc}")
        if threshold_low <= irr_orig_excl <= threshold_high:
            print(f"      • Original (Excluded): κ={irr_orig_excl:.3f} - {desc}")
    print()

    print("✅ CONCLUSION:")
    print(f"   Excluding the lowest-performing participant improved overall agreement by {orig_improve:.1%}")
    print(f"   and inter-rater reliability by {irr_orig_improve:.3f} (κ), moving from '{next(d for l, h, d in kappa_ranges if l <= irr_orig_all <= h)}'")
    print(f"   to '{next(d for l, h, d in kappa_ranges if l <= irr_orig_excl <= h)}' agreement level.")
    print()


def main():
    """Main execution function."""
    compare_all_scenarios()


if __name__ == "__main__":
    main()

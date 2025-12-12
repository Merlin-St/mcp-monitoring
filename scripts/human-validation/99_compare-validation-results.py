#!/usr/bin/env python3
"""
Compare human validation results between different LLM classification variants.

Loads two validation result JSON files and generates a comparison report.
"""

import json
import sys
from pathlib import Path


def load_results(path: Path) -> dict:
    """Load validation results from JSON file."""
    with open(path) as f:
        return json.load(f)


def compare_results(original: dict, alternative: dict) -> None:
    """Generate comparison report between two validation results."""
    print("=" * 80)
    print("HUMAN VALIDATION COMPARISON")
    print("Original (clservers_classified.csv.gz) vs Alternative (clservers_classified_alternative.csv.gz)")
    print("=" * 80)
    print()

    # Overall Summary
    print("OVERALL SUMMARY")
    print("-" * 80)
    orig_summary = original["summary"]
    alt_summary = alternative["summary"]

    print(f"{'Metric':<50} {'Original':>12} {'Alternative':>12} {'Diff':>8}")
    print("-" * 80)

    overall_agreement_orig = orig_summary["overall_agreement_with_llm"]
    overall_agreement_alt = alt_summary["overall_agreement_with_llm"]
    diff_agreement = overall_agreement_alt - overall_agreement_orig

    print(
        f"{'Overall Agreement with LLM':<50} {overall_agreement_orig:>11.1%} {overall_agreement_alt:>11.1%} {diff_agreement:>+7.1%}"
    )

    irr_orig = orig_summary["overall_inter_rater_reliability"]
    irr_alt = alt_summary["overall_inter_rater_reliability"]
    diff_irr = irr_alt - irr_orig

    print(f"{'Overall Inter-Rater Reliability (κ)':<50} {irr_orig:>12.3f} {irr_alt:>12.3f} {diff_irr:>+8.3f}")
    print()

    # Per-Question Comparison
    print("PER-QUESTION BREAKDOWN")
    print("-" * 80)
    print(f"{'Question':<15} {'Metric':<30} {'Original':>12} {'Alternative':>12} {'Diff':>8}")
    print("-" * 80)

    questions = [
        ("func_main", "Main functionality level"),
        ("func_sub", "Sub-functionality classification"),
        ("onet_l1", "O*NET Level 1 category"),
        ("q3", "Industry generality"),
        ("q4", "Environment generality"),
        ("q5", "Payment autonomy"),
    ]

    for q_id, q_name in questions:
        orig_q = original["by_question"][q_id]
        alt_q = alternative["by_question"][q_id]

        # Mean Kappa
        orig_kappa = orig_q["mean_kappa"]
        alt_kappa = alt_q["mean_kappa"]
        diff_kappa = alt_kappa - orig_kappa

        # Agreement %
        orig_agree = orig_q["agreement_pct"]
        alt_agree = alt_q["agreement_pct"]
        diff_agree = alt_agree - orig_agree

        # Fleiss Kappa
        orig_fleiss = orig_q["fleiss_kappa"]
        alt_fleiss = alt_q["fleiss_kappa"]
        diff_fleiss = alt_fleiss - orig_fleiss if (orig_fleiss is not None and alt_fleiss is not None) else None

        print(f"{q_id:<15} {'Mean Kappa (vs LLM)':<30} {orig_kappa:>12.3f} {alt_kappa:>12.3f} {diff_kappa:>+8.3f}")
        print(f"{'':<15} {'Agreement %':<30} {orig_agree:>11.1%} {alt_agree:>11.1%} {diff_agree:>+7.1%}")
        if diff_fleiss is not None:
            print(
                f"{'':<15} {'Fleiss Kappa (inter-rater)':<30} {orig_fleiss:>12.3f} {alt_fleiss:>12.3f} {diff_fleiss:>+8.3f}"
            )
        print()

    # Key Differences
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()

    # Find biggest improvements
    improvements = []
    declines = []

    for q_id, q_name in questions:
        orig_q = original["by_question"][q_id]
        alt_q = alternative["by_question"][q_id]

        diff_kappa = alt_q["mean_kappa"] - orig_q["mean_kappa"]
        diff_agree = alt_q["agreement_pct"] - orig_q["agreement_pct"]

        if diff_kappa > 0.1 or diff_agree > 0.1:
            improvements.append((q_name, q_id, diff_kappa, diff_agree))
        elif diff_kappa < -0.1 or diff_agree < -0.1:
            declines.append((q_name, q_id, diff_kappa, diff_agree))

    if improvements:
        print("✅ IMPROVEMENTS (Alternative performs better):")
        for name, q_id, diff_k, diff_a in improvements:
            print(f"   • {name} ({q_id}): Kappa {diff_k:+.3f}, Agreement {diff_a:+.1%}")
        print()

    if declines:
        print("⚠️  DECLINES (Original performs better):")
        for name, q_id, diff_k, diff_a in declines:
            print(f"   • {name} ({q_id}): Kappa {diff_k:+.3f}, Agreement {diff_a:+.1%}")
        print()

    if not improvements and not declines:
        print("📊 Performance is similar across both classification variants.")
        print()

    # Interpretation
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()

    if overall_agreement_alt > overall_agreement_orig:
        print(
            f"The alternative classification shows {(overall_agreement_alt - overall_agreement_orig):.1%} higher overall agreement"
        )
        print("with human validators, suggesting it may be more aligned with human judgment.")
    elif overall_agreement_alt < overall_agreement_orig:
        print(
            f"The original classification shows {(overall_agreement_orig - overall_agreement_alt):.1%} higher overall agreement"
        )
        print("with human validators, suggesting it may be more aligned with human judgment.")
    else:
        print("Both classifications show equivalent overall agreement with human validators.")

    print()

    # Kappa interpretation
    kappa_ranges = [
        (0.81, 1.00, "Almost perfect agreement"),
        (0.61, 0.80, "Substantial agreement"),
        (0.41, 0.60, "Moderate agreement"),
        (0.21, 0.40, "Fair agreement"),
        (0.00, 0.20, "Slight agreement"),
        (-1.00, 0.00, "Poor agreement (worse than chance)"),
    ]

    for irr, label in [(irr_orig, "Original"), (irr_alt, "Alternative")]:
        for low, high, desc in kappa_ranges:
            if low <= irr <= high:
                print(f"{label} inter-rater reliability (κ={irr:.3f}): {desc}")
                break


def main():
    """Main execution function."""
    project_root = Path(__file__).parent.parent.parent  # Go up to project root from scripts/human-validation/
    orig_path = project_root / "output-validation" / "cl-validation" / "human-validation-servers-scores.json"
    alt_path = project_root / "output-validation" / "cl-validation" / "human-validation-servers-scores-alternative.json"

    if not orig_path.exists():
        print(f"Error: Original validation results not found at {orig_path}", file=sys.stderr)
        sys.exit(1)

    if not alt_path.exists():
        print(f"Error: Alternative validation results not found at {alt_path}", file=sys.stderr)
        sys.exit(1)

    original = load_results(orig_path)
    alternative = load_results(alt_path)

    compare_results(original, alternative)


if __name__ == "__main__":
    main()

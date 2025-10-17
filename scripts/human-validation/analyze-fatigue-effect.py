#!/usr/bin/env python3
"""
Analyze fatigue effects in human validation by comparing agreement in first 50 vs. second 50 servers.

Examines whether participants showed declining agreement with LLM classifications
over time, which could indicate fatigue or reduced attention.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/analyze_fatigue_effect.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Question mapping from human validation to LLM fields
QUESTION_MAPPING = {
    "func_main": {
        "llm_field": "highest_automation_func",
        "mapping": {"perception": 1, "reasoning": 2, "action": 3},
    },
    "func_sub": {
        "llm_field": "main_automation_subfunc",
        "mapping": None,  # Keep as strings
    },
    "onet_l1": {
        "llm_field": "main_onet_task_level1",
        "mapping": None,  # Keep as strings
    },
    "q3": {
        "llm_field": "generality_industry",
        "mapping": None,  # Numeric already
    },
    "q4": {
        "llm_field": "generality_environment",
        "mapping": None,  # Numeric already
    },
    "q5": {
        "llm_field": "payments_autonomy",
        "mapping": None,  # Numeric already
    },
}


def load_gorilla_data(path: Path) -> pd.DataFrame:
    """Load human validation data from Gorilla CSV."""
    logger.info(f"Loading Gorilla data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows")
    return df


def load_llm_classifications(path: Path) -> pd.DataFrame:
    """Load LLM classifications from clservers_classified.csv."""
    logger.info(f"Loading LLM classifications from {path}")
    df = pd.read_csv(path, low_memory=False)
    # Normalize server names to lowercase
    df["server_name"] = df["server_name"].str.lower()
    logger.info(f"Loaded {len(df)} LLM classifications")
    return df


def load_onet_mapping(project_root: Path) -> dict:
    """Load O*NET L1 cluster mapping from task_clusters_names.csv."""
    onet_file = project_root / "data" / "internal-task-clusters" / "task_clusters_names.csv"

    if not onet_file.exists():
        logger.warning(f"O*NET mapping file not found: {onet_file}")
        return {}

    try:
        df = pd.read_csv(onet_file)
        mapping = df[["level1_cluster", "level1_name"]].drop_duplicates()
        mapping_dict = dict(zip(mapping["level1_cluster"], mapping["level1_name"]))
        logger.info(f"Loaded {len(mapping_dict)} O*NET L1 cluster mappings")
        return mapping_dict
    except Exception as e:
        logger.error(f"Error loading O*NET mapping: {e}")
        return {}


def prepare_data_for_question(
    gorilla_df: pd.DataFrame, llm_df: pd.DataFrame, question: str, onet_mapping: dict = None
) -> pd.DataFrame:
    """Prepare merged data for a specific question."""
    question_info = QUESTION_MAPPING.get(question)
    if not question_info:
        return pd.DataFrame()

    llm_field = question_info["llm_field"]
    if llm_field not in llm_df.columns:
        logger.warning(f"LLM field {llm_field} not found for question {question}")
        return pd.DataFrame()

    # Filter gorilla data for this question
    question_data = gorilla_df[gorilla_df["question"] == question].copy()
    question_data["servername"] = question_data["servername"].str.lower()

    # Merge with LLM data
    merged = question_data.merge(
        llm_df[["server_name", llm_field]],
        left_on="servername",
        right_on="server_name",
        how="inner",
    )

    if len(merged) == 0:
        return pd.DataFrame()

    # Map values to comparable scales
    mapping = question_info.get("mapping")
    if mapping:
        # Map human responses using provided mapping
        merged["human_value_mapped"] = merged["Value"].map(mapping)
        # Map LLM responses using same mapping
        merged["llm_value_mapped"] = merged[llm_field].map(mapping)
    else:
        # For numeric or string questions, convert to appropriate type
        if question in ["q3", "q4", "q5"]:
            # Numeric questions
            merged["human_value_mapped"] = pd.to_numeric(merged["Value"], errors="coerce")
            merged["llm_value_mapped"] = pd.to_numeric(merged[llm_field], errors="coerce")
        elif question == "onet_l1" and onet_mapping:
            # Special case: Map human codes (L1_01, L1_04, etc.) to full descriptions
            merged["human_value_mapped"] = merged["Value"].map(onet_mapping)
            # LLM already has descriptions, keep as is
            merged["llm_value_mapped"] = merged[llm_field].astype(str)
            logger.info(f"Mapped {merged['human_value_mapped'].notna().sum()} onet_l1 human codes to descriptions")
        else:
            # String questions - keep as is
            merged["human_value_mapped"] = merged["Value"].astype(str)
            merged["llm_value_mapped"] = merged[llm_field].astype(str)

    # Remove rows with any missing values
    merged = merged.dropna(subset=["human_value_mapped", "llm_value_mapped"])

    return merged


def calculate_kappa_by_half(data: pd.DataFrame, participant_id: str) -> dict:
    """Calculate Cohen's Kappa for first and second half of servers for a participant."""
    # Filter data for this participant
    participant_data = data[data["Participant Public ID"] == participant_id].copy()

    if len(participant_data) == 0:
        return None

    # Sort by event index to maintain chronological order
    participant_data = participant_data.sort_values("Event Index")

    # Get unique servers in chronological order (based on first encounter)
    server_first_appearance = participant_data.groupby("servername")["Event Index"].first().sort_values()
    n_servers = len(server_first_appearance)

    if n_servers < 20:  # Need reasonable sample size
        return None

    # Split into first and second half
    midpoint = n_servers // 2
    first_half_servers = set(server_first_appearance.iloc[:midpoint].index)
    second_half_servers = set(server_first_appearance.iloc[midpoint:].index)

    # Get data for each half
    first_half = participant_data[participant_data["servername"].isin(first_half_servers)]
    second_half = participant_data[participant_data["servername"].isin(second_half_servers)]

    # Calculate kappa for each half
    if len(first_half) < 5 or len(second_half) < 5:
        return None

    try:
        kappa_first = cohen_kappa_score(
            first_half["human_value_mapped"], first_half["llm_value_mapped"]
        )
        kappa_second = cohen_kappa_score(
            second_half["human_value_mapped"], second_half["llm_value_mapped"]
        )

        return {
            "kappa_first_half": float(kappa_first),
            "kappa_second_half": float(kappa_second),
            "n_first_half": len(first_half_servers),
            "n_second_half": len(second_half_servers),
            "diff": float(kappa_first - kappa_second),  # Positive = worse in second half (fatigue)
        }
    except Exception as e:
        logger.warning(f"Error calculating kappa for {participant_id}: {e}")
        return None


def analyze_fatigue(gorilla_df: pd.DataFrame, llm_df: pd.DataFrame, onet_mapping: dict = None) -> dict:
    """Analyze fatigue effects across all participants and questions."""
    participants = gorilla_df["Participant Public ID"].unique()
    questions = ["func_main", "func_sub", "onet_l1", "q3", "q4", "q5"]

    results = {
        "by_participant": {},
        "by_question": {},
        "overall": {},
    }

    # Analyze each question separately
    for question in questions:
        logger.info(f"Analyzing question: {question}")
        question_data = prepare_data_for_question(gorilla_df, llm_df, question, onet_mapping)

        if question_data.empty:
            logger.warning(f"No data for question {question}")
            continue

        question_results = []
        for participant_id in participants:
            kappa_results = calculate_kappa_by_half(question_data, participant_id)
            if kappa_results:
                question_results.append(
                    {
                        "participant_id": participant_id,
                        **kappa_results,
                    }
                )

                # Add to participant results
                if participant_id not in results["by_participant"]:
                    results["by_participant"][participant_id] = {"questions": {}}
                results["by_participant"][participant_id]["questions"][question] = kappa_results

        if question_results:
            avg_diff = sum(r["diff"] for r in question_results) / len(question_results)
            results["by_question"][question] = {
                "participants": question_results,
                "avg_diff": avg_diff,
                "n_participants": len(question_results),
            }

    # Calculate participant averages
    for participant_id, p_data in results["by_participant"].items():
        question_diffs = [q["diff"] for q in p_data["questions"].values()]
        if question_diffs:
            p_data["avg_diff"] = sum(question_diffs) / len(question_diffs)
            p_data["n_questions"] = len(question_diffs)

    # Overall statistics
    all_diffs = []
    for p_data in results["by_participant"].values():
        for q_data in p_data["questions"].values():
            all_diffs.append(q_data["diff"])

    if all_diffs:
        results["overall"] = {
            "mean_diff": float(np.mean(all_diffs)),
            "median_diff": float(np.median(all_diffs)),
            "max_diff": float(np.max(all_diffs)),
            "min_diff": float(np.min(all_diffs)),
            "n_observations": len(all_diffs),
            "positive_diffs": sum(1 for d in all_diffs if d > 0),  # Fatigue cases
            "negative_diffs": sum(1 for d in all_diffs if d < 0),  # Improvement cases
        }

    return results


def print_results(results: dict) -> None:
    """Print formatted results of fatigue analysis."""
    print("=" * 100)
    print("FATIGUE EFFECT ANALYSIS")
    print("Comparing Agreement (Cohen's Kappa) in First Half vs. Second Half of Servers")
    print("=" * 100)
    print()

    # Check if we have results
    if not results.get("overall"):
        print("No results found. Check that data was properly loaded and merged.")
        return

    # Overall summary
    overall = results["overall"]
    print("OVERALL STATISTICS")
    print("-" * 100)
    print(f"Mean difference (first - second): {overall['mean_diff']:.3f}")
    print(f"Median difference: {overall['median_diff']:.3f}")
    print(f"Range: [{overall['min_diff']:.3f}, {overall['max_diff']:.3f}]")
    print(f"")
    print(f"Cases showing fatigue (κ_first > κ_second): {overall['positive_diffs']}/{overall['n_observations']}")
    print(
        f"Cases showing improvement (κ_first < κ_second): {overall['negative_diffs']}/{overall['n_observations']}"
    )
    print()
    print(f"Interpretation:")
    if overall["mean_diff"] > 0.05:
        print(f"  ⚠️  Evidence of FATIGUE: Agreement declined in second half by {overall['mean_diff']:.3f} on average")
    elif overall["mean_diff"] < -0.05:
        print(
            f"  ✅ Evidence of LEARNING: Agreement improved in second half by {abs(overall['mean_diff']):.3f} on average"
        )
    else:
        print(f"  ➡️  NO STRONG EFFECT: Agreement remained stable throughout the task")
    print()
    print()

    # By participant
    print("BY PARTICIPANT")
    print("-" * 100)
    print(f"{'Participant ID':<28} {'Avg Δκ':<10} {'Questions':<12} {'Evidence':<30}")
    print("-" * 100)

    # Sort by average difference (descending)
    participants = sorted(
        results["by_participant"].items(), key=lambda x: x[1].get("avg_diff", 0), reverse=True
    )

    for participant_id, p_data in participants:
        avg_diff = p_data.get("avg_diff", 0)
        n_questions = p_data.get("n_questions", 0)

        if avg_diff > 0.1:
            evidence = "Strong fatigue"
        elif avg_diff > 0.05:
            evidence = "Moderate fatigue"
        elif avg_diff < -0.1:
            evidence = "Strong improvement"
        elif avg_diff < -0.05:
            evidence = "Moderate improvement"
        else:
            evidence = "Stable"

        print(f"{participant_id:<28} {avg_diff:>9.3f} {n_questions:<12} {evidence:<30}")

    print()
    print()

    # Detailed breakdown by participant
    print("DETAILED BREAKDOWN BY PARTICIPANT")
    print("=" * 100)
    print()

    for participant_id, p_data in participants:
        if "questions" not in p_data:
            continue

        print(f"Participant: {participant_id}")
        print(f"Average difference: {p_data.get('avg_diff', 0):.3f}")
        print("-" * 100)
        print(f"{'Question':<20} {'κ First Half':<15} {'κ Second Half':<16} {'Difference':<12} {'N (First/Second)'}")
        print("-" * 100)

        for question, q_data in sorted(p_data["questions"].items()):
            print(
                f"{question:<20} {q_data['kappa_first_half']:>14.3f} {q_data['kappa_second_half']:>15.3f} "
                f"{q_data['diff']:>11.3f} {q_data['n_first_half']:>8}/{q_data['n_second_half']:<8}"
            )

        print()
        print()

    # By question
    print("BY QUESTION")
    print("=" * 100)
    print()

    questions_sorted = sorted(results["by_question"].items(), key=lambda x: x[1]["avg_diff"], reverse=True)

    for question, q_data in questions_sorted:
        print(f"Question: {question}")
        print(f"Average difference across participants: {q_data['avg_diff']:.3f}")
        print("-" * 100)
        print(f"{'Participant ID':<28} {'κ First':<12} {'κ Second':<12} {'Difference':<12}")
        print("-" * 100)

        for p_result in sorted(q_data["participants"], key=lambda x: x["diff"], reverse=True):
            print(
                f"{p_result['participant_id']:<28} {p_result['kappa_first_half']:>11.3f} "
                f"{p_result['kappa_second_half']:>11.3f} {p_result['diff']:>11.3f}"
            )

        print()
        print()


def main():
    """Main execution function."""
    project_root = Path(__file__).parent.parent.parent

    # Create logs directory if it doesn't exist
    (project_root / "logs").mkdir(exist_ok=True)

    # Load data
    gorilla_path = project_root / "data/external-cl-human-valid/data_exp_242261-vall_tasks.csv"
    llm_path = project_root / "data/final/clservers_classified.csv"

    if not gorilla_path.exists():
        logger.error(f"Gorilla data not found at {gorilla_path}")
        return

    if not llm_path.exists():
        logger.error(f"LLM classifications not found at {llm_path}")
        return

    gorilla_df = load_gorilla_data(gorilla_path)
    llm_df = load_llm_classifications(llm_path)

    # Load O*NET mapping for onet_l1 question
    onet_mapping = load_onet_mapping(project_root)

    # Analyze fatigue
    results = analyze_fatigue(gorilla_df, llm_df, onet_mapping)

    # Print results
    print_results(results)

    # Save results
    output_path = project_root / "output-validation/cl-validation/fatigue-analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()

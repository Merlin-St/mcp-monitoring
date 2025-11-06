#!/usr/bin/env python3
"""
Human Validation Servers Scoring Script

Analyzes human validation data for MCP servers from Gorilla experiments and calculates
inter-rater agreement metrics comparing human ratings with LLM server classifications.

Features:
- Cohen's Kappa for pairwise agreement (human vs LLM)
- Fleiss' Kappa for inter-rater reliability across all human raters
- Confusion matrices showing disagreement patterns
- Percentage agreement statistics
- Weighted Kappa for ordinal scales (payment autonomy levels)
- Support for multiple LLM classification variants (e.g., different models)

Input:
- data/external-cl-human-valid/data_exp_242261-vall_tasks.csv - Servers validation data
- data/final/clservers_classified*.csv - CLServers LLM classifications

Output:
- human-validation-servers-scores.json - Comprehensive agreement statistics
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/human_validation_servers.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Question mapping: human validation question IDs to LLM field names
# Supports both clservers and cltools files with different column names
QUESTION_MAPPING = {
    "func_main": {
        "llm_field": "highest_automation_func",  # clservers field
        "llm_field_alt": "tool_functionality_main",  # cltools field
        "description": "Main functionality level (perception/reasoning/action)",
        "type": "categorical",
        "mapping": {"perception": 1, "reasoning": 2, "action": 3},
    },
    "func_sub": {
        "llm_field": "main_automation_subfunc",  # clservers field
        "llm_field_alt": "tool_functionality_sub",  # cltools field
        "description": "Sub-category functionality classification",
        "type": "categorical",
        "mapping": None,  # Dynamic mapping based on actual values
    },
    "onet_l1": {
        "llm_field": "main_onet_task_level1",  # clservers field
        "llm_field_alt": "level1_name",  # cltools field
        "description": "O*NET Level 1 occupational category",
        "type": "categorical",
        "mapping": None,  # Dynamic mapping based on actual values
    },
    "q3": {
        "llm_field": "generality_industry",
        "description": "Industry generality (cross-industry vs industry-specific)",
        "type": "binary",
        "mapping": {0: 0, 1: 1},
    },
    "q4": {
        "llm_field": "generality_environment",
        "description": "Environment generality (open/untrusted vs trusted)",
        "type": "binary",
        "mapping": {0: 0, 1: 1},
    },
    "q5": {
        "llm_field": "payments_autonomy",
        "description": "Payment autonomy level (0=not payment, 1-4=increasing autonomy)",
        "type": "ordinal",
        "mapping": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
    },
    "generality": {
        "llm_field": "generality_combined",
        "description": "Combined generality (Narrow-purpose/Cross-purpose/General-purpose/Other)",
        "type": "categorical",
        "mapping": None,  # Dynamic mapping based on derived values
    },
}

# Additional human-only questions (not in LLM data)
HUMAN_ONLY_QUESTIONS = {
    "q0_notes": {
        "description": "Free-text notes about server functionality",
        "type": "text",
    },
}

# Minimum ratings threshold for Fleiss Kappa calculation
MIN_RATINGS_FOR_FLEISS = 50


def derive_generality_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive generality classification from q3 (industry) and q4 (environment).

    Categories:
    - Narrow-purpose: Trusted (q4=0) AND industry-specific (q3=0)
    - Cross-purpose: Trusted (q4=0) AND cross-industry (q3=1)
    - General-purpose: Untrusted (q4=1) AND cross-industry (q3=1)
    - Other: Untrusted (q4=1) AND industry-specific (q3=0) - edge case

    Args:
        df: DataFrame with q3_mapped and q4_mapped columns

    Returns:
        DataFrame with generality_mapped column added
    """
    df = df.copy()

    # Check if required columns exist
    if "q3_mapped" not in df.columns or "q4_mapped" not in df.columns:
        logger.warning("Cannot derive generality: q3_mapped or q4_mapped columns missing")
        return df

    def classify_generality(row):
        q3 = row.get("q3_mapped")
        q4 = row.get("q4_mapped")

        # Handle missing values
        if pd.isna(q3) or pd.isna(q4):
            return np.nan

        # Convert to int for comparison
        q3 = int(q3)
        q4 = int(q4)

        # Derive category
        if q4 == 0 and q3 == 0:
            return "narrow-purpose"
        elif q4 == 0 and q3 == 1:
            return "cross-purpose"
        elif q4 == 1 and q3 == 1:
            return "general-purpose"
        elif q4 == 1 and q3 == 0:
            return "other"
        else:
            return "unknown"

    df["generality_mapped"] = df.apply(classify_generality, axis=1)

    logger.info(f"Derived generality classification: {df['generality_mapped'].value_counts().to_dict()}")

    return df


def load_onet_l1_mapping(project_root: Path) -> dict:
    """
    Load O*NET Level 1 cluster mapping from task_clusters_names.csv.
    Maps codes like 'L1_01' to full descriptions.

    Args:
        project_root: Path to project root directory

    Returns:
        Dictionary mapping L1 codes to descriptions
    """
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


def load_human_validation_data(data_dir: Path) -> pd.DataFrame:
    """
    Load all human validation CSV files from the directory.

    Args:
        data_dir: Path to directory containing data_exp_*.csv files

    Returns:
        Combined DataFrame with all human validation responses
    """
    logger.info(f"Loading human validation data from {data_dir}")

    csv_files = list(data_dir.glob("data_exp_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No data_exp_*.csv files found in {data_dir}")

    logger.info(f"Found {len(csv_files)} validation data files")

    # Load and combine all CSV files
    dfs = []
    for csv_file in csv_files:
        logger.info(f"Loading {csv_file.name}")
        df = pd.read_csv(csv_file)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    # Filter to only include actual task responses (exclude header rows)
    combined_df = combined_df[combined_df["question"] != "question"]

    logger.info(
        f"Loaded {len(combined_df)} total responses from {combined_df['Participant Public ID'].nunique()} participants"
    )

    return combined_df


def get_llm_field_name(df: pd.DataFrame, question_id: str) -> str:
    """
    Get the correct LLM field name for a question, checking both primary and alternative fields.

    Args:
        df: DataFrame with LLM classifications
        question_id: Question identifier

    Returns:
        The field name to use, or None if not found
    """
    question_info = QUESTION_MAPPING.get(question_id)
    if not question_info:
        return None

    # Check primary field
    primary_field = question_info.get("llm_field")
    if primary_field and primary_field in df.columns:
        return primary_field

    # Check alternative field (for cltools)
    alt_field = question_info.get("llm_field_alt")
    if alt_field and alt_field in df.columns:
        return alt_field

    return None


def load_llm_classifications(csv_path: Path) -> pd.DataFrame:
    """
    Load LLM classifications from classification CSV file.
    Supports both clservers and cltools files with different column names.

    Args:
        csv_path: Path to classification CSV file

    Returns:
        DataFrame with LLM classifications
    """
    logger.info(f"Loading LLM classifications from {csv_path.name}")

    df = pd.read_csv(csv_path, low_memory=False)

    # Build list of relevant columns based on what exists in the file
    relevant_cols = ["server_name"]

    for question_id, question_info in QUESTION_MAPPING.items():
        # Check primary field
        primary_field = question_info.get("llm_field")
        if primary_field and primary_field in df.columns:
            relevant_cols.append(primary_field)

        # Check alternative field
        alt_field = question_info.get("llm_field_alt")
        if alt_field and alt_field in df.columns:
            relevant_cols.append(alt_field)

    # Remove duplicates while preserving order
    relevant_cols = list(dict.fromkeys(relevant_cols))

    df = df[relevant_cols]

    # Normalize server names to lowercase for consistent matching
    df["server_name"] = df["server_name"].str.lower()

    # Derive generality classification if q3 and q4 fields exist
    if "generality_industry" in df.columns and "generality_environment" in df.columns:
        def classify_generality_llm(row):
            q3 = row.get("generality_industry")
            q4 = row.get("generality_environment")

            # Handle missing values
            if pd.isna(q3) or pd.isna(q4):
                return np.nan

            # Convert to int for comparison
            q3 = int(q3)
            q4 = int(q4)

            # Derive category
            if q4 == 0 and q3 == 0:
                return "narrow-purpose"
            elif q4 == 0 and q3 == 1:
                return "cross-purpose"
            elif q4 == 1 and q3 == 1:
                return "general-purpose"
            elif q4 == 1 and q3 == 0:
                return "other"
            else:
                return "unknown"

        df["generality_combined"] = df.apply(classify_generality_llm, axis=1)
        logger.info(f"Derived LLM generality classification: {df['generality_combined'].value_counts().to_dict()}")

    logger.info(f"Loaded {len(df)} server classifications with {len(relevant_cols)-1} relevant fields")

    return df


def transform_human_data(human_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform human validation data into structured format.

    Pivots the long-form data (one row per question response) into wide format
    (one row per server per participant).

    Args:
        human_df: Raw human validation data

    Returns:
        Transformed DataFrame with one row per server per participant
    """
    logger.info("Transforming human validation data to wide format")

    # Select relevant columns
    cols = ["Participant Public ID", "servername", "question", "Value"]
    pivot_df = human_df[cols].copy()

    # Normalize server names to lowercase for consistent matching
    pivot_df["servername"] = pivot_df["servername"].str.lower()

    # Pivot to wide format
    wide_df = pivot_df.pivot_table(
        index=["servername", "Participant Public ID"],
        columns="question",
        values="Value",
        aggfunc="first",
    ).reset_index()

    # Rename participant column
    wide_df = wide_df.rename(columns={"Participant Public ID": "participant_id"})

    logger.info(f"Transformed to {len(wide_df)} server-participant pairs")

    return wide_df


def map_human_to_llm_scale(human_df: pd.DataFrame, onet_mapping: dict = None) -> pd.DataFrame:
    """
    Map human responses to LLM scale for comparison.
    For categorical questions without predefined mapping, keeps original string values.

    Args:
        human_df: Wide-format human validation data
        onet_mapping: Optional O*NET L1 cluster mapping (code -> description)

    Returns:
        DataFrame with mapped values ready for comparison
    """
    logger.info("Mapping human responses to LLM scales")

    df = human_df.copy()

    for question_id, question_info in QUESTION_MAPPING.items():
        if question_id not in df.columns:
            continue

        mapping = question_info.get("mapping")

        if mapping:
            # Apply predefined mapping (e.g., func_main)
            # First try direct mapping, then try converting to appropriate type
            df[f"{question_id}_mapped"] = df[question_id].map(mapping)
            # If mapping failed (NaN), try converting strings to the key type first
            if df[f"{question_id}_mapped"].isna().all():
                # Try converting values to numeric if mapping keys are numeric
                if isinstance(list(mapping.keys())[0], (int, float)):
                    df[f"{question_id}_mapped"] = pd.to_numeric(df[question_id], errors="coerce").map(mapping)
        elif question_info["type"] in ["binary", "ordinal"]:
            # Convert to numeric
            df[f"{question_id}_mapped"] = pd.to_numeric(df[question_id], errors="coerce")
        else:
            # Categorical without mapping
            # Special case: onet_l1 - map codes to descriptions
            if question_id == "onet_l1" and onet_mapping:
                df[f"{question_id}_mapped"] = df[question_id].map(onet_mapping)
                logger.info(f"Mapped {df[f'{question_id}_mapped'].notna().sum()} onet_l1 codes to descriptions")
            else:
                # Keep as string
                df[f"{question_id}_mapped"] = df[question_id]

    return df


def calculate_cohens_kappa(human_ratings: np.ndarray, llm_ratings: np.ndarray) -> float:
    """
    Calculate Cohen's Kappa for pairwise agreement.

    Args:
        human_ratings: Array of human ratings
        llm_ratings: Array of LLM ratings

    Returns:
        Cohen's Kappa score (-1 to 1, where 1 is perfect agreement)
    """
    # Remove any pairs with missing data
    mask = ~(pd.isna(human_ratings) | pd.isna(llm_ratings))
    human_clean = human_ratings[mask]
    llm_clean = llm_ratings[mask]

    if len(human_clean) == 0:
        return np.nan

    return cohen_kappa_score(human_clean, llm_clean)


def calculate_weighted_kappa(human_ratings: np.ndarray, llm_ratings: np.ndarray) -> float:
    """
    Calculate weighted Cohen's Kappa for ordinal scales.

    Args:
        human_ratings: Array of human ratings
        llm_ratings: Array of LLM ratings

    Returns:
        Weighted Cohen's Kappa score
    """
    # Remove any pairs with missing data
    mask = ~(pd.isna(human_ratings) | pd.isna(llm_ratings))
    human_clean = human_ratings[mask]
    llm_clean = llm_ratings[mask]

    if len(human_clean) == 0:
        return np.nan

    return cohen_kappa_score(human_clean, llm_clean, weights="linear")


def calculate_fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """
    Calculate Fleiss' Kappa for inter-rater reliability.

    Args:
        ratings_matrix: Matrix of ratings (n_items x n_raters)

    Returns:
        Fleiss' Kappa score
    """
    # Implementation of Fleiss' Kappa
    # ratings_matrix: rows are items (servers), columns are raters (participants)

    n_items, n_raters = ratings_matrix.shape

    # Get unique categories
    categories = np.unique(ratings_matrix[~pd.isna(ratings_matrix)])

    if len(categories) == 0:
        return np.nan

    # Build contingency table: n_items x n_categories
    contingency = np.zeros((n_items, len(categories)))

    for i in range(n_items):
        for j, cat in enumerate(categories):
            contingency[i, j] = np.sum(ratings_matrix[i, :] == cat)

    # Calculate P_i (proportion of agreement for item i)
    P_i = (np.sum(contingency**2, axis=1) - n_raters) / (n_raters * (n_raters - 1))

    # Calculate P_bar (mean of P_i)
    P_bar = np.mean(P_i)

    # Calculate P_e (expected agreement by chance)
    p_j = np.sum(contingency, axis=0) / (n_items * n_raters)
    P_e = np.sum(p_j**2)

    # Calculate Fleiss' Kappa
    if P_e == 1.0:
        return np.nan

    kappa = (P_bar - P_e) / (1 - P_e)

    return kappa


def calculate_confusion_matrix(human_ratings: np.ndarray, llm_ratings: np.ndarray, labels: list) -> np.ndarray:
    """
    Calculate confusion matrix showing disagreement patterns.

    Args:
        human_ratings: Array of human ratings
        llm_ratings: Array of LLM ratings
        labels: List of possible label values

    Returns:
        Confusion matrix as numpy array
    """
    # Remove any pairs with missing data
    mask = ~(pd.isna(human_ratings) | pd.isna(llm_ratings))
    human_clean = human_ratings[mask]
    llm_clean = llm_ratings[mask]

    if len(human_clean) == 0:
        return np.array([])

    return confusion_matrix(llm_clean, human_clean, labels=labels)


def analyze_question(
    human_df: pd.DataFrame, llm_df: pd.DataFrame, question_id: str, question_info: dict
) -> dict[str, Any]:
    """
    Analyze agreement for a specific question across all participants.

    For func_sub (conditional question), only analyzes cases where there's agreement on func_main.

    Args:
        human_df: Human validation data (wide format)
        llm_df: LLM classifications
        question_id: Question identifier (e.g., 'func_main', 'q3')
        question_info: Question metadata from QUESTION_MAPPING

    Returns:
        Dictionary with agreement statistics for this question
    """
    logger.info(f"Analyzing question: {question_id} - {question_info['description']}")

    # Get the correct LLM field name for this file
    llm_field = get_llm_field_name(llm_df, question_id)
    if not llm_field:
        logger.warning(f"LLM field not found for question {question_id}")
        return {"error": "LLM field not found"}

    human_col = f"{question_id}_mapped"

    # Check if human column exists
    if human_col not in human_df.columns:
        logger.warning(f"Human column {human_col} not found")
        return {"error": "Human column not found"}

    # Merge on server name - need to include func_main fields for conditional filtering
    merge_cols = ["server_name", llm_field]

    # For func_sub, we need func_main fields to filter on agreement
    if question_id == "func_sub":
        func_main_llm_field = get_llm_field_name(llm_df, "func_main")
        if func_main_llm_field and func_main_llm_field in llm_df.columns:
            merge_cols.append(func_main_llm_field)

    merged = human_df.merge(llm_df[merge_cols], left_on="servername", right_on="server_name", how="inner")

    if len(merged) == 0:
        logger.warning(f"No matching servers found for question {question_id}")
        return {"error": "No matching servers"}

    # For func_sub, filter to only cases where there's agreement on func_main
    if question_id == "func_sub" and "func_main_mapped" in human_df.columns:
        func_main_llm_field = get_llm_field_name(llm_df, "func_main")
        if func_main_llm_field and func_main_llm_field in merged.columns:
            # Map LLM func_main to same scale as human
            func_main_mapping = QUESTION_MAPPING["func_main"].get("mapping")
            if func_main_mapping:
                merged[f"{func_main_llm_field}_mapped"] = merged[func_main_llm_field].map(func_main_mapping)
            else:
                merged[f"{func_main_llm_field}_mapped"] = merged[func_main_llm_field]

            # Filter to only rows where func_main agrees
            initial_count = len(merged)
            merged = merged[merged["func_main_mapped"] == merged[f"{func_main_llm_field}_mapped"]]
            filtered_count = len(merged)

            # Additional filtering: Remove LLM "sensors" predictions for non-perception cases
            # This is needed because the LLM incorrectly predicts "sensors" for reasoning/action cases
            sensors_before = len(merged[merged[llm_field] == "sensors"])
            merged = merged[
                ~(
                    (merged[f"{func_main_llm_field}_mapped"] != 1)  # Not perception (perception=1)
                    & (merged[llm_field] == "sensors")  # LLM predicts sensors
                )
            ]
            sensors_after = len(merged[merged[llm_field] == "sensors"])
            sensors_removed = sensors_before - sensors_after

            logger.info(
                f"Conditional filtering for func_sub: {initial_count} total responses, "
                f"{filtered_count} where func_main agrees ({filtered_count/initial_count:.1%})"
            )
            if sensors_removed > 0:
                logger.info(
                    f"Removed {sensors_removed} invalid LLM 'sensors' predictions for non-perception cases "
                    f"(final count: {len(merged)})"
                )

            if len(merged) == 0:
                logger.warning("No responses with func_main agreement for func_sub analysis")
                return {
                    "error": "No responses with func_main agreement",
                    "note": "func_sub is conditional on func_main agreement",
                }

    # Get unique participants
    participants = merged["participant_id"].unique()
    n_participants = len(participants)

    # Map LLM ratings if needed (for categorical with predefined mapping)
    if question_info.get("mapping"):
        merged[f"{llm_field}_mapped"] = merged[llm_field].map(question_info["mapping"])
        llm_field_to_use = f"{llm_field}_mapped"
    else:
        llm_field_to_use = llm_field

    # Calculate Cohen's Kappa for each participant vs LLM
    kappa_scores = []
    for participant in participants:
        participant_data = merged[merged["participant_id"] == participant]
        human_ratings = participant_data[human_col].values
        llm_ratings = participant_data[llm_field_to_use].values

        if question_info["type"] == "ordinal":
            kappa = calculate_weighted_kappa(human_ratings, llm_ratings)
        else:
            kappa = calculate_cohens_kappa(human_ratings, llm_ratings)

        kappa_scores.append({"participant": participant, "kappa": float(kappa) if not np.isnan(kappa) else None})

    # Calculate Fleiss' Kappa (inter-rater reliability among HUMAN raters only)
    # Build ratings matrix: servers x participants (excluding LLM)
    # Note: Different servers may have been rated by different numbers of participants
    servers = merged["servername"].unique()

    # Get all participants who rated any server
    all_participants = participants.tolist()

    # FILTER: Only include participants with at least 50 ratings for Fleiss Kappa
    # This ensures meaningful inter-rater reliability calculation
    participant_counts = merged.groupby("participant_id").size()
    substantial_participants = participant_counts[participant_counts >= MIN_RATINGS_FOR_FLEISS].index.tolist()

    # Use filtered participants for Fleiss Kappa calculation
    fleiss_participants = [p for p in all_participants if p in substantial_participants]

    # Update n_participants to reflect only substantial participants (used in Fleiss Kappa)
    n_participants = len(fleiss_participants)

    logger.info(f"Fleiss Kappa: Including {n_participants} participants with ≥{MIN_RATINGS_FOR_FLEISS} ratings (excluded {len(all_participants) - n_participants} with <{MIN_RATINGS_FOR_FLEISS})")

    ratings_matrix = []

    for server in servers:
        server_data = merged[merged["servername"] == server]

        # Build rating row: one rating per participant (or NaN if they didn't rate this server)
        # Use only substantial participants for Fleiss Kappa
        ratings = []
        for participant in fleiss_participants:
            participant_rating = server_data[server_data["participant_id"] == participant][human_col]
            if len(participant_rating) > 0:
                ratings.append(participant_rating.iloc[0])
            else:
                ratings.append(np.nan)

        # Do NOT add LLM rating - Fleiss' Kappa measures inter-rater reliability among humans only

        ratings_matrix.append(ratings)

    # Convert to numpy array - use object dtype for categorical data, float for numeric
    if question_info["type"] in ["binary", "ordinal"] or question_info.get("mapping"):
        ratings_matrix = np.array(ratings_matrix, dtype=float)
    else:
        ratings_matrix = np.array(ratings_matrix, dtype=object)

    fleiss_kappa = calculate_fleiss_kappa(ratings_matrix)

    # Calculate overall agreement percentage
    all_human = merged[human_col].values
    all_llm = merged[llm_field_to_use].values
    mask = ~(pd.isna(all_human) | pd.isna(all_llm))
    agreement_pct = np.mean(all_human[mask] == all_llm[mask])

    # Calculate confusion matrix (averaged across participants)
    if question_info.get("mapping"):
        labels = list(question_info["mapping"].values())
    else:
        # For categorical without mapping, get unique labels from data
        labels = sorted(set(all_human[mask].tolist() + all_llm[mask].tolist()))

    conf_matrix = calculate_confusion_matrix(all_human, all_llm, labels)

    return {
        "question_id": question_id,
        "description": question_info["description"],
        "type": question_info["type"],
        "n_participants": n_participants,
        "n_servers": len(servers),
        "n_responses": int(np.sum(mask)),
        "kappa_vs_llm": kappa_scores,
        "mean_kappa": float(np.nanmean([k["kappa"] for k in kappa_scores if k["kappa"] is not None])),
        "fleiss_kappa": float(fleiss_kappa) if not np.isnan(fleiss_kappa) else None,
        "agreement_pct": float(agreement_pct),
        "confusion_matrix": conf_matrix.tolist() if conf_matrix.size > 0 else [],
        "confusion_matrix_labels": labels,
    }


def visualize_confusion_matrices(question_stats: dict, output_dir: Path):
    """
    Visualize confusion matrices for each question and save as images.

    Args:
        question_stats: Dictionary with question-level statistics including confusion matrices
        output_dir: Directory to save confusion matrix images
    """
    logger.info("Generating confusion matrix visualizations")

    # Create subdirectory for confusion matrices
    cm_dir = output_dir / "confusion_matrices"
    cm_dir.mkdir(exist_ok=True)

    for question_id, stats in question_stats.items():
        if "error" in stats or not stats.get("confusion_matrix"):
            continue

        conf_matrix = np.array(stats["confusion_matrix"])
        labels = stats.get("confusion_matrix_labels", [])

        if conf_matrix.size == 0 or len(labels) == 0:
            continue

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Format labels for better display
        if isinstance(labels[0], str):
            # For string labels, truncate if too long
            display_labels = [str(label)[:30] + "..." if len(str(label)) > 30 else str(label) for label in labels]
        else:
            display_labels = labels

        # Create heatmap
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=display_labels,
            yticklabels=display_labels,
            ax=ax,
            cbar_kws={"label": "Count"},
        )

        # Set labels and title
        ax.set_xlabel("Human Rating", fontsize=12, fontweight="bold")
        ax.set_ylabel("LLM Rating", fontsize=12, fontweight="bold")

        # Create title with question info
        title = f"Question: {question_id}\n{stats['description']}"
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

        # Add agreement statistics as text
        fleiss = stats.get('fleiss_kappa')
        fleiss_str = f"{fleiss:.3f}" if fleiss is not None else "N/A"
        stats_text = (
            f"Mean Kappa: {stats.get('mean_kappa', 0):.3f}\n"
            f"Fleiss' Kappa: {fleiss_str}\n"
            f"Agreement: {stats.get('agreement_pct', 0):.1%}\n"
            f"N={stats.get('n_responses', 0)}"
        )
        ax.text(
            1.15,
            0.5,
            stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
            fontweight="bold",
        )

        # Rotate labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=0)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Save figure
        output_file = cm_dir / f"servers_{question_id}_confusion_matrix.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved confusion matrix for {question_id} to {output_file}")

    logger.info(f"All confusion matrices saved to {cm_dir}")


def calculate_participant_statistics(human_df: pd.DataFrame, llm_df: pd.DataFrame) -> dict[str, Any]:
    """
    Calculate per-participant agreement statistics.

    Args:
        human_df: Human validation data (wide format)
        llm_df: LLM classifications

    Returns:
        Dictionary with per-participant statistics
    """
    logger.info("Calculating per-participant statistics")

    participants = human_df["participant_id"].unique()
    participant_stats = {}

    for participant in participants:
        participant_data = human_df[human_df["participant_id"] == participant]

        # Calculate agreement for each question
        question_agreement = {}
        kappa_scores = []

        for question_id, question_info in QUESTION_MAPPING.items():
            llm_field = get_llm_field_name(llm_df, question_id)
            if not llm_field:
                continue

            human_col = f"{question_id}_mapped"

            if human_col not in participant_data.columns:
                continue

            # Merge with LLM data - include func_main for conditional filtering
            merge_cols = ["server_name", llm_field]
            if question_id == "func_sub":
                func_main_llm_field = get_llm_field_name(llm_df, "func_main")
                if func_main_llm_field and func_main_llm_field in llm_df.columns:
                    merge_cols.append(func_main_llm_field)

            merged = participant_data.merge(llm_df[merge_cols], left_on="servername", right_on="server_name", how="inner")

            if len(merged) == 0:
                continue

            # For func_sub, filter to only cases where there's agreement on func_main
            if question_id == "func_sub" and "func_main_mapped" in participant_data.columns:
                func_main_llm_field = get_llm_field_name(llm_df, "func_main")
                if func_main_llm_field and func_main_llm_field in merged.columns:
                    # Map LLM func_main to same scale as human
                    func_main_mapping = QUESTION_MAPPING["func_main"].get("mapping")
                    if func_main_mapping:
                        merged[f"{func_main_llm_field}_mapped"] = merged[func_main_llm_field].map(func_main_mapping)
                    else:
                        merged[f"{func_main_llm_field}_mapped"] = merged[func_main_llm_field]

                    # Filter to only rows where func_main agrees
                    merged = merged[merged["func_main_mapped"] == merged[f"{func_main_llm_field}_mapped"]]

                    # Additional filtering: Remove LLM "sensors" predictions for non-perception cases
                    merged = merged[
                        ~(
                            (merged[f"{func_main_llm_field}_mapped"] != 1)  # Not perception (perception=1)
                            & (merged[llm_field] == "sensors")  # LLM predicts sensors
                        )
                    ]

                    if len(merged) == 0:
                        continue

            # Map LLM ratings if needed
            if question_info.get("mapping"):
                merged[f"{llm_field}_mapped"] = merged[llm_field].map(question_info["mapping"])
                llm_field_to_use = f"{llm_field}_mapped"
            else:
                llm_field_to_use = llm_field

            human_ratings = merged[human_col].values
            llm_ratings = merged[llm_field_to_use].values

            if question_info["type"] == "ordinal":
                kappa = calculate_weighted_kappa(human_ratings, llm_ratings)
            else:
                kappa = calculate_cohens_kappa(human_ratings, llm_ratings)

            mask = ~(pd.isna(human_ratings) | pd.isna(llm_ratings))
            agreement_pct = np.mean(human_ratings[mask] == llm_ratings[mask]) if np.sum(mask) > 0 else 0

            question_agreement[question_id] = {
                "kappa": float(kappa) if not np.isnan(kappa) else None,
                "agreement_pct": float(agreement_pct),
                "n_responses": int(np.sum(mask)),
            }

            if not np.isnan(kappa):
                kappa_scores.append(kappa)

        participant_stats[participant] = {
            "n_servers_rated": len(participant_data),
            "overall_kappa": float(np.mean(kappa_scores)) if kappa_scores else None,
            "agreement_by_question": question_agreement,
        }

    return participant_stats


def check_server_name_matching(human_df: pd.DataFrame, llm_df: pd.DataFrame) -> None:
    """
    Check and report server name matching between human validation and LLM data.
    Warns about unmatched servers that might be case mismatches.

    Args:
        human_df: Wide-format human validation data with 'servername' column
        llm_df: LLM classifications with 'server_name' column
    """
    # Get unique server names from both datasets
    human_servers = set(human_df["servername"].dropna().unique())
    llm_servers = set(llm_df["server_name"].unique())

    # Create case-insensitive lookup for LLM servers
    llm_servers_lower = {name.lower(): name for name in llm_servers}

    # Find exact matches and mismatches
    exact_matches = human_servers & llm_servers
    unmatched = human_servers - llm_servers

    # Check for case-insensitive matches among unmatched
    case_mismatches = {}
    true_unmatched = []

    for human_name in unmatched:
        llm_name = llm_servers_lower.get(human_name.lower())
        if llm_name:
            case_mismatches[human_name] = llm_name
        else:
            true_unmatched.append(human_name)

    # Log results
    logger.info(f"Server name matching: {len(human_servers)} human servers, {len(llm_servers)} LLM servers")
    logger.info(f"  ✓ Exact matches: {len(exact_matches)}/{len(human_servers)}")

    if case_mismatches:
        logger.warning(f"  ⚠ Case mismatches found: {len(case_mismatches)}")
        for human_name, llm_name in case_mismatches.items():
            logger.warning(f"    - '{human_name}' (human) != '{llm_name}' (LLM) - case difference")
            logger.warning(f"      These servers will NOT be matched. Consider fixing case in human validation data.")

    if true_unmatched:
        logger.warning(f"  ✗ Unmatched servers: {len(true_unmatched)}")
        for name in true_unmatched:
            logger.warning(f"    - '{name}' not found in LLM classifications")


def main():
    """Main execution function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Human validation scoring analysis")
    parser.add_argument(
        "--llm-file",
        type=str,
        default="clservers_classified.csv",
        help="LLM classification file name (default: clservers_classified.csv)",
    )
    parser.add_argument(
        "--exclude-participants",
        type=str,
        nargs="+",
        default=[],
        help="List of participant IDs to exclude from analysis",
    )
    parser.add_argument(
        "--exclude-random",
        type=lambda x: x.lower() != "false",
        default=True,
        metavar="True/False",
        help="Exclude participants with worse-than-random (negative kappa) on any question (default: True)",
    )
    args = parser.parse_args()

    logger.info(f"Starting human validation scoring analysis with {args.llm_file}")
    if args.exclude_participants:
        logger.info(f"Excluding participants: {', '.join(args.exclude_participants)}")

    # Define paths
    project_root = Path(__file__).parent.parent.parent  # Go up to project root from scripts/human-validation/
    human_data_dir = project_root / "data" / "external-cl-human-valid"
    llm_data_path = project_root / "data" / "final" / args.llm_file

    # Create output directories if they don't exist
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    output_dir = project_root / "output-validation" / "cl-validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load O*NET L1 mapping for onet_l1 question
    onet_mapping = load_onet_l1_mapping(project_root)

    # Load data
    human_df = load_human_validation_data(human_data_dir)
    llm_df = load_llm_classifications(llm_data_path)

    # Filter out excluded participants
    if args.exclude_participants:
        initial_count = len(human_df)
        human_df = human_df[~human_df["Participant Public ID"].isin(args.exclude_participants)]
        filtered_count = len(human_df)
        logger.info(
            f"Filtered {initial_count - filtered_count} responses from {len(args.exclude_participants)} excluded participants"
        )
        logger.info(f"Remaining: {filtered_count} responses from {human_df['Participant Public ID'].nunique()} participants")

    # Transform human data
    human_wide = transform_human_data(human_df)
    human_mapped = map_human_to_llm_scale(human_wide, onet_mapping=onet_mapping)

    # Derive generality classification from q3 (industry) and q4 (environment)
    human_mapped = derive_generality_classification(human_mapped)

    # Check server name matching and report issues
    check_server_name_matching(human_wide, llm_df)

    # Calculate statistics for each question
    question_stats = {}
    for question_id, question_info in QUESTION_MAPPING.items():
        question_stats[question_id] = analyze_question(human_mapped, llm_df, question_id, question_info)

    # Calculate per-participant statistics (initial pass)
    participant_stats = calculate_participant_statistics(human_mapped, llm_df)

    # Identify and exclude participants with negative kappa on any question
    random_excluded = []
    if args.exclude_random:
        for participant_id, p_stats in participant_stats.items():
            for question_id, q_stats in p_stats["agreement_by_question"].items():
                kappa = q_stats["kappa"]
                if kappa is not None and kappa < 0:
                    random_excluded.append(participant_id)
                    logger.warning(
                        f"Participant {participant_id} has negative kappa ({kappa:.3f}) on {question_id} - excluding"
                    )
                    break  # One negative kappa is enough to exclude

        if random_excluded:
            logger.info(f"Excluding {len(random_excluded)} participants with worse-than-random performance")
            logger.info(f"Excluded participants: {', '.join(random_excluded)}")

            # Filter out excluded participants
            initial_count = len(human_df)
            human_df = human_df[~human_df["Participant Public ID"].isin(random_excluded)]
            filtered_count = len(human_df)
            logger.info(f"Filtered {initial_count - filtered_count} responses from excluded participants")
            logger.info(f"Remaining: {filtered_count} responses from {human_df['Participant Public ID'].nunique()} participants")

            # Recalculate everything with filtered data
            human_wide = transform_human_data(human_df)
            human_mapped = map_human_to_llm_scale(human_wide, onet_mapping=onet_mapping)

            # Derive generality classification from q3 (industry) and q4 (environment)
            human_mapped = derive_generality_classification(human_mapped)

            # Recalculate question statistics
            question_stats = {}
            for question_id, question_info in QUESTION_MAPPING.items():
                question_stats[question_id] = analyze_question(human_mapped, llm_df, question_id, question_info)

            # Recalculate participant statistics
            participant_stats = calculate_participant_statistics(human_mapped, llm_df)
        else:
            logger.info("No participants with worse-than-random performance found")
    else:
        logger.info("Skipping automatic exclusion of worse-than-random participants (--exclude-random False)")

    # Visualize confusion matrices for each question
    visualize_confusion_matrices(question_stats, output_dir)

    # Calculate summary statistics
    n_participants = human_wide["participant_id"].nunique()
    n_servers = human_wide["servername"].nunique()
    n_questions = len(QUESTION_MAPPING)

    # Number of actually included IDs in analysis (use minimum from question-level Fleiss Kappa calculations)
    # This represents participants who met the ≥50 rating threshold for at least one question
    question_n_participants = [
        q["n_participants"] for q in question_stats.values() if "n_participants" in q and "error" not in q
    ]
    n_participants_included = min(question_n_participants) if question_n_participants else 0

    # Overall agreement with LLM (mean of all question-level agreements)
    overall_agreement = np.mean([q["agreement_pct"] for q in question_stats.values() if "agreement_pct" in q])

    # Overall inter-rater reliability (mean of Fleiss' Kappa across questions)
    fleiss_kappas = [q["fleiss_kappa"] for q in question_stats.values() if q.get("fleiss_kappa") is not None]
    overall_fleiss_kappa = np.mean(fleiss_kappas) if fleiss_kappas else None

    # Generate output filename based on input file and exclusions
    all_excluded = list(set(args.exclude_participants + random_excluded))
    if all_excluded:
        # Create suffix for excluded participants
        exclude_suffix = "-excluded"
        output_filename = f"human-validation-servers-scores{exclude_suffix}.json"
        if "alternative" in args.llm_file:
            output_filename = f"human-validation-servers-scores-alternative{exclude_suffix}.json"
    elif "alternative" in args.llm_file:
        output_filename = "human-validation-servers-scores-alternative.json"
    else:
        output_filename = "human-validation-servers-scores.json"

    output_path = output_dir / output_filename

    # Build output
    output = {
        "summary": {
            "n_participants": n_participants,
            "n_participants_included": n_participants_included,
            "n_servers": n_servers,
            "n_questions": n_questions,
            "overall_agreement_with_llm": float(overall_agreement),
            "overall_inter_rater_reliability": float(overall_fleiss_kappa) if overall_fleiss_kappa else None,
            "excluded_participants": all_excluded,
            "excluded_random": random_excluded,
            "excluded_manual": args.exclude_participants,
        },
        "by_question": question_stats,
        "by_participant": participant_stats,
    }

    # Save output
    logger.info(f"Saving results to {output_path}")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Analysis complete!")
    logger.info(f"Summary: {n_participants} total participants ({n_participants_included} with ≥{MIN_RATINGS_FOR_FLEISS} ratings) rated {n_servers} servers across {n_questions} questions")
    logger.info(f"Overall agreement with LLM: {overall_agreement:.2%}")
    if overall_fleiss_kappa:
        logger.info(f"Overall inter-rater reliability: {overall_fleiss_kappa:.3f}")


if __name__ == "__main__":
    main()

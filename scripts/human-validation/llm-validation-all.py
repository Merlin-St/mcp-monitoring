#!/usr/bin/env python3
"""
LLM Validation Script

Compares LLM classifications between different model runs to assess inter-model agreement:
1. clservers_classified.csv vs clservers_classified_alternative.csv
2. cltools_classified.csv vs cltools_classified_gpt5.csv

Calculates:
- Cohen's Kappa for pairwise agreement between models
- Confusion matrices showing disagreement patterns
- Percentage agreement statistics

Output:
- output-validation/cl-validation/llm-validation.json - Combined agreement statistics
- output-validation/cl-validation/confusion_matrices_llm/ - Confusion matrix visualizations
"""

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
        logging.FileHandler("logs/llm_validation_all.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Field mappings for CLServers comparison
CLSERVERS_FIELDS = {
    "highest_automation_func": {
        "description": "Main functionality level (perception/reasoning/action)",
        "type": "categorical",
        "mapping": {"perception": 1, "reasoning": 2, "action": 3},
    },
    "main_automation_subfunc": {
        "description": "Sub-category functionality classification",
        "type": "categorical",
        "mapping": None,  # Dynamic mapping
    },
    "main_onet_task_level1": {
        "description": "O*NET Level 1 occupational category",
        "type": "categorical",
        "mapping": None,
    },
    "main_onet_task_level2": {
        "description": "O*NET Level 2 occupational category",
        "type": "categorical",
        "mapping": None,
    },
    "generality_industry": {
        "description": "Industry generality (cross-industry vs industry-specific)",
        "type": "binary",
        "mapping": {0: 0, 1: 1},
    },
    "generality_environment": {
        "description": "Environment generality (open/untrusted vs trusted)",
        "type": "binary",
        "mapping": {0: 0, 1: 1},
    },
    "payments_autonomy": {
        "description": "Payment autonomy level (0=not payment, 1-4=increasing autonomy)",
        "type": "ordinal",
        "mapping": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
    },
    "naics_code": {
        "description": "NAICS industry classification code",
        "type": "categorical",
        "mapping": None,
    },
    "is_finance_llm": {
        "description": "Finance-related binary classification",
        "type": "binary",
        "mapping": {0: 0, 1: 1, "yes": 1, "no": 0, True: 1, False: 0},
    },
}

# Field mappings for CLTools comparison
CLTOOLS_FIELDS = {
    "tool_functionality_main": {
        "description": "Main functionality level (perception/reasoning/action)",
        "type": "categorical",
        "mapping": {"perception": 1, "reasoning": 2, "action": 3},
    },
    "tool_functionality_sub": {
        "description": "Sub-category functionality classification",
        "type": "categorical",
        "mapping": None,
    },
    "level1_name": {
        "description": "O*NET Level 1 occupational category",
        "type": "categorical",
        "mapping": None,
    },
    "level2_name": {
        "description": "O*NET Level 2 occupational category",
        "type": "categorical",
        "mapping": None,
    },
}


def load_classification_file(file_path: Path, dataset_type: str) -> pd.DataFrame:
    """
    Load classification CSV file.

    Args:
        file_path: Path to classification CSV
        dataset_type: 'clservers' or 'cltools'

    Returns:
        DataFrame with classifications
    """
    logger.info(f"Loading {dataset_type} classifications from {file_path.name}")

    df = pd.read_csv(file_path, low_memory=False)

    # Get identifier column
    if dataset_type == "clservers":
        id_col = "server_id"  # Use server_id (unique) not server_name (has duplicates)
        fields = CLSERVERS_FIELDS
    else:
        id_col = "tool_id"
        fields = CLTOOLS_FIELDS

    # Select relevant columns
    relevant_cols = [id_col] + [field for field in fields.keys() if field in df.columns]
    df = df[relevant_cols]

    logger.info(f"Loaded {len(df)} records with {len(relevant_cols)-1} classification fields")

    return df


def calculate_cohens_kappa(model1_ratings: np.ndarray, model2_ratings: np.ndarray, weighted: bool = False) -> float:
    """
    Calculate Cohen's Kappa for pairwise agreement.

    Args:
        model1_ratings: Array of model 1 ratings
        model2_ratings: Array of model 2 ratings
        weighted: Use weighted kappa for ordinal data

    Returns:
        Cohen's Kappa score (-1 to 1, where 1 is perfect agreement)
    """
    # Remove any pairs with missing data
    mask = ~(pd.isna(model1_ratings) | pd.isna(model2_ratings))
    model1_clean = model1_ratings[mask]
    model2_clean = model2_ratings[mask]

    if len(model1_clean) == 0:
        return np.nan

    weights = "linear" if weighted else None
    return cohen_kappa_score(model1_clean, model2_clean, weights=weights)


def calculate_confusion_matrix(model1_ratings: np.ndarray, model2_ratings: np.ndarray, labels: list) -> np.ndarray:
    """
    Calculate confusion matrix showing disagreement patterns.

    Args:
        model1_ratings: Array of model 1 ratings
        model2_ratings: Array of model 2 ratings
        labels: List of possible label values

    Returns:
        Confusion matrix as numpy array
    """
    # Remove any pairs with missing data
    mask = ~(pd.isna(model1_ratings) | pd.isna(model2_ratings))
    model1_clean = model1_ratings[mask]
    model2_clean = model2_ratings[mask]

    if len(model1_clean) == 0:
        return np.array([])

    return confusion_matrix(model1_clean, model2_clean, labels=labels)


def analyze_field_agreement(
    df1: pd.DataFrame, df2: pd.DataFrame, field_name: str, field_info: dict, id_col: str, model1_name: str, model2_name: str
) -> dict[str, Any]:
    """
    Analyze agreement for a specific field between two model runs.

    Args:
        df1: Model 1 classifications
        df2: Model 2 classifications
        field_name: Field to compare
        field_info: Field metadata
        id_col: Identifier column name
        model1_name: Name of model 1
        model2_name: Name of model 2

    Returns:
        Dictionary with agreement statistics
    """
    logger.info(f"Analyzing field: {field_name} - {field_info['description']}")

    # Check if field exists in both dataframes
    if field_name not in df1.columns or field_name not in df2.columns:
        logger.warning(f"Field {field_name} not found in both datasets")
        return {"error": "Field not found in both datasets"}

    # Merge on identifier
    merged = df1[[id_col, field_name]].merge(
        df2[[id_col, field_name]], on=id_col, how="inner", suffixes=("_model1", "_model2")
    )

    if len(merged) == 0:
        logger.warning(f"No matching records found for field {field_name}")
        return {"error": "No matching records"}

    model1_col = f"{field_name}_model1"
    model2_col = f"{field_name}_model2"

    # Apply mapping if needed
    if field_info.get("mapping"):
        merged[f"{model1_col}_mapped"] = merged[model1_col].map(field_info["mapping"])
        merged[f"{model2_col}_mapped"] = merged[model2_col].map(field_info["mapping"])
        model1_ratings = merged[f"{model1_col}_mapped"].values
        model2_ratings = merged[f"{model2_col}_mapped"].values
    elif field_info["type"] in ["binary", "ordinal"]:
        # Convert to numeric
        model1_ratings = pd.to_numeric(merged[model1_col], errors="coerce").values
        model2_ratings = pd.to_numeric(merged[model2_col], errors="coerce").values
    else:
        # Keep as is for categorical
        model1_ratings = merged[model1_col].values
        model2_ratings = merged[model2_col].values

    # Calculate Cohen's Kappa
    weighted = field_info["type"] == "ordinal"
    kappa = calculate_cohens_kappa(model1_ratings, model2_ratings, weighted=weighted)

    # Calculate agreement percentage
    mask = ~(pd.isna(model1_ratings) | pd.isna(model2_ratings))
    agreement_pct = np.mean(model1_ratings[mask] == model2_ratings[mask]) if np.sum(mask) > 0 else 0

    # Calculate confusion matrix
    if field_info.get("mapping"):
        labels = sorted(list(field_info["mapping"].values()))
    else:
        labels = sorted(set(model1_ratings[mask].tolist() + model2_ratings[mask].tolist()))

    conf_matrix = calculate_confusion_matrix(model1_ratings, model2_ratings, labels)

    # Flag suspicious perfect agreement
    is_identical = bool(agreement_pct >= 0.999)  # 99.9% threshold, convert to Python bool

    return {
        "field_name": field_name,
        "description": field_info["description"],
        "type": field_info["type"],
        "n_records": int(np.sum(mask)),
        "cohens_kappa": float(kappa) if not np.isnan(kappa) else None,
        "agreement_pct": float(agreement_pct),
        "confusion_matrix": conf_matrix.tolist() if conf_matrix.size > 0 else [],
        "confusion_matrix_labels": labels,
        "model1_name": model1_name,
        "model2_name": model2_name,
        "is_identical": is_identical,
        "note": "Field appears identical between files (likely copied, not independently classified)" if is_identical else None,
    }


def visualize_confusion_matrices(field_stats: dict, output_dir: Path, dataset_name: str):
    """
    Visualize confusion matrices for each field and save as images.

    Args:
        field_stats: Dictionary with field-level statistics
        output_dir: Directory to save confusion matrix images
        dataset_name: Name of dataset (for file naming)
    """
    logger.info(f"Generating confusion matrix visualizations for {dataset_name}")

    # Create subdirectory for confusion matrices
    cm_dir = output_dir / "confusion_matrices_llm"
    cm_dir.mkdir(exist_ok=True)

    for field_name, stats in field_stats.items():
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
            display_labels = [str(label)[:30] + "..." if len(str(label)) > 30 else str(label) for label in labels]
        else:
            display_labels = labels

        # Create heatmap
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Greens",
            xticklabels=display_labels,
            yticklabels=display_labels,
            ax=ax,
            cbar_kws={"label": "Count"},
        )

        # Set labels and title
        ax.set_xlabel(f"{stats['model2_name']}", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"{stats['model1_name']}", fontsize=12, fontweight="bold")

        # Create title with field info
        title = f"Field: {field_name}\n{stats['description']}"
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

        # Add agreement statistics as text
        stats_text = (
            f"Cohen's Kappa: {stats.get('cohens_kappa', 0):.3f}\n"
            f"Agreement: {stats.get('agreement_pct', 0):.1%}\n"
            f"N={stats.get('n_records', 0)}"
        )
        ax.text(
            1.15,
            0.5,
            stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
            fontweight="bold",
        )

        # Rotate labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=0)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Save figure
        output_file = cm_dir / f"{dataset_name}_{field_name}_confusion_matrix.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved confusion matrix for {field_name} to {output_file}")

    logger.info(f"All confusion matrices saved to {cm_dir}")


def analyze_dataset_comparison(
    file1_path: Path, file2_path: Path, dataset_type: str, model1_name: str, model2_name: str
) -> dict[str, Any]:
    """
    Analyze agreement between two classification files for a dataset.

    Args:
        file1_path: Path to model 1 classification file
        file2_path: Path to model 2 classification file
        dataset_type: 'clservers' or 'cltools'
        model1_name: Name of model 1
        model2_name: Name of model 2

    Returns:
        Dictionary with comparison statistics
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Analyzing {dataset_type.upper()} comparison: {model1_name} vs {model2_name}")
    logger.info(f"{'='*80}\n")

    # Load data
    df1 = load_classification_file(file1_path, dataset_type)
    df2 = load_classification_file(file2_path, dataset_type)

    # Get field mappings
    fields = CLSERVERS_FIELDS if dataset_type == "clservers" else CLTOOLS_FIELDS
    id_col = "server_id" if dataset_type == "clservers" else "tool_id"

    # Analyze each field
    field_stats = {}
    for field_name, field_info in fields.items():
        field_stats[field_name] = analyze_field_agreement(df1, df2, field_name, field_info, id_col, model1_name, model2_name)

    # Identify suspicious fields with perfect or near-perfect agreement
    # These may indicate fields that were copied rather than independently classified
    SUSPICIOUS_THRESHOLD = 0.999  # 99.9% agreement is suspicious
    identical_fields = []
    variable_fields = []

    for field_name, stats in field_stats.items():
        if "error" not in stats and "agreement_pct" in stats:
            if stats["agreement_pct"] >= SUSPICIOUS_THRESHOLD:
                identical_fields.append(field_name)
                logger.warning(
                    f"⚠️  Field '{field_name}' shows {stats['agreement_pct']:.1%} agreement "
                    f"(kappa={stats['cohens_kappa']:.3f}) - likely identical/copied between files"
                )
            else:
                variable_fields.append(field_name)

    # Calculate summary statistics
    kappa_scores = [stats["cohens_kappa"] for stats in field_stats.values() if stats.get("cohens_kappa") is not None]
    agreement_pcts = [stats["agreement_pct"] for stats in field_stats.values() if "agreement_pct" in stats]

    # Calculate statistics excluding identical fields
    kappa_scores_variable = [
        field_stats[field]["cohens_kappa"]
        for field in variable_fields
        if field_stats[field].get("cohens_kappa") is not None
    ]
    agreement_pcts_variable = [
        field_stats[field]["agreement_pct"]
        for field in variable_fields
        if "agreement_pct" in field_stats[field]
    ]

    summary = {
        "model1_name": model1_name,
        "model2_name": model2_name,
        "n_fields": len(fields),
        "n_records": len(df1),
        "mean_cohens_kappa": float(np.mean(kappa_scores)) if kappa_scores else None,
        "mean_agreement_pct": float(np.mean(agreement_pcts)) if agreement_pcts else None,
        "identical_fields": identical_fields,
        "variable_fields": variable_fields,
        "mean_cohens_kappa_variable_only": float(np.mean(kappa_scores_variable)) if kappa_scores_variable else None,
        "mean_agreement_pct_variable_only": float(np.mean(agreement_pcts_variable)) if agreement_pcts_variable else None,
        "note": "Some fields show 100% agreement, suggesting they may be identical/copied between files rather than independently classified.",
    }

    # Log summary
    if identical_fields:
        logger.warning(
            f"\n⚠️  IMPORTANT: {len(identical_fields)} field(s) appear to be identical between files "
            f"(>{SUSPICIOUS_THRESHOLD:.1%} agreement):"
        )
        for field in identical_fields:
            logger.warning(f"    - {field}")
        logger.warning(
            f"\nStatistics excluding identical fields:"
        )
        logger.warning(f"    Mean Cohen's Kappa: {summary['mean_cohens_kappa_variable_only']:.3f} (was {summary['mean_cohens_kappa']:.3f} including identical)")
        logger.warning(f"    Mean Agreement: {summary['mean_agreement_pct_variable_only']:.1%} (was {summary['mean_agreement_pct']:.1%} including identical)")

    return {
        "summary": summary,
        "by_field": field_stats,
    }


def main():
    """Main execution function."""
    logger.info("Starting LLM validation analysis")

    # Define paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "final"
    output_dir = project_root / "output-validation" / "cl-validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create logs directory
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Define file pairs to compare
    comparisons = [
        {
            "dataset_type": "clservers",
            "file1": data_dir / "clservers_classified.csv.gz",
            "file2": data_dir / "clservers_classified_alternative.csv.gz",
            "model1_name": "Claude Sonnet 4.5",
            "model2_name": "GPT-5",
            "output_key": "clservers",
        },
        {
            "dataset_type": "cltools",
            "file1": data_dir / "cltools_classified.csv.gz",
            "file2": data_dir / "cltools_classified_gpt5.csv.gz",
            "model1_name": "Claude Sonnet 4.5",
            "model2_name": "GPT-5",
            "output_key": "cltools",
        },
    ]

    # Run comparisons
    results = {}
    for comparison in comparisons:
        dataset_name = comparison["output_key"]

        # Check if files exist
        if not comparison["file1"].exists():
            logger.error(f"File not found: {comparison['file1']}")
            continue
        if not comparison["file2"].exists():
            logger.error(f"File not found: {comparison['file2']}")
            continue

        # Analyze comparison
        comparison_results = analyze_dataset_comparison(
            comparison["file1"],
            comparison["file2"],
            comparison["dataset_type"],
            comparison["model1_name"],
            comparison["model2_name"],
        )

        results[dataset_name] = comparison_results

        # Visualize confusion matrices
        visualize_confusion_matrices(comparison_results["by_field"], output_dir, dataset_name)

    # Save combined output
    output_path = output_dir / "llm-validation.json"
    logger.info(f"\nSaving combined results to {output_path}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Log summary
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE - SUMMARY")
    logger.info("=" * 80)

    for dataset_name, result in results.items():
        summary = result["summary"]
        logger.info(f"\n{dataset_name.upper()}:")
        logger.info(f"  Models: {summary['model1_name']} vs {summary['model2_name']}")
        logger.info(f"  Records: {summary['n_records']}")
        logger.info(f"  Fields analyzed: {summary['n_fields']}")
        if summary["mean_cohens_kappa"] is not None:
            logger.info(f"  Mean Cohen's Kappa: {summary['mean_cohens_kappa']:.3f}")
        if summary["mean_agreement_pct"] is not None:
            logger.info(f"  Mean Agreement: {summary['mean_agreement_pct']:.1%}")

    logger.info(f"\nResults saved to: {output_path}")
    logger.info(f"Confusion matrices saved to: {output_dir / 'confusion_matrices_llm'}")


if __name__ == "__main__":
    main()

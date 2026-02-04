#!/usr/bin/env python3
"""
CLTools to CLServers Data Matching Tool

This module enriches CLServers CSV with aggregated tool-level classifications from CLTools.
For each server, it computes:
- highest_automation_func: Highest automation functionality (Actions > Reasoning > Perception)
- main_automation_subfunc: Most common automation sub-functionality
- main_onet_task_level1: Most common Level 1 O*NET task cluster name
- main_onet_task_level2: Most common Level 2 O*NET task cluster name
- main_onet_task_level3: Most common task_id
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/cltools_datamatch_toservers.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Automation functionality hierarchy (highest to lowest)
# Note: Using lowercase to match actual data values
AUTOMATION_HIERARCHY = {
    'action': 3,
    'reasoning': 2,
    'perception': 1
}


def get_most_common(values: pd.Series) -> Optional[str]:
    """
    Get the most common value from a series, excluding NaN.

    Args:
        values: Series of values to find mode of

    Returns:
        Most common value, or None if all values are NaN
    """
    # Remove NaN values
    clean_values = values.dropna()

    if len(clean_values) == 0:
        return None

    # Use Counter for most_common
    counter = Counter(clean_values)
    most_common = counter.most_common(1)

    return most_common[0][0] if most_common else None


def get_highest_automation_func(values: pd.Series) -> Optional[str]:
    """
    Get the highest automation functionality from a series.
    Hierarchy: Actions > Reasoning > Perception

    Args:
        values: Series of automation functionality values

    Returns:
        Highest automation functionality, or None if all values are NaN
    """
    # Remove NaN values
    clean_values = values.dropna()

    if len(clean_values) == 0:
        return None

    # Find the highest ranked value
    highest_rank = 0
    highest_func = None

    for value in clean_values.unique():
        rank = AUTOMATION_HIERARCHY.get(value, 0)
        if rank > highest_rank:
            highest_rank = rank
            highest_func = value

    return highest_func


def aggregate_tool_classifications(cltools_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate tool-level classifications by server_id.

    Args:
        cltools_df: CLTools DataFrame with tool-level classifications

    Returns:
        DataFrame with server_id and aggregated classifications
    """
    logger.info("Aggregating tool classifications by server_id...")

    # Group by server_id and aggregate
    aggregated = cltools_df.groupby('server_id').agg({
        'tool_functionality_main': lambda x: get_highest_automation_func(x),
        'tool_functionality_sub': lambda x: get_most_common(x),
        'level1_name': lambda x: get_most_common(x),
        'level2_name': lambda x: get_most_common(x),
    }).reset_index()

    # Check if task_id column exists for level3 aggregation
    if 'task_id' in cltools_df.columns:
        task_level3 = cltools_df.groupby('server_id')['task_id'].apply(
            lambda x: get_most_common(x)
        ).reset_index()
        aggregated = aggregated.merge(task_level3, on='server_id', how='left')
        aggregated = aggregated.rename(columns={'task_id': 'main_onet_task_level3'})
    else:
        logger.warning("Column 'task_id' not found in CLTools data - skipping level3 aggregation")
        aggregated['main_onet_task_level3'] = None

    # Rename columns to match desired output
    aggregated = aggregated.rename(columns={
        'tool_functionality_main': 'highest_automation_func',
        'tool_functionality_sub': 'main_automation_subfunc',
        'level1_name': 'main_onet_task_level1',
        'level2_name': 'main_onet_task_level2'
    })

    logger.info(f"Aggregated classifications for {len(aggregated)} servers")

    # Log distribution statistics
    logger.info("Distribution of highest_automation_func:")
    func_dist = aggregated['highest_automation_func'].value_counts()
    for func, count in func_dist.items():
        logger.info(f"  {func}: {count}")

    logger.info("Distribution of main_onet_task_level1:")
    level1_dist = aggregated['main_onet_task_level1'].value_counts()
    for level, count in level1_dist.head(5).items():
        logger.info(f"  {level}: {count}")
    logger.info(f"  ... ({len(level1_dist)} unique level1 values total)")

    return aggregated


def enrich_servers_with_tools(
    cltools_path: str = "data/final/cltools_classified.csv.gz",
    clservers_path: str = "data/final/clservers_classified.csv.gz",
    output_path: Optional[str] = None
) -> str:
    """
    Enrich CLServers CSV with aggregated tool classifications from CLTools.

    Args:
        cltools_path: Path to CLTools CSV file
        clservers_path: Path to CLServers CSV file to enrich
        output_path: Path for output file (default: adds '_enriched' suffix)

    Returns:
        str: Path to the enriched output file

    Raises:
        FileNotFoundError: If input files don't exist
        ValueError: If required columns are missing
    """
    logger.info("Starting server enrichment process")
    logger.info(f"CLTools file: {cltools_path}")
    logger.info(f"CLServers file: {clservers_path}")

    # Validate input files exist
    if not Path(cltools_path).exists():
        raise FileNotFoundError(f"CLTools file not found: {cltools_path}")
    if not Path(clservers_path).exists():
        raise FileNotFoundError(f"CLServers file not found: {clservers_path}")

    # Set default output path
    if output_path is None:
        clservers_stem = Path(clservers_path).stem
        clservers_suffix = Path(clservers_path).suffix
        output_path = f"{Path(clservers_path).parent}/{clservers_stem}_enriched{clservers_suffix}"

    logger.info(f"Output file: {output_path}")

    # Read CLTools data
    logger.info("Loading CLTools data...")
    try:
        cltools_df = pd.read_csv(cltools_path)
        logger.info(f"Loaded {len(cltools_df)} tools from CLTools file")
    except Exception as e:
        logger.error(f"Error reading CLTools file: {e}")
        raise

    # Validate CLTools has required columns
    required_cltools_cols = ['server_id', 'tool_functionality_main', 'tool_functionality_sub',
                              'level1_name', 'level2_name']
    missing_cols = [col for col in required_cltools_cols if col not in cltools_df.columns]
    if missing_cols:
        raise ValueError(f"CLTools file missing required columns: {missing_cols}")

    # Read CLServers data
    logger.info("Loading CLServers data...")
    try:
        clservers_df = pd.read_csv(clservers_path)
        logger.info(f"Loaded {len(clservers_df)} servers from CLServers file")
    except Exception as e:
        logger.error(f"Error reading CLServers file: {e}")
        raise

    # Validate CLServers has server_id
    if 'server_id' not in clservers_df.columns:
        raise ValueError("CLServers file missing required 'server_id' column")

    # Aggregate tool classifications
    aggregated_tools = aggregate_tool_classifications(cltools_df)

    # Drop existing O*NET columns if they exist (to allow overwriting)
    onet_columns = ['highest_automation_func', 'main_automation_subfunc',
                    'main_onet_task_level1', 'main_onet_task_level2', 'main_onet_task_level3']
    existing_onet_cols = [col for col in onet_columns if col in clservers_df.columns]
    if existing_onet_cols:
        logger.info(f"Dropping existing O*NET columns to overwrite: {existing_onet_cols}")
        clservers_df = clservers_df.drop(columns=existing_onet_cols)

    # Merge with CLServers data
    logger.info("Merging aggregated tool classifications with CLServers data...")
    enriched_df = clservers_df.merge(
        aggregated_tools,
        on='server_id',
        how='left'
    )

    # Log merge statistics
    total_servers = len(enriched_df)
    matched_servers = enriched_df['highest_automation_func'].notna().sum()
    unmatched_servers = total_servers - matched_servers

    logger.info("Merge complete:")
    logger.info(f"  Total servers: {total_servers}")
    logger.info(f"  Servers with tool classifications: {matched_servers}")
    logger.info(f"  Servers without tool classifications: {unmatched_servers}")

    if unmatched_servers > 0:
        logger.warning(
            f"{unmatched_servers} servers could not be matched with tool classifications "
            "(likely servers with no tools in CLTools dataset)"
        )
        # Log some examples
        unmatched_ids = enriched_df[enriched_df['highest_automation_func'].isna()]['server_id'].unique()[:5]
        logger.warning(f"Examples of unmatched server_ids: {list(unmatched_ids)}")

    # Save enriched data
    logger.info(f"Saving enriched data to {output_path}...")
    try:
        enriched_df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved {len(enriched_df)} rows to {output_path}")
    except Exception as e:
        logger.error(f"Error saving enriched file: {e}")
        raise

    # Log summary of new columns
    logger.info("Summary of new columns:")
    new_cols = ['highest_automation_func', 'main_automation_subfunc',
                'main_onet_task_level1', 'main_onet_task_level2', 'main_onet_task_level3']
    for col in new_cols:
        non_null = enriched_df[col].notna().sum()
        logger.info(f"  {col}: {non_null} non-null values ({non_null/total_servers*100:.1f}%)")

    return output_path


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enrich CLServers CSV with aggregated tool classifications from CLTools"
    )
    parser.add_argument(
        '--cltools',
        default="data/final/cltools_classified.csv.gz",
        help="Path to CLTools CSV file"
    )
    parser.add_argument(
        '--clservers',
        default="data/final/clservers_classified.csv.gz",
        help="Path to CLServers CSV file to enrich"
    )
    parser.add_argument(
        '--output',
        help="Output path (default: adds '_enriched' suffix to clservers path)"
    )

    args = parser.parse_args()

    try:
        output_file = enrich_servers_with_tools(
            cltools_path=args.cltools,
            clservers_path=args.clservers,
            output_path=args.output
        )
        logger.info("Enrichment completed successfully!")
        logger.info(f"Enriched file available at: {output_file}")

    except Exception as e:
        logger.error(f"Enrichment failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

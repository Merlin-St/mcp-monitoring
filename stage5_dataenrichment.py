#!/usr/bin/env python3
"""
Stage 4 Task Data Enrichment Tool

This module provides functionality to enrich the stage4 task output CSV with 
creation_date information from the stage2 CSV file and usage data from 
data_usage.json file.
"""

import pandas as pd
import logging
import json
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stage5_dataenrichment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def enrich_with_metadata(
    stage4_path: str = "stage5_task_output.csv",
    stage2_path: str = "server_classified.csv",
    usage_data_path: str = "data_usage.json",
    output_path: Optional[str] = None
) -> str:
    """
    Enrich task output CSV with creation_date and use_count from stage2 CSV,
    plus usage data from data_usage.json.
    
    Args:
        stage4_path: Path to the task output CSV file
        stage2_path: Path to the stage2 CSV file containing creation dates and use counts
        usage_data_path: Path to the data_usage.json file with detailed usage data
        output_path: Path for output file (default: adds '_enriched' suffix)
    
    Returns:
        str: Path to the enriched output file
        
    Raises:
        FileNotFoundError: If input files don't exist
        ValueError: If required columns are missing
    """
    logger.info("Starting enrichment process")
    logger.info(f"Task output file: {stage4_path}")
    logger.info(f"Stage2 file: {stage2_path}")
    logger.info(f"Usage data file: {usage_data_path}")
    
    # Validate input files exist
    if not Path(stage4_path).exists():
        raise FileNotFoundError(f"Task output file not found: {stage4_path}")
    if not Path(stage2_path).exists():
        raise FileNotFoundError(f"Stage2 file not found: {stage2_path}")
    if not Path(usage_data_path).exists():
        raise FileNotFoundError(f"Usage data file not found: {usage_data_path}")
    
    # Set default output path
    if output_path is None:
        stage4_stem = Path(stage4_path).stem
        stage4_suffix = Path(stage4_path).suffix
        output_path = f"{stage4_stem}_enriched{stage4_suffix}"
    
    logger.info(f"Output file: {output_path}")
    
    # Read task output data
    logger.info("Loading task output data...")
    try:
        stage4_df = pd.read_csv(stage4_path)
        logger.info(f"Loaded {len(stage4_df)} rows from task output file")
    except Exception as e:
        logger.error(f"Error reading task output file: {e}")
        raise
    
    # Validate task output has required columns
    if 'server_id' not in stage4_df.columns:
        raise ValueError("Stage4 file missing required 'server_id' column")
    
    # Read stage2 data (only needed columns for memory efficiency)
    logger.info("Loading stage2 metadata (creation date and use count)...")
    try:
        stage2_df = pd.read_csv(stage2_path, usecols=['server_id', 'created_at', 'use_count'])
        logger.info(f"Loaded {len(stage2_df)} rows from stage2 file")
    except Exception as e:
        logger.error(f"Error reading stage2 file: {e}")
        raise
    
    # Validate stage2 has required columns
    if 'created_at' not in stage2_df.columns:
        raise ValueError("Stage2 file missing required 'created_at' column")
    if 'use_count' not in stage2_df.columns:
        raise ValueError("Stage2 file missing required 'use_count' column")
    
    # Read and process usage data
    logger.info("Loading usage data from JSON...")
    try:
        with open(usage_data_path, 'r', encoding='utf-8') as f:
            usage_data = json.load(f)
        logger.info(f"Loaded {len(usage_data)} entries from usage data file")
        
        # Convert to DataFrame with relevant columns
        usage_df = pd.DataFrame(usage_data)
        # Rename 'id' to 'server_id' to match other datasets
        if 'id' in usage_df.columns:
            usage_df = usage_df.rename(columns={'id': 'server_id'})
        
        # Select relevant usage columns for enrichment
        usage_columns = ['server_id']
        
        # Priority usage fields (PyPI/npm download statistics)
        priority_usage_fields = ['usage_pypi_downloads', 'usage_npm_downloads', 'usage_total_downloads', 
                                'usage_monthly_breakdown', 'usage_matched_packages', 'usage_last_updated']
        
        # Additional useful columns
        additional_columns = ['stargazers_count', 'forks_count', 'language', 'owner_login', 
                           'is_finance_related', 'updated_at']
        
        # Add priority usage fields first
        for col in priority_usage_fields:
            if col in usage_df.columns:
                usage_columns.append(col)
                
        # Then add additional columns
        for col in additional_columns:
            if col in usage_df.columns:
                usage_columns.append(col)
        
        usage_df = usage_df[usage_columns]
        logger.info(f"Selected {len(usage_columns)-1} usage columns for enrichment: {usage_columns[1:]}")
        
    except Exception as e:
        logger.error(f"Error reading usage data file: {e}")
        raise
    
    # Perform the merges
    logger.info("Merging with stage2 data on server_id...")
    enriched_df = stage4_df.merge(
        stage2_df[['server_id', 'created_at', 'use_count']], 
        on='server_id', 
        how='left'
    )
    
    logger.info("Merging with usage data on server_id...")
    enriched_df = enriched_df.merge(
        usage_df,
        on='server_id',
        how='left'
    )
    
    # Add creation_date column (rename created_at for clarity)
    enriched_df = enriched_df.rename(columns={'created_at': 'creation_date'})
    
    # Log merge statistics
    total_rows = len(enriched_df)
    stage2_matched = enriched_df['creation_date'].notna().sum()
    stage2_unmatched = total_rows - stage2_matched
    
    # Check usage data matches (using priority usage field)
    usage_matched = 0
    usage_unmatched = total_rows
    if 'usage_last_updated' in enriched_df.columns:
        usage_matched = enriched_df['usage_last_updated'].notna().sum()
        usage_unmatched = total_rows - usage_matched
    elif 'stargazers_count' in enriched_df.columns:
        usage_matched = enriched_df['stargazers_count'].notna().sum()
        usage_unmatched = total_rows - usage_matched
    
    logger.info("Merge complete:")
    logger.info(f"  Total rows: {total_rows}")
    logger.info(f"  Stage2 matches (creation_date + use_count): {stage2_matched}")
    logger.info(f"  Stage2 unmatched: {stage2_unmatched}")
    logger.info(f"  Usage data matches: {usage_matched}")
    logger.info(f"  Usage data unmatched: {usage_unmatched}")
    
    if stage2_unmatched > 0:
        logger.warning(f"{stage2_unmatched} rows could not be matched with stage2 metadata")
        # Log some examples of unmatched server_ids for debugging
        unmatched_servers = enriched_df[enriched_df['creation_date'].isna()]['server_id'].unique()[:5]
        logger.warning(f"Examples of stage2 unmatched server_ids: {list(unmatched_servers)}")
    
    if usage_unmatched > 0:
        logger.warning(f"{usage_unmatched} rows could not be matched with usage data")
        if 'usage_last_updated' in enriched_df.columns:
            usage_unmatched_servers = enriched_df[enriched_df['usage_last_updated'].isna()]['server_id'].unique()[:5]
            logger.warning(f"Examples of usage data unmatched server_ids: {list(usage_unmatched_servers)}")
        elif 'stargazers_count' in enriched_df.columns:
            usage_unmatched_servers = enriched_df[enriched_df['stargazers_count'].isna()]['server_id'].unique()[:5]
            logger.warning(f"Examples of usage data unmatched server_ids: {list(usage_unmatched_servers)}")
    
    # Save enriched data
    logger.info(f"Saving enriched data to {output_path}...")
    try:
        enriched_df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved {len(enriched_df)} rows to {output_path}")
    except Exception as e:
        logger.error(f"Error saving enriched file: {e}")
        raise
    
    return output_path


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enrich task output with creation dates, use counts from stage2, and usage data"
    )
    parser.add_argument(
        '--stage4', 
        default="stage5_task_output.csv",
        help="Path to task output CSV file"
    )
    parser.add_argument(
        '--stage2', 
        default="server_classified.csv",
        help="Path to stage2 CSV file with creation dates and use counts"
    )
    parser.add_argument(
        '--usage-data', 
        default="data_usage.json",
        help="Path to data_usage.json file with detailed usage data"
    )
    parser.add_argument(
        '--output', 
        help="Output path (default: adds '_enriched' suffix)"
    )
    
    args = parser.parse_args()
    
    try:
        output_file = enrich_with_metadata(
            stage4_path=args.stage4,
            stage2_path=args.stage2,
            usage_data_path=getattr(args, 'usage_data'),
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
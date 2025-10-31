#!/usr/bin/env python3
"""
CLTools Data Matching Tool

This module provides functionality to enrich the CLTools task output CSV with 
creation_date information from the CLServers CSV file and usage data from 
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
        logging.FileHandler('logs/cltools_datamatch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def enrich_with_metadata(
    cltools_path: str = "data/internal-cl/cltools_3_results.csv",
    clservers_path: str = "data/final/clservers_classified.csv",
    usage_data_path: str = "data/initial/data_unified_filtered.json",
    output_path: Optional[str] = None
) -> str:
    """
    Enrich task output CSV with creation_date and use_count from CLServers CSV,
    plus usage data from data_unified_filtered.json.
    
    Args:
        cltools_path: Path to the task output CSV file
        clservers_path: Path to the CLServers CSV file containing creation dates and use counts
        usage_data_path: Path to the data_unified_filtered.json file with detailed usage data
        output_path: Path for output file (default: adds '_enriched' suffix)
    
    Returns:
        str: Path to the enriched output file
        
    Raises:
        FileNotFoundError: If input files don't exist
        ValueError: If required columns are missing
    """
    logger.info("Starting enrichment process")
    logger.info(f"Task output file: {cltools_path}")
    logger.info(f"CLServers file: {clservers_path}")
    logger.info(f"Usage data file: {usage_data_path}")
    
    # Validate input files exist
    if not Path(cltools_path).exists():
        raise FileNotFoundError(f"Task output file not found: {cltools_path}")
    if not Path(clservers_path).exists():
        raise FileNotFoundError(f"CLServers file not found: {clservers_path}")
    if not Path(usage_data_path).exists():
        raise FileNotFoundError(f"Usage data file not found: {usage_data_path}")
    
    # Set default output path
    if output_path is None:
        output_path = "data/final/cltools_classified.csv"
    
    logger.info(f"Output file: {output_path}")
    
    # Read task output data
    logger.info("Loading task output data...")
    try:
        cltools_df = pd.read_csv(cltools_path)
        logger.info(f"Loaded {len(cltools_df)} rows from task output file")
    except Exception as e:
        logger.error(f"Error reading task output file: {e}")
        raise
    
    # Validate task output has required columns
    if 'server_id' not in cltools_df.columns:
        raise ValueError("Stage4 file missing required 'server_id' column")
    
    # Read CLServers data (only needed columns for memory efficiency)
    logger.info("Loading CLServers metadata (creation date, use count, canonical_official, name, owner, repository_url)...")
    try:
        clservers_df = pd.read_csv(clservers_path, usecols=['server_id', 'created_at', 'use_count', 'canonical_official', 'name', 'owner', 'repository_url'])
        logger.info(f"Loaded {len(clservers_df)} rows from CLServers file")
    except Exception as e:
        logger.error(f"Error reading CLServers file: {e}")
        raise

    # Validate CLServers has required columns
    if 'created_at' not in clservers_df.columns:
        raise ValueError("Stage2 file missing required 'created_at' column")
    if 'use_count' not in clservers_df.columns:
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
    logger.info("Merging with CLServers data on server_id...")
    enriched_df = cltools_df.merge(
        clservers_df[['server_id', 'created_at', 'use_count', 'canonical_official', 'name', 'owner', 'repository_url']],
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

    # Filter out tools from non-MCP servers (those without CLServers metadata)
    tools_before_filter = len(enriched_df)
    enriched_df = enriched_df[enriched_df['creation_date'].notna()].copy()
    tools_after_filter = len(enriched_df)
    filtered_count = tools_before_filter - tools_after_filter

    if filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} tools from non-MCP servers (missing CLServers metadata)")
        logger.info(f"Remaining tools: {tools_after_filter}")
    
    # Log merge statistics
    total_rows = len(enriched_df)
    clservers_matched = enriched_df['creation_date'].notna().sum()
    clservers_unmatched = total_rows - clservers_matched
    
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
    logger.info(f"  CLServers matches (creation_date + use_count): {clservers_matched}")
    logger.info(f"  CLServers unmatched: {clservers_unmatched}")
    logger.info(f"  Usage data matches: {usage_matched}")
    logger.info(f"  Usage data unmatched: {usage_unmatched}")
    
    if clservers_unmatched > 0:
        logger.warning(f"{clservers_unmatched} rows could not be matched with CLServers metadata")
        # Log some examples of unmatched server_ids for debugging
        unmatched_servers = enriched_df[enriched_df['creation_date'].isna()]['server_id'].unique()[:5]
        logger.warning(f"Examples of CLServers unmatched server_ids: {list(unmatched_servers)}")
    
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
        description="Enrich task output with creation dates, use counts from CLServers, and usage data"
    )
    parser.add_argument(
        '--cltools', 
        default="data/internal-cl/cltools_3_results.csv",
        help="Path to task output CSV file"
    )
    parser.add_argument(
        '--clservers', 
        default="data/final/clservers_classified.csv",
        help="Path to CLServers CSV file with creation dates and use counts"
    )
    parser.add_argument(
        '--usage-data', 
        default="data/initial/data_unified_filtered.json",
        help="Path to data/initial/data_unified_filtered.json file with detailed usage data"
    )
    parser.add_argument(
        '--output', 
        help="Output path (default: adds '_enriched' suffix)"
    )
    
    args = parser.parse_args()
    
    try:
        output_file = enrich_with_metadata(
            cltools_path=args.cltools,
            clservers_path=args.clservers,
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
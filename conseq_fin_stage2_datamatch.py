#!/usr/bin/env python3
"""
Financial MCP Server Stage 2 - Data Matching

Matches the Stage 1 evaluation results with the unified dataset to add
creation dates and other metadata fields that weren't included in the
original evaluation.

This should be run after:
    python conseq_fin_stage1_dfprocessing.py

Usage:
    python conseq_fin_stage2_datamatch.py
"""

import json
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conseq_fin_stage2_datamatch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main data matching function"""
    logger.info("Starting Stage 2 Data Matching")
    
    # Load Stage 1 results
    stage1_csv = "conseq_fin_stage1_results.csv"
    if not Path(stage1_csv).exists():
        logger.error(f"Stage 1 results file {stage1_csv} not found. Run conseq_fin_stage1_dfprocessing.py first.")
        return
    
    results_df = pd.read_csv(stage1_csv)
    logger.info(f"Loaded {len(results_df)} Stage 1 results from {stage1_csv}")
    
    # Load the unified dataset for server metadata matching
    unified_file = 'data_unified_filtered.json'
    if not Path(unified_file).exists():
        logger.error(f"Unified dataset {unified_file} not found")
        return
    
    try:
        with open(unified_file, 'r', encoding='utf-8') as f:
            unified_data = json.load(f)
        # Create lookup dictionary: server_id -> server data
        server_lookup = {server['id']: server for server in unified_data}
        logger.info(f"Loaded {len(server_lookup)} servers for metadata matching")
    except Exception as e:
        logger.error(f"Could not load unified data for metadata matching: {e}")
        return
    
    # Match additional metadata fields
    logger.info("Matching creation dates and metadata...")
    
    # Add creation dates
    results_df['created_at'] = results_df['server_id'].apply(
        lambda server_id: server_lookup.get(server_id, {}).get('created_at', '')
    )
    
    # Add use count (Smithery usage metric)
    results_df['use_count'] = results_df['server_id'].apply(
        lambda server_id: server_lookup.get(server_id, {}).get('use_count', '')
    )
    
    # Add stargazers count (GitHub metric)
    results_df['stargazers_count'] = results_df['server_id'].apply(
        lambda server_id: server_lookup.get(server_id, {}).get('stargazers_count', '')
    )
    
    # Count successful matches
    matched_created_at = len(results_df[results_df['created_at'] != ''])
    matched_use_count = len(results_df[results_df['use_count'] != ''])
    matched_stars = len(results_df[results_df['stargazers_count'] != ''])
    
    logger.info(f"Matched creation dates: {matched_created_at}/{len(results_df)} ({matched_created_at/len(results_df)*100:.1f}%)")
    logger.info(f"Matched use counts: {matched_use_count}/{len(results_df)} ({matched_use_count/len(results_df)*100:.1f}%)")
    logger.info(f"Matched star counts: {matched_stars}/{len(results_df)} ({matched_stars/len(results_df)*100:.1f}%)")
    
    # Reorder columns to put metadata fields after basic server info
    input_columns = ['server_name', 'server_id', 'description', 'created_at', 'use_count', 'stargazers_count', 
                     'readme_filtered', 'readme_summary', 'tools', 'topics', 'data_sources']
    analysis_columns = ['server', 'analysis_notes', 'is_finance_llm', 'asset_type', 'confidence', 'level']
    capability_columns = [
        'research_and_risk_assessment', 'documentation_gathering', 'application_and_review',
        'identity_verification', 'authorization_account_transactions', 'account_opening'
    ]
    transfer_columns = [
        'transfer_bank_and_fund_bank_account', 'transfer_credit_card', 'transfer_paypal_stripe_payments',
        'transfer_stock_invest', 'transfer_crypto_and_stablecoin'
    ]
    other_columns = ['sensitive_data_required', 'sample_id', 'score', 'score_explanation', 'parsed_output']
    
    # Create ordered column list
    ordered_columns = input_columns + analysis_columns + capability_columns + transfer_columns + other_columns
    
    # Select only columns that exist in the DataFrame
    existing_columns = [col for col in ordered_columns if col in results_df.columns]
    results_df = results_df[existing_columns]
    
    # Save the enhanced results
    output_file = "conseq_fin_stage2.csv"
    results_df.to_csv(output_file, index=False)
    logger.info(f"Enhanced results saved to {output_file}")
    
    # Generate summary with metadata insights
    summary = {
        "processing_timestamp": datetime.now().isoformat(),
        "total_servers": len(results_df),
        "metadata_matching": {
            "created_at_matched": matched_created_at,
            "use_count_matched": matched_use_count,
            "stargazers_matched": matched_stars,
            "match_percentages": {
                "created_at": f"{matched_created_at/len(results_df)*100:.1f}%",
                "use_count": f"{matched_use_count/len(results_df)*100:.1f}%",
                "stargazers": f"{matched_stars/len(results_df)*100:.1f}%"
            }
        }
    }
    
    # Add creation date analysis if we have matches
    if matched_created_at > 0:
        # Parse creation dates and analyze
        valid_dates = results_df[results_df['created_at'] != '']['created_at']
        try:
            date_series = pd.to_datetime(valid_dates, errors='coerce')
            date_series = date_series.dropna()
            
            if len(date_series) > 0:
                summary["creation_date_analysis"] = {
                    "earliest_server": date_series.min().isoformat(),
                    "latest_server": date_series.max().isoformat(),
                    "servers_2024": len(date_series[date_series.dt.year == 2024]),
                    "servers_2025": len(date_series[date_series.dt.year == 2025])
                }
        except Exception as e:
            logger.warning(f"Could not analyze creation dates: {e}")
    
    # Save summary
    summary_file = "conseq_fin_stage2_datamatch_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary saved to {summary_file}")
    
    logger.info("Stage 2 data matching completed successfully!")

if __name__ == "__main__":
    main()
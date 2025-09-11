#!/usr/bin/env python3
"""
CLServers Step 4: Financial MCP Server - Data Matching

Matches the CLServers Step 3 evaluation results with the unified dataset to add
creation dates and other metadata fields that weren't included in the
original evaluation.

This should be run after:
    python clservers_3_dfprocessing.py

Usage:
    python clservers_4_datamatch.py
"""

import json
import logging
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import boto3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/clservers_4_datamatch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def expand_tools_columns(df):
    """
    Expand the tools JSON column into exactly 99 tool columns with sequential filling.
    Creates tool_01_name through tool_99_inputSchema, filling only when tools exist.
    
    Args:
        df: DataFrame with a 'tools' column containing JSON arrays
        
    Returns:
        DataFrame with exactly 99 sets of tool columns (297 total tool columns)
    """
    logger.info("Starting tools column expansion for exactly 99 tools...")
    
    # Parse tools column
    parsed_tools = []
    max_tools_found = 0
    
    for idx, row in df.iterrows():
        tools_str = row.get('tools', '[]')
        try:
            # Handle empty string or None
            if not tools_str or tools_str == '[]':
                tools = []
            else:
                if isinstance(tools_str, str):
                    # First try standard JSON parsing
                    try:
                        tools = json.loads(tools_str)
                    except json.JSONDecodeError:
                        # If that fails, try ast.literal_eval for Python literals with single quotes
                        import ast
                        tools = ast.literal_eval(tools_str)
                else:
                    tools = tools_str
                
                # Ensure it's a list
                if not isinstance(tools, list):
                    tools = []
        except (json.JSONDecodeError, TypeError, ValueError, SyntaxError) as e:
            logger.warning(f"Could not parse tools for row {idx} (server_id: {row.get('server_id', 'unknown')}): {e}")
            tools = []
        
        parsed_tools.append(tools)
        max_tools_found = max(max_tools_found, len(tools))
    
    logger.info(f"Found maximum of {max_tools_found} tools per server across {len(df)} servers")
    
    # Create exactly 99 tool column sets (tool_01_name to tool_99_inputSchema)
    new_columns = {}
    for tool_num in range(1, 100):  # 1 to 99 inclusive
        tool_num_str = f"{tool_num:02d}"  # Zero-padded (01, 02, etc.)
        new_columns[f'tool_{tool_num_str}_name'] = []
        new_columns[f'tool_{tool_num_str}_description'] = []
        new_columns[f'tool_{tool_num_str}_inputSchema'] = []
    
    # Populate the new columns with sequential filling
    for row_idx, tools in enumerate(parsed_tools):
        for tool_num in range(1, 100):  # 1 to 99 inclusive
            tool_num_str = f"{tool_num:02d}"  # Zero-padded
            tool_idx = tool_num - 1  # Convert to 0-based index
            
            if tool_idx < len(tools) and isinstance(tools[tool_idx], dict):
                # Tool exists at this position
                tool = tools[tool_idx]
                new_columns[f'tool_{tool_num_str}_name'].append(tool.get('name', ''))
                new_columns[f'tool_{tool_num_str}_description'].append(tool.get('description', ''))
                
                # Handle inputSchema - serialize to JSON string if present
                input_schema = tool.get('inputSchema', '')
                if input_schema and isinstance(input_schema, (dict, list)):
                    try:
                        input_schema = json.dumps(input_schema)
                    except (TypeError, ValueError):
                        input_schema = str(input_schema)
                elif not input_schema:
                    input_schema = ''
                
                new_columns[f'tool_{tool_num_str}_inputSchema'].append(input_schema)
            else:
                # No tool at this position - fill with empty strings
                new_columns[f'tool_{tool_num_str}_name'].append('')
                new_columns[f'tool_{tool_num_str}_description'].append('')
                new_columns[f'tool_{tool_num_str}_inputSchema'].append('')
    
    # Add tool_count column first
    tool_counts = [len(tools) for tools in parsed_tools]
    df['tool_count'] = tool_counts
    
    # Add new columns to DataFrame using concat for better performance
    new_df = pd.DataFrame(new_columns)
    df = pd.concat([df, new_df], axis=1)
    
    # Remove the original tools column
    if 'tools' in df.columns:
        df = df.drop('tools', axis=1)
    
    # Count statistics
    total_tools = sum(len(tools) for tools in parsed_tools)
    servers_with_tools = sum(1 for tools in parsed_tools if len(tools) > 0)
    
    logger.info(f"Expanded {total_tools} total tools across {servers_with_tools} servers")
    logger.info("Created exactly 297 tool columns (99 sets of name/description/inputSchema)")
    logger.info(f"Maximum tools found in any server: {max_tools_found}")
    
    return df

def main():
    """Main data matching function"""
    logger.info("Starting CLServers Step 4 Data Matching")
    
    # Load CLServers Step 3 results from JSON
    clservers_json = "data/internal-cl/clservers_3_results.json"
    if not Path(clservers_json).exists():
        logger.error(f"CLServers Step 3 results file {clservers_json} not found. Run clservers_3_dfprocessing.py first.")
        return
    
    # Load JSON and convert to DataFrame
    with open(clservers_json, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Handle both old format (list) and new format (dict with results key)
    if isinstance(json_data, list):
        results_data = json_data
    elif isinstance(json_data, dict) and 'results' in json_data:
        results_data = json_data['results']
    else:
        results_data = json_data
    
    results_df = pd.DataFrame(results_data)
    logger.info(f"Loaded {len(results_df)} CLServers Step 3 results from {clservers_json}")
    
    # Load the unified dataset for server metadata matching
    unified_file = 'data/initial/data_unified_filtered.json'
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
    
    # Extract fields from nested structures if they're not already columns
    if 'server_id' not in results_df.columns:
        results_df['server_id'] = results_df['input_data'].apply(
            lambda x: x.get('server_id', '') if isinstance(x, dict) else ''
        )
    
    if 'server_name' not in results_df.columns:
        results_df['server_name'] = results_df['input_data'].apply(
            lambda x: x.get('server_name', '') if isinstance(x, dict) else ''
        )
    
    if 'description' not in results_df.columns:
        results_df['description'] = results_df['input_data'].apply(
            lambda x: x.get('description', '') if isinstance(x, dict) else ''
        )
    
    if 'readme_filtered' not in results_df.columns:
        results_df['readme_filtered'] = results_df['input_data'].apply(
            lambda x: x.get('readme_filtered', '') if isinstance(x, dict) else ''
        )
    
    if 'readme_summary' not in results_df.columns:
        results_df['readme_summary'] = results_df['input_data'].apply(
            lambda x: x.get('readme_summary', '') if isinstance(x, dict) else ''
        )
    
    if 'tools' not in results_df.columns:
        results_df['tools'] = results_df['input_data'].apply(
            lambda x: str(x.get('tools', [])) if isinstance(x, dict) else '[]'
        )
    
    if 'topics' not in results_df.columns:
        results_df['topics'] = results_df['input_data'].apply(
            lambda x: str(x.get('topics', [])) if isinstance(x, dict) else '[]'
        )
    
    if 'data_sources' not in results_df.columns:
        results_df['data_sources'] = results_df['input_data'].apply(
            lambda x: str(x.get('data_sources', [])) if isinstance(x, dict) else '[]'
        )
    
    # Extract fields from parsed_output if they're not already columns
    parsed_fields = [
        'analysis_notes', 'is_finance_llm', 'asset_type', 'level',
        'research_and_risk_assessment', 'documentation_gathering', 'application_and_review',
        'identity_verification', 'authorization_account_transactions', 'account_opening',
        'transfer_bank_and_fund_bank_account', 'transfer_credit_card', 'transfer_paypal_stripe_payments',
        'transfer_stock_invest', 'transfer_crypto_and_stablecoin', 'sensitive_data_required'
    ]
    
    for field in parsed_fields:
        if field not in results_df.columns:
            results_df[field] = results_df['parsed_output'].apply(
                lambda x: x.get(field, '') if isinstance(x, dict) else ''
            )
    
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
    
    # Expand tools column into individual tool columns
    results_df = expand_tools_columns(results_df)
    
    # Reorder columns to put metadata fields after basic server info
    input_columns = ['server_name', 'server_id', 'description', 'created_at', 'use_count', 'stargazers_count', 
                     'readme_filtered', 'readme_summary', 'topics', 'data_sources']
    analysis_columns = ['analysis_notes', 'is_finance_llm', 'asset_type', 'level']
    capability_columns = [
        'research_and_risk_assessment', 'documentation_gathering', 'application_and_review',
        'identity_verification', 'authorization_account_transactions', 'account_opening'
    ]
    transfer_columns = [
        'transfer_bank_and_fund_bank_account', 'transfer_credit_card', 'transfer_paypal_stripe_payments',
        'transfer_stock_invest', 'transfer_crypto_and_stablecoin', 'sensitive_data_required'
    ]
    
    # Get tool_count column and all tool columns (dynamically created)
    tool_count_column = ['tool_count'] if 'tool_count' in results_df.columns else []
    tool_columns = [col for col in results_df.columns if col.startswith('tool_') and 
                   any(col.endswith(suffix) for suffix in ['_name', '_description', '_inputSchema'])]
    tool_columns.sort()  # Ensure consistent ordering
    
    # Create ordered column list (excluding other_columns)
    ordered_columns = input_columns + tool_count_column + tool_columns + analysis_columns + capability_columns + transfer_columns
    
    # Select only columns that exist in the DataFrame and exclude unwanted columns
    existing_columns = [col for col in ordered_columns if col in results_df.columns]
    results_df = results_df[existing_columns]
    
    # Save the enhanced results
    output_file = "data/final/clservers_classified.csv"
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
    
    # Summary generated for logging purposes only
    
    # Upload to AWS S3
    logger.info("Uploading results to AWS S3...")
    try:
        s3 = boto3.client('s3')
        s3.upload_file(
            'data/final/clservers_classified.csv',
            os.environ['AISI_PLATFORM_BUCKET'],
            f'users/{os.environ["AISI_PLATFORM_USER"]}/server_classified.csv'
        )
        logger.info("Successfully uploaded data/final/clservers_classified.csv to S3")
    except Exception as e:
        logger.error(f"Error during S3 upload: {e}")
    
    logger.info("CLServers Step 4 data matching completed successfully!")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
CLServers Step 4: Financial MCP Server - Data Matching

Matches the CLServers Step 3 evaluation results with the unified dataset to add
creation dates and other metadata fields that weren't included in the
original evaluation.

This should be run after:
    python clservers_3_dfprocessing.py
    python clservers_3_dfprocessing.py --task naics

Usage:
    python clservers_4_datamatch.py
"""

import json
import logging
import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
import boto3

# Add script directory to path to import naics_3digit_data
sys.path.insert(0, str(Path(__file__).parent))
from naics_3digit_data import get_naics_title

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
    import argparse
    parser = argparse.ArgumentParser(description='CLServers Step 4: Data Matching')
    parser.add_argument('--append-to', type=str, help='Path to existing clservers_classified.csv.gz to append new results to')
    args = parser.parse_args()

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
        'action_space_description', 'generality_industry', 'generality_environment',
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
    
    # Add creation dates and normalize to consistent ISO format
    results_df['created_at'] = results_df['server_id'].apply(
        lambda server_id: server_lookup.get(server_id, {}).get('created_at', '')
    )
    # Normalize mixed datetime formats (Smithery has microseconds, GitHub doesn't)
    # so downstream pd.to_datetime() works without format="mixed"
    results_df['created_at'] = pd.to_datetime(
        results_df['created_at'], format='mixed', errors='coerce'
    ).dt.strftime('%Y-%m-%dT%H:%M:%S+00:00').fillna('')
    
    # Add use count (Smithery usage metric)
    results_df['use_count'] = results_df['server_id'].apply(
        lambda server_id: server_lookup.get(server_id, {}).get('use_count', '')
    )
    
    # Add stargazers count (GitHub metric)
    results_df['stargazers_count'] = results_df['server_id'].apply(
        lambda server_id: server_lookup.get(server_id, {}).get('stargazers_count', '')
    )

    # Add canonical_official, name, owner, repository_url
    results_df['canonical_official'] = results_df['server_id'].apply(
        lambda server_id: server_lookup.get(server_id, {}).get('canonical_official', '')
    )

    results_df['name'] = results_df['server_id'].apply(
        lambda server_id: server_lookup.get(server_id, {}).get('name', '')
    )

    results_df['owner'] = results_df['server_id'].apply(
        lambda server_id: server_lookup.get(server_id, {}).get('owner', '')
    )

    results_df['repository_url'] = results_df['server_id'].apply(
        lambda server_id: server_lookup.get(server_id, {}).get('repository_url', '')
    )

    # Add download/usage data
    results_df['usage_pypi_downloads'] = results_df['server_id'].apply(
        lambda server_id: server_lookup.get(server_id, {}).get('usage_pypi_downloads', '')
    )

    results_df['usage_npm_downloads'] = results_df['server_id'].apply(
        lambda server_id: server_lookup.get(server_id, {}).get('usage_npm_downloads', '')
    )

    results_df['usage_total_downloads'] = results_df['server_id'].apply(
        lambda server_id: server_lookup.get(server_id, {}).get('usage_total_downloads', '')
    )

    results_df['usage_monthly_breakdown'] = results_df['server_id'].apply(
        lambda server_id: server_lookup.get(server_id, {}).get('usage_monthly_breakdown', '')
    )

    results_df['usage_matched_packages'] = results_df['server_id'].apply(
        lambda server_id: server_lookup.get(server_id, {}).get('usage_matched_packages', '')
    )

    results_df['usage_match_method'] = results_df['server_id'].apply(
        lambda server_id: server_lookup.get(server_id, {}).get('usage_match_method', '')
    )

    results_df['usage_last_updated'] = results_df['server_id'].apply(
        lambda server_id: server_lookup.get(server_id, {}).get('usage_last_updated', '')
    )

    # Note: pypi_by_country geo data is now inside usage_monthly_breakdown entries,
    # not at the server level. The breakdown already contains this data.

    # Add AI-created detection fields (from detect_ai_created.py → data_unified_filtered.json)
    results_df['ai_authored'] = results_df['server_id'].apply(
        lambda server_id: server_lookup.get(server_id, {}).get('ai_authored', '')
    )

    results_df['ai_authored_reasons'] = results_df['server_id'].apply(
        lambda server_id: server_lookup.get(server_id, {}).get('ai_authored_reasons', '')
    )

    results_df['likely_ai_agent'] = results_df['server_id'].apply(
        lambda server_id: server_lookup.get(server_id, {}).get('likely_ai_agent', '')
    )

    results_df['likely_creators_details'] = results_df['server_id'].apply(
        lambda server_id: server_lookup.get(server_id, {}).get('likely_creators_details', '')
    )

    # Add first-month AI-created detection fields
    results_df['ai_authored_first_month'] = results_df['server_id'].apply(
        lambda server_id: server_lookup.get(server_id, {}).get('ai_authored_first_month', '')
    )

    results_df['ai_authored_first_month_reasons'] = results_df['server_id'].apply(
        lambda server_id: server_lookup.get(server_id, {}).get('ai_authored_first_month_reasons', '')
    )

    results_df['first_month_likely_ai_agent'] = results_df['server_id'].apply(
        lambda server_id: server_lookup.get(server_id, {}).get('first_month_likely_ai_agent', '')
    )

    results_df['date_first_ai_evidence'] = results_df['server_id'].apply(
        lambda server_id: server_lookup.get(server_id, {}).get('date_first_ai_evidence', '')
    )

    # Count successful matches
    matched_created_at = len(results_df[results_df['created_at'] != ''])
    matched_use_count = len(results_df[results_df['use_count'] != ''])
    matched_stars = len(results_df[results_df['stargazers_count'] != ''])
    matched_canonical_official = len(results_df[results_df['canonical_official'] != ''])
    matched_name = len(results_df[results_df['name'] != ''])
    matched_owner = len(results_df[results_df['owner'] != ''])
    matched_repository_url = len(results_df[results_df['repository_url'] != ''])
    matched_usage_total_downloads = len(results_df[results_df['usage_total_downloads'] != ''])
    matched_usage_last_updated = len(results_df[results_df['usage_last_updated'] != ''])

    logger.info(f"Matched creation dates: {matched_created_at}/{len(results_df)} ({matched_created_at/len(results_df)*100:.1f}%)")
    logger.info(f"Matched use counts: {matched_use_count}/{len(results_df)} ({matched_use_count/len(results_df)*100:.1f}%)")
    logger.info(f"Matched star counts: {matched_stars}/{len(results_df)} ({matched_stars/len(results_df)*100:.1f}%)")
    logger.info(f"Matched canonical_official: {matched_canonical_official}/{len(results_df)} ({matched_canonical_official/len(results_df)*100:.1f}%)")
    logger.info(f"Matched name: {matched_name}/{len(results_df)} ({matched_name/len(results_df)*100:.1f}%)")
    logger.info(f"Matched owner: {matched_owner}/{len(results_df)} ({matched_owner/len(results_df)*100:.1f}%)")
    logger.info(f"Matched repository_url: {matched_repository_url}/{len(results_df)} ({matched_repository_url/len(results_df)*100:.1f}%)")
    logger.info(f"Matched usage_total_downloads: {matched_usage_total_downloads}/{len(results_df)} ({matched_usage_total_downloads/len(results_df)*100:.1f}%)")
    logger.info(f"Matched usage_last_updated: {matched_usage_last_updated}/{len(results_df)} ({matched_usage_last_updated/len(results_df)*100:.1f}%)")

    # Match NAICS classifications
    naics_file = 'data/internal-cl/clservers_naics_results.json'
    if Path(naics_file).exists():
        logger.info("Matching NAICS classifications...")
        try:
            with open(naics_file, 'r', encoding='utf-8') as f:
                naics_data = json.load(f)

            # Extract results (handle both formats)
            if isinstance(naics_data, dict) and 'results' in naics_data:
                naics_results = naics_data['results']
            else:
                naics_results = naics_data

            # Create lookup dictionary: server_id -> naics classification
            naics_lookup = {}
            for result in naics_results:
                parsed_output = result.get('parsed_output', {})
                if isinstance(parsed_output, dict):
                    # Get server_id from input_data
                    input_data = result.get('input_data', {})
                    if isinstance(input_data, dict):
                        server_id = input_data.get('server_id', '')
                        naics_code = parsed_output.get('naics_code', '')
                        reasoning = parsed_output.get('reasoning', '')

                        if server_id and naics_code:
                            naics_lookup[server_id] = {
                                'naics_code': naics_code,
                                'naics_reasoning': reasoning
                            }

            logger.info(f"Loaded {len(naics_lookup)} NAICS classifications")

            # Add NAICS code and title to results
            results_df['naics_code'] = results_df['server_id'].apply(
                lambda server_id: naics_lookup.get(server_id, {}).get('naics_code', '')
            )
            results_df['naics_title'] = results_df['naics_code'].apply(
                lambda code: get_naics_title(code) if code and code != 'cross-sector' else code
            )
            results_df['naics_reasoning'] = results_df['server_id'].apply(
                lambda server_id: naics_lookup.get(server_id, {}).get('naics_reasoning', '')
            )

            matched_naics = len(results_df[results_df['naics_code'] != ''])
            logger.info(f"Matched NAICS codes: {matched_naics}/{len(results_df)} ({matched_naics/len(results_df)*100:.1f}%)")

            # Show distribution of top NAICS codes
            if matched_naics > 0:
                naics_dist = results_df[results_df['naics_code'] != '']['naics_code'].value_counts()
                logger.info(f"Top 10 NAICS codes: {dict(list(naics_dist.items())[:10])}")

        except Exception as e:
            logger.warning(f"Could not load NAICS classifications: {e}")
            results_df['naics_code'] = ''
            results_df['naics_title'] = ''
            results_df['naics_reasoning'] = ''
    else:
        logger.warning(f"NAICS results file not found: {naics_file}")
        results_df['naics_code'] = ''
        results_df['naics_title'] = ''
        results_df['naics_reasoning'] = ''

    # Expand tools column into individual tool columns
    results_df = expand_tools_columns(results_df)
    
    # Reorder columns to put metadata fields after basic server info
    # Note: pypi_by_country geo data is now embedded inside usage_monthly_breakdown entries
    input_columns = ['server_name', 'server_id', 'name', 'owner', 'repository_url', 'canonical_official',
                     'description', 'created_at', 'use_count', 'stargazers_count',
                     'usage_pypi_downloads', 'usage_npm_downloads', 'usage_total_downloads',
                     'usage_monthly_breakdown', 'usage_matched_packages', 'usage_match_method', 'usage_last_updated',
                     'readme_filtered', 'readme_summary', 'topics', 'data_sources']
    naics_columns = ['naics_code', 'naics_title', 'naics_reasoning']
    analysis_columns = ['analysis_notes', 'is_finance_llm', 'asset_type', 'level', 'action_space_description', 'generality_industry', 'generality_environment']
    capability_columns = [
        'research_and_risk_assessment', 'documentation_gathering', 'application_and_review',
        'identity_verification', 'authorization_account_transactions', 'account_opening'
    ]
    transfer_columns = [
        'transfer_bank_and_fund_bank_account', 'transfer_credit_card', 'transfer_paypal_stripe_payments',
        'transfer_stock_invest', 'transfer_crypto_and_stablecoin', 'sensitive_data_required'
    ]
    payment_columns = ['payments_analysis', 'payments_autonomy']
    ai_created_columns = ['ai_authored', 'ai_authored_reasons', 'likely_ai_agent', 'likely_creators_details',
                          'ai_authored_first_month', 'ai_authored_first_month_reasons', 'first_month_likely_ai_agent',
                          'date_first_ai_evidence']

    # Get tool_count column and all tool columns (dynamically created)
    tool_count_column = ['tool_count'] if 'tool_count' in results_df.columns else []
    tool_columns = [col for col in results_df.columns if col.startswith('tool_') and
                   any(col.endswith(suffix) for suffix in ['_name', '_description', '_inputSchema'])]
    tool_columns.sort()  # Ensure consistent ordering

    # Create ordered column list (excluding other_columns)
    ordered_columns = input_columns + naics_columns + ai_created_columns + tool_count_column + tool_columns + analysis_columns + capability_columns + transfer_columns + payment_columns
    
    # Select only columns that exist in the DataFrame and exclude unwanted columns
    existing_columns = [col for col in ordered_columns if col in results_df.columns]
    results_df = results_df[existing_columns]

    # Convert list/dict columns to JSON strings for proper CSV serialization
    for col in ['usage_monthly_breakdown', 'usage_matched_packages', 'topics',
                'ai_authored_reasons', 'likely_creators_details',
                'ai_authored_first_month_reasons']:
        if col in results_df.columns:
            results_df[col] = results_df[col].apply(
                lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
            )

    # Append to existing file if requested (load BEFORE writing to avoid overwrite)
    output_file = "data/final/clservers_classified.csv.gz"
    if args.append_to and Path(args.append_to).exists():
        logger.info(f"Appending to existing file: {args.append_to}")
        existing_df = pd.read_csv(args.append_to, low_memory=False)
        logger.info(f"Existing data: {len(existing_df)} servers, new data: {len(results_df)} servers")
        results_df = pd.concat([existing_df, results_df], ignore_index=True)
        results_df = results_df.drop_duplicates(subset=['server_id'], keep='last')
        logger.info(f"Combined after dedup: {len(results_df)} servers")

    # Refresh usage and metadata fields for ALL servers from unified dataset
    # This ensures previously-processed servers get updated usage data too
    refreshable_fields = [
        'usage_pypi_downloads', 'usage_npm_downloads', 'usage_total_downloads',
        'usage_monthly_breakdown', 'usage_matched_packages', 'usage_match_method', 'usage_last_updated',
        'stargazers_count',
        'ai_authored', 'ai_authored_reasons', 'likely_ai_agent', 'likely_creators_details',
        'ai_authored_first_month', 'ai_authored_first_month_reasons', 'first_month_likely_ai_agent',
        'date_first_ai_evidence',
    ]
    refreshed = 0
    for idx, row in results_df.iterrows():
        server_data = server_lookup.get(row['server_id'])
        if server_data:
            refreshed += 1
            for field in refreshable_fields:
                val = server_data.get(field)
                if val is not None:
                    if field in ('usage_monthly_breakdown', 'usage_matched_packages',
                                 'ai_authored_reasons', 'likely_creators_details',
                                 'ai_authored_first_month_reasons'):
                        results_df.at[idx, field] = json.dumps(val) if isinstance(val, (list, dict)) else val
                    else:
                        results_df.at[idx, field] = val
    logger.info(f"Refreshed usage/metadata for {refreshed}/{len(results_df)} servers from unified dataset")

    # Save the enhanced results
    results_df.to_csv(output_file, index=False, compression='gzip')
    logger.info(f"Enhanced results saved to {output_file} ({len(results_df)} servers)")

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
            'data/final/clservers_classified.csv.gz',
            os.environ['AISI_PLATFORM_BUCKET'],
            f'users/{os.environ["AISI_PLATFORM_USER"]}/server_classified.csv.gz'
        )
        logger.info("Successfully uploaded data/final/clservers_classified.csv.gz to S3")
    except Exception as e:
        logger.error(f"Error during S3 upload: {e}")
    
    logger.info("CLServers Step 4 data matching completed successfully!")

if __name__ == "__main__":
    main()
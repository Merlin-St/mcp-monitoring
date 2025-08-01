#!/usr/bin/env python3
"""
Financial MCP Server Stage 4 - Data Preparation for O*NET Economic Task Classification

Extracts individual tools from data_unified_filtered.json and stage1 results
to create JSONL input for O*NET economic task classification.

This should be run after:
    python conseq_fin_stage1_dfprocessing.py

Usage:
    python conseq_fin_stage4_data_prep.py [--sample-size N] [--finance-only]
"""

import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conseq_fin_stage4_data_prep.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_stage1_results():
    """Load Stage 1 finance classification results"""
    stage1_file = "conseq_fin_stage1_results.json"
    if not Path(stage1_file).exists():
        logger.warning(f"Stage 1 results file {stage1_file} not found. Will proceed without finance classifications.")
        return {}
    
    try:
        with open(stage1_file, 'r', encoding='utf-8') as f:
            stage1_data = json.load(f)
        
        # Create lookup dictionary: server_id -> finance classification
        finance_lookup = {}
        for result in stage1_data.get('results', []):
            server_id = result.get('input_data', {}).get('server_id', '')
            parsed_output = result.get('parsed_output', {})
            if server_id and parsed_output:
                finance_lookup[server_id] = {
                    'is_finance_llm': parsed_output.get('is_finance_llm', 0),
                    'asset_type': parsed_output.get('asset_type', ''),
                    'level': parsed_output.get('level', 0),
                    'analysis_notes': parsed_output.get('analysis_notes', '')
                }
        
        logger.info(f"Loaded finance classifications for {len(finance_lookup)} servers from Stage 1")
        return finance_lookup
        
    except Exception as e:
        logger.error(f"Error loading Stage 1 results: {e}")
        return {}

def extract_tools_from_json(json_file, finance_lookup, sample_size=None, finance_only=False):
    """
    Extract individual tools from the unified JSON file
    
    Args:
        json_file: Path to data_unified_filtered.json
        finance_lookup: Dictionary of server_id -> finance classification
        sample_size: Maximum number of tools to extract (None for all)
        finance_only: If True, only extract tools from finance-related servers
        
    Returns:
        List of tool records
    """
    logger.info(f"Extracting tools from {json_file}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        servers = json.load(f)
    
    logger.info(f"Loaded {len(servers)} servers from unified dataset")
    
    tools_extracted = []
    servers_processed = 0
    servers_with_tools = 0
    finance_servers_found = 0
    
    for server in servers:
        servers_processed += 1
        server_id = server.get('id', '')
        server_name = server.get('name', '')
        
        # Get finance classification if available
        finance_info = finance_lookup.get(server_id, {})
        is_finance = finance_info.get('is_finance_llm', 0) == 1
        
        if finance_only and not is_finance:
            continue
            
        if is_finance:
            finance_servers_found += 1
        
        # Extract tools from this server
        tools = server.get('tools', [])
        if not tools:
            continue
            
        servers_with_tools += 1
        
        # Process each tool
        for tool_idx, tool in enumerate(tools):
            if not isinstance(tool, dict):
                continue
                
            tool_name = tool.get('name', '')
            tool_description = tool.get('description', '')
            tool_input_schema = tool.get('inputSchema', {})
            
            # Skip tools without basic information
            if not tool_name and not tool_description:
                continue
            
            # Create tool record
            tool_record = {
                # Tool-specific information
                'tool_id': f"{server_id}#{tool_idx + 1:02d}",
                'tool_name': tool_name,
                'tool_description': tool_description,
                'tool_input_schema': json.dumps(tool_input_schema) if tool_input_schema else '',
                'tool_position': tool_idx + 1,
                
                # Parent server context
                'server_id': server_id,
                'server_name': server_name,
                'server_description': server.get('description', ''),
                'server_readme_summary': server.get('readme_summary', ''),
                'server_readme_filtered': server.get('readme_filtered', ''),
                'server_created_at': server.get('created_at', ''),
                'server_stargazers_count': server.get('stargazers_count', 0),
                'server_topics': server.get('topics', []),
                'server_data_sources': server.get('data_sources', []),
                'server_tool_count': len(tools),
                
                # Finance classification from Stage 1
                'finance_is_finance_llm': finance_info.get('is_finance_llm', 0),
                'finance_asset_type': finance_info.get('asset_type', ''),
                'finance_level': finance_info.get('level', 0),
                'finance_analysis_notes': finance_info.get('analysis_notes', '')
            }
            
            tools_extracted.append(tool_record)
            
            # Check sample size limit
            if sample_size and len(tools_extracted) >= sample_size:
                logger.info(f"Reached sample size limit of {sample_size} tools")
                break
        
        if sample_size and len(tools_extracted) >= sample_size:
            break
    
    logger.info(f"Processed {servers_processed} servers")
    logger.info(f"Found {servers_with_tools} servers with tools")
    logger.info(f"Found {finance_servers_found} finance-related servers")
    logger.info(f"Extracted {len(tools_extracted)} individual tools")
    
    return tools_extracted

def save_tools_jsonl(tools, output_file):
    """Save tools to JSONL format for Inspect framework"""
    logger.info(f"Saving {len(tools)} tools to {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for tool in tools:
            # Create the input format expected by Inspect
            inspect_record = {
                'input': json.dumps(tool),
                'target': 'classify_economic_task'  # Placeholder target
            }
            f.write(json.dumps(inspect_record) + '\n')
    
    logger.info(f"Tools saved to {output_file}")

def generate_summary(tools, output_file, args):
    """Generate summary statistics"""
    summary = {
        'generation_timestamp': datetime.now().isoformat(),
        'source_file': 'data_unified_filtered.json',
        'parameters': {
            'sample_size': args.sample_size,
            'finance_only': args.finance_only
        },
        'total_tools': len(tools),
        'unique_servers': len(set(tool['server_id'] for tool in tools))
    }
    
    # Finance breakdown
    finance_tools = [t for t in tools if t['finance_is_finance_llm'] == 1]
    summary['finance_breakdown'] = {
        'finance_tools': len(finance_tools),
        'non_finance_tools': len(tools) - len(finance_tools),
        'finance_percentage': (len(finance_tools) / len(tools) * 100) if tools else 0
    }
    
    # Tool name analysis
    tools_with_names = [t for t in tools if t['tool_name']]
    tools_with_descriptions = [t for t in tools if t['tool_description']]
    summary['tool_completeness'] = {
        'tools_with_names': len(tools_with_names),
        'tools_with_descriptions': len(tools_with_descriptions),
        'tools_with_both': len([t for t in tools if t['tool_name'] and t['tool_description']])
    }
    
    # Server statistics
    server_tool_counts = {}
    for tool in tools:
        server_id = tool['server_id']
        if server_id not in server_tool_counts:
            server_tool_counts[server_id] = 0
        server_tool_counts[server_id] += 1
    
    tool_count_values = list(server_tool_counts.values())
    summary['server_statistics'] = {
        'servers_with_1_tool': sum(1 for count in tool_count_values if count == 1),
        'servers_with_2_5_tools': sum(1 for count in tool_count_values if 2 <= count <= 5),
        'servers_with_6_plus_tools': sum(1 for count in tool_count_values if count >= 6),
        'max_tools_per_server': max(tool_count_values) if tool_count_values else 0,
        'avg_tools_per_server': sum(tool_count_values) / len(tool_count_values) if tool_count_values else 0
    }
    
    # Top servers by tool count
    top_servers = sorted(server_tool_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    summary['top_servers_by_tool_count'] = [
        {
            'server_id': server_id,
            'tool_count': count,
            'server_name': next((t['server_name'] for t in tools if t['server_id'] == server_id), 'Unknown')
        }
        for server_id, count in top_servers
    ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Summary saved to {output_file}")
    return summary

def main():
    """Main data preparation function"""
    parser = argparse.ArgumentParser(description='Prepare MCP tool data for O*NET economic task classification')
    parser.add_argument('--sample-size', type=int, help='Maximum number of tools to extract')
    parser.add_argument('--finance-only', action='store_true', help='Only extract tools from finance-related servers')
    args = parser.parse_args()
    
    logger.info("Starting Stage 4 Data Preparation for O*NET Economic Task Classification")
    
    # Load unified dataset
    unified_file = 'data_unified_filtered.json'
    if not Path(unified_file).exists():
        logger.error(f"Unified dataset {unified_file} not found")
        return
    
    # Load Stage 1 finance classifications
    finance_lookup = load_stage1_results()
    
    # Extract tools
    tools = extract_tools_from_json(
        unified_file, 
        finance_lookup, 
        sample_size=args.sample_size,
        finance_only=args.finance_only
    )
    
    if not tools:
        logger.error("No tools extracted. Check your filters and data.")
        return
    
    # Save to JSONL for Inspect framework
    output_file = "conseq_fin_stage4_input.jsonl"
    save_tools_jsonl(tools, output_file)
    
    # Generate summary
    summary_file = "conseq_fin_stage4_data_prep_summary.json"
    summary = generate_summary(tools, summary_file, args)
    
    # Log key statistics
    logger.info("=== Data Preparation Summary ===")
    logger.info(f"Total tools extracted: {summary['total_tools']}")
    logger.info(f"Unique servers: {summary['unique_servers']}")
    logger.info(f"Finance tools: {summary['finance_breakdown']['finance_tools']} ({summary['finance_breakdown']['finance_percentage']:.1f}%)")
    logger.info(f"Tools with names: {summary['tool_completeness']['tools_with_names']}")
    logger.info(f"Tools with descriptions: {summary['tool_completeness']['tools_with_descriptions']}")
    logger.info(f"Max tools per server: {summary['server_statistics']['max_tools_per_server']}")
    logger.info(f"Avg tools per server: {summary['server_statistics']['avg_tools_per_server']:.1f}")
    
    logger.info("=== Top Servers by Tool Count ===")
    for server in summary['top_servers_by_tool_count'][:5]:
        logger.info(f"{server['server_name']}: {server['tool_count']} tools")
    
    logger.info("=== Next Steps ===")
    logger.info("1. Run: inspect eval conseq_fin_stage4_inspect.py --model anthropic/claude-sonnet-4-20250514")
    logger.info("2. Process results: python conseq_fin_stage4_dfprocessing.py")
    
    logger.info("Stage 4 data preparation completed successfully!")

if __name__ == "__main__":
    main()
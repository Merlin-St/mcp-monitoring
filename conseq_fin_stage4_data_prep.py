#!/usr/bin/env python3
"""
Stage 4 Data Preparation - Extract Tools for O*NET Classification

This script extracts individual tools from MCP servers for O*NET task classification.
Each tool is prepared with its context (server info, description, schema) for LLM analysis.

Usage:
    python conseq_fin_stage4_data_prep.py                    # Default: sample 1000 tools
    python conseq_fin_stage4_data_prep.py --samples 5000     # Custom sample size
    python conseq_fin_stage4_data_prep.py --all              # Process all tools
    python conseq_fin_stage4_data_prep.py --finance          # Only finance-related servers
"""

import json
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

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

def load_filtered_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load the filtered MCP server dataset"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} servers from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def extract_tools_from_servers(servers: List[Dict[str, Any]], 
                             finance_only: bool = False) -> List[Dict[str, Any]]:
    """Extract all tools from servers with their context"""
    tools = []
    
    for server in servers:
        # Skip if finance_only and server is not finance-related
        if finance_only and not server.get('is_sector_52', False):
            continue
        
        server_tools = server.get('tools', [])
        
        # Skip servers without tools
        if not server_tools:
            continue
        
        for tool in server_tools:
            # Create tool record with server context
            tool_record = {
                'tool_id': f"{server.get('id', 'unknown')}_{tool.get('name', 'unnamed')}",
                'tool_name': tool.get('name', ''),
                'tool_description': tool.get('description', ''),
                'tool_input_schema': tool.get('input_schema', {}),
                'server_id': server.get('id', ''),
                'server_name': server.get('name', ''),
                'server_description': server.get('canonical_description', ''),
                'readme_summary': server.get('readme_summary', ''),
                'server_data_sources': server.get('data_sources', [])
            }
            
            tools.append(tool_record)
    
    logger.info(f"Extracted {len(tools)} tools from {len(servers)} servers")
    return tools

def create_inspect_samples(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create samples in format expected by Inspect framework"""
    samples = []
    
    for tool in tools:
        # Create input text with all tool context
        input_data = {
            "tool_name": tool['tool_name'],
            "tool_description": tool['tool_description'],
            "tool_input_schema": json.dumps(tool['tool_input_schema']) if tool['tool_input_schema'] else "",
            "server_name": tool['server_name'],
            "server_description": tool['server_description'],
            "readme_summary": tool['readme_summary']
        }
        
        sample = {
            "input": json.dumps(input_data),
            "target": "",  # Empty target for generation task
            "id": tool['tool_id'],
            "metadata": {
                "stage": "onet_classification",
                "server_id": tool['server_id']
            }
        }
        
        samples.append(sample)
    
    return samples

def save_datasets(tools: List[Dict[str, Any]], samples: List[Dict[str, Any]]):
    """Save the prepared datasets"""
    # Save full tool dataset as JSON
    tools_file = "conseq_fin_stage4_tools_full.json"
    with open(tools_file, 'w', encoding='utf-8') as f:
        json.dump(tools, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(tools)} tools to {tools_file}")
    
    # Save Inspect samples as JSONL
    samples_file = "conseq_fin_stage4_input.jsonl"
    with open(samples_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(samples)} samples to {samples_file}")
    
    # Save summary statistics
    summary = {
        "created_at": datetime.now().isoformat(),
        "total_tools": len(tools),
        "total_samples": len(samples),
        "unique_servers": len(set(t['server_id'] for t in tools)),
        "tools_with_schema": sum(1 for t in tools if t['tool_input_schema']),
        "avg_tools_per_server": len(tools) / len(set(t['server_id'] for t in tools)) if tools else 0
    }
    
    # Tool name analysis
    tool_names = [t['tool_name'] for t in tools]
    summary['unique_tool_names'] = len(set(tool_names))
    summary['most_common_tools'] = pd.Series(tool_names).value_counts().head(10).to_dict()
    
    summary_file = "conseq_fin_stage4_data_prep_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Prepare tool data for O*NET classification')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of tools to sample (default: 1000)')
    parser.add_argument('--all', action='store_true',
                       help='Process all tools (no sampling)')
    parser.add_argument('--finance', action='store_true',
                       help='Only include tools from finance-related servers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sampling')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    logger.info("Starting Stage 4 data preparation")
    logger.info(f"Settings: samples={args.samples}, all={args.all}, finance={args.finance}")
    
    # Load server data
    dataset_file = "data_unified_filtered.json"
    if not Path(dataset_file).exists():
        logger.error(f"Dataset file {dataset_file} not found!")
        return
    
    servers = load_filtered_dataset(dataset_file)
    
    # Extract tools
    all_tools = extract_tools_from_servers(servers, finance_only=args.finance)
    
    if not all_tools:
        logger.error("No tools found!")
        return
    
    # Sample if requested
    if args.all:
        selected_tools = all_tools
        logger.info(f"Using all {len(selected_tools)} tools")
    else:
        n_samples = min(args.samples, len(all_tools))
        selected_tools = random.sample(all_tools, n_samples)
        logger.info(f"Sampled {len(selected_tools)} tools from {len(all_tools)} total")
    
    # Create Inspect samples
    samples = create_inspect_samples(selected_tools)
    
    # Save datasets
    save_datasets(selected_tools, samples)
    
    # Log statistics
    logger.info("\nDataset Statistics:")
    logger.info(f"- Total servers processed: {len(servers)}")
    logger.info(f"- Total tools found: {len(all_tools)}")
    logger.info(f"- Tools selected: {len(selected_tools)}")
    logger.info(f"- Unique servers in selection: {len(set(t['server_id'] for t in selected_tools))}")
    logger.info(f"- Tools with input schema: {sum(1 for t in selected_tools if t['tool_input_schema'])}")
    
    # Show sample tools
    logger.info("\nSample tools:")
    for tool in selected_tools[:5]:
        logger.info(f"  - {tool['tool_name']} ({tool['server_name']}): {tool['tool_description'][:100]}...")
    
    logger.info("\nData preparation complete!")

if __name__ == "__main__":
    main()
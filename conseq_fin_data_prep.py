#!/usr/bin/env python3
"""
Financial MCP Server Data Preparation

Prepares MCP server data for financial risk analysis by:
1. Loading filtered server dataset
2. Sampling servers based on command line flags
3. Cleaning data (removing sector classification fields)
4. Creating Inspect-compatible JSONL dataset

Usage:
    python conseq_fin_data_prep.py                    # Default: 100 servers
    python conseq_fin_data_prep.py --samples 500      # Custom sample size
    python conseq_fin_data_prep.py --all              # Process all servers
"""

import json
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conseq_fin_data_prep.log'),
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

def clean_server_data(server: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean server data by removing sector classification fields
    and keeping only relevant fields for financial analysis
    """
    # Fields to exclude (sector classification, language, dates)
    exclude_patterns = [
        'is_sector_', 'sector_', '_keywords', 'is_subsector_', 'subsector_',
        'language', 'languages', 'created_at', 'updated_at'
    ]
    
    # Fields to include
    include_fields = {
        'id', 'name', 'canonical_name', 'canonical_description', 
        'readme_content', 'extracted_tools', 'tools_count', 'tools_names',
        'tools_by_access', 'homepage', 'url', 'repository_url', 'github_url',
        'topics', 'data_sources', 'primary_source', 'embedding_text',
        'stargazers_count', 'forks_count', 'owner_login', 'fork', 'archived'
    }
    
    cleaned = {}
    for key, value in server.items():
        # Include if explicitly in include_fields
        if key in include_fields:
            cleaned[key] = value
        # Exclude if matches any exclude pattern
        elif not any(pattern in key for pattern in exclude_patterns):
            # Include other fields not matching exclude patterns
            if not key.startswith('is_') or key in include_fields:
                cleaned[key] = value
    
    return cleaned

def create_stage1_sample(server: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a Stage 1 sample for finance identification
    """
    # Truncate readme content to manage token limits
    readme = server.get('readme_content', '')
    if len(readme) > 5000:
        readme = readme[:5000] + "\n[...truncated for length...]"
    
    sample_input = {
        "server_name": server.get('name', ''),
        "server_id": server.get('id', ''),
        "description": server.get('canonical_description', ''),
        "readme": readme,
        "tools": server.get('extracted_tools', []),
        "tools_count": server.get('tools_count', 0),
        "homepage": server.get('homepage', ''),
        "url": server.get('url', ''),
        "topics": server.get('topics', []),
        "data_sources": server.get('data_sources', [])
    }
    
    return {
        "input": json.dumps(sample_input),
        "target": "",
        "id": server.get('id', ''),
        "metadata": {"stage": "finance_filter"}
    }

def sample_servers(servers: List[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
    """
    Sample servers ensuring diversity across data sources and tool counts
    """
    if sample_size >= len(servers):
        logger.info(f"Requested sample size ({sample_size}) >= total servers ({len(servers)}), using all servers")
        return servers
    
    # Stratified sampling by data source and tool presence
    github_servers = [s for s in servers if s.get('primary_source') == 'github']
    smithery_servers = [s for s in servers if s.get('primary_source') == 'smithery'] 
    official_servers = [s for s in servers if s.get('primary_source') == 'official']
    
    # Calculate proportional sample sizes
    total = len(servers)
    github_sample_size = int((len(github_servers) / total) * sample_size)
    smithery_sample_size = int((len(smithery_servers) / total) * sample_size)
    official_sample_size = sample_size - github_sample_size - smithery_sample_size
    
    # Sample from each group
    sampled = []
    if github_servers and github_sample_size > 0:
        sampled.extend(random.sample(github_servers, min(github_sample_size, len(github_servers))))
    if smithery_servers and smithery_sample_size > 0:
        sampled.extend(random.sample(smithery_servers, min(smithery_sample_size, len(smithery_servers))))
    if official_servers and official_sample_size > 0:
        sampled.extend(random.sample(official_servers, min(official_sample_size, len(official_servers))))
    
    # Fill remaining slots if needed
    while len(sampled) < sample_size and len(sampled) < len(servers):
        remaining = [s for s in servers if s not in sampled]
        if remaining:
            sampled.append(random.choice(remaining))
        else:
            break
    
    logger.info(f"Sampled {len(sampled)} servers: "
                f"{len([s for s in sampled if s.get('primary_source') == 'github'])} GitHub, "
                f"{len([s for s in sampled if s.get('primary_source') == 'smithery'])} Smithery, "
                f"{len([s for s in sampled if s.get('primary_source') == 'official'])} Official")
    
    return sampled

def main():
    parser = argparse.ArgumentParser(description='Prepare MCP server data for financial risk analysis')
    parser.add_argument('--samples', type=int, help='Number of servers to sample (default: 100)')
    parser.add_argument('--all', action='store_true', help='Process all servers')
    parser.add_argument('--finance', action='store_true', help='Only process finance-related servers (uses is_finance_related column)')
    args = parser.parse_args()
    
    # Determine sample size
    if args.all:
        sample_size = None  # Process all
        logger.info("Processing ALL servers")
    elif args.samples:
        sample_size = args.samples
        logger.info(f"Processing {sample_size} servers")
    else:
        sample_size = 100  # Default
        logger.info("Processing 100 servers (default)")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load dataset
    input_file = 'data_unified_filtered.json'
    if not Path(input_file).exists():
        logger.error(f"Input file {input_file} not found")
        return
    
    servers = load_filtered_dataset(input_file)
    
    # Filter for finance-related servers if flag is set
    if args.finance:
        finance_servers = [s for s in servers if s.get('is_finance_related') == True]
        logger.info(f"Filtered to {len(finance_servers)} finance-related servers (from {len(servers)} total)")
        servers = finance_servers
        
        if not servers:
            logger.error("No finance-related servers found")
            return
    
    # Sample servers if needed
    if sample_size is not None:
        servers = sample_servers(servers, sample_size)
    
    logger.info(f"Processing {len(servers)} servers")
    
    # Clean server data
    cleaned_servers = []
    for server in servers:
        cleaned = clean_server_data(server)
        cleaned_servers.append(cleaned)
    
    # Create Stage 1 samples for Inspect
    stage1_samples = []
    for server in cleaned_servers:
        sample = create_stage1_sample(server)
        stage1_samples.append(sample)
    
    # Save cleaned server data
    output_file = 'conseq_fin_servers_sample.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_servers, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(cleaned_servers)} cleaned servers to {output_file}")
    
    # Save Stage 1 input in JSONL format for Inspect
    stage1_file = 'conseq_fin_stage1_input.jsonl'
    with open(stage1_file, 'w', encoding='utf-8') as f:
        for sample in stage1_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(stage1_samples)} Stage 1 samples to {stage1_file}")
    
    # Generate summary statistics
    summary = {
        "total_servers": len(cleaned_servers),
        "sample_size_requested": sample_size,
        "data_sources": {},
        "tools_distribution": {},
        "processing_timestamp": datetime.now().isoformat()
    }
    
    # Data source distribution
    for server in cleaned_servers:
        source = server.get('primary_source', 'unknown')
        summary["data_sources"][source] = summary["data_sources"].get(source, 0) + 1
    
    # Tools distribution
    for server in cleaned_servers:
        tools_count = server.get('tools_count', 0)
        if tools_count == 0:
            category = "no_tools"
        elif tools_count <= 5:
            category = "1-5_tools"
        elif tools_count <= 10:
            category = "6-10_tools"
        else:
            category = "11+_tools"
        summary["tools_distribution"][category] = summary["tools_distribution"].get(category, 0) + 1
    
    # Save summary
    summary_file = 'conseq_fin_data_prep_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved processing summary to {summary_file}")
    
    logger.info("Data preparation completed successfully!")
    logger.info(f"Next step: Run Stage 1 evaluation with: inspect eval conseq_fin_stage1_filter.py --model claude-sonnet-4-20250514")

if __name__ == "__main__":
    main()
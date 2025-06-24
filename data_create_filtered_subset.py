#!/usr/bin/env python3
"""
Create a filtered subset of data_unified.json based on quality criteria:
- If GitHub is the only source: only include repos with stargazers_count >= 1
- If Smithery is the only source: only include repos with use_count >= 1
- Include all repos with multiple sources (they're likely higher quality)
"""

import json
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('filtered_subset_creation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_filtered_subset():
    """Create filtered subset based on quality criteria"""
    
    logger.info("Loading unified dashboard data...")
    with open('data_unified.json', 'r') as f:
        data = json.load(f)
    
    logger.info(f"Original dataset: {len(data)} servers")
    
    filtered_data = []
    stats = {
        'github_only_included': 0,
        'github_only_excluded': 0,
        'smithery_only_included': 0,
        'smithery_only_excluded': 0,
        'multi_source_included': 0,
        'other_cases': 0
    }
    
    for server in data:
        sources = server.get('data_sources', [])
        
        # Case 1: GitHub only - require stargazers >= 1
        if sources == ['github']:
            stargazers = server.get('stargazers_count', 0)
            if stargazers >= 1:
                filtered_data.append(server)
                stats['github_only_included'] += 1
            else:
                stats['github_only_excluded'] += 1
                
        # Case 2: Smithery only - require use_count >= 1  
        elif sources == ['smithery']:
            use_count = server.get('use_count', 0)
            if use_count >= 1:
                filtered_data.append(server)
                stats['smithery_only_included'] += 1
            else:
                stats['smithery_only_excluded'] += 1
                
        # Case 3: Multiple sources - include all (assume higher quality)
        elif len(sources) > 1:
            filtered_data.append(server)
            stats['multi_source_included'] += 1
            
        # Case 4: Other cases (official only, etc.)
        else:
            filtered_data.append(server)
            stats['other_cases'] += 1
    
    logger.info(f"Filtered dataset: {len(filtered_data)} servers")
    logger.info("Filtering statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # Calculate retention rate
    retention_rate = len(filtered_data) / len(data) * 100
    logger.info(f"Retention rate: {retention_rate:.1f}%")
    
    # Save filtered dataset
    output_file = 'data_unified_filtered.json'
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    logger.info(f"Filtered dataset saved to {output_file}")
    
    # Create summary of filtered data
    finance_count = len([s for s in filtered_data if s.get('is_finance_related', False)])
    source_counts = {}
    primary_source_counts = {}
    
    for server in filtered_data:
        for source in server.get('data_sources', []):
            source_counts[source] = source_counts.get(source, 0) + 1
        
        primary = server.get('primary_source')
        if primary:
            primary_source_counts[primary] = primary_source_counts.get(primary, 0) + 1
    
    summary = {
        'total_servers': len(filtered_data),
        'finance_related_servers': finance_count,
        'retention_rate_percent': round(retention_rate, 1),
        'filtering_statistics': stats,
        'source_coverage': source_counts,
        'primary_source_distribution': primary_source_counts,
        'processing_timestamp': '2025-06-20T00:00:00'
    }
    
    summary_file = 'data_unified_filtered_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved to {summary_file}")
    
    # Show some examples of included servers
    logger.info("Sample of included servers:")
    for i, server in enumerate(filtered_data[:5]):
        logger.info(f"  {i+1}. {server.get('name', 'N/A')} ({server.get('data_sources', [])})")
        if server.get('data_sources') == ['github']:
            logger.info(f"     Stars: {server.get('stargazers_count', 0)}")
        elif server.get('data_sources') == ['smithery']:
            logger.info(f"     Use count: {server.get('use_count', 0)}")

if __name__ == "__main__":
    create_filtered_subset()
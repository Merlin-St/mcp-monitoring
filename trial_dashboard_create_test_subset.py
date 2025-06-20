#!/usr/bin/env python3
"""
Create a smaller test subset of the unified MCP data for dashboard testing
"""

import json
import random
from collections import Counter

def create_test_subset():
    # Load full dataset
    with open('dashboard_mcp_servers_unified.json', 'r') as f:
        data = json.load(f)

    print(f'Original dataset: {len(data)} servers')

    # Create subset with good representation
    subset = []

    # Get all finance-related servers (priority)
    finance_servers = [s for s in data if s.get('is_finance_related', False)]
    print(f'Finance servers found: {len(finance_servers)}')

    # Add all finance servers
    subset.extend(finance_servers)

    # Add popular servers (with stars)
    popular_servers = [s for s in data if s.get('stargazers_count', 0) > 5 and not s.get('is_finance_related', False)]
    popular_servers.sort(key=lambda x: x.get('stargazers_count', 0), reverse=True)
    subset.extend(popular_servers[:200])  # Top 200 popular non-finance servers

    # Add some random servers for diversity
    added_ids = {s['id'] for s in subset}
    remaining_servers = [s for s in data if s['id'] not in added_ids]
    random.seed(42)  # For reproducible results
    random_sample = random.sample(remaining_servers, min(300, len(remaining_servers)))
    subset.extend(random_sample)

    print(f'Subset created: {len(subset)} servers')
    print(f'Finance servers: {len([s for s in subset if s.get("is_finance_related", False)])}')
    print(f'Servers with stars: {len([s for s in subset if s.get("stargazers_count", 0) > 0])}')

    # Save subset
    with open('dashboard_mcp_servers_unified_test.json', 'w') as f:
        json.dump(subset, f, indent=2)

    # Create test summary
    finance_count = len([s for s in subset if s.get('is_finance_related', False)])
    source_counts = Counter()
    primary_source_counts = Counter()
    language_counts = Counter()

    for server in subset:
        for source in server.get('data_sources', []):
            source_counts[source] += 1
        
        primary = server.get('primary_source')
        if primary:
            primary_source_counts[primary] += 1
        
        lang = server.get('language')
        if lang:
            language_counts[lang] += 1

    test_summary = {
        'total_servers': len(subset),
        'finance_related_servers': finance_count,
        'source_coverage': dict(source_counts),
        'primary_source_distribution': dict(primary_source_counts),
        'top_languages': dict(language_counts.most_common(10)),
        'top_topics': {},
        'processing_timestamp': '2025-06-16T00:00:00',
        'data_quality': {
            'servers_with_github_data': len([s for s in subset if 'github' in s.get('data_sources', [])]),
            'servers_with_smithery_data': len([s for s in subset if 'smithery' in s.get('data_sources', [])]),
            'servers_with_official_data': len([s for s in subset if 'official' in s.get('data_sources', [])]),
            'servers_with_multiple_sources': len([s for s in subset if len(s.get('data_sources', [])) > 1])
        }
    }

    with open('dashboard_mcp_servers_unified_test_summary.json', 'w') as f:
        json.dump(test_summary, f, indent=2)

    print('Test dataset and summary saved')
    print(f'Test file size: {len(json.dumps(subset))/1024/1024:.1f} MB')

if __name__ == "__main__":
    create_test_subset()
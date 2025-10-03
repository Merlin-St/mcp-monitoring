#!/usr/bin/env python3
"""
Analyze the similarity of O*NET task classifications for tools within the same server.
Computes metrics at each level: Level 1, Level 2, and Task ID.
"""

import logging
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/99_cltools_task_distribution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def calculate_homogeneity(values):
    """Calculate homogeneity score (proportion of most common value)."""
    if len(values) == 0:
        return 0.0
    counter = Counter(values)
    most_common_count = counter.most_common(1)[0][1]
    return most_common_count / len(values)

def calculate_entropy(values):
    """Calculate Shannon entropy (measure of diversity)."""
    if len(values) == 0:
        return 0.0
    counter = Counter(values)
    probs = np.array([count / len(values) for count in counter.values()])
    return -np.sum(probs * np.log2(probs + 1e-10))

def calculate_gini(values):
    """Calculate Gini coefficient (measure of concentration)."""
    if len(values) == 0:
        return 0.0
    counter = Counter(values)
    sorted_counts = sorted(counter.values())
    n = len(values)
    cumsum = np.cumsum(sorted_counts)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

def analyze_server_similarity(df, level_column, level_name):
    """Analyze classification similarity at a specific level."""
    logger.info(f"\nAnalyzing {level_name} similarity...")

    results = []

    # Group by server
    for server_id, group in df.groupby('server_id'):
        if len(group) <= 1:
            continue  # Skip single-tool servers

        values = group[level_column].dropna().tolist()
        if not values:
            continue

        num_tools = len(values)
        num_unique = len(set(values))
        homogeneity = calculate_homogeneity(values)
        entropy = calculate_entropy(values)
        gini = calculate_gini(values)

        results.append({
            'server_id': server_id,
            'server_name': group['server_name'].iloc[0],
            'num_tools': num_tools,
            'num_unique': num_unique,
            'homogeneity': homogeneity,
            'entropy': entropy,
            'gini': gini,
            'most_common': Counter(values).most_common(1)[0][0],
            'most_common_count': Counter(values).most_common(1)[0][1]
        })

    results_df = pd.DataFrame(results)

    # Overall statistics
    stats = {
        'level': level_name,
        'num_servers_analyzed': len(results_df),
        'avg_tools_per_server': results_df['num_tools'].mean(),
        'avg_unique_per_server': results_df['num_unique'].mean(),
        'avg_homogeneity': results_df['homogeneity'].mean(),
        'median_homogeneity': results_df['homogeneity'].median(),
        'avg_entropy': results_df['entropy'].mean(),
        'median_entropy': results_df['entropy'].median(),
        'avg_gini': results_df['gini'].mean(),
        'median_gini': results_df['gini'].median(),
        'perfect_homogeneity_pct': (results_df['homogeneity'] == 1.0).mean() * 100,
        'high_homogeneity_pct': (results_df['homogeneity'] >= 0.8).mean() * 100,
        'low_homogeneity_pct': (results_df['homogeneity'] < 0.5).mean() * 100
    }

    logger.info(f"\n{level_name} Statistics:")
    logger.info(f"  Servers analyzed: {stats['num_servers_analyzed']}")
    logger.info(f"  Avg tools per server: {stats['avg_tools_per_server']:.2f}")
    logger.info(f"  Avg unique classifications: {stats['avg_unique_per_server']:.2f}")
    logger.info(f"  Avg homogeneity: {stats['avg_homogeneity']:.3f} (median: {stats['median_homogeneity']:.3f})")
    logger.info(f"  Perfect homogeneity: {stats['perfect_homogeneity_pct']:.1f}% of servers")
    logger.info(f"  High homogeneity (≥0.8): {stats['high_homogeneity_pct']:.1f}% of servers")
    logger.info(f"  Low homogeneity (<0.5): {stats['low_homogeneity_pct']:.1f}% of servers")
    logger.info(f"  Avg entropy: {stats['avg_entropy']:.3f} (median: {stats['median_entropy']:.3f})")
    logger.info(f"  Avg Gini: {stats['avg_gini']:.3f} (median: {stats['median_gini']:.3f})")

    return results_df, stats

def main():
    """Main analysis pipeline."""
    logger.info("Starting CLTools task distribution analysis...")

    # Load data
    input_file = Path('data/final/cltools_classified.csv')
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return

    logger.info(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} tool records")

    # Count servers with multiple tools
    tools_per_server = df.groupby('server_id').size()
    multi_tool_servers = (tools_per_server > 1).sum()
    logger.info(f"Servers with multiple tools: {multi_tool_servers} / {len(tools_per_server)}")

    # Analyze at each level
    level1_results, level1_stats = analyze_server_similarity(df, 'level1_cluster', 'Level 1')
    level2_results, level2_stats = analyze_server_similarity(df, 'level2_cluster', 'Level 2')
    task_results, task_stats = analyze_server_similarity(df, 'task_id', 'Task ID')

    # Save detailed results
    output_dir = Path('data/internal-analysis')
    output_dir.mkdir(exist_ok=True)

    level1_results.to_csv(output_dir / '99_cltools_task_distribution_level1.csv', index=False)
    level2_results.to_csv(output_dir / '99_cltools_task_distribution_level2.csv', index=False)
    task_results.to_csv(output_dir / '99_cltools_task_distribution_task.csv', index=False)

    logger.info(f"\nDetailed results saved to {output_dir}/")

    # Save summary statistics
    summary = {
        'level1': level1_stats,
        'level2': level2_stats,
        'task': task_stats
    }

    with open(output_dir / '99_cltools_task_distribution_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary statistics saved to {output_dir / '99_cltools_task_distribution_summary.json'}")

    # Print comparison
    logger.info("\n" + "="*80)
    logger.info("COMPARISON ACROSS LEVELS")
    logger.info("="*80)

    comparison_df = pd.DataFrame([level1_stats, level2_stats, task_stats])
    comparison_df = comparison_df[['level', 'avg_homogeneity', 'median_homogeneity',
                                   'perfect_homogeneity_pct', 'high_homogeneity_pct',
                                   'avg_entropy', 'avg_unique_per_server']]

    logger.info("\n" + comparison_df.to_string(index=False))

    logger.info("\n" + "="*80)
    logger.info("INTERPRETATION:")
    logger.info("="*80)
    logger.info("Homogeneity: 1.0 = all tools in server have same classification, 0.0 = maximally diverse")
    logger.info("Entropy: 0.0 = no diversity (all same), higher = more diverse")
    logger.info("Gini: 0.0 = perfect equality, 1.0 = maximum inequality/concentration")
    logger.info("="*80)

if __name__ == '__main__':
    main()
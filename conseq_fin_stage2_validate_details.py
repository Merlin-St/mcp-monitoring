#!/usr/bin/env python3
"""
Finance Consequentiality Stage 2 Validation - Detailed Level Analysis

Deep-dive analysis of consequentiality level disagreements between human and LLM labels.
Analyzes specific cases where disagreements occurred to understand patterns and root causes.
Part of the conseq_fin stage2 validation pipeline.
"""

import pandas as pd
import json
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conseq_fin_stage2_validate_details.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_aligned_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and align the human and LLM labeled datasets."""
    logger.info("Loading datasets...")
    
    # Load datasets
    human_df = pd.read_csv("conseq_fin_stage2_validate_labelled.csv")
    llm_df = pd.read_csv("conseq_fin_stage2.csv", low_memory=False)
    
    # Clean server_id columns
    human_df['server_id'] = human_df['server_id'].astype(str).str.strip()
    llm_df['server_id'] = llm_df['server_id'].astype(str).str.strip()
    
    # Find common IDs and align
    common_ids = set(human_df['server_id'].unique()).intersection(set(llm_df['server_id'].unique()))
    
    human_aligned = human_df[human_df['server_id'].isin(common_ids)].copy()
    llm_aligned = llm_df[llm_df['server_id'].isin(common_ids)].copy()
    
    # Sort for consistent alignment
    human_aligned = human_aligned.sort_values('server_id').reset_index(drop=True)
    llm_aligned = llm_aligned.sort_values('server_id').reset_index(drop=True)
    
    logger.info(f"Aligned datasets: {len(human_aligned)} common records")
    return human_aligned, llm_aligned


def analyze_level_disagreements(human_df: pd.DataFrame, llm_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze specific disagreements in consequentiality level scoring."""
    logger.info("Analyzing consequentiality level disagreements...")
    
    # Extract level columns (note: human has trailing space)
    human_levels = pd.to_numeric(human_df['Level '], errors='coerce')
    llm_levels = pd.to_numeric(llm_df['level'], errors='coerce')
    
    # Find valid comparisons
    valid_mask = ~(human_levels.isna() | llm_levels.isna())
    valid_mask &= (human_levels >= 1) & (human_levels <= 5)
    valid_mask &= (llm_levels >= 1) & (llm_levels <= 5)
    
    # Filter to valid samples
    valid_human = human_df[valid_mask].copy()
    valid_llm = llm_df[valid_mask].copy()
    h_levels = human_levels[valid_mask].values
    l_levels = llm_levels[valid_mask].values
    
    logger.info(f"Valid level comparisons: {len(h_levels)}")
    
    # Find disagreements
    disagreements = []
    agreements = []
    
    for i, (h, l) in enumerate(zip(h_levels, l_levels)):
        # Helper to safely get string values and handle NaN
        def safe_str(val, max_len=300):
            if pd.isna(val) or val is None:
                return 'N/A'
            return str(val)[:max_len]
        
        server_data = {
            'index': i,
            'server_id': valid_human.iloc[i]['server_id'],
            'server_name': safe_str(valid_human.iloc[i].get('Title', 'N/A')),
            'human_level': int(h),
            'llm_level': int(l),
            'difference': int(l) - int(h),
            'human_description': safe_str(valid_human.iloc[i].get('description', ''), 200),
            'human_analysis_notes': safe_str(valid_human.iloc[i].get('analysis_notes', ''), 300),
            'llm_analysis_notes': safe_str(valid_llm.iloc[i].get('analysis_notes', ''), 300),
            'asset_type_human': safe_str(valid_human.iloc[i].get('Asset_type', 'N/A')),
            'asset_type_llm': safe_str(valid_llm.iloc[i].get('asset_type', 'N/A')),
            'is_finance_human': safe_str(valid_human.iloc[i].get('Is_finance', 'N/A')),
            'is_finance_llm': safe_str(valid_llm.iloc[i].get('is_finance_llm', 'N/A'))
        }
        
        if h != l:
            disagreements.append(server_data)
        else:
            agreements.append(server_data)
    
    logger.info(f"Found {len(disagreements)} disagreements and {len(agreements)} agreements")
    
    return {
        'disagreements': disagreements,
        'agreements': agreements,
        'total_valid': len(h_levels),
        'accuracy': len(agreements) / len(h_levels),
        'disagreement_rate': len(disagreements) / len(h_levels)
    }


def analyze_disagreement_patterns(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze patterns in the disagreements."""
    logger.info("Analyzing disagreement patterns...")
    
    disagreements = analysis['disagreements']
    
    # Pattern analysis
    patterns = {
        'by_difference': {},
        'by_human_level': {},
        'by_llm_level': {},
        'by_asset_type': {},
        'systematic_bias': {}
    }
    
    # Count by difference magnitude
    for d in disagreements:
        diff = d['difference']
        patterns['by_difference'][diff] = patterns['by_difference'].get(diff, 0) + 1
        
        h_level = d['human_level']
        l_level = d['llm_level']
        
        patterns['by_human_level'][h_level] = patterns['by_human_level'].get(h_level, 0) + 1
        patterns['by_llm_level'][l_level] = patterns['by_llm_level'].get(l_level, 0) + 1
        
        asset_type = d['asset_type_human']
        if asset_type not in patterns['by_asset_type']:
            patterns['by_asset_type'][asset_type] = {'count': 0, 'examples': []}
        patterns['by_asset_type'][asset_type]['count'] += 1
        if len(patterns['by_asset_type'][asset_type]['examples']) < 3:
            patterns['by_asset_type'][asset_type]['examples'].append(d)
    
    # Systematic bias analysis
    over_estimates = [d for d in disagreements if d['difference'] > 0]
    under_estimates = [d for d in disagreements if d['difference'] < 0]
    
    patterns['systematic_bias'] = {
        'over_estimates': len(over_estimates),
        'under_estimates': len(under_estimates),
        'net_bias': len(over_estimates) - len(under_estimates),
        'avg_over_estimate': sum(d['difference'] for d in over_estimates) / len(over_estimates) if over_estimates else 0,
        'avg_under_estimate': sum(d['difference'] for d in under_estimates) / len(under_estimates) if under_estimates else 0
    }
    
    return patterns


def print_detailed_analysis(analysis: Dict[str, Any], patterns: Dict[str, Any]):
    """Print comprehensive analysis of disagreements."""
    logger.info("=== CONSEQUENTIALITY LEVEL DISAGREEMENT ANALYSIS ===")
    
    total = analysis['total_valid']
    agreements = len(analysis['agreements'])
    disagreements = len(analysis['disagreements'])
    
    logger.info(f"Total valid comparisons: {total}")
    logger.info(f"Exact matches: {agreements} ({agreements/total:.1%})")
    logger.info(f"Disagreements: {disagreements} ({disagreements/total:.1%})")
    
    logger.info("\n=== DISAGREEMENT PATTERNS ===")
    
    # Difference magnitude analysis
    logger.info("Disagreement by magnitude:")
    for diff in sorted(patterns['by_difference'].keys()):
        count = patterns['by_difference'][diff]
        direction = "over-estimated" if diff > 0 else "under-estimated"
        logger.info(f"  LLM {direction} by {abs(diff)} level(s): {count} cases")
    
    # Systematic bias
    bias = patterns['systematic_bias']
    logger.info(f"\nSystematic bias analysis:")
    logger.info(f"  Over-estimates: {bias['over_estimates']}")
    logger.info(f"  Under-estimates: {bias['under_estimates']}")
    logger.info(f"  Net bias: {bias['net_bias']} (positive = LLM tends to over-estimate)")
    logger.info(f"  Avg over-estimate magnitude: {bias['avg_over_estimate']:.2f}")
    logger.info(f"  Avg under-estimate magnitude: {abs(bias['avg_under_estimate']):.2f}")
    
    # Level-specific patterns
    logger.info(f"\nDisagreements by human level:")
    for level in sorted(patterns['by_human_level'].keys()):
        count = patterns['by_human_level'][level]
        logger.info(f"  Human Level {level}: {count} disagreements")
    
    logger.info(f"\nDisagreements by LLM level:")
    for level in sorted(patterns['by_llm_level'].keys()):
        count = patterns['by_llm_level'][level]
        logger.info(f"  LLM Level {level}: {count} disagreements")
    
    # Asset type analysis
    logger.info(f"\nDisagreements by asset type:")
    for asset_type in sorted(patterns['by_asset_type'].keys(), 
                            key=lambda x: patterns['by_asset_type'][x]['count'], 
                            reverse=True):
        count = patterns['by_asset_type'][asset_type]['count']
        logger.info(f"  {asset_type}: {count} disagreements")


def print_specific_examples(analysis: Dict[str, Any]):
    """Print specific examples of disagreements for manual review."""
    logger.info("\n=== SPECIFIC DISAGREEMENT EXAMPLES ===")
    
    disagreements = analysis['disagreements']
    
    # Show examples by category
    categories = {
        'Major over-estimates (LLM > Human by 2+)': [d for d in disagreements if d['difference'] >= 2],
        'Major under-estimates (Human > LLM by 2+)': [d for d in disagreements if d['difference'] <= -2],
        'Minor over-estimates (LLM > Human by 1)': [d for d in disagreements if d['difference'] == 1],
        'Minor under-estimates (Human > LLM by 1)': [d for d in disagreements if d['difference'] == -1]
    }
    
    for category, examples in categories.items():
        if not examples:
            continue
            
        logger.info(f"\n--- {category} ({len(examples)} cases) ---")
        
        # Show up to 3 examples per category
        for i, example in enumerate(examples[:3]):
            logger.info(f"\nExample {i+1}:")
            logger.info(f"  Server: {example['server_name']} ({example['server_id']})")
            logger.info(f"  Human Level: {example['human_level']}, LLM Level: {example['llm_level']}")
            logger.info(f"  Asset Type: Human='{example['asset_type_human']}', LLM='{example['asset_type_llm']}'")
            logger.info(f"  Description: {example['human_description']}")
            if example['human_analysis_notes']:
                logger.info(f"  Human Notes: {example['human_analysis_notes']}")
            if example['llm_analysis_notes']:
                logger.info(f"  LLM Notes: {example['llm_analysis_notes']}")


def save_detailed_results(analysis: Dict[str, Any], patterns: Dict[str, Any]):
    """Save detailed analysis results to JSON."""
    logger.info("Saving detailed disagreement analysis...")
    
    output = {
        'summary': {
            'total_valid_comparisons': analysis['total_valid'],
            'exact_matches': len(analysis['agreements']),
            'disagreements': len(analysis['disagreements']),
            'accuracy': analysis['accuracy'],
            'disagreement_rate': analysis['disagreement_rate']
        },
        'patterns': patterns,
        'disagreement_examples': analysis['disagreements'][:20],  # Save top 20 for file size
        'agreement_examples': analysis['agreements'][:10]  # Save 10 agreements for comparison
    }
    
    with open('conseq_fin_stage2_validate_details_analysis.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info("Detailed analysis saved to conseq_fin_stage2_validate_details_analysis.json")


def main():
    """Main analysis workflow."""
    logger.info("Starting consequentiality level disagreement analysis")
    
    try:
        # Load data
        human_df, llm_df = load_aligned_data()
        
        # Analyze disagreements
        analysis = analyze_level_disagreements(human_df, llm_df)
        patterns = analyze_disagreement_patterns(analysis)
        
        # Print results
        print_detailed_analysis(analysis, patterns)
        print_specific_examples(analysis)
        
        # Save results
        save_detailed_results(analysis, patterns)
        
        logger.info("Disagreement analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
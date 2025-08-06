#!/usr/bin/env python3
"""
Process Level 1 cluster naming results and save them for comparison

This script reads the Inspect evaluation results and extracts the Level 1
cluster names generated based on Level 2 cluster names.

Usage:
    python conseq_fin_stage4_process_level1_names.py
    python conseq_fin_stage4_process_level1_names.py --eval-file specific.eval
"""

import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conseq_fin_stage4_process_level1_names.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_level1_naming_results(eval_file: Path) -> Dict[str, str]:
    """Extract Level 1 cluster names from evaluation results"""
    cluster_names = {}
    
    # Create temporary directory for analysis
    import tempfile
    import shutil
    temp_dir = tempfile.mkdtemp()
    temp_file = Path(temp_dir) / eval_file.name
    
    try:
        shutil.copy2(eval_file, temp_file)
        
        from inspect_ai.analysis.beta import samples_df, messages_df
        
        samples = samples_df(temp_dir)
        messages = messages_df(temp_dir)
        
        for idx, sample_row in samples.iterrows():
            # Get cluster ID from metadata
            cluster_id = sample_row.get('metadata_cluster_id', '')
            
            if cluster_id:
                # Get assistant response (the cluster name)
                sample_id = sample_row.get('sample_id', f'sample_{idx}')
                assistant_msg = messages[(messages['sample_id'] == sample_id) & (messages['role'] == 'assistant')]
                
                if not assistant_msg.empty:
                    cluster_name = assistant_msg.iloc[0]['content'].strip()
                    cluster_names[cluster_id] = cluster_name
                    logger.debug(f"{cluster_id}: {cluster_name}")
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return cluster_names

def save_level1_names(cluster_names: Dict[str, str]):
    """Save Level 1 cluster names and create comparison data"""
    
    # Save as JSON
    output_json = 'conseq_fin_stage4_level1_names_from_l2.json'
    with open(output_json, 'w') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'method': 'Generated from Level 2 cluster names',
            'cluster_names': cluster_names
        }, f, indent=2)
    logger.info(f"Saved Level 1 names to {output_json}")
    
    # Create CSV for easy comparison
    names_df = pd.DataFrame([
        {'cluster_id': k, 'cluster_name_from_l2': v} 
        for k, v in sorted(cluster_names.items())
    ])
    
    # Load existing Level 1 names if available
    metadata_file = 'conseq_fin_stage4_hierarchy_k12_summary.json'
    if Path(metadata_file).exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        if 'cluster_names' in metadata and 'level1' in metadata['cluster_names']:
            existing_names = metadata['cluster_names']['level1']
            for idx, row in names_df.iterrows():
                cluster_id = row['cluster_id']
                if cluster_id in existing_names:
                    names_df.at[idx, 'existing_name'] = existing_names[cluster_id]
    
    # Save comparison CSV
    comparison_csv = 'conseq_fin_stage4_level1_names_comparison.csv'
    names_df.to_csv(comparison_csv, index=False)
    logger.info(f"Saved comparison to {comparison_csv}")
    
    return names_df

def main():
    parser = argparse.ArgumentParser(description='Process Level 1 naming results')
    parser.add_argument('--eval-file', type=str,
                       help='Specific eval file to process')
    parser.add_argument('--logs-dir', type=str, default='logs',
                       help='Directory containing .eval files')
    
    args = parser.parse_args()
    
    logger.info("Processing Level 1 cluster naming results...")
    
    # Find eval file
    if args.eval_file:
        eval_file = Path(args.eval_file)
    else:
        log_dir = Path(args.logs_dir)
        eval_files = list(log_dir.glob('*level1-naming-task*.eval'))
        if not eval_files:
            logger.error("No Level 1 naming evaluation files found")
            return
        eval_file = max(eval_files, key=lambda x: x.stat().st_mtime)
    
    logger.info(f"Processing: {eval_file.name}")
    
    # Process results
    cluster_names = process_level1_naming_results(eval_file)
    logger.info(f"Extracted {len(cluster_names)} Level 1 cluster names")
    
    if cluster_names:
        # Save and display results
        comparison_df = save_level1_names(cluster_names)
        
        # Show results
        logger.info("\nGenerated Level 1 cluster names (based on Level 2 names):")
        for cluster_id, name in sorted(cluster_names.items()):
            logger.info(f"  {cluster_id}: {name}")
        
        if 'existing_name' in comparison_df.columns:
            logger.info("\nComparison with existing names:")
            for _, row in comparison_df.iterrows():
                logger.info(f"\n{row['cluster_id']}:")
                logger.info(f"  From L2: {row['cluster_name_from_l2']}")
                if pd.notna(row.get('existing_name')):
                    logger.info(f"  Existing: {row['existing_name']}")
    else:
        logger.warning("No cluster names found in evaluation results")
    
    logger.info("\nProcessing complete!")

if __name__ == "__main__":
    main()
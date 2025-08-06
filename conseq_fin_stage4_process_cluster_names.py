#!/usr/bin/env python3
"""
Process cluster naming results and update hierarchy metadata

This script reads the Inspect evaluation results and updates the hierarchy
metadata JSON with descriptive names for each Level 2 cluster.

Usage:
    python conseq_fin_stage4_process_cluster_names.py
    python conseq_fin_stage4_process_cluster_names.py --eval-file specific.eval
"""

import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conseq_fin_stage4_process_cluster_names.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_naming_results(eval_file: Path) -> Dict[str, str]:
    """Extract cluster names from evaluation results"""
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

def update_hierarchy_metadata(cluster_names: Dict[str, str]):
    """Update hierarchy metadata with cluster names"""
    metadata_file = 'conseq_fin_stage4_hierarchy_k12_summary.json'
    
    # Load existing metadata
    if Path(metadata_file).exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Initialize cluster names section if not exists
    if 'cluster_names' not in metadata:
        metadata['cluster_names'] = {
            'level1': {},
            'level2': {}
        }
    
    # Update Level 2 cluster names
    metadata['cluster_names']['level2'].update(cluster_names)
    metadata['cluster_names_generated_at'] = datetime.now().isoformat()
    metadata['cluster_names_count'] = len(cluster_names)
    
    # Save updated metadata
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Updated {metadata_file} with {len(cluster_names)} cluster names")
    
    # Also create a simple CSV mapping for easy reference
    import pandas as pd
    names_df = pd.DataFrame([
        {'cluster_id': k, 'cluster_name': v} 
        for k, v in sorted(cluster_names.items())
    ])
    names_csv = 'conseq_fin_stage4_cluster_names.csv'
    names_df.to_csv(names_csv, index=False)
    logger.info(f"Saved cluster names to {names_csv}")

def main():
    parser = argparse.ArgumentParser(description='Process cluster naming results')
    parser.add_argument('--eval-file', type=str,
                       help='Specific eval file to process')
    parser.add_argument('--logs-dir', type=str, default='logs',
                       help='Directory containing .eval files')
    
    args = parser.parse_args()
    
    logger.info("Processing cluster naming results...")
    
    # Find eval file
    if args.eval_file:
        eval_file = Path(args.eval_file)
    else:
        log_dir = Path(args.logs_dir)
        eval_files = list(log_dir.glob('*cluster-naming-task*.eval'))
        if not eval_files:
            logger.error("No cluster naming evaluation files found")
            return
        eval_file = max(eval_files, key=lambda x: x.stat().st_mtime)
    
    logger.info(f"Processing: {eval_file.name}")
    
    # Process results
    cluster_names = process_naming_results(eval_file)
    logger.info(f"Extracted {len(cluster_names)} cluster names")
    
    if cluster_names:
        # Update metadata
        update_hierarchy_metadata(cluster_names)
        
        # Show sample results
        logger.info("\nSample cluster names:")
        for cluster_id, name in list(cluster_names.items())[:10]:
            logger.info(f"  {cluster_id}: {name}")
    else:
        logger.warning("No cluster names found in evaluation results")
    
    logger.info("\nProcessing complete!")

if __name__ == "__main__":
    main()
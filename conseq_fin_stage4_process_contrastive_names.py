#!/usr/bin/env python3
"""
Process contrastive cluster naming results

This script processes the results from the contrastive naming approach
and creates validation datasets using these more distinctive names.

Usage:
    python conseq_fin_stage4_process_contrastive_names.py
    python conseq_fin_stage4_process_contrastive_names.py --source hierarchical
"""

import json
import argparse
import logging
import re
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conseq_fin_stage4_process_contrastive_names.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_naming_results(eval_file: Path, naming_type: str = 'contrastive') -> Dict[str, str]:
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

def create_validation_datasets_with_new_names(l1_names: Dict[str, str], l2_names: Dict[str, str], suffix: str):
    """Create new validation datasets using the distinctive names"""
    
    # Load original validation datasets
    validation_files = [
        'conseq_fin_stage4_validation_l3_to_l1.jsonl',
        'conseq_fin_stage4_validation_l2_to_l1.jsonl',
        'conseq_fin_stage4_validation_l3_to_l2.jsonl'
    ]
    
    for val_file in validation_files:
        if not Path(val_file).exists():
            logger.warning(f"Validation file not found: {val_file}")
            continue
        
        # Load samples
        samples = []
        with open(val_file, 'r') as f:
            for line in f:
                samples.append(json.loads(line))
        
        # Update samples with new names
        updated_samples = []
        for sample in samples:
            input_text = sample['input']
            
            # Replace Level 1 names
            for l1_id, l1_name in l1_names.items():
                # Find and replace in options
                old_pattern = f"{l1_id}: [^\\n]+"
                new_text = f"{l1_id}: {l1_name}"
                input_text = re.sub(old_pattern, new_text, input_text)
            
            # Replace Level 2 names
            for l2_id, l2_name in l2_names.items():
                old_pattern = f"{l2_id}: [^\\n]+"
                new_text = f"{l2_id}: {l2_name}"
                input_text = re.sub(old_pattern, new_text, input_text)
            
            # Create updated sample
            updated_sample = sample.copy()
            updated_sample['input'] = input_text
            updated_samples.append(updated_sample)
        
        # Save updated dataset
        output_file = val_file.replace('.jsonl', f'_{suffix}.jsonl')
        with open(output_file, 'w') as f:
            for sample in updated_samples:
                f.write(json.dumps(sample) + '\n')
        
        logger.info(f"Created {output_file} with {len(updated_samples)} samples")

def main():
    parser = argparse.ArgumentParser(description='Process contrastive naming results')
    parser.add_argument('--source', choices=['contrastive', 'hierarchical', 'distinctive'], 
                       default='contrastive',
                       help='Which naming approach to process')
    parser.add_argument('--logs-dir', type=str, default='logs',
                       help='Directory containing .eval files')
    
    args = parser.parse_args()
    
    logger.info(f"Processing {args.source} naming results...")
    
    l1_names = {}
    l2_names = {}
    
    # Find appropriate eval files based on source
    log_dir = Path(args.logs_dir)
    
    if args.source == 'contrastive':
        # Process Level 2 contrastive names
        l2_eval_files = list(log_dir.glob('*cluster-naming-contrastive*.eval'))
        if l2_eval_files:
            eval_file = max(l2_eval_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Processing L2 contrastive: {eval_file.name}")
            l2_names = process_naming_results(eval_file, 'contrastive')
    
    elif args.source == 'hierarchical':
        # Process Level 1 hierarchical names
        l1_eval_files = list(log_dir.glob('*level1-naming-hierarchical*.eval'))
        if l1_eval_files:
            eval_file = max(l1_eval_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Processing L1 hierarchical: {eval_file.name}")
            l1_names = process_naming_results(eval_file, 'hierarchical')
        
        # Also load existing L2 names
        if Path('conseq_fin_stage4_cluster_names_contrastive.csv').exists():
            df = pd.read_csv('conseq_fin_stage4_cluster_names_contrastive.csv')
            l2_names = dict(zip(df['cluster_id'], df['cluster_name']))
    
    elif args.source == 'distinctive':
        # Load from the combined distinctive names file
        dist_file = 'conseq_fin_stage4_distinctive_names.json'
        if Path(dist_file).exists():
            with open(dist_file, 'r') as f:
                data = json.load(f)
                l1_names = data.get('level1_names', {})
                l2_names = data.get('level2_names', {})
    
    # Save processed names
    if l2_names:
        l2_df = pd.DataFrame([
            {'cluster_id': k, 'cluster_name': v} 
            for k, v in sorted(l2_names.items())
        ])
        output_csv = f'conseq_fin_stage4_cluster_names_{args.source}.csv'
        l2_df.to_csv(output_csv, index=False)
        logger.info(f"Saved {len(l2_names)} Level 2 names to {output_csv}")
    
    if l1_names:
        output_json = f'conseq_fin_stage4_level1_names_{args.source}.json'
        with open(output_json, 'w') as f:
            json.dump({
                'generated_at': datetime.now().isoformat(),
                'method': f'{args.source}_generation',
                'cluster_names': l1_names
            }, f, indent=2)
        logger.info(f"Saved {len(l1_names)} Level 1 names to {output_json}")
    
    # Create validation datasets with new names
    if l1_names or l2_names:
        logger.info("Creating validation datasets with new names...")
        create_validation_datasets_with_new_names(l1_names, l2_names, args.source)
    
    logger.info("Processing complete!")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Process Level 1 Supercluster Names v2

Process Inspect evaluation results to extract Level 1 supercluster names
and update the hierarchy file.
"""

import json
import logging
import re
import pandas as pd
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conseq_fin_stage4_process_level1_names_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def find_latest_eval_file(pattern="*generate-level1-names*"):
    """Find the most recent evaluation file."""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        logs_dir = Path(".")
    
    eval_files = list(logs_dir.glob(f"*{pattern}*.eval"))
    if not eval_files:
        logger.error(f"No evaluation files found matching pattern: {pattern}")
        return None
    
    # Get the most recent file
    latest_file = max(eval_files, key=lambda f: f.stat().st_mtime)
    logger.info(f"Using evaluation file: {latest_file}")
    return latest_file

def extract_names_from_eval(eval_file):
    """Extract Level 1 supercluster names from evaluation results."""
    logger.info(f"Processing evaluation file: {eval_file}")
    
    names = {}
    
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
            # Get Level 1 ID from metadata or id field
            level1_id = sample_row.get('metadata_level1_id')
            if not level1_id:
                level1_id = sample_row.get('id')
            if not level1_id:
                level1_id = f'L1_{idx:02d}'
            
            if level1_id:
                # Get assistant response (the supercluster name)
                sample_id = sample_row.get('sample_id', f'sample_{idx}')
                assistant_msg = messages[(messages['sample_id'] == sample_id) & (messages['role'] == 'assistant')]
                
                if not assistant_msg.empty:
                    supercluster_name = assistant_msg.iloc[0]['content'].strip()
                    
                    # Clean up the name (remove quotes, extra whitespace, etc.)
                    supercluster_name = re.sub(r'^["\']|["\']$', '', supercluster_name)
                    supercluster_name = re.sub(r'\s+', ' ', supercluster_name).strip()
                    
                    names[level1_id] = supercluster_name
                    logger.info(f"  {level1_id}: {supercluster_name}")
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    logger.info(f"Extracted {len(names)} Level 1 supercluster names")
    return names

def update_hierarchy_with_names(names):
    """Update the hierarchy file with generated names."""
    logger.info("Updating hierarchy with generated names...")
    
    # Load current hierarchy
    with open('conseq_fin_stage4_hierarchy_k12.json', 'r') as f:
        hierarchy = json.load(f)
    
    # Update Level 1 supercluster names
    updated_count = 0
    for level1_id, name in names.items():
        if level1_id in hierarchy["level1_superclusters"]:
            hierarchy["level1_superclusters"][level1_id] = name
            updated_count += 1
            logger.info(f"Updated {level1_id}: {name}")
    
    # Update the level2_to_level1_mapping with new names
    for cluster_id, mapping in hierarchy["level2_to_level1_mapping"].items():
        l1_id = mapping["level1_id"]
        if l1_id in names:
            mapping["level1_name"] = names[l1_id]
    
    # Update metadata
    hierarchy["metadata"]["names_generated_at"] = datetime.now().isoformat()
    hierarchy["metadata"]["names_updated_count"] = updated_count
    
    # Save updated hierarchy
    with open('conseq_fin_stage4_hierarchy_k12.json', 'w') as f:
        json.dump(hierarchy, f, indent=2)
    
    logger.info(f"Updated hierarchy file with {updated_count} new names")
    
    # Create summary
    summary = {
        "updated_at": datetime.now().isoformat(),
        "total_superclusters": len(hierarchy["level1_superclusters"]),
        "names_updated": updated_count,
        "supercluster_names": dict(sorted(hierarchy["level1_superclusters"].items()))
    }
    
    with open('conseq_fin_stage4_hierarchy_k12_names_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return hierarchy

def create_readable_summary(hierarchy):
    """Create a readable summary of the Level 1 superclusters."""
    logger.info("Creating readable summary...")
    
    summary_lines = [
        "Level 1 Superclusters (K=20) - Generated from Level 2 Cluster Name Embeddings",
        "=" * 80,
        ""
    ]
    
    # Group Level 2 clusters by Level 1 for counting
    level1_sizes = {}
    for mapping in hierarchy["level2_to_level1_mapping"].values():
        l1_id = mapping["level1_id"]
        level1_sizes[l1_id] = level1_sizes.get(l1_id, 0) + 1
    
    # Sort by Level 1 ID
    for l1_id in sorted(hierarchy["level1_superclusters"].keys()):
        name = hierarchy["level1_superclusters"][l1_id]
        size = level1_sizes.get(l1_id, 0)
        
        summary_lines.append(f"{l1_id}: {name} ({size} Level 2 clusters)")
    
    summary_lines.extend([
        "",
        f"Total: {len(hierarchy['level1_superclusters'])} Level 1 superclusters",
        f"Covering: {len(hierarchy['level2_to_level1_mapping'])} Level 2 clusters",
        f"Silhouette Score: {hierarchy['metadata']['silhouette_score']:.3f}",
        f"Generated: {hierarchy['metadata']['created_at']}"
    ])
    
    summary_text = "\n".join(summary_lines)
    
    with open('conseq_fin_stage4_hierarchy_k12_readable_summary.txt', 'w') as f:
        f.write(summary_text)
    
    logger.info("Created readable summary file")
    
    # Also log the summary
    for line in summary_lines:
        logger.info(line)

def update_validation_datasets(names):
    """Update existing validation datasets with new Level 1 names."""
    logger.info("Updating validation datasets with new Level 1 names...")
    
    # Load current hierarchy to get L2 → L1 mapping
    with open('conseq_fin_stage4_hierarchy_k12.json', 'r') as f:
        hierarchy = json.load(f)
    
    # Create mapping from old cluster IDs to new Level 1 IDs and names
    l2_to_new_l1 = {}
    for cluster_id, mapping in hierarchy["level2_to_level1_mapping"].items():
        l1_id = mapping["level1_id"]
        l2_to_new_l1[cluster_id] = {
            "new_l1_id": l1_id,
            "new_l1_name": names.get(l1_id, f"Supercluster {l1_id}")
        }
    
    # Update L2 → L1 validation datasets
    validation_files = [
        'conseq_fin_stage4_validation_l2_to_l1.jsonl',
        'conseq_fin_stage4_validation_l2_to_l1_contrastive.jsonl',
        'conseq_fin_stage4_validation_l2_to_l1_hierarchical.jsonl'
    ]
    
    for file_path in validation_files:
        if Path(file_path).exists():
            logger.info(f"Updating {file_path}...")
            
            updated_samples = []
            with open(file_path, 'r') as f:
                for line in f:
                    sample = json.loads(line.strip())
                    
                    # Get L2 cluster ID from metadata
                    l2_cluster_id = sample.get('metadata', {}).get('l2_cluster_id', '')
                    
                    if l2_cluster_id in l2_to_new_l1:
                        # Add new Level 1 mapping to metadata
                        sample['metadata']['new_l1_id'] = l2_to_new_l1[l2_cluster_id]['new_l1_id']
                        sample['metadata']['new_l1_name'] = l2_to_new_l1[l2_cluster_id]['new_l1_name']
                    
                    updated_samples.append(sample)
            
            # Save updated file with _v2 suffix
            new_file_path = file_path.replace('.jsonl', '_v2.jsonl')
            with open(new_file_path, 'w') as f:
                for sample in updated_samples:
                    f.write(json.dumps(sample) + '\n')
            
            logger.info(f"Saved updated dataset to {new_file_path}")
    
    # Update L3 → L1 validation datasets (they also need L2 → L1 mapping)
    l3_l1_files = [
        'conseq_fin_stage4_validation_l3_to_l1.jsonl',
        'conseq_fin_stage4_validation_l3_to_l1_contrastive.jsonl',
        'conseq_fin_stage4_validation_l3_to_l1_hierarchical.jsonl'
    ]
    
    for file_path in l3_l1_files:
        if Path(file_path).exists():
            logger.info(f"Updating {file_path}...")
            
            updated_samples = []
            with open(file_path, 'r') as f:
                for line in f:
                    sample = json.loads(line.strip())
                    
                    # Get L2 cluster ID from metadata (L3 tasks have L2 mapping)
                    l2_cluster_id = sample.get('metadata', {}).get('l2_cluster_id', '')
                    
                    if l2_cluster_id in l2_to_new_l1:
                        # Add new Level 1 mapping to metadata
                        sample['metadata']['new_l1_id'] = l2_to_new_l1[l2_cluster_id]['new_l1_id']
                        sample['metadata']['new_l1_name'] = l2_to_new_l1[l2_cluster_id]['new_l1_name']
                    
                    updated_samples.append(sample)
            
            # Save updated file with _v2 suffix
            new_file_path = file_path.replace('.jsonl', '_v2.jsonl')
            with open(new_file_path, 'w') as f:
                for sample in updated_samples:
                    f.write(json.dumps(sample) + '\n')
            
            logger.info(f"Saved updated dataset to {new_file_path}")
    
    logger.info("Validation datasets updated successfully!")

def main():
    """Main processing function."""
    logger.info("Starting Level 1 supercluster names processing...")
    
    try:
        # Find the latest evaluation file
        eval_file = find_latest_eval_file("generate-level1-names")
        if not eval_file:
            logger.error("No evaluation file found. Run the inspect eval first.")
            return
        
        # Extract names from evaluation results
        names = extract_names_from_eval(eval_file)
        
        if not names:
            logger.error("No names were extracted from the evaluation file.")
            return
        
        # Update hierarchy with names
        hierarchy = update_hierarchy_with_names(names)
        
        # Create readable summary
        create_readable_summary(hierarchy)
        
        # Update validation datasets
        update_validation_datasets(names)
        
        logger.info("Level 1 supercluster names processing complete!")
        
    except Exception as e:
        logger.error(f"Error in main processing: {e}")
        raise

if __name__ == "__main__":
    main()
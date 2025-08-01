#!/usr/bin/env python3
"""
Create V2 Validation Datasets for New K=20 Hierarchy

Transform existing validation datasets to use the new Level 1 structure
with K=20 superclusters and updated targets.
"""

import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conseq_fin_stage4_create_v2_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_new_hierarchy():
    """Load the new K=12 hierarchy mapping."""
    with open('conseq_fin_stage4_hierarchy_k12.json', 'r') as f:
        hierarchy = json.load(f)
    return hierarchy

def create_new_l1_options(hierarchy):
    """Create the new Level 1 options text for prompts."""
    l1_options = []
    for l1_id in sorted(hierarchy["level1_superclusters"].keys()):
        l1_name = hierarchy["level1_superclusters"][l1_id]
        l1_options.append(f"- {l1_id}: {l1_name}")
    
    return "\n".join(l1_options)

def update_validation_dataset(input_file, output_file, hierarchy):
    """Update a validation dataset with new Level 1 structure."""
    logger.info(f"Updating {input_file} → {output_file}")
    
    l1_options_text = create_new_l1_options(hierarchy)
    
    # Create mapping from L2 cluster to new L1
    l2_to_new_l1 = {}
    for cluster_id, mapping in hierarchy["level2_to_level1_mapping"].items():
        l2_to_new_l1[cluster_id] = {
            "new_l1_id": mapping["level1_id"],
            "new_l1_name": mapping["level1_name"]
        }
    
    updated_samples = []
    samples_processed = 0
    samples_updated = 0
    
    with open(input_file, 'r') as f:
        for line in f:
            sample = json.loads(line.strip())
            samples_processed += 1
            
            # Get L2 cluster ID (try multiple fields for different validation types)
            l2_cluster_id = sample.get('metadata', {}).get('l2_cluster_id', '')
            if not l2_cluster_id:
                # For L3→L1 tasks, try correct_l2 field
                l2_cluster_id = sample.get('metadata', {}).get('correct_l2', '')
            
            if l2_cluster_id in l2_to_new_l1:
                # Update the target to new L1 ID
                new_l1_id = l2_to_new_l1[l2_cluster_id]["new_l1_id"]
                new_l1_name = l2_to_new_l1[l2_cluster_id]["new_l1_name"]
                
                sample['target'] = new_l1_id
                
                # Update metadata
                sample['metadata']['new_l1_id'] = new_l1_id
                sample['metadata']['new_l1_name'] = new_l1_name
                
                # Update the input prompt to use new Level 1 options
                input_text = sample['input']
                
                # Find where the old options start (try different patterns)
                options_start = input_text.find("Select the Level 1 parent cluster from the following options:")
                if options_start == -1:
                    options_start = input_text.find("Select the Level 1 cluster this task belongs to from the following options:")
                
                if options_start != -1:
                    # Replace everything from "Select..." onwards with new options
                    new_input = input_text[:options_start] + f"""Select the Level 1 supercluster this task belongs to from the following options:
{l1_options_text}"""
                    sample['input'] = new_input
                    samples_updated += 1
                else:
                    logger.warning(f"Could not find options section in sample: {l2_cluster_id}")
            
            updated_samples.append(sample)
    
    # Save updated dataset
    with open(output_file, 'w') as f:
        for sample in updated_samples:
            f.write(json.dumps(sample) + '\n')
    
    logger.info(f"Processed {samples_processed} samples, updated {samples_updated} samples")
    return samples_processed, samples_updated

def main():
    """Main function to create all v2 validation datasets."""
    logger.info("Creating K=12 validation datasets for new hierarchy...")
    
    # Load new hierarchy
    hierarchy = load_new_hierarchy()
    
    # Define input/output file pairs
    file_pairs = [
        ('conseq_fin_stage4_validation_l2_to_l1_v2.jsonl', 'conseq_fin_stage4_validation_l2_to_l1_k12.jsonl'),
        ('conseq_fin_stage4_validation_l3_to_l1_v2.jsonl', 'conseq_fin_stage4_validation_l3_to_l1_k12.jsonl'),
    ]
    
    total_processed = 0
    total_updated = 0
    
    for input_file, output_file in file_pairs:
        if Path(input_file).exists():
            processed, updated = update_validation_dataset(input_file, output_file, hierarchy)
            total_processed += processed
            total_updated += updated
        else:
            logger.warning(f"Input file not found: {input_file}")
    
    logger.info(f"K=12 validation dataset creation complete!")
    logger.info(f"Total samples processed: {total_processed}")
    logger.info(f"Total samples updated: {total_updated}")

if __name__ == "__main__":
    main()
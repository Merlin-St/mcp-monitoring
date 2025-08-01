#!/usr/bin/env python3
"""
Prepare validation datasets for cluster assignment accuracy testing

This script creates stratified samples for three validation scenarios:
1. Level 3->2: Task to Level 2 cluster assignment
2. Level 2->1: Level 2 cluster to Level 1 parent assignment  
3. Level 3->1: Task to Level 1 cluster direct assignment

Usage:
    python conseq_fin_stage4_validation_prep.py
    python conseq_fin_stage4_validation_prep.py --samples-per-cluster 5
"""

import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conseq_fin_stage4_validation_prep.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ValidationDataPrep:
    def __init__(self, samples_per_cluster: int = 3):
        self.samples_per_cluster = samples_per_cluster
        self.data_loaded = False
        
    def load_data(self):
        """Load cluster data and names"""
        logger.info("Loading cluster data...")
        
        # Load main cluster assignments
        self.df = pd.read_csv('conseq_fin_stage4_onetclusters.csv')
        logger.info(f"Loaded {len(self.df)} tasks")
        
        # Load cluster names
        self.l2_names = {}
        l2_names_df = pd.read_csv('conseq_fin_stage4_cluster_names.csv')
        self.l2_names = dict(zip(l2_names_df['cluster_id'], l2_names_df['cluster_name']))
        
        # Load Level 1 names from metadata
        metadata_file = 'conseq_fin_stage4_hierarchy_metadata.json'
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        self.l1_names = metadata.get('cluster_names', {}).get('level1', {})
        
        # If no Level 1 names in metadata, use the generated ones
        if not self.l1_names:
            l1_comparison_file = 'conseq_fin_stage4_level1_names_comparison.csv'
            if Path(l1_comparison_file).exists():
                l1_df = pd.read_csv(l1_comparison_file)
                # Use existing names if available, otherwise use generated
                for _, row in l1_df.iterrows():
                    cluster_id = row['cluster_id']
                    name = row.get('existing_name', row.get('cluster_name_from_l2', f"Level 1 Cluster {cluster_id}"))
                    if pd.notna(name):
                        self.l1_names[cluster_id] = name
        
        self.data_loaded = True
        logger.info(f"Loaded {len(self.l2_names)} Level 2 names and {len(self.l1_names)} Level 1 names")
        
    def create_l3_to_l2_samples(self) -> List[Dict[str, Any]]:
        """Create samples for Level 3 (task) to Level 2 cluster validation"""
        logger.info("Creating Level 3->2 validation samples...")
        
        samples = []
        
        # Sample tasks from each Level 2 cluster
        for l2_cluster in sorted(self.df['level2_cluster_id'].unique()):
            cluster_tasks = self.df[self.df['level2_cluster_id'] == l2_cluster]
            
            # Sample up to samples_per_cluster tasks
            n_samples = min(self.samples_per_cluster, len(cluster_tasks))
            sampled_tasks = cluster_tasks.sample(n=n_samples, random_state=42)
            
            for _, task in sampled_tasks.iterrows():
                # Get the correct answer
                correct_l2 = task['level2_cluster_id']
                correct_l1 = task['level1_cluster_id']
                
                # Create distractor options (other L2 clusters)
                # Include some from same L1 (harder) and some from different L1 (easier)
                same_l1_clusters = self.df[
                    (self.df['level1_cluster_id'] == correct_l1) & 
                    (self.df['level2_cluster_id'] != correct_l2)
                ]['level2_cluster_id'].unique()
                
                diff_l1_clusters = self.df[
                    self.df['level1_cluster_id'] != correct_l1
                ]['level2_cluster_id'].unique()
                
                # Select distractors
                n_same_l1 = min(4, len(same_l1_clusters))
                n_diff_l1 = 15 - n_same_l1  # Total 16 options including correct
                
                distractors = []
                if n_same_l1 > 0:
                    distractors.extend(np.random.choice(same_l1_clusters, n_same_l1, replace=False))
                if n_diff_l1 > 0:
                    distractors.extend(np.random.choice(diff_l1_clusters, n_diff_l1, replace=False))
                
                # Create option list
                options = [correct_l2] + list(distractors)
                np.random.shuffle(options)
                
                # Format options with names
                options_formatted = []
                for opt in options:
                    name = self.l2_names.get(opt, "Unknown")
                    options_formatted.append(f"{opt}: {name}")
                
                # Format input for inspect
                input_text = f"""Task: {task['Task']}

Select the Level 2 cluster this task belongs to from the following options:
{chr(10).join(f"- {opt}" for opt in options_formatted)}"""
                
                sample = {
                    'input': input_text,
                    'target': correct_l2,  # For includes() scorer
                    'metadata': {
                        'task_id': task['task_id'],
                        'correct_l2': correct_l2,
                        'correct_l2_name': self.l2_names.get(correct_l2, "Unknown"),
                        'correct_l1': correct_l1,
                        'validation_type': 'l3_to_l2'
                    }
                }
                samples.append(sample)
        
        logger.info(f"Created {len(samples)} Level 3->2 validation samples")
        return samples
    
    def create_l2_to_l1_samples(self) -> List[Dict[str, Any]]:
        """Create samples for Level 2 cluster to Level 1 parent validation"""
        logger.info("Creating Level 2->1 validation samples...")
        
        samples = []
        
        # Get unique L2-L1 mappings
        l2_to_l1 = self.df[['level2_cluster_id', 'level1_cluster_id']].drop_duplicates()
        
        # Sample Level 2 clusters
        n_samples = min(len(l2_to_l1), self.samples_per_cluster * 20)  # More samples since fewer L1 clusters
        sampled_l2 = l2_to_l1.sample(n=n_samples, random_state=42)
        
        # Get all Level 1 options
        all_l1_options = sorted(self.df['level1_cluster_id'].unique())
        
        for _, row in sampled_l2.iterrows():
            l2_cluster = row['level2_cluster_id']
            correct_l1 = row['level1_cluster_id']
            
            # Get sample tasks from this L2 cluster for context
            l2_tasks = self.df[self.df['level2_cluster_id'] == l2_cluster].sample(
                n=min(5, len(self.df[self.df['level2_cluster_id'] == l2_cluster])),
                random_state=42
            )['Task'].tolist()
            
            # Format all L1 options with names
            options_formatted = []
            for opt in all_l1_options:
                name = self.l1_names.get(opt, f"Level 1 Cluster {opt}")
                options_formatted.append(f"{opt}: {name}")
            
            # Format input for inspect
            input_text = f"""Level 2 Cluster: {l2_cluster} - {self.l2_names.get(l2_cluster, "Unknown")}

Sample tasks from this cluster:
{chr(10).join(f"- {task}" for task in l2_tasks)}

Select the Level 1 parent cluster from the following options:
{chr(10).join(f"- {opt}" for opt in options_formatted)}"""
            
            sample = {
                'input': input_text,
                'target': correct_l1,  # For includes() scorer
                'metadata': {
                    'l2_cluster_id': l2_cluster,
                    'l2_cluster_name': self.l2_names.get(l2_cluster, "Unknown"),
                    'correct_l1': correct_l1,
                    'correct_l1_name': self.l1_names.get(correct_l1, f"Level 1 Cluster {correct_l1}"),
                    'validation_type': 'l2_to_l1'
                }
            }
            samples.append(sample)
        
        logger.info(f"Created {len(samples)} Level 2->1 validation samples")
        return samples
    
    def create_l3_to_l1_samples(self) -> List[Dict[str, Any]]:
        """Create samples for Level 3 (task) to Level 1 cluster direct validation"""
        logger.info("Creating Level 3->1 validation samples...")
        
        samples = []
        
        # Sample tasks from each Level 1 cluster
        for l1_cluster in sorted(self.df['level1_cluster_id'].unique()):
            cluster_tasks = self.df[self.df['level1_cluster_id'] == l1_cluster]
            
            # Sample more tasks per L1 cluster since there are only 10
            n_samples = min(self.samples_per_cluster * 5, len(cluster_tasks))
            sampled_tasks = cluster_tasks.sample(n=n_samples, random_state=42)
            
            # Get all Level 1 options
            all_l1_options = sorted(self.df['level1_cluster_id'].unique())
            
            for _, task in sampled_tasks.iterrows():
                # Format options with names
                options_formatted = []
                for opt in all_l1_options:
                    name = self.l1_names.get(opt, f"Level 1 Cluster {opt}")
                    options_formatted.append(f"{opt}: {name}")
                
                # Format input for inspect
                input_text = f"""Task: {task['Task']}

Select the Level 1 cluster this task belongs to from the following options:
{chr(10).join(f"- {opt}" for opt in options_formatted)}"""
                
                sample = {
                    'input': input_text,
                    'target': task['level1_cluster_id'],  # For includes() scorer
                    'metadata': {
                        'task_id': task['task_id'],
                        'correct_l1': task['level1_cluster_id'],
                        'correct_l1_name': self.l1_names.get(task['level1_cluster_id'], f"Level 1 Cluster {task['level1_cluster_id']}"),
                        'correct_l2': task['level2_cluster_id'],  # For analysis
                        'validation_type': 'l3_to_l1'
                    }
                }
                samples.append(sample)
        
        logger.info(f"Created {len(samples)} Level 3->1 validation samples")
        return samples
    
    def save_samples(self, samples: List[Dict[str, Any]], output_file: str):
        """Save samples as JSONL for inspect evaluation"""
        with open(output_file, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        logger.info(f"Saved {len(samples)} samples to {output_file}")
    
    def generate_summary(self, all_samples: Dict[str, List[Dict[str, Any]]]):
        """Generate summary statistics"""
        summary = {
            'generated_at': datetime.now().isoformat(),
            'samples_per_cluster': self.samples_per_cluster,
            'validation_types': {}
        }
        
        for val_type, samples in all_samples.items():
            if val_type == 'l3_to_l2':
                unique_clusters = len(set(s['metadata']['correct_l2'] for s in samples))
            elif val_type == 'l2_to_l1':
                unique_clusters = len(set(s['metadata']['l2_cluster_id'] for s in samples))
            elif val_type == 'l3_to_l1':
                unique_clusters = len(set(s['metadata']['correct_l1'] for s in samples))
            else:
                unique_clusters = 0
                
            summary['validation_types'][val_type] = {
                'n_samples': len(samples),
                'unique_clusters': unique_clusters
            }
        
        with open('conseq_fin_stage4_validation_prep_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Summary statistics:")
        for val_type, stats in summary['validation_types'].items():
            logger.info(f"  {val_type}: {stats['n_samples']} samples covering {stats['unique_clusters']} clusters")
    
    def prepare_all(self):
        """Main method to prepare all validation datasets"""
        if not self.data_loaded:
            self.load_data()
        
        # Create samples for each validation type
        all_samples = {
            'l3_to_l2': self.create_l3_to_l2_samples(),
            'l2_to_l1': self.create_l2_to_l1_samples(),
            'l3_to_l1': self.create_l3_to_l1_samples()
        }
        
        # Save each dataset
        for val_type, samples in all_samples.items():
            output_file = f'conseq_fin_stage4_validation_{val_type}.jsonl'
            self.save_samples(samples, output_file)
        
        # Generate summary
        self.generate_summary(all_samples)
        
        logger.info("\nValidation data preparation complete!")

def main():
    parser = argparse.ArgumentParser(description='Prepare validation datasets')
    parser.add_argument('--samples-per-cluster', type=int, default=3,
                       help='Number of samples per cluster (default: 3)')
    
    args = parser.parse_args()
    
    prep = ValidationDataPrep(samples_per_cluster=args.samples_per_cluster)
    prep.prepare_all()

if __name__ == "__main__":
    main()
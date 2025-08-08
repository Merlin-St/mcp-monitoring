#!/usr/bin/env python3
"""
Data loading and management for ONET task clustering

This module handles:
- Loading ONET task data from CSV
- Managing the output CSV file with incremental updates
- Preparing validation samples in JSONL format
"""

import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from conseq_fin_stage4_task_llm import (
    VALIDATION_SYSTEM_PROMPT, 
    L3_TO_L2_VALIDATION_PROMPT,
    L2_TO_L1_VALIDATION_PROMPT, 
    L3_TO_L1_VALIDATION_PROMPT,
    SUBSET_L2_L3_VALIDATION_PROMPT
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_onet_tasks(csv_path: str = 'conseq_fin_stage4_onet_taskstatements.csv') -> pd.DataFrame:
    """
    Load ONET task statements from CSV
    
    Returns DataFrame with columns:
    - O*NET-SOC Code
    - Title
    - Task ID
    - Task
    - task_id (combined identifier)
    """
    logger.info(f"Loading ONET tasks from {csv_path}")
    
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"ONET task file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Create unique task identifier
    df['task_id'] = df['O*NET-SOC Code'] + '_' + df['Task ID'].astype(str)
    
    logger.info(f"Loaded {len(df)} ONET tasks")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df

def update_cluster_csv(
    df: pd.DataFrame,
    level2_clusters: Optional[np.ndarray] = None,
    level2_names: Optional[Dict[str, str]] = None,
    level1_clusters: Optional[np.ndarray] = None,
    level1_names: Optional[Dict[str, str]] = None,
    output_file: str = 'conseq_fin_stage4_tasks_cluster_names.csv'
) -> pd.DataFrame:
    """
    Update DataFrame with cluster assignments and names
    
    Args:
        df: DataFrame with ONET tasks
        level2_clusters: Array of Level 2 cluster assignments
        level2_names: Dict mapping cluster IDs to names
        level1_clusters: Array of Level 1 cluster assignments (for Level 2 clusters)
        level1_names: Dict mapping cluster IDs to names
        output_file: Path to save updated CSV
        
    Returns:
        Updated DataFrame
    """
    # Add Level 2 cluster assignments
    if level2_clusters is not None:
        logger.info(f"Adding Level 2 cluster assignments ({len(set(level2_clusters))} clusters)")
        df['level2_cluster'] = [f'L2_{i:03d}' for i in level2_clusters]
    
    # Add Level 2 cluster names
    if level2_names is not None:
        logger.info(f"Adding Level 2 cluster names ({len(level2_names)} names)")
        # Add quotes around names to handle commas and special characters
        quoted_names = {k: f"'{v}'" for k, v in level2_names.items()}
        df['level2_name'] = df['level2_cluster'].astype(str).map(quoted_names).fillna('')
    
    # Add Level 1 cluster assignments
    if level1_clusters is not None and 'level2_cluster' in df.columns:
        logger.info(f"Adding Level 1 cluster assignments ({len(set(level1_clusters))} clusters)")
        # Create mapping from L2 to L1
        l2_clusters = sorted(df['level2_cluster'].unique())
        l2_to_l1 = {l2: f'L1_{level1_clusters[i]:02d}' for i, l2 in enumerate(l2_clusters)}
        df['level1_cluster'] = df['level2_cluster'].map(l2_to_l1)
    
    # Add Level 1 cluster names
    if level1_names is not None:
        logger.info(f"Adding Level 1 cluster names ({len(level1_names)} names)")
        # Add quotes around names to handle commas and special characters
        quoted_l1_names = {k: f"'{v}'" for k, v in level1_names.items()}
        df['level1_name'] = df['level1_cluster'].map(quoted_l1_names).fillna('')
    
    # Save to CSV
    save_cluster_csv(df, output_file)
    
    return df

def save_cluster_csv(df: pd.DataFrame, output_file: str = 'conseq_fin_stage4_tasks_cluster_names.csv'):
    """Save DataFrame to CSV with proper column ordering"""
    # Define column order
    columns = ['task_id', 'O*NET-SOC Code', 'Task', 'Title']
    
    # Add cluster columns if they exist
    if 'level2_cluster' in df.columns:
        columns.extend(['level2_cluster', 'level2_name'])
    if 'level1_cluster' in df.columns:
        columns.extend(['level1_cluster', 'level1_name'])
    
    # Select only existing columns
    columns = [col for col in columns if col in df.columns]
    
    # Save to CSV
    df[columns].to_csv(output_file, index=False)
    logger.info(f"Saved {len(df)} tasks to {output_file}")

def prepare_validation_samples(
    df: pd.DataFrame,
    validation_type: str,
    n_samples: int = 50
) -> List[Dict[str, Any]]:
    """
    Prepare validation samples in JSONL format
    
    Args:
        df: DataFrame with cluster assignments
        validation_type: One of 'l3_to_l2', 'l2_to_l1', 'l3_to_l1', 'subset_l2_l3'
        n_samples: Number of samples per cluster
        
    Returns:
        List of samples ready for JSONL export
    """
    logger.info(f"Preparing {validation_type} validation samples")
    samples = []
    
    if validation_type == 'l3_to_l2':
        # Pre-compute cluster names for efficiency
        l2_names_map = df[['level2_cluster', 'level2_name']].drop_duplicates().set_index('level2_cluster')['level2_name'].to_dict()
        all_l2_options = ", ".join([f"{l2}: {name}" for l2, name in sorted(l2_names_map.items())])
        
        # Sample random tasks
        sampled_tasks = df.sample(n=min(n_samples, len(df)), random_state=42)
        
        for _, task in sampled_tasks.iterrows():
            prompt = L3_TO_L2_VALIDATION_PROMPT.format(
                task=task['Task'],
                options=all_l2_options
            )
            
            samples.append({
                'input': prompt,
                'target': task['level2_cluster'],
                'metadata': {
                    'task_id': task['task_id'],
                    'level2_cluster': task['level2_cluster'],
                    'validation_type': validation_type,
                    'system_prompt': VALIDATION_SYSTEM_PROMPT
                }
            })
    
    elif validation_type == 'l2_to_l1':
        # Pre-compute L1 names
        l1_names_map = df[['level1_cluster', 'level1_name']].drop_duplicates().set_index('level1_cluster')['level1_name'].to_dict()
        all_l1_options = ", ".join([f"{l1}: {name}" for l1, name in sorted(l1_names_map.items())])
        
        # Sample Level 2 clusters
        l2_clusters = df[['level2_cluster', 'level2_name', 'level1_cluster']].drop_duplicates()
        sampled = l2_clusters.sample(n=min(n_samples, len(l2_clusters)), random_state=42)
        
        for _, cluster in sampled.iterrows():
            prompt = L2_TO_L1_VALIDATION_PROMPT.format(
                cluster_name=cluster['level2_name'],
                options=all_l1_options
            )
            
            samples.append({
                'input': prompt,
                'target': cluster['level1_cluster'],
                'metadata': {
                    'level2_cluster': cluster['level2_cluster'],
                    'level1_cluster': cluster['level1_cluster'],
                    'validation_type': validation_type,
                    'system_prompt': VALIDATION_SYSTEM_PROMPT
                }
            })
    
    elif validation_type == 'l3_to_l1':
        # Pre-compute L1 names
        l1_names_map = df[['level1_cluster', 'level1_name']].drop_duplicates().set_index('level1_cluster')['level1_name'].to_dict()
        all_l1_options = ", ".join([f"{l1}: {name}" for l1, name in sorted(l1_names_map.items())])
        
        # Sample random tasks
        sampled_tasks = df.sample(n=min(n_samples, len(df)), random_state=42)
        
        for _, task in sampled_tasks.iterrows():
            prompt = L3_TO_L1_VALIDATION_PROMPT.format(
                task=task['Task'],
                options=all_l1_options
            )
            
            samples.append({
                'input': prompt,
                'target': task['level1_cluster'],
                'metadata': {
                    'task_id': task['task_id'],
                    'level1_cluster': task['level1_cluster'],
                    'validation_type': validation_type,
                    'system_prompt': VALIDATION_SYSTEM_PROMPT
                }
            })
    
    elif validation_type == 'subset_l2_l3':
        # Create subset L2 validation with focused options based on correct L1
        # Sample random tasks
        sampled_tasks = df.sample(n=min(n_samples, len(df)), random_state=42)
        
        for _, task in sampled_tasks.iterrows():
            correct_l1 = task['level1_cluster']
            correct_l2 = task['level2_cluster']
            correct_l2_name = task['level2_name']
            
            # Get all L2 clusters that belong to the same L1 as the correct answer
            l1_l2_clusters = df[df['level1_cluster'] == correct_l1][['level2_cluster', 'level2_name']].drop_duplicates()
            
            # Include some additional L2 clusters from other L1s as distractors (limit to ~16 total options)
            other_l2_clusters = df[df['level1_cluster'] != correct_l1][['level2_cluster', 'level2_name']].drop_duplicates()
            distractor_count = max(0, 16 - len(l1_l2_clusters))
            if distractor_count > 0 and len(other_l2_clusters) > 0:
                distractors = other_l2_clusters.sample(n=min(distractor_count, len(other_l2_clusters)), random_state=42)
                subset_l2_options = pd.concat([l1_l2_clusters, distractors])
            else:
                subset_l2_options = l1_l2_clusters
            
            # Create options string - format like other validation prompts
            options = []
            for _, option in subset_l2_options.iterrows():
                options.append(f"{option['level2_cluster']}: {option['level2_name']}")
            
            options_text = ", ".join(sorted(options))
            
            prompt = SUBSET_L2_L3_VALIDATION_PROMPT.format(
                task=task['Task'],
                options=options_text
            )
            
            samples.append({
                'input': prompt,
                'target': correct_l2,
                'metadata': {
                    'task_id': task['task_id'],
                    'correct_l2': correct_l2,
                    'correct_l2_name': correct_l2_name,
                    'correct_l1': correct_l1,
                    'validation_type': validation_type,
                    'system_prompt': VALIDATION_SYSTEM_PROMPT
                }
            })
    
    logger.info(f"Created {len(samples)} {validation_type} validation samples")
    return samples

def get_cluster_info(df: pd.DataFrame, level: str = 'level2') -> Dict[str, Dict[str, Any]]:
    """
    Get information about clusters for LLM naming
    
    Returns dict mapping cluster identifier to cluster info including sample tasks
    """
    cluster_col = f'{level}_cluster'
    clusters = {}
    
    for cluster_identifier in sorted(df[cluster_col].unique()):
        cluster_tasks = df[df[cluster_col] == cluster_identifier]
        
        # Use appropriate key name based on level
        if level == 'level2':
            key_name = 'level2_cluster'
        else:  # level1
            key_name = 'level1_cluster'
        
        clusters[cluster_identifier] = {
            key_name: cluster_identifier,
            'size': len(cluster_tasks),
            'tasks': cluster_tasks['Task'].tolist(),
            'sample_tasks': cluster_tasks.sample(min(10, len(cluster_tasks)), random_state=42)['Task'].tolist()
        }
    
    return clusters
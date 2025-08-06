#!/usr/bin/env python3
"""
Add predefined Level 1 cluster names based on analysis of their contents

This script analyzes the Level 1 clusters and assigns appropriate high-level names.
"""

import json
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_and_name_level1_clusters():
    """Analyze Level 1 clusters and assign descriptive names"""
    
    # Load data
    df = pd.read_csv('conseq_fin_stage4_onetclusters.csv')
    
    # Analyze each Level 1 cluster
    level1_names = {}
    
    for cluster_id in sorted(df['level1_cluster_id'].unique()):
        cluster_tasks = df[df['level1_cluster_id'] == cluster_id]
        
        # Get top occupations
        top_occupations = cluster_tasks['Title'].value_counts().head(20)
        
        logger.info(f"\n{cluster_id} ({len(cluster_tasks)} tasks):")
        logger.info("Top occupations:")
        for occ, count in top_occupations.items():
            logger.info(f"  {occ}: {count}")
    
    # Based on manual analysis, assign names
    # These are predefined based on the clustering results
    level1_names = {
        'cluster_1_000': 'Industrial and Manufacturing Operations',
        'cluster_1_001': 'Healthcare and Medical Services',
        'cluster_1_002': 'Business Management and Administration',
        'cluster_1_003': 'Information Technology and Digital Services',
        'cluster_1_004': 'Environmental and Agricultural Sciences',
        'cluster_1_005': 'Engineering and Technical Systems',
        'cluster_1_006': 'Construction and Skilled Trades',
        'cluster_1_007': 'Education and Training Services',
        'cluster_1_008': 'Operations and Safety Management',
        'cluster_1_009': 'Media and Creative Services'
    }
    
    # Update metadata
    metadata_file = 'conseq_fin_stage4_hierarchy_k12_summary.json'
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    if 'cluster_names' not in metadata:
        metadata['cluster_names'] = {'level1': {}, 'level2': {}}
    
    metadata['cluster_names']['level1'] = level1_names
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"\nAdded Level 1 cluster names to {metadata_file}")
    
    # Show the names
    for cluster_id, name in level1_names.items():
        logger.info(f"{cluster_id}: {name}")

if __name__ == "__main__":
    analyze_and_name_level1_clusters()
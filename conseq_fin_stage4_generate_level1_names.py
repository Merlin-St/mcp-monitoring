#!/usr/bin/env python3
"""
Generate Level 1 cluster names based on Level 2 cluster names using LLM

This script analyzes the Level 2 cluster names within each Level 1 cluster
and generates an appropriate overarching name for the Level 1 cluster.

Usage:
    inspect eval conseq_fin_stage4_generate_level1_names.py --model anthropic/claude-sonnet-4-20250514
"""

import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

from dotenv import load_dotenv
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, system_message

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conseq_fin_stage4_generate_level1_names.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def prepare_level1_data() -> List[Sample]:
    """Prepare Level 1 cluster data for LLM naming based on Level 2 names"""
    
    # Load cluster data and names
    clusters_csv = 'conseq_fin_stage4_onetclusters.csv'
    cluster_names_csv = 'conseq_fin_stage4_cluster_names.csv'
    
    if not Path(clusters_csv).exists():
        raise FileNotFoundError(f"Clusters CSV not found: {clusters_csv}")
    if not Path(cluster_names_csv).exists():
        raise FileNotFoundError(f"Cluster names CSV not found: {cluster_names_csv}")
    
    logger.info("Loading cluster data...")
    df = pd.read_csv(clusters_csv)
    names_df = pd.read_csv(cluster_names_csv)
    
    # Create mapping of Level 2 cluster IDs to names
    l2_names = dict(zip(names_df['cluster_id'], names_df['cluster_name']))
    
    # Get unique Level 1 clusters
    level1_clusters = sorted(df['level1_cluster_id'].unique())
    logger.info(f"Found {len(level1_clusters)} Level 1 clusters to name")
    
    samples = []
    
    for l1_cluster_id in level1_clusters:
        # Get all Level 2 clusters within this Level 1 cluster
        l1_data = df[df['level1_cluster_id'] == l1_cluster_id]
        l2_clusters_in_l1 = sorted(l1_data['level2_cluster_id'].unique())
        
        # Get the names of these Level 2 clusters
        l2_cluster_names = []
        for l2_id in l2_clusters_in_l1:
            if l2_id in l2_names:
                l2_cluster_names.append(l2_names[l2_id])
        
        # Calculate statistics
        total_tasks = len(l1_data)
        num_l2_clusters = len(l2_clusters_in_l1)
        
        # Create prompt for this Level 1 cluster
        prompt = f"""Here are the {num_l2_clusters} middle-level cluster names within this top-level cluster (containing {total_tasks} total tasks):

{chr(10).join([f"- {name}" for name in sorted(l2_cluster_names)])}

Based on these middle-level clusters, generate a broad, overarching name that captures the common theme across all these clusters.

Respond with ONLY the cluster name, no explanation or additional text."""
        
        samples.append(Sample(
            input=prompt,
            metadata={
                "cluster_id": l1_cluster_id,
                "task_count": total_tasks,
                "l2_cluster_count": num_l2_clusters
            }
        ))
    
    return samples

@task
def level1_naming_task():
    """Task to generate names for Level 1 clusters based on Level 2 names"""
    
    # Prepare samples
    samples = prepare_level1_data()
    logger.info(f"Prepared {len(samples)} Level 1 cluster naming prompts")
    
    system_prompt = """You are an expert at analyzing hierarchical categorizations and creating clear, high-level category names.
Your task is to generate broad, overarching names for top-level clusters based on the names of their constituent middle-level clusters.
The name should capture the common theme that unifies all the middle-level clusters.

The name should be:
- Broad and encompassing (capturing the full scope of activities)
- Professional and clear
- Focused on the primary domain or sector
- Typically 3-5 words
- At a higher level of abstraction than the middle-level clusters"""
    
    return Task(
        dataset=samples,
        solver=[
            system_message(system_prompt),
            generate()
        ],
        # No scorer needed - we just want the generated names
    )
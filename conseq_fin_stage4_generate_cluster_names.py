#!/usr/bin/env python3
"""
Generate descriptive names for Level 2 clusters using LLM

This script uses the Inspect framework to generate concise, descriptive names
for each of the 400 Level 2 clusters based on their task contents.

Usage:
    inspect eval conseq_fin_stage4_generate_cluster_names.py --model anthropic/claude-sonnet-4-20250514
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
        logging.FileHandler('conseq_fin_stage4_generate_cluster_names.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def prepare_cluster_data() -> List[Sample]:
    """Prepare cluster data for LLM naming"""
    
    # Load hierarchy data
    clusters_csv = 'conseq_fin_stage4_onetclusters.csv'
    metadata_json = 'conseq_fin_stage4_hierarchy_metadata.json'
    
    if not Path(clusters_csv).exists():
        raise FileNotFoundError(f"Clusters CSV not found: {clusters_csv}")
    
    logger.info("Loading cluster data...")
    df = pd.read_csv(clusters_csv)
    
    # Load existing metadata
    if Path(metadata_json).exists():
        with open(metadata_json, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Get unique Level 2 clusters
    level2_clusters = sorted(df['level2_cluster_id'].unique())
    logger.info(f"Found {len(level2_clusters)} Level 2 clusters to name")
    
    samples = []
    
    for cluster_id in level2_clusters:
        # Get ALL tasks in this cluster
        cluster_tasks = df[df['level2_cluster_id'] == cluster_id]
        
        # Get all unique tasks
        all_tasks = cluster_tasks['Task'].tolist()
        
        # Create prompt for this cluster
        prompt = f"""Here are the tasks in the cluster:

{chr(10).join([f"- {task}" for task in all_tasks])}

Respond with ONLY the cluster name, no explanation or additional text."""
        
        samples.append(Sample(
            input=prompt,
            metadata={
                "cluster_id": cluster_id,
                "task_count": len(cluster_tasks),
                "level1_cluster": cluster_tasks.iloc[0]['level1_cluster_id']
            }
        ))
    
    return samples

@task
def cluster_naming_task():
    """Task to generate names for Level 2 clusters"""
    
    # Prepare samples
    samples = prepare_cluster_data()
    logger.info(f"Prepared {len(samples)} cluster naming prompts")
    
    system_prompt = """You are an expert at analyzing occupational tasks and creating clear, descriptive category names.
Your task is to generate concise, professional names for task clusters based on their content.
Focus on the primary function or activity that unifies the tasks in each cluster. Provide a descriptive name that captures the common theme of these tasks.
The name should be:

Concise (3-7 words)
Professional and clear
Focused on the primary function/activity"""
    
    return Task(
        dataset=samples,
        solver=[
            system_message(system_prompt),
            generate()
        ],
        # No scorer needed - we just want the generated names
    )
#!/usr/bin/env python3
"""
Generate Level 1 cluster names with full hierarchical context

This script generates Level 1 names by showing the LLM the complete
hierarchical structure, including example paths from Level 1 → Level 2 → tasks,
to ensure names that reflect the full scope of each top-level cluster.

Usage:
    inspect eval conseq_fin_stage4_generate_level1_names_hierarchical.py --model anthropic/claude-sonnet-4-20250514
"""

import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict

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
        logging.FileHandler('conseq_fin_stage4_generate_level1_names_hierarchical.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def prepare_hierarchical_l1_data() -> List[Sample]:
    """Prepare Level 1 data with full hierarchical context"""
    
    # Load cluster data and names
    clusters_csv = 'conseq_fin_stage4_onetclusters.csv'
    cluster_names_csv = 'conseq_fin_stage4_cluster_names.csv'
    
    if not Path(clusters_csv).exists():
        raise FileNotFoundError(f"Clusters CSV not found: {clusters_csv}")
    
    logger.info("Loading cluster data...")
    df = pd.read_csv(clusters_csv)
    
    # Load Level 2 names
    l2_names = {}
    if Path(cluster_names_csv).exists():
        names_df = pd.read_csv(cluster_names_csv)
        l2_names = dict(zip(names_df['cluster_id'], names_df['cluster_name']))
    
    # Get unique Level 1 clusters
    level1_clusters = sorted(df['level1_cluster_id'].unique())
    logger.info(f"Found {len(level1_clusters)} Level 1 clusters to name")
    
    samples = []
    
    for l1_cluster_id in level1_clusters:
        # Get all data for this Level 1 cluster
        l1_data = df[df['level1_cluster_id'] == l1_cluster_id]
        
        # Organize by Level 2 clusters
        l2_clusters = defaultdict(list)
        for _, row in l1_data.iterrows():
            l2_clusters[row['level2_cluster_id']].append(row['Task'])
        
        # Create hierarchical representation
        hierarchy_examples = []
        
        # Show 5-10 example Level 2 clusters with their tasks
        for i, (l2_id, tasks) in enumerate(sorted(l2_clusters.items())[:10]):
            l2_name = l2_names.get(l2_id, "Unknown")
            example_tasks = tasks[:3]  # Show 3 tasks per L2 cluster
            
            hierarchy_examples.append(f"""
  └─ {l2_id}: {l2_name}
     Examples: 
{chr(10).join([f"       • {task}" for task in example_tasks])}""")
        
        # Statistics
        total_tasks = len(l1_data)
        num_l2_clusters = len(l2_clusters)
        
        # Get other Level 1 clusters for contrast
        other_l1_ids = [l1 for l1 in level1_clusters if l1 != l1_cluster_id]
        other_l1_examples = []
        for other_l1 in other_l1_ids[:3]:  # Show 3 other L1 clusters
            other_data = df[df['level1_cluster_id'] == other_l1]
            other_l2s = other_data['level2_cluster_id'].unique()[:3]
            other_examples = []
            for l2 in other_l2s:
                l2_name = l2_names.get(l2, "Unknown")
                other_examples.append(f"{l2}: {l2_name}")
            other_l1_examples.append(f"{other_l1} contains: {', '.join(other_examples[:2])}")
        
        # Create prompt
        prompt = f"""Create a Level 1 (top-level) cluster name based on this hierarchical structure:

LEVEL 1 CLUSTER: {l1_cluster_id}
├─ Contains {num_l2_clusters} Level 2 clusters
├─ Total of {total_tasks} occupational tasks
│
├─ SAMPLE LEVEL 2 CLUSTERS AND THEIR TASKS:
{''.join(hierarchy_examples)}
{f"|  ... and {num_l2_clusters - 10} more Level 2 clusters" if num_l2_clusters > 10 else ""}

CONTRAST WITH OTHER LEVEL 1 CLUSTERS:
To ensure distinctiveness, here are examples from OTHER top-level clusters:
{chr(10).join([f"- {ex}" for ex in other_l1_examples])}

Create a Level 1 name that:
1. Encompasses ALL the Level 2 clusters shown (and those not shown)
2. Is broad enough to cover the full scope
3. Is distinct from the other Level 1 clusters mentioned
4. Is professional and 3-5 words

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
def level1_naming_hierarchical():
    """Generate Level 1 names with full hierarchical context"""
    
    # Prepare samples
    samples = prepare_hierarchical_l1_data()
    logger.info(f"Prepared {len(samples)} hierarchical Level 1 naming prompts")
    
    system_prompt = """You are an expert at creating high-level category names that accurately reflect hierarchical structures.

When shown a Level 1 cluster with its Level 2 subclusters and example tasks, create a name that:
- Captures the full breadth of all subclusters
- Is appropriately high-level and encompassing
- Distinguishes clearly from other Level 1 clusters
- Reflects the hierarchical organization

Think of Level 1 names as broad sectors or domains that contain many specialized areas."""
    
    return Task(
        dataset=samples,
        solver=[
            system_message(system_prompt),
            generate()
        ]
    )
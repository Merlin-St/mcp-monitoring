#!/usr/bin/env python3
"""
Generate more distinctive Level 2 cluster names using contrastive approach

This script shows the LLM both tasks within the cluster AND nearby tasks
that are NOT in the cluster, to generate names that clearly distinguish
the cluster from its neighbors.

Usage:
    inspect eval conseq_fin_stage4_generate_cluster_names_contrastive.py --model anthropic/claude-sonnet-4-20250514
"""

import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

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
        logging.FileHandler('conseq_fin_stage4_generate_cluster_names_contrastive.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def find_boundary_tasks(df: pd.DataFrame, embeddings: np.ndarray, cluster_id: str, n_boundary: int = 10) -> List[Dict]:
    """Find tasks from other clusters that are closest to this cluster"""
    
    # Get tasks in this cluster
    cluster_mask = df['level2_cluster_id'] == cluster_id
    cluster_indices = df[cluster_mask].index.tolist()
    
    # Get embeddings for this cluster
    cluster_embeddings = embeddings[cluster_indices]
    
    # Calculate centroid of this cluster
    cluster_centroid = cluster_embeddings.mean(axis=0).reshape(1, -1)
    
    # Get tasks NOT in this cluster
    other_mask = ~cluster_mask
    other_indices = df[other_mask].index.tolist()
    other_embeddings = embeddings[other_indices]
    
    # Calculate distances to all other tasks
    distances = cosine_similarity(cluster_centroid, other_embeddings)[0]
    
    # Get indices of closest tasks from other clusters
    closest_indices = np.argsort(distances)[-n_boundary:][::-1]  # Highest similarity first
    
    # Get the actual tasks
    boundary_tasks = []
    for idx in closest_indices:
        actual_idx = other_indices[idx]
        task_data = df.iloc[actual_idx]
        boundary_tasks.append({
            'task': task_data['Task'],
            'cluster': task_data['level2_cluster_id'],
            'similarity': float(distances[idx])
        })
    
    return boundary_tasks

def prepare_contrastive_cluster_data() -> Tuple[List[Sample], np.ndarray]:
    """Prepare cluster data with boundary tasks for contrastive naming"""
    
    # Load hierarchy data
    clusters_csv = 'conseq_fin_stage4_onetclusters.csv'
    
    if not Path(clusters_csv).exists():
        raise FileNotFoundError(f"Clusters CSV not found: {clusters_csv}")
    
    logger.info("Loading cluster data...")
    df = pd.read_csv(clusters_csv)
    
    # Load or generate embeddings
    logger.info("Loading embeddings...")
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    # Check for cached embeddings
    embeddings_file = 'embeddings_cache/onet_task_embeddings.npy'
    if Path(embeddings_file).exists():
        embeddings = np.load(embeddings_file)
        logger.info(f"Loaded cached embeddings: {embeddings.shape}")
    else:
        logger.info("Generating embeddings...")
        texts = df['Task'].tolist()
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
        # Save for future use
        Path('embeddings_cache').mkdir(exist_ok=True)
        np.save(embeddings_file, embeddings)
    
    # Get unique Level 2 clusters
    level2_clusters = sorted(df['level2_cluster_id'].unique())
    logger.info(f"Found {len(level2_clusters)} Level 2 clusters to name")
    
    samples = []
    
    for i, cluster_id in enumerate(level2_clusters):
        if i % 50 == 0:
            logger.info(f"Processing cluster {i}/{len(level2_clusters)}...")
        
        # Get ALL tasks in this cluster
        cluster_tasks = df[df['level2_cluster_id'] == cluster_id]
        all_tasks = cluster_tasks['Task'].tolist()
        
        # Find boundary tasks from other clusters
        boundary_tasks = find_boundary_tasks(df, embeddings, cluster_id, n_boundary=10)
        
        # Create prompt for this cluster
        prompt = f"""You need to create a distinctive name for a cluster of occupational tasks.

TASKS IN THIS CLUSTER ({len(all_tasks)} tasks):
{chr(10).join([f"- {task}" for task in all_tasks[:50]])}  # Show first 50 for context
{f"... and {len(all_tasks) - 50} more tasks" if len(all_tasks) > 50 else ""}

IMPORTANT - TASKS NOT IN THIS CLUSTER (from neighboring clusters):
These similar tasks belong to OTHER clusters and should NOT be covered by your cluster name:
{chr(10).join([f"- {bt['task']} (from {bt['cluster']})" for bt in boundary_tasks[:5]])}

Create a cluster name that:
1. Accurately describes the tasks IN the cluster
2. EXCLUDES the boundary tasks shown above
3. Is specific enough to distinguish from neighboring clusters
4. Is 3-7 words long

Respond with ONLY the cluster name, no explanation or additional text."""
        
        samples.append(Sample(
            input=prompt,
            metadata={
                "cluster_id": cluster_id,
                "task_count": len(cluster_tasks),
                "level1_cluster": cluster_tasks.iloc[0]['level1_cluster_id']
            }
        ))
    
    return samples, embeddings

@task
def cluster_naming_contrastive():
    """Generate distinctive Level 2 cluster names using contrastive approach"""
    
    # Prepare samples with boundary tasks
    samples, _ = prepare_contrastive_cluster_data()
    logger.info(f"Prepared {len(samples)} contrastive cluster naming prompts")
    
    system_prompt = """You are an expert at creating precise, distinctive category names for clusters of occupational tasks.

Your goal is to create names that clearly distinguish each cluster from its neighbors.
When shown boundary tasks that are NOT in the cluster, ensure your name excludes them.

Focus on:
- The unique aspects of the tasks IN the cluster
- Creating boundaries that exclude similar but different tasks
- Being specific rather than generic"""
    
    return Task(
        dataset=samples,
        solver=[
            system_message(system_prompt),
            generate()
        ]
    )
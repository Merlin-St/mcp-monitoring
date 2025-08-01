#!/usr/bin/env python3
"""
Stage 4 Level 1 Supercluster Builder v2

New approach: Embed Level 2 cluster names to create K=20 Level 1 superclusters
- Use both basic and contrastive Level 2 cluster names as features
- Create embeddings of cluster names (not tasks)
- Cluster into K=20 superclusters
- Generate LLM names for superclusters based on member clusters
"""

import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import anthropic
import os
from typing import Dict, List, Tuple
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conseq_fin_stage4_build_level1_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_cluster_names() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load both basic and contrastive cluster names."""
    logger.info("Loading Level 2 cluster names...")
    
    basic_df = pd.read_csv('conseq_fin_stage4_cluster_names.csv')
    contrastive_df = pd.read_csv('conseq_fin_stage4_cluster_names_contrastive.csv')
    
    logger.info(f"Loaded {len(basic_df)} basic cluster names")
    logger.info(f"Loaded {len(contrastive_df)} contrastive cluster names")
    
    # Merge on cluster_id
    merged_df = basic_df.merge(
        contrastive_df, 
        on='cluster_id', 
        suffixes=('_basic', '_contrastive')
    )
    
    logger.info(f"Merged dataset: {len(merged_df)} clusters")
    return merged_df, basic_df, contrastive_df

def create_cluster_embeddings(merged_df: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Create embeddings for cluster names using both basic and contrastive names."""
    logger.info(f"Creating embeddings using {model_name}...")
    
    model = SentenceTransformer(model_name)
    
    # Combine basic and contrastive names for richer representation
    combined_texts = []
    for _, row in merged_df.iterrows():
        basic_name = row['cluster_name_basic']
        contrastive_name = row['cluster_name_contrastive']
        
        # Create combined text that incorporates both perspectives
        combined_text = f"{basic_name}. {contrastive_name}"
        combined_texts.append(combined_text)
    
    logger.info(f"Embedding {len(combined_texts)} combined cluster descriptions...")
    embeddings = model.encode(combined_texts, show_progress_bar=True)
    
    logger.info(f"Created embeddings with shape: {embeddings.shape}")
    return embeddings

def cluster_level2_to_level1(embeddings: np.ndarray, k: int = 20) -> Tuple[np.ndarray, float]:
    """Cluster Level 2 embeddings into K Level 1 superclusters."""
    logger.info(f"Clustering {len(embeddings)} Level 2 clusters into K={k} Level 1 superclusters...")
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Calculate silhouette score
    silhouette = silhouette_score(embeddings, cluster_labels)
    logger.info(f"Clustering complete. Silhouette score: {silhouette:.3f}")
    
    # Log cluster sizes
    unique, counts = np.unique(cluster_labels, return_counts=True)
    logger.info("Level 1 supercluster sizes:")
    for cluster_id, count in zip(unique, counts):
        logger.info(f"  L1_{cluster_id:02d}: {count} Level 2 clusters")
    
    return cluster_labels, silhouette

def generate_supercluster_names(merged_df: pd.DataFrame, level1_labels: np.ndarray) -> Dict[str, str]:
    """Generate descriptive names for Level 1 superclusters using LLM."""
    logger.info("Generating Level 1 supercluster names using LLM...")
    
    # Check for API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not found. Cannot generate supercluster names.")
        return {}
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # Group clusters by Level 1 assignment
    level1_groups = {}
    for i, l1_label in enumerate(level1_labels):
        if l1_label not in level1_groups:
            level1_groups[l1_label] = []
        level1_groups[l1_label].append(i)
    
    supercluster_names = {}
    
    for l1_id, cluster_indices in level1_groups.items():
        logger.info(f"Generating name for L1_{l1_id:02d} ({len(cluster_indices)} L2 clusters)...")
        
        # Get the Level 2 cluster names in this supercluster
        l2_names_basic = [merged_df.iloc[i]['cluster_name_basic'] for i in cluster_indices]
        l2_names_contrastive = [merged_df.iloc[i]['cluster_name_contrastive'] for i in cluster_indices]
        
        # Create prompt
        prompt = f"""You are analyzing Level 2 cluster names to create a Level 1 supercluster name.

Level 2 clusters in this supercluster ({len(cluster_indices)} total):

BASIC NAMES:
{chr(10).join(f"• {name}" for name in l2_names_basic)}

CONTRASTIVE NAMES:
{chr(10).join(f"• {name}" for name in l2_names_contrastive)}

Create a concise, descriptive Level 1 supercluster name (3-6 words) that captures the common theme across these Level 2 clusters. Focus on the broader occupational or functional category that unifies them.

Return only the supercluster name, nothing else."""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}]
            )
            
            supercluster_name = response.content[0].text.strip()
            supercluster_names[f"L1_{l1_id:02d}"] = supercluster_name
            logger.info(f"  Generated: {supercluster_name}")
            
        except Exception as e:
            logger.error(f"Error generating name for L1_{l1_id:02d}: {e}")
            supercluster_names[f"L1_{l1_id:02d}"] = f"Supercluster {l1_id}"
    
    return supercluster_names

def save_new_hierarchy(merged_df: pd.DataFrame, level1_labels: np.ndarray, 
                      supercluster_names: Dict[str, str], silhouette: float):
    """Save the new Level 1 hierarchy structure."""
    logger.info("Saving new Level 1 hierarchy...")
    
    # Create the new hierarchy structure
    hierarchy = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "method": "level2_name_embedding",
            "k_level1": 12,
            "n_level2_clusters": len(merged_df),
            "silhouette_score": float(silhouette),
            "embedding_model": "all-MiniLM-L6-v2",
            "clustering_method": "kmeans"
        },
        "level1_superclusters": supercluster_names,
        "level2_to_level1_mapping": {}
    }
    
    # Create the mapping from Level 2 to Level 1
    for i, (_, row) in enumerate(merged_df.iterrows()):
        cluster_id = row['cluster_id']
        l1_label = f"L1_{level1_labels[i]:02d}"
        
        hierarchy["level2_to_level1_mapping"][cluster_id] = {
            "level1_id": l1_label,
            "level1_name": supercluster_names.get(l1_label, f"Supercluster {level1_labels[i]}"),
            "basic_name": row['cluster_name_basic'],
            "contrastive_name": row['cluster_name_contrastive']
        }
    
    # Save hierarchy
    hierarchy_file = 'conseq_fin_stage4_hierarchy_k12.json'
    with open(hierarchy_file, 'w') as f:
        json.dump(hierarchy, f, indent=2)
    
    logger.info(f"Saved new hierarchy to {hierarchy_file}")
    
    # Create summary
    summary = {
        "total_level2_clusters": len(merged_df),
        "total_level1_superclusters": len(supercluster_names),
        "silhouette_score": float(silhouette),
        "level1_cluster_sizes": {}
    }
    
    # Calculate Level 1 cluster sizes
    unique, counts = np.unique(level1_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        l1_key = f"L1_{cluster_id:02d}"
        summary["level1_cluster_sizes"][l1_key] = {
            "name": supercluster_names.get(l1_key, f"Supercluster {cluster_id}"),
            "size": int(count)
        }
    
    # Save summary
    summary_file = 'conseq_fin_stage4_hierarchy_k12_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved hierarchy summary to {summary_file}")
    
    return hierarchy

def main():
    """Main execution function."""
    logger.info("Starting Level 1 supercluster building v2...")
    
    try:
        # Load cluster names
        merged_df, basic_df, contrastive_df = load_cluster_names()
        
        # Create embeddings
        embeddings = create_cluster_embeddings(merged_df)
        
        # Cluster into Level 1 superclusters
        level1_labels, silhouette = cluster_level2_to_level1(embeddings, k=12)
        
        # Generate supercluster names
        supercluster_names = generate_supercluster_names(merged_df, level1_labels)
        
        # Save new hierarchy
        hierarchy = save_new_hierarchy(merged_df, level1_labels, supercluster_names, silhouette)
        
        logger.info("Level 1 supercluster building complete!")
        logger.info(f"Created {len(supercluster_names)} Level 1 superclusters from {len(merged_df)} Level 2 clusters")
        logger.info(f"Silhouette score: {silhouette:.3f}")
        
        # Print Level 1 superclusters
        logger.info("\nLevel 1 Superclusters:")
        for l1_id, name in sorted(supercluster_names.items()):
            cluster_num = int(l1_id.split('_')[1])
            size = sum(1 for label in level1_labels if label == cluster_num)
            logger.info(f"  {l1_id}: {name} ({size} L2 clusters)")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
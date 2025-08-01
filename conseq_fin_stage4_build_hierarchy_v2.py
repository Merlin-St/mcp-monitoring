#!/usr/bin/env python3
"""
Build O*NET Task Hierarchy using Pure Clustering (No LLM)

Creates a 3-level hierarchy:
- Level 1: 10 top-level clusters (via k-means on Level 2 centroids)
- Level 2: 400 middle-level clusters (via k-means on task embeddings)
- Level 3: ~20,000 O*NET tasks (original data)

Output: CSV file mapping each task to its Level 1 and Level 2 clusters

Usage:
    python conseq_fin_stage4_build_hierarchy_v2.py
    python conseq_fin_stage4_build_hierarchy_v2.py --test-mode  # Use 1000 tasks
    python conseq_fin_stage4_build_hierarchy_v2.py --n-clusters-l2 300  # Fewer L2 clusters
"""

import json
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conseq_fin_stage4_build_hierarchy_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HierarchyBuilder:
    def __init__(self, n_clusters_l1: int = 10, n_clusters_l2: int = 400, test_mode: bool = False):
        """
        Initialize hierarchy builder
        
        Args:
            n_clusters_l1: Number of Level 1 clusters (default: 10)
            n_clusters_l2: Number of Level 2 clusters (default: 400)
            test_mode: If True, use only first 1000 tasks
        """
        self.n_clusters_l1 = n_clusters_l1
        self.n_clusters_l2 = n_clusters_l2
        self.test_mode = test_mode
        self.embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        # File paths
        self.tasks_csv = 'conseq_fin_stage4_onet_taskstatements.csv'
        self.output_csv = 'conseq_fin_stage4_onetclusters.csv'
        self.metadata_json = 'conseq_fin_stage4_hierarchy_metadata.json'
        self.embeddings_cache = 'conseq_fin_stage4_task_embeddings_cache.npz'
        
    def load_tasks(self) -> pd.DataFrame:
        """Load O*NET tasks from CSV"""
        logger.info("Loading O*NET tasks...")
        
        # Load task statements
        df = pd.read_csv(self.tasks_csv)
        
        # Create unique task ID
        df['task_id'] = df['O*NET-SOC Code'] + '_' + df['Task ID'].astype(str)
        
        # Create text for embedding (task + occupation context)
        df['embedding_text'] = df['Task'] + " [Context: " + df['Title'] + "]"
        
        if self.test_mode:
            logger.info("Test mode: Using first 1000 tasks")
            df = df.head(1000)
            
        logger.info(f"Loaded {len(df)} tasks")
        return df
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate or load cached embeddings"""
        cache_path = Path(self.embeddings_cache)
        
        if cache_path.exists() and not self.test_mode:
            logger.info("Loading cached embeddings...")
            with np.load(cache_path) as data:
                embeddings = data['embeddings']
                if len(embeddings) == len(texts):
                    logger.info(f"Loaded {len(embeddings)} cached embeddings")
                    return embeddings
                    
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.embedder.encode(texts, show_progress_bar=True, batch_size=32)
        
        # Cache embeddings
        if not self.test_mode:
            np.savez_compressed(cache_path, embeddings=embeddings)
            logger.info("Cached embeddings for future use")
            
        return embeddings
        
    def cluster_level2(self, embeddings: np.ndarray) -> Tuple[np.ndarray, KMeans]:
        """Create Level 2 clusters using k-means on task embeddings"""
        logger.info(f"Creating {self.n_clusters_l2} Level 2 clusters...")
        
        # K-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters_l2, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate silhouette score
        if len(embeddings) < 10000:  # Only for smaller datasets
            score = silhouette_score(embeddings, cluster_labels, sample_size=min(5000, len(embeddings)))
            logger.info(f"Level 2 clustering silhouette score: {score:.3f}")
        
        # Log cluster sizes
        unique, counts = np.unique(cluster_labels, return_counts=True)
        logger.info(f"Level 2 cluster sizes - Min: {counts.min()}, Max: {counts.max()}, Avg: {counts.mean():.1f}")
        
        return cluster_labels, kmeans
        
    def cluster_level1(self, l2_centroids: np.ndarray, l2_labels: np.ndarray) -> np.ndarray:
        """Create Level 1 clusters using k-means on Level 2 centroids"""
        logger.info(f"Creating {self.n_clusters_l1} Level 1 clusters from Level 2 centroids...")
        
        # K-means on centroids
        kmeans = KMeans(n_clusters=self.n_clusters_l1, random_state=42, n_init=10)
        l2_to_l1_mapping = kmeans.fit_predict(l2_centroids)
        
        # Map Level 1 assignments back to original tasks
        l1_labels = np.array([l2_to_l1_mapping[l2_label] for l2_label in l2_labels])
        
        # Log cluster sizes
        unique, counts = np.unique(l1_labels, return_counts=True)
        logger.info(f"Level 1 cluster sizes - Min: {counts.min()}, Max: {counts.max()}, Avg: {counts.mean():.1f}")
        
        return l1_labels
        
    def generate_cluster_descriptions(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Generate descriptions for each cluster based on common terms"""
        logger.info("Generating cluster descriptions...")
        
        descriptions = {
            'level1': {},
            'level2': {}
        }
        
        # Level 1 descriptions
        for cluster_id in df['level1_cluster_id'].unique():
            cluster_tasks = df[df['level1_cluster_id'] == cluster_id]
            
            # Get most common occupations
            top_occupations = cluster_tasks['Title'].value_counts().head(5).index.tolist()
            
            # Get sample tasks
            sample_tasks = cluster_tasks.sample(min(5, len(cluster_tasks)))['Task'].tolist()
            
            descriptions['level1'][cluster_id] = {
                'size': len(cluster_tasks),
                'top_occupations': top_occupations,
                'sample_tasks': sample_tasks
            }
            
        # Level 2 descriptions (sample for efficiency)
        for cluster_id in df['level2_cluster_id'].unique():
            cluster_tasks = df[df['level2_cluster_id'] == cluster_id]
            
            # Get most common occupation
            top_occupation = cluster_tasks['Title'].value_counts().head(1).index[0]
            
            # Get sample tasks
            sample_tasks = cluster_tasks.sample(min(3, len(cluster_tasks)))['Task'].tolist()
            
            descriptions['level2'][cluster_id] = {
                'size': len(cluster_tasks),
                'primary_occupation': top_occupation,
                'sample_tasks': sample_tasks
            }
            
        return descriptions
        
    def build_hierarchy(self):
        """Main method to build the complete hierarchy"""
        logger.info("Starting hierarchy construction...")
        start_time = datetime.now()
        
        # Load tasks
        df = self.load_tasks()
        
        # Generate embeddings
        embeddings = self.generate_embeddings(df['embedding_text'].tolist())
        
        # Level 2 clustering
        l2_labels, l2_kmeans = self.cluster_level2(embeddings)
        df['level2_cluster_id'] = [f'cluster_2_{i:03d}' for i in l2_labels]
        
        # Level 1 clustering (on Level 2 centroids)
        l1_labels = self.cluster_level1(l2_kmeans.cluster_centers_, l2_labels)
        df['level1_cluster_id'] = [f'cluster_1_{i:03d}' for i in l1_labels]
        
        # Generate cluster descriptions
        descriptions = self.generate_cluster_descriptions(df)
        
        # Save results
        logger.info("Saving results...")
        
        # Save main CSV with cluster assignments
        output_columns = ['task_id', 'O*NET-SOC Code', 'Title', 'Task', 
                         'level1_cluster_id', 'level2_cluster_id']
        df[output_columns].to_csv(self.output_csv, index=False)
        logger.info(f"Saved task-cluster mappings to {self.output_csv}")
        
        # Save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'n_tasks': len(df),
            'n_clusters_l1': self.n_clusters_l1,
            'n_clusters_l2': self.n_clusters_l2,
            'test_mode': self.test_mode,
            'cluster_descriptions': descriptions,
            'statistics': {
                'level1_cluster_sizes': df.groupby('level1_cluster_id').size().to_dict(),
                'level2_cluster_sizes': df.groupby('level2_cluster_id').size().to_dict()
            }
        }
        
        with open(self.metadata_json, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {self.metadata_json}")
        
        # Log summary
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"\nHierarchy construction complete in {elapsed:.1f} seconds")
        logger.info(f"Total tasks: {len(df)}")
        logger.info(f"Level 1 clusters: {self.n_clusters_l1}")
        logger.info(f"Level 2 clusters: {self.n_clusters_l2}")
        
        return df

def main():
    parser = argparse.ArgumentParser(description='Build O*NET task hierarchy using clustering')
    parser.add_argument('--test-mode', action='store_true',
                       help='Use only first 1000 tasks for testing')
    parser.add_argument('--n-clusters-l1', type=int, default=10,
                       help='Number of Level 1 clusters (default: 10)')
    parser.add_argument('--n-clusters-l2', type=int, default=400,
                       help='Number of Level 2 clusters (default: 400)')
    
    args = parser.parse_args()
    
    # Build hierarchy
    builder = HierarchyBuilder(
        n_clusters_l1=args.n_clusters_l1,
        n_clusters_l2=args.n_clusters_l2,
        test_mode=args.test_mode
    )
    
    df = builder.build_hierarchy()
    
    # Show sample results
    logger.info("\nSample cluster assignments:")
    sample = df.sample(5)
    for _, row in sample.iterrows():
        logger.info(f"\nTask: {row['Task'][:80]}...")
        logger.info(f"  Occupation: {row['Title']}")
        logger.info(f"  Level 1: {row['level1_cluster_id']}")
        logger.info(f"  Level 2: {row['level2_cluster_id']}")

if __name__ == "__main__":
    main()
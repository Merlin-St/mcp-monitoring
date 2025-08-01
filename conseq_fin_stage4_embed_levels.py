#!/usr/bin/env python3
"""
O*NET Task Hierarchy Builder using Embeddings and Clustering

This script creates a 3-level hierarchy of O*NET tasks following the methodology
from "Which Economic Tasks are Performed with AI?" (Anthropic, 2025).

Hierarchy:
- Level 1: 10 top-level economic task categories (predefined)
- Level 2: ~400 middle-level clusters (via k-means on embeddings)  
- Level 3: ~20,000 O*NET base tasks

Usage:
    python conseq_fin_stage4_embed_levels.py
    python conseq_fin_stage4_embed_levels.py --force-rebuild
    python conseq_fin_stage4_embed_levels.py --test-mode
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
from sklearn.metrics.pairwise import cosine_similarity
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conseq_fin_stage4_embed_levels.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Top-level task categories from the paper
TOP_LEVEL_TASKS = {
    "it_systems": "Design, implement, and maintain diverse information technology systems",
    "art_culture": "Create and preserve art, culture, and religious artifacts",
    "business_finance": "Business management, finance, and customer service operations",
    "education_hr": "Manage education, HR, and professional development programs",
    "scientific_research": "Conduct scientific research and technical analysis across disciplines",
    "government_safety": "Perform government regulatory enforcement and public safety operations",
    "industrial_agricultural": "Operate and manage diverse industrial and agricultural processes",
    "energy_management": "Manage diverse energy sources and optimize consumption",
    "environmental_systems": "Manage and improve environmental systems and sustainability practices",
    "healthcare_services": "Comprehensive healthcare services and medical treatment across specialties"
}

# Claude API configuration for cluster naming
ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"

class ONETTaskHierarchyBuilder:
    def __init__(self, force_rebuild: bool = False, test_mode: bool = False):
        self.force_rebuild = force_rebuild
        self.test_mode = test_mode
        self.embeddings_cache_file = "conseq_fin_stage4_embeddings_cache.npz"
        self.hierarchy_file = "conseq_fin_stage4_hierarchy.json"
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        # Load O*NET data
        self.tasks_df = None
        self.embeddings = None
        self.hierarchy = None
        
    def load_onet_tasks(self) -> pd.DataFrame:
        """Load O*NET task statements from CSV"""
        logger.info("Loading O*NET task statements...")
        
        # Load task statements
        tasks_df = pd.read_csv('conseq_fin_stage4_onet_taskstatements.csv')
        
        # Create combined text for embedding (task + occupation context)
        tasks_df['embedding_text'] = (
            tasks_df['Task'] + " [" + 
            tasks_df['Title'] + "]"
        )
        
        # Add unique identifier
        tasks_df['task_id'] = tasks_df['O*NET-SOC Code'] + '_' + tasks_df['Task ID'].astype(str)
        
        if self.test_mode:
            # Use subset for testing
            logger.info("Test mode: Using first 1000 tasks")
            tasks_df = tasks_df.head(1000)
        
        logger.info(f"Loaded {len(tasks_df)} O*NET tasks")
        self.tasks_df = tasks_df
        return tasks_df
    
    def generate_embeddings(self) -> np.ndarray:
        """Generate or load cached embeddings for all tasks"""
        cache_path = Path(self.embeddings_cache_file)
        
        if cache_path.exists() and not self.force_rebuild:
            logger.info("Loading cached embeddings...")
            data = np.load(cache_path)
            embeddings = data['embeddings']
            task_ids = data['task_ids']
            
            # Verify alignment with current tasks
            if len(embeddings) == len(self.tasks_df) and \
               all(tid == self.tasks_df.iloc[i]['task_id'] for i, tid in enumerate(task_ids)):
                logger.info(f"Loaded {len(embeddings)} cached embeddings")
                self.embeddings = embeddings
                return embeddings
            else:
                logger.warning("Cached embeddings don't match current tasks, regenerating...")
        
        logger.info("Generating embeddings for all tasks...")
        texts = self.tasks_df['embedding_text'].tolist()
        
        # Generate embeddings in batches
        batch_size = 100
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=True)
            embeddings.extend(batch_embeddings)
            
            if i % 1000 == 0:
                logger.info(f"Generated embeddings for {i}/{len(texts)} tasks")
        
        embeddings = np.array(embeddings)
        
        # Cache embeddings
        np.savez_compressed(
            cache_path,
            embeddings=embeddings,
            task_ids=self.tasks_df['task_id'].values
        )
        logger.info(f"Cached {len(embeddings)} embeddings to {cache_path}")
        
        self.embeddings = embeddings
        return embeddings
    
    def create_middle_level_clusters(self, n_clusters: int = 400) -> Dict[str, Any]:
        """Create middle-level clusters using k-means"""
        logger.info(f"Creating {n_clusters} middle-level clusters...")
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(self.embeddings)
        
        # Get cluster centers
        cluster_centers = kmeans.cluster_centers_
        
        # Organize tasks by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = {
                    'tasks': [],
                    'center': cluster_centers[label],
                    'size': 0
                }
            
            task_info = self.tasks_df.iloc[i].to_dict()
            clusters[label]['tasks'].append(task_info)
            clusters[label]['size'] += 1
        
        logger.info(f"Created {len(clusters)} clusters")
        logger.info(f"Average cluster size: {len(self.tasks_df) / len(clusters):.1f} tasks")
        
        return clusters
    
    def assign_clusters_to_top_level(self, clusters: Dict[int, Dict]) -> Dict[int, Dict]:
        """Assign middle-level clusters to top-level categories"""
        logger.info("Assigning clusters to top-level categories...")
        
        # Create embeddings for top-level categories
        top_level_texts = list(TOP_LEVEL_TASKS.values())
        top_level_embeddings = self.model.encode(top_level_texts)
        
        # For each cluster, find best matching top-level category
        for cluster_id, cluster_data in clusters.items():
            # Use cluster center for assignment
            cluster_center = cluster_data['center']
            
            # Calculate similarities to all top-level categories
            similarities = cosine_similarity([cluster_center], top_level_embeddings)[0]
            best_idx = np.argmax(similarities)
            
            # Assign to top-level category
            top_level_key = list(TOP_LEVEL_TASKS.keys())[best_idx]
            cluster_data['top_level'] = top_level_key
            cluster_data['top_level_similarity'] = float(similarities[best_idx])
        
        # Log distribution
        distribution = {}
        for cluster_data in clusters.values():
            top_level = cluster_data['top_level']
            distribution[top_level] = distribution.get(top_level, 0) + 1
        
        logger.info("Top-level distribution:")
        for key, count in sorted(distribution.items()):
            logger.info(f"  {key}: {count} clusters")
        
        return clusters
    
    def generate_cluster_descriptions(self, clusters: Dict[int, Dict]) -> Dict[int, Dict]:
        """Generate descriptive names for clusters based on their tasks"""
        logger.info("Generating cluster descriptions...")
        
        for cluster_id, cluster_data in clusters.items():
            # Get top 5 most representative tasks (closest to center)
            tasks = cluster_data['tasks']
            task_embeddings = self.embeddings[[self.tasks_df.index[self.tasks_df['task_id'] == t['task_id']].tolist()[0] 
                                              for t in tasks if not self.tasks_df.index[self.tasks_df['task_id'] == t['task_id']].empty]]
            
            if len(task_embeddings) > 0:
                distances = cosine_similarity([cluster_data['center']], task_embeddings)[0]
                top_indices = np.argsort(distances)[-5:][::-1]
                
                representative_tasks = [tasks[i]['Task'] for i in top_indices]
                representative_occupations = list(set([tasks[i]['Title'] for i in top_indices]))
                
                # Create simple description
                cluster_data['description'] = f"Tasks related to: {representative_occupations[0]}"
                cluster_data['representative_tasks'] = representative_tasks[:3]
                cluster_data['occupations'] = representative_occupations
            else:
                cluster_data['description'] = f"Cluster {cluster_id}"
                cluster_data['representative_tasks'] = []
                cluster_data['occupations'] = []
        
        return clusters
    
    def build_hierarchy(self) -> Dict[str, Any]:
        """Build the complete 3-level hierarchy"""
        logger.info("Building complete hierarchy...")
        
        # Load tasks
        self.load_onet_tasks()
        
        # Generate embeddings
        self.generate_embeddings()
        
        # Create middle-level clusters
        clusters = self.create_middle_level_clusters()
        
        # Assign to top-level categories
        clusters = self.assign_clusters_to_top_level(clusters)
        
        # Generate descriptions
        clusters = self.generate_cluster_descriptions(clusters)
        
        # Build hierarchy structure
        hierarchy = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_tasks': len(self.tasks_df),
                'n_clusters': len(clusters),
                'n_top_level': len(TOP_LEVEL_TASKS),
                'test_mode': self.test_mode
            },
            'top_level': TOP_LEVEL_TASKS,
            'middle_level': {},
            'task_lookup': {}  # For efficient task ID -> hierarchy path lookup
        }
        
        # Organize by top-level categories
        for top_key in TOP_LEVEL_TASKS:
            hierarchy['middle_level'][top_key] = []
        
        # Add clusters to hierarchy
        for cluster_id, cluster_data in clusters.items():
            top_level = cluster_data['top_level']
            
            cluster_info = {
                'cluster_id': int(cluster_id),
                'description': cluster_data['description'],
                'size': cluster_data['size'],
                'representative_tasks': cluster_data['representative_tasks'],
                'occupations': cluster_data['occupations'],
                'task_ids': [t['task_id'] for t in cluster_data['tasks']]
            }
            
            hierarchy['middle_level'][top_level].append(cluster_info)
            
            # Build lookup table
            for task in cluster_data['tasks']:
                hierarchy['task_lookup'][task['task_id']] = {
                    'top_level': top_level,
                    'cluster_id': int(cluster_id),
                    'task': task['Task'],
                    'occupation': task['Title']
                }
        
        self.hierarchy = hierarchy
        return hierarchy
    
    def save_hierarchy(self):
        """Save hierarchy to JSON file"""
        if self.hierarchy is None:
            logger.error("No hierarchy to save")
            return
        
        with open(self.hierarchy_file, 'w') as f:
            json.dump(self.hierarchy, f, indent=2)
        
        logger.info(f"Saved hierarchy to {self.hierarchy_file}")
        
        # Log summary statistics
        total_tasks = sum(len(clusters) for clusters in self.hierarchy['middle_level'].values())
        logger.info("\nHierarchy Summary:")
        logger.info(f"- Top-level categories: {len(self.hierarchy['top_level'])}")
        logger.info(f"- Middle-level clusters: {sum(len(c) for c in self.hierarchy['middle_level'].values())}")
        logger.info(f"- Total tasks: {self.hierarchy['metadata']['total_tasks']}")
        
        for top_key, top_desc in self.hierarchy['top_level'].items():
            n_clusters = len(self.hierarchy['middle_level'][top_key])
            n_tasks = sum(c['size'] for c in self.hierarchy['middle_level'][top_key])
            logger.info(f"\n{top_key}: {n_clusters} clusters, {n_tasks} tasks")
            logger.info(f"  Description: {top_desc}")

def main():
    parser = argparse.ArgumentParser(description='Build O*NET task hierarchy using embeddings')
    parser.add_argument('--force-rebuild', action='store_true', 
                       help='Force regeneration of embeddings even if cache exists')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode with subset of data')
    parser.add_argument('--n-clusters', type=int, default=400,
                       help='Number of middle-level clusters (default: 400)')
    
    args = parser.parse_args()
    
    # Check if O*NET data exists
    if not Path('conseq_fin_stage4_onet_taskstatements.csv').exists():
        logger.error("O*NET task statements file not found!")
        logger.error("Expected: conseq_fin_stage4_onet_taskstatements.csv")
        return
    
    # Build hierarchy
    builder = ONETTaskHierarchyBuilder(
        force_rebuild=args.force_rebuild,
        test_mode=args.test_mode
    )
    
    hierarchy = builder.build_hierarchy()
    builder.save_hierarchy()
    
    logger.info("\nHierarchy building complete!")

if __name__ == "__main__":
    main()
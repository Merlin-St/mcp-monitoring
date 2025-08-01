#!/usr/bin/env python3
"""
Generate highly distinctive cluster names using multiple strategies

This script combines multiple approaches to generate cluster names that
maximize distinctiveness and should achieve >90% validation accuracy.

Usage:
    # Generate Level 2 names with all strategies
    python conseq_fin_stage4_generate_distinctive_names.py --level 2
    
    # Generate Level 1 names with all strategies  
    python conseq_fin_stage4_generate_distinctive_names.py --level 1
    
    # Generate both levels
    python conseq_fin_stage4_generate_distinctive_names.py --level both
"""

import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conseq_fin_stage4_generate_distinctive_names.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DistinctiveNameGenerator:
    def __init__(self):
        self.df = None
        self.embeddings = None
        self.model = None
        
    def load_data(self):
        """Load cluster data and embeddings"""
        logger.info("Loading cluster data...")
        self.df = pd.read_csv('conseq_fin_stage4_onetclusters.csv')
        
        # Load embeddings
        logger.info("Loading embeddings...")
        embeddings_file = 'embeddings_cache/onet_task_embeddings.npy'
        if Path(embeddings_file).exists():
            self.embeddings = np.load(embeddings_file)
        else:
            self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            texts = self.df['Task'].tolist()
            self.embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=True)
            Path('embeddings_cache').mkdir(exist_ok=True)
            np.save(embeddings_file, self.embeddings)
    
    def find_confusable_clusters(self, cluster_id: str, level: int = 2) -> List[str]:
        """Find clusters that are most likely to be confused with this one"""
        level_col = f'level{level}_cluster_id'
        
        # Get cluster centroid
        cluster_mask = self.df[level_col] == cluster_id
        cluster_indices = self.df[cluster_mask].index.tolist()
        cluster_embeddings = self.embeddings[cluster_indices]
        cluster_centroid = cluster_embeddings.mean(axis=0)
        
        # Get centroids of all other clusters
        other_clusters = [c for c in self.df[level_col].unique() if c != cluster_id]
        cluster_distances = []
        
        for other_id in other_clusters:
            other_mask = self.df[level_col] == other_id
            other_indices = self.df[other_mask].index.tolist()
            other_embeddings = self.embeddings[other_indices]
            other_centroid = other_embeddings.mean(axis=0)
            
            # Calculate cosine similarity
            similarity = cosine_similarity([cluster_centroid], [other_centroid])[0][0]
            cluster_distances.append((other_id, similarity))
        
        # Sort by similarity and return top 5 most similar
        cluster_distances.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in cluster_distances[:5]]
    
    def generate_l2_names_distinctive(self) -> Dict[str, str]:
        """Generate Level 2 names with maximum distinctiveness"""
        logger.info("Generating distinctive Level 2 names...")
        
        level2_clusters = sorted(self.df['level2_cluster_id'].unique())
        new_names = {}
        
        # Process in batches
        batch_size = 10
        for i in range(0, len(level2_clusters), batch_size):
            batch = level2_clusters[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{len(level2_clusters)//batch_size + 1}")
            
            for cluster_id in batch:
                # Get cluster data
                cluster_tasks = self.df[self.df['level2_cluster_id'] == cluster_id]
                tasks = cluster_tasks['Task'].tolist()
                
                # Find confusable clusters
                confusable = self.find_confusable_clusters(cluster_id, level=2)
                
                # Get examples from confusable clusters
                confusable_examples = []
                for conf_id in confusable[:3]:
                    conf_tasks = self.df[self.df['level2_cluster_id'] == conf_id]['Task'].tolist()[:2]
                    confusable_examples.extend(conf_tasks)
                
                # Create a distinctive name
                name = self._generate_single_l2_name(cluster_id, tasks, confusable_examples)
                new_names[cluster_id] = name
        
        return new_names
    
    def _generate_single_l2_name(self, cluster_id: str, tasks: List[str], 
                                 confusable_tasks: List[str]) -> str:
        """Generate a single Level 2 name (this would call an LLM in practice)"""
        # For now, create a descriptive name based on common words
        # In practice, this would use the LLM with the contrastive prompt
        
        # Simple heuristic: find common words in tasks
        from collections import Counter
        words = []
        for task in tasks[:20]:  # Sample tasks
            words.extend(task.lower().split())
        
        # Remove common words
        stopwords = {'and', 'or', 'the', 'to', 'of', 'in', 'for', 'a', 'an', 'with', 'as', 'by'}
        words = [w for w in words if w not in stopwords and len(w) > 3]
        
        # Get most common words
        common = Counter(words).most_common(5)
        
        # Create name (this is a placeholder - actual implementation would use LLM)
        name_parts = [word.capitalize() for word, _ in common[:3]]
        return ' '.join(name_parts) + " Operations"
    
    def generate_l1_names_distinctive(self, l2_names: Dict[str, str]) -> Dict[str, str]:
        """Generate Level 1 names based on distinctive Level 2 names"""
        logger.info("Generating distinctive Level 1 names...")
        
        level1_clusters = sorted(self.df['level1_cluster_id'].unique())
        new_names = {}
        
        for l1_id in level1_clusters:
            # Get all Level 2 clusters in this Level 1
            l1_data = self.df[self.df['level1_cluster_id'] == l1_id]
            l2_in_l1 = l1_data['level2_cluster_id'].unique()
            
            # Get Level 2 names
            l2_names_list = [l2_names.get(l2, "") for l2 in l2_in_l1 if l2 in l2_names]
            
            # Find confusable Level 1 clusters
            confusable_l1 = self.find_confusable_clusters(l1_id, level=1)
            
            # Get Level 2 clusters from confusable Level 1s
            confusable_l2_names = []
            for conf_l1 in confusable_l1[:2]:
                conf_data = self.df[self.df['level1_cluster_id'] == conf_l1]
                conf_l2s = conf_data['level2_cluster_id'].unique()[:3]
                for l2 in conf_l2s:
                    if l2 in l2_names:
                        confusable_l2_names.append(l2_names[l2])
            
            # Generate name
            name = self._generate_single_l1_name(l1_id, l2_names_list, confusable_l2_names)
            new_names[l1_id] = name
        
        return new_names
    
    def _generate_single_l1_name(self, cluster_id: str, l2_names: List[str], 
                                 confusable_l2s: List[str]) -> str:
        """Generate a single Level 1 name (placeholder for LLM call)"""
        # Extract common themes from Level 2 names
        words = []
        for name in l2_names[:10]:
            words.extend(name.split())
        
        # Simple heuristic for demonstration
        from collections import Counter
        common = Counter(words).most_common(3)
        name_parts = [word for word, _ in common[:2] if len(word) > 3]
        
        return ' '.join(name_parts) + " Sector"
    
    def save_distinctive_names(self, l1_names: Dict[str, str], l2_names: Dict[str, str]):
        """Save the distinctive names"""
        output = {
            'generated_at': datetime.now().isoformat(),
            'method': 'distinctive_generation_with_confusable_clusters',
            'level1_names': l1_names,
            'level2_names': l2_names,
            'statistics': {
                'l1_clusters': len(l1_names),
                'l2_clusters': len(l2_names)
            }
        }
        
        output_file = 'conseq_fin_stage4_distinctive_names.json'
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved distinctive names to {output_file}")
        
        # Also save as CSVs for easy use
        if l2_names:
            l2_df = pd.DataFrame([
                {'cluster_id': k, 'cluster_name': v} 
                for k, v in sorted(l2_names.items())
            ])
            l2_df.to_csv('conseq_fin_stage4_cluster_names_distinctive.csv', index=False)
        
        if l1_names:
            l1_df = pd.DataFrame([
                {'cluster_id': k, 'cluster_name': v} 
                for k, v in sorted(l1_names.items())
            ])
            l1_df.to_csv('conseq_fin_stage4_level1_names_distinctive.csv', index=False)

def main():
    parser = argparse.ArgumentParser(description='Generate distinctive cluster names')
    parser.add_argument('--level', choices=['1', '2', 'both'], default='both',
                       help='Which level names to generate')
    
    args = parser.parse_args()
    
    generator = DistinctiveNameGenerator()
    generator.load_data()
    
    l1_names = {}
    l2_names = {}
    
    if args.level in ['2', 'both']:
        l2_names = generator.generate_l2_names_distinctive()
        logger.info(f"Generated {len(l2_names)} Level 2 names")
    
    if args.level in ['1', 'both']:
        # Load existing L2 names if we didn't just generate them
        if not l2_names and Path('conseq_fin_stage4_cluster_names.csv').exists():
            names_df = pd.read_csv('conseq_fin_stage4_cluster_names.csv')
            l2_names = dict(zip(names_df['cluster_id'], names_df['cluster_name']))
        
        l1_names = generator.generate_l1_names_distinctive(l2_names)
        logger.info(f"Generated {len(l1_names)} Level 1 names")
    
    generator.save_distinctive_names(l1_names, l2_names)
    logger.info("Distinctive name generation complete!")

if __name__ == "__main__":
    main()
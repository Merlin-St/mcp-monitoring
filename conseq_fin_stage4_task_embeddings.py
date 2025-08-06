#!/usr/bin/env python3
"""
Flexible embedding generation and clustering for ONET task hierarchy

This module provides reusable functions for:
- Embedding ONET tasks and clustering them into Level 2 clusters
- Embedding Level 2 cluster names and clustering them into Level 1 clusters
- Managing embedding caches to avoid recomputation
"""

import logging
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model instance (lazy loaded)
_embedding_model = None

# Predefined meaningful Level 1 categories (based on embed_levels.py approach)
LEVEL1_CATEGORIES = {
    "L1_01": "Business management, finance, and customer service operations",
    "L1_02": "Comprehensive healthcare services and medical specialties", 
    "L1_03": "Manage education, HR, and professional development programs",
    "L1_04": "Design, implement, and maintain diverse information technology systems",
    "L1_05": "Operate and manage diverse industrial and agricultural processes",
    "L1_06": "Perform government regulatory enforcement and public safety operations",
    "L1_07": "Conduct scientific research and technical analysis across disciplines",
    "L1_08": "Create and preserve art, culture, and religious artifacts",
    "L1_09": "Coordinate transportation networks and manage logistics supply chains",
    "L1_10": "Manage diverse energy sources and optimize power systems",
    "L1_11": "Design and construct infrastructure projects and engineering systems",
    "L1_12": "Manage and improve environmental systems and sustainability practices"
}

def get_embedding_model():
    """Get or create the shared SentenceTransformer model instance"""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading sentence transformer model...")
        _embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    return _embedding_model

def load_or_generate_embeddings(
    texts: list,
    cache_file: str,
    force_regenerate: bool = False
) -> np.ndarray:
    """Load embeddings from cache or generate new ones"""
    cache_path = Path(cache_file)
    
    if cache_path.exists() and not force_regenerate:
        logger.info(f"Loading cached embeddings from {cache_file}")
        data = np.load(cache_path)
        embeddings = data['embeddings']
        
        # Verify dimensions match
        if len(embeddings) == len(texts):
            return embeddings
        else:
            logger.warning(f"Cache size mismatch: {len(embeddings)} vs {len(texts)} texts")
    
    # Generate new embeddings
    logger.info(f"Generating embeddings for {len(texts)} texts...")
    model = get_embedding_model()
    
    # Process in batches for memory efficiency
    batch_size = 100
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=True)
        embeddings.extend(batch_embeddings)
        
        if i % 1000 == 0 and i > 0:
            logger.info(f"Generated embeddings for {i}/{len(texts)} texts")
    
    embeddings = np.array(embeddings)
    
    # Cache embeddings
    logger.info(f"Caching embeddings to {cache_file}")
    np.savez_compressed(cache_path, embeddings=embeddings)
    
    return embeddings

def cluster_embeddings(embeddings: np.ndarray, k: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform K-means clustering on embeddings
    
    Returns:
        cluster_labels: Array of cluster assignments for each embedding
        cluster_centers: Array of cluster center coordinates
    """
    logger.info(f"Performing K-means clustering with k={k}...")
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    cluster_centers = kmeans.cluster_centers_
    
    # Log cluster statistics
    unique, counts = np.unique(cluster_labels, return_counts=True)
    logger.info(f"Created {len(unique)} clusters")
    logger.info(f"Cluster size - min: {counts.min()}, max: {counts.max()}, mean: {counts.mean():.1f}")
    
    return cluster_labels, cluster_centers

def embed_onet_tasks(
    df: pd.DataFrame,
    text_column: str = 'Task',
    cache_file: str = 'conseq_fin_stage4_task_embeddings_onet.npz',
    k: int = 400,
    force_regenerate: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Embed ONET tasks and cluster them into Level 2 clusters
    
    Args:
        df: DataFrame with ONET tasks
        text_column: Column containing task text
        cache_file: Path to cache embeddings
        k: Number of Level 2 clusters to create
        force_regenerate: Force regeneration of embeddings
        
    Returns:
        embeddings: Task embeddings
        cluster_labels: Level 2 cluster assignments
        cluster_centers: Level 2 cluster centers (for Level 1 assignment)
    """
    logger.info(f"Processing {len(df)} ONET tasks for Level 2 clustering")
    
    # Create embedding text (task + occupation context)
    if 'Title' in df.columns:
        texts = (df[text_column] + " [" + df['Title'] + "]").tolist()
    else:
        texts = df[text_column].tolist()
    
    # Generate or load embeddings
    embeddings = load_or_generate_embeddings(texts, cache_file, force_regenerate)
    
    # Perform clustering
    cluster_labels, cluster_centers = cluster_embeddings(embeddings, k=k)
    
    # Log validation metrics
    log_cluster_validation_metrics(embeddings, cluster_labels, "Level 2 (ONET Tasks)")
    
    return embeddings, cluster_labels, cluster_centers

def assign_clusters_to_level1_categories(
    cluster_centers: np.ndarray,
    level2_clusters: np.ndarray,
    cache_file: str = 'conseq_fin_stage4_level1_embeddings.npz',
    force_regenerate: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Assign Level 2 clusters to Level 1 categories using cosine similarity
    
    Args:
        cluster_centers: Centers of Level 2 clusters  
        level2_clusters: Level 2 cluster indices for each center
        cache_file: Path to cache Level 1 category embeddings
        force_regenerate: Force regeneration of category embeddings
        
    Returns:
        level1_assignments: Array mapping each cluster to Level 1 category
        assignment_details: Dict with similarity scores and category info
    """
    logger.info(f"Assigning {len(cluster_centers)} Level 2 clusters to Level 1 categories using cosine similarity")
    
    # Generate embeddings for Level 1 categories
    category_texts = list(LEVEL1_CATEGORIES.values())
    category_embeddings = load_or_generate_embeddings(category_texts, cache_file, force_regenerate)
    
    # Calculate similarities between cluster centers and category embeddings
    similarities = cosine_similarity(cluster_centers, category_embeddings)
    
    # Assign each cluster to best matching category
    level1_assignments = np.argmax(similarities, axis=1)
    best_similarities = np.max(similarities, axis=1)
    
    # Create detailed assignment info
    category_keys = list(LEVEL1_CATEGORIES.keys())
    assignment_details = {
        'similarities': similarities,
        'best_similarities': best_similarities,
        'assignments': {}
    }
    
    # Log assignment distribution and quality
    logger.info("\n=== Level 1 Category Assignment Results ===")
    category_counts = {}
    total_similarity = 0
    
    for i, (level2_cluster, assignment, similarity) in enumerate(zip(level2_clusters, level1_assignments, best_similarities)):
        category_key = category_keys[assignment]
        category_name = LEVEL1_CATEGORIES[category_key]
        
        assignment_details['assignments'][int(level2_cluster)] = {
            'level1_cluster': category_key,
            'level1_name': category_name,
            'similarity': float(similarity)
        }
        
        category_counts[category_key] = category_counts.get(category_key, 0) + 1
        total_similarity += similarity
    
    # Log distribution
    logger.info("Category distribution:")
    for category_key, count in sorted(category_counts.items()):
        percentage = (count / len(cluster_centers)) * 100
        logger.info(f"  {category_key}: {count} clusters ({percentage:.1f}%)")
    
    # Log quality metrics
    avg_similarity = total_similarity / len(cluster_centers)
    min_similarity = best_similarities.min()
    max_similarity = best_similarities.max()
    
    logger.info(f"\nAssignment Quality:")
    logger.info(f"  Average similarity: {avg_similarity:.4f}")
    logger.info(f"  Min similarity: {min_similarity:.4f}")
    logger.info(f"  Max similarity: {max_similarity:.4f}")
    
    # Report average similarity by Level 1 category
    logger.info(f"\nAverage Similarity by Level 1 Category:")
    category_similarities = {}
    for category_key in category_keys:
        category_sims = []
        for level2_cluster, assignment, similarity in zip(level2_clusters, level1_assignments, best_similarities):
            if category_keys[assignment] == category_key:
                category_sims.append(similarity)
        
        if category_sims:
            avg_cat_sim = np.mean(category_sims)
            category_similarities[category_key] = avg_cat_sim
            category_name = LEVEL1_CATEGORIES[category_key]
            logger.info(f"  {category_key}: {avg_cat_sim:.4f} ({len(category_sims)} clusters) - {category_name}")
    
    # Report 20 lowest similarity assignments (poorest fits)
    logger.info(f"\n20 Lowest Similarity Assignments (Poorest Category Fits):")
    low_similarity_indices = np.argsort(best_similarities)[:20]
    for i, idx in enumerate(low_similarity_indices):
        level2_cluster = level2_clusters[idx]
        assignment_idx = level1_assignments[idx]
        similarity = best_similarities[idx]
        category_key = category_keys[assignment_idx]
        category_name = LEVEL1_CATEGORIES[category_key]
        logger.info(f"  {i+1:2d}. Cluster {level2_cluster}: {similarity:.4f} → {category_key} ({category_name})")
    
    if avg_similarity > 0.7:
        logger.info("✓ EXCELLENT: Very strong category-cluster alignment")
    elif avg_similarity > 0.5:
        logger.info("✓ GOOD: Strong category-cluster alignment")  
    elif avg_similarity > 0.3:
        logger.info("⚠ FAIR: Moderate category-cluster alignment")
    else:
        logger.info("✗ POOR: Weak category-cluster alignment")
    
    # Store additional metrics in assignment_details
    assignment_details.update({
        'avg_similarity': float(avg_similarity),
        'min_similarity': float(min_similarity),
        'max_similarity': float(max_similarity),
        'category_avg_similarities': {k: float(v) for k, v in category_similarities.items()},
        'lowest_similarity_clusters': [
            {
                'level2_cluster': int(level2_clusters[idx]),
                'similarity': float(best_similarities[idx]),
                'assigned_category': category_keys[level1_assignments[idx]],
                'category_name': LEVEL1_CATEGORIES[category_keys[level1_assignments[idx]]]
            }
            for idx in low_similarity_indices
        ]
    })
    
    logger.info("=== End Level 1 Assignment ===\n")
    
    return level1_assignments, assignment_details


def build_two_level_hierarchy(
    df: pd.DataFrame,
    text_column: str = 'Task',
    level2_clusters: int = 400,
    force_regenerate: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build complete 2-level hierarchy using improved approach
    
    Args:
        df: DataFrame with ONET tasks
        text_column: Column containing task text  
        level2_clusters: Number of Level 2 clusters
        force_regenerate: Force regeneration of all embeddings
        
    Returns:
        enhanced_df: DataFrame with Level 1 and Level 2 assignments
        metadata: Dictionary with hierarchy metadata and statistics
    """
    logger.info("=== Building 2-Level Task Hierarchy (Improved Approach) ===")
    
    # Step 1: Create Level 2 clusters from task embeddings
    embeddings, level2_labels, cluster_centers = embed_onet_tasks(
        df, text_column, k=level2_clusters, force_regenerate=force_regenerate
    )
    
    # Step 2: Assign Level 2 clusters to Level 1 categories using cosine similarity
    level2_cluster_indices = np.arange(level2_clusters)
    level1_assignments, assignment_details = assign_clusters_to_level1_categories(
        cluster_centers, level2_cluster_indices, force_regenerate=force_regenerate
    )
    
    # Step 3: Create enhanced DataFrame with both levels
    enhanced_df = df.copy()
    enhanced_df['level2_cluster'] = level2_labels
    
    # Map each task to its Level 1 category
    level1_mapping = {}
    for level2_cluster, details in assignment_details['assignments'].items():
        level1_mapping[level2_cluster] = details['level1_cluster']
        
    enhanced_df['level1_cluster'] = enhanced_df['level2_cluster'].map(level1_mapping)
    enhanced_df['level1_name'] = enhanced_df['level1_cluster'].map(LEVEL1_CATEGORIES)
    
    # Step 4: Create metadata
    metadata = {
        'total_tasks': len(df),
        'level2_clusters': level2_clusters,
        'level1_categories': len(LEVEL1_CATEGORIES),
        'assignment_details': assignment_details,
        'level1_categories': LEVEL1_CATEGORIES
    }
    
    logger.info("=== Hierarchy Building Complete ===")
    logger.info(f"Tasks: {len(df)}, Level 2 Clusters: {level2_clusters}, Level 1 Categories: {len(LEVEL1_CATEGORIES)}")
    
    return enhanced_df, metadata

def get_cluster_statistics(df: pd.DataFrame, cluster_column: str) -> Dict[str, Any]:
    """Generate statistics for a clustering result"""
    cluster_counts = df[cluster_column].value_counts().sort_index()
    
    stats = {
        'n_clusters': len(cluster_counts),
        'min_size': cluster_counts.min(),
        'max_size': cluster_counts.max(),
        'mean_size': cluster_counts.mean(),
        'std_size': cluster_counts.std(),
        'distribution': cluster_counts.to_dict()
    }
    
    return stats

def calculate_cluster_consistency_score(
    embeddings: np.ndarray, 
    cluster_labels: np.ndarray,
    method: str = 'silhouette'
) -> float:
    """
    Calculate cluster consistency/quality score
    
    Args:
        embeddings: The embeddings used for clustering
        cluster_labels: Cluster assignments
        method: 'silhouette' or 'intra_cluster_distance'
        
    Returns:
        consistency_score: Higher is better (0-1 for silhouette, lower is better for distance)
    """
    if method == 'silhouette':
        from sklearn.metrics import silhouette_score
        if len(set(cluster_labels)) > 1:
            score = silhouette_score(embeddings, cluster_labels)
            logger.info(f"Silhouette Score: {score:.4f} (range: -1 to 1, higher is better)")
            return score
        else:
            logger.warning("Only one cluster found, cannot calculate silhouette score")
            return 0.0
    
    elif method == 'intra_cluster_distance':
        from sklearn.metrics.pairwise import cosine_distances
        # Calculate average intra-cluster distance
        total_distance = 0
        total_pairs = 0
        
        for level2_cluster in set(cluster_labels):
            cluster_mask = cluster_labels == level2_cluster
            cluster_embeddings = embeddings[cluster_mask]
            
            if len(cluster_embeddings) > 1:
                distances = cosine_distances(cluster_embeddings)
                # Get upper triangle (avoid duplicates and self-distances)
                n = len(cluster_embeddings)
                upper_triangle = distances[np.triu_indices(n, k=1)]
                total_distance += upper_triangle.sum()
                total_pairs += len(upper_triangle)
        
        if total_pairs > 0:
            avg_distance = total_distance / total_pairs
            logger.info(f"Average Intra-cluster Distance: {avg_distance:.4f} (lower is better)")
            return avg_distance
        else:
            logger.warning("No cluster pairs found for distance calculation")
            return 1.0
    
    else:
        raise ValueError(f"Unknown method: {method}")

def log_cluster_validation_metrics(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    level_name: str = "Level"
):
    """Log comprehensive cluster validation metrics"""
    logger.info(f"\n=== {level_name} Cluster Validation Metrics ===")
    
    # Basic statistics
    unique_labels = set(cluster_labels)
    n_clusters = len(unique_labels)
    logger.info(f"Number of clusters: {n_clusters}")
    
    # Cluster size distribution
    cluster_counts = np.bincount(cluster_labels)
    logger.info(f"Cluster sizes - Min: {cluster_counts.min()}, Max: {cluster_counts.max()}, Mean: {cluster_counts.mean():.1f}")
    
    # Silhouette score
    silhouette = calculate_cluster_consistency_score(embeddings, cluster_labels, 'silhouette')
    
    # Intra-cluster distance
    intra_distance = calculate_cluster_consistency_score(embeddings, cluster_labels, 'intra_cluster_distance')
    
    # Log interpretation
    if silhouette > 0.5:
        logger.info("✓ GOOD: Strong cluster separation")
    elif silhouette > 0.25:
        logger.info("⚠ FAIR: Moderate cluster separation")
    else:
        logger.info("✗ POOR: Weak cluster separation")
    
    logger.info(f"=== End {level_name} Validation ===\n")
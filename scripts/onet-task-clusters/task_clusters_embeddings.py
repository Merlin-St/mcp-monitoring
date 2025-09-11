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
from typing import Tuple, Dict, Any
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model instance (lazy loaded)
_embedding_model = None

# Centralized model configuration - CHANGE HERE TO USE DIFFERENT MODEL
EMBEDDING_MODEL_NAME = 'NovaSearch/stella_en_400M_v5'  # 1024 dimensions, consistent across all embeddings

# GPU optimization setup
def setup_gpu_optimizations():
    """Configure GPU for maximum performance"""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        # Clear GPU cache
        torch.cuda.empty_cache()
        # Set to use TensorCores
        torch.set_float32_matmul_precision('medium')
    return torch.cuda.is_available()

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
    """Get or create the shared SentenceTransformer model instance with GPU optimization"""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading sentence transformer model...")
        
        # Set memory optimization environment variable
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Setup GPU optimizations
        has_gpu = setup_gpu_optimizations()
        device = 'cuda' if has_gpu else 'cpu'
        
        logger.info(f"Using device: {device}")
        logger.info(f"Using embedding model: {EMBEDDING_MODEL_NAME}")
        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device=device,
            trust_remote_code=True
        )
        
        if has_gpu:
            logger.info("GPU optimizations enabled")
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
    
    # Process in batches for memory efficiency (optimized for GPU)
    batch_size = 64  # GPU-optimized batch size
    embeddings = []
    
    # Clear GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
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
    cache_file: str = 'embeddings_cache/task_clusters_embeddings_onet.npz',
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
    cache_file: str = 'embeddings_cache/level1_embeddings.npz',
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
    
    logger.info("\nAssignment Quality:")
    logger.info(f"  Average similarity: {avg_similarity:.4f}")
    logger.info(f"  Min similarity: {min_similarity:.4f}")
    logger.info(f"  Max similarity: {max_similarity:.4f}")
    
    # Report average similarity by Level 1 category
    logger.info("\nAverage Similarity by Level 1 Category:")
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
    logger.info("\n20 Lowest Similarity Assignments (Poorest Category Fits):")
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


def cluster_level2_to_natural_level1(
    cluster_centers: np.ndarray,
    level2_clusters: np.ndarray,
    min_cluster_size: int = 10
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Cluster Level 2 cluster centers into natural Level 1 clusters using HDBSCAN
    
    Args:
        cluster_centers: Centers of Level 2 clusters  
        level2_clusters: Level 2 cluster indices for each center
        min_cluster_size: Minimum size for Level 1 clusters
        
    Returns:
        level1_assignments: Array mapping each Level 2 cluster to Level 1 category
        clustering_details: Dict with clustering info and statistics
    """
    logger.info(f"Creating natural Level 1 clusters from {len(cluster_centers)} Level 2 cluster centers using HDBSCAN")
    
    # Use HDBSCAN to find natural groupings among cluster centers
    import hdbscan
    
    # Scale min_cluster_size based on number of L2 clusters - aim for ~8-15 L1 clusters
    target_l1_clusters = min(15, max(8, len(cluster_centers) // 10))  # More aggressive division
    adjusted_min_size = max(2, len(cluster_centers) // target_l1_clusters)
    
    logger.info(f"Using HDBSCAN with min_cluster_size={adjusted_min_size}, targeting ~{target_l1_clusters} L1 clusters")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=adjusted_min_size,
        min_samples=max(1, adjusted_min_size // 3),  # Reduced min_samples
        metric='euclidean',
        cluster_selection_epsilon=0.005,  # Smaller epsilon for tighter clusters
        prediction_data=True,
        core_dist_n_jobs=-1,
        algorithm='prims_kdtree'
    )
    
    level1_labels = clusterer.fit_predict(cluster_centers)
    
    # Handle outliers by assigning them to nearest cluster
    outlier_mask = level1_labels == -1
    n_outliers = outlier_mask.sum()
    
    if n_outliers > 0:
        logger.info(f"Found {n_outliers} outlier Level 2 clusters, assigning to nearest Level 1 cluster")
        from sklearn.metrics.pairwise import euclidean_distances
        
        # Get valid cluster centers (non-outliers)
        valid_mask = level1_labels != -1
        if valid_mask.sum() > 0:
            valid_centers = cluster_centers[valid_mask]
            valid_labels = level1_labels[valid_mask]
            
            # For each outlier, find closest valid cluster center
            outlier_centers = cluster_centers[outlier_mask]
            distances = euclidean_distances(outlier_centers, valid_centers)
            closest_indices = distances.argmin(axis=1)
            
            # Assign outliers to closest cluster's label
            level1_labels[outlier_mask] = valid_labels[closest_indices]
        else:
            # If all are outliers, create a single cluster
            logger.warning("All Level 2 clusters were outliers, creating single Level 1 cluster")
            level1_labels[:] = 0
    
    # Create clustering details
    unique_labels = np.unique(level1_labels)
    n_l1_clusters = len(unique_labels)
    
    clustering_details = {
        'n_level1_clusters': n_l1_clusters,
        'level1_labels': level1_labels,
        'cluster_sizes': {},
        'assignments': {}
    }
    
    # Log distribution and create assignments
    logger.info("\n=== Natural Level 1 Clustering Results ===")
    logger.info(f"Created {n_l1_clusters} Level 1 clusters from {len(cluster_centers)} Level 2 clusters")
    
    for l1_cluster in unique_labels:
        l2_clusters_in_l1 = level2_clusters[level1_labels == l1_cluster]
        cluster_size = len(l2_clusters_in_l1)
        clustering_details['cluster_sizes'][int(l1_cluster)] = cluster_size
        
        logger.info(f"  L1 Cluster {l1_cluster}: {cluster_size} Level 2 clusters")
        
        # Create assignments for each L2 cluster in this L1 cluster
        for l2_cluster in l2_clusters_in_l1:
            clustering_details['assignments'][int(l2_cluster)] = {
                'level1_cluster': f'L1_Natural_{l1_cluster:02d}',
                'level1_name': f'Natural Cluster {l1_cluster}',
                'clustering_method': 'hdbscan'
            }
    
    logger.info("=== End Natural L1 Clustering ===\n")
    
    return level1_labels, clustering_details


def build_two_level_hierarchy(
    df: pd.DataFrame,
    text_column: str = 'Task',
    level2_clusters: int = 400,
    l1_approach: str = 'semantic',
    force_regenerate: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build complete 2-level hierarchy using chosen approach
    
    Args:
        df: DataFrame with ONET tasks
        text_column: Column containing task text  
        level2_clusters: Number of Level 2 clusters
        l1_approach: 'semantic' for predefined categories, 'natural' for HDBSCAN clustering
        force_regenerate: Force regeneration of all embeddings
        
    Returns:
        enhanced_df: DataFrame with Level 1 and Level 2 assignments
        metadata: Dictionary with hierarchy metadata and statistics
    """
    logger.info(f"=== Building 2-Level Task Hierarchy ({l1_approach} L1 approach) ===")
    
    # Step 1: Create Level 2 clusters from task embeddings
    embeddings, level2_labels, cluster_centers = embed_onet_tasks(
        df, text_column, k=level2_clusters, force_regenerate=force_regenerate
    )
    
    # Step 2: Create Level 1 clusters/categories using chosen approach
    level2_cluster_indices = np.arange(level2_clusters)
    
    if l1_approach == 'natural':
        # Use natural HDBSCAN clustering of Level 2 cluster centers
        level1_assignments, assignment_details = cluster_level2_to_natural_level1(
            cluster_centers, level2_cluster_indices
        )
    else:
        # Use semantic assignment to predefined categories (default)
        level1_assignments, assignment_details = assign_clusters_to_level1_categories(
            cluster_centers, level2_cluster_indices, force_regenerate=force_regenerate
        )
    
    # Step 3: Create enhanced DataFrame with both levels
    enhanced_df = df.copy()
    enhanced_df['level2_cluster'] = level2_labels
    
    # Map each task to its Level 1 category
    level1_mapping = {}
    level1_name_mapping = {}
    for level2_cluster, details in assignment_details['assignments'].items():
        level1_mapping[level2_cluster] = details['level1_cluster']
        level1_name_mapping[details['level1_cluster']] = details['level1_name']
        
    enhanced_df['level1_cluster'] = enhanced_df['level2_cluster'].map(level1_mapping)
    
    if l1_approach == 'natural':
        enhanced_df['level1_name'] = enhanced_df['level1_cluster'].map(level1_name_mapping)
    else:
        enhanced_df['level1_name'] = enhanced_df['level1_cluster'].map(LEVEL1_CATEGORIES)
    
    # Step 4: Create metadata
    if l1_approach == 'natural':
        l1_count = assignment_details['n_level1_clusters']
        metadata = {
            'total_tasks': len(df),
            'level2_clusters': level2_clusters,
            'level1_categories': l1_count,
            'assignment_details': assignment_details,
            'l1_approach': 'natural_hdbscan'
        }
        logger.info("=== Hierarchy Building Complete ===")
        logger.info(f"Tasks: {len(df)}, Level 2 Clusters: {level2_clusters}, Level 1 Clusters: {l1_count} (natural)")
    else:
        metadata = {
            'total_tasks': len(df),
            'level2_clusters': level2_clusters,
            'level1_categories': len(LEVEL1_CATEGORIES),
            'assignment_details': assignment_details,
            'level1_categories_definitions': LEVEL1_CATEGORIES,
            'l1_approach': 'semantic_cosine'
        }
        logger.info("=== Hierarchy Building Complete ===")
        logger.info(f"Tasks: {len(df)}, Level 2 Clusters: {level2_clusters}, Level 1 Categories: {len(LEVEL1_CATEGORIES)} (semantic)")
    
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
    
    
    # Log interpretation
    if silhouette > 0.5:
        logger.info("✓ GOOD: Strong cluster separation")
    elif silhouette > 0.25:
        logger.info("⚠ FAIR: Moderate cluster separation")
    else:
        logger.info("✗ POOR: Weak cluster separation")
    
    logger.info(f"=== End {level_name} Validation ===\n")
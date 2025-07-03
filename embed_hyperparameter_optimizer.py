#!/usr/bin/env python3
"""
Hyperparameter Optimization Script for embed_generate.py
Automatically tests different parameter combinations to minimize outliers and maximize coherence.
"""

import json
import logging
import os
import time
import argparse
from itertools import product
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import the original functions from embed_generate
from embed_generate import (
    generate_high_quality_embeddings,
    create_bertopic_model,
    calculate_topic_coherence,
    prepare_texts_parallel,
    setup_gpu_optimizations
)
from naics_classification_config import NAICS_SECTORS, NAICS_KEYWORDS

# Configuration constants
MIN_TOPICS_REQUIRED = 45  # Minimum number of topics required for optimization

@dataclass
class OptimizationResult:
    """Results from a single hyperparameter combination"""
    params: Dict[str, Any]
    train_outlier_pct: float
    test_outlier_pct: float
    train_coherence: float
    test_coherence: float
    num_topics: int
    avg_outlier_pct: float
    avg_coherence: float
    stability_score: float
    execution_time: float
    error: Optional[str] = None

class HyperparameterOptimizer:
    """Automated hyperparameter optimization for BERTopic models"""
    
    def __init__(self, data_file: str = 'data_unified_filtered.json', 
                 test_size: int = 1000, cache_embeddings: bool = True,
                 sector_filter: Optional[int] = None, include_kmeans: bool = False,
                 min_topics_sector: int = 10, min_topics_full: int = MIN_TOPICS_REQUIRED):
        self.data_file = data_file
        self.test_size = test_size
        self.cache_embeddings = cache_embeddings
        self.sector_filter = sector_filter
        self.include_kmeans = include_kmeans
        self.min_topics_sector = min_topics_sector
        self.min_topics_full = min_topics_full
        self.logger = self._setup_logging()
        self.results: List[OptimizationResult] = []
        self.rejected_count = 0  # Track rejected combinations due to insufficient topics
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        suffix = f"_sector_{self.sector_filter}" if self.sector_filter else ""
        log_filename = f'embed_hyperparameter_optimization{suffix}.log'
        
        # Clear existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # File handler
        file_handler = logging.FileHandler(log_filename, mode='w')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        
        # Configure logger
        logger = logging.getLogger('hyperopt')
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.propagate = False
        
        return logger
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load data and generate embeddings"""
        self.logger.info(f"Loading data from {self.data_file}")
        
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                servers_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file {self.data_file} not found")
        
        # Filter for sector-specific servers if specified
        if self.sector_filter:
            original_count = len(servers_data)
            sector_attr = f'is_sector_{self.sector_filter}'
            sector_name = NAICS_SECTORS.get(self.sector_filter, f"Sector {self.sector_filter}")
            
            servers_data = [server for server in servers_data if server.get(sector_attr, False)]
            self.logger.info(f"SECTOR {self.sector_filter} MODE: Filtered to {len(servers_data)} servers in {sector_name} from {original_count} total")
            
            if len(servers_data) == 0:
                raise ValueError(f"No servers found for sector {self.sector_filter} ({sector_name}) in the dataset.")
        
        # Use test subset if specified
        if self.test_size and self.test_size < len(servers_data):
            servers_data = servers_data[:self.test_size]
            sector_info = f" (sector {self.sector_filter})" if self.sector_filter else ""
            self.logger.info(f"Using test subset of {len(servers_data)} servers{sector_info}")
        
        # Prepare texts
        prepared_texts = prepare_texts_parallel(servers_data)
        df = pd.DataFrame(prepared_texts)
        
        if len(df) < 35:
            raise ValueError(f"Dataset too small: {len(df)} samples (need at least 35)")
        
        self.logger.info(f"Prepared {len(df)} servers for optimization")
        
        # Generate embeddings
        texts = df['text'].tolist()
        cache_dir = 'embeddings_cache' if self.cache_embeddings else None
        
        self.logger.info("Generating embeddings...")
        embeddings = generate_high_quality_embeddings(
            texts, 
            batch_size=64,
            cache_dir=cache_dir
        )
        
        self.logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return df, embeddings
    
    def validate_vectorizer_params(self, dataset_size: int, min_df: int, max_df: float) -> bool:
        """Validate that vectorizer parameters won't cause constraint violations"""
        # For very small datasets (test splits), use very lenient validation
        if dataset_size < 25:
            # For tiny datasets, only use min_df=1 to avoid constraint violations
            return min_df == 1 and max_df >= 0.8
        
        if dataset_size < 50:
            # For small datasets, be more lenient
            max_df_docs = int(dataset_size * max_df)
            # Ensure basic constraint satisfaction and don't be too strict on min_df
            return max_df_docs >= min_df and min_df <= max(2, dataset_size // 10)
        
        # For BERTopic, we need to account for potential reduction in document count
        # BERTopic creates "documents per topic" which can reduce effective document count
        # Use conservative estimate: assume 50% reduction in document count during processing
        effective_docs = max(1, int(dataset_size * 0.5))
        
        # Calculate actual thresholds
        max_df_docs = int(effective_docs * max_df)
        
        # Check constraint: max_df_docs >= min_df
        if max_df_docs < min_df:
            return False
            
        # Additional safety check: ensure reasonable ranges
        if min_df >= dataset_size * 0.1:  # min_df shouldn't be more than 10% of dataset
            return False
            
        if max_df <= 0.5:  # max_df should allow at least 50% of terms
            return False
            
        return True

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        """Define parameter grid for optimization"""
        # Enhanced grid with parameters that favor higher topic counts
        grid = {
            # UMAP parameters
            'n_neighbors': [5, 10, 15, 20, 30],
            'min_dist': [0.01, 0.05, 0.1, 0.2, 0.3],
            'n_components_clustering': [3, 5, 7],
            
            # HDBSCAN parameters - optimized for 40-60 topics range
            'min_cluster_size_factor': [0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.01],  # Smaller factors for more numerous topics
            'cluster_selection_epsilon': [0.003, 0.005, 0.008, 0.01, 0.015, 0.02],  # Smaller epsilons for more distinct clusters
            'min_samples_factor': [0.1, 0.2, 0.3, 0.4, 0.5],  # Include very small min_samples for maximum flexibility
            
            # Vectorizer parameters - optimized for 40-60 topic diversity
            'max_features': [800, 1000, 1500, 2000],  # Higher feature counts for richer topic vocabulary
            'max_df': [0.6, 0.7, 0.9],  # Higher values to retain more terms
            'min_df': [1, 2, 3],  # Include slightly higher min_df for better topic quality
            
            # Clustering algorithm (HDBSCAN by default, K-means if requested)
            'clustering_algorithm': ['hdbscan']
        }
        
        # Add K-means if requested (note: K-means doesn't produce outliers)
        if self.include_kmeans:
            grid['clustering_algorithm'].append('kmeans')
            
        return grid
    
    def create_optimized_bertopic_model(self, embeddings: np.ndarray, texts: List[str], 
                                      params: Dict[str, Any]) -> Tuple[Any, List[int], Any]:
        """Create BERTopic model with specified parameters"""
        import umap
        import hdbscan
        from bertopic import BERTopic
        from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
        from sklearn.cluster import KMeans
        
        dataset_size = len(texts)
        
        # Validate and adjust parameters
        n_neighbors = min(params['n_neighbors'], dataset_size - 1)
        n_neighbors = max(2, n_neighbors)
        
        min_cluster_size = max(5, int(dataset_size * params['min_cluster_size_factor']))
        min_samples = max(1, int(min_cluster_size * params['min_samples_factor']))
        
        # UMAP model
        try:
            from cuml.manifold import UMAP as cumlUMAP
            umap_model = cumlUMAP(
                n_neighbors=n_neighbors,
                min_dist=params['min_dist'],
                n_components=params['n_components_clustering'],
                metric='cosine',
                random_state=42,
                verbose=False
            )
        except ImportError:
            umap_model = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=params['min_dist'],
                n_components=params['n_components_clustering'],
                metric='cosine',
                random_state=42,
                verbose=False,
                n_jobs=-1,
                low_memory=False
            )
        
        # Clustering model
        if params['clustering_algorithm'] == 'kmeans':
            n_clusters = min(30, max(5, dataset_size // 40))
            cluster_model = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
                max_iter=300
            )
            nr_topics = n_clusters
        else:
            try:
                from cuml.cluster import HDBSCAN as cumlHDBSCAN
                cluster_model = cumlHDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric='euclidean',
                    cluster_selection_epsilon=params['cluster_selection_epsilon'],
                    cluster_selection_method='eom'
                )
            except ImportError:
                cluster_model = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric='euclidean',
                    cluster_selection_epsilon=params['cluster_selection_epsilon'],
                    prediction_data=True,
                    core_dist_n_jobs=-1,
                    algorithm='prims_kdtree'
                )
            nr_topics = 'auto'
        
        # Custom stop words
        custom_stop_words = [
            'api', 'demo', 'github', 'protocol', 'servers', 'mcp', 'openai', 'stdio', 'prompt',
            'server', 'tool', 'tools', 'service', 'client', 'function', 'functions', 'endpoint',
            'integration', 'plugin', 'extension', 'library', 'framework', 'sdk', 'connector',
            'interface', 'wrapper', 'bridge', 'adapter', 'manager', 'handler', 'provider',
            'assistant', 'chatgpt', 'gpt', 'claude', 'llm', 'ai', 'model', 'models',
            'request', 'response', 'http', 'https', 'json', 'xml', 'rest', 'graphql',
            'python', 'javascript', 'typescript', 'node', 'npm', 'pip', 'install',
            'example', 'sample', 'test', 'testing', 'mock', 'stub', 'dummy',
            'context', 'protocol', 'based', 'using', 'allows', 'provides', 'enables'
        ]
        
        all_stop_words = list(ENGLISH_STOP_WORDS) + custom_stop_words
        
        # Validate vectorizer parameters one more time in model creation
        if not self.validate_vectorizer_params(dataset_size, params['min_df'], params['max_df']):
            raise ValueError(f"Invalid vectorizer params: min_df={params['min_df']}, max_df={params['max_df']} for {dataset_size} documents")
        
        # Vectorizer
        vectorizer_model = TfidfVectorizer(
            max_features=params['max_features'],
            ngram_range=(1, 2),
            stop_words=all_stop_words,
            min_df=params['min_df'],
            max_df=params['max_df'],
            token_pattern=r'\b[a-zA-Z]{3,}\b',
            sublinear_tf=True,
            norm='l2',
            use_idf=True,
            smooth_idf=True
        )
        
        # Create BERTopic model
        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=cluster_model,
            vectorizer_model=vectorizer_model,
            verbose=False,
            calculate_probabilities=False,
            nr_topics=nr_topics,
            low_memory=True,
            min_topic_size=max(10, dataset_size // 35),
        )
        
        # Fit model
        topics, probs = topic_model.fit_transform(texts, embeddings)
        
        return topic_model, topics, probs
    
    def evaluate_parameters(self, df: pd.DataFrame, embeddings: np.ndarray, 
                          params: Dict[str, Any]) -> OptimizationResult:
        """Evaluate a single parameter combination"""
        start_time = time.time()
        
        try:
            # Validate vectorizer parameters first
            dataset_size = len(df)
            if not self.validate_vectorizer_params(dataset_size, params['min_df'], params['max_df']):
                execution_time = time.time() - start_time
                return OptimizationResult(
                    params=params.copy(),
                    train_outlier_pct=100.0,
                    test_outlier_pct=100.0,
                    train_coherence=0.0,
                    test_coherence=0.0,
                    num_topics=0,
                    avg_outlier_pct=100.0,
                    avg_coherence=0.0,
                    stability_score=-3.0,  # Very low score for invalid vectorizer params
                    execution_time=execution_time,
                    error=f"Invalid vectorizer params: min_df={params['min_df']}, max_df={params['max_df']} for {dataset_size} documents"
                )
            # Split data
            indices = range(len(df))
            train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
            
            df_train = df.iloc[train_indices].reset_index(drop=True)
            df_test = df.iloc[test_indices].reset_index(drop=True)
            
            embeddings_train = embeddings[train_indices]
            embeddings_test = embeddings[test_indices]
            
            texts_train = df_train['text'].tolist()
            texts_test = df_test['text'].tolist()
            
            # Additional validation: check vectorizer params work on both splits AND full dataset
            full_texts = df['text'].tolist()
            for dataset, name in [(texts_train, "train"), (texts_test, "test"), (full_texts, "full")]:
                if not self.validate_vectorizer_params(len(dataset), params['min_df'], params['max_df']):
                    execution_time = time.time() - start_time
                    return OptimizationResult(
                        params=params.copy(),
                        train_outlier_pct=100.0,
                        test_outlier_pct=100.0,
                        train_coherence=0.0,
                        test_coherence=0.0,
                        num_topics=0,
                        avg_outlier_pct=100.0,
                        avg_coherence=0.0,
                        stability_score=-3.0,
                        execution_time=execution_time,
                        error=f"Invalid vectorizer params for {name} dataset: min_df={params['min_df']}, max_df={params['max_df']} for {len(dataset)} documents"
                    )
            
            # Create and train model
            topic_model, train_topics, _ = self.create_optimized_bertopic_model(
                embeddings_train, texts_train, params
            )
            
            # Transform test set
            test_topics, _ = topic_model.transform(texts_test, embeddings_test)
            
            # Calculate metrics
            train_outliers = np.sum(np.array(train_topics) == -1)
            test_outliers = np.sum(np.array(test_topics) == -1)
            
            train_outlier_pct = (train_outliers / len(texts_train)) * 100
            test_outlier_pct = (test_outliers / len(texts_test)) * 100
            
            # Calculate coherence
            train_coherence, _ = calculate_topic_coherence(topic_model, texts_train, train_topics)
            test_coherence, _ = calculate_topic_coherence(topic_model, texts_test, test_topics)
            
            # Handle None coherence values
            train_coherence = train_coherence if train_coherence is not None else 0.0
            test_coherence = test_coherence if test_coherence is not None else 0.0
            
            # Calculate number of topics
            num_topics = len(set(train_topics)) - (1 if -1 in train_topics else 0)
            
            # Check minimum topic requirement
            min_required = self.min_topics_sector if self.sector_filter else self.min_topics_full
            if num_topics < min_required:
                self.rejected_count += 1
                execution_time = time.time() - start_time
                del topic_model
                return OptimizationResult(
                    params=params.copy(),
                    train_outlier_pct=train_outlier_pct,
                    test_outlier_pct=test_outlier_pct,
                    train_coherence=train_coherence,
                    test_coherence=test_coherence,
                    num_topics=num_topics,
                    avg_outlier_pct=100.0,  # Mark as failed due to insufficient topics
                    avg_coherence=0.0,
                    stability_score=-2.0,  # Very low score for insufficient topics
                    execution_time=execution_time,
                    error=f"Insufficient topics: {num_topics} < {min_required} required"
                )
            
            # Calculate composite scores
            avg_outlier_pct = (train_outlier_pct + test_outlier_pct) / 2
            avg_coherence = (train_coherence + test_coherence) / 2
            
            # Multi-objective stability score: prioritize topic count, then coherence, then outliers
            # Topic count score: reward meeting minimum + bonus for exceeding
            topic_count_score = min(1.0, num_topics / min_required)
            if num_topics > min_required:
                # Bonus for exceeding minimum (up to 20% bonus)
                excess_bonus = min(0.2, (num_topics - min_required) / min_required * 0.1)
                topic_count_score += excess_bonus
            
            # Coherence reward (0-1 scale typically)
            coherence_reward = avg_coherence
            
            # Outlier penalty (0-1 scale)
            outlier_penalty = avg_outlier_pct / 100.0
            
            # Combined score: topic_count * coherence * (1 - outlier_penalty)
            stability_score = topic_count_score * coherence_reward * (1 - outlier_penalty)
            
            execution_time = time.time() - start_time
            
            # Clean up model to free memory
            del topic_model
            
            return OptimizationResult(
                params=params.copy(),
                train_outlier_pct=train_outlier_pct,
                test_outlier_pct=test_outlier_pct,
                train_coherence=train_coherence,
                test_coherence=test_coherence,
                num_topics=num_topics,
                avg_outlier_pct=avg_outlier_pct,
                avg_coherence=avg_coherence,
                stability_score=stability_score,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return OptimizationResult(
                params=params.copy(),
                train_outlier_pct=100.0,
                test_outlier_pct=100.0,
                train_coherence=0.0,
                test_coherence=0.0,
                num_topics=0,
                avg_outlier_pct=100.0,
                avg_coherence=0.0,
                stability_score=-1.0,
                execution_time=execution_time,
                error=str(e)
            )
    
    def optimize(self, max_combinations: Optional[int] = None, 
                 sample_combinations: bool = True) -> List[OptimizationResult]:
        """Run hyperparameter optimization"""
        self.logger.info("Starting hyperparameter optimization...")
        
        # Log topic count requirements
        min_required = self.min_topics_sector if self.sector_filter else self.min_topics_full
        dataset_type = f"sector {self.sector_filter}" if self.sector_filter else "full dataset"
        self.logger.info(f"Minimum topics required for {dataset_type}: {min_required}")
        
        # Load data
        df, embeddings = self.load_and_prepare_data()
        
        # Get parameter grid
        param_grid = self.get_parameter_grid()
        
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combinations = list(product(*param_values))
        
        self.logger.info(f"Total parameter combinations: {len(all_combinations)}")
        
        # Sample combinations if requested
        if sample_combinations and max_combinations and len(all_combinations) > max_combinations:
            import random
            random.seed(42)
            sampled_combinations = random.sample(all_combinations, max_combinations)
            self.logger.info(f"Sampling {max_combinations} combinations")
        else:
            sampled_combinations = all_combinations[:max_combinations] if max_combinations else all_combinations
        
        # Run optimization
        self.results = []
        successful_runs = 0
        
        for i, combination in enumerate(sampled_combinations, 1):
            params = dict(zip(param_names, combination))
            
            self.logger.info(f"Testing combination {i}/{len(sampled_combinations)}: {params}")
            
            result = self.evaluate_parameters(df, embeddings, params)
            self.results.append(result)
            
            if result.error is None:
                successful_runs += 1
                self.logger.info(f"  ✓ Stability: {result.stability_score:.3f}, "
                               f"Outliers: {result.avg_outlier_pct:.1f}%, "
                               f"Coherence: {result.avg_coherence:.3f}, "
                               f"Topics: {result.num_topics}")
            else:
                self.logger.warning(f"  ✗ Failed: {result.error}")
            
            # Log intermediate progress every 10 combinations
            if i % 10 == 0:
                self.log_intermediate_progress(i, len(sampled_combinations))
        
        # Count different types of results
        rejected_by_topics = len([r for r in self.results if r.error and "Insufficient topics" in r.error])
        other_failures = len([r for r in self.results if r.error and "Insufficient topics" not in r.error])
        
        self.logger.info(f"Optimization complete: {successful_runs}/{len(sampled_combinations)} successful runs")
        if rejected_by_topics > 0:
            self.logger.info(f"Rejected due to insufficient topics: {rejected_by_topics}")
        if other_failures > 0:
            self.logger.info(f"Failed due to other errors: {other_failures}")
        
        return self.results
    
    def log_intermediate_progress(self, current: int, total: int):
        """Log intermediate progress"""
        successful_results = [r for r in self.results if r.error is None]
        if successful_results:
            best_result = max(successful_results, key=lambda x: x.stability_score)
            self.logger.info(f"Progress: {current}/{total} combinations tested")
            self.logger.info(f"Current best stability score: {best_result.stability_score:.3f}")
            self.logger.info(f"Current best outlier rate: {best_result.avg_outlier_pct:.1f}%")
        else:
            self.logger.info(f"Progress: {current}/{total} combinations tested (no successful runs yet)")
    
    def log_final_results(self):
        """Log final results and recommendations to the log file"""
        if not self.results:
            self.logger.info("No results to log")
            return
        
        successful_results = [r for r in self.results if r.error is None]
        failed_results = [r for r in self.results if r.error is not None]
        
        self.logger.info("="*60)
        self.logger.info("HYPERPARAMETER OPTIMIZATION FINAL RESULTS")
        self.logger.info("="*60)
        self.logger.info(f"Total combinations tested: {len(self.results)}")
        self.logger.info(f"Successful runs: {len(successful_results)}")
        self.logger.info(f"Failed runs: {len(failed_results)}")
        
        if successful_results:
            best_results = self.get_best_parameters(5)
            
            self.logger.info("")
            self.logger.info("TOP 5 PARAMETER COMBINATIONS:")
            self.logger.info("="*60)
            
            for i, result in enumerate(best_results, 1):
                self.logger.info("")
                self.logger.info(f"Rank {i}:")
                self.logger.info(f"  Stability Score: {result.stability_score:.3f}")
                self.logger.info(f"  Train Outliers: {result.train_outlier_pct:.1f}%")
                self.logger.info(f"  Test Outliers: {result.test_outlier_pct:.1f}%")
                self.logger.info(f"  Train Coherence: {result.train_coherence:.3f}")
                self.logger.info(f"  Test Coherence: {result.test_coherence:.3f}")
                self.logger.info(f"  Topics Found: {result.num_topics}")
                self.logger.info(f"  Execution Time: {result.execution_time:.1f}s")
                self.logger.info(f"  Parameters:")
                for param, value in result.params.items():
                    self.logger.info(f"    {param}: {value}")
        
        # Topic count analysis
        all_topic_counts = [r.num_topics for r in self.results if r.error is None]
        if all_topic_counts:
            min_required = self.min_topics_sector if self.sector_filter else self.min_topics_full
            self.logger.info("")
            self.logger.info("TOPIC COUNT ANALYSIS:")
            self.logger.info(f"  Minimum required: {min_required}")
            self.logger.info(f"  Topic counts found: {min(all_topic_counts)} - {max(all_topic_counts)}")
            self.logger.info(f"  Average topic count: {sum(all_topic_counts) / len(all_topic_counts):.1f}")
            
            # Count how many met the requirement
            meeting_requirement = len([c for c in all_topic_counts if c >= min_required])
            self.logger.info(f"  Combinations meeting requirement: {meeting_requirement}/{len(all_topic_counts)}")
        
        if failed_results:
            self.logger.info("")
            self.logger.info("COMMON FAILURE REASONS:")
            error_counts = {}
            for result in failed_results:
                error_type = str(result.error)[:50] + "..." if len(str(result.error)) > 50 else str(result.error)
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                self.logger.info(f"  {error_type}: {count} occurrences")
                
            # Add guidance for different failure types
            rejected_by_topics = len([r for r in failed_results if "Insufficient topics" in r.error])
            rejected_by_vectorizer = len([r for r in failed_results if "Invalid vectorizer params" in r.error])
            
            if rejected_by_vectorizer > 0:
                self.logger.info("")
                self.logger.info("GUIDANCE FOR VECTORIZER PARAMETER ISSUES:")
                self.logger.info("  Use larger max_df values (0.8, 0.9, 0.95) for better compatibility")
                self.logger.info("  Use smaller min_df values (1, 2) for smaller datasets")
                self.logger.info("  The optimizer now validates parameters against BERTopic constraints")
                
            if rejected_by_topics > 0:
                self.logger.info("")
                self.logger.info("GUIDANCE FOR INSUFFICIENT TOPICS:")
                self.logger.info("  Consider reducing min_cluster_size_factor to get smaller, more numerous clusters")
                self.logger.info("  Try smaller cluster_selection_epsilon values for more distinct topics")
                self.logger.info("  Increase max_features in vectorizer for richer topic vocabulary")
                self.logger.info("  Or reduce minimum topic requirements with --min-topics-* flags")
        
        # Log the recommended configuration
        if successful_results:
            best_result = best_results[0]
            self.logger.info("")
            self.logger.info("="*60)
            self.logger.info("RECOMMENDED CONFIGURATION")
            self.logger.info("="*60)
            self.logger.info("Based on highest stability score (coherence - outlier_rate):")
            self.logger.info("")
            self.logger.info("Use these parameters in embed_generate.py:")
            for param, value in best_result.params.items():
                if param == 'clustering_algorithm':
                    self.logger.info(f"  --clustering {value}")
                elif param in ['n_neighbors', 'min_dist', 'n_components_clustering']:
                    self.logger.info(f"  UMAP {param}: {value}")
                elif param in ['min_cluster_size_factor', 'cluster_selection_epsilon', 'min_samples_factor']:
                    self.logger.info(f"  HDBSCAN {param}: {value}")
                elif param in ['max_features', 'max_df', 'min_df']:
                    self.logger.info(f"  Vectorizer {param}: {value}")
            self.logger.info("")
            self.logger.info(f"Expected performance:")
            self.logger.info(f"  - Outlier rate: {best_result.avg_outlier_pct:.1f}%")
            self.logger.info(f"  - Topic coherence: {best_result.avg_coherence:.3f}")
            self.logger.info(f"  - Number of topics: {best_result.num_topics}")
            self.logger.info("="*60)
    
    def get_best_parameters(self, top_k: int = 5) -> List[OptimizationResult]:
        """Get top k parameter combinations"""
        # Filter successful results
        successful_results = [r for r in self.results if r.error is None]
        
        if not successful_results:
            self.logger.warning("No successful parameter combinations found")
            return []
        
        # Sort by stability score (higher is better)
        sorted_results = sorted(successful_results, key=lambda x: x.stability_score, reverse=True)
        
        return sorted_results[:top_k]
    
    def print_summary(self):
        """Print optimization summary"""
        if not self.results:
            print("No results to summarize")
            return
        
        successful_results = [r for r in self.results if r.error is None]
        failed_results = [r for r in self.results if r.error is not None]
        
        print(f"\n{'='*60}")
        print("HYPERPARAMETER OPTIMIZATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total combinations tested: {len(self.results)}")
        print(f"Successful runs: {len(successful_results)}")
        print(f"Failed runs: {len(failed_results)}")
        
        # Topic count analysis
        rejected_by_topics = len([r for r in failed_results if r.error and "Insufficient topics" in r.error])
        if rejected_by_topics > 0:
            min_required = self.min_topics_sector if self.sector_filter else self.min_topics_full
            print(f"Rejected due to insufficient topics (<{min_required}): {rejected_by_topics}")
        
        if successful_results:
            best_results = self.get_best_parameters(5)
            
            print(f"\nTOP 5 PARAMETER COMBINATIONS:")
            print(f"{'='*60}")
            
            for i, result in enumerate(best_results, 1):
                print(f"\nRank {i}:")
                print(f"  Stability Score: {result.stability_score:.3f}")
                print(f"  Avg Outliers: {result.avg_outlier_pct:.1f}%")
                print(f"  Avg Coherence: {result.avg_coherence:.3f}")
                print(f"  Topics Found: {result.num_topics}")
                print(f"  Execution Time: {result.execution_time:.1f}s")
                print(f"  Parameters:")
                for param, value in result.params.items():
                    print(f"    {param}: {value}")
        
        if failed_results:
            print(f"\nCOMMON FAILURE REASONS:")
            error_counts = {}
            for result in failed_results:
                error_type = type(result.error).__name__ if hasattr(result.error, '__class__') else 'Unknown'
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {error_type}: {count} occurrences")

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for BERTopic models')
    parser.add_argument('--test-size', type=int, default=1000, 
                       help='Number of samples to use for optimization (default: 1000)')
    parser.add_argument('--max-combinations', type=int, default=100,
                       help='Maximum number of parameter combinations to test (default: 100)')
    parser.add_argument('--data-file', type=str, default='data_unified_filtered.json',
                       help='Data file to use for optimization')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable embedding caching')
    parser.add_argument('--finance', action='store_true', 
                       help='Only optimize for finance-related MCP servers (equivalent to --52)')
    parser.add_argument('--kmeans', action='store_true',
                       help='Include K-means clustering in optimization (note: K-means has no outliers by definition)')
    parser.add_argument('--min-topics-sector', type=int, default=10,
                       help='Minimum number of topics required for sector-specific datasets (default: 10)')
    parser.add_argument('--min-topics-full', type=int, default=MIN_TOPICS_REQUIRED,
                       help=f'Minimum number of topics required for full dataset (default: {MIN_TOPICS_REQUIRED})')
    
    # Add sector-specific arguments
    for sector_code in NAICS_KEYWORDS.keys():
        sector_name = NAICS_SECTORS.get(sector_code, f"Sector {sector_code}")
        parser.add_argument(f'--{sector_code}', action='store_true', 
                          help=f'Only optimize for servers in sector {sector_code}: {sector_name}')
    
    args = parser.parse_args()
    
    # Determine which sector is selected
    selected_sector = None
    if args.finance:
        selected_sector = 52  # Finance and Insurance
    else:
        # Check which sector argument was provided
        for sector_code in NAICS_KEYWORDS.keys():
            if getattr(args, str(sector_code), False):
                if selected_sector is not None:
                    parser.error("Only one sector can be selected at a time")
                selected_sector = sector_code
    
    # Setup GPU optimizations
    setup_gpu_optimizations()
    
    # Run optimization
    optimizer = HyperparameterOptimizer(
        data_file=args.data_file,
        test_size=args.test_size,
        cache_embeddings=not args.no_cache,
        sector_filter=selected_sector,
        include_kmeans=args.kmeans,
        min_topics_sector=args.min_topics_sector,
        min_topics_full=args.min_topics_full
    )
    
    results = optimizer.optimize(max_combinations=args.max_combinations)
    
    # Log final results to log file
    optimizer.log_final_results()
    
    # Print brief summary to console
    optimizer.print_summary()
    
    # Display final file names
    suffix = f"_sector_{selected_sector}" if selected_sector else ""
    log_file = f'embed_hyperparameter_optimization{suffix}.log'
    
    print(f"\nAll results logged to: {log_file}")
    print(f"Check the end of the log file for recommended configuration.")
    
    if selected_sector:
        sector_name = NAICS_SECTORS.get(selected_sector, f"Sector {selected_sector}")
        print(f"Optimization completed for {sector_name} (NAICS {selected_sector})")

if __name__ == "__main__":
    main()
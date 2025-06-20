# %%
### 1. MAIN FUNCTION DEFINITIONS (OPTIMIZED FOR SPEED) ###

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import plotly.express as px
import json
import argparse
import logging
import os
import warnings
import time
from tqdm import tqdm
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import torch
from concurrent.futures import ThreadPoolExecutor
import hashlib
import pickle
import gc
import shutil
import re
from naics_classification_config import classify_naics_sector, NAICS_SECTORS, NAICS_COLORS

# OPTIMIZATION 1: Enable mixed precision and optimize GPU settings
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

seed_topic_list = [
    ["equity", "stocks", "shares", "dividend", "market cap", "P/E ratio"],
    ["debt", "bonds", "yield", "credit", "coupon", "maturity"],
    ["derivatives", "options", "futures", "swaps", "hedging"],
    ["risk", "volatility", "VaR", "beta", "standard deviation"],
    ["liquidity", "cash flow", "working capital", "current ratio"],
    ["profitability", "ROI", "ROE", "margin", "EBITDA"],
    ["valuation", "DCF", "multiples", "fair value", "intrinsic value"],
    ["regulation", "compliance", "Basel", "MiFID", "Dodd-Frank"]
    ]

# OPTIMIZATION 2: Implement proper batching with memory management
def generate_high_quality_embeddings(texts, model_name='ProsusAI/finbert', 
                                    device='cuda', batch_size=64, cache_dir='embeddings_cache'):
    """
    Converts a list of texts into high-quality numerical embeddings.
    Implements caching to avoid recomputation.
    """
    # Set memory optimization environment variable
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache key based on texts and model
    cache_key = hashlib.md5(f"{model_name}_{str(texts)}".encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    # Check if embeddings are cached
    if os.path.exists(cache_path):
        logging.info(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    # Clear GPU memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Setup GPU optimizations
    has_gpu = setup_gpu_optimizations()
    
    model = SentenceTransformer(
        model_name,
        device=device if has_gpu else 'cpu'
    )
    
    # Note: Removed model.half() as it can cause issues with some models
    
    # Process in batches with progress bar
    embeddings_list = []
    num_batches = (len(texts) + batch_size - 1) // batch_size
    
    with tqdm(total=num_batches, desc="Generating embeddings", unit="batch") as pbar:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                # Generate embeddings without autocast for stability
                batch_embeddings = model.encode(
                    batch_texts, 
                    show_progress_bar=False, 
                    normalize_embeddings=True,
                    convert_to_tensor=False
                )
                
                embeddings_list.append(batch_embeddings)
                pbar.update(1)
                
                # Aggressive memory cleanup after each batch
                if has_gpu:
                    torch.cuda.empty_cache()
                    
            except torch.cuda.OutOfMemoryError:
                logging.warning(f"OOM at batch {i//batch_size}, reducing batch size")
                # Try with smaller batch size
                if batch_size > 8:
                    smaller_batch = batch_size // 2
                    logging.info(f"Retrying with batch size {smaller_batch}")
                    # Recursive call with smaller batch size
                    del model
                    torch.cuda.empty_cache()
                    return generate_high_quality_embeddings(
                        texts, model_name, device, smaller_batch, cache_dir
                    )
                else:
                    raise
    
    # Cleanup model to free memory
    del model
    if has_gpu:
        torch.cuda.empty_cache()
    
    # Concatenate all embeddings
    embeddings = np.vstack(embeddings_list)
    
    # Cache the embeddings
    with open(cache_path, 'wb') as f:
        pickle.dump(embeddings, f)
    logging.info(f"Cached embeddings to {cache_path}")
    
    return embeddings

# OPTIMIZATION 3: Use GPU-accelerated UMAP if available, with optimized parameters
def reduce_dimensions(embeddings, n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine'):
    """Reduces the dimensionality of embeddings using UMAP with optimizations."""
    # Ensure n_neighbors is valid
    n_neighbors = min(n_neighbors, len(embeddings) - 1)
    n_neighbors = max(2, n_neighbors)
    
    try:
        # Try to use cuML UMAP for GPU acceleration
        from cuml.manifold import UMAP as cumlUMAP
        reducer = cumlUMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=42,
            verbose=False
        )
        logging.info("Using GPU-accelerated cuML UMAP")
    except ImportError:
        # Fall back to CPU UMAP with optimizations
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=42,
            verbose=False,
            n_jobs=-1,  # Use all CPU cores
            low_memory=False,  # Trade memory for speed
            angular_rp_forest=True,  # Use approximate nearest neighbor
            target_metric='euclidean'
        )
        logging.info("Using CPU UMAP with optimizations")
    
    embeddings_2d = reducer.fit_transform(embeddings)
    return embeddings_2d, reducer

# OPTIMIZATION 4: Parallelize data preparation
def prepare_texts_parallel(servers_data, num_workers=8):
    """Prepare texts using parallel processing"""
    def process_server(server):
        name = server.get('canonical_name', '')
        description = server.get('description', '')
        embedding_text = server.get('embedding_text', description)
        
        if embedding_text and len(embedding_text) > 20:
            return {
                'text': embedding_text,
                'canonical_name': name,
                'description': description,
                'url': server.get('url', ''),
                'stargazers_count': server.get('stargazers_count', 0),
                'created_at': server.get('created_at', '')
            }
        return None
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_server, servers_data))
    
    # Filter out None values
    return [r for r in results if r is not None]

# OPTIMIZATION 5: Optimized BERTopic with reduced vocabulary and GPU HDBSCAN if available
def create_bertopic_model(embeddings, texts, min_cluster_size=25, n_neighbors=15):
    """Create BERTopic model using pre-computed embeddings with optimizations."""
    
    # Ensure n_neighbors is valid
    n_neighbors = min(n_neighbors, len(embeddings) - 1)
    n_neighbors = max(2, n_neighbors)
    
    # Try GPU-accelerated UMAP
    try:
        from cuml.manifold import UMAP as cumlUMAP
        umap_model = cumlUMAP(
            n_neighbors=n_neighbors,
            min_dist=0.1,
            n_components=5,  # Higher dims for clustering, then reduce
            metric='cosine',
            random_state=42,
            verbose=False
        )
        logging.info("Using GPU-accelerated UMAP for BERTopic")
    except ImportError:
        umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=0.1,
            n_components=5,  # Higher dims for clustering
            metric='cosine',
            random_state=42,
            verbose=False,
            n_jobs=-1,
            low_memory=False
        )
    
    # Try GPU-accelerated HDBSCAN
    try:
        from cuml.cluster import HDBSCAN as cumlHDBSCAN
        hdbscan_model = cumlHDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean'
        )
        logging.info("Using GPU-accelerated HDBSCAN")
    except ImportError:
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            prediction_data=True,
            core_dist_n_jobs=-1,  # Use all cores
            algorithm='prims_kdtree'  # Faster for low dimensions
        )
    
    # Optimized vectorizer with parameters suitable for small datasets
    vectorizer_model = CountVectorizer(
        max_features=100,  # Much smaller for test datasets
        ngram_range=(1, 3),  # Only two-sentence words for small datasets
        stop_words='english',
        min_df=2,  # Accept single occurrences for small datasets
        max_df=0.95  # Accept all terms for small datasets
    )
    
    # Create BERTopic model with optimizations
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        verbose=False,
        calculate_probabilities=False,  # Faster without probability calculation
        nr_topics='auto',
        low_memory=True,  # Use less memory
        seed_topic_list=seed_topic_list
    )
    
    # Fit model with pre-computed embeddings
    topics, probs = topic_model.fit_transform(texts, embeddings)
    
    return topic_model, topics, probs

def create_interactive_plot(df, topic_info=None):
    """Creates an interactive scatter plot using Plotly with topic labels."""
    # Create color mapping for topics
    unique_topics = sorted(df['topic'].unique())
    
    # Create hover text with topic keywords if available
    hover_text = []
    for _, row in df.iterrows():
        topic_num = row['topic']
        topic_label = row.get('topic_label', f'Topic {topic_num}')
        hover_info = f"<b>{row['canonical_name']}</b><br>"
        hover_info += f"Topic: {topic_label}<br>"
        hover_info += f"Description: {row['text'][:100]}..."
        hover_text.append(hover_info)
    
    df['hover_text'] = hover_text
    
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='topic_label',
        custom_data=['canonical_name', 'text'],
        title='BERTopic Analysis of MCP Server Descriptions',
        labels={'topic_label': 'Topic'},
        template='plotly_dark'
    )
    
    fig.update_traces(
        marker=dict(size=4, opacity=0.7),
        hovertemplate='%{customdata[0]}<br>Topic: %{color}<br>%{customdata[1]}<extra></extra>'
    )
    
    fig.update_layout(
        xaxis=dict(showticklabels=False, title='UMAP Dimension 1'),
        yaxis=dict(showticklabels=False, title='UMAP Dimension 2'),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        width=1200,
        height=800
    )
    return fig

# %%
### 2. PARSE ARGUMENTS AND LOAD DATA FROM dashboard_mcp_servers_unified.json ###

def setup_logging(test_mode=False):
    """Configure logging with file handler and minimal console output."""
    log_filename = 'embed_test_generation.log' if test_mode else 'embed_generation.log'
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create file handler with detailed logging
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Create console handler with minimal output (only warnings and errors)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Configure logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Prevent duplicate logging
    logger.propagate = False
    
    return logger

def main():
    parser = argparse.ArgumentParser(description='Generate embeddings and visualizations for MCP servers')
    parser.add_argument('--test', action='store_true', help='Run in test mode with 1000 rows')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for embedding generation')
    parser.add_argument('--no-cache', action='store_true', help='Disable embedding caching')
    parser.add_argument('--clear-cache', action='store_true', help='Clear embedding cache before running')
    args = parser.parse_args()
    
    # Set memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    logger = setup_logging(args.test)
    
    # Clear cache if requested
    if args.clear_cache and os.path.exists('embeddings_cache'):
        import shutil
        shutil.rmtree('embeddings_cache')
        logger.info("Cleared embedding cache")
    
    # Setup GPU if available
    if torch.cuda.is_available():
        # Clear GPU memory at start
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        logger.info(f"GPU available: {torch.cuda.get_device_name()}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logger.info(f"Free GPU memory: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.2f} GB")
        setup_gpu_optimizations()
    else:
        logger.warning("No GPU available, using CPU")
    
    # Simple console message
    mode = "test mode" if args.test else "full mode"
    log_file = 'embed_test_generation.log' if args.test else 'embed_generation.log'
    print(f"Starting embedding generation in {mode}. Check {log_file} for detailed progress.")
    
    # Initialize progress bar for overall process and start timing
    start_time = time.time()
    total_steps = 5  # Data loading, embedding generation, BERTopic modeling, topic labeling, visualization
    progress_bar = tqdm(total=total_steps, desc="Overall Progress", unit="step", 
                       bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} steps [{elapsed}]')
    
    unified_data_file = 'dashboard_mcp_servers_unified.json'
    test_size = 1000

    try:
        with open(unified_data_file, 'r', encoding='utf-8') as f:
            servers_data = json.load(f)
        logger.info(f"Successfully loaded {len(servers_data):,} records from {unified_data_file}")

        # Use test mode if --test flag is provided
        if args.test:
            servers_data = servers_data[:test_size]
            logger.info(f"TEST MODE: Using only {len(servers_data)} records for testing")

        # Use parallel processing for data preparation
        prep_start = time.time()
        prepared_texts = prepare_texts_parallel(servers_data)
        prep_duration = time.time() - prep_start
        logger.info(f"Data preparation completed in {prep_duration:.1f} seconds")

        df = pd.DataFrame(prepared_texts)
        logger.info(f"Prepared {len(df):,} servers with meaningful embedding text for processing.")
        
        # Check if we have enough data for meaningful clustering
        if len(df) < 50:
            logger.warning(f"Only {len(df)} texts found. This may be too few for meaningful topic modeling.")
            if len(df) < 10:
                logger.error("Less than 10 texts found. Cannot proceed with topic modeling.")
                progress_bar.close()
                return
        
        progress_bar.update(1)  # Step 1: Data loading complete

    except FileNotFoundError:
        logger.error(f"The file '{unified_data_file}' was not found.")
        progress_bar.close()
        return
    except json.JSONDecodeError:
        logger.error(f"Could not decode JSON from '{unified_data_file}'.")
        progress_bar.close()
        return

    # %%
    ### 3. GENERATE HIGH-QUALITY EMBEDDINGS ###

    if not df.empty:
        texts = df['text'].tolist()
        
        logger.info(f"Generating embeddings for {len(texts)} texts with batch size {args.batch_size}")
        
        # Generate embeddings with optimizations
        embedding_start = time.time()
        cache_dir = None if args.no_cache else 'embeddings_cache'
        embeddings = generate_high_quality_embeddings(
            texts, 
            batch_size=args.batch_size,
            cache_dir=cache_dir
        )
        
        embedding_duration = time.time() - embedding_start
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        logger.info(f"Embedding generation completed in {embedding_duration:.1f} seconds")
        logger.info(f"Speed: {len(texts) / embedding_duration:.1f} texts/second")
        progress_bar.update(1)  # Step 2: Embedding generation complete
    else:
        logger.warning("Skipping embedding generation as no data was loaded.")
        progress_bar.close()
        return

    # %%
    ### 4. APPLY BERTOPIC MODELING ###

    if not df.empty and embeddings is not None:
        # Adjust clustering parameters for smaller datasets in test mode
        min_cluster_size = 10 if args.test else 25
        n_neighbors = min(15, max(2, len(df) // 10)) if args.test else 15
        
        # Apply BERTopic modeling
        logger.info("Starting BERTopic modeling...")
        bertopic_start = time.time()
        
        topic_model, topics, probs = create_bertopic_model(
            embeddings, 
            df['text'].tolist(), 
            min_cluster_size=min_cluster_size,
            n_neighbors=n_neighbors
        )
        
        bertopic_duration = time.time() - bertopic_start
        logger.info(f"BERTopic modeling completed in {bertopic_duration:.1f} seconds")
        progress_bar.update(1)  # Step 3: BERTopic modeling complete
        
        # Get topic information
        topic_info = topic_model.get_topic_info()
        logger.info(f"Found {len(topic_info)} topics (including outliers)")
        
        # Get 2D embeddings for visualization (use existing UMAP model for speed)
        viz_start = time.time()
        embeddings_2d = topic_model.umap_model.transform(embeddings)
        df['x'] = embeddings_2d[:, 0]
        df['y'] = embeddings_2d[:, 1]
        df['topic'] = topics
        
        # Create topic labels with keywords
        topic_labels = {}
        for topic_id in topic_info['Topic'].values:
            if topic_id == -1:
                topic_labels[topic_id] = "Outliers"
            else:
                # Get top 3 keywords for each topic
                topic_words = topic_model.get_topic(topic_id)
                if topic_words:
                    keywords = [word for word, _ in topic_words[:3]]
                    topic_labels[topic_id] = f"Topic {topic_id}: {', '.join(keywords)}"
                else:
                    topic_labels[topic_id] = f"Topic {topic_id}"
        
        df['topic_label'] = df['topic'].map(topic_labels)
        logger.info("Topic labeling complete.")
        progress_bar.update(1)  # Step 4: Topic labeling complete
        
        # Create visualization
        fig = create_interactive_plot(df, topic_info)
        
        # Save figure as HTML file
        html_filename = 'embed_test_visualization.html' if args.test else 'embed_visualization.html'
        fig.write_html(html_filename)
        viz_duration = time.time() - viz_start
        logger.info(f"Visualization created in {viz_duration:.1f} seconds")
        logger.info(f"Interactive visualization saved to {html_filename}")
        
        # Save detailed results
        output_file = 'embed_test_results.json' if args.test else 'embed_results.json'
        results = df.to_dict('records')
        
        # Add topic information to results
        topic_summary = {
            'topic_info': topic_info.to_dict('records'),
            'topic_labels': {str(k): v for k, v in topic_labels.items()},
            'num_topics': len(topic_info) - 1,  # Exclude outliers
            'servers': results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(topic_summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Results with topic analysis saved to {output_file}")
        
        # Print topic summary
        print(f"\nTopic Analysis Summary:")
        print(f"Found {len(topic_info) - 1} topics (excluding outliers)")
        for topic_id, label in sorted(topic_labels.items()):
            if topic_id != -1:
                count = len(df[df['topic'] == topic_id])
                print(f"  {label}: {count} servers")
        
        progress_bar.update(1)  # Step 5: Visualization and saving complete
        progress_bar.close()
        
        total_duration = time.time() - start_time
        print(f"\nBERTopic analysis completed successfully in {total_duration:.1f} seconds!")
        print(f"Performance summary:")
        print(f"  - Data preparation: {prep_duration:.1f}s")
        print(f"  - Embedding generation: {embedding_duration:.1f}s ({len(texts) / embedding_duration:.0f} texts/s)")
        print(f"  - BERTopic modeling: {bertopic_duration:.1f}s")
        print(f"  - Visualization: {viz_duration:.1f}s")
        
        # Memory usage report
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"\nGPU Memory usage:")
            print(f"  - Allocated: {allocated:.2f} GB")
            print(f"  - Reserved: {reserved:.2f} GB")
            
        print(f"\nCheck {html_filename} for visualization.")
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    else:
        logger.warning("Skipping analysis as no data was processed.")
        progress_bar.close()

if __name__ == "__main__":
    main()
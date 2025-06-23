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
from naics_classification_config import NAICS_SECTORS, NAICS_KEYWORDS

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



# OPTIMIZATION 2: Implement proper batching with memory management
def generate_high_quality_embeddings(texts, model_name='NovaSearch/stella_en_400M_v5', 
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
        device=device if has_gpu else 'cpu',
        trust_remote_code=True
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
def create_bertopic_model(embeddings, texts, min_cluster_size=5, n_neighbors=15, clustering_algorithm='hdbscan'):
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
    
    # Choose clustering algorithm
    if clustering_algorithm == 'kmeans':
        from sklearn.cluster import KMeans
        
        # Determine number of clusters based on dataset size
        n_clusters = min(30, max(5, len(texts) // 40))  # 5-30 clusters based on data size
        
        cluster_model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        logging.info(f"Using K-means clustering with {n_clusters} clusters")
        nr_topics = n_clusters
    else:
        # Use HDBSCAN (original algorithm)
        try:
            from cuml.cluster import HDBSCAN as cumlHDBSCAN
            cluster_model = cumlHDBSCAN(
                min_cluster_size=min_cluster_size,
                metric='euclidean',
                cluster_selection_epsilon=0.0,  
                cluster_selection_method='eom'  # 'eom' tends to find more clusters
            )
            logging.info("Using GPU-accelerated HDBSCAN")
        except ImportError:
            cluster_model = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                metric='euclidean',
                prediction_data=True,
                core_dist_n_jobs=-1,  # Use all cores
                algorithm='prims_kdtree'  # Faster for low dimensions
            )
            logging.info("Using CPU HDBSCAN")
        nr_topics = 'auto'
    
    # Optimized vectorizer with parameters suitable for small datasets
    # Adjust min_df and max_df based on dataset size to avoid conflicts
    n_docs = len(texts)
    
    vectorizer_model = CountVectorizer(
        max_features=min(300, n_docs * 2),  # Scale features with data size
        ngram_range=(1, 3),  
        stop_words='english',
        min_df=1,  
        max_df=0.3
    )
    
    # Create BERTopic model with chosen clustering algorithm
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=cluster_model,  # Using chosen clustering algorithm
        vectorizer_model=vectorizer_model,
        verbose=False,
        calculate_probabilities=False,  # Faster without probability calculation
        nr_topics=nr_topics,  # Auto or fixed number based on algorithm
        low_memory=True,  # Use less memory
    )
    
# Fit model with pre-computed embeddings
    topics, probs = topic_model.fit_transform(texts, embeddings)
    
    return topic_model, topics, probs

def create_interactive_plot(df, topic_info=None):
    """Creates an interactive scatter plot using Plotly with direct cluster labeling."""
    import numpy as np
    
    # Create hover text with topic keywords if available
    hover_text = []
    for _, row in df.iterrows():
        topic_num = row['topic']
        topic_label = row.get('topic_label', f'Topic {topic_num}')
        
        # Break description after every 5 words
        description = row['text'][:100]
        words = description.split()
        formatted_description = []
        for i in range(0, len(words), 5):
            formatted_description.append(' '.join(words[i:i+5]))
        formatted_desc = '<br>'.join(formatted_description) + "..."
        
        hover_info = f"<b>{row['canonical_name']}</b><br>"
        hover_info += f"Topic: {topic_label}<br>"
        hover_info += f"Stars: {row.get('stargazers_count', 0)}<br>"
        hover_info += f"Description: {formatted_desc}"
        hover_text.append(hover_info)
    
    df['hover_text'] = hover_text
    
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='topic',
        custom_data=['canonical_name', 'text', 'stargazers_count'],
        title='BERTopic Analysis of MCP Server Descriptions',
        template='plotly_dark'
    )
    
    fig.update_traces(
        marker=dict(size=6, opacity=0.7),
        hovertemplate='%{customdata[0]}<br>Stars: %{customdata[2]}<br>%{customdata[1]}<extra></extra>',
        showlegend=False  # Remove legend
    )
    
    # Calculate cluster information and place text annotations
    annotations = []
    used_positions = []  # Track used positions to avoid overlap
    
    for topic_id in df['topic'].unique():
        if topic_id == -1:  # Skip outliers
            continue
            
        topic_servers = df[df['topic'] == topic_id]
        if len(topic_servers) > 0:
            # Calculate cluster centroid
            center_x = topic_servers['x'].mean()
            center_y = topic_servers['y'].mean()
            
            # Find the most popular server in this topic
            most_popular = topic_servers.loc[topic_servers['stargazers_count'].idxmax()]
            
            # Get cluster keywords (first two terms only)
            cluster_name = most_popular.get('topic_label', f'Topic {topic_id}')
            if cluster_name.startswith(f'Topic {topic_id}:'):
                cluster_name = cluster_name.replace(f'Topic {topic_id}: ', '')
            
            # Extract first two terms
            keywords = cluster_name.split(', ')[:2]
            short_name = ', '.join(keywords)
            
            server_name = most_popular['canonical_name']
            server_count = len(topic_servers)
            stars = most_popular['stargazers_count']
            
            # Get the color for this topic from the scatter plot
            # Use a simple color mapping based on topic_id
            colors = px.colors.qualitative.Plotly
            topic_color = colors[topic_id % len(colors)]
            
            # Create annotation text in two lines: topic name, then example
            annotation_text = f"<b>{short_name}</b><br>(e.g. {server_name} {stars}★)"
            
            # Smart positioning to avoid overlap
            final_x, final_y = find_non_overlapping_position(
                center_x, center_y, used_positions, 
                df['x'].min(), df['x'].max(), df['y'].min(), df['y'].max()
            )
            used_positions.append((final_x, final_y))
            
            fig.add_annotation(
                x=final_x,
                y=final_y,
                text=annotation_text,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="white",
                ax=0,
                ay=0,
                bgcolor="rgba(0,0,0,0)",  # Fully transparent background
                bordercolor="rgba(0,0,0,0)",  # Fully transparent border
                borderwidth=0,
                font=dict(color="white", size=8),
                align="center"
            )
    
    fig.update_layout(
        xaxis=dict(showticklabels=False, title='UMAP Dimension 1'),
        yaxis=dict(showticklabels=False, title='UMAP Dimension 2'),
        width=1800,  # Much larger figure
        height=1200,
        showlegend=False  # No legend
    )
    return fig

def find_non_overlapping_position(center_x, center_y, used_positions, x_min, x_max, y_min, y_max, min_distance=0.3):
    """Find a position for annotation that doesn't overlap with existing ones."""
    import math
    
    # If no positions used yet, use center
    if not used_positions:
        return center_x, center_y
    
    # Check if center position is free
    if all(math.sqrt((center_x - ux)**2 + (center_y - uy)**2) > min_distance for ux, uy in used_positions):
        return center_x, center_y
    
    # Try positions in a spiral around the center
    data_width = x_max - x_min
    data_height = y_max - y_min
    step_size = min(data_width, data_height) * 0.05  # 5% of data range
    
    for radius in [step_size * i for i in range(1, 20)]:  # Try increasing radii
        for angle in [i * 45 for i in range(8)]:  # 8 directions
            test_x = center_x + radius * math.cos(math.radians(angle))
            test_y = center_y + radius * math.sin(math.radians(angle))
            
            # Check if this position is free
            if all(math.sqrt((test_x - ux)**2 + (test_y - uy)**2) > min_distance for ux, uy in used_positions):
                return test_x, test_y
    
    # If all else fails, return center (should rarely happen)
    return center_x, center_y

# %%
### 2. PARSE ARGUMENTS AND LOAD DATA FROM dashboard_mcp_servers_unified.json ###

def setup_logging(test_mode=False, sector_mode=None):
    """Configure logging with file handler and minimal console output."""
    filename_suffix = ""
    if test_mode:
        filename_suffix += "_test"
    if sector_mode:
        filename_suffix += f"_sector_{sector_mode}"
    log_filename = f'embed{filename_suffix}_generation.log'
    
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
    parser.add_argument('--finance', action='store_true', help='Only process finance-related MCP servers (equivalent to --52)')
    parser.add_argument('--clustering', choices=['hdbscan', 'kmeans'], default='kmeans', help='Clustering algorithm to use (default: kmeans)')
    
    # Add sector-specific arguments
    for sector_code in NAICS_KEYWORDS.keys():
        sector_name = NAICS_SECTORS.get(sector_code, f"Sector {sector_code}")
        parser.add_argument(f'--{sector_code}', action='store_true', 
                          help=f'Only process servers in sector {sector_code}: {sector_name}')
    
    args = parser.parse_args()
    
    # Set memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
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
    
    logger = setup_logging(args.test, selected_sector)
    
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
    mode_parts = []
    if args.test:
        mode_parts.append("test mode")
    if selected_sector:
        sector_name = NAICS_SECTORS.get(selected_sector, f"Sector {selected_sector}")
        mode_parts.append(f"sector {selected_sector} ({sector_name}) mode")
    if not mode_parts:
        mode_parts.append("full mode")
    mode = " + ".join(mode_parts)
    
    filename_suffix = ""
    if args.test:
        filename_suffix += "_test"
    if selected_sector:
        filename_suffix += f"_sector_{selected_sector}"
    log_file = f'embed{filename_suffix}_generation.log'
    print(f"Starting embedding generation in {mode}. Check {log_file} for detailed progress.")
    
    # Initialize progress bar for overall process and start timing
    start_time = time.time()
    total_steps = 5  # Data loading, embedding generation, BERTopic modeling, topic labeling, visualization
    progress_bar = tqdm(total=total_steps, desc="Overall Progress", unit="step", 
                       bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} steps [{elapsed}]')
    
    unified_data_file = 'dashboard_mcp_servers_unified_filtered.json'
    test_size = 1000

    try:
        with open(unified_data_file, 'r', encoding='utf-8') as f:
            servers_data = json.load(f)
        logger.info(f"Successfully loaded {len(servers_data):,} records from {unified_data_file}")

        # Filter for sector-specific servers if a sector flag is provided
        if selected_sector:
            original_count = len(servers_data)
            sector_attr = f'is_sector_{selected_sector}'
            sector_name = NAICS_SECTORS.get(selected_sector, f"Sector {selected_sector}")
            
            servers_data = [server for server in servers_data if server.get(sector_attr, False)]
            logger.info(f"SECTOR {selected_sector} MODE: Filtered to {len(servers_data)} servers in {sector_name} from {original_count} total")
            if len(servers_data) == 0:
                logger.error(f"No servers found for sector {selected_sector} ({sector_name}) in the dataset.")
                progress_bar.close()
                return

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
        min_cluster_size = 5
        n_neighbors = 10
        
        # Apply BERTopic modeling
        logger.info("Starting BERTopic modeling...")
        bertopic_start = time.time()
        
        topic_model, topics, probs = create_bertopic_model(
            embeddings, 
            df['text'].tolist(), 
            min_cluster_size=min_cluster_size,
            n_neighbors=n_neighbors,
            clustering_algorithm=args.clustering
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
        filename_suffix = ""
        if args.test:
            filename_suffix += "_test"
        if selected_sector:
            filename_suffix += f"_sector_{selected_sector}"
        
        html_filename = f'embed{filename_suffix}_visualization.html'
        fig.write_html(html_filename)
        viz_duration = time.time() - viz_start
        logger.info(f"Visualization created in {viz_duration:.1f} seconds")
        logger.info(f"Interactive visualization saved to {html_filename}")
        
        # Save detailed results
        output_file = f'embed{filename_suffix}_results.json'
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
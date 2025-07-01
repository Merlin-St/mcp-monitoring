#!/usr/bin/env python3
"""
Helper script to apply optimized parameters from hyperparameter optimization to embed_generate.py
Reads the recommended configuration from the log file and applies it to the code.
"""

import re
import argparse
import logging
from pathlib import Path

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def parse_log_file(log_file_path: str) -> dict:
    """Parse optimization log file to extract recommended parameters"""
    logger = logging.getLogger(__name__)
    
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Log file not found: {log_file_path}")
    
    # Find the RECOMMENDED CONFIGURATION section
    config_start = content.find("RECOMMENDED CONFIGURATION")
    if config_start == -1:
        raise ValueError("No RECOMMENDED CONFIGURATION section found in log file")
    
    config_section = content[config_start:]
    
    # Extract parameters using regex
    params = {}
    
    # UMAP parameters
    umap_patterns = {
        'n_neighbors': r'UMAP n_neighbors: (\d+)',
        'min_dist': r'UMAP min_dist: ([\d.]+)',
        'n_components_clustering': r'UMAP n_components_clustering: (\d+)'
    }
    
    # HDBSCAN parameters
    hdbscan_patterns = {
        'min_cluster_size_factor': r'HDBSCAN min_cluster_size_factor: ([\d.]+)',
        'cluster_selection_epsilon': r'HDBSCAN cluster_selection_epsilon: ([\d.]+)',
        'min_samples_factor': r'HDBSCAN min_samples_factor: ([\d.]+)'
    }
    
    # Vectorizer parameters
    vectorizer_patterns = {
        'max_features': r'Vectorizer max_features: (\d+)',
        'max_df': r'Vectorizer max_df: ([\d.]+)',
        'min_df': r'Vectorizer min_df: (\d+)'
    }
    
    # Clustering algorithm
    clustering_pattern = r'--clustering (\w+)'
    
    all_patterns = {**umap_patterns, **hdbscan_patterns, **vectorizer_patterns}
    
    for param_name, pattern in all_patterns.items():
        match = re.search(pattern, config_section)
        if match:
            value = match.group(1)
            # Convert to appropriate type
            if param_name in ['n_neighbors', 'n_components_clustering', 'max_features', 'min_df']:
                params[param_name] = int(value)
            else:
                params[param_name] = float(value)
        else:
            logger.warning(f"Could not find parameter: {param_name}")
    
    # Extract clustering algorithm
    clustering_match = re.search(clustering_pattern, config_section)
    if clustering_match:
        params['clustering_algorithm'] = clustering_match.group(1)
    
    logger.info(f"Extracted {len(params)} parameters from log file")
    return params

def apply_parameters_to_embed_generate(params: dict, embed_file_path: str = 'embed_generate.py', backup: bool = True):
    """Apply optimized parameters to embed_generate.py"""
    logger = logging.getLogger(__name__)
    
    # Read the current file
    try:
        with open(embed_file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"embed_generate.py not found: {embed_file_path}")
    
    # Create backup if requested
    if backup:
        backup_path = f"{embed_file_path}.backup"
        with open(backup_path, 'w') as f:
            f.write(content)
        logger.info(f"Created backup: {backup_path}")
    
    # Apply UMAP parameters
    if 'n_neighbors' in params and 'min_dist' in params and 'n_components_clustering' in params:
        # Update reduce_dimensions function signature
        old_pattern = r'def reduce_dimensions\(embeddings, n_neighbors=\d+, min_dist=[\d.]+, n_components=\d+, metric=\'cosine\'\):'
        new_signature = f"def reduce_dimensions(embeddings, n_neighbors={params['n_neighbors']}, min_dist={params['min_dist']}, n_components={params['n_components_clustering']}, metric='cosine'):"
        content = re.sub(old_pattern, new_signature, content)
        logger.info(f"Updated UMAP parameters: n_neighbors={params['n_neighbors']}, min_dist={params['min_dist']}, n_components={params['n_components_clustering']}")
    
    # Apply HDBSCAN parameters
    if 'min_cluster_size_factor' in params:
        # Update the scaling calculation
        old_pattern = r'scaled_min_cluster = max\(5, len\(texts\) // \(target_topics \* 2\)\)'
        new_calculation = f"scaled_min_cluster = max(5, int(len(texts) * {params['min_cluster_size_factor']}))"
        content = re.sub(old_pattern, new_calculation, content)
        logger.info(f"Updated HDBSCAN min_cluster_size_factor: {params['min_cluster_size_factor']}")
    
    if 'min_samples_factor' in params:
        # Update min_samples calculation
        old_pattern = r'min_samples=max\(3, scaled_min_cluster // 3\)'
        new_calculation = f"min_samples=max(3, int(scaled_min_cluster * {params['min_samples_factor']}))"
        content = re.sub(old_pattern, new_calculation, content)
        logger.info(f"Updated HDBSCAN min_samples_factor: {params['min_samples_factor']}")
    
    if 'cluster_selection_epsilon' in params:
        # Update cluster_selection_epsilon
        old_pattern = r'cluster_selection_epsilon=[\d.]+'
        new_value = f"cluster_selection_epsilon={params['cluster_selection_epsilon']}"
        content = re.sub(old_pattern, new_value, content)
        logger.info(f"Updated HDBSCAN cluster_selection_epsilon: {params['cluster_selection_epsilon']}")
    
    # Apply BERTopic model UMAP parameters (both GPU and CPU versions)
    if 'n_components_clustering' in params:
        # Update n_components in both cumlUMAP and umap.UMAP calls
        old_pattern = r'n_components=\d+,\s*#.*dims.*'
        new_value = f"n_components={params['n_components_clustering']},  # Optimized from hyperparameter tuning"
        content = re.sub(old_pattern, new_value, content)
        logger.info(f"Updated BERTopic UMAP n_components: {params['n_components_clustering']}")
    
    if 'min_samples_factor' in params:
        # Update the min_samples calculation in both HDBSCAN versions with new comment
        old_pattern = r'min_samples=max\(3, int\(scaled_min_cluster \* [\d.]+\)\), # Use optimized min_samples_factor'
        new_value = f"min_samples=max(3, int(scaled_min_cluster * {params['min_samples_factor']})), # Use optimized min_samples_factor"
        content = re.sub(old_pattern, new_value, content)
        logger.info(f"Updated BERTopic HDBSCAN min_samples_factor: {params['min_samples_factor']}")
    
    # Apply Vectorizer parameters
    if 'max_features' in params:
        old_pattern = r'max_features=\d+'
        new_value = f"max_features={params['max_features']}"
        content = re.sub(old_pattern, new_value, content)
        logger.info(f"Updated Vectorizer max_features: {params['max_features']}")
    
    if 'min_df' in params:
        old_pattern = r'min_df=\d+'
        new_value = f"min_df={params['min_df']}"
        content = re.sub(old_pattern, new_value, content)
        logger.info(f"Updated Vectorizer min_df: {params['min_df']}")
    
    if 'max_df' in params:
        old_pattern = r'max_df=[\d.]+'
        new_value = f"max_df={params['max_df']}"
        content = re.sub(old_pattern, new_value, content)
        logger.info(f"Updated Vectorizer max_df: {params['max_df']}")
    
    # Write the updated content
    with open(embed_file_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Successfully applied optimized parameters to {embed_file_path}")

def main():
    parser = argparse.ArgumentParser(description='Apply optimized parameters from hyperparameter optimization log')
    parser.add_argument('log_file', help='Path to the optimization log file')
    parser.add_argument('--embed-file', default='embed_generate.py', 
                       help='Path to embed_generate.py file (default: embed_generate.py)')
    parser.add_argument('--no-backup', action='store_true',
                       help='Do not create backup of original file')
    parser.add_argument('--dry-run', action='store_true',
                       help='Parse parameters but do not modify files')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    try:
        # Parse parameters from log file
        logger.info(f"Parsing parameters from: {args.log_file}")
        params = parse_log_file(args.log_file)
        
        logger.info("Extracted parameters:")
        for param, value in params.items():
            logger.info(f"  {param}: {value}")
        
        if args.dry_run:
            logger.info("Dry run mode - no files will be modified")
            return
        
        # Apply parameters to embed_generate.py
        apply_parameters_to_embed_generate(
            params, 
            args.embed_file, 
            backup=not args.no_backup
        )
        
        logger.info("✅ Successfully applied optimized parameters!")
        logger.info(f"You can now run: python {args.embed_file} with optimized settings")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
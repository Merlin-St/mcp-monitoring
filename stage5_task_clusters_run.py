#!/usr/bin/env python3
"""
Simplified ONET Task Clustering Pipeline - Semantic L1 Assignment

This script uses the new approach:
1. Build complete 2-level hierarchy with semantic Level 1 assignment
2. Generate Level 2 cluster names via LLM (Level 1 names are predefined)
3. Run validation tasks
4. Output results and summary

Usage:
    python stage5_task_clusters_run.py --k2 400
    python stage5_task_clusters_run.py --k2 400 --skip-validation
    python stage5_task_clusters_run.py --model openai/o3-mini --max-connections 100
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import tempfile

# Import inspect_ai analysis modules for confidence intervals
try:
    from inspect_ai.analysis.beta import evals_df, scores_df
    INSPECT_ANALYSIS_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("inspect_ai.analysis.beta not available - confidence intervals will not be calculated")
    INSPECT_ANALYSIS_AVAILABLE = False

# Import our modules
from stage5_task_clusters_embeddings import build_two_level_hierarchy, get_cluster_statistics
from stage5_task_clusters_data import (
    load_onet_tasks, update_cluster_csv, get_cluster_info, get_cluster_info_with_boundaries,
    prepare_validation_samples
)
from stage5_task_clusters_llm import (
    process_naming_results, 
    process_validation_results
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stage5_task_clusters_run.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_inspect_eval(script_path: str, task_name: str, eval_name: str, model: str = 'anthropic/claude-sonnet-4-20250514', max_connections: int = 50) -> str:
    """Run inspect eval and return log directory"""
    log_dir = f"logs/{eval_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    cmd = [
        'inspect', 'eval', f'{script_path}@{task_name}',
        '--model', model,
        '--max-connections', str(max_connections),
        '--temperature', '0',
        '--log-dir', log_dir
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error("Inspect eval failed:")
        logger.error(f"STDOUT: {result.stdout}")
        logger.error(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Inspect evaluation failed with return code {result.returncode}")
    
    logger.info(f"Inspect eval completed, logs in: {log_dir}")
    return log_dir

def calculate_confidence_interval(accuracy, stderr, confidence_level=0.95):
    """Calculate confidence interval for accuracy score using standard error"""
    if stderr is None or np.isnan(stderr):
        return None, None
    
    # For 95% confidence interval, use z-score of 1.96
    z_score = 1.96 if confidence_level == 0.95 else 2.576 if confidence_level == 0.99 else 1.645
    margin_of_error = z_score * stderr
    
    lower_bound = max(0.0, accuracy - margin_of_error)  # Clamp to [0, 1]
    upper_bound = min(1.0, accuracy + margin_of_error)
    
    return lower_bound, upper_bound

def get_validation_confidence_intervals(log_dir, task_type):
    """Extract confidence intervals from Inspect AI logs for validation task"""
    if not INSPECT_ANALYSIS_AVAILABLE:
        return None
    
    try:
        # Get evaluation summary with confidence metrics
        evals = evals_df(logs=log_dir)
        
        # Filter for the specific task type if multiple tasks exist
        if len(evals) > 0:
            # Take the most recent evaluation
            eval_row = evals.iloc[-1]
            
            accuracy = eval_row.get('score_accuracy')
            stderr = eval_row.get('score_accuracy_stderr')
            
            if accuracy is not None and stderr is not None:
                lower_95, upper_95 = calculate_confidence_interval(accuracy, stderr, 0.95)
                
                return {
                    'accuracy': accuracy,
                    'stderr': stderr,
                    'ci_95_lower': lower_95,
                    'ci_95_upper': upper_95,
                    'ci_95_formatted': f"[{lower_95:.3f}, {upper_95:.3f}]" if lower_95 is not None else None
                }
        
        return None
        
    except Exception as e:
        logger.warning(f"Could not extract confidence intervals for {task_type}: {e}")
        return None

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, 'item') and hasattr(obj, 'size'):  # numpy scalar or small array
        if obj.size == 1:
            return obj.item()
        else:
            return obj.tolist()  # Convert arrays to list
    elif hasattr(obj, 'item'):  # other numpy-like objects with .item()
        try:
            return obj.item()
        except ValueError:
            return str(obj)  # fallback to string representation
    else:
        return obj

def main():
    parser = argparse.ArgumentParser(description='ONET Task Clustering Pipeline')
    parser.add_argument('--k2', type=int, default=400,
                       help='Number of Level 2 clusters (default: 400)')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip validation tasks')
    parser.add_argument('--only-validation', action='store_true',
                       help='Only run validation tasks (skip clustering and naming)')
    parser.add_argument('--addcluster2names', action='store_true',
                       help='Only add Level 2 cluster names to existing CSV (skip clustering)')
    parser.add_argument('--log-dir', default='logs',
                       help='Directory for Inspect logs')
    parser.add_argument('--l1', choices=['semantic', 'natural'], default='semantic',
                       help='Level 1 clustering approach: semantic (cosine similarity to predefined categories) or natural (HDBSCAN clustering)')
    parser.add_argument('--model', default='anthropic/claude-sonnet-4-20250514',
                       help='Model to use for Inspect eval (default: anthropic/claude-sonnet-4-20250514)')
    parser.add_argument('--max-connections', type=int, default=50,
                       help='Maximum connections for Inspect eval (default: 50)')
    parser.add_argument('--contrastive', type=str, choices=['yes', 'no'], default='yes',
                       help='Include boundary tasks for contrastive cluster naming (default: yes)')
    parser.add_argument('--boundary-tasks', type=int, default=8,
                       help='Number of boundary tasks to include when contrastive=yes (default: 8)')
    
    args = parser.parse_args()
    
    # Check for conflicting arguments
    if args.skip_validation and args.only_validation:
        parser.error("Cannot use --skip-validation and --only-validation together")
    if args.addcluster2names and (args.only_validation or args.skip_validation):
        parser.error("Cannot use --addcluster2names with --only-validation or --skip-validation")
    if args.addcluster2names and args.only_validation:
        parser.error("Cannot use --addcluster2names and --only-validation together")
    
    if args.l1 == 'natural':
        logger.info(f"Starting ONET task clustering pipeline with k2={args.k2}, L1=natural HDBSCAN clustering")
        l1_approach = 'natural_l1_clustering'
        l1_count = 'auto'
    else:
        logger.info(f"Starting ONET task clustering pipeline with k2={args.k2}, L1=12 semantic categories")
        l1_approach = 'semantic_l1_assignment'
        l1_count = 12
    
    # Initialize summary
    summary = {
        'parameters': {'k2': args.k2, 'k1': l1_count, 'l1_approach': args.l1},
        'generated_at': datetime.now().isoformat(),
        'approach': l1_approach,
        'statistics': {},
        'validation_scores': {}
    }
    
    try:
        # Check if only adding cluster 2 names mode
        if args.addcluster2names:
            logger.info("Running in add-cluster2-names-only mode")
            # Load existing CSV with cluster assignments
            csv_file = 'stage5_tasks_cluster_names.csv'
            if not Path(csv_file).exists():
                logger.error(f"Cannot add cluster names: {csv_file} not found. Run full pipeline first.")
                sys.exit(1)
            df = pd.read_csv(csv_file)
            
            # Check if Level 2 cluster names already exist
            if 'level2_name' in df.columns and not df['level2_name'].isna().all():
                logger.warning("Level 2 cluster names already exist in CSV. Overwriting...")
            
            summary['statistics']['total_tasks'] = len(df)
            summary['statistics']['level2_clusters'] = df['level2_cluster'].nunique()
            summary['statistics']['level1_categories'] = df['level1_cluster'].nunique()
            logger.info(f"Loaded {len(df)} tasks from existing CSV")
            
            # Find existing Level 2 naming results
            logger.info("Looking for existing Level 2 cluster naming results")
            l2_cluster_info = get_cluster_info(df, level='level2')
            
            # Find the most recent l2_naming log directory
            logs_dir = Path('logs')
            l2_naming_dirs = list(logs_dir.glob('l2_naming_*'))
            
            if not l2_naming_dirs:
                logger.error("No existing l2_naming results found in logs/. Run the full pipeline first to generate cluster names.")
                sys.exit(1)
            
            # Use the most recent l2_naming directory
            log_dir = str(max(l2_naming_dirs, key=lambda p: p.stat().st_mtime))
            logger.info(f"Using existing l2_naming results from: {log_dir}")
            
            # Process results
            l2_names = process_naming_results(log_dir, expected_clusters=sorted(l2_cluster_info.keys()))
            df = update_cluster_csv(df, level2_names=l2_names)
            
            # Save updated CSV
            df.to_csv(csv_file, index=False)
            logger.info(f"✅ Level 2 naming complete - updated {csv_file}")
            
            # Update summary and exit
            summary['statistics']['level2_stats'] = get_cluster_statistics(df, 'level2_cluster')
            summary['statistics']['level1_stats'] = get_cluster_statistics(df, 'level1_cluster')
            
            logger.info("✅ Add cluster 2 names operation complete")
            return
        
        # Check if only validation mode
        elif args.only_validation:
            logger.info("Running in validation-only mode")
            # Load existing CSV with all cluster assignments
            csv_file = 'stage5_tasks_cluster_names.csv'
            if not Path(csv_file).exists():
                logger.error(f"Cannot run validation: {csv_file} not found. Run full pipeline first.")
                sys.exit(1)
            df = pd.read_csv(csv_file)
            summary['statistics']['total_tasks'] = len(df)
            summary['statistics']['level2_clusters'] = df['level2_cluster'].nunique()
            summary['statistics']['level1_categories'] = df['level1_cluster'].nunique()
            logger.info(f"Loaded {len(df)} tasks from existing CSV")
        else:
            # Step 1: Load ONET tasks
            logger.info("Step 1: Loading ONET tasks")
            df = load_onet_tasks()
            summary['statistics']['total_tasks'] = len(df)
        
        if not args.only_validation:
            # Step 2: Build complete 2-level hierarchy using chosen approach
            if args.l1 == 'natural':
                logger.info(f"Step 2: Building complete hierarchy - {args.k2} Level 2 clusters → natural Level 1 HDBSCAN clustering")
            else:
                logger.info(f"Step 2: Building complete hierarchy - {args.k2} Level 2 clusters → 12 Level 1 semantic categories")
            df, hierarchy_metadata = build_two_level_hierarchy(df, level2_clusters=args.k2, l1_approach=args.l1)
            
            # Update summary with hierarchy results
            summary['statistics']['level2_clusters'] = hierarchy_metadata['level2_clusters']
            summary['statistics']['level1_categories'] = hierarchy_metadata['level1_categories']
            
            # Add assignment quality metrics (only available for semantic approach)
            if args.l1 == 'semantic' and 'avg_similarity' in hierarchy_metadata['assignment_details']:
                summary['statistics']['assignment_quality'] = {
                    'avg_similarity': hierarchy_metadata['assignment_details']['avg_similarity'],
                    'min_similarity': hierarchy_metadata['assignment_details']['min_similarity'],
                    'max_similarity': hierarchy_metadata['assignment_details']['max_similarity'],
                    'category_avg_similarities': hierarchy_metadata['assignment_details']['category_avg_similarities']
                }
            summary['hierarchy_metadata'] = hierarchy_metadata
            
            # Get cluster statistics
            l2_stats = get_cluster_statistics(df, 'level2_cluster')
            l1_stats = get_cluster_statistics(df, 'level1_cluster')
            summary['statistics']['level2_stats'] = l2_stats
            summary['statistics']['level1_stats'] = l1_stats
            
            if args.l1 == 'natural':
                logger.info("✅ Hierarchy building complete - Level 1 clusters created via natural HDBSCAN clustering")
            else:
                logger.info("✅ Hierarchy building complete - Level 1 categories assigned via semantic similarity")
                logger.info("ℹ️  Level 1 names are predefined semantic categories (no LLM naming needed)")
        
        if not args.only_validation:
            # Step 3: Generate Level 2 cluster names via LLM
            logger.info("Step 3: Generating Level 2 cluster names via LLM")
            
            # Choose cluster info method based on contrastive flag
            if args.contrastive == 'yes':
                logger.info(f"Using contrastive naming with {args.boundary_tasks} boundary tasks per cluster")
                # Load task embeddings from cache (they should exist after Step 2)
                embeddings_file = 'stage5_task_clusters_embeddings_onet.npz'
                if not Path(embeddings_file).exists():
                    logger.error(f"Task embeddings cache not found: {embeddings_file}. Cannot use contrastive mode.")
                    raise FileNotFoundError(f"Task embeddings required for contrastive mode: {embeddings_file}")
                
                embeddings_data = np.load(embeddings_file)
                task_embeddings = embeddings_data['embeddings']
                logger.info(f"Loaded task embeddings: shape {task_embeddings.shape}")
                
                l2_cluster_info = get_cluster_info_with_boundaries(
                    df, task_embeddings, level='level2', boundary_tasks=args.boundary_tasks
                )
            else:
                logger.info("Using standard naming (tasks within cluster only)")
                l2_cluster_info = get_cluster_info(df, level='level2')
            
            # Create temporary module for Inspect
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(f"""
from inspect_ai import task
from stage5_task_clusters_llm import generate_cluster_names

clusters_info = {repr(l2_cluster_info)}

@task
def l2_naming_task():
    return generate_cluster_names(clusters_info, level='level2')
""")
                temp_module = f.name
            
            # Run Inspect evaluation
            log_dir = run_inspect_eval(temp_module, 'l2_naming_task', 'l2_naming', args.model, args.max_connections)
            Path(temp_module).unlink()  # Clean up temp file
            
            # Process results
            l2_names = process_naming_results(log_dir, expected_clusters=sorted(l2_cluster_info.keys()))
            df = update_cluster_csv(df, level2_names=l2_names)
            
            logger.info("✅ Level 2 naming complete")
        
        # Save updated CSV with cluster assignments and names
        if not args.only_validation:
            output_csv = 'stage5_tasks_cluster_names.csv'
            df.to_csv(output_csv, index=False)
            logger.info(f"Saved complete task hierarchy to: {output_csv}")
        
        # Step 4: Validation (unless skipped)
        if not args.skip_validation:
            logger.info("Step 4: Running validation tasks")
            
            # Define validation tasks
            validation_tasks = [
                ('l3_to_l2', 'Level 3 (task) to Level 2 cluster assignment'),
                ('l2_to_l1', 'Level 2 cluster to Level 1 category assignment'),
                ('l3_to_l1', 'Level 3 (task) to Level 1 category direct assignment'),
                ('subset_l2_l3', 'Level 3 (task) to Level 2 cluster assignment with focused subset')
            ]
            
            # Create JSONL files for each validation type
            logger.info("Creating validation JSONL files")
            for task_type, _ in validation_tasks:
                validation_samples = prepare_validation_samples(df, task_type)
                jsonl_file = f'stage5_task_clusters_validation_{task_type}.jsonl'
                
                with open(jsonl_file, 'w') as f:
                    for sample in validation_samples:
                        # Convert numpy types to native Python types for JSON serialization
                        sample_clean = {}
                        for key, value in sample.items():
                            if hasattr(value, 'item'):  # numpy scalar
                                sample_clean[key] = value.item()
                            elif isinstance(value, dict):
                                # Clean metadata dict
                                sample_clean[key] = {k: v.item() if hasattr(v, 'item') else v for k, v in value.items()}
                            else:
                                sample_clean[key] = value
                        f.write(json.dumps(sample_clean) + '\n')
                
                logger.info(f"Created {jsonl_file} with {len(validation_samples)} samples")
            
            # Run validation tasks using existing functions
            for task_type, description in validation_tasks:
                logger.info(f"Running {description}")
                
                # Use existing validation tasks directly
                log_dir = run_inspect_eval('stage5_task_clusters_llm.py', f'{task_type}_validation', f'{task_type}_validation', args.model, args.max_connections)
                
                # Process validation results with validation_type parameter
                validation_results = process_validation_results(log_dir, task_type)
                
                # Add confidence intervals to validation results
                confidence_data = get_validation_confidence_intervals(log_dir, task_type)
                if confidence_data:
                    validation_results.update(confidence_data)
                    ci_str = confidence_data.get('ci_95_formatted', '')
                    logger.info(f"✅ {description}: {validation_results['accuracy']:.3f} accuracy {ci_str}")
                else:
                    logger.info(f"✅ {description}: {validation_results['accuracy']:.3f} accuracy")
                
                summary['validation_scores'][task_type] = validation_results
        
        # Final summary
        logger.info("\n" + "="*50)
        logger.info("PIPELINE COMPLETE - SUMMARY")
        logger.info("="*50)
        logger.info(f"Total tasks: {summary['statistics']['total_tasks']}")
        logger.info(f"Level 2 clusters: {summary['statistics']['level2_clusters']}")
        if args.l1 == 'natural':
            l1_count = summary['statistics']['level1_categories']
            logger.info(f"Level 1 categories: {l1_count} (natural HDBSCAN)")
        else:
            logger.info("Level 1 categories: 12 (semantic)")
        
        if 'assignment_quality' in summary['statistics']:
            quality = summary['statistics']['assignment_quality']
            logger.info(f"L2→L1 assignment quality: {quality['avg_similarity']:.4f} avg similarity")
        
        if summary['validation_scores']:
            logger.info("Validation scores:")
            for task_type, results in summary['validation_scores'].items():
                accuracy = results.get('accuracy', 0)
                ci_formatted = results.get('ci_95_formatted', '')
                if ci_formatted:
                    logger.info(f"  {task_type}: {accuracy:.3f} {ci_formatted}")
                else:
                    logger.info(f"  {task_type}: {accuracy:.3f}")
        
        # Save summary with validation scores at top (convert numpy types to Python types for JSON serialization)
        summary_clean = convert_numpy_types(summary)
        
        # Reorder summary to put validation scores first
        summary_ordered = {}
        if summary_clean.get('validation_scores'):
            summary_ordered['validation_scores'] = summary_clean['validation_scores']
        
        # Add all other keys except validation_scores
        for key, value in summary_clean.items():
            if key != 'validation_scores':
                summary_ordered[key] = value
        
        summary_file = 'stage5_task_clusters_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary_ordered, f, indent=2)
        logger.info(f"Saved summary to: {summary_file}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        summary['error'] = str(e)
        summary['status'] = 'failed'
        
        # Save error summary (convert numpy types to Python types for JSON serialization)
        summary_clean = convert_numpy_types(summary)
        summary_file = 'stage5_task_clusters_summary_error.json'
        with open(summary_file, 'w') as f:
            json.dump(summary_clean, f, indent=2)
        
        sys.exit(1)

if __name__ == "__main__":
    main()
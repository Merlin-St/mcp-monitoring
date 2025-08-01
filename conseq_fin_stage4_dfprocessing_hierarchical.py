#!/usr/bin/env python3
"""
Stage 4 O*NET Classification - Hierarchical DataFrame Processing

Processes hierarchical task mapping results and extracts:
- Level 1: Top-level category (10 categories)
- Level 2: Middle-level cluster (~400 clusters)
- Level 3: Individual O*NET task

Usage:
    python conseq_fin_stage4_dfprocessing_hierarchical.py
"""

import json
import re
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conseq_fin_stage4_dfprocessing_hierarchical.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_mappings() -> Dict[str, Any]:
    """Load the option ID to task mappings"""
    mappings_file = "conseq_fin_stage4_mappings.json"
    if Path(mappings_file).exists():
        with open(mappings_file, 'r') as f:
            return json.load(f)
    return {}

def load_hierarchy() -> Dict[str, Any]:
    """Load the full task hierarchy"""
    hierarchy_file = "conseq_fin_stage4_hierarchy.json"
    with open(hierarchy_file, 'r') as f:
        return json.load(f)

def extract_hierarchical_classification(response: str, mappings: Dict[str, Any], hierarchy: Dict[str, Any]) -> Dict[str, str]:
    """Extract hierarchical classification from response"""
    result = {
        'level1_category': '',
        'level1_description': '',
        'level2_cluster': '',
        'level2_description': '',
        'level3_task': '',
        'classification_level': '',
        'raw_response': response
    }
    
    # Try to extract option ID
    response_clean = response.strip()
    
    # Look for pattern like L1_, L2_, L3_
    pattern = r'(L[123]_[\w_]+)'
    match = re.search(pattern, response_clean)
    
    if match:
        option_id = match.group(1)
        
        if option_id.startswith('L1_'):
            # Level 1 classification
            result['classification_level'] = '1'
            category_key = option_id.replace('L1_', '')
            if category_key in hierarchy['top_level']:
                result['level1_category'] = category_key
                result['level1_description'] = hierarchy['top_level'][category_key]
                
        elif option_id.startswith('L2_'):
            # Level 2 classification
            result['classification_level'] = '2'
            if option_id in mappings:
                mapping = mappings[option_id]
                result['level1_category'] = mapping['top_level']
                result['level1_description'] = hierarchy['top_level'].get(mapping['top_level'], '')
                result['level2_cluster'] = str(mapping['cluster_id'])
                result['level2_description'] = mapping['description']
                
        elif option_id.startswith('L3_'):
            # Level 3 classification
            result['classification_level'] = '3'
            if option_id in mappings:
                mapping = mappings[option_id]
                result['level1_category'] = mapping['top_level']
                result['level1_description'] = hierarchy['top_level'].get(mapping['top_level'], '')
                result['level2_cluster'] = str(mapping['cluster_id'])
                result['level3_task'] = mapping['task']
                
                # Find level 2 description
                for clusters in hierarchy['middle_level'].values():
                    for cluster in clusters:
                        if cluster['cluster_id'] == mapping['cluster_id']:
                            result['level2_description'] = cluster['description']
                            break
    else:
        # Fallback: Try to match against actual task text
        # This handles cases where the model returned the task description instead of ID
        response_lower = response_clean.lower()
        
        # Search through all tasks
        for top_key, clusters in hierarchy['middle_level'].items():
            for cluster in clusters:
                for task in cluster.get('representative_tasks', []):
                    if task.lower() in response_lower or response_lower in task.lower():
                        result['classification_level'] = '3'
                        result['level1_category'] = top_key
                        result['level1_description'] = hierarchy['top_level'].get(top_key, '')
                        result['level2_cluster'] = str(cluster['cluster_id'])
                        result['level2_description'] = cluster['description']
                        result['level3_task'] = task
                        return result
    
    return result

def process_hierarchical_eval(eval_file: Path, mappings: Dict[str, Any], hierarchy: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process hierarchical task mapping evaluation file"""
    results = []
    
    import tempfile
    temp_dir = tempfile.mkdtemp()
    temp_file = Path(temp_dir) / eval_file.name
    
    import shutil
    shutil.copy2(eval_file, temp_file)
    
    try:
        from inspect_ai.analysis.beta import samples_df, messages_df
        
        samples = samples_df(temp_dir)
        messages = messages_df(temp_dir)
        
        for idx, sample_row in samples.iterrows():
            sample_id = sample_row.get("sample_id", f"sample_{idx}")
            
            # Get tool info from user message
            user_msg = messages[(messages['sample_id'] == sample_id) & (messages['role'] == 'user')]
            tool_info = {}
            if not user_msg.empty:
                try:
                    content = user_msg.iloc[0]['content']
                    data = json.loads(content)
                    tool_info = {
                        'tool_name': data.get('tool_name', ''),
                        'tool_description': data.get('tool_description', ''),
                        'server_name': data.get('server_name', ''),
                        'server_description': data.get('server_description', '')
                    }
                except:
                    pass
            
            # Get classification from assistant response or scorer answer
            classification_response = ''
            
            # First try scorer answer
            if 'answer_task_mapping_scorer' in sample_row and sample_row['answer_task_mapping_scorer']:
                classification_response = str(sample_row['answer_task_mapping_scorer'])
            else:
                # Fallback to assistant message
                assistant_msg = messages[(messages['sample_id'] == sample_id) & (messages['role'] == 'assistant')]
                if not assistant_msg.empty:
                    classification_response = assistant_msg.iloc[0]['content']
            
            # Extract hierarchical classification
            classification = extract_hierarchical_classification(classification_response, mappings, hierarchy)
            
            # Combine results
            result = {
                'sample_id': sample_id,
                **tool_info,
                **classification,
                'score': sample_row.get('score_task_mapping_scorer', 0)
            }
            
            results.append(result)
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return results

def generate_hierarchical_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate summary statistics for hierarchical classification"""
    summary = {
        'processing_timestamp': datetime.now().isoformat(),
        'total_tools': len(df)
    }
    
    # Classification level distribution
    level_dist = df['classification_level'].value_counts().to_dict()
    summary['classification_levels'] = {
        'level_1': level_dist.get('1', 0),
        'level_2': level_dist.get('2', 0),
        'level_3': level_dist.get('3', 0),
        'unclassified': len(df[df['classification_level'] == ''])
    }
    
    # Level 1 category distribution
    level1_valid = df[df['level1_category'] != '']
    if len(level1_valid) > 0:
        level1_dist = level1_valid['level1_category'].value_counts().to_dict()
        summary['level1_distribution'] = level1_dist
        
        # Add descriptions
        summary['level1_categories'] = {}
        for cat in level1_dist.keys():
            desc = level1_valid[level1_valid['level1_category'] == cat]['level1_description'].iloc[0]
            summary['level1_categories'][cat] = {
                'count': level1_dist[cat],
                'description': desc
            }
    
    # Level 2 cluster analysis
    level2_valid = df[df['level2_cluster'] != '']
    if len(level2_valid) > 0:
        summary['unique_level2_clusters'] = level2_valid['level2_cluster'].nunique()
        
        # Top clusters
        cluster_counts = level2_valid.groupby(['level2_cluster', 'level2_description']).size().reset_index(name='count')
        cluster_counts = cluster_counts.sort_values('count', ascending=False)
        
        summary['top_level2_clusters'] = []
        for _, row in cluster_counts.head(10).iterrows():
            summary['top_level2_clusters'].append({
                'cluster_id': row['level2_cluster'],
                'description': row['level2_description'],
                'count': int(row['count'])
            })
    
    # Level 3 task analysis
    level3_valid = df[df['level3_task'] != '']
    if len(level3_valid) > 0:
        summary['tools_with_specific_tasks'] = len(level3_valid)
        
        # Sample of specific tasks
        task_counts = level3_valid['level3_task'].value_counts()
        summary['sample_specific_tasks'] = []
        for task, count in task_counts.head(10).items():
            summary['sample_specific_tasks'].append({
                'task': task,
                'count': int(count)
            })
    
    return summary

def main():
    parser = argparse.ArgumentParser(description='Process hierarchical O*NET classification results')
    parser.add_argument('--eval-file', type=str, 
                       help='Specific eval file to process')
    parser.add_argument('--logs-dir', type=str, default='logs',
                       help='Directory containing .eval files')
    
    args = parser.parse_args()
    
    logger.info("Starting Hierarchical Task Mapping Processing")
    
    # Load mappings and hierarchy
    mappings = load_mappings()
    hierarchy = load_hierarchy()
    
    if not mappings:
        logger.warning("No mappings file found. Run the hierarchical inspect script first.")
    
    # Find eval file
    if args.eval_file:
        eval_file = Path(args.eval_file)
    else:
        # Find latest hierarchical task mapping eval
        log_dir = Path(args.logs_dir)
        eval_files = list(log_dir.glob('*task-mapping-task*.eval'))
        if not eval_files:
            logger.error("No task mapping evaluation files found")
            return
        eval_file = max(eval_files, key=lambda x: x.stat().st_mtime)
    
    logger.info(f"Processing eval file: {eval_file.name}")
    
    # Process results
    results = process_hierarchical_eval(eval_file, mappings, hierarchy)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Generate summary
    summary = generate_hierarchical_summary(df)
    
    # Save results
    output_data = {
        "summary": summary,
        "results": results
    }
    
    # Save JSON
    json_file = "conseq_fin_stage4_hierarchical_results.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved results to {json_file}")
    
    # Save CSV
    csv_file = "conseq_fin_stage4_hierarchical_results.csv"
    df.to_csv(csv_file, index=False)
    logger.info(f"Saved DataFrame to {csv_file}")
    
    # Save summary
    summary_file = "conseq_fin_stage4_hierarchical_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_file}")
    
    # Log key statistics
    logger.info("\n=== Hierarchical Classification Results ===")
    logger.info(f"Total tools: {summary['total_tools']}")
    logger.info(f"\nClassification levels:")
    for level, count in summary['classification_levels'].items():
        logger.info(f"  {level}: {count}")
    
    if 'level1_categories' in summary:
        logger.info(f"\nLevel 1 Categories:")
        for cat, info in summary['level1_categories'].items():
            logger.info(f"  {cat}: {info['count']} tools - {info['description'][:60]}...")
    
    if 'top_level2_clusters' in summary:
        logger.info(f"\nTop Level 2 Clusters:")
        for cluster in summary['top_level2_clusters'][:5]:
            logger.info(f"  Cluster {cluster['cluster_id']}: {cluster['count']} tools - {cluster['description'][:50]}...")
    
    logger.info("\n=== Processing Complete ===")

if __name__ == "__main__":
    main()
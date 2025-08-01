#!/usr/bin/env python3
"""
Stage 4 O*NET Classification - Final Hierarchical DataFrame Processing

Processes hierarchical classification results from the dynamic inspect script.
Maps cluster IDs back to descriptions and generates comprehensive statistics.

Usage:
    python conseq_fin_stage4_dfprocessing_hierarchical_final.py
    python conseq_fin_stage4_dfprocessing_hierarchical_final.py --eval-file specific.eval
"""

import json
import re
import argparse
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conseq_fin_stage4_dfprocessing_hierarchical_final.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HierarchicalProcessor:
    def __init__(self):
        """Initialize with hierarchy data"""
        self.hierarchy_csv = 'conseq_fin_stage4_onetclusters.csv'
        self.metadata_json = 'conseq_fin_stage4_hierarchy_metadata.json'
        
        # Load hierarchy
        if Path(self.hierarchy_csv).exists():
            self.hierarchy_df = pd.read_csv(self.hierarchy_csv)
            logger.info(f"Loaded hierarchy with {len(self.hierarchy_df)} tasks")
        else:
            logger.warning(f"Hierarchy CSV not found: {self.hierarchy_csv}")
            self.hierarchy_df = None
            
        # Load metadata
        if Path(self.metadata_json).exists():
            with open(self.metadata_json, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
            
    def extract_tool_info(self, user_message: str) -> Dict[str, str]:
        """Extract tool information from user message"""
        info = {
            'tool_name': '',
            'tool_description': '',
            'server_name': '',
            'server_description': ''
        }
        
        # Try to parse as JSON
        try:
            data = json.loads(user_message)
            info['tool_name'] = data.get('tool_name', '')
            info['tool_description'] = data.get('tool_description', '')
            info['server_name'] = data.get('server_name', '')
            info['server_description'] = data.get('server_description', '')
        except json.JSONDecodeError:
            # Fallback to text parsing
            logger.debug("Failed to parse user message as JSON, using text parsing")
            
        return info
        
    def process_classification_response(self, response: str, level: str) -> Dict[str, Any]:
        """Process classification response for a specific level"""
        result = {
            'raw_response': response,
            'cluster_id': '',
            'valid': False,
            'level': level
        }
        
        response = response.strip()
        
        # Check for valid cluster/task ID format
        if level == 'level1' and response.startswith('cluster_1_'):
            result['cluster_id'] = response
            result['valid'] = True
        elif level == 'level2' and response.startswith('cluster_2_'):
            result['cluster_id'] = response
            result['valid'] = True
        elif level == 'level3':
            # Task IDs like "11-1011.00_8823"
            if '_' in response and '-' in response.split('_')[0]:
                result['cluster_id'] = response
                result['valid'] = True
                
        return result
        
    def enrich_with_hierarchy_info(self, classification: Dict[str, Any]) -> Dict[str, Any]:
        """Add hierarchy information to classification result"""
        if not self.hierarchy_df is not None:
            return classification
            
        enriched = classification.copy()
        
        # Level 1 enrichment
        if classification.get('level1_cluster_id'):
            cluster_id = classification['level1_cluster_id']
            cluster_info = self.metadata.get('cluster_descriptions', {}).get('level1', {}).get(cluster_id, {})
            enriched['level1_size'] = cluster_info.get('size', 0)
            enriched['level1_top_occupations'] = cluster_info.get('top_occupations', [])
            
        # Level 2 enrichment
        if classification.get('level2_cluster_id'):
            cluster_id = classification['level2_cluster_id']
            cluster_info = self.metadata.get('cluster_descriptions', {}).get('level2', {}).get(cluster_id, {})
            enriched['level2_size'] = cluster_info.get('size', 0)
            enriched['level2_primary_occupation'] = cluster_info.get('primary_occupation', '')
            
        # Level 3 enrichment (specific task)
        if classification.get('level3_task_id'):
            task_id = classification['level3_task_id']
            task_row = self.hierarchy_df[self.hierarchy_df['task_id'] == task_id]
            if not task_row.empty:
                enriched['level3_task'] = task_row.iloc[0]['Task']
                enriched['level3_occupation'] = task_row.iloc[0]['Title']
                enriched['level3_onet_code'] = task_row.iloc[0]['O*NET-SOC Code']
                
        return enriched
        
    def process_eval_file(self, eval_file: Path) -> List[Dict[str, Any]]:
        """Process a single evaluation file"""
        results = []
        
        # Create temporary directory for analysis
        import tempfile
        import shutil
        temp_dir = tempfile.mkdtemp()
        temp_file = Path(temp_dir) / eval_file.name
        
        try:
            shutil.copy2(eval_file, temp_file)
            
            from inspect_ai.analysis.beta import samples_df, messages_df
            
            samples = samples_df(temp_dir)
            messages = messages_df(temp_dir)
            
            for idx, sample_row in samples.iterrows():
                sample_id = sample_row.get("sample_id", f"sample_{idx}")
                
                # Extract tool info from user message
                user_msgs = messages[(messages['sample_id'] == sample_id) & (messages['role'] == 'user')]
                tool_info = {}
                if not user_msgs.empty:
                    tool_info = self.extract_tool_info(user_msgs.iloc[0]['content'])
                
                # Get classification response
                classification = {
                    'sample_id': sample_id,
                    **tool_info,
                    'level1_cluster_id': '',
                    'level2_cluster_id': '',
                    'level3_task_id': ''
                }
                
                # Extract classification from assistant response or scorer
                if 'answer_level_scorer' in sample_row:
                    answer = sample_row['answer_level_scorer']
                    if answer:
                        # Determine level from answer format
                        if answer.startswith('cluster_1_'):
                            classification['level1_cluster_id'] = answer
                        elif answer.startswith('cluster_2_'):
                            classification['level2_cluster_id'] = answer
                        else:
                            classification['level3_task_id'] = answer
                
                # Enrich with hierarchy information
                enriched = self.enrich_with_hierarchy_info(classification)
                results.append(enriched)
                
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        return results
        
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive summary statistics"""
        summary = {
            'processing_timestamp': datetime.now().isoformat(),
            'total_tools': len(df),
            'classification_completeness': {}
        }
        
        # Classification completeness
        summary['classification_completeness'] = {
            'level1': len(df[df['level1_cluster_id'] != '']),
            'level2': len(df[df['level2_cluster_id'] != '']),
            'level3': len(df[df['level3_task_id'] != ''])
        }
        
        # Level 1 distribution
        if 'level1_cluster_id' in df.columns:
            level1_dist = df[df['level1_cluster_id'] != '']['level1_cluster_id'].value_counts()
            summary['level1_distribution'] = {
                'clusters': level1_dist.to_dict(),
                'most_common': level1_dist.head(5).to_dict() if len(level1_dist) > 0 else {}
            }
            
        # Level 2 analysis
        if 'level2_cluster_id' in df.columns:
            level2_valid = df[df['level2_cluster_id'] != '']
            if len(level2_valid) > 0:
                summary['level2_statistics'] = {
                    'unique_clusters': level2_valid['level2_cluster_id'].nunique(),
                    'avg_cluster_size': len(level2_valid) / level2_valid['level2_cluster_id'].nunique()
                }
                
                # Top occupations from Level 2
                if 'level2_primary_occupation' in df.columns:
                    occupation_dist = level2_valid['level2_primary_occupation'].value_counts()
                    summary['top_occupations'] = occupation_dist.head(10).to_dict()
                    
        # Level 3 task analysis
        if 'level3_task_id' in df.columns:
            level3_valid = df[df['level3_task_id'] != '']
            if len(level3_valid) > 0:
                summary['level3_statistics'] = {
                    'tools_with_specific_tasks': len(level3_valid),
                    'unique_tasks': level3_valid['level3_task_id'].nunique()
                }
                
                # Sample specific tasks
                if 'level3_task' in df.columns:
                    sample_tasks = []
                    for _, row in level3_valid.head(10).iterrows():
                        if row.get('level3_task'):
                            sample_tasks.append({
                                'tool': row['tool_name'],
                                'task': row['level3_task'][:100] + '...',
                                'occupation': row.get('level3_occupation', '')
                            })
                    summary['sample_task_mappings'] = sample_tasks
                    
        return summary

def main():
    parser = argparse.ArgumentParser(description='Process hierarchical O*NET classification results')
    parser.add_argument('--eval-file', type=str,
                       help='Specific eval file to process')
    parser.add_argument('--logs-dir', type=str, default='logs',
                       help='Directory containing .eval files')
    
    args = parser.parse_args()
    
    logger.info("Starting Hierarchical Classification Processing")
    
    # Initialize processor
    processor = HierarchicalProcessor()
    
    # Find eval file(s)
    if args.eval_file:
        eval_files = [Path(args.eval_file)]
    else:
        log_dir = Path(args.logs_dir)
        # Look for hierarchical classification eval files
        eval_files = list(log_dir.glob('*hierarchical*.eval'))
        if not eval_files:
            logger.error("No hierarchical evaluation files found")
            return
            
    logger.info(f"Processing {len(eval_files)} evaluation file(s)")
    
    # Process all files
    all_results = []
    for eval_file in eval_files:
        logger.info(f"Processing: {eval_file.name}")
        results = processor.process_eval_file(eval_file)
        all_results.extend(results)
        
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Generate summary
    summary = processor.generate_summary_statistics(df)
    
    # Save results
    output_data = {
        "summary": summary,
        "results": all_results
    }
    
    # Save JSON
    json_file = "conseq_fin_stage4_hierarchical_final_results.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved results to {json_file}")
    
    # Save CSV
    csv_file = "conseq_fin_stage4_hierarchical_final_results.csv"
    df.to_csv(csv_file, index=False)
    logger.info(f"Saved DataFrame to {csv_file}")
    
    # Save summary separately
    summary_file = "conseq_fin_stage4_hierarchical_final_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_file}")
    
    # Log key statistics
    logger.info("\n=== Hierarchical Classification Results ===")
    logger.info(f"Total tools processed: {summary['total_tools']}")
    
    if 'classification_completeness' in summary:
        logger.info("\nClassification completeness:")
        for level, count in summary['classification_completeness'].items():
            percentage = (count / summary['total_tools'] * 100) if summary['total_tools'] > 0 else 0
            logger.info(f"  {level}: {count} ({percentage:.1f}%)")
            
    if 'level1_distribution' in summary:
        logger.info("\nTop Level 1 clusters:")
        for cluster, count in list(summary['level1_distribution']['most_common'].items())[:5]:
            logger.info(f"  {cluster}: {count} tools")
            
    if 'top_occupations' in summary:
        logger.info("\nTop occupations (from Level 2):")
        for occupation, count in list(summary['top_occupations'].items())[:5]:
            logger.info(f"  {occupation}: {count}")
            
    logger.info("\n=== Processing Complete ===")

if __name__ == "__main__":
    main()
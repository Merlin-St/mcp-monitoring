#!/usr/bin/env python3
"""
Stage 4 O*NET Classification - DataFrame Processing

Processes the evaluation results from conseq_fin_stage4_inspect.py
by reading the .eval files and converting them to structured JSON and CSV formats.

This should be run after:
    inspect eval conseq_fin_stage4_inspect.py --model anthropic/claude-sonnet-4-20250514

Usage:
    python conseq_fin_stage4_dfprocessing.py
    python conseq_fin_stage4_dfprocessing.py --logs-dir ./custom_logs
"""

import json
import re
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conseq_fin_stage4_dfprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MODEL = "anthropic/claude-sonnet-4-20250514"

def extract_json_objects(text: str, expected_count: int = 4) -> List[Dict[str, Any]]:
    """Extract multiple JSON objects from text"""
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    json_matches = re.findall(json_pattern, text, re.DOTALL)
    
    results = []
    for match in json_matches[:expected_count]:
        try:
            obj = json.loads(match)
            results.append(obj)
        except json.JSONDecodeError:
            continue
    
    return results

def process_classification_results(samples_df: pd.DataFrame, messages_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Process classification results from DataFrames"""
    results = []
    
    # Get assistant messages
    assistant_messages = messages_df[messages_df['role'] == 'assistant']
    
    for idx, sample_row in samples_df.iterrows():
        sample_id = sample_row.get("sample_id", f"sample_{idx}")
        
        # Find assistant message for this sample
        sample_messages = assistant_messages[assistant_messages['sample_id'] == sample_id]
        
        result = {
            "sample_id": sample_id,
            "tool_id": "",
            "input_data": {},
            "raw_output": "",
            "score": sample_row.get("score_onet_classifier_scorer", 0),
            "classifications": {
                "task_mapping": None,
                "collaboration_pattern": None,
                "automation_level": None,
                "tool_replacement": None
            },
            "errors": []
        }
        
        # Parse input data
        user_messages = messages_df[
            (messages_df['sample_id'] == sample_id) & 
            (messages_df['role'] == 'user')
        ]
        
        if not user_messages.empty:
            user_content = user_messages.iloc[0]['content']
            try:
                # Extract JSON from user content
                if "{" in user_content and "}" in user_content:
                    start = user_content.find('{')
                    end = user_content.rfind('}')
                    json_str = user_content[start:end+1]
                    input_data = json.loads(json_str)
                    result["input_data"] = input_data
                    
                    # Get tool_id from metadata if available
                    if "metadata" in sample_row:
                        metadata = sample_row["metadata"]
                        if isinstance(metadata, str):
                            metadata = json.loads(metadata)
                        result["tool_id"] = metadata.get("id", "")
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                result["errors"].append(f"Failed to parse input: {str(e)}")
        
        # Get assistant response
        if not sample_messages.empty:
            result["raw_output"] = sample_messages.iloc[0]['content']
            
            # Extract the 4 JSON responses
            try:
                json_objects = extract_json_objects(result["raw_output"], 4)
                
                if len(json_objects) >= 1:
                    result["classifications"]["task_mapping"] = json_objects[0]
                
                if len(json_objects) >= 2:
                    result["classifications"]["collaboration_pattern"] = json_objects[1]
                
                if len(json_objects) >= 3:
                    result["classifications"]["automation_level"] = json_objects[2]
                
                if len(json_objects) >= 4:
                    result["classifications"]["tool_replacement"] = json_objects[3]
                
                if len(json_objects) < 4:
                    result["errors"].append(f"Only found {len(json_objects)} JSON responses, expected 4")
                    
            except Exception as e:
                result["errors"].append(f"Failed to extract classifications: {str(e)}")
        
        results.append(result)
    
    return results

def create_analysis_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a flattened DataFrame for analysis"""
    rows = []
    
    for result in results:
        row = {
            'tool_id': result['tool_id'],
            'sample_id': result['sample_id'],
            'score': result['score'],
            'has_errors': len(result.get('errors', [])) > 0,
            'error_count': len(result.get('errors', []))
        }
        
        # Add input data fields
        input_data = result.get('input_data', {})
        row['tool_name'] = input_data.get('tool_name', '')
        row['tool_description'] = input_data.get('tool_description', '')
        row['server_name'] = input_data.get('server_name', '')
        row['server_description'] = input_data.get('server_description', '')
        
        # Task mapping classification
        task_mapping = result['classifications'].get('task_mapping', {})
        if task_mapping:
            row['top_level_category'] = task_mapping.get('top_level_category', '')
            row['top_level_number'] = task_mapping.get('top_level_number', '')
            row['specific_task'] = task_mapping.get('specific_task', '')
            row['occupation'] = task_mapping.get('occupation', '')
            row['task_confidence'] = task_mapping.get('confidence', '')
        
        # Collaboration pattern
        collab = result['classifications'].get('collaboration_pattern', {})
        if collab:
            row['collaboration_pattern'] = collab.get('pattern', '')
            row['collab_confidence'] = collab.get('confidence', '')
        
        # Automation level
        auto = result['classifications'].get('automation_level', {})
        if auto:
            row['automation_level'] = auto.get('level', -1)
            row['automation_description'] = auto.get('level_description', '')
        
        # Tool replacement
        replacement = result['classifications'].get('tool_replacement', {})
        if replacement:
            replaced_tools = replacement.get('replaced_tools', [])
            row['replaced_tools_count'] = len(replaced_tools)
            row['replaced_tools'] = ';'.join(replaced_tools) if replaced_tools else ''
            row['replacement_confidence'] = replacement.get('confidence', '')
        
        rows.append(row)
    
    return pd.DataFrame(rows)

def generate_summary_statistics(df: pd.DataFrame, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary statistics from processed results"""
    valid_results = df[df['score'] > 0]
    
    summary = {
        'processing_timestamp': datetime.now().isoformat(),
        'model': MODEL,
        'total_tools': len(results),
        'valid_classifications': len(valid_results),
        'error_rate': (len(df) - len(valid_results)) / len(df) * 100 if len(df) > 0 else 0
    }
    
    # Task mapping statistics
    if 'top_level_number' in valid_results.columns:
        top_level_dist = valid_results['top_level_number'].value_counts().to_dict()
        summary['top_level_distribution'] = {int(k): v for k, v in top_level_dist.items() if pd.notna(k)}
        summary['top_occupations'] = valid_results['occupation'].value_counts().head(10).to_dict()
    
    # Collaboration patterns
    if 'collaboration_pattern' in valid_results.columns:
        collab_dist = valid_results['collaboration_pattern'].value_counts().to_dict()
        summary['collaboration_patterns'] = collab_dist
        
        # Calculate automation vs augmentation
        automation_patterns = ['Directive', 'Feedback Loop']
        augmentation_patterns = ['Task Iteration', 'Learning', 'Validation']
        
        automation_count = valid_results[valid_results['collaboration_pattern'].isin(automation_patterns)].shape[0]
        augmentation_count = valid_results[valid_results['collaboration_pattern'].isin(augmentation_patterns)].shape[0]
        
        summary['automation_vs_augmentation'] = {
            'automation': automation_count,
            'augmentation': augmentation_count,
            'automation_percentage': automation_count / (automation_count + augmentation_count) * 100 if (automation_count + augmentation_count) > 0 else 0
        }
    
    # Automation levels
    if 'automation_level' in valid_results.columns:
        level_dist = valid_results['automation_level'].value_counts().sort_index().to_dict()
        summary['automation_levels'] = {int(k): v for k, v in level_dist.items() if k >= 0}
        summary['avg_automation_level'] = valid_results[valid_results['automation_level'] >= 0]['automation_level'].mean()
    
    # Tool replacement
    if 'replaced_tools_count' in valid_results.columns:
        summary['tools_replacing_onet'] = (valid_results['replaced_tools_count'] > 0).sum()
        summary['avg_tools_replaced'] = valid_results['replaced_tools_count'].mean()
        
        # Most commonly replaced tools
        all_replaced = []
        for tools_str in valid_results['replaced_tools'].dropna():
            if tools_str:
                all_replaced.extend(tools_str.split(';'))
        
        if all_replaced:
            from collections import Counter
            tool_counts = Counter(all_replaced)
            summary['most_replaced_tools'] = dict(tool_counts.most_common(10))
    
    return summary

def main():
    parser = argparse.ArgumentParser(description='Process O*NET classification results')
    parser.add_argument('--logs-dir', type=str, default='conseq_fin_stage4_logs',
                       help='Directory containing .eval files')
    parser.add_argument('--eval-file', type=str,
                       help='Specific .eval file to process')
    
    args = parser.parse_args()
    
    logger.info("Starting Stage 4 DataFrame Processing")
    
    # Find eval files
    if args.eval_file:
        eval_files = [Path(args.eval_file)]
    else:
        log_dir = Path(args.logs_dir)
        if not log_dir.exists():
            # Try default logs directory
            log_dir = Path('logs')
        
        if not log_dir.exists():
            logger.error(f"Log directory {args.logs_dir} not found")
            return
        
        eval_files = list(log_dir.glob("*onet-classification-task*.eval"))
        if not eval_files:
            logger.error(f"No O*NET classification .eval files found in {log_dir}")
            return
    
    # Process most recent file
    latest_file = max(eval_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Processing: {latest_file.name}")
    
    # Create temporary directory for single file processing
    import tempfile
    temp_dir = tempfile.mkdtemp()
    temp_file = Path(temp_dir) / latest_file.name
    
    import shutil
    shutil.copy2(latest_file, temp_file)
    
    try:
        # Load DataFrames
        from inspect_ai.analysis.beta import samples_df, messages_df
        
        samples = samples_df(temp_dir)
        messages = messages_df(temp_dir)
        
        logger.info(f"Loaded {len(samples)} samples")
        logger.info(f"Loaded {len(messages)} messages")
        
        # Process results
        results = process_classification_results(samples, messages)
        
        # Create analysis DataFrame
        analysis_df = create_analysis_dataframe(results)
        
        # Generate summary
        summary = generate_summary_statistics(analysis_df, results)
        
        # Add model usage if available
        if "model_usage" in samples.columns:
            total_input_tokens = 0
            total_output_tokens = 0
            
            for _, row in samples.iterrows():
                if row["model_usage"]:
                    try:
                        usage_data = json.loads(row["model_usage"]) if isinstance(row["model_usage"], str) else row["model_usage"]
                        if MODEL in usage_data:
                            total_input_tokens += usage_data[MODEL].get("input_tokens", 0)
                            total_output_tokens += usage_data[MODEL].get("output_tokens", 0)
                    except (KeyError, TypeError, AttributeError, json.JSONDecodeError):
                        continue
            
            summary["model_usage"] = {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens
            }
        
        # Save results
        output_data = {
            "summary": summary,
            "results": results
        }
        
        # Save JSON
        json_file = "conseq_fin_stage4_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved results to {json_file}")
        
        # Save CSV
        csv_file = "conseq_fin_stage4_results.csv"
        analysis_df.to_csv(csv_file, index=False)
        logger.info(f"Saved DataFrame to {csv_file}")
        
        # Save summary
        summary_file = "conseq_fin_stage4_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {summary_file}")
        
        # Log key statistics
        logger.info("\n=== Classification Results ===")
        logger.info(f"Total tools processed: {summary['total_tools']}")
        logger.info(f"Valid classifications: {summary['valid_classifications']}")
        logger.info(f"Error rate: {summary['error_rate']:.1f}%")
        
        if 'top_level_distribution' in summary:
            logger.info("\nTop-level task distribution:")
            for level, count in sorted(summary['top_level_distribution'].items()):
                logger.info(f"  Level {level}: {count} tools")
        
        if 'automation_vs_augmentation' in summary:
            logger.info(f"\nAutomation vs Augmentation:")
            logger.info(f"  Automation: {summary['automation_vs_augmentation']['automation']} ({summary['automation_vs_augmentation']['automation_percentage']:.1f}%)")
            logger.info(f"  Augmentation: {summary['automation_vs_augmentation']['augmentation']}")
        
        if 'automation_levels' in summary:
            logger.info("\nAutomation level distribution:")
            for level, count in sorted(summary['automation_levels'].items()):
                logger.info(f"  Level {level}: {count} tools")
        
        logger.info("\n=== Processing Complete ===")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
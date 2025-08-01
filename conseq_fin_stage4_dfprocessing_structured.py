#!/usr/bin/env python3
"""
Stage 4 O*NET Classification - Structured Output DataFrame Processing

Processes the evaluation results from conseq_fin_stage4_inspect_structured.py
by extracting the structured final answers from natural language responses.

Usage:
    python conseq_fin_stage4_dfprocessing_structured.py
    python conseq_fin_stage4_dfprocessing_structured.py --logs-dir ./custom_logs
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
        logging.FileHandler('conseq_fin_stage4_dfprocessing_structured.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MODEL = "anthropic/claude-sonnet-4-20250514"

# Category mappings
CATEGORY_NAMES = {
    "1": "Information technology systems",
    "2": "Art, culture, and creative work",
    "3": "Business management and finance",
    "4": "Education and HR",
    "5": "Scientific research",
    "6": "Government and public safety",
    "7": "Industrial and agricultural processes",
    "8": "Energy management",
    "9": "Environmental systems",
    "10": "Healthcare services"
}

def extract_tool_info_from_user_message(content: str) -> Dict[str, str]:
    """Extract tool information from user message"""
    info = {
        'tool_name': '',
        'tool_description': '',
        'server_name': '',
        'server_description': '',
        'input_schema': ''
    }
    
    # Try to find structured information in the user message
    lines = content.split('\n')
    for line in lines:
        if 'Server name:' in line:
            info['server_name'] = line.split('Server name:', 1)[1].strip()
        elif 'Server description:' in line:
            info['server_description'] = line.split('Server description:', 1)[1].strip()
        elif 'Tool name:' in line:
            info['tool_name'] = line.split('Tool name:', 1)[1].strip()
        elif 'Tool description:' in line:
            info['tool_description'] = line.split('Tool description:', 1)[1].strip()
        elif 'Input schema:' in line:
            info['input_schema'] = line.split('Input schema:', 1)[1].strip()
    
    return info

def process_structured_results(samples_df: pd.DataFrame, messages_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Process structured classification results from DataFrames"""
    results = []
    
    # Get assistant messages
    assistant_messages = messages_df[messages_df['role'] == 'assistant']
    
    for idx, sample_row in samples_df.iterrows():
        sample_id = sample_row.get("sample_id", f"sample_{idx}")
        
        # Find messages for this sample
        sample_assistant = assistant_messages[assistant_messages['sample_id'] == sample_id]
        sample_user = messages_df[
            (messages_df['sample_id'] == sample_id) & 
            (messages_df['role'] == 'user')
        ]
        
        result = {
            "sample_id": sample_id,
            "tool_id": "",
            "input_data": {},
            "raw_output": "",
            "reasoning": "",
            "score": sample_row.get("score_structured_scorer", 0),
            "classifications": {
                "top_level_category": "",
                "top_level_number": "",
                "collaboration_pattern": "",
                "automation_level": -1,
                "replaced_tools": []
            },
            "errors": []
        }
        
        # Extract tool info from user message
        if not sample_user.empty:
            user_content = sample_user.iloc[0]['content']
            result["input_data"] = extract_tool_info_from_user_message(user_content)
            
            # Try to get tool_id from metadata
            if "metadata" in sample_row:
                metadata = sample_row["metadata"]
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                        result["tool_id"] = metadata.get("id", "")
                    except json.JSONDecodeError:
                        pass
        
        # Get assistant response and parse structured output
        if not sample_assistant.empty:
            full_content = sample_assistant.iloc[0]['content']
            result["raw_output"] = full_content
            
            # Check if we have a structured answer from the scorer
            if "answer_structured_scorer" in sample_row and sample_row["answer_structured_scorer"]:
                try:
                    # The scorer stores the result as JSON
                    answer = sample_row["answer_structured_scorer"]
                    if isinstance(answer, str):
                        parsed = json.loads(answer)
                        result["reasoning"] = parsed.get("reasoning", "")
                        
                        structured = parsed.get("structured_output", {})
                        
                        # Extract top level category
                        top_level = structured.get("Top Level Category", "")
                        if top_level:
                            result["classifications"]["top_level_number"] = top_level
                            result["classifications"]["top_level_category"] = CATEGORY_NAMES.get(top_level, "")
                        
                        # Extract collaboration pattern
                        result["classifications"]["collaboration_pattern"] = structured.get("Collaboration Pattern", "")
                        
                        # Extract automation level
                        auto_level = structured.get("Automation Level", "")
                        if auto_level and auto_level.isdigit():
                            result["classifications"]["automation_level"] = int(auto_level)
                        
                        # Extract replaced tools
                        replaced = structured.get("Replaced Tools", "")
                        if replaced and replaced.lower() != "none":
                            # Split by comma and clean up
                            tools = [t.strip() for t in replaced.split(',') if t.strip()]
                            result["classifications"]["replaced_tools"] = tools
                            
                except (json.JSONDecodeError, TypeError, KeyError) as e:
                    result["errors"].append(f"Failed to parse structured answer: {str(e)}")
            else:
                # Fallback: try to extract from raw content
                if "FINAL ANSWER:" in full_content:
                    try:
                        final_start = full_content.find("FINAL ANSWER:")
                        final_section = full_content[final_start:]
                        lines = final_section.split('\n')[1:]  # Skip FINAL ANSWER line
                        
                        for line in lines:
                            if ':' in line:
                                key, value = line.split(':', 1)
                                key = key.strip()
                                value = value.strip()
                                
                                if key == "Top Level Category":
                                    result["classifications"]["top_level_number"] = value
                                    result["classifications"]["top_level_category"] = CATEGORY_NAMES.get(value, "")
                                elif key == "Collaboration Pattern":
                                    result["classifications"]["collaboration_pattern"] = value
                                elif key == "Automation Level" and value.isdigit():
                                    result["classifications"]["automation_level"] = int(value)
                                elif key == "Replaced Tools" and value.lower() != "none":
                                    result["classifications"]["replaced_tools"] = [t.strip() for t in value.split(',') if t.strip()]
                                    
                        result["reasoning"] = full_content[:final_start].strip()
                        
                    except Exception as e:
                        result["errors"].append(f"Failed to extract from FINAL ANSWER: {str(e)}")
                else:
                    result["errors"].append("No FINAL ANSWER section found")
        
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
        
        # Add classifications
        classifications = result.get('classifications', {})
        row['top_level_category'] = classifications.get('top_level_category', '')
        row['top_level_number'] = classifications.get('top_level_number', '')
        row['collaboration_pattern'] = classifications.get('collaboration_pattern', '')
        row['automation_level'] = classifications.get('automation_level', -1)
        
        replaced_tools = classifications.get('replaced_tools', [])
        row['replaced_tools_count'] = len(replaced_tools)
        row['replaced_tools'] = ';'.join(replaced_tools) if replaced_tools else ''
        
        rows.append(row)
    
    return pd.DataFrame(rows)

def generate_summary_statistics(df: pd.DataFrame, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary statistics from processed results"""
    # Filter for valid results (score > 0)
    valid_results = df[df['score'] > 0]
    
    summary = {
        'processing_timestamp': datetime.now().isoformat(),
        'model': MODEL,
        'total_tools': len(results),
        'valid_classifications': len(valid_results),
        'error_rate': (len(df) - len(valid_results)) / len(df) * 100 if len(df) > 0 else 0
    }
    
    # Task mapping statistics
    if len(valid_results) > 0:
        # Top level distribution
        top_level_valid = valid_results[valid_results['top_level_number'] != '']
        if len(top_level_valid) > 0:
            top_level_dist = top_level_valid['top_level_number'].value_counts().to_dict()
            summary['top_level_distribution'] = {int(k): v for k, v in top_level_dist.items() if k.isdigit()}
            
            # Most common categories
            category_counts = top_level_valid['top_level_category'].value_counts()
            summary['top_categories'] = category_counts.head(5).to_dict()
        
        # Collaboration patterns
        collab_valid = valid_results[valid_results['collaboration_pattern'] != '']
        if len(collab_valid) > 0:
            collab_dist = collab_valid['collaboration_pattern'].value_counts().to_dict()
            summary['collaboration_patterns'] = collab_dist
            
            # Calculate automation vs augmentation
            automation_patterns = ['Directive', 'Feedback Loop']
            augmentation_patterns = ['Task Iteration', 'Learning', 'Validation']
            
            automation_count = collab_valid[collab_valid['collaboration_pattern'].isin(automation_patterns)].shape[0]
            augmentation_count = collab_valid[collab_valid['collaboration_pattern'].isin(augmentation_patterns)].shape[0]
            
            if (automation_count + augmentation_count) > 0:
                summary['automation_vs_augmentation'] = {
                    'automation': automation_count,
                    'augmentation': augmentation_count,
                    'automation_percentage': automation_count / (automation_count + augmentation_count) * 100
                }
        
        # Automation levels
        auto_valid = valid_results[valid_results['automation_level'] >= 0]
        if len(auto_valid) > 0:
            level_dist = auto_valid['automation_level'].value_counts().sort_index().to_dict()
            summary['automation_levels'] = {int(k): v for k, v in level_dist.items()}
            summary['avg_automation_level'] = auto_valid['automation_level'].mean()
            
            # High risk tools (level 4-5)
            high_risk = auto_valid[auto_valid['automation_level'] >= 4]
            summary['high_risk_tools'] = len(high_risk)
            summary['high_risk_percentage'] = len(high_risk) / len(auto_valid) * 100
        
        # Tool replacement
        replace_valid = valid_results[valid_results['replaced_tools_count'] > 0]
        summary['tools_replacing_traditional'] = len(replace_valid)
        summary['avg_tools_replaced'] = valid_results['replaced_tools_count'].mean()
        
        # Most commonly replaced tools
        if len(replace_valid) > 0:
            all_replaced = []
            for tools_str in replace_valid['replaced_tools'].dropna():
                if tools_str:
                    all_replaced.extend(tools_str.split(';'))
            
            if all_replaced:
                from collections import Counter
                tool_counts = Counter(all_replaced)
                summary['most_replaced_tools'] = dict(tool_counts.most_common(10))
    
    return summary

def print_example_outputs(results: List[Dict[str, Any]], limit: int = 3):
    """Print example classification results"""
    logger.info("\n=== Example Classification Results ===")
    
    for i, result in enumerate(results[:limit]):
        if result['score'] > 0:
            logger.info(f"\n--- Example {i+1} ---")
            logger.info(f"Tool: {result['input_data'].get('tool_name', 'Unknown')}")
            logger.info(f"Server: {result['input_data'].get('server_name', 'Unknown')}")
            
            classifications = result['classifications']
            logger.info(f"\nClassifications:")
            logger.info(f"  Category: {classifications['top_level_category']} (Level {classifications['top_level_number']})")
            logger.info(f"  Collaboration: {classifications['collaboration_pattern']}")
            logger.info(f"  Automation Level: {classifications['automation_level']}")
            logger.info(f"  Replaces: {', '.join(classifications['replaced_tools']) if classifications['replaced_tools'] else 'None'}")
            
            if result.get('reasoning'):
                # Show first 200 chars of reasoning
                reasoning = result['reasoning'][:200] + "..." if len(result['reasoning']) > 200 else result['reasoning']
                logger.info(f"\nReasoning Preview: {reasoning}")

def main():
    parser = argparse.ArgumentParser(description='Process O*NET classification structured results')
    parser.add_argument('--logs-dir', type=str, default='conseq_fin_stage4_logs',
                       help='Directory containing .eval files')
    parser.add_argument('--eval-file', type=str,
                       help='Specific .eval file to process')
    
    args = parser.parse_args()
    
    logger.info("Starting Stage 4 Structured DataFrame Processing")
    
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
        results = process_structured_results(samples, messages)
        
        # Print examples
        print_example_outputs(results, limit=3)
        
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
        json_file = "conseq_fin_stage4_structured_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved results to {json_file}")
        
        # Save CSV
        csv_file = "conseq_fin_stage4_structured_results.csv"
        analysis_df.to_csv(csv_file, index=False)
        logger.info(f"Saved DataFrame to {csv_file}")
        
        # Save summary
        summary_file = "conseq_fin_stage4_structured_summary.json"
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
                category = CATEGORY_NAMES.get(str(level), "Unknown")
                logger.info(f"  Level {level} ({category}): {count} tools")
        
        if 'automation_vs_augmentation' in summary:
            logger.info(f"\nAutomation vs Augmentation:")
            logger.info(f"  Automation: {summary['automation_vs_augmentation']['automation']} ({summary['automation_vs_augmentation']['automation_percentage']:.1f}%)")
            logger.info(f"  Augmentation: {summary['automation_vs_augmentation']['augmentation']}")
        
        if 'automation_levels' in summary:
            logger.info("\nAutomation level distribution:")
            for level, count in sorted(summary['automation_levels'].items()):
                logger.info(f"  Level {level}: {count} tools")
            logger.info(f"  Average level: {summary.get('avg_automation_level', 0):.2f}")
            logger.info(f"  High risk tools (4-5): {summary.get('high_risk_tools', 0)} ({summary.get('high_risk_percentage', 0):.1f}%)")
        
        if 'most_replaced_tools' in summary:
            logger.info("\nMost commonly replaced tools:")
            for tool, count in list(summary['most_replaced_tools'].items())[:5]:
                logger.info(f"  {tool}: {count} times")
        
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
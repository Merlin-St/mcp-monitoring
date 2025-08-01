#!/usr/bin/env python3
"""
Stage 4 O*NET Classification - Natural Language DataFrame Processing

Processes the evaluation results from conseq_fin_stage4_inspect_simple.py
by reading the .eval files and extracting information from natural language responses.

This version handles natural language responses instead of expecting JSON.

Usage:
    python conseq_fin_stage4_dfprocessing_natural.py
    python conseq_fin_stage4_dfprocessing_natural.py --logs-dir ./custom_logs
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
        logging.FileHandler('conseq_fin_stage4_dfprocessing_natural.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MODEL = "anthropic/claude-sonnet-4-20250514"

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

def parse_onet_task_mapping(text: str) -> Dict[str, Any]:
    """Parse O*NET task mapping from natural language"""
    mapping = {
        'top_level_category': '',
        'top_level_number': '',
        'specific_task': '',
        'occupation': '',
        'confidence': 'medium'
    }
    
    # Extract top-level category - look for category mentions
    categories = {
        1: ['information technology', 'IT systems', 'software', 'computing'],
        2: ['art', 'culture', 'creative', 'design'],
        3: ['business', 'management', 'finance', 'accounting'],
        4: ['education', 'HR', 'human resources', 'training'],
        5: ['scientific', 'research', 'laboratory', 'analysis'],
        6: ['government', 'public safety', 'security', 'law enforcement'],
        7: ['industrial', 'agricultural', 'manufacturing', 'production'],
        8: ['energy', 'power', 'utilities'],
        9: ['environmental', 'sustainability', 'climate'],
        10: ['healthcare', 'medical', 'health services']
    }
    
    text_lower = text.lower()
    
    # Find which category is mentioned
    for num, keywords in categories.items():
        for keyword in keywords:
            if keyword in text_lower:
                mapping['top_level_number'] = str(num)
                # Set category name based on number
                category_names = {
                    1: 'Information technology systems',
                    2: 'Art, culture, and creative work',
                    3: 'Business management and finance',
                    4: 'Education and HR',
                    5: 'Scientific research',
                    6: 'Government and public safety',
                    7: 'Industrial and agricultural processes',
                    8: 'Energy management',
                    9: 'Environmental systems',
                    10: 'Healthcare services'
                }
                mapping['top_level_category'] = category_names.get(num, '')
                break
        if mapping['top_level_number']:
            break
    
    # Extract specific tasks mentioned
    task_patterns = [
        r'tasks? (?:include|such as|like|involving)\s*[:]\s*([^.]+)',
        r'supports?\s+([^.]+?)\s+tasks?',
        r'enables?\s+([^.]+?)\s+(?:tasks?|activities)',
        r'(?:used for|useful for)\s+([^.]+)'
    ]
    
    for pattern in task_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            mapping['specific_task'] = match.group(1).strip()
            break
    
    # Extract occupation mentions
    occupation_patterns = [
        r'(?:for|by)\s+(\w+\s*(?:analysts?|engineers?|developers?|managers?|specialists?))',
        r'(\w+\s*(?:analysts?|engineers?|developers?|managers?|specialists?))\s+(?:would|could|can)\s+use',
        r'occupations?\s*[:]\s*([^.]+)'
    ]
    
    for pattern in occupation_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            mapping['occupation'] = match.group(1).strip()
            break
    
    # Determine confidence based on how specific the analysis is
    if mapping['specific_task'] and mapping['occupation']:
        mapping['confidence'] = 'high'
    elif mapping['specific_task'] or mapping['occupation']:
        mapping['confidence'] = 'medium'
    else:
        mapping['confidence'] = 'low'
    
    return mapping

def parse_collaboration_pattern(text: str) -> Dict[str, Any]:
    """Parse collaboration pattern from natural language"""
    pattern = {
        'pattern': '',
        'confidence': 'medium'
    }
    
    text_lower = text.lower()
    
    # Define pattern keywords
    patterns = {
        'Directive': ['directive', 'complete delegation', 'autonomous', 'independently'],
        'Feedback Loop': ['feedback', 'iterative', 'back-and-forth', 'interactive'],
        'Task Iteration': ['iterative refinement', 'refine', 'improve', 'iterate'],
        'Learning': ['learning', 'understanding', 'educational', 'informational'],
        'Validation': ['validation', 'checking', 'verify', 'confirm', 'audit']
    }
    
    # Find which pattern is mentioned
    for pattern_name, keywords in patterns.items():
        for keyword in keywords:
            if keyword in text_lower:
                pattern['pattern'] = pattern_name
                pattern['confidence'] = 'high'
                return pattern
    
    # If no direct match, try to infer from description
    if 'human' in text_lower and 'minimal' in text_lower:
        pattern['pattern'] = 'Directive'
    elif 'collaborative' in text_lower or 'together' in text_lower:
        pattern['pattern'] = 'Feedback Loop'
    elif 'review' in text_lower or 'check' in text_lower:
        pattern['pattern'] = 'Validation'
    
    return pattern

def parse_automation_level(text: str) -> Dict[str, Any]:
    """Parse automation level from natural language"""
    auto = {
        'level': -1,
        'level_description': ''
    }
    
    # Look for level mentions
    level_match = re.search(r'level\s*(\d)', text, re.IGNORECASE)
    if level_match:
        auto['level'] = int(level_match.group(1))
    
    # Also try to find descriptive mentions
    level_descriptions = {
        0: ['not functional', 'broken', 'doesn\'t work'],
        1: ['monitoring', 'read-only', 'observation', 'view only'],
        2: ['analysis', 'process', 'recommend', 'analyze'],
        3: ['meta', 'coordinates', 'orchestrates', 'manages other'],
        4: ['restricted execution', 'specific environment', 'limited execution'],
        5: ['unrestricted', 'arbitrary', 'full execution', 'any action']
    }
    
    text_lower = text.lower()
    for level, keywords in level_descriptions.items():
        for keyword in keywords:
            if keyword in text_lower:
                auto['level'] = level
                auto['level_description'] = keyword
                break
        if auto['level'] != -1:
            break
    
    return auto

def parse_tool_replacement(text: str) -> Dict[str, Any]:
    """Parse tool replacement information from natural language"""
    replacement = {
        'replaced_tools': [],
        'confidence': 'medium'
    }
    
    # Look for tool mentions after key phrases
    patterns = [
        r'replace[s]?\s+(?:tools? like\s+)?([^.]+?)(?:\.|,|;|$)',
        r'instead of\s+([^.]+?)(?:\.|,|;|$)',
        r'replaces?\s+traditional\s+([^.]+?)(?:\.|,|;|$)',
        r'alternative to\s+([^.]+?)(?:\.|,|;|$)'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Split by common separators
            tools = re.split(r',|\sand\s|\sor\s', match)
            for tool in tools:
                tool = tool.strip()
                if tool and len(tool) < 100:  # Reasonable length for a tool name
                    replacement['replaced_tools'].append(tool)
    
    # Remove duplicates
    replacement['replaced_tools'] = list(set(replacement['replaced_tools']))
    
    # Set confidence based on findings
    if replacement['replaced_tools']:
        replacement['confidence'] = 'high'
    else:
        replacement['confidence'] = 'low'
    
    return replacement

def process_natural_language_results(samples_df: pd.DataFrame, messages_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Process natural language classification results from DataFrames"""
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
            "score": sample_row.get("score_simple_scorer", 0),
            "classifications": {
                "task_mapping": None,
                "collaboration_pattern": None,
                "automation_level": None,
                "tool_replacement": None
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
        
        # Get assistant response
        if not sample_assistant.empty:
            assistant_content = sample_assistant.iloc[0]['content']
            result["raw_output"] = assistant_content
            
            # Parse natural language response
            try:
                # Split response into sections based on numbered headers
                sections = re.split(r'\d+\.\s*\*\*[^*]+\*\*', assistant_content)
                
                # Find each section by header
                task_section = ""
                collab_section = ""
                auto_section = ""
                replace_section = ""
                
                # Look for sections more flexibly
                if '**O*NET Task Mapping**' in assistant_content:
                    start = assistant_content.find('**O*NET Task Mapping**')
                    end = assistant_content.find('2.', start)
                    if end == -1:
                        end = assistant_content.find('\n\n', start + 100)
                    task_section = assistant_content[start:end] if end != -1 else assistant_content[start:]
                
                if '**Collaboration Pattern**' in assistant_content:
                    start = assistant_content.find('**Collaboration Pattern**')
                    end = assistant_content.find('3.', start)
                    if end == -1:
                        end = assistant_content.find('\n\n', start + 100)
                    collab_section = assistant_content[start:end] if end != -1 else assistant_content[start:]
                
                if '**Automation Level**' in assistant_content:
                    start = assistant_content.find('**Automation Level**')
                    end = assistant_content.find('4.', start)
                    if end == -1:
                        end = assistant_content.find('\n\n', start + 100)
                    auto_section = assistant_content[start:end] if end != -1 else assistant_content[start:]
                
                if '**Tool Replacement**' in assistant_content:
                    start = assistant_content.find('**Tool Replacement**')
                    replace_section = assistant_content[start:]
                
                # Parse each section
                if task_section:
                    result["classifications"]["task_mapping"] = parse_onet_task_mapping(task_section)
                
                if collab_section:
                    result["classifications"]["collaboration_pattern"] = parse_collaboration_pattern(collab_section)
                
                if auto_section:
                    result["classifications"]["automation_level"] = parse_automation_level(auto_section)
                
                if replace_section:
                    result["classifications"]["tool_replacement"] = parse_tool_replacement(replace_section)
                    
            except Exception as e:
                result["errors"].append(f"Failed to parse natural language: {str(e)}")
        
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

def print_example_outputs(messages_df: pd.DataFrame, limit: int = 3):
    """Print example natural language outputs"""
    logger.info("\n=== Example Natural Language Outputs ===")
    
    assistant_messages = messages_df[messages_df['role'] == 'assistant']
    
    for i, (idx, msg) in enumerate(assistant_messages.head(limit).iterrows()):
        logger.info(f"\n--- Example {i+1} (Sample ID: {msg['sample_id']}) ---")
        
        # Get corresponding user message
        user_msg = messages_df[
            (messages_df['sample_id'] == msg['sample_id']) & 
            (messages_df['role'] == 'user')
        ]
        
        if not user_msg.empty:
            user_content = user_msg.iloc[0]['content']
            # Extract just tool info
            tool_info = extract_tool_info_from_user_message(user_content)
            logger.info(f"Tool: {tool_info['tool_name']}")
            logger.info(f"Server: {tool_info['server_name']}")
        
        # Show first 500 chars of response
        response = msg['content'][:500] + "..." if len(msg['content']) > 500 else msg['content']
        logger.info(f"\nAssistant Response Preview:\n{response}")

def generate_summary_statistics(df: pd.DataFrame, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary statistics from processed results"""
    # Filter for valid results (those with successful parsing)
    # First check if columns exist
    has_task = 'top_level_number' in df.columns
    has_collab = 'collaboration_pattern' in df.columns
    has_auto = 'automation_level' in df.columns
    has_replace = 'replaced_tools_count' in df.columns
    
    conditions = []
    if has_task:
        conditions.append(df['top_level_number'] != '')
    if has_collab:
        conditions.append(df['collaboration_pattern'] != '')
    if has_auto:
        conditions.append(df['automation_level'] >= 0)
    if has_replace:
        conditions.append(df['replaced_tools_count'] > 0)
    
    if conditions:
        from functools import reduce
        import operator
        valid_results = df[reduce(operator.or_, conditions)]
    else:
        valid_results = df[df['score'] > 0]  # Fallback to score-based filtering
    
    summary = {
        'processing_timestamp': datetime.now().isoformat(),
        'model': MODEL,
        'total_tools': len(results),
        'successfully_parsed': len(valid_results),
        'parse_success_rate': len(valid_results) / len(df) * 100 if len(df) > 0 else 0
    }
    
    # Task mapping statistics
    if 'top_level_number' in valid_results.columns:
        top_level_with_value = valid_results[valid_results['top_level_number'] != '']
        if len(top_level_with_value) > 0:
            top_level_dist = top_level_with_value['top_level_number'].value_counts().to_dict()
            summary['top_level_distribution'] = {int(k): v for k, v in top_level_dist.items() if k.isdigit()}
            summary['top_occupations'] = valid_results[valid_results['occupation'] != '']['occupation'].value_counts().head(10).to_dict()
    
    # Collaboration patterns
    if 'collaboration_pattern' in valid_results.columns:
        collab_with_value = valid_results[valid_results['collaboration_pattern'] != '']
        if len(collab_with_value) > 0:
            collab_dist = collab_with_value['collaboration_pattern'].value_counts().to_dict()
            summary['collaboration_patterns'] = collab_dist
            
            # Calculate automation vs augmentation
            automation_patterns = ['Directive', 'Feedback Loop']
            augmentation_patterns = ['Task Iteration', 'Learning', 'Validation']
            
            automation_count = collab_with_value[collab_with_value['collaboration_pattern'].isin(automation_patterns)].shape[0]
            augmentation_count = collab_with_value[collab_with_value['collaboration_pattern'].isin(augmentation_patterns)].shape[0]
            
            summary['automation_vs_augmentation'] = {
                'automation': automation_count,
                'augmentation': augmentation_count,
                'automation_percentage': automation_count / (automation_count + augmentation_count) * 100 if (automation_count + augmentation_count) > 0 else 0
            }
    
    # Automation levels
    if 'automation_level' in valid_results.columns:
        auto_with_value = valid_results[valid_results['automation_level'] >= 0]
        if len(auto_with_value) > 0:
            level_dist = auto_with_value['automation_level'].value_counts().sort_index().to_dict()
            summary['automation_levels'] = {int(k): v for k, v in level_dist.items()}
            summary['avg_automation_level'] = auto_with_value['automation_level'].mean()
    
    # Tool replacement
    if 'replaced_tools_count' in valid_results.columns:
        tools_with_replacement = valid_results[valid_results['replaced_tools_count'] > 0]
        summary['tools_replacing_traditional'] = len(tools_with_replacement)
        summary['avg_tools_replaced'] = valid_results['replaced_tools_count'].mean()
        
        # Most commonly replaced tools
        all_replaced = []
        for tools_str in tools_with_replacement['replaced_tools'].dropna():
            if tools_str:
                all_replaced.extend(tools_str.split(';'))
        
        if all_replaced:
            from collections import Counter
            tool_counts = Counter(all_replaced)
            summary['most_replaced_tools'] = dict(tool_counts.most_common(10))
    
    return summary

def main():
    parser = argparse.ArgumentParser(description='Process O*NET classification natural language results')
    parser.add_argument('--logs-dir', type=str, default='conseq_fin_stage4_logs',
                       help='Directory containing .eval files')
    parser.add_argument('--eval-file', type=str,
                       help='Specific .eval file to process')
    
    args = parser.parse_args()
    
    logger.info("Starting Stage 4 Natural Language DataFrame Processing")
    
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
        
        # Print example outputs
        print_example_outputs(messages, limit=3)
        
        # Process results
        results = process_natural_language_results(samples, messages)
        
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
        json_file = "conseq_fin_stage4_natural_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved results to {json_file}")
        
        # Save CSV
        csv_file = "conseq_fin_stage4_natural_results.csv"
        analysis_df.to_csv(csv_file, index=False)
        logger.info(f"Saved DataFrame to {csv_file}")
        
        # Save summary
        summary_file = "conseq_fin_stage4_natural_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {summary_file}")
        
        # Log key statistics
        logger.info("\n=== Natural Language Classification Results ===")
        logger.info(f"Total tools processed: {summary['total_tools']}")
        logger.info(f"Successfully parsed: {summary['successfully_parsed']}")
        logger.info(f"Parse success rate: {summary['parse_success_rate']:.1f}%")
        
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
#!/usr/bin/env python3
"""
Stage 4 O*NET Classification - Multi-Eval DataFrame Processing

Processes evaluation results from all 4 inspect scripts:
1. conseq_fin_stage4_inspect_1_task.py - Task mapping
2. conseq_fin_stage4_inspect_2_collab.py - Collaboration patterns
3. conseq_fin_stage4_inspect_3_auto.py - Automation levels
4. conseq_fin_stage4_inspect_4_tools.py - Tool replacement

Usage:
    python conseq_fin_stage4_dfprocessing_multi.py
    python conseq_fin_stage4_dfprocessing_multi.py --logs-dir ./custom_logs
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
        logging.FileHandler('conseq_fin_stage4_dfprocessing_multi.log'),
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
    
    # First try to parse as JSON (the actual format)
    try:
        data = json.loads(content)
        info['tool_name'] = data.get('tool_name', '')
        info['tool_description'] = data.get('tool_description', '')
        info['server_name'] = data.get('server_name', '')
        info['server_description'] = data.get('server_description', '')
        info['input_schema'] = data.get('tool_input_schema', '')
        return info
    except json.JSONDecodeError:
        pass
    
    # Fallback: Try to find structured information in the user message
    lines = content.split('\n')
    
    # Look for the pattern "Server name & description: name: description"
    for i, line in enumerate(lines):
        if 'Server name & description:' in line:
            # Extract server name and description
            match = re.search(r'Server name & description:\s*([^:]+):\s*(.+)', line)
            if match:
                info['server_name'] = match.group(1).strip()
                info['server_description'] = match.group(2).strip()
        elif 'Tool name, description and input schema' in line:
            # The actual tool info is typically on the next lines
            # Look for the next non-empty lines that contain tool info
            j = i + 1
            tool_info_lines = []
            while j < len(lines) and len(tool_info_lines) < 3:
                if lines[j].strip() and not lines[j].startswith(('Consider', 'Your', 'What', 'Based')):
                    tool_info_lines.append(lines[j].strip())
                j += 1
            
            # First line is usually tool name, second is description, third is input schema
            if len(tool_info_lines) > 0:
                info['tool_name'] = tool_info_lines[0]
            if len(tool_info_lines) > 1:
                info['tool_description'] = tool_info_lines[1]
            if len(tool_info_lines) > 2:
                info['input_schema'] = tool_info_lines[2]
            break
    
    return info

def extract_tool_info_from_eval(eval_file: Path) -> Dict[str, Dict[str, str]]:
    """Extract tool info from any evaluation file by parsing user messages"""
    tool_infos = {}
    
    import tempfile
    temp_dir = tempfile.mkdtemp()
    temp_file = Path(temp_dir) / eval_file.name
    
    import shutil
    shutil.copy2(eval_file, temp_file)
    
    try:
        from inspect_ai.analysis.beta import samples_df, messages_df
        
        samples = samples_df(temp_dir)
        messages = messages_df(temp_dir)
        
        # Extract tool info from user messages for all samples
        for idx, sample_row in samples.iterrows():
            sample_id = sample_row.get("sample_id", f"sample_{idx}")
            
            # Get user message for tool info
            user_msg = messages[(messages['sample_id'] == sample_id) & (messages['role'] == 'user')]
            if not user_msg.empty:
                tool_info = extract_tool_info_from_user_message(user_msg.iloc[0]['content'])
                tool_infos[sample_id] = tool_info
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return tool_infos

def process_task_mapping_eval(eval_file: Path) -> Dict[str, Dict[str, Any]]:
    """Process task mapping evaluation file"""
    results = {}
    
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
            
            # Get assistant response
            assistant_msg = messages[(messages['sample_id'] == sample_id) & (messages['role'] == 'assistant')]
            task_mapping = ""
            if not assistant_msg.empty:
                task_mapping = assistant_msg.iloc[0]['content'].strip()
            
            results[sample_id] = {
                'task_mapping': task_mapping,
                'score': sample_row.get('score_task_mapping_scorer', 0)
            }
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return results

def process_collab_pattern_eval(eval_file: Path) -> Dict[str, str]:
    """Process collaboration pattern evaluation file"""
    results = {}
    
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
            
            # Get pattern from answer_collab_pattern_scorer if available
            if 'answer_collab_pattern_scorer' in sample_row and sample_row['answer_collab_pattern_scorer']:
                results[sample_id] = sample_row['answer_collab_pattern_scorer']
            else:
                # Fallback to assistant message
                assistant_msg = messages[(messages['sample_id'] == sample_id) & (messages['role'] == 'assistant')]
                if not assistant_msg.empty:
                    results[sample_id] = assistant_msg.iloc[0]['content'].strip()
                else:
                    results[sample_id] = 'None'
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return results

def process_automation_level_eval(eval_file: Path) -> Dict[str, int]:
    """Process automation level evaluation file"""
    results = {}
    
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
            
            # Get level from answer_automation_level_scorer if available
            if 'answer_automation_level_scorer' in sample_row and sample_row['answer_automation_level_scorer'] is not None:
                answer = sample_row['answer_automation_level_scorer']
                if isinstance(answer, (int, float)):
                    results[sample_id] = int(answer)
                elif isinstance(answer, str) and answer.strip().isdigit():
                    results[sample_id] = int(answer.strip())
                else:
                    results[sample_id] = -1
            else:
                # Fallback to assistant message
                assistant_msg = messages[(messages['sample_id'] == sample_id) & (messages['role'] == 'assistant')]
                if not assistant_msg.empty:
                    response = assistant_msg.iloc[0]['content'].strip()
                    if response.isdigit() and 0 <= int(response) <= 5:
                        results[sample_id] = int(response)
                    else:
                        results[sample_id] = -1
                else:
                    results[sample_id] = -1
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return results

def process_tool_replacement_eval(eval_file: Path) -> Dict[str, List[str]]:
    """Process tool replacement evaluation file"""
    results = {}
    
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
            
            # Get tools from answer_tool_replacement_scorer if available
            if 'answer_tool_replacement_scorer' in sample_row and sample_row['answer_tool_replacement_scorer'] is not None:
                answer = sample_row['answer_tool_replacement_scorer']
                if isinstance(answer, str):
                    if answer.lower() == 'none':
                        results[sample_id] = []
                    else:
                        # Parse comma-separated tools
                        tools = [t.strip() for t in answer.split(',') if t.strip()]
                        results[sample_id] = tools
                elif isinstance(answer, list):
                    results[sample_id] = answer
                else:
                    results[sample_id] = []
            else:
                # Fallback to assistant message
                assistant_msg = messages[(messages['sample_id'] == sample_id) & (messages['role'] == 'assistant')]
                if not assistant_msg.empty:
                    response = assistant_msg.iloc[0]['content'].strip().lower()
                    if response == 'none':
                        results[sample_id] = []
                    else:
                        tools = [t.strip() for t in response.split(',') if t.strip()]
                        results[sample_id] = tools
                else:
                    results[sample_id] = []
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return results

def find_latest_eval_files(logs_dir: Path) -> Dict[str, Path]:
    """Find the latest eval file for each task type"""
    eval_files = {}
    
    # Patterns for each task type
    patterns = {
        'task': '*task-mapping-task*.eval',
        'collab': '*collaboration-pattern-task*.eval',
        'auto': '*automation-level-task*.eval',
        'tools': '*tool-replacement-task*.eval'
    }
    
    for task_type, pattern in patterns.items():
        files = list(logs_dir.glob(pattern))
        if files:
            # Get the most recent file
            latest = max(files, key=lambda x: x.stat().st_mtime)
            eval_files[task_type] = latest
            logger.info(f"Found {task_type} eval: {latest.name}")
        else:
            logger.warning(f"No eval file found for {task_type} with pattern {pattern}")
    
    return eval_files

def combine_results(task_results: Dict, collab_results: Dict, 
                   auto_results: Dict, tool_results: Dict,
                   tool_infos: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
    """Combine results from all 4 evaluations with tool info"""
    combined = []
    
    # Get all unique sample IDs
    all_sample_ids = set()
    all_sample_ids.update(task_results.keys())
    all_sample_ids.update(collab_results.keys())
    all_sample_ids.update(auto_results.keys())
    all_sample_ids.update(tool_results.keys())
    
    for sample_id in all_sample_ids:
        # Get tool info for this sample
        tool_info = tool_infos.get(sample_id, {})
        
        result = {
            'sample_id': sample_id,
            'tool_name': tool_info.get('tool_name', ''),
            'tool_description': tool_info.get('tool_description', ''),
            'server_name': tool_info.get('server_name', ''),
            'server_description': tool_info.get('server_description', ''),
            'task_mapping': task_results.get(sample_id, {}).get('task_mapping', ''),
            'collaboration_pattern': collab_results.get(sample_id, ''),
            'automation_level': auto_results.get(sample_id, -1),
            'replaced_tools': tool_results.get(sample_id, [])
        }
        
        combined.append(result)
    
    return combined

def create_analysis_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a DataFrame for analysis"""
    rows = []
    
    for result in results:
        row = {
            'sample_id': result['sample_id'],
            'tool_name': result['tool_name'],
            'tool_description': result['tool_description'],
            'server_name': result['server_name'],
            'server_description': result['server_description'],
            'task_mapping': result['task_mapping'],
            'collaboration_pattern': result['collaboration_pattern'],
            'automation_level': result['automation_level'],
            'replaced_tools_count': len(result['replaced_tools']),
            'replaced_tools': ';'.join(result['replaced_tools']) if result['replaced_tools'] else ''
        }
        
        rows.append(row)
    
    return pd.DataFrame(rows)

def generate_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate summary statistics from combined results"""
    summary = {
        'processing_timestamp': datetime.now().isoformat(),
        'model': MODEL,
        'total_tools': len(df)
    }
    
    # Task mapping statistics
    task_valid = df[df['task_mapping'] != '']
    summary['tools_with_task_mapping'] = len(task_valid)
    
    if len(task_valid) > 0:
        # Get distribution of task mappings
        task_dist = task_valid['task_mapping'].value_counts().to_dict()
        summary['task_mapping_distribution'] = dict(list(task_dist.items())[:20])  # Top 20 tasks
        summary['unique_tasks_mapped'] = len(task_dist)
    
    # Collaboration patterns
    collab_valid = df[df['collaboration_pattern'] != '']
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
    auto_valid = df[df['automation_level'] >= 0]
    if len(auto_valid) > 0:
        level_dist = auto_valid['automation_level'].value_counts().sort_index().to_dict()
        summary['automation_levels'] = {int(k): v for k, v in level_dist.items()}
        summary['avg_automation_level'] = auto_valid['automation_level'].mean()
        
        # High risk tools (level 4-5)
        high_risk = auto_valid[auto_valid['automation_level'] >= 4]
        summary['high_risk_tools'] = len(high_risk)
        summary['high_risk_percentage'] = len(high_risk) / len(auto_valid) * 100
    
    # Tool replacement
    replace_valid = df[df['replaced_tools_count'] > 0]
    summary['tools_replacing_traditional'] = len(replace_valid)
    summary['avg_tools_replaced'] = df['replaced_tools_count'].mean()
    
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

def main():
    parser = argparse.ArgumentParser(description='Process O*NET classification results from 4 evaluations')
    parser.add_argument('--logs-dir', type=str, default='logs',
                       help='Directory containing .eval files')
    
    args = parser.parse_args()
    
    logger.info("Starting Stage 4 Multi-Eval DataFrame Processing")
    
    # Find eval files
    log_dir = Path(args.logs_dir)
    if not log_dir.exists():
        logger.error(f"Log directory {args.logs_dir} not found")
        return
    
    eval_files = find_latest_eval_files(log_dir)
    
    # Check that we have all required files
    required = ['task', 'collab', 'auto', 'tools']
    missing = [t for t in required if t not in eval_files]
    if missing:
        logger.error(f"Missing evaluation files for: {missing}")
        logger.error("Please run all 4 inspect scripts first")
        return
    
    # First extract tool info from any evaluation (they all have the same user messages)
    logger.info("Extracting tool information from user messages...")
    tool_infos = extract_tool_info_from_eval(eval_files['task'])  # Use task eval as it exists
    logger.info(f"Extracted tool info for {len(tool_infos)} samples")
    
    # Process each evaluation
    logger.info("Processing task mapping evaluation...")
    task_results = process_task_mapping_eval(eval_files['task'])
    
    logger.info("Processing collaboration pattern evaluation...")
    collab_results = process_collab_pattern_eval(eval_files['collab'])
    
    logger.info("Processing automation level evaluation...")
    auto_results = process_automation_level_eval(eval_files['auto'])
    
    logger.info("Processing tool replacement evaluation...")
    tool_results = process_tool_replacement_eval(eval_files['tools'])
    
    # Combine results
    logger.info("Combining results from all evaluations...")
    combined_results = combine_results(task_results, collab_results, auto_results, tool_results, tool_infos)
    
    # Create analysis DataFrame
    analysis_df = create_analysis_dataframe(combined_results)
    
    # Generate summary
    summary = generate_summary_statistics(analysis_df)
    
    # Save results
    output_data = {
        "summary": summary,
        "results": combined_results
    }
    
    # Save JSON
    json_file = "conseq_fin_stage4_multi_results.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved results to {json_file}")
    
    # Save CSV
    csv_file = "conseq_fin_stage4_multi_results.csv"
    analysis_df.to_csv(csv_file, index=False)
    logger.info(f"Saved DataFrame to {csv_file}")
    
    # Save summary
    summary_file = "conseq_fin_stage4_multi_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_file}")
    
    # Log key statistics
    logger.info("\n=== Multi-Eval Classification Results ===")
    logger.info(f"Total tools processed: {summary['total_tools']}")
    logger.info(f"Tools with task mapping: {summary.get('tools_with_task_mapping', 0)}")
    
    if 'collaboration_patterns' in summary:
        logger.info("\nCollaboration patterns:")
        for pattern, count in summary['collaboration_patterns'].items():
            logger.info(f"  {pattern}: {count}")
    
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

if __name__ == "__main__":
    main()
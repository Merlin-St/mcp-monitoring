#!/usr/bin/env python3
"""
Financial MCP Server Stage 4 - DataFrame Processing for O*NET Economic Task Classification

Processes the evaluation results from conseq_fin_stage4_inspect.py
by reading the .eval files and converting them to JSON and CSV formats.

This should be run after:
    inspect eval conseq_fin_stage4_inspect.py --model anthropic/claude-sonnet-4-20250514

Usage:
    python conseq_fin_stage4_dfprocessing.py
"""

import json
import os
from pathlib import Path
from datetime import datetime
import logging
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

def main():
    """Main DataFrame processing function"""
    logger.info("Starting Stage 4 DataFrame Processing for O*NET Economic Task Classification")
    
    # Default log directory that Inspect uses
    log_dir = "logs"
    
    # Check if logs directory exists
    if not Path(log_dir).exists():
        logger.error(f"Log directory {log_dir} not found. Run inspect eval first.")
        return
    
    # Find the latest Stage 4 .eval file
    stage4_files = list(Path(log_dir).glob("*onet-economic-task-classification*.eval"))
    if not stage4_files:
        logger.error(f"No Stage 4 .eval files found in {log_dir}. Run inspect eval first.")
        return
    
    # Get the most recent Stage 4 file
    latest_stage4_file = max(stage4_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Using latest Stage 4 file: {latest_stage4_file.name}")
    
    # Create a temporary directory with just this file for DataFrame processing
    import tempfile
    import shutil
    temp_dir = tempfile.mkdtemp()
    temp_file = Path(temp_dir) / latest_stage4_file.name
    shutil.copy2(latest_stage4_file, temp_file)
    
    # Use the temp directory for DataFrame processing
    log_dir = temp_dir
    
    try:
        # Read results using messages DataFrame
        from inspect_ai.analysis.beta import samples_df, messages_df
        
        samples_df_data = samples_df(log_dir)
        messages_df_data = messages_df(log_dir)
        logger.info(f"Loaded samples DataFrame with {len(samples_df_data)} samples")
        logger.info(f"Loaded messages DataFrame with {len(messages_df_data)} messages")
        
        # Process results by joining samples and messages DataFrames
        results = []
        valid_responses = 0
        classification_counts = {
            'automation_levels': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            'economic_impacts': {1: 0, 2: 0, 3: 0},
            'confidence_levels': {'H': 0, 'M': 0, 'L': 0},
            'occupation_categories': {}
        }
        
        # Group messages by sample_id to get assistant responses
        assistant_messages = messages_df_data[messages_df_data['role'] == 'assistant']
        
        for idx, sample_row in samples_df_data.iterrows():
            sample_id = sample_row.get("sample_id", f"sample_{idx}")
            
            # Find the assistant message for this sample
            sample_messages = assistant_messages[assistant_messages['sample_id'] == sample_id]
            
            sample_result = {
                "sample_id": sample_id,
                "input_data": {},
                "raw_output": "",
                "score": sample_row.get("score_onet_classification_scorer", 0),
                "score_explanation": ""
            }
            
            # Parse input data from user message
            user_messages = messages_df_data[
                (messages_df_data['sample_id'] == sample_id) & 
                (messages_df_data['role'] == 'user')
            ]
            if not user_messages.empty:
                user_content = user_messages.iloc[0]['content']
                try:
                    # The input should be JSON containing tool data
                    sample_result["input_data"] = json.loads(user_content)
                except:
                    sample_result["input_data"] = {"raw_input": str(user_content)}
            
            # Get assistant response (the actual model output)
            if not sample_messages.empty:
                sample_result["raw_output"] = sample_messages.iloc[0]['content']
            
            # Try to parse the LLM output using robust JSON extraction
            try:
                if sample_result["raw_output"]:
                    completion = sample_result["raw_output"]
                    json_obj = None
                    
                    # First try: direct JSON parsing
                    try:
                        json_obj = json.loads(completion)
                    except json.JSONDecodeError:
                        # Second try: find JSON block in text (handle markdown code blocks)
                        import re
                        # Remove markdown code blocks
                        if completion.startswith('```'):
                            lines = completion.split('\n')
                            if len(lines) > 2:
                                # Remove first and last lines (```json and ```)
                                completion = '\n'.join(lines[1:-1])
                        
                        try:
                            json_obj = json.loads(completion)
                        except json.JSONDecodeError:
                            # Third try: find JSON pattern in text
                            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                            json_matches = re.findall(json_pattern, completion, re.DOTALL)
                            
                            for match in json_matches:
                                try:
                                    json_obj = json.loads(match)
                                    break
                                except json.JSONDecodeError:
                                    continue
                    
                    if json_obj:
                        sample_result["parsed_output"] = json_obj
                        
                        # Count valid responses (score > 0)
                        if sample_result["score"] > 0:
                            valid_responses += 1
                            
                            # Collect classification statistics
                            automation_level = json_obj.get("automation_level")
                            if automation_level in [1, 2, 3, 4, 5]:
                                classification_counts['automation_levels'][automation_level] += 1
                            
                            economic_impact = json_obj.get("economic_impact")
                            if economic_impact in [1, 2, 3]:
                                classification_counts['economic_impacts'][economic_impact] += 1
                            
                            confidence = json_obj.get("confidence")
                            if confidence in ['H', 'M', 'L']:
                                classification_counts['confidence_levels'][confidence] += 1
                            
                            occupation_category = json_obj.get("occupation_category", "")
                            if occupation_category:
                                if occupation_category not in classification_counts['occupation_categories']:
                                    classification_counts['occupation_categories'][occupation_category] = 0
                                classification_counts['occupation_categories'][occupation_category] += 1
                    else:
                        sample_result["parsed_output"] = None
                        sample_result["error"] = "Could not extract JSON"
                            
                else:
                    sample_result["parsed_output"] = None
                    sample_result["error"] = "No output generated"
                    
            except Exception as e:
                sample_result["parsed_output"] = None
                sample_result["error"] = f"Processing error: {str(e)}"
            
            results.append(sample_result)
        
        # Create summary
        summary = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "model": MODEL,
            "total_samples": len(results),
            "valid_responses": valid_responses,
            "invalid_responses": len(results) - valid_responses,
            "classification_statistics": classification_counts
        }
        
        # Add model usage from samples DataFrame if available
        if "model_usage" in samples_df_data.columns:
            total_input_tokens = 0
            total_output_tokens = 0
            for _, row in samples_df_data.iterrows():
                if row["model_usage"]:
                    try:
                        usage_data = json.loads(row["model_usage"]) if isinstance(row["model_usage"], str) else row["model_usage"]
                        if MODEL in usage_data:
                            total_input_tokens += usage_data[MODEL].get("input_tokens", 0)
                            total_output_tokens += usage_data[MODEL].get("output_tokens", 0)
                    except:
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
        
        # Convert any pandas NA values to None for JSON serialization
        def convert_na_to_none(obj):
            if isinstance(obj, dict):
                return {k: convert_na_to_none(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_na_to_none(v) for v in obj]
            elif pd.isna(obj):
                return None
            else:
                return obj
        
        output_data = convert_na_to_none(output_data)
        
        output_file = "conseq_fin_stage4_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Also save DataFrame as CSV for easy inspection using pandas
        results_df = pd.DataFrame(results)
        
        # Extract input_data fields into separate columns
        results_df['tool_id'] = results_df.apply(
            lambda row: json.loads(row['input_data']).get('tool_id', '') if isinstance(row['input_data'], str) else row['input_data'].get('tool_id', ''), 
            axis=1
        )
        results_df['tool_name'] = results_df.apply(
            lambda row: json.loads(row['input_data']).get('tool_name', '') if isinstance(row['input_data'], str) else row['input_data'].get('tool_name', ''), 
            axis=1
        )
        results_df['tool_description'] = results_df.apply(
            lambda row: json.loads(row['input_data']).get('tool_description', '') if isinstance(row['input_data'], str) else row['input_data'].get('tool_description', ''), 
            axis=1
        )
        results_df['server_name'] = results_df.apply(
            lambda row: json.loads(row['input_data']).get('server_name', '') if isinstance(row['input_data'], str) else row['input_data'].get('server_name', ''), 
            axis=1
        )
        results_df['server_description'] = results_df.apply(
            lambda row: json.loads(row['input_data']).get('server_description', '') if isinstance(row['input_data'], str) else row['input_data'].get('server_description', ''), 
            axis=1
        )
        results_df['finance_is_finance_llm'] = results_df.apply(
            lambda row: json.loads(row['input_data']).get('finance_is_finance_llm', 0) if isinstance(row['input_data'], str) else row['input_data'].get('finance_is_finance_llm', 0), 
            axis=1
        )
        
        # Add key parsed fields as separate columns for better CSV readability
        results_df['analysis_notes'] = results_df.apply(
            lambda row: row['parsed_output'].get('analysis_notes', '') if isinstance(row['parsed_output'], dict) else '', 
            axis=1
        )
        results_df['primary_onet_task'] = results_df.apply(
            lambda row: row['parsed_output'].get('primary_onet_task', '') if isinstance(row['parsed_output'], dict) else '', 
            axis=1
        )
        results_df['secondary_onet_tasks'] = results_df.apply(
            lambda row: row['parsed_output'].get('secondary_onet_tasks', '') if isinstance(row['parsed_output'], dict) else '', 
            axis=1
        )
        results_df['occupation_category'] = results_df.apply(
            lambda row: row['parsed_output'].get('occupation_category', '') if isinstance(row['parsed_output'], dict) else '', 
            axis=1
        )
        results_df['automation_level'] = results_df.apply(
            lambda row: row['parsed_output'].get('automation_level', '') if isinstance(row['parsed_output'], dict) else '', 
            axis=1
        )
        results_df['economic_impact'] = results_df.apply(
            lambda row: row['parsed_output'].get('economic_impact', '') if isinstance(row['parsed_output'], dict) else '', 
            axis=1
        )
        results_df['confidence'] = results_df.apply(
            lambda row: row['parsed_output'].get('confidence', '') if isinstance(row['parsed_output'], dict) else '', 
            axis=1
        )
        results_df['task_skills'] = results_df.apply(
            lambda row: row['parsed_output'].get('task_skills', '') if isinstance(row['parsed_output'], dict) else '', 
            axis=1
        )
        
        # Reorder columns to put key fields first
        input_columns = ['tool_id', 'tool_name', 'tool_description', 'server_name', 'server_description', 'finance_is_finance_llm']
        classification_columns = ['primary_onet_task', 'secondary_onet_tasks', 'occupation_category', 'automation_level', 'economic_impact', 'confidence', 'task_skills', 'analysis_notes']
        other_columns = ['sample_id', 'input_data', 'raw_output', 'score', 'score_explanation', 'parsed_output']
        
        # Create ordered column list
        ordered_columns = input_columns + classification_columns + other_columns
        
        # Select only columns that exist in the DataFrame
        existing_columns = [col for col in ordered_columns if col in results_df.columns]
        results_df = results_df[existing_columns]
        
        df_output_file = "conseq_fin_stage4_results.csv"
        results_df.to_csv(df_output_file, index=False)
        
        logger.info(f"Results saved to {output_file}")
        logger.info(f"DataFrame saved to {df_output_file}")
        logger.info(f"Summary: {valid_responses}/{len(results)} valid responses")
        
        # Log analysis overview
        logger.info("=== O*NET Economic Task Classification Analysis ===")
        
        # Automation levels
        logger.info(f"Automation Level Distribution:")
        total_classified = sum(classification_counts['automation_levels'].values())
        for level, count in classification_counts['automation_levels'].items():
            percentage = (count / total_classified * 100) if total_classified > 0 else 0
            level_desc = {1: "Monitoring", 2: "Analysis", 3: "Workflow", 4: "Execution", 5: "Autonomous"}
            logger.info(f"  Level {level} ({level_desc.get(level, 'Unknown')}): {count} ({percentage:.1f}%)")
        
        # Economic impact
        logger.info(f"Economic Impact Distribution:")
        total_impact = sum(classification_counts['economic_impacts'].values())
        for impact, count in classification_counts['economic_impacts'].items():
            percentage = (count / total_impact * 100) if total_impact > 0 else 0
            impact_desc = {1: "Low", 2: "Medium", 3: "High"}
            logger.info(f"  Impact {impact} ({impact_desc.get(impact, 'Unknown')}): {count} ({percentage:.1f}%)")
        
        # Confidence levels
        logger.info(f"Confidence Level Distribution:")
        total_conf = sum(classification_counts['confidence_levels'].values())
        for conf, count in classification_counts['confidence_levels'].items():
            percentage = (count / total_conf * 100) if total_conf > 0 else 0
            logger.info(f"  {conf}: {count} ({percentage:.1f}%)")
        
        # Top occupation categories
        logger.info(f"Top Occupation Categories:")
        sorted_categories = sorted(classification_counts['occupation_categories'].items(), key=lambda x: x[1], reverse=True)
        for category, count in sorted_categories[:5]:
            percentage = (count / total_classified * 100) if total_classified > 0 else 0
            logger.info(f"  {category}: {count} ({percentage:.1f}%)")
        
        # High-impact tools analysis
        high_impact_tools = results_df[
            (results_df['economic_impact'] == 3) & 
            (results_df['automation_level'].isin([4, 5]))
        ]
        if len(high_impact_tools) > 0:
            logger.info(f"=== High Economic Impact + High Automation Tools: {len(high_impact_tools)} ===")
            for _, row in high_impact_tools.head(5).iterrows():
                tool_name = row.get('tool_name', 'Unknown')
                primary_task = row.get('primary_onet_task', 'No task')[:80]
                logger.info(f"  {tool_name}: {primary_task}...")
        
        # Finance-specific analysis
        finance_tools = results_df[results_df['finance_is_finance_llm'] == 1]
        if len(finance_tools) > 0:
            logger.info(f"=== Finance-Specific Tools Analysis: {len(finance_tools)} tools ===")
            finance_categories = finance_tools['occupation_category'].value_counts()
            for category, count in finance_categories.head(3).items():
                logger.info(f"  {category}: {count} tools")
        
        logger.info("=== End Analysis ===")
        
        # Log next steps
        logger.info("=== Analysis Complete ===")
        logger.info(f"Results ready for further analysis and visualization")
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
            
    except Exception as e:
        logger.error(f"DataFrame processing failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
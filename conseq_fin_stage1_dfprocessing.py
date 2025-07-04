#!/usr/bin/env python3
"""
Financial MCP Server Stage 1 - DataFrame Processing

Processes the evaluation results from conseq_fin_stage1_inspect.py
by reading the .eval files and converting them to JSON and CSV formats.

This should be run after:
    inspect eval conseq_fin_stage1_inspect.py --model anthropic/claude-sonnet-4-20250514

Usage:
    python conseq_fin_stage1_dfprocessing.py
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
        logging.FileHandler('conseq_fin_stage1_dfprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MODEL = "anthropic/claude-sonnet-4-20250514"

def main():
    """Main DataFrame processing function"""
    logger.info("Starting Stage 1 DataFrame Processing")
    
    # Default log directory that Inspect uses
    log_dir = "logs"
    
    # Check if logs directory exists
    if not Path(log_dir).exists():
        logger.error(f"Log directory {log_dir} not found. Run inspect eval first.")
        return
    
    # Find the latest Stage 1 .eval file
    stage1_files = list(Path(log_dir).glob("*finance-identification-task*.eval"))
    if not stage1_files:
        logger.error(f"No Stage 1 .eval files found in {log_dir}. Run inspect eval first.")
        return
    
    # Get the most recent Stage 1 file
    latest_stage1_file = max(stage1_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Using latest Stage 1 file: {latest_stage1_file.name}")
    
    # Create a temporary directory with just this file for DataFrame processing
    import tempfile
    import shutil
    temp_dir = tempfile.mkdtemp()
    temp_file = Path(temp_dir) / latest_stage1_file.name
    shutil.copy2(latest_stage1_file, temp_file)
    
    # Use the temp directory for DataFrame processing
    log_dir = temp_dir
    
    try:
        # Read results using messages DataFrame
        from inspect_ai.analysis.beta import samples_df, messages_df
        
        samples_df_data = samples_df(log_dir)
        messages_df_data = messages_df(log_dir)
        logger.info(f"Loaded samples DataFrame with {len(samples_df_data)} samples")
        logger.info(f"Loaded messages DataFrame with {len(messages_df_data)} messages")
        logger.info(f"Messages columns: {list(messages_df_data.columns)}")
        
        # Process results by joining samples and messages DataFrames
        results = []
        valid_responses = 0
        finance_identified = 0
        
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
                "score": sample_row.get("score_finance_filter_scorer", 0),
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
                    # Extract the JSON part from the user content (after "MCP Server Data:")
                    if "MCP Server Data:" in user_content:
                        json_part = user_content.split("MCP Server Data:")[1].strip()
                        sample_result["input_data"] = json.loads(json_part)
                    else:
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
                            
                            # Count finance-identified servers
                            if json_obj.get("is_finance_llm") == "yes":
                                finance_identified += 1
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
            "finance_identified": finance_identified,
            "finance_percentage": (finance_identified / len(results) * 100) if results else 0,
            "log_directory": log_dir
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
        
        output_file = "conseq_fin_stage1_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Also save DataFrame as CSV for easy inspection using pandas
        results_df = pd.DataFrame(results)
        
        # Add key parsed fields as separate columns for better CSV readability
        results_df['is_finance_llm'] = results_df.apply(
            lambda row: row['parsed_output'].get('is_finance_llm', '') if isinstance(row['parsed_output'], dict) else '', 
            axis=1
        )
        results_df['confidence'] = results_df.apply(
            lambda row: row['parsed_output'].get('confidence', '') if isinstance(row['parsed_output'], dict) else '', 
            axis=1
        )
        results_df['threat_models'] = results_df.apply(
            lambda row: str(row['parsed_output'].get('threat_models', [])) if isinstance(row['parsed_output'], dict) else '', 
            axis=1
        )
        results_df['analysis_notes'] = results_df.apply(
            lambda row: row['parsed_output'].get('analysis_notes', '') if isinstance(row['parsed_output'], dict) else '', 
            axis=1
        )
        
        df_output_file = "conseq_fin_stage1_results.csv"
        results_df.to_csv(df_output_file, index=False)
        
        logger.info(f"Results saved to {output_file}")
        logger.info(f"DataFrame saved to {df_output_file}")
        logger.info(f"Summary: {valid_responses}/{len(results)} valid responses, {finance_identified} servers identified as finance-related")
        
        # Log next steps
        if finance_identified > 0:
            logger.info(f"Next step: Run Stage 2 evaluation with: inspect eval conseq_fin_stage2_inspect.py --model {MODEL}")
        else:
            logger.warning("No finance-related servers identified. Check results and consider adjusting criteria.")
            
    except Exception as e:
        logger.error(f"DataFrame processing failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
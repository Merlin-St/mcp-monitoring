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
                            if json_obj.get("is_finance_llm") == 1:
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
        
        # Extract input_data fields into separate columns
        results_df['server_name'] = results_df.apply(
            lambda row: row['input_data'].get('server_name', '') if isinstance(row['input_data'], dict) else '', 
            axis=1
        )
        results_df['server_id'] = results_df.apply(
            lambda row: row['input_data'].get('server_id', '') if isinstance(row['input_data'], dict) else '', 
            axis=1
        )
        results_df['description'] = results_df.apply(
            lambda row: row['input_data'].get('description', '') if isinstance(row['input_data'], dict) else '', 
            axis=1
        )
        results_df['readme_filtered'] = results_df.apply(
            lambda row: row['input_data'].get('readme_filtered', '') if isinstance(row['input_data'], dict) else '', 
            axis=1
        )
        results_df['readme_summary'] = results_df.apply(
            lambda row: row['input_data'].get('readme_summary', '') if isinstance(row['input_data'], dict) else '', 
            axis=1
        )
        results_df['tools'] = results_df.apply(
            lambda row: str(row['input_data'].get('tools', [])) if isinstance(row['input_data'], dict) else '', 
            axis=1
        )
        results_df['topics'] = results_df.apply(
            lambda row: str(row['input_data'].get('topics', [])) if isinstance(row['input_data'], dict) else '', 
            axis=1
        )
        results_df['data_sources'] = results_df.apply(
            lambda row: str(row['input_data'].get('data_sources', [])) if isinstance(row['input_data'], dict) else '', 
            axis=1
        )
        
        # Add key parsed fields as separate columns for better CSV readability
        
        # Basic classification fields
        results_df['server'] = results_df.apply(
            lambda row: row['parsed_output'].get('server', '') if isinstance(row['parsed_output'], dict) else '', 
            axis=1
        )
        results_df['analysis_notes'] = results_df.apply(
            lambda row: row['parsed_output'].get('analysis_notes', '') if isinstance(row['parsed_output'], dict) else '', 
            axis=1
        )
        results_df['is_finance_llm'] = results_df.apply(
            lambda row: row['parsed_output'].get('is_finance_llm', '') if isinstance(row['parsed_output'], dict) else '', 
            axis=1
        )
        results_df['asset_type'] = results_df.apply(
            lambda row: row['parsed_output'].get('asset_type', '') if isinstance(row['parsed_output'], dict) else '', 
            axis=1
        )
        results_df['confidence'] = results_df.apply(
            lambda row: row['parsed_output'].get('confidence', '') if isinstance(row['parsed_output'], dict) else '', 
            axis=1
        )
        results_df['level'] = results_df.apply(
            lambda row: row['parsed_output'].get('level', '') if isinstance(row['parsed_output'], dict) else '', 
            axis=1
        )
        
        # Financial capability fields
        results_df['research_and_risk_assessment'] = results_df.apply(
            lambda row: row['parsed_output'].get('research_and_risk_assessment', '') if isinstance(row['parsed_output'], dict) else '', 
            axis=1
        )
        results_df['documentation_gathering'] = results_df.apply(
            lambda row: row['parsed_output'].get('documentation_gathering', '') if isinstance(row['parsed_output'], dict) else '', 
            axis=1
        )
        results_df['application_and_review'] = results_df.apply(
            lambda row: row['parsed_output'].get('application_and_review', '') if isinstance(row['parsed_output'], dict) else '', 
            axis=1
        )
        results_df['identity_verification'] = results_df.apply(
            lambda row: row['parsed_output'].get('identity_verification', '') if isinstance(row['parsed_output'], dict) else '', 
            axis=1
        )
        results_df['authorization_account_transactions'] = results_df.apply(
            lambda row: row['parsed_output'].get('authorization_account_transactions', '') if isinstance(row['parsed_output'], dict) else '', 
            axis=1
        )
        results_df['account_opening'] = results_df.apply(
            lambda row: row['parsed_output'].get('account_opening', '') if isinstance(row['parsed_output'], dict) else '', 
            axis=1
        )
        
        # Transfer capability fields
        results_df['transfer_bank_and_fund_bank_account'] = results_df.apply(
            lambda row: row['parsed_output'].get('transfer_bank_and_fund_bank_account', '') if isinstance(row['parsed_output'], dict) else '', 
            axis=1
        )
        results_df['transfer_credit_card'] = results_df.apply(
            lambda row: row['parsed_output'].get('transfer_credit_card', '') if isinstance(row['parsed_output'], dict) else '', 
            axis=1
        )
        results_df['transfer_paypal_stripe_payments'] = results_df.apply(
            lambda row: row['parsed_output'].get('transfer_paypal_stripe_payments', '') if isinstance(row['parsed_output'], dict) else '', 
            axis=1
        )
        results_df['transfer_stock_invest'] = results_df.apply(
            lambda row: row['parsed_output'].get('transfer_stock_invest', '') if isinstance(row['parsed_output'], dict) else '', 
            axis=1
        )
        results_df['transfer_crypto_and_stablecoin'] = results_df.apply(
            lambda row: row['parsed_output'].get('transfer_crypto_and_stablecoin', '') if isinstance(row['parsed_output'], dict) else '', 
            axis=1
        )
        
        # Security and sensitive data field
        results_df['sensitive_data_required'] = results_df.apply(
            lambda row: row['parsed_output'].get('sensitive_data_required', '') if isinstance(row['parsed_output'], dict) else '', 
            axis=1
        )
        
        # Reorder columns to put input data fields first for better readability
        input_columns = ['server_name', 'server_id', 'description', 'readme_filtered', 'readme_summary', 'tools', 'topics', 'data_sources']
        analysis_columns = ['server', 'analysis_notes', 'is_finance_llm', 'asset_type', 'confidence', 'level']
        capability_columns = [
            'research_and_risk_assessment', 'documentation_gathering', 'application_and_review',
            'identity_verification', 'authorization_account_transactions', 'account_opening'
        ]
        transfer_columns = [
            'transfer_bank_and_fund_bank_account', 'transfer_credit_card', 'transfer_paypal_stripe_payments',
            'transfer_stock_invest', 'transfer_crypto_and_stablecoin'
        ]
        other_columns = ['sensitive_data_required', 'sample_id', 'input_data', 'raw_output', 'score', 'score_explanation', 'parsed_output']
        
        # Create ordered column list
        ordered_columns = input_columns + analysis_columns + capability_columns + transfer_columns + other_columns
        
        # Select only columns that exist in the DataFrame
        existing_columns = [col for col in ordered_columns if col in results_df.columns]
        results_df = results_df[existing_columns]
        
        df_output_file = "conseq_fin_stage1_results.csv"
        results_df.to_csv(df_output_file, index=False)
        
        logger.info(f"Results saved to {output_file}")
        logger.info(f"DataFrame saved to {df_output_file}")
        logger.info(f"Summary: {valid_responses}/{len(results)} valid responses, {finance_identified} servers identified as finance-related")
        
        # Quick analysis overview for the 18 CSV fields
        logger.info("=== Field Analysis Overview ===")
        
        # is_finance_llm (binary)
        finance_1 = len(results_df[results_df['is_finance_llm'] == 1])
        logger.info(f"is_finance_llm: {finance_1}/{len(results_df)} ({finance_1/len(results_df)*100:.0f}%)")
        
        # Confidence (H/M/L)
        conf_counts = results_df['confidence'].value_counts()
        total = len(results_df)
        h_count = conf_counts.get('H', 0)
        m_count = conf_counts.get('M', 0)
        l_count = conf_counts.get('L', 0)
        logger.info(f"confidence: H={h_count} ({h_count/total*100:.0f}%), M={m_count} ({m_count/total*100:.0f}%), L={l_count} ({l_count/total*100:.0f}%)")
        
        # Level (0-5)
        level_counts = results_df['level'].value_counts()
        level_summary = {}
        for level in [0, 1, 2, 3, 4, 5]:
            count = level_counts.get(level, 0)
            level_summary[str(level)] = f"{count} ({count/total*100:.0f}%)"
        logger.info(f"level: {level_summary}")
        
        asset_counts = results_df['asset_type'].value_counts()
        top_assets = dict(list(asset_counts.items())[:3])
        logger.info(f"asset_type: {top_assets}")
        
        # Financial capabilities (binary 0/1)
        capabilities = [
            'research_and_risk_assessment', 'documentation_gathering', 'application_and_review',
            'identity_verification', 'authorization_account_transactions', 'account_opening'
        ]
        for cap in capabilities:
            cap_1 = len(results_df[results_df[cap] == 1])
            logger.info(f"{cap}: {cap_1}/{len(results_df)} ({cap_1/len(results_df)*100:.0f}%)")
        
        # Transfer capabilities (binary 0/1)
        transfers = [
            'transfer_bank_and_fund_bank_account', 'transfer_credit_card', 'transfer_paypal_stripe_payments',
            'transfer_stock_invest', 'transfer_crypto_and_stablecoin'
        ]
        for trans in transfers:
            trans_1 = len(results_df[results_df[trans] == 1])
            logger.info(f"{trans}: {trans_1}/{len(results_df)} ({trans_1/len(results_df)*100:.0f}%)")
        
        # Sensitive data (show examples)
        sensitive_vals = results_df['sensitive_data_required'].dropna()
        sensitive_vals = sensitive_vals[sensitive_vals != '']
        examples = list(sensitive_vals.unique())[:3]
        logger.info(f"sensitive_data_required: {len(sensitive_vals)} servers, examples: {examples}")
        
        # High-level servers (level 4-5) with examples
        high_level = results_df[results_df['level'].isin([4, 5])]
        if len(high_level) > 0:
            logger.info(f"=== High-Level Servers (4-5): {len(high_level)} servers ===")
            for _, row in high_level.head(3).iterrows():
                server_name = row.get('server', 'Unknown')
                # Get description from input_data
                input_data = row.get('input_data', {})
                description = input_data.get('description', input_data.get('readme_content', 'No description'))[:100]
                logger.info(f"Level {row['level']}: {server_name} - {description}...")
        
        # Transfer capability servers with examples
        transfer_fields = [
            'transfer_bank_and_fund_bank_account', 'transfer_credit_card', 'transfer_paypal_stripe_payments',
            'transfer_stock_invest', 'transfer_crypto_and_stablecoin'
        ]
        transfer_servers = results_df[results_df[transfer_fields].eq(1).any(axis=1)]
        if len(transfer_servers) > 0:
            logger.info(f"=== Transfer Capability Servers: {len(transfer_servers)} servers ===")
            for _, row in transfer_servers.head(3).iterrows():
                server_name = row.get('server', 'Unknown')
                # Get description from input_data
                input_data = row.get('input_data', {})
                description = input_data.get('description', input_data.get('readme_content', 'No description'))[:100]
                # Find which transfer capabilities are enabled
                enabled_transfers = [field.replace('transfer_', '') for field in transfer_fields if row.get(field) == 1]
                logger.info(f"{server_name} - {description}... [Transfers: {', '.join(enabled_transfers)}]")
        
        logger.info("=== End Field Analysis ===")
        
        # Log next steps
        
            
    except Exception as e:
        logger.error(f"DataFrame processing failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
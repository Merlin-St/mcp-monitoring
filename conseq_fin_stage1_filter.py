#!/usr/bin/env python3
"""
Financial MCP Server Stage 1 Filter

Uses Inspect framework to evaluate MCP servers for financial relevance
using LLM-based analysis with threat model mapping.

System Prompt 1: Finance Tool Identification and Threat Model Mapping
"""

import asyncio
import json
import os
from pathlib import Path

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from inspect_ai import Task, eval_async, task
from inspect_ai.dataset import json_dataset, Sample
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import TaskState, generate
from datetime import datetime
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conseq_fin_stage1_filter.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MODEL = "anthropic/claude-sonnet-4-20250514"

FINANCE_IDENTIFICATION_SYSTEM_PROMPT = """
ONLY RESPOND WITH JSON. NO EXPLANATIONS. NO MARKDOWN. NO TEXT OUTSIDE JSON.

You are a financial technology risk analyst identifying AI-enabled MCP tools that interact
  with financial systems and mapping them to three threat models.

  Threat Models:
  - TM1 - Correlated Credit Risk: AI tools in credit decisioning, potentially causing
  systematic mispricing across institutions.
  - TM2 - Deposit Stickiness Erosion: AI tools enabling rapid deposit movement between banks,
   destabilizing funding bases.
  - TM3 - Autonomous Payment Systems: AI agents with direct payment capabilities creating
  systemic risks.

  Specific Tasks:

  TM1 - Correlated Credit:
  - loan_application_intake - Processing loan applications
  - kyc_fraud_checks - KYC or fraud detection
  - credit_report_retrieval - Pulling credit scores/reports
  - identity_verification - Verifying employment/income/identity
  - affordability_assessment - Analyzing bank statements
  - risk_modeling - Credit risk modeling/pricing
  - collateral_valuation - Assessing collateral value
  - credit_decisioning - Approve/decline decisions
  - condition_setting - Setting loan conditions
  - terms_generation - Creating offers/payment schedules
  - documentation_drafting - Generating agreements
  - fund_disbursement - Distributing funds
  - loan_monitoring - Post-disbursement monitoring
  - payment_servicing - Processing loan payments

  TM2 - Deposit Stickiness:
  - rate_comparison - Comparing bank interest rates
  - bank_risk_monitoring - Detecting institution risk signals
  - automated_transfers - Moving funds between banks
  - sentiment_analysis - Analyzing bank safety from media
  - deposit_optimization - Optimizing deposit allocation
  - risk_alerts - Bank safety alerts
  - account_management - Direct bank account control
  - agent_behavior_tracking - Monitoring other AI agents
  - information_propagation - Sharing bank concerns
  - rate_arbitrage - Exploiting rate differentials

  TM3 - Payment Systems:
  - payment_execution - Making autonomous payments
  - virtual_card_management - Managing virtual cards
  - agent_authentication - AI agent identity verification
  - transaction_authorization - Payment approval
  - open_banking_payments - PSD2/Open Banking payments
  - fund_routing - Moving money between accounts
  - crypto_payments - Cryptocurrency transactions
  - stablecoin_operations - Fiat/stablecoin conversion
  - payment_api_integration - Payment provider integration
  - agent_transactions - Agent-to-agent payments
  - compliance_monitoring - AML/KYC monitoring
  - resource_acquisition - Using payments for resources

  Output Format:
  {
    "tool_name": "string",
    "server": "string",
    "is_finance_llm": "yes|no",
    "confidence": "high|medium|low",
    "threat_models": [
      {
        "model": "TM1|TM2|TM3",
        "tasks": ["task_id_1", "task_id_2"],
        "relevance_explanation": "Brief explanation"
      }
    ],
    "analysis_notes": "Additional context"
  }

RESPOND ONLY WITH JSON.
""".strip()

def setup_api_key():
    """Setup Anthropic API key with AWS proxy support"""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        raise ValueError("ANTHROPIC_API_KEY environment variable required")
    
    # Handle AWS proxy keys
    if api_key.startswith("aws"):
        try:
            # Try to import aisitools for AWS proxy support
            from aisitools.api_key import get_api_key_for_proxy
            api_key = get_api_key_for_proxy(api_key)
            os.environ["ANTHROPIC_API_KEY"] = api_key
            logger.info("Successfully converted AWS proxy API key")
        except ImportError:
            logger.warning("aisitools not available, using AWS proxy key directly")
    
    return api_key

@scorer(metrics=[accuracy()])
def finance_filter_scorer() -> Scorer:
    """
    Custom scorer for validating JSON structure and extracting results
    Tries to extract JSON from responses that might contain additional text
    """
    async def _scorer(state: TaskState, target: Target):
        completion = state.output.completion
        
        # Try to extract JSON from the completion text
        json_obj = None
        
        # First try: direct JSON parsing
        try:
            json_obj = json.loads(completion)
        except json.JSONDecodeError:
            # Second try: find JSON block in text
            import re
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_matches = re.findall(json_pattern, completion, re.DOTALL)
            
            for match in json_matches:
                try:
                    json_obj = json.loads(match)
                    break
                except json.JSONDecodeError:
                    continue
        
        if json_obj is None:
            # Third try: more aggressive JSON extraction
            try:
                # Look for content between first { and last }
                start = completion.find('{')
                end = completion.rfind('}')
                if start != -1 and end != -1 and end > start:
                    potential_json = completion[start:end+1]
                    json_obj = json.loads(potential_json)
            except json.JSONDecodeError:
                pass
        
        if json_obj is None:
            return Score(
                value=0,
                answer=completion,
                explanation="No valid JSON found in response"
            )
        
        # Validate core fields
        core_fields = ["is_finance_llm"]
        has_core_fields = all(field in json_obj for field in core_fields)
        
        if not has_core_fields:
            missing = [field for field in core_fields if field not in json_obj]
            return Score(
                value=0,
                answer=completion,
                explanation=f"Missing core fields: {missing}"
            )
        
        # Validate is_finance_llm value
        valid_finance_values = ["yes", "no", "unclear"]
        if json_obj["is_finance_llm"] not in valid_finance_values:
            return Score(
                value=0,
                answer=completion,
                explanation=f"Invalid is_finance_llm value: {json_obj['is_finance_llm']}"
            )
        
        return Score(
            value=1,
            answer=completion,
            explanation="Valid JSON with required fields extracted"
        )
    
    return _scorer

@task
def finance_identification_task():
    """
    Inspect task for identifying finance-related MCP servers
    """
    dataset_file = "conseq_fin_stage1_input.jsonl"
    
    if not Path(dataset_file).exists():
        logger.error(f"Dataset file {dataset_file} not found. Run conseq_fin_data_prep.py first.")
        raise FileNotFoundError(f"Dataset file {dataset_file} not found")
    
    # Modify the JSONL file to include system prompt in each sample
    modified_dataset_file = "conseq_fin_stage1_input_with_prompt.jsonl"
    
    # Create modified dataset file if it doesn't exist
    if not Path(modified_dataset_file).exists():
        import json
        with open(dataset_file, 'r') as f_in, open(modified_dataset_file, 'w') as f_out:
            for line in f_in:
                sample = json.loads(line)
                # Combine system prompt with the original input
                combined_message = f"{FINANCE_IDENTIFICATION_SYSTEM_PROMPT}\n\nMCP Server Data:\n{sample['input']}"
                sample['input'] = combined_message
                f_out.write(json.dumps(sample) + '\n')
        logger.info(f"Created modified dataset with system prompt: {modified_dataset_file}")
    
    return Task(
        dataset=json_dataset(modified_dataset_file),
        solver=generate(),
        scorer=[finance_filter_scorer()],
        message_limit=50
    )

async def main():
    """Main evaluation function"""
    logger.info("Starting Stage 1: Finance Tool Identification")
    
    # Setup API key
    setup_api_key()
    
    # Check if input file exists
    input_file = "conseq_fin_stage1_input.jsonl"
    if not Path(input_file).exists():
        logger.error(f"Input file {input_file} not found. Please run conseq_fin_data_prep.py first.")
        return
    
    # Count samples
    with open(input_file, 'r') as f:
        sample_count = sum(1 for _ in f)
    logger.info(f"Processing {sample_count} MCP servers for finance identification")
    
    try:
        # Run evaluation with explicit log directory
        log_dir = "conseq_fin_stage1_logs"
        eval_logs = await eval_async(
            tasks=[finance_identification_task()],
            model=MODEL,
            log_dir=log_dir
        )
        
        logger.info(f"Evaluation completed, processing results from {log_dir}")
        
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
                    sample_result["input_data"] = json.loads(user_content)
                except:
                    sample_result["input_data"] = {"raw_input": str(user_content)}
            
            # Get assistant response (the actual model output)
            if not sample_messages.empty:
                sample_result["raw_output"] = sample_messages.iloc[0]['content']
            
            # Try to parse the LLM output
            try:
                if sample_result["raw_output"]:
                    parsed_output = json.loads(sample_result["raw_output"])
                    sample_result["parsed_output"] = parsed_output
                    
                    # Count valid responses (score > 0)
                    if sample_result["score"] > 0:
                        valid_responses += 1
                        
                        # Count finance-identified servers
                        if parsed_output.get("is_finance_llm") == "yes":
                            finance_identified += 1
                            
                else:
                    sample_result["parsed_output"] = None
                    sample_result["error"] = "No output generated"
                    
            except json.JSONDecodeError:
                sample_result["parsed_output"] = None
                sample_result["error"] = "Invalid JSON output"
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
        
        output_file = "conseq_fin_stage1_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Also save DataFrame as CSV for easy inspection using pandas
        import pandas as pd
        results_df = pd.DataFrame(results)
        df_output_file = "conseq_fin_stage1_results.csv"
        results_df.to_csv(df_output_file, index=False)
        
        logger.info(f"Results saved to {output_file}")
        logger.info(f"DataFrame saved to {df_output_file}")
        logger.info(f"Summary: {valid_responses}/{len(results)} valid responses, {finance_identified} servers identified as finance-related")
        
        # Log next steps
        if finance_identified > 0:
            logger.info(f"Next step: Run Stage 2 evaluation with: inspect eval conseq_fin_stage2_assess.py --model {MODEL}")
        else:
            logger.warning("No finance-related servers identified. Check results and consider adjusting criteria.")
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    asyncio.run(main())
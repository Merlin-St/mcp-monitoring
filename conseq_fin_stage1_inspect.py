#!/usr/bin/env python3
"""
Financial MCP Server Stage 1 Filter - Inspect Task Definition

Uses Inspect framework to evaluate MCP servers for financial relevance
using LLM-based analysis with threat model mapping.

This file contains only the task definition for Inspect to run.
Use conseq_fin_stage1_dfprocessing.py to process the results.

Usage:
    inspect eval conseq_fin_stage1_inspect.py --model anthropic/claude-sonnet-4-20250514
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import TaskState, generate
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conseq_fin_stage1_inspect.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

def count_dataset_size(dataset_file):
    """Count the number of samples in the dataset file"""
    if not Path(dataset_file).exists():
        return 0
    
    with open(dataset_file, 'r') as f:
        count = sum(1 for _ in f)
    
    logger.info(f"Dataset {dataset_file} contains {count} samples")
    return count

@task
def finance_identification_task():
    """
    Inspect task for identifying finance-related MCP servers
    """
    dataset_file = "conseq_fin_stage1_input.jsonl"
    
    if not Path(dataset_file).exists():
        logger.error(f"Dataset file {dataset_file} not found. Run conseq_fin_data_prep.py first.")
        raise FileNotFoundError(f"Dataset file {dataset_file} not found")
    
    # Count samples in dataset to set appropriate message limit
    dataset_size = count_dataset_size(dataset_file)
    dynamic_message_limit = dataset_size + 10  # Add buffer for safety
    
    # Modify the JSONL file to include system prompt in each sample
    modified_dataset_file = "conseq_fin_stage1_input_with_prompt.jsonl"
    
    # Always remove existing intermediate file to ensure fresh creation
    if Path(modified_dataset_file).exists():
        Path(modified_dataset_file).unlink()
        logger.info(f"Removed existing intermediate file: {modified_dataset_file}")
    
    # Create modified dataset file
    import json
    with open(dataset_file, 'r') as f_in, open(modified_dataset_file, 'w') as f_out:
        for line in f_in:
            sample = json.loads(line)
            # Combine system prompt with the original input
            combined_message = f"{FINANCE_IDENTIFICATION_SYSTEM_PROMPT}\n\nMCP Server Data:\n{sample['input']}"
            sample['input'] = combined_message
            f_out.write(json.dumps(sample) + '\n')
    logger.info(f"Created modified dataset with system prompt: {modified_dataset_file}")
    
    logger.info(f"Setting message_limit to {dynamic_message_limit} for {dataset_size} samples")
    
    return Task(
        dataset=json_dataset(modified_dataset_file),
        solver=generate(),
        scorer=[finance_filter_scorer()],
        message_limit=dynamic_message_limit
    )

# Setup API key when module is imported
setup_api_key()
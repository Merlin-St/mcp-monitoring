#!/usr/bin/env python3
"""
Financial MCP Server Stage 2 - Inspect Task Definition

Uses Inspect framework to evaluate MCP servers identified as finance-related
in Stage 1 for their consequentiality levels and risk assessment.

This file contains only the task definition for Inspect to run.
Use conseq_fin_stage2_dfprocessing.py to process the results.

Usage:
    inspect eval conseq_fin_stage2_inspect.py --model anthropic/claude-sonnet-4-20250514
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
        logging.FileHandler('conseq_fin_stage2_inspect.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

CONSEQUENTIALITY_ASSESSMENT_SYSTEM_PROMPT = """
You are a financial risk specialist evaluating the consequentiality level of finance-related MCP tools based on their execution capabilities.

Consequentiality Levels:
Level 1 - MONITORING: Read-only access, no modification ability
Level 2 - ADVISING: Analyzes data and provides recommendations, cannot execute
Level 3 - PREPARING: Creates/stages operations for human approval
Level 4 - EXECUTING WITH CONSTRAINTS: Executes within limits or with authentication
Level 5 - EXECUTING WITHOUT CONSTRAINTS: Full autonomous execution, no limits

Assessment Dimensions:
- Execution Capability: What actions can it perform?
- Access Scope: What financial systems/data accessed?
- Autonomy Level: Human oversight required?
- Reversibility: Can actions be undone?
- Financial Impact: Maximum monetary impact
- Regulatory Exposure: Affected regulations

You will be provided with an MCP server that has already been identified as finance-related in Stage 1, along with the threat model analysis. Your task is to assess its consequentiality level.

Output Format:
You must respond with valid JSON in exactly this format:
{
  "server_name": "string",
  "server_id": "string",
  "threat_model": "TM1|TM2|TM3|multiple",
  "specific_tasks": ["task_id_1", "task_id_2"],
  "consequentiality_level": 1-5,
  "confidence": "high|medium|low",
  "capabilities": {
    "can_read_financial_data": true|false,
    "can_modify_financial_data": true|false,
    "can_execute_transactions": true|false,
    "can_make_binding_decisions": true|false,
    "requires_human_approval": true|false,
    "has_monetary_limits": true|false,
    "sensitive_data_required": ["API keys", "account numbers", "credit card numbers", "SSN", "none"]
  },
  "reversibility": "fully|partially|irreversible",
  "regulatory_concerns": ["PSD2", "GDPR", "Basel III", "SOX", "PCI DSS", "etc"],
  "analysis_reasoning": "Brief explanation of level assignment and key risk factors"
}

Important:
- consequentiality_level must be an integer from 1 to 5
- sensitive_data_required should be an array of strings
- regulatory_concerns should be an array of strings
- Provide thorough reasoning for your assessment
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

def load_stage1_results():
    """Load Stage 1 results and filter for finance-related servers"""
    stage1_file = "conseq_fin_stage1_results.json"
    
    if not Path(stage1_file).exists():
        logger.error(f"Stage 1 results file {stage1_file} not found. Run conseq_fin_stage1_dfprocessing.py first.")
        raise FileNotFoundError(f"Stage 1 results file not found")
    
    with open(stage1_file, 'r', encoding='utf-8') as f:
        stage1_data = json.load(f)
    
    # Filter for servers identified as finance-related
    finance_servers = []
    for result in stage1_data["results"]:
        if (result.get("parsed_output") and 
            result["parsed_output"].get("is_finance_llm") == "yes"):
            finance_servers.append(result)
    
    logger.info(f"Found {len(finance_servers)} finance-related servers from Stage 1")
    return finance_servers

def create_stage2_dataset(finance_servers):
    """Create Stage 2 dataset JSONL file from finance-identified servers"""
    stage2_file = "conseq_fin_stage2_input.jsonl"
    
    with open(stage2_file, 'w') as f:
        for server_result in finance_servers:
            # Combine original server data with Stage 1 analysis
            input_data = server_result["input_data"]
            stage1_analysis = server_result["parsed_output"]
            
            # Create enhanced input for Stage 2
            stage2_input = {
                **input_data,  # Include all original server data
                "stage1_analysis": stage1_analysis,
                "identified_threat_models": stage1_analysis.get("threat_models", []),
                "stage1_confidence": stage1_analysis.get("confidence", ""),
                "stage1_notes": stage1_analysis.get("analysis_notes", "")
            }
            
            sample = {
                "input": json.dumps(stage2_input, ensure_ascii=False),
                "target": "",
                "id": server_result["sample_id"],
                "metadata": {"stage": "consequentiality_assessment"}
            }
            
            f.write(json.dumps(sample) + '\n')
    
    logger.info(f"Created Stage 2 dataset: {stage2_file}")
    return stage2_file

@scorer(metrics=[accuracy()])
def consequentiality_scorer() -> Scorer:
    """
    Custom scorer for validating consequentiality assessment JSON structure
    """
    async def _scorer(state: TaskState, target: Target):
        completion = state.output.completion
        
        # Try to extract JSON from the completion text (same robust pattern as Stage 1)
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
        core_fields = ["consequentiality_level"]
        has_core_fields = all(field in json_obj for field in core_fields)
        
        if not has_core_fields:
            missing = [field for field in core_fields if field not in json_obj]
            return Score(
                value=0,
                answer=completion,
                explanation=f"Missing core fields: {missing}"
            )
        
        # Validate consequentiality_level value
        level = json_obj["consequentiality_level"]
        if not (isinstance(level, int) and 1 <= level <= 5):
            return Score(
                value=0,
                answer=completion,
                explanation=f"Invalid consequentiality_level: {level} (must be 1-5)"
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
def consequentiality_assessment_task():
    """
    Inspect task for assessing consequentiality of finance-related MCP servers
    """
    # Load Stage 1 results and create dataset
    finance_servers = load_stage1_results()
    
    if not finance_servers:
        raise ValueError("No finance-related servers found in Stage 1 results")
    
    dataset_file = create_stage2_dataset(finance_servers)
    
    # Count samples to set appropriate message limit
    dataset_size = len(finance_servers)  # Use finance_servers count directly
    dynamic_message_limit = dataset_size + 10  # Add buffer for safety
    
    # Modify the JSONL file to include system prompt in each sample
    modified_dataset_file = "conseq_fin_stage2_input_with_prompt.jsonl"
    
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
            combined_message = f"{CONSEQUENTIALITY_ASSESSMENT_SYSTEM_PROMPT}\n\nMCP Server Data:\n{sample['input']}"
            sample['input'] = combined_message
            f_out.write(json.dumps(sample) + '\n')
    logger.info(f"Created modified Stage 2 dataset with system prompt: {modified_dataset_file}")
    
    logger.info(f"Setting message_limit to {dynamic_message_limit} for {dataset_size} finance servers")
    
    return Task(
        dataset=json_dataset(modified_dataset_file),
        solver=generate(),
        scorer=[consequentiality_scorer()],
        message_limit=dynamic_message_limit
    )

# Setup API key when module is imported
setup_api_key()
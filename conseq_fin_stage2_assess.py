#!/usr/bin/env python3
"""
Financial MCP Server Stage 2 Consequentiality Assessment

Uses Inspect framework to evaluate MCP servers identified as finance-related
in Stage 1 for their consequentiality levels and risk assessment.

System Prompt 2: Financial Tool Consequentiality Assessment
"""

import asyncio
import json
import os
from pathlib import Path

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from inspect_ai import Task, eval_async, task
from inspect_ai.dataset import Dataset, Sample, json_dataset
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
        logging.FileHandler('conseq_fin_stage2_assess.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MODEL = "anthropic/claude-sonnet-4-20250514"

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
        logger.error(f"Stage 1 results file {stage1_file} not found. Run conseq_fin_stage1_filter.py first.")
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
        
        # Validate core fields (simplified from Stage 1 pattern)
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
    
    # Modify the JSONL file to include system prompt in each sample
    modified_dataset_file = "conseq_fin_stage2_input_with_prompt.jsonl"
    
    if not Path(modified_dataset_file).exists():
        import json
        with open(dataset_file, 'r') as f_in, open(modified_dataset_file, 'w') as f_out:
            for line in f_in:
                sample = json.loads(line)
                # Combine system prompt with the original input
                combined_message = f"{CONSEQUENTIALITY_ASSESSMENT_SYSTEM_PROMPT}\n\nMCP Server Data:\n{sample['input']}"
                sample['input'] = combined_message
                f_out.write(json.dumps(sample) + '\n')
        logger.info(f"Created modified Stage 2 dataset with system prompt: {modified_dataset_file}")
    
    return Task(
        dataset=json_dataset(modified_dataset_file),
        solver=generate(),
        scorer=[consequentiality_scorer()],
        message_limit=50
    )

async def main():
    """Main evaluation function"""
    logger.info("Starting Stage 2: Consequentiality Assessment")
    
    # Setup API key
    setup_api_key()
    
    # Check if Stage 1 results exist
    stage1_file = "conseq_fin_stage1_results.json"
    if not Path(stage1_file).exists():
        logger.error(f"Stage 1 results file {stage1_file} not found. Please run conseq_fin_stage1_filter.py first.")
        return
    
    # Load and count finance servers
    try:
        finance_servers = load_stage1_results()
        if not finance_servers:
            logger.warning("No finance-related servers found in Stage 1 results. Nothing to assess.")
            return
            
        logger.info(f"Processing {len(finance_servers)} finance-related MCP servers for consequentiality assessment")
        
    except Exception as e:
        logger.error(f"Error loading Stage 1 results: {e}")
        return
    
    try:
        # Run evaluation with explicit log directory
        log_dir = "conseq_fin_stage2_logs"
        eval_logs = await eval_async(
            tasks=[consequentiality_assessment_task()],
            model=MODEL,
            log_dir=log_dir
        )
        
        logger.info(f"Evaluation completed, processing results from {log_dir}")
        
        # Import DataFrame functions
        from inspect_ai.analysis.beta import samples_df
        
        # Read results as DataFrame
        df = samples_df(log_dir)
        logger.info(f"Loaded DataFrame with {len(df)} samples and columns: {list(df.columns)}")
        
        # Process DataFrame results
        results = []
        valid_responses = 0
        consequentiality_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        for idx, row in df.iterrows():
            sample_result = {
                "sample_id": row.get("id", f"sample_{idx}"),
                "input_data": {},
                "raw_output": row.get("completion", ""),
                "scores": {}
            }
            
            # Parse input data if available
            if "input" in row and row["input"]:
                try:
                    sample_result["input_data"] = json.loads(row["input"])
                except:
                    sample_result["input_data"] = {"raw_input": str(row["input"])}
            
            # Extract scores from DataFrame
            score_columns = [col for col in df.columns if 'score' in col.lower()]
            for col in score_columns:
                if row.get(col) is not None:
                    sample_result["scores"][col] = row[col]
            
            # Try to parse the LLM output
            try:
                if row.get("completion"):
                    parsed_output = json.loads(row["completion"])
                    sample_result["parsed_output"] = parsed_output
                    
                    # Check if response is valid (using general score)
                    main_score = row.get("score", 0)
                    if main_score > 0.7:  # Threshold for valid response
                        valid_responses += 1
                        
                        # Count consequentiality levels
                        level = parsed_output.get("consequentiality_level")
                        if isinstance(level, int) and 1 <= level <= 5:
                            consequentiality_distribution[level] += 1
                            
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
            "consequentiality_distribution": consequentiality_distribution,
            "average_consequentiality": sum(level * count for level, count in consequentiality_distribution.items()) / sum(consequentiality_distribution.values()) if sum(consequentiality_distribution.values()) > 0 else 0,
            "log_directory": log_dir
        }
        
        # Add model usage if available in DataFrame
        if "input_tokens" in df.columns:
            summary["model_usage"] = {
                "input_tokens": df["input_tokens"].sum() if "input_tokens" in df.columns else 0,
                "output_tokens": df["output_tokens"].sum() if "output_tokens" in df.columns else 0,
                "total_tokens": (df["input_tokens"].sum() if "input_tokens" in df.columns else 0) + 
                              (df["output_tokens"].sum() if "output_tokens" in df.columns else 0)
            }
        
        # Save results
        output_data = {
            "summary": summary,
            "results": results
        }
        
        output_file = "conseq_fin_stage2_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Also save DataFrame as CSV for easy inspection
        df_output_file = "conseq_fin_stage2_results.csv"
        df.to_csv(df_output_file, index=False)
        
        logger.info(f"Results saved to {output_file}")
        logger.info(f"DataFrame saved to {df_output_file}")
        logger.info(f"Summary: {valid_responses}/{len(results)} valid responses")
        logger.info(f"Consequentiality distribution: {consequentiality_distribution}")
        logger.info(f"Average consequentiality level: {summary['average_consequentiality']:.2f}")
        
        # Log next steps
        logger.info(f"Next step: Run results merger with: python conseq_fin_results_merger.py")
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    asyncio.run(main())
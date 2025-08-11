#!/usr/bin/env python3
"""
Stage 4 O*NET Classification - Part 3: Automation Level
Classifies MCP tools by their execution capabilities (0-5 scale).

Usage:
    inspect eval conseq_fin_stage4_inspect_3_auto.py --model anthropic/claude-sonnet-4-20250514 --message-limit 50
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from dotenv import load_dotenv
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import TaskState, generate, system_message

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conseq_fin_stage4_inspect_3_auto.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# System prompt for automation level analysis
AUTOMATION_LEVEL_PROMPT = """The following is a description of an AI agent tool from a Model Context Protocol server:

Server name & description: {server_name}: {server_description}
Tool name, description and input schema: {tool_name} {tool_description} {input_schema}

Your task is to classify this AI agent tool's (not the full mcp server - only the specific tool) automation level based on its execution capabilities:

0: Not a functional tool (e.g., just documentation or links)
1: MONITORING - Read-only access, information retrieval only
2: ANALYSIS - Processes and analyzes data, provides recommendations
3: META - Coordinates other tools/servers for execution
4: RESTRICTED EXECUTION - Modifies specific external environments (e.g., executes transactions on specific systems)
5: UNRESTRICTED EXECUTION - Full execution capabilities (e.g., arbitrary code execution, system-wide modifications)

Examples:
Tool: get_account_balance - Retrieves current balance from bank account
Level: 1 (MONITORING - only reads data, no modifications)

Tool: analyze_portfolio - Analyzes investment holdings and provides recommendations
Level: 2 (ANALYSIS - processes data and advises, but doesn't modify anything)

Tool: execute_trade - Places buy/sell orders on stock exchange
Level: 4 (RESTRICTED EXECUTION - modifies external trading environment)

Tool: run_shell_command - Executes arbitrary system commands
Level: 5 (UNRESTRICTED EXECUTION - can modify anything on the system or arbitrary internet actions)

Based on your analysis, classify this tool's automation level. Consider what the tool actually does and its potential impact on external systems. ONLY REPLY WITH A SINGLE DIGIT NUMBER 0-5."""

@scorer(metrics=[accuracy()])
def automation_level_scorer() -> Scorer:
    """Score automation level responses"""
    async def _scorer(state: TaskState, target: Target):
        completion = state.output.completion
        
        if not completion:
            return Score(
                value=0,
                answer=None,
                explanation="No response received"
            )
        
        # Clean the response
        answer = completion.strip()
        
        # Check if it's a valid level (0-5)
        if answer in ['0', '1', '2', '3', '4', '5']:
            return Score(
                value=1,
                answer=answer,  # Keep as string
                explanation=f"Valid automation level: {answer}"
            )
        else:
            # Try to extract a number from the response
            import re
            numbers = re.findall(r'[0-5]', answer)
            if numbers:
                return Score(
                    value=0.5,
                    answer=numbers[0],  # Keep as string
                    explanation=f"Number found in response but not exact match"
                )
            
            return Score(
                value=0,
                answer=answer,
                explanation="Invalid automation level"
            )
    
    return _scorer

@task
def automation_level_task():
    """Task 3: Automation Level Classification"""
    dataset_file = "conseq_fin_stage4_input.jsonl"
    
    if not Path(dataset_file).exists():
        raise FileNotFoundError(f"Dataset file {dataset_file} not found. Run conseq_fin_stage4_data_prep.py first.")
    
    # Count samples
    with open(dataset_file, 'r') as f:
        sample_count = sum(1 for _ in f)
    
    logger.info(f"Loading {sample_count} tool samples for automation level classification")
    
    return Task(
        dataset=json_dataset(dataset_file),
        solver=[
            system_message("""You are an expert at classifying AI tool automation levels.

Automation levels:
0 - Not functional (documentation/links only)
1 - MONITORING (read-only access)
2 - ANALYSIS (process data, provide recommendations)
3 - META (coordinate other tools)
4 - RESTRICTED EXECUTION (modify specific systems)
5 - UNRESTRICTED EXECUTION (arbitrary execution)

Analyze the tool's actual capabilities and potential system impact.
Respond with ONLY a single digit: 0, 1, 2, 3, 4, or 5."""),
            generate()
        ],
        scorer=[automation_level_scorer()],
        message_limit=100
    )
#!/usr/bin/env python3
"""
Stage 4 O*NET Task Classification - Simple Version

Uses Inspect framework to classify MCP tools with natural language responses.
The model will provide analysis in plain text, which we'll process separately.

Usage:
    inspect eval conseq_fin_stage4_inspect_simple.py --model anthropic/claude-sonnet-4-20250514
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
        logging.FileHandler('conseq_fin_stage4_inspect.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Simple prompt that asks for natural language analysis
ANALYSIS_PROMPT = """
You are an expert at analyzing AI agent tools and their capabilities. Analyze the following MCP server tool and provide a comprehensive assessment.

Tool Information:
Server name: {server_name}
Server description: {server_description}
Tool name: {tool_name}
Tool description: {tool_description}
Input schema: {tool_input_schema}

Please provide your analysis covering these aspects:

1. **O*NET Task Mapping**: What occupational tasks does this tool support? Consider which of these economic categories it fits:
   - Information technology systems
   - Art, culture, and creative work
   - Business management and finance
   - Education and HR
   - Scientific research
   - Government and public safety
   - Industrial and agricultural processes
   - Energy management
   - Environmental systems
   - Healthcare services

2. **Collaboration Pattern**: How do humans interact with this tool?
   - Is it directive (complete delegation)?
   - Does it involve feedback loops?
   - Is it iterative refinement?
   - Is it for learning/understanding?
   - Is it for validation/checking?

3. **Automation Level**: Rate the tool's capabilities (0-5):
   - 0: Not functional
   - 1: Monitoring only (read-only)
   - 2: Analysis (process and recommend)
   - 3: Meta (coordinates other tools)
   - 4: Restricted execution (specific environments)
   - 5: Unrestricted execution (arbitrary actions)

4. **Tool Replacement**: What traditional workplace tools or processes could this replace? Consider software, hardware, or manual processes.

Provide a thoughtful analysis addressing each area.
"""

@scorer(metrics=[accuracy()])
def simple_scorer() -> Scorer:
    """
    Simple scorer that just checks if we got a response
    """
    async def _scorer(state: TaskState, target: Target):
        completion = state.output.completion
        
        # Just check if we got a reasonable response
        if completion and len(completion) > 100:
            return Score(
                value=1,
                answer=completion,
                explanation="Response received"
            )
        else:
            return Score(
                value=0,
                answer=completion,
                explanation="No response or too short"
            )
    
    return _scorer

@task
def onet_classification_task():
    """
    Simplified task for O*NET classification of MCP tools
    """
    dataset_file = "conseq_fin_stage4_input.jsonl"
    
    if not Path(dataset_file).exists():
        raise FileNotFoundError(f"Dataset file {dataset_file} not found. Run conseq_fin_stage4_data_prep.py first.")
    
    # Count samples
    with open(dataset_file, 'r') as f:
        sample_count = sum(1 for _ in f)
    
    logger.info(f"Loading {sample_count} tool samples for classification")
    
    return Task(
        dataset=json_dataset(dataset_file),
        solver=[
            system_message("You are an expert at analyzing AI tools and mapping them to occupational tasks and patterns. Provide comprehensive, thoughtful analysis."),
            generate()
        ],
        scorer=[simple_scorer()],
        message_limit=100  # Higher limit for natural language responses
    )
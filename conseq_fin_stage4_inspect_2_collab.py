#!/usr/bin/env python3
"""
Stage 4 O*NET Classification - Part 2: Collaboration Pattern
Analyzes how humans interact with MCP tools.

Usage:
    inspect eval conseq_fin_stage4_inspect_2_collab.py --model anthropic/claude-sonnet-4-20250514 --message-limit 50
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
        logging.FileHandler('conseq_fin_stage4_inspect_2_collab.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# System prompt for collaboration pattern analysis
COLLAB_PATTERN_PROMPT = """The following is a description of an AI agent tool from a Model Context Protocol server:

Server name & description: {server_name}: {server_description}
Tool name, description and input schema: {tool_name} {tool_description} {input_schema}

Your task is to analyze how humans would interact with this AI agent tool (not the full mcp server - only the specific tool) to identify the primary collaboration pattern. Focus on how humans would structure their requests and engage with this tool.

Analyze the tool according to these collaboration patterns:

Directive - Human delegates complete task execution to AI tool with minimal interaction
Feedback Loop - Human and AI tool engage in iterative dialogue to complete task with human mainly providing feedback from the environment
Task Iteration - Human and AI tool engage in iterative dialogue to complete a task with the human refining the AI tool outputs
Learning - Human seeks understanding and explanation rather than direct task completion
Validation - Human uses AI tool to check or validate their own work

Based on your analysis, identify which of the above is most representative of how users would interact with this tool. If multiple patterns are present, select the one that appears most frequently. If you are unsure or there is not enough context to determine the most representative pattern, return 'None' as your answer. Use 'None' liberally---for only some tools will this task be possible."""

@scorer(metrics=[accuracy()])
def collab_pattern_scorer() -> Scorer:
    """Score collaboration pattern responses"""
    async def _scorer(state: TaskState, target: Target):
        completion = state.output.completion
        
        if not completion:
            return Score(
                value=0,
                answer=None,
                explanation="No response received"
            )
        
        # Valid patterns
        valid_patterns = ['Directive', 'Feedback Loop', 'Task Iteration', 'Learning', 'Validation', 'None']
        
        # Clean the response
        answer = completion.strip()
        
        # Check if it's a valid pattern
        if answer in valid_patterns:
            return Score(
                value=1,
                answer=answer,
                explanation=f"Valid collaboration pattern: {answer}"
            )
        else:
            # Try to find a pattern in the response
            for pattern in valid_patterns:
                if pattern.lower() in answer.lower():
                    return Score(
                        value=0.5,
                        answer=pattern,
                        explanation=f"Pattern found in response but not exact match"
                    )
            
            return Score(
                value=0,
                answer=answer,
                explanation="Invalid collaboration pattern"
            )
    
    return _scorer

@task
def collaboration_pattern_task():
    """Task 2: Collaboration Pattern Analysis"""
    dataset_file = "conseq_fin_stage4_input.jsonl"
    
    if not Path(dataset_file).exists():
        raise FileNotFoundError(f"Dataset file {dataset_file} not found. Run conseq_fin_stage4_data_prep.py first.")
    
    # Count samples
    with open(dataset_file, 'r') as f:
        sample_count = sum(1 for _ in f)
    
    logger.info(f"Loading {sample_count} tool samples for collaboration pattern analysis")
    
    return Task(
        dataset=json_dataset(dataset_file),
        solver=[
            system_message("""You are an expert at analyzing human-AI collaboration patterns.

Valid collaboration patterns are:
- Directive: Human delegates complete task execution to AI tool with minimal interaction
- Feedback Loop: Human and AI tool engage in iterative dialogue with human providing environmental feedback
- Task Iteration: Human and AI tool iterate with human refining AI outputs
- Learning: Human seeks understanding and explanation
- Validation: Human uses AI to check or validate their work
- None: Cannot determine pattern from available information

Respond with ONLY one of these exact terms: Directive, Feedback Loop, Task Iteration, Learning, Validation, or None."""),
            generate()
        ],
        scorer=[collab_pattern_scorer()],
        message_limit=100
    )
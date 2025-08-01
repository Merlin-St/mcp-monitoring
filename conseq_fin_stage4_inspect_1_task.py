#!/usr/bin/env python3
"""
Stage 4 O*NET Classification - Part 1: Task Mapping
Maps MCP tools to O*NET occupational tasks using hierarchical search.

Usage:
    inspect eval conseq_fin_stage4_inspect_1_task.py --model anthropic/claude-sonnet-4-20250514 --message-limit 50
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
        logging.FileHandler('conseq_fin_stage4_inspect_1_task.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_task_hierarchy() -> Dict[str, Any]:
    """Load the O*NET task hierarchy"""
    hierarchy_file = "conseq_fin_stage4_hierarchy.json"
    if not Path(hierarchy_file).exists():
        raise FileNotFoundError(f"Hierarchy file {hierarchy_file} not found. Run conseq_fin_stage4_embed_levels.py first.")
    
    with open(hierarchy_file, 'r') as f:
        return json.load(f)

def get_task_options(hierarchy: Dict[str, Any], max_options: int = 20) -> List[str]:
    """Get a diverse sample of O*NET tasks for classification"""
    # For now, get a sample of tasks from different clusters
    # In production, you'd use the hierarchical search approach
    all_tasks = []
    
    # Sample from each top-level category
    for category in hierarchy.get('top_level_categories', []):
        category_tasks = category.get('sample_tasks', [])[:2]  # Get 2 tasks per category
        all_tasks.extend(category_tasks)
    
    # If we need more, add from middle-level clusters
    if len(all_tasks) < max_options:
        for cluster in hierarchy.get('middle_level_clusters', {}).values():
            if len(all_tasks) >= max_options:
                break
            cluster_tasks = cluster.get('tasks', [])[:1]
            all_tasks.extend([t for t in cluster_tasks if t not in all_tasks])
    
    return all_tasks[:max_options]

# System prompt template for task mapping
TASK_MAPPING_PROMPT = """The following is a description of an AI agent tool from a Model Context Protocol server:

Server name & description: {server_name}: {server_description}
Tool name, description and input schema: {tool_name} {tool_description} {input_schema}

Consider the following list of classification options:
{options_str}

Your job is to identify which option best describes the AI agent tool (not the full mcp server - only the specific tool). In this case, the provided options are occupational tasks. Your job is to identify which task is performed by this AI agent tool.

What is the answer? You MUST provide an option exactly as written above. If multiple options apply, choose the single-most pertinent one."""

@scorer(metrics=[accuracy()])
def task_mapping_scorer() -> Scorer:
    """Score task mapping responses"""
    async def _scorer(state: TaskState, target: Target):
        completion = state.output.completion
        
        if not completion:
            return Score(
                value=0,
                answer=None,
                explanation="No response received"
            )
        
        # For task mapping, we expect the model to return one of the provided options
        # Store the response for processing
        return Score(
            value=1,  # We'll validate the response in post-processing
            answer=completion.strip(),
            explanation="Task mapping response received"
        )
    
    return _scorer

@task
def task_mapping_task():
    """Task 1: O*NET Task Mapping"""
    dataset_file = "conseq_fin_stage4_input.jsonl"
    
    if not Path(dataset_file).exists():
        raise FileNotFoundError(f"Dataset file {dataset_file} not found. Run conseq_fin_stage4_data_prep.py first.")
    
    # Load hierarchy for task options
    hierarchy = load_task_hierarchy()
    task_options = get_task_options(hierarchy)
    options_str = "\n".join([f"- {task}" for task in task_options])
    
    # Count samples
    with open(dataset_file, 'r') as f:
        sample_count = sum(1 for _ in f)
    
    logger.info(f"Loading {sample_count} tool samples for task mapping")
    logger.info(f"Using {len(task_options)} task options for classification")
    
    # Create system message with task options
    system_msg = f"""You are an expert at mapping AI tools to occupational tasks.

Available task options:
{options_str}

When analyzing a tool, identify which occupational task from the list above best matches what the tool does.
You MUST respond with one of the exact task descriptions from the list above."""
    
    return Task(
        dataset=json_dataset(dataset_file),
        solver=[
            system_message(system_msg),
            generate()
        ],
        scorer=[task_mapping_scorer()],
        message_limit=100
    )
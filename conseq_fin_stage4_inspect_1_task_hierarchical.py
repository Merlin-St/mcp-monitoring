#!/usr/bin/env python3
"""
Stage 4 O*NET Classification - Part 1: Task Mapping (Hierarchical)
Maps MCP tools to O*NET occupational tasks using proper 3-level hierarchy.

Level 1: 10 top-level categories
Level 2: ~400 middle-level clusters  
Level 3: Individual O*NET tasks

Usage:
    inspect eval conseq_fin_stage4_inspect_1_task_hierarchical.py --model anthropic/claude-sonnet-4-20250514 --message-limit 50
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
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
        logging.FileHandler('conseq_fin_stage4_inspect_1_task_hierarchical.log'),
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

def format_hierarchy_for_prompt(hierarchy: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
    """Format the hierarchy for the classification prompt"""
    # Level 1: Top-level categories
    level1_options = []
    level1_mapping = {}
    
    for key, desc in hierarchy['top_level'].items():
        option_id = f"L1_{key}"
        level1_options.append(f"{option_id}: {desc}")
        level1_mapping[option_id] = key
    
    # Level 2: Sample of middle-level clusters (for demonstration)
    level2_options = []
    level2_mapping = {}
    
    # Get a representative sample from each top-level category
    for top_key, clusters in hierarchy['middle_level'].items():
        for i, cluster in enumerate(clusters[:5]):  # Take first 5 clusters per category
            option_id = f"L2_{top_key}_{cluster['cluster_id']}"
            desc = cluster['description']
            # Include sample tasks to help the model understand
            sample_tasks = cluster['representative_tasks'][:2]
            task_examples = " | ".join(sample_tasks)
            level2_options.append(f"{option_id}: {desc} (Examples: {task_examples})")
            level2_mapping[option_id] = {
                'top_level': top_key,
                'cluster_id': cluster['cluster_id'],
                'description': desc
            }
    
    # Level 3: Individual tasks (sample)
    level3_options = []
    level3_mapping = {}
    
    # Get sample tasks from various clusters
    task_count = 0
    for top_key, clusters in hierarchy['middle_level'].items():
        for cluster in clusters:
            if task_count >= 30:  # Limit to 30 individual tasks
                break
            for task in cluster['representative_tasks'][:1]:
                option_id = f"L3_{cluster['cluster_id']}_{task_count}"
                level3_options.append(f"{option_id}: {task}")
                level3_mapping[option_id] = {
                    'top_level': top_key,
                    'cluster_id': cluster['cluster_id'],
                    'task': task
                }
                task_count += 1
    
    # Combine all options
    all_options = (
        "=== LEVEL 1: TOP-LEVEL CATEGORIES ===\n" +
        "\n".join(level1_options) + "\n\n" +
        "=== LEVEL 2: MIDDLE-LEVEL CLUSTERS ===\n" +
        "\n".join(level2_options[:20]) + "\n\n" +  # Limit for readability
        "=== LEVEL 3: INDIVIDUAL TASKS ===\n" +
        "\n".join(level3_options[:20])  # Limit for readability
    )
    
    all_mappings = {**level1_mapping, **level2_mapping, **level3_mapping}
    
    return all_options, all_mappings

# System prompt template for hierarchical task mapping
TASK_MAPPING_PROMPT = """You are an expert at mapping AI tools to O*NET occupational tasks using a 3-level hierarchy.

The hierarchy has:
- Level 1: 10 broad categories (e.g., "it_systems", "business_finance")
- Level 2: ~400 middle-level clusters of related tasks
- Level 3: Individual specific O*NET tasks

For each tool, select the MOST SPECIFIC classification that accurately describes what the tool does:
- Use Level 3 (individual task) if the tool performs a specific, well-defined task
- Use Level 2 (task cluster) if the tool performs multiple related tasks within a cluster
- Use Level 1 (category) only if the tool is very broad and spans multiple clusters

IMPORTANT: You must respond with ONLY the option ID (e.g., "L3_29_5" or "L2_it_systems_153" or "L1_business_finance"). Do not include any explanation or additional text.

Here are the available options:

{options}

Now classify this tool:

Server name & description: {server_name}: {server_description}
Tool name, description and input schema: {tool_name} {tool_description} {input_schema}

What is the most specific classification for this tool? Respond with ONLY the option ID."""

@scorer(metrics=[accuracy()])
def task_mapping_scorer() -> Scorer:
    """Score hierarchical task mapping responses"""
    async def _scorer(state: TaskState, target: Target):
        completion = state.output.completion
        
        if not completion:
            return Score(
                value=0,
                answer=None,
                explanation="No response received"
            )
        
        # Extract the option ID from the response
        response = completion.strip()
        
        # Check if it's a valid option ID format
        if response.startswith(('L1_', 'L2_', 'L3_')):
            return Score(
                value=1,
                answer=response,
                explanation=f"Valid hierarchical classification: {response}"
            )
        else:
            # Try to extract an ID if the model included extra text
            import re
            pattern = r'(L[123]_[\w_]+)'
            match = re.search(pattern, response)
            if match:
                extracted_id = match.group(1)
                return Score(
                    value=1,
                    answer=extracted_id,
                    explanation=f"Extracted valid classification: {extracted_id}"
                )
            else:
                return Score(
                    value=0,
                    answer=response,
                    explanation="Invalid response format - no valid option ID found"
                )
    
    return _scorer

@task
def task_mapping_task():
    """Task 1: Hierarchical O*NET Task Mapping"""
    dataset_file = "conseq_fin_stage4_input.jsonl"
    
    if not Path(dataset_file).exists():
        raise FileNotFoundError(f"Dataset file {dataset_file} not found. Run conseq_fin_stage4_data_prep.py first.")
    
    # Load hierarchy and format options
    hierarchy = load_task_hierarchy()
    options_str, mappings = format_hierarchy_for_prompt(hierarchy)
    
    # Save mappings for reference
    with open('conseq_fin_stage4_mappings.json', 'w') as f:
        json.dump(mappings, f, indent=2)
    
    # Count samples
    with open(dataset_file, 'r') as f:
        sample_count = sum(1 for _ in f)
    
    logger.info(f"Loading {sample_count} tool samples for hierarchical task mapping")
    logger.info(f"Using 3-level hierarchy with {len(mappings)} total options")
    
    # Create system message
    system_msg = f"""You are an expert at mapping AI tools to O*NET occupational tasks using a hierarchical classification system.

Always select the MOST SPECIFIC level that accurately describes the tool:
- Level 3 (L3_): Individual specific tasks
- Level 2 (L2_): Clusters of related tasks
- Level 1 (L1_): Broad categories

Respond with ONLY the option ID, no additional text."""
    
    return Task(
        dataset=json_dataset(dataset_file),
        solver=[
            system_message(system_msg),
            generate()
        ],
        scorer=[task_mapping_scorer()],
        message_limit=100
    )
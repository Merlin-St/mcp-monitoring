#!/usr/bin/env python3
"""
Stage 4 O*NET Classification - Part 1: Task Mapping (Hierarchical V2)
Implements proper hierarchical classification following the paper's approach.

The model navigates through a 3-level hierarchy:
- Level 1: 10 top-level categories  
- Level 2: ~400 middle-level clusters
- Level 3: ~20k individual O*NET tasks

Usage:
    inspect eval conseq_fin_stage4_inspect_1_task_hierarchical_v2.py --model anthropic/claude-sonnet-4-20250514 --message-limit 50
"""

import json
import logging
import random
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
        logging.FileHandler('conseq_fin_stage4_inspect_1_task_hierarchical_v2.log'),
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

def build_hierarchical_prompt(hierarchy: Dict[str, Any]) -> str:
    """Build a hierarchical prompt showing the taxonomy structure"""
    
    prompt_parts = []
    
    # Add Level 1 categories (always show all 10)
    prompt_parts.append("=== LEVEL 1: TOP-LEVEL CATEGORIES ===")
    prompt_parts.append("Choose the broad category that best describes the tool's function:\n")
    
    for key, desc in hierarchy['top_level'].items():
        prompt_parts.append(f"• {key}: {desc}")
    
    prompt_parts.append("\n=== LEVEL 2: MIDDLE-LEVEL CLUSTERS ===")
    prompt_parts.append("After selecting Level 1, identify the specific cluster within that category.")
    prompt_parts.append("Below are representative clusters for each top-level category:\n")
    
    # For each Level 1 category, show 3-5 representative Level 2 clusters
    for top_key, top_desc in hierarchy['top_level'].items():
        prompt_parts.append(f"\n[{top_key}] clusters:")
        
        clusters = hierarchy['middle_level'].get(top_key, [])
        # Sample up to 5 clusters, ensuring diversity
        sample_size = min(5, len(clusters))
        if sample_size > 0:
            # Try to get clusters of different sizes for diversity
            sorted_clusters = sorted(clusters, key=lambda x: x['size'], reverse=True)
            sampled_clusters = []
            
            # Get one large, one medium, one small cluster if possible
            if len(sorted_clusters) >= 3:
                sampled_clusters.append(sorted_clusters[0])  # Large
                sampled_clusters.append(sorted_clusters[len(sorted_clusters)//2])  # Medium
                sampled_clusters.append(sorted_clusters[-1])  # Small
                
                # Add more if needed
                remaining = [c for c in sorted_clusters if c not in sampled_clusters]
                if len(sampled_clusters) < sample_size and remaining:
                    random.seed(42)  # For reproducibility
                    additional = random.sample(remaining, min(sample_size - len(sampled_clusters), len(remaining)))
                    sampled_clusters.extend(additional)
            else:
                sampled_clusters = sorted_clusters[:sample_size]
            
            for cluster in sampled_clusters:
                cluster_desc = cluster['description'].replace("Tasks related to: ", "")
                prompt_parts.append(f"  - cluster_{cluster['cluster_id']}: {cluster_desc}")
                
                # Show 2 example tasks for this cluster
                example_tasks = cluster['representative_tasks'][:2]
                if example_tasks:
                    prompt_parts.append(f"    Examples: {' | '.join(example_tasks)}")
    
    prompt_parts.append("\n=== LEVEL 3: SPECIFIC TASKS ===")
    prompt_parts.append("The most specific level - individual O*NET tasks within a cluster.")
    prompt_parts.append("Only select this level if the tool performs a very specific, well-defined task.")
    
    prompt_parts.append("\n=== CLASSIFICATION INSTRUCTIONS ===")
    prompt_parts.append("1. First, identify the Level 1 category")
    prompt_parts.append("2. Then, find the most appropriate Level 2 cluster within that category")
    prompt_parts.append("3. If the tool is very specific, identify the exact Level 3 task")
    prompt_parts.append("4. Stop at the most specific level that accurately describes the tool")
    prompt_parts.append("\nIMPORTANT: Many tools should stop at Level 2 (cluster). Only use Level 3 for very specific tools.")
    
    prompt_parts.append("\n=== RESPONSE FORMAT ===")
    prompt_parts.append("You MUST respond in exactly this format:")
    prompt_parts.append("CLASSIFICATION_PATH: L1:<category> | L2:cluster_<id> | L3:<specific_task>")
    prompt_parts.append("\nExamples:")
    prompt_parts.append("- CLASSIFICATION_PATH: L1:business_finance | L2:cluster_234")
    prompt_parts.append("- CLASSIFICATION_PATH: L1:it_systems | L2:cluster_29 | L3:Manage backup, security and user help systems")
    prompt_parts.append("\nIf the tool only fits Level 1 or 2, stop there. Don't force a Level 3 classification.")
    
    return "\n".join(prompt_parts)

# System prompt for hierarchical classification
HIERARCHICAL_SYSTEM_PROMPT = """You are an expert at classifying AI tools using the O*NET occupational task hierarchy.

The hierarchy has 3 levels:
1. Level 1: Broad categories (10 total)
2. Level 2: Task clusters (~400 total, ~40 per category)  
3. Level 3: Specific individual tasks (~20k total)

Your job is to navigate through this hierarchy to find the most appropriate classification for each tool.

IMPORTANT GUIDELINES:
- Start with Level 1 (category)
- Progress to Level 2 (cluster) if it fits
- Only use Level 3 (specific task) for very specific, narrow tools
- Most tools should be classified at Level 2
- Always use the EXACT format specified in the prompt"""

@scorer(metrics=[accuracy()])
def hierarchical_task_scorer() -> Scorer:
    """Score hierarchical classification responses"""
    async def _scorer(state: TaskState, target: Target):
        completion = state.output.completion
        
        if not completion:
            return Score(
                value=0,
                answer=None,
                explanation="No response received"
            )
        
        # Look for the classification path pattern
        import re
        pattern = r'CLASSIFICATION_PATH:\s*(.+)'
        match = re.search(pattern, completion)
        
        if not match:
            return Score(
                value=0,
                answer=completion.strip(),
                explanation="No valid CLASSIFICATION_PATH found in response"
            )
        
        classification_path = match.group(1).strip()
        
        # Validate the path format
        # Should be like: L1:category | L2:cluster_123 | L3:task
        parts = [p.strip() for p in classification_path.split('|')]
        
        valid_format = True
        extracted_levels = {}
        
        for part in parts:
            if ':' not in part:
                valid_format = False
                break
                
            level, value = part.split(':', 1)
            level = level.strip()
            value = value.strip()
            
            if level not in ['L1', 'L2', 'L3']:
                valid_format = False
                break
                
            extracted_levels[level] = value
        
        if not valid_format or 'L1' not in extracted_levels:
            return Score(
                value=0,
                answer=classification_path,
                explanation="Invalid classification path format"
            )
        
        # Valid classification - store the structured result
        return Score(
            value=1,
            answer=json.dumps(extracted_levels),  # Store as JSON for easy parsing
            explanation=f"Valid hierarchical classification with {len(extracted_levels)} levels"
        )
    
    return _scorer

@task
def task_mapping_task():
    """Task 1: Hierarchical O*NET Task Mapping"""
    dataset_file = "conseq_fin_stage4_input.jsonl"
    
    if not Path(dataset_file).exists():
        raise FileNotFoundError(f"Dataset file {dataset_file} not found. Run conseq_fin_stage4_data_prep.py first.")
    
    # Load hierarchy
    hierarchy = load_task_hierarchy()
    
    # Build hierarchical prompt
    hierarchy_prompt = build_hierarchical_prompt(hierarchy)
    
    # Count samples
    with open(dataset_file, 'r') as f:
        sample_count = sum(1 for _ in f)
    
    logger.info(f"Loading {sample_count} tool samples for hierarchical task mapping")
    logger.info(f"Hierarchy: {len(hierarchy['top_level'])} top-level, ~{sum(len(clusters) for clusters in hierarchy['middle_level'].values())} clusters")
    
    # Create the full user prompt template
    user_prompt_template = f"""{hierarchy_prompt}

=== TOOL TO CLASSIFY ===
Server name & description: {{server_name}}: {{server_description}}
Tool name, description and input schema: {{tool_name}} {{tool_description}} {{input_schema}}

Navigate through the hierarchy and provide your classification in the specified format.
Remember: Most tools should be classified at Level 2 (cluster level), not Level 3."""
    
    return Task(
        dataset=json_dataset(dataset_file),
        solver=[
            system_message(HIERARCHICAL_SYSTEM_PROMPT),
            generate()
        ],
        scorer=[hierarchical_task_scorer()],
        message_limit=100
    )
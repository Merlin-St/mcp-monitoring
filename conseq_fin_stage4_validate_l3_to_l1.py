#!/usr/bin/env python3
"""
Validation Task: Level 3 (Task) to Level 1 Cluster Direct Assignment

Tests how accurately an LLM can directly assign individual tasks to their
Level 1 clusters (out of 10 total), bypassing Level 2.

Includes two tasks to test both sets of Level 1 names:
- validate_l3_to_l1_original: Uses original/existing Level 1 names
- validate_l3_to_l1_from_l2: Uses Level 1 names generated from Level 2 clusters

Usage:
    # Test with original names
    inspect eval conseq_fin_stage4_validate_l3_to_l1.py:validate_l3_to_l1_original --model anthropic/claude-sonnet-4-20250514
    
    # Test with names generated from L2
    inspect eval conseq_fin_stage4_validate_l3_to_l1.py:validate_l3_to_l1_from_l2 --model anthropic/claude-sonnet-4-20250514
    
    # Test with contrastive names
    inspect eval conseq_fin_stage4_validate_l3_to_l1.py:validate_l3_to_l1_contrastive --model anthropic/claude-sonnet-4-20250514
    
    # Test with hierarchical names
    inspect eval conseq_fin_stage4_validate_l3_to_l1.py:validate_l3_to_l1_hierarchical --model anthropic/claude-sonnet-4-20250514
    
    # Test with distinctive names
    inspect eval conseq_fin_stage4_validate_l3_to_l1.py:validate_l3_to_l1_distinctive --model anthropic/claude-sonnet-4-20250514
"""

import json
import pandas as pd
from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset, Sample
from inspect_ai.solver import generate, system_message, chain
from inspect_ai.scorer import includes

def load_alternative_l1_names():
    """Load Level 1 names generated from Level 2 clusters"""
    l1_names_file = 'conseq_fin_stage4_level1_names_from_l2.json'
    if Path(l1_names_file).exists():
        with open(l1_names_file, 'r') as f:
            data = json.load(f)
            return data.get('cluster_names', {})
    return {}

def create_dataset_with_alternative_names():
    """Create dataset with alternative Level 1 names"""
    # Load original dataset
    original_samples = []
    with open('conseq_fin_stage4_validation_l3_to_l1.jsonl', 'r') as f:
        for line in f:
            original_samples.append(json.loads(line))
    
    # Load alternative names
    alt_names = load_alternative_l1_names()
    
    # Create new samples with updated options
    new_samples = []
    for sample in original_samples:
        # Parse the input to replace the Level 1 names
        input_text = sample['input']
        
        # Split into parts
        parts = input_text.split('\n\nSelect the Level 1 cluster this task belongs to from the following options:\n')
        if len(parts) == 2:
            task_part = parts[0]
            options_part = parts[1]
            
            # Replace each cluster name in options
            new_options_lines = []
            for line in options_part.split('\n'):
                if line.startswith('- cluster_1_'):
                    # Extract cluster ID
                    cluster_id = line.split(':')[0].replace('- ', '')
                    # Get alternative name
                    alt_name = alt_names.get(cluster_id, line.split(': ', 1)[1] if ': ' in line else "Unknown")
                    new_options_lines.append(f"- {cluster_id}: {alt_name}")
                else:
                    new_options_lines.append(line)
            
            # Reconstruct input
            new_input = task_part + '\n\nSelect the Level 1 cluster this task belongs to from the following options:\n' + '\n'.join(new_options_lines)
            
            # Create new sample
            new_samples.append(Sample(
                input=new_input,
                target=sample['target'],
                metadata=sample['metadata']
            ))
        else:
            # Fallback if parsing fails
            new_samples.append(Sample(
                input=sample['input'],
                target=sample['target'],
                metadata=sample['metadata']
            ))
    
    return new_samples

@task
def validate_l3_to_l1_original():
    """Task to validate Level 3 to Level 1 assignment with original/existing names"""
    
    # Load the prepared validation dataset
    dataset = json_dataset("conseq_fin_stage4_validation_l3_to_l1.jsonl")
    
    # System prompt
    system_prompt = """You are an expert at analyzing occupational tasks and categorizing them into broad categories.

You will be given an individual task and asked to identify which Level 1 (top-level) cluster it belongs to from a list of 10 options.

Level 1 clusters are broad, high-level categories that encompass many related occupations and tasks.
Each option shows the cluster ID and descriptive name in format: "cluster_1_XXX: Cluster Name"

Analyze the task carefully and select the most appropriate high-level category based on:
1. The general domain or sector
2. The type of work being performed
3. The broad occupational field

Respond with ONLY the cluster ID (e.g., "cluster_1_001") of your selection."""
    
    # Create solver chain
    solver = chain(
        system_message(system_prompt),
        generate()
    )
    
    return Task(
        dataset=dataset,
        solver=solver,
        scorer=includes()  # Will check if correct cluster ID is included in response
    )

@task
def validate_l3_to_l1_from_l2():
    """Task to validate Level 3 to Level 1 assignment with names generated from Level 2"""
    
    # Create dataset with alternative names
    dataset = create_dataset_with_alternative_names()
    
    # System prompt (same as original)
    system_prompt = """You are an expert at analyzing occupational tasks and categorizing them into broad categories.

You will be given an individual task and asked to identify which Level 1 (top-level) cluster it belongs to from a list of 10 options.

Level 1 clusters are broad, high-level categories that encompass many related occupations and tasks.
Each option shows the cluster ID and descriptive name in format: "cluster_1_XXX: Cluster Name"

Analyze the task carefully and select the most appropriate high-level category based on:
1. The general domain or sector
2. The type of work being performed
3. The broad occupational field

Respond with ONLY the cluster ID (e.g., "cluster_1_001") of your selection."""
    
    # Create solver chain
    solver = chain(
        system_message(system_prompt),
        generate()
    )
    
    return Task(
        dataset=dataset,
        solver=solver,
        scorer=includes()  # Will check if correct cluster ID is included in response
    )
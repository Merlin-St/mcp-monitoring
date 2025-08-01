#!/usr/bin/env python3
"""
Validation Task: Level 3 (Task) to Level 2 Cluster Assignment

Tests how accurately an LLM can assign individual tasks to their
Level 2 clusters (out of 400 total).

Usage:
    # Test with original names
    inspect eval conseq_fin_stage4_validate_l3_to_l2.py --model anthropic/claude-sonnet-4-20250514
    
    # Test with contrastive names
    inspect eval "conseq_fin_stage4_validate_l3_to_l2.py@validate_l3_to_l2_contrastive" --model anthropic/claude-sonnet-4-20250514
"""

from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.solver import generate, system_message, chain
from inspect_ai.scorer import includes

@task
def validate_l3_to_l2():
    """Task to validate Level 3 to Level 2 cluster assignment accuracy with original names"""
    
    # Load the prepared validation dataset
    dataset = json_dataset("conseq_fin_stage4_validation_l3_to_l2.jsonl")
    
    # System prompt
    system_prompt = """You are an expert at analyzing occupational tasks and categorizing them into appropriate clusters.

You will be given an individual task and asked to identify which Level 2 cluster it belongs to from a provided list of options.

Level 2 clusters are specific groupings of related tasks (400 total clusters).
Each option shows the cluster ID and descriptive name in format: "cluster_2_XXX: Cluster Name"

Analyze the task carefully and select the most appropriate cluster based on:
1. The primary function or activity described
2. The skills and knowledge required
3. The occupational context

Respond with ONLY the cluster ID (e.g., "cluster_2_001") of your selection."""
    
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
def validate_l3_to_l2_contrastive():
    """Task to validate Level 3 to Level 2 cluster assignment with contrastive names"""
    
    # Check if contrastive dataset exists
    contrastive_file = "conseq_fin_stage4_validation_l3_to_l2_contrastive.jsonl"
    if not Path(contrastive_file).exists():
        # Fallback to original
        print(f"Warning: {contrastive_file} not found, using original dataset")
        dataset = json_dataset("conseq_fin_stage4_validation_l3_to_l2.jsonl")
    else:
        dataset = json_dataset(contrastive_file)
    
    # Same system prompt - we're testing the names, not the prompt
    system_prompt = """You are an expert at analyzing occupational tasks and categorizing them into appropriate clusters.

You will be given an individual task and asked to identify which Level 2 cluster it belongs to from a provided list of options.

Level 2 clusters are specific groupings of related tasks (400 total clusters).
Each option shows the cluster ID and descriptive name in format: "cluster_2_XXX: Cluster Name"

Analyze the task carefully and select the most appropriate cluster based on:
1. The primary function or activity described
2. The skills and knowledge required
3. The occupational context

Respond with ONLY the cluster ID (e.g., "cluster_2_001") of your selection."""
    
    solver = chain(
        system_message(system_prompt),
        generate()
    )
    
    return Task(
        dataset=dataset,
        solver=solver,
        scorer=includes()
    )
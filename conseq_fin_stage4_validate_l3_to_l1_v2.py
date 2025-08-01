#!/usr/bin/env python3
"""
Validation Task: Level 3 (Task) to Level 1 Cluster Direct Assignment V2

Enhanced version with support for multiple naming approaches including
contrastive, hierarchical, and distinctive names.

Usage:
    # Test with contrastive names
    inspect eval conseq_fin_stage4_validate_l3_to_l1_v2.py:validate_l3_to_l1_contrastive --model anthropic/claude-sonnet-4-20250514
    
    # Test with hierarchical names
    inspect eval conseq_fin_stage4_validate_l3_to_l1_v2.py:validate_l3_to_l1_hierarchical --model anthropic/claude-sonnet-4-20250514
    
    # Test with distinctive names
    inspect eval conseq_fin_stage4_validate_l3_to_l1_v2.py:validate_l3_to_l1_distinctive --model anthropic/claude-sonnet-4-20250514
"""

import json
from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.solver import generate, system_message, chain
from inspect_ai.scorer import includes

@task
def validate_l3_to_l1_contrastive():
    """Validate with contrastive Level 2 names"""
    dataset_file = "conseq_fin_stage4_validation_l3_to_l1_contrastive.jsonl"
    if not Path(dataset_file).exists():
        # Fallback to original if contrastive not generated yet
        dataset = json_dataset("conseq_fin_stage4_validation_l3_to_l1.jsonl")
    else:
        dataset = json_dataset(dataset_file)
    
    system_prompt = """You are an expert at analyzing occupational tasks and categorizing them into broad categories.

You will be given an individual task and asked to identify which Level 1 (top-level) cluster it belongs to from a list of 10 options.

Level 1 clusters are broad, high-level categories that encompass many related occupations and tasks.
Each option shows the cluster ID and descriptive name in format: "cluster_1_XXX: Cluster Name"

Analyze the task carefully and select the most appropriate high-level category based on:
1. The general domain or sector
2. The type of work being performed
3. The broad occupational field

Respond with ONLY the cluster ID (e.g., "cluster_1_001") of your selection."""
    
    solver = chain(
        system_message(system_prompt),
        generate()
    )
    
    return Task(
        dataset=dataset,
        solver=solver,
        scorer=includes()
    )

@task
def validate_l3_to_l1_hierarchical():
    """Validate with hierarchical Level 1 names"""
    dataset_file = "conseq_fin_stage4_validation_l3_to_l1_hierarchical.jsonl"
    if not Path(dataset_file).exists():
        dataset = json_dataset("conseq_fin_stage4_validation_l3_to_l1.jsonl")
    else:
        dataset = json_dataset(dataset_file)
    
    system_prompt = """You are an expert at analyzing occupational tasks and categorizing them into broad categories.

You will be given an individual task and asked to identify which Level 1 (top-level) cluster it belongs to from a list of 10 options.

Level 1 clusters are broad, high-level categories that encompass many related occupations and tasks.
Each option shows the cluster ID and descriptive name in format: "cluster_1_XXX: Cluster Name"

Analyze the task carefully and select the most appropriate high-level category based on:
1. The general domain or sector
2. The type of work being performed
3. The broad occupational field

Respond with ONLY the cluster ID (e.g., "cluster_1_001") of your selection."""
    
    solver = chain(
        system_message(system_prompt),
        generate()
    )
    
    return Task(
        dataset=dataset,
        solver=solver,
        scorer=includes()
    )

@task
def validate_l3_to_l1_distinctive():
    """Validate with distinctive names from combined approach"""
    dataset_file = "conseq_fin_stage4_validation_l3_to_l1_distinctive.jsonl"
    if not Path(dataset_file).exists():
        dataset = json_dataset("conseq_fin_stage4_validation_l3_to_l1.jsonl")
    else:
        dataset = json_dataset(dataset_file)
    
    system_prompt = """You are an expert at analyzing occupational tasks and categorizing them into broad categories.

You will be given an individual task and asked to identify which Level 1 (top-level) cluster it belongs to from a list of 10 options.

Level 1 clusters are broad, high-level categories that encompass many related occupations and tasks.
Each option shows the cluster ID and descriptive name in format: "cluster_1_XXX: Cluster Name"

Analyze the task carefully and select the most appropriate high-level category based on:
1. The general domain or sector
2. The type of work being performed
3. The broad occupational field

Respond with ONLY the cluster ID (e.g., "cluster_1_001") of your selection."""
    
    solver = chain(
        system_message(system_prompt),
        generate()
    )
    
    return Task(
        dataset=dataset,
        solver=solver,
        scorer=includes()
    )
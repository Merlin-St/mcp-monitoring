#!/usr/bin/env python3
"""
Validation Task: Level 2 to Level 1 Assignment V2

Enhanced version with support for multiple naming approaches including
contrastive Level 2 names and hierarchical Level 1 names.

Usage:
    # Test with contrastive L2 names + original L1
    inspect eval conseq_fin_stage4_validate_l2_to_l1_v2.py:validate_l2_to_l1_contrastive --model anthropic/claude-sonnet-4-20250514
    
    # Test with contrastive L2 + hierarchical L1
    inspect eval conseq_fin_stage4_validate_l2_to_l1_v2.py:validate_l2_to_l1_hierarchical --model anthropic/claude-sonnet-4-20250514
    
    # Test with all distinctive names
    inspect eval conseq_fin_stage4_validate_l2_to_l1_v2.py:validate_l2_to_l1_distinctive --model anthropic/claude-sonnet-4-20250514
"""

import json
from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.solver import generate, system_message, chain
from inspect_ai.scorer import includes

@task
def validate_l2_to_l1_contrastive():
    """Validate with contrastive Level 2 names"""
    dataset_file = "conseq_fin_stage4_validation_l2_to_l1_contrastive.jsonl"
    if not Path(dataset_file).exists():
        dataset = json_dataset("conseq_fin_stage4_validation_l2_to_l1.jsonl")
    else:
        dataset = json_dataset(dataset_file)
    
    system_prompt = """You are an expert at understanding hierarchical categorization systems.

You will be given:
1. A Level 2 cluster name and ID
2. Sample tasks from that cluster
3. A list of all 10 Level 1 (top-level) cluster options

Your task is to identify which Level 1 cluster is the parent of the given Level 2 cluster.

Level 1 clusters are broad, high-level categories that encompass multiple Level 2 clusters.
Each Level 2 cluster belongs to exactly one Level 1 parent cluster.

Analyze the Level 2 cluster name and sample tasks to determine which broad Level 1 category it falls under.

Respond with ONLY the cluster ID (e.g., "cluster_1_001") of the Level 1 parent."""
    
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
def validate_l2_to_l1_hierarchical():
    """Validate with hierarchical Level 1 names and contrastive L2"""
    dataset_file = "conseq_fin_stage4_validation_l2_to_l1_hierarchical.jsonl"
    if not Path(dataset_file).exists():
        dataset = json_dataset("conseq_fin_stage4_validation_l2_to_l1.jsonl")
    else:
        dataset = json_dataset(dataset_file)
    
    system_prompt = """You are an expert at understanding hierarchical categorization systems.

You will be given:
1. A Level 2 cluster name and ID
2. Sample tasks from that cluster
3. A list of all 10 Level 1 (top-level) cluster options

Your task is to identify which Level 1 cluster is the parent of the given Level 2 cluster.

Level 1 clusters are broad, high-level categories that encompass multiple Level 2 clusters.
Each Level 2 cluster belongs to exactly one Level 1 parent cluster.

Analyze the Level 2 cluster name and sample tasks to determine which broad Level 1 category it falls under.

Respond with ONLY the cluster ID (e.g., "cluster_1_001") of the Level 1 parent."""
    
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
def validate_l2_to_l1_distinctive():
    """Validate with all distinctive names"""
    dataset_file = "conseq_fin_stage4_validation_l2_to_l1_distinctive.jsonl"
    if not Path(dataset_file).exists():
        dataset = json_dataset("conseq_fin_stage4_validation_l2_to_l1.jsonl")
    else:
        dataset = json_dataset(dataset_file)
    
    system_prompt = """You are an expert at understanding hierarchical categorization systems.

You will be given:
1. A Level 2 cluster name and ID
2. Sample tasks from that cluster
3. A list of all 10 Level 1 (top-level) cluster options

Your task is to identify which Level 1 cluster is the parent of the given Level 2 cluster.

Level 1 clusters are broad, high-level categories that encompass multiple Level 2 clusters.
Each Level 2 cluster belongs to exactly one Level 1 parent cluster.

Analyze the Level 2 cluster name and sample tasks to determine which broad Level 1 category it falls under.

Respond with ONLY the cluster ID (e.g., "cluster_1_001") of the Level 1 parent."""
    
    solver = chain(
        system_message(system_prompt),
        generate()
    )
    
    return Task(
        dataset=dataset,
        solver=solver,
        scorer=includes()
    )
#!/usr/bin/env python3
"""
Validation Task: Level 2 Cluster to Level 1 Parent Assignment

Tests how accurately an LLM can identify the Level 1 parent cluster
for a given Level 2 cluster.

Includes two tasks to test both sets of Level 1 names:
- validate_l2_to_l1_original: Uses original/existing Level 1 names
- validate_l2_to_l1_from_l2: Uses Level 1 names generated from Level 2 clusters

Usage:
    # Test with original names
    inspect eval "conseq_fin_stage4_validate_l2_to_l1.py@validate_l2_to_l1_original" --model anthropic/claude-sonnet-4-20250514
    
    # Test with names generated from L2
    inspect eval "conseq_fin_stage4_validate_l2_to_l1.py@validate_l2_to_l1_from_l2" --model anthropic/claude-sonnet-4-20250514
"""

import json
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

def create_l2_l1_dataset_with_alternative_names():
    """Create dataset with alternative Level 1 names for L2->L1 validation"""
    # Load original dataset
    original_samples = []
    with open('conseq_fin_stage4_validation_l2_to_l1.jsonl', 'r') as f:
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
        parts = input_text.split('\n\nSelect the Level 1 parent cluster from the following options:\n')
        if len(parts) == 2:
            cluster_info_part = parts[0]
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
            new_input = cluster_info_part + '\n\nSelect the Level 1 parent cluster from the following options:\n' + '\n'.join(new_options_lines)
            
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
def validate_l2_to_l1_original():
    """Task to validate Level 2 to Level 1 parent assignment with original names"""
    
    # Load the prepared validation dataset
    dataset = json_dataset("conseq_fin_stage4_validation_l2_to_l1.jsonl")
    
    # System prompt
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
def validate_l2_to_l1_from_l2():
    """Task to validate Level 2 to Level 1 parent assignment with names from L2"""
    
    # Create dataset with alternative names
    dataset = create_l2_l1_dataset_with_alternative_names()
    
    # System prompt (same as original)
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
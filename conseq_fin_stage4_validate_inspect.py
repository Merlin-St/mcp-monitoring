#!/usr/bin/env python3
"""
Unified Stage 4 Validation Script - All Validation Tasks

Available Tasks (grouped by hierarchy type):

    # Original 10-cluster hierarchy
    inspect eval conseq_fin_stage4_validate_inspect.py@validate_l3_to_l1_original --model anthropic/claude-sonnet-4-20250514
    inspect eval conseq_fin_stage4_validate_inspect.py@validate_l2_to_l1_original --model anthropic/claude-sonnet-4-20250514
    inspect eval conseq_fin_stage4_validate_inspect.py@validate_l3_to_l2_original --model anthropic/claude-sonnet-4-20250514
    
    # K10 with L2-generated names
    inspect eval conseq_fin_stage4_validate_inspect.py@validate_l3_to_l1_k10_from_l2 --model anthropic/claude-sonnet-4-20250514
    inspect eval conseq_fin_stage4_validate_inspect.py@validate_l2_to_l1_k10_from_l2 --model anthropic/claude-sonnet-4-20250514
    
    # K12 hierarchy
    inspect eval conseq_fin_stage4_validate_inspect.py@validate_l3_to_l1_k12 --model anthropic/claude-sonnet-4-20250514
    inspect eval conseq_fin_stage4_validate_inspect.py@validate_l2_to_l1_k12 --model anthropic/claude-sonnet-4-20250514
"""

import json
from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset, Sample
from inspect_ai.solver import generate, system_message, chain
from inspect_ai.scorer import includes

# =============================================================================
# Utility Functions
# =============================================================================

def load_cluster_names(hierarchy_type="original"):
    """Load cluster names from data files"""
    if hierarchy_type == "k12":
        k12_file = 'conseq_fin_stage4_hierarchy_k12_names_summary.json'
        if Path(k12_file).exists():
            with open(k12_file, 'r') as f:
                return json.load(f).get('supercluster_names', {})
    
    elif hierarchy_type == "from_l2":
        l1_names_file = 'conseq_fin_stage4_level1_names_from_l2.json'
        if Path(l1_names_file).exists():
            with open(l1_names_file, 'r') as f:
                return json.load(f).get('cluster_names', {})
    
    else:  # original - extract from validation dataset
        validation_file = 'conseq_fin_stage4_validation_l3_to_l1.jsonl'
        if Path(validation_file).exists():
            with open(validation_file, 'r') as f:
                sample = json.loads(f.readline())
                input_text = sample['input']
                options_start = input_text.find('from the following options:\n')
                if options_start != -1:
                    options_text = input_text[options_start + len('from the following options:\n'):]
                    cluster_names = {}
                    for line in options_text.split('\n'):
                        if line.startswith('- cluster_1_'):
                            parts = line.split(': ', 1)
                            if len(parts) == 2:
                                cluster_names[parts[0].replace('- ', '')] = parts[1]
                    return cluster_names
    return {}

def create_system_prompt():
    """Single consistent system prompt template"""
    return """The following is a description of an occupational task: [insert task name and description (either onet tasks for l3->l1/l3->l2 validation or l2 cluster names for l2->l1 validation)]. Consider the following list of classification options: [insert fitting cluster names, either l2 for l3->l2 validation, or l1 names for l3->l1 or l2->l1 validation]. Your job is to identify which option best describes the occupational task. What is the answer? You MUST provide an option exactly as written above. If multiple options apply, choose the single-most pertinent one. Respond ONLY with the cluster ID (e.g. L1_06 or similar)."""

def create_dataset_with_alternative_names(dataset_file, split_text, hierarchy_type="from_l2"):
    """Replace cluster names in dataset with alternative names"""
    original_samples = []
    with open(dataset_file, 'r') as f:
        for line in f:
            original_samples.append(json.loads(line))
    
    alt_names = load_cluster_names(hierarchy_type)
    new_samples = []
    
    for sample in original_samples:
        parts = sample['input'].split(split_text)
        if len(parts) == 2:
            task_part, options_part = parts
            new_options_lines = []
            for line in options_part.split('\n'):
                if line.startswith('- cluster_1_'):
                    cluster_id = line.split(':')[0].replace('- ', '')
                    alt_name = alt_names.get(cluster_id, line.split(': ', 1)[1] if ': ' in line else "Unknown")
                    new_options_lines.append(f"- {cluster_id}: {alt_name}")
                else:
                    new_options_lines.append(line)
            new_input = task_part + split_text + '\n'.join(new_options_lines)
            new_samples.append(Sample(input=new_input, target=sample['target'], metadata=sample['metadata']))
        else:
            new_samples.append(Sample(input=sample['input'], target=sample['target'], metadata=sample['metadata']))
    
    return new_samples

# =============================================================================
# Validation Tasks (grouped by hierarchy type)
# =============================================================================

# Original 10-cluster hierarchy
@task
def validate_l3_to_l1_original():
    """L3→L1 validation with original cluster names"""
    return Task(
        dataset=json_dataset("conseq_fin_stage4_validation_l3_to_l1.jsonl"),
        solver=chain(system_message(create_system_prompt()), generate()),
        scorer=includes()
    )

@task
def validate_l2_to_l1_original():
    """L2→L1 validation with original cluster names"""
    return Task(
        dataset=json_dataset("conseq_fin_stage4_validation_l2_to_l1.jsonl"),
        solver=chain(system_message(create_system_prompt()), generate()),
        scorer=includes()
    )

@task
def validate_l3_to_l2_original():
    """L3→L2 validation with original names"""
    return Task(
        dataset=json_dataset("conseq_fin_stage4_validation_l3_to_l2.jsonl"),
        solver=chain(system_message(create_system_prompt()), generate()),
        scorer=includes()
    )

# K10 with L2-generated names
@task
def validate_l3_to_l1_k10_from_l2():
    """L3→L1 validation with K10 names generated from L2"""
    dataset = create_dataset_with_alternative_names(
        'conseq_fin_stage4_validation_l3_to_l1.jsonl',
        '\n\nSelect the Level 1 cluster this task belongs to from the following options:\n'
    )
    return Task(
        dataset=dataset,
        solver=chain(system_message(create_system_prompt()), generate()),
        scorer=includes()
    )

@task
def validate_l2_to_l1_k10_from_l2():
    """L2→L1 validation with K10 names generated from L2"""
    dataset = create_dataset_with_alternative_names(
        'conseq_fin_stage4_validation_l2_to_l1.jsonl',
        '\n\nSelect the Level 1 parent cluster from the following options:\n'
    )
    return Task(
        dataset=dataset,
        solver=chain(system_message(create_system_prompt()), generate()),
        scorer=includes()
    )

# K12 hierarchy
@task
def validate_l3_to_l1_k12():
    """L3→L1 validation using K=12 hierarchy"""
    return Task(
        dataset=json_dataset("conseq_fin_stage4_validation_l3_to_l1_k12.jsonl"),
        solver=chain(system_message(create_system_prompt()), generate()),
        scorer=includes()
    )

@task
def validate_l2_to_l1_k12():
    """L2→L1 validation using K=12 hierarchy"""
    return Task(
        dataset=json_dataset("conseq_fin_stage4_validation_l2_to_l1_k12.jsonl"),
        solver=chain(system_message(create_system_prompt()), generate()),
        scorer=includes()
    )

if __name__ == "__main__":
    # Run via: inspect eval conseq_fin_stage4_validate_inspect.py@validate_l3_to_l1_original --model anthropic/claude-sonnet-4-20250514
    pass
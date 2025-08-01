#!/usr/bin/env python3
"""
Stage 4 L2 → L1 Validation K=12 - New K=12 Hierarchy

Test the new K=12 Level 1 supercluster hierarchy accuracy.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import includes
from inspect_ai.solver import generate, system_message

def create_l2_to_l1_system_prompt():
    """Create system prompt for L2 → L1 validation using K=12 hierarchy."""
    return """You are an expert at categorizing occupational tasks. You will be given a Level 2 cluster containing related tasks and asked to select the correct Level 1 parent supercluster.

The Level 1 superclusters (K=12) are:
- L1_00: Academic Education and Teaching
- L1_01: Management and Professional Services
- L1_02: Equipment Operations and Maintenance
- L1_03: Customer-Facing Business Operations Management
- L1_04: Installation and Systems Construction
- L1_05: Data and Records Management
- L1_06: Healthcare and Wellness Services
- L1_07: Manufacturing and Design Engineering
- L1_08: Environmental and Natural Resources Management
- L1_09: Media and Communications Services
- L1_10: Testing and Quality Assurance
- L1_11: Operations Management and Service Delivery

Analyze the tasks carefully and select the most appropriate Level 1 supercluster. Return only the Level 1 ID (e.g., "L1_06")."""

def create_l2_to_l1_prompt(sample):
    """Create prompt for L2 → L1 classification using K=12 hierarchy."""
    l2_cluster_id = sample.metadata["l2_cluster_id"]
    l2_cluster_name = sample.metadata["l2_cluster_name"]
    
    # Extract task examples from the input
    input_text = sample.input
    tasks_start = input_text.find("Sample tasks from this cluster:")
    tasks_end = input_text.find("Select the Level 1 parent supercluster")
    
    if tasks_start != -1 and tasks_end != -1:
        tasks_section = input_text[tasks_start:tasks_end].strip()
    else:
        tasks_section = "No task examples available."
    
    prompt = f"""Level 2 Cluster: {l2_cluster_id} - {l2_cluster_name}

{tasks_section}

Select the correct Level 1 supercluster from the options above. Consider the overall functional category and occupational domain that best encompasses these tasks."""
    
    return prompt

@task
def validate_l2_to_l1_k12():
    """Validate L2 → L1 classification accuracy using K=12 hierarchy."""
    dataset = json_dataset("conseq_fin_stage4_validation_l2_to_l1_k12.jsonl")
    
    return Task(
        dataset=dataset,
        plan=[
            system_message(create_l2_to_l1_system_prompt()),
            generate(create_l2_to_l1_prompt)
        ],
        scorer=includes()
    )

if __name__ == "__main__":
    # Run with: inspect eval conseq_fin_stage4_validate_l2_to_l1_k12.py --model anthropic/claude-sonnet-4-20250514
    pass
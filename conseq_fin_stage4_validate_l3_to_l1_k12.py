#!/usr/bin/env python3
"""
Stage 4 L3 → L1 Validation K=12 - New K=12 Hierarchy

Test direct task-to-supercluster classification using the K=12 Level 1 hierarchy.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import includes
from inspect_ai.solver import generate, system_message

def create_l3_to_l1_system_prompt():
    """Create system prompt for L3 → L1 validation using K=12 hierarchy."""
    return """You are an expert at categorizing occupational tasks. You will be given a specific task and asked to select which broad Level 1 supercluster it belongs to.

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

Analyze the task carefully and select the most appropriate Level 1 supercluster. Return only the Level 1 ID (e.g., "L1_06")."""

def create_l3_to_l1_prompt(sample):
    """Create prompt for L3 → L1 classification using K=12 hierarchy."""
    task_description = sample.input.split("\n")[0].replace("Task: ", "")
    
    prompt = f"""Task: {task_description}

Select the correct Level 1 supercluster from the options above. Consider the overall functional category and occupational domain that best encompasses this specific task."""
    
    return prompt

@task
def validate_l3_to_l1_k12():
    """Validate L3 → L1 classification accuracy using K=12 hierarchy."""
    dataset = json_dataset("conseq_fin_stage4_validation_l3_to_l1_k12.jsonl")
    
    return Task(
        dataset=dataset,
        plan=[
            system_message(create_l3_to_l1_system_prompt()),
            generate(create_l3_to_l1_prompt)
        ],
        scorer=includes()
    )

if __name__ == "__main__":
    # Run with: inspect eval conseq_fin_stage4_validate_l3_to_l1_k12.py --model anthropic/claude-sonnet-4-20250514
    pass
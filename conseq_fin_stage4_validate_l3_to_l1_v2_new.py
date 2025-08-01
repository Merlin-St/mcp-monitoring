#!/usr/bin/env python3
"""
Stage 4 L3 → L1 Validation v2 - New K=20 Hierarchy

Test direct task-to-supercluster classification using the new K=20 Level 1 hierarchy.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import includes
from inspect_ai.solver import generate, system_message

def create_l3_to_l1_system_prompt():
    """Create system prompt for L3 → L1 validation using new hierarchy."""
    return """You are an expert at categorizing occupational tasks. You will be given a specific task and asked to select which broad Level 1 supercluster it belongs to.

The Level 1 superclusters (K=20) are:
- L1_00: Educational Services and Student Support
- L1_01: Equipment Operations and Maintenance  
- L1_02: Technology Systems Development and Management
- L1_03: Financial Analysis and Management
- L1_04: Service and Care Operations
- L1_05: Research and Development Engineering
- L1_06: Manufacturing Operations and Machine Control
- L1_07: Academic and Educational Leadership
- L1_08: Manufacturing and Logistics Operations
- L1_09: Entertainment and Marketing Management
- L1_10: Healthcare and Medical Services
- L1_11: Customer Service and Operations Management
- L1_12: Manufacturing and Craft Production Operations
- L1_13: Installation and Technical Systems
- L1_14: Project and Team Coordination Management
- L1_15: Records and Documentation Management
- L1_16: Natural Resources and Environmental Operations
- L1_17: Safety and Security Operations Management
- L1_18: Environmental Compliance and Sustainability Management
- L1_19: Quality Testing and Inspection Services

Analyze the task carefully and select the most appropriate Level 1 supercluster. Return only the Level 1 ID (e.g., "L1_10")."""

def create_l3_to_l1_prompt(sample):
    """Create prompt for L3 → L1 classification using new hierarchy."""
    task_description = sample.input.split("\n")[0].replace("Task: ", "")
    
    prompt = f"""Task: {task_description}

Select the correct Level 1 supercluster from the options above. Consider the overall functional category and occupational domain that best encompasses this specific task."""
    
    return prompt

@task
def validate_l3_to_l1_v2_new():
    """Validate L3 → L1 classification accuracy using new K=20 hierarchy."""
    dataset = json_dataset("conseq_fin_stage4_validation_l3_to_l1_v2_new.jsonl")
    
    return Task(
        dataset=dataset,
        plan=[
            system_message(create_l3_to_l1_system_prompt()),
            generate(create_l3_to_l1_prompt)
        ],
        scorer=includes()
    )

if __name__ == "__main__":
    # Run with: inspect eval conseq_fin_stage4_validate_l3_to_l1_v2_new.py --model anthropic/claude-sonnet-4-20250514
    pass
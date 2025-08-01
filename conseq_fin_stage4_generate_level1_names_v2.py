#!/usr/bin/env python3
"""
Generate Level 1 Supercluster Names v2 - Inspect Framework

Simple Inspect script to generate names for the new K=20 Level 1 superclusters
based on their Level 2 cluster memberships.
"""

import json
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate

def load_hierarchy_data():
    """Load the new Level 1 hierarchy data."""
    with open('conseq_fin_stage4_hierarchy_k12.json', 'r') as f:
        hierarchy = json.load(f)
    return hierarchy

def create_naming_dataset():
    """Create dataset for Level 1 supercluster naming."""
    hierarchy = load_hierarchy_data()
    
    # Group Level 2 clusters by Level 1 assignment
    level1_groups = {}
    for cluster_id, mapping in hierarchy["level2_to_level1_mapping"].items():
        l1_id = mapping["level1_id"]
        if l1_id not in level1_groups:
            level1_groups[l1_id] = []
        level1_groups[l1_id].append({
            "cluster_id": cluster_id,
            "basic_name": mapping["basic_name"],
            "contrastive_name": mapping["contrastive_name"]
        })
    
    samples = []
    for l1_id, clusters in sorted(level1_groups.items()):
        # Create combined text of all Level 2 cluster names
        basic_names = [c["basic_name"] for c in clusters]
        contrastive_names = [c["contrastive_name"] for c in clusters]
        
        input_text = f"""Level 1 Supercluster: {l1_id}
Contains {len(clusters)} Level 2 clusters:

BASIC LEVEL 2 CLUSTER NAMES:
{chr(10).join(f"• {name}" for name in basic_names)}

CONTRASTIVE LEVEL 2 CLUSTER NAMES:
{chr(10).join(f"• {name}" for name in contrastive_names)}

Create a concise, descriptive Level 1 supercluster name (3-6 words) that captures the common theme across these Level 2 clusters. Focus on the broader occupational or functional category that unifies them.

Return only the supercluster name, nothing else."""

        samples.append(Sample(
            input=input_text,
            target="",  # We'll extract the name from the response
            id=l1_id,
            metadata={
                "level1_id": l1_id,
                "n_clusters": len(clusters),
                "cluster_ids": [c["cluster_id"] for c in clusters]
            }
        ))
    
    return samples

@task
def generate_level1_names():
    """Task to generate Level 1 supercluster names."""
    dataset = create_naming_dataset()
    
    return Task(
        dataset=dataset,
        plan=[
            generate()
        ]
    )

if __name__ == "__main__":
    # Can be run with: inspect eval conseq_fin_stage4_generate_level1_names_v2.py --model anthropic/claude-sonnet-4-20250514
    pass
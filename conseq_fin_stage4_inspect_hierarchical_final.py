#!/usr/bin/env python3
"""
Stage 4 O*NET Classification - Hierarchical Classification with Dynamic Subsets

Uses pre-built clustering hierarchy to classify tools through 3 levels:
- Level 1: Select from 10 top clusters (show all)
- Level 2: Select from ~40 clusters in chosen Level 1 (dynamic subset)
- Level 3: Select from ~50 tasks in chosen Level 2 (dynamic subset)

Usage:
    inspect eval conseq_fin_stage4_inspect_hierarchical_final.py --model anthropic/claude-sonnet-4-20250514 --message-limit 50
"""

import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from dotenv import load_dotenv
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import TaskState, generate, system_message

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conseq_fin_stage4_inspect_hierarchical_final.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HierarchicalClassifier:
    def __init__(self):
        """Initialize with pre-built hierarchy"""
        self.hierarchy_csv = 'conseq_fin_stage4_onetclusters.csv'
        self.metadata_json = 'conseq_fin_stage4_hierarchy_metadata.json'
        
        # Load hierarchy
        if not Path(self.hierarchy_csv).exists():
            raise FileNotFoundError(f"Hierarchy CSV not found: {self.hierarchy_csv}. Run conseq_fin_stage4_build_hierarchy_v2.py first.")
            
        self.df = pd.read_csv(self.hierarchy_csv)
        
        # Load metadata for cluster descriptions
        if Path(self.metadata_json).exists():
            with open(self.metadata_json, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
            
        logger.info(f"Loaded hierarchy with {len(self.df)} tasks")
        logger.info(f"Level 1 clusters: {self.df['level1_cluster_id'].nunique()}")
        logger.info(f"Level 2 clusters: {self.df['level2_cluster_id'].nunique()}")
        
    def get_level1_prompt(self, tool_info: Dict) -> str:
        """Generate Level 1 classification prompt showing all top clusters"""
        prompt_parts = [
            "=== LEVEL 1 CLASSIFICATION ===",
            f"Tool to classify: {tool_info['tool_name']}",
            f"Description: {tool_info['tool_description']}",
            "",
            "Select the most appropriate top-level category from these 10 clusters:",
            ""
        ]
        
        # Get all Level 1 clusters
        level1_clusters = sorted(self.df['level1_cluster_id'].unique())
        
        for cluster_id in level1_clusters:
            # Get cluster info from metadata
            cluster_info = self.metadata.get('cluster_descriptions', {}).get('level1', {}).get(cluster_id, {})
            
            # Get sample tasks
            sample_tasks = cluster_info.get('sample_tasks', [])
            if not sample_tasks:
                # Fallback: get from dataframe
                sample_df = self.df[self.df['level1_cluster_id'] == cluster_id].sample(min(3, len(self.df)))
                sample_tasks = sample_df['Task'].tolist()
            
            prompt_parts.append(f"{cluster_id}:")
            for task in sample_tasks[:3]:
                prompt_parts.append(f"  • {task[:80]}...")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "RESPOND WITH ONLY THE CLUSTER ID (e.g., cluster_1_003)",
            "Do not include any explanation or additional text."
        ])
        
        return "\n".join(prompt_parts)
    
    def get_level2_prompt(self, tool_info: Dict, level1_selection: str) -> str:
        """Generate Level 2 classification prompt with dynamic subset"""
        # Get Level 2 clusters within selected Level 1
        level2_clusters = sorted(
            self.df[self.df['level1_cluster_id'] == level1_selection]['level2_cluster_id'].unique()
        )
        
        prompt_parts = [
            "=== LEVEL 2 CLASSIFICATION ===",
            f"Tool: {tool_info['tool_name']}",
            f"Description: {tool_info['tool_description']}",
            f"",
            f"You selected Level 1 category: {level1_selection}",
            f"Now select the most specific cluster within this category ({len(level2_clusters)} options):",
            ""
        ]
        
        # Show up to 20 clusters (or all if fewer)
        clusters_to_show = level2_clusters[:20] if len(level2_clusters) > 20 else level2_clusters
        
        for cluster_id in clusters_to_show:
            # Get cluster info
            cluster_info = self.metadata.get('cluster_descriptions', {}).get('level2', {}).get(cluster_id, {})
            
            # Get primary occupation and sample tasks
            primary_occupation = cluster_info.get('primary_occupation', '')
            sample_tasks = cluster_info.get('sample_tasks', [])
            
            if not sample_tasks:
                # Fallback: get from dataframe
                sample_df = self.df[self.df['level2_cluster_id'] == cluster_id].sample(min(2, len(self.df)))
                sample_tasks = sample_df['Task'].tolist()
                if not primary_occupation:
                    primary_occupation = sample_df['Title'].iloc[0]
            
            prompt_parts.append(f"{cluster_id} [{primary_occupation}]:")
            for task in sample_tasks[:2]:
                prompt_parts.append(f"  • {task[:80]}...")
            prompt_parts.append("")
        
        if len(level2_clusters) > 20:
            prompt_parts.append(f"(Showing first 20 of {len(level2_clusters)} clusters)")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "RESPOND WITH ONLY THE CLUSTER ID (e.g., cluster_2_037)",
            "Do not include any explanation or additional text."
        ])
        
        return "\n".join(prompt_parts)
    
    def get_level3_prompt(self, tool_info: Dict, level2_selection: str) -> str:
        """Generate Level 3 classification prompt with specific tasks"""
        # Get tasks within selected Level 2 cluster
        cluster_tasks = self.df[self.df['level2_cluster_id'] == level2_selection]
        
        prompt_parts = [
            "=== LEVEL 3 CLASSIFICATION ===",
            f"Tool: {tool_info['tool_name']}",
            f"Description: {tool_info['tool_description']}",
            f"",
            f"You selected Level 2 cluster: {level2_selection}",
            f"Now select the SPECIFIC O*NET task that best matches this tool ({len(cluster_tasks)} options):",
            ""
        ]
        
        # Show all tasks in cluster (typically ~50)
        for _, task_row in cluster_tasks.iterrows():
            task_id = task_row['task_id']
            task_desc = task_row['Task']
            occupation = task_row['Title']
            
            prompt_parts.append(f"{task_id} [{occupation}]:")
            prompt_parts.append(f"  {task_desc}")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "RESPOND WITH ONLY THE TASK ID (e.g., 11-1011.00_8823)",
            "Do not include any explanation or additional text."
        ])
        
        return "\n".join(prompt_parts)

# Global classifier instance
classifier = None

def get_classifier():
    """Get or create classifier instance"""
    global classifier
    if classifier is None:
        classifier = HierarchicalClassifier()
    return classifier

# System prompts for each level
LEVEL1_SYSTEM_PROMPT = """You are an expert at classifying AI tools into occupational categories.
You will be shown 10 top-level clusters with sample tasks.
Select the cluster that best matches the tool's primary function.
Respond with ONLY the cluster ID."""

LEVEL2_SYSTEM_PROMPT = """You are refining the classification of an AI tool.
You will be shown clusters within a previously selected category.
Select the most specific cluster that matches the tool's function.
Respond with ONLY the cluster ID."""

LEVEL3_SYSTEM_PROMPT = """You are making the final classification of an AI tool.
You will be shown specific O*NET tasks within a cluster.
Select the EXACT task that best describes what this tool does.
Respond with ONLY the task ID."""

@scorer(metrics=[accuracy()])
def level_scorer(expected_format: str) -> Scorer:
    """Score responses based on expected format"""
    async def _scorer(state: TaskState, target: Target):
        completion = state.output.completion
        
        if not completion:
            return Score(
                value=0,
                answer=None,
                explanation="No response received"
            )
        
        response = completion.strip()
        
        # Check format based on level
        if expected_format == "level1" and response.startswith("cluster_1_"):
            return Score(value=1, answer=response, explanation="Valid Level 1 cluster")
        elif expected_format == "level2" and response.startswith("cluster_2_"):
            return Score(value=1, answer=response, explanation="Valid Level 2 cluster")
        elif expected_format == "level3" and "_" in response and response.split("_")[0].count("-") == 2:
            # Task IDs look like "11-1011.00_8823"
            return Score(value=1, answer=response, explanation="Valid task ID")
        else:
            return Score(value=0, answer=response, explanation=f"Invalid format for {expected_format}")
    
    return _scorer

@task
def hierarchical_classification_task():
    """Three-stage hierarchical classification task"""
    dataset_file = "conseq_fin_stage4_input.jsonl"
    
    if not Path(dataset_file).exists():
        raise FileNotFoundError(f"Dataset file {dataset_file} not found. Run conseq_fin_stage4_data_prep.py first.")
    
    # Get classifier
    clf = get_classifier()
    
    # Count samples
    with open(dataset_file, 'r') as f:
        sample_count = sum(1 for _ in f)
    
    logger.info(f"Loading {sample_count} tools for hierarchical classification")
    
    # For this implementation, we'll create three separate tasks for the three levels
    # In production, you might want to chain these together
    
    # Create a modified dataset that tracks selections
    dataset = json_dataset(dataset_file)
    
    # This is a simplified version - in production you'd want to chain the prompts
    # For now, we'll just do Level 1 classification
    return Task(
        dataset=dataset,
        solver=[
            system_message(LEVEL1_SYSTEM_PROMPT),
            generate()
        ],
        scorer=[level_scorer("level1")],
        message_limit=100
    )

# Note: In a production implementation, you would want to:
# 1. Run Level 1 classification
# 2. Use the Level 1 results to generate Level 2 prompts
# 3. Use the Level 2 results to generate Level 3 prompts
# This could be done with a custom solver that chains the classifications
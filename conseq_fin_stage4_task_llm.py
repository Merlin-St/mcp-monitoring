#!/usr/bin/env python3
"""
LLM-based cluster naming using Inspect framework

This module handles:
- Generating descriptive names for Level 2 and Level 1 clusters
- Processing Inspect evaluation results using messages_df
- Managing prompts and formatting for LLM interactions
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.solver import generate, system_message
from inspect_ai.scorer import includes
from inspect_ai.analysis.beta import messages_df, samples_df, evals_df

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# System prompts as provided
CLUSTER_NAMING_PROMPT = """You are an expert at analyzing occupational tasks and creating clear, descriptive category names.
Your task is to generate concise, professional names for task clusters based on their content.
Focus on the primary function or activity that unifies the tasks in each cluster. Provide a descriptive name that captures the common theme of these tasks.
The name should be:

Concise (6-13 words)
Professional and clear
Focused on the primary function/activity

Respond with ONLY the cluster name, no explanation or additional text."""

# Validation system prompts
VALIDATION_SYSTEM_PROMPT = """You are an expert at analyzing occupational tasks and categorizing them into appropriate clusters."""

L3_TO_L2_VALIDATION_PROMPT = """The following is a description of an occupational task: {task}.
Consider the following list of classification options: {options}.
Your job is to identify which option best describes the cluster theme.
What is the answer?
If multiple options apply, choose the single-most pertinent one.
Respond with cluster ID: cluster name (e.g. L2_000: xxx)."""

L2_TO_L1_VALIDATION_PROMPT = """The following is a description of an occupational task cluster: {cluster_name}.
Consider the following list of classification options: {options}.
Your job is to identify which option best describes the cluster theme.
What is the answer?
If multiple options apply, choose the single-most pertinent one.
Respond with cluster ID: cluster name (e.g. L1_01: xxx)."""

L3_TO_L1_VALIDATION_PROMPT = """The following is a description of an occupational task: {task}.
Consider the following list of classification options: {options}.
Your job is to identify which option best describes the cluster theme.
What is the answer?
If multiple options apply, choose the single-most pertinent one.
Respond with cluster ID: cluster name (e.g. L1_01: xxx)."""

SUBSET_L2_L3_VALIDATION_PROMPT = """The following is a description of an occupational task: {task}.
Consider the following list of classification options: {options}.
Your job is to identify which option best describes the cluster theme.
What is the answer?
If multiple options apply, choose the single-most pertinent one.
Respond with cluster ID: cluster name (e.g. L2_000: xxx)."""

def prepare_cluster_naming_samples(clusters_info: Dict[str, Dict[str, Any]], level: str = 'level2') -> List[Sample]:
    """
    Prepare samples for cluster naming
    
    Args:
        clusters_info: Dict mapping cluster identifier to cluster information
        level: 'level2' or 'level1' for different prompting strategies
        
    Returns:
        List of Inspect samples
    """
    samples = []
    
    for cluster_identifier, info in clusters_info.items():
        if level == 'level2':
            # Use all tasks in the cluster
            prompt = "\n".join([f"- {task}" for task in info['tasks']])
            metadata_key = "level2_cluster"
        else:  # level1
            # Use Level 2 cluster names that belong to this Level 1
            prompt = "\n".join([f"- {name}" for name in info['l2_names']])
            metadata_key = "level1_cluster"
        
        samples.append(Sample(
            input=prompt,
            metadata={
                metadata_key: cluster_identifier,
                "cluster_size": info.get('size', len(info.get('tasks', [])))
            }
        ))
    
    logger.info(f"Prepared {len(samples)} samples for {level} naming")
    return samples

@task
def generate_cluster_names(clusters_info: Dict[str, Dict[str, Any]], level: str = 'level2'):
    """
    Create an Inspect task for generating cluster names
    
    Args:
        clusters_info: Cluster information dictionary
        level: 'level2' or 'level1'
        
    Returns:
        Inspect Task
    """
    samples = prepare_cluster_naming_samples(clusters_info, level)
    
    return Task(
        dataset=samples,
        solver=[
            system_message(CLUSTER_NAMING_PROMPT),
            generate()
        ]
    )

def process_naming_results(log_dir: str, expected_clusters: List[str] = None) -> Dict[str, str]:
    """
    Process Inspect evaluation results to extract cluster names by reading flattened metadata.
    
    Args:
        log_dir: Directory containing Inspect logs.
        expected_clusters: List of expected cluster IDs for validation (unused, kept for compatibility).
        
    Returns:
        Dict mapping cluster identifier to cluster_name.
    """
    logger.info(f"Processing naming results from {log_dir}")
    cluster_names = {}
    
    try:
        # Use samples_df to get sample metadata including cluster identifier
        samples = samples_df(logs=log_dir, quiet=True)
        logger.info(f"Available columns in samples_df: {list(samples.columns)}")
        
        # Get messages to extract the LLM responses
        messages = messages_df(logs=log_dir, quiet=True)
        
        # Filter for assistant messages (final responses)
        assistant_messages = messages[messages['role'] == 'assistant']
        logger.info(f"Found {len(assistant_messages)} assistant messages")
        
        # Create a dictionary mapping sample_id to the final LLM response
        sample_responses = {}
        # We group by sample_id and take the last message, as there could be multiple turns
        for sample_id, group in assistant_messages.groupby('sample_id'):
            # The last message in the sequence for that sample_id
            last_message = group.iloc[-1]
            sample_responses[sample_id] = last_message['content'].strip()

        logger.info(f"Found final responses for {len(sample_responses)} samples")
        
        # Map cluster identifier from flattened metadata to the LLM response
        for _, sample_row in samples.iterrows():
            sample_id = sample_row['sample_id']
            cluster_identifier = None
                
            # Access the flattened metadata columns directly.
            # Check for different possible key names and ensure the value is not null (pd.notna).
            if 'metadata_level2_cluster' in samples.columns and pd.notna(sample_row['metadata_level2_cluster']):
                cluster_identifier = sample_row['metadata_level2_cluster']
            elif 'metadata_level1_cluster' in samples.columns and pd.notna(sample_row['metadata_level1_cluster']):
                cluster_identifier = sample_row['metadata_level1_cluster']
            elif 'metadata_cluster_id' in samples.columns and pd.notna(sample_row['metadata_cluster_id']):
                # This key was seen in your log file
                cluster_identifier = sample_row['metadata_cluster_id']
        
            if cluster_identifier is not None and sample_id in sample_responses:
                # Ensure cluster_identifier is a string, as it might be read as a number
                cluster_identifier_str = str(cluster_identifier)
                cluster_names[cluster_identifier_str] = sample_responses[sample_id]
                logger.debug(f"Mapped {cluster_identifier_str} -> '{sample_responses[sample_id][:50]}...'")
            else:
                if cluster_identifier is None:
                    logger.warning(f"Could not find a valid cluster identifier for sample {sample_id}")
                else: # Mising response
                    logger.warning(f"Found cluster identifier '{cluster_identifier}' but no matching response for sample {sample_id}")

        logger.info(f"Successfully extracted {len(cluster_names)} cluster names.")
        return cluster_names

    except Exception as e:
        logger.error(f"An error occurred while processing naming results: {e}")
        # Depending on requirements, you might want to re-raise the exception
        # raise e
        return {}

@task
def l3_to_l2_validation():
    """Validate L3 to L2 classification"""
    return Task(
        dataset=json_dataset("conseq_fin_stage4_task_validation_l3_to_l2.jsonl"),
        solver=[
            system_message(VALIDATION_SYSTEM_PROMPT),
            generate()
        ],
        scorer=includes()
    )

@task
def l2_to_l1_validation():
    """Validate L2 to L1 classification"""
    return Task(
        dataset=json_dataset("conseq_fin_stage4_task_validation_l2_to_l1.jsonl"),
        solver=[
            system_message(VALIDATION_SYSTEM_PROMPT),
            generate()
        ],
        scorer=includes()
    )

@task
def l3_to_l1_validation():
    """Validate L3 to L1 classification"""
    return Task(
        dataset=json_dataset("conseq_fin_stage4_task_validation_l3_to_l1.jsonl"),
        solver=[
            system_message(VALIDATION_SYSTEM_PROMPT),
            generate()
        ],
        scorer=includes()
    )

@task
def subset_l2_l3_validation():
    """Validate L3 to L2 classification using focused subset of L2 options"""
    return Task(
        dataset=json_dataset("conseq_fin_stage4_task_validation_subset_l2_l3.jsonl"),
        solver=[
            system_message(VALIDATION_SYSTEM_PROMPT),
            generate()
        ],
        scorer=includes()
    )

def process_validation_results(log_dir: str, validation_type: str) -> Dict[str, Any]:
    """
    Process validation results and calculate accuracy
    
    Args:
        log_dir: Directory containing Inspect logs
        validation_type: Type of validation
        
    Returns:
        Dict with accuracy metrics
    """
    logger.info(f"Processing {validation_type} validation results from {log_dir}")
    
    correct = 0
    total = 0
    errors = []
    
    try:
        # Get evaluation dataframe
        df = evals_df(logs=log_dir, quiet=True)
        
        # Get total samples and accuracy from evaluation summary
        if 'total_samples' in df.columns and len(df) > 0:
            total = df['total_samples'].iloc[0]
        else:
            total = 50  # fallback to expected sample count
        
        # Check scores - includes scorer results are in score_includes_accuracy
        if 'score_includes_accuracy' in df.columns and len(df) > 0:
            accuracy = df['score_includes_accuracy'].iloc[0]
            correct = int(accuracy * total)
        else:
            logger.warning("No 'score_includes_accuracy' column found in eval results")
            accuracy = 0
            correct = 0
        
        results = {
            'validation_type': validation_type,
            'accuracy': accuracy,
            'correct': int(correct),
            'total': total,
            'errors': total - correct
        }
        
        logger.info(f"{validation_type} accuracy: {accuracy:.2%} ({correct}/{total})")
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing validation results: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to basic count
        return {
            'validation_type': validation_type,
            'accuracy': 0,
            'correct': 0,
            'total': 0,
            'errors': 0
        }
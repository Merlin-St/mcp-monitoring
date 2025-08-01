#!/usr/bin/env python3
"""
Custom scorer for cluster assignment validation

Implements an 'includes' scorer that checks if the correct cluster
appears in the model's response, with support for partial credit.
"""

from inspect_ai.scorer import Scorer, Score, scorer
from inspect_ai.solver import TaskState
import re
from typing import Optional

@scorer(name="cluster_includes")
def cluster_includes_scorer() -> Scorer:
    """
    Scorer that checks if the correct cluster ID is included in the response.
    
    Supports partial credit for hierarchically close answers:
    - Full credit (1.0): Exact match
    - Partial credit (0.5): Same parent cluster (for L3->L2 validation)
    - No credit (0.0): Wrong answer
    """
    
    async def score(state: TaskState, target: Optional[str] = None) -> Score:
        # Get the answer from model
        answer = state.output.completion if state.output else ""
        
        # Get metadata about the question
        metadata = state.metadata or {}
        validation_type = metadata.get('validation_type', '')
        
        # Extract cluster IDs from the answer
        # Look for patterns like "cluster_1_000" or "cluster_2_000"
        cluster_pattern = r'cluster_[12]_\d{3}'
        found_clusters = re.findall(cluster_pattern, answer)
        
        # Determine the correct answer based on validation type
        if validation_type == 'l3_to_l2':
            correct = metadata.get('correct_l2', '')
            correct_parent = metadata.get('correct_l1', '')
        elif validation_type == 'l2_to_l1':
            correct = metadata.get('correct_l1', '')
            correct_parent = None
        elif validation_type == 'l3_to_l1':
            correct = metadata.get('correct_l1', '')
            correct_parent = None
        else:
            # Fallback to target if validation type not specified
            correct = target or ''
            correct_parent = None
        
        # Check for exact match
        if correct in found_clusters:
            return Score(
                value=1.0,
                answer=answer,
                explanation=f"Correct! Found {correct} in response."
            )
        
        # Check for partial credit (same parent) - only for L3->L2
        if validation_type == 'l3_to_l2' and correct_parent:
            # Check if any found cluster has the same parent
            for cluster in found_clusters:
                if cluster.startswith('cluster_2_'):  # It's a Level 2 cluster
                    # We'd need the mapping to check parent - for now, no partial credit
                    pass
        
        # Check if correct answer appears anywhere in response (even without cluster_ prefix)
        correct_id = correct.split('_')[-1]  # Get just the number part
        if correct_id in answer:
            return Score(
                value=0.8,
                answer=answer,
                explanation=f"Partially correct - found cluster number {correct_id} but not in proper format."
            )
        
        # Wrong answer
        found_str = ', '.join(found_clusters) if found_clusters else "no valid clusters"
        return Score(
            value=0.0,
            answer=answer,
            explanation=f"Incorrect. Expected {correct}, found {found_str}."
        )
    
    return score


@scorer(name="cluster_includes_detailed")
def cluster_includes_detailed_scorer() -> Scorer:
    """
    Enhanced scorer with more detailed partial credit logic.
    """
    
    async def score(state: TaskState, target: Optional[str] = None) -> Score:
        answer = state.output.completion if state.output else ""
        metadata = state.metadata or {}
        
        # Extract all cluster mentions
        l1_pattern = r'cluster_1_\d{3}'
        l2_pattern = r'cluster_2_\d{3}'
        
        found_l1 = re.findall(l1_pattern, answer)
        found_l2 = re.findall(l2_pattern, answer)
        all_found = found_l1 + found_l2
        
        validation_type = metadata.get('validation_type', '')
        
        # Get correct answers
        if validation_type == 'l3_to_l2':
            correct = metadata.get('correct_l2', '')
            expected_level = 2
        elif validation_type == 'l2_to_l1':
            correct = metadata.get('correct_l1', '')
            expected_level = 1
        elif validation_type == 'l3_to_l1':
            correct = metadata.get('correct_l1', '')
            expected_level = 1
        else:
            correct = target or ''
            expected_level = 0
        
        # Exact match - full credit
        if correct in all_found:
            # Extra credit if it's the only/first cluster mentioned
            if len(all_found) == 1:
                return Score(
                    value=1.0,
                    answer=answer,
                    explanation=f"Perfect! Correctly identified {correct} as the only answer."
                )
            elif all_found[0] == correct:
                return Score(
                    value=0.95,
                    answer=answer,
                    explanation=f"Correct! {correct} was the first cluster mentioned."
                )
            else:
                return Score(
                    value=0.9,
                    answer=answer,
                    explanation=f"Correct, but {correct} was mentioned among other clusters."
                )
        
        # Check for level confusion (answered with wrong level)
        if expected_level == 2 and found_l1:
            return Score(
                value=0.3,
                answer=answer,
                explanation=f"Level confusion - answered with Level 1 cluster instead of Level 2."
            )
        elif expected_level == 1 and found_l2:
            return Score(
                value=0.3,
                answer=answer,
                explanation=f"Level confusion - answered with Level 2 cluster instead of Level 1."
            )
        
        # No valid clusters found
        if not all_found:
            return Score(
                value=0.0,
                answer=answer,
                explanation="No valid cluster IDs found in response."
            )
        
        # Wrong clusters mentioned
        return Score(
            value=0.0,
            answer=answer,
            explanation=f"Incorrect. Expected {correct}, found {', '.join(all_found)}."
        )
    
    return score
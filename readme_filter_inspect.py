#!/usr/bin/env python3
"""
README Content Filter - Inspect Task Definition

Uses Inspect framework to refine README content filtering using LLM-based analysis.
Removes installation tips while preserving functional descriptions, tool information,
and sector-relevant content for embedding analysis and consequentiality scoring.

MODIFIED: Uses simple includes() scorer instead of custom LLM-based scorer to save API tokens.
The scorer just checks that the output contains markdown headers (# ) as a basic validation.

This file contains only the task definition for Inspect to run.
Use readme_filter_dfprocessing.py to process the results.

Usage:
    python readme_content_filter.py                    # Run Stage 1 first
    inspect eval readme_filter_inspect.py --model anthropic/claude-sonnet-4-20250514
    python readme_filter_dfprocessing.py               # Process results
"""

import json
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import includes
from inspect_ai.solver import generate
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('readme_filter_inspect.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

README_FILTER_SYSTEM_PROMPT = """
Filter README content to retain only what’s useful for embedding analysis and consequentiality scoring.

KEEP:
- Tool features and functionality
- API docs and capabilities
- Use cases and application areas
- Integrations and connected services
- Sector- or task-specific context

REMOVE:
- Install/setup commands (e.g., npm, pip, docker)
- Prerequisites or system requirements
- Code examples for setup/config
- Directory layout, license, contributing

GUIDELINES:
1. Preserve markdown format and structure
2. If setup is mixed with function, keep function
3. Focus on WHAT the tool does, not HOW to install
4. Keep anything useful for classification

OUTPUT: Clean markdown only. No explanations.

Original README content:
""".strip()


# No custom scorer needed - using simple includes() to avoid LLM evaluation

def prepare_readme_dataset():
    """
    Prepare dataset from data_unified_filtered.json for README filtering
    """
    input_file = 'data_unified_filtered.json'
    dataset_file = 'readme_filter_input.jsonl'
    
    if not Path(input_file).exists():
        logger.error(f"Input file {input_file} not found. Run readme_content_filter.py first.")
        raise FileNotFoundError(f"Input file {input_file} not found")
    
    # Load the dataset
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} servers from {input_file}")
    
    # Filter to servers that have readme content and were processed in Stage 1
    servers_with_readme = [
        server for server in data 
        if server.get('readme_content') and server.get('readme_content').strip()
    ]
    
    logger.info(f"Found {len(servers_with_readme)} servers with README content")
    
    # Create dataset samples
    samples = []
    for server in servers_with_readme:
        # Use Stage 1 filtered content if available, otherwise original
        readme_content = server.get('readme_filteredinitial', server.get('readme_content', ''))
        
        if readme_content and readme_content.strip():
            # Truncate very long content to manage token limits
            if len(readme_content) > 8000:
                readme_content = readme_content[:8000] + "\n[...truncated for length...]"
            
            sample = {
                "input": f"{README_FILTER_SYSTEM_PROMPT}\n\n{readme_content}",
                "target": "# ",  # Simple target for includes() scorer
                "id": server.get('id', ''),
                "metadata": {
                    "stage": "readme_filter",
                    "server_name": server.get('name', ''),
                    "original_length": len(server.get('readme_content', '')),
                    "stage1_length": len(readme_content)
                }
            }
            samples.append(sample)
    
    logger.info(f"Created {len(samples)} samples for README filtering")
    
    # Save dataset
    with open(dataset_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved dataset to {dataset_file}")
    return dataset_file, len(samples)

def count_dataset_size(dataset_file):
    """Count the number of samples in the dataset file"""
    if not Path(dataset_file).exists():
        return 0
    
    with open(dataset_file, 'r') as f:
        count = sum(1 for _ in f)
    
    logger.info(f"Dataset {dataset_file} contains {count} samples")
    return count

@task
def readme_filter_task():
    """
    Inspect task for filtering README content
    """
    # Prepare dataset
    dataset_file, sample_count = prepare_readme_dataset()
    
    # Set appropriate message limit
    dynamic_message_limit = sample_count + 10  # Add buffer for safety
    
    logger.info(f"Setting message_limit to {dynamic_message_limit} for {sample_count} samples")
    
    return Task(
        dataset=json_dataset(dataset_file),
        solver=generate(),
        scorer=includes("# "),  # Simple scorer that checks for markdown headers
        message_limit=dynamic_message_limit
    )


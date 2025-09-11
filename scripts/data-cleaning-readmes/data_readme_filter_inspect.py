#!/usr/bin/env python3
"""
README Content Filter - Inspect Task Definition

Uses Inspect framework to refine README content filtering using LLM-based analysis.
Removes installation tips while preserving functional descriptions, tool information,
and sector-relevant content for embedding analysis and consequentiality scoring.

MODIFIED: Uses custom JSON structure validator scorer similar to clservers_2_inspect.py.
The scorer validates JSON structure, required fields, field types, and content quality.

This file contains only the task definition for Inspect to run.
Use data_readme_filter_dfprocessing.py to process the results.

Usage:
    python readme_content_filter.py                    # Run initial filter first
    inspect eval data_readme_filter_inspect.py --model anthropic/claude-3-5-haiku-latest --max-connections 300
    python data_readme_filter_dfprocessing.py               # Process results
"""

import json
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import scorer, Score, Target, accuracy, Scorer
from inspect_ai.solver import generate, system_message, TaskState
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_readme_filter_inspect.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

README_FILTER_SYSTEM_PROMPT = """
Filter README content and extract structured information useful for embedding analysis and consequentiality scoring.

Step 1: Create filtered_content
KEEP in filtered_content:
- Tool features and functionality
- API docs and capabilities
- Use cases and application areas
- Integrations and connected services
- Sector- or task-specific context

REMOVE from filtered_content:
- Install/setup commands (e.g., npm, pip, docker)
- Prerequisites or system requirements
- Code examples for setup/config
- Directory layout, license, contributing
- All URLs e.g. [Github](https://github.com) -> Github

Step 2: CLASSIFY server type:
- is_mcp_server: 1 if this is an actual MCP server with tools/capabilities, 0 if it's just documentation, links, or references to MCP servers

Step 3: 
EXTRACT tools information (try to copy the relevant exact text from the README):
- Identify each distinct tool/function/capability mentioned
- Extract name and description for each tool
- Look for tool definitions, API endpoints, functions, commands, etc.

OUTPUT: Valid JSON object only, with this exact structure:
{
  "summary": "Brief 1 sentence summary of what this MCP server and its tools does",
  "is_mcp_server": 1,
  "filtered_content": "Clean markdown content following the filtering rules above",
  "tools": [
    {
      "name": "first_tool_name",
      "description": "what this specific tool does and its purpose"
    },
    {
      "name": "second_tool_name", 
      "description": "what this other tool does and its purpose"
    },
    ...
  ]
}

CRITICAL: Output ONLY the JSON object - no explanations, comments, or additional text.

GUIDELINES:
1. Preserve markdown format in filtered_content
2. Focus on WHAT the tool does, not HOW to install
3. Set is_mcp_server to 1 for actual MCP servers, 0 for lists/references
4. Extract ALL distinct tools/functions/capabilities mentioned - create separate entries for each tool
5. Each tool should have a unique name - never duplicate tool names
6. If no specific tools are found, tools array can be empty
7. Tool descriptions should be copied, plus required input data such as API-keys

ONLY JSON. 
Original README content:
""".strip()


@scorer(metrics=[accuracy()])
def readme_json_scorer() -> Scorer:
    """
    Custom scorer for validating JSON structure with required fields
    Similar to clservers_2_inspect.py scorer but for README filtering
    """
    async def _scorer(state: TaskState, target: Target):
        completion = state.output.completion.strip()
        
        # Only basic cleanup - remove code block wrappers if present
        if completion.startswith('```json'):
            completion = completion[7:]
        elif completion.startswith('```'):
            completion = completion[3:]
        
        if completion.endswith('```'):
            completion = completion[:-3]
        
        completion = completion.strip()
        
        # Try to parse JSON - no aggressive extraction
        try:
            json_obj = json.loads(completion)
        except json.JSONDecodeError as e:
            return Score(
                value=0,
                answer=completion,
                explanation=f"Invalid JSON: {str(e)}"
            )
        
        # Check required fields exist
        required_fields = ["summary", "is_mcp_server", "filtered_content", "tools"]
        missing_fields = [field for field in required_fields if field not in json_obj]
        
        if missing_fields:
            return Score(
                value=0,
                answer=completion,
                explanation=f"Missing required fields: {missing_fields}"
            )
        
        # Check basic field types
        if not isinstance(json_obj["tools"], list):
            return Score(
                value=0,
                answer=completion,
                explanation="Tools field must be an array"
            )
        
        if json_obj["is_mcp_server"] not in [0, 1]:
            return Score(
                value=0,
                answer=completion,
                explanation="is_mcp_server must be 0 or 1"
            )
        
        return Score(
            value=1,
            answer=completion,
            explanation=f"Valid JSON with {len(json_obj['tools'])} tools"
        )
    
    return _scorer


def prepare_readme_dataset():
    """
    Prepare dataset from data_unified_filtered.json for README filtering
    """
    input_file = 'data/initial/data_unified_filtered.json'
    dataset_file = 'data/internal-cl/data_readme_filter_input.jsonl'
    
    if not Path(input_file).exists():
        logger.error(f"Input file {input_file} not found. Run readme_content_filter.py first.")
        raise FileNotFoundError(f"Input file {input_file} not found")
    
    # Load the dataset
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} servers from {input_file}")
    
    # Filter to servers that have initial filtered README content
    servers_with_readme = [
        server for server in data 
        if server.get('readme_filteredinitial') and server.get('readme_filteredinitial').strip()
    ]
    
    logger.info(f"Found {len(servers_with_readme)} servers with initial filtered README content")
    
    # Create dataset samples
    samples = []
    for server in servers_with_readme:
        # Use only initial filtered content
        readme_content = server.get('readme_filteredinitial', '')
        
        if readme_content and readme_content.strip():
            # Truncate very long content to manage token limits
            if len(readme_content) > 20000:
                readme_content = readme_content[:20000] + "\n[...truncated for length...]"
            
            sample = {
                "input": readme_content,
                "target": "valid_json",  # Target for custom JSON scorer
                "id": server.get('id', ''),
                "metadata": {
                    "phase": "readme_filter",
                    "server_name": server.get('name', ''),
                    "original_length": len(server.get('readme_filteredinitial', '')),
                    "initial_length": len(readme_content)
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
        solver=[
            system_message(README_FILTER_SYSTEM_PROMPT),
            generate()
        ],
        scorer=readme_json_scorer(),  # Custom scorer that validates JSON structure
        message_limit=dynamic_message_limit
    )


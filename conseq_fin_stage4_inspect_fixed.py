#!/usr/bin/env python3
"""
Stage 4 O*NET Task Classification - Inspect Task Definition (Fixed)

Uses Inspect framework to classify MCP tools across 4 dimensions:
1. O*NET task mapping (hierarchical)
2. Collaboration pattern analysis
3. Automation level assessment
4. Tool replacement mapping

Usage:
    inspect eval conseq_fin_stage4_inspect_fixed.py --model anthropic/claude-sonnet-4-20250514
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
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
        logging.FileHandler('conseq_fin_stage4_inspect.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# System prompt for all 4 classifications combined
COMBINED_CLASSIFICATION_PROMPT = """
You are an expert at analyzing AI agent tools and their capabilities. You will analyze a Model Context Protocol (MCP) server tool across 4 different dimensions.

Tool Information:
Server name & description: {server_name}: {server_description}
Tool name: {tool_name}
Tool description: {tool_description}
Input schema: {tool_input_schema}

Server context (README summary): {server_readme_summary}

Please provide 4 separate JSON responses for the following analyses:

### ANALYSIS 1: O*NET Task Mapping
Map this tool to the most appropriate O*NET occupational task.

Top-level categories:
1. Design, implement, and maintain diverse information technology systems
2. Create and preserve art, culture, and religious artifacts
3. Business management, finance, and customer service operations
4. Manage education, HR, and professional development programs
5. Conduct scientific research and technical analysis across disciplines
6. Perform government regulatory enforcement and public safety operations
7. Operate and manage diverse industrial and agricultural processes
8. Manage diverse energy sources and optimize consumption
9. Manage and improve environmental systems and sustainability practices
10. Comprehensive healthcare services and medical treatment across specialties

Respond with:
{{
    "reasoning": "Your step-by-step analysis",
    "top_level_category": "The selected category name",
    "top_level_number": 1-10,
    "specific_task": "The most specific O*NET task description",
    "occupation": "The primary occupation",
    "confidence": "high/medium/low"
}}

### ANALYSIS 2: Collaboration Pattern
Identify the primary human-AI collaboration pattern.

Patterns:
- Directive: Complete task delegation
- Feedback Loop: Iterative with environment feedback
- Task Iteration: Iterative refinement
- Learning: Seeking understanding
- Validation: Checking work
- None: Cannot determine

Respond with:
{{
    "reasoning": "Your analysis",
    "pattern": "Selected pattern or None",
    "confidence": "high/medium/low"
}}

### ANALYSIS 3: Automation Level
Classify the automation level (0-5).

Levels:
0: Not functional
1: MONITORING (read-only)
2: ANALYSIS (process & recommend)
3: META (coordinates other tools)
4: RESTRICTED EXECUTION (specific environment)
5: UNRESTRICTED EXECUTION (arbitrary actions)

Respond with:
{{
    "reasoning": "Your analysis",
    "level": 0-5,
    "level_description": "Why this level"
}}

### ANALYSIS 4: Tool Replacement
Identify which O*NET occupational tools this could replace.

Consider common workplace tools like:
- Desktop computers
- Laptop computers
- Smartphones
- Calculators
- Scanners
- Printers
- Photocopiers
- Desktop publishing software
- Word processing software
- Spreadsheet software
- Database software
- Project management software
- Customer relationship management software
- Enterprise resource planning software
- Computer aided design software
- Statistical analysis software
- Geographic information system software
- Medical software
- Laboratory equipment
- Measuring instruments
- Hand tools
- Power tools
- Industrial machinery
- Vehicles
- Communication devices
- Safety equipment

Respond with:
{{
    "reasoning": "Your analysis",
    "replaced_tools": ["tool1", "tool2", ...] or [],
    "confidence": "high/medium/low"
}}

Provide all 4 JSON responses in order.
"""

@scorer(metrics=[accuracy()])
def stage4_scorer() -> Scorer:
    """
    Custom scorer for validating all 4 classification outputs
    """
    async def _scorer(state: TaskState, target: Target):
        completion = state.output.completion
        
        # Try to extract JSON from the completion
        try:
            # Look for all JSON objects in the response
            import re
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_matches = re.findall(json_pattern, completion, re.DOTALL)
            
            if len(json_matches) < 4:
                return Score(
                    value=0,
                    answer=completion,
                    explanation=f"Expected 4 JSON responses, found {len(json_matches)}"
                )
            
            # Parse each JSON response
            results = []
            for i, match in enumerate(json_matches[:4]):
                try:
                    results.append(json.loads(match))
                except json.JSONDecodeError:
                    return Score(
                        value=0,
                        answer=completion,
                        explanation=f"Invalid JSON in response {i+1}"
                    )
            
            # Basic validation - just check that we have 4 valid JSON objects
            # More detailed validation could be added here
            
            return Score(
                value=1,
                answer=completion,
                explanation="All 4 classifications valid"
            )
            
        except Exception as e:
            return Score(
                value=0,
                answer=completion,
                explanation=f"Error processing response: {str(e)}"
            )
    
    return _scorer

@task
def onet_classification_task():
    """
    Inspect task for O*NET classification of MCP tools
    """
    dataset_file = "conseq_fin_stage4_input.jsonl"
    
    if not Path(dataset_file).exists():
        raise FileNotFoundError(f"Dataset file {dataset_file} not found. Run conseq_fin_stage4_data_prep.py first.")
    
    # Count samples
    with open(dataset_file, 'r') as f:
        sample_count = sum(1 for _ in f)
    
    logger.info(f"Loading {sample_count} tool samples for classification")
    
    # Dynamic message limit
    dynamic_message_limit = min(sample_count * 2 + 10, 50)  # Cap at 50 for testing
    
    return Task(
        dataset=json_dataset(dataset_file),
        solver=[
            system_message("You are an expert at analyzing AI tools and mapping them to occupational tasks and patterns."),
            generate()
        ],
        scorer=[stage4_scorer()],
        message_limit=dynamic_message_limit
    )
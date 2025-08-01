#!/usr/bin/env python3
"""
Stage 4 O*NET Classification - Part 4: Tool Replacement
Identifies which traditional tools could be replaced by MCP tools.

Usage:
    inspect eval conseq_fin_stage4_inspect_4_tools.py --model anthropic/claude-sonnet-4-20250514 --message-limit 50
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

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
        logging.FileHandler('conseq_fin_stage4_inspect_4_tools.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_tool_list() -> List[str]:
    """Get list of traditional tools from O*NET data"""
    # Common workplace tools that AI might replace
    # This is a curated list of tools that are more likely to be replaced by AI/software
    tools = [
        "10-key calculators",
        "Accounting software",
        "Analytical software",
        "Audio editing software",
        "Bar code readers",
        "Calendar and scheduling software",
        "Compliance software",
        "Computer aided design CAD software",
        "Contact management software",
        "Customer relationship management CRM software",
        "Data base management system software",
        "Desktop computers",
        "Desktop publishing software",
        "Development environment software",
        "Document management software",
        "Electronic mail software",
        "Enterprise resource planning ERP software",
        "Expert system software",
        "Facilities management software",
        "File servers",
        "Financial analysis software",
        "Flowcharting software",
        "Forms design software",
        "Geographic information system software",
        "Graphics or photo imaging software",
        "Human resources software",
        "Industrial control software",
        "Instant messaging software",
        "Internet browser software",
        "Inventory management software",
        "Label making software",
        "Laptop computers",
        "Map creation software",
        "Medical software",
        "Messaging software",
        "Network monitoring software",
        "Notebook computers",
        "Object or component oriented development software",
        "Operating system software",
        "Optical character reader OCR software",
        "Order processing software",
        "Personal computers",
        "Personal digital assistants PDA",
        "Photocopying equipment",
        "Point of sale POS systems",
        "Presentation software",
        "Procurement software",
        "Program testing software",
        "Project management software",
        "Query software",
        "Report generators",
        "Risk management software",
        "Route navigation software",
        "Sales and marketing software",
        "Scanners",
        "Scientific software",
        "Security software",
        "Smartphones",
        "Spell checkers",
        "Spreadsheet software",
        "Statistical software",
        "Tablet computers",
        "Tax preparation software",
        "Time accounting software",
        "Transaction processing software",
        "Translation software",
        "Video conferencing software",
        "Voice recognition software",
        "Web page creation software",
        "Web platform development software",
        "Word processing software",
        "Workflow software"
    ]
    return tools

# System prompt for tool replacement analysis
TOOL_REPLACEMENT_PROMPT = """The following is a description of an AI agent tool from a Model Context Protocol server:

Server name & description: {server_name}: {server_description}
Tool name, description and input schema: {tool_name} {tool_description} {input_schema}

Please identify which categories best describe this AI agent tool (not the full mcp server - only the specific tool). Consider the provided list of occupational tools. Your job is to identify ALL tools that could be replaced by this AI agent tool. You can select multiple tools if appropriate.

Available tools to consider:
{tools_list}

Select all that apply. Please comma-separate your selections (e.g., 'Accounting software, Spreadsheet software, Financial analysis software') and provide no additional commentary. If no tools are applicable, return 'none'."""

@scorer(metrics=[accuracy()])
def tool_replacement_scorer() -> Scorer:
    """Score tool replacement responses"""
    async def _scorer(state: TaskState, target: Target):
        completion = state.output.completion
        
        if not completion:
            return Score(
                value=0,
                answer=None,
                explanation="No response received"
            )
        
        # Clean the response
        answer = completion.strip().lower()
        
        # Check for "none" response
        if answer == 'none':
            return Score(
                value=1,
                answer='none',  # Keep as string
                explanation="No tools replaced"
            )
        
        # Parse comma-separated tools
        tools = [t.strip() for t in answer.split(',') if t.strip()]
        
        if tools:
            # Return the original answer string (comma-separated)
            return Score(
                value=1,
                answer=completion.strip(),  # Use original response
                explanation=f"Identified {len(tools)} replaced tools"
            )
        else:
            return Score(
                value=0,
                answer=answer,
                explanation="Could not parse tool list"
            )
    
    return _scorer

@task
def tool_replacement_task():
    """Task 4: Tool Replacement Analysis"""
    dataset_file = "conseq_fin_stage4_input.jsonl"
    
    if not Path(dataset_file).exists():
        raise FileNotFoundError(f"Dataset file {dataset_file} not found. Run conseq_fin_stage4_data_prep.py first.")
    
    # Get tool list
    tools = get_tool_list()
    tools_list_str = "\n".join([f"- {tool}" for tool in tools])
    
    # Count samples
    with open(dataset_file, 'r') as f:
        sample_count = sum(1 for _ in f)
    
    logger.info(f"Loading {sample_count} tool samples for replacement analysis")
    logger.info(f"Using {len(tools)} traditional tools for comparison")
    
    return Task(
        dataset=json_dataset(dataset_file),
        solver=[
            system_message(f"""You are an expert at identifying which traditional workplace tools can be replaced by AI tools.

Consider these traditional tools:
{tools_list_str}

When analyzing an AI tool, identify ALL traditional tools from the list above that it could replace.
Respond with a comma-separated list of tools exactly as they appear above.
If the AI tool doesn't replace any traditional tools, respond with 'none'.

Examples:
- An AI code generator might replace: "Development environment software, Program testing software, Code debugging software"
- A data analysis AI might replace: "Statistical software, Spreadsheet software, Report generators"
- A documentation tool might replace: "none" (if it's just for reading docs)"""),
            generate()
        ],
        scorer=[tool_replacement_scorer()],
        message_limit=100
    )
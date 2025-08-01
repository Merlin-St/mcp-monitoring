#!/usr/bin/env python3
"""
Stage 4 O*NET Task Classification - Inspect Task Definition

Uses Inspect framework to classify MCP tools across 4 dimensions:
1. O*NET task mapping (hierarchical)
2. Collaboration pattern analysis
3. Automation level assessment
4. Tool replacement mapping

Usage:
    inspect eval conseq_fin_stage4_inspect.py --model anthropic/claude-sonnet-4-20250514
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
from inspect_ai.solver import TaskState, generate, system_message, chain

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

# Load O*NET hierarchy
def load_onet_hierarchy():
    """Load the pre-built O*NET task hierarchy"""
    hierarchy_file = "conseq_fin_stage4_hierarchy.json"
    if not Path(hierarchy_file).exists():
        logger.error(f"Hierarchy file {hierarchy_file} not found! Run conseq_fin_stage4_embed_levels.py first.")
        return None
    
    with open(hierarchy_file, 'r') as f:
        return json.load(f)

# Load O*NET tools
def load_onet_tools():
    """Load O*NET tools from CSV for System Prompt 4"""
    tools_file = "conseq_fin_stage4_onet_toolsused.csv"
    if not Path(tools_file).exists():
        logger.error(f"Tools file {tools_file} not found!")
        return []
    
    import pandas as pd
    tools_df = pd.read_csv(tools_file)
    # Get unique tool examples
    unique_tools = tools_df['Example'].unique().tolist()
    return unique_tools

ONET_HIERARCHY = load_onet_hierarchy()
ONET_TOOLS = load_onet_tools()

# System Prompt 1: Tool to O*NET Task Mapping
TASK_MAPPING_PROMPT = """
The following is a description of an AI agent tool from a Model Context Protocol server:

Server name & description: {server_name}: {server_description}
Tool name, description and input schema: {tool_name} {tool_description} {input_schema}

You need to map this tool to the most appropriate O*NET occupational task through a hierarchical search.

First, consider these top-level task categories:
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

Think step by step:
1. Which top-level category best describes this tool's purpose?
2. What specific occupational tasks within that category does this tool perform?
3. What is the single most relevant O*NET task?

Respond with a JSON object containing:
{
    "reasoning": "Your step-by-step analysis",
    "top_level_category": "The selected category name from the list above",
    "top_level_number": 1-10,
    "specific_task": "The most specific O*NET task description",
    "occupation": "The primary occupation associated with this task",
    "confidence": "high/medium/low"
}
"""

# System Prompt 2: Collaboration Pattern Analysis
COLLABORATION_PATTERN_PROMPT = """
The following is a description of an AI agent tool from a Model Context Protocol server:

Server name & description: {server_name}: {server_description}
Tool name, description and input schema: {tool_name} {tool_description} {input_schema}

Your task is to analyze how humans would interact with this AI agent tool (not the full mcp server - only the specific tool) to identify the primary collaboration pattern. Focus on how humans would structure their requests and engage with this tool.

Analyze the tool according to these collaboration patterns:
- Directive: Human delegates complete task execution to AI tool with minimal interaction
- Feedback Loop: Human and AI tool engage in iterative dialogue to complete task with human mainly providing feedback from the environment
- Task Iteration: Human and AI tool engage in iterative dialogue to complete a task with the human refining the AI tool outputs
- Learning: Human seeks understanding and explanation rather than direct task completion
- Validation: Human uses AI tool to check or validate their own work

Based on your analysis, identify which of the above is most representative of how users would interact with this tool. If multiple patterns are present, select the one that appears most frequently. If you are unsure or there is not enough context to determine the most representative pattern, return 'None' as your answer. Use 'None' liberally---for only some tools will this task be possible.

Think step by step about:
1. What is the primary purpose of this tool?
2. How would users typically interact with it?
3. What pattern best describes this interaction?

Respond with a JSON object containing:
{
    "reasoning": "Your analysis of the interaction pattern",
    "pattern": "Directive/Feedback Loop/Task Iteration/Learning/Validation/None",
    "confidence": "high/medium/low"
}
"""

# System Prompt 3: Automation Level Analysis
AUTOMATION_LEVEL_PROMPT = """
The following is a description of an AI agent tool from a Model Context Protocol server:

Server name & description: {server_name}: {server_description}
Tool name, description and input schema: {tool_name} {tool_description} {input_schema}

Your task is to classify this AI agent tool's (not the full mcp server - only the specific tool) automation level based on its execution capabilities:

0: Not a functional tool (e.g., just documentation or links)
1: MONITORING - Read-only access, information retrieval only
2: ANALYSIS - Processes and analyzes data, provides recommendations
3: META - Coordinates other tools/servers for execution
4: RESTRICTED EXECUTION - Modifies specific external environments (e.g., executes transactions on specific systems)
5: UNRESTRICTED EXECUTION - Full execution capabilities (e.g., arbitrary code execution, system-wide modifications)

Examples:
- Tool: get_account_balance - Retrieves current balance from bank account
  Level: 1 (MONITORING - only reads data, no modifications)

- Tool: analyze_portfolio - Analyzes investment holdings and provides recommendations
  Level: 2 (ANALYSIS - processes data and advises, but doesn't modify anything)

- Tool: execute_trade - Places buy/sell orders on stock exchange
  Level: 4 (RESTRICTED EXECUTION - modifies external trading environment)

- Tool: run_shell_command - Executes arbitrary system commands
  Level: 5 (UNRESTRICTED EXECUTION - can modify anything on the system)

Think step by step:
1. What actions can this tool perform?
2. Can it modify external systems or just read/analyze?
3. If it can modify, is it restricted to specific systems or unrestricted?

Respond with a JSON object containing:
{
    "reasoning": "Your analysis of the tool's capabilities",
    "level": 0-5,
    "level_description": "Brief description of why this level was chosen"
}
"""

# System Prompt 4: Tool Match Analysis
TOOL_MATCH_PROMPT = """
The following is a description of an AI agent tool from a Model Context Protocol server:

Server name & description: {server_name}: {server_description}
Tool name, description and input schema: {tool_name} {tool_description} {input_schema}

Please identify which occupational tools from the O*NET database could be replaced or augmented by this AI agent tool. Consider the provided list of occupational tools and identify ALL tools that this AI agent tool could potentially replace or significantly augment.

Consider these occupational tools:
{onet_tools_list}

Think about:
1. What traditional tools or equipment does this AI tool replicate or replace?
2. What manual processes does it automate?
3. What physical tools become unnecessary because of this AI capability?

Select all that apply. Respond with a JSON object containing:
{
    "reasoning": "Your analysis of which tools are replaced",
    "replaced_tools": ["tool1", "tool2", ...] or [],
    "confidence": "high/medium/low"
}

If no tools are applicable, return an empty list for replaced_tools.
"""

@scorer(metrics=[accuracy()])
def onet_classifier_scorer() -> Scorer:
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
            
            # Validate each result
            # Result 1: Task mapping
            if not all(key in results[0] for key in ["reasoning", "top_level_category", "top_level_number", "specific_task", "occupation", "confidence"]):
                return Score(value=0, answer=completion, explanation="Missing fields in task mapping")
            
            # Result 2: Collaboration pattern
            if not all(key in results[1] for key in ["reasoning", "pattern", "confidence"]):
                return Score(value=0, answer=completion, explanation="Missing fields in collaboration pattern")
            
            valid_patterns = ["Directive", "Feedback Loop", "Task Iteration", "Learning", "Validation", "None"]
            if results[1]["pattern"] not in valid_patterns:
                return Score(value=0, answer=completion, explanation=f"Invalid collaboration pattern: {results[1]['pattern']}")
            
            # Result 3: Automation level
            if not all(key in results[2] for key in ["reasoning", "level", "level_description"]):
                return Score(value=0, answer=completion, explanation="Missing fields in automation level")
            
            if not isinstance(results[2]["level"], int) or results[2]["level"] not in range(6):
                return Score(value=0, answer=completion, explanation=f"Invalid automation level: {results[2]['level']}")
            
            # Result 4: Tool match
            if not all(key in results[3] for key in ["reasoning", "replaced_tools", "confidence"]):
                return Score(value=0, answer=completion, explanation="Missing fields in tool match")
            
            if not isinstance(results[3]["replaced_tools"], list):
                return Score(value=0, answer=completion, explanation="replaced_tools must be a list")
            
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

def create_combined_prompt(sample_data: Dict[str, Any]) -> str:
    """Create a combined prompt for all 4 classifications"""
    # Parse input data
    input_json = json.loads(sample_data['input'])
    
    # Format the tool list for prompt 4
    onet_tools_formatted = "\n".join([f"- {tool}" for tool in ONET_TOOLS[:200]])  # Limit to avoid context overflow
    
    combined_prompt = f"""You are an expert at analyzing AI agent tools and their capabilities. You will analyze a Model Context Protocol (MCP) server tool across 4 different dimensions.

Tool Information:
Server name & description: {input_json['server_name']}: {input_json['server_description']}
Tool name: {input_json['tool_name']}
Tool description: {input_json['tool_description']}
Input schema: {input_json['tool_input_schema']}

Server context (README summary): {input_json['server_readme_summary']}

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

Sample tools to consider:
{onet_tools_formatted[:2000]}
[... and more]

Respond with:
{{
    "reasoning": "Your analysis",
    "replaced_tools": ["tool1", "tool2", ...] or [],
    "confidence": "high/medium/low"
}}

Provide all 4 JSON responses in order."""
    
    return combined_prompt

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
    dynamic_message_limit = sample_count * 5 + 50  # Allow for multiple prompts per sample
    
    # Create custom system message that formats all prompts
    def combined_solver():
        @chain
        async def solve(state: TaskState) -> TaskState:
            # Get the sample data
            sample_data = state.metadata.get("sample", {})
            
            # Create combined prompt
            prompt = create_combined_prompt(sample_data)
            
            # Update the user message
            state.messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Generate response
            state = await generate()(state)
            
            return state
        
        return solve
    
    return Task(
        dataset=json_dataset(dataset_file),
        solver=[
            system_message("You are an expert at analyzing AI tools and mapping them to occupational tasks and patterns."),
            combined_solver()
        ],
        scorer=[onet_classifier_scorer()],
        message_limit=dynamic_message_limit
    )

# Entry point for Inspect
if __name__ == "__main__":
    # Verify dependencies
    if not ONET_HIERARCHY:
        logger.error("Failed to load O*NET hierarchy. Run conseq_fin_stage4_embed_levels.py first.")
        exit(1)
    
    if not ONET_TOOLS:
        logger.error("Failed to load O*NET tools.")
        exit(1)
    
    logger.info(f"Loaded O*NET hierarchy with {len(ONET_HIERARCHY.get('task_lookup', {}))} tasks")
    logger.info(f"Loaded {len(ONET_TOOLS)} O*NET occupational tools")
    logger.info("Ready for classification task")
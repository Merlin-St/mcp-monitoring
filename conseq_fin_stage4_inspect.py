#!/usr/bin/env python3
"""
Financial MCP Server Stage 4 - O*NET Economic Task Classification

Uses Inspect framework to classify individual MCP server tools according to 
O*NET economic tasks, following the Anthropic Economic Index methodology.

This should be run after:
    python conseq_fin_stage4_data_prep.py

Usage:
    inspect eval conseq_fin_stage4_inspect.py --model anthropic/claude-sonnet-4-20250514
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import TaskState, generate, system_message

# Load environment variables
load_dotenv()

ONET_ECONOMIC_TASK_CLASSIFICATION_SYSTEM_PROMPT = """
ONLY RESPOND WITH JSON. NO MARKDOWN. NO TEXT OUTSIDE JSON.

You are an economic task analyst classifying AI agent tools (from MCP servers) according to O*NET economic tasks, following the methodology from "Which Economic Tasks are Performed with AI?" by Anthropic.

Your goal is to map each tool to the most relevant economic tasks that it performs or automates, using the O*NET occupational framework.

STEP 1: First, write analysis_notes
Analyze the tool and document your findings in the analysis_notes field. Include:
- What specific economic function this tool performs
- Which occupation(s) would typically use this tool
- The level of automation vs. augmentation it provides
- Any uncertainty about the tool's purpose or capabilities

STEP 2: Identify primary_onet_task
Map to the single most relevant O*NET task statement. Use your knowledge of the O*NET database structure to identify specific task statements. Examples:
- "Debug software programs using manual and automated debugging procedures"
- "Create technical documentation for software applications and systems"
- "Develop and maintain databases and database management systems"
- "Analyze financial data to identify trends, risks, and opportunities"
- "Design and implement security measures to protect computer networks"

If you're unsure of the exact O*NET wording, provide the most specific task description you can.

STEP 3: Identify secondary_onet_tasks (up to 2)
List up to 2 additional relevant O*NET tasks, separated by semicolons.

STEP 4: Determine occupation_category
Select the most relevant broad occupational category:
- "Computer and Mathematical Occupations"
- "Architecture and Engineering Occupations"
- "Life, Physical, and Social Science Occupations"
- "Business and Financial Operations Occupations"
- "Management Occupations"
- "Sales and Related Occupations"
- "Office and Administrative Support Occupations"
- "Arts, Design, Entertainment, Sports, and Media Occupations"
- "Healthcare Practitioners and Technical Occupations"
- "Education, Training, and Library Occupations"
- "Legal Occupations"
- "Other" (specify in notes)

STEP 5: Set automation_level (1-5)
Rate the level of automation this tool provides:
- 1: MONITORING - Read-only information gathering, displays data
- 2: ANALYSIS - Processes and analyzes data, provides insights, no external changes
- 3: WORKFLOW - Coordinates between systems, manages processes, limited execution
- 4: EXECUTION - Performs specific tasks with external system changes
- 5: AUTONOMOUS - Full autonomous operation with minimal human oversight

STEP 6: Set economic_impact (1-3)
Assess potential economic impact:
- 1: LOW - Affects individual productivity, limited scope
- 2: MEDIUM - Affects team/department efficiency, moderate scope
- 3: HIGH - Affects organizational operations, broad economic implications

STEP 7: Set confidence (H/M/L)
- "H" (High): Clear tool description, obvious economic task mapping
- "M" (Medium): Adequate description, reasonable task inference
- "L" (Low): Limited description, uncertain task mapping

STEP 8: Identify task_skills (up to 5)
List key skills from the O*NET Skills taxonomy that this tool exhibits or supports:
Examples: "Critical Thinking", "Programming", "Systems Analysis", "Writing", "Complex Problem Solving", "Reading Comprehension", "Mathematics", "Active Learning"

EXAMPLES:

Example 1 - Software Development Tool:
Input Tool: {"tool_name": "debug_code", "tool_description": "Debug Python applications by analyzing stack traces and suggesting fixes", "server_name": "python-dev-server"}
{
    "tool_id": "python-dev-server#01",
    "analysis_notes": "This tool performs software debugging tasks, which directly maps to software development work. It automates part of the debugging process by analyzing stack traces and providing fix suggestions, but likely requires developer oversight and implementation.",
    "primary_onet_task": "Debug software programs using manual and automated debugging procedures",
    "secondary_onet_tasks": "Analyze code to identify errors and performance issues;Create technical documentation for software troubleshooting",
    "occupation_category": "Computer and Mathematical Occupations",
    "automation_level": 2,
    "economic_impact": 1,
    "confidence": "H",
    "task_skills": "Programming;Critical Thinking;Complex Problem Solving;Systems Analysis;Active Learning"
}

Example 2 - Financial Analysis Tool:
Input Tool: {"tool_name": "risk_calculator", "tool_description": "Calculate portfolio risk metrics including VaR, beta, and correlation analysis", "server_name": "finance-analytics"}
{
    "tool_id": "finance-analytics#01",
    "analysis_notes": "This tool performs quantitative financial analysis, specifically risk assessment calculations. It would typically be used by financial analysts, portfolio managers, or risk managers to evaluate investment portfolios. The tool automates complex mathematical calculations but requires professional interpretation.",
    "primary_onet_task": "Analyze financial data to identify trends, risks, and opportunities",
    "secondary_onet_tasks": "Develop financial models and forecasting tools;Prepare reports and presentations on financial analysis findings",
    "occupation_category": "Business and Financial Operations Occupations",
    "automation_level": 2,
    "economic_impact": 2,
    "confidence": "H",
    "task_skills": "Mathematics;Critical Thinking;Complex Problem Solving;Systems Analysis;Active Learning"
}

Example 3 - Content Creation Tool:
Input Tool: {"tool_name": "generate_blog_post", "tool_description": "Generate marketing blog posts based on topic and target audience", "server_name": "content-creator"}
{
    "tool_id": "content-creator#01",
    "analysis_notes": "This tool automates content creation for marketing purposes, which would typically be performed by marketing specialists, content writers, or copywriters. It appears to handle the entire content generation process, though quality review would likely be needed.",
    "primary_onet_task": "Create written materials for marketing and promotional purposes",
    "secondary_onet_tasks": "Develop content strategies for target audiences;Edit and revise written materials for publication",
    "occupation_category": "Arts, Design, Entertainment, Sports, and Media Occupations",
    "automation_level": 4,
    "economic_impact": 1,
    "confidence": "H",
    "task_skills": "Writing;Critical Thinking;Active Learning;Reading Comprehension;Persuasion"
}

Example 4 - General Purpose Tool:
Input Tool: {"tool_name": "file_manager", "tool_description": "Basic file operations: create, read, update, delete files", "server_name": "filesystem-server"}
{
    "tool_id": "filesystem-server#01",
    "analysis_notes": "This is a general-purpose file management tool that could be used across many occupations. While not tied to a specific economic task, it supports various administrative and technical functions that involve document management and file organization.",
    "primary_onet_task": "Organize and maintain files and records using computer systems",
    "secondary_onet_tasks": "Perform administrative tasks using computer applications;Manage digital information and documentation systems",
    "occupation_category": "Office and Administrative Support Occupations",
    "automation_level": 1,
    "economic_impact": 1,
    "confidence": "M",
    "task_skills": "Active Learning;Technology Design;Reading Comprehension"
}

Output Format:
{
    "tool_id": "string",
    "analysis_notes": "String describing the economic task analysis",
    "primary_onet_task": "Most relevant O*NET task statement",
    "secondary_onet_tasks": "Additional relevant tasks separated by semicolons (up to 2)",
    "occupation_category": "Broad occupational category from the list above",
    "automation_level": 1|2|3|4|5,
    "economic_impact": 1|2|3,
    "confidence": "H|M|L",
    "task_skills": "Relevant O*NET skills separated by semicolons (up to 5)"
}

RESPOND ONLY WITH JSON.
""".strip()

@scorer(metrics=[accuracy()])
def onet_classification_scorer() -> Scorer:
    """
    Custom scorer for validating O*NET classification JSON structure
    """
    async def _scorer(state: TaskState, target: Target):
        completion = state.output.completion
        
        # Try to extract JSON from the completion text
        json_obj = None
        
        # First try: direct JSON parsing
        try:
            json_obj = json.loads(completion)
        except json.JSONDecodeError:
            # Second try: find JSON block in text
            import re
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_matches = re.findall(json_pattern, completion, re.DOTALL)
            
            for match in json_matches:
                try:
                    json_obj = json.loads(match)
                    break
                except json.JSONDecodeError:
                    continue
        
        if json_obj is None:
            # Third try: more aggressive JSON extraction
            try:
                # Look for content between first { and last }
                start = completion.find('{')
                end = completion.rfind('}')
                if start != -1 and end != -1 and end > start:
                    potential_json = completion[start:end+1]
                    json_obj = json.loads(potential_json)
            except json.JSONDecodeError:
                pass
        
        if json_obj is None:
            return Score(
                value=0,
                answer=completion,
                explanation="No valid JSON found in response"
            )
        
        # Validate all required fields
        required_fields = [
            "tool_id",
            "analysis_notes",
            "primary_onet_task",
            "secondary_onet_tasks",
            "occupation_category",
            "automation_level",
            "economic_impact",
            "confidence",
            "task_skills"
        ]
        
        missing_fields = [field for field in required_fields if field not in json_obj]
        
        if missing_fields:
            return Score(
                value=0,
                answer=completion,
                explanation=f"Missing required fields: {missing_fields}"
            )
        
        # Validate automation_level (should be 1-5)
        if json_obj["automation_level"] not in [1, 2, 3, 4, 5]:
            return Score(
                value=0,
                answer=completion,
                explanation=f"Invalid automation_level value: {json_obj['automation_level']} (expected 1-5)"
            )
        
        # Validate economic_impact (should be 1-3)
        if json_obj["economic_impact"] not in [1, 2, 3]:
            return Score(
                value=0,
                answer=completion,
                explanation=f"Invalid economic_impact value: {json_obj['economic_impact']} (expected 1-3)"
            )
        
        # Validate confidence field (should be H, M, or L)
        if json_obj["confidence"] not in ["H", "M", "L"]:
            return Score(
                value=0,
                answer=completion,
                explanation=f"Invalid confidence value: {json_obj['confidence']} (expected H, M, or L)"
            )
        
        # Validate occupation_category is not empty
        valid_categories = [
            "Computer and Mathematical Occupations",
            "Architecture and Engineering Occupations", 
            "Life, Physical, and Social Science Occupations",
            "Business and Financial Operations Occupations",
            "Management Occupations",
            "Sales and Related Occupations",
            "Office and Administrative Support Occupations",
            "Arts, Design, Entertainment, Sports, and Media Occupations",
            "Healthcare Practitioners and Technical Occupations",
            "Education, Training, and Library Occupations",
            "Legal Occupations",
            "Other"
        ]
        
        if json_obj["occupation_category"] not in valid_categories:
            return Score(
                value=0,
                answer=completion,
                explanation=f"Invalid occupation_category: {json_obj['occupation_category']} (must be one of the predefined categories)"
            )
        
        return Score(
            value=1,
            answer=completion,
            explanation="Valid JSON with required fields extracted"
        )
    
    return _scorer

def count_dataset_size(dataset_file):
    """Count the number of samples in the dataset file"""
    if not Path(dataset_file).exists():
        return 0
    
    with open(dataset_file, 'r') as f:
        count = sum(1 for _ in f)
    
    return count

@task
def onet_economic_task_classification():
    """
    Inspect task for classifying MCP tools according to O*NET economic tasks
    """
    dataset_file = "conseq_fin_stage4_input.jsonl"
    
    if not Path(dataset_file).exists():
        raise FileNotFoundError(f"Dataset file {dataset_file} not found. Run conseq_fin_stage4_data_prep.py first.")
    
    # Count samples in dataset to set appropriate message limit
    dataset_size = count_dataset_size(dataset_file)
    dynamic_message_limit = dataset_size + 10  # Add buffer for safety
    
    return Task(
        dataset=json_dataset(dataset_file),
        solver=[
            system_message(ONET_ECONOMIC_TASK_CLASSIFICATION_SYSTEM_PROMPT),
            generate()
        ],
        scorer=[onet_classification_scorer()],
        message_limit=dynamic_message_limit
    )

# API key handling is managed automatically by AISI environment variables:
# - INSPECT_API_KEY_OVERRIDE=aisitools.api_key.override_api_key  
# - ANTHROPIC_API_KEY (handled automatically by inspect framework)
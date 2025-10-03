#!/usr/bin/env python3
"""
Study Composer Config Generator for MCP Server and Tool Classification Study

This script generates a Gorilla Study Composer configuration from CSV data containing
MCP servers and tools, creating questionnaire pages for human classification.
"""

import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Autonomy classification taxonomy (2-step hierarchical structure)
AUTONOMY_TAXONOMY = {
    "perception": {
        "label": "1. PERCEPTION (gathering information)",
        "subcategories": {
            "sensors": "1.1 Sensors - database queries, monitoring, diagnostics, GUI reading, voice, search, physical sensing"
        }
    },
    "reasoning": {
        "label": "2. REASONING (processing/analysis)",
        "subcategories": {
            "planning": "2.1 Planning - task decomposition, path-finding, workflow orchestration",
            "analysis": "2.2 Analysis - calculations, simulations, data processing",
            "resource_management": "2.3 Resource Management - memory, self-management, resource allocation"
        }
    },
    "action": {
        "label": "3. ACTION (directly affecting the environment)",
        "subcategories": {
            "authentication": "3.1 Authentication - login, CAPTCHA, wallet operations",
            "computer_use": "3.2 Computer Use - GUI interaction, website automation, computer control",
            "code_execution": "3.3 Code Execution - interpreters, IDE, file operations, running code",
            "software_extensions": "3.4 Software Extensions - calendar, social media APIs, third-party services",
            "physical_extensions": "3.5 Physical Extensions - robotics, laboratory tools, physical world",
            "human_interaction": "3.6 Human Interaction - phone calls, messaging, direct communication",
            "agent_interaction": "3.7 Agent Interaction - multi-agent coordination, sub-agents, third-party agents"
        }
    }
}


def load_onet_taxonomy(csv_path: str) -> Dict[str, Any]:
    """
    Load O*NET task taxonomy from CSV into hierarchical structure.

    Returns dict with structure:
    {
        'L1_01': {
            'name': 'Level 1 name',
            'level2': {
                'cluster_id': 'Level 2 name',
                ...
            }
        },
        ...
    }
    """
    logger.info(f"Loading O*NET taxonomy from {csv_path}")
    taxonomy = {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            l1_cluster = row['level1_cluster']
            l1_name = row['level1_name']
            l2_cluster = row['level2_cluster']
            l2_name = row['level2_name']

            if l1_cluster not in taxonomy:
                taxonomy[l1_cluster] = {
                    'name': l1_name,
                    'level2': {}
                }

            if l2_cluster not in taxonomy[l1_cluster]['level2']:
                taxonomy[l1_cluster]['level2'][l2_cluster] = l2_name

    logger.info(f"Loaded {len(taxonomy)} Level 1 clusters")
    return taxonomy


def create_server_classification_page(server_data: Dict[str, Any], page_number: int) -> Dict[str, Any]:
    """Create a questionnaire page for server classification."""

    server_name = server_data.get('server_name', 'Unknown')
    description = server_data.get('description', 'No description available')
    readme_filtered = server_data.get('readme_filtered', '')
    readme_summary = server_data.get('readme_summary', '')

    # Build tools list - iterate through ALL possible tool slots
    tools_list = []
    try:
        tool_count = int(server_data.get('tool_count', 0))
    except (ValueError, TypeError):
        tool_count = 0

    # Always check up to 99 tools to catch all entries
    for i in range(1, 100):
        tool_name = server_data.get(f'tool_{i:02d}_name', '').strip()
        if tool_name:
            tool_desc = server_data.get(f'tool_{i:02d}_description', '').strip()
            # Include both name and description, handle missing descriptions
            if tool_desc:
                tools_list.append(f"- **{tool_name}**: {tool_desc}")
            else:
                tools_list.append(f"- **{tool_name}**")

    tools_text = "\n".join(tools_list) if tools_list else "No tools documented"

    # Add tool count summary
    if tools_list:
        tools_header = f"**Tool Count:** {len(tools_list)}\n\n"
        tools_text = tools_header + tools_text

    # Instructions markdown
    instructions_md = f"""## Server Classification

### Server Information:
**Name:** {server_name}

**Description:** {description}

**README Summary:** {readme_summary}

**Tools:**
{tools_text}

---

Please classify this server based on the information provided above.
"""

    # Create questions
    questions = [
        {
            "id": f"server_{page_number}_industry_generality",
            "type": "multiple-choice",
            "text": "Q1: Industry Generality - Is this server cross-industry (usable across many sectors) or industry-specific?",
            "choices": [
                {
                    "text": "1 - Cross-industry (desktop commander, file management, email client, calendar tools, note-taking)",
                    "value": "1"
                },
                {
                    "text": "0 - Industry-specific (crypto transaction tools, payment platforms, medical records systems, legal document processors)",
                    "value": "0"
                }
            ],
            "required": True
        },
        {
            "id": f"server_{page_number}_environment_generality",
            "type": "multiple-choice",
            "text": "Q2: Environment Generality - Does this server operate in an open/untrusted environment or a trusted/pre-specified environment?",
            "choices": [
                {
                    "text": "1 - Open/untrusted (computer use, browser automation, web scraping, generic file system access)",
                    "value": "1"
                },
                {
                    "text": "0 - Trusted/pre-specified (specific API like Stripe, GitHub, Slack; internal database; pre-configured service endpoint)",
                    "value": "0"
                }
            ],
            "required": True
        },
        {
            "id": f"server_{page_number}_payment_autonomy",
            "type": "multiple-choice",
            "text": "Q3: Payment Autonomy Level - What is the payment autonomy level of this server?",
            "choices": [
                {
                    "text": "0 - Not a payment processing server (no payment functionality)",
                    "value": "0"
                },
                {
                    "text": "1 - Only information about payments (invoice view, payment history, read-only)",
                    "value": "1"
                },
                {
                    "text": "2 - Payment request or link created (generates payment links/requests but doesn't execute)",
                    "value": "2"
                },
                {
                    "text": "3 - Payment processing via third-party (executes payments through external API like Stripe, PayPal)",
                    "value": "3"
                },
                {
                    "text": "4 - Payment processing directly (direct payment execution with full control, e.g., blockchain transactions)",
                    "value": "4"
                }
            ],
            "required": True
        }
    ]

    page = {
        "type": "Questionnaire",
        "title": f"Server Classification - {page_number}",
        "id": f"server_classification_{page_number}",
        "content": {
            "instructionsMarkdown": instructions_md,
            "questionnaire": {
                "id": f"server_questionnaire_{page_number}",
                "questions": questions
            }
        }
    }

    return page


def create_tool_classification_page(tool_data: Dict[str, Any], page_number: int, onet_taxonomy: Dict[str, Any]) -> Dict[str, Any]:
    """Create a questionnaire page for tool classification with autonomy and O*NET task mapping."""

    tool_name = tool_data.get('tool_name', 'Unknown')
    tool_description = tool_data.get('tool_description', 'No description available')
    tool_input_schema = tool_data.get('tool_input_schema', '{}')
    server_name = tool_data.get('server_name', 'Unknown')

    # Instructions markdown
    instructions_md = f"""## Tool Classification

### Tool Information:
**Tool Name:** {tool_name}

**Server:** {server_name}

**Description:** {tool_description}

**Input Schema:**
```json
{tool_input_schema}
```

---

Please classify this tool's autonomy level and associated O*NET occupational task.
"""

    # Q1: Autonomy Level - Step 1 (Category selection)
    autonomy_step1_choices = [
        {"text": "1. PERCEPTION (gathering information)", "value": "perception"},
        {"text": "2. REASONING (processing/analysis)", "value": "reasoning"},
        {"text": "3. ACTION (directly affecting the environment)", "value": "action"}
    ]

    # Q2: Autonomy Level - Step 2 (Subcategory - conditional based on Q1)
    perception_subcategories = [
        {"text": "1.1 Sensors - database queries, monitoring, diagnostics, GUI reading, voice, search, physical sensing", "value": "sensors"}
    ]

    reasoning_subcategories = [
        {"text": "2.1 Planning - task decomposition, path-finding, workflow orchestration", "value": "planning"},
        {"text": "2.2 Analysis - calculations, simulations, data processing", "value": "analysis"},
        {"text": "2.3 Resource Management - memory, self-management, resource allocation", "value": "resource_management"}
    ]

    action_subcategories = [
        {"text": "3.1 Authentication - login, CAPTCHA, wallet operations", "value": "authentication"},
        {"text": "3.2 Computer Use - GUI interaction, website automation, computer control", "value": "computer_use"},
        {"text": "3.3 Code Execution - interpreters, IDE, file operations, running code", "value": "code_execution"},
        {"text": "3.4 Software Extensions - calendar, social media APIs, third-party services", "value": "software_extensions"},
        {"text": "3.5 Physical Extensions - robotics, laboratory tools, physical world", "value": "physical_extensions"},
        {"text": "3.6 Human Interaction - phone calls, messaging, direct communication", "value": "human_interaction"},
        {"text": "3.7 Agent Interaction - multi-agent coordination, sub-agents, third-party agents", "value": "agent_interaction"}
    ]

    # O*NET Classification - Level 1
    onet_level1_choices = []
    for l1_cluster_id in sorted(onet_taxonomy.keys()):
        l1_name = onet_taxonomy[l1_cluster_id]['name']
        onet_level1_choices.append({
            "text": f"{l1_cluster_id}: {l1_name}",
            "value": l1_cluster_id
        })

    # Create all O*NET Level 2 questions (conditional on Level 1)
    onet_level2_questions = []
    for l1_cluster_id in sorted(onet_taxonomy.keys()):
        l2_choices = []
        for l2_cluster_id in sorted(onet_taxonomy[l1_cluster_id]['level2'].keys()):
            l2_name = onet_taxonomy[l1_cluster_id]['level2'][l2_cluster_id]
            l2_choices.append({
                "text": f"{l2_cluster_id}: {l2_name}",
                "value": str(l2_cluster_id)
            })

        if l2_choices:
            onet_level2_questions.append({
                "id": f"tool_{page_number}_onet_level2_{l1_cluster_id}",
                "type": "multiple-choice",
                "text": f"Select Level 2 task cluster for {l1_cluster_id}:",
                "choices": l2_choices,
                "required": True,
                "visibleIf": {
                    "questionId": f"tool_{page_number}_onet_level1",
                    "answers": [str(list(sorted(onet_taxonomy.keys())).index(l1_cluster_id))]
                }
            })

    # Create questions
    questions = [
        {
            "id": f"tool_{page_number}_autonomy_category",
            "type": "multiple-choice",
            "text": "Q1: Autonomy Category - Select the primary autonomy category for this tool:",
            "choices": autonomy_step1_choices,
            "required": True
        },
        # Conditional subcategory questions
        {
            "id": f"tool_{page_number}_autonomy_perception_sub",
            "type": "multiple-choice",
            "text": "Select PERCEPTION subcategory:",
            "choices": perception_subcategories,
            "required": True,
            "visibleIf": {
                "questionId": f"tool_{page_number}_autonomy_category",
                "answers": ["0"]  # Index 0 = perception
            }
        },
        {
            "id": f"tool_{page_number}_autonomy_reasoning_sub",
            "type": "multiple-choice",
            "text": "Select REASONING subcategory:",
            "choices": reasoning_subcategories,
            "required": True,
            "visibleIf": {
                "questionId": f"tool_{page_number}_autonomy_category",
                "answers": ["1"]  # Index 1 = reasoning
            }
        },
        {
            "id": f"tool_{page_number}_autonomy_action_sub",
            "type": "multiple-choice",
            "text": "Select ACTION subcategory:",
            "choices": action_subcategories,
            "required": True,
            "visibleIf": {
                "questionId": f"tool_{page_number}_autonomy_category",
                "answers": ["2"]  # Index 2 = action
            }
        },
        # O*NET Classification
        {
            "id": f"tool_{page_number}_onet_level1",
            "type": "multiple-choice",
            "text": "Q2: O*NET Task Classification - Level 1 - Select the primary occupational task cluster:",
            "choices": onet_level1_choices,
            "required": True
        }
    ]

    # Add all conditional Level 2 questions
    questions.extend(onet_level2_questions)

    page = {
        "type": "Questionnaire",
        "title": f"Tool Classification - {page_number}",
        "id": f"tool_classification_{page_number}",
        "content": {
            "instructionsMarkdown": instructions_md,
            "questionnaire": {
                "id": f"tool_questionnaire_{page_number}",
                "questions": questions
            }
        }
    }

    return page


def generate_study_config(
    servers_csv_path: str,
    tools_csv_path: str,
    onet_csv_path: str,
    output_path: str,
    num_server_samples: int = 2,
    num_tool_samples: int = 2
):
    """
    Generate complete Study Composer configuration.

    Args:
        servers_csv_path: Path to clservers_classified.csv
        tools_csv_path: Path to cltools_classified.csv
        onet_csv_path: Path to task_clusters_names.csv
        output_path: Path for output JSON config
        num_server_samples: Number of server classification pages (default: 2)
        num_tool_samples: Number of tool classification pages (default: 2)
    """
    logger.info("Starting study config generation")

    # Load O*NET taxonomy
    onet_taxonomy = load_onet_taxonomy(onet_csv_path)

    # Load server data - prioritize servers with tools for samples
    logger.info(f"Loading server data from {servers_csv_path}")
    all_servers = []
    with open(servers_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_servers.append(row)

    # Sort servers: prioritize those with tools (tool_count > 0)
    servers_with_tools = [s for s in all_servers if int(s.get('tool_count', 0) or 0) > 0]
    servers_without_tools = [s for s in all_servers if int(s.get('tool_count', 0) or 0) == 0]

    # Take samples: prioritize servers with tools
    servers = (servers_with_tools + servers_without_tools)[:num_server_samples]

    logger.info(f"Loaded {len(servers)} servers")

    # Load tool data
    logger.info(f"Loading tool data from {tools_csv_path}")
    tools = []
    with open(tools_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= num_tool_samples:
                break
            tools.append(row)

    logger.info(f"Loaded {len(tools)} tools")

    # Create pages
    pages = []

    # Intro page
    intro_page = {
        "type": "TextMarkdown",
        "title": "MCP Classification Study",
        "id": "intro_page",
        "content": {
            "markdownText": """# MCP Server and Tool Classification Study

Welcome to the MCP (Model Context Protocol) Classification Study!

## Study Overview
In this study, you will be classifying MCP servers and tools based on:

### For Servers:
1. **Industry Generality**: Is it cross-industry or industry-specific?
2. **Environment Generality**: Does it operate in open/untrusted or trusted environments?
3. **Payment Autonomy**: What level of payment processing capability does it have?

### For Tools:
1. **Autonomy Level**: Classify tools into Perception, Reasoning, or Action categories with subcategories
2. **O*NET Task Mapping**: Map tools to occupational tasks from the O*NET database

## Instructions
- Read all information carefully before answering
- All questions are required
- Some questions will appear conditionally based on your previous answers

Click **Next** to begin the classification tasks.
"""
        }
    }
    pages.append(intro_page)

    # Server classification pages
    logger.info("Generating server classification pages")
    for i, server_data in enumerate(servers, start=1):
        page = create_server_classification_page(server_data, i)
        pages.append(page)

    # Tool classification pages
    logger.info("Generating tool classification pages")
    for i, tool_data in enumerate(tools, start=1):
        page = create_tool_classification_page(tool_data, i, onet_taxonomy)
        pages.append(page)

    # Completion page
    completion_page = {
        "type": "TextMarkdown",
        "title": "Study Complete",
        "id": "completion_page",
        "content": {
            "markdownText": """# Thank You!

You have completed the MCP Classification Study.

Your responses have been recorded and will help us better understand the landscape of MCP servers and tools.

If you have any questions or feedback, please contact the research team.

Click **Finish** to submit your responses.
"""
        }
    }
    pages.append(completion_page)

    # Build complete study config
    study_config = {
        "name": "mcp-classification-study",
        "studyContentsConfig": {
            "navigation": {
                "allowBack": True,
                "showProgress": True,
                "showPageNumbers": True,
                "exitRedirectUrl": "https://www.prolific.com",
                "showElapsedTime": True
            },
            "pages": pages
        },
        "model": "claude-sonnet-4-20250514",
        "maxTokens": 2048,
        "providerOptions": None,
        "provider": "Anthropic",
        "responseMode": "stream",
        "props": {}
    }

    # Write output
    logger.info(f"Writing config to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(study_config, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Study config generated successfully!")
    logger.info(f"   - Total pages: {len(pages)}")
    logger.info(f"   - Server classification pages: {len(servers)}")
    logger.info(f"   - Tool classification pages: {len(tools)}")
    logger.info(f"   - Output: {output_path}")


def main():
    """Main entry point for script."""
    # Paths - relative to project root (one level up from script location)
    base_dir = Path(__file__).parent.parent
    servers_csv = base_dir / "data" / "final" / "clservers_classified.csv"
    tools_csv = base_dir / "data" / "final" / "cltools_classified.csv"
    onet_csv = base_dir / "data" / "internal-task-clusters" / "task_clusters_names.csv"
    output_json = Path(__file__).parent / "99_study_config_mcp_classification.json"

    # Validate input files exist
    for filepath in [servers_csv, tools_csv, onet_csv]:
        if not filepath.exists():
            logger.error(f"Input file not found: {filepath}")
            return

    # Generate config with 2 samples each (for testing)
    generate_study_config(
        servers_csv_path=str(servers_csv),
        tools_csv_path=str(tools_csv),
        onet_csv_path=str(onet_csv),
        output_path=str(output_json),
        num_server_samples=2,
        num_tool_samples=2
    )

    logger.info("To generate full config with 100 samples each, edit the script and change:")
    logger.info("  num_server_samples=100, num_tool_samples=100")


if __name__ == "__main__":
    main()

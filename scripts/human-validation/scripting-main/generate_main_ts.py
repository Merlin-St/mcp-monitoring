#!/usr/bin/env python3
"""
Generate main.ts file dynamically from CLServers CSV and CLTools data.

This script reads:
1. data/final/clservers_classified.csv - MCP server data with NAICS + finance classification
2. data/final/cltools_classified.csv - Tool-level data with O*NET + functionality classification
3. questions_config.csv - Additional classification questions with choices

And generates a main.ts file with hierarchical O*NET and functionality validation questions
followed by the standard classification questions.
"""

import csv
import json
import random
import sys
from pathlib import Path
from collections import OrderedDict, defaultdict
import re


def has_non_english_chars(text):
    """Check if text contains non-English characters (excluding common punctuation/symbols)."""
    if not text:
        return False
    # Allow ASCII printable chars, common punctuation, and accented chars
    # Flag if we see CJK, Cyrillic, Arabic, etc.
    non_english_pattern = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\u0400-\u04ff\u0600-\u06ff\u0900-\u097f]')
    return bool(non_english_pattern.search(text))


def load_clservers_csv(csv_path, limit=None, filter_non_english=False, functionality_map=None):
    """
    Load server data from CLServers CSV.

    Args:
        csv_path: Path to CSV file
        limit: Maximum number of servers to load
        filter_non_english: If True, skip servers with non-English tool descriptions
        functionality_map: Optional dict {server_id: {tool_name: functionality_main}} for enriching tools

    Returns list of server dicts with selected columns.
    """
    servers = []
    filtered_count = 0
    filtered_too_few_tools = 0
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            server_id = row.get('server_id', '')

            # Extract tool columns (tool_01 through tool_99)
            tools = []
            has_non_english = False
            for i in range(1, 100):
                tool_name_col = f'tool_{i:02d}_name'
                tool_desc_col = f'tool_{i:02d}_description'

                if tool_name_col in row and row[tool_name_col]:
                    tool_name = row[tool_name_col]
                    tool_desc = row.get(tool_desc_col, '')

                    # Check for non-English characters in tool description
                    if filter_non_english and has_non_english_chars(tool_desc):
                        has_non_english = True
                        break

                    # Lookup functionality classification from CLTools data
                    functionality = ''
                    if functionality_map and server_id in functionality_map:
                        functionality = functionality_map[server_id].get(tool_name, '')

                    tools.append({
                        'name': tool_name,
                        'description': tool_desc,
                        'functionality': functionality
                    })

            # Skip this server if it has non-English tool descriptions
            if filter_non_english and has_non_english:
                filtered_count += 1
                continue

            # Skip servers with less than 2 tools
            if len(tools) < 2:
                filtered_too_few_tools += 1
                continue

            # Only include servers with tools and descriptions
            if tools and row.get('description'):
                servers.append({
                    'server_name': row.get('server_name', 'Unknown'),
                    'server_id': server_id,
                    'description': row.get('description', ''),
                    'tools': tools,
                    'naics_code': row.get('naics_code', ''),
                    'naics_title': row.get('naics_title', ''),
                    'is_finance_llm': row.get('is_finance_llm', ''),
                    'created_at': row.get('created_at', ''),
                    'stargazers_count': row.get('stargazers_count', '')
                })

            if limit and len(servers) >= limit:
                break

    if filter_non_english and filtered_count > 0:
        print(f"Filtered out {filtered_count} servers with non-English tool descriptions")
    if filtered_too_few_tools > 0:
        print(f"Filtered out {filtered_too_few_tools} servers with less than 2 tools")

    return servers


def load_cltools_csv(csv_path, limit=None):
    """
    Load individual tool data from CLTools CSV.

    Returns list of tool dicts with tool and server information.
    """
    tools = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tool_name = row.get('tool_name', '').strip()
            tool_desc = row.get('tool_description', '').strip()
            server_name = row.get('server_name', '').strip()
            server_desc = row.get('server_description', '').strip()

            # Only include tools with name and description
            if tool_name and tool_desc and server_name:
                tools.append({
                    'tool_id': row.get('tool_id', ''),
                    'tool_name': tool_name,
                    'tool_description': tool_desc,
                    'server_id': row.get('server_id', ''),
                    'server_name': server_name,
                    'server_description': server_desc,
                    'level1_cluster': row.get('level1_cluster', ''),
                    'level1_name': row.get('level1_name', ''),
                    'tool_functionality_main': row.get('tool_functionality_main', ''),
                    'tool_functionality_sub': row.get('tool_functionality_sub', '')
                })

            if limit and len(tools) >= limit:
                break

    return tools


def load_cltools_functionality_map(cltools_csv_path):
    """
    Load functionality classification from CLTools CSV into a fast lookup dictionary.

    Returns a nested dict: {server_id: {tool_name: functionality_main}}
    This enables quick lookup of autonomy level for any tool.
    """
    functionality_map = defaultdict(dict)

    with open(cltools_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            server_id = row.get('server_id', '').strip()
            tool_name = row.get('tool_name', '').strip()
            functionality_main = row.get('tool_functionality_main', '').strip()

            if server_id and tool_name:
                functionality_map[server_id][tool_name] = functionality_main

    return functionality_map


def build_onet_hierarchy(cltools_csv_path):
    """
    Build O*NET 3-level hierarchy from CLTools CSV.

    Returns:
        dict: {
            'l1': {cluster_id: {'name': str, 'l2_children': [cluster_ids]}},
            'l2': {cluster_id: {'name': str, 'parent_l1': str, 'tasks': [task_ids]}},
            'tasks': {task_id: {'name': str, 'parent_l2': str}}
        }
    """
    l1 = {}
    l2 = {}
    tasks = {}

    with open(cltools_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            l1_id = row.get('level1_cluster', '').strip()
            l1_name = row.get('level1_name', '').strip()
            l2_id = row.get('level2_cluster', '').strip()
            l2_name = row.get('level2_name', '').strip()
            task_id = row.get('task_id', '').strip()
            task_name = row.get('task_name', '').strip()

            # Skip rows with missing data
            if not all([l1_id, l1_name, l2_id, l2_name, task_id, task_name]):
                continue

            # Build L1
            if l1_id not in l1:
                l1[l1_id] = {'name': l1_name, 'l2_children': set()}
            l1[l1_id]['l2_children'].add(l2_id)

            # Build L2
            if l2_id not in l2:
                l2[l2_id] = {'name': l2_name, 'parent_l1': l1_id, 'tasks': set()}
            l2[l2_id]['tasks'].add(task_id)

            # Build tasks
            if task_id not in tasks:
                tasks[task_id] = {'name': task_name, 'parent_l2': l2_id}

    # Convert sets to sorted lists
    for l1_id in l1:
        l1[l1_id]['l2_children'] = sorted(l1[l1_id]['l2_children'])
    for l2_id in l2:
        l2[l2_id]['tasks'] = sorted(l2[l2_id]['tasks'])

    return {'l1': l1, 'l2': l2, 'tasks': tasks}


def build_functionality_hierarchy_from_csv(questions):
    """
    Build functionality 2-level hierarchy from questions CSV.

    Args:
        questions: OrderedDict from load_questions_with_choices()

    Returns:
        dict: {
            'main_categories': {'perception': 'text', 'reasoning': 'text', 'action': 'text'},
            'sub_categories': {
                'perception': [{'value': 'sensors', 'text': 'Sensors - e.g., ...'}],
                'reasoning': [...],
                'action': [...]
            }
        }
    """
    result = {
        'main_categories': {},
        'sub_categories': {}
    }

    # Extract Q2.1 (main categories)
    if 'q2.1' in questions:
        for choice in questions['q2.1']['choices']:
            result['main_categories'][choice['value']] = choice['text']

    # Extract Q2.2 subcategories for each main category
    for q_id in questions:
        if q_id.startswith('q2.2_'):
            main_cat = q_id.split('_')[1]  # e.g., 'perception', 'reasoning', 'action'
            if main_cat not in result['sub_categories']:
                result['sub_categories'][main_cat] = []

            for choice in questions[q_id]['choices']:
                result['sub_categories'][main_cat].append({
                    'value': choice['value'],
                    'text': choice['text']
                })

    return result


def load_questions_with_choices(questions_csv_path):
    """
    Load questions and their choices from CSV file.

    CSV format:
    question_id,question_text,choice_value,choice_text
    q1,Q1: Industry Generality...,1,1 - Cross-industry...

    Returns:
        OrderedDict of {question_id: {'text': str, 'choices': [{'value': str, 'text': str}]}}
        Includes ALL questions (q2.1, q2.2_*, q3, q4, q5, etc.)
    """
    questions = OrderedDict()

    with open(questions_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            q_id = row['question_id']
            q_text = row['question_text']
            choice_value = row['choice_value']
            choice_text = row['choice_text']

            if q_id not in questions:
                questions[q_id] = {
                    'text': q_text,
                    'choices': []
                }

            questions[q_id]['choices'].append({
                'value': choice_value,
                'text': choice_text
            })

    return questions


def escape_typescript_string(s):
    """Escape string for TypeScript."""
    if not s:
        return ''
    s = str(s)
    s = s.replace('\\', '\\\\')
    s = s.replace("'", "\\'")
    s = s.replace('\n', '\\n')
    s = s.replace('\r', '\\r')
    s = s.replace('`', '\\`')
    return s


def format_tools_array(tools):
    """
    Format tools as TypeScript array of strings, prioritizing by autonomy level.

    Selects up to 5 tools with highest autonomy:
    1. action (highest autonomy)
    2. reasoning (medium autonomy)
    3. perception (lowest autonomy)
    4. unclassified (no functionality data)

    If server has <5 tools, shows all tools.
    """
    if not tools:
        return "[]"

    # Define autonomy priority (lower number = higher priority)
    autonomy_priority = {
        'action': 1,
        'reasoning': 2,
        'perception': 3,
        '': 4  # unclassified/empty
    }

    # Sort tools by autonomy level (highest autonomy first)
    sorted_tools = sorted(tools, key=lambda t: autonomy_priority.get(t.get('functionality', ''), 4))

    # Select up to 5 tools
    selected_tools = sorted_tools[:5]

    tool_strings = []
    for tool in selected_tools:
        name = tool.get('name', '')
        desc = tool.get('description', '')
        tool_str = f"{name}: {desc}" if desc else name

        # Truncate if too long
        if len(tool_str) > 150:
            tool_str = tool_str[:147] + '...'

        escaped = escape_typescript_string(tool_str)
        tool_strings.append(f"'{escaped}'")

    return "[\n            " + ",\n            ".join(tool_strings) + "\n        ]"


def format_choices_array(choices):
    """Format choices as TypeScript array."""
    choice_strings = []
    for choice in choices:
        value = escape_typescript_string(str(choice['value']))
        text = escape_typescript_string(choice['text'])
        choice_strings.append(f"            {{ value: '{value}', text: '{text}' }}")

    return "[\n" + ",\n".join(choice_strings) + "\n        ]"


def generate_onet_data_structures(onet_hierarchy):
    """
    Generate TypeScript data structures for O*NET hierarchy.

    Returns TypeScript code string.
    """
    ts_code = "// O*NET Hierarchy Data Structures\n"
    ts_code += "var onetData = {\n"

    # L1 clusters
    ts_code += "    l1Clusters: {\n"
    for l1_id, l1_data in sorted(onet_hierarchy['l1'].items()):
        name_escaped = escape_typescript_string(l1_data['name'])
        ts_code += f"        '{l1_id}': '{name_escaped}',\n"
    ts_code += "    },\n\n"

    # L2 to L1 mapping
    ts_code += "    l2ToL1: {\n"
    for l2_id, l2_data in sorted(onet_hierarchy['l2'].items()):
        ts_code += f"        '{l2_id}': '{l2_data['parent_l1']}',\n"
    ts_code += "    },\n\n"

    # L2 clusters with names
    ts_code += "    l2Clusters: {\n"
    for l2_id, l2_data in sorted(onet_hierarchy['l2'].items()):
        name_escaped = escape_typescript_string(l2_data['name'])
        ts_code += f"        '{l2_id}': '{name_escaped}',\n"
    ts_code += "    },\n\n"

    # Tasks to L2 mapping
    ts_code += "    taskToL2: {\n"
    for task_id, task_data in sorted(onet_hierarchy['tasks'].items()):
        ts_code += f"        '{task_id}': '{task_data['parent_l2']}',\n"
    ts_code += "    },\n\n"

    # Tasks with names
    ts_code += "    tasks: {\n"
    for task_id, task_data in sorted(onet_hierarchy['tasks'].items()):
        name_escaped = escape_typescript_string(task_data['name'])
        ts_code += f"        '{task_id}': '{name_escaped}',\n"
    ts_code += "    }\n"

    ts_code += "};\n\n"

    # Helper functions
    ts_code += """// Get L2 clusters for a given L1 cluster
function getL2ClustersForL1(l1Id) {
    var l2Clusters = [];
    for (var l2Id in onetData.l2ToL1) {
        if (onetData.l2ToL1[l2Id] === l1Id) {
            l2Clusters.push({
                value: l2Id,
                text: onetData.l2Clusters[l2Id]
            });
        }
    }
    return l2Clusters;
}

// Get tasks for a given L2 cluster
function getTasksForL2(l2Id) {
    var tasks = [];
    for (var taskId in onetData.taskToL2) {
        if (onetData.taskToL2[taskId] === l2Id) {
            tasks.push({
                value: taskId,
                text: onetData.tasks[taskId]
            });
        }
    }
    return tasks;
}

"""

    return ts_code


def generate_functionality_data_structures(func_hierarchy):
    """
    Generate TypeScript data structures for functionality hierarchy from CSV data.

    Args:
        func_hierarchy: dict with 'main_categories' and 'sub_categories' keys

    Returns TypeScript code string.
    """
    ts_code = "// Functionality Hierarchy Data Structures\n"
    ts_code += "var functionalityData = {\n"

    # Main categories from CSV
    ts_code += "    mainCategories: {\n"
    for cat_value, cat_text in func_hierarchy['main_categories'].items():
        cat_text_escaped = cat_text.replace("'", "\\'")
        ts_code += f"        '{cat_value}': '{cat_text_escaped}',\n"
    ts_code += "    },\n\n"

    # Sub-categories from CSV
    ts_code += "    subCategories: {\n"
    for main_cat, sub_cats in func_hierarchy['sub_categories'].items():
        ts_code += f"        '{main_cat}': [\n"
        for sub_cat in sub_cats:
            value = sub_cat['value']
            text = sub_cat['text'].replace("'", "\\'")
            ts_code += f"            {{ value: '{value}', text: '{text}' }},\n"
        ts_code += "        ],\n"
    ts_code += "    }\n"

    ts_code += "};\n\n"

    # Helper function
    ts_code += """// Get sub-categories for a given main category
function getSubCategoriesFor(mainCategory) {
    return functionalityData.subCategories[mainCategory] || [];
}

"""

    return ts_code


def generate_instruction_page_content(all_onet_levels=False):
    """Generate the detailed instruction page content with examples.

    Args:
        all_onet_levels: If True, show all 3 O*NET levels. If False, only show level 1.
    """
    # O*NET section changes based on flag
    if all_onet_levels:
        onet_section = """### Q1: O*NET Occupational Classification (3 Levels)

**Q1.1 - Broad Category**: Choose the general occupational area (e.g., "Business Operations", "Computer/Mathematical")

**Q1.2 - Sub-Category**: Narrow down within your Q1.1 choice (e.g., within Computer/Mathematical: "Software Development")

**Q1.3 - Specific Task**: Select the precise task this server helps with (e.g., "Design and develop software applications")"""
        task_list = """* **Classify occupational function** (Q1.1-Q1.3) - What job task does it support? (Note: Many servers will be in the category 'Design, implement, and maintain diverse information technology systems')

* **Classify functionality type** (Q2.1-Q2.2) - How does it work?

* **Answer standard questions** (Q3-Q5) - Industry specificity, environment, and payment autonomy"""
    else:
        task_list = """* **Classify occupational function** (Q1.1) - What job task does it support? (Note: Many servers will be in the category 'Design, implement, and maintain diverse information technology systems')

* **Classify functionality type** (Q2.1-Q2.2) - How does it work?

* **Answer standard questions** (Q3-Q5) - Industry specificity, environment, and payment autonomy"""

    return f"""    {{
        type: 'instructions',
        title: 'Study Instructions',
        content: `# How to Classify MCP Servers

Thank you for participating! You will be evaluating MCP (Model Context Protocol) servers - tools that give AI agents specific capabilities.

## Your Task

For each server, you will:

{task_list}
* Based on our testing, the first servers will take you 2-10mins until you get used to it, then we expect each server to take around 1-1.5mins.
* Click **Next** to see example classifications, then you'll practice with feedback before starting the real study.`
    }},
"""


def generate_tool_instruction_page_content(all_onet_levels=False):
    """Generate the simplified instruction page content for tool classification.

    Tools study only includes Q1 (O*NET L1) and Q2 (Functionality) questions.
    No Q0, Q3-Q5 questions.
    """
    onet_desc = "2 questions" if not all_onet_levels else "2-4 questions"
    onet_detail = "* **O*NET Occupational Category** (Q1.1) - What job task does this tool support?" if not all_onet_levels else """* **O*NET Occupational Category** (Q1.1-Q1.3) - What job task does this tool support?
  - Q1.1: Broad category (e.g., 'Computer/Mathematical')
  - Q1.2: Sub-category (e.g., 'Software Development') - conditional
  - Q1.3: Specific task - conditional"""

    return f"""    {{
        type: 'instructions',
        title: 'Study Instructions',
        content: `# How to Classify MCP Tools

Thank you for participating! You will be evaluating individual MCP (Model Context Protocol) tools - specific functions within AI agent toolkits.

## Your Task

* For each tool, you will answer {onet_desc}:

{onet_detail}

* **Functionality Classification** (Q2.1-Q2.2) - How does this tool work?

* **Focus on the TOOL** - Not the entire server, just this specific tool's function

* Based on our testing, the first tools will take you 2-10mins until you get used to it, then we expect each tool to take around 1-1.5mins
* Click **Next** to see an example classification, then you'll start the study.`
    }},
"""


def generate_tutorial_practice_questions(server_name, description, tools, correct_answers, practice_num, total_practice, questions, all_questions):
    """
    Generate full question sequence for a practice tutorial server WITH feedback pages after each question.

    Args:
        server_name: Name of the practice server
        description: Server description
        tools: List of tool strings
        correct_answers: Dict with correct answers for all questions (should include text labels)
        practice_num: Practice number (2 or 3)
        total_practice: Total number of practices (usually 4)
        questions: OrderedDict of standard questions from CSV (q3, q4, q5, etc.)
        all_questions: All questions including q2.1 and q2.2_*

    Returns:
        String of TypeScript pages for this practice server
    """
    # Use special serverIndex for practice servers
    server_idx = f"'tutorial_{practice_num}'"

    # Format tools array
    tools_str = "[\n            " + ",\n            ".join([f"'{t}'" for t in tools]) + "\n        ]"

    pages = ""

    # Q1.1: O*NET Level 1 (tutorials always ONLY include level 1)
    onet_l1_correct = correct_answers.get('onet_l1', '')
    pages += f"""    // Tutorial Practice {practice_num}: Q1.1 - O*NET Level 1
    {{
        type: 'onet_l1',
        serverIndex: {server_idx},
        title: 'Tutorial Example {practice_num}/{total_practice} - Practice Question 1/6',
        serverName: '{server_name}',
        serverId: 'tutorial_{practice_num}',
        description: '{description}',
        tools: {tools_str},
        question: 'onet_l1',
        questionText: 'Q1.1: Which broad occupational category best describes the primary function of this server?',
        isPractice: true,
        correctAnswer: '{onet_l1_correct}'
    }},
    // Tutorial Practice {practice_num}: Q1.1 Feedback
    {{
        type: 'tutorial_feedback',
        serverIndex: {server_idx},
        title: 'Tutorial Example {practice_num}/{total_practice} - Feedback for Q1.1',
        questionKey: 'onet_l1',
        questionTitle: 'O*NET Level 1 - Broad Category',
        correctValue: '{onet_l1_correct}',
        feedbackTip: 'Focus on the PRIMARY occupational function. What job role would most commonly use this server? If it\\'s for general software tasks, choose Computer/Mathematical. If it\\'s for business operations, choose Business Operations.',
        isPractice: true
    }},
"""

    # Q2.1: Functionality Main (from CSV)
    func_main_correct = correct_answers.get('func_main', '')
    q21_text = all_questions['q2.1']['text'].replace("'", "\\'")
    q21_choices_list = []
    for choice in all_questions['q2.1']['choices']:
        value = choice['value'].replace("'", "\\'")
        text = choice['text'].replace("'", "\\'")
        q21_choices_list.append(f"            {{ value: '{value}', text: '{text}' }}")
    q21_choices_str = "[\n" + ",\n".join(q21_choices_list) + "\n        ]"

    pages += f"""    // Tutorial Practice {practice_num}: Q2.1 - Functionality Main
    {{
        type: 'func_main',
        serverIndex: {server_idx},
        title: 'Tutorial Example {practice_num}/{total_practice} - Practice Question 2/6',
        serverName: '{server_name}',
        serverId: 'tutorial_{practice_num}',
        description: '{description}',
        tools: {tools_str},
        question: 'func_main',
        questionText: '{q21_text}',
        choices: {q21_choices_str},
        isPractice: true,
        correctAnswer: '{func_main_correct}'
    }},
    // Tutorial Practice {practice_num}: Q2.1 Feedback
    {{
        type: 'tutorial_feedback',
        serverIndex: {server_idx},
        title: 'Tutorial Example {practice_num}/{total_practice} - Feedback for Q2.1',
        questionKey: 'func_main',
        questionTitle: 'Functionality Main - Primary Type',
        correctValue: '{func_main_correct}',
        feedbackTip: 'Perception = gathering data, Reasoning = processing/analyzing, Action = modifying/executing. Choose based on what the server PRIMARILY does.',
        isPractice: true
    }},
"""

    # Q2.2: Functionality Sub
    func_sub_correct = correct_answers.get('func_sub', '')
    pages += f"""    // Tutorial Practice {practice_num}: Q2.2 - Functionality Sub
    {{
        type: 'func_sub',
        serverIndex: {server_idx},
        title: 'Tutorial Example {practice_num}/{total_practice} - Practice Question 3/6',
        serverName: '{server_name}',
        serverId: 'tutorial_{practice_num}',
        description: '{description}',
        tools: {tools_str},
        question: 'func_sub',
        questionText: 'Q2.2: Which specific sub-category best describes this server\\'s functionality?',
        conditionalOn: 'func_main',
        isPractice: true,
        correctAnswer: '{func_sub_correct}'
    }},
    // Tutorial Practice {practice_num}: Q2.2 Feedback
    {{
        type: 'tutorial_feedback',
        serverIndex: {server_idx},
        title: 'Tutorial Example {practice_num}/{total_practice} - Feedback for Q2.2',
        questionKey: 'func_sub',
        questionTitle: 'Functionality Sub - Specific Mechanism',
        correctValue: '{func_sub_correct}',
        feedbackTip: 'Within your main category, identify the specific implementation mechanism. Consider how the server technically accomplishes its function.',
        isPractice: true
    }},
"""

    # Standard questions (Q3-Q5 from CSV) - tutorials only have 7 total questions now
    question_titles = {
        'q3': 'Industry Generality',
        'q4': 'Environment Generality',
        'q5': 'Payment Autonomy Level'
    }

    feedback_tips = {
        'q3': 'Cross-industry tools work across many sectors (like file managers). Industry-specific tools are designed for particular domains (like medical, finance, crypto).',
        'q4': 'Open/untrusted means generic internet/file access. Trusted means specific pre-configured APIs or internal systems.',
        'q5': 'Level 0 = no payment capability. Level 1 = read payment data. Level 2 = payment requests or links. Level 3 = third-party processing (e.g., Stripe, PayPal). Level 4 = direct execution (e.g., crypto, credit cards).'
    }

    for q_idx, (q_id, q_data) in enumerate(questions.items(), 4):
        q_text = q_data['text'].replace("'", "\\'")
        q_correct = correct_answers.get(q_id, '')

        # Format choices
        choices_list = []
        for choice in q_data['choices']:
            value = str(choice['value']).replace("'", "\\'")
            text = choice['text'].replace("'", "\\'")
            choices_list.append(f"            {{ value: '{value}', text: '{text}' }}")
        choices_str = "[\n" + ",\n".join(choices_list) + "\n        ]"

        q_title = question_titles.get(q_id, q_id.upper())
        q_tip = feedback_tips.get(q_id, 'Compare your answer with the correct answer.')

        pages += f"""    // Tutorial Practice {practice_num}: Question {q_idx}/6 - {q_id}
    {{
        type: 'server',
        serverIndex: {server_idx},
        title: 'Tutorial Example {practice_num}/{total_practice} - Practice Question {q_idx}/6',
        serverName: '{server_name}',
        serverId: 'tutorial_{practice_num}',
        description: '{description}',
        tools: {tools_str},
        question: '{q_id}',
        questionText: '{q_text}',
        choices: {choices_str},
        isPractice: true,
        correctAnswer: '{q_correct}'
    }},
    // Tutorial Practice {practice_num}: {q_id} Feedback
    {{
        type: 'tutorial_feedback',
        serverIndex: {server_idx},
        title: 'Tutorial Example {practice_num}/{total_practice} - Feedback for Q{q_idx-3}',
        questionKey: '{q_id}',
        questionTitle: '{q_title}',
        correctValue: '{q_correct}',
        feedbackTip: '{q_tip}',
        isPractice: true
    }},
"""

    return pages


def generate_tutorial_examples(questions, all_questions):
    """Generate tutorial example pages with full practice question sequences.

    Args:
        questions: Standard questions (q3, q4, q5, etc.)
        all_questions: All questions including q2.1 and q2.2_*
    """

    # Example 1: asher-mcp (Pre-answered)
    example1 = """    // Tutorial Example 1: Pre-answered (asher-mcp)
    {
        type: 'tutorial_intro',
        exampleNumber: 1,
        title: 'Tutorial Example 1/2 (Pre-answered)',
        content: `# Example 1: Pre-answered Classification

We'll show you a complete classification with all correct answers. Study this carefully to understand the reasoning.

* **Server**: asher-mcp

* **Description**: Financial data aggregation tool that connects to banking APIs to retrieve account information, balances, transactions, and investment holdings

* **Tools (sample)**:
* get_accounts: Retrieve list of all connected bank accounts
* get_account_balance: Get current balance for a specific account
* get_transactions: Retrieve transaction history for an account
* get_investment_holdings: View investment portfolio holdings

All these tools are **read-only** - they retrieve information but don't execute any actions.

Click **Next** to see the correct classifications.`
    },
    {
        type: 'tutorial_preanswered',
        exampleNumber: 1,
        title: 'Tutorial Example 1/2 - Correct Answers',
        serverName: 'asher-mcp',
        description: 'Financial data aggregation tool that connects to banking APIs to retrieve account information, balances, transactions, and investment holdings',
        tools: [
            'get_accounts: Retrieve list of all connected bank accounts',
            'get_account_balance: Get current balance for a specific account',
            'get_transactions: Retrieve transaction history for an account',
            'get_investment_holdings: View investment portfolio holdings'
        ],
        correctAnswers: {
            onet_l1: '13',
            onet_l1_text: 'Business and Financial Operations',
            onet_l1_explanation: `- PRIMARY function: retrieving and monitoring financial account information
- Maps directly to financial operations tasks
- Not general software development (that would be Computer/Mathematical)`,

            onet_l2: '13-2',
            onet_l2_text: 'Financial Specialists',
            onet_l2_explanation: `- Specifically handles financial data aggregation and account monitoring
- Aligns with financial specialist roles (not general business operations)
- Requires understanding of financial systems and banking APIs`,

            onet_task: '4.A.2.b.4',
            onet_task_text: 'Monitor financial data and prepare financial reports',
            onet_task_explanation: `- Core function is monitoring account balances, transactions, and holdings
- Provides data that feeds into financial reporting
- Read-only monitoring rather than active trading or execution`,

            func_main: 'perception',
            func_main_text: 'Perception (gathering information)',
            func_main_explanation: `- All tools are read-only data retrieval (no calculations, no actions)
- "Perception" = gathering information from external systems
- NOT "Action" because nothing is modified or executed`,

            func_sub: 'sensors',
            func_sub_text: 'Sensors',
            func_sub_explanation: `- Acts as sensors connecting to banking APIs
- Retrieves data without processing or transforming it
- No autonomous execution - just data collection`,

            q3: '0',
            q3_text: 'Industry-specific (finance)',
            q3_explanation: `- Only useful for finance sector
- Tools specifically designed for banking/financial accounts
- Not cross-industry like general file management tools`,

            q4: '0',
            q4_text: 'Trusted/pre-specified (specific banking APIs)',
            q4_explanation: `- Uses specific banking APIs (Plaid, Yodlee, etc.)
- Not open web scraping or generic file access
- Requires authentication to specific financial institutions`,

            q5: '1',
            q5_text: 'Information only (read-only payment data)',
            q5_explanation: `- Can VIEW payment data and transaction history
- Cannot initiate, generate, or execute payments
- Read-only access to financial information`
        }
    },
"""

    # Example 2: base-mcp (Practice with actual questions)
    example2_intro = """    // Tutorial Example 2: Practice intro (base-mcp)
    {
        type: 'tutorial_intro',
        exampleNumber: 2,
        title: 'Tutorial Example 2/2 (Practice)',
        content: `# Example 2: Practice Classification

Now it's your turn! Classify this server just as you would in the real study.

**Server**: base-mcp

**Description**: Blockchain interaction tool for Base network (Ethereum L2). Enables checking wallet balances, sending ETH/tokens, and deploying/interacting with smart contracts.

**Tools (sample)**:
* get_balance: Check wallet balance for ETH and tokens
* send_transaction: Send ETH or tokens to another address
* deploy_contract: Deploy new smart contracts to the blockchain
* call_contract: Execute functions on existing smart contracts

Notice the key difference from Example 1: this server can **EXECUTE** transactions, not just read data.

Click **Next** to start classifying (you'll answer all 6 questions).`
    },
"""

    # Generate all 6 question pages for Practice 2 (tutorials use only L1)
    example2_questions = generate_tutorial_practice_questions(
        server_name='base-mcp',
        description='Blockchain interaction tool for Base network (Ethereum L2). Enables checking wallet balances, sending ETH/tokens, and deploying/interacting with smart contracts.',
        tools=[
            'get_balance: Check wallet balance for ETH and tokens',
            'send_transaction: Send ETH or tokens to another address',
            'deploy_contract: Deploy new smart contracts to the blockchain',
            'call_contract: Execute functions on existing smart contracts'
        ],
        correct_answers={
            'onet_l1': 'L1_01',
            'onet_l2': 'L2_15-1',
            'onet_task': '4.A.4.a.6',
            'func_main': 'action',
            'func_sub': 'software_extensions',
            'q3': '0',
            'q4': '0',
            'q5': '4'
        },
        practice_num=2,
        total_practice=2,
        questions=questions,
        all_questions=all_questions
    )

    return example1 + example2_intro + example2_questions


def generate_tool_tutorial_example(all_questions):
    """Generate simplified tutorial example for tool classification.

    Args:
        all_questions: All questions including q2.1 and q2.2_*

    Returns:
        String of TypeScript pages for tool tutorial (2 examples: 1 pre-answered + 1 practice with Q1.1 + Q2.1/Q2.2)
    """
    # Example 1: get_account_balance from asher-mcp (Pre-answered)
    example1 = """    // Tool Tutorial Example 1: Pre-answered (get_account_balance)
    {
        type: 'tutorial_intro',
        exampleNumber: 1,
        title: 'Tutorial Example 1/2 (Pre-answered)',
        content: `# Example 1: Pre-answered Tool Classification

We'll show you how to classify a single tool. Study this example to understand the reasoning.

* **Tool**: get_account_balance

* **Tool Description**: Retrieves the current balance for a specific bank account by account ID

* **Server Context**: This tool is part of asher-mcp, a financial data aggregation server that connects to banking APIs

* **Key Point**: Focus on what THIS TOOL does (retrieve account balance), not what the overall server does (aggregate all financial data).

* Click **Next** to see the correct classification.`
    },
    {
        type: 'tutorial_preanswered',
        exampleNumber: 1,
        title: 'Tutorial Example 1/2 - Correct Answers',
        toolName: 'get_account_balance',
        toolDescription: 'Retrieves the current balance for a specific bank account by account ID',
        serverName: 'asher-mcp',
        serverDescription: 'Financial data aggregation server that connects to banking APIs',
        correctAnswers: {
            onet_l1: '13',
            onet_l1_text: 'Business and Financial Operations',
            onet_l1_explanation: `- This tool retrieves financial account balance information
- Maps directly to financial operations tasks
- Used by financial professionals to monitor accounts`,

            func_main: 'perception',
            func_main_text: 'Perception (gathering information)',
            func_main_explanation: `- This tool is read-only data retrieval
- "Perception" = gathering information from external systems (banking APIs)
- NOT "Action" because nothing is modified or executed`,

            func_sub: 'sensors',
            func_sub_text: 'Sensors',
            func_sub_explanation: `- Acts as a sensor connecting to banking APIs
- Retrieves balance data without processing or transforming it
- Simple data collection`
        }
    },
"""

    # Example 2: send_transaction tool (Practice)
    example2_intro = """    // Tool Tutorial Example 2: Practice (send_transaction)
    {
        type: 'tutorial_intro',
        exampleNumber: 2,
        title: 'Tutorial Example 2/2 (Practice)',
        content: `# Example 2: Practice Tool Classification

Now it's your turn! Classify this tool as you would in the real study.

* **Tool**: send_transaction

* **Tool Description**: Sends ETH or ERC-20 tokens to another wallet address on the blockchain

* **Server Context**: This tool is part of base-mcp, a blockchain interaction server for the Base network (Ethereum L2)

* Click **Next** to start classifying.`
    },
"""

    # Generate Q1.1, Q2.1, Q2.2 for practice tool
    # Get Q2.1 choices
    q21_choices_list = []
    for choice in all_questions['q2.1']['choices']:
        value = choice['value'].replace("'", "\\'")
        text = choice['text'].replace("'", "\\'")
        q21_choices_list.append(f"            {{ value: '{value}', text: '{text}' }}")
    q21_choices_str = "[\n" + ",\n".join(q21_choices_list) + "\n        ]"
    q21_text = all_questions['q2.1']['text'].replace("'", "\\'")

    example2_questions = f"""    // Tool Tutorial Practice: Q1.1 - O*NET Level 1
    {{
        type: 'onet_l1',
        toolIndex: 'tutorial_tool_2',
        title: 'Tutorial Example 2/2 - Practice Question 1/3',
        toolName: 'send_transaction',
        toolId: 'tutorial_tool_2',
        toolDescription: 'Sends ETH or ERC-20 tokens to another wallet address on the blockchain',
        serverName: 'base-mcp',
        serverDescription: 'Blockchain interaction server for the Base network (Ethereum L2)',
        question: 'onet_l1',
        questionText: 'Q1.1: Which broad occupational category best describes what this tool does?',
        isPractice: true,
        correctAnswer: 'L1_01'
    }},
    // Tool Tutorial Practice: Q1.1 Feedback
    {{
        type: 'tutorial_feedback',
        toolIndex: 'tutorial_tool_2',
        title: 'Tutorial Example 2/2 - Feedback for Q1.1',
        questionKey: 'onet_l1',
        questionTitle: 'O*NET Level 1 - Broad Category',
        correctValue: 'L1_01',
        feedbackTip: 'Blockchain/crypto tools typically fall under Computer/Mathematical since they involve software systems and computational operations.',
        isPractice: true
    }},
    // Tool Tutorial Practice: Q2.1 - Functionality Main
    {{
        type: 'func_main',
        toolIndex: 'tutorial_tool_2',
        title: 'Tutorial Example 2/2 - Practice Question 2/3',
        toolName: 'send_transaction',
        toolId: 'tutorial_tool_2',
        toolDescription: 'Sends ETH or ERC-20 tokens to another wallet address on the blockchain',
        serverName: 'base-mcp',
        serverDescription: 'Blockchain interaction server for the Base network (Ethereum L2)',
        question: 'func_main',
        questionText: '{q21_text}',
        choices: {q21_choices_str},
        isPractice: true,
        correctAnswer: 'action'
    }},
    // Tool Tutorial Practice: Q2.1 Feedback
    {{
        type: 'tutorial_feedback',
        toolIndex: 'tutorial_tool_2',
        title: 'Tutorial Example 2/2 - Feedback for Q2.1',
        questionKey: 'func_main',
        questionTitle: 'Functionality Main - Primary Type',
        correctValue: 'action',
        feedbackTip: 'This tool SENDS transactions (executes actions that modify blockchain state). Perception = reading data, Reasoning = processing, Action = executing/modifying.',
        isPractice: true
    }},
    // Tool Tutorial Practice: Q2.2 - Functionality Sub
    {{
        type: 'func_sub',
        toolIndex: 'tutorial_tool_2',
        title: 'Tutorial Example 2/2 - Practice Question 3/3',
        toolName: 'send_transaction',
        toolId: 'tutorial_tool_2',
        toolDescription: 'Sends ETH or ERC-20 tokens to another wallet address on the blockchain',
        serverName: 'base-mcp',
        serverDescription: 'Blockchain interaction server for the Base network (Ethereum L2)',
        question: 'func_sub',
        questionText: 'Q2.2: Which specific sub-category best describes this tool\\'s functionality?',
        conditionalOn: 'func_main',
        isPractice: true,
        correctAnswer: 'software_extensions'
    }},
    // Tool Tutorial Practice: Q2.2 Feedback
    {{
        type: 'tutorial_feedback',
        toolIndex: 'tutorial_tool_2',
        title: 'Tutorial Example 2/2 - Feedback for Q2.2',
        questionKey: 'func_sub',
        questionTitle: 'Functionality Sub - Specific Mechanism',
        correctValue: 'software_extensions',
        feedbackTip: 'Blockchain interactions extend software capabilities by interfacing with external blockchain networks. This is a software extension rather than direct code execution or computer use.',
        isPractice: true
    }},
"""

    return example1 + example2_intro + example2_questions


def generate_main_ts(servers, onet_hierarchy, func_hierarchy, questions, all_questions, output_path, num_servers=5, all_onet_levels=False):
    """Generate main.ts file content with hierarchical validation questions.

    Args:
        servers: List of server data
        onet_hierarchy: O*NET hierarchy data structure
        func_hierarchy: Functionality hierarchy data structure (from CSV)
        questions: Standard questions configuration (q3, q4, q5, etc.)
        all_questions: All questions including q2.1 and q2.2_*
        output_path: Path to output file
        num_servers: Number of servers to sample
        all_onet_levels: If True, include all 3 O*NET levels (L1, L2, task). If False, only L1.
    """

    # Sample random servers
    if len(servers) > num_servers:
        servers = random.sample(servers, num_servers)
    else:
        servers = servers[:num_servers]

    # Start building output
    output = """// MCP Classification Study - With CLTools Validation Questions
// Generated dynamically from CLServers CSV and CLTools hierarchy
// DO NOT EDIT MANUALLY - Regenerate using generate_main_ts.py

// import gorilla = require("gorilla/gorilla");  // Commented out for browser compatibility

"""

    # Add O*NET and functionality data structures
    output += generate_onet_data_structures(onet_hierarchy)
    output += generate_functionality_data_structures(func_hierarchy)

    # Start studyPages array
    output += """// Study data - pages with hierarchical validation questions
var studyPages = [
"""

    # Add instruction page as first page (pass all_onet_levels flag)
    output += generate_instruction_page_content(all_onet_levels)

    # Add tutorial examples after instructions (pass questions for practice pages)
    output += generate_tutorial_examples(questions, all_questions)

    # Generate pages for each server
    # Now using single combined page per server
    total_questions_per_server = 1  # All questions on one page

    for server_idx, server in enumerate(servers, 1):
        server_name = escape_typescript_string(server['server_name'])
        description = escape_typescript_string(server['description'])
        tools = server['tools']
        server_id = server['server_id']

        # Build all questions data for the combined page
        # Q2.1 choices
        q21_text = escape_typescript_string(all_questions['q2.1']['text'])
        q21_choices = format_choices_array(all_questions['q2.1']['choices'])

        # Standard questions (q3, q4, q5)
        standard_questions_data = []
        for q_id, q_data in questions.items():
            standard_questions_data.append(f"""{{
                questionId: '{q_id}',
                questionText: '{escape_typescript_string(q_data['text'])}',
                choices: {format_choices_array(q_data['choices'])}
            }}""")

        standard_questions_str = ",\n            ".join(standard_questions_data)

        # Create single combined page with all questions
        output += f"""    // Server {server_idx} - Combined page with all questions
    {{
        type: 'server_combined',
        serverIndex: {server_idx},
        title: 'Server {server_idx}/{len(servers)}',
        serverName: '{server_name}',
        serverId: '{server_id}',
        description: '{description}',
        tools: {format_tools_array(tools)},
        allOnetLevels: {str(all_onet_levels).lower()},
        q21Text: '{q21_text}',
        q21Choices: {q21_choices},
        standardQuestions: [
            {standard_questions_str}
        ]
    }},
"""

    # Add completion page
    output += """    {
        type: 'completion',
        title: 'Thank You',
        content: `# Study Complete!

Thank you for participating. Your responses have been recorded.

Click **Finish** to complete.`
    }
];

"""

    # Add the rest of the TypeScript code (state management, page rendering, etc.)
    # This will be added in the next part due to length

    return output


def generate_main_tools_ts(tools, onet_hierarchy, func_hierarchy, all_questions, output_path, num_tools=10, all_onet_levels=False):
    """Generate main_tools.ts file content for tool classification (simplified study).

    Args:
        tools: List of tool data from CLTools CSV
        onet_hierarchy: O*NET hierarchy data structure
        func_hierarchy: Functionality hierarchy data structure (from CSV)
        all_questions: All questions including q2.1 and q2.2_*
        output_path: Path to output file
        num_tools: Number of tools to sample

    Note: Tools study only includes Q1.1 (O*NET L1) + Q2.1/Q2.2 (Functionality).
          No Q0 (analysis notes), no Q3-Q5 (standard questions).
          With all_onet_levels=True, adds Q1.2 (L2) and Q1.3 (Task) questions.
    """

    # Sample random tools
    if len(tools) > num_tools:
        tools = random.sample(tools, num_tools)
    else:
        tools = tools[:num_tools]

    # Start building output
    output = """// MCP Tool Classification Study - Simplified (Q1.1 + Q2.1/Q2.2 only)
// Generated dynamically from CLTools CSV
// DO NOT EDIT MANUALLY - Regenerate using generate_main_ts.py --type tools

// import gorilla = require("gorilla/gorilla");  // Commented out for browser compatibility

"""

    # Add O*NET and functionality data structures
    output += generate_onet_data_structures(onet_hierarchy)
    output += generate_functionality_data_structures(func_hierarchy)

    # Start studyPages array
    output += """// Study data - pages with simplified tool classification
var studyPages = [
"""

    # Add instruction page (tool-specific)
    output += generate_tool_instruction_page_content(all_onet_levels)

    # Add tutorial example (tools tutorial)
    output += generate_tool_tutorial_example(all_questions)

    # Generate pages for each tool (only Q1.1 + Q2.1/Q2.2) - using combined page
    total_questions_per_tool = 1  # All questions on one page

    for tool_idx, tool in enumerate(tools, 1):
        tool_name = escape_typescript_string(tool['tool_name'])
        tool_desc = escape_typescript_string(tool['tool_description'])
        server_name = escape_typescript_string(tool['server_name'])
        server_desc = escape_typescript_string(tool['server_description'])
        tool_id = tool['tool_id']

        # Q2.1 choices
        q21_text = escape_typescript_string(all_questions['q2.1']['text'])
        q21_choices = format_choices_array(all_questions['q2.1']['choices'])

        # Create single combined page with all questions for tools
        output += f"""    // Tool {tool_idx} - Combined page with all questions
    {{
        type: 'tool_combined',
        toolIndex: {tool_idx},
        title: 'Tool {tool_idx}/{len(tools)}',
        toolName: '{tool_name}',
        toolId: '{tool_id}',
        toolDescription: '{tool_desc}',
        serverName: '{server_name}',
        serverDescription: '{server_desc}',
        q21Text: '{q21_text}',
        q21Choices: {q21_choices},
        allOnetLevels: {str(all_onet_levels).lower()}
    }},
"""

    # Add completion page
    output += """    {
        type: 'completion',
        title: 'Thank You',
        content: `# Study Complete!

Thank you for participating. Your responses have been recorded.

Click **Finish** to complete.`
    }
];

"""

    # Add allOnetLevels configuration variable
    output += f"""// Configuration
var allOnetLevels = {str(all_onet_levels).lower()};

"""

    return output


# Continuation function for the TypeScript runtime code
def generate_typescript_runtime():
    """Generate the TypeScript runtime code for page rendering and state management."""

    return """// State
var currentPage = 0;
var responses = {};
var serverResponses = {};  // Track responses per server/tool: {index: {question: answer}}

// Main entry point
gorilla.ready(function() {
    console.log("Study ready");
    // Initialize serverResponses for both servers and tools
    for (var i = 0; i < studyPages.length; i++) {
        var page = studyPages[i];
        var idx = page.serverIndex || page.toolIndex;
        if (idx && !serverResponses[idx]) {
            serverResponses[idx] = {};
        }
    }
    showPage(0);
});

// Show a page
function showPage(pageIndex) {
    currentPage = pageIndex;
    var page = studyPages[pageIndex];

    // Auto-skip func_sub pages when func_main is 'perception'
    if (page.type === 'func_sub') {
        var idx = page.serverIndex || page.toolIndex;
        var funcMain = serverResponses[idx] && serverResponses[idx]['func_main'];

        if (funcMain === 'perception') {
            // Auto-fill with 'sensors' and skip to next page
            serverResponses[idx]['func_sub'] = 'sensors';
            var nameKey = page.serverName || page.toolName;
            var responseKey = nameKey + '_func_sub';
            responses[responseKey] = 'sensors';
            console.log('Auto-skipped func_sub for perception, set to sensors');

            // Continue to next page
            showPage(pageIndex + 1);
            return;
        }
    }

    // Clear screen
    $('#gorilla').empty();

    // Show progress
    var progress = ((pageIndex + 1) / studyPages.length) * 100;
    $('#gorilla').append(`
        <div class="progress-bar-container">
            <div class="progress-bar" style="width: ${progress}%"></div>
            <div class="progress-text">Page ${pageIndex + 1} of ${studyPages.length}</div>
        </div>
    `);

    // Show page content based on type
    if (page.type === 'completion' || page.type === 'instructions') {
        showTextPage(page);
    } else if (page.type === 'tutorial_intro') {
        showTextPage(page);
    } else if (page.type === 'tutorial_preanswered') {
        showTutorialPreansweredPage(page, pageIndex);
    } else if (page.type === 'tutorial_practice') {
        showTutorialPracticePage(page, pageIndex);
    } else if (page.type === 'tutorial_feedback') {
        showTutorialFeedbackPage(page, pageIndex);
    } else if (page.type === 'server_combined') {
        showServerCombinedPage(page, pageIndex);
    } else if (page.type === 'tool_combined') {
        showToolCombinedPage(page, pageIndex);
    } else if (page.type === 'onet_l1') {
        showONetL1Page(page, pageIndex);
    } else if (page.type === 'onet_l2') {
        showONetL2Page(page, pageIndex);
    } else if (page.type === 'onet_task') {
        showONetTaskPage(page, pageIndex);
    } else if (page.type === 'func_main') {
        showFuncMainPage(page, pageIndex);
    } else if (page.type === 'func_sub') {
        showFuncSubPage(page, pageIndex);
    } else if (page.type === 'server') {
        showServerQuestionPage(page, pageIndex);
    }

    // Show navigation
    showNavigation(pageIndex);
}

// Show text page (completion/instructions/tutorial_intro)
function showTextPage(page) {
    // Instructions and tutorial intro pages need markdown conversion
    var needsMarkdown = (page.type === 'instructions' || page.type === 'tutorial_intro');
    var content = needsMarkdown ? markdownToHtml(page.content) : page.content;

    $('#gorilla').append(`
        <div class="page-container">
            <h1>${page.title}</h1>
            <div class="content">${content}</div>
        </div>
    `);
}

// Convert markdown to HTML (basic conversion)
function markdownToHtml(markdown) {
    var html = markdown;

    // Convert headers
    html = html.replace(/^#### (.*$)/gim, '<h4>$1</h4>');
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

    // Convert bold
    html = html.replace(/\\*\\*(.*?)\\*\\*/gim, '<strong>$1</strong>');

    // Convert lists
    html = html.replace(/^\\* (.*$)/gim, '<li>$1</li>');
    html = html.replace(/(<li>.*<\\/li>)/gim, '<ul>$1</ul>');

    // Convert paragraphs (split by double newlines)
    var paragraphs = html.split('\\\\n\\\\n');
    html = paragraphs.map(p => {
        p = p.trim();
        // Don't wrap if already has HTML tags
        if (p.startsWith('<')) return p;
        // Replace single newlines with <br>
        p = p.replace(/\\\\n/g, '<br>');
        return '<p>' + p + '</p>';
    }).join('\\\\n');

    return html;
}

// Show server info block (reusable)
function getServerInfoHtml(page) {
    // Check if this is a tool page (has toolName) or server page (has tools array)
    if (page.toolName) {
        // Tool page
        return `
            <div class="server-info">
                <h2>Tool Information</h2>
                <p><strong>Tool Name:</strong> ${page.toolName}</p>
                <p><strong>Tool Description:</strong> ${page.toolDescription || page.description}</p>
                <p><strong>Server:</strong> ${page.serverName}</p>
                <p><strong>Server Description:</strong> ${page.serverDescription}</p>
            </div>
        `;
    } else {
        // Server page
        var toolsList = page.tools ? page.tools.map(t => `<li>${t}</li>`).join('') : '';
        return `
            <div class="server-info">
                <h2>Server Information</h2>
                <p><strong>Name:</strong> ${page.serverName}</p>
                <p><strong>Description:</strong> ${page.description}</p>
                <p><strong>Tools (sample):</strong></p>
                <ul>${toolsList}</ul>
            </div>
        `;
    }
}

// Show tutorial pre-answered page
function showTutorialPreansweredPage(page, pageIndex) {
    var answers = page.correctAnswers;
    var answersHtml = '';

    // Q1.1: O*NET Broad Category (both studies)
    answersHtml += `
        <div class="answer-block" style="background: #e8f5e9; padding: 10px; margin: 10px 0; border-left: 4px solid #4caf50;">
            <strong>Q1.1: O*NET Broad Category</strong><br>
            ✓ ${answers.onet_l1_text}
        </div>
        <div class="explanation-block" style="background: #fff3e0; padding: 10px; margin: 0 0 15px 0; border-left: 4px solid #ff9800;">
            <strong>Why this answer?</strong><br>
            ${markdownToHtml(answers.onet_l1_explanation)}
        </div>
    `;

    // Q2.1: Functionality Main (both studies)
    answersHtml += `
        <div class="answer-block" style="background: #e8f5e9; padding: 10px; margin: 10px 0; border-left: 4px solid #4caf50;">
            <strong>Q2.1: Functionality Main</strong><br>
            ✓ ${answers.func_main_text}
        </div>
        <div class="explanation-block" style="background: #fff3e0; padding: 10px; margin: 0 0 15px 0; border-left: 4px solid #ff9800;">
            <strong>Why this answer?</strong><br>
            ${markdownToHtml(answers.func_main_explanation)}
        </div>
    `;

    // Q2.2: Functionality Sub (both studies)
    answersHtml += `
        <div class="answer-block" style="background: #e8f5e9; padding: 10px; margin: 10px 0; border-left: 4px solid #4caf50;">
            <strong>Q2.2: Functionality Sub</strong><br>
            ✓ ${answers.func_sub_text}
        </div>
        <div class="explanation-block" style="background: #fff3e0; padding: 10px; margin: 0 0 15px 0; border-left: 4px solid #ff9800;">
            <strong>Why this answer?</strong><br>
            ${markdownToHtml(answers.func_sub_explanation)}
        </div>
    `;

    // Q3, Q4, Q5: Standard questions (only for servers study)
    if (answers.q3_text) {
        answersHtml += `
            <div class="answer-block" style="background: #e8f5e9; padding: 10px; margin: 10px 0; border-left: 4px solid #4caf50;">
                <strong>Q3: Industry Generality</strong><br>
                ✓ ${answers.q3_text}
            </div>
            <div class="explanation-block" style="background: #fff3e0; padding: 10px; margin: 0 0 15px 0; border-left: 4px solid #ff9800;">
                <strong>Why this answer?</strong><br>
                ${markdownToHtml(answers.q3_explanation)}
            </div>
        `;
    }

    if (answers.q4_text) {
        answersHtml += `
            <div class="answer-block" style="background: #e8f5e9; padding: 10px; margin: 10px 0; border-left: 4px solid #4caf50;">
                <strong>Q4: Environment Generality</strong><br>
                ✓ ${answers.q4_text}
            </div>
            <div class="explanation-block" style="background: #fff3e0; padding: 10px; margin: 0 0 15px 0; border-left: 4px solid #ff9800;">
                <strong>Why this answer?</strong><br>
                ${markdownToHtml(answers.q4_explanation)}
            </div>
        `;
    }

    if (answers.q5_text) {
        answersHtml += `
            <div class="answer-block" style="background: #e8f5e9; padding: 10px; margin: 10px 0; border-left: 4px solid #4caf50;">
                <strong>Q5: Payment Autonomy Level</strong><br>
                ✓ ${answers.q5_text}
            </div>
            <div class="explanation-block" style="background: #fff3e0; padding: 10px; margin: 0 0 15px 0; border-left: 4px solid #ff9800;">
                <strong>Why this answer?</strong><br>
                ${markdownToHtml(answers.q5_explanation)}
            </div>
        `;
    }

    $('#gorilla').append(`
        <div class="page-container">
            <h1>${page.title}</h1>
            ${getServerInfoHtml(page)}
            <div class="tutorial-answers">
                <h3>Correct Classifications:</h3>
                ${answersHtml}
            </div>
        </div>
    `);
}

// Show tutorial practice page
function showTutorialPracticePage(page, pageIndex) {
    $('#gorilla').append(`
        <div class="page-container">
            <h1>${page.title}</h1>
            <p style="background: #e3f2fd; padding: 10px; border-left: 4px solid #2196f3;">
                <strong>Practice Mode:</strong> Answer the questions as you normally would.
            </p>
            ${getServerInfoHtml(page)}
            <p><em>Click Next to begin classifying this server.</em></p>
        </div>
    `);
}

// Show tutorial feedback page
function showTutorialFeedbackPage(page, pageIndex) {
    var idx = page.serverIndex || page.toolIndex;  // Support both servers and tools
    var questionKey = page.questionKey;

    console.log('Feedback page - idx:', idx, 'questionKey:', questionKey);
    console.log('serverResponses[idx]:', serverResponses[idx]);

    var userAnswer = serverResponses[idx] ? serverResponses[idx][questionKey] : undefined;
    var correctValue = page.correctValue;

    console.log('userAnswer:', userAnswer, 'correctValue:', correctValue);

    var isCorrect = (userAnswer === correctValue);

    // Get text labels for ONET and functionality questions
    var userAnswerText = userAnswer;
    var correctAnswerText = correctValue;

    if (questionKey === 'onet_l1') {
        userAnswerText = onetData.l1Clusters[userAnswer] || userAnswer;
        correctAnswerText = onetData.l1Clusters[correctValue] || correctValue;
    } else if (questionKey === 'onet_l2') {
        userAnswerText = onetData.l2Clusters[userAnswer] || userAnswer;
        correctAnswerText = onetData.l2Clusters[correctValue] || correctValue;
    } else if (questionKey === 'onet_task') {
        userAnswerText = onetData.tasks[userAnswer] || userAnswer;
        correctAnswerText = onetData.tasks[correctValue] || correctValue;
    } else if (questionKey === 'func_main') {
        userAnswerText = functionalityData.mainCategories[userAnswer] || userAnswer;
        correctAnswerText = functionalityData.mainCategories[correctValue] || correctValue;
    } else if (questionKey === 'func_sub') {
        // Find subcategory text from functionalityData
        for (var mainCat in functionalityData.subCategories) {
            var subCats = functionalityData.subCategories[mainCat];
            for (var i = 0; i < subCats.length; i++) {
                if (subCats[i].value === userAnswer) {
                    userAnswerText = subCats[i].text;
                }
                if (subCats[i].value === correctValue) {
                    correctAnswerText = subCats[i].text;
                }
            }
        }
    } else if (questionKey === 'q3' || questionKey === 'q4' || questionKey === 'q5') {
        // For standard questions, find the preceding question page to get choices
        var questionPage = studyPages[pageIndex - 1];
        if (questionPage && questionPage.choices) {
            for (var j = 0; j < questionPage.choices.length; j++) {
                if (questionPage.choices[j].value === userAnswer) {
                    userAnswerText = questionPage.choices[j].text;
                }
                if (questionPage.choices[j].value === correctValue) {
                    correctAnswerText = questionPage.choices[j].text;
                }
            }
        }
    }

    var feedbackColor = isCorrect ? '#4caf50' : '#f44336';
    var feedbackBg = isCorrect ? '#e8f5e9' : '#ffebee';
    var feedbackIcon = isCorrect ? '✓' : '✗';
    var feedbackText = isCorrect ? 'Correct!' : 'Not quite right';

    var yourAnswerHtml = `<p style="background: ${feedbackBg}; padding: 10px; margin: 10px 0; border-left: 4px solid ${feedbackColor};">
               <strong>${feedbackIcon} Your answer:</strong> ${userAnswerText}
           </p>`;

    var correctAnswerHtml = isCorrect
            ? ''
            : `<p style="background: #e8f5e9; padding: 10px; margin: 10px 0; border-left: 4px solid #4caf50;">
                   <strong>✓ Correct answer:</strong> ${correctAnswerText}
               </p>`;

    $('#gorilla').append(`
        <div class="page-container">
            <h1>${page.title}</h1>
            <div style="background: ${feedbackBg}; padding: 15px; margin: 20px 0; border-left: 4px solid ${feedbackColor}; border-radius: 4px;">
                <h2 style="color: ${feedbackColor}; margin-top: 0;">${feedbackIcon} ${feedbackText}</h2>
                <h3>${page.questionTitle}</h3>
            </div>

            ${yourAnswerHtml}
            ${correctAnswerHtml}

            <div style="background: #fff3e0; padding: 15px; margin: 20px 0; border-left: 4px solid #ff9800; border-radius: 4px;">
                <h4 style="margin-top: 0;">💡 Tip:</h4>
                <p>${page.feedbackTip}</p>
            </div>

            <p style="text-align: center; margin-top: 30px;">
                <em>Click Next to continue with the practice questions.</em>
            </p>
        </div>
    `);
}

// Show O*NET Level 1 page
function showONetL1Page(page, pageIndex) {
    var choices = [];
    for (var l1Id in onetData.l1Clusters) {
        choices.push({
            value: l1Id,
            text: onetData.l1Clusters[l1Id]
        });
    }

    var choicesHtml = choices.map(choice => `
        <label>
            <input type="radio" name="question_${pageIndex}" value="${choice.value}">
            ${choice.text}
        </label>
    `).join('');

    $('#gorilla').append(`
        <div class="page-container">
            <h1>${page.title}</h1>
            ${getServerInfoHtml(page)}
            <div class="question-block">
                <p class="question-text">${page.questionText} <span class="required">*</span></p>
                ${choicesHtml}
            </div>
        </div>
    `);
}

// Show O*NET Level 2 page (conditional on L1)
function showONetL2Page(page, pageIndex) {
    // Get previous L1 selection
    var serverIdx = page.serverIndex;
    var l1Selection = serverResponses[serverIdx]['onet_l1'];

    if (!l1Selection) {
        $('#gorilla').append(`
            <div class="page-container">
                <h1>Error</h1>
                <p>Missing L1 selection. Please go back.</p>
            </div>
        `);
        return;
    }

    var choices = getL2ClustersForL1(l1Selection);

    var choicesHtml = choices.map(choice => `
        <label>
            <input type="radio" name="question_${pageIndex}" value="${choice.value}">
            ${choice.text}
        </label>
    `).join('');

    $('#gorilla').append(`
        <div class="page-container">
            <h1>${page.title}</h1>
            ${getServerInfoHtml(page)}
            <div class="question-block">
                <p class="question-text">${page.questionText} <span class="required">*</span></p>
                <p><em>Based on your L1 selection: ${onetData.l1Clusters[l1Selection]}</em></p>
                ${choicesHtml}
            </div>
        </div>
    `);
}

// Show O*NET Task page (conditional on L2)
function showONetTaskPage(page, pageIndex) {
    // Get previous L2 selection
    var serverIdx = page.serverIndex;
    var l2Selection = serverResponses[serverIdx]['onet_l2'];

    if (!l2Selection) {
        $('#gorilla').append(`
            <div class="page-container">
                <h1>Error</h1>
                <p>Missing L2 selection. Please go back.</p>
            </div>
        `);
        return;
    }

    var choices = getTasksForL2(l2Selection);

    var choicesHtml = choices.map(choice => `
        <label>
            <input type="radio" name="question_${pageIndex}" value="${choice.value}">
            ${choice.text}
        </label>
    `).join('');

    $('#gorilla').append(`
        <div class="page-container">
            <h1>${page.title}</h1>
            ${getServerInfoHtml(page)}
            <div class="question-block">
                <p class="question-text">${page.questionText} <span class="required">*</span></p>
                <p><em>Based on your L2 selection: ${onetData.l2Clusters[l2Selection]}</em></p>
                ${choicesHtml}
            </div>
        </div>
    `);
}

// Show functionality main category page
function showFuncMainPage(page, pageIndex) {
    var choicesHtml = page.choices.map(choice => `
        <label>
            <input type="radio" name="question_${pageIndex}" value="${choice.value}">
            ${choice.text}
        </label>
    `).join('');

    $('#gorilla').append(`
        <div class="page-container">
            <h1>${page.title}</h1>
            ${getServerInfoHtml(page)}
            <div class="question-block">
                <p class="question-text">${page.questionText} <span class="required">*</span></p>
                ${choicesHtml}
            </div>
        </div>
    `);
}

// Show functionality sub-category page (conditional on main)
function showFuncSubPage(page, pageIndex) {
    // Get previous main category selection
    var idx = page.serverIndex || page.toolIndex;  // Support both servers and tools
    var mainSelection = serverResponses[idx] ? serverResponses[idx]['func_main'] : undefined;

    if (!mainSelection) {
        $('#gorilla').append(`
            <div class="page-container">
                <h1>Error</h1>
                <p>Missing main functionality selection. Please go back.</p>
            </div>
        `);
        return;
    }

    var choices = getSubCategoriesFor(mainSelection);

    var choicesHtml = choices.map(choice => `
        <label>
            <input type="radio" name="question_${pageIndex}" value="${choice.value}">
            ${choice.text}
        </label>
    `).join('');

    $('#gorilla').append(`
        <div class="page-container">
            <h1>${page.title}</h1>
            ${getServerInfoHtml(page)}
            <div class="question-block">
                <p class="question-text">${page.questionText} <span class="required">*</span></p>
                <p><em>Based on your main category: ${functionalityData.mainCategories[mainSelection]}</em></p>
                ${choicesHtml}
            </div>
        </div>
    `);
}

// Show combined server page with all questions
function showServerCombinedPage(page, pageIndex) {
    var serverIdx = page.serverIndex;
    var toolsList = page.tools.map(t => `<li>${t}</li>`).join('');

    // Build server info section
    var serverInfoHtml = `
        <div class="server-info">
            <h2>Server Information</h2>
            <p><strong>Name:</strong> ${page.serverName}</p>
            <p><strong>Description:</strong> ${page.description}</p>
            <p><strong>Tools (sample):</strong></p>
            <ul>${toolsList}</ul>
        </div>
    `;

    // Q1.1: O*NET Level 1 (always visible)
    var onetL1Choices = [];
    for (var l1Id in onetData.l1Clusters) {
        onetL1Choices.push({value: l1Id, text: onetData.l1Clusters[l1Id]});
    }
    var onetL1Html = onetL1Choices.map(choice => `
        <label>
            <input type="radio" name="onet_l1_${serverIdx}" value="${choice.value}" class="onet-l1-radio">
            ${choice.text}
        </label>
    `).join('');

    var q1Html = `
        <div class="question-block" id="q1_block">
            <h3>Q1.1: O*NET Broad Category</h3>
            <p class="question-text">Which broad occupational category best describes the primary function of this server? <span class="required">*</span></p>
            ${onetL1Html}
        </div>
    `;

    // Q1.2 and Q1.3: O*NET Level 2 and Task (conditionally visible)
    var q12Html = '';
    var q13Html = '';
    if (page.allOnetLevels) {
        q12Html = `
            <div class="question-block conditional-question" id="q12_block" style="display: none; margin-left: 20px; padding-left: 20px; border-left: 3px solid #ccc;">
                <h3>Q1.2: O*NET Sub-Category</h3>
                <p class="question-text">Which specific occupational sub-category best fits this server? <span class="required">*</span></p>
                <div id="onet_l2_choices_${serverIdx}"></div>
            </div>
        `;

        q13Html = `
            <div class="question-block conditional-question" id="q13_block" style="display: none; margin-left: 40px; padding-left: 20px; border-left: 3px solid #ccc;">
                <h3>Q1.3: O*NET Task</h3>
                <p class="question-text">Which specific occupational task most closely matches this server's functionality? <span class="required">*</span></p>
                <div id="onet_task_choices_${serverIdx}"></div>
            </div>
        `;
    }

    // Q2.1: Functionality Main Category (always visible)
    var q21Html = page.q21Choices.map(choice => `
        <label>
            <input type="radio" name="q21_${serverIdx}" value="${choice.value}" class="q21-radio">
            ${choice.text}
        </label>
    `).join('');

    var q2Html = `
        <div class="question-block" id="q2_block">
            <h3>Q2.1: Autonomy Level</h3>
            <p class="question-text">${page.q21Text} <span class="required">*</span></p>
            ${q21Html}
        </div>
    `;

    // Q2.2: Functionality Sub-Category (conditionally visible)
    var q22Html = `
        <div class="question-block conditional-question" id="q22_block" style="display: none; margin-left: 20px; padding-left: 20px; border-left: 3px solid #ccc;">
            <h3>Q2.2: Sub-Category</h3>
            <p class="question-text">Which specific sub-category best describes this server's functionality? <span class="required">*</span></p>
            <div id="q22_choices_${serverIdx}"></div>
        </div>
    `;

    // Standard questions (Q3, Q4, Q5)
    var standardQuestionsHtml = page.standardQuestions.map(function(q, idx) {
        var qNum = idx + 3;
        var choicesHtml = q.choices.map(choice => `
            <label>
                <input type="radio" name="${q.questionId}_${serverIdx}" value="${choice.value}">
                ${choice.text}
            </label>
        `).join('');

        return `
            <div class="question-block" id="q${qNum}_block">
                <h3>${q.questionId.toUpperCase()}</h3>
                <p class="question-text">${q.questionText} <span class="required">*</span></p>
                ${choicesHtml}
            </div>
        `;
    }).join('');

    // Assemble the full page
    $('#gorilla').append(`
        <div class="page-container">
            <h1>${page.title}</h1>
            ${serverInfoHtml}
            ${q1Html}
            ${q12Html}
            ${q13Html}
            ${q2Html}
            ${q22Html}
            ${standardQuestionsHtml}
        </div>
    `);

    // Add event listeners for conditional display

    // O*NET L1 -> L2 conditional display
    if (page.allOnetLevels) {
        $(`.onet-l1-radio`).on('change', function() {
            var selectedL1 = $('input[name="onet_l1_' + serverIdx + '"]:checked').val();
            if (selectedL1) {
                // Show L2 block and populate choices
                $('#q12_block').show();
                var l2Choices = getL2ClustersForL1(selectedL1);
                var l2Html = l2Choices.map(choice => `
                    <label>
                        <input type="radio" name="onet_l2_${serverIdx}" value="${choice.value}" class="onet-l2-radio">
                        ${choice.text}
                    </label>
                `).join('');
                $('#onet_l2_choices_' + serverIdx).html(l2Html);

                // Add listener for L2 -> Task
                $(`.onet-l2-radio`).on('change', function() {
                    var selectedL2 = $('input[name="onet_l2_' + serverIdx + '"]:checked').val();
                    if (selectedL2) {
                        // Show Task block and populate choices
                        $('#q13_block').show();
                        var taskChoices = getTasksForL2(selectedL2);
                        var taskHtml = taskChoices.map(choice => `
                            <label>
                                <input type="radio" name="onet_task_${serverIdx}" value="${choice.value}">
                                ${choice.text}
                            </label>
                        `).join('');
                        $('#onet_task_choices_' + serverIdx).html(taskHtml);
                    }
                });
            }
        });
    }

    // Q2.1 -> Q2.2 conditional display
    $(`.q21-radio`).on('change', function() {
        var selectedQ21 = $('input[name="q21_' + serverIdx + '"]:checked').val();
        if (selectedQ21) {
            // Auto-fill perception with sensors
            if (selectedQ21 === 'perception') {
                $('#q22_block').hide();
                // Auto-set response
                if (!serverResponses[serverIdx]) serverResponses[serverIdx] = {};
                serverResponses[serverIdx]['func_sub'] = 'sensors';
                responses[page.serverName + '_func_sub'] = 'sensors';
            } else {
                // Show Q2.2 block and populate choices
                $('#q22_block').show();
                var subChoices = getSubCategoriesFor(selectedQ21);
                var subHtml = subChoices.map(choice => `
                    <label>
                        <input type="radio" name="q22_${serverIdx}" value="${choice.value}">
                        ${choice.text}
                    </label>
                `).join('');
                $('#q22_choices_' + serverIdx).html(subHtml);
            }
        }
    });
}

// Show combined tool page with all questions (Q1.1, Q2.1, Q2.2)
function showToolCombinedPage(page, pageIndex) {
    var toolIdx = page.toolIndex;

    // Build tool info section
    var toolInfoHtml = `
        <div class="server-info">
            <h2>Tool Information</h2>
            <p><strong>Tool Name:</strong> ${page.toolName}</p>
            <p><strong>Tool Description:</strong> ${page.toolDescription}</p>
            <p><strong>Server:</strong> ${page.serverName}</p>
            <p><strong>Server Description:</strong> ${page.serverDescription}</p>
        </div>
    `;

    // Q1.1: O*NET Level 1 (always visible)
    var onetL1Choices = [];
    for (var l1Id in onetData.l1Clusters) {
        onetL1Choices.push({value: l1Id, text: onetData.l1Clusters[l1Id]});
    }
    var onetL1Html = onetL1Choices.map(choice => `
        <label>
            <input type="radio" name="onet_l1_${toolIdx}" value="${choice.value}">
            ${choice.text}
        </label>
    `).join('');

    var q1Html = `
        <div class="question-block" id="q1_block">
            <h3>Q1.1: O*NET Broad Category</h3>
            <p class="question-text">Which broad occupational category best describes what this tool does? <span class="required">*</span></p>
            ${onetL1Html}
        </div>
    `;

    // Q1.2 and Q1.3: O*NET Level 2 and Task (conditionally visible)
    var q12Html = '';
    var q13Html = '';
    if (page.allOnetLevels) {
        q12Html = `
            <div class="question-block conditional-question" id="q12_block" style="display: none; margin-left: 20px; padding-left: 20px; border-left: 3px solid #ccc;">
                <h3>Q1.2: O*NET Sub-Category</h3>
                <p class="question-text">Which specific occupational sub-category best fits this tool? <span class="required">*</span></p>
                <div id="onet_l2_choices_${toolIdx}"></div>
            </div>
        `;

        q13Html = `
            <div class="question-block conditional-question" id="q13_block" style="display: none; margin-left: 40px; padding-left: 20px; border-left: 3px solid #ccc;">
                <h3>Q1.3: O*NET Task</h3>
                <p class="question-text">Which specific occupational task most closely matches this tool's functionality? <span class="required">*</span></p>
                <div id="onet_task_choices_${toolIdx}"></div>
            </div>
        `;
    }

    // Q2.1: Functionality Main Category (always visible)
    var q21Html = page.q21Choices.map(choice => `
        <label>
            <input type="radio" name="q21_${toolIdx}" value="${choice.value}" class="q21-radio">
            ${choice.text}
        </label>
    `).join('');

    var q2Html = `
        <div class="question-block" id="q2_block">
            <h3>Q2.1: Autonomy Level</h3>
            <p class="question-text">${page.q21Text} <span class="required">*</span></p>
            ${q21Html}
        </div>
    `;

    // Q2.2: Functionality Sub-Category (conditionally visible)
    var q22Html = `
        <div class="question-block conditional-question" id="q22_block" style="display: none; margin-left: 20px; padding-left: 20px; border-left: 3px solid #ccc;">
            <h3>Q2.2: Sub-Category</h3>
            <p class="question-text">Which specific sub-category best describes this tool's functionality? <span class="required">*</span></p>
            <div id="q22_choices_${toolIdx}"></div>
        </div>
    `;

    // Assemble the full page
    $('#gorilla').append(`
        <div class="page-container">
            <h1>${page.title}</h1>
            ${toolInfoHtml}
            ${q1Html}
            ${q12Html}
            ${q13Html}
            ${q2Html}
            ${q22Html}
        </div>
    `);

    // Add event listeners for conditional display

    // O*NET L1 -> L2 conditional display
    if (page.allOnetLevels) {
        $('input[name="onet_l1_' + toolIdx + '"]').on('change', function() {
            var selectedL1 = $('input[name="onet_l1_' + toolIdx + '"]:checked').val();
            if (selectedL1) {
                // Show L2 block and populate choices
                $('#q12_block').show();
                var l2Choices = getL2ClustersForL1(selectedL1);
                var l2Html = l2Choices.map(choice => `
                    <label>
                        <input type="radio" name="onet_l2_${toolIdx}" value="${choice.value}" class="onet-l2-radio">
                        ${choice.text}
                    </label>
                `).join('');
                $('#onet_l2_choices_' + toolIdx).html(l2Html);

                // Add listener for L2 -> Task
                $(`.onet-l2-radio`).on('change', function() {
                    var selectedL2 = $('input[name="onet_l2_' + toolIdx + '"]:checked').val();
                    if (selectedL2) {
                        // Show Task block and populate choices
                        $('#q13_block').show();
                        var taskChoices = getTasksForL2(selectedL2);
                        var taskHtml = taskChoices.map(choice => `
                            <label>
                                <input type="radio" name="onet_task_${toolIdx}" value="${choice.value}">
                                ${choice.text}
                            </label>
                        `).join('');
                        $('#onet_task_choices_' + toolIdx).html(taskHtml);
                    }
                });
            }
        });
    }

    // Q2.1 -> Q2.2 conditional display
    $(`.q21-radio`).on('change', function() {
        var selectedQ21 = $('input[name="q21_' + toolIdx + '"]:checked').val();
        if (selectedQ21) {
            // Auto-fill perception with sensors
            if (selectedQ21 === 'perception') {
                $('#q22_block').hide();
                // Auto-set response
                if (!serverResponses[toolIdx]) serverResponses[toolIdx] = {};
                serverResponses[toolIdx]['func_sub'] = 'sensors';
                responses[page.toolName + '_func_sub'] = 'sensors';
            } else {
                // Show Q2.2 block and populate choices
                $('#q22_block').show();
                var subChoices = getSubCategoriesFor(selectedQ21);
                var subHtml = subChoices.map(choice => `
                    <label>
                        <input type="radio" name="q22_${toolIdx}" value="${choice.value}">
                        ${choice.text}
                    </label>
                `).join('');
                $('#q22_choices_' + toolIdx).html(subHtml);
            }
        }
    });
}

// Show standard server question page
function showServerQuestionPage(page, pageIndex) {
    var toolsList = page.tools.map(t => `<li>${t}</li>`).join('');

    var choicesHtml = page.choices.map(choice => `
        <label>
            <input type="radio" name="question_${pageIndex}" value="${choice.value}">
            ${choice.text}
        </label>
    `).join('');

    $('#gorilla').append(`
        <div class="page-container">
            <h1>${page.title}</h1>
            ${getServerInfoHtml(page)}
            <div class="question-block">
                <p class="question-text">${page.questionText} <span class="required">*</span></p>
                ${choicesHtml}
            </div>
        </div>
    `);
}

// Show navigation buttons
function showNavigation(pageIndex) {
    var isFirst = pageIndex === 0;
    var isLast = pageIndex === studyPages.length - 1;

    var backButton = isFirst ? '' : '<button id="back-btn" class="btn">Back</button>';
    var nextText = isLast ? 'Finish' : 'Next';

    $('#gorilla').append(`
        <div class="navigation">
            ${backButton}
            <button id="next-btn" class="btn btn-primary">${nextText}</button>
        </div>
    `);

    // Back button
    if (!isFirst) {
        $('#back-btn').on('click', function() {
            var targetPage = currentPage - 1;

            // Skip backwards over auto-skipped func_sub pages
            if (targetPage >= 0) {
                var prevPage = studyPages[targetPage];
                if (prevPage.type === 'func_sub') {
                    var serverIdx = prevPage.serverIndex;
                    var funcMain = serverResponses[serverIdx] && serverResponses[serverIdx]['func_main'];
                    if (funcMain === 'perception') {
                        // Skip one more page back to func_main
                        targetPage = targetPage - 1;
                    }
                }
            }

            showPage(targetPage);
        });
    }

    // Next button
    $('#next-btn').on('click', function() {
        if (validatePage()) {
            if (isLast) {
                finishStudy();
            } else {
                showPage(currentPage + 1);
            }
        }
    });
}

// Validate current page
function validatePage() {
    var page = studyPages[currentPage];

    // Text pages don't need validation
    if (page.type === 'completion' || page.type === 'instructions') {
        return true;
    }

    // Tutorial pages don't need validation
    if (page.type === 'tutorial_intro' || page.type === 'tutorial_preanswered' || page.type === 'tutorial_practice' || page.type === 'tutorial_feedback') {
        return true;
    }

    // Combined tool page validation
    if (page.type === 'tool_combined') {
        var toolIdx = page.toolIndex;
        var isValid = true;
        var missingFields = [];

        // Q1.1: O*NET L1
        var onetL1 = $('input[name="onet_l1_' + toolIdx + '"]:checked').val();
        if (!onetL1) {
            missingFields.push('Q1.1: O*NET Broad Category');
            isValid = false;
        } else {
            serverResponses[toolIdx]['onet_l1'] = onetL1;
            responses[page.toolName + '_onet_l1'] = onetL1;
        }

        // Q1.2 and Q1.3 (if allOnetLevels is true)
        if (page.allOnetLevels) {
            // Q1.2: O*NET L2 (only required if L1 is selected and L2 block is visible)
            if (onetL1 && $('#q12_block').is(':visible')) {
                var onetL2 = $('input[name="onet_l2_' + toolIdx + '"]:checked').val();
                if (!onetL2) {
                    missingFields.push('Q1.2: O*NET Sub-Category');
                    isValid = false;
                } else {
                    serverResponses[toolIdx]['onet_l2'] = onetL2;
                    responses[page.toolName + '_onet_l2'] = onetL2;

                    // Q1.3: O*NET Task (only required if L2 is selected and task block is visible)
                    if ($('#q13_block').is(':visible')) {
                        var onetTask = $('input[name="onet_task_' + toolIdx + '"]:checked').val();
                        if (!onetTask) {
                            missingFields.push('Q1.3: O*NET Task');
                            isValid = false;
                        } else {
                            serverResponses[toolIdx]['onet_task'] = onetTask;
                            responses[page.toolName + '_onet_task'] = onetTask;
                        }
                    }
                }
            }
        }

        // Q2.1: Functionality Main
        var q21 = $('input[name="q21_' + toolIdx + '"]:checked').val();
        if (!q21) {
            missingFields.push('Q2.1: Autonomy Level');
            isValid = false;
        } else {
            serverResponses[toolIdx]['func_main'] = q21;
            responses[page.toolName + '_func_main'] = q21;

            // Q2.2: Functionality Sub (not required for perception - auto-filled)
            if (q21 !== 'perception' && $('#q22_block').is(':visible')) {
                var q22 = $('input[name="q22_' + toolIdx + '"]:checked').val();
                if (!q22) {
                    missingFields.push('Q2.2: Sub-Category');
                    isValid = false;
                } else {
                    serverResponses[toolIdx]['func_sub'] = q22;
                    responses[page.toolName + '_func_sub'] = q22;
                }
            }
        }

        if (!isValid) {
            alert('Please answer all required questions:\\n\\n' + missingFields.join('\\n'));
            return false;
        }

        return true;
    }

    // Combined server page validation
    if (page.type === 'server_combined') {
        var serverIdx = page.serverIndex;
        var isValid = true;
        var missingFields = [];

        // Q1.1: O*NET L1
        var onetL1 = $('input[name="onet_l1_' + serverIdx + '"]:checked').val();
        if (!onetL1) {
            missingFields.push('Q1.1: O*NET Broad Category');
            isValid = false;
        } else {
            serverResponses[serverIdx]['onet_l1'] = onetL1;
            responses[page.serverName + '_onet_l1'] = onetL1;
        }

        // Q1.2 and Q1.3 (if allOnetLevels is true)
        if (page.allOnetLevels) {
            // Q1.2: O*NET L2 (only required if L1 is selected and L2 block is visible)
            if (onetL1 && $('#q12_block').is(':visible')) {
                var onetL2 = $('input[name="onet_l2_' + serverIdx + '"]:checked').val();
                if (!onetL2) {
                    missingFields.push('Q1.2: O*NET Sub-Category');
                    isValid = false;
                } else {
                    serverResponses[serverIdx]['onet_l2'] = onetL2;
                    responses[page.serverName + '_onet_l2'] = onetL2;

                    // Q1.3: O*NET Task (only required if L2 is selected and task block is visible)
                    if ($('#q13_block').is(':visible')) {
                        var onetTask = $('input[name="onet_task_' + serverIdx + '"]:checked').val();
                        if (!onetTask) {
                            missingFields.push('Q1.3: O*NET Task');
                            isValid = false;
                        } else {
                            serverResponses[serverIdx]['onet_task'] = onetTask;
                            responses[page.serverName + '_onet_task'] = onetTask;
                        }
                    }
                }
            }
        }

        // Q2.1: Functionality Main
        var q21 = $('input[name="q21_' + serverIdx + '"]:checked').val();
        if (!q21) {
            missingFields.push('Q2.1: Autonomy Level');
            isValid = false;
        } else {
            serverResponses[serverIdx]['func_main'] = q21;
            responses[page.serverName + '_func_main'] = q21;

            // Q2.2: Functionality Sub (not required for perception - auto-filled)
            if (q21 !== 'perception' && $('#q22_block').is(':visible')) {
                var q22 = $('input[name="q22_' + serverIdx + '"]:checked').val();
                if (!q22) {
                    missingFields.push('Q2.2: Sub-Category');
                    isValid = false;
                } else {
                    serverResponses[serverIdx]['func_sub'] = q22;
                    responses[page.serverName + '_func_sub'] = q22;
                }
            }
        }

        // Standard questions (Q3, Q4, Q5, etc.)
        for (var i = 0; i < page.standardQuestions.length; i++) {
            var q = page.standardQuestions[i];
            var qAnswer = $('input[name="' + q.questionId + '_' + serverIdx + '"]:checked').val();
            if (!qAnswer) {
                missingFields.push(q.questionId.toUpperCase());
                isValid = false;
            } else {
                serverResponses[serverIdx][q.questionId] = qAnswer;
                responses[page.serverName + '_' + q.questionId] = qAnswer;
            }
        }

        if (!isValid) {
            alert('Please answer all required questions:\\n\\n' + missingFields.join('\\n'));
            return false;
        }

        return true;
    }

    // All other question pages
    var answer = $('input[name="question_' + currentPage + '"]:checked').val();

    // For practice pages, allow skipping but save answer if provided
    if (!answer && !page.isPractice) {
        alert('Please answer the question');
        return false;
    }

    // Store response if an answer was provided
    if (answer) {
        var idx = page.serverIndex || page.toolIndex;  // Support both servers and tools
        if (idx) {
            // Initialize if needed
            if (!serverResponses[idx]) {
                serverResponses[idx] = {};
                console.log('Initialized serverResponses[' + idx + ']');
            }

            // Store in server/tool-specific responses
            serverResponses[idx][page.question] = answer;
            console.log('Stored answer: serverResponses[' + idx + '][' + page.question + '] = ' + answer);

            // Also store in global responses with server/tool name prefix
            var nameKey = page.serverName || page.toolName || 'unknown';
            var responseKey = nameKey + '_' + page.question;
            responses[responseKey] = answer;
        } else {
            responses[page.question] = answer;
        }
    } else if (page.isPractice) {
        // For practice pages, still initialize the response object even if no answer
        var idx = page.serverIndex || page.toolIndex;
        if (idx && !serverResponses[idx]) {
            serverResponses[idx] = {};
            console.log('Initialized serverResponses[' + idx + '] for practice page (no answer yet)');
        }
    }

    return true;
}

// Finish study
function finishStudy() {
    console.log("Final responses:", responses);
    console.log("Server responses:", serverResponses);

    // Upload metrics
    for (var key in responses) {
        // Extract server name and question ID
        var lastUnderscoreQIndex = key.lastIndexOf('_q');
        var lastUnderscoreOIndex = key.lastIndexOf('_onet');
        var lastUnderscoreFIndex = key.lastIndexOf('_func');

        var serverName = key;
        var questionId = '';

        if (lastUnderscoreQIndex !== -1) {
            serverName = key.substring(0, lastUnderscoreQIndex);
            questionId = key.substring(lastUnderscoreQIndex + 1);
        } else if (lastUnderscoreOIndex !== -1) {
            serverName = key.substring(0, lastUnderscoreOIndex);
            questionId = key.substring(lastUnderscoreOIndex + 1);
        } else if (lastUnderscoreFIndex !== -1) {
            serverName = key.substring(0, lastUnderscoreFIndex);
            questionId = key.substring(lastUnderscoreFIndex + 1);
        }

        gorilla.metric({
            name: key,
            value: responses[key],
            checked: '1',
            servername: serverName,
            question: questionId
        });
    }

    gorilla.finish();
}
"""


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate TypeScript file from CLServers CSV with hierarchical questions')
    parser.add_argument('--type', choices=['servers', 'tools'], default='servers',
                       help='Study type: servers (full Q0-Q5) or tools (Q1+Q2 only)')
    parser.add_argument('--servers', type=int, default=5,
                       help='Number of servers/tools to include (default: 5)')
    parser.add_argument('--clservers', default='../../data/final/clservers_classified.csv.gz',
                       help='Path to CLServers CSV file')
    parser.add_argument('--cltools', default='../../data/final/cltools_classified.csv.gz',
                       help='Path to CLTools CSV file')
    parser.add_argument('--questions', default='questions_config.csv',
                       help='Path to questions CSV file with choices')
    parser.add_argument('--output', default=None,
                       help='Output TypeScript file name (default: main_servers.ts or main_tools.ts based on type)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for server sampling (for reproducibility)')
    parser.add_argument('--all-onet-levels', action='store_true',
                       help='Include all 3 O*NET levels (L1, L2, Task) for each server. Default: only L1. Note: tutorials always use only L1 regardless of this flag.')

    args = parser.parse_args()

    # Set default output filename based on type
    if args.output is None:
        args.output = 'main_servers.ts' if args.type == 'servers' else 'main_tools.ts'

    # Set random seed if provided
    if args.seed:
        random.seed(args.seed)

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    clservers_path = (script_dir / args.clservers).resolve()
    cltools_path = (script_dir / args.cltools).resolve()
    questions_path = (script_dir / args.questions).resolve()
    output_path = (script_dir / args.output).resolve()

    # Load O*NET hierarchy and questions (needed for both types)
    print(f"\nBuilding O*NET hierarchy from: {cltools_path}")
    onet_hierarchy = build_onet_hierarchy(cltools_path)
    print(f"  - L1 clusters: {len(onet_hierarchy['l1'])}")
    print(f"  - L2 clusters: {len(onet_hierarchy['l2'])}")
    print(f"  - Tasks: {len(onet_hierarchy['tasks'])}")

    print("\nLoading questions from:", questions_path)
    all_questions = load_questions_with_choices(questions_path)
    print(f"Loaded {len(all_questions)} question configurations:")
    for q_id in all_questions:
        print(f"  - {q_id}: {len(all_questions[q_id]['choices'])} choices")

    print("\nBuilding functionality hierarchy from CSV")
    func_hierarchy = build_functionality_hierarchy_from_csv(all_questions)
    print(f"  - Main categories: {len(func_hierarchy['main_categories'])}")
    total_subs = sum(len(subs) for subs in func_hierarchy['sub_categories'].values())
    print(f"  - Sub-categories: {total_subs}")

    # Extract standard questions (q3, q4, q5, etc.) - exclude q2.1 and q2.2_* questions
    questions = OrderedDict()
    for q_id, q_data in all_questions.items():
        if not q_id.startswith('q2.'):
            questions[q_id] = q_data
    print(f"\nStandard questions (excluding Q2.1, Q2.2): {len(questions)}")

    # Load functionality map for tool autonomy prioritization
    print(f"\nLoading functionality map from: {cltools_path}")
    functionality_map = load_cltools_functionality_map(cltools_path)
    print(f"  - Loaded functionality data for {len(functionality_map)} servers")

    # Route to correct generation function based on type
    if args.type == 'servers':
        # Server classification study
        print(f"\nLoading CLServers from: {clservers_path}")
        servers = load_clservers_csv(clservers_path, filter_non_english=True, functionality_map=functionality_map)
        print(f"Loaded {len(servers)} servers (with >=2 tools)")

        print(f"\nGenerating {args.output} with {args.servers} servers...")
        content = generate_main_ts(servers, onet_hierarchy, func_hierarchy, questions, all_questions, output_path,
                                  num_servers=args.servers, all_onet_levels=args.all_onet_levels)

        # Add TypeScript runtime code
        content += generate_typescript_runtime()

        print(f"Writing to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # Calculate total pages for servers study
        if args.all_onet_levels:
            questions_per_server = 5 + len(questions)  # 5 validation (O*NET L1/L2/Task + Func Main/Sub) + standard questions
            onet_questions_desc = "Q1.1: O*NET L1, Q1.2: O*NET L2 (conditional), Q1.3: O*NET Task (conditional)"
        else:
            questions_per_server = 3 + len(questions)  # 3 validation (O*NET L1 + Func Main/Sub) + standard questions
            onet_questions_desc = "Q1.1: O*NET L1 only"

        # Tutorials always have 6 questions: O*NET L1 + Func Main/Sub + 3 standard questions
        tutorial_practice_pages = (1 * 6 * 2)  # 1 practice × 6 questions × 2 (question + feedback) = 12 pages
        tutorial_pages = 2 + 1 + tutorial_practice_pages  # 2 intro pages + 1 preanswered + practice pages
        total_pages = 1 + tutorial_pages + args.servers + 1  # instructions + tutorials + servers (1 combined page each) + completion

        print(f"\n✓ Generated {output_path}")
        print(f"  - {args.servers} servers (randomly sampled)")
        print(f"  - {questions_per_server} questions per server (validation questions + {len(questions)} standard)")
        print(f"  - {tutorial_pages} tutorial pages (2 intros + 1 preanswered + {tutorial_practice_pages} practice pages)")
        print(f"  - {total_pages} total pages")
        print(f"\nO*NET Configuration:")
        print(f"  - Server questions: {'All 3 levels (L1, L2, Task)' if args.all_onet_levels else 'Level 1 only (L1)'}")
        print(f"  - Tutorial questions: Level 1 only (L1) - always simplified for practice")
        print(f"\nQuestions per server:")
        print(f"  - {onet_questions_desc}")
        print(f"  - Q2.1: Functionality Main (3 choices)")
        print(f"  - Q2.2: Functionality Sub (conditional, 2-7 choices)")
        print(f"  - {len(questions)} standard questions (Q3-Q{2+len(questions)})")

    elif args.type == 'tools':
        # Tool classification study (simplified)
        print(f"Loading CLTools from: {cltools_path}")
        tools = load_cltools_csv(cltools_path)
        print(f"Loaded {len(tools)} tools")

        print(f"\nGenerating {args.output} with {args.servers} tools...")
        content = generate_main_tools_ts(tools, onet_hierarchy, func_hierarchy, all_questions, output_path,
                                        num_tools=args.servers, all_onet_levels=args.all_onet_levels)

        # Add TypeScript runtime code
        content += generate_typescript_runtime()

        print(f"Writing to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # Calculate total pages for tools study
        questions_per_tool = 3  # Q1.1 + Q2.1 + Q2.2
        tutorial_practice_pages = (1 * 3 * 2)  # 1 practice × 3 questions × 2 (question + feedback) = 6 pages
        tutorial_pages = 2 + 1 + tutorial_practice_pages  # 2 intro pages + 1 preanswered + practice pages = 9
        total_pages = 1 + tutorial_pages + args.servers + 1  # instructions + tutorial + tools (1 combined page each) + completion

        print(f"\n✓ Generated {output_path}")
        print(f"  - {args.servers} tools (randomly sampled)")
        print(f"  - {questions_per_tool} questions per tool (Q1.1 + Q2.1 + Q2.2 only)")
        print(f"  - {tutorial_pages} tutorial pages (2 intros + 1 preanswered + {tutorial_practice_pages} practice pages)")
        print(f"  - {total_pages} total pages")
        print(f"\nO*NET Configuration:")
        print(f"  - Tool questions: Level 1 only (L1)")
        print(f"  - Tutorial: Simplified (1 pre-answered example)")
        print(f"\nQuestions per tool:")
        print(f"  - Q1.1: O*NET L1 (broad occupational category)")
        print(f"  - Q2.1: Functionality Main (3 choices)")
        print(f"  - Q2.2: Functionality Sub (conditional, 2-7 choices)")
        print(f"\nNote: Tools study excludes Q3-Q5 (industry/environment/payment)")

    if args.seed:
        print(f"\nRandom seed used: {args.seed} (for reproducibility)")


if __name__ == '__main__':
    main()

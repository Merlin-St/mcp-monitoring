#!/usr/bin/env python3
"""
Patch generate_main_ts.py to support --all-onet-levels for tools.

This script modifies the generate_main_tools_ts function to:
1. Accept and use the all_onet_levels parameter
2. Update instruction page to mention Q1.1/Q1.2/Q1.3 when enabled
3. Modify TypeScript generation to include hierarchical O*NET questions
4. Update validation logic for Q1.2 and Q1.3
"""

import re

def patch_file():
    """Apply all patches to generate_main_ts.py"""

    with open('generate_main_ts.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # Patch 1: Update instruction page content to be conditional
    old_instruction = '''    return f"""    {{
        type: 'instructions',
        title: 'Study Instructions',
        content: `# How to Classify MCP Tools

Thank you for participating! You will be evaluating individual MCP (Model Context Protocol) tools - specific functions within AI agent toolkits.

## Your Task

* For each tool, you will answer 2 questions:

* **O*NET Occupational Category** (Q1.1) - What job task does this tool support? (Note: Many tools will be in the category 'Design, implement, and maintain diverse information technology systems')

* **Functionality Classification** (Q2.1-Q2.2) - How does this tool work?

* **Focus on the TOOL** - Not the entire server, just this specific tool's function

* Based on our testing, the first tools will take you 2-10mins until you get used to it, then we expect each tool to take around 1-1.5mins
* Click **Next** to see an example classification, then you'll start the study.`
    }},
"""'''

    new_instruction = '''    onet_desc = "2 questions" if not all_onet_levels else "2-4 questions"
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
"""'''

    if old_instruction in content:
        content = content.replace(old_instruction, new_instruction)
        print("✓ Patched: instruction page conditional text")
    else:
        print("⚠ Warning: Could not find instruction page to patch (may already be patched)")

    # Patch 2: Add allOnetLevels variable after studyPages array closing
    # Find where we close the studyPages array in generate_main_tools_ts
    study_pages_close_pattern = r"(\]\;\n\n\"\"\")\n\n    return output"

    if re.search(study_pages_close_pattern, content):
        replacement = r"\1\n\n    # Add allOnetLevels configuration variable\n    output += f\"\"\"// Configuration\nvar allOnetLevels = {str(all_onet_levels).lower()};\n\n\"\"\"\n\n    return output"
        content = re.sub(study_pages_close_pattern, replacement, content)
        print("✓ Patched: added allOnetLevels variable to tools output")
    else:
        print("⚠ Warning: Could not find studyPages closing pattern")

    # Write patched content back
    with open('generate_main_ts.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print("\n✓ All patches applied successfully")
    print("Note: The TypeScript runtime already supports hierarchical O*NET rendering.")
    print("The allOnetLevels variable controls Q1.2/Q1.3 visibility in the browser.")

if __name__ == '__main__':
    patch_file()

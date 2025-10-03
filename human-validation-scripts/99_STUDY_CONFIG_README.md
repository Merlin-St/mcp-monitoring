# Study Composer Config Generator - Usage Guide

## Overview

This generator creates Gorilla Study Composer configurations for human classification of MCP servers and tools.

## Generated Files

- **`99_study_config_generator.py`** - Python script to generate study configs
- **`99_study_config_mcp_classification.json`** - Generated study config (ready to upload to Gorilla)

## Quick Start

### 1. Generate Config with Default Samples (2 servers + 2 tools)

```bash
source .venv/bin/activate
python 99_study_config_generator.py
```

### 2. Generate Config with 100 Samples Each

Edit `99_study_config_generator.py` and change the last section:

```python
# Generate config with 100 samples each
generate_study_config(
    servers_csv_path=str(servers_csv),
    tools_csv_path=str(tools_csv),
    onet_csv_path=str(onet_csv),
    output_path=str(output_json),
    num_server_samples=100,  # ← Change from 2 to 100
    num_tool_samples=100     # ← Change from 2 to 100
)
```

Then run:

```bash
python 99_study_config_generator.py
```

## Study Structure

### Page Types

1. **Intro Page** - Welcome and instructions
2. **Server Classification Pages** (N pages)
   - Industry Generality (Q1)
   - Environment Generality (Q2)
   - Payment Autonomy Level (Q3)
3. **Tool Classification Pages** (M pages)
   - Autonomy Level Classification (Q1)
     - Step 1: Category (Perception/Reasoning/Action)
     - Step 2: Subcategory (conditional based on Step 1)
   - O*NET Task Mapping (Q2)
     - Level 1: Primary task cluster
     - Level 2: Specific task (conditional based on Level 1)
4. **Completion Page** - Thank you message

### Classification Taxonomies

#### Server Classification

**Q1: Industry Generality**
- `1` = Cross-industry (desktop tools, file management, calendars, etc.)
- `0` = Industry-specific (crypto, payments, medical, legal systems)

**Q2: Environment Generality**
- `1` = Open/untrusted (browser automation, web scraping, generic file access)
- `0` = Trusted/pre-specified (specific APIs like Stripe/GitHub/Slack, internal databases)

**Q3: Payment Autonomy**
- `0` = Not a payment server
- `1` = Information only (read-only payment data)
- `2` = Payment request generation (creates links but doesn't execute)
- `3` = Third-party payment processing (via Stripe/PayPal APIs)
- `4` = Direct payment processing (full control, e.g., blockchain)

#### Tool Classification

**Q1: Autonomy Level (2-step hierarchical)**

1. **PERCEPTION** (gathering information)
   - 1.1 Sensors - database queries, monitoring, diagnostics, GUI reading, search

2. **REASONING** (processing/analysis)
   - 2.1 Planning - task decomposition, path-finding, workflow orchestration
   - 2.2 Analysis - calculations, simulations, data processing
   - 2.3 Resource Management - memory, self-management, resource allocation

3. **ACTION** (directly affecting the environment)
   - 3.1 Authentication - login, CAPTCHA, wallet operations
   - 3.2 Computer Use - GUI interaction, website automation, computer control
   - 3.3 Code Execution - interpreters, IDE, file operations, running code
   - 3.4 Software Extensions - calendar, social media APIs, third-party services
   - 3.5 Physical Extensions - robotics, laboratory tools, physical world
   - 3.6 Human Interaction - phone calls, messaging, direct communication
   - 3.7 Agent Interaction - multi-agent coordination, sub-agents, third-party agents

**Q2: O*NET Task Mapping (2-level hierarchical)**

- **Level 1**: 12 primary occupational clusters (L1_01 through L1_12)
  - Examples: "Business management, finance, and customer service operations"
- **Level 2**: Specific task clusters (conditional based on Level 1 selection)
  - Examples: "Securities Trading and Investment Advisory Services"

## Features

### ✅ Supported Features

- **Conditional Questions**: Uses `visibleIf` to show subcategories based on parent selection
- **Hierarchical Classification**: 2-level autonomy and O*NET taxonomies
- **Dynamic Data Loading**: Reads from CSV files
- **Markdown Formatting**: Rich text display of server/tool information
- **Progress Tracking**: Shows progress bar and page numbers
- **Navigation**: Allows back navigation and elapsed time display

### ⚠️ Limitations

- **Static Configuration**: Cannot dynamically load CSV at runtime in Gorilla
- **Fixed Sample Size**: Must regenerate config to change number of samples
- **No 3rd Level O*NET**: Task-level classification not included (only L1 and L2 clusters)

## Data Sources

- **Servers**: `data/final/clservers_classified.csv`
- **Tools**: `data/final/cltools_classified.csv`
- **O*NET Tasks**: `data/internal-task-clusters/task_clusters_names.csv`

## Output Format

The generated JSON follows the Gorilla Study Composer schema with:

```json
{
  "name": "mcp-classification-study",
  "studyContentsConfig": {
    "navigation": { ... },
    "pages": [ ... ]
  },
  "model": "claude-sonnet-4-20250514",
  "provider": "Anthropic",
  ...
}
```

## Uploading to Gorilla

1. Generate the config: `python 99_study_config_generator.py`
2. Open Gorilla Study Composer: https://gorilla.sc/
3. Create new study or edit existing
4. Import JSON config: Upload `99_study_config_mcp_classification.json`
5. Preview and test the study
6. Publish when ready

## Validation

The script automatically validates that:
- All required CSV files exist
- JSON output is valid
- All question IDs are unique
- Conditional logic references valid question IDs

## Customization

To modify classifications:

1. **Change Server Questions**: Edit `create_server_classification_page()` function
2. **Change Tool Questions**: Edit `create_tool_classification_page()` function
3. **Update Autonomy Taxonomy**: Modify `AUTONOMY_TAXONOMY` dictionary
4. **Add Instructions**: Edit intro/completion page markdown text

## Troubleshooting

**Issue**: CSV file not found
- **Solution**: Run script from project root directory

**Issue**: JSON validation fails
- **Solution**: Check CSV data for special characters, escape quotes properly

**Issue**: Too many conditional questions
- **Solution**: Gorilla has limits on page complexity, reduce O*NET levels or split into multiple pages

## Example Output Stats

With default settings (2 + 2 samples):
- Total pages: 6 (intro + 2 servers + 2 tools + completion)
- File size: ~168 KB
- Questions per server page: 3
- Questions per tool page: 17 (4 autonomy + 13 conditional O*NET)

With 100 + 100 samples:
- Total pages: 202
- File size: ~8-10 MB (estimated)
- Total questions: 2000+ (3×100 + 17×100)

## Contact

For questions about this generator or the study design, contact the MCP research team.

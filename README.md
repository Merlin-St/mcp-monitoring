# MCP server monitoring project

**Status: Highly WIP. First full pipeline built, with outputs on https://sr-mcp-dashboard.apps.aisi.org.uk/. Not yet fully validated, readme needs clarification, ask Merlin for details**

## Overview: What AI agents do - and how autonomously: Evidence from 70k MCP tools
AI agents are starting to interact with the external environment, mostly via Model Context Protocol servers (MCPs) that provide tools to agents. We analyse over 70k MCP tools sourced from online repositories(Github, Smithery, official MCP repo) and a web server search (Shodan, tbd). We classify servers and tools through the lens of tasks in the O*NET Database and by their ability to modify external environments, and weight them by usage (as per pypi/npm downloads, and smithery API usage).

## 🚀 Quick Start
```bash
uv sync
make help

# Complete data collection and processing
make workflow-data-creation

# Full data collection, processing, analysis and classification
make workflow-complete
```

**Note**: Dashboard components have been moved to https://github.com/AI-Safety-Institute/sr-mcp-dashboard. This repository focuses on data collection, processing, topic modelling analysis and classification. This repo creates the main output data files in final/ and topic modelling figures.

## 📁 Repository Structure

```
mcp-monitoring/
├── data/                              # Organized data storage
│   ├── external-servers/              # Raw server data from 3 sources
│   ├── external-usage/                # Usage statistics (npm, PyPI)
│   ├── external-cl/                   # External classification datasets (NAICS, O*NET)
│   ├── internal-task-clusters/        # O*NET task clustering results
│   ├── internal-cl/                   # Internal classification results
│   ├── initial/                       # Unified and filtered datasets
│   └── final/                         # Final classified datasets
├── scripts/                           # Processing and analysis scripts
│   ├── data-collection-servers/       # Server data collection (3 sources)
│   ├── data-collection-usage/         # Usage statistics collection
│   ├── data-unification/              # Data merging and processing
│   ├── data-cleaning-readmes/         # LLM-based README filtering
│   ├── data-analysis-topics/          # ML topic analysis and embeddings
│   ├── data-classification-servers/   # CLServers: Finance server classification
│   ├── data-classification-tools/     # CLTools: O*NET task mapping
│   ├── onet-task-clusters/            # O*NET occupational task clustering
│   └── 99_playground/                 # Experimental scripts
├── output-visuals/                    # Visualization outputs
│   └── topics-embedding/              # Interactive topic analysis visualizations
├── output-validation/                 # Analysis validation and findings
│   ├── cl-validation/                 # Classification validation results
│   └── task-validation/               # Task mapping validation
├── logs/                              # Execution logs
├── embeddings_cache/                  # Cached embeddings for performance
├── Makefile                           # Common commands and workflows
├── CLAUDE.md                          # Claude Code instructions
└── README.md                          # This file
```

## Data Processing Pipeline: 3 Sources → Unified Dataset

### Raw Data Sources & Structure

#### 1. **Smithery API** (`data/external-servers/smithery_data.json`)
- **Shape**: 6,434 servers × 6 columns
- **Columns**: `qualifiedName`, `displayName`, `description`, `createdAt`, `useCount`, `homepage`  
- **Sample Data**:
  - `qualifiedName`: `@wonderwhy-er/desktop-commander`
  - `displayName`: `Desktop Commander`
  - `useCount`: `579226` (usage metrics)

#### 2. **GitHub Repositories** (`data/external-servers/github_data.json`)
- **Shape**: 21,053 repos × 83 columns (GitHub API fields)
- **Key Columns**: `readme_content`, `owner`, `license`, `topics`, `stargazers_count`, `language`, `permissions`
- **Sample Data**:
  - `readme_content`: Full README files (up to 22,000 characters)
  - `owner`: `{'login': 'phil65', 'id': 110931, 'avatar_url': '...'}`  
  - `topics`: `['mcp-server', 'ai-tools', 'claude']`

#### 3. **Official MCP List** (`data/external-servers/officiallist_data.json`)
- **Shape**: 966 servers × 5 columns per server
- **Structure**: `{fetch_date, total_servers, servers: [...]}`
- **Server Columns**: `name`, `url`, `description`, `is_github`, `extracted_date`
- **Sample Data**:
  - `name`: `Everything`
  - `url`: `https://github.com/modelcontextprotocol/src/everything`
  - `description`: `Reference / test server with prompts, resources, and tools`

### Data processing, filtering & cleaning Pipeline (`make data-initial-clean`)

```
Load Data→ Deduplicate → Enhance (→data_unified.json) → Filter, add usage & cleaned readmes → Update to data_unified_filtered.json
     ↓          ↓           ↓                           ↓         ↓          ↓ 
3 JSON files  Merge     Keyword sector                  >=1 star  pypi/npm   llm cleaning
             Conflicts  classification                                       & tool extraction


## 📦 Usage Statistics Collection

Strict 1:1 package-to-repository matching of monthly download statistics from November 2024 to present from PyPI and npm 
`data/external-usage/data_usage.json`

- **Package Coverage**: 70.6% of discovered MCP packages matched to repositories (3,450/4,886, 1000 in exact match, 2500 in 90%+ confidence fuzzy indexed scoring), which is 80% of downloads
- **Repository Coverage**: 1,431 repositories with download statistics (8.4% of 16,940 total)
- **PyPI Coverage**: 97.6% match rate (3,018/3,092 packages matched), 80.4% of all PyPI downloads captured
- **npm Coverage**: 59.5% match rate (213/358 packages matched), 100% of matched packages have data
- **Data Quality**: 90+ confidence threshold, strict 1:1 repository matching, monthly breakdown Nov 2024-Aug 2025

# Package discovery used search terms across PyPI and npm:
# - 'mcp server', 'mcp-server', 'modelcontextprotocol', 'mcp'
# - Found: 4,886 total packages (PyPI: 4,528, npm: 358)
# - Matched: 3,450 packages via competitive 1:1 matching (70.6% coverage)
```
### Data labelling Pipeline (`make data-cl-all`)
- Before: Filtering and cleaning (see above) with `make data-clean-readmes`.
  - So far: Readme clean version, is mcp server?, 1 sentence summary, tool extraction.
  - To add: Which are narrow-purpose vs. general-purpose tools (first say what is the 'action space' then narrow vs general purpose). Which are official vs. unofficial tools as judged by the author of the github repo.
- Servers classification: Autonomy level
  - So far: Is_finance, Finance asset type, Automation level, task capabilities, sensitive data inputs 
- Tools classification:
  - So far: Automation level, Functionality perception/reasoning/action, access write/read, O-Net cluster assignment


## 📁 Key Files
### Main Data
- `scripts/data-unification/data_unified_processor.py` - **Creates** `data/initial/data_unified.json` from 3 sources (27,899 servers)
- `scripts/data-unification/data_unified_create_filtered_subset.py` - **Creates** `data/initial/data_unified_filtered.json` (filtered subset)
- `scripts/data-cleaning-readmes/data_readme_filter_dfprocessing.py` - **Edits** `data/initial/data_unified_filtered.json` (adds readme_filtered column)
- `scripts/data-unification/data_unified_add_usage.py` - **Modifies** `data/initial/data_unified_filtered.json` (adds usage statistics in place)
- `data/initial/data_unified.json` - Full unified dataset (343MB, 27,899 servers)
- `data/initial/data_unified_summary.json` - Basic dataset statistics and metadata
- `data/initial/data_unified_filtered.json` - Filtered dataset (225MB, core fields only)

### Topic modelling Analysis
- `scripts/data-analysis-topics/embed_generate.py` - GPU-accelerated embedding generation
- `scripts/data-analysis-topics/embed_hyperparameter_optimizer.py` - Automated hyperparameter optimization
- `scripts/data-analysis-topics/embed_apply_optimized_parameters.py` - Helper to apply optimized parameters automatically

### Consequentiality Scoring

The consequentiality scoring system evaluates MCP servers and tools for their potential impact on financial systems through three integrated pipelines:

#### CLServers Pipeline - Server Classification
**Purpose**: Identifies and classifies finance-relevant MCP servers by autonomy level and consequentiality
- `scripts/data-classification-servers/clservers_1_dataprep.py` - Data preparation and finance server sampling
- `scripts/data-classification-servers/clservers_2_inspect.py` - LLM-based finance relevance classification using Inspect framework
- `scripts/data-classification-servers/clservers_3_dfprocessing.py` - Process evaluation results from .eval files to JSON
- `scripts/data-classification-servers/clservers_4_datamatch.py` - Add metadata and create final classified server CSV
- `scripts/data-classification-servers/clservers_validate.py` - Human vs LLM validation comparison script

#### CLTools Pipeline - O*NET Task Mapping  
**Purpose**: Maps MCP tools to occupational tasks using O*NET database for consequentiality assessment
- `scripts/data-classification-tools/cltools_main.py` - Main O*NET task classification pipeline
- `scripts/data-classification-tools/cltools_datamatch.py` - Enriches tool classifications with server metadata
- `scripts/data-classification-tools/cltools.md` - Documentation for CLTools approach and methodology

#### O*NET Task Clustering - Occupational Analysis
**Purpose**: Creates task clusters from O*NET database for systematic tool classification
- `scripts/onet-task-clusters/task_clusters_run.py` - Main task clustering pipeline runner
- `scripts/onet-task-clusters/task_clusters_data.py` - O*NET task data loading and processing
- `scripts/onet-task-clusters/task_clusters_embeddings.py` - Generate embeddings for task clustering
- `scripts/onet-task-clusters/task_clusters_llm.py` - LLM-based cluster naming and validation
- `scripts/onet-task-clusters/task_clusters_embed_match.py` - Match MCP tools to O*NET task clusters
- `scripts/onet-task-clusters/task_clusters.md` - Task clustering methodology documentation

#### Validation & Quality Assurance
- `data/external-cl/clservers_validate_labelled.csv` - Human-labeled validation dataset (92 servers)
- `output-validation/cl-validation/clservers_validation.json` - Validation results comparing human vs LLM labels
- `output-validation/task-validation/task_clusters_embed_match_findings.md` - Task clustering validation findings

Human-labeled validation against LLM consequentiality scoring across 92 servers shows **83.2% mean accuracy** across 15 fields with systematic over-estimation bias in level classification.

**5-Level Classification:**
- **Exact Accuracy**: 55.7% (49/88 servers)
- **Off-by-one Accuracy**: 86.4% (acceptable for ordinal data)
- **Mean Absolute Error**: 0.63 levels

**Confusion Matrix (Rows=Human, Cols=LLM):**
```
     L1  L2  L3  L4  L5   (Human Distribution)
L1:  15  22   1   3   0   (41 servers, 46.6%)
L2:   2   9   1   3   0   (15 servers, 17.0%) 
L3:   0   0   2   2   1   (5 servers, 5.7%)
L4:   1   3   0  21   0   (25 servers, 28.4%)
L5:   0   0   0   0   2   (2 servers, 2.3%)
```

**4-Level Classification (L1+L2 Combined):**
- **Combined Accuracy**: 83.0% (73/88 servers) 
- **Improvement**: +27.3 percentage points
- **Key Fix**: 24/39 disagreements resolved by treating L1/L2 as single "Low Risk" category

**Systematic Bias**: LLM over-estimates by 1+ levels in 33/39 disagreements (84.6%), particularly confusing information-gathering tools (Human L1) with limited-interaction tools (Human L2).

**Validation Scripts**: `stage_stage2_validate.py`, `conseq_level_disagreement_analysis.py`


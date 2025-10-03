# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

**IMPORTANT**: Always activate the virtual environment before running Python scripts:
```bash
source ~/mcp-monitoring/.venv/bin/activate
```
If libraries are missing, add them to the pyproject.toml and run uv sync

## Project Dependencies

This project uses **pyproject.toml** for dependency management (modern Python standard):

```bash
# Install dependencies
uv sync                    # Install runtime dependencies
uv sync --group dev        # Install with development dependencies (includes ruff, pytest)

# Alternative with pip
pip install -e .           # Install runtime dependencies  
pip install -e .[dev]      # Install with development dependencies
```

## Code Quality & Linting

**Always run code quality checks before committing:**

```bash
# Check code quality (linting)
ruff check .

# Auto-fix issues where possible
ruff check . --fix

# Fix unsafe issues (like bare except statements)
ruff check . --fix --unsafe-fixes

# Format code
ruff format .

# Fix specific error types
ruff check . --select E722 --fix --unsafe-fixes  # Fix bare except statements
```

**Configuration**: Code quality rules are defined in `pyproject.toml` under `[tool.ruff.lint]`

## Project Overview

This is a **MCP Server Monitoring Dashboard** that provides comprehensive analysis of Model Context Protocol (MCP) servers across multiple data sources. The project monitors MCP server ecosystem growth, tool availability, and finance-specific capabilities to answer key research questions about AI tool proliferation in financial systems.

### Research Questions (RQs)
1. **Which tools are available for AI power-users in finance currently?**
2. **What is the uptake for these tools over time?**
3. **How many new tools are made available over time?**

### Project Goals
- Understand most consequential available tools for agents to externals or open-source local tools
- Initial picture of finance sector tool availability
- Monitor MCP server creation and usage trends over time
- Classify servers by sectors, finance use cases, autonomy levels, and consequentiality

## Data Collection Strategy (4 Approaches)

### 1. Smithery MCP Server Database
```bash
# Download all MCP server data from Smithery Registry
python scripts/data-collection-servers/smithery_data_run.py
```

### 2. GitHub Repository Scanning
```bash
# Scan GitHub for 'mcp server' repositories
python scripts/data-collection-servers/github_data_run.py

# Default: REST API with daily date-based searches
```

### 3. Official ModelContextProtocol/Servers List
```bash
# Scrape official MCP servers list
python scripts/data-collection-servers/officiallist_data_run.py
```

### 4. Awesome MCP Servers List
```bash
# Scrape curated awesome-mcp-servers list from GitHub
python scripts/data-collection-servers/awesomelist_data_run.py
```

## Core Architecture

**Data Flow:**
1. **Collection**: Three parallel data collection streams → JSON data files
2. **Analysis**: Unified analysis processing server data with financial risk categorization
3. **Visualization**: Dashboard components moved to https://github.com/AI-Safety-Institute/sr-mcp-dashboard

**Key Files:**

**Data Collection:**
- `scripts/data-collection-servers/smithery_data_run.py` - Smithery API collection entry point
- `scripts/data-collection-servers/smithery_bulk_mcp_downloader.py` - Core Smithery download logic
- `scripts/data-collection-servers/github_data_run.py` - GitHub repository scanning
- `scripts/data-collection-servers/github_mcp_repo_searcher.py` - GitHub search functionality
- `scripts/data-collection-servers/officiallist_data_run.py` - Official list scraping
- `scripts/data-collection-servers/officiallist_html_fetcher.py` - HTML content fetching
- `scripts/data-collection-servers/officiallist_url_extractor.py` - URL extraction from HTML
- `scripts/data-collection-servers/awesomelist_data_run.py` - Awesome MCP Servers curated list scraping with emoji metadata parsing

**Data Processing & Analysis:**
- `data_unified_processor.py` - Unified data processing (27,899 servers)
- `data_unified_create_filtered_subset.py` - Create filtered subsets for analysis

**ML Analysis & Embeddings:**
- `scripts/data-analysis-topics/embed_generate.py` - GPU-accelerated embedding generation with NAICS classification
- `scripts/data-analysis-topics/embed_hyperparameter_optimizer.py` - Automated hyperparameter optimization for BERTopic models
- `scripts/data-analysis-topics/embed_apply_optimized_parameters.py` - Helper script to apply optimized parameters to embed_generate.py
- `scripts/data-analysis-topics/naics_classification_config.py` - NAICS sector definitions and keyword mappings

**Consequentiality Scoring - CLServers Pipeline:**
- `scripts/data-classification-servers/clservers_1_dataprep.py` - CLServers Step 1: Finance data preparation for analysis
- `scripts/data-classification-servers/clservers_2_inspect.py` - CLServers Step 2: Inspect task for finance-relevant server filtering
- `scripts/data-classification-servers/clservers_3_dfprocessing.py` - CLServers Step 3: DataFrame processing for .eval files to JSON
- `scripts/data-classification-servers/clservers_4_datamatch.py` - CLServers Step 4: Data matching and final CSV generation
- `scripts/data-classification-servers/clservers_validate.py` - Validation script comparing human vs LLM labels
- `clservers_validate_details.py` - Detailed analysis of validation disagreements

**Consequentiality Scoring - CLTools Pipeline:**
- `scripts/data-classification-tools/cltools_main.py` - Main O*NET task classification pipeline
- `scripts/data-classification-tools/cltools_datamatch.py` - Enriches CLTools output with metadata from CLServers
- `scripts/onet-task-clusters/task_clusters_data.py` - O*NET task data loading and processing
- `scripts/onet-task-clusters/task_clusters_embeddings.py` - Embedding generation for task clustering
- `scripts/onet-task-clusters/task_clusters_llm.py` - LLM-based cluster naming and validation
- `scripts/onet-task-clusters/task_clusters_run.py` - Main runner for task clustering pipeline
- `scripts/onet-task-clusters/task_clusters_embed_match.py` - Matching MCP tools to O*NET tasks
- `scripts/onet-task-clusters/task_clusters.md` - Documentation for task clustering approach

**README Content Filtering:**
- `scripts/data-cleaning-readmes/data_readme_filter_inspect.py` - LLM-based refinement using Inspect framework
- `scripts/data-cleaning-readmes/data_readme_filter_dfprocessing.py` - Process Inspect results back to JSON format

**Utilities:**
- `scripts/data-collection-servers/smithery_bulk_mcp_config.py` - Configuration management
- `scripts/data-collection-servers/officiallist_github_fetcher.py` - GitHub metadata collection for officiallist servers

## Analysis & Classification

The system extracts and classifies:

### Data Extraction
- **Tools** available in each server
- **Usage statistics** over time (stars, forks, creation dates)
- **Creation date** and growth trends
- **Official/unofficial** status

### Classification Categories
- **NAICS Sectors**: Full 20-sector classification (Agriculture, Finance, Professional Services, etc.)
- **Finance Use Cases**: Payment execution, market data, risk analysis
- **Autonomy Levels**: Information gathering, execution capabilities, agent interactions
- **Consequentiality**: Risk assessment for financial system impact

### ML-Powered Analysis
- **Semantic Embeddings**: High-quality text embeddings using sentence-transformers
- **Topic Modeling**: BERTopic for discovering server clusters and themes
- **Dimensional Reduction**: UMAP for 2D/3D visualization of server relationships
- **Clustering**: HDBSCAN for identifying server groups and outliers
- **Sector Classification**: Automated NAICS sector assignment using keyword matching
- **Hyperparameter Optimization**: Automated tuning of model parameters to minimize outliers and maximize topic coherence

## Dashboard & Visualization

**Note**: Dashboard components have been moved to https://github.com/AI-Safety-Institute/sr-mcp-dashboard from data/final/clservers_classified.csv onwards. This repository focuses on data collection, processing, and analysis.

## Key Dependencies

**Required Python Libraries:**
- `requests` - API interactions
- `pandas` - data manipulation
- `streamlit` - web dashboard
- `plotly` - visualizations
- `nltk` - text processing (auto-downloads required data)
- `aiohttp` - async HTTP requests for GitHub API
- `selenium` - web scraping for official list
- `sentence-transformers` - embedding generation for ML analysis
- `umap-learn` - dimensionality reduction for visualization
- `hdbscan` - clustering for topic analysis
- `bertopic` - topic modeling
- `torch` - GPU acceleration for embeddings

**API Authentication:**
- Smithery API token: `~/.cache/smithery-api/token`
- GitHub token: `GH_TOKEN` environment variable

## Data Files

### Smithery Data Files
- `data/external-servers/smithery_data.json` - Complete Smithery server data

### GitHub Data Files
- `data/external-servers/github_data.json` - Complete GitHub repository data
- `data/external-servers/github_data_summary.json` - Collection statistics

### Official List Data Files
- `data/external-servers/officiallist_data.json` - Complete official server list with GitHub metadata
- `data/external-servers/officiallist_history.json` - Historical tracking data
- `data/external-servers/officiallist_monthly_history.json` - Monthly snapshots
- `data/external-servers/officiallist_urls.json` - Extracted URLs

### Awesome List Data Files
- `data/external-servers/awesomelist_data.json` - Curated awesome-mcp-servers list with emoji metadata (languages, scope, platforms)
- `data/external-servers/awesomelist_data_summary.json` - Summary statistics by category, language, and deployment type

### Dashboard Data Files
- `data/initial/data_unified.json` - Unified dashboard data (27MB, 27,899 servers)
- `data/initial/data_unified_summary.json` - Basic dashboard summary
- `data/initial/data_unified_filtered.json` - Filtered subset for analysis
- `data/initial/data_unified_filtered_summary.json` - Filtered summary

### Embedding & Analysis Data Files
- `output-visuals/topics-embedding/embed_results.json` - Complete embedding analysis results
- `output-visuals/topics-embedding/embed_finance_results.json` - Finance-specific embedding analysis
- `output-visuals/topics-embedding/embed_sector_52_results.json` - Finance & Insurance sector analysis (NAICS 52)
- `embed_*.html` - Interactive visualization files for each analysis
- `embeddings_cache/` - Cached embeddings to avoid recomputation

### Consequentiality Scoring Data Files

**CLServers Pipeline Files:**
- `data/internal-cl/clservers_1_dataprep_summary.json` - CLServers Step 1 data preparation summary
- `data/internal-cl/clservers_1_dataprep_servers_sample.json` - CLServers Step 1 finance server sample
- `clservers_input.jsonl` - Input for CLServers Inspect evaluation
- `data/internal-cl/clservers_3_results.json` - CLServers Step 3 filtering results (JSON)
- `data/final/clservers_classified.csv` - CLServers final classified servers with metadata (CSV)
- `output-validation/cl-validation/clservers_validation.json` - Validation results comparing human vs LLM labels
- `data/external-cl/clservers_validate_labelled.csv` - Human-labeled validation dataset

**CLTools Pipeline Files:**
- `cltools_samples.jsonl` - MCP tool samples for O*NET classification
- `data/internal-cl/cltools_3_results.csv` - CLTools task classification results
- `data/final/cltools_classified.csv` - CLTools enriched with metadata from CLServers
- `data/internal-cl/cltools_prep.json` - Snapshot of all tool records (preprocessed data)
- `data/internal-task-clusters/task_clusters_names.csv` - O*NET task clusters with generated names
- `task_clusters_*.json` - Various clustering summary and result files
- `data/external-cl/cl_onet_taskstatements.csv` - O*NET task statements database
- `data/external-cl/cl_onet_toolsused.csv` - O*NET tools usage database

### Test Data Files
- Various test files for development and validation

## Development Guidelines

### Environment
- **Always use**: `source ~/si_setup/.venv/bin/activate` before running Python
- All commands assume virtual environment is activated

### Logging Standards
- **ALWAYS use logging instead of print() statements** for all terminal output
- Configure logging with both file and console handlers for development visibility
- Use appropriate log levels:
  - `logger.info()` for progress and status messages
  - `logger.warning()` for rate limits and recoverable issues
  - `logger.error()` for errors and exceptions
  - `logger.debug()` for detailed debugging information
- Log files should be named descriptively (e.g., `logs/github_data_run.log`, `logs/bulk_mcp_download.log`)

### Code Quality
- Replace all print() statements with appropriate logging calls
- Maintain existing functionality while improving observability
- Use structured logging format with timestamps for better analysis

### Commit Message Guidelines
- Write concise, direct commit messages that focus on what changed
- Avoid overly descriptive words like "comprehensive", "detailed", "enhanced", "improved"
- Do not mention Claude or AI assistance in commit messages
- Use imperative mood (e.g., "Add feature" not "Added feature")
- Keep messages under 50 characters for the title when possible
- Examples:
  - Good: "Fix rate limiting bug"
  - Bad: "Comprehensive fix for detailed rate limiting issues with enhanced error handling"

### Git and GitHub Setup
- **Remote Configuration**: Use SSH for authentication: `git@github.com:Merlin-St/mcp-monitoring.git`
- **Push to GitHub**: Use `git push origin main` (requires SSH key setup via GitHub CLI)
- **Authentication**: GitHub CLI (`gh`) should be configured with SSH protocol
- **Check auth status**: `gh auth status` to verify SSH configuration
- **Update remote**: `git remote set-url origin git@github.com:Merlin-St/mcp-monitoring.git` if needed
- **IMPORTANT**: Always ensure remote uses SSH, not HTTPS. If push takes too long, check `git remote -v` and update to SSH if needed.

## README Content Filtering

### Overview
The README filtering pipeline removes installation tips and setup instructions while preserving functional descriptions, tool information, and sector-relevant content for embedding analysis and consequentiality scoring.

### Pipeline Stages
1. **Stage 1**: Keyword-based filtering using pattern matching
2. **CLServers Step 2**: LLM-based refinement using Inspect framework

### What Gets Removed
- Package manager commands (`npm install`, `pip install`, `docker run`, etc.)
- Setup instructions and getting started guides
- Environment variable configuration for setup
- Build and compilation instructions
- Development environment setup
- Prerequisites and system requirements
- Code blocks with installation commands

### What Gets Preserved
- Tool descriptions and functionality
- API documentation and capabilities
- Feature lists and what the software does
- Use cases and application domains
- Integration possibilities
- Business logic and workflow descriptions
- Security and compliance features
- Service connections and external integrations

### Example Results

**Example 1: 05-make-your-mcp-server (49.2% reduction)**
- **Before**: 5,685 characters with Docker setup commands, build instructions, and installation steps
- **After**: 2,886 characters focused on MCP server functionality, tools, and usage
- **Removed**: Docker build commands, curl installation, configuration files, deployment instructions
- **Preserved**: Tool descriptions, API explanations, server capabilities, integration examples

**Example 2: 12306-mcp (13.2% reduction)**
- **Before**: 3,018 characters with npm installation and CLI setup
- **After**: 2,619 characters focused on ticket search functionality
- **Removed**: `git clone`, `npm i`, CLI installation commands, configuration setup
- **Preserved**: Feature table, API capabilities, service descriptions, documentation references

### Performance Statistics
- LLM-based filtering removes installation content while preserving functional descriptions
- Servers processed: 9,141 total servers with 7,000+ containing README content
- Output: `readme_filtered` column added to `data/initial/data_unified_filtered.json`

## Rate Limiting

### GitHub API
- Simple rate limiting: Wait 10 seconds when <10 requests remaining
- Automatic rate limit reset waiting when exhausted
- No complex throttling - focus on efficiency

## Common Commands

### Full Data Collection Pipeline
```bash
source ~/si_setup/.venv/bin/activate

# Collect from all 3 sources
python scripts/data-collection-servers/smithery_data_run.py
python scripts/data-collection-servers/github_data_run.py
python scripts/data-collection-servers/officiallist_data_run.py

# Enhance officiallist with GitHub metadata
python scripts/data-collection-servers/officiallist_github_fetcher.py

# Process unified data (27,899 servers)
python data_unified_processor.py

# Create filtered subset for analysis
python data_unified_create_filtered_subset.py
```

### ML Analysis Pipeline
```bash
source ~/si_setup/.venv/bin/activate

# Generate embeddings and sector analysis (requires GPU for optimal performance)
python scripts/data-analysis-topics/embed_generate.py                             # Full dataset analysis
python scripts/data-analysis-topics/embed_generate.py --filter finance           # Finance-only analysis
python scripts/data-analysis-topics/embed_generate.py --filter sector_52         # Finance & Insurance sector (NAICS 52)
python scripts/data-analysis-topics/embed_generate.py --filter sector_54         # Professional Services sector (NAICS 54)

# Results saved as JSON and interactive HTML visualizations
```

### Hyperparameter Optimization
```bash
source ~/si_setup/.venv/bin/activate

# Optimize BERTopic parameters to reduce outliers and improve coherence (HDBSCAN by default)
python scripts/data-analysis-topics/embed_hyperparameter_optimizer.py                        # Full dataset optimization (HDBSCAN only)
python scripts/data-analysis-topics/embed_hyperparameter_optimizer.py --finance              # Finance sector optimization
python scripts/data-analysis-topics/embed_hyperparameter_optimizer.py --52                   # Finance & Insurance sector (NAICS 52)
python scripts/data-analysis-topics/embed_hyperparameter_optimizer.py --54                   # Professional Services sector (NAICS 54)

# Options:
python scripts/data-analysis-topics/embed_hyperparameter_optimizer.py --kmeans               # Include K-means clustering (note: no outliers by definition)
python scripts/data-analysis-topics/embed_hyperparameter_optimizer.py --test-size 500        # Use smaller dataset for faster testing
python scripts/data-analysis-topics/embed_hyperparameter_optimizer.py --max-combinations 50  # Limit parameter combinations to test
python scripts/data-analysis-topics/embed_hyperparameter_optimizer.py --no-cache             # Disable embedding caching

# All results saved to:
# - logs/embed_hyperparameter_optimization.log (or with sector suffix)
# - Check end of log file for recommended configuration

# Apply optimized parameters to scripts/data-analysis-topics/embed_generate.py:
python scripts/data-analysis-topics/embed_apply_optimized_parameters.py logs/embed_hyperparameter_optimization_sector_52.log
python scripts/data-analysis-topics/embed_apply_optimized_parameters.py logs/embed_hyperparameter_optimization.log --dry-run  # Preview changes

# The helper script:
# 1. Parses the "RECOMMENDED CONFIGURATION" section from the optimization log
# 2. Extracts optimized UMAP, HDBSCAN, and Vectorizer parameters
# 3. Automatically modifies scripts/data-analysis-topics/embed_generate.py with the optimal values
# 4. Creates a backup (.backup) before making changes
# 5. Uses regex to find and replace parameter values in the source code
```

### README Content Filtering Pipeline
```bash
source ~/si_setup/.venv/bin/activate

# LLM-based README filtering using Inspect framework (requires ANTHROPIC_API_KEY)
inspect eval scripts/data-cleaning-readmes/data_readme_filter_inspect.py --model anthropic/claude-sonnet-4-5-20250929 --temperature 0

# Process results back to JSON
python scripts/data-cleaning-readmes/data_readme_filter_dfprocessing.py
python scripts/data-cleaning-readmes/data_readme_filter_dfprocessing.py --logs-dir ./logs          # Custom logs directory
python scripts/data-cleaning-readmes/data_readme_filter_dfprocessing.py --eval-file specific.eval  # Process specific eval file

# Complete pipeline workflow:
# 1. Prepare JSONL dataset from data/initial/data_unified_filtered.json
# 2. Run LLM refinement via Inspect framework to filter installation content
# 3. Processing script updates 'readme_filtered' column with refined content
# 4. Output ready for embedding analysis and consequentiality scoring

# Key outputs:
# - data/initial/data_unified_filtered.json (updated with readme_filtered column)
# - data/initial/data_readme_filter_dfprocessing_summary.json (Processing statistics)
# - data_readme_filter_input.jsonl (Inspect input dataset)
# - logs/readme_filter_*.eval (Inspect evaluation results)
```

### Consequentiality Analysis Pipeline (4-Stage Process with NAICS Classification)
```bash
source ~/si_setup/.venv/bin/activate

# 1. Data Preparation - Create filtered dataset for analysis
python scripts/data-classification-servers/clservers_1_dataprep.py                    # Default: 100 servers
python scripts/data-classification-servers/clservers_1_dataprep.py --samples 500      # Custom sample size (more samples)
python scripts/data-classification-servers/clservers_1_dataprep.py --samples 1000     # Large sample for comprehensive analysis
python scripts/data-classification-servers/clservers_1_dataprep.py --all              # Process all servers
python scripts/data-classification-servers/clservers_1_dataprep.py --finance          # Only finance-related servers
python scripts/data-classification-servers/clservers_1_dataprep.py --samples 1000 --finance  # Large finance-focused sample

# 2. CLServers Step 2 - Dual Classification (evaluates ALL servers)
# Finance Identification: Determines if each server is finance-related (binary: yes/no)
inspect eval scripts/data-classification-servers/clservers_2_inspect.py --model anthropic/claude-sonnet-4-5-20250929 --temperature 0

# NAICS Classification: Assigns 3-digit NAICS industry code to each server
inspect eval scripts/data-classification-servers/clservers_2_inspect.py@naics_classification_task --model anthropic/claude-sonnet-4-5-20250929 --temperature 0

# 3. CLServers Step 3 - DataFrame Processing (Both Tasks)
python scripts/data-classification-servers/clservers_3_dfprocessing.py --task finance-identification  # Process finance yes/no classification
python scripts/data-classification-servers/clservers_3_dfprocessing.py --task naics                   # Process NAICS industry codes

# 4. CLServers Step 4 - Data Matching & Merging
python scripts/data-classification-servers/clservers_4_datamatch.py                   # Merges finance + NAICS + metadata

# CLTools Pipeline - O*NET Task Classification
python scripts/data-classification-tools/cltools_main.py --run                      # Run full pipeline (costly)
python scripts/data-classification-tools/cltools_main.py \
    --onet data/internal-task-clusters/task_clusters_names.csv \
    --data data/initial/data_unified_filtered.json \
    --out data/internal-cl/cltools_3_results.csv \
    --model openai/gpt-4o-mini \
    --finance \
    --limit 100

# Enrich CLTools output with metadata
python scripts/data-classification-tools/cltools_datamatch.py \
    --stage4 data/internal-cl/cltools_3_results.csv \
    --stage2 data/final/clservers_classified.csv \
    --usage data/initial/data_unified_filtered.json \
    --output data/final/cltools_classified.csv

# Task Clustering Pipeline
python scripts/onet-task-clusters/task_clusters_run.py --k2 400              # Run clustering with 400 L2 clusters
python scripts/onet-task-clusters/task_clusters_embed_match.py               # Match MCP tools to O*NET tasks

# Complete Pipeline Workflow:
# CLServers Pipeline: Finance Identification + NAICS Classification (ALL servers evaluated)
#   - Step 1: Creates data/internal-cl/clservers_input.jsonl with ALL server data
#   - Step 2a: Finance identification - binary yes/no for each server (finance-related or not)
#   - Step 2b: NAICS classification - assigns 3-digit NAICS industry code to each server
#   - Step 3a: Processes finance .eval files to JSON (data/internal-cl/clservers_3_results.json)
#   - Step 3b: Processes NAICS .eval files to JSON (data/internal-cl/clservers_naics_results.json)
#   - Step 4: Merges finance + NAICS + metadata → final CSV (data/final/clservers_classified.csv)

# CLTools Pipeline: O*NET Task Mapping
#   - Main: Maps MCP tools to O*NET occupational tasks
#   - DataMatch: Enriches results with creation dates and usage data
#   - Output: data/final/cltools_classified.csv with full task and metadata mapping

# Outputs:
# - CLServers: data/final/clservers_classified.csv (ALL servers with finance yes/no + NAICS code + metadata)
# - CLTools: data/final/cltools_classified.csv (O*NET task mappings with metadata)

# Requirements:
# - ANTHROPIC_API_KEY environment variable set
# - Inspect framework installed (pip install inspect_ai)
# - data/initial/data_unified_filtered.json must exist (run scripts/data-unification/data_unified_create_filtered_subset.py first)
# - matplotlib, seaborn, pandas for CLServers visualizations
```

### Testing & Validation
```bash
# Test GitHub collection
python scripts/data-collection-servers/github_data_run.py --test
```

## Known Issues

- No formal testing framework - validation is done through individual script testing
- The data/initial/data_unified.json and data/initial/data_unified_filtered.json are very big files, so you cant use read() for them directly


# GENERAL GUIDELINES

The blow provides general coding guidelines for Claude Code across all projects.

## Git Commit Messages

Write concise commit messages that add value beyond what's obvious from the code:
- Avoid redundancy with the commit title
- Don't list low-level changes that are self-evident
- Focus on the "why" when it's not obvious
- Include nuanced details that might be misunderstood
- Keep messages brief and to the point
- No need to talk at length about claude code

## Collaborative Development

### Understanding Intent
- Humans often ask for what they think they want, not necessarily what they need
- Initial requests are starting points for collaborative refinement
- Take time to understand the underlying problem before jumping to solutions
- Ask clarifying questions early to avoid building the wrong thing
- Consider the broader context and goals behind specific requests

### Asking Clarifying Questions
- When requirements seem ambiguous, ask for clarification
- If multiple valid interpretations exist, present them and ask which is intended
- Question assumptions that might lead to suboptimal solutions
- Ask about edge cases and error handling expectations
- Clarify scope when it's unclear (e.g., "Should this handle X case as well?")

### Providing Guidance
- Suggest best practices even when not explicitly asked
- Offer alternative approaches when you see better solutions
- Explain trade-offs between different implementation choices
- Share relevant patterns from other parts of the codebase
- Point out potential issues or improvements proactively

## Development Guidelines

### Before Writing Code
- Always explore the codebase first to find existing patterns, utilities, or related implementations
- Plan your approach before implementing - don't jump straight to coding
- When asked to implement something, first understand the context and existing code
- Use extended thinking ("think", "think hard", etc.) for complex problems that need deeper analysis
- Present your understanding and plan before implementing if the task is complex

### Implementation Approach
- Start with a simple, minimal solution and iterate to improve
- Verify the reasonableness of your solution as you implement
- Follow existing patterns and conventions in the codebase
- Reuse existing libraries and utilities rather than reimplementing functionality
- Consider edge cases and error handling from the start
- Be open about limitations or concerns with the current approach

### Test-Driven Development
- When possible, write tests before implementation
- Create tests based on expected behavior, not current implementation
- Ensure tests fail before implementing the solution
- Don't modify tests to make them pass - fix the implementation instead
- Write tests that clearly express intent and expected behavior
- Suggest adding tests even when not explicitly requested

### Iteration and Refinement
- Expect to iterate - first versions are rarely perfect
- When given visual targets or test cases, iterate until the solution matches
- Verify implementations aren't overfitting to specific test cases
- Take feedback and refine solutions through multiple iterations
- Be open to changing direction based on new understanding

### Code Quality
- Write self-documenting code that minimizes cognitive load
- Prefer editing existing files over creating new ones
- Never create documentation files unless explicitly requested
- Ensure linting and formatting tools are properly configured
- Never introduce code that exposes or logs secrets
- Consider maintainability and future developers
- Don't commit things from servers or other production state, always resolve the changes properly through the development workflow

### Communication Best Practices
- Explain your reasoning when making non-obvious choices
- Be transparent about uncertainties or areas where you need guidance
- Acknowledge when you're making assumptions and verify them
- Provide context for why certain approaches are preferred
- Admit when something is outside your expertise and suggest resources

## General Principles
- Quality over speed - take time to understand before implementing
- Collaboration over isolation - work with the human to refine solutions
- Clarity over cleverness - write code that's easy to understand
- Pragmatism over perfection - balance ideal solutions with practical constraints
- Learning over knowing - be open about knowledge gaps and learn from the codebase
- NEVER Do unrelated changes
- ALWAYS specifically report any fallbacks you have implemented

### Command Line Tools
- Use ag instead of rg for code searching
- rg is not installed. use ag instead
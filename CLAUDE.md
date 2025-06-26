# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

**IMPORTANT**: Always activate the virtual environment before running Python scripts:
```bash
source ~/si_setup/.venv/bin/activate
```

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

## Data Collection Strategy (3 Approaches)

### 1. Smithery MCP Server Database
```bash
# Download all MCP server data from Smithery Registry
python smithery_run_bulk_mcp_download.py
```

### 2. GitHub Repository Scanning
```bash
# Scan GitHub for 'mcp server' repositories
python github_mcp_repo_collector.py

# Options:
python github_mcp_repo_collector.py --test          # Test mode (10 repos)
python github_mcp_repo_collector.py --graphql       # Use GraphQL API
# Default: REST API with daily date-based searches
```

### 3. Official ModelContextProtocol/Servers List
```bash
# Scrape official MCP servers list
python officiallist_url_scraping.py
```

## Core Architecture

**Data Flow:**
1. **Collection**: Three parallel data collection streams → JSON data files
2. **Analysis**: Unified analysis processing server data with financial risk categorization
3. **Visualization**: Streamlit dashboard with 2 graphs & 2 tables

**Key Files:**

**Data Collection:**
- `smithery_run_bulk_mcp_download.py` - Smithery API collection entry point
- `smithery_bulk_mcp_downloader.py` - Core Smithery download logic
- `smithery_mcp_api_handler.py` - Smithery API interaction handler
- `github_mcp_repo_collector.py` - GitHub repository scanning
- `github_mcp_repo_searcher.py` - GitHub search functionality
- `officiallist_url_scraping.py` - Official list scraping
- `officiallist_html_fetcher.py` - HTML content fetching
- `officiallist_url_extractor.py` - URL extraction from HTML

**Dashboard & Analysis:**
- `dashboard_unified_mcp.py` - Main comprehensive dashboard (supports --filtered flag for filtered dataset)
- `dashboard_unified_mcp_data_processor.py` - Unified data processing (27,899 servers)
- `dashboard_launch.py` - Simple dashboard launcher with --filtered support
- `dashboard_tmux_launcher.py` - Persistent tmux session launcher with --filtered support
- `dashboard_verify_data.py` - Data validation utility
- `data_unified_mcp_data_processor.py` - Enhanced data processing pipeline
- `data_create_filtered_subset.py` - Create filtered subsets for analysis

**ML Analysis & Embeddings:**
- `embed_generate.py` - GPU-accelerated embedding generation with NAICS classification
- `naics_classification_config.py` - NAICS sector definitions and keyword mappings

**Utilities:**
- `smithery_quickcheck_bulk_mcp_data.py` - Data validation
- `smithery_bulk_mcp_config.py` - Configuration management
- `officiallist_github_fetcher.py` - GitHub metadata collection for officiallist servers
- `verify_github_coverage.py` - Analyze GitHub coverage gaps

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

## Dashboard Outputs

### 2 Graphs
1. **MCP servers creation and usage over time** - Shows trend analysis with sub-graphs focusing on:
   - Servers executing payments
   - Other finance servers
   - Clickable example MCP servers

2. **Finance tool availability trends** - Growth of finance-specific capabilities

### 2 Tables
1. **Overview of all tools relevant to automatic payments**
2. **Overview table of all MCP servers and tools** - Filterable by sector, use case, autonomy level

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
- `smithery_all_mcp_server_summaries.json` - Complete Smithery server data

### GitHub Data Files
- `github_mcp_repositories.json` - Complete GitHub repository data
- `github_mcp_repositories_partial.json` - Partial/in-progress data
- `github_mcp_collection_summary.json` - Collection statistics

### Official List Data Files
- `officiallist_mcp_servers_full.json` - Complete official server list with GitHub metadata
- `officiallist_mcp_servers.json` - Processed official server data
- `officiallist_history.json` - Historical tracking data
- `officiallist_monthly_history.json` - Monthly snapshots
- `officiallist_urls.json` - Extracted URLs
- `officiallist_github_metadata.json` - GitHub metadata for officiallist servers

### Dashboard Data Files
- `data_unified.json` - Unified dashboard data (27MB, 27,899 servers)
- `data_unified_summary.json` - Dashboard summary
- `data_unified_filtered.json` - Filtered subset for analysis
- `data_unified_filtered_summary.json` - Filtered summary

### Embedding & Analysis Data Files
- `embed_results.json` - Complete embedding analysis results
- `embed_finance_results.json` - Finance-specific embedding analysis
- `embed_sector_52_results.json` - Finance & Insurance sector analysis (NAICS 52)
- `embed_sector_54_results.json` - Professional Services sector analysis (NAICS 54)
- `embed_*.html` - Interactive visualization files for each analysis
- `embeddings_cache/` - Cached embeddings to avoid recomputation
- `smithery_all_mcp_server_details_complete.json` - Enhanced server details

### Test Data Files
- `officiallist_mcp_servers_test*.json` - Test datasets
- `officiallist_urls_test*.json` - Test URL datasets
- `officiallist_test_results.json` - Test results
- `embed_test_*.json` - Embedding test results

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
- Log files should be named descriptively (e.g., `github_mcp_collection.log`, `bulk_mcp_download.log`)

### Code Quality
- Replace all print() statements with appropriate logging calls
- Maintain existing functionality while improving observability
- Use structured logging format with timestamps for better analysis

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
python smithery_run_bulk_mcp_download.py
python github_mcp_repo_collector.py  
python officiallist_url_scraping.py

# Enhance officiallist with GitHub metadata
python officiallist_github_fetcher.py

# Process unified data (27,899 servers)
python dashboard_unified_mcp_data_processor.py

# Launch dashboards
streamlit run dashboard_unified_mcp.py               # Main unified dashboard (full dataset)
streamlit run dashboard_unified_mcp.py --filtered    # Main dashboard with filtered dataset
python dashboard_launch.py                           # Simple launcher (supports --filtered)
python dashboard_launch.py --filtered                # Simple launcher with filtered dataset
python dashboard_tmux_launcher.py start unified      # Persistent tmux session (full dataset)
python dashboard_tmux_launcher.py start unified --filtered  # Persistent session with filtered dataset
```

### ML Analysis Pipeline
```bash
source ~/si_setup/.venv/bin/activate

# Generate embeddings and sector analysis (requires GPU for optimal performance)
python embed_generate.py                             # Full dataset analysis
python embed_generate.py --filter finance           # Finance-only analysis
python embed_generate.py --filter sector_52         # Finance & Insurance sector (NAICS 52)
python embed_generate.py --filter sector_54         # Professional Services sector (NAICS 54)

# Results saved as JSON and interactive HTML visualizations
```

### Testing & Validation
```bash
# Quick validation
python smithery_quickcheck_bulk_mcp_data.py

# Test GitHub collection
python github_mcp_repo_collector.py --test
```

## Known Issues

- `smithery_mcp_api_handler.py:4` has broken import: `from hf_models_monitoring_test.config_utils` - this module path needs to be updated
- No formal testing framework - validation is done through quickcheck scripts
- GitHub rate limiting may require patience for full collection runs
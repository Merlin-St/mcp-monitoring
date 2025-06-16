# MCP Server Monitoring Dashboard

## Overview
Comprehensive dashboard tracking **27,899 MCP servers** across 3 data sources, with focus on **966 finance-related servers**. Analyzes AI tool proliferation and finance sector adoption.

## Quick Start
```bash
# Activate environment
source ~/si_setup/.venv/bin/activate

# Process data (if needed)
python dashboard_unified_mcp_data_processor.py

# Launch dashboard
streamlit run dashboard_unified_mcp.py
```

Dashboard available at: http://localhost:8501

## Data Collection (3 Sources)
1. **Smithery API** (6,434 servers) - `smithery_run_bulk_mcp_download.py`
2. **GitHub Search** (21,053 repos) - `github_mcp_repo_collector.py`  
3. **Official List** (966 servers) - `officiallist_url_scraping.py`

## Dashboard Features
- **5 Sections**: Overview, Growth Trends, Technology Analysis, Finance Analysis, Server Explorer
- **Interactive Filtering**: By source, language, finance relevance
- **Visualizations**: Growth charts, distribution analysis, technology trends

## Key Files
- `dashboard_unified_mcp.py` - Main dashboard
- `dashboard_unified_mcp_data_processor.py` - Data unification
- `dashboard_mcp_servers_unified.json` - Unified dataset (27MB)

## Research Focus
Tracks AI tool ecosystem growth with specific attention to finance sector tools, autonomous capabilities, and consequential system impact.
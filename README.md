# MCP Server Monitoring Dashboard

## Overview
Comprehensive dashboard tracking **27,899 MCP servers** across 3 data sources, with focus on **966 finance-related servers**. Analyzes AI tool proliferation and finance sector adoption using advanced ML techniques.

## 🚀 Quick Start
```bash
# Activate environment
source ~/si_setup/.venv/bin/activate

# Process data (if needed)
python dashboard_unified_mcp_data_processor.py

# Launch dashboard in persistent tmux session (recommended)
python dashboard_tmux_launcher.py start unified

# Use filtered dataset (smaller, faster loading)
python dashboard_tmux_launcher.py start unified --filtered

# Stop when done
python dashboard_tmux_launcher.py stop unified
```

## 🔬 ML Analysis Features
- **Semantic Embeddings**: High-quality text analysis using sentence-transformers
- **NAICS Classification**: Sector classification across 20 industries, based on the [latest official US classification](https://www.census.gov/naics/reference_files_tools/2022_NAICS_Manual.pdf). It can be mapped to [O*NET](https://www.onetonline.org/find/industry?i=52)
- **Topic Modeling**: BERTopic for discovering server themes and clusters
- **Interactive Visualizations**: 2D/3D embeddings with clustering analysis
- **GPU Acceleration**: Optimized for CUDA with caching for fast iterations

## Data Processing Pipeline: 3 Sources → Unified Dataset

### Raw Data Sources & Structure

#### 1. **Smithery API** (`smithery_all_mcp_server_summaries.json`)
- **Shape**: 6,434 servers × 6 columns
- **Columns**: `qualifiedName`, `displayName`, `description`, `createdAt`, `useCount`, `homepage`  
- **Sample Data**:
  - `qualifiedName`: `@wonderwhy-er/desktop-commander`
  - `displayName`: `Desktop Commander`
  - `useCount`: `579226` (usage metrics)

#### 2. **GitHub Repositories** (`github_mcp_repositories.json`)
- **Shape**: 21,053 repos × 83 columns (GitHub API fields)
- **Key Columns**: `readme_content`, `owner`, `license`, `topics`, `stargazers_count`, `language`, `permissions`
- **Sample Data**:
  - `readme_content`: Full README files (up to 22,000 characters)
  - `owner`: `{'login': 'phil65', 'id': 110931, 'avatar_url': '...'}`  
  - `topics`: `['mcp-server', 'ai-tools', 'claude']`

#### 3. **Official MCP List** (`officiallist_mcp_servers_full.json`)
- **Shape**: 966 servers × 5 columns per server
- **Structure**: `{fetch_date, total_servers, servers: [...]}`
- **Server Columns**: `name`, `url`, `description`, `is_github`, `extracted_date`
- **Sample Data**:
  - `name`: `Everything`
  - `url`: `https://github.com/modelcontextprotocol/src/everything`
  - `description`: `Reference / test server with prompts, resources, and tools`

### Processing Pipeline (`dashboard_unified_mcp_data_processor.py`)

```
Raw Sources → Load Data → Process Sources → Deduplicate → Enhance → Save Unified & Filtered versions
     ↓             ↓           ↓             ↓          ↓         ↓
  3 JSON files   Parse    UnifiedMCPServer   Merge    Classify   Final JSON
  83+ columns    Data      Objects         Conflicts  Sectors   
```

#### **Key Processing Steps**:

1. **Load Data** (lines 89-135): Read all 3 JSON files with error handling
2. **Process Sources** (lines 214-378): Convert each source to standardized `UnifiedMCPServer` objects
3. **Deduplication** (lines 176-193): Generate unique IDs from URLs/names, merge duplicates by priority
4. **Enhancement** (lines 380-422): Add finance classification, determine primary source, set canonical names
5. **Save Filtered** (lines 424-478): Export ~25 core fields, **exclude large content fields**

#### **Sector Classification** (e.g. for Finance)
- **Keywords**: `finance`, `trading`, `payment`, `bank`, `crypto`, `market`, `investment`, etc.
- **Text Sources**: `name + description + qualified_name + topics`
- **Result**: `is_finance_related` boolean flag


**Current Output Structure** (`data_unified.json`):
```json
{
  "id": "phil65/llmling", 
  "name": "LLMling",
  "description": "Easy MCP servers and AI agents, defined as YAML",
  "github_url": "https://github.com/phil65/LLMling",
  "stargazers_count": 42,
}
```

## Dashboard Features
- **5 Sections**: Overview, Growth Trends, Technology Analysis, Finance Analysis, Server Explorer
- **Interactive Filtering**: By source, language, finance relevance
- **Visualizations**: Growth charts, distribution analysis, technology trends

## 🧬 Topic modelling
```bash
# Generate embeddings and sector analysis (GPU recommended)
python embed_generate.py                     # Full dataset analysis
python embed_generate.py --filter sector_52 # Finance & Insurance (NAICS 52)

# Results: JSON data + interactive HTML visualizations
```

## 📁 Key Files
### Dashboard Files
- `dashboard_unified_mcp.py` - Main comprehensive dashboard (supports --filtered flag)
- `dashboard_launch.py` - Simple launcher with --filtered support
- `dashboard_tmux_launcher.py` - Persistent tmux session launcher with --filtered support
- `dashboard_verify_data.py` - Data validation utility

### Main Data
- `dashboard_unified_mcp_data_processor.py` - Data unification (27,899 servers)
- `data_unified.json` - Full unified dataset (343MB, 27,899 servers)
- `data_unified_filtered.json` - Filtered dataset (225MB, core fields only)
- `data_unified_summary.json` - Dataset statistics and metadata

### ML Analysis
- `embed_generate.py` - GPU-accelerated embedding generation
- `naics_classification_config.py` - NAICS sector definitions
- `embed_*.json` - Analysis results by sector/filter
- `embed_*.html` - Interactive visualizations

### Data Collection
- `smithery_run_bulk_mcp_download.py` - Smithery API (6,434 servers)
- `github_mcp_repo_collector.py` - GitHub scanning (21,053 repos) 
- `officiallist_url_scraping.py` - Official list (966 servers)

## 🎯 Research Focus
Tracks AI tool ecosystem growth with specific attention to:
- **Finance sector tools** and autonomous capabilities
- **Consequential system impact** assessment
- **NAICS sector classification** across 20 industries
- **Semantic clustering** of server capabilities and use cases
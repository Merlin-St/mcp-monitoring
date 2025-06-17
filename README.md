# MCP Server Monitoring Dashboard

## Overview
Comprehensive dashboard tracking **27,899 MCP servers** across 3 data sources, with focus on **966 finance-related servers**. Analyzes AI tool proliferation and finance sector adoption.

## Quick Start
```bash
# Process data (if needed)
python dashboard_unified_mcp_data_processor.py

# Launch dashboard in persistent tmux session (recommended)
python dashboard_tmux_launcher.py start unified

# Stop when done
python dashboard_tmux_launcher.py stop unified
```

## Data Processing Pipeline: 3 Sources → Unified Dataset

### Raw Data Sources & Structure

#### 1. **Smithery API** (`smithery_all_mcp_server_summaries.json`)
- **Shape**: 6,434 servers × 6 columns
- **Columns**: `qualifiedName`, `displayName`, `description`, `createdAt`, `useCount`, `homepage`  
- **Sample Data**:
  - `qualifiedName`: `@wonderwhy-er/desktop-commander`
  - `displayName`: `Desktop Commander`
  - `useCount`: `579226` (usage metrics)
- **Content**: Clean, structured server metadata with usage statistics
- **🟢 Data Retention**: All fields preserved in unified dataset

#### 2. **GitHub Repositories** (`github_mcp_repositories.json`)
- **Shape**: 21,053 repos × 83 columns (GitHub API fields)
- **Key Columns**: `readme_content`, `owner`, `license`, `topics`, `stargazers_count`, `language`, `permissions`
- **Sample Data**:
  - `readme_content`: Full README files (up to 22,000 characters)
  - `owner`: `{'login': 'phil65', 'id': 110931, 'avatar_url': '...'}`  
  - `topics`: `['mcp-server', 'ai-tools', 'claude']`
- **Content**: Rich repository metadata, full README content, licensing, social metrics
- **🔴 Data Loss**: `readme_content`, `license`, `owner` details, `permissions`, 40+ GitHub fields excluded

#### 3. **Official MCP List** (`officiallist_mcp_servers_full.json`)
- **Shape**: 966 servers × 5 columns per server
- **Structure**: `{fetch_date, total_servers, servers: [...]}`
- **Server Columns**: `name`, `url`, `description`, `is_github`, `extracted_date`
- **Sample Data**:
  - `name`: `Everything`
  - `url`: `https://github.com/modelcontextprotocol/src/everything`
  - `description`: `Reference / test server with prompts, resources, and tools`
- **Content**: Curated official servers with verified descriptions
- **🟢 Data Retention**: All server fields preserved (meta fields like `fetch_date` excluded)

### Processing Pipeline (`dashboard_unified_mcp_data_processor.py`)

```
Raw Sources → Load Data → Process Sources → Deduplicate → Enhance → Save Unified
     ↓             ↓           ↓             ↓          ↓         ↓
  3 JSON files   Parse    UnifiedMCPServer   Merge    Classify   Final JSON
  83+ columns    Data      Objects         Conflicts  Finance   ~25 columns
```

#### **Key Processing Steps**:

1. **Load Data** (lines 89-135): Read all 3 JSON files with error handling
2. **Process Sources** (lines 214-378): Convert each source to standardized `UnifiedMCPServer` objects
3. **Deduplication** (lines 176-193): Generate unique IDs from URLs/names, merge duplicates by priority
4. **Enhancement** (lines 380-422): Add finance classification, determine primary source, set canonical names
5. **Save Filtered** (lines 424-478): Export ~25 core fields, **exclude large content fields**

#### **Deduplication Strategy**:
- **ID Generation**: `qualified_name > repo_name > normalized_name > url_hash`  
- **Merge Priority**: Smithery > GitHub > Official (preserves highest-quality metadata)
- **Conflict Resolution**: First-come-wins for most fields, append data sources list

#### **Finance Classification**:
- **Keywords**: `finance`, `trading`, `payment`, `bank`, `crypto`, `market`, `investment`, etc.
- **Text Sources**: `name + description + qualified_name + topics`
- **Result**: `is_finance_related` boolean flag

### **Critical Data Losses in Unified Dataset**

| Source | **Excluded Fields** | **Impact** |
|--------|-------------------|-----------|
| **GitHub** | `readme_content` | ❌ **No README content** (up to 22K chars) |
| **GitHub** | `license`, `permissions` | ❌ **No licensing info** |  
| **GitHub** | `owner` (detailed), `topics`, `languages` | ❌ **Limited repo metadata** |
| **GitHub** | `has_issues`, `has_wiki`, `has_pages` | ❌ **No GitHub features info** |
| **Smithery** | None | ✅ **All data preserved** |
| **Official** | `fetch_date`, `total_servers` | ✅ **Core data preserved** |

### **Why README/Tools Missing**

The processor **does collect** README content from GitHub (`line 294: server.readme_content = item.get('readme_content')`) but **excludes it** from final output due to:

1. **File Size**: README content would increase dataset from 27MB to ~200MB+
2. **Processing Speed**: Large text fields slow dashboard loading  
3. **Memory Usage**: Browser/Streamlit performance impact

**Current Output Structure** (`dashboard_mcp_servers_unified.json`):
```json
{
  "id": "phil65/llmling", 
  "name": "LLMling",
  "description": "Easy MCP servers and AI agents, defined as YAML",
  "github_url": "https://github.com/phil65/LLMling",
  "stargazers_count": 42,
  // ❌ readme_content: EXCLUDED
  // ❌ license: EXCLUDED  
  // ❌ detailed owner: EXCLUDED
}
```

### **Solution: Include Rich Content**

To restore README content and licensing info, modify `dashboard_unified_mcp_data_processor.py:432-464`:

```python
server_dict = {
    # ... existing fields ...
    'readme_content': server.readme_content,  # ADD THIS
    'license': getattr(server, 'license', None),  # ADD THIS
    'owner_details': getattr(server, 'owner_details', None),  # ADD THIS
}
```

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
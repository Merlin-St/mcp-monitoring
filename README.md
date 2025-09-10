# MCP Server Monitoring Dashboard

## Overview
Comprehensive dashboard tracking **27,899 MCP servers** across 3 data sources, with focus on **966 finance-related servers**. Analyzes AI tool proliferation and finance sector adoption using advanced ML techniques including semantic embeddings, topic modeling, and automated consequentiality scoring.

## 🚀 Quick Start
```bash
# Activate environment
uv sync # or old via source ~/si_setup/.venv/bin/activate

# Process data
python data_unified_processor.py

# Create filtered subset for analysis
python data_unified_create_filtered_subset.py
```

**Note**: Dashboard components have been moved to https://github.com/AI-Safety-Institute/sr-mcp-dashboard. This repository focuses on data collection, processing, and ML analysis.

## 🔬 ML Analysis Features
- **Semantic Embeddings**: High-quality text analysis using sentence-transformers
- **NAICS Classification**: Sector classification across 20 industries, based on the [latest official US classification](https://www.census.gov/naics/reference_files_tools/2022_NAICS_Manual.pdf). It can be mapped to [O*NET](https://www.onetonline.org/find/industry?i=52)
- **Topic Modeling**: BERTopic for discovering server themes and clusters
- **Hyperparameter Optimization**: Automated tuning to minimize outliers and maximize topic coherence
- **Interactive Visualizations**: 2D/3D embeddings with clustering analysis
- **GPU Acceleration**: Optimized for CUDA with caching for fast iterations

## Data Processing Pipeline: 3 Sources → Unified Dataset

### Raw Data Sources & Structure

#### 1. **Smithery API** (`smithery_data.json`)
- **Shape**: 6,434 servers × 6 columns
- **Columns**: `qualifiedName`, `displayName`, `description`, `createdAt`, `useCount`, `homepage`  
- **Sample Data**:
  - `qualifiedName`: `@wonderwhy-er/desktop-commander`
  - `displayName`: `Desktop Commander`
  - `useCount`: `579226` (usage metrics)

#### 2. **GitHub Repositories** (`github_data.json`)
- **Shape**: 21,053 repos × 83 columns (GitHub API fields)
- **Key Columns**: `readme_content`, `owner`, `license`, `topics`, `stargazers_count`, `language`, `permissions`
- **Sample Data**:
  - `readme_content`: Full README files (up to 22,000 characters)
  - `owner`: `{'login': 'phil65', 'id': 110931, 'avatar_url': '...'}`  
  - `topics`: `['mcp-server', 'ai-tools', 'claude']`

#### 3. **Official MCP List** (`officiallist_data.json`)
- **Shape**: 966 servers × 5 columns per server
- **Structure**: `{fetch_date, total_servers, servers: [...]}`
- **Server Columns**: `name`, `url`, `description`, `is_github`, `extracted_date`
- **Sample Data**:
  - `name`: `Everything`
  - `url`: `https://github.com/modelcontextprotocol/src/everything`
  - `description`: `Reference / test server with prompts, resources, and tools`

### Processing Pipeline (`data_unified_processor.py`)

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

## Core Features
- **Data Collection**: 3 parallel sources (Smithery, GitHub, Official MCP list)
- **ML Analysis**: Semantic embeddings, topic modeling, NAICS classification
- **Finance Focus**: 966 finance-related servers with consequentiality scoring
- **Processing Pipeline**: Unified dataset with deduplication and enhancement
- **Usage Statistics**: Download statistics for PyPI and npm packages with 70.6% coverage

## 📦 Usage Statistics Collection

### Overview
The project now includes comprehensive package usage statistics through a **strict 1:1 package-to-repository matching system** that replaces the previous Libraries.io approach. This system provides accurate download statistics from PyPI and npm while ensuring no repository is matched to multiple packages.

### Key Improvements Over Previous System
- **70.6% Coverage**: 3,450 matched packages out of 4,886 total discovered packages
- **Strict 1:1 Matching**: Each repository matched to only ONE package (eliminates many-to-one conflicts)
- **Direct API Sources**: PyPI BigQuery + npm registry APIs (replacing outdated Libraries.io data)
- **Monthly Breakdown**: Download statistics from November 2024 to present

### Package Discovery & Matching
```bash
# Final package matching results stored in:
usage_match.json

# Package discovery used search terms across PyPI and npm:
# - 'mcp server', 'mcp-server', 'modelcontextprotocol', 'mcp'
# - Found: 4,886 total packages (PyPI: 4,528, npm: 358)
# - Matched: 3,450 packages via competitive 1:1 matching (70.6% coverage)
```

### Usage Statistics Integration
The matched packages are integrated into the main dataset with download statistics:

```bash
# Collect and integrate usage statistics
python usage_run.py                    # Full PyPI + npm collection (uses pre-downloaded PyPI data)
python usage_run.py --skip-pypi        # npm only (faster testing)
python usage_run.py --skip-npm         # PyPI only (uses pre-downloaded data)

# Output: data_usage.json (890MB file with integrated usage fields)

# PyPI Data Collection: 
# Run this BigQuery query in Google Cloud Console and save as 'usage_bigquery_webresults_pypi.json':
# SELECT file.project AS package_name, FORMAT_DATE('%Y-%m', DATE_TRUNC(DATE(timestamp), MONTH)) AS month, COUNT(*) AS downloads
# FROM `bigquery-public-data.pypi.file_downloads` 
# WHERE LOWER(file.project) LIKE '%mcp%' AND DATE(timestamp) >= '2024-11-01' AND DATE(timestamp) < '2025-09-01'
# GROUP BY package_name, month ORDER BY package_name, month
```

### Dataset Enhancement
The final `data_usage.json` (890MB) contains 16,940 repositories with integrated download statistics:

- `usage_pypi_downloads`: Total PyPI downloads since Nov 2024
- `usage_npm_downloads`: Total npm downloads since Nov 2024  
- `usage_total_downloads`: Combined download count
- `usage_monthly_breakdown`: Month-by-month download statistics (Nov 2024 - Aug 2025)
- `usage_matched_packages`: List of matched PyPI/npm packages for each repository
- `usage_last_updated`: Statistics collection date

### Matching Strategy
**Two-Tier Competitive Matching:**
1. **Tier 1 (Confirmed)**: 783 exact URL/name matches from enhanced fuzzy matching
2. **Tier 2 (Strict 1:1)**: 2,667 competitive matches using prefix indexing + fuzzy scoring

**Quality Thresholds:**
- 90+ confidence: 2,667 matches used for download statistics
- Competitive assignment ensures each repository → single best package match
- Eliminated previous many-to-one conflicts (e.g., 87+ packages to one repo)

### Final Results Summary
- **Total Downloads Collected**: 87.7 million (74.5M PyPI + 13.1M npm)
- **Package Coverage**: 70.6% of discovered MCP packages matched to repositories (3,450/4,886)
- **Repository Coverage**: 1,431 repositories with download statistics (8.4% of 16,940 total)
- **PyPI Coverage**: 97.6% match rate (3,018/3,092 packages matched), 80.4% of all PyPI downloads captured
- **npm Coverage**: 59.5% match rate (213/358 packages matched), 100% of matched packages have data
- **Data Quality**: 90+ confidence threshold, strict 1:1 repository matching, monthly breakdown Nov 2024-Aug 2025
- **Missing Downloads**: 18.2M PyPI downloads (19.6%) from 3,723 non-matched packages - requires manual mapping or dataset expansion

## 🧬 Topic Modeling & Optimization
```bash
# Generate embeddings and sector analysis (GPU recommended)
python embed_generate.py --clustering hdbscan                     # Full dataset analysis
python embed_generate.py --filter sector_52 --clustering hdbscan # Finance & Insurance (NAICS 52)

# Optimize BERTopic parameters for better results (HDBSCAN by default)
python embed_hyperparameter_optimizer.py                        # Full dataset optimization (≥50 topics required)
python embed_hyperparameter_optimizer.py --finance              # Finance sector optimization (≥10 topics required)
python embed_hyperparameter_optimizer.py --kmeans               # Include K-means (note: no outliers)
python embed_hyperparameter_optimizer.py --test-size 500        # Faster testing with smaller dataset
python embed_hyperparameter_optimizer.py --max-combinations 50  # Limit parameter combinations
python embed_hyperparameter_optimizer.py --min-topics-sector 5  # Custom minimum topics for sectors
python embed_hyperparameter_optimizer.py --min-topics-full 25   # Custom minimum topics for full dataset

# Apply optimized parameters to embed_generate.py
python embed_apply_optimized_parameters.py embed_hyperparameter_optimization_sector_52.log

# Complete optimization pipeline (one command) - ensures ≥10 topics for finance sector
python embed_hyperparameter_optimizer.py --finance --test-size 500 --max-combinations 20 && python embed_apply_optimized_parameters.py embed_hyperparameter_optimization_sector_52.log && python embed_generate.py --52 --clustering hdbscan

# Results: JSON data + interactive HTML visualizations + optimization logs
```

## 📁 Key Files
### Main Data
- `data_unified_processor.py` - Data unification (27,899 servers)
- `data_unified.json` - Full unified dataset (343MB, 27,899 servers)
- `data_unified_filtered.json` - Filtered dataset (225MB, core fields only)
- `data_unified_summary.json` - Dataset statistics and metadata

### ML Analysis
- `embed_generate.py` - GPU-accelerated embedding generation
- `embed_hyperparameter_optimizer.py` - Automated hyperparameter optimization
- `embed_apply_optimized_parameters.py` - Helper to apply optimized parameters automatically
- `naics_classification_config.py` - NAICS sector definitions
- `embed_*.json` - Analysis results by sector/filter
- `embed_*.html` - Interactive visualizations
- `embed_hyperparameter_optimization_*.log` - Optimization results and recommendations

### Consequentiality Scoring (2-Stage Pipeline)
- `data_tools_extraction_utils.py` - Tool extraction and access level classification utilities
- `conseq_ground_truth_tools_sample.json` - Random tools sample for ground truth scoring
- `stage_data_prep.py` - Stage 1: Data preparation with sampling options (--samples 500, --all, --finance)
- `stage_stage1_inspect.py` - Stage 1: Finance tool identification using Inspect framework
- `stage_stage1_dfprocessing.py` - Stage 1: Process .eval files to JSON/CSV
- `stage_stage3_visual.py` - Stage 3: Visualization and top tools analysis

### Data Collection
- `smithery_data_run.py` - Smithery API (6,434 servers)
- `github_data_run.py` - GitHub scanning (21,053 repos) 
- `officiallist_data_run.py` - Official list (966 servers)

### Usage Statistics
- `usage_run.py` - Package download statistics collection (pre-downloaded PyPI data + npm API)
- `usage_match.json` - Final 1:1 matched packages (3,450 packages, 70.6% coverage)
- `usage_bigquery_webresults_pypi.json` - Pre-downloaded PyPI statistics (5,024 packages, 92.7M downloads)
- `data_usage.json` - Final dataset with integrated download statistics (890MB, 87.7M downloads)

## 🎯 Research Focus
Tracks AI tool ecosystem growth with specific attention to:
- **Finance sector tools** and autonomous capabilities
- **Consequential system impact** assessment through ground truth scoring
- **NAICS sector classification** across 20 industries with keyword-based automation
- **Semantic clustering** of server capabilities and use cases using advanced embeddings
- **2-stage analysis pipeline** for finance-specific filtering and visualization

## 🔍 Consequentiality Analysis Pipeline (2-Stage Process)

### Stage 1: Data Preparation & Finance Filtering
```bash
# Data preparation with various sampling options
python stage_data_prep.py --samples 500           # Analyze 500 servers
python stage_data_prep.py --samples 1000          # Analyze 1000 servers  
python stage_data_prep.py --all                   # Analyze all servers
python stage_data_prep.py --finance               # Only finance-related servers
python stage_data_prep.py --samples 1000 --finance # Large finance-focused sample

# Finance tool identification using LLM evaluation
inspect eval stage_stage1_inspect.py --model anthropic/claude-sonnet-4-20250514
python stage_stage1_dfprocessing.py               # Convert .eval files to JSON/CSV
```

### Stage 3: Visualization & Analysis
```bash
# Generate charts and analysis based on Stage 1 results
python stage_stage3_visual.py
```

**Pipeline Output:**
- **Stage 1**: `stage_stage1_results.json/csv` (finance-relevant servers)
- **Stage 3**: PNG charts + finance server analysis + summary statistics

## 📊 LLM Validation Results

Human-labeled validation against LLM consequentiality scoring across 92 servers shows **83.2% mean accuracy** across 15 fields with systematic over-estimation bias in level classification.

### Consequentiality Level Performance

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

**Takeaway**: LLM struggles to distinguish between low-risk information gathering (L1) and limited interaction (L2) categories, suggesting these should be combined for practical consequentiality assessment.
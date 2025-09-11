# Hierarchical O*NET Task Classification

## Quick start
  1. (If data/internal-task-clusters/task_clusters_names.csv does not yet exist) ../onet-task-clusters/task_clusters_run.py --k2 400 (once, to create task hierarchy)
  2. cltools_main.py --run (main classification, takes hours/days)
  3. cltools_datamatch.py (optional datamatch)
  Output file is ../../data/final/cltools_classified.csv


## Overview

Hierarchical classification of MCP tools using O*NET occupational task data through a 3-step process:
1. **Level 1**: Select high-level occupational category (12 options)
2. **Level 2**: Select specific cluster within category (~33 options per L1)  
3. **Task ID**: Select specific occupational task (~8 options per L2)
## Main Script

**File**: `cltools_main.py`

### Usage
```bash
# Minimal run (uses all defaults)
python cltools_main.py --run

# Common options
python cltools_main.py \
  --finance \
  --limit 100 \
  --run
```

### Key Flags
- `--finance`: Only process finance-related servers (sector 52)
- `--model`: model name (default: anthropic/claude-sonnet-4-20250514)
- `--run`: Execute inspect evaluation automatically
- `--out FILE`: Output CSV filename (default: `../../data/internal-cl/cltools_3_results.csv`)
- `--limit N`: Limit number of samples processed by inspect eval (no limit by default)

### Input Files
- `../../data/initial/data_unified_filtered.json`: MCP servers dataset (16,940 servers)
- `../../data/internal-task-clusters/task_clusters_names.csv`: O*NET hierarchy (18,796 tasks)

### Output
- Clean CSV with hierarchical classifications (Level 1 → Level 2 → Task ID)
- Timestamped log directories for each run

### Cost
- Each sample is at roughly 2k input and 100 output tokens, with 70k current total tools this will cost ~$500

## Task cluster creation & Validation

The `../onet-task-clusters/task_clusters_run.py` system creates the O*NET task hierarchy and validates clustering quality:

### Task Cluster Creation
- Creates 2-level hierarchy: 400 Level 2 clusters → 12 Level 1 categories
- Uses embeddings + K-means clustering + LLM naming
- Generates `../../data/internal-task-clusters/task_clusters_names.csv` (required input for main classification)

### Validation System
- Tests hierarchy consistency across levels (L3→L2, L2→L1, L3→L1)
- Reports accuracy metrics with confidence intervals
- Can be skipped with `--skip-validation` flag

## Data Enrichment

**File**: `cltools_datamatch.py`

Enriches classification results with metadata:
- Adds creation dates and usage counts from `../../data/final/clservers_classified.csv`
- Adds GitHub metrics (stars, forks, language) and download statistics from `../../data/external-usage/data_usage.json` 
- Creates final enriched dataset: `../../data/final/cltools_classified.csv`
- Optional post-processing step after main classification
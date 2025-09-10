# Hierarchical O*NET Task Classification

## Quick start
  1. (If stage5_task_clusters_names.csv does not yet exist) stage5_task_clusters_run.py --k2 400 (once, to create task hierarchy)
  2. stage5_main.py --run (main classification, takes hours/days)
  3. stage5_dataenrichment.py (optional datamatch)
  Output file is stage5_task_output_enriched.csv


## Overview

Hierarchical classification of MCP tools using O*NET occupational task data through a 3-step process:
1. **Level 1**: Select high-level occupational category (12 options)
2. **Level 2**: Select specific cluster within category (~33 options per L1)  
3. **Task ID**: Select specific occupational task (~8 options per L2)
## Main Script

**File**: `stage5_main.py`

### Usage
```bash
# Minimal run (uses all defaults)
python stage5_main.py --run

# Common options
python stage5_main.py \
  --finance \
  --limit 100 \
  --run
```

### Key Flags
- `--finance`: Only process finance-related servers (sector 52)
- `--model`: model name (default: anthropic/claude-sonnet-4-20250514)
- `--run`: Execute inspect evaluation automatically
- `--out FILE`: Output CSV filename (default: `stage5_task_output.csv`)
- `--limit N`: Limit number of samples processed by inspect eval (no limit by default)

### Input Files
- `data_unified_filtered.json`: MCP servers dataset (16,940 servers)
- `stage5_task_clusters_names.csv`: O*NET hierarchy (18,796 tasks)

### Output
- Clean CSV with hierarchical classifications (Level 1 → Level 2 → Task ID)
- Timestamped log directories for each run

### Cost
- Each sample is at roughly 2k input and 100 output tokens, with 70k current total tools this will cost ~$500

## Task cluster creation & Validation

The `stage5_task_clusters_run.py` system creates the O*NET task hierarchy and validates clustering quality:

### Task Cluster Creation
- Creates 2-level hierarchy: 400 Level 2 clusters → 12 Level 1 categories
- Uses embeddings + K-means clustering + LLM naming
- Generates `stage5_task_clusters_names.csv` (required input for main classification)

### Validation System
- Tests hierarchy consistency across levels (L3→L2, L2→L1, L3→L1)
- Reports accuracy metrics with confidence intervals
- Can be skipped with `--skip-validation` flag

## Data Enrichment

**File**: `stage5_dataenrichment.py`

Enriches classification results with metadata:
- Adds creation dates and usage counts from `server_classified.csv`
- Adds GitHub metrics (stars, forks, language) and download statistics from `data_usage.json` 
- Creates final enriched dataset: `stage5_task_output_enriched.csv`
- Optional post-processing step after main classification
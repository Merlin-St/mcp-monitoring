# Stage 4: Hierarchical O*NET Task Classification

## Overview

Stage 4 performs hierarchical classification of MCP tools using O*NET occupational task data through a 3-step process:
1. **Level 1**: Select high-level occupational category (12 options)
2. **Level 2**: Select specific cluster within category (~33 options per L1)  
3. **Task ID**: Select specific occupational task (~8 options per L2)

## Main Script

**File**: `conseq_fin_stage4_task_main.py`

### Usage
```bash
# Minimal run (uses all defaults)
python conseq_fin_stage4_task_main.py --run

# Common options
python conseq_fin_stage4_task_main.py \
  --finance \
  --limit 100 \
  --run
```

### Key Flags
- `--finance`: Only process finance-related servers (sector 52)
- `--model`: model name (default: anthropic/claude-sonnet-4-20250514)
- `--run`: Execute inspect evaluation automatically
- `--out FILE`: Output CSV filename (default: `conseq_fin_stage4_task_output.csv`)
- `--limit N`: Limit number of samples processed by inspect eval (no limit by default)

### Input Files
- `data_unified_filtered.json`: MCP servers dataset (16,940 servers)
- `conseq_fin_stage4_tasks_cluster_names.csv`: O*NET hierarchy (18,796 tasks)

### Output
- Clean CSV with hierarchical classifications (Level 1 → Level 2 → Task ID)
- Timestamped log directories for each run

### Cost
- Each sample is at roughly 2k input and 100 output tokens, with 70k current total tools this will cost ~$500
## Validation System

The `conseq_fin_stage4_task_clusters_run.py` system provides validation of the O*NET task clustering:

### Validation Files
- `conseq_fin_stage4_task_clusters_validation_l2_to_l1.jsonl`: Validates Level 2→Level 1 mappings
- `conseq_fin_stage4_task_clusters_validation_l3_to_l2.jsonl`: Validates Level 3→Level 2 mappings  
- `conseq_fin_stage4_task_clusters_validation_l3_to_l1.jsonl`: Validates Level 3→Level 1 mappings
- `conseq_fin_stage4_task_clusters_validation_subset_l2_l3.jsonl`: Validates L2/L3 subset relationships

### Purpose
These validation files ensure the hierarchical clustering structure is consistent:
- Higher level categories properly contain lower levels
- No orphaned tasks or broken hierarchies
- Semantic coherence across hierarchy levels

The cluster validation system helps maintain data quality for the main classification pipeline.
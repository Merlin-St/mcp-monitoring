# Human Validation Scoring

This directory contains scripts for analyzing human validation data from Gorilla experiments and calculating inter-rater agreement metrics.

## Overview

The human validation scoring system compares human ratings of MCP servers with LLM classifications to assess agreement and reliability.

## Scripts

### `human-validation-scoring.py`

Main script for calculating agreement metrics between human validators and LLM classifications.

**Features:**
- **Cohen's Kappa**: Pairwise agreement between each human rater and LLM
- **Fleiss' Kappa**: Inter-rater reliability across all human raters and LLM
- **Weighted Kappa**: For ordinal scales (e.g., payment autonomy levels)
- **Confusion Matrices**: Show disagreement patterns for each question
- **Per-participant Statistics**: Individual agreement scores for each validator
- **Per-question Statistics**: Aggregated metrics for each classification question

**Usage:**
```bash
source ~/mcp-monitoring/.venv/bin/activate
python human-validation-scripts/human-validation-scoring.py
```

**Input Files:**
- `data/external-cl-human-valid/data_exp_*.csv` - Human validation data from Gorilla experiments
- `data/final/clservers_classified.csv` - LLM classifications

**Output Files:**
- `output-validation/cl-validation/human-validation-scores.json` - Comprehensive agreement statistics
- `logs/human_validation_scoring.log` - Execution log with detailed progress

## Question Mapping

The script compares human responses to LLM classifications for the following questions:

| Human Question | LLM Field (clservers) | LLM Field (cltools) | Description | Type |
|----------------|-----------------------|---------------------|-------------|------|
| `func_main` | `highest_automation_func` | `tool_functionality_main` | Main functionality level (perception/reasoning/action) | Categorical |
| `func_sub` | `main_automation_subfunc` | `tool_functionality_sub` | Sub-category functionality classification | Categorical |
| `onet_l1` | `main_onet_task_level1` | `level1_name` | O*NET Level 1 occupational category | Categorical |
| `q3` | `generality_industry` | N/A | Industry generality (0=industry-specific, 1=cross-industry) | Binary |
| `q4` | `generality_environment` | N/A | Environment generality (0=trusted, 1=open/untrusted) | Binary |
| `q5` | `payments_autonomy` | N/A | Payment autonomy level (0-4 scale) | Ordinal |

**O*NET L1 Mapping:**
- Human responses use codes (e.g., `L1_01`, `L1_04`)
- LLM responses use full descriptions (e.g., `"Business management, finance, and customer service operations"`)
- Script automatically loads mapping from `data/internal-task-clusters/task_clusters_names.csv`
- Maps human codes to descriptions for proper comparison

Additional human-only questions (not compared with LLM):
- `q0_notes` - Free-text notes about server functionality

## Output Format

### Summary Statistics

```json
{
  "summary": {
    "n_participants": 3,
    "n_servers": 12,
    "n_questions": 4,
    "overall_agreement_with_llm": 0.75,
    "overall_inter_rater_reliability": 0.68
  }
}
```

### Per-Question Statistics

```json
{
  "by_question": {
    "func_main": {
      "question_id": "func_main",
      "description": "Main functionality level",
      "type": "categorical",
      "n_participants": 3,
      "n_servers": 12,
      "n_responses": 36,
      "kappa_vs_llm": [
        {"participant": "P1", "kappa": 0.72},
        {"participant": "P2", "kappa": 0.65},
        {"participant": "P3", "kappa": 0.80}
      ],
      "mean_kappa": 0.72,
      "fleiss_kappa": 0.71,
      "agreement_pct": 0.83,
      "confusion_matrix": [[...]]
    }
  }
}
```

### Per-Participant Statistics

```json
{
  "by_participant": {
    "P1": {
      "n_servers_rated": 12,
      "overall_kappa": 0.72,
      "agreement_by_question": {
        "func_main": {
          "kappa": 0.72,
          "agreement_pct": 0.83,
          "n_responses": 12
        }
      }
    }
  }
}
```

## Interpreting Kappa Scores

Cohen's Kappa and Fleiss' Kappa range from -1 to 1:

| Kappa Range | Interpretation |
|-------------|----------------|
| < 0 | Poor agreement (worse than chance) |
| 0.00 - 0.20 | Slight agreement |
| 0.21 - 0.40 | Fair agreement |
| 0.41 - 0.60 | Moderate agreement |
| 0.61 - 0.80 | Substantial agreement |
| 0.81 - 1.00 | Almost perfect agreement |

## Confusion Matrix Interpretation

Confusion matrices show disagreement patterns between LLM (rows) and human (columns):

```
Example for func_main (3 categories):
             Human: 1  Human: 2  Human: 3
LLM: 1 (perception)    2         0         0
LLM: 2 (reasoning)     1         1         1
LLM: 3 (action)        0         0         0
```

- Diagonal values = agreements
- Off-diagonal values = disagreements
- Row sums = LLM classification totals
- Column sums = human classification totals

## Scalability

The script is designed to scale with:

1. **Multiple participants** - Automatically detects all participants in data files
2. **Multiple data files** - Processes all `data_exp_*.csv` files in the input directory
3. **Additional questions** - Easily add new questions to `QUESTION_MAPPING`
4. **New metrics** - Modular design for adding custom agreement metrics

## Adding New Questions

To analyze additional questions:

1. Add question to `QUESTION_MAPPING` in `human-validation-scoring.py`:
```python
QUESTION_MAPPING = {
    "new_question": {
        "llm_field": "llm_field_name",
        "description": "Question description",
        "type": "categorical",  # or "binary", "ordinal"
        "mapping": {"value1": 1, "value2": 2}
    }
}
```

2. Ensure LLM field exists in `clservers_classified.csv`
3. Ensure human question exists in validation CSV files
4. Run the script - new question will be automatically analyzed

## Dependencies

- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Cohen's Kappa calculation
- Standard library: `json`, `pathlib`, `logging`

All dependencies are managed via `pyproject.toml` and installed with:
```bash
uv sync
```

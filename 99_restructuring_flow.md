# Repository Restructuring Flow

## Directory Structure Creation
First, create the following directory structure:

```
├── data/
│   ├── external-servers/
│   ├── external-usage/
│   ├── external-cl/
│   ├── internal-task-clusters/
│   ├── internal-cl/
│   ├── initial/
│   └── final/
├── scripts/
│   ├── data-collection-servers/
│   ├── data-collection-usage/
│   ├── data-unification/
│   ├── data-cleaning-readmes/
│   ├── data-classification-servers/
│   ├── data-classification-tools/
│   ├── onet-task-clusters/
│   └── data-analysis-topics/
├── output-visuals/
│   └── topics-embedding/
├── output-validation/
│   ├── cl-validation/
│   └── task-validation/
└── logs/
```

## File Movement and Renaming Tasks

### Task 1: Move Smithery Data Collection Files
- [x] 1a. Rename and move: `smithery_data_run.py` → `scripts/data-collection-servers/smithery_data_run.py`
- [x] 1b. Check dependencies: Update imports in files that reference this script
- [x] 1c. Check references: Update README.md, CLAUDE.md references

### Task 2: Move Smithery Downloader
- [x] 2a. Rename and move: `smithery_bulk_mcp_downloader.py` → `scripts/data-collection-servers/smithery_bulk_mcp_downloader.py`
- [x] 2b. Check dependencies: Update imports in smithery_data_run.py
- [x] 2c. Check references: Update documentation references

### Task 3: Move Smithery Config
- [x] 3a. Rename and move: `smithery_bulk_mcp_config.py` → `scripts/data-collection-servers/smithery_bulk_mcp_config.py`
- [x] 3b. Check dependencies: Update imports in smithery_bulk_mcp_downloader.py
- [x] 3c. Check references: Update documentation references

### Task 4: Move GitHub Data Collection
- [x] 4a. Rename and move: `github_data_run.py` → `scripts/data-collection-servers/github_data_run.py`
- [x] 4b. Check dependencies: Update imports in files that reference this script
- [x] 4c. Check references: Update README.md, CLAUDE.md references

### Task 5: Move GitHub Repo Searcher
- [x] 5a. Rename and move: `github_mcp_repo_searcher.py` → `scripts/data-collection-servers/github_mcp_repo_searcher.py`
- [x] 5b. Check dependencies: Update imports in github_data_run.py
- [x] 5c. Check references: Update documentation references

### Task 6: Move Official List Data Collection
- [x] 6a. Rename and move: `officiallist_data_run.py` → `scripts/data-collection-servers/officiallist_data_run.py`
- [x] 6b. Check dependencies: Update imports in files that reference this script
- [x] 6c. Check references: Update README.md, CLAUDE.md references

### Task 7: Move Official List HTML Fetcher
- [x] 7a. Rename and move: `officiallist_html_fetcher.py` → `scripts/data-collection-servers/officiallist_html_fetcher.py`
- [x] 7b. Check dependencies: Update imports in officiallist_data_run.py
- [x] 7c. Check references: Update documentation references

### Task 8: Move Official List URL Extractor
- [x] 8a. Rename and move: `officiallist_url_extractor.py` → `scripts/data-collection-servers/officiallist_url_extractor.py`
- [x] 8b. Check dependencies: Update imports in officiallist_data_run.py
- [x] 8c. Check references: Update documentation references

### Task 9: Move Official List GitHub Fetcher
- [x] 9a. Rename and move: `officiallist_github_fetcher.py` → `scripts/data-collection-servers/officiallist_github_fetcher.py`
- [x] 9b. Check dependencies: Update imports that reference this script
- [x] 9c. Check references: Update documentation references

### Task 10: Move Usage Collection NPM
- [x] 10a. Rename and move: `usage_collect_npm.py` → `scripts/data-collection-usage/usage_collect_npm.py`
- [x] 10b. Check dependencies: Update imports in files that reference this script
- [x] 10c. Check references: Update documentation references

### Task 11: Move Data Unified Processor
- [x] 11a. Rename and move: `data_unified_processor.py` → `scripts/data-unification/data_unified_processor.py`
- [x] 11b. Check dependencies: Update imports and file paths
- [x] 11c. Check references: Update README.md, CLAUDE.md references

### Task 12: Move Data Unified Create Filtered Subset
- [x] 12a. Rename and move: `data_unified_create_filtered_subset.py` → `scripts/data-unification/data_unified_create_filtered_subset.py`
- [x] 12b. Check dependencies: Update imports and file paths
- [x] 12c. Check references: Update documentation references

### Task 13: Move Data Unified Add Usage
- [x] 13a. Rename and move: `data_unified_add_usage.py` → `scripts/data-unification/data_unified_add_usage.py`
- [x] 13b. Check dependencies: Update imports and file paths
- [x] 13c. Check references: Update documentation references

### Task 14: Move README Filter Inspect
- [x] 14a. Rename and move: `data_readme_filter_inspect.py` → `scripts/data-cleaning-readmes/data_readme_filter_inspect.py`
- [x] 14b. Check dependencies: Update imports and file paths
- [x] 14c. Check references: Update documentation references

### Task 15: Move README Filter DF Processing
- [x] 15a. Rename and move: `data_readme_filter_dfprocessing.py` → `scripts/data-cleaning-readmes/data_readme_filter_dfprocessing.py`
- [x] 15b. Check dependencies: Update imports and file paths
- [x] 15c. Check references: Update documentation references

### Task 16: Move CLServers Step 1
- [x] 16a. Rename and move: `clservers_1_dataprep.py` → `scripts/data-classification-servers/clservers_1_dataprep.py`
- [x] 16b. Check dependencies: Update imports and file paths
- [x] 16c. Check references: Update documentation references

### Task 17: Move CLServers Step 2
- [x] 17a. Rename and move: `clservers_2_inspect.py` → `scripts/data-classification-servers/clservers_2_inspect.py`
- [x] 17b. Check dependencies: Update imports and file paths
- [x] 17c. Check references: Update documentation references

### Task 18: Move CLServers Step 3
- [x] 18a. Rename and move: `clservers_3_dfprocessing.py` → `scripts/data-classification-servers/clservers_3_dfprocessing.py`
- [x] 18b. Check dependencies: Update imports and file paths
- [x] 18c. Check references: Update documentation references

### Task 19: Move CLServers Step 4
- [x] 19a. Rename and move: `clservers_4_datamatch.py` → `scripts/data-classification-servers/clservers_4_datamatch.py`
- [x] 19b. Check dependencies: Update imports and file paths
- [x] 19c. Check references: Update documentation references

### Task 20: Move CLServers Validate
- [x] 20a. Rename and move: `clservers_validate.py` → `scripts/data-classification-servers/clservers_validate.py`
- [x] 20b. Check dependencies: Update imports and file paths
- [x] 20c. Check references: Update documentation references

### Task 21: Move CLTools Main
- [x] 21a. Rename and move: `cltools_main.py` → `scripts/data-classification-tools/cltools_main.py`
- [x] 21b. Check dependencies: Update imports and file paths
- [x] 21c. Check references: Update documentation references

### Task 22: Move CLTools DataMatch
- [x] 22a. Rename and move: `cltools_datamatch.py` → `scripts/data-classification-tools/cltools_datamatch.py`
- [x] 22b. Check dependencies: Update imports and file paths
- [x] 22c. Check references: Update documentation references

### Task 23: Move CLTools Documentation
- [x] 23a. Rename and move: `cltools.md` → `scripts/data-classification-tools/cltools.md`
- [x] 23b. Check dependencies: None
- [x] 23c. Check references: Update documentation references

### Task 24: Move Task Clusters Data
- [x] 24a. Rename and move: `task_clusters_data.py` → `scripts/onet-task-clusters/task_clusters_data.py`
- [x] 24b. Check dependencies: Update imports in other task_clusters scripts
- [x] 24c. Check references: Update documentation references

### Task 25: Move Task Clusters Embeddings
- [x] 25a. Rename and move: `task_clusters_embeddings.py` → `scripts/onet-task-clusters/task_clusters_embeddings.py`
- [x] 25b. Check dependencies: Update imports in other task_clusters scripts
- [x] 25c. Check references: Update documentation references

### Task 26: Move Task Clusters LLM
- [x] 26a. Rename and move: `task_clusters_llm.py` → `scripts/onet-task-clusters/task_clusters_llm.py`
- [x] 26b. Check dependencies: Update imports in other task_clusters scripts
- [x] 26c. Check references: Update documentation references

### Task 27: Move Task Clusters Run
- [x] 27a. Rename and move: `task_clusters_run.py` → `scripts/onet-task-clusters/task_clusters_run.py`
- [x] 27b. Check dependencies: Update imports and file paths
- [x] 27c. Check references: Update documentation references

### Task 28: Move Task Clusters Embed Match
- [x] 28a. Rename and move: `task_clusters_embed_match.py` → `scripts/onet-task-clusters/task_clusters_embed_match.py`
- [x] 28b. Check dependencies: Update imports and file paths
- [x] 28c. Check references: Update documentation references

### Task 29: Move Task Clusters Documentation
- [x] 29a. Rename and move: `task_clusters.md` → `scripts/onet-task-clusters/task_clusters.md`
- [x] 29b. Check dependencies: None
- [x] 29c. Check references: Update documentation references

### Task 30: Move Embed Generate
- [x] 30a. Rename and move: `embed_generate.py` → `scripts/data-analysis-topics/embed_generate.py`
- [x] 30b. Check dependencies: Update imports and file paths
- [x] 30c. Check references: Update README.md, CLAUDE.md references

### Task 31: Move Embed Hyperparameter Optimizer
- [x] 31a. Rename and move: `embed_hyperparameter_optimizer.py` → `scripts/data-analysis-topics/embed_hyperparameter_optimizer.py`
- [x] 31b. Check dependencies: Update imports and file paths
- [x] 31c. Check references: Update documentation references

### Task 32: Move Embed Apply Optimized Parameters
- [x] 32a. Rename and move: `embed_apply_optimized_parameters.py` → `scripts/data-analysis-topics/embed_apply_optimized_parameters.py`
- [x] 32b. Check dependencies: Update imports and file paths
- [x] 32c. Check references: Update documentation references

### Task 33: Move NAICS Classification Config
- [x] 33a. Rename and move: `naics_classification_config.py` → `scripts/data-analysis-topics/naics_classification_config.py`
- [x] 33b. Check dependencies: Update imports in embed_generate.py
- [x] 33c. Check references: Update documentation references

### Task 34: Move Smithery Data JSON
- [x] 34a. Rename and move: `smithery_data.json` → `data/external-servers/smithery_data.json`
- [x] 34b. Check dependencies: Update file paths in smithery scripts
- [x] 34c. Check references: Update documentation references

### Task 35: Move GitHub Data JSON
- [x] 35a. Rename and move: `github_data.json` → `data/external-servers/github_data.json`
- [x] 35b. Check dependencies: Update file paths in github scripts
- [x] 35c. Check references: Update documentation references

### Task 36: Move GitHub Data Summary
- [x] 36a. Rename and move: `github_data_summary.json` → `data/external-servers/github_data_summary.json`
- [x] 36b. Check dependencies: Update file paths in scripts
- [x] 36c. Check references: Update documentation references

### Task 37: Move Official List Data JSON
- [x] 37a. Rename and move: `officiallist_data.json` → `data/external-servers/officiallist_data.json`
- [x] 37b. Check dependencies: Update file paths in officiallist scripts
- [x] 37c. Check references: Update documentation references

### Task 38: Move Official List Summary
- [x] 38a. Rename and move: `officiallist_data_summary.json` → `data/external-servers/officiallist_data_summary.json`
- [x] 38b. Check dependencies: Update file paths in scripts
- [x] 38c. Check references: Update documentation references

### Task 39: Move Official List History
- [x] 39a. Rename and move: `officiallist_history.json` → `data/external-servers/officiallist_history.json`
- [x] 39b. Check dependencies: Update file paths in scripts
- [x] 39c. Check references: Update documentation references

### Task 40: Move Official List Monthly History
- [x] 40a. Rename and move: `officiallist_monthly_history.json` → `data/external-servers/officiallist_monthly_history.json`
- [x] 40b. Check dependencies: Update file paths in scripts
- [x] 40c. Check references: Update documentation references

### Task 41: Move Usage NPM Data
- [x] 41a. Rename and move: `usage_npm.json` → `data/external-usage/usage_npm.json`
- [x] 41b. Check dependencies: Update file paths in usage scripts
- [x] 41c. Check references: Update documentation references

### Task 42: Move Usage BigQuery Data
- [x] 42a. Rename and move: `usage_bigquery_webresults_pypi.json` → `data/external-usage/usage_bigquery_webresults_pypi.json`
- [x] 42b. Check dependencies: Update file paths in scripts
- [x] 42c. Check references: Update documentation references

### Task 43: Move Usage Data
- [x] 43a. Rename and move: `data_usage.json` → `data/external-usage/data_usage.json`
- [x] 43b. Check dependencies: Update file paths in scripts
- [x] 43c. Check references: Update documentation references

### Task 44: Move Usage Match
- [x] 44a. Rename and move: `usage_match.json` → `data/external-usage/usage_match.json`
- [x] 44b. Check dependencies: Update file paths in scripts
- [x] 44c. Check references: Update documentation references

### Task 45: Move NAICS Files
- [x] 45a. Rename and move: `naics_2022_4digit_subsectors.csv` → `data/external-cl/naics_2022_4digit_subsectors.csv`
- [x] 45b. Check dependencies: Update file paths in naics_classification_config.py
- [x] 45c. Check references: Update documentation references

### Task 46: Move NAICS Full Descriptions
- [x] 46a. Rename and move: `naics_2022_full_descriptions.csv` → `data/external-cl/naics_2022_full_descriptions.csv`
- [x] 46b. Check dependencies: Update file paths in scripts
- [x] 46c. Check references: Update documentation references

### Task 47: Move NAICS CNI Percentages
- [x] 47a. Rename and move: `naics_cni_percentages.csv` → `data/external-cl/naics_cni_percentages.csv`
- [x] 47b. Check dependencies: Update file paths in scripts
- [x] 47c. Check references: Update documentation references

### Task 48: Move O*NET Task Statements
- [x] 48a. Rename and move: `cl_onet_taskstatements.csv` → `data/external-cl/cl_onet_taskstatements.csv`
- [x] 48b. Check dependencies: Update file paths in task_clusters scripts
- [x] 48c. Check references: Update documentation references

### Task 49: Move O*NET Tools Used
- [x] 49a. Rename and move: `cl_onet_toolsused.csv` → `data/external-cl/cl_onet_toolsused.csv`
- [x] 49b. Check dependencies: Update file paths in task_clusters scripts
- [x] 49c. Check references: Update documentation references

### Task 50: Move CLServers Validate Labelled
- [x] 50a. Rename and move: `clservers_validate_labelled.csv` → `data/external-cl/clservers_validate_labelled.csv`
- [x] 50b. Check dependencies: Update file paths in clservers_validate.py
- [x] 50c. Check references: Update documentation references

### Task 51: Move Task Clusters Names
- [x] 51a. Rename and move: `task_clusters_names.csv` → `data/internal-task-clusters/task_clusters_names.csv`
- [x] 51b. Check dependencies: Update file paths in task_clusters scripts
- [x] 51c. Check references: Update documentation references

### Task 52: Move Task Clusters Summary Files
- [x] 52a. Rename and move: `task_clusters_summary_20250808_1308.json` → `data/internal-task-clusters/task_clusters_summary_20250808_1308.json`
- [x] 52b. Check dependencies: Update file paths if referenced
- [x] 52c. Check references: Update documentation references

### Task 53: Move Task Clusters Summary Contrastive
- [x] 53a. Rename and move: `task_clusters_summary_contrastive-40.json` → `data/internal-task-clusters/task_clusters_summary_contrastive-40.json`
- [x] 53b. Check dependencies: Update file paths if referenced
- [x] 53c. Check references: Update documentation references

### Task 54: Move CLServers Results
- [x] 54a. Rename and move: `clservers_3_results.json` → `data/internal-cl/clservers_3_results.json`
- [x] 54b. Check dependencies: Update file paths in clservers_4_datamatch.py
- [x] 54c. Check references: Update documentation references

### Task 55: Move CLTools Results
- [x] 55a. Rename and move: `cltools_3_results.csv` → `data/internal-cl/cltools_3_results.csv`
- [x] 55b. Check dependencies: Update file paths in cltools_datamatch.py
- [x] 55c. Check references: Update documentation references

### Task 56: Move CLTools Prep
- [x] 56a. Rename and move: `cltools_prep.json` → `data/internal-cl/cltools_prep.json`
- [x] 56b. Check dependencies: Update file paths in scripts
- [x] 56c. Check references: Update documentation references

### Task 57: Move Data Unified JSON
- [x] 57a. Rename and move: `data_unified.json` → `data/initial/data_unified.json`
- [x] 57b. Check dependencies: Update file paths in all scripts that read this file
- [x] 57c. Check references: Update documentation references

### Task 58: Move Data Unified Filtered
- [x] 58a. Rename and move: `data_unified_filtered.json` → `data/initial/data_unified_filtered.json`
- [x] 58b. Check dependencies: Update file paths in all scripts that read this file
- [x] 58c. Check references: Update documentation references

### Task 59: Move Data Unified Filtered Backup
- [x] 59a. Rename and move: `data_unified_filtered_backup.json` → `data/initial/data_unified_filtered_backup.json`
- [x] 59b. Check dependencies: Update file paths if referenced
- [x] 59c. Check references: Update documentation references

### Task 60: Move Data Unified Filtered Summary
- [x] 60a. Rename and move: `data_unified_filtered_summary.json` → `data/initial/data_unified_filtered_summary.json`
- [x] 60b. Check dependencies: Update file paths in scripts
- [x] 60c. Check references: Update documentation references

### Task 61: Move Data README Filter Content Summary
- [x] 61a. Rename and move: `data_readme_filter_content_summary.json` → `data/initial/data_readme_filter_content_summary.json`
- [x] 61b. Check dependencies: Update file paths in scripts
- [x] 61c. Check references: Update documentation references

### Task 62: Move Data README Filter DF Processing Summary
- [x] 62a. Rename and move: `data_readme_filter_dfprocessing_summary.json` → `data/initial/data_readme_filter_dfprocessing_summary.json`
- [x] 62b. Check dependencies: Update file paths in scripts
- [x] 62c. Check references: Update documentation references

### Task 63: Move CLServers Classified
- [x] 63a. Rename and move: `clservers_classified.csv` → `data/final/clservers_classified.csv`
- [x] 63b. Check dependencies: Update file paths in scripts that use this as input
- [x] 63c. Check references: Update documentation references

### Task 64: Move CLTools Classified
- [x] 64a. Rename and move: `cltools_classified.csv` → `data/final/cltools_classified.csv`
- [x] 64b. Check dependencies: Update file paths in scripts that use this as input
- [x] 64c. Check references: Update documentation references

### Task 65: Move Embed Results
- [x] 65a. Rename and move: `embed_results.json` → `output-visuals/topics-embedding/embed_results.json`
- [x] 65b. Check dependencies: Update file paths in scripts
- [x] 65c. Check references: Update documentation references

### Task 66: Move Embed Finance Results
- [x] 66a. Rename and move: `embed_finance_results.json` → `output-visuals/topics-embedding/embed_finance_results.json`
- [x] 66b. Check dependencies: Update file paths in scripts
- [x] 66c. Check references: Update documentation references

### Task 67: Move Embed Sector 52 Results
- [x] 67a. Rename and move: `embed_sector_52_results.json` → `output-visuals/topics-embedding/embed_sector_52_results.json`
- [x] 67b. Check dependencies: Update file paths in scripts
- [x] 67c. Check references: Update documentation references

### Task 68: Move Embed Visualization HTML
- [x] 68a. Rename and move: `embed_visualization.html` → `output-visuals/topics-embedding/embed_visualization.html`
- [x] 68b. Check dependencies: Update file paths in scripts
- [x] 68c. Check references: Update documentation references

### Task 69: Move Embed Finance Visualization
- [x] 69a. Rename and move: `embed_finance_visualization.html` → `output-visuals/topics-embedding/embed_finance_visualization.html`
- [x] 69b. Check dependencies: Update file paths in scripts
- [x] 69c. Check references: Update documentation references

### Task 70: Move Embed Sector 52 Visualization
- [x] 70a. Rename and move: `embed_sector_52_visualization.html` → `output-visuals/topics-embedding/embed_sector_52_visualization.html`
- [x] 70b. Check dependencies: Update file paths in scripts
- [x] 70c. Check references: Update documentation references

### Task 71: Move Embed Sector 54 Visualization
- [x] 71a. Rename and move: `embed_sector_54_visualization.html` → `output-visuals/topics-embedding/embed_sector_54_visualization.html`
- [x] 71b. Check dependencies: Update file paths in scripts
- [x] 71c. Check references: Update documentation references

### Task 72: Move CLServers Validation
- [x] 72a. Rename and move: `clservers_validation.json` → `output-validation/cl-validation/clservers_validation.json`
- [x] 72b. Check dependencies: Update file paths in scripts
- [x] 72c. Check references: Update documentation references

### Task 73: Move Task Clusters Embed Match Findings
- [x] 73a. Rename and move: `task_clusters_embed_match_findings.md` → `output-validation/task-validation/task_clusters_embed_match_findings.md`
- [x] 73b. Check dependencies: None
- [x] 73c. Check references: Update documentation references

### Task 74: Move Task Clusters Embed Match Results
- [x] 74a. Rename and move: `task_clusters_embed_match_results.json` → `output-validation/task-validation/task_clusters_embed_match_results.json`
- [x] 74b. Check dependencies: Update file paths in scripts
- [x] 74c. Check references: Update documentation references

### Task 75: Move Log Files
- [x] 75a. Rename and move: `cltools_datamatch.log` → `logs/cltools_datamatch.log`
- [x] 75b. Check dependencies: Update log file paths in scripts
- [x] 75c. Check references: Update documentation references

### Task 76: Move Data Unified Add Usage Log
- [x] 76a. Rename and move: `data_unified_add_usage.log` → `logs/data_unified_add_usage.log`
- [x] 76b. Check dependencies: Update log file paths in scripts
- [x] 76c. Check references: Update documentation references

### Task 77: Move Usage Collect NPM Log
- [x] 77a. Rename and move: `usage_collect_npm.log` → `logs/usage_collect_npm.log`
- [x] 77b. Check dependencies: Update log file paths in scripts
- [x] 77c. Check references: Update documentation references

### Task 78: Move Inspect MCP Tasks (if needed)
- [x] 78a. Rename and move: `inspect_mcp_tasks1.py` → `scripts/99_playground/inspect_mcp_tasks1.py`
- [x] 78b. Check dependencies: Update imports and file paths
- [x] 78c. Check references: Update documentation references

## Files to Keep in Root
- README.md
- CLAUDE.md
- LICENSE (to be created)
- Makefile (to be created)
- pyproject.toml
- uv.lock
- setup.cfg
- .gitignore
- .gitattributes
- 99_restructuring_flow.md (this file)

## Notes on Uncertain Files
- `=1.10.0` and `=4.3.2` - These appear to be accidentally created files, should be removed
- `.mypy_cache/` - Should be added to .gitignore if not already
- `.claude/` - Configuration folder, should remain in root

## Summary
Total tasks: 78
Each task has 3 sub-steps (rename/move, check dependencies, check references)
Total sub-steps: 234
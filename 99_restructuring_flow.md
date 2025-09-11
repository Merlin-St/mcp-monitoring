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
- [ ] 1a. Rename and move: `smithery_data_run.py` → `scripts/data-collection-servers/smithery_data_run.py`
- [ ] 1b. Check dependencies: Update imports in files that reference this script
- [ ] 1c. Check references: Update README.md, CLAUDE.md references

### Task 2: Move Smithery Downloader
- [ ] 2a. Rename and move: `smithery_bulk_mcp_downloader.py` → `scripts/data-collection-servers/smithery_bulk_mcp_downloader.py`
- [ ] 2b. Check dependencies: Update imports in smithery_data_run.py
- [ ] 2c. Check references: Update documentation references

### Task 3: Move Smithery Config
- [ ] 3a. Rename and move: `smithery_bulk_mcp_config.py` → `scripts/data-collection-servers/smithery_bulk_mcp_config.py`
- [ ] 3b. Check dependencies: Update imports in smithery_bulk_mcp_downloader.py
- [ ] 3c. Check references: Update documentation references

### Task 4: Move GitHub Data Collection
- [ ] 4a. Rename and move: `github_data_run.py` → `scripts/data-collection-servers/github_data_run.py`
- [ ] 4b. Check dependencies: Update imports in files that reference this script
- [ ] 4c. Check references: Update README.md, CLAUDE.md references

### Task 5: Move GitHub Repo Searcher
- [ ] 5a. Rename and move: `github_mcp_repo_searcher.py` → `scripts/data-collection-servers/github_mcp_repo_searcher.py`
- [ ] 5b. Check dependencies: Update imports in github_data_run.py
- [ ] 5c. Check references: Update documentation references

### Task 6: Move Official List Data Collection
- [ ] 6a. Rename and move: `officiallist_data_run.py` → `scripts/data-collection-servers/officiallist_data_run.py`
- [ ] 6b. Check dependencies: Update imports in files that reference this script
- [ ] 6c. Check references: Update README.md, CLAUDE.md references

### Task 7: Move Official List HTML Fetcher
- [ ] 7a. Rename and move: `officiallist_html_fetcher.py` → `scripts/data-collection-servers/officiallist_html_fetcher.py`
- [ ] 7b. Check dependencies: Update imports in officiallist_data_run.py
- [ ] 7c. Check references: Update documentation references

### Task 8: Move Official List URL Extractor
- [ ] 8a. Rename and move: `officiallist_url_extractor.py` → `scripts/data-collection-servers/officiallist_url_extractor.py`
- [ ] 8b. Check dependencies: Update imports in officiallist_data_run.py
- [ ] 8c. Check references: Update documentation references

### Task 9: Move Official List GitHub Fetcher
- [ ] 9a. Rename and move: `officiallist_github_fetcher.py` → `scripts/data-collection-servers/officiallist_github_fetcher.py`
- [ ] 9b. Check dependencies: Update imports that reference this script
- [ ] 9c. Check references: Update documentation references

### Task 10: Move Usage Collection NPM
- [ ] 10a. Rename and move: `usage_collect_npm.py` → `scripts/data-collection-usage/usage_collect_npm.py`
- [ ] 10b. Check dependencies: Update imports in files that reference this script
- [ ] 10c. Check references: Update documentation references

### Task 11: Move Data Unified Processor
- [ ] 11a. Rename and move: `data_unified_processor.py` → `scripts/data-unification/data_unified_processor.py`
- [ ] 11b. Check dependencies: Update imports and file paths
- [ ] 11c. Check references: Update README.md, CLAUDE.md references

### Task 12: Move Data Unified Create Filtered Subset
- [ ] 12a. Rename and move: `data_unified_create_filtered_subset.py` → `scripts/data-unification/data_unified_create_filtered_subset.py`
- [ ] 12b. Check dependencies: Update imports and file paths
- [ ] 12c. Check references: Update documentation references

### Task 13: Move Data Unified Add Usage
- [ ] 13a. Rename and move: `data_unified_add_usage.py` → `scripts/data-unification/data_unified_add_usage.py`
- [ ] 13b. Check dependencies: Update imports and file paths
- [ ] 13c. Check references: Update documentation references

### Task 14: Move README Filter Inspect
- [ ] 14a. Rename and move: `data_readme_filter_inspect.py` → `scripts/data-cleaning-readmes/data_readme_filter_inspect.py`
- [ ] 14b. Check dependencies: Update imports and file paths
- [ ] 14c. Check references: Update documentation references

### Task 15: Move README Filter DF Processing
- [ ] 15a. Rename and move: `data_readme_filter_dfprocessing.py` → `scripts/data-cleaning-readmes/data_readme_filter_dfprocessing.py`
- [ ] 15b. Check dependencies: Update imports and file paths
- [ ] 15c. Check references: Update documentation references

### Task 16: Move CLServers Step 1
- [ ] 16a. Rename and move: `clservers_1_dataprep.py` → `scripts/data-classification-servers/clservers_1_dataprep.py`
- [ ] 16b. Check dependencies: Update imports and file paths
- [ ] 16c. Check references: Update documentation references

### Task 17: Move CLServers Step 2
- [ ] 17a. Rename and move: `clservers_2_inspect.py` → `scripts/data-classification-servers/clservers_2_inspect.py`
- [ ] 17b. Check dependencies: Update imports and file paths
- [ ] 17c. Check references: Update documentation references

### Task 18: Move CLServers Step 3
- [ ] 18a. Rename and move: `clservers_3_dfprocessing.py` → `scripts/data-classification-servers/clservers_3_dfprocessing.py`
- [ ] 18b. Check dependencies: Update imports and file paths
- [ ] 18c. Check references: Update documentation references

### Task 19: Move CLServers Step 4
- [ ] 19a. Rename and move: `clservers_4_datamatch.py` → `scripts/data-classification-servers/clservers_4_datamatch.py`
- [ ] 19b. Check dependencies: Update imports and file paths
- [ ] 19c. Check references: Update documentation references

### Task 20: Move CLServers Validate
- [ ] 20a. Rename and move: `clservers_validate.py` → `scripts/data-classification-servers/clservers_validate.py`
- [ ] 20b. Check dependencies: Update imports and file paths
- [ ] 20c. Check references: Update documentation references

### Task 21: Move CLTools Main
- [ ] 21a. Rename and move: `cltools_main.py` → `scripts/data-classification-tools/cltools_main.py`
- [ ] 21b. Check dependencies: Update imports and file paths
- [ ] 21c. Check references: Update documentation references

### Task 22: Move CLTools DataMatch
- [ ] 22a. Rename and move: `cltools_datamatch.py` → `scripts/data-classification-tools/cltools_datamatch.py`
- [ ] 22b. Check dependencies: Update imports and file paths
- [ ] 22c. Check references: Update documentation references

### Task 23: Move CLTools Documentation
- [ ] 23a. Rename and move: `cltools.md` → `scripts/data-classification-tools/cltools.md`
- [ ] 23b. Check dependencies: None
- [ ] 23c. Check references: Update documentation references

### Task 24: Move Task Clusters Data
- [ ] 24a. Rename and move: `task_clusters_data.py` → `scripts/onet-task-clusters/task_clusters_data.py`
- [ ] 24b. Check dependencies: Update imports in other task_clusters scripts
- [ ] 24c. Check references: Update documentation references

### Task 25: Move Task Clusters Embeddings
- [ ] 25a. Rename and move: `task_clusters_embeddings.py` → `scripts/onet-task-clusters/task_clusters_embeddings.py`
- [ ] 25b. Check dependencies: Update imports in other task_clusters scripts
- [ ] 25c. Check references: Update documentation references

### Task 26: Move Task Clusters LLM
- [ ] 26a. Rename and move: `task_clusters_llm.py` → `scripts/onet-task-clusters/task_clusters_llm.py`
- [ ] 26b. Check dependencies: Update imports in other task_clusters scripts
- [ ] 26c. Check references: Update documentation references

### Task 27: Move Task Clusters Run
- [ ] 27a. Rename and move: `task_clusters_run.py` → `scripts/onet-task-clusters/task_clusters_run.py`
- [ ] 27b. Check dependencies: Update imports and file paths
- [ ] 27c. Check references: Update documentation references

### Task 28: Move Task Clusters Embed Match
- [ ] 28a. Rename and move: `task_clusters_embed_match.py` → `scripts/onet-task-clusters/task_clusters_embed_match.py`
- [ ] 28b. Check dependencies: Update imports and file paths
- [ ] 28c. Check references: Update documentation references

### Task 29: Move Task Clusters Documentation
- [ ] 29a. Rename and move: `task_clusters.md` → `scripts/onet-task-clusters/task_clusters.md`
- [ ] 29b. Check dependencies: None
- [ ] 29c. Check references: Update documentation references

### Task 30: Move Embed Generate
- [ ] 30a. Rename and move: `embed_generate.py` → `scripts/data-analysis-topics/embed_generate.py`
- [ ] 30b. Check dependencies: Update imports and file paths
- [ ] 30c. Check references: Update README.md, CLAUDE.md references

### Task 31: Move Embed Hyperparameter Optimizer
- [ ] 31a. Rename and move: `embed_hyperparameter_optimizer.py` → `scripts/data-analysis-topics/embed_hyperparameter_optimizer.py`
- [ ] 31b. Check dependencies: Update imports and file paths
- [ ] 31c. Check references: Update documentation references

### Task 32: Move Embed Apply Optimized Parameters
- [ ] 32a. Rename and move: `embed_apply_optimized_parameters.py` → `scripts/data-analysis-topics/embed_apply_optimized_parameters.py`
- [ ] 32b. Check dependencies: Update imports and file paths
- [ ] 32c. Check references: Update documentation references

### Task 33: Move NAICS Classification Config
- [ ] 33a. Rename and move: `naics_classification_config.py` → `scripts/data-analysis-topics/naics_classification_config.py`
- [ ] 33b. Check dependencies: Update imports in embed_generate.py
- [ ] 33c. Check references: Update documentation references

### Task 34: Move Smithery Data JSON
- [ ] 34a. Rename and move: `smithery_data.json` → `data/external-servers/smithery_data.json`
- [ ] 34b. Check dependencies: Update file paths in smithery scripts
- [ ] 34c. Check references: Update documentation references

### Task 35: Move GitHub Data JSON
- [ ] 35a. Rename and move: `github_data.json` → `data/external-servers/github_data.json`
- [ ] 35b. Check dependencies: Update file paths in github scripts
- [ ] 35c. Check references: Update documentation references

### Task 36: Move GitHub Data Summary
- [ ] 36a. Rename and move: `github_data_summary.json` → `data/external-servers/github_data_summary.json`
- [ ] 36b. Check dependencies: Update file paths in scripts
- [ ] 36c. Check references: Update documentation references

### Task 37: Move Official List Data JSON
- [ ] 37a. Rename and move: `officiallist_data.json` → `data/external-servers/officiallist_data.json`
- [ ] 37b. Check dependencies: Update file paths in officiallist scripts
- [ ] 37c. Check references: Update documentation references

### Task 38: Move Official List Summary
- [ ] 38a. Rename and move: `officiallist_data_summary.json` → `data/external-servers/officiallist_data_summary.json`
- [ ] 38b. Check dependencies: Update file paths in scripts
- [ ] 38c. Check references: Update documentation references

### Task 39: Move Official List History
- [ ] 39a. Rename and move: `officiallist_history.json` → `data/external-servers/officiallist_history.json`
- [ ] 39b. Check dependencies: Update file paths in scripts
- [ ] 39c. Check references: Update documentation references

### Task 40: Move Official List Monthly History
- [ ] 40a. Rename and move: `officiallist_monthly_history.json` → `data/external-servers/officiallist_monthly_history.json`
- [ ] 40b. Check dependencies: Update file paths in scripts
- [ ] 40c. Check references: Update documentation references

### Task 41: Move Usage NPM Data
- [ ] 41a. Rename and move: `usage_npm.json` → `data/external-usage/usage_npm.json`
- [ ] 41b. Check dependencies: Update file paths in usage scripts
- [ ] 41c. Check references: Update documentation references

### Task 42: Move Usage BigQuery Data
- [ ] 42a. Rename and move: `usage_bigquery_webresults_pypi.json` → `data/external-usage/usage_bigquery_webresults_pypi.json`
- [ ] 42b. Check dependencies: Update file paths in scripts
- [ ] 42c. Check references: Update documentation references

### Task 43: Move Usage Data
- [ ] 43a. Rename and move: `data_usage.json` → `data/external-usage/data_usage.json`
- [ ] 43b. Check dependencies: Update file paths in scripts
- [ ] 43c. Check references: Update documentation references

### Task 44: Move Usage Match
- [ ] 44a. Rename and move: `usage_match.json` → `data/external-usage/usage_match.json`
- [ ] 44b. Check dependencies: Update file paths in scripts
- [ ] 44c. Check references: Update documentation references

### Task 45: Move NAICS Files
- [ ] 45a. Rename and move: `naics_2022_4digit_subsectors.csv` → `data/external-cl/naics_2022_4digit_subsectors.csv`
- [ ] 45b. Check dependencies: Update file paths in naics_classification_config.py
- [ ] 45c. Check references: Update documentation references

### Task 46: Move NAICS Full Descriptions
- [ ] 46a. Rename and move: `naics_2022_full_descriptions.csv` → `data/external-cl/naics_2022_full_descriptions.csv`
- [ ] 46b. Check dependencies: Update file paths in scripts
- [ ] 46c. Check references: Update documentation references

### Task 47: Move NAICS CNI Percentages
- [ ] 47a. Rename and move: `naics_cni_percentages.csv` → `data/external-cl/naics_cni_percentages.csv`
- [ ] 47b. Check dependencies: Update file paths in scripts
- [ ] 47c. Check references: Update documentation references

### Task 48: Move O*NET Task Statements
- [ ] 48a. Rename and move: `cl_onet_taskstatements.csv` → `data/external-cl/cl_onet_taskstatements.csv`
- [ ] 48b. Check dependencies: Update file paths in task_clusters scripts
- [ ] 48c. Check references: Update documentation references

### Task 49: Move O*NET Tools Used
- [ ] 49a. Rename and move: `cl_onet_toolsused.csv` → `data/external-cl/cl_onet_toolsused.csv`
- [ ] 49b. Check dependencies: Update file paths in task_clusters scripts
- [ ] 49c. Check references: Update documentation references

### Task 50: Move CLServers Validate Labelled
- [ ] 50a. Rename and move: `clservers_validate_labelled.csv` → `data/external-cl/clservers_validate_labelled.csv`
- [ ] 50b. Check dependencies: Update file paths in clservers_validate.py
- [ ] 50c. Check references: Update documentation references

### Task 51: Move Task Clusters Names
- [ ] 51a. Rename and move: `task_clusters_names.csv` → `data/internal-task-clusters/task_clusters_names.csv`
- [ ] 51b. Check dependencies: Update file paths in task_clusters scripts
- [ ] 51c. Check references: Update documentation references

### Task 52: Move Task Clusters Summary Files
- [ ] 52a. Rename and move: `task_clusters_summary_20250808_1308.json` → `data/internal-task-clusters/task_clusters_summary_20250808_1308.json`
- [ ] 52b. Check dependencies: Update file paths if referenced
- [ ] 52c. Check references: Update documentation references

### Task 53: Move Task Clusters Summary Contrastive
- [ ] 53a. Rename and move: `task_clusters_summary_contrastive-40.json` → `data/internal-task-clusters/task_clusters_summary_contrastive-40.json`
- [ ] 53b. Check dependencies: Update file paths if referenced
- [ ] 53c. Check references: Update documentation references

### Task 54: Move CLServers Results
- [ ] 54a. Rename and move: `clservers_3_results.json` → `data/internal-cl/clservers_3_results.json`
- [ ] 54b. Check dependencies: Update file paths in clservers_4_datamatch.py
- [ ] 54c. Check references: Update documentation references

### Task 55: Move CLTools Results
- [ ] 55a. Rename and move: `cltools_3_results.csv` → `data/internal-cl/cltools_3_results.csv`
- [ ] 55b. Check dependencies: Update file paths in cltools_datamatch.py
- [ ] 55c. Check references: Update documentation references

### Task 56: Move CLTools Prep
- [ ] 56a. Rename and move: `cltools_prep.json` → `data/internal-cl/cltools_prep.json`
- [ ] 56b. Check dependencies: Update file paths in scripts
- [ ] 56c. Check references: Update documentation references

### Task 57: Move Data Unified JSON
- [ ] 57a. Rename and move: `data_unified.json` → `data/initial/data_unified.json`
- [ ] 57b. Check dependencies: Update file paths in all scripts that read this file
- [ ] 57c. Check references: Update documentation references

### Task 58: Move Data Unified Filtered
- [ ] 58a. Rename and move: `data_unified_filtered.json` → `data/initial/data_unified_filtered.json`
- [ ] 58b. Check dependencies: Update file paths in all scripts that read this file
- [ ] 58c. Check references: Update documentation references

### Task 59: Move Data Unified Filtered Backup
- [ ] 59a. Rename and move: `data_unified_filtered_backup.json` → `data/initial/data_unified_filtered_backup.json`
- [ ] 59b. Check dependencies: Update file paths if referenced
- [ ] 59c. Check references: Update documentation references

### Task 60: Move Data Unified Filtered Summary
- [ ] 60a. Rename and move: `data_unified_filtered_summary.json` → `data/initial/data_unified_filtered_summary.json`
- [ ] 60b. Check dependencies: Update file paths in scripts
- [ ] 60c. Check references: Update documentation references

### Task 61: Move Data README Filter Content Summary
- [ ] 61a. Rename and move: `data_readme_filter_content_summary.json` → `data/initial/data_readme_filter_content_summary.json`
- [ ] 61b. Check dependencies: Update file paths in scripts
- [ ] 61c. Check references: Update documentation references

### Task 62: Move Data README Filter DF Processing Summary
- [ ] 62a. Rename and move: `data_readme_filter_dfprocessing_summary.json` → `data/initial/data_readme_filter_dfprocessing_summary.json`
- [ ] 62b. Check dependencies: Update file paths in scripts
- [ ] 62c. Check references: Update documentation references

### Task 63: Move CLServers Classified
- [ ] 63a. Rename and move: `clservers_classified.csv` → `data/final/clservers_classified.csv`
- [ ] 63b. Check dependencies: Update file paths in scripts that use this as input
- [ ] 63c. Check references: Update documentation references

### Task 64: Move CLTools Classified
- [ ] 64a. Rename and move: `cltools_classified.csv` → `data/final/cltools_classified.csv`
- [ ] 64b. Check dependencies: Update file paths in scripts that use this as input
- [ ] 64c. Check references: Update documentation references

### Task 65: Move Embed Results
- [ ] 65a. Rename and move: `embed_results.json` → `output-visuals/topics-embedding/embed_results.json`
- [ ] 65b. Check dependencies: Update file paths in scripts
- [ ] 65c. Check references: Update documentation references

### Task 66: Move Embed Finance Results
- [ ] 66a. Rename and move: `embed_finance_results.json` → `output-visuals/topics-embedding/embed_finance_results.json`
- [ ] 66b. Check dependencies: Update file paths in scripts
- [ ] 66c. Check references: Update documentation references

### Task 67: Move Embed Sector 52 Results
- [ ] 67a. Rename and move: `embed_sector_52_results.json` → `output-visuals/topics-embedding/embed_sector_52_results.json`
- [ ] 67b. Check dependencies: Update file paths in scripts
- [ ] 67c. Check references: Update documentation references

### Task 68: Move Embed Visualization HTML
- [ ] 68a. Rename and move: `embed_visualization.html` → `output-visuals/topics-embedding/embed_visualization.html`
- [ ] 68b. Check dependencies: Update file paths in scripts
- [ ] 68c. Check references: Update documentation references

### Task 69: Move Embed Finance Visualization
- [ ] 69a. Rename and move: `embed_finance_visualization.html` → `output-visuals/topics-embedding/embed_finance_visualization.html`
- [ ] 69b. Check dependencies: Update file paths in scripts
- [ ] 69c. Check references: Update documentation references

### Task 70: Move Embed Sector 52 Visualization
- [ ] 70a. Rename and move: `embed_sector_52_visualization.html` → `output-visuals/topics-embedding/embed_sector_52_visualization.html`
- [ ] 70b. Check dependencies: Update file paths in scripts
- [ ] 70c. Check references: Update documentation references

### Task 71: Move Embed Sector 54 Visualization
- [ ] 71a. Rename and move: `embed_sector_54_visualization.html` → `output-visuals/topics-embedding/embed_sector_54_visualization.html`
- [ ] 71b. Check dependencies: Update file paths in scripts
- [ ] 71c. Check references: Update documentation references

### Task 72: Move CLServers Validation
- [ ] 72a. Rename and move: `clservers_validation.json` → `output-validation/cl-validation/clservers_validation.json`
- [ ] 72b. Check dependencies: Update file paths in scripts
- [ ] 72c. Check references: Update documentation references

### Task 73: Move Task Clusters Embed Match Findings
- [ ] 73a. Rename and move: `task_clusters_embed_match_findings.md` → `output-validation/task-validation/task_clusters_embed_match_findings.md`
- [ ] 73b. Check dependencies: None
- [ ] 73c. Check references: Update documentation references

### Task 74: Move Task Clusters Embed Match Results
- [ ] 74a. Rename and move: `task_clusters_embed_match_results.json` → `output-validation/task-validation/task_clusters_embed_match_results.json`
- [ ] 74b. Check dependencies: Update file paths in scripts
- [ ] 74c. Check references: Update documentation references

### Task 75: Move Log Files
- [ ] 75a. Rename and move: `cltools_datamatch.log` → `logs/cltools_datamatch.log`
- [ ] 75b. Check dependencies: Update log file paths in scripts
- [ ] 75c. Check references: Update documentation references

### Task 76: Move Data Unified Add Usage Log
- [ ] 76a. Rename and move: `data_unified_add_usage.log` → `logs/data_unified_add_usage.log`
- [ ] 76b. Check dependencies: Update log file paths in scripts
- [ ] 76c. Check references: Update documentation references

### Task 77: Move Usage Collect NPM Log
- [ ] 77a. Rename and move: `usage_collect_npm.log` → `logs/usage_collect_npm.log`
- [ ] 77b. Check dependencies: Update log file paths in scripts
- [ ] 77c. Check references: Update documentation references

### Task 78: Move Inspect MCP Tasks (if needed)
- [ ] 78a. Rename and move: `inspect_mcp_tasks1.py` → `scripts/99_playground/inspect_mcp_tasks1.py`
- [ ] 78b. Check dependencies: Update imports and file paths
- [ ] 78c. Check references: Update documentation references

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
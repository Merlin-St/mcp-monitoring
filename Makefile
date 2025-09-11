# MCP Monitoring Dashboard Makefile
# Common commands for data collection, processing, and analysis
# ############################
# NOTE: THIS IS NOT YET TESTED! 
# ############################
# Prerequisites:
# - Virtual environment: uv sync / source ~/mcp-monitoring/.venv/bin/activate

.PHONY: help data-collect-servers data-collect-usage data-collect-all data-initial data-clean-readmes data-initial-clean data-embed-analysis data-embed-analysis-finance data-embed-optimize data-embed-analysis-all data-cl-servers data-cl-tools data-cl-all data-task-clusters clean lint lint-fix workflow-data-creation workflow-complete

# Default target
help:
	@echo "MCP Monitoring Dashboard - Available Commands:"
	@echo ""
	@echo "  Data Collection (External Sources):"
	@echo "    data-collect-servers    Collect from all 3 server sources"
	@echo "    data-collect-usage      Collect usage statistics"
	@echo "    data-collect-all        Run complete data collection pipeline"
	@echo ""
	@echo "  Data Processing (Internal):"
	@echo "    data-initial           Create unified and filtered datasets"
	@echo "    data-clean-readmes     Filter README content using LLM"
	@echo "    data-initial-clean     Run data-initial + data-clean-readmes"
	@echo ""
	@echo "  Analysis & Visualization:"
	@echo "    data-embed-analysis          Generate embeddings and topic analysis"
	@echo "    data-embed-analysis-finance  Finance-specific analysis"
	@echo "    data-embed-optimize          Hyperparameter optimization"
	@echo "    data-embed-analysis-all      Run all embedding analysis steps"
	@echo ""
	@echo "  Classification (Consequentiality Scoring):"
	@echo "    data-cl-servers        CLServers pipeline (server classification)"
	@echo "    data-cl-tools          CLTools pipeline (O*NET task mapping)"
	@echo "    data-cl-all            Complete classification pipeline"
	@echo "    data-task-clusters     O*NET task clustering"
	@echo ""
	@echo "  Maintenance:"
	@echo "    clean                  Clean logs and temporary files"
	@echo "    lint                   Run code quality checks"
	@echo "    lint-fix               Auto-fix code quality issues"
	@echo ""
	@echo "  Complete Workflows:"
	@echo "    workflow-data-creation Data collection + initial processing"
	@echo "    workflow-complete      Full pipeline (all steps)"
	@echo ""
	@echo "  Prerequisites: source ~/mcp-monitoring/.venv/bin/activate"

# ===============================
# Data Collection (External)
# ===============================

data-collect-servers:
	@echo "🔄 Collecting MCP server data from all 3 sources..."
	python scripts/data-collection-servers/smithery_data_run.py
	python scripts/data-collection-servers/github_data_run.py
	python scripts/data-collection-servers/officiallist_data_run.py
	python scripts/data-collection-servers/officiallist_github_fetcher.py
	@echo "✅ Server data collection complete"

data-collect-usage:
	@echo "🔄 Collecting usage statistics..."
	python scripts/data-collection-usage/usage_collect_npm.py
	@echo "✅ Usage data collection complete"

data-collect-all: data-collect-servers data-collect-usage

# ===============================
# Data Processing (Internal)
# ===============================

data-initial:
	@echo "🔄 Creating unified datasets..."
	python scripts/data-unification/data_unified_processor.py
	python scripts/data-unification/data_unified_create_filtered_subset.py
	python scripts/data-unification/data_unified_add_usage.py
	@echo "✅ Unified datasets created: data/initial/"

data-clean-readmes:
	@echo "🔄 Filtering README content using LLM..."
	inspect eval scripts/data-cleaning-readmes/data_readme_filter_inspect.py --model anthropic/claude-sonnet-4-20250514
	python scripts/data-cleaning-readmes/data_readme_filter_dfprocessing.py
	@echo "✅ README filtering complete"

data-initial-clean: data-initial data-clean-readmes

# ===============================
# Analysis & Visualization
# ===============================

data-embed-optimize:
	@echo "🔄 Running hyperparameter optimization..."
	python scripts/data-analysis-topics/embed_hyperparameter_optimizer.py --finance --test-size 500 --max-combinations 20
	python scripts/data-analysis-topics/embed_apply_optimized_parameters.py logs/embed_hyperparameter_optimization_sector_52.log
	@echo "✅ Optimization complete - parameters applied to embed_generate.py"

data-embed-analysis:
	@echo "🔄 Generating embeddings and topic analysis (full dataset)..."
	python scripts/data-analysis-topics/embed_generate.py --clustering hdbscan
	@echo "✅ Analysis complete: output-visuals/topics-embedding/"

data-embed-analysis-finance:
	@echo "🔄 Generating finance-specific analysis..."
	python scripts/data-analysis-topics/embed_generate.py --filter sector_52 --clustering hdbscan
	@echo "✅ Finance analysis complete"

data-embed-analysis-all: data-embed-optimize data-embed-analysis data-embed-analysis-finance

# ===============================
# Classification Pipelines
# ===============================

data-task-clusters:
	@echo "🔄 Running O*NET task clustering..."
	@if [ ! -f "data/internal-task-clusters/task_clusters_names.csv" ]; then \
		echo "Task clusters file not found, running clustering..."; \
		python scripts/onet-task-clusters/task_clusters_run.py --k2 400; \
	else \
		echo "Task clusters file exists, skipping clustering step"; \
	fi
	python scripts/onet-task-clusters/task_clusters_embed_match.py
	@echo "✅ Task clustering complete"

data-cl-servers:
	@echo "🔄 Running CLServers pipeline (server classification)..."
	python scripts/data-classification-servers/clservers_1_dataprep.py
	inspect eval scripts/data-classification-servers/clservers_2_inspect.py --model anthropic/claude-sonnet-4-20250514
	python scripts/data-classification-servers/clservers_3_dfprocessing.py
	python scripts/data-classification-servers/clservers_4_datamatch.py
	@echo "✅ CLServers pipeline complete: data/final/clservers_classified.csv"

data-cl-tools:
	@echo "🔄 Running CLTools pipeline (O*NET task mapping)..."
	python scripts/data-classification-tools/cltools_main.py --run --finance --limit 1000
	python scripts/data-classification-tools/cltools_datamatch.py --stage4 data/internal-cl/cltools_3_results.csv --stage2 data/final/clservers_classified.csv --usage data/initial/data_unified_filtered.json --output data/final/cltools_classified.csv
	@echo "✅ CLTools pipeline complete: data/final/cltools_classified.csv"

data-cl-all: data-task-clusters data-cl-servers data-cl-tools



# ===============================
# Maintenance
# ===============================

clean:
	@echo "🧹 Cleaning temporary files and logs..."
	find logs/ -name "*.log" -mtime +7 -delete 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name ".eval" -delete 2>/dev/null || true
	@echo "✅ Cleanup complete"

lint:
	@echo "🔍 Running code quality checks..."
	ruff check .
	ruff format . --check
	@echo "✅ Code quality checks complete"

lint-fix:
	@echo "🔧 Auto-fixing code quality issues..."
	ruff check . --fix
	ruff format .
	@echo "✅ Code formatting complete"


# ===============================
# Complete Workflows
# ===============================

workflow-data-creation: data-collect-all data-initial-clean
	@echo "✅ Initial workflow complete - ready for analysis"

workflow-complete: data-collect-all data-initial-clean data-embed-analysis-all data-cl-all
	@echo "✅ Analysis workflow complete"

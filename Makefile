# MCP Monitoring Dashboard Makefile
# Common commands for data collection, processing, and analysis
# ############################
# NOTE: THIS IS NOT YET TESTED! 
# ############################
# Prerequisites:
# - Virtual environment: uv sync / source ~/mcp-monitoring/.venv/bin/activate

.PHONY: help data-collect-servers data-collect-usage data-collect-all data-initial data-clean-readmes data-initial-clean data-cl-aicreated data-cl-servers data-cl-tools data-cl-servers-enrich data-cl-all data-task-clusters clean lint lint-fix workflow-data-creation workflow-complete data-update-initial data-update-clean-readmes data-update-cl-aicreated data-update-cl-servers data-update-cl-tools data-update-cl-servers-enrich data-update-cl-all data-update-all data-update-test

# Configurable parameters for incremental updates
DATE_AFTER ?= 2025-10-01
DATE_BEFORE ?= 2026-01-31
MODEL ?= anthropic/claude-sonnet-4-5-20250929
MAX_CONNECTIONS ?= 50

# Default target
help:
	@echo "MCP Monitoring Dashboard - Available Commands:"
	@echo ""
	@echo "  Data Collection (External Sources):"
	@echo "    data-collect-servers    Collect from all 4 server sources"
	@echo "    data-collect-usage      Collect usage statistics"
	@echo "    data-collect-all        Run complete data collection pipeline"
	@echo ""
	@echo "  Data Processing (Internal):"
	@echo "    data-initial           Create unified and filtered datasets"
	@echo "    data-clean-readmes     Filter README content using LLM"
	@echo "    data-initial-clean     Run data-initial + data-clean-readmes"
	@echo ""
	@echo "  Classification (Consequentiality Scoring):"
	@echo "    data-cl-aicreated      Detect AI-created servers via git history"
	@echo "    data-cl-servers        CLServers pipeline (server classification)"
	@echo "    data-cl-tools          CLTools pipeline (O*NET task mapping)"
	@echo "    data-cl-servers-enrich Enrich CLServers with tool aggregations"
	@echo "    data-cl-all            Complete classification pipeline"
	@echo "    data-task-clusters     O*NET task clustering"
	@echo ""
	@echo "  Maintenance:"
	@echo "    clean                  Clean logs and temporary files"
	@echo "    lint                   Run code quality checks"
	@echo "    lint-fix               Auto-fix code quality issues"
	@echo ""
	@echo "  Incremental Update (classify only new servers by date):"
	@echo "    data-update-initial    Re-unify data, preserve LLM fields"
	@echo "    data-update-clean-readmes  Filter READMEs for new servers only"
	@echo "    data-update-cl-aicreated   Detect AI-created servers for date range"
	@echo "    data-update-cl-servers Classify new servers, append to existing"
	@echo "    data-update-cl-tools   Classify new tools, append to existing"
	@echo "    data-update-cl-servers-enrich  Re-enrich with tool aggregations"
	@echo "    data-update-cl-all     Full incremental classification pipeline"
	@echo "    data-update-all        Full incremental pipeline (collect+process+classify)"
	@echo "    data-update-test       Mini-test: 10 servers from date range"
	@echo ""
	@echo "  Complete Workflows:"
	@echo "    workflow-data-creation Data collection + initial processing"
	@echo "    workflow-complete      Full pipeline (all steps)"
	@echo ""
	@echo "  Config: DATE_AFTER=$(DATE_AFTER) DATE_BEFORE=$(DATE_BEFORE) MODEL=$(MODEL)"
	@echo "  Prerequisites: source ~/mcp-monitoring/.venv/bin/activate"

# ===============================
# Data Collection (External)
# ===============================

data-collect-servers:
	@echo "🔄 Collecting MCP server data from all 4 sources..."
	python scripts/data-collection-servers/smithery_data_run.py --resume
	python scripts/data-collection-servers/github_data_run.py --resume
	python scripts/data-collection-servers/officiallist_data_run.py
	python scripts/data-collection-servers/officiallist_data_run.py --awesomelist
	@echo "✅ Server data collection complete"

data-collect-usage:
	@echo "🔄 Collecting usage statistics..."
	@echo "  Step 1: Searching npm registry for MCP packages..."
	python scripts/data-collection-usage/usage_npm_search.py --detailed
	@echo "  Step 2: Fetching download statistics from npm API..."
	python scripts/data-collection-usage/usage_collect_npm.py
	@echo "✅ Usage data collection complete"

data-collect-all: data-collect-servers data-collect-usage

# ===============================
# Data Processing (Internal)
# ===============================

data-initial:
	@echo "🔄 Creating unified datasets..."
	python scripts/data-unification/data_unified_processor.py
	python scripts/data-unification/data_unified_add_usage.py
	python scripts/data-unification/data_unified_create_filtered_subset.py
	@echo "✅ Unified datasets created: data/initial/"

data-clean-readmes:
	@echo "🔄 Filtering README content using LLM..."
	inspect eval scripts/data-cleaning-readmes/data_readme_filter_inspect.py --model anthropic/claude-sonnet-4-5-20250929 --temperature 0 --max-connections 50
	python scripts/data-cleaning-readmes/data_readme_filter_dfprocessing.py
	python scripts/data-unification/data_unified_update_filtered_subset.py
	@echo "✅ README filtering complete"

data-initial-clean: data-initial data-clean-readmes


# ===============================
# AI-Created Detection
# ===============================

data-cl-aicreated:
	@echo "🔄 Detecting AI-created servers via git history mining..."
	python scripts/data-classification-aicreatedmcp/detect_ai_created.py --resume
	@echo "🔄 Validating AI-created detection results..."
	python -m pytest tests/test_ai_created.py -q
	@echo "✅ AI-created detection complete: data/internal-cl/aicreated_results.json"


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
	@echo "  Running finance identification (evaluates all servers)..."
	inspect eval scripts/data-classification-servers/clservers_2_inspect.py --model anthropic/claude-sonnet-4-5-20250929 --temperature 0 --max-connections 50
	@echo "  Running NAICS industry classification..."
	inspect eval scripts/data-classification-servers/clservers_2_inspect.py@naics_classification_task --model anthropic/claude-sonnet-4-5-20250929 --temperature 0
	@echo "  Processing finance identification results..."
	python scripts/data-classification-servers/clservers_3_dfprocessing.py --task finance-identification
	@echo "  Processing NAICS classification results..."
	python scripts/data-classification-servers/clservers_3_dfprocessing.py --task naics
	@echo "  Matching and merging all results..."
	python scripts/data-classification-servers/clservers_4_datamatch.py
	@echo "✅ CLServers pipeline complete: data/final/clservers_classified.csv.gz"

data-cl-tools:
	@echo "🔄 Running CLTools pipeline (O*NET task mapping)..."
	python scripts/data-classification-tools/cltools_main.py --run
	python scripts/data-classification-tools/cltools_datamatch.py --stage4 data/internal-cl/cltools_3_results.csv --stage2 data/final/clservers_classified.csv.gz --usage data/initial/data_unified_filtered.json --output data/final/cltools_classified.csv.gz
	@echo "✅ CLTools pipeline complete: data/final/cltools_classified.csv.gz"

data-cl-servers-enrich:
	@echo "🔄 Enriching CLServers with aggregated tool classifications..."
	python scripts/data-classification-tools/cltools_datamatch_toservers.py --cltools data/final/cltools_classified.csv.gz --clservers data/final/clservers_classified.csv.gz --output data/final/clservers_classified.csv.gz
	@echo "✅ CLServers enrichment complete: data/final/clservers_classified.csv.gz"

data-cl-all: data-cl-aicreated data-task-clusters data-cl-servers data-cl-tools data-cl-servers-enrich



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

workflow-complete: data-collect-all data-initial-clean data-cl-all
	@echo "✅ Complete workflow finished"


# ===============================
# Incremental Update Pipeline
# ===============================
# Only classifies NEW servers (by creation date) and appends to existing data.
# Usage: make data-update-all DATE_AFTER=2025-10-01 DATE_BEFORE=2026-01-31
# Mini-test: make data-update-test DATE_AFTER=2025-10-01 DATE_BEFORE=2026-01-31

data-update-initial:
	@echo "🔄 Re-unifying data with LLM field preservation..."
	python scripts/data-unification/data_unified_processor.py
	python scripts/data-unification/data_unified_add_usage.py
	python scripts/data-unification/data_unified_create_filtered_subset.py --preserve-llm-fields
	@echo "✅ Unified datasets created with LLM fields preserved"

data-update-clean-readmes:
	@echo "🔄 Filtering READMEs for new servers only ($(DATE_AFTER) to $(DATE_BEFORE))..."
	MCP_CREATED_AFTER=$(DATE_AFTER) MCP_CREATED_BEFORE=$(DATE_BEFORE) MCP_SKIP_EXISTING_FILTERED=1 \
		inspect eval scripts/data-cleaning-readmes/data_readme_filter_inspect.py --model $(MODEL) --temperature 0 --max-connections $(MAX_CONNECTIONS)
	python scripts/data-cleaning-readmes/data_readme_filter_dfprocessing.py
	python scripts/data-unification/data_unified_update_filtered_subset.py
	@echo "✅ README filtering complete for new servers"

data-update-cl-aicreated:
	@echo "🔄 Detecting AI-created servers ($(DATE_AFTER) to $(DATE_BEFORE))..."
	python scripts/data-classification-aicreatedmcp/detect_ai_created.py --resume --created-after $(DATE_AFTER) --created-before $(DATE_BEFORE) --append-to data/internal-cl/aicreated_results.json
	@echo "🔄 Validating AI-created detection results..."
	python -m pytest tests/test_ai_created.py -q
	@echo "✅ AI-created detection complete for new servers"

data-update-cl-servers:
	@echo "🔄 Classifying new servers ($(DATE_AFTER) to $(DATE_BEFORE))..."
	python scripts/data-classification-servers/clservers_1_dataprep.py --all --created-after $(DATE_AFTER) --created-before $(DATE_BEFORE)
	@echo "  Running finance identification on new servers..."
	inspect eval scripts/data-classification-servers/clservers_2_inspect.py --model $(MODEL) --temperature 0 --max-connections $(MAX_CONNECTIONS)
	@echo "  Running NAICS industry classification on new servers..."
	inspect eval scripts/data-classification-servers/clservers_2_inspect.py@naics_classification_task --model $(MODEL) --temperature 0
	@echo "  Processing results..."
	python scripts/data-classification-servers/clservers_3_dfprocessing.py --task finance-identification
	python scripts/data-classification-servers/clservers_3_dfprocessing.py --task naics
	@echo "  Matching and appending to existing data..."
	python scripts/data-classification-servers/clservers_4_datamatch.py --append-to data/final/clservers_classified.csv.gz
	@echo "✅ CLServers incremental update complete"

data-update-cl-tools:
	@echo "🔄 Classifying new tools ($(DATE_AFTER) to $(DATE_BEFORE))..."
	python scripts/data-classification-tools/cltools_main.py --run --created-after $(DATE_AFTER) --created-before $(DATE_BEFORE) --model $(MODEL) --max-connections $(MAX_CONNECTIONS)
	python scripts/data-classification-tools/cltools_datamatch.py \
		--cltools data/internal-cl/cltools_3_results.csv \
		--clservers data/final/clservers_classified.csv.gz \
		--usage data/initial/data_unified_filtered.json \
		--output data/final/cltools_classified.csv.gz \
		--append-to data/final/cltools_classified.csv.gz
	@echo "✅ CLTools incremental update complete"

data-update-cl-servers-enrich:
	@echo "🔄 Re-enriching CLServers with aggregated tool classifications..."
	python scripts/data-classification-tools/cltools_datamatch_toservers.py --cltools data/final/cltools_classified.csv.gz --clservers data/final/clservers_classified.csv.gz --output data/final/clservers_classified.csv.gz
	@echo "✅ CLServers enrichment complete"

data-update-cl-all: data-update-cl-aicreated data-update-cl-servers data-update-cl-tools data-update-cl-servers-enrich

data-update-all: data-collect-servers data-update-initial data-update-clean-readmes data-update-cl-all
	@echo "✅ Full incremental update complete"

data-update-test:
	@echo "🧪 Mini-test: classifying 10 new servers from $(DATE_AFTER) to $(DATE_BEFORE)..."
	python scripts/data-classification-servers/clservers_1_dataprep.py --samples 10 --created-after $(DATE_AFTER) --created-before $(DATE_BEFORE)
	@echo "  Running finance identification (test)..."
	inspect eval scripts/data-classification-servers/clservers_2_inspect.py --model $(MODEL) --temperature 0 --max-connections 5
	@echo "  Running NAICS classification (test)..."
	inspect eval scripts/data-classification-servers/clservers_2_inspect.py@naics_classification_task --model $(MODEL) --temperature 0
	@echo "  Processing test results..."
	python scripts/data-classification-servers/clservers_3_dfprocessing.py --task finance-identification
	python scripts/data-classification-servers/clservers_3_dfprocessing.py --task naics
	python scripts/data-classification-servers/clservers_4_datamatch.py
	@echo "✅ Mini-test complete - check data/final/clservers_classified.csv.gz"

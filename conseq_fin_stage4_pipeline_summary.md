# O*NET Task Classification Pipeline

## Overview
This pipeline implements the methodology from Anthropic's paper "Which Economic Tasks are Performed with AI?" to classify MCP server tools according to O*NET occupational tasks. It creates a 3-level hierarchy of ~20,000 O*NET tasks and uses it to classify MCP tools.

## Data Flow & Architecture

### Input Data
- **`conseq_fin_stage4_onet_taskstatements.csv`** - Source O*NET task statements (~20K tasks)
- **`conseq_fin_stage4_onet_toolsused.csv`** - O*NET tools data
- **`data_unified_filtered.json`** - Filtered MCP server dataset (16,940 servers with tools)

### 3-Level Task Hierarchy
```
Level 1: 10-12 top-level economic categories (predefined)
    ↓
Level 2: ~400 middle-level clusters (k-means)
    ↓  
Level 3: ~20,000 individual O*NET tasks
```

## Core Pipeline Components

### 1. Hierarchical Task Clustering
**`conseq_fin_stage4_embed_levels.py`** - Creates the 3-level hierarchy

**Process:**
1. **Load O*NET Tasks**: Reads `onet_taskstatements.csv` with ~20K tasks
2. **Generate Embeddings**: Creates sentence embeddings using `all-mpnet-base-v2`
3. **Level 2 Clustering**: K-means clustering (K=400) to group similar tasks
4. **Level 1 Assignment**: Assigns Level 2 clusters to predefined economic categories using embedding similarity
5. **Output**: `conseq_fin_stage4_hierarchy_k12.json` with complete hierarchy structure

### 2. Cluster Naming (LLM-Enhanced)

**Level 2 Naming Process:**
1. **`conseq_fin_stage4_generate_cluster_names.py`** - LLM generates descriptive names
   - Analyzes all Level 3 tasks within each Level 2 cluster
   - Creates concise, descriptive names for all 400 clusters
2. **`conseq_fin_stage4_process_cluster_names.py`** - Processes and validates LLM results

**Level 1 Naming Process:**
1. **`conseq_fin_stage4_generate_level1_names.py`** - Bottom-up naming approach
   - Uses Level 2 cluster names to generate overarching Level 1 names
   - Creates coherent high-level category names
2. **`conseq_fin_stage4_process_level1_names.py`** - Processes Level 1 naming results
3. **`conseq_fin_stage4_add_level1_names.py`** - Applies predefined names (alternative approach)
   - Uses occupation analysis for name assignment

### 3. Tool Data Preparation
**`conseq_fin_stage4_data_prep.py`** - Transforms MCP server data into tool classification format

**Data Flow:**
```
data_unified_filtered.json (16,940 servers)
↓
Extract individual tools from each server's tools[] array
↓  
Combine tool info with server context (README, description, metadata)
↓
Format for LLM analysis → conseq_fin_stage4_input.jsonl
```

**Tool Context Extraction:**
- **Tool Data**: name, description, input schema from `tools[]` array
- **Server Context**: name, description, readme_summary, data_sources
- **Sampling Options**: `--all`, `--samples N`, `--finance` for focused analysis

**Output Formats:**
- `conseq_fin_stage4_input.jsonl` - LLM-ready samples for Inspect framework
- `conseq_fin_stage4_tools_full.json` - Complete tool dataset with metadata
- `conseq_fin_stage4_data_prep_summary.json` - Extraction statistics

### 4. MCP Tool Classification
**`conseq_fin_stage4_inspect.py`** - Main hierarchical classification
- Uses the 3-level hierarchy to classify MCP tools
- Maps tools to specific O*NET tasks (Level 3)
- Provides Level 1 and Level 2 category assignments

**Alternative: 4 Separate Analysis Dimensions**
- **`conseq_fin_stage4_inspect_1_task.py`** - O*NET task mapping
- **`conseq_fin_stage4_inspect_2_collab.py`** - Collaboration patterns  
- **`conseq_fin_stage4_inspect_3_auto.py`** - Automation levels (0-5 scale)
- **`conseq_fin_stage4_inspect_4_tools.py`** - Tool replacement analysis

### 5. Hierarchy Validation Framework
- **`conseq_fin_stage4_validation_prep.py`** - Prepare validation datasets
  - Creates stratified samples for 3 validation types
  - Formats data for includes() scorer
  
- **Validation Scripts**:
  - `conseq_fin_stage4_validate_l3_to_l1.py` - Task to Level 1 validation
  - `conseq_fin_stage4_validate_l2_to_l1.py` - Level 2 to Level 1 validation
  - `conseq_fin_stage4_validate_l3_to_l2.py` - Task to Level 2 validation
  - `conseq_fin_stage4_validate_l3_to_l1_k12.py` - K=12 L3→L1 validation
  - `conseq_fin_stage4_validate_l2_to_l1_k12.py` - K=12 L2→L1 validation
  
- **`conseq_fin_stage4_validation_analysis.py`** - Analyze validation results
  - Calculates accuracy metrics
  - Identifies problematic clusters
  - Generates validation reports

- **`conseq_fin_stage4_validation_scorer.py`** - Custom scoring logic (deprecated)

### 6. Results Processing & Analysis
- **`conseq_fin_stage4_dfprocessing.py`** - Process Inspect results
  - Extracts all classification results
  - Creates comprehensive DataFrame
  - Outputs JSON and CSV formats
  
- **`conseq_fin_stage4_dfprocessing_multi.py`** - Process multiple inspect runs
  - Handles 4 separate analysis dimensions
  - Combines results into unified output

- **`conseq_fin_stage4_dfprocessing_natural.py`** - Natural language processing
- **`conseq_fin_stage4_dfprocessing_structured.py`** - Structured data processing

- **`conseq_fin_stage4_analysis.py`** - Generate visualizations
  - Task distribution across categories
  - Automation vs augmentation analysis
  - Tool replacement patterns
  - Occupation-level impact

### 7. Supporting Utilities
- **`conseq_fin_stage4_test_pipeline.py`** - Pipeline testing and validation
- **`conseq_fin_stage4_inspect_fixed.py`** - Fixed version of inspect script
- **`conseq_fin_stage4_inspect_simple.py`** - Simplified inspect script

## Validation Results Summary

### Baseline Performance (Original/Basic Names)
- **L3→L2**: **90% accuracy** ✅ (Excellent!)
- **L3→L1**: 62.8% accuracy ⚠️
- **L2→L1**: 61.0% accuracy ⚠️

### Hierarchy Performance Comparison

| Clustering Approach | Level 1 Clusters | L2→L1 Accuracy | L3→L1 Accuracy | Best For |
|---------------------|------------------|------------------|------------------|----------|
| **K=10 (Original)** | 10 | 64% | 58% | Baseline |
| **K=12 (Recommended)** | 12 | 58% | **62%** | **End-to-end tool classification** |
| **K=20** | 20 | **68%** | 46% | High-level categorization |

### Key Insights & Recommendations

**For MCP Tool Classification:**
- **Use K=12 hierarchy** - Provides best L3→L1 accuracy (62%) for end-to-end tool classification
- **L3→L1 matters most** - This represents the actual classification task (individual tasks to broad categories)
- **400 Level 2 clusters work well** - Good granularity without over-segmentation

**Technical Findings:**
- **Trade-off exists** - More Level 1 clusters improve L2→L1 but hurt L3→L1 performance  
- **Embedding similarity works** - Cosine similarity effectively assigns clusters to economic categories
- **LLM naming adds value** - Human-readable names significantly improve interpretability

## Step-by-Step Pipeline Execution

### Step 1: Build O*NET Task Hierarchy
```bash
# Prerequisite: Ensure O*NET data files exist
# - conseq_fin_stage4_onet_taskstatements.csv
# - conseq_fin_stage4_onet_toolsused.csv

# Build 3-level hierarchy with embeddings and clustering
python conseq_fin_stage4_embed_levels.py

# Outputs:
# - conseq_fin_stage4_hierarchy_k12.json (complete hierarchy structure)
# - conseq_fin_stage4_embeddings_cache.npz (cached embeddings)
```

### Step 2: Generate Human-Readable Cluster Names
```bash
# Step 2a: Generate Level 2 cluster names via LLM
inspect eval conseq_fin_stage4_generate_cluster_names.py --model anthropic/claude-sonnet-4-20250514
python conseq_fin_stage4_process_cluster_names.py

# Step 2b: Generate Level 1 category names via LLM  
inspect eval conseq_fin_stage4_generate_level1_names.py --model anthropic/claude-sonnet-4-20250514
python conseq_fin_stage4_process_level1_names.py

# Outputs:
# - conseq_fin_stage4_cluster_names.csv (Level 2 names)
# - conseq_fin_stage4_hierarchy_k12_names_summary.json (Level 1 names)
```

### Step 3: Validate Hierarchy Quality (Optional)
```bash
# Prepare validation datasets (stratified sampling)
python conseq_fin_stage4_validation_prep.py --samples-per-cluster 5

# Test hierarchy classification accuracy
inspect eval conseq_fin_stage4_validate_l3_to_l1.py --model anthropic/claude-sonnet-4-20250514 --message-limit 50
inspect eval conseq_fin_stage4_validate_l2_to_l1.py --model anthropic/claude-sonnet-4-20250514 --message-limit 50
inspect eval conseq_fin_stage4_validate_l3_to_l1_k12.py --model anthropic/claude-sonnet-4-20250514 --message-limit 50

# Analyze validation results and generate accuracy reports
python conseq_fin_stage4_validation_analysis.py

# Key Validation Results:
# - L3→L1 (K=12): 62% accuracy (best for end-to-end classification)
# - L2→L1 (K=10): 64% accuracy  
# - L2→L1 (K=20): 68% accuracy
```

### Step 4: Prepare MCP Tool Data for Classification
```bash
# Prerequisite: Ensure MCP server data exists
# - data_unified_filtered.json (16,940 servers with tools)

# Extract tools from MCP servers for classification
python conseq_fin_stage4_data_prep.py --all                    # All available tools
python conseq_fin_stage4_data_prep.py --samples 1000           # Sample for testing  
python conseq_fin_stage4_data_prep.py --samples 500 --finance  # Finance-focused sample

# Outputs:
# - conseq_fin_stage4_input.jsonl (formatted for Inspect framework)
# - conseq_fin_stage4_tools_full.json (complete tool dataset)
# - conseq_fin_stage4_data_prep_summary.json (extraction statistics)
```

### Step 5: Classify MCP Tools Using O*NET Hierarchy
```bash
# Main hierarchical classification (recommended)
inspect eval conseq_fin_stage4_inspect.py --model anthropic/claude-sonnet-4-20250514

# Alternative: Run 4 separate analysis dimensions
inspect eval conseq_fin_stage4_inspect_1_task.py --model anthropic/claude-sonnet-4-20250514      # O*NET task mapping
inspect eval conseq_fin_stage4_inspect_2_collab.py --model anthropic/claude-sonnet-4-20250514    # Collaboration analysis  
inspect eval conseq_fin_stage4_inspect_3_auto.py --model anthropic/claude-sonnet-4-20250514      # Automation levels (0-5)
inspect eval conseq_fin_stage4_inspect_4_tools.py --model anthropic/claude-sonnet-4-20250514     # Tool replacement patterns

# Classification results saved in logs/ directory as .eval files
```

### Step 6: Process Classification Results  
```bash
# Process main classification results
python conseq_fin_stage4_dfprocessing.py

# Or process multiple analysis dimensions
python conseq_fin_stage4_dfprocessing_multi.py

# Alternative processing approaches
python conseq_fin_stage4_dfprocessing_natural.py      # Natural language processing
python conseq_fin_stage4_dfprocessing_structured.py   # Structured data processing

# Outputs:
# - conseq_fin_stage4_results.json/csv (complete results)
# - conseq_fin_stage4_multi_summary.json (combined analysis)
```

### Step 7: Generate Analysis & Visualizations
```bash
# Generate comprehensive analysis and visualizations
python conseq_fin_stage4_analysis.py

# Outputs:
# - conseq_fin_stage4_visualizations/ (charts and graphs)
# - Task distribution across economic categories
# - Automation vs augmentation analysis  
# - Tool replacement patterns
# - Occupation-level impact analysis
```

## Key Output Files

### Core Hierarchy Files
- **`conseq_fin_stage4_hierarchy_k12.json`** - Complete 3-level hierarchy structure
- **`conseq_fin_stage4_hierarchy_k12_summary.json`** - Hierarchy metadata and statistics
- **`conseq_fin_stage4_hierarchy_k12_names_summary.json`** - Human-readable cluster names
- **`conseq_fin_stage4_cluster_names.csv`** - Level 2 cluster names (400 clusters)
- **`conseq_fin_stage4_embeddings_cache.npz`** - Cached task embeddings (~20K tasks)

### Validation
- `conseq_fin_stage4_validation_*.jsonl` - Validation datasets
- `conseq_fin_stage4_validation_report.json` - Validation results
- `conseq_fin_stage4_validation_accuracy.png` - Accuracy visualization

### Classification Results
- `conseq_fin_stage4_input.jsonl` - Tool dataset for classification
- `conseq_fin_stage4_results.json` - Complete classification results
- `conseq_fin_stage4_results.csv` - CSV format
- `conseq_fin_stage4_multi_summary.json` - Combined analysis summary

### Analysis Outputs
- `conseq_fin_stage4_visualizations/` - All charts and graphs
- `conseq_fin_stage4_analysis_report.json` - Detailed insights

## Technical Implementation Details

### Embedding & Clustering Pipeline
```
O*NET Tasks (~20K) 
↓ 
Sentence Embeddings (all-mpnet-base-v2, 768-dim)
↓
K-means Clustering (K=400) → Level 2 Clusters
↓
Cosine Similarity Assignment → Level 1 Categories (K=10/12/20)
↓
LLM Naming → Human-readable cluster names
```

### Model Configuration
- **Embedding Model**: `sentence-transformers/all-mpnet-base-v2`
- **Clustering**: K-means with K=400 (Level 2), K=12 (Level 1 recommended)
- **Assignment**: Cosine similarity between cluster centers and predefined categories
- **Caching**: Embeddings cached in `conseq_fin_stage4_embeddings_cache.npz`

### LLM Integration
- **Classification Model**: Claude Sonnet 4 (anthropic/claude-sonnet-4-20250514)
- **Framework**: Inspect AI for evaluation and batch processing
- **Context Management**: Hierarchical filtering to handle 20K+ tasks efficiently
- **Output Format**: Natural language responses with structured extraction

### Performance & Cost Estimates
- **Embedding Generation**: 30-60 minutes (one-time, GPU recommended)
- **K-means Clustering**: 5-10 minutes for 400 clusters  
- **LLM Cluster Naming**: ~$50-100 API costs (400 Level 2 + 12 Level 1)
- **Tool Classification**: ~$350-700 for complete MCP dataset
- **Validation Testing**: ~$10-20 per validation run

### Alignment with Anthropic Methodology
- ✅ **3-level hierarchy**: Top categories → Middle clusters → Individual tasks
- ✅ **Embedding-based clustering**: Semantic similarity for task grouping
- ✅ **K-means methodology**: Consistent with paper's approach
- ✅ **Economic task categories**: Meaningful groupings for economic analysis
- ✅ **Validation framework**: Empirical accuracy testing (addition to paper)
- ✅ **Tool classification**: Applied to MCP ecosystem (novel application)

## Usage Recommendations

### For MCP Tool Classification
1. **Use K=12 hierarchy** - Best balance of granularity and accuracy
2. **Focus on L3→L1 performance** - Most relevant for end-to-end tool classification  
3. **Leverage full pipeline** - Embeddings + clustering + LLM naming provides best results
4. **Cache embeddings** - Reuse expensive embedding computation across experiments

### For Economic Analysis
1. **400 Level 2 clusters** provide good task granularity without over-segmentation
2. **12 Level 1 categories** offer meaningful economic groupings
3. **Validation framework** enables empirical comparison of different approaches
4. **LLM naming** significantly improves interpretability of clusters

## Summary

This pipeline successfully implements and extends the Anthropic "Which Economic Tasks are Performed with AI?" methodology for MCP tool classification:

**✅ Robust 3-level hierarchy** - 20K O*NET tasks organized into 400 middle clusters and 12 top categories  
**✅ High-quality embeddings** - Semantic task similarity using state-of-the-art sentence transformers  
**✅ Validated performance** - Empirical testing shows 62% L3→L1 accuracy with K=12 approach  
**✅ LLM-enhanced naming** - Human-readable cluster names for interpretability  
**✅ MCP tool application** - Ready for large-scale classification of AI tools and capabilities  
 -  
          -  | Hierarchy     | L2→L1 Accuracy | L3→L1 Accuracy | Key Insight                    |
        -    |---------------|----------------|----------------|--------------------------------|
        -    | Original K=10 | 64%            | 58%            | Baseline performance           |
        -    | K=20          | 68%            | 46%            | Better L2→L1, much worse L3→L1 |
        -    | K=12          | 58%            | 62%            | Best L3→L1 performance         |
The K=12 hierarchy provides the optimal balance for classifying MCP tools, offering sufficient granularity while maintaining reasonable accuracy for end-to-end task assignment.
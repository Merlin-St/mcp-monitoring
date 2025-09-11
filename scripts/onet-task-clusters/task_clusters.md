# ONET Task Clustering Pipeline

## Overview

This pipeline implements a unified approach to clustering ~20,000 ONET occupational tasks into a two-level hierarchy using embeddings and LLM-generated names. The system creates:

- **Level 2**: 400 task clusters (via K-means on task embeddings)
- **Level 1**: 12 supercluster categories (via K-means on Level 2 cluster name embeddings)

## Quick start
  python task_clusters_run.py --k1 12 --k2 400 --skip-validation

  What to expect:

  1. Step 1: Load ~20K ONET tasks
  2. Step 2: Generate embeddings (5-10 min first time, cached after)
  3. Step 3: LLM generates Level 2 names (API calls)
  4. Step 4: Embed Level 2 names and cluster
  5. Step 5: LLM generates Level 1 names (API calls)
  6. Step 6: Run validations (optional)

## Architecture

### Core Components

1. **`task_clusters_embeddings.py`** - Embedding generation and clustering
   - Flexible functions for embedding both ONET tasks and cluster names
   - K-means clustering with configurable k values
   - Embedding cache management

2. **`task_clusters_data.py`** - Data loading and management
   - ONET task CSV loading
   - Incremental CSV updates throughout pipeline
   - Validation sample preparation

3. **`task_clusters_llm.py`** - LLM cluster naming
   - Inspect framework integration
   - Cluster name generation prompts
   - Result processing via messages_df

4. **`task_clusters_run.py`** - Main orchestration script
   - Complete pipeline execution
   - Validation task coordination
   - Summary generation

### Data Flow

```
ONET Tasks CSV
    ↓
Task Embeddings → K-means (k=400) → Level 2 Clusters
    ↓
LLM Name Generation → Level 2 Cluster Names
    ↓
Name Embeddings → K-means (k=12) → Level 1 Clusters
    ↓
LLM Name Generation → Level 1 Cluster Names
    ↓
Validation Tasks → Accuracy Scores
    ↓
Final CSV + Summary
```

## Usage

### Basic Execution

```bash
# Default configuration (k1=12, k2=400)
python task_clusters_run.py

# Custom cluster counts
python task_clusters_run.py --k1 10 --k2 300

# Skip validation for faster execution
python task_clusters_run.py --skip-validation
```

### Prerequisites

- ONET task data: `../../data/external-cl/cl_onet_taskstatements.csv`
- Anthropic API key set in environment
- Inspect framework installed

## Output Files

### Primary Output
- **`../../data/internal-task-clusters/task_clusters_names.csv`** - Complete task assignments with cluster names
  ```
  task_id | onet_code | task | title | level2_cluster | level2_name | level1_cluster | level1_name
  ```

### Summary
- **`../../data/internal-task-clusters/task_clusters_summary.json`** - Pipeline statistics and validation scores
  ```json
  {
    "parameters": {"k1": 12, "k2": 400},
    "statistics": {
      "total_tasks": 20000,
      "level2_clusters": 400,
      "level1_clusters": 12
    },
    "validation_scores": {
      "l3_to_l2": 0.90,
      "l2_to_l1": 0.62,
      "l3_to_l1": 0.58
    }
  }
  ```

### Cache Files
- **`../../embeddings_cache/task_clusters_embeddings_onet.npz`** - Cached ONET task embeddings
- **`../../embeddings_cache/task_clusters_embeddings_l2.npz`** - Cached Level 2 name embeddings

### Logs
- **`logs/task_clusters_run.log`** - Main pipeline execution log
- **`logs/`** - Inspect evaluation logs for naming and validation

## Implementation Details

### Embedding Generation
- Model: `sentence-transformers/all-mpnet-base-v2`
- Task text includes occupation context: `"[Task] [Occupation Title]"`
- Batch processing for memory efficiency
- Automatic caching to avoid recomputation

### Clustering
- K-means with 10 initializations for stability
- Level 2: 400 clusters from ~20K tasks
- Level 1: 12 clusters from 400 Level 2 names

### LLM Naming
- Model: Claude Sonnet 4 (anthropic/claude-sonnet-4-20250514)
- Concise names (3-7 words) focused on primary function
- Direct generation without explanations

### Validation
- Three validation types: L3→L2, L2→L1, L3→L1
- Uses exact prompts as specified
- Inspect framework with includes() scorer
- Accuracy reported in summary

## System Prompts

### Cluster Naming
```
You are an expert at analyzing occupational tasks and creating clear, descriptive category names.
Your task is to generate concise, professional names for task clusters based on their content.
Focus on the primary function or activity that unifies the tasks in each cluster. Provide a descriptive name that captures the common theme of these tasks.
The name should be:

Concise (3-7 words)
Professional and clear
Focused on the primary function/activity
```

### Validation
```
The following is a description of an occupational task: [task/cluster name]. 
Consider the following list of classification options: [cluster options]. 
Your job is to identify which option best describes the occupational task. 
What is the answer? You MUST provide an option exactly as written above. 
If multiple options apply, choose the single-most pertinent one. 
Respond ONLY with the cluster ID (e.g. L1_06 or similar).
```

## Performance

- Task embedding: ~5-10 minutes (one-time with caching)
- Level 2 clustering: ~2-3 minutes
- Level 2 naming: ~$20-30 API cost
- Level 1 clustering: <1 minute
- Level 1 naming: ~$5-10 API cost
- Validation: ~$10-20 per validation type
- Total runtime: ~30-45 minutes (first run)

## Key Features

- **Unified Pipeline**: Single script orchestrates entire process
- **Flexible Configuration**: Adjustable k values for different granularities
- **Efficient Caching**: Embeddings cached to avoid recomputation
- **Clean Output**: Single CSV with all assignments and names
- **Rigorous Validation**: Three-level validation with accuracy metrics
- **Modular Design**: Reusable components for future extensions


## Level 1 Category Assignment Process - Step by Step


  1. Input Data

  - Level 2 cluster centers: 400 embedding vectors (one
   per Level 2 cluster)
  - Level 1 categories: 12 predefined semantic
  categories

  2. Embed Category Names

  category_texts = [
      "Business Management and Financial Operations",
      "Healthcare and Medical Services",
      "Education and Training Services",
      # ... all 12 categories
  ]
  category_embeddings = embed(category_texts)  # 12 
  embedding vectors

  3. Calculate Similarity Matrix

  similarities = cosine_similarity(cluster_centers,
  category_embeddings)
  # Result: 400x12 matrix where similarities[i][j] = 
  similarity between Level2_cluster_i and 
  Level1_category_j

  4. Assign Each Level 2 Cluster to Best Level 1 
  Category

  level1_assignments = np.argmax(similarities, axis=1)
   # For each L2 cluster, find index of most similar L1
   category
  best_similarities = np.max(similarities, axis=1)
   # Store the similarity score

  5. Example Assignment Process

  For Level 2 cluster cluster_2_042 (hypothetically 
  about "Medical diagnoses"):

  | Level 1 Category              | Similarity Score |
  |-------------------------------|------------------|
  | L1_01: Business Management    | 0.12             |
  | L1_02: Healthcare and Medical | 0.87 ← HIGHEST   |
  | L1_03: Education              | 0.23             |
  | L1_04: IT                     | 0.15             |
  | ...                           | ...              |

  Result: cluster_2_042 gets assigned to L1_02: 
  Healthcare and Medical Services

  6. Assignment Results

  assignment_details['assignments'][42] = {
      'level1_id': 'L1_02',
      'level1_name': 'Healthcare and Medical Services',

      'similarity': 0.87
  }
#!/usr/bin/env python3
"""
Embedding-based MCP tool to ONET task matching using cosine similarity

This script:
1. Loads 100 samples from stage4_input.jsonl (MCP server tools)
2. Embeds the MCP tool descriptions using functions from stage4_task_clusters_embeddings.py
3. Loads pre-embedded ONET tasks from the NPZ cache
4. Computes cosine similarities between MCP tools and ONET tasks
5. Finds best matches and generates analysis results

This is an embedding-based alternative to the LLM-based matching approach.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.metrics.pairwise import cosine_similarity

# Import functions from the stage4 task embeddings module
from conseq_fin_stage4_task_clusters_embeddings import (
    get_embedding_model,
    load_or_generate_embeddings,
    EMBEDDING_MODEL_NAME
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conseq_fin_stage4_task_clusters_embed_match.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MCPToONETMatcher:
    """Main class for matching MCP tools to ONET tasks using embeddings"""
    
    def __init__(self):
        """Initialize the matcher with required data paths"""
        self.mcp_input_file = "conseq_fin_stage4_input.jsonl"
        self.onet_tasks_file = "conseq_fin_stage4_onet_taskstatements.csv"
        self.onet_embeddings_file = "conseq_fin_stage4_task_clusters_embeddings_onet.npz"
        self.mcp_embeddings_cache = "conseq_fin_stage4_task_clusters_mcp_tool_embeddings.npz"
        
        # Data containers
        self.mcp_samples = []
        self.onet_tasks_df = None
        self.onet_embeddings = None
        self.mcp_embeddings = None
        
        # Results
        self.similarity_matrix = None
        self.best_matches = []
        
    def load_mcp_samples(self, limit: int = 100) -> None:
        """Load MCP tool samples from stage4_input.jsonl"""
        logger.info(f"Loading MCP samples from {self.mcp_input_file}")
        
        samples = []
        with open(self.mcp_input_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                sample = json.loads(line)
                samples.append(sample)
        
        self.mcp_samples = samples
        logger.info(f"Loaded {len(self.mcp_samples)} MCP tool samples")
    
    def load_onet_data(self) -> None:
        """Load ONET tasks and their pre-computed embeddings"""
        logger.info(f"Loading ONET tasks from {self.onet_tasks_file}")
        
        # Load ONET tasks CSV
        self.onet_tasks_df = pd.read_csv(self.onet_tasks_file)
        logger.info(f"Loaded {len(self.onet_tasks_df)} ONET tasks")
        
        # Load pre-computed ONET embeddings
        logger.info(f"Loading pre-computed ONET embeddings from {self.onet_embeddings_file}")
        onet_data = np.load(self.onet_embeddings_file)
        self.onet_embeddings = onet_data['embeddings']
        
        logger.info(f"Loaded ONET embeddings: shape {self.onet_embeddings.shape}")
        
        # Verify alignment between tasks and embeddings
        if len(self.onet_tasks_df) != len(self.onet_embeddings):
            logger.warning(f"Mismatch: {len(self.onet_tasks_df)} tasks vs {len(self.onet_embeddings)} embeddings")
            # Take the minimum to avoid index errors
            min_len = min(len(self.onet_tasks_df), len(self.onet_embeddings))
            self.onet_tasks_df = self.onet_tasks_df.iloc[:min_len]
            self.onet_embeddings = self.onet_embeddings[:min_len]
            logger.info(f"Aligned to {min_len} tasks and embeddings")
    
    def prepare_mcp_text_for_embedding(self) -> List[str]:
        """Prepare MCP tool text for embedding generation"""
        logger.info("Preparing MCP tool text for embedding")
        
        mcp_texts = []
        for sample in self.mcp_samples:
            # Parse the input JSON string
            tool_data = json.loads(sample['input'])
            
            # Create comprehensive text representation
            text_parts = []
            
            # Add tool name and description (primary content)
            if tool_data.get('tool_name'):
                text_parts.append(f"Tool: {tool_data['tool_name']}")
            if tool_data.get('tool_description'):
                text_parts.append(f"Description: {tool_data['tool_description']}")
            
            # Add server context for better understanding
            if tool_data.get('server_name'):
                text_parts.append(f"Server: {tool_data['server_name']}")
            if tool_data.get('server_description'):
                text_parts.append(f"Server Description: {tool_data['server_description']}")
            
            # Add README summary if available (additional context)
            if tool_data.get('server_readme_summary'):
                text_parts.append(f"Summary: {tool_data['server_readme_summary']}")
            
            # Combine all parts
            full_text = " | ".join(text_parts)
            mcp_texts.append(full_text)
        
        logger.info(f"Prepared {len(mcp_texts)} MCP tool texts for embedding")
        return mcp_texts
    
    def generate_mcp_embeddings(self) -> None:
        """Generate embeddings for MCP tools using the same model as ONET tasks"""
        logger.info("Generating embeddings for MCP tools")
        
        mcp_texts = self.prepare_mcp_text_for_embedding()
        
        # Use the same embedding function as used for ONET tasks
        self.mcp_embeddings = load_or_generate_embeddings(
            texts=mcp_texts,
            cache_file=self.mcp_embeddings_cache,
            force_regenerate=False
        )
        
        logger.info(f"Generated MCP embeddings: shape {self.mcp_embeddings.shape}")
    
    def compute_similarity_matrix(self) -> None:
        """Compute cosine similarity matrix between MCP tools and ONET tasks"""
        logger.info("Computing cosine similarity matrix")
        
        # Compute cosine similarity between MCP tools (rows) and ONET tasks (columns)
        self.similarity_matrix = cosine_similarity(self.mcp_embeddings, self.onet_embeddings)
        
        logger.info(f"Computed similarity matrix: shape {self.similarity_matrix.shape}")
    
    def find_best_matches(self, top_k: int = 5) -> None:
        """Find best ONET task matches for each MCP tool"""
        logger.info(f"Finding top {top_k} matches for each MCP tool")
        
        self.best_matches = []
        
        for i, sample in enumerate(self.mcp_samples):
            tool_data = json.loads(sample['input'])
            tool_name = tool_data.get('tool_name', f'Tool_{i}')
            
            # Get similarity scores for this MCP tool across all ONET tasks
            similarities = self.similarity_matrix[i]
            
            # Find top-k most similar ONET tasks
            top_indices = np.argsort(similarities)[-top_k:][::-1]  # Sort descending
            top_scores = similarities[top_indices]
            
            matches = []
            for j, (idx, score) in enumerate(zip(top_indices, top_scores)):
                onet_task = self.onet_tasks_df.iloc[idx]
                matches.append({
                    'rank': j + 1,
                    'onet_soc_code': str(onet_task['O*NET-SOC Code']),
                    'onet_title': str(onet_task['Title']),
                    'onet_task_id': int(onet_task['Task ID']) if pd.notna(onet_task['Task ID']) else None,
                    'onet_task': str(onet_task['Task']),
                    'similarity_score': float(score)
                })
            
            self.best_matches.append({
                'mcp_index': i,
                'mcp_tool_name': tool_name,
                'mcp_tool_description': tool_data.get('tool_description', ''),
                'mcp_server_name': tool_data.get('server_name', ''),
                'is_finance': sample.get('metadata', {}).get('is_finance', False),
                'matches': matches
            })
        
        logger.info(f"Found best matches for {len(self.best_matches)} MCP tools")
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze matching results and generate statistics"""
        logger.info("Analyzing matching results")
        
        analysis = {
            'total_mcp_tools': len(self.mcp_samples),
            'total_onet_tasks': len(self.onet_tasks_df),
            'embedding_model': EMBEDDING_MODEL_NAME,
            'embedding_dimensions': self.mcp_embeddings.shape[1],
            'similarity_statistics': {},
            'top_onet_occupations': {},
            'finance_tool_analysis': {},
            'match_quality_distribution': {}
        }
        
        # Collect all similarity scores
        all_top_scores = []
        finance_scores = []
        non_finance_scores = []
        onet_occupation_counts = {}
        
        for match_result in self.best_matches:
            top_match_score = match_result['matches'][0]['similarity_score']
            all_top_scores.append(top_match_score)
            
            if match_result['is_finance']:
                finance_scores.append(top_match_score)
            else:
                non_finance_scores.append(top_match_score)
            
            # Count ONET occupations
            for match in match_result['matches']:
                occ_title = match['onet_title']
                onet_occupation_counts[occ_title] = onet_occupation_counts.get(occ_title, 0) + 1
        
        # Similarity statistics
        analysis['similarity_statistics'] = {
            'mean_top_match_score': float(np.mean(all_top_scores)),
            'median_top_match_score': float(np.median(all_top_scores)),
            'std_top_match_score': float(np.std(all_top_scores)),
            'min_top_match_score': float(np.min(all_top_scores)),
            'max_top_match_score': float(np.max(all_top_scores))
        }
        
        # Finance vs non-finance analysis
        if finance_scores:
            analysis['finance_tool_analysis'] = {
                'num_finance_tools': len(finance_scores),
                'mean_finance_score': float(np.mean(finance_scores)),
                'mean_non_finance_score': float(np.mean(non_finance_scores)) if non_finance_scores else 0.0,
                'finance_vs_non_finance_diff': float(np.mean(finance_scores)) - float(np.mean(non_finance_scores)) if non_finance_scores else 0.0
            }
        
        # Top ONET occupations
        analysis['top_onet_occupations'] = dict(
            sorted(onet_occupation_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        )
        
        # Match quality distribution
        score_ranges = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
        analysis['match_quality_distribution'] = {}
        
        for low, high in score_ranges:
            count = sum(1 for score in all_top_scores if low <= score < high)
            analysis['match_quality_distribution'][f'{low:.1f}-{high:.1f}'] = {
                'count': count,
                'percentage': float(count) / len(all_top_scores) * 100
            }
        
        logger.info(f"Analysis complete: mean top match score = {analysis['similarity_statistics']['mean_top_match_score']:.4f}")
        
        return analysis
    
    def save_results(self, output_file: str = 'conseq_fin_stage4_task_clusters_embed_match_results.json') -> None:
        """Save all results to JSON file"""
        logger.info(f"Saving results to {output_file}")
        
        results = {
            'metadata': {
                'embedding_model': EMBEDDING_MODEL_NAME,
                'mcp_samples_count': len(self.mcp_samples),
                'onet_tasks_count': len(self.onet_tasks_df),
                'embedding_dimensions': self.mcp_embeddings.shape[1],
                'similarity_matrix_shape': list(self.similarity_matrix.shape)
            },
            'analysis': self.analyze_results(),
            'detailed_matches': self.best_matches
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    def run_complete_analysis(self, mcp_sample_limit: int = 100) -> None:
        """Run the complete embedding-based matching analysis"""
        logger.info("=== Starting Embedding-Based MCP to ONET Task Matching ===")
        
        # Load data
        self.load_mcp_samples(limit=mcp_sample_limit)
        self.load_onet_data()
        
        # Generate embeddings
        self.generate_mcp_embeddings()
        
        # Compute similarities and find matches
        self.compute_similarity_matrix()
        self.find_best_matches(top_k=5)
        
        # Save results
        self.save_results()
        
        logger.info("=== Embedding-Based Matching Complete ===")


def main():
    """Main execution function"""
    try:
        # Initialize matcher and run analysis
        matcher = MCPToONETMatcher()
        matcher.run_complete_analysis(mcp_sample_limit=100)
        
        # Log some key findings
        analysis = matcher.analyze_results()
        logger.info("\n=== KEY FINDINGS ===")
        logger.info(f"• Total MCP tools analyzed: {analysis['total_mcp_tools']}")
        logger.info(f"• Total ONET tasks in comparison: {analysis['total_onet_tasks']}")
        logger.info(f"• Average top match similarity: {analysis['similarity_statistics']['mean_top_match_score']:.4f}")
        logger.info(f"• Similarity score range: {analysis['similarity_statistics']['min_top_match_score']:.4f} - {analysis['similarity_statistics']['max_top_match_score']:.4f}")
        
        if analysis['finance_tool_analysis']:
            logger.info(f"• Finance tools: {analysis['finance_tool_analysis']['num_finance_tools']} tools")
            logger.info(f"• Finance tools avg similarity: {analysis['finance_tool_analysis']['mean_finance_score']:.4f}")
            logger.info(f"• Non-finance tools avg similarity: {analysis['finance_tool_analysis']['mean_non_finance_score']:.4f}")
        
        # Show top 5 most matched ONET occupations
        top_occs = list(analysis['top_onet_occupations'].items())[:5]
        logger.info("• Top 5 most matched ONET occupations:")
        for occ, count in top_occs:
            logger.info(f"  - {occ}: {count} matches")
        
        logger.info("\nDetailed results saved to: conseq_fin_stage4_task_clusters_embed_match_results.json")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
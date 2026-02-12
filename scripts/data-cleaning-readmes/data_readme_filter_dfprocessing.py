#!/usr/bin/env python3
"""
README Filter Results Processing

Processes Inspect evaluation results from data_readme_filter_inspect.py and updates
the data_unified_filtered.json file with LLM-refined README content.

This script:
1. Loads Inspect .eval files from the logs directory
2. Extracts filtered README content from LLM responses
3. Updates the original dataset with refined content
4. Generates processing statistics and validation

Usage:
    python data_readme_filter_dfprocessing.py
    python data_readme_filter_dfprocessing.py --logs-dir ./logs
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_readme_filter_dfprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def find_latest_eval_file(logs_dir: str = "logs") -> Optional[str]:
    """Find the latest README filter evaluation file"""
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        logger.error(f"Logs directory {logs_dir} not found")
        return None
    
    # Look for eval files with readme_filter in the name
    eval_files = list(logs_path.glob("*readme-filter-task*.eval"))
    
    if not eval_files:
        logger.error(f"No README filter evaluation files found in {logs_dir}")
        return None
    
    # Sort by modification time and get the latest
    latest_file = max(eval_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Found latest evaluation file: {latest_file}")
    return str(latest_file)

def load_eval_results(logs_dir: str) -> List[Dict[str, Any]]:
    """Load and parse Inspect evaluation results using proper DataFrame processing"""
    try:
        # Find the latest evaluation file instead of loading all files
        latest_eval_file = find_latest_eval_file(logs_dir)
        if not latest_eval_file:
            logger.error("No evaluation file found")
            return []
        
        logger.info(f"Processing only the latest evaluation file: {latest_eval_file}")
        
        # Use Inspect's DataFrame processing functions on the specific file
        from inspect_ai.analysis import samples_df, messages_df
        
        samples_df_data = samples_df(latest_eval_file)
        messages_df_data = messages_df(latest_eval_file)
        logger.info(f"Loaded samples DataFrame with {len(samples_df_data)} samples")
        logger.info(f"Loaded messages DataFrame with {len(messages_df_data)} messages")
        
        # Optimize: Pre-filter and group messages by sample_id and role
        assistant_messages = messages_df_data[messages_df_data['role'] == 'assistant']
        user_messages = messages_df_data[messages_df_data['role'] == 'user']
        
        # Create lookup dictionaries for O(1) access
        assistant_lookup = assistant_messages.set_index('sample_id')['content'].to_dict()
        user_messages.set_index('sample_id')['content'].to_dict()
        
        # Process results using vectorized operations
        results = []
        
        for idx, sample_row in samples_df_data.iterrows():
            sample_id = sample_row.get("sample_id", f"sample_{idx}")
            
            sample_result = {
                "sample_id": sample_id,
                "input_data": {},
                "raw_output": assistant_lookup.get(sample_id, ""),
                "score": sample_row.get("score_readme_filter_scorer", 0),
                "metadata": {}
            }
            
            # Parse metadata only if it exists
            if hasattr(sample_row, 'metadata') and sample_row.metadata:
                try:
                    sample_result["metadata"] = json.loads(sample_row.metadata) if isinstance(sample_row.metadata, str) else sample_row.metadata
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
            
            # Extract server ID from sample metadata
            if hasattr(sample_row, 'id') and sample_row.id:
                sample_result["server_id"] = sample_row.id
            
            results.append(sample_result)
        
        logger.info(f"Processed {len(results)} evaluation results")
        return results
        
    except Exception as e:
        logger.error(f"Error loading evaluation results: {e}")
        raise

def extract_filtered_content(sample: Dict[str, Any]) -> Optional[str]:
    """Extract filtered README content from evaluation sample"""
    try:
        # Get the output from the sample
        output = sample.get('output', {})
        completion = output.get('completion', '')
        
        if not completion:
            logger.warning(f"No completion found for sample {sample.get('id', 'unknown')}")
            return None
        
        # Clean up the completion - remove any potential system messages or formatting
        filtered_content = completion.strip()
        
        # Remove any potential markdown code block wrappers
        if filtered_content.startswith('```'):
            # Remove opening code block
            lines = filtered_content.split('\n')
            if len(lines) > 1:
                filtered_content = '\n'.join(lines[1:])
        
        if filtered_content.endswith('```'):
            # Remove closing code block
            filtered_content = filtered_content[:-3].strip()
        
        return filtered_content
        
    except Exception as e:
        logger.error(f"Error extracting content from sample {sample.get('id', 'unknown')}: {e}")
        return None

def calculate_content_stats(original: str, stage1: str, stage2: str) -> Dict[str, Any]:
    """Calculate statistics comparing original, stage1, and stage2 content"""
    return {
        'original_length': len(original) if original else 0,
        'stage1_length': len(stage1) if stage1 else 0,
        'stage2_length': len(stage2) if stage2 else 0,
        'stage1_reduction_pct': ((len(original) - len(stage1)) / len(original) * 100) if original else 0,
        'stage2_reduction_pct': ((len(original) - len(stage2)) / len(original) * 100) if original else 0,
        'total_reduction_pct': ((len(original) - len(stage2)) / len(original) * 100) if original else 0,
        'stage1_to_stage2_change': len(stage2) - len(stage1) if stage1 else 0,
    }

def process_evaluation_results(eval_results: List[Dict[str, Any]], 
                             dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Process evaluation results and update dataset
    
    This function preserves the full dataset and only updates the readme_filtered
    field for servers that were processed in the LLM evaluation. All other servers
    remain unchanged in the dataset.
    """
    
    # Create lookup for quick access to servers by ID
    server_lookup = {str(server.get('id', '')): i for i, server in enumerate(dataset)}
    
    processing_stats = {
        'total_samples': len(eval_results),
        'successful_extractions': 0,
        'failed_extractions': 0,
        'servers_updated': 0,
        'average_reduction_pct': 0,
        'content_stats': []
    }
    
    total_reduction = 0
    content_stats = []
    
    # Batch process results to reduce overhead
    for result in eval_results:
        # Get server ID from result
        server_id = str(result.get('server_id', ''))
        if not server_id:
            # Try to get from metadata
            metadata = result.get('metadata', {})
            server_id = str(metadata.get('server_id', ''))
        
        # Extract structured content from LLM JSON response
        raw_output = result.get('raw_output', '')
        
        if not raw_output or not raw_output.strip():
            processing_stats['failed_extractions'] += 1
            continue
        
        # Parse JSON response from LLM
        try:
            # Clean up potential code block wrappers
            cleaned_output = raw_output.strip()
            if cleaned_output.startswith('```json'):
                cleaned_output = cleaned_output[7:]
            elif cleaned_output.startswith('```'):
                cleaned_output = cleaned_output[3:]
            
            if cleaned_output.endswith('```'):
                cleaned_output = cleaned_output[:-3]
            
            cleaned_output = cleaned_output.strip()
            
            # Parse JSON
            parsed_response = json.loads(cleaned_output)
            
            # Extract fields
            summary = parsed_response.get('summary', '')
            is_mcp_server = parsed_response.get('is_mcp_server', 1)  # Default to 1 if missing
            filtered_content = parsed_response.get('filtered_content', '')
            tools = parsed_response.get('tools', [])
            
            if not filtered_content and not summary:
                processing_stats['failed_extractions'] += 1
                continue
                
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse JSON response for server {server_id}: {e}")
            # Fallback to treating raw output as filtered content
            filtered_content = raw_output.strip()
            summary = ""
            is_mcp_server = 1  # Default to 1 for fallback
            tools = []
        
        processing_stats['successful_extractions'] += 1
        
        # Find corresponding server in dataset using O(1) lookup
        server_index = server_lookup.get(server_id)
        if server_index is not None:
            server = dataset[server_index]
            
            # Get original and stage1 content for comparison
            original_content = server.get('readme_content', '')
            stage1_content = server.get('readme_filteredinitial', '')
            
            # Calculate statistics
            stats = calculate_content_stats(original_content, stage1_content, filtered_content)
            content_stats.append(stats)
            
            total_reduction += stats['total_reduction_pct']
            
            # Update server with LLM-refined structured content
            dataset[server_index]['readme_filtered'] = filtered_content
            dataset[server_index]['readme_summary'] = summary
            dataset[server_index]['readme_is_mcp_server'] = is_mcp_server

            # Only add README-extracted tools if no existing tools
            existing_tools = dataset[server_index].get('tools', [])
            if not existing_tools and tools and isinstance(tools, list):
                dataset[server_index]['tools'] = tools
            processing_stats['servers_updated'] += 1
            
            if processing_stats['servers_updated'] % 100 == 0:
                logger.info(f"Updated {processing_stats['servers_updated']} servers")
        
        else:
            logger.warning(f"Server with ID {server_id} not found in dataset")
    
    # Set content stats in batch
    processing_stats['content_stats'] = content_stats
    
    # Calculate average reduction
    if processing_stats['successful_extractions'] > 0:
        processing_stats['average_reduction_pct'] = total_reduction / processing_stats['successful_extractions']
    
    logger.info(f"Processing complete: {processing_stats['servers_updated']} servers updated")
    logger.info(f"Average total reduction: {processing_stats['average_reduction_pct']:.1f}%")
    
    return processing_stats

def validate_filtering_results(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate the filtering results"""
    validation_stats = {
        'total_servers': len(dataset),
        'servers_with_original_readme': 0,
        'servers_with_filtered_readme': 0,
        'servers_with_summary': 0,
        'servers_with_tools': 0,
        'servers_classified_as_mcp': 0,
        'servers_classified_as_non_mcp': 0,
        'servers_with_both': 0,
        'total_tools_extracted': 0,
        'smithery_servers_with_tools': 0,
        'non_smithery_servers_with_tools': 0,
        'smithery_total_tools': 0,
        'non_smithery_total_tools': 0,
        'average_compression_ratio': 0,
        'common_remaining_patterns': [],
    }
    
    compression_ratios = []
    
    # Common patterns that should be removed (pre-compiled for performance)
    check_patterns = [
        'npm install', 'pip install', 'docker run', 'git clone',
        'yarn add', 'make install', 'sudo ', 'export PATH=',
        'cd ', 'mkdir ', 'chmod +x', 'source ', 'virtualenv'
    ]
    
    pattern_counts = {pattern: 0 for pattern in check_patterns}
    
    # Batch process servers to reduce overhead
    for server in dataset:
        original = server.get('readme_content', '')
        filtered = server.get('readme_filtered', '')
        summary = server.get('readme_summary', '')
        is_mcp_server = server.get('readme_is_mcp_server')
        tools = server.get('tools', [])
        
        if original and original.strip():
            validation_stats['servers_with_original_readme'] += 1
            
            if filtered and filtered.strip():
                validation_stats['servers_with_filtered_readme'] += 1
                validation_stats['servers_with_both'] += 1
                
                # Calculate compression ratio
                if len(original) > 0:
                    ratio = len(filtered) / len(original)
                    compression_ratios.append(ratio)
                
                # Check for remaining patterns (case-insensitive, single pass)
                filtered_lower = filtered.lower()
                for pattern in check_patterns:
                    if pattern.lower() in filtered_lower:
                        pattern_counts[pattern] += 1
        
        # Count new fields
        if summary and summary.strip():
            validation_stats['servers_with_summary'] += 1

        # Count MCP server classification
        if is_mcp_server == 1:
            validation_stats['servers_classified_as_mcp'] += 1
        elif is_mcp_server == 0:
            validation_stats['servers_classified_as_non_mcp'] += 1

        if tools and isinstance(tools, list) and len(tools) > 0:
            validation_stats['servers_with_tools'] += 1
            validation_stats['total_tools_extracted'] += len(tools)
            
            # Differentiate by source (Smithery vs non-Smithery)
            data_sources = server.get('data_sources', [])
            is_smithery = 'smithery' in data_sources
            
            if is_smithery:
                validation_stats['smithery_servers_with_tools'] += 1
                validation_stats['smithery_total_tools'] += len(tools)
            else:
                validation_stats['non_smithery_servers_with_tools'] += 1
                validation_stats['non_smithery_total_tools'] += len(tools)
    
    # Calculate average compression ratio
    if compression_ratios:
        validation_stats['average_compression_ratio'] = sum(compression_ratios) / len(compression_ratios)
    
    # Report patterns that still appear frequently
    validation_stats['common_remaining_patterns'] = [
        {"pattern": pattern, "count": count}
        for pattern, count in pattern_counts.items()
        if count > 0
    ]
    
    logger.info("Validation complete:")
    logger.info(f"  - {validation_stats['servers_with_both']} servers have both original and filtered content")
    logger.info(f"  - {validation_stats['servers_with_summary']} servers have summaries extracted")
    logger.info(f"  - {validation_stats['servers_classified_as_mcp']} servers classified as MCP servers (1)")
    logger.info(f"  - {validation_stats['servers_classified_as_non_mcp']} servers classified as non-MCP (0)")
    logger.info(f"  - {validation_stats['servers_with_tools']} servers have tools extracted")
    logger.info(f"    - Smithery servers: {validation_stats['smithery_servers_with_tools']} servers, {validation_stats['smithery_total_tools']} tools")
    logger.info(f"    - Non-Smithery servers: {validation_stats['non_smithery_servers_with_tools']} servers, {validation_stats['non_smithery_total_tools']} tools")
    logger.info(f"  - {validation_stats['total_tools_extracted']} total tools extracted")
    logger.info(f"  - Average compression ratio: {validation_stats['average_compression_ratio']:.2f}")
    logger.info(f"  - {len([p for p in validation_stats['common_remaining_patterns'] if p['count'] > 0])} patterns still present")
    
    return validation_stats

def main():
    parser = argparse.ArgumentParser(description='Process README filter evaluation results')
    parser.add_argument('--logs-dir', default='logs', help='Directory containing .eval files')
    args = parser.parse_args()
    
    # Check if logs directory exists
    if not Path(args.logs_dir).exists():
        logger.error(f"Logs directory {args.logs_dir} not found")
        return
    
    # Find evaluation file to verify we have the right task
    eval_file = find_latest_eval_file(args.logs_dir)
    if not eval_file:
        logger.error("No evaluation file found")
        return
    
    # Load evaluation results using DataFrame processing
    eval_results = load_eval_results(args.logs_dir)
    
    # Load original dataset (preserves full dataset regardless of evaluation sample size)
    dataset_file = 'data/initial/data_unified_filtered.json'
    if not Path(dataset_file).exists():
        logger.error(f"Dataset file {dataset_file} not found")
        return
    
    logger.info(f"Loading dataset from {dataset_file}")
    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    logger.info(f"Loaded {len(dataset)} servers from dataset (full dataset preserved)")
    
    # Process evaluation results
    processing_stats = process_evaluation_results(eval_results, dataset)
    
    # Validate results
    validation_stats = validate_filtering_results(dataset)
    
    # Save updated dataset
    with open(dataset_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Updated dataset saved to {dataset_file}")
    
    # Save processing summary
    summary = {
        "processing_timestamp": datetime.now().isoformat(),
        "stage": "llm_refinement",
        "eval_file": eval_file,
        "dataset_file": dataset_file,
        "processing_stats": processing_stats,
        "validation_stats": validation_stats
    }
    
    summary_file = 'data/initial/data_readme_filter_dfprocessing_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Processing summary saved to {summary_file}")
    logger.info("README filtering pipeline completed successfully!")

if __name__ == "__main__":
    main()
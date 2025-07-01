#!/usr/bin/env python3
"""
Extract Random Tools for Ground Truth Consequentiality Scoring

This script extracts 30 random tools from the filtered dataset to create
a ground truth dataset for consequentiality scoring validation.
"""

import json
import logging
import random
from typing import List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extract_random_tools.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_filtered_data() -> List[Dict]:
    """Load the filtered dataset"""
    logger.info("Loading data_unified_filtered.json...")
    
    try:
        with open('data_unified_filtered.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} servers from filtered dataset")
        return data
    
    except FileNotFoundError:
        logger.error("data_unified_filtered.json not found")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON: {e}")
        raise

def extract_all_tools(servers: List[Dict]) -> List[Dict]:
    """Extract all tools from all servers"""
    logger.info("Extracting tools from all servers...")
    
    all_tools = []
    servers_with_tools = 0
    
    for server in servers:
        server_id = server.get('id', 'unknown')
        server_name = server.get('name', server_id)
        
        extracted_tools = server.get('extracted_tools', [])
        
        if extracted_tools:
            servers_with_tools += 1
            
            for tool in extracted_tools:
                # Add server context to each tool
                tool_with_context = {
                    'tool_name': tool.get('name', ''),
                    'description': tool.get('description', ''),
                    'inputSchema': tool.get('inputSchema', {}),
                    'source': tool.get('source', ''),
                    'server_id': server_id,
                    'server_name': server_name
                }
                all_tools.append(tool_with_context)
    
    logger.info(f"Extracted {len(all_tools)} tools from {servers_with_tools} servers")
    return all_tools

def load_existing_ground_truth() -> List[Dict]:
    """Load existing ground truth tools to preserve existing scores"""
    logger.info("Loading existing ground truth tools...")
    
    try:
        with open('conseq_ground_truth_tools_sample.json', 'r', encoding='utf-8') as f:
            existing_tools = json.load(f)
        
        logger.info(f"Loaded {len(existing_tools)} existing ground truth tools")
        return existing_tools
    
    except FileNotFoundError:
        logger.info("No existing ground truth file found, starting fresh")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing existing ground truth JSON: {e}")
        return []

def get_smithery_tools(all_tools: List[Dict]) -> List[Dict]:
    """Filter tools from Smithery sources"""
    smithery_tools = [tool for tool in all_tools if tool.get('source') == 'smithery']
    logger.info(f"Found {len(smithery_tools)} Smithery tools")
    return smithery_tools

def get_execution_tools(all_tools: List[Dict]) -> List[Dict]:
    """Filter tools that likely have execution capabilities based on name/description"""
    execution_keywords = [
        'execute', 'run', 'create', 'delete', 'update', 'send', 'post', 'put', 
        'patch', 'write', 'save', 'deploy', 'install', 'remove', 'start', 'stop',
        'kill', 'restart', 'modify', 'change', 'set', 'configure', 'submit',
        'transfer', 'move', 'copy', 'upload', 'download', 'publish', 'commit',
        'push', 'pull', 'merge', 'branch', 'checkout', 'reset', 'rebase'
    ]
    
    execution_tools = []
    for tool in all_tools:
        tool_name = tool.get('tool_name', '').lower()
        description = tool.get('description', '').lower()
        
        if any(keyword in tool_name or keyword in description for keyword in execution_keywords):
            execution_tools.append(tool)
    
    logger.info(f"Found {len(execution_tools)} tools with execution capabilities")
    return execution_tools

def sample_additional_tools(all_tools: List[Dict], existing_tools: List[Dict], n: int = 20) -> List[Dict]:
    """Sample n additional tools, prioritizing Smithery and execution tools"""
    logger.info(f"Sampling {n} additional tools from {len(all_tools)} total tools...")
    
    # Get existing tool identifiers to avoid duplicates
    existing_tool_ids = set()
    for tool in existing_tools:
        tool_id = f"{tool.get('server_id', '')}:{tool.get('tool_name', '')}"
        existing_tool_ids.add(tool_id)
    
    # Filter out tools that already exist
    available_tools = []
    for tool in all_tools:
        tool_id = f"{tool.get('server_id', '')}:{tool.get('tool_name', '')}"
        if tool_id not in existing_tool_ids:
            available_tools.append(tool)
    
    logger.info(f"Found {len(available_tools)} available tools (excluding existing ones)")
    
    if len(available_tools) < n:
        logger.warning(f"Only {len(available_tools)} tools available, returning all")
        return available_tools
    
    # Prioritize Smithery tools
    smithery_tools = get_smithery_tools(available_tools)
    execution_tools = get_execution_tools(available_tools)
    
    # Combine and deduplicate priority tools
    priority_tools = []
    priority_tool_ids = set()
    
    for tool in smithery_tools + execution_tools:
        tool_id = f"{tool.get('server_id', '')}:{tool.get('tool_name', '')}"
        if tool_id not in priority_tool_ids:
            priority_tools.append(tool)
            priority_tool_ids.add(tool_id)
    
    logger.info(f"Found {len(priority_tools)} priority tools (Smithery + execution)")
    
    # Set random seed for reproducibility
    random.seed(43)  # Different seed from original to get different tools
    
    selected_tools = []
    
    # First, select from priority tools
    priority_sample_size = min(n // 2, len(priority_tools))
    if priority_tools:
        selected_tools.extend(random.sample(priority_tools, priority_sample_size))
    
    # Then fill remaining slots with random tools
    remaining_slots = n - len(selected_tools)
    if remaining_slots > 0:
        # Get tools not already selected
        remaining_tools = [tool for tool in available_tools 
                         if f"{tool.get('server_id', '')}:{tool.get('tool_name', '')}" 
                         not in [f"{t.get('server_id', '')}:{t.get('tool_name', '')}" for t in selected_tools]]
        
        if remaining_tools:
            additional_sample_size = min(remaining_slots, len(remaining_tools))
            selected_tools.extend(random.sample(remaining_tools, additional_sample_size))
    
    logger.info(f"Selected {len(selected_tools)} additional tools")
    return selected_tools

def sample_random_tools(all_tools: List[Dict], n: int = 30) -> List[Dict]:
    """Sample n random tools from the complete tool list"""
    logger.info(f"Sampling {n} random tools from {len(all_tools)} total tools...")
    
    if len(all_tools) < n:
        logger.warning(f"Only {len(all_tools)} tools available, returning all")
        return all_tools
    
    # Set random seed for reproducibility
    random.seed(42)
    
    sampled_tools = random.sample(all_tools, n)
    logger.info(f"Selected {len(sampled_tools)} random tools")
    
    return sampled_tools

def format_schema_summary(schema: Dict) -> str:
    """Format inputSchema into a readable summary"""
    if not schema or not isinstance(schema, dict):
        return "No schema"
    
    properties = schema.get('properties', {})
    required = schema.get('required', [])
    
    if not properties:
        return "No parameters"
    
    param_strs = []
    for param, details in properties.items():
        param_type = details.get('type', 'unknown')
        is_required = param in required
        req_str = " (required)" if is_required else ""
        param_strs.append(f"{param}: {param_type}{req_str}")
    
    return ", ".join(param_strs)

def display_tools(tools: List[Dict]):
    """Display tools in a format suitable for manual scoring"""
    logger.info("Displaying selected tools for manual consequentiality scoring:")
    
    print("\n" + "="*80)
    print("RANDOM TOOLS FOR GROUND TRUTH CONSEQUENTIALITY SCORING")
    print("="*80)
    print("Please score each tool as: HIGH, MEDIUM, or LOW consequentiality")
    print("="*80 + "\n")
    
    for i, tool in enumerate(tools, 1):
        tool_name = tool['tool_name']
        description = tool['description']
        schema_summary = format_schema_summary(tool['inputSchema'])
        server_name = tool['server_name']
        
        print(f"Tool #{i}: {tool_name}")
        print(f"Description: {description}")
        print(f"Parameters: {schema_summary}")
        print(f"Source Server: {server_name}")
        print("-" * 80)
        print()

def add_score_me_field(tools: List[Dict]) -> List[Dict]:
    """Add score_me field to tools that don't have it"""
    for tool in tools:
        if 'score_me' not in tool:
            tool['score_me'] = 'unclear'  # Default value for new tools
    return tools

def save_tools_json(tools: List[Dict], filename: str = 'conseq_ground_truth_tools_sample.json'):
    """Save sampled tools to JSON file for further processing"""
    logger.info(f"Saving tools to {filename}...")
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(tools, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(tools)} tools to {filename}")

def main():
    """Main function"""
    try:
        # Load existing ground truth tools to preserve scores
        existing_tools = load_existing_ground_truth()
        
        # Load filtered data
        servers = load_filtered_data()
        
        # Extract all tools
        all_tools = extract_all_tools(servers)
        
        if not all_tools:
            logger.error("No tools found in the filtered dataset")
            return
        
        if existing_tools:
            # Add 20 more tools to existing ones
            additional_tools = sample_additional_tools(all_tools, existing_tools, 20)
            
            # Add score_me field to new tools
            additional_tools = add_score_me_field(additional_tools)
            
            # Combine existing and new tools
            all_ground_truth_tools = existing_tools + additional_tools
            
            logger.info(f"Combined {len(existing_tools)} existing tools with {len(additional_tools)} new tools")
            logger.info(f"Total ground truth tools: {len(all_ground_truth_tools)}")
            
        else:
            # No existing tools, sample 30 random tools (original behavior)
            sampled_tools = sample_random_tools(all_tools, 30)
            sampled_tools = add_score_me_field(sampled_tools)
            all_ground_truth_tools = sampled_tools
            
            logger.info(f"Created initial ground truth with {len(all_ground_truth_tools)} tools")
        
        # Display tools for manual scoring (only new ones if existing tools present)
        if existing_tools:
            logger.info("Displaying only NEW tools for manual scoring:")
            display_tools(additional_tools)
        else:
            display_tools(all_ground_truth_tools)
        
        # Save complete set to JSON
        save_tools_json(all_ground_truth_tools)
        
        logger.info("Tool extraction completed successfully")
        
    except Exception as e:
        logger.error(f"Error during tool extraction: {e}")
        raise

if __name__ == "__main__":
    main()
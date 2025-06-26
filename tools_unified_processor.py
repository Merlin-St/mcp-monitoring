#!/usr/bin/env python3
"""
Tools Unified Processor - Utility Functions

Extracts and classifies individual tools from MCP servers by access level:
- read: Information gathering, fetching data  
- write: Data modification, storage operations
- execute: Action execution, external system interaction

This module provides utility functions for tool extraction and classification.
"""

import json
import logging
import re
from typing import Dict, List, Optional
import argparse

logger = logging.getLogger(__name__)

# Utility functions for tool extraction and classification

def extract_tools_from_smithery(server: Dict) -> List[Dict]:
    """Extract tools from Smithery data"""
    tools = []
    
    # Check if server has Smithery tools data
    if server.get('tools') and isinstance(server['tools'], list):
        for tool in server['tools']:
            if isinstance(tool, dict) and tool.get('name'):
                tools.append({
                    'name': tool.get('name'),
                    'description': tool.get('description', ''),
                    'inputSchema': tool.get('inputSchema', {}),
                    'source': 'smithery',
                    'access_level': classify_tool_access_level(
                        tool.get('name', ''), 
                        tool.get('description', '')
                    )
                })
    
    return tools

def extract_tools_from_readme(server: Dict) -> List[Dict]:
    """Extract tools from README content using pattern matching"""
    tools = []
    readme = server.get('readme_content', '')
    
    if not readme or len(readme) < 50:
        return tools
    
    # Patterns to match tool definitions
    patterns = [
        r'##\s+Tools?\s*\n(.*?)(?=\n##|\n#|\Z)',  # ## Tools section
        r'###\s+Available\s+Tools?\s*\n(.*?)(?=\n##|\n#|\Z)',  # ### Available Tools
        r'`([a-zA-Z_][a-zA-Z0-9_]*)`[:\s]*([^\n]*)',  # `tool_name`: description
        r'-\s+`([a-zA-Z_][a-zA-Z0-9_]*)`[:\s]*([^\n]*)',  # - `tool_name`: description
        r'\*\s+`([a-zA-Z_][a-zA-Z0-9_]*)`[:\s]*([^\n]*)',  # * `tool_name`: description
        r'(\w+)\(\)\s*[:-]\s*([^\n]*)',  # function_name(): description
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, readme, re.MULTILINE | re.DOTALL | re.IGNORECASE)
        for match in matches:
            if len(match.groups()) >= 2:
                name = match.group(1).strip()
                desc = match.group(2).strip() if len(match.groups()) > 1 else ''
            else:
                # Handle section matches
                section = match.group(1)
                # Extract individual tools from section
                tool_matches = re.finditer(r'`([a-zA-Z_][a-zA-Z0-9_]*)`[:\s]*([^\n]*)', section)
                for tool_match in tool_matches:
                    name = tool_match.group(1).strip()
                    desc = tool_match.group(2).strip()
                    if name and len(name) > 1:
                        tools.append({
                            'name': name,
                            'description': desc,
                            'inputSchema': generate_schema_from_description(name, desc),
                            'source': 'readme',
                            'access_level': classify_tool_access_level(name, desc)
                        })
                continue
            
            if name and len(name) > 1:
                tools.append({
                    'name': name,
                    'description': desc,
                    'inputSchema': generate_schema_from_description(name, desc),
                    'source': 'readme',
                    'access_level': classify_tool_access_level(name, desc)
                })
    
    return tools

def extract_tools_from_html(server: Dict) -> List[Dict]:
    """Extract tools from HTML content"""
    tools = []
    html = server.get('html_content', '')
    
    if not html or len(html) < 100:
        return tools
    
    # Basic HTML tool extraction patterns
    patterns = [
        r'<code>([a-zA-Z_][a-zA-Z0-9_]*)</code>[:\s]*([^<\n]*)',
        r'<strong>([a-zA-Z_][a-zA-Z0-9_]*)</strong>[:\s]*([^<\n]*)',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, html, re.IGNORECASE)
        for match in matches:
            name = match.group(1).strip()
            desc = match.group(2).strip()
            if name and len(name) > 1:
                tools.append({
                    'name': name,
                    'description': desc,
                    'inputSchema': generate_schema_from_description(name, desc),
                    'source': 'html',
                    'access_level': classify_tool_access_level(name, desc)
                })
    
    return tools

def generate_schema_from_description(name: str, description: str) -> Dict:
    """Generate a basic inputSchema from tool name and description"""
    schema = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    # Basic schema generation based on common patterns
    text = f"{name} {description}".lower()
    
    # Common parameter patterns
    if any(word in text for word in ['file', 'path', 'url', 'link']):
        if 'file' in text or 'path' in text:
            schema["properties"]["file_path"] = {
                "type": "string",
                "description": "File path or file to process"
            }
            schema["required"].append("file_path")
        if 'url' in text or 'link' in text:
            schema["properties"]["url"] = {
                "type": "string",
                "description": "URL or link to process"
            }
            schema["required"].append("url")
    
    if any(word in text for word in ['query', 'search', 'find']):
        schema["properties"]["query"] = {
            "type": "string", 
            "description": "Search query or term"
        }
        schema["required"].append("query")
    
    if any(word in text for word in ['message', 'text', 'content', 'input']):
        schema["properties"]["message"] = {
            "type": "string",
            "description": "Text content or message"
        }
        schema["required"].append("message")
    
    if any(word in text for word in ['amount', 'price', 'cost', 'value']):
        schema["properties"]["amount"] = {
            "type": "number",
            "description": "Numeric amount or value"
        }
        schema["required"].append("amount")
    
    # If no specific patterns found, add a generic parameter
    if not schema["properties"]:
        schema["properties"]["input"] = {
            "type": "string",
            "description": f"Input parameter for {name}"
        }
        schema["required"].append("input")
    
    return schema

def classify_tool_access_level(name: str, description: str) -> str:
    """Classify tool by access level: read, write, execute"""
    text = f"{name} {description}".lower()
    
    # Execute keywords (highest priority)
    execute_keywords = [
        'send', 'execute', 'run', 'launch', 'kill', 'restart',
        'deploy', 'install', 'uninstall', 'delete', 'download',
        'buy', 'sell', 'trade', 'payment', 'transfer_money'
    ]
    
    # Write keywords (medium priority)
    write_keywords = [
        'write', 'save', 'store', 'update', 'modify', 'edit', 'change',
        'set', 'insert', 'add', 'append', 'create', 'generate', 'build',
        'make', 'produce', 'output', 'export', 'backup', 'sync'
    ]
    
    # Read keywords (lowest priority - default)
    read_keywords = [
        'read', 'get', 'fetch', 'retrieve', 'load', 'query', 'search',
        'find', 'list', 'show', 'display', 'view', 'check', 'verify',
        'analyze', 'parse', 'scan', 'monitor', 'watch', 'observe'
    ]
    
    # Check in order of risk/impact
    if any(keyword in text for keyword in execute_keywords):
        return 'execute'
    elif any(keyword in text for keyword in write_keywords):
        return 'write'
    else:
        return 'read'  # Default to read

class ToolsProcessor:
    """Main processor class for extracting tools from MCP servers"""
    
    def __init__(self):
        self.servers_data = []
        self.processed_servers = []
        self.tool_stats = {
            'servers_with_tools': 0,
            'total_tools': 0,
            'tools_by_source': {'smithery': 0, 'readme': 0, 'html': 0},
            'tools_by_access_level': {'read': 0, 'write': 0, 'execute': 0}
        }
    
    def load_data(self, input_file: str = 'data_unified.json') -> bool:
        """Load the unified MCP server data"""
        logger.info(f"Loading data from {input_file}")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                self.servers_data = json.load(f)
            logger.info(f"Loaded {len(self.servers_data)} servers")
            
            # Check if servers already have extracted_tools
            has_extracted = sum(1 for s in self.servers_data[:100] if s.get('extracted_tools'))
            logger.info(f"Servers already with extracted_tools (sample of 100): {has_extracted}")
            
            # Check existing tools field structure
            has_smithery_tools = sum(1 for s in self.servers_data[:100] if s.get('tools'))
            logger.info(f"Servers with existing 'tools' field (sample of 100): {has_smithery_tools}")
            
            return True
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def process_server(self, server: Dict) -> Dict:
        """Process a single server to extract and classify tools"""
        server_tools = []
        
        # Extract from different sources
        server_tools.extend(extract_tools_from_smithery(server))
        server_tools.extend(extract_tools_from_readme(server))
        server_tools.extend(extract_tools_from_html(server))
        
        # Deduplicate tools by name (keep first occurrence)
        seen_tools = set()
        unique_tools = []
        for tool in server_tools:
            tool_key = tool['name'].lower()
            if tool_key not in seen_tools:
                seen_tools.add(tool_key)
                unique_tools.append(tool)
        
        # Update statistics
        if unique_tools:
            self.tool_stats['servers_with_tools'] += 1
        
        self.tool_stats['total_tools'] += len(unique_tools)
        
        for tool in unique_tools:
            self.tool_stats['tools_by_source'][tool['source']] += 1
            self.tool_stats['tools_by_access_level'][tool['access_level']] += 1
        
        # Add tools to server data
        enhanced_server = server.copy()
        enhanced_server['extracted_tools'] = unique_tools
        enhanced_server['tools_count'] = len(unique_tools)
        enhanced_server['tools_by_access'] = {
            'read': len([t for t in unique_tools if t['access_level'] == 'read']),
            'write': len([t for t in unique_tools if t['access_level'] == 'write']),
            'execute': len([t for t in unique_tools if t['access_level'] == 'execute'])
        }
        
        return enhanced_server
    
    def process_all_servers(self, limit: Optional[int] = None, test_mode: bool = False):
        """Process all servers to extract tools"""
        servers_to_process = self.servers_data
        
        if test_mode:
            servers_to_process = servers_to_process[:100]
            logger.info("Running in test mode - processing first 100 servers")
        
        if limit:
            servers_to_process = servers_to_process[:limit]
        
        total_servers = len(servers_to_process)
        logger.info(f"Processing {total_servers} servers")
        
        for i, server in enumerate(servers_to_process):
            if i % 1000 == 0:
                logger.info(f"Processed {i}/{total_servers} servers")
            
            try:
                processed_server = self.process_server(server)
                self.processed_servers.append(processed_server)
            except Exception as e:
                logger.error(f"Error processing server {server.get('id', 'unknown')}: {e}")
                continue
    
    def save_results(self, output_file: str):
        """Save processed results to JSON file"""
        logger.info(f"Saving processed data to {output_file}")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.processed_servers, f, indent=2, ensure_ascii=False)
            
            logger.info("Processing complete! Statistics:")
            logger.info(f"  Total servers: {len(self.processed_servers)}")
            logger.info(f"  Servers with tools: {self.tool_stats['servers_with_tools']} ({self.tool_stats['servers_with_tools']/len(self.processed_servers)*100:.1f}%)")
            logger.info(f"  Total tools extracted: {self.tool_stats['total_tools']}")
            logger.info(f"  Tools by source: {self.tool_stats['tools_by_source']}")
            logger.info(f"  Tools by access level: {self.tool_stats['tools_by_access_level']}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Extract and classify tools from MCP servers')
    parser.add_argument('--test', action='store_true', help='Run in test mode (first 100 servers)')
    parser.add_argument('--limit', type=int, help='Limit number of servers to process')
    parser.add_argument('--output', help='Output file name', default='data_unified.json')
    
    args = parser.parse_args()
    
    processor = ToolsProcessor()
    
    # Load data
    if not processor.load_data():
        return 1
    
    # Process servers
    processor.process_all_servers(limit=args.limit, test_mode=args.test)
    
    # Generate output filename
    if args.test:
        output_file = 'data_unified_with_tools_test.json'
    elif args.output != 'data_unified.json':
        output_file = args.output
    else:
        output_file = 'data_unified.json'  # Update original by default
    
    # Save results
    processor.save_results(output_file)
    
    return 0

if __name__ == "__main__":
    exit(main())
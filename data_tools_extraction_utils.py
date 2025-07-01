#!/usr/bin/env python3
"""
Tools Extraction Utilities

Utility functions for extracting and classifying tools from MCP servers.
Used by data_unified_mcp_data_processor.py for tool processing.
"""

import re
from typing import Dict, List

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
    """Extract tools from README content using optimized pattern matching"""
    tools = []
    readme = server.get('readme_content', '')
    
    if not readme or len(readme) < 50:
        return tools
    
    # Pre-compiled patterns for better performance
    if not hasattr(extract_tools_from_readme, '_compiled_patterns'):
        extract_tools_from_readme._compiled_patterns = [
            re.compile(r'##\s+Tools?\s*\n(.*?)(?=\n##|\n#|\Z)', re.MULTILINE | re.DOTALL | re.IGNORECASE),
            re.compile(r'###\s+Available\s+Tools?\s*\n(.*?)(?=\n##|\n#|\Z)', re.MULTILINE | re.DOTALL | re.IGNORECASE),
            re.compile(r'`([a-zA-Z_][a-zA-Z0-9_]*)`[:\s]*([^\n]*)', re.MULTILINE | re.IGNORECASE),
            re.compile(r'-\s+`([a-zA-Z_][a-zA-Z0-9_]*)`[:\s]*([^\n]*)', re.MULTILINE | re.IGNORECASE),
            re.compile(r'\*\s+`([a-zA-Z_][a-zA-Z0-9_]*)`[:\s]*([^\n]*)', re.MULTILINE | re.IGNORECASE),
            re.compile(r'(\w+)\(\)\s*[:-]\s*([^\n]*)', re.MULTILINE | re.IGNORECASE),
        ]
        extract_tools_from_readme._section_pattern = re.compile(r'`([a-zA-Z_][a-zA-Z0-9_]*)`[:\s]*([^\n]*)')
    
    patterns = extract_tools_from_readme._compiled_patterns
    section_pattern = extract_tools_from_readme._section_pattern
    
    for pattern in patterns:
        matches = pattern.finditer(readme)
        for match in matches:
            if len(match.groups()) >= 2:
                name = match.group(1).strip()
                desc = match.group(2).strip() if len(match.groups()) > 1 else ''
            else:
                # Handle section matches - process in batch
                section = match.group(1)
                tool_matches = section_pattern.finditer(section)
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
    """Extract tools from HTML content using optimized patterns"""
    tools = []
    html = server.get('html_content', '')
    
    if not html or len(html) < 100:
        return tools
    
    # Pre-compiled patterns for better performance
    if not hasattr(extract_tools_from_html, '_compiled_patterns'):
        extract_tools_from_html._compiled_patterns = [
            re.compile(r'<code>([a-zA-Z_][a-zA-Z0-9_]*)</code>[:\s]*([^<\n]*)', re.IGNORECASE),
            re.compile(r'<strong>([a-zA-Z_][a-zA-Z0-9_]*)</strong>[:\s]*([^<\n]*)', re.IGNORECASE),
        ]
    
    patterns = extract_tools_from_html._compiled_patterns
    
    for pattern in patterns:
        matches = pattern.finditer(html)
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
    
    # Check in order of risk/impact
    if any(keyword in text for keyword in execute_keywords):
        return 'execute'
    elif any(keyword in text for keyword in write_keywords):
        return 'write'
    else:
        return 'read'  # Default to read

def extract_and_classify_tools(server: Dict) -> List[Dict]:
    """Extract and classify all tools from a server"""
    all_tools = []
    
    # Extract from different sources
    all_tools.extend(extract_tools_from_smithery(server))
    all_tools.extend(extract_tools_from_readme(server))
    all_tools.extend(extract_tools_from_html(server))
    
    # Deduplicate tools by name (keep first occurrence)
    seen_tools = set()
    unique_tools = []
    for tool in all_tools:
        tool_key = tool['name'].lower()
        if tool_key not in seen_tools:
            seen_tools.add(tool_key)
            unique_tools.append(tool)
    
    return unique_tools
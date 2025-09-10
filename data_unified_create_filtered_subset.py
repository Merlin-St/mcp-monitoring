#!/usr/bin/env python3
"""
Create a filtered subset of data_unified.json based on quality criteria:
- If GitHub is the only source: only include repos with stargazers_count >= 1
- If Smithery is the only source: only include repos with use_count >= 1
- Include all repos with multiple sources (they're likely higher quality)
- Apply README content filtering to remove installation tips while preserving functional descriptions
"""

import json
import logging
import re
import argparse
from typing import Dict, List, Any
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_unified_create_filtered_subset.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Installation patterns to remove
INSTALLATION_PATTERNS = [
    # Package managers
    r'npm install.*',
    r'yarn add.*',
    r'pip install.*',
    r'pip3 install.*',
    r'composer install.*',
    r'gem install.*',
    r'cargo install.*',
    r'go get.*',
    r'docker run.*',
    r'docker build.*',
    r'docker-compose.*',
    
    # Git operations
    r'git clone.*',
    r'git pull.*',
    r'git checkout.*',
    
    # Environment setup
    r'export [A-Z_]+=[^\\n]*',
    r'set [A-Z_]+=[^\\n]*',
    r'source.*venv.*',
    r'source.*activate.*',
    r'conda activate.*',
    r'virtualenv.*',
    r'python -m venv.*',
    
    # Build commands
    r'make install.*',
    r'make build.*',
    r'./configure.*',
    r'cmake.*',
    r'mvn install.*',
    r'gradle build.*',
    
    # Development setup
    r'cd [^\\n]*',
    r'mkdir [^\\n]*',
    r'chmod [^\\n]*',
    r'sudo [^\\n]*',
]

# Section headers to remove entirely
INSTALLATION_SECTIONS = [
    'installation',
    'setup',
    'getting started',
    'quick start',
    'prerequisites',
    'requirements',
    'configuration',
    'environment setup',
    'development setup',
    'building',
    'compiling',
    'deploying',
    'running',
    'starting',
    'launch',
    'usage examples',  # Often contains setup commands
    'quickstart',
    'how to use',
    'first steps',
    'initial setup',
]

# Patterns for code blocks that are likely installation-related
CODE_BLOCK_PATTERNS = [
    r'```(?:bash|sh|shell|zsh|fish|powershell|cmd|terminal).*?```',
    r'```.*?(?:npm|yarn|pip|docker|git|make|cargo|go get).*?```',
    r'`[^`]*(?:npm|yarn|pip|docker|git|make|cargo|go get)[^`]*`',
]

def remove_installation_sections(content: str) -> str:
    """Remove entire sections that are installation-related"""
    if not content:
        return content
    
    lines = content.split('\n')
    filtered_lines = []
    skip_section = False
    section_level = 0
    
    for line in lines:
        # Check if this is a header line
        header_match = re.match(r'^(#{1,6})\s*(.*)', line)
        if header_match:
            current_level = len(header_match.group(1))
            header_text = header_match.group(2).lower().strip()
            
            # Check if this header indicates an installation section
            if any(section in header_text for section in INSTALLATION_SECTIONS):
                skip_section = True
                section_level = current_level
                continue
            
            # If we're skipping and we hit a header at same or higher level, stop skipping
            if skip_section and current_level <= section_level:
                skip_section = False
                section_level = 0
        
        # Add line if we're not skipping
        if not skip_section:
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)

def remove_installation_patterns(content: str) -> str:
    """Remove lines matching installation patterns"""
    if not content:
        return content
    
    lines = content.split('\n')
    filtered_lines = []
    
    for line in lines:
        # Check if line matches any installation pattern
        line_matches = False
        for pattern in INSTALLATION_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                line_matches = True
                break
        
        if not line_matches:
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)

def remove_installation_code_blocks(content: str) -> str:
    """Remove code blocks that contain installation commands"""
    if not content:
        return content
    
    # Remove code blocks with installation patterns
    for pattern in CODE_BLOCK_PATTERNS:
        content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
    
    return content

def clean_whitespace(content: str) -> str:
    """Clean up excessive whitespace and empty lines"""
    if not content:
        return content
    
    # Replace multiple consecutive newlines with max 2
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Remove trailing whitespace from lines
    lines = [line.rstrip() for line in content.split('\n')]
    
    # Remove empty lines at start and end
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    
    return '\n'.join(lines)

def filter_readme_content_stage1(content: str) -> str:
    """
    Stage 1: Keyword-based filtering to remove installation content
    """
    if not content or not content.strip():
        return content
    
    # Step 1: Remove installation sections
    content = remove_installation_sections(content)
    
    # Step 2: Remove installation patterns
    content = remove_installation_patterns(content)
    
    # Step 3: Remove installation code blocks
    content = remove_installation_code_blocks(content)
    
    # Step 4: Clean up whitespace
    content = clean_whitespace(content)
    
    return content

def calculate_filtering_stats(original: str, filtered: str) -> Dict[str, Any]:
    """Calculate statistics about the filtering process"""
    orig_lines = len(original.split('\n')) if original else 0
    filt_lines = len(filtered.split('\n')) if filtered else 0
    
    orig_chars = len(original) if original else 0
    filt_chars = len(filtered) if filtered else 0
    
    return {
        'original_lines': orig_lines,
        'filtered_lines': filt_lines,
        'lines_removed': orig_lines - filt_lines,
        'lines_reduction_pct': ((orig_lines - filt_lines) / orig_lines * 100) if orig_lines > 0 else 0,
        'original_chars': orig_chars,
        'filtered_chars': filt_chars,
        'chars_removed': orig_chars - filt_chars,
        'chars_reduction_pct': ((orig_chars - filt_chars) / orig_chars * 100) if orig_chars > 0 else 0,
    }

def apply_readme_filtering(data: List[Dict[str, Any]], apply_filtering: bool = True) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply README filtering to servers if requested"""
    if not apply_filtering:
        # Just add empty readme_filteredinitial field
        for server in data:
            server['readme_filteredinitial'] = ''
        return data, {}
    
    logger.info("Applying README content filtering...")
    
    filtering_stats = {
        'total_servers': len(data),
        'servers_with_readme': 0,
        'servers_processed': 0,
        'total_lines_removed': 0,
        'total_chars_removed': 0,
        'avg_reduction_pct': 0,
    }
    
    for i, server in enumerate(data):
        if i % 100 == 0 and i > 0:
            logger.info(f"Processing README filtering {i+1}/{len(data)}")
        
        # Get original readme content
        original_readme = server.get('readme_content', '')
        
        if original_readme and original_readme.strip():
            filtering_stats['servers_with_readme'] += 1
            
            # Apply Stage 1 filtering
            filtered_readme = filter_readme_content_stage1(original_readme)
            
            # Calculate stats
            stats = calculate_filtering_stats(original_readme, filtered_readme)
            
            # Update totals
            filtering_stats['total_lines_removed'] += stats['lines_removed']
            filtering_stats['total_chars_removed'] += stats['chars_removed']
            filtering_stats['avg_reduction_pct'] += stats['chars_reduction_pct']
            
            # Set filtered content
            server['readme_filteredinitial'] = filtered_readme
            filtering_stats['servers_processed'] += 1
        else:
            # No readme content, set empty filtered content
            server['readme_filteredinitial'] = ''
    
    # Calculate average reduction
    if filtering_stats['servers_processed'] > 0:
        filtering_stats['avg_reduction_pct'] /= filtering_stats['servers_processed']
    
    logger.info(f"README filtering complete: {filtering_stats['servers_processed']} servers processed")
    logger.info(f"Average content reduction: {filtering_stats['avg_reduction_pct']:.1f}%")
    logger.info(f"Total lines removed: {filtering_stats['total_lines_removed']}")
    logger.info(f"Total chars removed: {filtering_stats['total_chars_removed']}")
    
    return data, filtering_stats

def create_filtered_subset(enable_readme_filtering: bool = True):
    """Create filtered subset based on quality criteria"""
    
    logger.info("Loading unified dashboard data...")
    with open('data_unified.json', 'r') as f:
        data = json.load(f)
    
    logger.info(f"Original dataset: {len(data)} servers")
    
    filtered_data = []
    stats = {
        'github_only_included': 0,
        'github_only_excluded': 0,
        'smithery_only_included': 0,
        'smithery_only_excluded': 0,
        'multi_source_included': 0,
        'other_cases': 0
    }
    
    for server in data:
        sources = server.get('data_sources', [])
        
        # Case 1: GitHub only - require stargazers >= 1
        if sources == ['github']:
            stargazers = server.get('stargazers_count', 0)
            if stargazers >= 1:
                filtered_data.append(server)
                stats['github_only_included'] += 1
            else:
                stats['github_only_excluded'] += 1
                
        # Case 2: Smithery only - include all servers from Smithery source
        elif sources == ['smithery']:
            filtered_data.append(server)
            stats['smithery_only_included'] += 1
                
        # Case 3: Multiple sources - include all (assume higher quality)
        elif len(sources) > 1:
            filtered_data.append(server)
            stats['multi_source_included'] += 1
            
        # Case 4: Other cases (official only, etc.)
        else:
            filtered_data.append(server)
            stats['other_cases'] += 1
    
    logger.info(f"Filtered dataset: {len(filtered_data)} servers")
    logger.info("Filtering statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # Calculate retention rate
    retention_rate = len(filtered_data) / len(data) * 100
    logger.info(f"Retention rate: {retention_rate:.1f}%")
    
    # Apply README filtering if requested
    filtered_data, filtering_stats = apply_readme_filtering(filtered_data, enable_readme_filtering)
    
    # Save filtered dataset
    output_file = 'data_unified_filtered.json'
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    logger.info(f"Filtered dataset saved to {output_file}")
    
    # Create summary of filtered data
    finance_count = len([s for s in filtered_data if s.get('is_finance_related', False)])
    source_counts = {}
    primary_source_counts = {}
    
    for server in filtered_data:
        for source in server.get('data_sources', []):
            source_counts[source] = source_counts.get(source, 0) + 1
        
        primary = server.get('primary_source')
        if primary:
            primary_source_counts[primary] = primary_source_counts.get(primary, 0) + 1
    
    summary = {
        'total_servers': len(filtered_data),
        'finance_related_servers': finance_count,
        'retention_rate_percent': round(retention_rate, 1),
        'filtering_statistics': stats,
        'source_coverage': source_counts,
        'primary_source_distribution': primary_source_counts,
        'readme_filtering_applied': enable_readme_filtering,
        'readme_filtering_stats': filtering_stats if enable_readme_filtering else {},
        'processing_timestamp': datetime.now().isoformat()
    }
    
    summary_file = 'data_unified_filtered_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved to {summary_file}")
    
    # Show some examples of included servers
    logger.info("Sample of included servers:")
    for i, server in enumerate(filtered_data[:5]):
        logger.info(f"  {i+1}. {server.get('name', 'N/A')} ({server.get('data_sources', [])})")
        if server.get('data_sources') == ['github']:
            logger.info(f"     Stars: {server.get('stargazers_count', 0)}")
        elif server.get('data_sources') == ['smithery']:
            logger.info(f"     Use count: {server.get('use_count', 0)}")

def main():
    parser = argparse.ArgumentParser(description='Create filtered subset of MCP servers with optional README filtering')
    parser.add_argument('--no-readme-filtering', action='store_true', 
                       help='Skip README content filtering (default: apply filtering)')
    args = parser.parse_args()
    
    # Apply README filtering by default, unless explicitly disabled
    apply_filtering = not args.no_readme_filtering
    
    if apply_filtering:
        logger.info("README content filtering will be applied")
    else:
        logger.info("README content filtering will be skipped")
    
    create_filtered_subset(enable_readme_filtering=apply_filtering)

if __name__ == "__main__":
    main()
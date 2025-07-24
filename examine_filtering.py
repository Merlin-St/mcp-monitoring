#!/usr/bin/env python3
"""
Examine specific filtering examples to understand what was removed.
"""

import json
import re
from difflib import unified_diff

def examine_filtering_example(server_name, data):
    """Examine what was filtered in a specific server."""
    for server in data:
        if server.get('name') == server_name:
            orig = server['readme_content']
            filt = server['readme_filteredinitial']
            
            print(f'=== {server_name} FILTERING ANALYSIS ===')
            print(f'Original length: {len(orig)} chars')
            print(f'Filtered length: {len(filt)} chars')
            print(f'Reduction: {((len(orig) - len(filt)) / len(orig)) * 100:.1f}%')
            
            # Split into lines
            orig_lines = orig.split('\n')
            filt_lines = filt.split('\n')
            
            print(f'\nOriginal lines: {len(orig_lines)}')
            print(f'Filtered lines: {len(filt_lines)}')
            print(f'Lines removed: {len(orig_lines) - len(filt_lines)}')
            
            # Find installation-related lines in original
            install_keywords = ['install', 'clone', 'docker', 'npm', 'pip', 'yarn', 'setup', 'requirements.txt', 'curl', 'wget']
            install_lines = []
            
            for i, line in enumerate(orig_lines):
                if any(keyword in line.lower() for keyword in install_keywords):
                    install_lines.append((i, line.strip()))
            
            print(f'\nInstallation-related lines in original ({len(install_lines)}):')
            for line_num, line in install_lines[:10]:  # Show first 10
                print(f'  Line {line_num}: {line}')
            
            # Check if these lines are in filtered version
            install_in_filtered = []
            for i, line in enumerate(filt_lines):
                if any(keyword in line.lower() for keyword in install_keywords):
                    install_in_filtered.append((i, line.strip()))
            
            print(f'\nInstallation-related lines remaining in filtered ({len(install_in_filtered)}):')
            for line_num, line in install_in_filtered[:5]:
                print(f'  Line {line_num}: {line}')
            
            # Show a diff sample (first 50 lines)
            print(f'\n=== DIFF SAMPLE (first 50 lines) ===')
            orig_sample = orig_lines[:50]
            filt_sample = filt_lines[:50]
            
            diff = list(unified_diff(orig_sample, filt_sample, 
                                   fromfile='original', tofile='filtered', lineterm=''))
            
            for line in diff[:30]:  # Show first 30 diff lines
                print(line)
            
            return server
    
    return None

def main():
    """Main function to examine filtering examples."""
    file_path = '/home/ubuntu/mcp-monitoring/data_unified_filtered.json'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Examine several examples
    examples = ['A2A-MCP-Server', 'a2a-mcp-with-security', '05-make-your-mcp-server']
    
    for example in examples:
        examine_filtering_example(example, data)
        print('\n' + '='*100 + '\n')

if __name__ == "__main__":
    main()
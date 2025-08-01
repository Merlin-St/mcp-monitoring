#!/usr/bin/env python3

import json
import pandas as pd

# Load results
with open('conseq_fin_stage4_multi_results.json', 'r') as f:
    data = json.load(f)

# Create summary
print('=== O*NET Task Classification Results Summary ===')
print(f'\nTotal tools processed: {len(data["results"])}')
print(f'Tools with task mapping: {data["summary"]["tools_with_task_mapping"]}')
print(f'Unique O*NET tasks mapped: {data["summary"]["unique_tasks_mapped"]}')

print(f'\n=== Collaboration Patterns ===')
for pattern, count in data['summary']['collaboration_patterns'].items():
    if len(pattern) < 50:  # Skip the long one
        print(f'{pattern}: {count}')

print(f'\n=== Automation vs Augmentation ===')
auto_info = data['summary']['automation_vs_augmentation']
print(f'Automation tools: {auto_info["automation"]} ({auto_info["automation_percentage"]:.1f}%)')
print(f'Augmentation tools: {auto_info["augmentation"]}')

print(f'\n=== Automation Levels ===')
for level, count in sorted(data['summary']['automation_levels'].items()):
    print(f'Level {level}: {count} tools')
print(f'Average level: {data["summary"]["avg_automation_level"]:.2f}')
print(f'High risk tools (4-5): {data["summary"]["high_risk_tools"]} ({data["summary"]["high_risk_percentage"]:.1f}%)')

print(f'\n=== Tool Replacement ===')
print(f'Tools replacing traditional workplace tools: {data["summary"]["tools_replacing_traditional"]}')
print(f'Average tools replaced per MCP tool: {data["summary"]["avg_tools_replaced"]:.2f}')
print(f'\nTop 5 most commonly replaced tools:')
for tool, count in list(data['summary']['most_replaced_tools'].items())[:5]:
    print(f'  - {tool}: {count} times')

# Show sample of task mappings
print(f'\n=== Sample O*NET Task Mappings ===')
df = pd.read_csv('conseq_fin_stage4_multi_results.csv')
task_samples = df[df['task_mapping'].notna() & (df['task_mapping'] != '')].head(5)
for _, row in task_samples.iterrows():
    task = row['task_mapping']
    # Extract just the task name, not the full LLM response
    if '**' in task:
        parts = task.split('**')
        if len(parts) > 1:
            task = parts[1]
    elif ':' in task:
        task = task.split(':')[-1].strip()
    print(f'\n- Tool: {row["tool_name"]}')
    print(f'  Task: {task[:80]}...')
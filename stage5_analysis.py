#!/usr/bin/env python3
"""
Stage 4 O*NET Classification - Analysis and Visualization

Analyzes the classification results and generates visualizations similar to the
Anthropic paper "Which Economic Tasks are Performed with AI?"

Usage:
    python stage5_analysis.py
    python stage5_analysis.py --input-file custom_results.json
"""

import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stage5_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Top-level task names
TOP_LEVEL_TASKS = {
    1: "IT Systems",
    2: "Art & Culture",
    3: "Business & Finance",
    4: "Education & HR",
    5: "Scientific Research",
    6: "Government & Safety",
    7: "Industrial & Agricultural",
    8: "Energy Management",
    9: "Environmental Systems",
    10: "Healthcare Services"
}

class ONETAnalyzer:
    def __init__(self, results_file: str, csv_file: str):
        self.results_file = results_file
        self.csv_file = csv_file
        self.results = None
        self.df = None
        self.output_dir = Path("stage5_visualizations")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_data(self):
        """Load classification results"""
        logger.info(f"Loading results from {self.results_file}")
        with open(self.results_file, 'r') as f:
            self.results = json.load(f)
        
        logger.info(f"Loading DataFrame from {self.csv_file}")
        self.df = pd.read_csv(self.csv_file)
        
        # Filter to valid results
        self.df_valid = self.df[self.df['score'] > 0].copy()
        logger.info(f"Loaded {len(self.df)} total results, {len(self.df_valid)} valid")
        
    def analyze_task_distribution(self):
        """Analyze distribution across top-level tasks"""
        logger.info("Analyzing task distribution...")
        
        # Create figure similar to paper's Figure 2
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left: Raw counts
        task_counts = self.df_valid['top_level_number'].value_counts().sort_index()
        task_labels = [TOP_LEVEL_TASKS.get(int(i), f"Task {i}") for i in task_counts.index]
        
        bars1 = ax1.barh(task_labels, task_counts.values)
        ax1.set_xlabel('Number of MCP Tools')
        ax1.set_title('MCP Tool Distribution Across Economic Task Categories')
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars1, task_counts.values)):
            ax1.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                    f'{count} ({count/len(self.df_valid)*100:.1f}%)', 
                    va='center')
        
        # Right: Percentage comparison (placeholder for comparison with economy)
        task_pcts = (task_counts / len(self.df_valid) * 100).values
        
        # Create stacked bar for comparison
        width = 0.35
        x = np.arange(len(task_labels))
        
        bars2 = ax2.bar(x - width/2, task_pcts, width, label='MCP Tools')
        
        # Add hypothetical economy distribution (would need real data)
        # For now, using uniform distribution as placeholder
        economy_pcts = [10] * len(task_labels)  # Placeholder
        bars3 = ax2.bar(x + width/2, economy_pcts, width, label='U.S. Economy (placeholder)', alpha=0.5)
        
        ax2.set_ylabel('Percentage')
        ax2.set_xlabel('Task Categories')
        ax2.set_title('Task Distribution: MCP Tools vs Economy')
        ax2.set_xticks(x)
        ax2.set_xticklabels([t.replace(' & ', '\n& ') for t in task_labels], rotation=45, ha='right')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'stage5_task_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log statistics
        logger.info("\nTop-level task distribution:")
        for idx, count in task_counts.items():
            logger.info(f"  {TOP_LEVEL_TASKS.get(int(idx), idx)}: {count} tools ({count/len(self.df_valid)*100:.1f}%)")
        
    def analyze_automation_patterns(self):
        """Analyze automation vs augmentation patterns"""
        logger.info("Analyzing automation patterns...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left: Overall automation vs augmentation (pie chart like paper)
        automation_patterns = ['Directive', 'Feedback Loop']
        augmentation_patterns = ['Task Iteration', 'Learning', 'Validation']
        
        pattern_counts = self.df_valid['collaboration_pattern'].value_counts()
        
        automation_count = sum(pattern_counts.get(p, 0) for p in automation_patterns)
        augmentation_count = sum(pattern_counts.get(p, 0) for p in augmentation_patterns)
        none_count = pattern_counts.get('None', 0)
        
        # Pie chart
        sizes = [automation_count, augmentation_count]
        labels = [f'Automation\n({automation_count}, {automation_count/(automation_count+augmentation_count)*100:.1f}%)',
                 f'Augmentation\n({augmentation_count}, {augmentation_count/(automation_count+augmentation_count)*100:.1f}%)']
        colors = ['#ff9999', '#66b3ff']
        
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='',
                                           startangle=90, textprops={'fontsize': 12})
        ax1.set_title('Automation vs Augmentation in MCP Tools')
        
        # Right: Breakdown by pattern
        pattern_order = ['Directive', 'Feedback Loop', 'Task Iteration', 'Learning', 'Validation', 'None']
        pattern_data = []
        for pattern in pattern_order:
            count = pattern_counts.get(pattern, 0)
            pattern_data.append({
                'Pattern': pattern,
                'Count': count,
                'Percentage': count / len(self.df_valid) * 100,
                'Type': 'Automation' if pattern in automation_patterns else 
                       ('Augmentation' if pattern in augmentation_patterns else 'Unknown')
            })
        
        pattern_df = pd.DataFrame(pattern_data)
        
        # Create grouped bar chart
        colors_map = {'Automation': '#ff9999', 'Augmentation': '#66b3ff', 'Unknown': '#gray'}
        bars = ax2.bar(pattern_df['Pattern'], pattern_df['Count'], 
                       color=[colors_map[t] for t in pattern_df['Type']])
        
        ax2.set_xlabel('Collaboration Pattern')
        ax2.set_ylabel('Number of Tools')
        ax2.set_title('Distribution of Collaboration Patterns')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, row in zip(bars, pattern_df.itertuples()):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{row.Count}\n({row.Percentage:.1f}%)', 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'stage5_automation_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"\nAutomation patterns:")
        logger.info(f"  Automation: {automation_count} ({automation_count/(automation_count+augmentation_count)*100:.1f}%)")
        logger.info(f"  Augmentation: {augmentation_count} ({augmentation_count/(automation_count+augmentation_count)*100:.1f}%)")
        
    def analyze_automation_levels(self):
        """Analyze automation level distribution"""
        logger.info("Analyzing automation levels...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left: Distribution of automation levels
        level_counts = self.df_valid['automation_level'].value_counts().sort_index()
        level_labels = {
            0: "Not Functional",
            1: "Monitoring",
            2: "Analysis",
            3: "Meta-Coordination",
            4: "Restricted Execution",
            5: "Unrestricted Execution"
        }
        
        bars = ax1.bar(level_counts.index, level_counts.values)
        ax1.set_xlabel('Automation Level')
        ax1.set_ylabel('Number of Tools')
        ax1.set_title('Distribution of MCP Tools by Automation Level')
        ax1.set_xticks(range(6))
        ax1.set_xticklabels([f"{i}\n{level_labels[i]}" for i in range(6)], rotation=45, ha='right')
        
        # Add value labels
        for bar, count in zip(bars, level_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{count}\n({count/len(self.df_valid)*100:.1f}%)', 
                    ha='center', va='bottom')
        
        # Right: Automation level by top task category
        pivot_data = pd.crosstab(self.df_valid['top_level_number'], 
                                self.df_valid['automation_level'], 
                                normalize='index') * 100
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', 
                   ax=ax2, cbar_kws={'label': 'Percentage'})
        ax2.set_xlabel('Automation Level')
        ax2.set_ylabel('Top-Level Task Category')
        ax2.set_title('Automation Levels by Task Category (%)')
        ax2.set_yticklabels([TOP_LEVEL_TASKS.get(int(i), f"Task {i}") for i in pivot_data.index], rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'stage5_automation_levels.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate statistics
        avg_level = self.df_valid['automation_level'].mean()
        logger.info(f"\nAutomation level statistics:")
        logger.info(f"  Average level: {avg_level:.2f}")
        for level, count in level_counts.items():
            logger.info(f"  Level {level} ({level_labels[level]}): {count} tools ({count/len(self.df_valid)*100:.1f}%)")
        
    def analyze_tool_replacement(self):
        """Analyze which O*NET tools are being replaced"""
        logger.info("Analyzing tool replacement...")
        
        # Get all replaced tools
        all_replaced = []
        for tools_str in self.df_valid['replaced_tools'].dropna():
            if tools_str:
                all_replaced.extend(tools_str.split(';'))
        
        if not all_replaced:
            logger.warning("No replaced tools found")
            return
        
        # Count occurrences
        tool_counts = Counter(all_replaced)
        top_tools = tool_counts.most_common(20)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 10))
        
        tools, counts = zip(*top_tools)
        y_pos = np.arange(len(tools))
        
        bars = ax.barh(y_pos, counts)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([t[:50] + '...' if len(t) > 50 else t for t in tools])
        ax.invert_yaxis()
        ax.set_xlabel('Number of MCP Tools Replacing This')
        ax.set_title('Top 20 Most Commonly Replaced O*NET Tools')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   str(count), va='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'stage5_replaced_tools.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Statistics
        tools_with_replacement = (self.df_valid['replaced_tools_count'] > 0).sum()
        avg_replaced = self.df_valid['replaced_tools_count'].mean()
        
        logger.info(f"\nTool replacement statistics:")
        logger.info(f"  Tools replacing O*NET tools: {tools_with_replacement} ({tools_with_replacement/len(self.df_valid)*100:.1f}%)")
        logger.info(f"  Average tools replaced per MCP tool: {avg_replaced:.2f}")
        logger.info(f"  Total unique O*NET tools replaced: {len(tool_counts)}")
        logger.info("\nTop 10 most replaced tools:")
        for tool, count in top_tools[:10]:
            logger.info(f"  {tool}: {count} MCP tools")
        
    def analyze_occupations(self):
        """Analyze occupation distribution"""
        logger.info("Analyzing occupation distribution...")
        
        # Get top occupations
        occupation_counts = self.df_valid['occupation'].value_counts().head(20)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 10))
        
        y_pos = np.arange(len(occupation_counts))
        bars = ax.barh(y_pos, occupation_counts.values)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(occupation_counts.index)
        ax.invert_yaxis()
        ax.set_xlabel('Number of MCP Tools')
        ax.set_title('Top 20 Occupations by MCP Tool Count')
        
        # Add value labels
        for bar, count in zip(bars, occupation_counts.values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{count} ({count/len(self.df_valid)*100:.1f}%)', va='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'stage5_occupations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("\nTop 10 occupations:")
        for occ, count in occupation_counts.head(10).items():
            logger.info(f"  {occ}: {count} tools ({count/len(self.df_valid)*100:.1f}%)")
        
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        logger.info("Generating summary report...")
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'analysis_summary': {
                'total_tools_analyzed': len(self.df_valid),
                'total_servers_represented': len(set(self.df_valid['server_name'])),
                'classification_success_rate': len(self.df_valid) / len(self.df) * 100
            },
            'key_findings': {}
        }
        
        # Task distribution
        task_dist = self.df_valid['top_level_number'].value_counts().to_dict()
        report['key_findings']['dominant_task_category'] = {
            'category': TOP_LEVEL_TASKS.get(int(task_dist.keys()[0]), 'Unknown'),
            'percentage': list(task_dist.values())[0] / len(self.df_valid) * 100
        }
        
        # Automation patterns
        automation_patterns = ['Directive', 'Feedback Loop']
        augmentation_patterns = ['Task Iteration', 'Learning', 'Validation']
        pattern_counts = self.df_valid['collaboration_pattern'].value_counts()
        
        automation_count = sum(pattern_counts.get(p, 0) for p in automation_patterns)
        augmentation_count = sum(pattern_counts.get(p, 0) for p in augmentation_patterns)
        
        report['key_findings']['automation_ratio'] = {
            'automation_percentage': automation_count / (automation_count + augmentation_count) * 100,
            'augmentation_percentage': augmentation_count / (automation_count + augmentation_count) * 100
        }
        
        # Automation levels
        report['key_findings']['average_automation_level'] = self.df_valid['automation_level'].mean()
        report['key_findings']['execution_capable_tools'] = (self.df_valid['automation_level'] >= 4).sum()
        
        # Tool replacement
        report['key_findings']['tools_replacing_onet'] = (self.df_valid['replaced_tools_count'] > 0).sum()
        report['key_findings']['average_tools_replaced'] = self.df_valid['replaced_tools_count'].mean()
        
        # Save report
        report_file = self.output_dir / 'stage5_analysis_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Summary report saved to {report_file}")
        
        return report
    
    def run_full_analysis(self):
        """Run all analyses"""
        logger.info("Starting full O*NET classification analysis...")
        
        self.load_data()
        self.analyze_task_distribution()
        self.analyze_automation_patterns()
        self.analyze_automation_levels()
        self.analyze_tool_replacement()
        self.analyze_occupations()
        report = self.generate_summary_report()
        
        logger.info("\n=== Analysis Complete ===")
        logger.info(f"Visualizations saved to: {self.output_dir}")
        logger.info("\nKey Findings:")
        logger.info(f"- Dominant task: {report['key_findings']['dominant_task_category']['category']} "
                   f"({report['key_findings']['dominant_task_category']['percentage']:.1f}%)")
        logger.info(f"- Automation vs Augmentation: {report['key_findings']['automation_ratio']['automation_percentage']:.1f}% vs "
                   f"{report['key_findings']['automation_ratio']['augmentation_percentage']:.1f}%")
        logger.info(f"- Average automation level: {report['key_findings']['average_automation_level']:.2f}")
        logger.info(f"- Execution-capable tools: {report['key_findings']['execution_capable_tools']}")

def main():
    parser = argparse.ArgumentParser(description='Analyze O*NET classification results')
    parser.add_argument('--results-file', type=str, default='stage5_results.json',
                       help='JSON results file from processing')
    parser.add_argument('--csv-file', type=str, default='stage5_results.csv',
                       help='CSV results file from processing')
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.results_file).exists():
        logger.error(f"Results file {args.results_file} not found")
        return
    
    if not Path(args.csv_file).exists():
        logger.error(f"CSV file {args.csv_file} not found")
        return
    
    # Run analysis
    analyzer = ONETAnalyzer(args.results_file, args.csv_file)
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()
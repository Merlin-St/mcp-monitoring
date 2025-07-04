#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conseq_fin_stage3_visual.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_stage2_data():
    """Load and clean the stage 2 results data, including threat model information."""
    try:
        # Load CSV data
        df = pd.read_csv('conseq_fin_stage2_results.csv')
        logger.info(f"Loaded {len(df)} rows from stage 2 results")
        
        # Load JSON data to get threat model information
        import json
        with open('conseq_fin_stage2_results.json', 'r') as f:
            json_data = json.load(f)
        
        # Create threat model mapping from JSON data
        threat_model_map = {}
        for result in json_data['results']:
            sample_id = result['sample_id']
            # Extract threat model from parsed output
            parsed_output = result.get('parsed_output')
            if parsed_output is None:
                threat_model = 'unknown'
            else:
                threat_model = parsed_output.get('threat_model', 'unknown')
            threat_model_map[sample_id] = threat_model
        
        # Add threat model column to dataframe
        df['threat_model_clean'] = df['sample_id'].map(threat_model_map).fillna('unknown')
        
        # Convert capability columns to boolean
        capability_cols = [col for col in df.columns if col.startswith('capability_')]
        for col in capability_cols:
            df[col] = df[col].astype(bool)
        
        # Convert consequentiality_level to numeric
        df['consequentiality_level'] = pd.to_numeric(df['consequentiality_level'], errors='coerce')
        
        logger.info(f"Threat model distribution: {df['threat_model_clean'].value_counts().to_dict()}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def create_consequentiality_chart(df):
    """Create stacked bar chart showing distribution of consequentiality levels by threat models."""
    plt.figure(figsize=(12, 6))
    
    # Create cross-tabulation of consequentiality levels and threat models
    cross_tab = pd.crosstab(df['consequentiality_level'], df['threat_model_clean'])
    
    # Define base color mapping for consequentiality levels (keep consistent)
    level_color_map = {
        1: 'lightgrey',
        2: 'grey', 
        3: 'darkgrey',
        4: 'lightcoral',
        5: 'darkred'
    }
    
    # Define pattern mapping for threat models
    threat_model_patterns = {
        'TM1': '\\\\\\',         # Diagonal lines (30 degree angle, left-leaning)
        'TM2': '///',           # Diagonal lines (30 degree angle, right-leaning)
        'TM3': None,           # No pattern (clear/solid)
        'multiple': 'xx',      # Cross-over lines
        'unknown': '***'       # Star pattern
    }
    
    # Ensure all levels 1-5 are represented
    for level in range(1, 6):
        if level not in cross_tab.index:
            # Add empty row for missing level
            cross_tab.loc[level] = 0
    
    cross_tab = cross_tab.sort_index()
    
    # Create stacked bar chart
    bottom = np.zeros(len(cross_tab.index))
    bars = []
    
    for tm in cross_tab.columns:
        # Use base color according to level, pattern according to threat model
        for level in cross_tab.index:
            if cross_tab.loc[level, tm] > 0:
                color = level_color_map.get(level, 'lightgrey')
                pattern = threat_model_patterns.get(tm, None)
                
                bars.append(plt.bar(level, cross_tab.loc[level, tm], 
                                  bottom=bottom[list(cross_tab.index).index(level)], 
                                  color=color, hatch=pattern, alpha=0.8,
                                  label=tm if level == cross_tab.index[0] else "",
                                  edgecolor='black', linewidth=0.5))
        
        # Update bottom for stacking
        for i, level in enumerate(cross_tab.index):
            bottom[i] += cross_tab.loc[level, tm]
    
    # Add value labels on bars
    for level in cross_tab.index:
        total = cross_tab.loc[level].sum()
        if total > 0:
            plt.text(level, total + 0.1, f'{int(total)}',
                    ha='center', va='bottom', fontweight='bold')
    
    plt.title('Distribution of Consequentiality Levels by Threat Models', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Consequentiality Level', fontsize=12)
    plt.ylabel('Number of Servers', fontsize=12)
    
    # Set x-axis to show all levels 1-5
    plt.xlim(0.5, 5.5)
    plt.xticks(range(1, 6))
    
    # Add level descriptions
    level_descriptions = {
        1: 'MONITORING\n(Read-only)',
        2: 'ADVISING\n(Recommendations)',
        3: 'PREPARING\n(Staging operations)',
        4: 'EXECUTING\n(With constraints)',
        5: 'EXECUTING\n(No constraints)'
    }
    
    ax = plt.gca()
    max_height = max(cross_tab.sum(axis=1)) if len(cross_tab) > 0 else 1
    # Show descriptions for all levels 1-5, even if no data exists
    for level, desc in level_descriptions.items():
        ax.text(level, -max_height * 0.15, desc, 
               ha='center', va='top', fontsize=9, style='italic')
    
    # Define threat model descriptions (1-3 words as requested)
    threat_model_descriptions = {
        'TM1': 'Credit Risk',
        'TM2': 'Deposit Movement', 
        'TM3': 'Payment Systems',
        'multiple': 'Multiple Models',
        'unknown': 'Unknown'
    }
    
    # Create custom legend with patterns
    legend_elements = []
    for tm in cross_tab.columns:
        pattern = threat_model_patterns.get(tm, None)
        description = threat_model_descriptions.get(tm, tm)
        label = f"{tm} ({description})"
        legend_elements.append(plt.Rectangle((0,0),1,1, hatch=pattern, 
                                           facecolor='lightgrey', edgecolor='black',
                                           label=label))
    
    plt.legend(handles=legend_elements, title='Threat Models', loc='upper right')
    
    plt.tight_layout()
    plt.savefig('conseq_fin_stage3_consequentiality_levels.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Cross-tabulation of levels and threat models:\n{cross_tab}")
    
    return cross_tab

def create_capability_chart(df):
    """Create horizontal bar chart showing affordance distributions with percentages."""
    plt.figure(figsize=(12, 4))  # Reduced height by half
    
    # Get capability columns
    capability_cols = [col for col in df.columns if col.startswith('capability_') and col != 'capability_sensitive_data_required']
    
    # Clean up column names for display - change to affordance terminology
    affordance_names = [col.replace('capability_', '').replace('_', ' ').title() for col in capability_cols]
    
    # Calculate percentages for each affordance
    affordance_data = []
    for col in capability_cols:
        true_count = df[col].sum()
        false_count = len(df) - true_count
        true_pct = (true_count / len(df)) * 100
        false_pct = (false_count / len(df)) * 100
        affordance_data.append({
            'affordance': col.replace('capability_', '').replace('_', ' ').title(),
            'true_count': true_count,
            'false_count': false_count,
            'true_pct': true_pct,
            'false_pct': false_pct
        })
    
    # Create horizontal stacked bar chart
    affordances = [item['affordance'] for item in affordance_data]
    true_counts = [item['true_count'] for item in affordance_data]
    false_counts = [item['false_count'] for item in affordance_data]
    true_pcts = [item['true_pct'] for item in affordance_data]
    
    y = np.arange(len(affordances))
    height = 0.3  # Very narrow lines as requested
    
    # Create horizontal stacked bars - black for present, light grey for absent
    bars1 = plt.barh(y, true_counts, height, label='Affordance Present', color='black')
    bars2 = plt.barh(y, false_counts, height, left=true_counts, label='Affordance Absent', color='lightgrey')
    
    # Add percentage labels only for black bars (affordance present)
    for i, (bar1, pct) in enumerate(zip(bars1, true_pcts)):
        # Label for true percentage only
        if pct > 5:  # Only show if percentage is significant
            plt.text(bar1.get_width()/2, bar1.get_y() + bar1.get_height()/2,
                    f'{pct:.1f}%', ha='center', va='center', fontweight='bold', color='white')
    
    plt.title('Affordance Distribution Across Finance MCP Servers', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Servers', fontsize=12)
    plt.ylabel('Affordances', fontsize=12)
    plt.yticks(y, affordances)
    plt.legend(loc='upper right')  # Move legend to top right
    
    plt.tight_layout()
    plt.savefig('conseq_fin_stage3_capabilities.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return affordance_data

def find_most_execution_level_tools(df):
    """Find and display the 5 most execution-level tools."""
    logger.info("Finding most execution-level tools...")
    
    # Filter for high execution capability (level 4 or 5, can execute transactions)
    execution_servers = df[
        (df['consequentiality_level'] >= 4) & 
        (df['capability_can_execute_transactions'] == True)
    ].copy()
    
    if len(execution_servers) == 0:
        logger.warning("No servers found with execution capabilities")
        # Fallback to level 3+ with transaction capability
        execution_servers = df[
            (df['consequentiality_level'] >= 3) & 
            (df['capability_can_execute_transactions'] == True)
        ].copy()
    
    if len(execution_servers) == 0:
        logger.warning("No servers found with transaction capabilities")
        # Fallback to highest consequentiality levels
        execution_servers = df[df['consequentiality_level'] == df['consequentiality_level'].max()].copy()
    
    # Sort by consequentiality level and select top 5
    execution_servers = execution_servers.sort_values('consequentiality_level', ascending=False).head(5)
    
    logger.info(f"Found {len(execution_servers)} execution-level servers")
    
    print("\n" + "="*80)
    print("TOP 5 MOST EXECUTION-LEVEL FINANCE MCP TOOLS")
    print("="*80)
    
    for idx, (_, server) in enumerate(execution_servers.iterrows(), 1):
        print(f"\n{idx}. {server['server_name']} ({server['server_id']})")
        print(f"   Consequentiality Level: {server['consequentiality_level']}")
        print(f"   Affordances:")
        print(f"     • Can Read Financial Data: {server['capability_can_read_financial_data']}")
        print(f"     • Can Modify Financial Data: {server['capability_can_modify_financial_data']}")
        print(f"     • Can Execute Transactions: {server['capability_can_execute_transactions']}")
        print(f"     • Can Make Binding Decisions: {server['capability_can_make_binding_decisions']}")
        print(f"     • Requires Human Approval: {server['capability_requires_human_approval']}")
        print(f"     • Has Monetary Limits: {server['capability_has_monetary_limits']}")
        print(f"   Reversibility: {server['reversibility']}")
        print(f"   Threat Model: {server['threat_model']}")
        if pd.notna(server['analysis_reasoning']):
            reasoning = str(server['analysis_reasoning'])[:200] + "..." if len(str(server['analysis_reasoning'])) > 200 else str(server['analysis_reasoning'])
            print(f"   Analysis: {reasoning}")
    
    return execution_servers

def generate_summary_stats(df):
    """Generate and display summary statistics."""
    logger.info("Generating summary statistics...")
    
    print("\n" + "="*80)
    print("FINANCE MCP SERVERS CONSEQUENTIALITY ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"Total Servers Analyzed: {len(df)}")
    print(f"Average Consequentiality Level: {df['consequentiality_level'].mean():.2f}")
    print(f"Median Consequentiality Level: {df['consequentiality_level'].median():.1f}")
    
    print("\nConsequentiality Level Distribution:")
    level_counts = df['consequentiality_level'].value_counts().sort_index()
    for level, count in level_counts.items():
        pct = (count / len(df)) * 100
        print(f"  Level {level}: {count} servers ({pct:.1f}%)")
    
    print("\nAffordance Statistics:")
    capability_cols = [col for col in df.columns if col.startswith('capability_') and col != 'capability_sensitive_data_required']
    for col in capability_cols:
        true_count = df[col].sum()
        pct = (true_count / len(df)) * 100
        clean_name = col.replace('capability_', '').replace('_', ' ').title()
        print(f"  {clean_name}: {true_count} servers ({pct:.1f}%)")
    
    print("\nHigh-Risk Servers (Level 4-5):")
    high_risk = df[df['consequentiality_level'] >= 4]
    print(f"  Count: {len(high_risk)} servers ({len(high_risk)/len(df)*100:.1f}%)")
    
    if len(high_risk) > 0:
        print(f"  Can Execute Transactions: {high_risk['capability_can_execute_transactions'].sum()}")
        print(f"  Can Make Binding Decisions: {high_risk['capability_can_make_binding_decisions'].sum()}")
        print(f"  Require Human Approval: {high_risk['capability_requires_human_approval'].sum()}")

def main():
    """Main visualization function."""
    logger.info("Starting Finance MCP Server Consequentiality Visualization")
    
    try:
        # Load data
        df = load_stage2_data()
        
        # Generate summary statistics
        generate_summary_stats(df)
        
        # Create consequentiality level chart
        logger.info("Creating consequentiality level chart...")
        level_counts = create_consequentiality_chart(df)
        
        # Create affordance chart
        logger.info("Creating affordance distribution chart...")
        affordance_data = create_capability_chart(df)
        
        # Find most execution-level tools
        execution_servers = find_most_execution_level_tools(df)
        
        logger.info("Visualization complete! Generated files:")
        logger.info("  - conseq_fin_stage3_consequentiality_levels.png")
        logger.info("  - conseq_fin_stage3_capabilities.png")
        logger.info("  - conseq_fin_stage3_visual.log")
        
    except Exception as e:
        logger.error(f"Error in visualization: {e}")
        raise

if __name__ == "__main__":
    main()
#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging

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

def load_stage1_data():
    """Load and clean the stage 1 results data."""
    try:
        # Load CSV data
        df = pd.read_csv('conseq_fin_stage1_results.csv')
        logger.info(f"Loaded {len(df)} rows from stage 1 results")
        
        # Load JSON data to get additional information
        import json
        with open('conseq_fin_stage1_results.json', 'r') as f:
            json_data = json.load(f)
        
        logger.info(f"JSON summary: {json_data.get('summary', {})}")
        
        # Add finance relevance based on is_finance_llm field
        df['finance_relevant'] = df['is_finance_llm'] == 1
        
        # Identify transfer capability columns
        transfer_cols = [col for col in df.columns if col.startswith('transfer_')]
        df['has_transfer_capability'] = df[transfer_cols].sum(axis=1) > 0
        
        # Identify level 4+ servers
        df['is_level_4_plus'] = df['level'] >= 4
        
        logger.info(f"Finance relevant servers: {df['finance_relevant'].sum()}")
        logger.info(f"Servers with transfer capabilities: {df['has_transfer_capability'].sum()}")
        logger.info(f"Level 4+ servers: {df['is_level_4_plus'].sum()}")
        
        return df, transfer_cols
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def create_transfer_capabilities_chart(df, transfer_cols):
    """Create chart showing distribution of transfer capabilities."""
    plt.figure(figsize=(14, 8))
    
    # Count servers with each transfer capability
    transfer_counts = {}
    for col in transfer_cols:
        count = df[df[col] == 1].shape[0]
        clean_name = col.replace('transfer_', '').replace('_', ' ').title()
        transfer_counts[clean_name] = count
    
    # Sort by count
    sorted_transfers = sorted(transfer_counts.items(), key=lambda x: x[1], reverse=True)
    labels, values = zip(*sorted_transfers) if sorted_transfers else ([], [])
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    bars = plt.bar(labels, values, color=colors)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        if value > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                    f'{int(value)}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('MCP Servers by Transfer Capability Type', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Servers', fontsize=12)
    plt.xlabel('Transfer Capability Type', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('conseq_fin_stage3_transfer_capabilities.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    total_with_transfer = df['has_transfer_capability'].sum()
    logger.info(f"Total servers with transfer capabilities: {total_with_transfer}")
    logger.info(f"Transfer capability breakdown: {transfer_counts}")
    
    return transfer_counts

def create_level_distribution_chart(df):
    """Create chart showing distribution of consequentiality levels."""
    plt.figure(figsize=(12, 6))
    
    # Count servers by level
    level_counts = df['level'].value_counts().sort_index()
    
    colors = ['lightblue', 'lightgreen', 'yellow', 'orange', 'red', 'darkred'][:len(level_counts)]
    bars = plt.bar(level_counts.index, level_counts.values, color=colors)
    
    # Add value labels on bars
    for bar, value in zip(bars, level_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(level_counts.values)*0.01,
                f'{int(value)}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('MCP Servers by Consequentiality Level', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Servers', fontsize=12)
    plt.xlabel('Consequentiality Level', fontsize=12)
    
    # Add level 4+ highlight
    level4_count = df['is_level_4_plus'].sum()
    plt.axvline(x=3.5, color='red', linestyle='--', alpha=0.7, linewidth=2)
    plt.text(4.5, max(level_counts.values)*0.8, f'Level 4+: {level4_count} servers', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.3),
             fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('conseq_fin_stage3_level_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Level distribution: {level_counts.to_dict()}")
    logger.info(f"Level 4+ servers: {level4_count}")
    
    return level_counts

def create_finance_relevance_chart(df):
    """Create bar chart showing distribution of finance-relevant servers."""
    plt.figure(figsize=(10, 6))
    
    # Count finance relevant vs non-relevant
    finance_counts = df['finance_relevant'].value_counts()
    labels = ['Finance Relevant', 'Not Finance Relevant']
    values = [finance_counts.get(True, 0), finance_counts.get(False, 0)]
    
    colors = ['darkgreen', 'lightgray']
    bars = plt.bar(labels, values, color=colors)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        if value > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                    f'{int(value)}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Finance-Relevant MCP Servers Distribution', 
              fontsize=14, fontweight='bold')
    plt.ylabel('Number of Servers', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('conseq_fin_stage3_finance_relevance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Finance relevance distribution: {dict(zip(labels, values))}")
    
    return dict(zip(labels, values))

def create_server_overview_chart(df):
    """Create overview chart of finance-relevant servers."""
    plt.figure(figsize=(12, 6))
    
    # For stage 1, we'll show basic server information
    # Check what columns are available
    available_cols = df.columns.tolist()
    logger.info(f"Available columns: {available_cols}")
    
    # Create a simple overview based on available data
    if 'finance_relevant' in df.columns:
        df[df['finance_relevant']]
        plt.figure(figsize=(10, 6))
        
        # Show finance relevance distribution
        relevance_counts = df['finance_relevant'].value_counts()
        labels = ['Finance Relevant', 'Not Finance Relevant']
        values = [relevance_counts.get(True, 0), relevance_counts.get(False, 0)]
        colors = ['darkgreen', 'lightgray']
        
        bars = plt.bar(labels, values, color=colors)
        
        # Add value labels
        for bar, value in zip(bars, values):
            if value > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                        f'{int(value)}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Finance MCP Servers Overview', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Servers', fontsize=12)
        
    else:
        # All servers assumed finance relevant
        plt.bar(['Finance Relevant Servers'], [len(df)], color='darkgreen')
        plt.text(0, len(df) + len(df)*0.01, f'{len(df)}', 
                ha='center', va='bottom', fontweight='bold')
        plt.title('Finance MCP Servers Overview', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Servers', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('conseq_fin_stage3_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def display_transfer_capability_servers(df, transfer_cols):
    """Display servers with transfer capabilities."""
    logger.info("Finding servers with transfer capabilities...")
    
    transfer_servers = df[df['has_transfer_capability']].copy()
    
    if len(transfer_servers) == 0:
        logger.warning("No servers with transfer capabilities found")
        return pd.DataFrame()
    
    # Sort by level (highest first) and get top 10
    top_transfer_servers = transfer_servers.sort_values('level', ascending=False).head(10)
    
    print("\n" + "="*80)
    print("SERVERS WITH TRANSFER CAPABILITIES")
    print("="*80)
    print(f"Total servers with transfer capabilities: {len(transfer_servers)}")
    print("Top 10 by consequentiality level:")
    
    for idx, (_, server) in enumerate(top_transfer_servers.iterrows(), 1):
        server_name = server.get('server', f"Server {idx}")
        level = server.get('level', 0)
        confidence = server.get('confidence', 'Unknown')
        
        print(f"\n{idx}. {server_name} (Level {level}, Confidence: {confidence})")
        
        # Show which transfer capabilities they have
        capabilities = []
        for col in transfer_cols:
            if server[col] == 1:
                cap_name = col.replace('transfer_', '').replace('_', ' ').title()
                capabilities.append(cap_name)
        
        if capabilities:
            print(f"   Transfer Capabilities: {', '.join(capabilities)}")
        
        # Show analysis notes
        analysis = server.get('analysis_notes', '')
        if analysis and len(str(analysis)) > 0:
            analysis_text = str(analysis)[:150] + "..." if len(str(analysis)) > 150 else str(analysis)
            print(f"   Analysis: {analysis_text}")
    
    return top_transfer_servers

def display_level4_plus_servers(df):
    """Display level 4+ servers."""
    logger.info("Finding level 4+ servers...")
    
    level4_servers = df[df['is_level_4_plus']].copy()
    
    if len(level4_servers) == 0:
        logger.warning("No level 4+ servers found")
        return pd.DataFrame()
    
    # Sort by level (highest first) and get top 15
    top_level4_servers = level4_servers.sort_values('level', ascending=False).head(15)
    
    print("\n" + "="*80)
    print("LEVEL 4+ SERVERS (HIGH CONSEQUENTIALITY)")
    print("="*80)
    print(f"Total level 4+ servers: {len(level4_servers)}")
    print("Top 15 by consequentiality level:")
    
    for idx, (_, server) in enumerate(top_level4_servers.iterrows(), 1):
        server_name = server.get('server', f"Server {idx}")
        level = server.get('level', 0)
        confidence = server.get('confidence', 'Unknown')
        is_finance = server.get('finance_relevant', False)
        
        finance_indicator = " [FINANCE]" if is_finance else ""
        
        print(f"\n{idx}. {server_name} (Level {level}, Confidence: {confidence}){finance_indicator}")
        
        # Show analysis notes
        analysis = server.get('analysis_notes', '')
        if analysis and len(str(analysis)) > 0:
            analysis_text = str(analysis)[:150] + "..." if len(str(analysis)) > 150 else str(analysis)
            print(f"   Analysis: {analysis_text}")
    
    return top_level4_servers

def find_top_finance_tools(df):
    """Find and display the top finance-relevant tools."""
    logger.info("Finding top finance-relevant tools...")
    
    finance_servers = df[df['finance_relevant']].copy()
    
    if len(finance_servers) == 0:
        logger.warning("No finance-relevant servers found")
        return pd.DataFrame()
    
    # Sort by level (highest first) and get top 10
    top_servers = finance_servers.sort_values('level', ascending=False).head(10)
    
    logger.info(f"Found {len(top_servers)} top finance servers")
    
    print("\n" + "="*80)
    print("TOP FINANCE-RELEVANT MCP SERVERS")
    print("="*80)
    print(f"Total finance-relevant servers: {len(finance_servers)}")
    print("Top 10 by consequentiality level:")
    
    for idx, (_, server) in enumerate(top_servers.iterrows(), 1):
        server_name = server.get('server', f"Server {idx}")
        level = server.get('level', 0)
        confidence = server.get('confidence', 'Unknown')
        
        print(f"\n{idx}. {server_name} (Level {level}, Confidence: {confidence})")
        
        # Show analysis notes
        analysis = server.get('analysis_notes', '')
        if analysis and len(str(analysis)) > 0:
            analysis_text = str(analysis)[:150] + "..." if len(str(analysis)) > 150 else str(analysis)
            print(f"   Analysis: {analysis_text}")
    
    return top_servers

def generate_summary_stats(df, transfer_cols):
    """Generate and display summary statistics."""
    logger.info("Generating summary statistics...")
    
    print("\n" + "="*80)
    print("FINANCE MCP SERVERS STAGE 1 ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"Total Servers Analyzed: {len(df)}")
    
    # Finance relevance statistics
    finance_count = df['finance_relevant'].sum()
    finance_pct = (finance_count / len(df)) * 100
    print(f"Finance Relevant Servers: {finance_count} ({finance_pct:.1f}%)")
    print(f"Non-Finance Relevant Servers: {len(df) - finance_count} ({100-finance_pct:.1f}%)")
    
    # Transfer capability statistics
    transfer_count = df['has_transfer_capability'].sum()
    transfer_pct = (transfer_count / len(df)) * 100
    print(f"Servers with Transfer Capabilities: {transfer_count} ({transfer_pct:.1f}%)")
    
    # Level 4+ statistics
    level4_count = df['is_level_4_plus'].sum()
    level4_pct = (level4_count / len(df)) * 100
    print(f"Level 4+ Servers: {level4_count} ({level4_pct:.1f}%)")
    
    # Level distribution
    print("\nConsequentiality Level Distribution:")
    level_dist = df['level'].value_counts().sort_index()
    for level, count in level_dist.items():
        pct = (count / len(df)) * 100
        print(f"  Level {level}: {count} servers ({pct:.1f}%)")
    
    # Transfer capability breakdown
    print("\nTransfer Capability Breakdown:")
    for col in transfer_cols:
        count = df[df[col] == 1].shape[0]
        if count > 0:
            pct = (count / len(df)) * 100
            clean_name = col.replace('transfer_', '').replace('_', ' ').title()
            print(f"  {clean_name}: {count} servers ({pct:.1f}%)")
    
    # Confidence level distribution
    print("\nConfidence Level Distribution:")
    confidence_dist = df['confidence'].value_counts()
    for conf, count in confidence_dist.items():
        pct = (count / len(df)) * 100
        print(f"  {conf}: {count} servers ({pct:.1f}%)")
    
    # Cross-analysis: Finance + Transfer + Level 4+
    finance_transfer = df[(df['finance_relevant']) & (df['has_transfer_capability'])]
    finance_level4 = df[(df['finance_relevant']) & (df['is_level_4_plus'])]
    transfer_level4 = df[(df['has_transfer_capability']) & (df['is_level_4_plus'])]
    all_three = df[(df['finance_relevant']) & (df['has_transfer_capability']) & (df['is_level_4_plus'])]
    
    print("\nCross-Analysis:")
    print(f"  Finance + Transfer Capabilities: {len(finance_transfer)} servers")
    print(f"  Finance + Level 4+: {len(finance_level4)} servers")
    print(f"  Transfer + Level 4+: {len(transfer_level4)} servers")
    print(f"  Finance + Transfer + Level 4+: {len(all_three)} servers")

def main():
    """Main visualization function."""
    logger.info("Starting Finance MCP Server Consequentiality Visualization")
    
    try:
        # Load data
        df, transfer_cols = load_stage1_data()
        
        # Generate summary statistics
        generate_summary_stats(df, transfer_cols)
        
        # Create transfer capabilities chart
        logger.info("Creating transfer capabilities chart...")
        create_transfer_capabilities_chart(df, transfer_cols)
        
        # Create level distribution chart
        logger.info("Creating level distribution chart...")
        create_level_distribution_chart(df)
        
        # Create finance relevance chart
        logger.info("Creating finance relevance chart...")
        create_finance_relevance_chart(df)
        
        # Create server overview chart
        logger.info("Creating server overview chart...")
        create_server_overview_chart(df)
        
        # Display servers with transfer capabilities
        display_transfer_capability_servers(df, transfer_cols)
        
        # Display level 4+ servers
        display_level4_plus_servers(df)
        
        # Find top finance tools
        find_top_finance_tools(df)
        
        logger.info("Visualization complete! Generated files:")
        logger.info("  - conseq_fin_stage3_transfer_capabilities.png")
        logger.info("  - conseq_fin_stage3_level_distribution.png")
        logger.info("  - conseq_fin_stage3_finance_relevance.png")
        logger.info("  - conseq_fin_stage3_overview.png")
        logger.info("  - conseq_fin_stage3_visual.log")
        
    except Exception as e:
        logger.error(f"Error in visualization: {e}")
        raise

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Analyze MCP servers by NAICS industry sectors from data_unified.json
"""

import json
import pandas as pd
from collections import defaultdict
from naics_classification_config import NAICS_SECTORS


def analyze_naics_sectors():
    """Analyze MCP servers by NAICS sectors"""
    
    print("Loading MCP server data...")
    
    # Load the JSON data
    with open('data_unified_filtered.json', 'r') as f:
        data = json.load(f)
    
    total_servers = len(data)
    print(f"Total MCP servers: {total_servers:,}")
    
    # Count servers by sector
    sector_counts = defaultdict(int)
    
    for server in data:
        for sector_code in NAICS_SECTORS.keys():
            sector_key = f"is_sector_{sector_code}"
            if server.get(sector_key, False):
                sector_counts[sector_code] += 1
    
    # Create summary table
    summary_data = []
    for sector_code in sorted(NAICS_SECTORS.keys()):
        count = sector_counts[sector_code]
        percentage = (count / total_servers) * 100 if total_servers > 0 else 0
        
        summary_data.append({
            'NAICS_Code': sector_code,
            'Sector_Name': NAICS_SECTORS[sector_code],
            'Server_Count': count,
            'Percentage': percentage
        })
    
    # Sort by count (highest first)
    summary_data.sort(key=lambda x: x['Server_Count'], reverse=True)
    
    # Create DataFrame for better formatting
    df = pd.DataFrame(summary_data)
    
    print("\n" + "="*100)
    print("MCP SERVERS BY NAICS INDUSTRY SECTOR")
    print("="*100)
    print(f"{'Code':<4} {'Sector Name':<50} {'Count':<8} {'Percentage':<10}")
    print("-"*100)
    
    for _, row in df.iterrows():
        print(f"{row['NAICS_Code']:<4} {row['Sector_Name']:<50} {row['Server_Count']:<8,} {row['Percentage']:<10.2f}%")
    
    print("-"*100)
    print(f"{'TOTAL':<4} {'All Servers':<50} {total_servers:<8,} {'100.00%':<10}")
    print("="*100)
    
    # Additional statistics
    servers_with_sector = sum(1 for server in data if any(server.get(f"is_sector_{code}", False) for code in NAICS_SECTORS.keys()))
    servers_without_sector = total_servers - servers_with_sector
    
    print(f"\nServers classified in at least one sector: {servers_with_sector:,} ({(servers_with_sector/total_servers)*100:.2f}%)")
    print(f"Servers not classified in any sector: {servers_without_sector:,} ({(servers_without_sector/total_servers)*100:.2f}%)")
    
    # Top sectors summary
    top_5 = df.head(5)
    print(f"\nTop 5 Sectors by MCP Server Count:")
    for i, (_, row) in enumerate(top_5.iterrows(), 1):
        print(f"{i}. {row['Sector_Name']} ({row['NAICS_Code']}): {row['Server_Count']:,} servers ({row['Percentage']:.2f}%)")
    
    return df

if __name__ == "__main__":
    result_df = analyze_naics_sectors()
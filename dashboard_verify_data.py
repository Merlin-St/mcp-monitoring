#!/usr/bin/env python3
"""
Data Verification Script for MCP Dashboard
"""
import json
import pandas as pd
from pathlib import Path

def verify_unified_data():
    """Verify the unified data file"""
    print("🔍 Verifying unified MCP data...")
    
    data_file = Path("dashboard_mcp_servers_unified.json")
    summary_file = Path("dashboard_mcp_servers_unified_summary.json")
    
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        return False
    
    if not summary_file.exists():
        print(f"❌ Summary file not found: {summary_file}")
        return False
    
    try:
        # Load and verify main data
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        print(f"✅ Loaded {len(data)} servers")
        
        # Create DataFrame
        df = pd.DataFrame(data)
        print(f"✅ DataFrame shape: {df.shape}")
        
        # Check required columns
        required_cols = ['id', 'name', 'data_sources']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return False
        
        print("✅ All required columns present")
        
        # Check data sources
        sources = df['data_sources'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
        source_counts = sources.value_counts()
        print(f"✅ Data source distribution:")
        for source, count in source_counts.head().items():
            print(f"   {source}: {count}")
        
        # Check finance classification
        finance_count = len(df[df.get('is_finance_related', False) == True])
        print(f"✅ Finance-related servers: {finance_count}")
        
        # Load summary
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        print(f"✅ Summary data loaded")
        print(f"   Total servers: {summary.get('total_servers')}")
        print(f"   Finance servers: {summary.get('finance_related_servers')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verifying data: {e}")
        return False

def main():
    """Main verification"""
    success = verify_unified_data()
    
    if success:
        print("\n✅ Data verification passed! Dashboard should work correctly.")
        print("Launch with: python dashboard_launch.py")
    else:
        print("\n❌ Data verification failed!")
        print("Run data processor: python dashboard_unified_mcp_data_processor.py")
    
    return success

if __name__ == "__main__":
    main()
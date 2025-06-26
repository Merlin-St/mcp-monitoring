#!/usr/bin/env python3
"""
Dashboard Launcher Script

This script helps launch the MCP dashboard with proper error checking.
"""
import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check if we're in the right directory
    if not Path("data_unified.json").exists():
        print("❌ Unified data file not found!")
        print("Please run: python dashboard_unified_mcp_data_processor.py")
        return False
    
    # Check virtual environment
    if 'VIRTUAL_ENV' not in os.environ:
        print("❌ Virtual environment not activated!")
        print("Please run: source ~/si_setup/.venv/bin/activate")
    return False
    
    # Check Streamlit
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__} found")
    except ImportError:
        print("❌ Streamlit not installed!")
        print("Please install: pip install streamlit")
        return False
    
    # Check data size
    data_size = Path("data_unified.json").stat().st_size / (1024*1024)
    print(f"✅ Data file: {data_size:.1f} MB")
    
    return True

def launch_dashboard(port=8501):
    """Launch the dashboard"""
    if not check_requirements():
        return False
    
    print(f"🚀 Launching dashboard on port {port}...")
    print(f"📱 Access at: http://localhost:{port}")
    print("   Or external: http://18.132.184.2:{port}")
    print("🛑 Press Ctrl+C to stop")
    
    try:
        # Launch Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "dashboard_unified_mcp.py",
            "--server.port", str(port),
            "--server.address", "0.0.0.0",
            "--server.headless", "true"
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        return False
    
    return True

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch MCP Dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Port to run on")
    parser.add_argument("--dashboard", choices=["unified"], 
                       default="unified", help="Dashboard type to launch")
    parser.add_argument("--filtered", action="store_true", 
                       help="Use filtered dataset (data_unified_filtered.json)")
    
    args = parser.parse_args()
    
    dashboard_files = {
        "unified": "dashboard_unified_mcp.py"
    }
    
    dashboard_file = dashboard_files[args.dashboard]
    
    print(f"🚀 Launching {args.dashboard} dashboard on port {args.port}...")
    print(f"📱 Access at: http://localhost:{args.port}")
    print("   Or external: http://18.132.184.2:{args.port}")
    print("🛑 Press Ctrl+C to stop")
    
    if args.dashboard != "test" and not check_requirements():
        return False
    
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            dashboard_file,
            "--server.port", str(args.port),
            "--server.address", "0.0.0.0",
            "--server.headless", "true"
        ]
        
        # Add --filtered flag if specified
        if args.filtered:
            cmd.append("--")
            cmd.append("--filtered")
        
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print(f"\n👋 {args.dashboard.title()} dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
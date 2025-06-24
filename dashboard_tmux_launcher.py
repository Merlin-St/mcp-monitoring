#!/usr/bin/env python3
"""
Tmux-based Dashboard Launcher Script

This script manages persistent Streamlit sessions using tmux for reliable dashboard hosting.
Solves the issue where Streamlit loads forever after the first run by maintaining persistent sessions.
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

class TmuxDashboardManager:
    def __init__(self):
        self.session_prefix = "streamlit_"
        self.default_port = 8501
        
    def check_tmux(self):
        """Check if tmux is available"""
        try:
            result = subprocess.run(['tmux', '-V'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {result.stdout.strip()}")
                return True
        except FileNotFoundError:
            print("❌ tmux not found! Please install: sudo apt install tmux")
            return False
        return False
    
    def check_virtual_env(self):
        """Check if virtual environment is activated"""
        if 'VIRTUAL_ENV' not in os.environ:
            print("❌ Virtual environment not activated!")
            print("Please run: source ~/si_setup/.venv/bin/activate")
            return False
        print("✅ Virtual environment activated")
        return True
    
    def check_data_files(self, dashboard_type):
        """Check if required data files exist"""
        required_files = {
            'unified': 'data_unified.json',
            'finance': 'data_unified.json',
            'smithery': 'smithery_all_mcp_server_summaries.json'
        }
        
        if dashboard_type in required_files:
            file_path = Path(required_files[dashboard_type])
            if not file_path.exists():
                print(f"❌ Data file not found: {file_path}")
                return False
            size_mb = file_path.stat().st_size / (1024*1024)
            print(f"✅ Data file: {file_path} ({size_mb:.1f} MB)")
        return True
    
    def get_session_name(self, dashboard_type, port):
        """Generate tmux session name"""
        return f"{self.session_prefix}{dashboard_type}_{port}"
    
    def session_exists(self, session_name):
        """Check if tmux session exists"""
        try:
            result = subprocess.run(['tmux', 'has-session', '-t', session_name], 
                                  capture_output=True)
            return result.returncode == 0
        except:
            return False
    
    def list_sessions(self):
        """List all streamlit tmux sessions"""
        try:
            result = subprocess.run(['tmux', 'list-sessions'], capture_output=True, text=True)
            if result.returncode == 0:
                sessions = [line for line in result.stdout.split('\n') 
                           if self.session_prefix in line]
                if sessions:
                    print("🖥️  Active Streamlit Sessions:")
                    for session in sessions:
                        print(f"   {session}")
                    return sessions
                else:
                    print("📭 No active Streamlit sessions found")
                    return []
            else:
                print("📭 No tmux sessions running")
                return []
        except Exception as e:
            print(f"❌ Error listing sessions: {e}")
            return []
    
    def start_dashboard(self, dashboard_type, port=None):
        """Start dashboard in tmux session"""
        if port is None:
            port = self.default_port
        
        session_name = self.get_session_name(dashboard_type, port)
        
        # Check prerequisites
        if not self.check_tmux():
            return False
        if not self.check_virtual_env():
            return False
        if not self.check_data_files(dashboard_type):
            return False
        
        # Check if session already exists
        if self.session_exists(session_name):
            print(f"⚠️  Session '{session_name}' already exists!")
            print(f"   Use 'attach' to connect or 'stop' to terminate first")
            return False
        
        # Dashboard file mapping
        dashboard_files = {
            'unified': 'dashboard_unified_mcp.py',
            'finance': 'dashboard_finance_mcp.py',
            'smithery': 'dashboard_smithery_local_mcp.py'
        }
        
        if dashboard_type not in dashboard_files:
            print(f"❌ Unknown dashboard type: {dashboard_type}")
            print(f"   Available: {', '.join(dashboard_files.keys())}")
            return False
        
        dashboard_file = dashboard_files[dashboard_type]
        
        # Prepare streamlit command
        venv_python = os.path.join(os.environ['VIRTUAL_ENV'], 'bin', 'python')
        streamlit_cmd = f"{venv_python} -m streamlit run {dashboard_file} --server.port {port} --server.address 0.0.0.0 --server.headless true"
        
        # Create tmux session
        try:
            print(f"🚀 Starting {dashboard_type} dashboard in tmux session '{session_name}'...")
            
            # Create new detached session
            subprocess.run(['tmux', 'new-session', '-d', '-s', session_name])
            
            # Set working directory
            subprocess.run(['tmux', 'send-keys', '-t', session_name, f'cd {os.getcwd()}', 'Enter'])
            
            # Activate virtual environment in the session
            venv_path = os.environ['VIRTUAL_ENV']
            subprocess.run(['tmux', 'send-keys', '-t', session_name, f'source {venv_path}/bin/activate', 'Enter'])
            
            # Start streamlit
            subprocess.run(['tmux', 'send-keys', '-t', session_name, streamlit_cmd, 'Enter'])
            
            # Wait a moment for startup
            time.sleep(3)
            
            print(f"✅ Dashboard started successfully!")
            print(f"📱 Access at: http://localhost:{port}")
            print(f"🔗 External: http://18.132.184.2:{port}")
            print(f"📺 Session: {session_name}")
            print(f"🔗 To attach: tmux attach -t {session_name}")
            print(f"🛑 To stop: python {sys.argv[0]} stop {dashboard_type} --port {port}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error starting dashboard: {e}")
            return False
    
    def stop_dashboard(self, dashboard_type, port=None):
        """Stop dashboard tmux session"""
        if port is None:
            port = self.default_port
        
        session_name = self.get_session_name(dashboard_type, port)
        
        if not self.session_exists(session_name):
            print(f"❌ Session '{session_name}' not found")
            return False
        
        try:
            subprocess.run(['tmux', 'kill-session', '-t', session_name])
            print(f"🛑 Dashboard session '{session_name}' stopped")
            return True
        except Exception as e:
            print(f"❌ Error stopping session: {e}")
            return False
    
    def attach_dashboard(self, dashboard_type, port=None):
        """Attach to existing dashboard session"""
        if port is None:
            port = self.default_port
        
        session_name = self.get_session_name(dashboard_type, port)
        
        if not self.session_exists(session_name):
            print(f"❌ Session '{session_name}' not found")
            self.list_sessions()
            return False
        
        print(f"🔗 Attaching to session '{session_name}'...")
        print("   Press Ctrl+B then D to detach")
        
        try:
            # Replace current process with tmux attach
            os.execvp('tmux', ['tmux', 'attach', '-t', session_name])
        except Exception as e:
            print(f"❌ Error attaching to session: {e}")
            return False
    
    def status(self):
        """Show status of all dashboard sessions"""
        print("🖥️  Dashboard Session Status")
        print("=" * 40)
        sessions = self.list_sessions()
        
        if sessions:
            print("\n🔧 Management Commands:")
            for session_line in sessions:
                session_name = session_line.split(':')[0]
                parts = session_name.replace(self.session_prefix, '').split('_')
                if len(parts) >= 2:
                    dashboard_type = parts[0]
                    port = parts[1]
                    print(f"   Attach: python {sys.argv[0]} attach {dashboard_type} --port {port}")
                    print(f"   Stop:   python {sys.argv[0]} stop {dashboard_type} --port {port}")
        
        return len(sessions) > 0

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Tmux-based MCP Dashboard Manager")
    parser.add_argument('action', choices=['start', 'stop', 'attach', 'status', 'list'],
                       help='Action to perform')
    parser.add_argument('dashboard', nargs='?', 
                       choices=['unified', 'finance', 'smithery'],
                       help='Dashboard type (required for start/stop/attach)')
    parser.add_argument('--port', type=int, default=8501,
                       help='Port number (default: 8501)')
    
    args = parser.parse_args()
    
    manager = TmuxDashboardManager()
    
    if args.action == 'status' or args.action == 'list':
        manager.status()
    elif args.action in ['start', 'stop', 'attach']:
        if not args.dashboard:
            print("❌ Dashboard type required for this action")
            print("   Available: unified, finance, smithery")
            return False
        
        if args.action == 'start':
            return manager.start_dashboard(args.dashboard, args.port)
        elif args.action == 'stop':
            return manager.stop_dashboard(args.dashboard, args.port)
        elif args.action == 'attach':
            return manager.attach_dashboard(args.dashboard, args.port)
    
    return True

if __name__ == "__main__":
    main()
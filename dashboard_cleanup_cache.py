#!/usr/bin/env python3
"""
Dashboard cache cleanup utility to reduce zombie connections and clear caches.
Handles cache cleanup and session cleanup for better resource management.
"""

import os
import sys
import subprocess
import logging
import shutil
import glob
import signal
import time
import argparse
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dashboard_cleanup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def cleanup_streamlit_cache():
    """Clear Streamlit cache directories"""
    logger.info("Starting Streamlit cache cleanup...")
    
    # Common Streamlit cache locations
    cache_paths = [
        os.path.expanduser("~/.streamlit"),
        ".streamlit",
        "/tmp/.streamlit",
        "/tmp/streamlit*",
        os.path.expanduser("~/.cache/streamlit"),
    ]
    
    for cache_path in cache_paths:
        if "*" in cache_path:
            # Handle glob patterns
            for path in glob.glob(cache_path):
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                        logger.info(f"Removed cache directory: {path}")
                    elif os.path.isfile(path):
                        os.remove(path)
                        logger.info(f"Removed cache file: {path}")
                except Exception as e:
                    logger.warning(f"Could not remove {path}: {e}")
        else:
            try:
                if os.path.exists(cache_path):
                    if os.path.isdir(cache_path):
                        shutil.rmtree(cache_path)
                        logger.info(f"Removed cache directory: {cache_path}")
                    else:
                        os.remove(cache_path)
                        logger.info(f"Removed cache file: {cache_path}")
            except Exception as e:
                logger.warning(f"Could not remove {cache_path}: {e}")

def cleanup_streamlit_sessions():
    """Gracefully stop all Streamlit processes"""
    logger.info("Starting Streamlit session cleanup...")
    
    try:
        # Find all Streamlit processes
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        streamlit_pids = []
        
        for line in result.stdout.split('\n'):
            if 'streamlit' in line and 'python' in line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        pid = int(parts[1])
                        streamlit_pids.append(pid)
                        logger.info(f"Found Streamlit process: PID {pid}")
                    except ValueError:
                        continue
        
        # Gracefully terminate processes
        for pid in streamlit_pids:
            try:
                logger.info(f"Sending SIGTERM to process {pid}")
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                logger.info(f"Process {pid} already terminated")
            except Exception as e:
                logger.warning(f"Could not terminate process {pid}: {e}")
        
        # Wait a bit for graceful shutdown
        if streamlit_pids:
            logger.info("Waiting 3 seconds for graceful shutdown...")
            time.sleep(3)
        
        # Force kill any remaining processes
        for pid in streamlit_pids:
            try:
                os.kill(pid, signal.SIGKILL)
                logger.info(f"Force killed process {pid}")
            except ProcessLookupError:
                pass  # Already terminated
            except Exception as e:
                logger.warning(f"Could not force kill process {pid}: {e}")
                
    except Exception as e:
        logger.error(f"Error during session cleanup: {e}")

def cleanup_port_connections():
    """Clean up connections on Streamlit ports"""
    logger.info("Cleaning up port connections...")
    
    ports = [8501, 8502, 8503, 8504, 8505, 8506, 8507, 8508]
    
    for port in ports:
        try:
            # Kill processes using the port
            result = subprocess.run(['lsof', '-ti', f':{port}'], 
                                 capture_output=True, text=True)
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid:
                        try:
                            subprocess.run(['kill', '-9', pid], check=True)
                            logger.info(f"Killed process {pid} using port {port}")
                        except subprocess.CalledProcessError:
                            logger.warning(f"Could not kill process {pid}")
        except FileNotFoundError:
            logger.warning("lsof not available")
        except Exception as e:
            logger.warning(f"Error cleaning port {port}: {e}")

def cleanup_tmux_sessions():
    """Clean up tmux sessions related to Streamlit"""
    logger.info("Cleaning up tmux sessions...")
    
    try:
        # List tmux sessions
        result = subprocess.run(['tmux', 'list-sessions'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'streamlit' in line.lower():
                    session_name = line.split(':')[0]
                    try:
                        subprocess.run(['tmux', 'kill-session', '-t', session_name], 
                                     check=True)
                        logger.info(f"Killed tmux session: {session_name}")
                    except subprocess.CalledProcessError:
                        logger.warning(f"Could not kill tmux session: {session_name}")
        else:
            logger.info("No tmux server running")
            
    except FileNotFoundError:
        logger.info("tmux not available")
    except Exception as e:
        logger.warning(f"Error cleaning tmux sessions: {e}")

def quick_cleanup():
    """Quick cleanup of Streamlit processes and caches"""
    print("🧹 Quick dashboard cleanup starting...")
    
    # 1. Kill Streamlit processes
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        streamlit_pids = []
        
        for line in result.stdout.split('\n'):
            if 'streamlit' in line and 'python' in line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        pid = int(parts[1])
                        streamlit_pids.append(pid)
                    except ValueError:
                        continue
        
        for pid in streamlit_pids:
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"  ✓ Stopped Streamlit process {pid}")
            except:
                pass
        
        if streamlit_pids:
            time.sleep(2)  # Wait for graceful shutdown
            
    except Exception as e:
        print(f"  ⚠ Error stopping processes: {e}")
    
    # 2. Kill port connections
    for port in [8501, 8502, 8503, 8504, 8505, 8506, 8507, 8508]:
        try:
            subprocess.run(['lsof', '-ti', f':{port}'], 
                         capture_output=True, text=True, check=True)
            subprocess.run(f'lsof -ti:{port} | xargs -r kill -9', 
                         shell=True, capture_output=True)
            print(f"  ✓ Cleaned port {port}")
        except:
            pass
    
    # 3. Kill tmux sessions
    try:
        subprocess.run(['tmux', 'kill-server'], 
                      capture_output=True, text=True)
        print("  ✓ Killed tmux sessions")
    except:
        pass
    
    # 4. Clear Streamlit cache
    cache_dirs = [
        os.path.expanduser("~/.streamlit"),
        ".streamlit"
    ]
    
    for cache_dir in cache_dirs:
        try:
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                print(f"  ✓ Cleared cache: {cache_dir}")
        except:
            pass
    
    print("🎉 Quick cleanup complete!")

def main():
    """Run cleanup operations"""
    parser = argparse.ArgumentParser(description='Dashboard cleanup utility')
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick cleanup (faster, less verbose)')
    args = parser.parse_args()
    
    if args.quick:
        quick_cleanup()
    else:
        logger.info("=" * 50)
        logger.info("Starting dashboard cleanup operations")
        logger.info("=" * 50)
        
        # Run all cleanup operations
        cleanup_streamlit_sessions()
        cleanup_tmux_sessions() 
        cleanup_port_connections()
        cleanup_streamlit_cache()
        
        logger.info("=" * 50)
        logger.info("Dashboard cleanup completed")
        logger.info("=" * 50)

if __name__ == "__main__":
    main()
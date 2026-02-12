#!/bin/bash
# Wrapper to run pangram_detect.py with the API key from a file
export pangram_key=$(cat /home/ubuntu/.pangram_key)
cd /home/ubuntu/mcp-monitoring
exec uv run python3 scripts/data-classification-aicreatedmcp/agent1-pangram/pangram_detect.py 2>&1

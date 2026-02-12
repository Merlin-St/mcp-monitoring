#!/usr/bin/env python3
"""
Post-cleaning filter for data_unified_filtered.json.

Run AFTER the README cleaning LLM has populated readme_is_mcp_server and tools fields.
Removes:
  1. Servers explicitly marked as non-MCP (readme_is_mcp_server == 0)
  2. Servers with no tools extracted
"""

import json
import logging
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_unified_update_filtered_subset.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

FILTERED_PATH = 'data/initial/data_unified_filtered.json'
SUMMARY_PATH = 'data/initial/data_unified_filtered_summary.json'


def main():
    parser = argparse.ArgumentParser(description='Post-cleaning filter for data_unified_filtered.json')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be filtered without saving')
    args = parser.parse_args()

    logger.info(f"Loading {FILTERED_PATH}...")
    with open(FILTERED_PATH, 'r', encoding='utf-8') as f:
        servers = json.load(f)
    total_before = len(servers)
    logger.info(f"Loaded {total_before} servers")

    # Filter out servers explicitly marked as NOT MCP servers
    non_mcp = [s for s in servers if s.get('readme_is_mcp_server') == 0]
    servers = [s for s in servers if s.get('readme_is_mcp_server') != 0]
    logger.info(f"Filtered out {len(non_mcp)} non-MCP servers, keeping {len(servers)}")

    # Filter out servers with no tools
    before_tools = len(servers)
    servers = [s for s in servers if s.get('tools') and len(s['tools']) > 0]
    no_tools = before_tools - len(servers)
    logger.info(f"Filtered out {no_tools} servers with no tools, keeping {len(servers)}")

    total_removed = total_before - len(servers)
    logger.info(f"Total removed: {total_removed} ({total_removed / total_before * 100:.1f}%), remaining: {len(servers)}")

    if args.dry_run:
        logger.info("Dry run — not saving")
        return

    logger.info(f"Saving {len(servers)} servers to {FILTERED_PATH}...")
    with open(FILTERED_PATH, 'w', encoding='utf-8') as f:
        json.dump(servers, f, indent=2, ensure_ascii=False)

    # Update summary
    summary_path = Path(SUMMARY_PATH)
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        summary['total_servers'] = len(servers)
        summary['post_cleaning_filter'] = {
            'servers_before': total_before,
            'non_mcp_removed': len(non_mcp),
            'no_tools_removed': no_tools,
            'servers_after': len(servers),
        }
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Updated summary at {SUMMARY_PATH}")

    logger.info("Done")


if __name__ == '__main__':
    main()

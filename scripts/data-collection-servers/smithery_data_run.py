# %%
# =============================================================================
# Main Script to Run Bulk Download of All MCP Server Data
# =============================================================================
import json
import time
import sys
import os
from datetime import datetime

# Handle both direct execution and module imports
try:
    from .smithery_bulk_mcp_config import (
        bulk_logger as logger, SMITHERY_API_TOKEN, MCP_API_BASE_URL,
        MCP_PAGE_SIZE_BULK, API_DELAY_BULK, MCP_MAX_WORKERS_BULK
    )
    from .smithery_bulk_mcp_downloader import get_all_server_summaries, get_details_for_all_servers
except ImportError:
    from smithery_bulk_mcp_config import (
        bulk_logger as logger, SMITHERY_API_TOKEN, MCP_API_BASE_URL,
        MCP_PAGE_SIZE_BULK, API_DELAY_BULK, MCP_MAX_WORKERS_BULK
    )
    from smithery_bulk_mcp_downloader import get_all_server_summaries, get_details_for_all_servers

# Define the details file name locally since it's only used here
ALL_SERVERS_DETAILS_COMPLETE_JSON = "data/external-servers/smithery_data.json"
# If you switched to config_utils.py, make sure to import the correct variables
# e.g., API_DELAY_BULK_LIST as API_DELAY_BULK

def load_existing_data(filename):
    """Load existing smithery data from file"""
    if not os.path.exists(filename):
        logger.info(f"No existing file found at {filename}")
        return []

    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            logger.info(f"Loaded {len(data)} existing servers from {filename}")
            return data
    except Exception as e:
        logger.error(f"Error loading existing data from {filename}: {e}")
        return []

def merge_server_data(existing_servers, new_servers):
    """Merge existing and new server data, preferring newer data for duplicates"""
    # Create lookup by qualifiedName for existing servers
    existing_lookup = {s.get('qualifiedName'): s for s in existing_servers if s.get('qualifiedName')}

    # Update with new servers (overwrites duplicates)
    for new_server in new_servers:
        qname = new_server.get('qualifiedName')
        if qname:
            existing_lookup[qname] = new_server

    merged = list(existing_lookup.values())
    logger.info(f"Merged data: {len(existing_servers)} existing + {len(new_servers)} new = {len(merged)} unique servers")
    return merged

def main_bulk_download(resume_mode=False):
    logger.info("--- Starting Bulk MCP Server Data Download Process ---")
    if resume_mode:
        logger.info("RESUME MODE: Will merge with existing data")

    start_time_total = time.time()

    if not SMITHERY_API_TOKEN:
        logger.error("SMITHERY_API_TOKEN is not set. Please configure it in bulk_mcp_config.py or as an environment variable.")
        logger.error("Aborting bulk download.")
        return

    # --- Step 1: Fetch all server summaries ---
    logger.info("Step 1: Fetching all server summaries...")
    start_time_summaries = time.time()
    all_summaries = get_all_server_summaries(
        api_key=SMITHERY_API_TOKEN,
        base_url=MCP_API_BASE_URL,
        page_size=MCP_PAGE_SIZE_BULK,
        delay=API_DELAY_BULK # This should be API_DELAY_BULK_LIST if using config_utils.py
    )
    end_time_summaries = time.time()
    logger.info(f"Fetched {len(all_summaries)} server summaries in {end_time_summaries - start_time_summaries:.2f} seconds.")

    if not all_summaries:
        logger.error("No server summaries were fetched. Cannot proceed to download details. Exiting.")
        return



    # --- Step 2: Fetch full details for all servers ---
    logger.info("\nStep 2: Fetching full details for all collected server summaries...")
    start_time_details = time.time()
    
    # CRITICAL FIX: Pass the list of summary dictionaries directly
    # The get_details_for_all_servers function will handle extracting qnames internally if needed
    # or more accurately, its helper _fetch_single_server_details_and_merge expects the summary dict.
    
    # No need to pre-extract qualified_names_to_fetch here for the main call.
    # The validation for 'dict with qualifiedName' is inside get_details_for_all_servers.
    logger.info(f"Attempting to fetch details for {len(all_summaries)} server summaries.")
    
    all_server_details_list = get_details_for_all_servers(
        server_summaries=all_summaries, # Pass the list of summary dictionaries
        api_key=SMITHERY_API_TOKEN,
        base_url=MCP_API_BASE_URL,
        max_workers=MCP_MAX_WORKERS_BULK
    )
    end_time_details = time.time()
    logger.info(f"Fetched details for {len(all_server_details_list)} servers in {end_time_details - start_time_details:.2f} seconds.")

    if not all_server_details_list:
        logger.warning("No server details were successfully fetched.")
        return

    # --- Step 3: Merge with existing data if in resume mode ---
    final_server_list = all_server_details_list

    if resume_mode:
        logger.info("\nStep 3: Merging with existing data...")
        existing_servers = load_existing_data(ALL_SERVERS_DETAILS_COMPLETE_JSON)
        if existing_servers:
            final_server_list = merge_server_data(existing_servers, all_server_details_list)
        else:
            logger.info("No existing data to merge, using new data only")

    # --- Step 4: Save the final data ---
    try:
        with open(ALL_SERVERS_DETAILS_COMPLETE_JSON, 'w') as f:
            json.dump(final_server_list, f, indent=2)
        logger.info(f"Final data saved to {ALL_SERVERS_DETAILS_COMPLETE_JSON} ({len(final_server_list)} servers)")
    except IOError as e:
        logger.error(f"Error saving server details to {ALL_SERVERS_DETAILS_COMPLETE_JSON}: {e}")

    end_time_total = time.time()
    logger.info(f"--- Bulk MCP Server Data Download Process Finished in {end_time_total - start_time_total:.2f} seconds ---")

if __name__ == "__main__":
    # Check for --resume flag
    resume_mode = '--resume' in sys.argv or '-r' in sys.argv

    if resume_mode:
        logger.info("Running with --resume flag: will merge new data with existing smithery_data.json")

    main_bulk_download(resume_mode=resume_mode)

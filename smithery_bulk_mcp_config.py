# %%
# =============================================================================
# Configuration for Bulk MCP Server Data Download
# =============================================================================
import logging
import os

# --- API and File Configuration ---
# IMPORTANT: Set SMITHERY_API_TOKEN environment variable or replace "your-smithery-api-token-here"
with open(os.path.expanduser("~/.cache/smithery-api/token")) as f:
    SMITHERY_API_TOKEN = f.read().strip() # IMPORTANT: Set this environment variable in the setup file as aws secret
MCP_API_BASE_URL = "https://registry.smithery.ai"
MCP_REQUEST_TIMEOUT = 30  # Seconds for API request timeout
MCP_PAGE_SIZE_BULK = 10000 # Number of items per page for Smithery API
MCP_MAX_WORKERS_BULK = 20 # Max concurrent workers for fetching server details

# Output file for storing the list of all server summaries (from /servers endpoint)
ALL_SERVERS_SUMMARIES_JSON = "smithery_all_mcp_server_summaries.json"

API_DELAY_BULK = 0.05     # Small delay (in seconds) between paginated calls to the /servers list endpoint

# --- Logging Setup for Bulk Download ---
LOG_FILE_BULK = "bulk_mcp_download.log"

# Remove existing handlers if re-running in an interactive environment
for handler in logging.root.handlers[:]:
    if isinstance(handler, logging.FileHandler) and handler.baseFilename.endswith(LOG_FILE_BULK):
        logging.root.removeHandler(handler)
    # Avoid removing general console StreamHandler if shared across modules
    # For simplicity here, we'll re-add. In a larger app, manage logger instances.






logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
                    handlers=[
                        logging.FileHandler(LOG_FILE_BULK, mode='w'),
                        logging.StreamHandler(), # Console output
                    ])

bulk_logger = logging.getLogger(__name__)
bulk_logger.info("Bulk MCP Download Configuration and Logging Setup Complete.")

if SMITHERY_API_TOKEN == "your-smithery-api-token-here":
    bulk_logger.warning("SMITHERY_API_TOKEN is using the placeholder value. Bulk download will likely fail authentication.")


#!/usr/bin/env python3
"""
Official List Data Collection - Main Execution Script
Streamlined version that orchestrates URL extraction, HTML fetching, and GitHub metadata enhancement
"""

import os
import json
import time
import logging
import argparse
import asyncio
from datetime import datetime

# Import our modular components
from officiallist_url_extractor import MCPServerURLExtractor
from officiallist_html_fetcher import MCPServerHTMLFetcher
from officiallist_github_fetcher import OfficialistGitHubFetcher


class OfficialistDataRunner:
    def __init__(self):
        self.github_token = os.environ.get('GH_TOKEN')
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            file_handler = logging.FileHandler('officiallist_data_run.log')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.INFO)
    
    def run_url_extraction(self, fetch_history=True, history_months=None):
        """Step 1: Extract URLs from official README"""
        self.logger.info("=== STEP 1: URL Extraction ===")
        
        extractor = MCPServerURLExtractor()
        
        # Extract current URLs
        extractor.extract_current_urls()
        
        # Optionally fetch historical data
        if fetch_history:
            if history_months:
                extractor.fetch_readme_monthly_history(months_back=history_months)
            else:
                extractor.fetch_readme_history(days_back=30)
        
        # Save results
        extractor.save_urls('officiallist_urls.json')
        if extractor.historical_data:
            extractor.save_historical_data('officiallist_history.json')
        
        extractor.print_summary()
        
        self.logger.info(f"✓ URL extraction complete: {len(extractor.servers)} servers found")
        return len(extractor.servers)
    
    def run_html_fetching(self, test_mode=False):
        """Step 2: Fetch HTML content from external servers"""
        self.logger.info("=== STEP 2: HTML Content Fetching ===")
        
        # Load URLs from step 1
        try:
            with open('officiallist_urls.json', 'r', encoding='utf-8') as f:
                url_data = json.load(f)
            servers = url_data.get('servers', [])
        except FileNotFoundError:
            self.logger.error("officiallist_urls.json not found. Run URL extraction first.")
            return 0
        
        # Filter for external (non-GitHub) servers
        external_servers = [s for s in servers if not s.get('is_github', False)]
        
        if not external_servers:
            self.logger.info("No external servers found to fetch HTML content")
            return 0
        
        if test_mode:
            external_servers = external_servers[:5]
            self.logger.info(f"TEST MODE: Processing only {len(external_servers)} external servers")
        
        # Fetch HTML content
        fetcher = MCPServerHTMLFetcher(headless=True)
        
        try:
            fetcher.servers = servers  # Load all servers
            fetcher.fetch_external_html(max_external=len(external_servers) if not test_mode else 5)
            
            # Save enhanced results
            output_filename = 'officiallist_mcp_servers_with_html.json' if not test_mode else 'officiallist_mcp_servers_with_html_test.json'
            fetcher.save_results(output_filename)
            
            self.logger.info(f"✓ HTML fetching complete: {len(external_servers)} external servers processed")
            return len(external_servers)
            
        except Exception as e:
            self.logger.error(f"Error during HTML fetching: {e}")
            return 0
    
    async def run_github_enhancement(self, test_mode=False):
        """Step 3: Enhance GitHub servers with metadata"""
        self.logger.info("=== STEP 3: GitHub Metadata Enhancement ===")
        
        if not self.github_token:
            self.logger.warning("GH_TOKEN not found - skipping GitHub enhancement")
            return 0
        
        # Initialize GitHub fetcher
        github_fetcher = OfficialistGitHubFetcher(self.github_token)
        
        try:
            # Process GitHub servers
            repositories = await github_fetcher.process_github_servers(
                test_mode=test_mode, 
                resume=True
            )
            
            # Save final results
            await github_fetcher.save_progress(repositories, [])
            
            # Integrate with full dataset
            github_fetcher.integrate_with_officiallist_full()
            
            self.logger.info(f"✓ GitHub enhancement complete: {len(repositories)} repositories processed")
            return len(repositories)
            
        except Exception as e:
            self.logger.error(f"Error during GitHub enhancement: {e}")
            return 0
    
    
    async def run_complete_pipeline(self, test_mode=False, fetch_history=True, history_months=None, skip_html=False, skip_github=False):
        """Run the complete data collection pipeline"""
        self.logger.info("Starting complete officiallist data collection pipeline")
        self.logger.info(f"Test mode: {test_mode}")
        self.logger.info(f"Fetch history: {fetch_history}")
        self.logger.info(f"Skip HTML: {skip_html}")
        self.logger.info(f"Skip GitHub: {skip_github}")
        
        start_time = time.time()
        
        try:
            # Step 1: URL Extraction
            url_count = self.run_url_extraction(fetch_history=fetch_history, history_months=history_months)
            if url_count == 0:
                self.logger.error("URL extraction failed - stopping pipeline")
                return False
            
            # Step 2: HTML Fetching (optional)
            html_count = 0
            if not skip_html:
                html_count = self.run_html_fetching(test_mode=test_mode)
            else:
                self.logger.info("=== STEP 2: HTML Content Fetching (SKIPPED) ===")
            
            # Step 3: GitHub Enhancement (optional)
            github_count = 0
            if not skip_github:
                github_count = await self.run_github_enhancement(test_mode=test_mode)
            else:
                self.logger.info("=== STEP 3: GitHub Metadata Enhancement (SKIPPED) ===")
            
            
            # Summary
            duration = time.time() - start_time
            self.logger.info("\n" + "="*60)
            self.logger.info("PIPELINE COMPLETION SUMMARY")
            self.logger.info("="*60)
            self.logger.info(f"Duration: {duration:.1f} seconds")
            self.logger.info(f"URLs extracted: {url_count}")
            self.logger.info(f"HTML fetched: {html_count}")
            self.logger.info(f"GitHub enhanced: {github_count}")
            
            self.logger.info(f"\n✓ PIPELINE SUCCESSFUL - Data saved to: officiallist_data.json")
            return True
                
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False


async def main():
    parser = argparse.ArgumentParser(description='Complete officiallist MCP server data collection pipeline')
    parser.add_argument('--test', action='store_true', 
                       help='Test mode: process limited number of servers')
    parser.add_argument('--no-history', action='store_true',
                       help='Skip historical data collection')
    parser.add_argument('--history-months', type=int,
                       help='Number of months of history to collect (default: 30 days)')
    parser.add_argument('--skip-html', action='store_true',
                       help='Skip HTML content fetching from external servers')
    parser.add_argument('--skip-github', action='store_true',
                       help='Skip GitHub metadata enhancement')
    parser.add_argument('--urls-only', action='store_true',
                       help='Only extract URLs (skip HTML and GitHub steps)')
    args = parser.parse_args()
    
    # Setup logging for main function
    log_level = logging.DEBUG if args.test else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('officiallist_data_collection.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Log mode information
    if args.test:
        logger.info("Running in TEST MODE - processing limited servers")
    else:
        logger.info("Running in FULL MODE - processing all servers")
    
    # Check for GitHub token
    github_token = os.environ.get('GH_TOKEN')
    if not github_token and not args.skip_github:
        logger.warning("GH_TOKEN environment variable not set")
        logger.warning("Set it with: export GH_TOKEN=your_github_token")
        logger.warning("GitHub enhancement will be skipped...")
        args.skip_github = True
    
    # Handle urls-only mode
    if args.urls_only:
        args.skip_html = True
        args.skip_github = True
        logger.info("URLs-only mode: skipping HTML and GitHub steps")
    
    try:
        # Create and run pipeline
        runner = OfficialistDataRunner()
        
        success = await runner.run_complete_pipeline(
            test_mode=args.test,
            fetch_history=not args.no_history,
            history_months=args.history_months,
            skip_html=args.skip_html,
            skip_github=args.skip_github
        )
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
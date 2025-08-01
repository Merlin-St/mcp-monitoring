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


class OfficialistDataRunner:
    def __init__(self):
        self.github_token = os.environ.get('GH_TOKEN')
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            file_handler = logging.FileHandler('officiallist_data_run.log')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
            output_filename = 'officiallist_data_onlyhtml.json' if not test_mode else 'officiallist_data_onlyhtml_test.json'
            fetcher.save_results(output_filename)
            
            self.logger.info(f"✓ HTML fetching complete: {len(external_servers)} external servers processed")
            return len(external_servers)
            
        except Exception as e:
            self.logger.error(f"Error during HTML fetching: {e}")
            return 0
    
    async def run_github_enhancement(self, test_mode=False):
        """Step 3: Process GitHub servers with lean fetcher"""
        self.logger.info("=== STEP 3: GitHub Metadata Enhancement ===")
        
        if not self.github_token:
            self.logger.warning("GH_TOKEN not found - skipping GitHub enhancement")
            return 0
        
        # Initialize lean GitHub fetcher
        from officiallist_github_fetcher import OfficiallistGitHubFetcherLean
        github_fetcher = OfficiallistGitHubFetcherLean(self.github_token)
        
        try:
            # Process GitHub servers
            servers = await github_fetcher.process_all_github_servers(test_mode=test_mode)
            
            # Save results
            github_fetcher.save_results(servers)
            
            self.logger.info(f"✓ GitHub enhancement complete: {len(servers)} repositories processed")
            return len(servers)
            
        except Exception as e:
            self.logger.error(f"Error during GitHub enhancement: {e}")
            return 0
    
    def run_data_merger(self):
        """Step 4: Merge all data sources into final dataset"""
        self.logger.info("=== STEP 4: Data Merger ===")
        
        try:
            success = self.merge_all_data()
            if success:
                self.logger.info("✓ Data merger complete")
                return True
            else:
                self.logger.error("Data merger failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during data merger: {e}")
            return False
    
    def load_data_sources(self):
        """Load all data sources"""
        data_sources = {}
        
        # Load URL data (base data)
        try:
            with open('officiallist_urls.json', 'r', encoding='utf-8') as f:
                data_sources['urls'] = json.load(f)
                self.logger.info(f"Loaded URL data: {data_sources['urls']['total_servers']} servers")
        except FileNotFoundError:
            self.logger.error("officiallist_urls.json not found")
            return None
        
        # Load HTML data (check both test and production files)
        html_files = ['officiallist_data_onlyhtml.json', 'officiallist_data_onlyhtml_test.json']
        html_loaded = False
        
        for html_file in html_files:
            try:
                with open(html_file, 'r', encoding='utf-8') as f:
                    data_sources['html'] = json.load(f)
                    self.logger.info(f"Loaded HTML data from {html_file}: {data_sources['html']['processed_count']} servers")
                    html_loaded = True
                    break
            except FileNotFoundError:
                continue
        
        if not html_loaded:
            self.logger.warning("No HTML data files found - continuing without HTML data")
            data_sources['html'] = {'servers': []}
        
        # Load GitHub data  
        try:
            with open('officiallist_data_onlygithub.json', 'r', encoding='utf-8') as f:
                data_sources['github'] = json.load(f)
                self.logger.info(f"Loaded GitHub data: {data_sources['github']['processed_count']} servers")
        except FileNotFoundError:
            self.logger.warning("officiallist_data_onlygithub.json not found - continuing without GitHub data")
            data_sources['github'] = {'servers': []}
        
        return data_sources
    
    def create_lookups(self, data_sources):
        """Create URL-based lookups for efficient merging"""
        lookups = {}
        
        # HTML lookup by URL
        lookups['html'] = {}
        for server in data_sources['html'].get('servers', []):
            url = server.get('url')
            if url:
                lookups['html'][url] = server
        
        # GitHub lookup by URL
        lookups['github'] = {}
        for server in data_sources['github'].get('servers', []):
            url = server.get('url')
            if url:
                lookups['github'][url] = server
        
        self.logger.info(f"Created lookups: {len(lookups['html'])} HTML, {len(lookups['github'])} GitHub")
        return lookups
    
    def merge_server_data(self, base_server, html_data=None, github_data=None):
        """Merge data from different sources for a single server"""
        # Start with base server data from URLs
        merged = {
            'name': base_server.get('name'),
            'url': base_server.get('url'),
            'description': base_server.get('description', ''),
            'is_github': base_server.get('is_github', False),
            'extracted_date': base_server.get('extracted_date')
        }
        
        # Add HTML data if available
        if html_data:
            # HTML content and metadata
            if html_data.get('html_content'):
                merged['html_content'] = html_data['html_content']
            if html_data.get('readme_content'):  # HTML servers copy content to readme_content
                merged['readme_content'] = html_data['readme_content']
            if html_data.get('content_source'):
                merged['content_source'] = html_data['content_source']
            if html_data.get('fetch_status'):
                merged['fetch_status'] = html_data['fetch_status']
            if html_data.get('html_length'):
                merged['html_length'] = html_data['html_length']
            if html_data.get('fetch_timestamp'):
                merged['fetch_timestamp'] = html_data['fetch_timestamp']
        
        # Add GitHub data if available
        if github_data:
            # README content (overrides HTML content for GitHub servers)
            if github_data.get('readme_content'):
                merged['readme_content'] = github_data['readme_content']
            if github_data.get('content_source'):
                merged['content_source'] = github_data['content_source']
            if github_data.get('readme_path'):
                merged['readme_path'] = github_data['readme_path']
            
            # Processing status
            if github_data.get('processing_status'):
                merged['processing_status'] = github_data['processing_status']
            if github_data.get('error_message'):
                merged['error_message'] = github_data['error_message']
            
            # GitHub metadata
            if github_data.get('github_metadata'):
                merged['github_metadata'] = github_data['github_metadata']
            if github_data.get('github_info'):
                merged['github_info'] = github_data['github_info']
        
        # Set processing status if not already set
        if 'processing_status' not in merged:
            if merged.get('readme_content') or merged.get('html_content'):
                merged['processing_status'] = 'success'
            else:
                merged['processing_status'] = 'no_content'
        
        return merged
    
    def merge_all_data(self):
        """Merge all data sources into final dataset"""
        self.logger.info("Starting data merger for officiallist_data.json")
        
        # Load all data sources
        data_sources = self.load_data_sources()
        if not data_sources:
            return False
        
        # Create lookups
        lookups = self.create_lookups(data_sources)
        
        # Merge all servers
        merged_servers = []
        
        # Process each server from the base URL list
        for base_server in data_sources['urls'].get('servers', []):
            url = base_server.get('url')
            
            # Get corresponding data from other sources
            html_data = lookups['html'].get(url)
            github_data = lookups['github'].get(url)
            
            # Merge server data
            merged_server = self.merge_server_data(base_server, html_data, github_data)
            merged_servers.append(merged_server)
        
        # Calculate statistics
        total_servers = len(merged_servers)
        github_servers = sum(1 for s in merged_servers if s.get('is_github'))
        external_servers = total_servers - github_servers
        servers_with_content = sum(1 for s in merged_servers if s.get('readme_content') or s.get('html_content'))
        github_enhanced = sum(1 for s in merged_servers if s.get('github_metadata'))
        
        # Create final output
        final_data = {
            'fetch_date': datetime.now().isoformat(),
            'total_servers': total_servers,
            'github_servers': github_servers,
            'external_servers': external_servers,
            'servers_with_content': servers_with_content,
            'github_enhanced_count': github_enhanced,
            'processed_count': servers_with_content,
            'data_sources_merged': {
                'urls': data_sources['urls'].get('total_servers', 0),
                'html': len(lookups['html']),
                'github': len(lookups['github'])
            },
            'servers': merged_servers
        }
        
        # Save merged data
        with open('officiallist_data.json', 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        
        # Create and save summary
        self.create_summary(final_data, merged_servers)
        
        # Clean up intermediate files
        self.cleanup_intermediate_files()
        
        # Log summary
        self.logger.info("="*60)
        self.logger.info("DATA MERGER COMPLETION SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Total servers merged: {total_servers}")
        self.logger.info(f"  GitHub servers: {github_servers}")
        self.logger.info(f"  External servers: {external_servers}")
        self.logger.info(f"Servers with content: {servers_with_content}")
        self.logger.info(f"GitHub enhanced: {github_enhanced}")
        self.logger.info("Data sources merged:")
        self.logger.info(f"  URLs: {data_sources['urls'].get('total_servers', 0)}")
        self.logger.info(f"  HTML: {len(lookups['html'])}")
        self.logger.info(f"  GitHub: {len(lookups['github'])}")
        self.logger.info("Final dataset saved to: officiallist_data.json")
        
        return True
    
    def cleanup_intermediate_files(self):
        """Remove intermediate files after successful merger"""
        intermediate_files = [
            'officiallist_urls.json',
            'officiallist_data_onlyhtml.json', 
            'officiallist_data_onlyhtml_test.json',
            'officiallist_data_onlygithub.json'
        ]
        
        cleaned_count = 0
        for filename in intermediate_files:
            try:
                if os.path.exists(filename):
                    os.remove(filename)
                    self.logger.info(f"Cleaned up intermediate file: {filename}")
                    cleaned_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to remove {filename}: {e}")
        
        if cleaned_count > 0:
            self.logger.info(f"Successfully cleaned up {cleaned_count} intermediate files")
    
    def create_summary(self, final_data, merged_servers):
        """Create a summary file with key statistics"""
        # Extract GitHub repositories with language info
        github_repos_with_lang = []
        for server in merged_servers:
            if server.get('github_metadata') and server.get('is_github'):
                github_meta = server['github_metadata']
                github_repos_with_lang.append({
                    'name': server.get('name'),
                    'url': server.get('url'),
                    'language': github_meta.get('language'),
                    'stargazers_count': github_meta.get('stargazers_count', 0),
                    'forks_count': github_meta.get('forks_count', 0),
                    'created_at': github_meta.get('created_at'),
                    'has_readme': bool(server.get('readme_content'))
                })
        
        # Count repositories by language
        language_counts = {}
        for repo in github_repos_with_lang:
            lang = repo['language'] or 'null'
            language_counts[lang] = language_counts.get(lang, 0) + 1
        
        # Sort languages by count
        sorted_languages = dict(sorted(language_counts.items(), key=lambda x: x[1], reverse=True))
        
        # Count servers by processing status
        status_counts = {}
        for server in merged_servers:
            status = server.get('processing_status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Count content types
        content_stats = {
            'with_readme': sum(1 for s in merged_servers if s.get('readme_content')),
            'with_html': sum(1 for s in merged_servers if s.get('html_content')),
            'with_github_metadata': sum(1 for s in merged_servers if s.get('github_metadata')),
            'no_content': sum(1 for s in merged_servers if not s.get('readme_content') and not s.get('html_content'))
        }
        
        # Top repositories by stars (if available)
        top_starred = sorted(
            [repo for repo in github_repos_with_lang if repo['stargazers_count'] > 0],
            key=lambda x: x['stargazers_count'],
            reverse=True
        )[:20]
        
        # Create summary data with clearer distinctions
        summary_data = {
            'total_servers': final_data['total_servers'],
            'collection_date': final_data['fetch_date'],
            'github_servers': final_data['github_servers'],
            'external_servers': final_data['external_servers'],
            'servers_with_content': final_data['servers_with_content'],
            'github_enhanced_count': final_data['github_enhanced_count'],
            'pipeline_processing': {
                'urls_extracted': final_data['data_sources_merged']['urls'],
                'html_processed': final_data['data_sources_merged']['html'],
                'github_processed': final_data['data_sources_merged']['github']
            },
            'final_results': {
                'servers_with_readme': content_stats['with_readme'],
                'servers_with_html': content_stats['with_html'],
                'servers_with_github_metadata': content_stats['with_github_metadata'],
                'servers_no_content': content_stats['no_content']
            },
            'processing_status_counts': status_counts,
            'github_repositories_by_language': sorted_languages,
            'top_starred_repositories': top_starred,
            'language_distribution': {
                'total_languages': len([lang for lang in sorted_languages.keys() if lang != 'null']),
                'most_popular_language': max(sorted_languages.items(), key=lambda x: x[1])[0] if sorted_languages else None,
                'repositories_with_language': sum(count for lang, count in sorted_languages.items() if lang != 'null'),
                'repositories_without_language': sorted_languages.get('null', 0)
            },
            'server_breakdown': {
                'github_with_readme': sum(1 for s in merged_servers if s.get('is_github') and s.get('readme_content')),
                'github_without_readme': sum(1 for s in merged_servers if s.get('is_github') and not s.get('readme_content')),
                'external_with_html': sum(1 for s in merged_servers if not s.get('is_github') and s.get('html_content')),
                'external_without_content': sum(1 for s in merged_servers if not s.get('is_github') and not s.get('html_content'))
            }
        }
        
        # Save summary
        with open('officiallist_data_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info("Summary statistics saved to: officiallist_data_summary.json")
    
    
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
            
            # Step 4: Data Merger - Always run to merge available data
            self.logger.info("=== STEP 4: Data Merger ===")
            merger_success = self.run_data_merger()
            if not merger_success:
                self.logger.error("Data merger failed - pipeline incomplete")
                return False
            
            # Summary
            duration = time.time() - start_time
            self.logger.info("\n" + "="*60)
            self.logger.info("PIPELINE COMPLETION SUMMARY")
            self.logger.info("="*60)
            self.logger.info(f"Duration: {duration:.1f} seconds")
            self.logger.info(f"URLs extracted: {url_count}")
            self.logger.info(f"HTML fetched: {html_count}")
            self.logger.info(f"GitHub enhanced: {github_count}")
            self.logger.info(f"Data merger: {'✓ Success' if merger_success else '✗ Failed'}")
            
            self.logger.info("\n✓ PIPELINE SUCCESSFUL - Data saved to: officiallist_data.json")
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
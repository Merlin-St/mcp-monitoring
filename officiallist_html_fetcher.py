#!/usr/bin/env python3
"""
Official MCP Server HTML Fetcher
Fetches full HTML content from external (non-GitHub) MCP server URLs
"""

import json
import time
import logging
import argparse
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, WebDriverException

class MCPServerHTMLFetcher:
    def __init__(self, headless=True, batch_size=50):
        self.headless = headless
        self.batch_size = batch_size
        self.driver = None
        self.servers = []
        self.processed_count = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            file_handler = logging.FileHandler('officiallist_html_fetching.log')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.INFO)
    
    def setup_driver(self):
        """Setup Selenium Chrome driver with optimizations"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        
        # Performance optimizations
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1280,720")
        chrome_options.add_argument("--page-load-strategy=eager")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        # Disable unnecessary features for speed
        chrome_options.add_experimental_option("useAutomationExtension", False)
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.set_page_load_timeout(30)  # 30 second timeout
            self.logger.info("Chrome driver initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing Chrome driver: {e}")
            self.logger.error("Make sure you have Chrome and ChromeDriver installed")
            raise
    
    def close_driver(self):
        """Close the Selenium driver"""
        if self.driver:
            self.driver.quit()
            self.logger.info("Chrome driver closed")
    
    def wait_for_page_load(self, timeout=15):
        """Wait for basic page load"""
        try:
            WebDriverWait(self.driver, timeout).until(
                lambda driver: driver.execute_script("return document.readyState") == "complete"
            )
            # Brief wait for dynamic content
            time.sleep(1)
            return True
        except Exception as e:
            self.logger.warning(f"Error waiting for page load: {e}")
            return False
    
    def fetch_html_content(self, server):
        """Fetch HTML content for a single server"""
        url = server['url']
        self.logger.info(f"Fetching HTML for: {server['name']}")
        
        # Skip GitHub URLs
        if server.get('is_github', False):
            server['html_content'] = None
            server['fetch_status'] = 'skipped_github'
            server['html_length'] = 0
            server['fetch_timestamp'] = time.time()
            self.logger.info(f"  Skipped GitHub URL: {url}")
            return
        
        # Skip if already processed successfully
        if server.get('fetch_status') == 'success' and server.get('html_content'):
            self.logger.info(f"  Already processed: {server['name']}")
            return
        
        try:
            # Navigate to the URL
            self.driver.get(url)
            
            # Wait for page to load
            self.wait_for_page_load()
            
            # Get the full HTML content
            html_content = self.driver.page_source
            
            # Update server data
            server['html_content'] = html_content
            server['fetch_status'] = 'success'
            server['html_length'] = len(html_content)
            server['fetch_timestamp'] = time.time()
            
            self.logger.info(f"  Successfully fetched {len(html_content):,} characters")
            
        except TimeoutException:
            self.logger.warning(f"  Timeout for {url}")
            server['html_content'] = None
            server['fetch_status'] = 'timeout'
            server['html_length'] = 0
            server['fetch_timestamp'] = time.time()
            
        except WebDriverException as e:
            self.logger.warning(f"  WebDriver error for {url}: {e}")
            server['html_content'] = None
            server['fetch_status'] = 'webdriver_error'
            server['html_length'] = 0
            server['fetch_timestamp'] = time.time()
            
        except Exception as e:
            self.logger.error(f"  Unexpected error for {url}: {e}")
            server['html_content'] = None
            server['fetch_status'] = 'error'
            server['html_length'] = 0
            server['fetch_timestamp'] = time.time()
    
    def load_urls(self, filename):
        """Load URLs from extractor output JSON"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.servers = data.get('servers', [])
            self.logger.info(f"Loaded {len(self.servers)} servers from {filename}")
            
            # Count external (non-GitHub) servers
            external_servers = [s for s in self.servers if not s.get('is_github', False)]
            self.logger.info(f"  External servers to fetch: {len(external_servers)}")
            
            return True
            
        except FileNotFoundError:
            self.logger.error(f"URL file not found: {filename}")
            return False
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in {filename}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error loading URLs: {e}")
            return False
    
    def save_progress(self, filename, force=False):
        """Save current progress to file"""
        if not force and self.processed_count % self.batch_size != 0:
            return
        
        output_data = {
            'fetch_date': datetime.now().isoformat(),
            'total_servers': len(self.servers),
            'processed_count': self.processed_count,
            'servers': self.servers
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Progress saved to {filename} ({self.processed_count}/{len(self.servers)} servers)")
    
    def fetch_all_html(self, output_file, resume=False):
        """Fetch HTML for all external servers"""
        external_servers = [s for s in self.servers if not s.get('is_github', False)]
        
        if not external_servers:
            self.logger.warning("No external servers to fetch")
            return
        
        self.logger.info(f"Starting HTML fetch for {len(external_servers)} external servers")
        
        # Setup selenium driver
        self.setup_driver()
        
        try:
            start_time = time.time()
            
            for i, server in enumerate(external_servers):
                # Skip if resuming and already processed
                if resume and server.get('fetch_status') in ['success', 'skipped_github']:
                    continue
                
                # Fetch HTML content
                self.fetch_html_content(server)
                self.processed_count += 1
                
                # Progress reporting
                if i > 0 and (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed * 60  # servers per minute
                    remaining = len(external_servers) - (i + 1)
                    eta_minutes = remaining / rate if rate > 0 else 0
                    
                    self.logger.info(f"Progress: {i + 1}/{len(external_servers)} ({rate:.1f}/min, ETA: {eta_minutes:.0f}m)")
                
                # Save progress periodically
                if (i + 1) % self.batch_size == 0:
                    self.save_progress(output_file)
                
                # Rate limiting between requests
                time.sleep(0.5)
            
            # Final save
            self.save_progress(output_file, force=True)
            
            end_time = time.time()
            self.logger.info(f"HTML fetching completed in {(end_time - start_time)/60:.1f} minutes")
            
        finally:
            self.close_driver()
    
    def print_summary(self):
        """Print summary of HTML fetching results"""
        external_servers = [s for s in self.servers if not s.get('is_github', False)]
        
        if not external_servers:
            self.logger.info("No external servers processed")
            return
        
        # Count by status
        status_counts = {}
        total_html_size = 0
        
        for server in external_servers:
            status = server.get('fetch_status', 'pending')
            status_counts[status] = status_counts.get(status, 0) + 1
            
            if server.get('html_length', 0) > 0:
                total_html_size += server['html_length']
        
        self.logger.info("\n" + "="*60)
        self.logger.info("HTML FETCHING SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"External servers processed: {len(external_servers)}")
        
        for status, count in status_counts.items():
            self.logger.info(f"  {status}: {count}")
        
        if total_html_size > 0:
            self.logger.info(f"Total HTML collected: {total_html_size:,} characters")
        
        # Show successful fetches
        successful = [s for s in external_servers if s.get('fetch_status') == 'success']
        if successful:
            self.logger.info(f"\nSuccessfully fetched {len(successful)} servers:")
            for server in successful[:5]:  # Show first 5
                self.logger.info(f"  - {server['name']}: {server.get('html_length', 0):,} chars")

def main():
    parser = argparse.ArgumentParser(description='Fetch HTML from external MCP server URLs')
    parser.add_argument('--input', default='officiallist_urls.json',
                       help='Input JSON file with URLs from extractor')
    parser.add_argument('--output', default='officiallist_servers_with_html.json',
                       help='Output JSON file with HTML content')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Save progress every N servers')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous run (skip already processed)')
    parser.add_argument('--no-headless', action='store_true',
                       help='Run browser in non-headless mode (for debugging)')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('officiallist_html_fetching.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Create fetcher
    fetcher = MCPServerHTMLFetcher(
        headless=not args.no_headless,
        batch_size=args.batch_size
    )
    
    try:
        # Load URLs
        if not fetcher.load_urls(args.input):
            return 1
        
        # Fetch HTML content
        logger.info("Starting HTML fetching process...")
        fetcher.fetch_all_html(args.output, resume=args.resume)
        
        # Print summary
        fetcher.print_summary()
        
        logger.info(f"Results saved to {args.output}")
        
    except KeyboardInterrupt:
        logger.warning("HTML fetching interrupted by user")
        # Save current progress before exiting
        fetcher.save_progress(args.output, force=True)
        return 1
        
    except Exception as e:
        logger.error(f"Error during HTML fetching: {e}")
        # Save current progress before exiting
        try:
            fetcher.save_progress(args.output, force=True)
        except:
            pass
        raise

if __name__ == "__main__":
    main()
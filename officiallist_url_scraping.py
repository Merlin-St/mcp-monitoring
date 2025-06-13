import os
import re
import json
import time
import logging
import argparse
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict

class MCPServerScraper:
    def __init__(self, headless=True):
        self.base_url = "https://github.com/modelcontextprotocol/servers"
        self.servers = []
        self.github_token = os.environ.get('GH_TOKEN')
        self.headless = headless
        self.driver = None
        self.historical_data = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            file_handler = logging.FileHandler('officiallist_mcp_scraping.log')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.INFO)
        
    def setup_driver(self):
        """Setup Selenium Chrome driver with optimizations for speed"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        
        # Performance optimizations
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-images")
        chrome_options.add_argument("--disable-javascript")
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-css")
        chrome_options.add_argument("--window-size=1280,720")  # Smaller window
        chrome_options.add_argument("--page-load-strategy=eager")  # Don't wait for all resources
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        # Disable unnecessary features
        chrome_options.add_experimental_option("useAutomationExtension", False)
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
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
    
    def get_github_api_headers(self):
        """Get headers for GitHub API requests"""
        headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'MCP-Server-Scraper'
        }
        if self.github_token:
            headers['Authorization'] = f'token {self.github_token}'
        return headers
    
    def fetch_readme_history(self, days_back=30):
        """Fetch README commit history and content at different dates"""
        self.logger.info(f"Fetching README history for the last {days_back} days...")
        
        # Get commits for README.md
        commits_url = "https://api.github.com/repos/modelcontextprotocol/servers/commits"
        params = {
            'path': 'README.md',
            'since': (datetime.now() - timedelta(days=days_back)).isoformat()
        }
        
        try:
            response = requests.get(commits_url, headers=self.get_github_api_headers(), params=params)
            response.raise_for_status()
            commits = response.json()
            
            self.logger.info(f"Found {len(commits)} commits for README.md")
            
            # Fetch README content for each commit
            for commit in commits:
                commit_sha = commit['sha']
                commit_date = datetime.strptime(commit['commit']['author']['date'], '%Y-%m-%dT%H:%M:%SZ')
                
                # Get README content at this commit
                readme_url = f"https://api.github.com/repos/modelcontextprotocol/servers/contents/README.md?ref={commit_sha}"
                
                try:
                    response = requests.get(readme_url, headers=self.get_github_api_headers())
                    response.raise_for_status()
                    
                    # Decode base64 content
                    import base64
                    content = base64.b64decode(response.json()['content']).decode('utf-8')
                    
                    # Count MCP servers in this version
                    pattern = r'\[([^\]]+)\]\(([^)]+)\)\s*-\s*([^\n]+)'
                    matches = re.findall(pattern, content)
                    
                    self.historical_data.append({
                        'date': commit_date,
                        'commit_sha': commit_sha,
                        'server_count': len(matches),
                        'commit_message': commit['commit']['message']
                    })
                    
                    time.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    self.logger.warning(f"Error fetching README for commit {commit_sha}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error fetching commit history: {e}")
            
        # Sort by date
        self.historical_data.sort(key=lambda x: x['date'])
        
    def plot_mcp_growth(self):
        """Create a visualization of MCP server growth over time"""
        if not self.historical_data:
            self.logger.warning("No historical data available for plotting")
            return
            
        dates = [item['date'] for item in self.historical_data]
        counts = [item['server_count'] for item in self.historical_data]
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, counts, marker='o', linestyle='-', linewidth=2, markersize=6)
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.gcf().autofmt_xdate()
        
        # Labels and title
        plt.xlabel('Date')
        plt.ylabel('Number of MCP Servers')
        plt.title('MCP Server Growth Over Time')
        
        # Add annotations for significant changes
        for i in range(1, len(counts)):
            if counts[i] - counts[i-1] >= 5:  # Significant increase
                plt.annotate(f'+{counts[i] - counts[i-1]}', 
                           xy=(dates[i], counts[i]), 
                           xytext=(5, 5), 
                           textcoords='offset points',
                           fontsize=8,
                           color='green')
        
        plt.tight_layout()
        plt.savefig('mcp_growth_chart.png', dpi=300)
        self.logger.info("Growth chart saved as 'mcp_growth_chart.png'")
        
        # Also create a daily additions chart
        self.plot_daily_additions()
    
    def plot_daily_additions(self):
        """Create a bar chart of daily MCP server additions"""
        if len(self.historical_data) < 2:
            return
            
        dates = []
        additions = []
        
        for i in range(1, len(self.historical_data)):
            dates.append(self.historical_data[i]['date'])
            additions.append(self.historical_data[i]['server_count'] - 
                           self.historical_data[i-1]['server_count'])
        
        plt.figure(figsize=(12, 6))
        plt.bar(dates, additions, color=['green' if x > 0 else 'red' for x in additions])
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.gcf().autofmt_xdate()
        
        # Labels and title
        plt.xlabel('Date')
        plt.ylabel('Number of Servers Added')
        plt.title('Daily MCP Server Additions')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('mcp_daily_additions.png', dpi=300)
        self.logger.info("Daily additions chart saved as 'mcp_daily_additions.png'")
    
    def fetch_github_readme_api(self):
        """Fetch README content using GitHub API"""
        if not self.github_token:
            self.logger.warning("GH_TOKEN not found in environment variables")
            self.logger.warning("Using unauthenticated requests (lower rate limit)")
        
        api_url = "https://api.github.com/repos/modelcontextprotocol/servers/readme"
        headers = self.get_github_api_headers()
        headers['Accept'] = 'application/vnd.github.v3.raw'
        
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching README via API: {e}")
            return None
    
    def wait_for_basic_page_load(self, timeout=10):
        """Wait for basic page load - faster version"""
        try:
            # Just wait for document ready state - no scrolling or extra waits
            WebDriverWait(self.driver, timeout).until(
                lambda driver: driver.execute_script("return document.readyState") == "complete"
            )
            # Minimal wait for basic content
            time.sleep(1)
            return True
        except Exception as e:
            self.logger.warning(f"Error waiting for page load: {e}")
            return False
    
    # REMOVED: extract_tools_from_page method - moved to later analysis phase
    
    def fetch_page_selenium(self, url, max_wait=10):
        """Fetch a webpage using Selenium and get full HTML - optimized for speed"""
        try:
            self.driver.get(url)
            
            # Wait for basic page load only
            self.wait_for_basic_page_load(max_wait)
            
            # Get the full HTML immediately
            full_html = self.driver.page_source
            
            return full_html
            
        except TimeoutException:
            self.logger.warning(f"Timeout waiting for page: {url}")
            return None
        except WebDriverException as e:
            self.logger.error(f"WebDriver error for {url}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error fetching {url}: {e}")
            return None
    
    def parse_github_readme(self):
        """Parse the main GitHub README to extract MCP servers"""
        self.logger.info("Fetching README via GitHub API...")
        
        content = self.fetch_github_readme_api()
        
        if not content:
            self.logger.warning("Failed to fetch README via API, trying direct fetch...")
            readme_url = "https://raw.githubusercontent.com/modelcontextprotocol/servers/main/README.md"
            try:
                response = requests.get(readme_url)
                response.raise_for_status()
                content = response.text
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Failed to fetch README: {e}")
                return
        
        # Parse all markdown links from bullet points, with comprehensive pattern matching
        lines = content.split('\n')
        matches = []
        
        for line in lines:
            # Skip lines that don't start with bullet points
            if not line.strip().startswith('-'):
                continue
                
            # Skip navigation/header lines
            if any(skip in line.lower() for skip in ['github discussions', 'contributing', 'license', 'documentation']):
                continue
            
            # Extract all links from the line with improved regex
            link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
            link_matches = re.findall(link_pattern, line)
            
            for link_match in link_matches:
                name, url = link_match
                
                # Skip if it's clearly not a server (contains markdown formatting artifacts)
                if name.startswith('#') or url.startswith('#'):
                    continue
                
                # Extract description - everything after the first link
                desc_match = re.search(r'\[' + re.escape(name) + r'\]\([^)]+\)\*?\*?[^-]*(?:-\s*)?([^\[\n]*)', line)
                description = desc_match.group(1).strip() if desc_match else ""
                
                # Clean up description
                description = re.sub(r'<[^>]*>', '', description)  # Remove HTML tags
                description = re.sub(r'\*\*', '', description)      # Remove bold markdown
                description = description.strip()
                
                # If no description, try to get it from context around the link
                if not description and '-' in line:
                    parts = line.split('-', 1)
                    if len(parts) > 1:
                        description = parts[1].strip()
                        description = re.sub(r'<[^>]*>', '', description)
                        description = re.sub(r'\*\*', '', description)
                        description = description.strip()
                
                # Use "No description available" if still empty
                if not description:
                    description = "No description available"
                
                matches.append((name.strip(), url.strip(), description))
        
        self.logger.info(f"Total links found: {len(matches)}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_matches = []
        for match in matches:
            key = (match[0], match[1])  # name, url tuple
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)
        
        matches = unique_matches
        self.logger.info(f"Unique servers after deduplication: {len(matches)}")
        
        for match in matches:
            name, url, description = match
            
            # Clean up the description
            description = description.strip()
            
            # Handle relative URLs
            if url.startswith('/'):
                url = f"https://github.com{url}"
            elif not url.startswith('http'):
                url = urljoin(self.base_url, url)
            
            server_info = {
                'name': name.strip(),
                'url': url,
                'description': description,
                'full_html': None,
                'fetch_status': 'pending',
                'html_length': 0,
                'fetch_timestamp': None
            }
            
            self.servers.append(server_info)
        
        self.logger.info(f"Found {len(self.servers)} MCP servers")
    
    def fetch_full_html(self, server):
        """Fetch full HTML content from the server's URL - simplified for speed"""
        url = server['url']
        self.logger.info(f"Fetching HTML for: {server['name']}")
        
        # Skip if it's a local path reference
        if url.startswith('/modelcontextprotocol/servers/blob/'):
            server['full_html'] = "Local repository reference - part of the main MCP servers collection"
            server['fetch_status'] = "skipped_local"
            return
        
        try:
            full_html = self.fetch_page_selenium(url)
            if not full_html:
                server['full_html'] = "Failed to fetch HTML content"
                server['fetch_status'] = "failed"
                return
            
            # Store full HTML content (no truncation for now)
            server['full_html'] = full_html
            server['fetch_status'] = "success"
            server['html_length'] = len(full_html)
            server['fetch_timestamp'] = time.time()
            
            self.logger.info(f"  Successfully fetched {len(full_html)} characters for {server['name']}")
            
        except Exception as e:
            self.logger.error(f"Failed to fetch HTML for {server['name']} at {url}: {e}")
            server['full_html'] = f"Error fetching HTML: {str(e)}"
            server['fetch_status'] = "error"
    
    def scrape_all(self, fetch_context=True, max_contexts=None, fetch_history=True):
        """Main method to scrape all MCP servers"""
        # First, fetch historical data if requested
        if fetch_history:
            self.fetch_readme_history(days_back=30)
            self.plot_mcp_growth()
        
        # Parse the main README
        self.parse_github_readme()
        
        # Optionally fetch HTML content for each server
        if fetch_context:
            self.logger.info("Fetching HTML content for each server...")
            self.setup_driver()
            
            try:
                servers_to_process = self.servers[:max_contexts] if max_contexts else self.servers
                
                for i, server in enumerate(servers_to_process):
                    if i > 0 and i % 10 == 0:  # Progress every 10 servers
                        self.logger.info(f"Progress: {i}/{len(servers_to_process)}")
                        time.sleep(1)  # Minimal rate limiting
                    
                    try:
                        self.fetch_full_html(server)
                    except Exception as e:
                        self.logger.error(f"Error processing server {server['name']}: {e}")
                        server['full_html'] = f"Error processing: {str(e)}"
                        server['fetch_status'] = "error"
                    time.sleep(0.5)  # Shorter delay between requests
                    
            finally:
                self.close_driver()
        
        return self.servers
    
    def save_results(self, filename='mcp_servers.json', format='json'):
        """Save the results to a file - simplified for HTML storage"""
        if format == 'json':
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.servers, f, indent=2, ensure_ascii=False, default=str)
            self.logger.info(f"Results saved to {filename}")
        else:
            self.logger.info(f"Format {format} not supported in simplified version")
        
        # Also save historical data
        if self.historical_data:
            with open('mcp_history.json', 'w', encoding='utf-8') as f:
                json.dump(self.historical_data, f, indent=2, default=str)
            self.logger.info("Historical data saved to 'mcp_history.json'")
    
    def print_summary(self):
        """Print a summary of the scraped servers"""
        self.logger.info("\n" + "="*60)
        self.logger.info(f"Total MCP servers found: {len(self.servers)}")
        self.logger.info("="*60 + "\n")
        
        # Group by category
        categories = defaultdict(list)
        for server in self.servers:
            if '/modelcontextprotocol/servers/blob/' in server['url']:
                category = 'Official Reference Implementations'
            elif 'github.com' in server['url']:
                category = 'GitHub Projects'
            else:
                category = 'External Resources'
            categories[category].append(server)
        
        # Print category summary with fetch status
        for category, servers in categories.items():
            self.logger.info(f"\n{category}: {len(servers)} servers")
            
            # Show fetch status summary
            status_counts = defaultdict(int)
            total_html_size = 0
            for server in servers:
                status = server.get('fetch_status', 'unknown')
                status_counts[status] += 1
                if server.get('html_length'):
                    total_html_size += server['html_length']
            
            self.logger.info("  Fetch status:")
            for status, count in status_counts.items():
                self.logger.info(f"    - {status}: {count}")
            
            if total_html_size > 0:
                self.logger.info(f"  Total HTML collected: {total_html_size:,} characters")
        
        # Historical summary
        if self.historical_data:
            self.logger.info("\n" + "="*60)
            self.logger.info("Historical Growth Summary:")
            self.logger.info(f"  Period: {self.historical_data[0]['date'].date()} to {self.historical_data[-1]['date'].date()}")
            self.logger.info(f"  Total growth: {self.historical_data[-1]['server_count'] - self.historical_data[0]['server_count']} servers")
            self.logger.info(f"  Number of updates: {len(self.historical_data)}")

def main():
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description='Scrape MCP servers from the official repository')
    parser.add_argument('--test', action='store_true', 
                       help='Test mode: process only the first 10 servers')
    parser.add_argument('--no-history', action='store_true',
                       help='Skip historical data collection')
    parser.add_argument('--no-context', action='store_true',
                       help='Skip fetching additional context from individual server pages')
    args = parser.parse_args()
    
    # Setup logging for main function
    log_level = logging.DEBUG if args.test else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('officiallist_mcp_scraping.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Log mode information
    if args.test:
        logger.info("Running in TEST MODE - processing only first 10 servers")
    else:
        logger.info("Running in FULL MODE - processing all servers")
    
    # Check for GitHub token
    if not os.environ.get('GH_TOKEN'):
        logger.warning("GH_TOKEN environment variable not set")
        logger.warning("Set it with: export GH_TOKEN=your_github_token")
        logger.warning("Continuing without authentication...\n")
    
    # Create scraper instance
    scraper = MCPServerScraper(headless=True)
    
    try:
        # Determine processing parameters based on arguments
        max_contexts = 10 if args.test else None
        fetch_context = not args.no_context
        fetch_history = not args.no_history
        
        # Scrape servers with parameters
        servers = scraper.scrape_all(
            fetch_context=fetch_context,
            max_contexts=max_contexts,
            fetch_history=fetch_history
        )
        
        # Save results with appropriate filenames
        if args.test:
            filename_base = 'mcp_servers_test'
        else:
            filename_base = 'mcp_servers'
            
        scraper.save_results(f'{filename_base}.json', format='json')
        
        # Print summary
        scraper.print_summary()
        
        # Example: Print first server with basic details
        logger.info("\n\nExample server details:")
        for server in servers[:1]:
            logger.info(f"\nName: {server['name']}")
            logger.info(f"URL: {server['url']}")
            logger.info(f"Description: {server['description']}")
            logger.info(f"Fetch Status: {server.get('fetch_status', 'N/A')}")
            html_len = server.get('html_length', 0)
            if html_len > 0:
                logger.info(f"HTML Length: {html_len:,} characters")
            else:
                logger.info("HTML Length: No HTML content")
        
        # Log completion message
        total_processed = len(servers)
        if args.test:
            logger.info(f"\nTEST MODE COMPLETE: Processed {total_processed} servers (limited to first 10)")
        else:
            logger.info(f"\nFULL SCRAPE COMPLETE: Processed {total_processed} servers")
                
    except Exception as e:
        logger.error(f"Error during scraping: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
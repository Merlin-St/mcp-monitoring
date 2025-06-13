#!/usr/bin/env python3
"""
Official MCP Server URL Extractor
Extracts all URLs from modelcontextprotocol/servers README with optional historical analysis
"""

import os
import re
import json
import time
import logging
import argparse
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
import requests

class MCPServerURLExtractor:
    def __init__(self):
        self.github_token = os.environ.get('GH_TOKEN')
        self.base_url = "https://github.com/modelcontextprotocol/servers"
        self.servers = []
        self.historical_data = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            file_handler = logging.FileHandler('officiallist_url_extraction.log')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.INFO)
    
    def get_github_api_headers(self):
        """Get headers for GitHub API requests"""
        headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'MCP-URL-Extractor'
        }
        if self.github_token:
            headers['Authorization'] = f'token {self.github_token}'
        return headers
    
    def fetch_readme_content(self):
        """Fetch current README content using GitHub API"""
        self.logger.info("Fetching current README via GitHub API...")
        
        # Try GitHub API first
        api_url = "https://api.github.com/repos/modelcontextprotocol/servers/readme"
        headers = self.get_github_api_headers()
        headers['Accept'] = 'application/vnd.github.v3.raw'
        
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"GitHub API failed: {e}, trying direct fetch...")
            
            # Fallback to direct raw URL
            try:
                readme_url = "https://raw.githubusercontent.com/modelcontextprotocol/servers/main/README.md"
                response = requests.get(readme_url)
                response.raise_for_status()
                return response.text
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Failed to fetch README: {e}")
                return None
    
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
            
            self.logger.info(f"Found {len(commits)} README commits in the last {days_back} days")
            
            for commit in commits:
                commit_date = datetime.fromisoformat(commit['commit']['committer']['date'].replace('Z', '+00:00'))
                commit_sha = commit['sha']
                
                # Get README content at this commit
                content_url = f"https://api.github.com/repos/modelcontextprotocol/servers/contents/README.md"
                params = {'ref': commit_sha}
                
                try:
                    content_response = requests.get(content_url, headers=self.get_github_api_headers(), params=params)
                    if content_response.status_code == 200:
                        content_data = content_response.json()
                        if content_data.get('encoding') == 'base64':
                            import base64
                            content = base64.b64decode(content_data['content']).decode('utf-8', errors='ignore')
                            
                            # Parse servers from this historical content
                            servers = self.parse_servers_from_content(content)
                            
                            self.historical_data.append({
                                'date': commit_date,
                                'commit_sha': commit_sha,
                                'server_count': len(servers),
                                'commit_message': commit['commit']['message'],
                                'servers': servers
                            })
                            
                            self.logger.info(f"  {commit_date.date()}: {len(servers)} servers")
                            
                except requests.exceptions.RequestException as e:
                    self.logger.warning(f"Failed to get content for commit {commit_sha}: {e}")
                    continue
                    
                # Rate limiting
                time.sleep(0.1)
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching commit history: {e}")
    
    def fetch_readme_monthly_history(self, months_back=8):
        """Fetch README content at monthly intervals going back N months"""
        self.logger.info(f"Fetching README history for the last {months_back} months...")
        
        # Calculate target dates (1st of each month going back)
        current_date = datetime.now()
        target_dates = []
        
        for i in range(months_back + 1):  # Include current month
            if i == 0:
                # Current version
                target_date = current_date
                target_dates.append(('current', target_date))
            else:
                # Go back i months
                year = current_date.year
                month = current_date.month - i
                
                # Handle year boundary
                while month <= 0:
                    month += 12
                    year -= 1
                
                # Use first day of the month
                target_date = datetime(year, month, 1)
                target_dates.append((f'{i}_months_ago', target_date))
        
        date_strs = [f"{label}: {date.strftime('%Y-%m-%d')}" for label, date in target_dates]
        self.logger.info(f"Target dates: {date_strs}")
        
        # For each target date, find the closest commit
        for label, target_date in target_dates:
            try:
                if label == 'current':
                    # Get current README
                    content = self.fetch_readme_content()
                    if content:
                        servers = self.parse_servers_from_content(content)
                        self.historical_data.append({
                            'label': label,
                            'date': target_date,
                            'commit_sha': 'current',
                            'server_count': len(servers),
                            'commit_message': 'Current version',
                            'servers': servers
                        })
                        self.logger.info(f"  {label} ({target_date.strftime('%Y-%m-%d')}): {len(servers)} servers")
                else:
                    # Find closest commit to target date
                    commit_sha = self.find_closest_commit_to_date(target_date)
                    
                    if commit_sha:
                        content = self.fetch_readme_at_commit(commit_sha)
                        if content:
                            servers = self.parse_servers_from_content(content)
                            
                            # Get commit info
                            commit_info = self.get_commit_info(commit_sha)
                            actual_date = commit_info.get('date', target_date) if commit_info else target_date
                            commit_message = commit_info.get('message', 'Unknown') if commit_info else 'Unknown'
                            
                            self.historical_data.append({
                                'label': label,
                                'date': actual_date,
                                'commit_sha': commit_sha,
                                'server_count': len(servers),
                                'commit_message': commit_message,
                                'servers': servers
                            })
                            
                            self.logger.info(f"  {label} ({actual_date.strftime('%Y-%m-%d')}): {len(servers)} servers")
                    else:
                        self.logger.warning(f"  No commits found for {label} ({target_date.strftime('%Y-%m-%d')})")
                        
            except Exception as e:
                self.logger.error(f"Error processing {label}: {e}")
                continue
                
            # Rate limiting
            time.sleep(0.2)
    
    def find_closest_commit_to_date(self, target_date):
        """Find the commit closest to (but not after) the target date"""
        commits_url = "https://api.github.com/repos/modelcontextprotocol/servers/commits"
        params = {
            'path': 'README.md',
            'until': target_date.isoformat(),
            'per_page': 1
        }
        
        try:
            response = requests.get(commits_url, headers=self.get_github_api_headers(), params=params)
            response.raise_for_status()
            commits = response.json()
            
            if commits:
                return commits[0]['sha']
            else:
                return None
                
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Error finding commit for {target_date}: {e}")
            return None
    
    def fetch_readme_at_commit(self, commit_sha):
        """Fetch README content at a specific commit"""
        content_url = f"https://api.github.com/repos/modelcontextprotocol/servers/contents/README.md"
        params = {'ref': commit_sha}
        
        try:
            response = requests.get(content_url, headers=self.get_github_api_headers(), params=params)
            response.raise_for_status()
            content_data = response.json()
            
            if content_data.get('encoding') == 'base64':
                import base64
                return base64.b64decode(content_data['content']).decode('utf-8', errors='ignore')
            else:
                return content_data.get('content', '')
                
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Failed to get README at commit {commit_sha}: {e}")
            return None
    
    def get_commit_info(self, commit_sha):
        """Get commit information (date, message)"""
        commit_url = f"https://api.github.com/repos/modelcontextprotocol/servers/commits/{commit_sha}"
        
        try:
            response = requests.get(commit_url, headers=self.get_github_api_headers())
            response.raise_for_status()
            commit_data = response.json()
            
            return {
                'date': datetime.fromisoformat(commit_data['commit']['committer']['date'].replace('Z', '+00:00')),
                'message': commit_data['commit']['message']
            }
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Failed to get commit info for {commit_sha}: {e}")
            return None

    def parse_servers_from_content(self, content):
        """Parse servers from README content with comprehensive pattern matching"""
        lines = content.split('\n')
        servers = []
        
        for line in lines:
            # Skip lines that don't start with bullet points
            if not line.strip().startswith('-'):
                continue
                
            # Skip navigation/header lines
            if any(skip in line.lower() for skip in ['github discussions', 'contributing', 'license', 'documentation']):
                continue
            
            # Extract all links from the line
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
                
                # Handle relative URLs
                if url.startswith('/'):
                    url = f"https://github.com{url}"
                elif not url.startswith('http'):
                    url = urljoin(self.base_url, url)
                
                # Determine if it's a GitHub URL
                is_github = 'github.com' in url
                
                server_info = {
                    'name': name.strip(),
                    'url': url.strip(),
                    'description': description,
                    'is_github': is_github,
                    'extracted_date': datetime.now().isoformat()
                }
                
                servers.append(server_info)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_servers = []
        for server in servers:
            key = (server['name'], server['url'])
            if key not in seen:
                seen.add(key)
                unique_servers.append(server)
        
        return unique_servers
    
    def extract_current_urls(self):
        """Extract URLs from current README"""
        content = self.fetch_readme_content()
        
        if not content:
            self.logger.error("Failed to fetch README content")
            return
        
        self.servers = self.parse_servers_from_content(content)
        
        self.logger.info(f"Total servers found: {len(self.servers)}")
        
        # Count by type
        github_count = sum(1 for s in self.servers if s['is_github'])
        external_count = len(self.servers) - github_count
        
        self.logger.info(f"  GitHub URLs: {github_count}")
        self.logger.info(f"  External URLs: {external_count}")
    
    def save_urls(self, filename='officiallist_urls.json'):
        """Save extracted URLs to JSON file"""
        output_data = {
            'extraction_date': datetime.now().isoformat(),
            'total_servers': len(self.servers),
            'github_servers': sum(1 for s in self.servers if s['is_github']),
            'external_servers': sum(1 for s in self.servers if not s['is_github']),
            'servers': self.servers
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"URLs saved to {filename}")
    
    def save_historical_data(self, filename='officiallist_history.json'):
        """Save historical data to JSON file"""
        if not self.historical_data:
            self.logger.info("No historical data to save")
            return
        
        # Sort by date (handle timezone-aware and naive datetimes)
        def get_sort_date(x):
            date = x['date']
            if hasattr(date, 'replace') and date.tzinfo is not None:
                return date.replace(tzinfo=None)
            return date
        self.historical_data.sort(key=get_sort_date)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.historical_data, f, indent=2, default=str, ensure_ascii=False)
        
        self.logger.info(f"Historical data saved to {filename}")
        self.logger.info(f"  Date range: {self.historical_data[0]['date'].date()} to {self.historical_data[-1]['date'].date()}")
        self.logger.info(f"  Total snapshots: {len(self.historical_data)}")
    
    def print_summary(self):
        """Print summary of extracted URLs"""
        self.logger.info("\n" + "="*60)
        self.logger.info(f"URL EXTRACTION COMPLETE")
        self.logger.info("="*60)
        self.logger.info(f"Total servers: {len(self.servers)}")
        
        # Count by type
        github_count = sum(1 for s in self.servers if s['is_github'])
        external_count = len(self.servers) - github_count
        
        self.logger.info(f"  GitHub URLs: {github_count}")
        self.logger.info(f"  External URLs: {external_count}")
        
        # Show sample external URLs
        external_servers = [s for s in self.servers if not s['is_github']]
        if external_servers:
            self.logger.info(f"\nSample external servers:")
            for server in external_servers[:5]:
                self.logger.info(f"  - {server['name']}: {server['url']}")
        
        # Historical summary
        if self.historical_data:
            self.logger.info(f"\nHistorical data:")
            self.logger.info(f"  Snapshots collected: {len(self.historical_data)}")
            if len(self.historical_data) > 1:
                growth = self.historical_data[-1]['server_count'] - self.historical_data[0]['server_count']
                self.logger.info(f"  Server growth: +{growth} servers")

def main():
    parser = argparse.ArgumentParser(description='Extract URLs from MCP servers README')
    parser.add_argument('--history', type=int, default=0, 
                       help='Days of history to collect (0 = current only)')
    parser.add_argument('--monthly-history', type=int, default=0,
                       help='Months of history to collect (0 = current only)')
    parser.add_argument('--output', default='officiallist_urls.json',
                       help='Output JSON file for URLs')
    parser.add_argument('--history-output', default='officiallist_history.json',
                       help='Output JSON file for historical data')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('officiallist_url_extraction.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Check for GitHub token
    if not os.environ.get('GH_TOKEN'):
        logger.warning("GH_TOKEN environment variable not set")
        logger.warning("Set it with: export GH_TOKEN=your_github_token")
        logger.warning("Continuing without authentication (lower rate limits)...\n")
    
    # Create extractor
    extractor = MCPServerURLExtractor()
    
    try:
        start_time = time.time()
        
        # Extract current URLs
        logger.info("Starting URL extraction...")
        extractor.extract_current_urls()
        
        # Extract historical data if requested
        if args.monthly_history > 0:
            logger.info(f"Collecting {args.monthly_history} months of historical data...")
            extractor.fetch_readme_monthly_history(months_back=args.monthly_history)
        elif args.history > 0:
            logger.info(f"Collecting {args.history} days of historical data...")
            extractor.fetch_readme_history(days_back=args.history)
        
        end_time = time.time()
        logger.info(f"Extraction completed in {end_time - start_time:.1f} seconds")
        
        # Save results
        extractor.save_urls(args.output)
        if args.monthly_history > 0 or args.history > 0:
            extractor.save_historical_data(args.history_output)
        
        # Print summary
        extractor.print_summary()
        
    except KeyboardInterrupt:
        logger.warning("Extraction interrupted by user")
    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        raise

if __name__ == "__main__":
    main()
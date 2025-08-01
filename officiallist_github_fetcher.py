#!/usr/bin/env python3
"""
Lean Official MCP Server GitHub Fetcher
Processes GitHub URLs from officiallist_urls.json to collect README and metadata
"""

import json
import asyncio
import logging
import aiohttp
import base64
from datetime import datetime

class OfficiallistGitHubFetcherLean:
    def __init__(self, github_token):
        self.github_token = github_token
        self.session = None
        self.github_servers = []
        self.processed_count = 0
        
        # Setup logging - use shared log file
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
    
    async def create_session(self):
        """Create aiohttp session with GitHub auth"""
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'MCP-Monitoring-Tool'
        }
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            headers=headers, 
            connector=connector, 
            timeout=timeout
        )
    
    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
    
    def extract_github_info(self, url):
        """Extract owner, repo, and subdirectory from GitHub URL"""
        try:
            # Clean up URL
            url = url.strip()
            if not url.startswith('https://github.com/'):
                return None
                
            # Remove github.com and split path
            path = url.replace('https://github.com/', '')
            parts = path.split('/')
            
            if len(parts) < 2:
                return None
                
            owner = parts[0]
            repo = parts[1]
            
            # Check for subdirectory patterns
            subdirectory = None
            if len(parts) > 2:
                # Handle tree/main/subdir patterns
                if len(parts) >= 5 and parts[2] == 'tree':
                    subdirectory = '/'.join(parts[4:])
                # Handle direct subdirectory references
                elif parts[2] not in ['issues', 'pulls', 'wiki', 'releases']:
                    subdirectory = '/'.join(parts[2:])
            
            return {
                'owner': owner,
                'repo': repo,
                'full_name': f"{owner}/{repo}",
                'subdirectory': subdirectory,
                'url': url
            }
        except Exception as e:
            self.logger.error(f"Error parsing GitHub URL {url}: {e}")
            return None
    
    async def get_repo_metadata(self, owner, repo):
        """Get repository metadata from GitHub API"""
        url = f"https://api.github.com/repos/{owner}/{repo}"
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'full_name': data.get('full_name'),
                        'description': data.get('description'),
                        'stargazers_count': data.get('stargazers_count'),
                        'forks_count': data.get('forks_count'),
                        'language': data.get('language'),
                        'created_at': data.get('created_at'),
                        'updated_at': data.get('updated_at'),
                        'default_branch': data.get('default_branch', 'main'),
                        'topics': data.get('topics', []),
                        'license': data.get('license', {}).get('name') if data.get('license') else None
                    }
                elif response.status == 404:
                    self.logger.warning(f"Repository {owner}/{repo} not found")
                    return None
                else:
                    self.logger.warning(f"Failed to fetch {owner}/{repo}: {response.status}")
                    return None
        except Exception as e:
            self.logger.error(f"Error fetching metadata for {owner}/{repo}: {e}")
            return None
    
    async def get_readme_content(self, owner, repo, subdirectory=None, default_branch='main'):
        """Get README content from repository or subdirectory"""
        # Try different README filename variations
        readme_files = ['README.md', 'readme.md', 'Readme.md', 'README.txt', 'README']
        
        for readme_file in readme_files:
            if subdirectory:
                path = f"{subdirectory}/{readme_file}"
            else:
                path = readme_file
                
            url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
            
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('content'):
                            # Decode base64 content
                            content = base64.b64decode(data['content']).decode('utf-8')
                            return {
                                'readme_content': content,
                                'readme_path': path,
                                'content_source': 'github_readme'
                            }
                    elif response.status == 404:
                        continue  # Try next README file
                    else:
                        self.logger.warning(f"Failed to fetch README {path}: {response.status}")
                        continue
            except Exception as e:
                self.logger.error(f"Error fetching README {path}: {e}")
                continue
        
        # If subdirectory README not found, return empty content (no fallback to main repo)
        if subdirectory:
            return {
                'readme_content': '',
                'readme_path': f"{subdirectory}/README.md",
                'content_source': 'github_readme'
            }
            
        return None
    
    async def process_github_server(self, server):
        """Process a single GitHub server"""
        github_info = self.extract_github_info(server['url'])
        if not github_info:
            return None
            
        # Get repository metadata
        repo_metadata = await self.get_repo_metadata(github_info['owner'], github_info['repo'])
        if not repo_metadata:
            return {
                'name': server['name'],
                'url': server['url'],
                'description': server.get('description', ''),
                'is_github': True,
                'processing_status': 'failed_metadata',
                'error_message': 'Repository not found or API error'
            }
        
        # Get README content
        readme_data = await self.get_readme_content(
            github_info['owner'], 
            github_info['repo'],
            github_info['subdirectory'],
            repo_metadata.get('default_branch', 'main')
        )
        
        # Build server data
        result = {
            'name': server['name'],
            'url': server['url'],
            'description': server.get('description', ''),
            'is_github': True,
            'extracted_date': server.get('extracted_date'),
            'processing_status': 'success',
            'github_metadata': repo_metadata
        }
        
        # Add README content if found
        if readme_data:
            result.update(readme_data)
        else:
            result['readme_content'] = None
            result['content_source'] = None
        
        # Add GitHub-specific fields
        result['github_info'] = github_info
        
        return result
    
    async def process_all_github_servers(self, test_mode=False):
        """Process all GitHub servers from officiallist_urls.json"""
        # Load URL data
        try:
            with open('officiallist_urls.json', 'r', encoding='utf-8') as f:
                url_data = json.load(f)
        except FileNotFoundError:
            self.logger.error("officiallist_urls.json not found")
            return []
        
        # Filter GitHub servers
        github_servers = [s for s in url_data.get('servers', []) if s.get('is_github', False)]
        
        if test_mode:
            github_servers = github_servers[:5]
            
        self.logger.info(f"Processing {len(github_servers)} GitHub servers")
        
        # Create session
        await self.create_session()
        
        # Process servers with concurrency control
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
        
        async def process_with_semaphore(server):
            async with semaphore:
                return await self.process_github_server(server)
        
        # Process all servers
        results = await asyncio.gather(*[process_with_semaphore(server) for server in github_servers])
        
        # Filter successful results
        processed_servers = [r for r in results if r is not None]
        self.processed_count = len(processed_servers)
        
        self.logger.info(f"Successfully processed {self.processed_count} GitHub servers")
        
        # Close session
        await self.close_session()
        
        return processed_servers
    
    def save_results(self, servers):
        """Save GitHub processing results"""
        output_data = {
            'collection_date': datetime.now().isoformat(),
            'total_github_servers': len(servers),
            'processed_count': self.processed_count,
            'servers': servers
        }
        
        with open('officiallist_data_onlygithub.json', 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info("Results saved to officiallist_data_onlygithub.json")


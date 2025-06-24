#!/usr/bin/env python3
"""
Official List GitHub Fetcher
Fetches comprehensive GitHub metadata and README content for all officiallist GitHub servers
Borrows functions from github_mcp_repo_collector.py to ensure consistent data structure
"""

import aiohttp
import asyncio
import json
import time
import base64
from datetime import datetime
import os
from typing import Dict, List, Optional
import logging
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('officiallist_github_fetching.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OfficialistGitHubFetcher:
    def __init__(self, token: str):
        self.token = token
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.base_url = 'https://api.github.com'
        self.session = None
        self.rate_limit_remaining = 5000
        self.rate_limit_reset_time = 0
        self.processed_count = 0
        
    async def init_session(self):
        """Initialize aiohttp session"""
        if not self.session:
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=self.headers
            )
    
    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            
    async def check_rate_limit(self, response_headers):
        """Update rate limit info from response headers"""
        old_remaining = self.rate_limit_remaining
        
        if 'X-RateLimit-Remaining' in response_headers:
            self.rate_limit_remaining = int(response_headers['X-RateLimit-Remaining'])
        if 'X-RateLimit-Reset' in response_headers:
            self.rate_limit_reset_time = int(response_headers['X-RateLimit-Reset'])
            
        # Log rate limit updates
        if old_remaining != self.rate_limit_remaining and self.rate_limit_remaining % 100 == 0:
            logger.info(f"Rate limit updated: {self.rate_limit_remaining} requests remaining")
            
    async def wait_for_rate_limit(self):
        """Wait if rate limit is low"""
        if self.rate_limit_remaining <= 0:
            wait_time = max(0, self.rate_limit_reset_time - time.time() + 60)
            if wait_time > 0:
                logger.warning(f"Rate limit exhausted, waiting {wait_time:.0f} seconds until reset...")
                await asyncio.sleep(wait_time)
            return
            
        if self.rate_limit_remaining < 10:
            logger.warning(f"Rate limit very low ({self.rate_limit_remaining}), waiting 10 seconds...")
            await asyncio.sleep(10)

    def extract_owner_repo_and_path(self, github_url: str) -> Optional[tuple]:
        """Extract owner, repo name, and subdirectory path from GitHub URL"""
        try:
            parsed = urlparse(github_url)
            if parsed.netloc.lower() != 'github.com':
                return None
            
            path_parts = parsed.path.strip('/').split('/')
            if len(path_parts) >= 2:
                owner = path_parts[0]
                repo = path_parts[1]
                subdirectory = None
                
                # Handle subdirectory URLs (tree/main/path or blob/main/path)
                if len(path_parts) > 4 and path_parts[2] in ['tree', 'blob']:
                    # URL like: github.com/owner/repo/tree/branch/path/to/subdir
                    # Extract the subdirectory path
                    subdirectory = '/'.join(path_parts[4:])  # Skip owner/repo/tree/branch
                elif len(path_parts) > 2 and path_parts[2] not in ['tree', 'blob']:
                    # Direct subdirectory reference like github.com/owner/repo/subdir
                    subdirectory = '/'.join(path_parts[2:])
                    
                return (owner, repo, subdirectory)
            return None
        except Exception as e:
            logger.error(f"Error parsing GitHub URL '{github_url}': {e}")
            return None

    def extract_owner_repo(self, github_url: str) -> Optional[tuple]:
        """Extract owner and repo name from GitHub URL (legacy compatibility)"""
        result = self.extract_owner_repo_and_path(github_url)
        if result:
            return (result[0], result[1])  # Return only owner and repo
        return None

    async def get_repository_metadata(self, owner: str, repo: str) -> Optional[Dict]:
        """Get comprehensive repository metadata from GitHub API"""
        await self.init_session()
        
        try:
            await self.wait_for_rate_limit()
            
            # Get main repository data
            async with self.session.get(
                f"{self.base_url}/repos/{owner}/{repo}"
            ) as response:
                await self.check_rate_limit(response.headers)
                
                if response.status == 200:
                    repo_data = await response.json()
                    
                    # Get additional data in parallel
                    tasks = [
                        self.get_repository_topics(owner, repo),
                        self.get_repository_languages(owner, repo),
                        self.get_readme_content(owner, repo)
                    ]
                    
                    topics, languages, readme_content = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Handle exceptions in parallel tasks
                    if isinstance(topics, Exception):
                        logger.warning(f"Error getting topics for {owner}/{repo}: {topics}")
                        topics = []
                    if isinstance(languages, Exception):
                        logger.warning(f"Error getting languages for {owner}/{repo}: {languages}")
                        languages = {}
                    if isinstance(readme_content, Exception):
                        logger.warning(f"Error getting README for {owner}/{repo}: {readme_content}")
                        readme_content = None
                    
                    # Construct comprehensive metadata matching GitHub collector format
                    enriched_repo = {
                        'id': repo_data['id'],
                        'node_id': repo_data['node_id'],
                        'name': repo_data['name'],
                        'full_name': repo_data['full_name'],
                        'private': repo_data['private'],
                        'owner': {
                            'login': repo_data['owner']['login'],
                            'id': repo_data['owner']['id'],
                            'node_id': repo_data['owner']['node_id'],
                            'avatar_url': repo_data['owner']['avatar_url'],
                            'gravatar_id': repo_data['owner']['gravatar_id'],
                            'url': repo_data['owner']['url'],
                            'html_url': repo_data['owner']['html_url'],
                            'type': repo_data['owner']['type'],
                            'site_admin': repo_data['owner']['site_admin']
                        },
                        'html_url': repo_data['html_url'],
                        'description': repo_data['description'],
                        'fork': repo_data['fork'],
                        'url': repo_data['url'],
                        'archive_url': repo_data['archive_url'],
                        'assignees_url': repo_data['assignees_url'],
                        'blobs_url': repo_data['blobs_url'],
                        'branches_url': repo_data['branches_url'],
                        'collaborators_url': repo_data['collaborators_url'],
                        'comments_url': repo_data['comments_url'],
                        'commits_url': repo_data['commits_url'],
                        'compare_url': repo_data['compare_url'],
                        'contents_url': repo_data['contents_url'],
                        'contributors_url': repo_data['contributors_url'],
                        'deployments_url': repo_data['deployments_url'],
                        'downloads_url': repo_data['downloads_url'],
                        'events_url': repo_data['events_url'],
                        'forks_url': repo_data['forks_url'],
                        'git_commits_url': repo_data['git_commits_url'],
                        'git_refs_url': repo_data['git_refs_url'],
                        'git_tags_url': repo_data['git_tags_url'],
                        'git_url': repo_data['git_url'],
                        'issue_comment_url': repo_data['issue_comment_url'],
                        'issue_events_url': repo_data['issue_events_url'],
                        'issues_url': repo_data['issues_url'],
                        'keys_url': repo_data['keys_url'],
                        'labels_url': repo_data['labels_url'],
                        'languages_url': repo_data['languages_url'],
                        'merges_url': repo_data['merges_url'],
                        'milestones_url': repo_data['milestones_url'],
                        'notifications_url': repo_data['notifications_url'],
                        'pulls_url': repo_data['pulls_url'],
                        'releases_url': repo_data['releases_url'],
                        'ssh_url': repo_data['ssh_url'],
                        'stargazers_url': repo_data['stargazers_url'],
                        'statuses_url': repo_data['statuses_url'],
                        'subscribers_url': repo_data['subscribers_url'],
                        'subscription_url': repo_data['subscription_url'],
                        'tags_url': repo_data['tags_url'],
                        'teams_url': repo_data['teams_url'],
                        'trees_url': repo_data['trees_url'],
                        'clone_url': repo_data['clone_url'],
                        'mirror_url': repo_data['mirror_url'],
                        'hooks_url': repo_data['hooks_url'],
                        'svn_url': repo_data['svn_url'],
                        'homepage': repo_data['homepage'],
                        'language': repo_data['language'],
                        'forks_count': repo_data['forks_count'],
                        'stargazers_count': repo_data['stargazers_count'],
                        'watchers_count': repo_data['watchers_count'],
                        'size': repo_data['size'],
                        'default_branch': repo_data['default_branch'],
                        'open_issues_count': repo_data['open_issues_count'],
                        'is_template': repo_data.get('is_template', False),
                        'has_issues': repo_data['has_issues'],
                        'has_projects': repo_data['has_projects'],
                        'has_wiki': repo_data['has_wiki'],
                        'has_pages': repo_data['has_pages'],
                        'has_downloads': repo_data['has_downloads'],
                        'archived': repo_data['archived'],
                        'disabled': repo_data['disabled'],
                        'visibility': repo_data.get('visibility', 'public'),
                        'pushed_at': repo_data['pushed_at'],
                        'created_at': repo_data['created_at'],
                        'updated_at': repo_data['updated_at'],
                        'permissions': repo_data.get('permissions', {}),
                        'allow_rebase_merge': repo_data.get('allow_rebase_merge', True),
                        'template_repository': repo_data.get('template_repository'),
                        'temp_clone_token': repo_data.get('temp_clone_token'),
                        'allow_squash_merge': repo_data.get('allow_squash_merge', True),
                        'allow_auto_merge': repo_data.get('allow_auto_merge', False),
                        'delete_branch_on_merge': repo_data.get('delete_branch_on_merge', False),
                        'allow_merge_commit': repo_data.get('allow_merge_commit', True),
                        'subscribers_count': repo_data.get('subscribers_count', 0),
                        'network_count': repo_data.get('network_count', 0),
                        'license': repo_data.get('license'),
                        'score': 1.0,  # Placeholder score
                        
                        # Additional data from our API calls
                        'topics': topics,
                        'languages': languages,
                        'readme_content': readme_content,
                        'collected_at': datetime.now().isoformat(),
                        'source': 'officiallist_github_fetcher'
                    }
                    
                    return enriched_repo
                    
                elif response.status == 404:
                    logger.warning(f"Repository {owner}/{repo} not found (404)")
                    return None
                else:
                    logger.error(f"Error fetching {owner}/{repo}: HTTP {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting repository metadata for {owner}/{repo}: {e}")
            return None

    async def get_repository_topics(self, owner: str, repo: str) -> List[str]:
        """Get repository topics/tags"""
        await self.init_session()
        
        try:
            await self.wait_for_rate_limit()
            
            # Use the preview API for topics
            headers = {**self.headers, 'Accept': 'application/vnd.github.mercy-preview+json'}
            
            async with self.session.get(
                f"{self.base_url}/repos/{owner}/{repo}/topics",
                headers=headers
            ) as response:
                await self.check_rate_limit(response.headers)
                
                if response.status == 200:
                    data = await response.json()
                    return data.get('names', [])
                else:
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting topics for {owner}/{repo}: {e}")
            return []

    async def get_repository_languages(self, owner: str, repo: str) -> Dict[str, int]:
        """Get repository programming languages"""
        await self.init_session()
        
        try:
            await self.wait_for_rate_limit()
            
            async with self.session.get(
                f"{self.base_url}/repos/{owner}/{repo}/languages"
            ) as response:
                await self.check_rate_limit(response.headers)
                
                if response.status == 200:
                    return await response.json()
                else:
                    return {}
                    
        except Exception as e:
            logger.error(f"Error getting languages for {owner}/{repo}: {e}")
            return {}

    async def get_readme_content(self, owner: str, repo: str) -> Optional[str]:
        """Get README content"""
        await self.init_session()
        
        try:
            await self.wait_for_rate_limit()
            
            async with self.session.get(
                f"{self.base_url}/repos/{owner}/{repo}/readme"
            ) as response:
                await self.check_rate_limit(response.headers)
                
                if response.status == 200:
                    data = await response.json()
                    # Decode base64 content
                    if data.get('encoding') == 'base64':
                        content = base64.b64decode(data['content']).decode('utf-8', errors='ignore')
                        return content
                    return data.get('content', '')
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting README for {owner}/{repo}: {e}")
            return None

    def load_officiallist_servers(self, filename: str = 'officiallist_urls.json') -> List[Dict]:
        """Load officiallist servers and filter for GitHub URLs"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Filter for GitHub servers only
            github_servers = [
                server for server in data.get('servers', [])
                if server.get('is_github', False)
            ]
            
            logger.info(f"Loaded {len(github_servers)} GitHub servers from {filename}")
            return github_servers
            
        except FileNotFoundError:
            logger.error(f"File not found: {filename}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {filename}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading officiallist servers: {e}")
            return []

    def load_existing_metadata(self, filename: str = 'officiallist_github_metadata.json') -> Dict[str, Dict]:
        """Load existing GitHub metadata to avoid re-fetching"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Create lookup by unique_key (handles subdirectories)
            metadata_lookup = {}
            for item in data.get('repositories', []):
                # Try to get unique_key from officiallist_context, fallback to full_name
                officiallist_context = item.get('officiallist_context', {})
                unique_key = officiallist_context.get('unique_key')
                if not unique_key:
                    # Fallback for older data
                    full_name = item.get('full_name')
                    if full_name:
                        unique_key = full_name
                
                if unique_key:
                    metadata_lookup[unique_key] = item
            
            logger.info(f"Loaded existing metadata for {len(metadata_lookup)} repositories")
            return metadata_lookup
            
        except FileNotFoundError:
            logger.info("No existing metadata file found - starting fresh")
            return {}
        except Exception as e:
            logger.warning(f"Error loading existing metadata: {e}")
            return {}

    async def process_github_servers(self, test_mode: bool = False, resume: bool = True) -> List[Dict]:
        """Process all GitHub servers from officiallist and fetch their metadata"""
        # Load officiallist GitHub servers
        github_servers = self.load_officiallist_servers()
        
        if not github_servers:
            logger.error("No GitHub servers found in officiallist")
            return []
        
        # Load existing metadata for resume capability
        existing_metadata = self.load_existing_metadata() if resume else {}
        
        # Limit for test mode
        if test_mode:
            github_servers = github_servers[:5]
            logger.info(f"=== TEST MODE: Processing only {len(github_servers)} servers ===")
        
        enriched_repositories = []
        failed_servers = []
        skipped_count = 0
        
        logger.info(f"Starting to process {len(github_servers)} GitHub servers...")
        
        try:
            await self.init_session()
            
            for i, server in enumerate(github_servers, 1):
                url = server.get('url', '')
                name = server.get('name', 'Unknown')
                
                # Extract owner, repo, and subdirectory from URL
                url_parts = self.extract_owner_repo_and_path(url)
                if not url_parts:
                    logger.warning(f"Could not parse GitHub URL: {url}")
                    failed_servers.append({
                        'server': server,
                        'error': 'Invalid GitHub URL format'
                    })
                    continue
                
                owner, repo, subdirectory = url_parts
                full_name = f"{owner}/{repo}"
                
                # Create unique key for subdirectories
                unique_key = f"{full_name}#{subdirectory}" if subdirectory else full_name
                
                # Check if we already have metadata (resume mode)
                if unique_key in existing_metadata:
                    logger.info(f"[{i}/{len(github_servers)}] Skipping {name} ({unique_key}) - already processed")
                    enriched_repositories.append(existing_metadata[unique_key])
                    skipped_count += 1
                    continue
                
                logger.info(f"[{i}/{len(github_servers)}] Processing {name} ({full_name})")
                
                # Fetch comprehensive metadata
                repo_metadata = await self.get_repository_metadata(owner, repo)
                
                if repo_metadata:
                    # Add original officiallist context and subdirectory info
                    repo_metadata['officiallist_context'] = {
                        'name': server.get('name'),
                        'url': server.get('url'),
                        'description': server.get('description'),
                        'extracted_date': server.get('extracted_date'),
                        'subdirectory': subdirectory,
                        'unique_key': unique_key
                    }
                    
                    enriched_repositories.append(repo_metadata)
                    logger.info(f"  ✓ Successfully processed {unique_key}")
                else:
                    failed_servers.append({
                        'server': server,
                        'error': 'Failed to fetch metadata'
                    })
                    logger.warning(f"  ✗ Failed to process {unique_key}")
                
                self.processed_count += 1
                
                # Progress reporting
                if i % 10 == 0:
                    success_rate = (len(enriched_repositories) / i) * 100
                    logger.info(f"Progress: {i}/{len(github_servers)} ({success_rate:.1f}% success rate)")
                
                # Save progress periodically
                if i % 50 == 0:
                    await self.save_progress(enriched_repositories, failed_servers)
                
                # Rate limiting courtesy pause
                await asyncio.sleep(0.1)
                
        finally:
            await self.close_session()
        
        logger.info(f"""
=== PROCESSING COMPLETE ===
Total servers: {len(github_servers)}
Successfully processed: {len(enriched_repositories)}
Skipped (already had): {skipped_count}
Failed: {len(failed_servers)}
""")
        
        if failed_servers:
            logger.warning("Failed servers:")
            for failed in failed_servers[:10]:  # Show first 10
                server = failed['server']
                logger.warning(f"  - {server.get('name')} ({server.get('url')}): {failed['error']}")
        
        return enriched_repositories

    async def save_progress(self, repositories: List[Dict], failed_servers: List[Dict]):
        """Save current progress"""
        output_data = {
            'collection_date': datetime.now().isoformat(),
            'total_repositories': len(repositories),
            'failed_count': len(failed_servers),
            'repositories': repositories,
            'failed_servers': failed_servers,
            'source': 'officiallist_github_fetcher'
        }
        
        with open('officiallist_github_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Progress saved: {len(repositories)} repositories processed")

    def integrate_with_officiallist_full(self, github_metadata_file: str = 'officiallist_github_metadata.json'):
        """Integrate GitHub metadata into officiallist_mcp_servers_full.json"""
        try:
            # Load GitHub metadata
            with open(github_metadata_file, 'r', encoding='utf-8') as f:
                github_data = json.load(f)
            
            github_repos = {repo['full_name']: repo for repo in github_data.get('repositories', [])}
            
            # Load existing officiallist full data
            try:
                with open('officiallist_mcp_servers_full.json', 'r', encoding='utf-8') as f:
                    officiallist_data = json.load(f)
            except FileNotFoundError:
                # Create new structure if file doesn't exist
                officiallist_data = {
                    'collection_date': datetime.now().isoformat(),
                    'total_servers': 0,
                    'github_enhanced_count': 0,
                    'servers': []
                }
            
            enhanced_count = 0
            
            # Enhance servers with GitHub metadata
            for server in officiallist_data.get('servers', []):
                if server.get('is_github', False):
                    url = server.get('url', '')
                    owner_repo = self.extract_owner_repo(url)
                    
                    if owner_repo:
                        owner, repo = owner_repo
                        full_name = f"{owner}/{repo}"
                        
                        if full_name in github_repos:
                            # Merge GitHub metadata
                            server['github_metadata'] = github_repos[full_name]
                            enhanced_count += 1
                            logger.debug(f"Enhanced {server.get('name')} with GitHub metadata")
            
            # Update collection metadata
            officiallist_data['github_enhanced_count'] = enhanced_count
            officiallist_data['last_github_integration'] = datetime.now().isoformat()
            
            # Save integrated data
            with open('officiallist_mcp_servers_full.json', 'w', encoding='utf-8') as f:
                json.dump(officiallist_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Successfully integrated GitHub metadata for {enhanced_count} servers into officiallist_mcp_servers_full.json")
            
        except Exception as e:
            logger.error(f"Error integrating GitHub metadata: {e}")
            raise


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch GitHub metadata for officiallist GitHub servers')
    parser.add_argument('--test', action='store_true', help='Test mode - process only 5 servers')
    parser.add_argument('--no-resume', action='store_true', help='Start fresh, ignore existing metadata')
    parser.add_argument('--integrate-only', action='store_true', help='Only integrate existing metadata, skip fetching')
    args = parser.parse_args()
    
    # Check for GitHub token
    github_token = os.environ.get('GH_TOKEN')
    if not github_token:
        logger.error("GH_TOKEN environment variable is required")
        logger.error("Set it with: export GH_TOKEN=your_github_token")
        return 1
    
    fetcher = OfficialistGitHubFetcher(github_token)
    
    try:
        if args.integrate_only:
            logger.info("Integration mode: updating officiallist_mcp_servers_full.json with existing metadata")
            fetcher.integrate_with_officiallist_full()
        else:
            # Fetch GitHub metadata
            logger.info("Starting GitHub metadata collection for officiallist servers...")
            repositories = await fetcher.process_github_servers(
                test_mode=args.test,
                resume=not args.no_resume
            )
            
            # Save final results
            await fetcher.save_progress(repositories, [])
            
            # Integrate with officiallist full data
            logger.info("Integrating GitHub metadata into officiallist_mcp_servers_full.json...")
            fetcher.integrate_with_officiallist_full()
        
        logger.info("Process completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
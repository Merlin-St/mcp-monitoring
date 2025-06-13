#!/usr/bin/env python3
"""
GitHub MCP Repository Collector
Searches for MCP repositories and collects metadata, topics, and README content
"""

import aiohttp
import asyncio
import json
import time
import base64
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('github_mcp_collection.log'),
        logging.StreamHandler()  # Keep console output for immediate feedback
    ]
)
logger = logging.getLogger(__name__)


class GitHubMCPCollector:
    def __init__(self, token: str):
        self.token = token
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.base_url = 'https://api.github.com'
        self.graphql_url = 'https://api.github.com/graphql'
        self.data = []
        self.session = None
        self.rate_limit_remaining = 5000
        self.rate_limit_reset_time = 0
        
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
        if old_remaining != self.rate_limit_remaining:
            logger.info(f"Rate limit updated: {self.rate_limit_remaining} requests remaining")
            
    async def wait_for_rate_limit(self):
        """Wait if rate limit is low or when hitting thresholds"""
        # Only wait for actual rate limit exhaustion
        if self.rate_limit_remaining <= 0:
            # Calculate wait time until reset (add 1 minute buffer)
            wait_time = max(0, self.rate_limit_reset_time - time.time() + 60)
            if wait_time > 0:
                logger.warning(f"Rate limit exhausted ({self.rate_limit_remaining}), waiting {wait_time:.0f} seconds until reset...")
                await asyncio.sleep(wait_time)
            return
            
        # Simple check: wait 10 seconds if very low
        if self.rate_limit_remaining < 10:
            logger.warning(f"Rate limit very low ({self.rate_limit_remaining}), waiting 10 seconds...")
            await asyncio.sleep(10)
                
    async def search_repositories_graphql(self, query: str, max_results: int = 1000) -> List[Dict]:
        """Search GitHub repositories using GraphQL API for better performance"""
        await self.init_session()
        repositories = []
        
        graphql_query = """
        query($query: String!, $first: Int!, $after: String) {
            search(query: $query, type: REPOSITORY, first: $first, after: $after) {
                repositoryCount
                edges {
                    node {
                        ... on Repository {
                            id
                            name
                            nameWithOwner
                            description
                            url
                            homepageUrl
                            createdAt
                            updatedAt
                            stargazerCount
                            forkCount
                            primaryLanguage {
                                name
                            }
                            owner {
                                login
                                ... on User {
                                    name
                                }
                                ... on Organization {
                                    name
                                }
                            }
                            repositoryTopics(first: 20) {
                                nodes {
                                    topic {
                                        name
                                    }
                                }
                            }
                            languages(first: 10) {
                                edges {
                                    node {
                                        name
                                    }
                                    size
                                }
                            }
                        }
                    }
                }
                pageInfo {
                    hasNextPage
                    endCursor
                }
            }
            rateLimit {
                remaining
                resetAt
            }
        }
        """
        
        after_cursor = None
        page = 1
        
        while len(repositories) < max_results:
            variables = {
                "query": query,
                "first": min(100, max_results - len(repositories)),
                "after": after_cursor
            }
            
            try:
                await self.wait_for_rate_limit()
                
                async with self.session.post(
                    self.graphql_url,
                    json={"query": graphql_query, "variables": variables}
                ) as response:
                    await self.check_rate_limit(response.headers)
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'errors' in data:
                            logger.error(f"GraphQL errors: {data['errors']}")
                            break
                            
                        search_data = data['data']['search']
                        
                        # Update rate limit info
                        if 'rateLimit' in data['data']:
                            self.rate_limit_remaining = data['data']['rateLimit']['remaining']
                        
                        if not search_data['edges']:
                            break
                            
                        # Convert GraphQL response to REST-like format
                        for edge in search_data['edges']:
                            repo = edge['node']
                            # Transform to match REST API format
                            rest_repo = {
                                'id': repo['id'],
                                'name': repo['name'],
                                'full_name': repo['nameWithOwner'],
                                'description': repo['description'],
                                'html_url': repo['url'],
                                'homepage': repo['homepageUrl'],
                                'created_at': repo['createdAt'],
                                'updated_at': repo['updatedAt'],
                                'stargazers_count': repo['stargazerCount'],
                                'forks_count': repo['forkCount'],
                                'language': repo['primaryLanguage']['name'] if repo['primaryLanguage'] else None,
                                'owner': {
                                    'login': repo['owner']['login'],
                                    'name': repo['owner'].get('name')
                                },
                                'topics': [topic['topic']['name'] for topic in repo['repositoryTopics']['nodes']],
                                'languages': {edge['node']['name']: edge['size'] for edge in repo['languages']['edges']}
                            }
                            repositories.append(rest_repo)
                        
                        # Log progress after processing results
                        logger.info(f"GraphQL search '{query}' - Page {page} (found {len(repositories)} repos)")
                            
                        if not search_data['pageInfo']['hasNextPage']:
                            break
                            
                        after_cursor = search_data['pageInfo']['endCursor']
                        page += 1
                        
                    else:
                        logger.error(f"HTTP {response.status}: {await response.text()}")
                        break
                        
            except asyncio.TimeoutError:
                logger.error(f"Timeout error in GraphQL search for query '{query}' page {page}")
                # Try to continue with next page after timeout
                await asyncio.sleep(2)
                continue
            except Exception as e:
                logger.error(f"Error in GraphQL search for query '{query}' page {page}: {e}")
                break
                
        return repositories[:max_results]
    
    async def get_repository_topics(self, owner: str, repo: str) -> List[str]:
        """Get repository topics/tags - now handled by GraphQL"""
        # This method is kept for compatibility but GraphQL handles topics
        return []
    
    async def get_readme_content(self, owner: str, repo: str) -> Optional[str]:
        """Get README content asynchronously"""
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
    
    async def get_languages(self, owner: str, repo: str) -> Dict[str, int]:
        """Get repository languages - now handled by GraphQL"""
        # This method is kept for compatibility but GraphQL handles languages
        return {}
    
    async def enrich_repository_data(self, repo: Dict) -> Dict:
        """Enrich repository data with additional information"""
        owner = repo['owner']['login']
        repo_name = repo['name']
        
        logger.debug(f"Enriching data for {owner}/{repo_name}")
        
        # Get README content (topics and languages now come from GraphQL)
        readme = await self.get_readme_content(owner, repo_name)
        
        # Add enriched data
        enriched_repo = {
            **repo,
            'readme_content': readme,
            'collected_at': datetime.now().isoformat()
        }
        
        return enriched_repo
    
    async def search_repositories_rest(self, query: str, max_results: int = 10000) -> List[Dict]:
        """Search GitHub repositories using REST API with full pagination"""
        await self.init_session()
        repositories = []
        page = 1
        per_page = 100  # Max allowed by GitHub
        
        while len(repositories) < max_results:
            try:
                await self.wait_for_rate_limit()
                
                params = {
                    'q': query,
                    'sort': 'updated',
                    'order': 'desc',
                    'page': page,
                    'per_page': per_page
                }
                
                async with self.session.get(
                    f"{self.base_url}/search/repositories",
                    params=params
                ) as response:
                    await self.check_rate_limit(response.headers)
                    
                    if response.status == 200:
                        data = await response.json()
                        items = data.get('items', [])
                        
                        if not items:
                            logger.info(f"No more results for '{query}' at page {page}")
                            break
                            
                        repositories.extend(items)
                        logger.info(f"REST search '{query}' - Page {page} (total: {len(repositories)}/{data.get('total_count', 0)})")
                        
                        # Check if we've reached the end
                        if len(items) < per_page or len(repositories) >= data.get('total_count', 0):
                            break
                            
                        page += 1
                        
                    elif response.status == 422:
                        logger.warning(f"Query '{query}' returned 422 - may be too broad or have syntax issues")
                        break
                    else:
                        logger.error(f"HTTP {response.status}: {await response.text()}")
                        break
                        
            except asyncio.TimeoutError:
                logger.error(f"Timeout error in REST search for query '{query}' page {page}")
                await asyncio.sleep(2)
                continue
            except Exception as e:
                logger.error(f"Error in REST search for query '{query}' page {page}: {e}")
                break
                
        return repositories[:max_results]

    async def collect_mcp_repositories(self, test_mode=False, use_rest=False, resume_mode=False):
        """Main collection process using async/await for better performance"""
        if use_rest:
            # Daily date-based MECE search strategy using "mcp server"
            search_queries = []
            
            # Generate daily searches from 2012 to current date
            from datetime import datetime, timedelta
            
            # Handle resume mode
            existing_data = []
            if resume_mode:
                partial_filename = 'github_mcp_repositories_test_partial.json' if test_mode else 'github_mcp_repositories_partial.json'
                existing_data = self.load_existing_data(partial_filename)
                start_date = self.get_resume_date(existing_data)
            else:
                start_date = datetime(2024, 11, 1)
            
            end_date = datetime.now()
            current_date = start_date
            
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                search_queries.append(f'mcp server created:{date_str}')
                current_date += timedelta(days=1)
            
            logger.info(f"Generated {len(search_queries)} daily search queries from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        else:
            # Original GraphQL search queries
            search_queries = [
                'mcp server',
                'model context protocol', 
                'mcp-server',
                'modelcontextprotocol',
                '"mcp" in:name',
                '"mcp" in:description',
                'topic:mcp',
                'topic:model-context-protocol'
            ]
        
        if test_mode:
            logger.info("=== RUNNING IN TEST MODE - LIMITED TO 10 REPOS ===")
            if use_rest:
                # For REST test mode, use only recent high-activity dates
                test_queries = [q for q in search_queries if '2025-06' in q or '2025-05' in q]
                search_queries = test_queries[:5]  # Use only 5 recent dates
            else:
                search_queries = search_queries[:2]  # Use only first 2 queries
        
        all_repos = {}  # Use dict to avoid duplicates
        
        if use_rest:
            logger.info("Using REST API for repository search...")
            # For REST API, process queries sequentially to respect rate limits
            search_results = []
            for query in search_queries:
                max_results = 10 if test_mode else 10000  # REST can handle more
                logger.info(f"Searching with REST API: '{query}'")
                result = await self.search_repositories_rest(query, max_results=max_results)
                search_results.append(result)
        else:
            # Original GraphQL batch processing
            max_concurrent = 3  # Reduced from 8 to avoid GitHub abuse detection
            
            logger.info(f"Starting GraphQL searches with max {max_concurrent} concurrent requests...")
            
            # Process searches in batches
            search_results = []
            for i in range(0, len(search_queries), max_concurrent):
                batch_queries = search_queries[i:i+max_concurrent]
                batch_tasks = []
                
                for query in batch_queries:
                    max_results = 10 if test_mode else 500
                    batch_tasks.append(self.search_repositories_graphql(query, max_results=max_results))
                
                logger.info(f"Processing search batch {i//max_concurrent + 1}/{(len(search_queries) + max_concurrent - 1)//max_concurrent}")
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                search_results.extend(batch_results)
                
                # Small delay between batches to be respectful to GitHub API
                if i + max_concurrent < len(search_queries):
                    await asyncio.sleep(1)
        
        # Process search results with validation
        for i, (query, result) in enumerate(zip(search_queries, search_results)):
            if isinstance(result, Exception):
                logger.error(f"Error searching '{query}': {result}")
                continue
                
            # Validate and filter results
            valid_repos = []
            for repo in result:
                # Basic validation - ensure required fields exist
                if (repo.get('full_name') and repo.get('name') and 
                    repo.get('owner', {}).get('login')):
                    valid_repos.append(repo)
                else:
                    logger.warning(f"Skipping invalid repo data: {repo.get('full_name', 'unknown')}")
            
            logger.info(f"Query '{query}' returned {len(valid_repos)} valid repositories (filtered from {len(result)} total)")
            
            # Add to our collection (using full_name as key to avoid duplicates)
            for repo in valid_repos:
                all_repos[repo['full_name']] = repo
                if test_mode and len(all_repos) >= 10:
                    break
            
            if test_mode and len(all_repos) >= 10:
                break
        
        # Limit to 10 repos in test mode
        if test_mode:
            all_repos = dict(list(all_repos.items())[:10])
        
        logger.info(f"Found {len(all_repos)} unique repositories")
        
        # Enrich repositories with README content concurrently
        logger.info("Enriching repositories with README content...")
        repos_list = list(all_repos.values())
        
        # Process in batches to avoid overwhelming the API
        batch_size = 20
        # Start with existing data if in resume mode
        enriched_repos = existing_data if resume_mode and use_rest else []
        
        for i in range(0, len(repos_list), batch_size):
            batch = repos_list[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(repos_list) + batch_size - 1)//batch_size}")
            
            # Process batch concurrently
            enrich_tasks = [self.enrich_repository_data(repo) for repo in batch]
            batch_results = await asyncio.gather(*enrich_tasks, return_exceptions=True)
            
            for repo, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error enriching {repo['full_name']}: {result}")
                    # Add repo without enrichment
                    enriched_repos.append({**repo, 'collected_at': datetime.now().isoformat()})
                else:
                    enriched_repos.append(result)
            
            # Save periodically
            if len(enriched_repos) % 50 == 0:
                filename = 'github_mcp_repositories_test_partial.json' if test_mode else 'github_mcp_repositories_partial.json'
                self.save_data(enriched_repos, filename)
        
        self.data = enriched_repos
        return enriched_repos
    
    def load_existing_data(self, filename: str) -> List[Dict]:
        """Load existing partial data if it exists"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Loaded {len(data)} existing repositories from {filename}")
                return data
        except FileNotFoundError:
            logger.info(f"No existing file {filename} found, starting fresh")
            return []
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return []

    def get_resume_date(self, existing_data: List[Dict]) -> datetime:
        """Get the date to resume from based on existing data"""
        if not existing_data:
            return datetime(2024, 11, 1)
        
        # Find the latest creation date in existing data
        dates = []
        for repo in existing_data:
            if repo.get('created_at'):
                try:
                    # Parse ISO format: 2025-04-16T23:30:01Z
                    date_obj = datetime.fromisoformat(repo['created_at'].replace('Z', '+00:00'))
                    dates.append(date_obj.date())
                except Exception:
                    continue
        
        if dates:
            latest_date = max(dates)
            # Resume from the day after the latest found date
            resume_date = datetime.combine(latest_date, datetime.min.time()) + timedelta(days=1)
            logger.info(f"Resuming collection from {resume_date.strftime('%Y-%m-%d')} (day after latest found: {latest_date})")
            return resume_date
        else:
            logger.warning("No valid creation dates found in existing data, starting from default date")
            return datetime(2024, 11, 1)

    def save_data(self, data: List[Dict], filename: str = 'github_mcp_repositories.json'):
        """Save collected data to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(data)} repositories to {filename}")


async def main():
    # Check for command line arguments
    import sys
    test_mode = '--test' in sys.argv or '-t' in sys.argv
    resume_mode = '--resume' in sys.argv or '-r' in sys.argv
    
    # Get GitHub token from environment
    token = os.environ.get('GH_TOKEN')
    
    if not token:
        logger.error("GitHub token is required!")
        logger.error("Set GH_TOKEN environment variable with your GitHub personal access token")
        logger.error("Create one at: https://github.com/settings/tokens")
        logger.error("Required scopes: public_repo")
        return
    
    # Create collector and run
    collector = GitHubMCPCollector(token)
    
    try:
        use_rest = '--graphql' not in sys.argv and '--gql' not in sys.argv  # Default to REST, override with --graphql
        
        if test_mode:
            logger.info("Starting MCP repository collection in TEST MODE...")
            logger.info("Will collect only 10 repositories for testing...")
        else:
            logger.info("Starting MCP repository collection...")
        
        if resume_mode:
            logger.info("RESUME MODE: Will load existing data and continue from last processed date")
        
        if use_rest:
            logger.info("Using REST API with full pagination to get all MCP repositories...")
        else:
            logger.info("Using async GraphQL API for faster collection...")
        logger.info("Use --test or -t flag to run a quick test with 10 repos")
        logger.info("Use --graphql or --gql flag to use GraphQL API (REST is default)")
        logger.info("Use --resume or -r flag to resume from existing partial data")
        
        start_time = time.time()
        repositories = await collector.collect_mcp_repositories(test_mode=test_mode, use_rest=use_rest, resume_mode=resume_mode)
        end_time = time.time()
        
        logger.info(f"Collection completed in {end_time - start_time:.1f} seconds")
        
        # Close the session
        await collector.close_session()
        
        # Save final data with appropriate filename
        if test_mode:
            filename = 'github_mcp_repositories_test_rest.json' if use_rest else 'github_mcp_repositories_test.json'
        else:
            filename = 'github_mcp_repositories_rest.json' if use_rest else 'github_mcp_repositories.json'
        collector.save_data(repositories, filename)
        
        # Also save a summary
        summary = {
            'total_repositories': len(repositories),
            'collection_date': datetime.now().isoformat(),
            'collection_time_seconds': end_time - start_time,
            'test_mode': test_mode,
            'repositories_by_language': {},
            'repositories_with_topics': sum(1 for r in repositories if r.get('topics')),
            'repositories_with_readme': sum(1 for r in repositories if r.get('readme_content'))
        }
        
        # Count repositories by primary language
        for repo in repositories:
            lang = repo.get('language', 'Unknown')
            summary['repositories_by_language'][lang] = summary['repositories_by_language'].get(lang, 0) + 1
        
        summary_filename = 'github_mcp_collection_summary_test.json' if test_mode else 'github_mcp_collection_summary.json'
        with open(summary_filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Collection complete!")
        logger.info(f"Total repositories: {len(repositories)}")
        logger.info(f"Data saved to: {filename}")
        logger.info(f"Summary saved to: {summary_filename}")
        
        if test_mode:
            logger.info("To run full collection, remove --test flag")
        
    except KeyboardInterrupt:
        logger.warning("Collection interrupted by user")
        await collector.close_session()
        if collector.data:
            interrupt_filename = 'github_mcp_repositories_test_interrupted.json' if test_mode else 'github_mcp_repositories_interrupted.json'
            collector.save_data(collector.data, interrupt_filename)
    except Exception as e:
        logger.error(f"Error during collection: {e}")
        await collector.close_session()
        if collector.data:
            error_filename = 'github_mcp_repositories_test_error.json' if test_mode else 'github_mcp_repositories_error.json'
            collector.save_data(collector.data, error_filename)


def run_main():
    """Wrapper to run async main function"""
    asyncio.run(main())


if __name__ == "__main__":
    run_main()
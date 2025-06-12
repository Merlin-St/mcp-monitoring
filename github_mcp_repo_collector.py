#!/usr/bin/env python3
"""
GitHub MCP Repository Collector
Searches for MCP repositories and collects metadata, topics, and README content
"""

import requests
import json
import time
import base64
from datetime import datetime
import os
from typing import Dict, List, Optional

class GitHubMCPCollector:
    def __init__(self, token: str):
        self.token = token
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.base_url = 'https://api.github.com'
        self.data = []
        
    def search_repositories(self, query: str, max_results: int = 1000) -> List[Dict]:
        """Search GitHub repositories with pagination"""
        repositories = []
        page = 1
        per_page = 100  # GitHub's max per page
        
        while len(repositories) < max_results:
            print(f"Searching with query '{query}' - Page {page}")
            
            try:
                response = requests.get(
                    f"{self.base_url}/search/repositories",
                    headers=self.headers,
                    params={
                        'q': query,
                        'per_page': per_page,
                        'page': page,
                        'sort': 'stars',
                        'order': 'desc'
                    }
                )
                
                if response.status_code == 403:
                    print("Rate limit hit. Waiting 60 seconds...")
                    time.sleep(60)
                    continue
                    
                response.raise_for_status()
                data = response.json()
                
                if 'items' not in data or not data['items']:
                    break
                    
                repositories.extend(data['items'])
                print(f"Found {len(data['items'])} repositories (Total: {len(repositories)})")
                
                # Check if we've retrieved all available results
                if len(data['items']) < per_page:
                    break
                    
                page += 1
                time.sleep(2)  # Be respectful of rate limits
                
            except requests.exceptions.RequestException as e:
                print(f"Error searching repositories: {e}")
                break
                
        return repositories[:max_results]
    
    def get_repository_topics(self, owner: str, repo: str) -> List[str]:
        """Get repository topics/tags"""
        try:
            response = requests.get(
                f"{self.base_url}/repos/{owner}/{repo}/topics",
                headers={**self.headers, 'Accept': 'application/vnd.github.mercy-preview+json'}
            )
            
            if response.status_code == 200:
                return response.json().get('names', [])
            else:
                return []
                
        except Exception as e:
            print(f"Error getting topics for {owner}/{repo}: {e}")
            return []
    
    def get_readme_content(self, owner: str, repo: str) -> Optional[str]:
        """Get README content"""
        try:
            response = requests.get(
                f"{self.base_url}/repos/{owner}/{repo}/readme",
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                # Decode base64 content
                if data.get('encoding') == 'base64':
                    content = base64.b64decode(data['content']).decode('utf-8', errors='ignore')
                    return content
                return data.get('content', '')
            else:
                return None
                
        except Exception as e:
            print(f"Error getting README for {owner}/{repo}: {e}")
            return None
    
    def get_languages(self, owner: str, repo: str) -> Dict[str, int]:
        """Get repository languages"""
        try:
            response = requests.get(
                f"{self.base_url}/repos/{owner}/{repo}/languages",
                headers=self.headers
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {}
                
        except Exception as e:
            print(f"Error getting languages for {owner}/{repo}: {e}")
            return {}
    
    def enrich_repository_data(self, repo: Dict) -> Dict:
        """Enrich repository data with additional information"""
        owner = repo['owner']['login']
        repo_name = repo['name']
        
        print(f"Enriching data for {owner}/{repo_name}")
        
        # Get topics
        topics = self.get_repository_topics(owner, repo_name)
        
        # Get README
        readme = self.get_readme_content(owner, repo_name)
        
        # Get languages
        languages = self.get_languages(owner, repo_name)
        
        # Add enriched data
        enriched_repo = {
            **repo,
            'topics': topics,
            'readme_content': readme,
            'languages': languages,
            'collected_at': datetime.now().isoformat()
        }
        
        return enriched_repo
    
    def collect_mcp_repositories(self, test_mode=False):
        """Main collection process"""
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
            print("=== RUNNING IN TEST MODE - LIMITED TO 10 REPOS ===")
            search_queries = search_queries[:2]  # Use only first 2 queries
        
        all_repos = {}  # Use dict to avoid duplicates
        
        for query in search_queries:
            print(f"\n=== Searching for: {query} ===")
            max_results = 10 if test_mode else 500
            repos = self.search_repositories(query, max_results=max_results)
            
            # Add to our collection (using full_name as key to avoid duplicates)
            for repo in repos:
                all_repos[repo['full_name']] = repo
                if test_mode and len(all_repos) >= 10:
                    break
            
            print(f"Total unique repositories so far: {len(all_repos)}")
            
            if test_mode and len(all_repos) >= 10:
                print("Reached 10 repositories limit for test mode")
                break
                
            time.sleep(5)  # Pause between different searches
        
        # Limit to 10 repos in test mode
        if test_mode:
            all_repos = dict(list(all_repos.items())[:10])
        
        print(f"\n=== Found {len(all_repos)} unique repositories ===")
        
        # Enrich each repository with additional data
        enriched_repos = []
        for i, (full_name, repo) in enumerate(all_repos.items(), 1):
            print(f"\nProcessing {i}/{len(all_repos)}: {full_name}")
            
            enriched_repo = self.enrich_repository_data(repo)
            enriched_repos.append(enriched_repo)
            
            # Save periodically in case of interruption (every 5 in test mode)
            save_interval = 5 if test_mode else 10
            if i % save_interval == 0:
                filename = 'mcp_repositories_test_partial.json' if test_mode else 'mcp_repositories_partial.json'
                self.save_data(enriched_repos, filename)
            
            time.sleep(1)  # Rate limiting
        
        self.data = enriched_repos
        return enriched_repos
    
    def save_data(self, data: List[Dict], filename: str = 'mcp_repositories.json'):
        """Save collected data to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(data)} repositories to {filename}")


def main():
    # Check for test mode argument
    import sys
    test_mode = '--test' in sys.argv or '-t' in sys.argv
    
    # Get GitHub token from environment or prompt
    token = os.environ.get('GH_TOKEN')
    if not token:
        token = input("Enter your GitHub personal access token: ").strip()
    
    if not token:
        print("Error: GitHub token is required!")
        print("Create one at: https://github.com/settings/tokens")
        print("Required scopes: public_repo")
        return
    
    # Create collector and run
    collector = GitHubMCPCollector(token)
    
    try:
        if test_mode:
            print("Starting MCP repository collection in TEST MODE...")
            print("Will collect only 10 repositories for testing...")
        else:
            print("Starting MCP repository collection...")
            print("This may take a while due to rate limits...")
            print("Use --test or -t flag to run a quick test with 10 repos")
        
        repositories = collector.collect_mcp_repositories(test_mode=test_mode)
        
        # Save final data
        filename = 'mcp_repositories_test.json' if test_mode else 'mcp_repositories.json'
        collector.save_data(repositories, filename)
        
        # Also save a summary
        summary = {
            'total_repositories': len(repositories),
            'collection_date': datetime.now().isoformat(),
            'test_mode': test_mode,
            'repositories_by_language': {},
            'repositories_with_topics': sum(1 for r in repositories if r.get('topics')),
            'repositories_with_readme': sum(1 for r in repositories if r.get('readme_content'))
        }
        
        # Count repositories by primary language
        for repo in repositories:
            lang = repo.get('language', 'Unknown')
            summary['repositories_by_language'][lang] = summary['repositories_by_language'].get(lang, 0) + 1
        
        summary_filename = 'mcp_collection_summary_test.json' if test_mode else 'mcp_collection_summary.json'
        with open(summary_filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nCollection complete!")
        print(f"Total repositories: {len(repositories)}")
        print(f"Data saved to: {filename}")
        print(f"Summary saved to: {summary_filename}")
        
        if test_mode:
            print("\nTo run full collection, remove --test flag")
        
    except KeyboardInterrupt:
        print("\nCollection interrupted by user")
        if collector.data:
            interrupt_filename = 'mcp_repositories_test_interrupted.json' if test_mode else 'mcp_repositories_interrupted.json'
            collector.save_data(collector.data, interrupt_filename)
    except Exception as e:
        print(f"Error during collection: {e}")
        if collector.data:
            error_filename = 'mcp_repositories_test_error.json' if test_mode else 'mcp_repositories_error.json'
            collector.save_data(collector.data, error_filename)


if __name__ == "__main__":
    main()
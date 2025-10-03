#!/usr/bin/env python3
"""
NPM Package Search for MCP-related packages

Searches the npm registry for packages related to Model Context Protocol (MCP)
using the npm registry API. Creates a comprehensive list of potential MCP packages
with metadata for matching and download statistics collection.

Features:
- Searches npm registry API for MCP-related packages
- Multiple search strategies (keyword, text search)
- Extracts package metadata (description, repository, keywords)
- Filters for MCP relevance
- Exports results for usage collection pipeline

Usage:
    python usage_npm_search.py                          # Default search
    python usage_npm_search.py --output custom.json    # Custom output
    python usage_npm_search.py --limit 1000            # Limit results
"""

import argparse
import json
import time
import logging
from typing import Dict, List, Set
from datetime import datetime

import requests

# Configuration
DEFAULT_OUTPUT_FILE = "data/external-usage/usage_npm_search_results.json"
NPM_REGISTRY_API = "https://registry.npmjs.org"
NPM_SEARCH_API = "https://registry.npmjs.com/-/v1/search"

# Search terms for MCP packages
SEARCH_TERMS = [
    "mcp",
    "model-context-protocol",
    "modelcontextprotocol",
    "mcp-server",
    "mcp server"
]

def setup_logging():
    """Setup logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/usage_npm_search.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def search_npm_packages(search_term: str, size: int = 250, max_results: int = 2500) -> List[Dict]:
    """
    Search npm registry for packages matching the search term.

    Args:
        search_term: Term to search for
        size: Number of results per page (max 250)
        max_results: Maximum total results to fetch

    Returns:
        List of package objects with metadata
    """
    logger = logging.getLogger(__name__)

    packages = []
    from_offset = 0

    logger.info(f"Searching npm for: '{search_term}'")

    while from_offset < max_results:
        try:
            # npm search API endpoint
            # https://github.com/npm/registry/blob/master/docs/REGISTRY-API.md
            params = {
                'text': search_term,
                'size': min(size, max_results - from_offset),
                'from': from_offset,
                'quality': 0.65,
                'popularity': 0.98,
                'maintenance': 0.5
            }

            response = requests.get(NPM_SEARCH_API, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            results = data.get('objects', [])

            if not results:
                logger.info(f"  No more results for '{search_term}' at offset {from_offset}")
                break

            packages.extend(results)
            logger.info(f"  Fetched {len(results)} packages (total: {len(packages)})")

            # Check if we've reached the end
            total = data.get('total', 0)
            if from_offset + len(results) >= total:
                logger.info(f"  Reached end of results (total: {total})")
                break

            from_offset += len(results)

            # Rate limiting - minimal wait
            time.sleep(0.3)

        except requests.RequestException as e:
            # Check if it's a rate limit error
            if '429' in str(e):
                logger.warning(f"Rate limit hit for '{search_term}' at offset {from_offset}. Waiting 10 seconds...")
                time.sleep(10)
                # Try one more time
                try:
                    response = requests.get(NPM_SEARCH_API, params=params, timeout=30)
                    response.raise_for_status()
                    data = response.json()
                    results = data.get('objects', [])

                    if results:
                        packages.extend(results)
                        logger.info(f"  Retry successful: Fetched {len(results)} packages (total: {len(packages)})")
                        from_offset += len(results)
                        time.sleep(3)
                        continue
                except:
                    logger.warning(f"Retry failed for '{search_term}' at offset {from_offset}")
                    break
            else:
                logger.warning(f"Error searching npm for '{search_term}' at offset {from_offset}: {e}")
                break
        except Exception as e:
            logger.error(f"Unexpected error searching npm for '{search_term}': {e}")
            break

    logger.info(f"Total packages found for '{search_term}': {len(packages)}")
    return packages

def extract_package_metadata(package_obj: Dict) -> Dict:
    """
    Extract relevant metadata from npm search result object.

    Args:
        package_obj: Package object from npm search API

    Returns:
        Cleaned package metadata dictionary
    """
    package_data = package_obj.get('package', {})

    # Extract repository info
    repo_info = package_data.get('links', {}).get('repository', '')
    repo_url = ''
    github_repo = ''

    if repo_info:
        repo_url = repo_info
        # Extract GitHub repo path if it's a GitHub URL
        if 'github.com' in repo_info:
            # Parse: https://github.com/owner/repo
            try:
                parts = repo_info.replace('https://github.com/', '').replace('http://github.com/', '')
                parts = parts.split('#')[0].split('?')[0]  # Remove anchors/queries
                parts = parts.rstrip('/')
                if '/' in parts:
                    github_repo = '/'.join(parts.split('/')[:2])  # owner/repo
            except:
                pass

    # Extract keywords
    keywords = package_data.get('keywords', [])
    keywords_str = ', '.join(keywords) if keywords else ''

    # Extract maintainers
    maintainers = package_data.get('maintainers', [])
    maintainer_names = [m.get('username', '') for m in maintainers]

    return {
        'name': package_data.get('name', ''),
        'version': package_data.get('version', ''),
        'description': package_data.get('description', ''),
        'keywords': keywords_str,
        'keywords_list': keywords,
        'author': package_data.get('author', {}).get('name', '') if isinstance(package_data.get('author'), dict) else str(package_data.get('author', '')),
        'maintainers': maintainer_names,
        'repository': repo_url,
        'github_repo': github_repo,
        'homepage': package_data.get('links', {}).get('homepage', ''),
        'npm_url': package_data.get('links', {}).get('npm', ''),
        'published_date': package_data.get('date', ''),
        'publisher': package_data.get('publisher', {}).get('username', ''),
        'scope': package_data.get('scope', ''),
        'search_score': package_obj.get('score', {}).get('final', 0),
        'quality_score': package_obj.get('score', {}).get('detail', {}).get('quality', 0),
        'popularity_score': package_obj.get('score', {}).get('detail', {}).get('popularity', 0),
        'maintenance_score': package_obj.get('score', {}).get('detail', {}).get('maintenance', 0),
    }

def is_mcp_relevant(package: Dict) -> bool:
    """
    Check if package is relevant to MCP (Model Context Protocol).

    Args:
        package: Package metadata dictionary

    Returns:
        True if package appears to be MCP-related
    """
    # Search terms that indicate MCP relevance
    mcp_indicators = [
        'mcp',
        'model context protocol',
        'modelcontextprotocol',
        'model-context-protocol',
        'mcp-server',
        'mcp server',
        'claude mcp',
        'anthropic mcp'
    ]

    # Combine searchable text
    searchable_text = ' '.join([
        package.get('name', ''),
        package.get('description', ''),
        package.get('keywords', ''),
        package.get('repository', '')
    ]).lower()

    # Check if any MCP indicator is present
    for indicator in mcp_indicators:
        if indicator in searchable_text:
            return True

    return False

def deduplicate_packages(packages: List[Dict]) -> List[Dict]:
    """
    Remove duplicate packages based on package name.

    Args:
        packages: List of package metadata dictionaries

    Returns:
        Deduplicated list of packages
    """
    logger = logging.getLogger(__name__)

    seen_names = set()
    unique_packages = []

    for package in packages:
        name = package.get('name', '')
        if name and name not in seen_names:
            seen_names.add(name)
            unique_packages.append(package)

    duplicates_removed = len(packages) - len(unique_packages)
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate packages")

    return unique_packages

def search_all_npm_packages(search_terms: List[str], max_per_term: int = 2500) -> List[Dict]:
    """
    Search npm for all MCP-related packages using multiple search terms.

    Args:
        search_terms: List of search terms to use
        max_per_term: Maximum results per search term

    Returns:
        List of unique package metadata dictionaries
    """
    logger = logging.getLogger(__name__)

    all_packages = []

    logger.info(f"Starting npm package search with {len(search_terms)} search terms")

    for term in search_terms:
        results = search_npm_packages(term, size=250, max_results=max_per_term)

        # Extract metadata from each result
        for result in results:
            try:
                package = extract_package_metadata(result)
                all_packages.append(package)
            except Exception as e:
                logger.warning(f"Error extracting metadata for package: {e}")
                continue

        time.sleep(1)  # Rate limiting between search terms

    # Deduplicate
    unique_packages = deduplicate_packages(all_packages)

    # Filter for MCP relevance
    mcp_packages = [p for p in unique_packages if is_mcp_relevant(p)]

    logger.info(f"Total packages before filtering: {len(unique_packages)}")
    logger.info(f"MCP-relevant packages after filtering: {len(mcp_packages)}")

    return mcp_packages

def categorize_packages(packages: List[Dict]) -> Dict:
    """
    Categorize packages by various criteria for analysis.

    Args:
        packages: List of package metadata dictionaries

    Returns:
        Dictionary with categorized package lists and statistics
    """
    logger = logging.getLogger(__name__)

    # Categorize by GitHub presence
    with_github = [p for p in packages if p.get('github_repo')]
    without_github = [p for p in packages if not p.get('github_repo')]

    # Categorize by scope (scoped vs unscoped)
    scoped = [p for p in packages if p.get('scope')]
    unscoped = [p for p in packages if not p.get('scope')]

    # Categorize by name pattern
    server_packages = [p for p in packages if 'server' in p.get('name', '').lower()]
    client_packages = [p for p in packages if 'client' in p.get('name', '').lower()]
    tool_packages = [p for p in packages if 'tool' in p.get('name', '').lower()]

    # High quality packages (score > 0.5)
    high_quality = [p for p in packages if p.get('search_score', 0) > 0.5]

    logger.info("\n=== PACKAGE CATEGORIZATION ===")
    logger.info(f"Packages with GitHub repo: {len(with_github)}")
    logger.info(f"Packages without GitHub repo: {len(without_github)}")
    logger.info(f"Scoped packages (@org/name): {len(scoped)}")
    logger.info(f"Unscoped packages: {len(unscoped)}")
    logger.info(f"Server packages: {len(server_packages)}")
    logger.info(f"Client packages: {len(client_packages)}")
    logger.info(f"Tool packages: {len(tool_packages)}")
    logger.info(f"High quality packages (score > 0.5): {len(high_quality)}")

    return {
        'with_github': with_github,
        'without_github': without_github,
        'scoped': scoped,
        'unscoped': unscoped,
        'server_packages': server_packages,
        'client_packages': client_packages,
        'tool_packages': tool_packages,
        'high_quality': high_quality
    }

def fetch_detailed_package_info(package_name: str) -> Dict:
    """
    Fetch detailed package information from npm registry API.

    Args:
        package_name: Name of the package

    Returns:
        Dictionary with detailed package information
    """
    logger = logging.getLogger(__name__)

    try:
        # Fetch from npm registry API
        # https://github.com/npm/registry/blob/master/docs/REGISTRY-API.md
        url = f"{NPM_REGISTRY_API}/{package_name}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Get latest version info
        latest_version = data.get('dist-tags', {}).get('latest', '')
        version_data = data.get('versions', {}).get(latest_version, {})

        # Extract repository info
        repo = version_data.get('repository', {})
        if isinstance(repo, dict):
            repo_url = repo.get('url', '')
        elif isinstance(repo, str):
            repo_url = repo
        else:
            repo_url = ''

        # Clean up GitHub URL
        github_url = ''
        if repo_url:
            repo_url = repo_url.replace('git+', '').replace('git://', 'https://').replace('.git', '')
            if 'github.com' in repo_url:
                github_url = repo_url

        # Extract bugs and homepage
        bugs = version_data.get('bugs', {})
        if isinstance(bugs, dict):
            bugs_url = bugs.get('url', '')
        elif isinstance(bugs, str):
            bugs_url = bugs
        else:
            bugs_url = ''

        homepage = version_data.get('homepage', '')

        # Extract author
        author = version_data.get('author', {})
        if isinstance(author, dict):
            author_name = author.get('name', '')
            author_email = author.get('email', '')
        elif isinstance(author, str):
            author_name = author
            author_email = ''
        else:
            author_name = ''
            author_email = ''

        # Extract maintainers
        maintainers_list = data.get('maintainers', [])
        maintainers = [m.get('name', '') for m in maintainers_list if isinstance(m, dict)]

        return {
            'name': data.get('name', ''),
            'version': latest_version,
            'description': version_data.get('description', ''),
            'keywords': version_data.get('keywords', []),
            'repository': github_url or repo_url,
            'homepage': homepage,
            'bugs': bugs_url,
            'author': {
                'name': author_name,
                'email': author_email
            },
            'maintainers': maintainers,
            'dist_tags': data.get('dist-tags', {}),
            'license': version_data.get('license', ''),
            'created_at': data.get('time', {}).get('created', ''),
            'modified_at': data.get('time', {}).get('modified', '')
        }

    except Exception as e:
        logger.debug(f"Error fetching detailed info for {package_name}: {e}")
        return {}

def save_results(packages: List[Dict], output_file: str):
    """
    Save search results to JSON file with metadata.

    Args:
        packages: List of package metadata dictionaries
        output_file: Path to output JSON file
    """
    logger = logging.getLogger(__name__)

    # Categorize packages
    categories = categorize_packages(packages)

    # Create output structure
    output_data = {
        'metadata': {
            'collection_date': datetime.now().isoformat(),
            'total_packages': len(packages),
            'search_terms_used': SEARCH_TERMS,
            'api_source': 'npm_registry_search_api',
            'categories': {
                'with_github': len(categories['with_github']),
                'without_github': len(categories['without_github']),
                'scoped': len(categories['scoped']),
                'unscoped': len(categories['unscoped']),
                'server_packages': len(categories['server_packages']),
                'client_packages': len(categories['client_packages']),
                'tool_packages': len(categories['tool_packages']),
                'high_quality': len(categories['high_quality'])
            }
        },
        'packages': packages
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"\n✓ Saved {len(packages)} npm packages to {output_file}")

    # Show top 10 packages by search score
    logger.info("\n=== TOP 10 PACKAGES BY SEARCH SCORE ===")
    sorted_packages = sorted(packages, key=lambda p: p.get('search_score', 0), reverse=True)
    for i, pkg in enumerate(sorted_packages[:10], 1):
        logger.info(f"{i}. {pkg['name']} (score: {pkg['search_score']:.3f})")
        logger.info(f"   {pkg.get('description', 'No description')[:80]}")

def save_usage_compatible_format(packages: List[Dict], output_file: str):
    """
    Save results in usage_npm.json compatible format with detailed package info.

    Format matches existing usage_npm.json structure with package metadata.

    Args:
        packages: List of package metadata dictionaries
        output_file: Path to output JSON file (e.g., usage_npm_detailed.json)
    """
    logger = logging.getLogger(__name__)

    logger.info(f"\n=== Fetching detailed package information ===")
    logger.info(f"This may take a while for {len(packages)} packages...")

    # Create usage-compatible structure
    usage_packages = {}
    successful_fetches = 0
    failed_fetches = 0

    for i, pkg in enumerate(packages, 1):
        package_name = pkg.get('name', '')

        if (i % 50 == 0) or (i == 1):
            logger.info(f"Fetching detailed info {i}/{len(packages)}: {package_name}")

        # Fetch detailed info
        detailed_info = fetch_detailed_package_info(package_name)

        if detailed_info:
            # Create entry matching usage_npm.json format
            usage_packages[package_name] = {
                'monthly': {
                    '2024-11': None,
                    '2024-12': None,
                    '2025-01': None,
                    '2025-02': None,
                    '2025-03': None,
                    '2025-04': None,
                    '2025-05': None,
                    '2025-06': None,
                    '2025-07': None,
                    '2025-08': None,
                    '2025-09': None,
                    '2025-10': None
                },
                'total': None,
                'metadata': {
                    'name': detailed_info.get('name', package_name),
                    'version': detailed_info.get('version', ''),
                    'description': detailed_info.get('description', ''),
                    'keywords': detailed_info.get('keywords', []),
                    'repository': detailed_info.get('repository', ''),
                    'homepage': detailed_info.get('homepage', ''),
                    'bugs': detailed_info.get('bugs', ''),
                    'author': detailed_info.get('author', {}),
                    'maintainers': detailed_info.get('maintainers', []),
                    'dist_tags': detailed_info.get('dist_tags', {}),
                    'license': detailed_info.get('license', ''),
                    'created_at': detailed_info.get('created_at', ''),
                    'modified_at': detailed_info.get('modified_at', ''),
                    'search_score': pkg.get('search_score', 0)
                }
            }
            successful_fetches += 1
        else:
            # Create minimal entry if fetch failed
            usage_packages[package_name] = {
                'monthly': {
                    '2024-11': None,
                    '2024-12': None,
                    '2025-01': None,
                    '2025-02': None,
                    '2025-03': None,
                    '2025-04': None,
                    '2025-05': None,
                    '2025-06': None,
                    '2025-07': None,
                    '2025-08': None,
                    '2025-09': None,
                    '2025-10': None
                },
                'total': None,
                'metadata': {
                    'name': package_name,
                    'description': pkg.get('description', ''),
                    'repository': pkg.get('repository', ''),
                    'search_score': pkg.get('search_score', 0)
                }
            }
            failed_fetches += 1

        # Rate limiting - minimal wait for detailed fetch
        time.sleep(0.05)

    # Create final structure
    output_data = {
        'metadata': {
            'collection_date': datetime.now().isoformat(),
            'date_range': {
                'start_date': '2024-11-01',
                'end_date': datetime.now().date().isoformat()
            },
            'total_packages_processed': len(packages),
            'packages_with_data': 0,  # Will be updated by usage_collect_npm.py
            'total_downloads': 0,  # Will be updated by usage_collect_npm.py
            'api_source': 'npm_registry_api',
            'detailed_info_fetched': successful_fetches,
            'detailed_info_failed': failed_fetches
        },
        'packages': usage_packages
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"\n✓ Saved {len(usage_packages)} packages to {output_file}")
    logger.info(f"  Successful detailed fetches: {successful_fetches}")
    logger.info(f"  Failed fetches: {failed_fetches}")

def main():
    """Main npm search workflow."""
    parser = argparse.ArgumentParser(description="Search npm registry for MCP-related packages")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE,
                       help="Output JSON file for search results")
    parser.add_argument("--limit", type=int, default=2500,
                       help="Maximum results per search term")
    parser.add_argument("--terms", nargs='+',
                       help="Custom search terms (overrides default)")
    parser.add_argument("--detailed", action='store_true',
                       help="Fetch detailed package info and save in usage-compatible format")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()
    logger.info("Starting npm package search for MCP-related packages")

    try:
        # Use custom search terms if provided
        search_terms = args.terms if args.terms else SEARCH_TERMS
        logger.info(f"Search terms: {search_terms}")

        # Search npm registry
        packages = search_all_npm_packages(search_terms, max_per_term=args.limit)

        if not packages:
            logger.warning("No MCP-related packages found")
            return

        # Save basic results
        save_results(packages, args.output)

        # Save detailed usage-compatible format
        if args.detailed:
            usage_npm_output = "data/external-usage/usage_npm.json"

            # Backup existing file if it exists
            import shutil
            import os
            if os.path.exists(usage_npm_output):
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = f"data/external-usage/usage_npm_backup_{timestamp}.json"
                shutil.copy2(usage_npm_output, backup_path)
                logger.info(f"Backed up existing {usage_npm_output} to {backup_path}")

            # Save directly to usage_npm.json
            save_usage_compatible_format(packages, usage_npm_output)

        logger.info("\n✓ npm search completed successfully")
        logger.info(f"Found {len(packages)} MCP-related npm packages")
        logger.info(f"Results saved to: {args.output}")
        if args.detailed and 'usage_npm_output' in locals():
            logger.info(f"Detailed usage format saved to: {usage_npm_output}")

    except Exception as e:
        logger.error(f"npm search failed: {e}")
        raise

if __name__ == "__main__":
    main()

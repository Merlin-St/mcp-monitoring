#!/usr/bin/env python3
"""
MCP Package Usage Statistics Collection

Collects download statistics for MCP packages using the final 1:1 matched packages
and integrates them into the unified dataset. This replaces the old Libraries.io approach
with direct PyPI and npm API queries based on strict 1:1 package-to-repository matching.

Features:
- Uses final matched packages from strict 1:1 matching (70.6% coverage)
- Uses pre-downloaded PyPI stats from BigQuery data
- Uses pre-collected npm stats from usage_npm.json
- Monthly breakdown from Nov 2024 to present
- Modifies data_unified_filtered.json in place

PyPI Data Collection:
To get PyPI download statistics, run this query in Google Cloud Console BigQuery web UI:

SELECT 
    file.project AS package_name,
    FORMAT_DATE('%Y-%m', DATE_TRUNC(DATE(timestamp), MONTH)) AS month,
    COUNT(*) AS downloads
FROM 
    `bigquery-public-data.pypi.file_downloads`
WHERE 
    LOWER(file.project) LIKE '%mcp%'
    AND DATE(timestamp) >= '2024-11-01'
    AND DATE(timestamp) < '2025-09-01'
GROUP BY 
    package_name, 
    month
ORDER BY 
    package_name, 
    month

Export the results as JSON and save as 'usage_bigquery_webresults_pypi.json'
"""

import argparse
import datetime as dt
import json
from typing import Dict, List, Tuple
import logging


# File paths
MATCHED_PACKAGES_FILE = "data/external-usage/usage_match.json"
DATASET_FILE = "data/initial/data_unified_filtered.json"

# Data source files
pypi_data_file = "data/external-usage/usage_bigquery_webresults_pypi.json"
npm_data_file = "data/external-usage/usage_npm.json"


# Reporting window 
START_DATE = dt.date(2024, 11, 1)
END_DATE = dt.date.today()

def setup_logging():
    """Setup logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/data_unified_add_usage.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_matched_packages(file_path: str) -> Tuple[List[Dict], List[Dict]]:
    """Load final matched packages from strict 1:1 matching results."""
    logger = logging.getLogger(__name__)
    
    with open(file_path) as f:
        data = json.load(f)
    
    tier1_matches = data.get('tier1_confirmed_matches', [])
    tier2_matches = data.get('tier2_strict_matches', [])
    
    logger.info(f"Loaded {len(tier1_matches)} Tier 1 matches and {len(tier2_matches)} Tier 2 matches")
    
    return tier1_matches, tier2_matches

def get_all_packages(tier1_matches: List[Dict], tier2_matches: List[Dict]) -> Dict[str, List[Dict]]:
    """Extract all packages organized by platform and create repo mapping."""
    pypi_packages = []
    npm_packages = []
    package_to_repo_mapping = {}
    
    # Process Tier 1 matches
    for match in tier1_matches:
        platform = match.get('platform', '')
        package_name = match.get('package_name', '')
        repo_id = match.get('dataset_id', '')
        
        if platform == 'pypi' and package_name and repo_id:
            pypi_packages.append({
                'name': package_name,
                'repo_id': repo_id,
                'tier': 'confirmed',
                'score': match.get('match_score', 100)
            })
            package_to_repo_mapping[('pypi', package_name)] = repo_id
            
        elif platform == 'npm' and package_name and repo_id:
            npm_packages.append({
                'name': package_name,
                'repo_id': repo_id,
                'tier': 'confirmed',
                'score': match.get('match_score', 100)
            })
            package_to_repo_mapping[('npm', package_name)] = repo_id
    
    # Process Tier 2 matches
    for match in tier2_matches:
        platform = match.get('platform', '')
        package_name = match.get('package_name', '')
        repo_id = match.get('repo_id', '')
        
        if platform == 'pypi' and package_name and repo_id:
            pypi_packages.append({
                'name': package_name,
                'repo_id': repo_id,
                'tier': match.get('tier', 'strict'),
                'score': match.get('score', 0)
            })
            package_to_repo_mapping[('pypi', package_name)] = repo_id
            
        elif platform == 'npm' and package_name and repo_id:
            npm_packages.append({
                'name': package_name,
                'repo_id': repo_id,
                'tier': match.get('tier', 'strict'),
                'score': match.get('score', 0)
            })
            package_to_repo_mapping[('npm', package_name)] = repo_id
    
    return {
        'pypi': pypi_packages,
        'npm': npm_packages,
        'mapping': package_to_repo_mapping
    }

def collect_pypi_downloads(pypi_packages: List[Dict], start_date: dt.date, end_date: dt.date, batch_size: int = 50) -> Dict[str, Dict]:
    """Collect PyPI download statistics using pre-downloaded BigQuery data."""
    logger = logging.getLogger(__name__)
    
    if not pypi_packages:
        logger.info("No PyPI packages to process")
        return {}
    
    logger.info(f"Collecting PyPI download stats for {len(pypi_packages)} packages")
    
    # Load pre-downloaded PyPI data
    try:
        with open(pypi_data_file) as f:
            raw_pypi_data = json.load(f)
        logger.info(f"Loaded {len(raw_pypi_data):,} PyPI download records from {pypi_data_file}")
    except Exception as e:
        logger.error(f"Failed to load pre-downloaded PyPI data from {pypi_data_file}: {e}")
        return {}
    
    # Create a lookup for fast access: package_name -> {month: downloads}
    pypi_lookup = {}
    for record in raw_pypi_data:
        package_name = record['package_name']
        month = record['month']
        downloads = int(record['downloads'])
        
        if package_name not in pypi_lookup:
            pypi_lookup[package_name] = {}
        pypi_lookup[package_name][month] = downloads
    
    logger.info(f"Built PyPI lookup for {len(pypi_lookup):,} unique packages")
    
    # Match our packages with the pre-downloaded data
    download_stats = {}
    total_downloads = 0
    matched_packages = 0
    
    for pkg in pypi_packages:
        package_name = pkg['name']
        
        if package_name in pypi_lookup:
            matched_packages += 1
            package_monthly = pypi_lookup[package_name]
            package_total = sum(package_monthly.values())
            
            download_stats[package_name] = {
                'monthly': package_monthly,
                'total': package_total
            }
            
            total_downloads += package_total
        else:
            # Package not found in pre-downloaded data
            download_stats[package_name] = {
                'monthly': {},
                'total': 0
            }
    
    logger.info(f"Matched {matched_packages}/{len(pypi_packages)} packages ({matched_packages/len(pypi_packages)*100:.1f}%)")
    logger.info(f"Collected PyPI stats: {len(download_stats)} packages, {total_downloads:,} total downloads")
    
    return download_stats

def load_npm_downloads(npm_packages: List[Dict]) -> Dict[str, Dict]:
    """Load npm download statistics from pre-collected usage_npm.json."""
    logger = logging.getLogger(__name__)
    
    if not npm_packages:
        logger.info("No npm packages to process")
        return {}
    
    logger.info(f"Loading npm download stats for {len(npm_packages)} packages")
    
    # Load pre-collected npm data
    try:
        with open(npm_data_file) as f:
            npm_data = json.load(f)
        
        packages_data = npm_data.get('packages', {})
        logger.info(f"Loaded npm data from {npm_data_file} ({len(packages_data)} packages)")
        
        # Show collection metadata if available
        if 'metadata' in npm_data:
            metadata = npm_data['metadata']
            logger.info(f"npm data collection date: {metadata.get('collection_date', 'unknown')}")
            logger.info(f"npm data date range: {metadata.get('date_range', {}).get('start_date')} to {metadata.get('date_range', {}).get('end_date')}")
            logger.info(f"npm total downloads in source: {metadata.get('total_downloads', 0):,}")
        
    except Exception as e:
        logger.error(f"Failed to load npm data from {npm_data_file}: {e}")
        return {}
    
    # Match our packages with the pre-collected data
    download_stats = {}
    matched_packages = 0
    total_downloads = 0
    
    for pkg in npm_packages:
        package_name = pkg['name']
        
        if package_name in packages_data:
            # Package found in pre-collected data
            package_data = packages_data[package_name]
            download_stats[package_name] = {
                'monthly': package_data.get('monthly', {}),
                'total': package_data.get('total', 0)
            }
            matched_packages += 1
            total_downloads += package_data.get('total', 0)
        else:
            # Package not found in pre-collected data
            download_stats[package_name] = {
                'monthly': {},
                'total': 0
            }
    
    logger.info(f"Matched {matched_packages}/{len(npm_packages)} packages ({matched_packages/len(npm_packages)*100:.1f}%)")
    logger.info(f"Loaded npm stats: {len(download_stats)} packages, {total_downloads:,} total downloads")
    
    return download_stats

def integrate_download_stats(dataset_file: str, pypi_stats: Dict, npm_stats: Dict, 
                           package_mapping: Dict, start_date: dt.date, end_date: dt.date) -> List[Dict]:
    """Integrate download statistics into the unified dataset."""
    logger = logging.getLogger(__name__)
    
    # Load dataset
    with open(dataset_file) as f:
        dataset = json.load(f)
    
    logger.info(f"Integrating download stats into dataset with {len(dataset)} repositories")
    
    # Generate complete month list
    months = []
    current = start_date.replace(day=1)
    end_month = end_date.replace(day=1)
    
    while current <= end_month:
        months.append(current.strftime('%Y-%m'))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    
    # Create reverse mapping: repo_id -> packages
    repo_to_packages = {}
    for (platform, package_name), repo_id in package_mapping.items():
        if repo_id not in repo_to_packages:
            repo_to_packages[repo_id] = {'pypi': [], 'npm': []}
        repo_to_packages[repo_id][platform].append(package_name)
    
    # Integrate stats into dataset
    updated_count = 0
    
    for repo in dataset:
        repo_id = repo.get('id', '')
        
        if repo_id in repo_to_packages:
            # This repository has matched packages
            matched_packages = repo_to_packages[repo_id]
            
            # Initialize usage fields
            pypi_total = 0
            npm_total = 0
            monthly_breakdown = []
            
            # Calculate totals and monthly breakdown
            for month in months:
                pypi_month = 0
                npm_month = 0
                
                # Sum PyPI downloads for this month
                for pkg_name in matched_packages['pypi']:
                    if pkg_name in pypi_stats:
                        pypi_month += pypi_stats[pkg_name]['monthly'].get(month, 0)
                
                # Sum npm downloads for this month  
                for pkg_name in matched_packages['npm']:
                    if pkg_name in npm_stats:
                        npm_month += npm_stats[pkg_name]['monthly'].get(month, 0)
                
                monthly_breakdown.append({
                    'month': month,
                    'pypi': pypi_month,
                    'npm': npm_month
                })
                
                pypi_total += pypi_month
                npm_total += npm_month
            
            # Add usage fields to repository
            repo['usage_pypi_downloads'] = pypi_total
            repo['usage_npm_downloads'] = npm_total
            repo['usage_total_downloads'] = pypi_total + npm_total
            repo['usage_monthly_breakdown'] = monthly_breakdown
            repo['usage_matched_packages'] = matched_packages
            repo['usage_last_updated'] = dt.date.today().isoformat()
            
            updated_count += 1
        else:
            # Repository has no matched packages
            repo['usage_pypi_downloads'] = 0
            repo['usage_npm_downloads'] = 0
            repo['usage_total_downloads'] = 0
            repo['usage_monthly_breakdown'] = [{'month': month, 'pypi': 0, 'npm': 0} for month in months]
            repo['usage_matched_packages'] = {'pypi': [], 'npm': []}
            repo['usage_last_updated'] = dt.date.today().isoformat()
    
    logger.info(f"Updated {updated_count} repositories with download statistics")
    
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Collect and integrate MCP package usage statistics")
    parser.add_argument("--matched-packages", default="data/external-usage/usage_match.json", 
                       help="Final matched packages file")
    parser.add_argument("--dataset", default=DATASET_FILE, 
                       help="Dataset file to integrate stats into (modified in place)")
    parser.add_argument("--skip-pypi", action="store_true", 
                       help="Skip PyPI download collection")
    parser.add_argument("--skip-npm", action="store_true", 
                       help="Skip npm download loading")
    parser.add_argument("--pypi-batch-size", type=int, default=50,
                       help="Batch size for PyPI queries (smaller = less quota usage)")
    parser.add_argument("--test-pypi", action="store_true",
                       help="Test with only first 10 PyPI packages")
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("=== MCP Package Usage Statistics Collection ===")
    logger.info(f"Date range: {START_DATE} to {END_DATE}")
    
    # Load matched packages
    tier1_matches, tier2_matches = load_matched_packages(args.matched_packages)
    packages_data = get_all_packages(tier1_matches, tier2_matches)
    
    # Apply test mode if requested
    if args.test_pypi:
        packages_data['pypi'] = packages_data['pypi'][:10]
        logger.info("TEST MODE: Using only first 10 PyPI packages")
    
    logger.info(f"Packages to process: {len(packages_data['pypi'])} PyPI, {len(packages_data['npm'])} npm")
    
    # Collect download statistics
    pypi_stats = {}
    npm_stats = {}
    
    if not args.skip_pypi:
        pypi_stats = collect_pypi_downloads(packages_data['pypi'], START_DATE, END_DATE, args.pypi_batch_size)
    else:
        logger.info("Skipping PyPI download collection")
    
    if not args.skip_npm:
        npm_stats = load_npm_downloads(packages_data['npm'])
    else:
        logger.info("Skipping npm download loading")
    
    # Integrate statistics into dataset
    updated_dataset = integrate_download_stats(
        args.dataset, pypi_stats, npm_stats, packages_data['mapping'], START_DATE, END_DATE
    )
    
    # Save updated dataset back to the same file
    with open(args.dataset, 'w') as f:
        json.dump(updated_dataset, f, indent=2)
    
    logger.info(f"Updated {args.dataset} with usage statistics")
    
    # Summary statistics
    total_pypi_downloads = sum(stats['total'] for stats in pypi_stats.values())
    total_npm_downloads = sum(stats['total'] for stats in npm_stats.values())
    repos_with_stats = sum(1 for repo in updated_dataset if repo.get('usage_total_downloads', 0) > 0)
    
    logger.info("=== SUMMARY ===")
    logger.info(f"Total PyPI downloads collected: {total_pypi_downloads:,}")
    logger.info(f"Total npm downloads collected: {total_npm_downloads:,}")
    logger.info(f"Repositories with download stats: {repos_with_stats}")
    logger.info(f"Coverage: {repos_with_stats/len(updated_dataset)*100:.1f}% of {len(updated_dataset)} repositories")

if __name__ == "__main__":
    main()
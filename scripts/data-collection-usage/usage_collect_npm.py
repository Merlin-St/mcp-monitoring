#!/usr/bin/env python3
"""
NPM Package Download Statistics Collector

Collects download statistics for npm packages using the npm API.
Updates usage_npm.json with real download numbers.

Features:
- Reads package list from usage_npm.json
- Fetches download statistics from npm API
- Updates monthly breakdown and totals
- Preserves all existing metadata
- Skips packages that already have download data
- Rate limiting to respect API limits

Usage:
    python usage_collect_npm.py                           # Update usage_npm.json
    python usage_collect_npm.py --input custom.json      # Custom input file
    python usage_collect_npm.py --start-date 2024-01-01  # Custom date range
    python usage_collect_npm.py --force                   # Refetch all packages
"""

import argparse
import datetime as dt
import json
import time
from typing import Dict, List, Set
import logging

import requests

# Configuration
DEFAULT_INPUT_FILE = "data/external-usage/usage_npm.json"
DEFAULT_START_DATE = dt.date(2024, 11, 1)
DEFAULT_END_DATE = dt.date.today()

def setup_logging():
    """Setup logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/usage_collect_npm.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_usage_npm_file(file_path: str) -> Dict:
    """Load usage_npm.json file with package data."""
    logger = logging.getLogger(__name__)

    with open(file_path) as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data.get('packages', {}))} packages from {file_path}")

    return data

def get_packages_to_fetch(data: Dict, force: bool = False) -> List[str]:
    """
    Get list of package names that need download data.

    Args:
        data: Usage npm data dictionary
        force: If True, refetch all packages regardless of existing data

    Returns:
        List of package names to fetch
    """
    logger = logging.getLogger(__name__)
    packages_to_fetch = []
    packages_with_data = []

    for package_name, package_data in data['packages'].items():
        # Check if package already has download data
        has_data = False
        if not force and 'total' in package_data:
            # Check if total is not None/null
            if package_data['total'] is not None and package_data['total'] > 0:
                has_data = True
                packages_with_data.append(package_name)

        if not has_data:
            packages_to_fetch.append(package_name)

    logger.info(f"Packages with existing data: {len(packages_with_data)}")
    logger.info(f"Packages to fetch: {len(packages_to_fetch)}")

    if force:
        logger.info("Force mode: refetching all packages")

    return packages_to_fetch

def collect_npm_downloads(package_names: List[str], start_date: dt.date, end_date: dt.date) -> Dict[str, Dict]:
    """Collect npm download statistics using npm API."""
    logger = logging.getLogger(__name__)

    if not package_names:
        logger.info("No packages to process")
        return {}

    logger.info(f"Collecting npm download stats for {len(package_names)} packages")
    logger.info(f"Date range: {start_date} to {end_date}")

    download_stats = {}
    successful_requests = 0
    failed_requests = 0
    not_found_packages = 0

    # Generate month list for reference
    months = []
    current = start_date.replace(day=1)
    end_month = end_date.replace(day=1)

    while current <= end_month:
        months.append(current.strftime('%Y-%m'))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    logger.info(f"Tracking downloads across {len(months)} months: {months[0]} to {months[-1]}")

    # Collect stats for each package
    for i, package_name in enumerate(package_names):
        if (i + 1) % 50 == 0 or i == 0:
            logger.info(f"Processing npm package {i+1}/{len(package_names)}: {package_name}")

        try:
            # Use npm API to get download stats
            # Format: YYYY-MM-DD:YYYY-MM-DD for date range
            range_str = f"{start_date}:{end_date}"
            url = f"https://api.npmjs.org/downloads/range/{range_str}/{package_name}"

            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()

                if 'downloads' in data:
                    monthly_stats = {}
                    total = 0

                    # Initialize all months with 0
                    for month in months:
                        monthly_stats[month] = 0

                    # Aggregate daily downloads by month
                    for download_entry in data['downloads']:
                        date_str = download_entry['day']  # YYYY-MM-DD
                        downloads = download_entry['downloads']
                        month = date_str[:7]  # YYYY-MM

                        if month in monthly_stats:
                            monthly_stats[month] += downloads
                        total += downloads

                    download_stats[package_name] = {
                        'monthly': monthly_stats,
                        'total': total
                    }
                    successful_requests += 1
                else:
                    # API returned 200 but no downloads data
                    download_stats[package_name] = {
                        'monthly': {month: 0 for month in months},
                        'total': 0
                    }
                    successful_requests += 1

            elif response.status_code == 404:
                # Package not found - this is normal for some packages
                download_stats[package_name] = {
                    'monthly': {month: 0 for month in months},
                    'total': 0
                }
                not_found_packages += 1
            else:
                # Other HTTP error
                logger.warning(f"Error fetching {package_name}: HTTP {response.status_code}")
                download_stats[package_name] = {
                    'monthly': {month: 0 for month in months},
                    'total': 0
                }
                failed_requests += 1

        except requests.RequestException as e:
            logger.warning(f"Network error fetching {package_name}: {e}")
            download_stats[package_name] = {
                'monthly': {month: 0 for month in months},
                'total': 0
            }
            failed_requests += 1
        except Exception as e:
            logger.error(f"Unexpected error processing {package_name}: {e}")
            download_stats[package_name] = {
                'monthly': {month: 0 for month in months},
                'total': 0
            }
            failed_requests += 1

        # Rate limiting - be respectful to npm API
        time.sleep(0.1)

    # Calculate summary statistics
    total_downloads = sum(stats['total'] for stats in download_stats.values())
    packages_with_data = sum(1 for stats in download_stats.values() if stats['total'] > 0)

    logger.info("=== NPM COLLECTION SUMMARY ===")
    logger.info(f"Total packages processed: {len(package_names)}")
    logger.info(f"Successful API requests: {successful_requests}")
    logger.info(f"Failed requests: {failed_requests}")
    logger.info(f"Packages not found (404): {not_found_packages}")
    logger.info(f"Packages with download data: {packages_with_data}")
    logger.info(f"Total npm downloads collected: {total_downloads:,}")

    return download_stats

def update_usage_npm_file(data: Dict, download_stats: Dict[str, Dict], output_file: str,
                         start_date: dt.date, end_date: dt.date):
    """
    Update usage_npm.json with download statistics.

    Preserves all existing metadata and only updates monthly/total fields.
    """
    logger = logging.getLogger(__name__)

    # Update packages with download data
    updated_count = 0
    for package_name, stats in download_stats.items():
        if package_name in data['packages']:
            # Update monthly and total, preserve metadata
            data['packages'][package_name]['monthly'] = stats['monthly']
            data['packages'][package_name]['total'] = stats['total']
            updated_count += 1

    # Update metadata
    packages_with_data = sum(1 for pkg in data['packages'].values()
                            if pkg.get('total') is not None and pkg.get('total', 0) > 0)
    total_downloads = sum(pkg.get('total', 0) for pkg in data['packages'].values()
                         if pkg.get('total') is not None)

    data['metadata']['collection_date'] = dt.datetime.now().isoformat()
    data['metadata']['date_range'] = {
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat()
    }
    data['metadata']['packages_with_data'] = packages_with_data
    data['metadata']['total_downloads'] = total_downloads

    # Save updated data
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"✓ Updated {updated_count} packages in {output_file}")
    logger.info(f"✓ Total packages with data: {packages_with_data}")
    logger.info(f"✓ Total downloads: {total_downloads:,}")

def main():
    """Main npm collection workflow."""
    parser = argparse.ArgumentParser(description="Collect npm package download statistics")
    parser.add_argument("--input", default=DEFAULT_INPUT_FILE,
                       help="Path to usage_npm.json file")
    parser.add_argument("--output",
                       help="Output JSON file (defaults to same as input)")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE.isoformat(),
                       help="Start date for collection (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE.isoformat(),
                       help="End date for collection (YYYY-MM-DD)")
    parser.add_argument("--force", action='store_true',
                       help="Refetch all packages, ignoring existing data")

    args = parser.parse_args()

    # Default output to same as input
    output_file = args.output or args.input

    # Setup logging
    logger = setup_logging()
    logger.info("Starting npm package download statistics collection")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output file: {output_file}")

    try:
        # Parse dates
        start_date = dt.datetime.strptime(args.start_date, '%Y-%m-%d').date()
        end_date = dt.datetime.strptime(args.end_date, '%Y-%m-%d').date()

        logger.info(f"Collection period: {start_date} to {end_date}")

        # Load existing usage_npm.json
        data = load_usage_npm_file(args.input)

        if not data.get('packages'):
            logger.error("No packages found in input file")
            return

        # Get packages that need download data
        packages_to_fetch = get_packages_to_fetch(data, force=args.force)

        if not packages_to_fetch:
            logger.info("All packages already have download data. Use --force to refetch.")
            return

        # Collect download statistics
        download_stats = collect_npm_downloads(packages_to_fetch, start_date, end_date)

        # Update and save results
        update_usage_npm_file(data, download_stats, output_file, start_date, end_date)

        logger.info("✓ npm collection completed successfully")

    except Exception as e:
        logger.error(f"npm collection failed: {e}")
        raise

if __name__ == "__main__":
    main()

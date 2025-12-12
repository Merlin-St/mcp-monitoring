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
- Supports incremental updates for specific months

Usage:
    python usage_collect_npm.py                           # Update usage_npm.json
    python usage_collect_npm.py --input custom.json      # Custom input file
    python usage_collect_npm.py --start-date 2024-01-01  # Custom date range
    python usage_collect_npm.py --force                   # Refetch all packages
    python usage_collect_npm.py --months 2025-10 2025-11 # Collect specific months only
"""

import argparse
import datetime as dt
import json
import time
from typing import Dict, List, Tuple
import re
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


def parse_months_to_date_range(months: List[str]) -> Tuple[dt.date, dt.date, List[str]]:
    """
    Convert list of YYYY-MM month strings to start and end dates.

    Args:
        months: List of month strings like ['2025-10', '2025-11']

    Returns:
        Tuple of (start_date, end_date, validated_months)

    Raises:
        ValueError: If month format is invalid
    """
    logger = logging.getLogger(__name__)

    pattern = re.compile(r'^(\d{4})-(0[1-9]|1[0-2])$')
    parsed_months = []

    for month in months:
        match = pattern.match(month)
        if not match:
            raise ValueError(f"Invalid month format: {month}. Expected YYYY-MM.")
        year, month_num = int(match.group(1)), int(match.group(2))
        parsed_months.append((year, month_num, month))

    # Sort by date
    parsed_months.sort(key=lambda x: (x[0], x[1]))
    validated_months = [m[2] for m in parsed_months]

    # Get start date (first day of earliest month)
    first_year, first_month = parsed_months[0][0], parsed_months[0][1]
    start_date = dt.date(first_year, first_month, 1)

    # Get end date (last day of latest month)
    last_year, last_month = parsed_months[-1][0], parsed_months[-1][1]
    if last_month == 12:
        end_date = dt.date(last_year + 1, 1, 1) - dt.timedelta(days=1)
    else:
        end_date = dt.date(last_year, last_month + 1, 1) - dt.timedelta(days=1)

    logger.info(f"Parsed months {validated_months} to date range: {start_date} to {end_date}")
    return start_date, end_date, validated_months


def merge_monthly_data(existing_monthly: Dict[str, int], new_monthly: Dict[str, int],
                       target_months: List[str]) -> Dict[str, int]:
    """
    Merge new monthly data into existing data, only updating specified months.

    Args:
        existing_monthly: Existing monthly download counts
        new_monthly: New monthly download counts from API
        target_months: List of months to update (e.g., ['2025-10', '2025-11'])

    Returns:
        Merged monthly data with target months updated
    """
    merged = dict(existing_monthly)  # Copy existing

    for month in target_months:
        if month in new_monthly:
            merged[month] = new_monthly[month]
        else:
            # New month not in API response - set to 0
            merged[month] = 0

    return merged


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
            elif response.status_code == 429:
                # Rate limited - wait and retry once
                logger.warning(f"Rate limited on {package_name}, waiting 60s and retrying...")
                time.sleep(60)
                retry_response = requests.get(url, timeout=10)
                if retry_response.status_code == 200:
                    data = retry_response.json()
                    if 'downloads' in data:
                        monthly_stats = {}
                        total = 0
                        for month in months:
                            monthly_stats[month] = 0
                        for download_entry in data['downloads']:
                            date_str = download_entry['day']
                            downloads = download_entry['downloads']
                            month = date_str[:7]
                            if month in monthly_stats:
                                monthly_stats[month] += downloads
                            total += downloads
                        download_stats[package_name] = {
                            'monthly': monthly_stats,
                            'total': total
                        }
                        successful_requests += 1
                    else:
                        download_stats[package_name] = {
                            'monthly': {month: 0 for month in months},
                            'total': 0
                        }
                        successful_requests += 1
                else:
                    logger.warning(f"Retry failed for {package_name}: HTTP {retry_response.status_code}")
                    download_stats[package_name] = {
                        'monthly': {month: 0 for month in months},
                        'total': 0
                    }
                    failed_requests += 1
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
        time.sleep(1.0)

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
                         start_date: dt.date, end_date: dt.date,
                         target_months: List[str] = None):
    """
    Update usage_npm.json with download statistics.

    Preserves all existing metadata and only updates monthly/total fields.
    If target_months is provided, only those months are updated (incremental mode).

    Args:
        data: Usage npm data dictionary
        download_stats: Download statistics from npm API
        output_file: Path to output file
        start_date: Collection start date
        end_date: Collection end date
        target_months: Optional list of months to update (incremental mode)
    """
    logger = logging.getLogger(__name__)

    if target_months:
        logger.info(f"Incremental mode: updating only months {target_months}")

    # Update packages with download data
    updated_count = 0
    for package_name, stats in download_stats.items():
        if package_name in data['packages']:
            existing_pkg = data['packages'][package_name]

            if target_months:
                # Incremental update: merge only target months
                existing_monthly = existing_pkg.get('monthly', {})
                new_monthly = stats['monthly']
                merged_monthly = merge_monthly_data(existing_monthly, new_monthly, target_months)
                existing_pkg['monthly'] = merged_monthly
                existing_pkg['total'] = sum(merged_monthly.values())
            else:
                # Full update (original behavior)
                existing_pkg['monthly'] = stats['monthly']
                existing_pkg['total'] = stats['total']

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
    parser.add_argument("--months", nargs='+', type=str,
                       help="Specific months to collect (YYYY-MM format, e.g., --months 2025-10 2025-11). "
                            "If provided, only these months are collected and merged with existing data.")
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
        # Handle --months flag
        target_months = None
        if args.months:
            start_date, end_date, target_months = parse_months_to_date_range(args.months)
            logger.info(f"Month-specific mode: collecting {target_months}")
            # Force refetch when using --months to get latest data for those months
            force = True
        else:
            # Parse dates from arguments
            start_date = dt.datetime.strptime(args.start_date, '%Y-%m-%d').date()
            end_date = dt.datetime.strptime(args.end_date, '%Y-%m-%d').date()
            force = args.force

        logger.info(f"Collection period: {start_date} to {end_date}")

        # Load existing usage_npm.json
        data = load_usage_npm_file(args.input)

        if not data.get('packages'):
            logger.error("No packages found in input file")
            return

        # Get packages that need download data
        packages_to_fetch = get_packages_to_fetch(data, force=force)

        if not packages_to_fetch:
            logger.info("All packages already have download data. Use --force or --months to refetch.")
            return

        # Collect download statistics
        download_stats = collect_npm_downloads(packages_to_fetch, start_date, end_date)

        # Update and save results (pass target_months for incremental mode)
        update_usage_npm_file(data, download_stats, output_file, start_date, end_date,
                             target_months=target_months)

        logger.info("npm collection completed successfully")

    except Exception as e:
        logger.error(f"npm collection failed: {e}")
        raise

if __name__ == "__main__":
    main()

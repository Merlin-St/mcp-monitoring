#!/usr/bin/env python3
"""
NPM Package Download Statistics Collector

Collects download statistics for npm packages using the npm API.
Creates usage_npm.json for integration into the main usage pipeline.

Features:
- Fetches download statistics from npm API
- Monthly breakdown from configurable date range
- Rate limiting to respect API limits
- Error handling for missing packages
- JSON output compatible with main usage pipeline

Usage:
    python usage_collect_npm.py                           # Use default matched packages
    python usage_collect_npm.py --packages custom.json   # Use custom package list
    python usage_collect_npm.py --start-date 2024-01-01  # Custom date range
"""

import argparse
import datetime as dt
import json
import time
from typing import Dict, List
import logging

import requests

# Configuration
DEFAULT_MATCHED_PACKAGES_FILE = "usage_match.json"
DEFAULT_OUTPUT_FILE = "usage_npm.json"
DEFAULT_START_DATE = dt.date(2024, 11, 1)
DEFAULT_END_DATE = dt.date.today()

def setup_logging():
    """Setup logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('usage_collect_npm.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_npm_packages(file_path: str) -> List[Dict]:
    """Load npm package list from matched packages JSON."""
    logger = logging.getLogger(__name__)
    
    with open(file_path) as f:
        data = json.load(f)
    
    # Extract npm packages from all matching arrays
    npm_packages = []
    
    # Check tier1_confirmed_matches
    tier1_matches = data.get('tier1_confirmed_matches', [])
    tier1_npm = [pkg for pkg in tier1_matches if pkg.get('platform') == 'npm']
    npm_packages.extend(tier1_npm)
    
    # Check tier2_strict_matches  
    tier2_matches = data.get('tier2_strict_matches', [])
    tier2_npm = [pkg for pkg in tier2_matches if pkg.get('platform') == 'npm']
    npm_packages.extend(tier2_npm)
    
    # Convert to expected format for npm API calls
    npm_package_list = []
    for pkg in npm_packages:
        npm_package_list.append({
            'name': pkg.get('package_name', ''),
            'platform': 'npm',
            'repository': pkg.get('repository', ''),
            'dataset_id': pkg.get('dataset_id', '')
        })
    
    logger.info(f"Loaded {len(npm_package_list)} npm packages from {file_path}")
    logger.info(f"Found {len(tier1_npm)} tier1 + {len(tier2_npm)} tier2 npm packages")
    
    return npm_package_list

def collect_npm_downloads(npm_packages: List[Dict], start_date: dt.date, end_date: dt.date) -> Dict[str, Dict]:
    """Collect npm download statistics using npm API."""
    logger = logging.getLogger(__name__)
    
    if not npm_packages:
        logger.info("No npm packages to process")
        return {}
    
    logger.info(f"Collecting npm download stats for {len(npm_packages)} packages")
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
    for i, pkg in enumerate(npm_packages):
        package_name = pkg['name']
        
        if (i + 1) % 50 == 0 or i == 0:
            logger.info(f"Processing npm package {i+1}/{len(npm_packages)}: {package_name}")
        
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
                    
                    # Aggregate daily downloads by month
                    for download_entry in data['downloads']:
                        date_str = download_entry['day']  # YYYY-MM-DD
                        downloads = download_entry['downloads']
                        month = date_str[:7]  # YYYY-MM
                        
                        if month not in monthly_stats:
                            monthly_stats[month] = 0
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
                        'monthly': {},
                        'total': 0
                    }
                    successful_requests += 1
                    
            elif response.status_code == 404:
                # Package not found - this is normal for some packages
                download_stats[package_name] = {
                    'monthly': {},
                    'total': 0
                }
                not_found_packages += 1
            else:
                # Other HTTP error
                logger.warning(f"Error fetching {package_name}: HTTP {response.status_code}")
                download_stats[package_name] = {
                    'monthly': {},
                    'total': 0
                }
                failed_requests += 1
                
        except requests.RequestException as e:
            logger.warning(f"Network error fetching {package_name}: {e}")
            download_stats[package_name] = {
                'monthly': {},
                'total': 0
            }
            failed_requests += 1
        except Exception as e:
            logger.error(f"Unexpected error processing {package_name}: {e}")
            download_stats[package_name] = {
                'monthly': {},
                'total': 0
            }
            failed_requests += 1
        
        # Rate limiting - be respectful to npm API
        time.sleep(0.1)
    
    # Calculate summary statistics
    total_downloads = sum(stats['total'] for stats in download_stats.values())
    packages_with_data = sum(1 for stats in download_stats.values() if stats['total'] > 0)
    
    logger.info("=== NPM COLLECTION SUMMARY ===")
    logger.info(f"Total packages processed: {len(npm_packages)}")
    logger.info(f"Successful API requests: {successful_requests}")
    logger.info(f"Failed requests: {failed_requests}")
    logger.info(f"Packages not found (404): {not_found_packages}")
    logger.info(f"Packages with download data: {packages_with_data}")
    logger.info(f"Total npm downloads collected: {total_downloads:,}")
    
    return download_stats

def save_npm_stats(download_stats: Dict[str, Dict], output_file: str, 
                   start_date: dt.date, end_date: dt.date, npm_packages: List[Dict]):
    """Save npm download statistics to JSON file."""
    logger = logging.getLogger(__name__)
    
    # Prepare output data with metadata
    output_data = {
        'metadata': {
            'collection_date': dt.date.today().isoformat(),
            'date_range': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'total_packages_processed': len(npm_packages),
            'packages_with_data': sum(1 for stats in download_stats.values() if stats['total'] > 0),
            'total_downloads': sum(stats['total'] for stats in download_stats.values()),
            'api_source': 'npm_api'
        },
        'packages': download_stats
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Saved npm download statistics to {output_file}")
    logger.info(f"File contains data for {len(download_stats)} packages")

def main():
    """Main npm collection workflow."""
    parser = argparse.ArgumentParser(description="Collect npm package download statistics")
    parser.add_argument("--packages", default=DEFAULT_MATCHED_PACKAGES_FILE,
                       help="Path to matched packages JSON file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE,
                       help="Output JSON file for npm statistics")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE.isoformat(),
                       help="Start date for collection (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE.isoformat(),
                       help="End date for collection (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting npm package download statistics collection")
    
    try:
        # Parse dates
        start_date = dt.datetime.strptime(args.start_date, '%Y-%m-%d').date()
        end_date = dt.datetime.strptime(args.end_date, '%Y-%m-%d').date()
        
        logger.info(f"Collection period: {start_date} to {end_date}")
        
        # Load npm packages
        npm_packages = load_npm_packages(args.packages)
        
        if not npm_packages:
            logger.error("No npm packages found in input file")
            return
        
        # Collect download statistics
        download_stats = collect_npm_downloads(npm_packages, start_date, end_date)
        
        # Save results
        save_npm_stats(download_stats, args.output, start_date, end_date, npm_packages)
        
        logger.info("npm collection completed successfully")
        
    except Exception as e:
        logger.error(f"npm collection failed: {e}")
        raise

if __name__ == "__main__":
    main()
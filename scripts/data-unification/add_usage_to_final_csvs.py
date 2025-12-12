#!/usr/bin/env python3
"""
Add Usage Data to Final CSVs

Updates clservers_classified.csv.gz and cltools_classified.csv.gz with
monthly usage data for specific months from data_unified_filtered.json.

IMPORTANT: The --months flag is REQUIRED to prevent accidental full overwrites.

Usage:
    python add_usage_to_final_csvs.py --months 2025-10 2025-11
    python add_usage_to_final_csvs.py --months 2025-10 2025-11 --add-to-filtered  # Update filtered JSON first
"""

import argparse
import ast
import gzip
import json
import logging
import re
import datetime as dt
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd


# File paths
UNIFIED_FILTERED_FILE = "data/initial/data_unified_filtered.json"
CLSERVERS_FILE = "data/final/clservers_classified.csv.gz"
CLTOOLS_FILE = "data/final/cltools_classified.csv.gz"
NPM_DATA_FILE = "data/external-usage/usage_npm.json"
PYPI_DATA_FILE = "data/external-usage/usage_bigquery_webresults_pypi.json.gz"


def setup_logging() -> logging.Logger:
    """Setup logging for the script."""
    Path('logs').mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/add_usage_to_final_csvs.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def validate_months(months: List[str]) -> List[str]:
    """
    Validate month format (YYYY-MM).

    Args:
        months: List of month strings

    Returns:
        Validated and sorted list of months

    Raises:
        ValueError: If any month has invalid format
    """
    pattern = re.compile(r'^\d{4}-(0[1-9]|1[0-2])$')
    for month in months:
        if not pattern.match(month):
            raise ValueError(f"Invalid month format: {month}. Expected YYYY-MM.")
    return sorted(months)


def load_unified_usage_data(file_path: str, logger: logging.Logger) -> Dict[str, Dict]:
    """
    Load usage data from unified filtered JSON.

    Args:
        file_path: Path to data_unified_filtered.json
        logger: Logger instance

    Returns:
        Dict mapping server_id to usage data
    """
    logger.info(f"Loading unified data from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    usage_lookup = {}
    for server in data:
        server_id = server.get('id', '')
        if server_id:
            usage_lookup[server_id] = {
                'usage_pypi_downloads': server.get('usage_pypi_downloads', 0),
                'usage_npm_downloads': server.get('usage_npm_downloads', 0),
                'usage_total_downloads': server.get('usage_total_downloads', 0),
                'usage_monthly_breakdown': server.get('usage_monthly_breakdown', []),
                'usage_matched_packages': server.get('usage_matched_packages', {}),
                'usage_last_updated': server.get('usage_last_updated', '')
            }

    logger.info(f"Loaded usage data for {len(usage_lookup)} servers")
    return usage_lookup


def parse_monthly_breakdown(breakdown_str) -> List[Dict]:
    """
    Parse usage_monthly_breakdown from CSV string format.

    Args:
        breakdown_str: String representation of monthly breakdown list

    Returns:
        List of monthly breakdown dicts
    """
    if not breakdown_str or pd.isna(breakdown_str):
        return []

    if isinstance(breakdown_str, list):
        return breakdown_str

    breakdown_str = str(breakdown_str)
    if breakdown_str == '[]' or breakdown_str == '':
        return []

    try:
        return ast.literal_eval(breakdown_str)
    except (ValueError, SyntaxError):
        try:
            return json.loads(breakdown_str)
        except json.JSONDecodeError:
            return []


def merge_monthly_breakdowns(existing_breakdown: List[Dict],
                             new_breakdown: List[Dict],
                             target_months: List[str]) -> List[Dict]:
    """
    Merge new monthly data into existing breakdown, only updating target months.

    Args:
        existing_breakdown: Current monthly breakdown list
        new_breakdown: New monthly breakdown from unified data
        target_months: List of months to update

    Returns:
        Merged breakdown with target months updated
    """
    existing_by_month = {entry['month']: entry for entry in existing_breakdown}
    new_by_month = {entry['month']: entry for entry in new_breakdown}

    for month in target_months:
        if month in new_by_month:
            existing_by_month[month] = new_by_month[month]
        elif month not in existing_by_month:
            existing_by_month[month] = {'month': month, 'pypi': 0, 'npm': 0}

    return [existing_by_month[m] for m in sorted(existing_by_month.keys())]


def calculate_totals(breakdown: List[Dict]) -> tuple:
    """
    Calculate total downloads from monthly breakdown.

    Args:
        breakdown: List of monthly breakdown dicts

    Returns:
        Tuple of (pypi_total, npm_total, combined_total)
    """
    pypi_total = sum(entry.get('pypi', 0) for entry in breakdown)
    npm_total = sum(entry.get('npm', 0) for entry in breakdown)
    return pypi_total, npm_total, pypi_total + npm_total


def normalize_github_url(url: str) -> str:
    """Normalize GitHub URL to owner/repo format."""
    if not url:
        return ""
    github_pattern = r'github\.com[/:]([^/]+)/([^/\s#?]+)'
    match = re.search(github_pattern, str(url).lower())
    if match:
        owner, repo = match.groups()
        repo = re.sub(r'\.(git|issues|wiki).*$', '', repo)
        return f"{owner}/{repo}"
    return ""


def load_npm_data(file_path: str, logger: logging.Logger) -> Tuple[Dict, Dict]:
    """
    Load npm package data and build GitHub URL mapping.

    Returns:
        Tuple of (github_to_npm, npm_package_metadata)
    """
    logger.info(f"Loading npm data from {file_path}")
    with open(file_path) as f:
        npm_data = json.load(f)

    github_to_npm = {}
    package_metadata = {}

    for package_name, package_data in npm_data.get('packages', {}).items():
        metadata = package_data.get('metadata', {})

        # Extract GitHub URLs
        github_urls = set()
        for field in ['repository', 'homepage', 'bugs']:
            url = metadata.get(field, '')
            if url:
                normalized = normalize_github_url(url)
                if normalized:
                    github_urls.add(normalized)

        for github_url in github_urls:
            if github_url not in github_to_npm:
                github_to_npm[github_url] = []
            github_to_npm[github_url].append(package_name)

        package_metadata[('npm', package_name)] = {
            'monthly': package_data.get('monthly', {}),
            'total': package_data.get('total', 0)
        }

    logger.info(f"Loaded {len(npm_data.get('packages', {}))} npm packages, {len(github_to_npm)} with GitHub URLs")
    return github_to_npm, package_metadata


def load_pypi_data(file_path: str, logger: logging.Logger) -> Tuple[Dict, Dict]:
    """
    Load PyPI package data and build GitHub URL mapping.

    Returns:
        Tuple of (github_to_pypi, pypi_package_metadata)
    """
    logger.info(f"Loading PyPI data from {file_path}")

    pypi_by_name = {}
    open_func = gzip.open if file_path.endswith('.gz') else open
    mode = 'rt' if file_path.endswith('.gz') else 'r'

    with open_func(file_path, mode) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            package_name = record.get('name')
            month = record.get('month', '')
            country = record.get('country_code', '')
            downloads = int(record.get('monthly_downloads', 0))

            if not package_name:
                continue

            if package_name not in pypi_by_name:
                # Handle both field name formats: 'Project-URLs' and 'project_urls'
                project_urls = record.get('Project-URLs') or record.get('project_urls', [])
                description = record.get('description', '')
                pypi_by_name[package_name] = {
                    'monthly': {},
                    'Project-URLs': project_urls,
                    'description': description
                }

            if month:
                if month not in pypi_by_name[package_name]['monthly']:
                    pypi_by_name[package_name]['monthly'][month] = {}
                if country:
                    # Country-level breakdown
                    pypi_by_name[package_name]['monthly'][month][country] = \
                        pypi_by_name[package_name]['monthly'][month].get(country, 0) + downloads
                else:
                    # No country_code - use '__total__' key for aggregate downloads
                    pypi_by_name[package_name]['monthly'][month]['__total__'] = \
                        pypi_by_name[package_name]['monthly'][month].get('__total__', 0) + downloads

    github_to_pypi = {}
    package_metadata = {}

    for package_name, pkg_data in pypi_by_name.items():
        # Extract GitHub URLs from Project-URLs
        github_urls = set()
        for url in pkg_data.get('Project-URLs', []):
            normalized = normalize_github_url(url)
            if normalized:
                github_urls.add(normalized)

        # Also check description as fallback (same as data_unified_add_usage.py)
        description = pkg_data.get('description', '')
        if description:
            normalized = normalize_github_url(description)
            if normalized:
                github_urls.add(normalized)

        for github_url in github_urls:
            if github_url not in github_to_pypi:
                github_to_pypi[github_url] = []
            github_to_pypi[github_url].append(package_name)

        package_metadata[('pypi', package_name)] = {
            'monthly': pkg_data.get('monthly', {}),
            'total': sum(
                sum(country_data.values()) if isinstance(country_data, dict) else country_data
                for country_data in pkg_data.get('monthly', {}).values()
            )
        }

    logger.info(f"Loaded {len(pypi_by_name)} PyPI packages, {len(github_to_pypi)} with GitHub URLs")
    return github_to_pypi, package_metadata


def update_filtered_json(file_path: str, github_to_npm: Dict, github_to_pypi: Dict,
                         package_metadata: Dict, target_months: List[str],
                         logger: logging.Logger) -> int:
    """
    Update data_unified_filtered.json with usage data for specific months.

    Returns:
        Number of servers updated
    """
    logger.info(f"Loading filtered data from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"Updating {len(data)} servers for months {target_months}")
    updated_count = 0

    for server in data:
        server_id = server.get('id', '')
        repo_url = server.get('repository_url', '')

        if not repo_url:
            continue

        normalized_url = normalize_github_url(repo_url)
        if not normalized_url:
            continue

        npm_packages = github_to_npm.get(normalized_url, [])
        pypi_packages = github_to_pypi.get(normalized_url, [])

        if not npm_packages and not pypi_packages:
            continue

        # Get existing breakdown
        existing_breakdown = server.get('usage_monthly_breakdown', [])
        existing_by_month = {entry['month']: entry for entry in existing_breakdown}

        # Calculate new data for target months
        for month in target_months:
            pypi_month = 0
            npm_month = 0
            pypi_by_country = {}

            for pkg_name in pypi_packages:
                pkg_meta = package_metadata.get(('pypi', pkg_name), {})
                month_data = pkg_meta.get('monthly', {}).get(month, {})
                if isinstance(month_data, dict):
                    for country, downloads in month_data.items():
                        if country != '__total__':
                            pypi_by_country[country] = pypi_by_country.get(country, 0) + downloads
                        pypi_month += downloads

            for pkg_name in npm_packages:
                pkg_meta = package_metadata.get(('npm', pkg_name), {})
                npm_month += pkg_meta.get('monthly', {}).get(month, 0)

            entry = {'month': month, 'pypi': pypi_month, 'npm': npm_month}
            if pypi_by_country:
                entry['pypi_by_country'] = pypi_by_country
            existing_by_month[month] = entry

        # Rebuild breakdown sorted by month
        monthly_breakdown = [existing_by_month[m] for m in sorted(existing_by_month.keys())]

        # Calculate totals
        pypi_total = sum(entry.get('pypi', 0) for entry in monthly_breakdown)
        npm_total = sum(entry.get('npm', 0) for entry in monthly_breakdown)

        # Update server
        server['usage_pypi_downloads'] = pypi_total
        server['usage_npm_downloads'] = npm_total
        server['usage_total_downloads'] = pypi_total + npm_total
        server['usage_monthly_breakdown'] = monthly_breakdown
        server['usage_matched_packages'] = {'npm': npm_packages, 'pypi': pypi_packages}
        server['usage_last_updated'] = dt.date.today().isoformat()

        updated_count += 1

    # Save updated data
    logger.info(f"Saving updated filtered data to {file_path}")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Updated {updated_count} servers in filtered JSON")
    return updated_count


def update_csv_usage(csv_path: str, usage_lookup: Dict[str, Dict],
                     target_months: List[str], logger: logging.Logger) -> int:
    """
    Update usage columns in a CSV file.

    Args:
        csv_path: Path to the CSV file
        usage_lookup: Dict mapping server_id to usage data
        target_months: Months to update
        logger: Logger instance

    Returns:
        Number of rows updated
    """
    logger.info(f"Processing {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")

    if 'server_id' not in df.columns:
        raise ValueError(f"Missing 'server_id' column in {csv_path}")
    if 'usage_monthly_breakdown' not in df.columns:
        raise ValueError(f"Missing 'usage_monthly_breakdown' column in {csv_path}")

    updated_count = 0
    matched_count = 0

    for idx, row in df.iterrows():
        server_id = row['server_id']

        if server_id not in usage_lookup:
            continue

        matched_count += 1
        new_usage = usage_lookup[server_id]

        # Parse existing breakdown
        existing_breakdown = parse_monthly_breakdown(row['usage_monthly_breakdown'])
        new_breakdown = new_usage.get('usage_monthly_breakdown', [])

        # Merge new data into existing breakdown
        merged_breakdown = merge_monthly_breakdowns(
            existing_breakdown, new_breakdown, target_months
        )

        # Calculate totals from merged breakdown
        pypi_total, npm_total, total = calculate_totals(merged_breakdown)

        # Update columns
        df.at[idx, 'usage_monthly_breakdown'] = str(merged_breakdown)
        df.at[idx, 'usage_pypi_downloads'] = pypi_total
        df.at[idx, 'usage_npm_downloads'] = npm_total
        df.at[idx, 'usage_total_downloads'] = total

        if 'usage_last_updated' in df.columns:
            df.at[idx, 'usage_last_updated'] = new_usage.get('usage_last_updated', '')

        updated_count += 1

    # Save with gzip compression if original was compressed
    compression = 'gzip' if csv_path.endswith('.gz') else None
    df.to_csv(csv_path, index=False, compression=compression)
    logger.info(f"Matched {matched_count} servers, updated {updated_count} rows in {csv_path}")

    return updated_count


def main():
    parser = argparse.ArgumentParser(
        description="Update final CSVs with monthly usage data"
    )
    parser.add_argument("--months", nargs='+', type=str, required=True,
                       help="Months to update (YYYY-MM format). REQUIRED.")
    parser.add_argument("--add-to-filtered", action="store_true",
                       help="Update data_unified_filtered.json directly from npm/PyPI source data first")
    parser.add_argument("--unified", default=UNIFIED_FILTERED_FILE,
                       help="Path to data_unified_filtered.json")
    parser.add_argument("--npm-data", default=NPM_DATA_FILE,
                       help="Path to usage_npm.json")
    parser.add_argument("--pypi-data", default=PYPI_DATA_FILE,
                       help="Path to usage_bigquery_webresults_pypi.json.gz")
    parser.add_argument("--clservers", default=CLSERVERS_FILE,
                       help="Path to clservers_classified.csv.gz")
    parser.add_argument("--cltools", default=CLTOOLS_FILE,
                       help="Path to cltools_classified.csv.gz")

    args = parser.parse_args()
    logger = setup_logging()

    logger.info("=== Add Usage to Final CSVs ===")
    logger.info(f"Target months: {args.months}")

    try:
        # Validate months
        target_months = validate_months(args.months)
        logger.info(f"Validated months: {target_months}")

        # If --add-to-filtered, update the filtered JSON first from source data
        filtered_updated = 0
        if args.add_to_filtered:
            logger.info("=== Updating filtered JSON from source data ===")

            # Load npm and PyPI source data
            github_to_npm, npm_metadata = load_npm_data(args.npm_data, logger)
            github_to_pypi, pypi_metadata = load_pypi_data(args.pypi_data, logger)

            # Merge metadata
            package_metadata = {**npm_metadata, **pypi_metadata}

            # Update filtered JSON
            filtered_updated = update_filtered_json(
                args.unified, github_to_npm, github_to_pypi,
                package_metadata, target_months, logger
            )

        # Load unified usage data (now updated if --add-to-filtered was used)
        usage_lookup = load_unified_usage_data(args.unified, logger)

        # Update CLServers CSV
        clservers_updated = 0
        if Path(args.clservers).exists():
            clservers_updated = update_csv_usage(
                args.clservers, usage_lookup, target_months, logger
            )
        else:
            logger.warning(f"CLServers file not found: {args.clservers}")

        # Update CLTools CSV
        cltools_updated = 0
        if Path(args.cltools).exists():
            cltools_updated = update_csv_usage(
                args.cltools, usage_lookup, target_months, logger
            )
        else:
            logger.warning(f"CLTools file not found: {args.cltools}")

        logger.info("=== Summary ===")
        logger.info(f"Months updated: {target_months}")
        if args.add_to_filtered:
            logger.info(f"Filtered JSON servers updated: {filtered_updated}")
        logger.info(f"CLServers rows updated: {clservers_updated}")
        logger.info(f"CLTools rows updated: {cltools_updated}")
        logger.info("Usage update completed successfully")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()

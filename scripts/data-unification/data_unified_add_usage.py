#!/usr/bin/env python3
"""
MCP Package Usage Statistics Collection

Collects download statistics for MCP packages and integrates them into the unified dataset.
Uses multi-strategy matching approach:
1. Primary: GitHub URL matching (repository, homepage, bugs, Project-URLs)
2. Secondary: Author/email matching

Features:
- Direct GitHub URL matching from npm and PyPI metadata
- Author/maintainer email matching as fallback
- Monthly breakdown from Nov 2024 to present
- Modifies data_unified.json in place (run BEFORE filtering)
- Supports incremental updates for specific months via --months flag

Usage:
    python data_unified_add_usage.py                        # Full refresh
    python data_unified_add_usage.py --months 2025-10 2025-11  # Update specific months only
"""

import argparse
import datetime as dt
import json
from typing import Dict, List, Set, Tuple
import logging
import re


# File paths
DATASET_FILE = "data/initial/data_unified.json"

# Data source files
pypi_data_file = "data/external-usage/usage_bigquery_webresults_pypi.json.gz"
pypi_geo_file = "data/external-usage/usage_bigquery_webresults_pypi_geo.json.gz"
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


def validate_months(months: List[str]) -> List[str]:
    """
    Validate and sort month strings in YYYY-MM format.

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


def merge_monthly_breakdown(existing_breakdown: List[Dict],
                            new_months_data: Dict[str, Dict],
                            target_months: List[str]) -> List[Dict]:
    """
    Merge new monthly data into existing breakdown, only updating target months.

    Args:
        existing_breakdown: Current usage_monthly_breakdown list
        new_months_data: Dict with month -> {'month': m, 'pypi': X, 'npm': Y, ...}
        target_months: Months to update

    Returns:
        Updated breakdown with target months merged, others preserved
    """
    # Convert existing to dict by month
    existing_by_month = {entry['month']: entry for entry in existing_breakdown}

    # Update target months with new data
    for month in target_months:
        if month in new_months_data:
            existing_by_month[month] = new_months_data[month]
        elif month not in existing_by_month:
            # Add new month with zeros if not in new data
            existing_by_month[month] = {'month': month, 'pypi': 0, 'npm': 0}

    # Sort by month and return as list
    return [existing_by_month[m] for m in sorted(existing_by_month.keys())]


def normalize_github_url(url: str) -> str:
    """
    Normalize GitHub URL to owner/repo format.

    Examples:
        https://github.com/owner/repo -> owner/repo
        git://github.com/owner/repo.git -> owner/repo
        github.com/owner/repo/issues -> owner/repo
    """
    if not url:
        return ""

    # Match github.com URLs
    github_pattern = r'github\.com[/:]([^/]+)/([^/\s#?]+)'
    match = re.search(github_pattern, str(url).lower())
    if match:
        owner, repo = match.groups()
        # Clean repo name (remove .git, issues, wiki, etc.)
        repo = re.sub(r'\.(git|issues|wiki).*$', '', repo)
        return f"{owner}/{repo}"
    return ""


def normalize_package_name(package_name: str) -> str:
    """
    Normalize package name for matching (lowercase, handle scoped packages).

    Examples:
        @author/package -> package
        MyPackage -> mypackage
    """
    if not package_name:
        return ""

    # Remove scope from npm packages (e.g., @author/package → package)
    if package_name.startswith('@'):
        parts = package_name.split('/')
        if len(parts) > 1:
            package_name = parts[1]

    return package_name.lower()


def extract_github_urls_from_npm(package_data: Dict) -> Set[str]:
    """Extract all GitHub URLs from npm package metadata."""
    urls = set()
    metadata = package_data.get('metadata', {})

    # Check repository field
    repo = metadata.get('repository', '')
    if repo:
        normalized = normalize_github_url(repo)
        if normalized:
            urls.add(normalized)

    # Check homepage field
    homepage = metadata.get('homepage', '')
    if homepage:
        normalized = normalize_github_url(homepage)
        if normalized:
            urls.add(normalized)

    # Check bugs field
    bugs = metadata.get('bugs', '')
    if bugs:
        normalized = normalize_github_url(bugs)
        if normalized:
            urls.add(normalized)

    return urls


def extract_github_urls_from_pypi(package_record: Dict) -> Set[str]:
    """Extract all GitHub URLs from PyPI package metadata."""
    urls = set()

    # Check Project-URLs
    project_urls = package_record.get('Project-URLs', [])
    for url in project_urls:
        normalized = normalize_github_url(url)
        if normalized:
            urls.add(normalized)

    # Check description as fallback
    description = package_record.get('description', '')
    if description:
        normalized = normalize_github_url(description)
        if normalized:
            urls.add(normalized)

    return urls


def build_github_to_packages_mapping(npm_data: Dict, pypi_by_name: Dict[str, Dict], logger) -> Tuple[Dict, Dict, Dict]:
    """
    Build mappings from GitHub URLs to package names.

    Args:
        npm_data: npm package data
        pypi_by_name: PyPI data already fully aggregated (package -> monthly totals)
        logger: Logger instance

    Returns:
        - github_to_npm: {github_url: [package_names]}
        - github_to_pypi: {github_url: [package_names]}
        - package_metadata: {(platform, package_name): metadata}
    """
    github_to_npm = {}
    github_to_pypi = {}
    package_metadata = {}

    # Process npm packages
    npm_packages = npm_data.get('packages', {})
    logger.info(f"Processing {len(npm_packages)} npm packages for GitHub URL matching")

    npm_url_matches = 0
    for package_name, package_data in npm_packages.items():
        github_urls = extract_github_urls_from_npm(package_data)

        if github_urls:
            npm_url_matches += 1
            for github_url in github_urls:
                if github_url not in github_to_npm:
                    github_to_npm[github_url] = []
                github_to_npm[github_url].append(package_name)

        # Store metadata including author info
        metadata = package_data.get('metadata', {})
        package_metadata[('npm', package_name)] = {
            'author': metadata.get('author', {}),
            'maintainers': metadata.get('maintainers', []),
            'monthly': package_data.get('monthly', {}),
            'total': package_data.get('total', 0)
        }

    logger.info(f"npm: {npm_url_matches} packages with GitHub URLs ({npm_url_matches/len(npm_packages)*100:.1f}%)")

    # Process PyPI packages (already fully aggregated with monthly totals)

    logger.info(f"Processing {len(pypi_by_name)} unique PyPI packages for GitHub URL matching")

    pypi_url_matches = 0
    for package_name, pkg_data in pypi_by_name.items():
        # Extract GitHub URLs from Project-URLs and description
        github_urls = extract_github_urls_from_pypi(pkg_data)

        if github_urls:
            pypi_url_matches += 1
            for github_url in github_urls:
                if github_url not in github_to_pypi:
                    github_to_pypi[github_url] = []
                github_to_pypi[github_url].append(package_name)

        # Store metadata (already aggregated with monthly totals or country breakdown)
        monthly_downloads = pkg_data.get('monthly', {})

        # Calculate total downloads (handle both aggregated and country-level formats)
        total_downloads = 0
        for month_data in monthly_downloads.values():
            if isinstance(month_data, dict):
                # Country-level breakdown: {month: {country: downloads}}
                total_downloads += sum(month_data.values())
            else:
                # Aggregated monthly total: {month: downloads}
                total_downloads += month_data

        package_metadata[('pypi', package_name)] = {
            'author_email': pkg_data.get('author_email', ''),
            'author': pkg_data.get('author', ''),
            'monthly': monthly_downloads,
            'total': total_downloads
        }

    logger.info(f"PyPI: {pypi_url_matches} packages with GitHub URLs ({pypi_url_matches/len(pypi_by_name)*100:.1f}%)")
    logger.info(f"Total unique GitHub URLs: npm={len(github_to_npm)}, PyPI={len(github_to_pypi)}")

    return github_to_npm, github_to_pypi, package_metadata


def build_author_email_indices(package_metadata: Dict, logger) -> Tuple[Dict, Dict]:
    """
    Build indices for author/email matching.

    Returns:
        - email_to_packages: {email: {'npm': [packages], 'pypi': [packages]}}
        - author_to_packages: {author_name: {'npm': [packages], 'pypi': [packages]}}
    """
    email_to_packages = {}
    author_to_packages = {}

    npm_with_email = 0
    npm_with_author = 0
    pypi_with_email = 0
    pypi_with_author = 0

    for (platform, package_name), metadata in package_metadata.items():
        if platform == 'npm':
            # npm author email
            author = metadata.get('author', {})
            if isinstance(author, dict):
                email = author.get('email', '').strip().lower()
                author_name = author.get('name', '').strip().lower()
            else:
                email = ''
                author_name = ''

            if email and '@' in email:
                npm_with_email += 1
                if email not in email_to_packages:
                    email_to_packages[email] = {'npm': [], 'pypi': []}
                email_to_packages[email]['npm'].append(package_name)

            if author_name:
                npm_with_author += 1
                if author_name not in author_to_packages:
                    author_to_packages[author_name] = {'npm': [], 'pypi': []}
                author_to_packages[author_name]['npm'].append(package_name)

        elif platform == 'pypi':
            # PyPI author_email
            author_email = metadata.get('author_email', '').strip().lower()
            author_name = metadata.get('author', '').strip().lower()

            # Extract email from "Name <email@example.com>" format
            email_match = re.search(r'<([^>]+)>', author_email)
            if email_match:
                email = email_match.group(1).strip()
            elif '@' in author_email:
                email = author_email
            else:
                email = ''

            if email and '@' in email:
                pypi_with_email += 1
                if email not in email_to_packages:
                    email_to_packages[email] = {'npm': [], 'pypi': []}
                email_to_packages[email]['pypi'].append(package_name)

            if author_name:
                pypi_with_author += 1
                if author_name not in author_to_packages:
                    author_to_packages[author_name] = {'npm': [], 'pypi': []}
                author_to_packages[author_name]['pypi'].append(package_name)

    logger.info("Author/Email Index Stats:")
    logger.info(f"  npm: {npm_with_email} packages with email, {npm_with_author} with author name")
    logger.info(f"  PyPI: {pypi_with_email} packages with email, {pypi_with_author} with author name")
    logger.info(f"  Unique emails: {len(email_to_packages)}")
    logger.info(f"  Unique author names: {len(author_to_packages)}")

    return email_to_packages, author_to_packages


def extract_repo_author_info(repo: Dict) -> Tuple[str, str]:
    """
    Extract author name and email from repository owner information.

    Returns:
        - owner_login: GitHub username/org name
        - owner_email: Email if available (usually not in API)
    """
    owner = repo.get('owner', {})
    owner_login = owner.get('login', '').strip().lower()

    # GitHub API doesn't typically provide owner email
    # But we can try from other fields if available
    owner_email = ''

    return owner_login, owner_email


def match_packages_to_repos(dataset: List[Dict], github_to_npm: Dict, github_to_pypi: Dict,
                            package_metadata: Dict, email_to_packages: Dict,
                            author_to_packages: Dict, logger) -> Tuple[Dict, Dict]:
    """
    Match packages to repositories using simplified case-insensitive matching:
    1. Primary: repository_url matching (normalized github.com/owner/repo format)
    2. Secondary: owner+name combination (for servers without repository_url)

    Returns:
        - repo_to_packages: {repo_id: {'npm': [package_names], 'pypi': [package_names]}}
        - match_stats: Statistics about matching
    """
    repo_to_packages = {}

    stats = {
        'repos_with_npm': 0,
        'repos_with_pypi': 0,
        'repos_with_both': 0,
        'repos_with_any': 0,
        'total_npm_packages': 0,
        'total_pypi_packages': 0,
        'url_match_npm': 0,
        'url_match_pypi': 0,
        'owner_name_match_npm': 0,
        'owner_name_match_pypi': 0
    }

    # Track matched servers to ensure 1 package → 1 server constraint
    matched_npm_packages = set()
    matched_pypi_packages = set()

    for repo in dataset:
        repo_id = repo.get('id', '')
        if not repo_id:
            continue

        npm_packages = []
        pypi_packages = []
        match_method = 'none'

        # Strategy 1: repository_url matching (primary)
        repo_url = repo.get('repository_url', '')
        if repo_url:
            # Normalize repository_url to owner/repo format (case-insensitive)
            normalized_url = normalize_github_url(repo_url)

            if normalized_url:
                npm_packages = github_to_npm.get(normalized_url, [])
                pypi_packages = github_to_pypi.get(normalized_url, [])

                # Filter out already matched packages (enforce 1 package → 1 server)
                npm_packages = [p for p in npm_packages if p not in matched_npm_packages]
                pypi_packages = [p for p in pypi_packages if p not in matched_pypi_packages]

                if npm_packages or pypi_packages:
                    match_method = 'repository_url'
                    if npm_packages:
                        stats['url_match_npm'] += 1
                        matched_npm_packages.update(npm_packages)
                    if pypi_packages:
                        stats['url_match_pypi'] += 1
                        matched_pypi_packages.update(pypi_packages)

        # Strategy 2: owner+name matching (secondary, only for servers without repository_url)
        if not npm_packages and not pypi_packages and not repo_url:
            owner = repo.get('owner', '').strip().lower()
            name = repo.get('name', '').strip().lower()

            if owner and name:
                # Try matching by owner:name combination
                owner_packages = author_to_packages.get(owner, {})
                npm_from_owner = owner_packages.get('npm', [])
                pypi_from_owner = owner_packages.get('pypi', [])

                # Filter out already matched packages
                npm_from_owner = [p for p in npm_from_owner if p not in matched_npm_packages]
                pypi_from_owner = [p for p in pypi_from_owner if p not in matched_pypi_packages]

                # Also check if package name matches server name
                npm_packages = [p for p in npm_from_owner if normalize_package_name(p) == name]
                pypi_packages = [p for p in pypi_from_owner if normalize_package_name(p) == name]

                if npm_packages:
                    match_method = 'owner_name'
                    stats['owner_name_match_npm'] += 1
                    matched_npm_packages.update(npm_packages)

                if pypi_packages:
                    if match_method == 'none':
                        match_method = 'owner_name'
                    stats['owner_name_match_pypi'] += 1
                    matched_pypi_packages.update(pypi_packages)

        # Store matches if any found
        if npm_packages or pypi_packages:
            repo_to_packages[repo_id] = {
                'npm': list(set(npm_packages)),  # Remove duplicates
                'pypi': list(set(pypi_packages)),
                'match_method': match_method
            }

            if npm_packages:
                stats['repos_with_npm'] += 1
                stats['total_npm_packages'] += len(set(npm_packages))

            if pypi_packages:
                stats['repos_with_pypi'] += 1
                stats['total_pypi_packages'] += len(set(pypi_packages))

            if npm_packages and pypi_packages:
                stats['repos_with_both'] += 1

            stats['repos_with_any'] += 1

    logger.info("Matching Results:")
    logger.info(f"  Total repos matched: {stats['repos_with_any']}")
    logger.info(f"  Repos with npm packages: {stats['repos_with_npm']}")
    logger.info(f"  Repos with PyPI packages: {stats['repos_with_pypi']}")
    logger.info(f"  Repos with both: {stats['repos_with_both']}")
    logger.info("")
    logger.info(f"  Total npm packages matched: {stats['total_npm_packages']}")
    logger.info(f"  Total PyPI packages matched: {stats['total_pypi_packages']}")
    logger.info("")
    logger.info("  Match breakdown:")
    logger.info(f"    - repository_url (npm): {stats['url_match_npm']}")
    logger.info(f"    - repository_url (PyPI): {stats['url_match_pypi']}")
    logger.info(f"    - owner+name (npm): {stats['owner_name_match_npm']}")
    logger.info(f"    - owner+name (PyPI): {stats['owner_name_match_pypi']}")

    return repo_to_packages, stats


def integrate_download_stats(dataset: List[Dict], repo_to_packages: Dict, package_metadata: Dict,
                           start_date: dt.date, end_date: dt.date, logger,
                           target_months: List[str] = None,
                           pypi_geo_data: Dict[str, Dict[str, Dict[str, int]]] = None) -> List[Dict]:
    """
    Integrate download statistics into the unified dataset.

    Args:
        dataset: List of repository records
        repo_to_packages: Mapping of repo_id to matched packages
        package_metadata: Package download metadata
        start_date: Start of date range
        end_date: End of date range
        logger: Logger instance
        target_months: Optional list of months to update (incremental mode).
                      If None, performs full refresh.
        pypi_geo_data: Optional geo data from separate file.
                      Dict mapping package_name -> month -> {country: downloads}.
                      If provided, geo data is added to breakdown entries where available,
                      and totals are validated against main PyPI data.
    """
    if pypi_geo_data is None:
        pypi_geo_data = {}
    logger.info(f"Integrating download stats into dataset with {len(dataset)} repositories")

    # Generate complete month list (for full mode or reference)
    months = []
    current = start_date.replace(day=1)
    end_month = end_date.replace(day=1)

    while current <= end_month:
        months.append(current.strftime('%Y-%m'))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    if target_months:
        logger.info(f"Incremental mode: updating only {target_months}")

    # Integrate stats into dataset
    updated_count = 0

    for repo in dataset:
        repo_id = repo.get('id', '')

        if repo_id in repo_to_packages:
            # This repository has matched packages
            matched_packages = repo_to_packages[repo_id]
            npm_packages = matched_packages.get('npm', [])
            pypi_packages = matched_packages.get('pypi', [])

            if target_months:
                # INCREMENTAL MODE: merge with existing breakdown
                existing_breakdown = repo.get('usage_monthly_breakdown', [])

                # Calculate new data for target months only
                new_months_data = {}
                for month in target_months:
                    pypi_month = 0
                    npm_month = 0

                    # Sum PyPI downloads from main aggregated data
                    for pkg_name in pypi_packages:
                        pkg_meta = package_metadata.get(('pypi', pkg_name), {})
                        month_data = pkg_meta.get('monthly', {}).get(month, 0)
                        if isinstance(month_data, int):
                            pypi_month += month_data
                        elif isinstance(month_data, dict):
                            pypi_month += sum(month_data.values())

                    for pkg_name in npm_packages:
                        pkg_meta = package_metadata.get(('npm', pkg_name), {})
                        npm_month += pkg_meta.get('monthly', {}).get(month, 0)

                    entry = {'month': month, 'pypi': pypi_month, 'npm': npm_month}

                    # Add geo data from separate geo file if available
                    if pypi_geo_data:
                        pypi_by_country = {}
                        geo_total = 0
                        for pkg_name in pypi_packages:
                            pkg_geo = pypi_geo_data.get(pkg_name, {}).get(month, {})
                            for country, downloads in pkg_geo.items():
                                pypi_by_country[country] = pypi_by_country.get(country, 0) + downloads
                                geo_total += downloads

                        if pypi_by_country:
                            entry['pypi_by_country'] = pypi_by_country
                            # Validate totals match (warn if mismatch > 1%)
                            if pypi_month > 0 and abs(geo_total - pypi_month) / pypi_month > 0.01:
                                logger.warning(f"Geo total mismatch for {repo_id} month {month}: "
                                             f"main={pypi_month}, geo={geo_total}")

                    new_months_data[month] = entry

                # Merge new data into existing breakdown
                monthly_breakdown = merge_monthly_breakdown(
                    existing_breakdown, new_months_data, target_months
                )

            else:
                # FULL MODE: original behavior
                monthly_breakdown = []
                for month in months:
                    pypi_month = 0
                    npm_month = 0

                    # Sum PyPI downloads for this month (from main aggregated data)
                    for pkg_name in pypi_packages:
                        pkg_meta = package_metadata.get(('pypi', pkg_name), {})
                        month_data = pkg_meta.get('monthly', {}).get(month, 0)
                        if isinstance(month_data, int):
                            pypi_month += month_data
                        elif isinstance(month_data, dict):
                            # Shouldn't happen with keep_countries=False, but handle it
                            pypi_month += sum(month_data.values())

                    # Sum npm downloads for this month (npm has no country breakdown)
                    for pkg_name in npm_packages:
                        pkg_meta = package_metadata.get(('npm', pkg_name), {})
                        npm_month += pkg_meta.get('monthly', {}).get(month, 0)

                    # Build monthly breakdown entry
                    breakdown_entry = {
                        'month': month,
                        'pypi': pypi_month,
                        'npm': npm_month
                    }

                    # Add geo data from separate geo file if available
                    if pypi_geo_data:
                        pypi_by_country = {}
                        geo_total = 0
                        for pkg_name in pypi_packages:
                            pkg_geo = pypi_geo_data.get(pkg_name, {}).get(month, {})
                            for country, downloads in pkg_geo.items():
                                pypi_by_country[country] = pypi_by_country.get(country, 0) + downloads
                                geo_total += downloads

                        if pypi_by_country:
                            breakdown_entry['pypi_by_country'] = pypi_by_country
                            # Validate totals match (warn if mismatch > 1%)
                            if pypi_month > 0 and abs(geo_total - pypi_month) / pypi_month > 0.01:
                                logger.warning(f"Geo total mismatch for {repo_id} month {month}: "
                                             f"main={pypi_month}, geo={geo_total}")

                    monthly_breakdown.append(breakdown_entry)

            # Calculate totals from breakdown (source of truth)
            pypi_total = sum(entry.get('pypi', 0) for entry in monthly_breakdown)
            npm_total = sum(entry.get('npm', 0) for entry in monthly_breakdown)

            # Add usage fields to repository
            repo['usage_pypi_downloads'] = pypi_total
            repo['usage_npm_downloads'] = npm_total
            repo['usage_total_downloads'] = pypi_total + npm_total
            repo['usage_monthly_breakdown'] = monthly_breakdown
            repo['usage_matched_packages'] = {
                'npm': npm_packages,
                'pypi': pypi_packages
            }
            repo['usage_match_method'] = matched_packages.get('match_method', 'unknown')
            repo['usage_last_updated'] = dt.date.today().isoformat()

            updated_count += 1
        else:
            # Repository has no matched packages - handle differently for incremental vs full
            if target_months:
                # Preserve existing data, just update target months to 0
                existing_breakdown = repo.get('usage_monthly_breakdown', [])
                if existing_breakdown:
                    # Merge zeros for target months
                    new_months_data = {m: {'month': m, 'pypi': 0, 'npm': 0} for m in target_months}
                    monthly_breakdown = merge_monthly_breakdown(
                        existing_breakdown, new_months_data, target_months
                    )
                    repo['usage_monthly_breakdown'] = monthly_breakdown
                    # Recalculate totals
                    repo['usage_pypi_downloads'] = sum(entry.get('pypi', 0) for entry in monthly_breakdown)
                    repo['usage_npm_downloads'] = sum(entry.get('npm', 0) for entry in monthly_breakdown)
                    repo['usage_total_downloads'] = repo['usage_pypi_downloads'] + repo['usage_npm_downloads']
                    repo['usage_last_updated'] = dt.date.today().isoformat()
                # If no existing breakdown, skip this repo in incremental mode
            else:
                # Full mode: set all to zeros
                repo['usage_pypi_downloads'] = 0
                repo['usage_npm_downloads'] = 0
                repo['usage_total_downloads'] = 0
                repo['usage_monthly_breakdown'] = [{'month': month, 'pypi': 0, 'npm': 0} for month in months]
                repo['usage_matched_packages'] = {'pypi': [], 'npm': []}
                repo['usage_match_method'] = 'none'
                repo['usage_last_updated'] = dt.date.today().isoformat()

    logger.info(f"Updated {updated_count} repositories with download statistics")

    return dataset

def load_pypi_data_as_jsonl(file_path: str, logger=None, keep_countries: bool = True) -> Dict[str, Dict]:
    """
    Load and aggregate PyPI data from JSONL file (supports gzip compression).

    Args:
        file_path: Path to PyPI data file (supports .gz)
        logger: Logger instance
        keep_countries: If True, preserve country-level breakdown; if False, aggregate to monthly totals

    Returns:
        Dict mapping package_name -> {
            'monthly': {month: downloads} OR {month: {country: downloads}} (if keep_countries=True),
            'author': str,
            'author_email': str,
            'Project-URLs': list,
            'summary': str,
            'description': str,
            'upload_time': str
        }
    """
    import gzip

    if logger:
        mode_str = "with country breakdown" if keep_countries else "aggregated by month"
        logger.info(f"Loading PyPI data {mode_str}...")

    # Aggregate by package name AND month while loading
    pypi_by_name = {}

    # Auto-detect gzip compression by file extension
    open_func = gzip.open if file_path.endswith('.gz') else open
    mode = 'rt' if file_path.endswith('.gz') else 'r'

    line_count = 0
    with open_func(file_path, mode) as f:
        for line in f:
            line_count += 1
            if line_count % 100000 == 0 and logger:
                logger.info(f"  Processed {line_count:,} records -> {len(pypi_by_name):,} unique packages")

            if not line.strip():
                continue

            record = json.loads(line)
            package_name = record.get('name')
            month = record.get('month', '')
            country = record.get('country_code', '')
            downloads = int(record.get('monthly_downloads', 0))

            if not package_name:
                continue

            # Initialize package entry if first time seeing it
            if package_name not in pypi_by_name:
                pypi_by_name[package_name] = {
                    'monthly': {},
                    'author': record.get('author', ''),
                    'author_email': record.get('author_email', ''),
                    'Project-URLs': record.get('Project-URLs', []),
                    'summary': record.get('summary', ''),
                    'description': record.get('description', ''),
                    'upload_time': record.get('upload_time', '')
                }

            pkg_data = pypi_by_name[package_name]

            # Aggregate downloads by month and optionally by country
            if month:
                if keep_countries:
                    # Keep country-level breakdown: monthly[month][country] = downloads
                    if month not in pkg_data['monthly']:
                        pkg_data['monthly'][month] = {}
                    if country:
                        pkg_data['monthly'][month][country] = pkg_data['monthly'][month].get(country, 0) + downloads
                else:
                    # Aggregate to monthly totals: monthly[month] = total_downloads
                    pkg_data['monthly'][month] = pkg_data['monthly'].get(month, 0) + downloads

            # Keep most recent upload_time
            if record.get('upload_time', '') > pkg_data['upload_time']:
                pkg_data['upload_time'] = record.get('upload_time', '')

    if logger:
        logger.info(f"  Loaded {line_count:,} records aggregated to {len(pypi_by_name):,} unique packages")

    return pypi_by_name


def load_pypi_geo_data(file_path: str, logger=None) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Load PyPI geo data from separate geo file.

    The geo file has country-level breakdown that may cover different months
    than the main PyPI data file. This function loads it as a separate lookup.

    Args:
        file_path: Path to geo data file (supports .gz)
        logger: Logger instance

    Returns:
        Dict mapping package_name -> month -> {country_code: downloads}
    """
    import gzip
    import os

    if not os.path.exists(file_path):
        if logger:
            logger.warning(f"Geo data file not found: {file_path}, skipping geo data")
        return {}

    if logger:
        logger.info(f"Loading PyPI geo data from {file_path}...")

    geo_data = {}  # {package_name: {month: {country: downloads}}}

    open_func = gzip.open if file_path.endswith('.gz') else open
    mode = 'rt' if file_path.endswith('.gz') else 'r'

    line_count = 0
    geo_months = set()
    with open_func(file_path, mode) as f:
        for line in f:
            line_count += 1
            if not line.strip():
                continue

            record = json.loads(line)
            package_name = record.get('name')
            month = record.get('month', '')
            country = record.get('country_code', '')
            downloads = int(record.get('monthly_downloads', 0))

            if not package_name or not month or not country:
                continue

            geo_months.add(month)

            if package_name not in geo_data:
                geo_data[package_name] = {}
            if month not in geo_data[package_name]:
                geo_data[package_name][month] = {}

            geo_data[package_name][month][country] = geo_data[package_name][month].get(country, 0) + downloads

    if logger:
        logger.info(f"  Loaded {line_count:,} geo records for {len(geo_data):,} packages")
        logger.info(f"  Geo data covers months: {sorted(geo_months)}")

    return geo_data


def main():
    parser = argparse.ArgumentParser(description="Collect and integrate MCP package usage statistics")
    parser.add_argument("--dataset", default=DATASET_FILE,
                       help="Dataset file to integrate stats into (modified in place)")
    parser.add_argument("--output", default=None,
                       help="Output file (default: same as dataset)")
    parser.add_argument("--months", nargs='+', type=str,
                       help="Specific months to update (YYYY-MM format, e.g., --months 2025-10 2025-11). "
                            "If provided, only these months are updated and merged with existing data.")
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()

    # Validate months if provided
    target_months = None
    if args.months:
        target_months = validate_months(args.months)
        logger.info(f"Incremental mode: updating only {target_months}")

    logger.info("=== MCP Package Usage Statistics Collection ===")
    logger.info(f"Date range: {START_DATE} to {END_DATE}")
    logger.info("Using multi-strategy matching: 1) GitHub URL, 2) Author/Email")

    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}")
    with open(args.dataset) as f:
        dataset = json.load(f)
    logger.info(f"Loaded {len(dataset)} repositories")

    # Load npm data
    logger.info(f"Loading npm data from {npm_data_file}")
    with open(npm_data_file) as f:
        npm_data = json.load(f)

    # Load PyPI data (JSONL format) - returns aggregated data by package name
    logger.info(f"Loading PyPI data from {pypi_data_file}")
    pypi_by_name = load_pypi_data_as_jsonl(pypi_data_file, logger, keep_countries=False)

    # Load PyPI geo data separately (may have different month coverage)
    pypi_geo_data = load_pypi_geo_data(pypi_geo_file, logger)

    # Build GitHub URL to package mappings
    logger.info("Building GitHub URL to package mappings...")
    github_to_npm, github_to_pypi, package_metadata = build_github_to_packages_mapping(
        npm_data, pypi_by_name, logger
    )

    # Build author/email indices for fallback matching
    logger.info("Building author/email indices for fallback matching...")
    email_to_packages, author_to_packages = build_author_email_indices(
        package_metadata, logger
    )

    # Match packages to repositories
    logger.info("Matching packages to repositories...")
    repo_to_packages, match_stats = match_packages_to_repos(
        dataset, github_to_npm, github_to_pypi, package_metadata,
        email_to_packages, author_to_packages, logger
    )

    # Integrate statistics into dataset
    logger.info("Integrating download statistics into dataset...")
    updated_dataset = integrate_download_stats(
        dataset, repo_to_packages, package_metadata, START_DATE, END_DATE, logger,
        target_months=target_months, pypi_geo_data=pypi_geo_data
    )

    # Save updated dataset
    output_file = args.output or args.dataset
    logger.info(f"Saving updated dataset to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(updated_dataset, f, indent=2)

    # Summary statistics
    total_pypi_downloads = sum(
        pkg_meta.get('total', 0)
        for (platform, _), pkg_meta in package_metadata.items()
        if platform == 'pypi'
    )
    total_npm_downloads = sum(
        pkg_meta.get('total', 0)
        for (platform, _), pkg_meta in package_metadata.items()
        if platform == 'npm'
    )
    repos_with_stats = sum(1 for repo in updated_dataset if repo.get('usage_total_downloads', 0) > 0)

    logger.info("=== SUMMARY ===")
    logger.info(f"Total PyPI downloads in matched packages: {total_pypi_downloads:,}")
    logger.info(f"Total npm downloads in matched packages: {total_npm_downloads:,}")
    logger.info(f"Repositories with download stats: {repos_with_stats}")
    logger.info(f"Coverage: {repos_with_stats/len(updated_dataset)*100:.1f}% of {len(updated_dataset)} repositories")
    logger.info("Match breakdown:")
    logger.info(f"  - GitHub URL matches (npm): {match_stats['url_match_npm']}")
    logger.info(f"  - GitHub URL matches (PyPI): {match_stats['url_match_pypi']}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Analyze GitHub URL coverage weighted by download volume.
Shows what percentage of total downloads come from packages with GitHub URLs.
"""

import json
import logging
from pathlib import Path
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/usage_download_weighted_coverage.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def extract_github_url(url_string):
    """Extract GitHub repo URL from various URL formats."""
    if not url_string:
        return None

    # Match github.com URLs
    github_pattern = r'github\.com[/:]([^/]+)/([^/\s#?]+)'
    match = re.search(github_pattern, str(url_string))
    if match:
        owner, repo = match.groups()
        # Clean repo name (remove .git, issues, etc.)
        repo = re.sub(r'\.(git|issues|wiki).*$', '', repo)
        return f"{owner}/{repo}"
    return None


def analyze_npm_download_coverage(npm_file):
    """Analyze npm downloads by GitHub URL availability."""
    logger.info("Analyzing npm download-weighted coverage...")

    with open(npm_file, 'r') as f:
        data = json.load(f)

    packages = data.get('packages', {})
    total_packages = len(packages)

    total_downloads = 0
    downloads_with_github = 0
    downloads_without_github = 0

    packages_with_github = 0
    packages_without_github = 0

    top_packages_without_github = []

    for package_name, package_data in packages.items():
        metadata = package_data.get('metadata', {})
        package_downloads = package_data.get('total', 0)
        total_downloads += package_downloads

        # Check for GitHub URLs
        repository = metadata.get('repository')
        homepage = metadata.get('homepage')
        bugs = metadata.get('bugs')

        has_github = False
        github_url = None

        repo_url = extract_github_url(repository)
        if repo_url:
            has_github = True
            github_url = repo_url

        if not has_github:
            home_url = extract_github_url(homepage)
            if home_url:
                has_github = True
                github_url = home_url

        if not has_github:
            bugs_url = extract_github_url(bugs)
            if bugs_url:
                has_github = True
                github_url = bugs_url

        if has_github:
            downloads_with_github += package_downloads
            packages_with_github += 1
        else:
            downloads_without_github += package_downloads
            packages_without_github += 1
            top_packages_without_github.append({
                'name': package_name,
                'downloads': package_downloads,
                'repository': repository,
                'homepage': homepage
            })

    # Sort packages without GitHub by downloads
    top_packages_without_github.sort(key=lambda x: x['downloads'], reverse=True)

    stats = {
        'total_packages': total_packages,
        'packages_with_github': packages_with_github,
        'packages_without_github': packages_without_github,
        'total_downloads': total_downloads,
        'downloads_with_github': downloads_with_github,
        'downloads_without_github': downloads_without_github,
        'percent_packages_with_github': (packages_with_github / total_packages * 100) if total_packages > 0 else 0,
        'percent_downloads_with_github': (downloads_with_github / total_downloads * 100) if total_downloads > 0 else 0,
        'top_10_packages_without_github': top_packages_without_github[:10]
    }

    return stats


def analyze_pypi_download_coverage(pypi_file):
    """Analyze PyPI downloads by GitHub URL availability."""
    logger.info("Analyzing PyPI download-weighted coverage...")

    # Read JSONL file and aggregate by package
    packages_data = {}
    with open(pypi_file, 'r') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                package_name = record.get('name')
                monthly_downloads = int(record.get('monthly_downloads', 0))

                # Aggregate monthly downloads per package
                if package_name not in packages_data:
                    packages_data[package_name] = {
                        'total_downloads': 0,
                        'metadata': record
                    }
                packages_data[package_name]['total_downloads'] += monthly_downloads

    total_packages = len(packages_data)

    total_downloads = 0
    downloads_with_github = 0
    downloads_without_github = 0

    packages_with_github = 0
    packages_without_github = 0

    top_packages_without_github = []

    for package_name, package_info in packages_data.items():
        package_downloads = package_info['total_downloads']
        record = package_info['metadata']

        total_downloads += package_downloads

        # Check for GitHub URLs
        project_urls = record.get('Project-URLs', [])
        description = record.get('description', '')

        has_github = False
        github_url = None

        # Check Project-URLs
        for url in project_urls:
            github_url = extract_github_url(url)
            if github_url:
                has_github = True
                break

        # Check description as fallback
        if not has_github:
            github_url = extract_github_url(description)
            if github_url:
                has_github = True

        if has_github:
            downloads_with_github += package_downloads
            packages_with_github += 1
        else:
            downloads_without_github += package_downloads
            packages_without_github += 1
            top_packages_without_github.append({
                'name': package_name,
                'downloads': package_downloads,
                'project_urls': project_urls,
                'summary': record.get('summary', '')
            })

    # Sort packages without GitHub by downloads
    top_packages_without_github.sort(key=lambda x: x['downloads'], reverse=True)

    stats = {
        'total_packages': total_packages,
        'packages_with_github': packages_with_github,
        'packages_without_github': packages_without_github,
        'total_downloads': total_downloads,
        'downloads_with_github': downloads_with_github,
        'downloads_without_github': downloads_without_github,
        'percent_packages_with_github': (packages_with_github / total_packages * 100) if total_packages > 0 else 0,
        'percent_downloads_with_github': (downloads_with_github / total_downloads * 100) if total_downloads > 0 else 0,
        'top_10_packages_without_github': top_packages_without_github[:10]
    }

    return stats


def main():
    """Main analysis function."""
    logger.info("Starting download-weighted coverage analysis")

    # File paths
    npm_file = Path('data/external-usage/usage_npm.json')
    pypi_file = Path('data/external-usage/usage_bigquery_webresults_pypi.json')
    output_file = Path('data/external-usage/usage_download_weighted_coverage.json')

    # Analyze npm
    npm_stats = analyze_npm_download_coverage(npm_file)

    logger.info(f"\n=== NPM Download-Weighted Coverage ===")
    logger.info(f"Total packages: {npm_stats['total_packages']:,}")
    logger.info(f"Packages with GitHub: {npm_stats['packages_with_github']:,} ({npm_stats['percent_packages_with_github']:.1f}%)")
    logger.info(f"Packages without GitHub: {npm_stats['packages_without_github']:,}")
    logger.info(f"\nTotal downloads: {npm_stats['total_downloads']:,}")
    logger.info(f"Downloads with GitHub: {npm_stats['downloads_with_github']:,} ({npm_stats['percent_downloads_with_github']:.1f}%)")
    logger.info(f"Downloads without GitHub: {npm_stats['downloads_without_github']:,}")
    logger.info(f"\nTop 10 packages without GitHub URLs:")
    for i, pkg in enumerate(npm_stats['top_10_packages_without_github'], 1):
        logger.info(f"  {i}. {pkg['name']}: {pkg['downloads']:,} downloads")

    # Analyze PyPI
    pypi_stats = analyze_pypi_download_coverage(pypi_file)

    logger.info(f"\n=== PyPI Download-Weighted Coverage ===")
    logger.info(f"Total packages: {pypi_stats['total_packages']:,}")
    logger.info(f"Packages with GitHub: {pypi_stats['packages_with_github']:,} ({pypi_stats['percent_packages_with_github']:.1f}%)")
    logger.info(f"Packages without GitHub: {pypi_stats['packages_without_github']:,}")
    logger.info(f"\nTotal downloads: {pypi_stats['total_downloads']:,}")
    logger.info(f"Downloads with GitHub: {pypi_stats['downloads_with_github']:,} ({pypi_stats['percent_downloads_with_github']:.1f}%)")
    logger.info(f"Downloads without GitHub: {pypi_stats['downloads_without_github']:,}")
    logger.info(f"\nTop 10 packages without GitHub URLs:")
    for i, pkg in enumerate(pypi_stats['top_10_packages_without_github'], 1):
        logger.info(f"  {i}. {pkg['name']}: {pkg['downloads']:,} downloads")

    # Combined stats
    combined_total_downloads = npm_stats['total_downloads'] + pypi_stats['total_downloads']
    combined_downloads_with_github = npm_stats['downloads_with_github'] + pypi_stats['downloads_with_github']
    combined_percent = (combined_downloads_with_github / combined_total_downloads * 100) if combined_total_downloads > 0 else 0

    logger.info(f"\n=== Combined Coverage ===")
    logger.info(f"Total downloads (npm + PyPI): {combined_total_downloads:,}")
    logger.info(f"Downloads with GitHub: {combined_downloads_with_github:,} ({combined_percent:.1f}%)")

    # Save results
    results = {
        'npm': npm_stats,
        'pypi': pypi_stats,
        'combined': {
            'total_downloads': combined_total_downloads,
            'downloads_with_github': combined_downloads_with_github,
            'percent_downloads_with_github': combined_percent
        }
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()

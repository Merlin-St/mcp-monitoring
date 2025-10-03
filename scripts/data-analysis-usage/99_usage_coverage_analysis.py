#!/usr/bin/env python3
"""
Analyze coverage of GitHub matching fields in npm and PyPI usage data.
Checks what percentage of packages have usable fields for matching with GitHub repos.
"""

import json
import logging
from pathlib import Path
from collections import Counter
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/usage_coverage_analysis.log'),
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


def analyze_npm_coverage(npm_file):
    """Analyze npm package coverage for GitHub matching fields."""
    logger.info("Analyzing npm data coverage...")

    with open(npm_file, 'r') as f:
        data = json.load(f)

    packages = data.get('packages', {})
    total_packages = len(packages)

    stats = {
        'total_packages': total_packages,
        'with_repository': 0,
        'with_homepage': 0,
        'with_bugs': 0,
        'with_author_name': 0,
        'with_author_email': 0,
        'with_maintainers': 0,
        'with_github_repository': 0,
        'with_github_homepage': 0,
        'with_github_bugs': 0,
        'with_any_github_url': 0,
        'github_urls_found': []
    }

    for package_name, package_data in packages.items():
        metadata = package_data.get('metadata', {})

        # Check basic fields
        repository = metadata.get('repository')
        homepage = metadata.get('homepage')
        bugs = metadata.get('bugs')
        author = metadata.get('author', {})
        maintainers = metadata.get('maintainers', [])

        if repository:
            stats['with_repository'] += 1
        if homepage:
            stats['with_homepage'] += 1
        if bugs:
            stats['with_bugs'] += 1
        if author.get('name'):
            stats['with_author_name'] += 1
        if author.get('email'):
            stats['with_author_email'] += 1
        if maintainers:
            stats['with_maintainers'] += 1

        # Check GitHub URLs
        github_found = False

        repo_url = extract_github_url(repository)
        if repo_url:
            stats['with_github_repository'] += 1
            stats['github_urls_found'].append(repo_url)
            github_found = True

        home_url = extract_github_url(homepage)
        if home_url:
            stats['with_github_homepage'] += 1
            if not github_found:
                stats['github_urls_found'].append(home_url)
                github_found = True

        bugs_url = extract_github_url(bugs)
        if bugs_url:
            stats['with_github_bugs'] += 1
            if not github_found:
                stats['github_urls_found'].append(bugs_url)
                github_found = True

        if github_found:
            stats['with_any_github_url'] += 1

    # Calculate percentages
    percentages = {}
    for key, value in stats.items():
        if key not in ['total_packages', 'github_urls_found']:
            percentages[key] = (value / total_packages * 100) if total_packages > 0 else 0

    stats['percentages'] = percentages
    stats['unique_github_repos'] = len(set(stats['github_urls_found']))

    return stats


def analyze_pypi_coverage(pypi_file):
    """Analyze PyPI package coverage for GitHub matching fields."""
    logger.info("Analyzing PyPI data coverage...")

    # Read JSONL file
    packages_data = {}
    with open(pypi_file, 'r') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                package_name = record.get('name')

                # Store unique package (latest upload_time)
                if package_name not in packages_data:
                    packages_data[package_name] = record
                else:
                    # Keep the more recent one
                    if record.get('upload_time', '') > packages_data[package_name].get('upload_time', ''):
                        packages_data[package_name] = record

    total_packages = len(packages_data)

    stats = {
        'total_packages': total_packages,
        'with_project_urls': 0,
        'with_author_email': 0,
        'with_author': 0,
        'with_description': 0,
        'with_github_in_project_urls': 0,
        'with_github_in_description': 0,
        'with_any_github_url': 0,
        'github_urls_found': []
    }

    for package_name, record in packages_data.items():
        project_urls = record.get('Project-URLs', [])
        author_email = record.get('author_email')
        author = record.get('author')
        description = record.get('description', '')

        if project_urls and len(project_urls) > 0:
            stats['with_project_urls'] += 1
        if author_email:
            stats['with_author_email'] += 1
        if author:
            stats['with_author'] += 1
        if description:
            stats['with_description'] += 1

        # Check for GitHub URLs
        github_found = False

        # In Project-URLs
        for url in project_urls:
            github_url = extract_github_url(url)
            if github_url:
                stats['with_github_in_project_urls'] += 1
                stats['github_urls_found'].append(github_url)
                github_found = True
                break

        # In description
        if not github_found and description:
            github_url = extract_github_url(description)
            if github_url:
                stats['with_github_in_description'] += 1
                stats['github_urls_found'].append(github_url)
                github_found = True

        if github_found:
            stats['with_any_github_url'] += 1

    # Calculate percentages
    percentages = {}
    for key, value in stats.items():
        if key not in ['total_packages', 'github_urls_found']:
            percentages[key] = (value / total_packages * 100) if total_packages > 0 else 0

    stats['percentages'] = percentages
    stats['unique_github_repos'] = len(set(stats['github_urls_found']))

    return stats


def main():
    """Main analysis function."""
    logger.info("Starting usage data coverage analysis")

    # File paths
    npm_file = Path('data/external-usage/usage_npm.json')
    pypi_file = Path('data/external-usage/usage_bigquery_webresults_pypi.json')
    output_file = Path('data/external-usage/usage_coverage_analysis.json')

    # Analyze npm
    npm_stats = analyze_npm_coverage(npm_file)

    logger.info(f"\n=== NPM Coverage Results ===")
    logger.info(f"Total packages: {npm_stats['total_packages']}")
    logger.info(f"With repository field: {npm_stats['with_repository']} ({npm_stats['percentages']['with_repository']:.1f}%)")
    logger.info(f"With GitHub in repository: {npm_stats['with_github_repository']} ({npm_stats['percentages']['with_github_repository']:.1f}%)")
    logger.info(f"With GitHub in homepage: {npm_stats['with_github_homepage']} ({npm_stats['percentages']['with_github_homepage']:.1f}%)")
    logger.info(f"With GitHub in bugs: {npm_stats['with_github_bugs']} ({npm_stats['percentages']['with_github_bugs']:.1f}%)")
    logger.info(f"With ANY GitHub URL: {npm_stats['with_any_github_url']} ({npm_stats['percentages']['with_any_github_url']:.1f}%)")
    logger.info(f"Unique GitHub repos found: {npm_stats['unique_github_repos']}")
    logger.info(f"With author name: {npm_stats['with_author_name']} ({npm_stats['percentages']['with_author_name']:.1f}%)")
    logger.info(f"With author email: {npm_stats['with_author_email']} ({npm_stats['percentages']['with_author_email']:.1f}%)")
    logger.info(f"With maintainers: {npm_stats['with_maintainers']} ({npm_stats['percentages']['with_maintainers']:.1f}%)")

    # Analyze PyPI
    pypi_stats = analyze_pypi_coverage(pypi_file)

    logger.info(f"\n=== PyPI Coverage Results ===")
    logger.info(f"Total packages: {pypi_stats['total_packages']}")
    logger.info(f"With Project-URLs: {pypi_stats['with_project_urls']} ({pypi_stats['percentages']['with_project_urls']:.1f}%)")
    logger.info(f"With GitHub in Project-URLs: {pypi_stats['with_github_in_project_urls']} ({pypi_stats['percentages']['with_github_in_project_urls']:.1f}%)")
    logger.info(f"With GitHub in description: {pypi_stats['with_github_in_description']} ({pypi_stats['percentages']['with_github_in_description']:.1f}%)")
    logger.info(f"With ANY GitHub URL: {pypi_stats['with_any_github_url']} ({pypi_stats['percentages']['with_any_github_url']:.1f}%)")
    logger.info(f"Unique GitHub repos found: {pypi_stats['unique_github_repos']}")
    logger.info(f"With author_email: {pypi_stats['with_author_email']} ({pypi_stats['percentages']['with_author_email']:.1f}%)")
    logger.info(f"With author: {pypi_stats['with_author']} ({pypi_stats['percentages']['with_author']:.1f}%)")

    # Save results
    results = {
        'npm': npm_stats,
        'pypi': pypi_stats
    }

    # Remove large arrays before saving
    results['npm']['github_urls_found'] = f"{npm_stats['unique_github_repos']} unique URLs"
    results['pypi']['github_urls_found'] = f"{pypi_stats['unique_github_repos']} unique URLs"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()

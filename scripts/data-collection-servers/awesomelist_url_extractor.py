#!/usr/bin/env python3
"""
Awesome MCP Servers List URL Extractor
Extracts all URLs from awesome-mcp-servers README with emoji metadata
"""

import os
import re
import json
import time
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urljoin

class AwesomelistURLExtractor:
    def __init__(self):
        self.github_token = os.environ.get('GH_TOKEN')
        self.base_url = "https://github.com/punkpeye/awesome-mcp-servers"
        self.servers = []

        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            file_handler = logging.FileHandler('logs/awesomelist_data_run.log')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.INFO)

        # Emoji mappings
        self.official_emoji = '🎖️'

        self.language_emojis = {
            '🐍': 'Python',
            '📇': 'TypeScript/JavaScript',
            '🏎️': 'Go',
            '🦀': 'Rust',
            '#️⃣': 'C#',
            '☕': 'Java',
            '🌊': 'C/C++',
            '💎': 'Ruby'
        }

        self.scope_emojis = {
            '☁️': 'cloud',
            '🏠': 'local',
            '📟': 'embedded'
        }

        self.platform_emojis = {
            '🍎': 'macOS',
            '🪟': 'Windows',
            '🐧': 'Linux'
        }

    def get_github_api_headers(self):
        """Get headers for GitHub API requests"""
        headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'MCP-Awesomelist-Extractor'
        }
        if self.github_token:
            headers['Authorization'] = f'token {self.github_token}'
        return headers

    def fetch_readme_content(self):
        """Fetch current README content using GitHub API"""
        self.logger.info("Fetching awesome-mcp-servers README via GitHub API...")

        # Try GitHub API first
        api_url = "https://api.github.com/repos/punkpeye/awesome-mcp-servers/readme"
        headers = self.get_github_api_headers()
        headers['Accept'] = 'application/vnd.github.v3.raw'

        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"GitHub API failed: {e}, trying direct fetch...")

            # Fallback to direct raw URL
            try:
                readme_url = "https://raw.githubusercontent.com/punkpeye/awesome-mcp-servers/main/README.md"
                response = requests.get(readme_url)
                response.raise_for_status()
                return response.text
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Failed to fetch README: {e}")
                return None

    def extract_emojis_from_line(self, line: str) -> Dict[str, any]:
        """Extract all emoji metadata from a line"""
        metadata = {
            'is_official': False,
            'languages': [],
            'scope': [],
            'platforms': []
        }

        # Check for official emoji
        if self.official_emoji in line:
            metadata['is_official'] = True

        # Extract language emojis
        for emoji, language in self.language_emojis.items():
            if emoji in line:
                metadata['languages'].append(language)

        # Extract scope emojis
        for emoji, scope in self.scope_emojis.items():
            if emoji in line:
                metadata['scope'].append(scope)

        # Extract platform emojis
        for emoji, platform in self.platform_emojis.items():
            if emoji in line:
                metadata['platforms'].append(platform)

        return metadata

    def parse_servers_from_content(self, content: str) -> List[Dict]:
        """Parse servers from README content with emoji metadata"""
        lines = content.split('\n')
        servers = []
        current_category = None

        for line in lines:
            # Track current category
            if line.startswith('###'):
                # Extract category name
                category_match = re.search(r'###.*?<a name="([^"]+)"></a>(.+)', line)
                if category_match:
                    current_category = category_match.group(2).strip()
                    self.logger.debug(f"Found category: {current_category}")
                continue

            # Skip lines that don't start with bullet points
            if not line.strip().startswith('-'):
                continue

            # Extract emoji metadata first (before markdown links)
            emoji_metadata = self.extract_emojis_from_line(line)

            # Extract all links from the line
            link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
            link_matches = re.findall(link_pattern, line)

            for link_match in link_matches:
                name, url = link_match

                # Skip if it's clearly not a server (internal links, badges, etc.)
                if url.startswith('#') or 'awesome.re' in url or 'shields.io' in url:
                    continue

                # Skip navigation/header/community links
                if any(skip in name.lower() for skip in ['awesome', 'badge', 'discord', 'reddit', 'thai', 'english', 'chinese', 'japanese', 'korean', 'português']):
                    continue

                # Extract description - everything after the emojis and link
                desc_match = re.search(r'\[' + re.escape(name) + r'\]\([^)]+\)\s*-\s*([^\n]*)', line)
                description = desc_match.group(1).strip() if desc_match else ""

                # Clean up description
                description = re.sub(r'<[^>]*>', '', description)  # Remove HTML tags
                description = re.sub(r'\*\*', '', description)      # Remove bold markdown

                # Remove any remaining emojis from description
                for emoji in list(self.language_emojis.keys()) + list(self.scope_emojis.keys()) + list(self.platform_emojis.keys()) + [self.official_emoji]:
                    description = description.replace(emoji, '')

                description = description.strip()

                # Use "No description available" if still empty
                if not description:
                    description = "No description available"

                # Handle relative URLs
                if url.startswith('/'):
                    url = f"https://github.com{url}"
                elif not url.startswith('http'):
                    url = urljoin(self.base_url, url)

                # Determine if it's a GitHub URL
                is_github = 'github.com' in url

                server_info = {
                    'name': name.strip(),
                    'url': url.strip(),
                    'description': description,
                    'is_github': is_github,
                    'is_official': emoji_metadata['is_official'],
                    'languages': emoji_metadata['languages'],
                    'scope': emoji_metadata['scope'],
                    'platforms': emoji_metadata['platforms'],
                    'extracted_date': datetime.now().isoformat(),
                    'category': current_category
                }

                servers.append(server_info)

        # Remove duplicates while preserving order
        seen = set()
        unique_servers = []
        for server in servers:
            key = (server['name'], server['url'])
            if key not in seen:
                seen.add(key)
                unique_servers.append(server)

        return unique_servers

    def extract_current_urls(self):
        """Extract URLs from current README"""
        content = self.fetch_readme_content()

        if not content:
            self.logger.error("Failed to fetch README content")
            return

        self.servers = self.parse_servers_from_content(content)

        self.logger.info(f"Total servers found: {len(self.servers)}")

        # Count by type
        github_count = sum(1 for s in self.servers if s['is_github'])
        external_count = len(self.servers) - github_count
        official_count = sum(1 for s in self.servers if s.get('is_official', False))

        self.logger.info(f"  GitHub URLs: {github_count}")
        self.logger.info(f"  External URLs: {external_count}")
        self.logger.info(f"  Official servers (⭐): {official_count}")

        # Count by language
        language_counts = {}
        for server in self.servers:
            for lang in server.get('languages', []):
                language_counts[lang] = language_counts.get(lang, 0) + 1

        if language_counts:
            self.logger.info(f"  Top languages: {dict(sorted(language_counts.items(), key=lambda x: x[1], reverse=True)[:5])}")

        # Count by scope
        scope_counts = {}
        for server in self.servers:
            for scope in server.get('scope', []):
                scope_counts[scope] = scope_counts.get(scope, 0) + 1

        if scope_counts:
            self.logger.info(f"  Scope distribution: {scope_counts}")

    def save_urls(self, filename='data/external-servers/awesomelist_urls.json'):
        """Save extracted URLs to JSON file"""
        output_data = {
            'extraction_date': datetime.now().isoformat(),
            'source': 'https://github.com/punkpeye/awesome-mcp-servers',
            'total_servers': len(self.servers),
            'github_servers': sum(1 for s in self.servers if s['is_github']),
            'external_servers': sum(1 for s in self.servers if not s['is_github']),
            'official_servers': sum(1 for s in self.servers if s.get('is_official', False)),
            'servers': self.servers
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"URLs saved to {filename}")

    def print_summary(self):
        """Print summary of extracted URLs"""
        self.logger.info("\n" + "="*60)
        self.logger.info("AWESOMELIST URL EXTRACTION COMPLETE")
        self.logger.info("="*60)
        self.logger.info(f"Total servers: {len(self.servers)}")

        # Count by type
        github_count = sum(1 for s in self.servers if s['is_github'])
        external_count = len(self.servers) - github_count
        official_count = sum(1 for s in self.servers if s.get('is_official', False))

        self.logger.info(f"  GitHub URLs: {github_count}")
        self.logger.info(f"  External URLs: {external_count}")
        self.logger.info(f"  Official servers (⭐): {official_count}")

        # Show sample servers
        if self.servers:
            self.logger.info("\nSample servers:")
            for server in self.servers[:5]:
                langs = ', '.join(server.get('languages', [])) or 'unknown'
                scope = ', '.join(server.get('scope', [])) or 'unknown'
                official = "⭐" if server.get('is_official') else ""
                self.logger.info(f"  - {official} {server['name']}: {langs} | {scope}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Extract URLs from awesome-mcp-servers README')
    parser.add_argument('--output', default='data/external-servers/awesomelist_urls.json',
                       help='Output JSON file for URLs')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/awesomelist_data_run.log')
        ]
    )
    logger = logging.getLogger(__name__)

    # Check for GitHub token
    if not os.environ.get('GH_TOKEN'):
        logger.warning("GH_TOKEN environment variable not set")
        logger.warning("Set it with: export GH_TOKEN=your_github_token")
        logger.warning("Continuing without authentication (lower rate limits)...\n")

    # Create extractor
    extractor = AwesomelistURLExtractor()

    try:
        start_time = time.time()

        # Extract current URLs
        logger.info("Starting awesomelist URL extraction...")
        extractor.extract_current_urls()

        end_time = time.time()
        logger.info(f"Extraction completed in {end_time - start_time:.1f} seconds")

        # Save results
        extractor.save_urls(args.output)

        # Print summary
        extractor.print_summary()

    except KeyboardInterrupt:
        logger.warning("Extraction interrupted by user")
    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        raise


if __name__ == "__main__":
    main()

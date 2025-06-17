#!/usr/bin/env python3
"""
Unified MCP Data Processor

This script consolidates data from all 3 MCP server collection sources:
1. Smithery API (smithery_all_mcp_server_summaries.json)
2. GitHub repositories (github_mcp_repositories.json)  
3. Official MCP servers list (officiallist_mcp_servers.json)

It merges, deduplicates, and creates a comprehensive unified dataset.
"""

import json
import logging
import re
import urllib.parse
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dashboard_unified_mcp_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class UnifiedMCPServer:
    """Unified data structure for MCP servers from all sources"""
    # Core identifiers
    id: str  # Unique identifier derived from name/url
    name: str
    qualified_name: Optional[str] = None
    display_name: Optional[str] = None
    
    # URLs and links
    url: Optional[str] = None
    homepage: Optional[str] = None
    github_url: Optional[str] = None
    repository_url: Optional[str] = None
    
    # Descriptive info
    description: Optional[str] = None
    readme_content: Optional[str] = None
    
    # Dates
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    # Metrics
    use_count: Optional[int] = None
    stargazers_count: Optional[int] = None
    forks_count: Optional[int] = None
    
    # Technical details
    language: Optional[str] = None
    languages: Optional[Dict] = None
    topics: Optional[List[str]] = None
    
    # Owner/organization
    owner_login: Optional[str] = None
    owner_name: Optional[str] = None
    
    # Repository status
    fork: Optional[bool] = None
    archived: Optional[bool] = None
    
    # Source metadata
    data_sources: List[str] = None
    fetch_status: Optional[str] = None
    html_length: Optional[int] = None
    fetch_timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.data_sources is None:
            self.data_sources = []
        if self.topics is None:
            self.topics = []

class UnifiedMCPDataProcessor:
    def __init__(self):
        self.smithery_data = []
        self.github_data = []
        self.official_data = []
        self.unified_servers: Dict[str, UnifiedMCPServer] = {}
        
    def load_data_files(self) -> bool:
        """Load all data files"""
        try:
            # Load Smithery data
            smithery_file = Path("smithery_all_mcp_server_summaries.json")
            if smithery_file.exists():
                with open(smithery_file, 'r', encoding='utf-8') as f:
                    self.smithery_data = json.load(f)
                logger.info(f"Loaded {len(self.smithery_data)} Smithery servers")
            else:
                logger.warning("Smithery data file not found")
                
            # Load GitHub data
            github_file = Path("github_mcp_repositories.json")
            if github_file.exists():
                with open(github_file, 'r', encoding='utf-8') as f:
                    self.github_data = json.load(f)
                logger.info(f"Loaded {len(self.github_data)} GitHub repositories")
            else:
                logger.warning("GitHub data file not found")
                
            # Load Official list data - try full file first, fallback to processed file
            official_full_file = Path("officiallist_mcp_servers_full.json")
            official_file = Path("officiallist_mcp_servers.json")
            
            if official_full_file.exists():
                with open(official_full_file, 'r', encoding='utf-8') as f:
                    full_data = json.load(f)
                    # Extract servers list from the full file structure
                    if isinstance(full_data, dict) and 'servers' in full_data:
                        self.official_data = full_data['servers']
                        logger.info(f"Loaded {len(self.official_data)} Official list servers from full dataset")
                    else:
                        self.official_data = full_data if isinstance(full_data, list) else []
                        logger.warning(f"Unexpected full file structure, loaded {len(self.official_data)} servers")
            elif official_file.exists():
                with open(official_file, 'r', encoding='utf-8') as f:
                    self.official_data = json.load(f)
                logger.info(f"Loaded {len(self.official_data)} Official list servers from processed dataset")
            else:
                logger.warning("Official list data files not found")
                
            return True
            
        except Exception as e:
            logger.error(f"Error loading data files: {e}")
            return False
    
    def normalize_url(self, url: str) -> str:
        """Normalize URL for comparison and deduplication"""
        if not url:
            return ""
        
        # Remove trailing slashes, normalize protocol
        url = url.rstrip('/')
        if url.startswith('http://'):
            url = url.replace('http://', 'https://', 1)
        
        # Parse and reconstruct to normalize
        parsed = urllib.parse.urlparse(url)
        normalized = urllib.parse.urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path,
            parsed.params,
            parsed.query,
            ''  # Remove fragment
        ))
        return normalized
    
    def extract_repo_name_from_url(self, url: str) -> str:
        """Extract repository name from GitHub URL"""
        if not url:
            return ""
        
        # Handle GitHub URLs
        github_match = re.search(r'github\.com/([^/]+/[^/]+)', url)
        if github_match:
            return github_match.group(1).lower()
        
        # Handle npm package names
        npm_match = re.search(r'@([^/]+)/([^/\s]+)', url)
        if npm_match:
            return f"{npm_match.group(1)}/{npm_match.group(2)}".lower()
            
        return ""
    
    def generate_server_id(self, name: str, url: str = "", qualified_name: str = "") -> str:
        """Generate unique ID for server"""
        # Priority: qualified_name > repo_name > name > url
        if qualified_name:
            return qualified_name.lower()
        
        repo_name = self.extract_repo_name_from_url(url)
        if repo_name:
            return repo_name
            
        if name:
            return re.sub(r'[^\w\-]', '-', name.lower())
        
        if url:
            normalized = self.normalize_url(url)
            return re.sub(r'[^\w\-]', '-', normalized.split('/')[-1].lower())
        
        return f"unknown-{hash(str((name, url, qualified_name))) % 10000}"
    
    def parse_datetime(self, date_str: str) -> Optional[datetime]:
        """Parse datetime string from various formats"""
        if not date_str:
            return None
            
        try:
            # ISO format with Z
            if date_str.endswith('Z'):
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            # ISO format
            return datetime.fromisoformat(date_str.replace('+00:00', ''))
        except:
            try:
                # GitHub format
                return datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ')
            except:
                logger.warning(f"Could not parse datetime: {date_str}")
                return None
    
    def process_smithery_data(self):
        """Process Smithery data into unified format"""
        logger.info("Processing Smithery data...")
        
        for item in self.smithery_data:
            try:
                server_id = self.generate_server_id(
                    name=item.get('displayName', ''),
                    qualified_name=item.get('qualifiedName', ''),
                    url=item.get('homepage', '')
                )
                
                if server_id in self.unified_servers:
                    # Merge with existing
                    server = self.unified_servers[server_id]
                    if 'smithery' not in server.data_sources:
                        server.data_sources.append('smithery')
                    
                    # Update fields from Smithery if not already set
                    if not server.qualified_name:
                        server.qualified_name = item.get('qualifiedName')
                    if not server.display_name:
                        server.display_name = item.get('displayName')
                    if not server.use_count:
                        server.use_count = item.get('useCount')
                    if not server.homepage:
                        server.homepage = item.get('homepage')
                else:
                    # Create new server
                    server = UnifiedMCPServer(
                        id=server_id,
                        name=item.get('displayName', item.get('qualifiedName', '')),
                        qualified_name=item.get('qualifiedName'),
                        display_name=item.get('displayName'),
                        description=item.get('description'),
                        homepage=item.get('homepage'),
                        created_at=self.parse_datetime(item.get('createdAt')),
                        use_count=item.get('useCount'),
                        data_sources=['smithery']
                    )
                    self.unified_servers[server_id] = server
                    
            except Exception as e:
                logger.error(f"Error processing Smithery item: {e}")
                continue
    
    def process_github_data(self):
        """Process GitHub data into unified format"""
        logger.info("Processing GitHub data...")
        
        for item in self.github_data:
            try:
                github_url = item.get('html_url', '')
                server_id = self.generate_server_id(
                    name=item.get('name', ''),
                    url=github_url
                )
                
                if server_id in self.unified_servers:
                    # Merge with existing
                    server = self.unified_servers[server_id]
                    if 'github' not in server.data_sources:
                        server.data_sources.append('github')
                    
                    # Update fields from GitHub if not already set
                    if not server.github_url:
                        server.github_url = github_url
                    if not server.repository_url:
                        server.repository_url = github_url
                    if not server.stargazers_count:
                        server.stargazers_count = item.get('stargazers_count')
                    if not server.forks_count:
                        server.forks_count = item.get('forks_count')
                    if not server.language:
                        server.language = item.get('language')
                    if not server.languages:
                        server.languages = item.get('languages', {})
                    if not server.topics:
                        server.topics = item.get('topics', [])
                    if not server.readme_content:
                        server.readme_content = item.get('readme_content')
                    if not server.owner_login and item.get('owner'):
                        server.owner_login = item['owner'].get('login')
                        server.owner_name = item['owner'].get('name')
                    if server.fork is None:
                        server.fork = item.get('fork')
                    if server.archived is None:
                        server.archived = item.get('archived')
                else:
                    # Create new server
                    owner = item.get('owner', {})
                    server = UnifiedMCPServer(
                        id=server_id,
                        name=item.get('name', ''),
                        description=item.get('description'),
                        url=github_url,
                        github_url=github_url,
                        repository_url=github_url,
                        homepage=item.get('homepage'),
                        created_at=self.parse_datetime(item.get('created_at')),
                        updated_at=self.parse_datetime(item.get('updated_at')),
                        stargazers_count=item.get('stargazers_count'),
                        forks_count=item.get('forks_count'),
                        language=item.get('language'),
                        languages=item.get('languages', {}),
                        topics=item.get('topics', []),
                        readme_content=item.get('readme_content'),
                        owner_login=owner.get('login'),
                        owner_name=owner.get('name'),
                        fork=item.get('fork'),
                        archived=item.get('archived'),
                        data_sources=['github']
                    )
                    self.unified_servers[server_id] = server
                    
            except Exception as e:
                logger.error(f"Error processing GitHub item: {e}")
                continue
    
    def process_official_data(self):
        """Process Official list data into unified format"""
        logger.info("Processing Official list data...")
        
        for item in self.official_data:
            try:
                server_id = self.generate_server_id(
                    name=item.get('name', ''),
                    url=item.get('url', '')
                )
                
                if server_id in self.unified_servers:
                    # Merge with existing
                    server = self.unified_servers[server_id]
                    if 'official' not in server.data_sources:
                        server.data_sources.append('official')
                    
                    # Update fields from Official list if not already set
                    if not server.url:
                        server.url = item.get('url')
                    # Handle both file formats
                    if not server.fetch_status:
                        server.fetch_status = item.get('fetch_status')
                    if not server.html_length:
                        server.html_length = item.get('html_length')
                    if not server.fetch_timestamp:
                        server.fetch_timestamp = item.get('fetch_timestamp')
                    # Handle new format from full file
                    if not hasattr(server, 'is_github') or server.is_github is None:
                        server.is_github = item.get('is_github', False)
                    if not hasattr(server, 'extracted_date') or server.extracted_date is None:
                        server.extracted_date = item.get('extracted_date')
                else:
                    # Create new server - handle both file formats
                    server = UnifiedMCPServer(
                        id=server_id,
                        name=item.get('name', ''),
                        description=item.get('description'),
                        url=item.get('url'),
                        fetch_status=item.get('fetch_status'),
                        html_length=item.get('html_length'),
                        fetch_timestamp=item.get('fetch_timestamp'),
                        data_sources=['official']
                    )
                    # Add new fields from full file format
                    server.is_github = item.get('is_github', False)
                    server.extracted_date = item.get('extracted_date')
                    self.unified_servers[server_id] = server
                    
            except Exception as e:
                logger.error(f"Error processing Official item: {e}")
                continue
    
    def enhance_metadata(self):
        """Enhance metadata and classify servers"""
        logger.info("Enhancing metadata and classifying servers...")
        
        finance_keywords = [
            'finance', 'trading', 'payment', 'bank', 'stock', 'forex', 'crypto',
            'money', 'price', 'market', 'investment', 'portfolio', 'wallet',
            'financial', 'fintech', 'trading', 'alphavantage', 'coinbase'
        ]
        
        for server in self.unified_servers.values():
            try:
                # Classify as finance-related
                text_to_check = ' '.join(filter(None, [
                    server.name,
                    server.description,
                    server.qualified_name,
                    ' '.join(server.topics) if server.topics else ''
                ])).lower()
                
                server.is_finance_related = any(keyword in text_to_check for keyword in finance_keywords)
                
                # Determine primary source
                if len(server.data_sources) == 1:
                    server.primary_source = server.data_sources[0]
                elif 'smithery' in server.data_sources:
                    server.primary_source = 'smithery'
                elif 'github' in server.data_sources:
                    server.primary_source = 'github'
                else:
                    server.primary_source = 'official'
                
                # Set canonical name
                server.canonical_name = (
                    server.display_name or 
                    server.qualified_name or 
                    server.name or 
                    server.id
                )
                
            except Exception as e:
                logger.error(f"Error enhancing metadata for {server.id}: {e}")
                continue
    
    def save_unified_data(self, output_file: str = "dashboard_mcp_servers_unified.json"):
        """Save unified data to JSON file"""
        logger.info(f"Saving unified data to {output_file}...")
        
        try:
            # Convert to serializable format
            serializable_data = []
            for server in self.unified_servers.values():
                server_dict = {
                    'id': server.id,
                    'canonical_name': getattr(server, 'canonical_name', server.name),
                    'name': server.name,
                    'qualified_name': server.qualified_name,
                    'display_name': server.display_name,
                    'description': server.description,
                    'url': server.url,
                    'homepage': server.homepage,
                    'github_url': server.github_url,
                    'repository_url': server.repository_url,
                    'created_at': server.created_at.isoformat() if server.created_at else None,
                    'updated_at': server.updated_at.isoformat() if server.updated_at else None,
                    'use_count': server.use_count,
                    'stargazers_count': server.stargazers_count,
                    'forks_count': server.forks_count,
                    'language': server.language,
                    'languages': server.languages,
                    'topics': server.topics,
                    'owner_login': server.owner_login,
                    'owner_name': server.owner_name,
                    'fork': server.fork,
                    'archived': server.archived,
                    'data_sources': server.data_sources,
                    'primary_source': getattr(server, 'primary_source', None),
                    'is_finance_related': getattr(server, 'is_finance_related', False),
                    'fetch_status': server.fetch_status,
                    'html_length': server.html_length,
                    'fetch_timestamp': server.fetch_timestamp,
                    'is_github': getattr(server, 'is_github', None),
                    'extracted_date': getattr(server, 'extracted_date', None)
                }
                # Remove None values to reduce file size
                server_dict = {k: v for k, v in server_dict.items() if v is not None}
                serializable_data.append(server_dict)
            
            # Sort by canonical name
            serializable_data.sort(key=lambda x: x.get('canonical_name', '').lower())
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Successfully saved {len(serializable_data)} unified servers to {output_file}")
            
            # Generate summary statistics
            self.generate_summary_stats(serializable_data, output_file.replace('.json', '_summary.json'))
            
        except Exception as e:
            logger.error(f"Error saving unified data: {e}")
    
    def generate_summary_stats(self, data: List[Dict], summary_file: str):
        """Generate summary statistics"""
        logger.info("Generating summary statistics...")
        
        try:
            total_servers = len(data)
            
            # Count by source
            source_counts = {}
            primary_source_counts = {}
            finance_count = 0
            
            for server in data:
                # Count data sources
                for source in server.get('data_sources', []):
                    source_counts[source] = source_counts.get(source, 0) + 1
                
                # Count primary sources
                primary = server.get('primary_source')
                if primary:
                    primary_source_counts[primary] = primary_source_counts.get(primary, 0) + 1
                
                # Count finance-related
                if server.get('is_finance_related'):
                    finance_count += 1
            
            # Language distribution
            language_counts = {}
            for server in data:
                lang = server.get('language')
                if lang:
                    language_counts[lang] = language_counts.get(lang, 0) + 1
            
            # Top topics
            topic_counts = {}
            for server in data:
                for topic in server.get('topics', []):
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
            summary = {
                'total_servers': total_servers,
                'finance_related_servers': finance_count,
                'source_coverage': source_counts,
                'primary_source_distribution': primary_source_counts,
                'top_languages': dict(sorted(language_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
                'top_topics': dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:20]),
                'processing_timestamp': datetime.now().isoformat(),
                'data_quality': {
                    'servers_with_github_data': len([s for s in data if 'github' in s.get('data_sources', [])]),
                    'servers_with_smithery_data': len([s for s in data if 'smithery' in s.get('data_sources', [])]),
                    'servers_with_official_data': len([s for s in data if 'official' in s.get('data_sources', [])]),
                    'servers_with_multiple_sources': len([s for s in data if len(s.get('data_sources', [])) > 1])
                }
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Summary statistics saved to {summary_file}")
            logger.info(f"Total unified servers: {total_servers}")
            logger.info(f"Finance-related servers: {finance_count}")
            logger.info(f"Servers with multiple sources: {summary['data_quality']['servers_with_multiple_sources']}")
            
        except Exception as e:
            logger.error(f"Error generating summary statistics: {e}")
    
    def process_all(self):
        """Main processing pipeline"""
        logger.info("Starting unified MCP data processing...")
        
        if not self.load_data_files():
            logger.error("Failed to load data files")
            return False
        
        # Process each data source
        self.process_smithery_data()
        self.process_github_data()
        self.process_official_data()
        
        # Enhance and classify
        self.enhance_metadata()
        
        # Save results
        self.save_unified_data()
        
        logger.info("Unified MCP data processing completed successfully!")
        return True

def main():
    """Main function"""
    processor = UnifiedMCPDataProcessor()
    success = processor.process_all()
    
    if success:
        logger.info("✅ All processing completed successfully!")
    else:
        logger.error("❌ Processing failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
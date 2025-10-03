#!/usr/bin/env python3
"""
Unified MCP Data Processor

This script consolidates data from all 3 MCP server collection sources:
1. Smithery API (smithery_data.json)
2. GitHub repositories (github_data.json)  
3. Official MCP servers list (officiallist_data.json)

It merges, deduplicates, and creates a comprehensive unified dataset.
"""

import json
import logging
import re
import urllib.parse
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_unified_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class UnifiedMCPServer:
    """Unified data structure for MCP servers from all sources"""
    # Core identifiers
    id: str  # Unique identifier derived from name/url
    name: str  # Canonical package/repo name (lowercase, extracted from repo URL or package name)
    owner: Optional[str] = None  # Repository owner/author (lowercase)
    qualified_name: Optional[str] = None
    display_name: Optional[str] = None

    # URLs and links
    url: Optional[str] = None
    homepage: Optional[str] = None
    github_url: Optional[str] = None
    repository_url: Optional[str] = None  # GitHub repository URL (https://github.com/owner/repo)
    
    # Descriptive info
    readme_content: Optional[str] = None
    html_content: Optional[str] = None
    embedding_text: Optional[str] = None

    # Internal fields for canonical_description computation (not serialized)
    _official_description: Optional[str] = None
    _smithery_description: Optional[str] = None
    _github_description: Optional[str] = None
    
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
    
    # Tools information
    tools: Optional[List[Dict]] = None
    
    
    # Source metadata
    data_sources: List[str] = None
    fetch_status: Optional[str] = None
    html_length: Optional[int] = None
    fetch_timestamp: Optional[float] = None
    
    # Lean pipeline fields
    content_source: Optional[str] = None  # "github_readme" or "html_content"
    processing_status: Optional[str] = None  # "success", "no_content", etc.
    github_info: Optional[Dict] = None  # Structured GitHub parsing info
    readme_path: Optional[str] = None  # Path to README file
    is_github: Optional[bool] = None  # GitHub server flag
    extracted_date: Optional[str] = None  # URL extraction date

    # Awesomelist fields
    is_official: Optional[bool] = None  # Official status from awesomelist (⭐ emoji)
    scope: Optional[List[str]] = None  # Scope from awesomelist (cloud, local, embedded)
    platforms: Optional[List[str]] = None  # Platforms from awesomelist (macOS, Windows, Linux)
    category: Optional[str] = None  # Category from awesomelist
    awesomelist_languages: Optional[List[str]] = None  # Languages from awesomelist emojis

    # Computed fields
    canonical_official: Optional[bool] = None  # True if in officiallist OR (in awesomelist AND is_official=True)

    def __post_init__(self):
        if self.data_sources is None:
            self.data_sources = []
        if self.topics is None:
            self.topics = []
        if self.tools is None:
            self.tools = []
        if self.scope is None:
            self.scope = []
        if self.platforms is None:
            self.platforms = []
        if self.awesomelist_languages is None:
            self.awesomelist_languages = []

class UnifiedMCPDataProcessor:
    def __init__(self):
        self.smithery_data = []
        self.github_data = []
        self.official_data = []
        self.awesomelist_data = []
        self.unified_servers: Dict[str, UnifiedMCPServer] = {}
        
    def _update_server_fields(self, server: UnifiedMCPServer, data: Dict, field_mapping: Dict[str, str]):
        """Helper to update server fields from data source if not already set"""
        for server_attr, data_key in field_mapping.items():
            if not getattr(server, server_attr, None) and data.get(data_key):
                setattr(server, server_attr, data[data_key])
        
    def load_data_files(self) -> bool:
        """Load all data files"""
        try:
            # Load Smithery data (detailed version with tools)
            smithery_file = Path("data/external-servers/smithery_data.json")
            if smithery_file.exists():
                with open(smithery_file, 'r', encoding='utf-8') as f:
                    self.smithery_data = json.load(f)
                logger.info(f"Loaded {len(self.smithery_data)} Smithery servers")
            else:
                logger.warning("Smithery data file not found")
                
            # Load GitHub data
            github_file = Path("data/external-servers/github_data.json")
            if github_file.exists():
                with open(github_file, 'r', encoding='utf-8') as f:
                    self.github_data = json.load(f)
                logger.info(f"Loaded {len(self.github_data)} GitHub repositories")
            else:
                logger.warning("GitHub data file not found")
                
            # Load Official list data - only use the full file with GitHub metadata
            official_full_file = Path("data/external-servers/officiallist_data.json")
            
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
            else:
                logger.error("officiallist_mcp_servers_full.json not found - this is required for processing")

            # Load Awesomelist data
            awesomelist_file = Path("data/external-servers/awesomelist_data.json")

            if awesomelist_file.exists():
                with open(awesomelist_file, 'r', encoding='utf-8') as f:
                    full_data = json.load(f)
                    # Extract servers list from the full file structure
                    if isinstance(full_data, dict) and 'servers' in full_data:
                        self.awesomelist_data = full_data['servers']
                        logger.info(f"Loaded {len(self.awesomelist_data)} Awesomelist servers from full dataset")
                    else:
                        self.awesomelist_data = full_data if isinstance(full_data, list) else []
                        logger.warning(f"Unexpected awesomelist file structure, loaded {len(self.awesomelist_data)} servers")
            else:
                logger.warning("Awesomelist data file not found - continuing without awesomelist data")

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
        except (ValueError, TypeError):
            try:
                # GitHub format
                return datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ')
            except (ValueError, TypeError):
                logger.warning(f"Could not parse datetime: {date_str}")
                return None

    def extract_github_url_from_smithery(self, item: Dict) -> Optional[str]:
        """
        Extract GitHub URL from Smithery item fields.
        Searches description, homepage, and connections fields for github.com URLs.
        Returns normalized https://github.com/owner/repo format.
        """
        # GitHub URL patterns to match
        github_patterns = [
            r'https?://github\.com/([^/\s]+)/([^/\s#?]+)',  # https://github.com/owner/repo
            r'github\.com/([^/\s]+)/([^/\s#?]+)',  # github.com/owner/repo
            r'git@github\.com:([^/\s]+)/([^/\s\.]+)',  # git@github.com:owner/repo.git
        ]

        # Fields to search (in priority order)
        search_fields = [
            item.get('description', ''),
            item.get('homepage', ''),
            str(item.get('connections', ''))
        ]

        for field_value in search_fields:
            if not field_value:
                continue

            for pattern in github_patterns:
                match = re.search(pattern, str(field_value), re.IGNORECASE)
                if match:
                    owner, repo = match.groups()
                    # Clean repo name (remove .git suffix if present)
                    repo = re.sub(r'\.git$', '', repo)
                    # Return normalized URL
                    return f"https://github.com/{owner}/{repo}"

        return None

    def extract_name_from_repo_url(self, repository_url: str) -> Optional[str]:
        """Extract repository name from GitHub URL (e.g., https://github.com/owner/repo -> repo)"""
        if not repository_url:
            return None

        match = re.search(r'github\.com/[^/]+/([^/\s#?]+)', repository_url, re.IGNORECASE)
        if match:
            repo_name = match.group(1)
            # Remove .git suffix if present
            repo_name = re.sub(r'\.git$', '', repo_name)
            return repo_name.lower()

        return None

    def extract_owner_from_repo_url(self, repository_url: str) -> Optional[str]:
        """Extract owner from GitHub URL (e.g., https://github.com/owner/repo -> owner)"""
        if not repository_url:
            return None

        match = re.search(r'github\.com/([^/\s]+)/', repository_url, re.IGNORECASE)
        if match:
            return match.group(1).lower()

        return None

    def extract_name_and_owner_from_qualified_name(self, qualified_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract name and owner from Smithery qualifiedName (e.g., @author/my-server -> (my-server, author))"""
        if not qualified_name:
            return None, None

        # Match @author/package or author/package patterns
        match = re.match(r'^@?([^/]+)/(.+)$', qualified_name)
        if match:
            owner, name = match.groups()
            return name.lower(), owner.lower()

        # If no slash, treat entire string as name with no owner
        return qualified_name.lower(), None
    
    def process_smithery_data(self):
        """Process Smithery data into unified format"""
        logger.info("Processing Smithery data...")

        for item in self.smithery_data:
            try:
                # Extract repository_url from Smithery fields
                repository_url = self.extract_github_url_from_smithery(item)

                server_id = self.generate_server_id(
                    name=item.get('displayName', ''),
                    qualified_name=item.get('qualifiedName', ''),
                    url=repository_url or item.get('homepage', '')
                )

                if server_id in self.unified_servers:
                    # Merge with existing
                    server = self.unified_servers[server_id]
                    if 'smithery' not in server.data_sources:
                        server.data_sources.append('smithery')

                    # Update fields from Smithery if not already set
                    if not server.repository_url and repository_url:
                        server.repository_url = repository_url
                    if not server.qualified_name:
                        server.qualified_name = item.get('qualifiedName')
                    if not server.display_name:
                        server.display_name = item.get('displayName')
                    if not server.use_count:
                        server.use_count = item.get('useCount')
                    if not server.homepage:
                        server.homepage = item.get('homepage')
                    # Store Smithery description for canonical_description computation
                    if item.get('description') and not server._smithery_description:
                        server._smithery_description = item.get('description')

                    # Process tools data
                    if item.get('tools') and not server.tools:
                        server.tools = item.get('tools')
                else:
                    # Create new server
                    server = UnifiedMCPServer(
                        id=server_id,
                        name=item.get('displayName', item.get('qualifiedName', '')),
                        repository_url=repository_url,
                        qualified_name=item.get('qualifiedName'),
                        display_name=item.get('displayName'),
                        homepage=item.get('homepage'),
                        created_at=self.parse_datetime(item.get('createdAt')),
                        use_count=item.get('useCount'),
                        data_sources=['smithery']
                    )
                    # Store description for later canonical_description computation
                    server._smithery_description = item.get('description')

                    # Process tools data for new server
                    if item.get('tools'):
                        server.tools = item.get('tools')

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
                    field_mapping = {
                        'github_url': github_url,
                        'repository_url': github_url,
                        'stargazers_count': 'stargazers_count',
                        'forks_count': 'forks_count',
                        'language': 'language',
                        'languages': 'languages',
                        'topics': 'topics',
                        'readme_content': 'readme_content'
                    }

                    # Handle simple field mappings
                    for server_attr, data_key in field_mapping.items():
                        if server_attr in ['github_url', 'repository_url']:
                            if not getattr(server, server_attr, None):
                                setattr(server, server_attr, github_url)
                        else:
                            if not getattr(server, server_attr, None) and item.get(data_key):
                                setattr(server, server_attr, item[data_key])

                    # Store GitHub description for canonical_description computation
                    if item.get('description') and not server._github_description:
                        server._github_description = item.get('description')
                    
                    # Handle special cases
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
                    # Store GitHub description for canonical_description computation
                    server._github_description = item.get('description')
                    self.unified_servers[server_id] = server
                    
            except Exception as e:
                logger.error(f"Error processing GitHub item: {e}")
                continue
    
    def process_official_data(self):
        """Process Official list data into unified format"""
        logger.info("Processing Official list data...")

        for item in self.official_data:
            try:
                # Extract repository_url by checking multiple fields for GitHub URLs
                github_info = item.get('github_info', {})
                github_meta = item.get('github_metadata', {})

                # Priority order for extraction:
                # 1. github_info.url
                # 2. github_metadata.html_url
                # 3. url field (if it contains github.com)
                # 4. homepage field (if it contains github.com)
                repository_url = None

                if github_info.get('url'):
                    repository_url = github_info.get('url')
                elif github_meta.get('html_url'):
                    repository_url = github_meta.get('html_url')
                elif item.get('url') and 'github.com' in str(item.get('url', '')).lower():
                    repository_url = item.get('url')
                elif item.get('homepage') and 'github.com' in str(item.get('homepage', '')).lower():
                    repository_url = item.get('homepage')

                server_id = self.generate_server_id(
                    name=item.get('name', ''),
                    url=repository_url or item.get('url', '')
                )

                if server_id in self.unified_servers:
                    # Merge with existing
                    server = self.unified_servers[server_id]
                    if 'official' not in server.data_sources:
                        server.data_sources.append('official')

                    # Update fields from Official list if not already set
                    if not server.repository_url and repository_url:
                        server.repository_url = repository_url
                    if not server.url:
                        server.url = item.get('url')
                    # Handle basic officiallist fields
                    if not server.fetch_status:
                        server.fetch_status = item.get('fetch_status')
                    if not server.html_length:
                        server.html_length = item.get('html_length')
                    if not server.fetch_timestamp:
                        server.fetch_timestamp = item.get('fetch_timestamp')
                    if not server.html_content:
                        server.html_content = item.get('html_content')
                    # Handle new format from full file
                    if not hasattr(server, 'is_github') or server.is_github is None:
                        server.is_github = item.get('is_github', False)
                    if not hasattr(server, 'extracted_date') or server.extracted_date is None:
                        server.extracted_date = item.get('extracted_date')
                    # Store Official description for canonical_description computation
                    if item.get('description') and not server._official_description:
                        server._official_description = item.get('description')

                    # Handle new fields from lean pipeline
                    if not hasattr(server, 'content_source') or not server.content_source:
                        server.content_source = item.get('content_source')
                    if not hasattr(server, 'processing_status') or not server.processing_status:
                        server.processing_status = item.get('processing_status')
                    if not hasattr(server, 'github_info') or not server.github_info:
                        server.github_info = item.get('github_info')
                    if not hasattr(server, 'readme_path') or not server.readme_path:
                        server.readme_path = item.get('readme_path')
                    
                    # Consolidate readme content from all sources into readme_content field
                    # Priority: root-level readme_content > github_metadata readme_content > html_content
                    if not server.readme_content:
                        if item.get('readme_content'):
                            server.readme_content = item.get('readme_content')
                        elif item.get('github_metadata', {}).get('readme_content'):
                            server.readme_content = item.get('github_metadata', {}).get('readme_content')
                        elif item.get('html_content'):
                            server.readme_content = item.get('html_content')
                    
                    # Extract GitHub metadata from officiallist if available and not already set from direct GitHub source
                    github_meta = item.get('github_metadata', {})
                    if github_meta:
                        # Only update if we don't have better data from direct GitHub source
                        if not server.stargazers_count and github_meta.get('stargazers_count') is not None:
                            server.stargazers_count = github_meta.get('stargazers_count')
                        if not server.forks_count and github_meta.get('forks_count') is not None:
                            server.forks_count = github_meta.get('forks_count')
                        if not server.language and github_meta.get('language'):
                            server.language = github_meta.get('language')
                        if not server.languages and github_meta.get('languages'):
                            server.languages = github_meta.get('languages')
                        if not server.topics and github_meta.get('topics'):
                            server.topics = github_meta.get('topics', [])
                        if not server.created_at and github_meta.get('created_at'):
                            server.created_at = self.parse_datetime(github_meta.get('created_at'))
                        if not server.updated_at and github_meta.get('updated_at'):
                            server.updated_at = self.parse_datetime(github_meta.get('updated_at'))
                        if not server.github_url and github_meta.get('html_url'):
                            server.github_url = github_meta.get('html_url')
                        if not server.repository_url and github_meta.get('html_url'):
                            server.repository_url = github_meta.get('html_url')
                        if server.fork is None and github_meta.get('fork') is not None:
                            server.fork = github_meta.get('fork')
                        if server.archived is None and github_meta.get('archived') is not None:
                            server.archived = github_meta.get('archived')
                        if not server.owner_login and github_meta.get('owner', {}).get('login'):
                            server.owner_login = github_meta['owner'].get('login')
                            server.owner_name = github_meta['owner'].get('name')
                        # Store GitHub description for canonical_description computation
                        if github_meta.get('description') and not server._github_description:
                            server._github_description = github_meta.get('description')
                else:
                    # Create new server - handle full file format with GitHub metadata
                    server = UnifiedMCPServer(
                        id=server_id,
                        name=item.get('name', ''),
                        repository_url=repository_url,
                        url=item.get('url'),
                        html_content=item.get('html_content'),
                        fetch_status=item.get('fetch_status'),
                        html_length=item.get('html_length'),
                        fetch_timestamp=item.get('fetch_timestamp'),
                        data_sources=['official']
                    )

                    # Store Official description for canonical_description computation
                    server._official_description = item.get('description')

                    # Add fields from full file format
                    server.is_github = item.get('is_github', False)
                    server.extracted_date = item.get('extracted_date')
                    
                    # Add new fields from lean pipeline
                    server.content_source = item.get('content_source')
                    server.processing_status = item.get('processing_status')
                    server.github_info = item.get('github_info')
                    server.readme_path = item.get('readme_path')
                    
                    # Consolidate readme content from all sources into readme_content field
                    # Priority: root-level readme_content > github_metadata readme_content > html_content
                    if item.get('readme_content'):
                        server.readme_content = item.get('readme_content')
                    elif github_meta.get('readme_content'):
                        server.readme_content = github_meta.get('readme_content')
                    elif item.get('html_content'):
                        server.readme_content = item.get('html_content')
                    
                    # Extract GitHub metadata if available
                    if github_meta:
                        server.stargazers_count = github_meta.get('stargazers_count')
                        server.forks_count = github_meta.get('forks_count')
                        server.language = github_meta.get('language')
                        server.languages = github_meta.get('languages', {})
                        server.topics = github_meta.get('topics', [])
                        server.created_at = self.parse_datetime(github_meta.get('created_at'))
                        server.updated_at = self.parse_datetime(github_meta.get('updated_at'))
                        # Only set github_url and repository_url if not already set
                        if not server.github_url and github_meta.get('html_url'):
                            server.github_url = github_meta.get('html_url')
                        if not server.repository_url and github_meta.get('html_url'):
                            server.repository_url = github_meta.get('html_url')
                        server.fork = github_meta.get('fork')
                        server.archived = github_meta.get('archived')
                        if github_meta.get('owner'):
                            server.owner_login = github_meta['owner'].get('login')
                            server.owner_name = github_meta['owner'].get('name')
                        # Store GitHub description for canonical_description computation
                        if github_meta.get('description'):
                            server._github_description = github_meta.get('description')
                    
                    self.unified_servers[server_id] = server
                    
            except Exception as e:
                logger.error(f"Error processing Official item: {e}")
                continue

    def process_awesomelist_data(self):
        """Process Awesomelist data into unified format"""
        logger.info("Processing Awesomelist data...")

        for item in self.awesomelist_data:
            try:
                # Extract repository_url by checking multiple fields for GitHub URLs
                github_info = item.get('github_info', {})
                github_meta = item.get('github_metadata', {})

                # Priority order for extraction:
                # 1. github_info.url
                # 2. github_metadata.html_url
                # 3. url field (if it contains github.com)
                # 4. homepage field (if it contains github.com)
                repository_url = None

                if github_info.get('url'):
                    repository_url = github_info.get('url')
                elif github_meta.get('html_url'):
                    repository_url = github_meta.get('html_url')
                elif item.get('url') and 'github.com' in str(item.get('url', '')).lower():
                    repository_url = item.get('url')
                elif item.get('homepage') and 'github.com' in str(item.get('homepage', '')).lower():
                    repository_url = item.get('homepage')

                server_id = self.generate_server_id(
                    name=item.get('name', ''),
                    url=repository_url or item.get('url', '')
                )

                if server_id in self.unified_servers:
                    # Merge with existing
                    server = self.unified_servers[server_id]
                    if 'awesomelist' not in server.data_sources:
                        server.data_sources.append('awesomelist')

                    # Update awesomelist-specific fields if not already set
                    if not server.repository_url and repository_url:
                        server.repository_url = repository_url
                    if server.is_official is None and item.get('is_official') is not None:
                        server.is_official = item.get('is_official')
                    if not server.scope and item.get('scope'):
                        server.scope = item.get('scope', [])
                    if not server.platforms and item.get('platforms'):
                        server.platforms = item.get('platforms', [])
                    if not server.category and item.get('category'):
                        server.category = item.get('category')
                    if not server.awesomelist_languages and item.get('languages'):
                        server.awesomelist_languages = item.get('languages', [])

                    # Update basic fields from awesomelist if not already set
                    if not server.url:
                        server.url = item.get('url')
                    if not server.description and item.get('description'):
                        server._official_description = item.get('description')  # Store for canonical_description

                    # Handle awesomelist GitHub metadata (if available)
                    if github_meta:
                        if not server.stargazers_count and github_meta.get('stargazers_count') is not None:
                            server.stargazers_count = github_meta.get('stargazers_count')
                        if not server.forks_count and github_meta.get('forks_count') is not None:
                            server.forks_count = github_meta.get('forks_count')
                        if not server.language and github_meta.get('language'):
                            server.language = github_meta.get('language')
                        if not server.created_at and github_meta.get('created_at'):
                            server.created_at = self.parse_datetime(github_meta.get('created_at'))
                        if not server.updated_at and github_meta.get('updated_at'):
                            server.updated_at = self.parse_datetime(github_meta.get('updated_at'))

                    # Add readme content if available
                    if not server.readme_content and item.get('readme_content'):
                        server.readme_content = item.get('readme_content')

                else:
                    # Create new server
                    server = UnifiedMCPServer(
                        id=server_id,
                        name=item.get('name', ''),
                        repository_url=repository_url,
                        url=item.get('url'),
                        is_github=item.get('is_github', False),
                        is_official=item.get('is_official', False),
                        scope=item.get('scope', []),
                        platforms=item.get('platforms', []),
                        category=item.get('category'),
                        awesomelist_languages=item.get('languages', []),
                        extracted_date=item.get('extracted_date'),
                        readme_content=item.get('readme_content'),
                        data_sources=['awesomelist']
                    )

                    # Store description for canonical_description computation
                    server._official_description = item.get('description')

                    # Add GitHub metadata if available
                    if github_meta:
                        server.stargazers_count = github_meta.get('stargazers_count')
                        server.forks_count = github_meta.get('forks_count')
                        server.language = github_meta.get('language')
                        server.languages = github_meta.get('languages', {})
                        server.topics = github_meta.get('topics', [])
                        server.created_at = self.parse_datetime(github_meta.get('created_at'))
                        server.updated_at = self.parse_datetime(github_meta.get('updated_at'))
                        # Only set github_url and repository_url if not already set
                        if not server.github_url and github_meta.get('html_url'):
                            server.github_url = github_meta.get('html_url')
                        if not server.repository_url and github_meta.get('html_url'):
                            server.repository_url = github_meta.get('html_url')
                        server.fork = github_meta.get('fork')
                        server.archived = github_meta.get('archived')
                        if github_meta.get('owner'):
                            server.owner_login = github_meta['owner'].get('login')
                            server.owner_name = github_meta['owner'].get('name')
                        if github_meta.get('description'):
                            server._github_description = github_meta.get('description')

                    self.unified_servers[server_id] = server

            except Exception as e:
                logger.error(f"Error processing Awesomelist item: {e}")
                continue

    def enhance_metadata(self):
        """Enhance metadata for servers"""
        logger.info("Enhancing metadata...")

        servers_list = list(self.unified_servers.values())
        total_servers = len(servers_list)

        # Process metadata in optimized batches
        logger.info("Processing metadata...")
        batch_size = 1000
        for i in range(0, total_servers, batch_size):
            batch = servers_list[i:i+batch_size]
            batch_end = min(i + batch_size, total_servers)
            logger.info(f"Processing metadata batch {i//batch_size + 1} ({i+1}-{batch_end}/{total_servers})")

            for server in batch:
                try:
                    # Determine primary source
                    if len(server.data_sources) == 1:
                        server.primary_source = server.data_sources[0]
                    elif 'smithery' in server.data_sources:
                        server.primary_source = 'smithery'
                    elif 'github' in server.data_sources:
                        server.primary_source = 'github'
                    else:
                        server.primary_source = 'official'

                    # Compute canonical name and owner fields
                    # Priority: repository_url > qualified_name > existing name/owner_login
                    if server.repository_url:
                        # Extract from repository URL
                        extracted_name = self.extract_name_from_repo_url(server.repository_url)
                        extracted_owner = self.extract_owner_from_repo_url(server.repository_url)
                        if extracted_name:
                            server.name = extracted_name
                        if extracted_owner:
                            server.owner = extracted_owner
                    elif server.qualified_name:
                        # Extract from Smithery qualifiedName (e.g., @author/my-server)
                        extracted_name, extracted_owner = self.extract_name_and_owner_from_qualified_name(server.qualified_name)
                        if extracted_name and not server.name:
                            server.name = extracted_name
                        if extracted_owner and not server.owner:
                            server.owner = extracted_owner
                    else:
                        # Fallback: use existing name (lowercase) and owner_login
                        if server.name:
                            server.name = server.name.lower()
                        if not server.owner and server.owner_login:
                            server.owner = server.owner_login.lower()
                        elif not server.owner:
                            # Last resort: use name as owner (for officiallist/awesomelist without other info)
                            server.owner = server.name

                    # Ensure name is always set and lowercase
                    if not server.name:
                        # Absolute fallback: use id or display_name
                        server.name = (server.display_name or server.id).lower()
                    
                    # Set canonical description with priority: officiallist > smithery > github
                    server.canonical_description = (
                        server._official_description or
                        server._smithery_description or
                        server._github_description or
                        ""
                    )

                    # Compute canonical_official
                    # True if server is in officiallist OR (in awesomelist AND marked as official)
                    # Priority: officiallist takes precedence over awesomelist
                    if 'official' in server.data_sources:
                        server.canonical_official = True
                    elif 'awesomelist' in server.data_sources and server.is_official:
                        server.canonical_official = True
                    else:
                        server.canonical_official = False

                    # Create embedding text (must be last operation, uses canonical fields)
                    server.embedding_text = self.create_embedding_text(server)

                except Exception as e:
                    logger.error(f"Error enhancing metadata for {server.id}: {e}")
                    continue

        # Log field coverage statistics
        self._log_field_coverage()


    def _log_field_coverage(self):
        """Log field coverage statistics for repository_url, name, and owner"""
        total = len(self.unified_servers)
        if total == 0:
            return

        # Count coverage by field
        repo_url_count = sum(1 for s in self.unified_servers.values() if s.repository_url)
        name_count = sum(1 for s in self.unified_servers.values() if s.name)
        owner_count = sum(1 for s in self.unified_servers.values() if s.owner)

        # Count coverage by source
        source_stats = {}
        for server in self.unified_servers.values():
            primary = server.primary_source
            if primary not in source_stats:
                source_stats[primary] = {'total': 0, 'with_repo_url': 0, 'with_name': 0, 'with_owner': 0}

            source_stats[primary]['total'] += 1
            if server.repository_url:
                source_stats[primary]['with_repo_url'] += 1
            if server.name:
                source_stats[primary]['with_name'] += 1
            if server.owner:
                source_stats[primary]['with_owner'] += 1

        logger.info("=== Field Coverage Statistics ===")
        logger.info(f"Total servers: {total}")
        logger.info(f"repository_url coverage: {repo_url_count} ({repo_url_count/total*100:.1f}%)")
        logger.info(f"name coverage: {name_count} ({name_count/total*100:.1f}%)")
        logger.info(f"owner coverage: {owner_count} ({owner_count/total*100:.1f}%)")
        logger.info("")
        logger.info("=== Coverage by Source ===")
        for source, stats in sorted(source_stats.items()):
            total_src = stats['total']
            logger.info(f"{source}:")
            logger.info(f"  Total: {total_src}")
            logger.info(f"  repository_url: {stats['with_repo_url']} ({stats['with_repo_url']/total_src*100:.1f}%)")
            logger.info(f"  name: {stats['with_name']} ({stats['with_name']/total_src*100:.1f}%)")
            logger.info(f"  owner: {stats['with_owner']} ({stats['with_owner']/total_src*100:.1f}%)")

    def deduplicate_servers(self):
        """
        Deduplicate servers in two stages:
        Stage 1: By repository_url (case-insensitive)
        Stage 2: By owner+name combination (for servers without repository_url)
        """
        logger.info("Starting deduplication...")
        total_before = len(self.unified_servers)

        # Stage 1: Deduplicate by repository_url
        logger.info("Stage 1: Deduplicating by repository_url...")
        stage1_removed = 0

        # Group servers by normalized repository_url
        url_groups = {}
        for server_id, server in list(self.unified_servers.items()):
            if server.repository_url:
                normalized_url = server.repository_url.lower()
                if normalized_url not in url_groups:
                    url_groups[normalized_url] = []
                url_groups[normalized_url].append((server_id, server))

        # For each group with duplicates, keep the best one
        for normalized_url, servers in url_groups.items():
            if len(servers) > 1:
                # Sort servers by priority: (1) most data_sources, (2) highest stargazers_count, (3) longest readme_content
                servers.sort(key=lambda x: (
                    -len(x[1].data_sources),
                    -(x[1].stargazers_count or 0),
                    -(len(x[1].readme_content) if x[1].readme_content else 0)
                ))

                # Keep the first (best) server, merge data_sources from duplicates
                keep_id, keep_server = servers[0]
                for dup_id, dup_server in servers[1:]:
                    # Merge data_sources
                    for source in dup_server.data_sources:
                        if source not in keep_server.data_sources:
                            keep_server.data_sources.append(source)

                    # Remove duplicate
                    del self.unified_servers[dup_id]
                    stage1_removed += 1

        logger.info(f"Stage 1: Removed {stage1_removed} duplicates by repository_url")

        # Stage 2: Deduplicate by owner+name (only for servers WITHOUT repository_url)
        logger.info("Stage 2: Deduplicating by owner+name...")
        stage2_removed = 0

        # Group servers without repository_url by owner+name
        owner_name_groups = {}
        for server_id, server in list(self.unified_servers.items()):
            if not server.repository_url and server.owner and server.name:
                key = f"{server.owner.lower()}:{server.name.lower()}"
                if key not in owner_name_groups:
                    owner_name_groups[key] = []
                owner_name_groups[key].append((server_id, server))

        # For each group with duplicates, keep the best one
        for key, servers in owner_name_groups.items():
            if len(servers) > 1:
                # Sort servers by priority: (1) most data_sources, (2) highest use_count or stargazers_count
                servers.sort(key=lambda x: (
                    -len(x[1].data_sources),
                    -max(x[1].use_count or 0, x[1].stargazers_count or 0)
                ))

                # Keep the first (best) server, merge data_sources from duplicates
                keep_id, keep_server = servers[0]
                for dup_id, dup_server in servers[1:]:
                    # Merge data_sources
                    for source in dup_server.data_sources:
                        if source not in keep_server.data_sources:
                            keep_server.data_sources.append(source)

                    # Remove duplicate
                    del self.unified_servers[dup_id]
                    stage2_removed += 1

        logger.info(f"Stage 2: Removed {stage2_removed} duplicates by owner+name")

        total_after = len(self.unified_servers)
        total_removed = stage1_removed + stage2_removed

        logger.info("=== Deduplication Summary ===")
        logger.info(f"Total servers before: {total_before}")
        logger.info(f"Duplicates removed (Stage 1 - repository_url): {stage1_removed}")
        logger.info(f"Duplicates removed (Stage 2 - owner+name): {stage2_removed}")
        logger.info(f"Total duplicates removed: {total_removed}")
        logger.info(f"Total servers after: {total_after}")
        logger.info(f"Deduplication rate: {total_removed/total_before*100:.2f}%")

    def create_embedding_text(self, server: UnifiedMCPServer) -> str:
        """Create preprocessed text for embeddings using name and canonical_description"""
        try:
            text_parts = []

            # Add name (display_name if available, otherwise lowercase name)
            name = server.display_name or server.name
            if name:
                text_parts.append(name)

            # Add canonical description
            canonical_description = getattr(server, 'canonical_description', None)
            if canonical_description:
                text_parts.append(canonical_description)

            # Clean and combine text
            combined_text = ' '.join(text_parts)

            # Basic text cleaning
            import re
            # Remove excessive whitespace
            combined_text = re.sub(r'\s+', ' ', combined_text)
            # Remove special characters that might interfere with embedding
            combined_text = re.sub(r'[^\w\s\-\.\,\!\?]', ' ', combined_text)

            return combined_text.strip()

        except Exception as e:
            logger.warning(f"Error creating embedding text for server {server.id}: {e}")
            # Fallback to basic name and description
            return f"{server.name or ''} {getattr(server, 'canonical_description', '')}".strip()
    
    def _filter_servers_by_date(self):
        """Filter servers by creation date (must be after 2024-11-01)"""
        from datetime import timezone
        cutoff_date = datetime(2024, 11, 1, tzinfo=timezone.utc)
        filtered_count = 0
        
        for server in self.unified_servers.values():
            # Check if server has a creation date before cutoff
            if server.created_at:
                try:
                    # Ensure both dates are timezone-aware for comparison
                    server_date = server.created_at
                    if server_date.tzinfo is None:
                        server_date = server_date.replace(tzinfo=timezone.utc)
                    
                    if server_date < cutoff_date:
                        filtered_count += 1
                        continue  # Skip this server
                except Exception as e:
                    logger.warning(f"Error comparing date for server {server.id}: {e}")
                    # If we can't compare dates, include the server to be safe
            
            yield server
        
        logger.info(f"Filtered out {filtered_count} servers created before 2024-11-01")
    
    def _serialize_server(self, server: UnifiedMCPServer) -> Dict:
        """Convert server object to serializable dictionary"""
        server_dict = {
            'id': server.id,
            'name': server.name,
            'owner': server.owner,
            'canonical_description': getattr(server, 'canonical_description', ''),
            'qualified_name': server.qualified_name,
            'display_name': server.display_name,
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
            'fetch_status': server.fetch_status,
            'html_length': server.html_length,
            'fetch_timestamp': server.fetch_timestamp,
            'is_github': getattr(server, 'is_github', None),
            'extracted_date': getattr(server, 'extracted_date', None),
            'content_source': getattr(server, 'content_source', None),
            'processing_status': getattr(server, 'processing_status', None),
            'github_info': getattr(server, 'github_info', None),
            'readme_path': getattr(server, 'readme_path', None),
            'readme_content': server.readme_content,
            'html_content': server.html_content,
            'embedding_text': server.embedding_text,
            'tools': server.tools,
            'is_official': getattr(server, 'is_official', None),
            'scope': getattr(server, 'scope', None),
            'platforms': getattr(server, 'platforms', None),
            'category': getattr(server, 'category', None),
            'awesomelist_languages': getattr(server, 'awesomelist_languages', None),
            'canonical_official': getattr(server, 'canonical_official', None)
        }

        # Remove None values to reduce file size
        return {k: v for k, v in server_dict.items() if v is not None}
    
    def save_unified_data(self, output_file: str = "data/initial/data_unified.json"):
        """Save unified data to JSON file with streaming for memory efficiency"""
        logger.info(f"Saving unified data to {output_file}...")
        
        try:
            # Use generator for memory-efficient processing
            filtered_servers = list(self._filter_servers_by_date())
            total_servers = len(self.unified_servers)
            
            logger.info(f"Processing {len(filtered_servers)} servers (was {total_servers})")
            
            # Convert to serializable format and sort by name
            serializable_data = [self._serialize_server(server) for server in filtered_servers]
            serializable_data.sort(key=lambda x: x.get('name', '').lower())
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Successfully saved {len(serializable_data)} unified servers to {output_file}")
            
            # Generate summary statistics
            if output_file == "data/initial/data_unified.json":
                # Create basic summary for main file
                self.generate_basic_summary_stats(serializable_data, output_file.replace('.json', '_summary.json'))
            else:
                # Create detailed summary for other files
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

            for server in data:
                # Count data sources
                for source in server.get('data_sources', []):
                    source_counts[source] = source_counts.get(source, 0) + 1

                # Count primary sources
                primary = server.get('primary_source')
                if primary:
                    primary_source_counts[primary] = primary_source_counts.get(primary, 0) + 1

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
                'source_coverage': source_counts,
                'primary_source_distribution': primary_source_counts,
                'top_languages': dict(sorted(language_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
                'top_topics': dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:20]),
                'processing_timestamp': datetime.now().isoformat(),
                'data_quality': {
                    'servers_with_github_data': len([s for s in data if 'github' in s.get('data_sources', [])]),
                    'servers_with_smithery_data': len([s for s in data if 'smithery' in s.get('data_sources', [])]),
                    'servers_with_official_data': len([s for s in data if 'official' in s.get('data_sources', [])]),
                    'servers_with_awesomelist_data': len([s for s in data if 'awesomelist' in s.get('data_sources', [])]),
                    'servers_with_multiple_sources': len([s for s in data if len(s.get('data_sources', [])) > 1]),
                    'servers_canonical_official': len([s for s in data if s.get('canonical_official', False)])
                }
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Summary statistics saved to {summary_file}")
            logger.info(f"Total unified servers: {total_servers}")
            logger.info(f"Servers with multiple sources: {summary['data_quality']['servers_with_multiple_sources']}")
            
        except Exception as e:
            logger.error(f"Error generating summary statistics: {e}")
    
    def generate_basic_summary_stats(self, data: List[Dict], summary_file: str):
        """Generate basic summary statistics for main data_unified.json"""
        logger.info("Generating basic summary statistics...")
        
        try:
            total_servers = len(data)
            
            # Count by primary source
            primary_source_counts = {}
            
            for server in data:
                # Count primary sources
                primary = server.get('primary_source')
                if primary:
                    primary_source_counts[primary] = primary_source_counts.get(primary, 0) + 1
            
            # Create basic summary
            summary = {
                'total_servers': total_servers,
                'primary_source_distribution': primary_source_counts,
                'processing_timestamp': datetime.now().isoformat(),
                'data_quality': {
                    'servers_with_github_data': len([s for s in data if 'github' in s.get('data_sources', [])]),
                    'servers_with_smithery_data': len([s for s in data if 'smithery' in s.get('data_sources', [])]),
                    'servers_with_official_data': len([s for s in data if 'official' in s.get('data_sources', [])]),
                    'servers_with_awesomelist_data': len([s for s in data if 'awesomelist' in s.get('data_sources', [])]),
                    'servers_with_multiple_sources': len([s for s in data if len(s.get('data_sources', [])) > 1]),
                    'servers_canonical_official': len([s for s in data if s.get('canonical_official', False)])
                }
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Basic summary statistics saved to {summary_file}")
            logger.info(f"Total unified servers: {total_servers}")
            logger.info(f"Primary source distribution: {primary_source_counts}")
            
        except Exception as e:
            logger.error(f"Error generating basic summary statistics: {e}")
    
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
        self.process_awesomelist_data()

        # Enhance and classify
        self.enhance_metadata()

        # Deduplicate servers
        self.deduplicate_servers()

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
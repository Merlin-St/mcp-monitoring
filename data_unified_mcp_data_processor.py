#!/usr/bin/env python3
"""
Unified MCP Data Processor

This script consolidates data from all 3 MCP server collection sources:
1. Smithery API (smithery_all_mcp_server_summaries.json)
2. GitHub repositories (github_mcp_repositories.json)  
3. Official MCP servers list (officiallist_mcp_servers_full.json)

It merges, deduplicates, and creates a comprehensive unified dataset.
"""

import json
import logging
import re
import time
import urllib.parse
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from pathlib import Path
from naics_classification_config import NAICS_KEYWORDS, NAICS_KEYWORDS_SUB
from tools_extraction_utils import extract_and_classify_tools

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
    official_description: Optional[str] = None
    smithery_description: Optional[str] = None
    github_description: Optional[str] = None
    readme_content: Optional[str] = None
    html_content: Optional[str] = None
    embedding_text: Optional[str] = None
    
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
    tools_count: Optional[int] = None
    tools_names: Optional[List[str]] = None
    tools_summary: Optional[str] = None
    
    # Extracted and classified tools
    extracted_tools: Optional[List[Dict]] = None
    tools_by_access: Optional[Dict[str, int]] = None
    
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
        if self.tools is None:
            self.tools = []
        if self.tools_names is None:
            self.tools_names = []
        if self.extracted_tools is None:
            self.extracted_tools = []
        if self.tools_by_access is None:
            self.tools_by_access = {'read': 0, 'write': 0, 'execute': 0}

class UnifiedMCPDataProcessor:
    def __init__(self):
        self.smithery_data = []
        self.github_data = []
        self.official_data = []
        self.unified_servers: Dict[str, UnifiedMCPServer] = {}
        
    def load_data_files(self) -> bool:
        """Load all data files"""
        try:
            # Load Smithery data (detailed version with tools)
            smithery_file = Path("smithery_all_mcp_server_details_complete.json")
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
                
            # Load Official list data - only use the full file with GitHub metadata
            official_full_file = Path("officiallist_mcp_servers_full.json")
            
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
                    # Store Smithery description
                    if item.get('description'):
                        server.smithery_description = item.get('description')
                    
                    # Process tools data
                    if item.get('tools') and not server.tools:
                        server.tools = item.get('tools')
                        server.tools_count = len(server.tools)
                        server.tools_names = [tool.get('name', '') for tool in server.tools if tool.get('name')]
                        server.tools_summary = f"{server.tools_count} tools: {', '.join(server.tools_names[:3])}" + ("..." if len(server.tools_names) > 3 else "")
                else:
                    # Create new server
                    server = UnifiedMCPServer(
                        id=server_id,
                        name=item.get('displayName', item.get('qualifiedName', '')),
                        qualified_name=item.get('qualifiedName'),
                        display_name=item.get('displayName'),
                        description=item.get('description'),
                        smithery_description=item.get('description'),
                        homepage=item.get('homepage'),
                        created_at=self.parse_datetime(item.get('createdAt')),
                        use_count=item.get('useCount'),
                        data_sources=['smithery']
                    )
                    
                    # Process tools data for new server
                    if item.get('tools'):
                        server.tools = item.get('tools')
                        server.tools_count = len(server.tools)
                        server.tools_names = [tool.get('name', '') for tool in server.tools if tool.get('name')]
                        server.tools_summary = f"{server.tools_count} tools: {', '.join(server.tools_names[:3])}" + ("..." if len(server.tools_names) > 3 else "")
                    
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
                    # Store GitHub description
                    if item.get('description'):
                        server.github_description = item.get('description')
                else:
                    # Create new server
                    owner = item.get('owner', {})
                    server = UnifiedMCPServer(
                        id=server_id,
                        name=item.get('name', ''),
                        description=item.get('description'),
                        github_description=item.get('description'),
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
                    # Store Official description
                    if item.get('description'):
                        server.official_description = item.get('description')
                    
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
                        if not server.readme_content and github_meta.get('readme_content'):
                            server.readme_content = github_meta.get('readme_content')
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
                        # Use GitHub description as fallback if official description not available
                        if not server.github_description and github_meta.get('description'):
                            server.github_description = github_meta.get('description')
                else:
                    # Create new server - handle full file format with GitHub metadata
                    github_meta = item.get('github_metadata', {})
                    
                    server = UnifiedMCPServer(
                        id=server_id,
                        name=item.get('name', ''),
                        description=item.get('description'),
                        official_description=item.get('description'),
                        url=item.get('url'),
                        html_content=item.get('html_content'),
                        fetch_status=item.get('fetch_status'),
                        html_length=item.get('html_length'),
                        fetch_timestamp=item.get('fetch_timestamp'),
                        data_sources=['official']
                    )
                    
                    # Add fields from full file format
                    server.is_github = item.get('is_github', False)
                    server.extracted_date = item.get('extracted_date')
                    
                    # Extract GitHub metadata if available
                    if github_meta:
                        server.stargazers_count = github_meta.get('stargazers_count')
                        server.forks_count = github_meta.get('forks_count')
                        server.language = github_meta.get('language')
                        server.languages = github_meta.get('languages', {})
                        server.topics = github_meta.get('topics', [])
                        server.readme_content = github_meta.get('readme_content')
                        server.created_at = self.parse_datetime(github_meta.get('created_at'))
                        server.updated_at = self.parse_datetime(github_meta.get('updated_at'))
                        server.github_url = github_meta.get('html_url')
                        server.repository_url = github_meta.get('html_url')
                        server.fork = github_meta.get('fork')
                        server.archived = github_meta.get('archived')
                        if github_meta.get('owner'):
                            server.owner_login = github_meta['owner'].get('login')
                            server.owner_name = github_meta['owner'].get('name')
                        # Store GitHub description from metadata
                        if github_meta.get('description'):
                            server.github_description = github_meta.get('description')
                    
                    self.unified_servers[server_id] = server
                    
            except Exception as e:
                logger.error(f"Error processing Official item: {e}")
                continue
    
    def enhance_metadata(self):
        """Enhance metadata and classify servers using optimized batch processing"""
        logger.info("Enhancing metadata and classifying servers...")
        
        servers_list = list(self.unified_servers.values())
        total_servers = len(servers_list)
        
        # Prepare all text data for batch processing
        logger.info("Preparing text data for batch classification...")
        server_texts = []
        
        for server in servers_list:
            text_to_check = ' '.join(filter(None, [
                server.name or '',
                server.description or '',
                server.qualified_name or '',
                ' '.join(server.topics) if server.topics else ''
            ])).lower()
            server_texts.append(text_to_check)
        
        # Optimize NAICS classification with vectorized operations
        logger.info("Running batch NAICS classification...")
        self._classify_servers_batch(servers_list, server_texts)
        
        # Optimize NAICS subsector classification with vectorized operations
        logger.info("Running batch NAICS subsector classification...")
        self._classify_servers_subsector_batch(servers_list, server_texts)
        
        # Process other metadata in optimized batches
        logger.info("Processing remaining metadata...")
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
                    
                    # Set canonical name
                    server.canonical_name = (
                        server.display_name or 
                        server.qualified_name or 
                        server.name or 
                        server.id
                    )
                    
                    # Set canonical description with priority: officiallist > smithery > github
                    server.canonical_description = (
                        getattr(server, 'official_description', None) or
                        getattr(server, 'smithery_description', None) or
                        getattr(server, 'github_description', None) or
                        server.description or
                        ""
                    )
                    
                    # Extract and classify tools (most expensive operation)
                    server.extracted_tools = extract_and_classify_tools(server.__dict__)
                    server.tools_count = len(server.extracted_tools) if server.extracted_tools else 0
                    
                    # Add tools access level summary
                    if server.extracted_tools:
                        server.tools_by_access = {
                            'read': len([t for t in server.extracted_tools if t['access_level'] == 'read']),
                            'write': len([t for t in server.extracted_tools if t['access_level'] == 'write']),
                            'execute': len([t for t in server.extracted_tools if t['access_level'] == 'execute'])
                        }
                    else:
                        server.tools_by_access = {'read': 0, 'write': 0, 'execute': 0}
                    
                    # Create embedding text (must be last operation, uses canonical fields)
                    server.embedding_text = self.create_embedding_text(server)
                    
                except Exception as e:
                    logger.error(f"Error enhancing metadata for {server.id}: {e}")
                    continue
    
    def _classify_servers_batch(self, servers_list, server_texts):
        """Optimized batch NAICS classification using vectorized operations"""
        import re
        
        # Pre-compile all regex patterns for significant speedup
        logger.info("Pre-compiling NAICS patterns...")
        compiled_patterns = {}
        
        # Define stopwords - overly generic terms to skip
        stopwords = {
            'git', 'github', 'server', 'pos', 'directory', 'api', 'storage', 'data', 
            'natural', 'power', 'professional', 'infrastructure', 'architecture', 
            'used', 'system', 'technology', 'tech', 'platform', 'online', 'digital', 
            'internet', 'website', 'application', 'app', 'computer', 'code', 
            'programming', 'development', 'developer'
        }
        
        for sector_code, keywords in NAICS_KEYWORDS.items():
            filtered_keywords = []
            patterns = []
            
            for keyword in keywords:
                # Skip generic stopwords for single-word terms
                if len(keyword.split()) == 1 and keyword.lower() in stopwords:
                    continue
                
                filtered_keywords.append(keyword)
                
                # Pre-compile patterns for speed
                if len(keyword.split()) > 1:  # Multi-word keywords - exact phrase match
                    patterns.append((keyword, 'phrase'))
                else:  # Single word - use word boundaries
                    pattern = re.compile(r'\b' + re.escape(keyword) + r'\b')
                    patterns.append((pattern, 'regex'))
            
            compiled_patterns[sector_code] = (filtered_keywords, patterns)
        
        # Batch process all servers for each sector
        total_sectors = len(NAICS_KEYWORDS)
        for sector_idx, (sector_code, (keywords, patterns)) in enumerate(compiled_patterns.items()):
            logger.info(f"Classifying sector {sector_code} ({sector_idx + 1}/{total_sectors})")
            
            # Vectorized matching for all servers at once
            for server_idx, (server, text) in enumerate(zip(servers_list, server_texts)):
                matched_keywords = []
                
                for idx, (pattern_data, keyword) in enumerate(zip(patterns, filtered_keywords)):
                    if pattern_data[1] == 'phrase':
                        # Simple string contains check for phrases
                        if pattern_data[0] in text:
                            matched_keywords.append(pattern_data[0])
                    else:
                        # Compiled regex for single words
                        if pattern_data[0].search(text):
                            matched_keywords.append(keyword)
                
                sector_match = len(matched_keywords) > 0
                setattr(server, f'is_sector_{sector_code}', sector_match)
                setattr(server, f'sector_{sector_code}_keywords', matched_keywords)
        
        # Set finance classification for backward compatibility
        for server in servers_list:
            server.is_finance_related = getattr(server, 'is_sector_52', False)
    
    def _classify_servers_subsector_batch(self, servers_list, server_texts):
        """Optimized batch NAICS subsector classification using vectorized operations"""
        import re
        
        # Pre-compile all regex patterns for significant speedup
        logger.info("Pre-compiling NAICS subsector patterns...")
        compiled_patterns = {}
        
        # Define stopwords - overly generic terms to skip
        stopwords = {
            'git', 'github', 'server', 'pos', 'directory', 'api', 'storage', 'data', 
            'natural', 'power', 'professional', 'infrastructure', 'architecture', 
            'used', 'system', 'technology', 'tech', 'platform', 'online', 'digital', 
            'internet', 'website', 'application', 'app', 'computer', 'code', 
            'programming', 'development', 'developer'
        }
        
        for subsector_code, keywords in NAICS_KEYWORDS_SUB.items():
            filtered_keywords = []
            patterns = []
            
            for keyword in keywords:
                # Skip generic stopwords for single-word terms
                if len(keyword.split()) == 1 and keyword.lower() in stopwords:
                    continue
                
                filtered_keywords.append(keyword)
                
                # Pre-compile patterns for speed
                if len(keyword.split()) > 1:  # Multi-word keywords - exact phrase match
                    patterns.append((keyword, 'phrase'))
                else:  # Single word - use word boundaries
                    pattern = re.compile(r'\b' + re.escape(keyword) + r'\b')
                    patterns.append((pattern, 'regex'))
            
            compiled_patterns[subsector_code] = (filtered_keywords, patterns)
        
        # Batch process all servers for each subsector
        total_subsectors = len(NAICS_KEYWORDS_SUB)
        for subsector_idx, (subsector_code, (keywords, patterns)) in enumerate(compiled_patterns.items()):
            logger.info(f"Classifying subsector {subsector_code} ({subsector_idx + 1}/{total_subsectors})")
            
            # Vectorized matching for all servers at once
            for server_idx, (server, text) in enumerate(zip(servers_list, server_texts)):
                matched_keywords = []
                
                for idx, (pattern_data, keyword) in enumerate(zip(patterns, filtered_keywords)):
                    if pattern_data[1] == 'phrase':
                        # Simple string contains check for phrases
                        if pattern_data[0] in text:
                            matched_keywords.append(pattern_data[0])
                    else:
                        # Compiled regex for single words
                        if pattern_data[0].search(text):
                            matched_keywords.append(keyword)
                
                subsector_match = len(matched_keywords) > 0
                setattr(server, f'is_subsector_{subsector_code}', subsector_match)
                setattr(server, f'subsector_{subsector_code}_keywords', matched_keywords)
    
    def create_embedding_text(self, server: UnifiedMCPServer) -> str:
        """Create preprocessed text for embeddings using only canonical name and description"""
        try:
            # Only use canonical fields for embedding text
            text_parts = []
            
            # Add canonical name
            canonical_name = getattr(server, 'canonical_name', None)
            if canonical_name:
                text_parts.append(canonical_name)
            
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
            # Fallback to basic name and description if canonical fields not available
            return f"{getattr(server, 'canonical_name', server.name or '')} {getattr(server, 'canonical_description', server.description or '')}".strip()
    
    def save_unified_data(self, output_file: str = "data_unified.json"):
        """Save unified data to JSON file"""
        logger.info(f"Saving unified data to {output_file}...")
        
        try:
            # Convert to serializable format
            serializable_data = []
            for server in self.unified_servers.values():
                server_dict = {
                    'id': server.id,
                    'canonical_name': getattr(server, 'canonical_name', server.name),
                    'canonical_description': getattr(server, 'canonical_description', ''),
                    'name': server.name,
                    'qualified_name': server.qualified_name,
                    'display_name': server.display_name,
                    'description': server.description,
                    'official_description': getattr(server, 'official_description', None),
                    'smithery_description': getattr(server, 'smithery_description', None),
                    'github_description': getattr(server, 'github_description', None),
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
                    'extracted_date': getattr(server, 'extracted_date', None),
                    'readme_content': server.readme_content,
                    'html_content': server.html_content,
                    'embedding_text': server.embedding_text,
                    'tools': server.tools,
                    'tools_count': server.tools_count,
                    'tools_names': server.tools_names,
                    'tools_summary': server.tools_summary,
                    'extracted_tools': server.extracted_tools,
                    'tools_by_access': server.tools_by_access
                }
                
                # Add all sector classifications and matched keywords
                for sector_code in NAICS_KEYWORDS.keys():
                    sector_attr = f'is_sector_{sector_code}'
                    keywords_attr = f'sector_{sector_code}_keywords'
                    server_dict[sector_attr] = getattr(server, sector_attr, False)
                    server_dict[keywords_attr] = getattr(server, keywords_attr, [])
                
                # Add all subsector classifications and matched keywords
                for subsector_code in NAICS_KEYWORDS_SUB.keys():
                    subsector_attr = f'is_subsector_{subsector_code}'
                    keywords_attr = f'subsector_{subsector_code}_keywords'
                    server_dict[subsector_attr] = getattr(server, subsector_attr, False)
                    server_dict[keywords_attr] = getattr(server, keywords_attr, [])
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
            sector_counts = {}
            subsector_counts = {}
            
            # Tool statistics
            total_tools = 0
            servers_with_tools = 0
            tools_by_source = {'smithery': 0, 'readme': 0, 'html': 0}
            tools_by_access = {'read': 0, 'write': 0, 'execute': 0}
            
            for server in data:
                # Count data sources
                for source in server.get('data_sources', []):
                    source_counts[source] = source_counts.get(source, 0) + 1
                
                # Count primary sources
                primary = server.get('primary_source')
                if primary:
                    primary_source_counts[primary] = primary_source_counts.get(primary, 0) + 1
                
                # Count finance-related (backward compatibility)
                if server.get('is_finance_related'):
                    finance_count += 1
                
                # Count by sector
                for sector_code in NAICS_KEYWORDS.keys():
                    sector_attr = f'is_sector_{sector_code}'
                    if server.get(sector_attr, False):
                        if sector_code not in sector_counts:
                            sector_counts[sector_code] = 0
                        sector_counts[sector_code] += 1
                
                # Count by subsector
                for subsector_code in NAICS_KEYWORDS_SUB.keys():
                    subsector_attr = f'is_subsector_{subsector_code}'
                    if server.get(subsector_attr, False):
                        if subsector_code not in subsector_counts:
                            subsector_counts[subsector_code] = 0
                        subsector_counts[subsector_code] += 1
                
                # Count tools
                extracted_tools = server.get('extracted_tools', [])
                if extracted_tools:
                    servers_with_tools += 1
                    total_tools += len(extracted_tools)
                    
                    # Count by source
                    for tool in extracted_tools:
                        source = tool.get('source', 'unknown')
                        if source in tools_by_source:
                            tools_by_source[source] += 1
                    
                    # Count by access level
                    tools_access = server.get('tools_by_access', {})
                    for access_type in ['read', 'write', 'execute']:
                        tools_by_access[access_type] += tools_access.get(access_type, 0)
            
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
            
            # Create sector summary with names
            from naics_classification_config import NAICS_SECTORS, NAICS_SUBSECTORS
            sector_summary = {}
            for sector_code, count in sector_counts.items():
                sector_name = NAICS_SECTORS.get(sector_code, f"Sector {sector_code}")
                sector_summary[f"{sector_code} - {sector_name}"] = count
            
            # Create subsector summary with names
            subsector_summary = {}
            for subsector_code, count in subsector_counts.items():
                subsector_name = NAICS_SUBSECTORS.get(subsector_code, f"Subsector {subsector_code}")
                subsector_summary[f"{subsector_code} - {subsector_name}"] = count
            
            summary = {
                'total_servers': total_servers,
                'finance_related_servers': finance_count,
                'sector_distribution': dict(sorted(sector_summary.items(), key=lambda x: x[1], reverse=True)),
                'sector_counts_raw': sector_counts,
                'subsector_distribution': dict(sorted(subsector_summary.items(), key=lambda x: x[1], reverse=True)),
                'subsector_counts_raw': subsector_counts,
                'source_coverage': source_counts,
                'primary_source_distribution': primary_source_counts,
                'top_languages': dict(sorted(language_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
                'top_topics': dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:20]),
                'processing_timestamp': datetime.now().isoformat(),
                'tool_statistics': {
                    'total_extracted_tools': total_tools,
                    'servers_with_tools': servers_with_tools,
                    'tools_by_source': tools_by_source,
                    'tools_by_access_level': tools_by_access
                },
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
            logger.info(f"Servers with extracted tools: {servers_with_tools} ({servers_with_tools/total_servers*100:.1f}%)")
            logger.info(f"Total extracted tools: {total_tools}")
            logger.info(f"Tools by source: {tools_by_source}")
            logger.info(f"Tools by access level: {tools_by_access}")
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
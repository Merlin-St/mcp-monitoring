#!/usr/bin/env python3
"""
Finance MCP Repository Searcher
Searches collected MCP repository data for finance-related repositories
"""

import json
import re
from typing import Dict, List, Tuple
import csv
from datetime import datetime

class FinanceMCPSearcher:
    def __init__(self, data_file: str = 'mcp_repositories.json'):
        self.data_file = data_file
        self.repositories = []
        self.finance_keywords = [
            # Core finance terms
            'finance', 'financial', 'fintech', 'banking', 'bank', 'payment', 'payments',
            'money', 'currency', 'monetary', 'fiscal', 'treasury', 'accounting',
            
            # Trading and markets
            'trading', 'trader', 'trade', 'market', 'markets', 'stock', 'stocks',
            'equity', 'equities', 'exchange', 'ticker', 'portfolio', 'investment',
            'investing', 'investor', 'asset', 'assets', 'securities', 'commodity',
            'commodities', 'forex', 'fx', 'derivative', 'derivatives', 'option',
            'options', 'futures', 'hedge', 'alpha', 'beta', 'quant', 'quantitative',
            
            # Crypto and blockchain
            'crypto', 'cryptocurrency', 'bitcoin', 'ethereum', 'blockchain',
            'defi', 'decentralized finance', 'wallet', 'token', 'tokens',
            'ledger', 'smart contract', 'web3', 'nft', 'dao', 'dex',
            
            # Financial data providers
            'yahoo finance', 'yfinance', 'alpha vantage', 'alphavantage',
            'bloomberg', 'reuters', 'nasdaq', 'nyse', 'coinbase', 'binance',
            'kraken', 'polygon', 'coingecko', 'coinmarketcap',
            
            # Financial metrics
            'price', 'pricing', 'valuation', 'revenue', 'profit', 'loss',
            'balance sheet', 'income statement', 'cash flow', 'financial statement',
            'earnings', 'dividend', 'yield', 'return', 'roi', 'irr', 'npv',
            
            # Other financial terms
            'budget', 'budgeting', 'expense', 'expenses', 'transaction',
            'transactions', 'invoice', 'invoicing', 'billing', 'subscription',
            'tax', 'taxes', 'audit', 'compliance', 'risk', 'credit', 'debit',
            'loan', 'mortgage', 'insurance', 'pension', 'retirement'
        ]
        
        # Compile regex patterns for efficient searching
        self.keyword_patterns = [
            re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            for keyword in self.finance_keywords
        ]
    
    def load_data(self):
        """Load repository data from JSON file"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                self.repositories = json.load(f)
            print(f"Loaded {len(self.repositories)} repositories from {self.data_file}")
        except FileNotFoundError:
            print(f"Error: {self.data_file} not found. Run the collector script first.")
            return False
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {self.data_file}")
            return False
        return True
    
    def calculate_relevance_score(self, repo: Dict) -> Tuple[int, List[str]]:
        """Calculate finance relevance score and return matched keywords"""
        score = 0
        matched_keywords = set()
        
        # Check repository name (highest weight)
        name = repo.get('name', '').lower()
        for pattern in self.keyword_patterns:
            if pattern.search(name):
                score += 5
                matched_keywords.add(pattern.pattern.strip('\\b').lower())
        
        # Check description (high weight)
        description = repo.get('description', '') or ''
        for pattern in self.keyword_patterns:
            if pattern.search(description):
                score += 3
                matched_keywords.add(pattern.pattern.strip('\\b').lower())
        
        # Check topics (high weight)
        topics = repo.get('topics', [])
        for topic in topics:
            for pattern in self.keyword_patterns:
                if pattern.search(topic):
                    score += 4
                    matched_keywords.add(pattern.pattern.strip('\\b').lower())
        
        # Check README content (medium weight)
        readme = repo.get('readme_content', '') or ''
        # Only check first 5000 characters to avoid too much processing
        readme_sample = readme[:5000].lower()
        for pattern in self.keyword_patterns:
            if pattern.search(readme_sample):
                score += 1
                matched_keywords.add(pattern.pattern.strip('\\b').lower())
        
        # Bonus for specific finance-related languages
        languages = repo.get('languages', {})
        finance_languages = ['Python', 'R', 'Java', 'C++', 'TypeScript', 'JavaScript']
        if any(lang in languages for lang in finance_languages):
            score += 1
        
        return score, list(matched_keywords)
    
    def search_finance_repositories(self) -> List[Dict]:
        """Search for finance-related repositories"""
        finance_repos = []
        
        print("Searching for finance-related repositories...")
        
        for repo in self.repositories:
            score, keywords = self.calculate_relevance_score(repo)
            
            if score > 0:  # Found finance-related content
                finance_repo = {
                    'name': repo.get('name'),
                    'full_name': repo.get('full_name'),
                    'description': repo.get('description'),
                    'html_url': repo.get('html_url'),
                    'topics': repo.get('topics', []),
                    'language': repo.get('language'),
                    'stars': repo.get('stargazers_count', 0),
                    'relevance_score': score,
                    'matched_keywords': keywords,
                    'has_readme': bool(repo.get('readme_content')),
                    'created_at': repo.get('created_at'),
                    'updated_at': repo.get('updated_at')
                }
                finance_repos.append(finance_repo)
        
        # Sort by relevance score (descending) and then by stars
        finance_repos.sort(key=lambda x: (-x['relevance_score'], -x['stars']))
        
        return finance_repos
    
    def generate_report(self, finance_repos: List[Dict]):
        """Generate a detailed report of findings"""
        report = {
            'summary': {
                'total_mcp_repositories': len(self.repositories),
                'finance_related_repositories': len(finance_repos),
                'percentage': f"{(len(finance_repos) / len(self.repositories) * 100):.1f}%",
                'report_generated': datetime.now().isoformat()
            },
            'top_repositories': finance_repos[:20],  # Top 20
            'keyword_frequency': {},
            'language_distribution': {},
            'score_distribution': {
                'high_relevance': 0,  # score >= 10
                'medium_relevance': 0,  # score 5-9
                'low_relevance': 0  # score 1-4
            }
        }
        
        # Analyze keyword frequency
        for repo in finance_repos:
            for keyword in repo['matched_keywords']:
                report['keyword_frequency'][keyword] = report['keyword_frequency'].get(keyword, 0) + 1
        
        # Analyze language distribution
        for repo in finance_repos:
            lang = repo.get('language', 'Unknown')
            report['language_distribution'][lang] = report['language_distribution'].get(lang, 0) + 1
        
        # Analyze score distribution
        for repo in finance_repos:
            score = repo['relevance_score']
            if score >= 10:
                report['score_distribution']['high_relevance'] += 1
            elif score >= 5:
                report['score_distribution']['medium_relevance'] += 1
            else:
                report['score_distribution']['low_relevance'] += 1
        
        # Sort keyword frequency
        report['keyword_frequency'] = dict(
            sorted(report['keyword_frequency'].items(), key=lambda x: x[1], reverse=True)
        )
        
        return report
    
    def save_results(self, finance_repos: List[Dict], report: Dict):
        """Save results to multiple formats"""
        # Save detailed JSON
        with open('finance_mcp_repositories.json', 'w', encoding='utf-8') as f:
            json.dump(finance_repos, f, indent=2, ensure_ascii=False)
        
        # Save report
        with open('finance_mcp_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Save CSV for easy viewing
        with open('finance_mcp_repositories.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'name', 'full_name', 'description', 'html_url', 'language',
                'stars', 'relevance_score', 'matched_keywords', 'topics'
            ])
            writer.writeheader()
            
            for repo in finance_repos:
                row = repo.copy()
                row['matched_keywords'] = ', '.join(row['matched_keywords'])
                row['topics'] = ', '.join(row.get('topics', []))
                writer.writerow(row)
        
        # Save a simple text summary
        with open('finance_mcp_summary.txt', 'w', encoding='utf-8') as f:
            f.write("FINANCE-RELATED MCP REPOSITORIES SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total MCP repositories analyzed: {report['summary']['total_mcp_repositories']}\n")
            f.write(f"Finance-related repositories found: {report['summary']['finance_related_repositories']}\n")
            f.write(f"Percentage: {report['summary']['percentage']}\n\n")
            
            f.write("TOP 20 FINANCE MCP REPOSITORIES:\n")
            f.write("-" * 50 + "\n")
            for i, repo in enumerate(report['top_repositories'], 1):
                f.write(f"\n{i}. {repo['full_name']}\n")
                f.write(f"   Stars: {repo['stars']} | Score: {repo['relevance_score']}\n")
                f.write(f"   Description: {repo['description']}\n")
                f.write(f"   Keywords: {', '.join(repo['matched_keywords'])}\n")
                f.write(f"   URL: {repo['html_url']}\n")
            
            f.write("\n\nMOST COMMON FINANCE KEYWORDS:\n")
            f.write("-" * 50 + "\n")
            for keyword, count in list(report['keyword_frequency'].items())[:20]:
                f.write(f"{keyword}: {count}\n")


def main():
    searcher = FinanceMCPSearcher()
    
    # Load data
    if not searcher.load_data():
        return
    
    # Search for finance repositories
    finance_repos = searcher.search_finance_repositories()
    
    print(f"\nFound {len(finance_repos)} finance-related repositories")
    
    if finance_repos:
        # Generate report
        report = searcher.generate_report(finance_repos)
        
        # Save results
        searcher.save_results(finance_repos, report)
        
        print("\nResults saved to:")
        print("  - finance_mcp_repositories.json (detailed data)")
        print("  - finance_mcp_report.json (analysis report)")
        print("  - finance_mcp_repositories.csv (spreadsheet format)")
        print("  - finance_mcp_summary.txt (human-readable summary)")
        
        print(f"\nTop 5 finance MCP repositories:")
        for i, repo in enumerate(finance_repos[:5], 1):
            print(f"{i}. {repo['full_name']} (Score: {repo['relevance_score']}, Stars: {repo['stars']})")
            print(f"   Keywords: {', '.join(repo['matched_keywords'][:5])}")
    else:
        print("No finance-related repositories found.")


if __name__ == "__main__":
    main()
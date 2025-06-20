# NAICS Classification Configuration for MCP Server Embedding Analysis
# This file contains the NAICS sector definitions and keyword mappings for classifying MCP servers

# NAICS Sector Definitions
NAICS_SECTORS = {
    11: "Agriculture, Forestry, Fishing and Hunting",
    21: "Mining, Quarrying, and Oil and Gas Extraction", 
    22: "Utilities",
    23: "Construction",
    31: "Manufacturing",
    32: "Manufacturing", 
    33: "Manufacturing",
    42: "Wholesale Trade",
    44: "Retail Trade",
    45: "Retail Trade", 
    48: "Transportation and Warehousing",
    49: "Transportation and Warehousing",
    51: "Information",
    52: "Finance and Insurance",
    53: "Real Estate and Rental and Leasing",
    54: "Professional, Scientific, and Technical Services",
    55: "Management of Companies and Enterprises",
    56: "Administrative and Support and Waste Management and Remediation Services",
    61: "Educational Services", 
    62: "Health Care and Social Assistance",
    71: "Arts, Entertainment, and Recreation",
    72: "Accommodation and Food Services",
    81: "Other Services (except Public Administration)",
    92: "Public Administration",
    99: "Unclassified"
}

# Keyword mappings for each NAICS sector
NAICS_KEYWORDS = {
    # Finance and Insurance (52) - High priority detection
    52: [
        'finance', 'financial', 'bank', 'banking', 'payment', 'pay', 'money', 'trading', 'trade',
        'investment', 'invest', 'stock', 'market', 'crypto', 'cryptocurrency', 'blockchain',
        'insurance', 'loan', 'credit', 'debit', 'wallet', 'portfolio', 'exchange', 'forex',
        'accounting', 'budget', 'expense', 'revenue', 'profit', 'loss', 'tax', 'treasury',
        'payroll', 'invoice', 'billing', 'stripe', 'paypal', 'venmo', 'cashapp', 'square',
        'plaid', 'mint', 'quickbooks', 'xero', 'sage', 'robinhood', 'coinbase', 'binance',
        'fintech', 'ledger', 'cashflow', 'mortgage', 'refinance', 'apr', 'interest rate',
        'compound interest', 'dividend', 'equity', 'bond', 'mutual fund', 'etf', 'ira',
        '401k', 'retirement', 'pension', 'annuity', 'actuarial', 'underwriting'
    ],
    
    # Information (51) - Technology, software, data
    51: [
        'software', 'app', 'application', 'system', 'platform', 'server', 'database', 'data',
        'api', 'web', 'website', 'internet', 'online', 'digital', 'tech', 'technology',
        'computer', 'programming', 'code', 'development', 'developer', 'github', 'git',
        'cloud', 'aws', 'azure', 'google', 'microsoft', 'facebook', 'twitter', 'social',
        'media', 'content', 'cms', 'blog', 'news', 'information', 'search', 'analytics',
        'ai', 'ml', 'machine learning', 'artificial intelligence', 'chatbot', 'bot',
        'saas', 'paas', 'iaas', 'devops', 'automation', 'integration', 'workflow',
        'notification', 'messaging', 'communication', 'collaboration', 'productivity'
    ],
    
    # Professional, Scientific, and Technical Services (54)
    54: [
        'consulting', 'consultant', 'legal', 'law', 'attorney', 'lawyer', 'engineering',
        'architect', 'design', 'research', 'science', 'scientific', 'analysis', 'analytics',
        'audit', 'tax preparation', 'bookkeeping', 'marketing', 'advertising', 'creative',
        'seo', 'optimization', 'strategy', 'business intelligence', 'data analysis'
    ],
    
    # Health Care and Social Assistance (62)
    62: [
        'health', 'healthcare', 'medical', 'hospital', 'clinic', 'doctor', 'patient',
        'medicine', 'pharmaceutical', 'drug', 'therapy', 'treatment', 'care', 'wellness',
        'fitness', 'mental health', 'dental', 'vision', 'insurance', 'medicare', 'medicaid',
        'telemedicine', 'telehealth', 'ehr', 'emr', 'hipaa', 'medical records'
    ],
    
    # Retail Trade (44-45)
    44: [
        'retail', 'store', 'shop', 'shopping', 'ecommerce', 'e-commerce', 'marketplace',
        'sales', 'selling', 'buy', 'purchase', 'product', 'catalog', 'inventory',
        'amazon', 'ebay', 'shopify', 'woocommerce', 'magento', 'cart', 'checkout',
        'pos', 'point of sale', 'merchant', 'customer', 'order', 'fulfillment'
    ],
    
    # Transportation and Warehousing (48-49)
    48: [
        'transport', 'transportation', 'shipping', 'delivery', 'logistics', 'warehouse',
        'freight', 'cargo', 'vehicle', 'truck', 'car', 'auto', 'uber', 'lyft', 'taxi',
        'ride', 'travel', 'trip', 'route', 'navigation', 'gps', 'maps', 'fedex', 'ups',
        'supply chain', 'distribution', 'fleet', 'tracking', 'dispatch'
    ],
    
    # Real Estate and Rental and Leasing (53)
    53: [
        'real estate', 'property', 'rent', 'rental', 'lease', 'housing', 'home', 'house',
        'apartment', 'condo', 'mortgage', 'realtor', 'broker', 'listing', 'mls',
        'zillow', 'redfin', 'airbnb', 'vrbo', 'booking', 'property management'
    ],
    
    # Educational Services (61)
    61: [
        'education', 'educational', 'school', 'university', 'college', 'student', 'teacher',
        'learning', 'course', 'training', 'tutorial', 'lesson', 'curriculum', 'academic',
        'research', 'study', 'exam', 'grade', 'classroom', 'online learning', 'elearning',
        'lms', 'learning management', 'mooc', 'certification'
    ],
    
    # Arts, Entertainment, and Recreation (71)
    71: [
        'entertainment', 'game', 'gaming', 'music', 'video', 'movie', 'film', 'art',
        'creative', 'media', 'streaming', 'netflix', 'spotify', 'youtube', 'twitch',
        'social media', 'instagram', 'tiktok', 'recreation', 'sport', 'fitness', 'gym',
        'podcast', 'photography', 'design', 'animation'
    ],
    
    # Accommodation and Food Services (72)
    72: [
        'hotel', 'motel', 'accommodation', 'hospitality', 'restaurant', 'food', 'dining',
        'menu', 'recipe', 'cooking', 'kitchen', 'cafe', 'bar', 'catering', 'delivery',
        'takeout', 'reservation', 'booking', 'yelp', 'doordash', 'ubereats', 'grubhub',
        'pos', 'table management', 'guest services'
    ],
    
    # Manufacturing (31-33)
    31: [
        'manufacturing', 'factory', 'production', 'assembly', 'industrial', 'machinery',
        'equipment', 'supply chain', 'inventory', 'quality control', 'automation',
        'lean', 'six sigma', 'erp', 'mrp', 'production planning'
    ],
    
    # Utilities (22)
    22: [
        'utility', 'utilities', 'power', 'electricity', 'energy', 'gas', 'water', 'electric',
        'renewable', 'solar', 'wind', 'grid', 'infrastructure', 'smart grid', 'meter'
    ],
    
    # Agriculture, Forestry, Fishing and Hunting (11)
    11: [
        'agriculture', 'farming', 'farm', 'crop', 'livestock', 'forestry', 'fishing',
        'hunting', 'agricultural', 'food production', 'organic', 'sustainable',
        'precision agriculture', 'agtech', 'irrigation'
    ],
    
    # Wholesale Trade (42)
    42: [
        'wholesale', 'distributor', 'distribution', 'supplier', 'bulk', 'b2b',
        'business to business', 'trade', 'import', 'export', 'procurement'
    ],
    
    # Construction (23)
    23: [
        'construction', 'building', 'contractor', 'builder', 'renovation', 'repair',
        'maintenance', 'infrastructure', 'architecture', 'engineering', 'project management'
    ],
    
    # Mining, Quarrying, and Oil and Gas Extraction (21)
    21: [
        'mining', 'oil', 'gas', 'petroleum', 'extraction', 'drilling', 'quarry',
        'mineral', 'coal', 'natural gas', 'energy extraction', 'refinery'
    ],
    
    # Public Administration (92)
    92: [
        'government', 'public', 'municipal', 'federal', 'state', 'city', 'county',
        'administration', 'civic', 'policy', 'regulation', 'compliance', 'tax',
        'social services', 'public safety', 'emergency', 'permits', 'licensing'
    ],
    
    # Other Services (81)
    81: [
        'repair', 'maintenance', 'personal services', 'salon', 'beauty', 'laundry',
        'dry cleaning', 'pet', 'veterinary', 'nonprofit', 'association', 'organization'
    ],
    
    # Administrative and Support Services (56)
    56: [
        'administrative', 'support', 'temp', 'staffing', 'human resources', 'hr',
        'payroll', 'recruiting', 'cleaning', 'security', 'waste', 'facility',
        'office management', 'virtual assistant'
    ],
    
    # Management of Companies (55)
    55: [
        'management', 'holding company', 'corporate', 'headquarters', 'subsidiary',
        'parent company', 'conglomerate', 'enterprise', 'portfolio management'
    ]
}

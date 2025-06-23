# NAICS Classification Configuration for MCP Server Embedding Analysis
# This file contains the NAICS sector definitions and keyword mappings for classifying MCP servers

# NAICS Sector Definitions
NAICS_SECTORS = {
    11: "Agriculture, Forestry, Fishing and Hunting",
    21: "Mining, Quarrying, and Oil and Gas Extraction",
    22: "Utilities",
    23: "Construction",
    31: "Manufacturing",
    42: "Wholesale Trade",
    44: "Retail Trade",
    48: "Transportation and Warehousing",
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
    # Agriculture, Forestry, Fishing and Hunting (11)
    11: [
        'farming', 'production', 'animal', 'forest', 'oilseed', 'grain', 'vegetable',
        'melon', 'fruit', 'tree', 'greenhouse', 'nursery', 'floriculture', 'cattle',
        'ranching', 'hog', 'poultry', 'egg', 'sheep', 'goat', 'aquaculture', 'timber',
        'tract', 'nurseries', 'gathering', 'logging', 'fishing', 'hunting', 'trapping',
        'forestry', 'agriculture', 'farm', 'crop', 'livestock', 'agricultural',
        'food production', 'organic', 'sustainable', 'precision agriculture', 'agtech',
        'irrigation'
    ],

    # Mining, Quarrying, and Oil and Gas Extraction (21)
    21: [
        'mining', 'oil', 'gas', 'coal', 'metal', 'ore', 'nonmetallic', 'mineral',
        'quarrying', 'petroleum', 'extraction', 'drilling', 'quarry', 'natural gas',
        'energy extraction', 'refinery'
    ],

    # Utilities (22)
    22: [
        'distribution', 'electric', 'power', 'transmission', 'natural', 'gas', 'water',
        'sewage', 'utility', 'utilities', 'electricity', 'energy', 'renewable', 'solar',
        'wind', 'grid', 'infrastructure', 'smart grid', 'meter'
    ],

    # Construction (23)
    23: [
        'building', 'construction', 'contractors', 'residential', 'nonresidential',
        'utility', 'land', 'subdivision', 'highway', 'street', 'bridge', 'heavy',
        'civil', 'engineering', 'exterior', 'finishing', 'specialty', 'trade',
        'contractor', 'builder', 'renovation', 'repair', 'maintenance', 'infrastructure',
        'architecture', 'project management'
    ],

    # Manufacturing (31)
    31: [
        'manufacturing', 'factory', 'production', 'assembly', 'industrial',
        'machinery', 'equipment', 'supply chain', 'inventory', 'quality control',
        'automation', 'lean', 'six sigma', 'erp', 'mrp', 'production planning'
    ],

    # Wholesale Trade (42)
    42: [
        'goods', 'petroleum', 'motor', 'vehicle', 'furniture', 'home', 'furnishing',
        'lumber', 'construction', 'materials', 'commercial', 'metal', 'mineral',
        'household', 'appliances', 'electrical', 'electronic', 'hardware', 'plumbing',
        'heating', 'machinery', 'durable', 'drugs', 'druggists', 'sundries', 'apparel',
        'piece', 'notions', 'grocery', 'farm', 'wholesale', 'distributor',
        'distribution', 'supplier', 'bulk', 'b2b', 'business to business', 'trade',
        'import', 'export', 'procurement'
    ],

    # Retail Trade (44)
    44: [
        'retail', 'store', 'shop', 'shopping', 'ecommerce', 'e-commerce',
        'marketplace', 'sales', 'selling', 'buy', 'purchase', 'product', 'catalog',
        'inventory', 'amazon', 'ebay', 'shopify', 'woocommerce', 'magento', 'cart',
        'checkout', 'pos', 'point of sale', 'merchant', 'customer', 'order',
        'fulfillment'
    ],

    # Transportation and Warehousing (48)
    48: [
        'transport', 'transportation', 'shipping', 'delivery', 'logistics',
        'warehouse', 'freight', 'cargo', 'vehicle', 'truck', 'car', 'auto', 'uber',
        'lyft', 'taxi', 'ride', 'travel', 'trip', 'route', 'navigation', 'gps', 'maps',
        'fedex', 'ups', 'supply chain', 'distribution', 'fleet', 'tracking', 'dispatch'
    ],

    # Information (51)
    51: [
        'telecommunications', 'publishers', 'media', 'networks', 'providers',
        'satellite', 'web', 'motion', 'picture', 'video', 'sound', 'recording',
        'newspaper', 'periodical', 'book', 'directory', 'software', 'radio',
        'television', 'broadcasting', 'stations', 'streaming', 'distribution', 'social',
        'content', 'wired', 'wireless', 'computing', 'processing', 'hosting', 'app',
        'application', 'system', 'platform', 'server', 'database', 'data', 'api',
        'website', 'internet', 'online', 'digital', 'tech', 'technology', 'computer',
        'programming', 'code', 'development', 'developer', 'github', 'git', 'cloud',
        'aws', 'azure', 'google', 'microsoft', 'facebook', 'twitter', 'cms', 'blog',
        'news', 'information', 'search', 'analytics', 'ai', 'ml', 'machine learning',
        'artificial intelligence', 'chatbot', 'bot', 'saas', 'paas', 'iaas', 'devops',
        'automation', 'integration', 'workflow', 'notification', 'messaging',
        'communication', 'collaboration', 'productivity'
    ],

    # Finance and Insurance (52)
    52: [
        'finance', 'financial', 'bank', 'banking', 'payment', 'pay', 'money',
        'trading', 'trade', 'investment', 'invest', 'stock', 'crypto', 'cryptocurrency',
        'blockchain', 'insurance', 'loan', 'credit', 'debit', 'wallet', 'portfolio',
        'exchange', 'forex', 'budget', 'expense', 'revenue', 'treasury', 'payroll',
        'invoice', 'billing', 'stripe', 'paypal', 'venmo', 'cashapp', 'square', 'plaid',
        'quickbooks', 'xero', 'sage', 'robinhood', 'coinbase', 'binance', 'fintech',
        'ledger', 'cashflow', 'mortgage', 'refinance', 'apr', 'interest rate',
        'compound interest', 'dividend', 'equity', 'bond', 'mutual fund', 'etf', 'ira',
        '401k', 'retirement', 'pension', 'annuity', 'actuarial', 'underwriting',
        'intermediation', 'securities', 'commodity', 'funds', 'monetary',
        'authorities-central', 'depository', 'nondepository', 'contracts', 'brokerage',
        'exchanges', 'carriers', 'agencies', 'brokerages', 'employee', 'benefit',
        'pools'
    ],

    # Real Estate and Rental and Leasing (53)
    53: [
        'rental', 'estate', 'lessors', 'leasing', 'brokers', 'automotive', 'consumer',
        'goods', 'centers', 'commercial', 'industrial', 'machinery', 'nonfinancial',
        'intangible', 'assets', 'copyrighted', 'real estate', 'property', 'rent',
        'lease', 'housing', 'home', 'house', 'apartment', 'condo', 'mortgage', 'realtor',
        'broker', 'listing', 'mls', 'zillow', 'redfin', 'airbnb', 'vrbo', 'booking',
        'property management'
    ],

    # Professional, Scientific, and Technical Services (54)
    54: [
        'scientific', 'legal', 'accounting', 'tax', 'preparation', 'bookkeeping',
        'payroll', 'architectural', 'engineering', 'computer', 'consulting',
        'advertising', 'public', 'consultant', 'law', 'attorney', 'lawyer', 'architect',
        'design', 'research', 'science', 'analysis', 'analytics', 'audit',
        'tax preparation', 'marketing', 'creative', 'seo', 'optimization', 'strategy',
        'business intelligence', 'data analysis'
    ],

    # Management of Companies and Enterprises (55)
    55: [
        'companies', 'enterprises', 'management', 'holding company', 'corporate',
        'headquarters', 'subsidiary', 'parent company', 'conglomerate', 'enterprise',
        'portfolio management'
    ],

    # Administrative and Support and Waste Management and Remediation Services (56)
    56: [
        'waste', 'office', 'administrative', 'employment', 'travel', 'arrangement',
        'reservation', 'investigation', 'buildings', 'dwellings', 'treatment',
        'disposal', 'remediation', 'support', 'temp', 'staffing', 'human resources',
        'hr', 'payroll', 'recruiting', 'cleaning', 'security', 'facility',
        'office management', 'virtual assistant'
    ],

    # Educational Services (61)
    61: [
        'schools', 'colleges', 'elementary', 'secondary', 'junior', 'universities',
        'computer', 'training', 'trade', 'instruction', 'educational', 'education',
        'school', 'university', 'college', 'student', 'teacher', 'learning', 'course',
        'tutorial', 'lesson', 'curriculum', 'academic', 'research', 'study', 'exam',
        'grade', 'classroom', 'online learning', 'elearning', 'lms',
        'learning management', 'mooc', 'certification'
    ],

    # Health Care and Social Assistance (62)
    62: [
        'care', 'health', 'hospitals', 'substance', 'abuse', 'medical', 'psychiatric',
        'nursing', 'residential', 'physicians', 'dentists', 'practitioners',
        'outpatient', 'centers', 'diagnostic', 'laboratories', 'home', 'ambulatory',
        'surgical', 'specialty', 'skilled', 'intellectual', 'developmental',
        'disability', 'mental', 'continuing', 'retirement', 'communities', 'assisted',
        'living', 'healthcare', 'hospital', 'clinic', 'doctor', 'patient', 'medicine',
        'pharmaceutical', 'drug', 'therapy', 'treatment', 'wellness', 'fitness',
        'mental health', 'dental', 'vision', 'insurance', 'medicare', 'medicaid',
        'telemedicine', 'telehealth', 'ehr', 'emr', 'hipaa', 'medical records'
    ],

    # Arts, Entertainment, and Recreation (71)
    71: [
        'performing', 'arts', 'sports', 'artists', 'amusement', 'companies',
        'spectator', 'promoters', 'events', 'managers', 'athletes', 'entertainers',
        'public', 'figures', 'independent', 'writers', 'performers', 'museums',
        'historical', 'sites', 'institutions', 'parks', 'arcades', 'gambling',
        'recreation', 'entertainment', 'game', 'gaming', 'music', 'video', 'movie',
        'film', 'art', 'creative', 'media', 'streaming', 'netflix', 'spotify', 'youtube',
        'twitch', 'social media', 'instagram', 'tiktok', 'sport', 'fitness', 'gym',
        'podcast', 'photography', 'design', 'animation'
    ],

    # Accommodation and Food Services (72)
    72: [
        'recreational', 'camps', 'places', 'traveler', 'accommodation', 'vehicle',
        'parks', 'rooming', 'boarding', 'houses', 'dormitories', 'workers', 'special',
        'food', 'drinking', 'alcoholic', 'beverages', 'restaurants', 'eating', 'hotel',
        'motel', 'hospitality', 'restaurant', 'dining', 'menu', 'recipe', 'cooking',
        'kitchen', 'cafe', 'bar', 'catering', 'delivery', 'takeout', 'reservation',
        'booking', 'yelp', 'doordash', 'ubereats', 'grubhub', 'pos', 'table management',
        'guest services'
    ],

    # Other Services (except Public Administration) (81)
    81: [
        'repair', 'maintenance', 'organizations', 'personal', 'automotive',
        'electronic', 'care', 'social', 'commercial', 'industrial', 'machinery',
        'household', 'goods', 'death', 'drycleaning', 'laundry', 'religious',
        'grantmaking', 'giving', 'advocacy', 'civic', 'labor', 'political', 'private',
        'households', 'personal services', 'salon', 'beauty', 'dry cleaning', 'pet',
        'veterinary', 'nonprofit', 'association', 'organization'
    ],

    # Public Administration (92)
    92: [
        'programs', 'executive', 'legislative', 'government', 'justice', 'public',
        'order', 'safety', 'environmental', 'housing', 'urban', 'community', 'economic',
        'space', 'technology', 'national', 'international', 'affairs', 'municipal',
        'federal', 'state', 'city', 'county', 'administration', 'civic', 'policy',
        'regulation', 'compliance', 'tax', 'social services', 'public safety',
        'emergency', 'permits', 'licensing'
    ],

}

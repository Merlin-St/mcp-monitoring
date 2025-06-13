# MCP Server Monitoring Dashboard

Interactive web dashboards for monitoring Model Context Protocol (MCP) server ecosystem growth and finance-specific tool availability.

## Quick Start

```bash
# Launch dashboard (choose one)
    # Multi-source unified view
streamlit run dashboard_smithery_local_mcp.py  # Smithery-focused analysis
streamlit run dashboard_finance_mcp.py      # Finance-specific tools
```

## Available Dashboards

### 1. Unified Dashboard (`dashboard_unified_mcp.py`)
- **Purpose**: Comprehensive view across all data sources
- **Features**: 
  - MCP server creation trends over time
  - Cross-platform tool availability analysis
  - Finance vs. general-purpose server comparison
  - Interactive filtering by sector and use case

### 2. Smithery Dashboard (`dashboard_smithery_local_mcp.py`)
- **Purpose**: Deep dive into Smithery registry data
- **Features**:
  - Detailed server categorization
  - Tool popularity metrics
  - Financial risk assessment
  - Autonomy level classification

### 3. Finance Dashboard (`dashboard_finance_mcp.py`)
- **Purpose**: Finance-specific tool analysis
- **Features**:
  - Payment execution capabilities
  - Banking and trading tools
  - Risk assessment dashboard
  - Regulatory compliance tracking

## Key Visualizations

### Graphs
1. **Server Growth Timeline** - Creation trends with finance vs. general breakdown
2. **Tool Availability Trends** - Growth in finance-specific capabilities

### Tables
1. **Payment Tools Overview** - All tools relevant to automatic payments
2. **Complete Server Catalog** - Filterable by sector, use case, autonomy level

## Data Sources

- **Smithery Registry**: Official MCP server database
- **GitHub Repositories**: Community-developed servers
- **Official ModelContextProtocol List**: Curated server collection

## Research Focus

Answers key questions about AI tool proliferation in financial systems:
- Which tools are available for AI power-users in finance?
- What is the uptake for these tools over time?
- How many new tools are made available over time?
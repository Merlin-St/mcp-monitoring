#!/usr/bin/env python3
"""
Finance-Focused MCP Server Dashboard

Addresses the specific research questions from CLAUDE.md:
1. Which tools are available for AI power-users in finance currently?
2. What is the uptake for these tools over time? 
3. How many new tools are made available over time?

Delivers: 2 Graphs + 2 Tables as specified in requirements
"""

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import re
import numpy as np
from pathlib import Path
from naics_classification_config import NAICS_KEYWORDS

# Page configuration
st.set_page_config(
    page_title="MCP Finance Tools Monitor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_and_analyze_finance_data():
    """Load unified data and perform finance-specific analysis"""
    data_file = Path("data_unified.json")
    
    if not data_file.exists():
        st.error("Unified data file not found. Please run unified_mcp_data_processor.py first.")
        return pd.DataFrame()
    
    # Load data
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Convert dates
    for col in ['created_at', 'updated_at']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Get finance keywords from central configuration
    finance_keywords_base = NAICS_KEYWORDS[52]  # Finance and Insurance sector
    
    # Enhanced finance classification with specific categories
    finance_keywords = {
        'payment_execution': [
            'payment', 'pay', 'transaction', 'transfer', 'send money', 'wire', 
            'paypal', 'stripe', 'square', 'venmo', 'cashapp', 'zelle',
            'cryptocurrency', 'bitcoin', 'ethereum', 'wallet', 'blockchain'
        ],
        'market_data': [
            'market', 'stock', 'trading', 'ticker', 'price', 'quote', 'financial data',
            'bloomberg', 'reuters', 'yahoo finance', 'alpha vantage', 'polygon',
            'forex', 'currency', 'exchange rate', 'commodities', 'futures'
        ],
        'risk_analysis': [
            'risk', 'credit', 'fraud', 'compliance', 'kyc', 'aml', 'audit',
            'portfolio', 'investment', 'analysis', 'valuation', 'assessment'
        ],
        'banking': [
            'bank', 'account', 'balance', 'deposit', 'withdrawal', 'loan',
            'mortgage', 'credit card', 'debit', 'savings', 'checking'
        ],
        'insurance': [
            'insurance', 'policy', 'claim', 'coverage', 'premium', 'actuarial'
        ]
    }
    
    # Classify servers by finance category
    def classify_finance_category(row):
        text = ' '.join(filter(None, [
            str(row.get('name', '')),
            str(row.get('description', '')),
            str(row.get('canonical_name', '')),
            ' '.join(row.get('topics', []) if row.get('topics') else [])
        ])).lower()
        
        categories = []
        for category, keywords in finance_keywords.items():
            if any(keyword in text for keyword in keywords):
                categories.append(category)
        
        return categories
    
    # Apply classification
    df['finance_categories'] = df.apply(classify_finance_category, axis=1)
    df['has_finance_category'] = df['finance_categories'].apply(lambda x: len(x) > 0)
    
    # Autonomy level classification
    def classify_autonomy_level(row):
        text = ' '.join(filter(None, [
            str(row.get('name', '')),
            str(row.get('description', '')),
            str(row.get('readme_content', ''))[:500]  # First 500 chars only
        ])).lower()
        
        execution_keywords = ['execute', 'send', 'transfer', 'buy', 'sell', 'trade', 'place order']
        information_keywords = ['fetch', 'get', 'retrieve', 'search', 'query', 'read', 'view']
        agent_keywords = ['agent', 'autonomous', 'ai', 'llm', 'claude', 'gpt']
        
        levels = []
        if any(keyword in text for keyword in execution_keywords):
            levels.append('execution')
        if any(keyword in text for keyword in information_keywords):
            levels.append('information_gathering')
        if any(keyword in text for keyword in agent_keywords):
            levels.append('agent_interaction')
        
        return levels
    
    # Apply autonomy classification to finance servers
    finance_df = df[df['has_finance_category']].copy()
    if len(finance_df) > 0:
        finance_df['autonomy_levels'] = finance_df.apply(classify_autonomy_level, axis=1)
        finance_df['has_execution'] = finance_df['autonomy_levels'].apply(lambda x: 'execution' in x)
        finance_df['has_information'] = finance_df['autonomy_levels'].apply(lambda x: 'information_gathering' in x)
        finance_df['has_agent'] = finance_df['autonomy_levels'].apply(lambda x: 'agent_interaction' in x)
    
    return df, finance_df

def create_graph_1_mcp_growth_trends(df, finance_df):
    """Graph 1: MCP servers creation and usage over time"""
    st.header("📈 Graph 1: MCP Server Creation and Usage Trends")
    
    # Filter data with dates
    df_with_dates = df[df['created_at'].notna()].copy()
    finance_with_dates = finance_df[finance_df['created_at'].notna()].copy()
    
    if len(df_with_dates) == 0:
        st.warning("No creation date data available")
        return
    
    # Create monthly aggregations
    df_with_dates['month'] = df_with_dates['created_at'].dt.to_period('M')
    finance_with_dates['month'] = finance_with_dates['created_at'].dt.to_period('M')
    
    # Overall monthly counts
    monthly_all = df_with_dates.groupby('month').size().reset_index(name='count')
    monthly_all['month_str'] = monthly_all['month'].astype(str)
    monthly_all['cumulative'] = monthly_all['count'].cumsum()
    monthly_all['type'] = 'All MCP Servers'
    
    # Finance monthly counts
    monthly_finance = finance_with_dates.groupby('month').size().reset_index(name='count')
    monthly_finance['month_str'] = monthly_finance['month'].astype(str)
    monthly_finance['cumulative'] = monthly_finance['count'].cumsum()
    monthly_finance['type'] = 'Finance Servers'
    
    # Payment execution servers
    payment_servers = finance_with_dates[
        finance_with_dates['finance_categories'].apply(
            lambda x: 'payment_execution' in x if x else False
        )
    ]
    if len(payment_servers) > 0:
        monthly_payments = payment_servers.groupby('month').size().reset_index(name='count')
        monthly_payments['month_str'] = monthly_payments['month'].astype(str)
        monthly_payments['cumulative'] = monthly_payments['count'].cumsum()
        monthly_payments['type'] = 'Payment Execution'
    else:
        monthly_payments = pd.DataFrame(columns=['month_str', 'cumulative', 'type'])
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Cumulative Server Growth Over Time',
            'Monthly New Servers',
            'Finance vs Non-Finance Growth',
            'Payment Execution Servers Focus'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot 1: Cumulative growth
    fig.add_trace(
        go.Scatter(
            x=monthly_all['month_str'],
            y=monthly_all['cumulative'],
            mode='lines+markers',
            name='All Servers',
            line=dict(color='#1f77b4', width=3)
        ), row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=monthly_finance['month_str'],
            y=monthly_finance['cumulative'],
            mode='lines+markers',
            name='Finance Servers',
            line=dict(color='#ff7f0e', width=3)
        ), row=1, col=1
    )
    
    # Plot 2: Monthly new servers
    fig.add_trace(
        go.Bar(
            x=monthly_all['month_str'],
            y=monthly_all['count'],
            name='New Servers (Monthly)',
            marker_color='#2ca02c',
            showlegend=False
        ), row=1, col=2
    )
    
    # Plot 3: Finance breakdown
    if len(monthly_finance) > 0:
        fig.add_trace(
            go.Bar(
                x=monthly_finance['month_str'],
                y=monthly_finance['count'],
                name='Finance Servers',
                marker_color='#ff7f0e',
                showlegend=False
            ), row=2, col=1
        )
    
    # Plot 4: Payment execution focus
    if len(monthly_payments) > 0:
        fig.add_trace(
            go.Scatter(
                x=monthly_payments['month_str'],
                y=monthly_payments['cumulative'],
                mode='lines+markers',
                name='Payment Execution',
                line=dict(color='#d62728', width=3),
                showlegend=False
            ), row=2, col=2
        )
    
    fig.update_layout(
        height=800,
        title_text="MCP Server Growth Analysis - Addressing RQ2 & RQ3",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total MCP Servers", len(df))
    with col2:
        st.metric("Finance Servers", len(finance_df))
    with col3:
        payment_count = len(payment_servers) if len(payment_servers) > 0 else 0
        st.metric("Payment Execution Servers", payment_count)

def create_graph_2_finance_tool_trends(finance_df):
    """Graph 2: Finance tool availability trends"""
    st.header("💰 Graph 2: Finance Tool Availability Trends")
    
    if len(finance_df) == 0:
        st.warning("No finance servers found")
        return
    
    # Category distribution
    col1, col2 = st.columns(2)
    
    with col1:
        # Finance categories pie chart
        category_counts = {}
        for _, row in finance_df.iterrows():
            for category in row['finance_categories']:
                category_counts[category] = category_counts.get(category, 0) + 1
        
        if category_counts:
            fig = px.pie(
                values=list(category_counts.values()),
                names=list(category_counts.keys()),
                title="Finance Tool Categories Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Autonomy levels
        autonomy_counts = {
            'Execution Capable': len(finance_df[finance_df.get('has_execution', False)]),
            'Information Gathering': len(finance_df[finance_df.get('has_information', False)]),
            'Agent Interaction': len(finance_df[finance_df.get('has_agent', False)])
        }
        
        fig = px.bar(
            x=list(autonomy_counts.keys()),
            y=list(autonomy_counts.values()),
            title="Autonomy Levels in Finance Tools",
            color=list(autonomy_counts.values()),
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Growth trends by category
    finance_with_dates = finance_df[finance_df['created_at'].notna()].copy()
    if len(finance_with_dates) > 0:
        finance_with_dates['month'] = finance_with_dates['created_at'].dt.to_period('M')
        
        # Create category trends
        category_trends = {}
        for category in ['payment_execution', 'market_data', 'risk_analysis', 'banking']:
            cat_servers = finance_with_dates[
                finance_with_dates['finance_categories'].apply(
                    lambda x: category in x if x else False
                )
            ]
            if len(cat_servers) > 0:
                monthly = cat_servers.groupby('month').size().reset_index(name='count')
                monthly['month_str'] = monthly['month'].astype(str)
                monthly['cumulative'] = monthly['count'].cumsum()
                category_trends[category] = monthly
        
        if category_trends:
            fig = go.Figure()
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            for i, (category, data) in enumerate(category_trends.items()):
                fig.add_trace(go.Scatter(
                    x=data['month_str'],
                    y=data['cumulative'],
                    mode='lines+markers',
                    name=category.replace('_', ' ').title(),
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
            
            fig.update_layout(
                title="Finance Category Growth Trends Over Time",
                xaxis_title="Month",
                yaxis_title="Cumulative Servers",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

def create_table_1_payment_tools(finance_df):
    """Table 1: Overview of all tools relevant to automatic payments"""
    st.header("💳 Table 1: Automatic Payment Tools Overview")
    
    # Filter for payment-related tools
    payment_tools = finance_df[
        finance_df['finance_categories'].apply(
            lambda x: 'payment_execution' in x if x else False
        )
    ].copy()
    
    if len(payment_tools) == 0:
        st.warning("No payment execution tools found in the dataset")
        return
    
    # Prepare display data
    display_cols = []
    for col in ['canonical_name', 'description', 'primary_source', 'language', 
               'stargazers_count', 'use_count', 'created_at', 'autonomy_levels']:
        if col in payment_tools.columns:
            display_cols.append(col)
    
    # Sort by popularity (stars or use count)
    if 'stargazers_count' in payment_tools.columns:
        payment_tools = payment_tools.sort_values('stargazers_count', ascending=False, na_position='last')
    elif 'use_count' in payment_tools.columns:
        payment_tools = payment_tools.sort_values('use_count', ascending=False, na_position='last')
    
    # Format the table
    display_df = payment_tools[display_cols].copy()
    
    # Clean up autonomy levels for display
    if 'autonomy_levels' in display_df.columns:
        display_df['autonomy_levels'] = display_df['autonomy_levels'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else str(x)
        )
    
    st.dataframe(
        display_df.head(20),
        use_container_width=True,
        hide_index=True
    )
    
    st.info(f"Showing top 20 of {len(payment_tools)} payment execution tools found")

def create_table_2_all_servers_overview(df, finance_df):
    """Table 2: Overview table of all MCP servers and tools"""
    st.header("🗂️ Table 2: All MCP Servers Overview")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sector_filter = st.selectbox(
            "Filter by Sector",
            options=["All", "Finance Only", "Non-Finance"],
            index=0
        )
    
    with col2:
        source_filter = st.multiselect(
            "Data Source",
            options=df['primary_source'].unique() if 'primary_source' in df.columns else [],
            default=[]
        )
    
    with col3:
        language_filter = st.multiselect(
            "Language",
            options=sorted(df['language'].dropna().unique()) if 'language' in df.columns else [],
            default=[]
        )
    
    with col4:
        min_stars = st.number_input(
            "Min Stars",
            min_value=0,
            value=0,
            step=1
        )
    
    # Apply filters
    filtered_df = df.copy()
    
    if sector_filter == "Finance Only":
        # Use the finance classification from our analysis
        finance_ids = set(finance_df['id']) if len(finance_df) > 0 else set()
        filtered_df = filtered_df[filtered_df['id'].isin(finance_ids)]
    elif sector_filter == "Non-Finance":
        finance_ids = set(finance_df['id']) if len(finance_df) > 0 else set()
        filtered_df = filtered_df[~filtered_df['id'].isin(finance_ids)]
    
    if source_filter:
        filtered_df = filtered_df[filtered_df['primary_source'].isin(source_filter)]
    
    if language_filter:
        filtered_df = filtered_df[filtered_df['language'].isin(language_filter)]
    
    if min_stars > 0:
        filtered_df = filtered_df[
            (filtered_df['stargazers_count'].fillna(0) >= min_stars) |
            (filtered_df['use_count'].fillna(0) >= min_stars)
        ]
    
    # Display results
    st.write(f"Found {len(filtered_df):,} servers matching filters")
    
    # Prepare display columns
    display_cols = []
    for col in ['canonical_name', 'description', 'primary_source', 'language', 
               'stargazers_count', 'use_count', 'created_at']:
        if col in filtered_df.columns:
            display_cols.append(col)
    
    # Sort by popularity
    if 'stargazers_count' in filtered_df.columns:
        filtered_df = filtered_df.sort_values('stargazers_count', ascending=False, na_position='last')
    elif 'use_count' in filtered_df.columns:
        filtered_df = filtered_df.sort_values('use_count', ascending=False, na_position='last')
    
    if len(filtered_df) > 0:
        st.dataframe(
            filtered_df[display_cols].head(100),  # Limit for performance
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No servers match the selected filters")

def main():
    """Main dashboard application"""
    st.title("💰 MCP Finance Tools Monitoring Dashboard")
    st.markdown("""
    **Research Questions Addressed:**
    1. Which tools are available for AI power-users in finance currently?
    2. What is the uptake for these tools over time?
    3. How many new tools are made available over time?
    """)
    
    # Load data
    with st.spinner("Loading and analyzing MCP server data..."):
        result = load_and_analyze_finance_data()
        if isinstance(result, tuple):
            df, finance_df = result
        else:
            df = result
            finance_df = pd.DataFrame()
    
    if df.empty:
        st.error("No data available. Please run the unified data processor first.")
        return
    
    # Sidebar with key metrics
    with st.sidebar:
        st.header("📊 Key Metrics")
        st.metric("Total MCP Servers", f"{len(df):,}")
        st.metric("Finance Servers", f"{len(finance_df):,}")
        
        if len(finance_df) > 0:
            payment_servers = finance_df[
                finance_df['finance_categories'].apply(
                    lambda x: 'payment_execution' in x if x else False
                )
            ]
            st.metric("Payment Tools", f"{len(payment_servers):,}")
        
        st.markdown("---")
        st.markdown("**Navigation:**")
        view = st.radio(
            "Select View",
            ["📈 Growth Trends", "💰 Finance Tools", "💳 Payment Tools", "🗂️ All Servers"],
            index=0
        )
    
    # Main content based on navigation
    if view == "📈 Growth Trends":
        create_graph_1_mcp_growth_trends(df, finance_df)
        create_graph_2_finance_tool_trends(finance_df)
    
    elif view == "💰 Finance Tools":
        create_graph_2_finance_tool_trends(finance_df)
    
    elif view == "💳 Payment Tools":
        create_table_1_payment_tools(finance_df)
    
    elif view == "🗂️ All Servers":
        create_table_2_all_servers_overview(df, finance_df)
    
    # Footer with research context
    st.markdown("---")
    st.markdown("""
    **Dashboard Purpose:** Monitor MCP server ecosystem growth, tool availability, and finance-specific capabilities 
    to answer key research questions about AI tool proliferation in financial systems.
    
    **Classification:** Servers classified by finance sectors (payment execution, market data, risk analysis, banking, insurance), 
    autonomy levels (information gathering, execution capabilities, agent interactions), and consequentiality for financial system impact.
    """)

if __name__ == "__main__":
    main()
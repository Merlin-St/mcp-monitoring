#!/usr/bin/env python3
"""
Unified MCP Server Dashboard

A comprehensive Streamlit dashboard for monitoring and analyzing MCP servers
from all data sources (Smithery, GitHub, Official list).
"""

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path
from naics_classification_config import NAICS_SECTORS, NAICS_SUBSECTORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="MCP Server Monitoring Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
    }
    .finance-highlight {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

def get_data_file_path(use_filtered=False):
    """
    Get the appropriate data file path based on command line arguments or parameter.
    
    Args:
        use_filtered (bool): Use filtered dataset instead of full dataset
        
    Returns:
        tuple: (data_file_path, summary_file_path)
    """
    # Check command line arguments for --filtered flag
    if '--filtered' in sys.argv:
        use_filtered = True
    
    if use_filtered:
        data_file = Path("data_unified_filtered.json")
        summary_file = Path("data_unified_filtered_summary.json")
    else:
        data_file = Path("data_unified.json") 
        summary_file = Path("data_unified_summary.json")
    
    return data_file, summary_file

@st.cache_data
def load_unified_data(use_filtered=False):
    """
    Load unified MCP server data with support for filtered dataset.
    
    Args:
        use_filtered (bool): Use filtered dataset instead of full dataset
        
    Returns:
        pd.DataFrame: Loaded and processed MCP server data
    """
    data_file, summary_file = get_data_file_path(use_filtered)
    
    if not data_file.exists():
        dataset_type = "filtered" if use_filtered else "full"
        st.error(f"Unified {dataset_type} data file not found: {data_file}")
        st.info("Please run data_unified_mcp_data_processor.py first.")
        return pd.DataFrame()
    
    try:
        # Load data
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} servers from {data_file}")
        
        # Convert dates if present
        for col in ['created_at', 'updated_at']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data from {data_file}: {str(e)}")
        return pd.DataFrame()

def load_summary_data(use_filtered=False):
    """
    Load summary statistics for the dataset.
    
    Args:
        use_filtered (bool): Use filtered dataset summary
        
    Returns:
        dict: Summary statistics
    """
    _, summary_file = get_data_file_path(use_filtered)
    
    if not summary_file.exists():
        return {}
    
    try:
        with open(summary_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading summary from {summary_file}: {str(e)}")
        return {}

def display_data_info():
    """Display information about which dataset is being used."""
    use_filtered = '--filtered' in sys.argv
    
    if use_filtered:
        st.info("📊 Using **filtered dataset** (data_unified_filtered.json)")
        st.caption("To use full dataset, remove --filtered flag")
    else:
        st.info("📊 Using **full dataset** (data_unified.json)")  
        st.caption("To use filtered dataset, add --filtered flag")
    
    return use_filtered

@st.cache_data
def load_unified_data_with_progress():
    """Load the unified MCP server data with progress tracking and filtering support"""
    use_filtered = display_data_info()
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Loading MCP server data...")
        progress_bar.progress(25)
        
        # Use shared data loader
        df = load_unified_data(use_filtered)
        
        if df.empty:
            return pd.DataFrame(), {}
        
        status_text.text("Loading summary data...")
        progress_bar.progress(75)
        
        summary = load_summary_data(use_filtered)
        
        progress_bar.progress(100)
        status_text.text(f"Successfully loaded {len(df):,} servers!")
        
        # Clear progress indicators after a short delay
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return df, summary
        
    except Exception as e:
        logger.error(f"Error loading unified data: {e}")
        st.error(f"Unexpected error loading data: {e}")
        return pd.DataFrame(), {}
    finally:
        # Always clean up progress indicators
        try:
            progress_bar.empty()
            status_text.empty()
        except:
            pass

def display_overview_metrics(df, summary):
    """Display key metrics overview"""
    st.header("📊 Overview Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total MCP Servers",
            f"{len(df):,}",
            help="Total number of unique MCP servers across all sources"
        )
    
    with col2:
        finance_count = len(df[df.get('is_finance_related', False) == True])
        finance_pct = (finance_count / len(df) * 100) if len(df) > 0 else 0
        st.metric(
            "Finance-Related Servers",
            f"{finance_count:,}",
            f"{finance_pct:.1f}% of total"
        )
    
    with col3:
        multi_source = len(df[df['data_sources'].apply(lambda x: len(x) > 1)])
        st.metric(
            "Multi-Source Servers",
            f"{multi_source:,}",
            help="Servers found in multiple data sources"
        )
    
    with col4:
        github_with_stars = len(df[(df['stargazers_count'].notna()) & (df['stargazers_count'] > 0)])
        st.metric(
            "GitHub Servers with Stars",
            f"{github_with_stars:,}",
            help="GitHub repositories with at least 1 star"
        )

def display_source_distribution(df, summary):
    """Display data source distribution"""
    st.header("📈 Data Source Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Source coverage pie chart
        if 'source_coverage' in summary:
            source_data = summary['source_coverage']
            fig = px.pie(
                values=list(source_data.values()),
                names=list(source_data.keys()),
                title="Servers by Data Source",
                color_discrete_map={
                    'smithery': '#FF6B6B',
                    'github': '#4ECDC4', 
                    'official': '#45B7D1'
                }
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Primary source distribution
        if 'primary_source_distribution' in summary:
            primary_data = summary['primary_source_distribution']
            fig = px.bar(
                x=list(primary_data.keys()),
                y=list(primary_data.values()),
                title="Primary Data Source Distribution",
                color=list(primary_data.keys()),
                color_discrete_map={
                    'smithery': '#FF6B6B',
                    'github': '#4ECDC4',
                    'official': '#45B7D1'
                }
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

def display_growth_trends(df):
    """Display creation and growth trends"""
    st.header("📈 MCP Server Growth Trends")
    
    # Filter data with creation dates
    df_with_dates = df[df['created_at'].notna()].copy()
    
    if len(df_with_dates) == 0:
        st.warning("No creation date data available for trend analysis")
        return
    
    # Create monthly growth chart
    df_with_dates['month'] = df_with_dates['created_at'].dt.to_period('M')
    monthly_counts = df_with_dates.groupby(['month', 'is_finance_related']).size().reset_index(name='count')
    monthly_counts['month_str'] = monthly_counts['month'].astype(str)
    
    # Separate finance and non-finance
    finance_monthly = monthly_counts[monthly_counts['is_finance_related'] == True]
    non_finance_monthly = monthly_counts[monthly_counts['is_finance_related'] == False]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('All MCP Servers Creation Over Time', 'Finance vs Non-Finance Servers'),
        vertical_spacing=0.1
    )
    
    # All servers trend
    all_monthly = df_with_dates.groupby('month').size().reset_index(name='count')
    all_monthly['month_str'] = all_monthly['month'].astype(str)
    all_monthly['cumulative'] = all_monthly['count'].cumsum()
    
    fig.add_trace(
        go.Scatter(
            x=all_monthly['month_str'],
            y=all_monthly['cumulative'],
            mode='lines+markers',
            name='Cumulative Servers',
            line=dict(color='#45B7D1', width=3)
        ),
        row=1, col=1
    )
    
    # Finance vs non-finance
    if len(finance_monthly) > 0:
        fig.add_trace(
            go.Bar(
                x=finance_monthly['month_str'],
                y=finance_monthly['count'],
                name='Finance-Related',
                marker_color='#FFC107'
            ),
            row=2, col=1
        )
    
    if len(non_finance_monthly) > 0:
        fig.add_trace(
            go.Bar(
                x=non_finance_monthly['month_str'],
                y=non_finance_monthly['count'],
                name='Non-Finance',
                marker_color='#4ECDC4'
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="MCP Server Creation Trends"
    )
    
    st.plotly_chart(fig, use_container_width=True)



def display_naics_classification(df):
    """Display NAICS sectoral classification of all MCP servers"""
    st.header("🏢 NAICS Sectoral Classification")
    st.markdown("All MCP servers classified by North American Industry Classification System (NAICS) sectors.")
    
    # Display embed visualization at the top
    embed_file = Path("embed_visualization.html")
    if embed_file.exists():
        st.subheader("🎯 Interactive Embedding Visualization")
        st.markdown("Explore MCP servers in embedding space - similar servers cluster together:")
        
        # Read and display the HTML file
        with open(embed_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Scale down the visualization by wrapping in a div with transform
        scaled_html = f"""
        <div style="transform: scale(0.8); transform-origin: top left; width: 125%; height: 125%;">
            {html_content}
        </div>
        """
        st.components.v1.html(scaled_html, height=800, scrolling=False)
        
        st.markdown("---")
    else:
        st.info("💡 Run `python embed_generate.py` to generate interactive embedding visualization")
        st.markdown("---")
    
    # Create a list to hold all server-sector relationships
    classification_data = []
    
    for _, server in df.iterrows():
        server_info = {
            'Server Name': server.get('canonical_name', server.get('name', 'Unknown')),
            'Description': server.get('canonical_description', server.get('description', ''))[:100] + '...' if server.get('canonical_description', server.get('description', '')) else '',
            'Primary Source': server.get('primary_source', 'Unknown'),
            'Usage Count': server.get('use_count', 0) or 0,
            'Owner/Creator': server.get('owner_login', server.get('owner_name', 'Unknown')),
            'Stars': server.get('stargazers_count', 0) or 0,
            'Sectors': [],
            'Matched Keywords': []
        }
        
        # Check each NAICS sector
        for sector_code, sector_name in NAICS_SECTORS.items():
            if sector_code == 99:  # Skip "Unclassified"
                continue
                
            is_sector_col = f'is_sector_{sector_code}'
            keywords_col = f'sector_{sector_code}_keywords'
            
            if server.get(is_sector_col, False):
                server_info['Sectors'].append(f"{sector_code}: {sector_name}")
                keywords = server.get(keywords_col, [])
                if keywords:
                    server_info['Matched Keywords'].extend([f"{sector_code}: {', '.join(keywords[:3])}" + ("..." if len(keywords) > 3 else "")])
        
        # Only include servers that match at least one sector
        if server_info['Sectors']:
            classification_data.append(server_info)
    
    if not classification_data:
        st.warning("No servers found with NAICS sector classifications")
        return
    
    # Convert to DataFrame for display
    display_df = pd.DataFrame(classification_data)
    
    # Summary statistics
    total_all_servers = len(df)  # Total servers in dataset
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Classified Servers", len(display_df))
    with col2:
        total_sectors = sum(len(sectors) for sectors in display_df['Sectors'])
        st.metric("Total Sector Assignments", total_sectors)
    with col3:
        classification_pct = (len(display_df) / total_all_servers * 100) if total_all_servers > 0 else 0
        st.metric("% of All Servers Classified", f"{classification_pct:.1f}%")
    
    # Sector distribution chart (as percentages)
    st.subheader("📊 Sector Distribution (% of Total Servers)")
    sector_counts = {}
    for sectors_list in display_df['Sectors']:
        for sector in sectors_list:
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
    
    if sector_counts:
        sector_df = pd.DataFrame(list(sector_counts.items()), columns=['Sector', 'Count'])
        sector_df['Percentage'] = (sector_df['Count'] / total_all_servers * 100).round(2)
        sector_df = sector_df.sort_values('Percentage', ascending=True)
        
        fig = px.bar(
            sector_df, 
            x='Percentage', 
            y='Sector',
            orientation='h',
            title=f"Percentage of All Servers by NAICS Sector (of {total_all_servers:,} total servers)",
            color='Percentage',
            color_continuous_scale='viridis',
            text='Percentage'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(
            height=max(400, len(sector_df) * 25),
            xaxis_title="Percentage of Total Servers (%)",
            showlegend=False,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Filters
    st.subheader("🔍 Filter Servers")
    col1, col2 = st.columns(2)
    
    with col1:
        # Sector filter
        all_sectors = list(set([sector for sectors_list in display_df['Sectors'] for sector in sectors_list]))
        selected_sectors = st.multiselect("Filter by Sectors", options=all_sectors)
    
    with col2:
        # Source filter
        sources = display_df['Primary Source'].unique()
        selected_sources = st.multiselect("Filter by Source", options=sources)
    
    # Apply filters
    filtered_df = display_df.copy()
    if selected_sectors:
        filtered_df = filtered_df[filtered_df['Sectors'].apply(
            lambda x: any(sector in x for sector in selected_sectors)
        )]
    if selected_sources:
        filtered_df = filtered_df[filtered_df['Primary Source'].isin(selected_sources)]
    
    # Prepare display dataframe
    display_table = filtered_df.copy()
    display_table['Sectors'] = display_table['Sectors'].apply(lambda x: ' | '.join(x))
    display_table['Matched Keywords'] = display_table['Matched Keywords'].apply(lambda x: ' | '.join(x))
    
    # Show the table
    st.subheader(f"📋 MCP Servers NAICS Classification ({len(display_table)} servers)")
    st.dataframe(
        display_table[['Server Name', 'Description', 'Sectors', 'Matched Keywords', 'Primary Source', 'Usage Count', 'Owner/Creator', 'Stars']],
        use_container_width=True,
        hide_index=True
    )

def display_server_explorer(df):
    """Interactive server explorer"""
    st.header("🔍 Server Explorer")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        source_filter = st.multiselect(
            "Filter by Data Source",
            options=df['primary_source'].unique() if 'primary_source' in df.columns else [],
            default=[]
        )
    
    with col2:
        finance_filter = st.selectbox(
            "Finance-Related",
            options=["All", "Finance Only", "Non-Finance Only"],
            index=0
        )
    
    with col3:
        language_filter = st.multiselect(
            "Filter by Language",
            options=sorted(df['language'].dropna().unique()) if 'language' in df.columns else [],
            default=[]
        )
    
    # Apply filters
    filtered_df = df.copy()
    
    if source_filter:
        filtered_df = filtered_df[filtered_df['primary_source'].isin(source_filter)]
    
    if finance_filter == "Finance Only":
        filtered_df = filtered_df[filtered_df.get('is_finance_related', False) == True]
    elif finance_filter == "Non-Finance Only":
        filtered_df = filtered_df[filtered_df.get('is_finance_related', False) == False]
    
    if language_filter:
        filtered_df = filtered_df[filtered_df['language'].isin(language_filter)]
    
    # Display results
    st.write(f"Found {len(filtered_df):,} servers matching filters")
    
    if len(filtered_df) > 0:
        # Select columns to display
        display_cols = []
        for col in ['canonical_name', 'description', 'primary_source', 'language', 'stargazers_count', 'use_count', 'is_finance_related']:
            if col in filtered_df.columns:
                display_cols.append(col)
        
        if display_cols:
            st.dataframe(
                filtered_df[display_cols].head(100),  # Limit for performance
                use_container_width=True,
                hide_index=True
            )
    else:
        st.info("No servers match the selected filters")

def display_finance_analysis(df):
    """Display comprehensive analysis of Finance and Insurance (NAICS 52) MCP servers"""
    st.header("💰 Finance MCP Servers Analysis")
    st.markdown("**Finance and Insurance (NAICS Sector 52) - Comprehensive Analysis**")
    
    # Filter to sector 52 servers only
    finance_df = df[df.get('is_sector_52', False) == True].copy()
    
    if len(finance_df) == 0:
        st.warning("No Finance and Insurance (Sector 52) servers found in the dataset")
        return
    
    # Display embed visualization at the top
    embed_file = Path("embed_sector_52_visualization.html")
    if embed_file.exists():
        st.subheader("🎯 Interactive Finance Sector Embedding Visualization")
        st.markdown("Explore Finance and Insurance MCP servers in embedding space - similar financial services cluster together:")
        
        # Read and display the HTML file
        with open(embed_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Scale down the visualization by wrapping in a div with transform
        scaled_html = f"""
        <div style="transform: scale(0.8); transform-origin: top left; width: 125%; height: 125%;">
            {html_content}
        </div>
        """
        st.components.v1.html(scaled_html, height=800, scrolling=False)
        
        st.markdown("---")
    else:
        st.info("💡 Run `python embed_generate.py --filter sector_52` to generate interactive finance sector embedding visualization")
        st.markdown("---")
    
    # Finance sector overview metrics
    st.subheader("📊 Finance Sector Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Finance Servers", len(finance_df))
    
    with col2:
        total_servers = len(df)
        finance_pct = (len(finance_df) / total_servers * 100) if total_servers > 0 else 0
        st.metric("% of Total Servers", f"{finance_pct:.1f}%")
    
    with col3:
        avg_tools_finance = finance_df['tools_count'].mean() if 'tools_count' in finance_df.columns else 0
        st.metric("Avg Tools per Server", f"{avg_tools_finance:.1f}")
    
    with col4:
        total_stars = finance_df['stargazers_count'].sum() if 'stargazers_count' in finance_df.columns else 0
        st.metric("Total GitHub Stars", f"{total_stars:,}")
    
    # Subsector 52xx Analysis
    st.subheader("🏦 Finance Subsector Distribution (52xx)")
    
    # Extract subsector data for sector 52
    subsector_data = {}
    for _, server in finance_df.iterrows():
        # Look for subsector columns (52xx format)
        for col in server.index:
            if col.startswith('is_subsector_52') and server[col]:
                subsector_code = col.replace('is_subsector_', '')
                # Get subsector name from NAICS config if available
                try:
                    subsector_num = int(subsector_code)
                    subsector_name = f"{subsector_code}: {NAICS_SUBSECTORS.get(subsector_num, 'Unknown')}"
                except:
                    subsector_name = subsector_code
                    
                if subsector_name not in subsector_data:
                    subsector_data[subsector_name] = 0
                subsector_data[subsector_name] += 1
    
    if subsector_data:
        # Create subsector distribution chart
        subsector_df = pd.DataFrame(list(subsector_data.items()), columns=['Subsector', 'Count'])
        subsector_df['Percentage'] = (subsector_df['Count'] / len(finance_df) * 100).round(2)
        subsector_df = subsector_df.sort_values('Count', ascending=True)
        
        fig = px.bar(
            subsector_df, 
            x='Count', 
            y='Subsector',
            orientation='h',
            title=f"Finance Subsector Distribution ({len(finance_df):,} servers)",
            color='Count',
            color_continuous_scale='viridis',
            text='Count'
        )
        fig.update_traces(texttemplate='%{text} (%{customdata:.1f}%)', textposition='outside', customdata=subsector_df['Percentage'])
        fig.update_layout(
            height=max(400, len(subsector_df) * 30),
            xaxis_title="Number of Servers",
            showlegend=False,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No subsector classification data available for finance servers")
    
    # Top Finance Servers Table
    st.subheader("⭐ Top Finance MCP Servers")
    
    # Calculate ranking score combining stars, tools count, and usage
    finance_display = finance_df.copy()
    finance_display['ranking_score'] = (
        (finance_display.get('stargazers_count', 0) or 0) * 0.4 +
        (finance_display.get('tools_count', 0) or 0) * 0.4 +
        (finance_display.get('use_count', 0) or 0) * 0.2
    )
    
    # Get top 10 servers
    top_finance = finance_display.nlargest(10, 'ranking_score')
    
    # Prepare display table
    display_data = []
    for _, server in top_finance.iterrows():
        # Get subsectors for this server
        server_subsectors = []
        for col in server.index:
            if col.startswith('is_subsector_52') and server[col]:
                subsector_code = col.replace('is_subsector_', '')
                try:
                    subsector_num = int(subsector_code)
                    subsector_name = f"{subsector_code}: {NAICS_SUBSECTORS.get(subsector_num, 'Unknown')}"
                except:
                    subsector_name = subsector_code
                server_subsectors.append(subsector_name)
        
        display_data.append({
            'Server Name': server.get('canonical_name', server.get('name', 'Unknown')),
            'Description': (server.get('canonical_description', server.get('description', ''))[:80] + '...' 
                          if server.get('canonical_description', server.get('description', '')) else ''),
            'Subsectors': ', '.join(server_subsectors) if server_subsectors else 'None',
            'Tools': server.get('tools_count', 0) or 0,
            'Stars': server.get('stargazers_count', 0) or 0,
            'Usage': server.get('use_count', 0) or 0,
            'Source': server.get('primary_source', 'Unknown')
        })
    
    if display_data:
        display_df = pd.DataFrame(display_data)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No finance servers found for ranking")
    
    # Tools Analysis for Finance Sector
    st.subheader("🔧 Tools Analysis - Finance Sector")
    
    if 'tools_by_access' in finance_df.columns:
        # Aggregate tools by access level
        total_tools_by_access = {'read': 0, 'write': 0, 'execute': 0}
        
        for _, server in finance_df.iterrows():
            tools_access = server.get('tools_by_access', {})
            if isinstance(tools_access, dict):
                for access_type in total_tools_by_access.keys():
                    total_tools_by_access[access_type] += tools_access.get(access_type, 0)
        
        # Create tools distribution chart
        col1, col2 = st.columns(2)
        
        with col1:
            # Tools by access level pie chart
            if sum(total_tools_by_access.values()) > 0:
                fig = px.pie(
                    values=list(total_tools_by_access.values()),
                    names=list(total_tools_by_access.keys()),
                    title="Finance Sector Tools by Access Level",
                    color_discrete_map={
                        'read': '#4ECDC4',
                        'write': '#FFC107', 
                        'execute': '#FF6B6B'
                    }
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Tools statistics
            total_tools = sum(total_tools_by_access.values())
            servers_with_tools = len(finance_df[finance_df.get('tools_count', 0) > 0])
            
            st.metric("Total Finance Tools", f"{total_tools:,}")
            st.metric("Servers with Tools", f"{servers_with_tools:,}")
            if len(finance_df) > 0:
                st.metric("Avg Tools per Finance Server", f"{total_tools/len(finance_df):.1f}")
    
    # Finance sector specific insights
    st.subheader("💡 Finance Sector Insights")
    
    # Source distribution for finance servers
    if 'primary_source' in finance_df.columns:
        source_counts = finance_df['primary_source'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                values=source_counts.values,
                names=source_counts.index,
                title="Finance Servers by Data Source",
                color_discrete_map={
                    'smithery': '#FF6B6B',
                    'github': '#4ECDC4',
                    'official': '#45B7D1'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Key Statistics:**")
            st.write(f"• **Total Finance Servers:** {len(finance_df):,}")
            st.write(f"• **Most Common Subsector:** {list(subsector_data.keys())[0] if subsector_data else 'N/A'}")
            st.write(f"• **Average Tools per Server:** {avg_tools_finance:.1f}")
            st.write(f"• **Servers with GitHub Stars:** {len(finance_df[finance_df.get('stargazers_count', 0) > 0]):,}")


def main():
    """Main dashboard application"""
    st.title("🤖 MCP Server Monitoring Dashboard")
    st.markdown("**Comprehensive analysis of Model Context Protocol servers across all data sources**")
    
    # Debug information in expander
    with st.expander("🔧 Debug Information", expanded=False):
        st.write("**Current Working Directory:**", Path.cwd())
        st.write("**Data Files Status:**")
        data_file = Path("data_unified.json")
        summary_file = Path("data_unified_summary.json")
        st.write(f"- Main data file exists: {data_file.exists()}")
        if data_file.exists():
            st.write(f"- Main data file size: {data_file.stat().st_size / (1024*1024):.1f} MB")
        st.write(f"- Summary file exists: {summary_file.exists()}")
        
        # Test data loading
        if st.button("Test Data Loading"):
            try:
                with open(data_file, 'r') as f:
                    test_data = json.load(f)
                st.success(f"✅ Successfully loaded {len(test_data)} servers for testing")
            except Exception as e:
                st.error(f"❌ Error loading data: {e}")
    
    # Load data
    try:
        df, summary = load_unified_data_with_progress()
    except Exception as e:
        st.error(f"Critical error loading data: {e}")
        st.stop()
    
    if df.empty:
        st.error("No data available.")
        st.info("Please run the unified data processor first: `python unified_mcp_data_processor.py`")
        st.stop()
    
    # Sidebar with data info
    with st.sidebar:
        st.header("📋 Data Information")
        
        if summary:
            st.metric("Total Servers", f"{summary.get('total_servers', 0):,}")
            st.metric("Finance Servers", f"{summary.get('finance_related_servers', 0):,}")
            
            if 'processing_timestamp' in summary:
                proc_time = datetime.fromisoformat(summary['processing_timestamp'])
                st.write(f"**Last Updated:** {proc_time.strftime('%Y-%m-%d %H:%M')}")
        
        st.subheader("🎛️ Navigation")
        page = st.radio(
            "Select View",
            ["Overview", "Growth Trends", "Sector Classification", "Finance MCP Servers", "Server Explorer"],
            index=0
        )
    
    # Main content area
    if page == "Overview":
        display_overview_metrics(df, summary)
        display_source_distribution(df, summary)
    
    elif page == "Growth Trends":
        display_growth_trends(df)
    
    
    elif page == "Sector Classification":
        display_naics_classification(df)
    
    elif page == "Finance MCP Servers":
        display_finance_analysis(df)
    
    elif page == "Server Explorer":
        display_server_explorer(df)
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"**Dashboard Info:** Displaying {len(df):,} unified MCP servers from Smithery, GitHub, and Official sources. "
        f"Data last processed: {summary.get('processing_timestamp', 'Unknown')}"
    )

if __name__ == "__main__":
    main()
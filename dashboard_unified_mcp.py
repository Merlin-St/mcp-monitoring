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
from pathlib import Path

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

@st.cache_data
def load_unified_data():
    """Load the unified MCP server data with progress tracking"""
    data_file = Path("dashboard_mcp_servers_unified_filtered.json")
    summary_file = Path("dashboard_mcp_servers_unified_filtered_summary.json")
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Checking data files...")
        progress_bar.progress(10)
        
        if not data_file.exists():
            st.error(f"Unified data file not found: {data_file.absolute()}")
            st.info("Please run the unified data processor first: `python unified_mcp_data_processor.py`")
            return pd.DataFrame(), {}
        
        status_text.text(f"Loading main data file ({data_file.stat().st_size / (1024*1024):.1f} MB)...")
        progress_bar.progress(25)
        
        # Load main data with error handling
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            st.error(f"Data format error: Expected list, got {type(data)}")
            return pd.DataFrame(), {}
        
        status_text.text("Loading summary data...")
        progress_bar.progress(50)
        
        # Load summary
        summary = {}
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
        else:
            st.warning("Summary file not found - some metrics may be unavailable")
        
        status_text.text("Creating DataFrame...")
        progress_bar.progress(75)
        
        df = pd.DataFrame(data)
        
        if df.empty:
            st.error("No data found in the unified data file")
            return pd.DataFrame(), {}
        
        status_text.text("Processing date columns...")
        progress_bar.progress(90)
        
        # Convert date columns safely
        date_cols = ['created_at', 'updated_at']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Ensure required columns exist
        required_cols = ['id', 'name', 'primary_source', 'data_sources']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.warning(f"Missing expected columns: {missing_cols}")
        
        progress_bar.progress(100)
        status_text.text(f"Successfully loaded {len(df):,} servers!")
        
        # Clear progress indicators after a short delay
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        logger.info(f"Loaded {len(df)} unified MCP servers")
        return df, summary
        
    except json.JSONDecodeError as e:
        st.error(f"JSON decode error: {e}")
        st.info("The data file may be corrupted. Try regenerating it.")
        return pd.DataFrame(), {}
    except Exception as e:
        logger.error(f"Error loading unified data: {e}")
        st.error(f"Unexpected error loading data: {e}")
        st.info("Please check the logs and try again.")
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

def display_technology_analysis(df, summary):
    """Display technology and language analysis"""
    st.header("💻 Technology Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top programming languages
        if 'top_languages' in summary and summary['top_languages']:
            lang_data = summary['top_languages']
            fig = px.bar(
                x=list(lang_data.values()),
                y=list(lang_data.keys()),
                orientation='h',
                title="Top Programming Languages",
                color=list(lang_data.values()),
                color_continuous_scale='viridis'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No language data available")
    
    with col2:
        # Top topics/tags
        if 'top_topics' in summary and summary['top_topics']:
            topic_data = dict(list(summary['top_topics'].items())[:10])  # Top 10
            fig = px.bar(
                x=list(topic_data.values()),
                y=list(topic_data.keys()),
                orientation='h',
                title="Top Topics/Tags",
                color=list(topic_data.values()),
                color_continuous_scale='plasma'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No topic data available")

def display_finance_analysis(df):
    """Display finance-specific analysis"""
    st.header("💰 Finance-Related MCP Servers Analysis")
    
    finance_df = df[df.get('is_finance_related', False) == True].copy()
    
    if len(finance_df) == 0:
        st.warning("No finance-related servers found in the dataset")
        return
    
    # Finance servers by source
    col1, col2 = st.columns(2)
    
    with col1:
        finance_by_source = finance_df['primary_source'].value_counts()
        fig = px.pie(
            values=finance_by_source.values,
            names=finance_by_source.index,
            title=f"Finance Servers by Source ({len(finance_df)} total)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top finance languages
        finance_langs = finance_df['language'].value_counts().head(8)
        if len(finance_langs) > 0:
            fig = px.bar(
                x=finance_langs.values,
                y=finance_langs.index,
                orientation='h',
                title="Programming Languages in Finance Servers"
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No language data for finance servers")
    
    # Top finance servers by popularity
    st.subheader("🌟 Top Finance-Related Servers")
    
    # Sort by stars or use count
    finance_display = finance_df.copy()
    if 'stargazers_count' in finance_display.columns:
        finance_display = finance_display.sort_values('stargazers_count', ascending=False, na_position='last')
    elif 'use_count' in finance_display.columns:
        finance_display = finance_display.sort_values('use_count', ascending=False, na_position='last')
    
    # Display top finance servers
    cols_to_show = ['canonical_name', 'description', 'primary_source', 'stargazers_count', 'use_count', 'language']
    available_cols = [col for col in cols_to_show if col in finance_display.columns]
    
    if available_cols:
        st.dataframe(
            finance_display[available_cols].head(20),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("No suitable columns found for finance server display")

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

def main():
    """Main dashboard application"""
    st.title("🤖 MCP Server Monitoring Dashboard")
    st.markdown("**Comprehensive analysis of Model Context Protocol servers across all data sources**")
    
    # Debug information in expander
    with st.expander("🔧 Debug Information", expanded=False):
        st.write("**Current Working Directory:**", Path.cwd())
        st.write("**Data Files Status:**")
        data_file = Path("dashboard_mcp_servers_unified.json")
        summary_file = Path("dashboard_mcp_servers_unified_summary.json")
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
        df, summary = load_unified_data()
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
            ["Overview", "Growth Trends", "Technology Analysis", "Finance Analysis", "Server Explorer"],
            index=0
        )
    
    # Main content area
    if page == "Overview":
        display_overview_metrics(df, summary)
        display_source_distribution(df, summary)
    
    elif page == "Growth Trends":
        display_growth_trends(df)
    
    elif page == "Technology Analysis":
        display_technology_analysis(df, summary)
    
    elif page == "Finance Analysis":
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
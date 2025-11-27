from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from config.settings import DEFAULT_DATA_SOURCE
from core.data.loaders import ParquetDataLoader, SupabaseDataLoader, load_data
from ui.components.header import render_header
from ui.components.sidebar import render_sidebar

st.set_page_config(page_title="Nodal Analysis", page_icon="🗺️", layout="wide")

render_header()
render_sidebar()

st.title("🗺️ Nodal Arbitrage Assessment")
st.markdown("""
Identify the most profitable grid locations by analyzing historical volatility and price spreads across settlement points.
This tool scans available nodes to find the best opportunities for battery storage deployment.
""")

# Configuration
source = st.session_state.get('data_source', DEFAULT_DATA_SOURCE)
if source not in ('database', 'local_parquet'):
    st.error(f"Invalid data source: {source}. Must be 'database' or 'local_parquet'.")
    st.stop()
st.info(f"Current Data Source: **{source}**")

if source == 'local_parquet':
    st.success("✅ Local Parquet Optimized for Nodal Scanning (Powered by Polars)")


@st.cache_data(ttl=3600, show_spinner=False)
def load_node_data_cached(source: Literal['database', 'local_parquet'], node: str) -> pd.DataFrame:
    """
    Load and cache node data for 1 hour.
    Uses polars for fast parquet I/O when source is 'local_parquet'.
    """
    return load_data(source=source, node=node)


def analyze_single_node(source: Literal['database', 'local_parquet'], node: str) -> Optional[dict]:
    """
    Analyze a single node and return metrics.
    Returns None if node has no data or error occurs.
    """
    try:
        df = load_node_data_cached(source, node)

        if df.empty:
            return None

        # Calculate Metrics
        volatility = df['price_mwh_rt'].std()
        avg_spread = df['price_spread'].mean()

        # Revenue Score: Sum of spreads > $20/MWh
        profitable_spreads = df[df['price_spread'] > 20]['price_spread']
        revenue_score = profitable_spreads.sum()

        return {
            'Node': node,
            'Volatility ($/MWh)': round(volatility, 2),
            'Avg Spread ($/MWh)': round(avg_spread, 2),
            'Revenue Score': round(revenue_score, 2),
            'Data Points': len(df)
        }
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def run_nodal_assessment(source: Literal['database', 'local_parquet']) -> tuple[pd.DataFrame, dict]:
    """
    Run nodal assessment with caching.
    Returns (results_df, node_data_cache) where node_data_cache contains data for top nodes.
    """
    # Get list of nodes
    if source == 'local_parquet':
        data_dir = Path(__file__).parent.parent / 'data'
        loader = ParquetDataLoader(data_dir)
        nodes = loader.get_available_nodes()
    elif source == 'database':
        loader = SupabaseDataLoader()
        nodes = loader.get_available_nodes()
        nodes = nodes[:50]  # Limit for database performance
    else:
        return pd.DataFrame(), {}

    if not nodes:
        return pd.DataFrame(), {}

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    results = []
    node_data_cache = {}

    # Analyze each node
    total = len(nodes)
    for i, node in enumerate(nodes):
        status_text.text(f"Analyzing node {i + 1}/{total}: {node}")

        result = analyze_single_node(source, node)
        if result:
            results.append(result)
            # Cache data for top nodes (we'll need top 3 for visualization)
            if len(results) <= 10:  # Cache top 10 to be safe
                node_data_cache[node] = load_node_data_cached(source, node)

        progress_bar.progress((i + 1) / total)

    # Clear progress indicators
    status_text.empty()
    progress_bar.empty()

    # Create DataFrame and sort by revenue score
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Revenue Score', ascending=False).reset_index(drop=True)
        results_df.index += 1  # 1-based ranking

        # Update cache with actual top nodes
        top_nodes = results_df.head(10)['Node'].tolist()
        for node in top_nodes:
            if node not in node_data_cache:
                node_data_cache[node] = load_node_data_cached(source, node)

        return results_df, node_data_cache

    return pd.DataFrame(), {}


# UI Controls
col1, col2 = st.columns([1, 4])
with col1:
    run_analysis = st.button("🔍 Run Analysis", type="primary", width="stretch")
with col2:
    if st.button("🗑️ Clear Cache", width="stretch"):
        st.cache_data.clear()
        st.success("Cache cleared! Click 'Run Analysis' to rescan.")
        st.rerun()

# Run analysis on button click or if results already cached
if run_analysis or 'nodal_results' in st.session_state:
    with st.spinner("🔄 Scanning nodes..."):
        results_df, node_data_cache = run_nodal_assessment(source)

    # Store in session state
    st.session_state['nodal_results'] = results_df
    st.session_state['node_data_cache'] = node_data_cache

    if not results_df.empty:
        # Display results
        st.subheader("🏆 Top Performing Nodes")

        # Show summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Nodes Analyzed", len(results_df))
        with col2:
            st.metric("Avg Revenue Score", f"${results_df['Revenue Score'].mean():,.0f}")
        with col3:
            st.metric("Top Node", results_df.iloc[0]['Node'])

        st.markdown("---")

        # Display styled dataframe
        st.dataframe(
            results_df.style.background_gradient(
                subset=['Revenue Score', 'Volatility ($/MWh)'],
                cmap='Greens'
            ),
            width='stretch',
            height=400
        )

        st.caption("""
        **Metric Definitions:**
        - **Volatility:** Standard deviation of Real-Time Market (RTM) prices. Higher volatility indicates more arbitrage opportunities.
        - **Avg Spread:** Average absolute difference between Day-Ahead and Real-Time prices.
        - **Revenue Score:** Sum of all price spreads > $20/MWh. Higher score = more frequent and larger arbitrage opportunities.
        - **Data Points:** Number of 15-minute intervals analyzed for the node.
        """)

        # Visual Comparison of Top 3
        st.subheader("📊 Top 3 Nodes: Price Spread Distribution")

        top_3_nodes = results_df.head(3)['Node'].tolist()

        # Use cached data if available
        dfs = []
        for node in top_3_nodes:
            if node in node_data_cache:
                d = node_data_cache[node].copy()
            else:
                d = load_node_data_cached(source, node)
            d['Node'] = node
            dfs.append(d)

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)

            # Box plot for spread distribution
            fig = px.box(
                combined,
                x='Node',
                y='price_spread',
                title="Price Spread Distribution (Higher = More Volatility)",
                labels={'price_spread': 'Price Spread ($/MWh)', 'Node': 'Settlement Point'},
                color='Node'
            )
            fig.update_layout(showlegend=False, height=500)
            st.plotly_chart(fig, width='stretch')

            # Additional metrics comparison
            st.subheader("📈 Top 3 Nodes: Detailed Comparison")

            comparison_data = []
            for node in top_3_nodes:
                if node in node_data_cache:
                    df = node_data_cache[node]
                else:
                    df = load_node_data_cached(source, node)

                # Count unique days with extreme events (>$50 spread)
                extreme_events_df = df[df['price_spread'] > 50]
                if not extreme_events_df.empty and 'timestamp' in df.columns:
                    days_with_extreme = extreme_events_df['timestamp'].dt.date.nunique()
                else:
                    days_with_extreme = 0

                comparison_data.append({
                    'Node': node,
                    'Max RT Price': f"${df['price_mwh_rt'].max():.2f}",
                    'Min RT Price': f"${df['price_mwh_rt'].min():.2f}",
                    'Max Spread': f"${df['price_spread'].max():.2f}",
                    'Days with Extreme Events (>$50 spread)': days_with_extreme,
                })

            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, width='stretch')

    else:
        st.warning("No data found. Please check your data source configuration.")
else:
    st.info("👆 Click **'Run Analysis'** to scan all available nodes and identify the best locations for battery storage.")

    # Show example of what will be analyzed
    if source == 'local_parquet':
        data_dir = Path(__file__).parent.parent / 'data'
        loader = ParquetDataLoader(data_dir)
        nodes = loader.get_available_nodes()
        if nodes:
            st.success(f"✅ Ready to analyze **{len(nodes)}** nodes from local parquet files.")
    elif source == 'database':
        loader = SupabaseDataLoader()
        nodes = loader.get_available_nodes()
        if nodes:
            st.success(
                f"✅ Ready to analyze **{min(50, len(nodes))}** nodes from database (max 50 for performance).")

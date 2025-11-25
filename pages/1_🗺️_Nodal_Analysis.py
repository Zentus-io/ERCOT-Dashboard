import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from core.data.loaders import load_data, ParquetDataLoader, SupabaseDataLoader
from ui.components.header import render_header
from ui.components.sidebar import render_sidebar
from config.settings import DEFAULT_DATA_SOURCE

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
st.info(f"Current Data Source: **{source}**")

if source == 'local_parquet':
    st.success("✅ Local Parquet Optimized for Nodal Scanning")
def run_nodal_assessment(source: str):
    """
    Run nodal assessment with progress bar.
    """
    # 1. Get List of Nodes
    nodes = []
    if source == 'local_parquet':
        # Use the optimized loader
        from pathlib import Path
        data_dir = Path(__file__).parent.parent / 'data'
        loader = ParquetDataLoader(data_dir)
        nodes = loader.get_available_nodes()
    elif source == 'database':
        loader = SupabaseDataLoader()
        nodes = loader.get_available_nodes()
        nodes = nodes[:50] # Limit for performance
    else: # CSV
        # Fallback
        pass
        
    if not nodes:
        return []
        
    results = []
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 2. Analyze each node
    for i, node in enumerate(nodes):
        status_text.text(f"Analyzing {node}...")
        try:
            # Load data for node
            df = load_data(source=source, node=node)
            
            if df.empty:
                continue
                
            # Calculate Metrics
            # 1. Volatility (Std Dev of RTM)
            volatility = df['price_mwh_rt'].std()
            
            # 2. Average Spread (Abs(DA - RT))
            avg_spread = df['price_spread'].mean()
            
            # 3. Revenue Potential (Simple approximation: Sum of top 10% spreads)
            profitable_spreads = df[df['price_spread'] > 20]['price_spread']
            revenue_score = profitable_spreads.sum()
            
            results.append({
                'Node': node,
                'Volatility ($/MWh)': volatility,
                'Avg Spread ($/MWh)': avg_spread,
                'Revenue Score': revenue_score,
                'Data Points': len(df)
            })
            
        except Exception as e:
            pass
        
        # Update progress
        progress_bar.progress((i + 1) / len(nodes))
            
    # Clear progress
    status_text.empty()
    progress_bar.empty()
    
    return results

# Auto-run analysis
results_data = run_nodal_assessment(source)

if results_data:
    results_df = pd.DataFrame(results_data)
    
    # Ranking
    results_df = results_df.sort_values('Revenue Score', ascending=False).reset_index(drop=True)
    results_df.index += 1 # 1-based ranking
    
    st.subheader("🏆 Top Performing Nodes")
    
    # Metrics styling
    st.dataframe(
        results_df.style.background_gradient(subset=['Revenue Score', 'Volatility ($/MWh)'], cmap='Greens'),
        width="stretch"
    )

    st.caption("""
    **Metric Definitions:**
    - **Volatility:** Standard deviation of Real-Time Market (RTM) prices. Higher volatility often implies more arbitrage opportunities.
    - **Avg Spread:** Average absolute difference between Day-Ahead and Real-Time prices.
    - **Revenue Score:** A simplified proxy for profitability. It is the sum of all price spreads greater than $20/MWh. A higher score indicates more frequent and larger arbitrage opportunities.
    """)
    
    # Visual Comparison
    top_3 = results_df.head(3)['Node'].tolist()
    
    st.subheader("📊 Top 3 Nodes Comparison")
    
    # Load data for top 3 to plot
    dfs = []
    for node in top_3:
        d = load_data(source=source, node=node)
        d['Node'] = node
        dfs.append(d)
    
    if dfs:
        combined = pd.concat(dfs)
        
        # Plot Spread Distribution
        fig = px.box(combined, x='Node', y='price_spread', title="Price Spread Distribution (Volatility)")
        st.plotly_chart(fig, width="stretch")
        
else:
    st.info("No data found. Please check your data source or configuration.")

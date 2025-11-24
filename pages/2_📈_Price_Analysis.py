"""
Price Analysis Page
Zentus - ERCOT Battery Revenue Dashboard

This page provides detailed price dynamics analysis including
DA vs RT price comparison, statistics, and forecast error distribution.
"""

import streamlit as st
from config.page_config import configure_page
from ui.styles.custom_css import apply_custom_styles
from ui.components.header import render_header
from ui.components.sidebar import render_sidebar
from utils.state import get_state, has_valid_config, get_date_range_str
from core.data.loaders import DataLoader
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

configure_page("Price Analysis")
apply_custom_styles()

# ============================================================================
# HEADER AND SIDEBAR
# ============================================================================

render_header()
render_sidebar()

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.header("📈 Price Analysis")

# Check if configuration is valid
if not has_valid_config():
    st.warning("⚠️ Please configure battery specifications and select a settlement point in the sidebar to begin analysis.")
    st.stop()

# Get state
state = get_state()

if state.price_data is None:
    st.error("⚠️ Price data not loaded. Please refresh the page or check data availability.")
    st.stop()

if state.selected_node is None:
    st.error("⚠️ No settlement point selected. Please select a node in the sidebar.")
    st.stop()

# Load node data
loader = DataLoader(Path(__file__).parent.parent / 'data')
node_data = loader.filter_by_node(state.price_data, state.selected_node)

# Calculate dynamic thresholds for threshold-based strategy
if state.strategy_type == "Threshold-Based":
    # Calculate baseline thresholds (for visualization)
    decision_prices = node_data['price_mwh_da']
    charge_thresh = decision_prices.quantile(state.charge_percentile)
    discharge_thresh = decision_prices.quantile(state.discharge_percentile)

    # Ensure minimum spread
    if discharge_thresh - charge_thresh < 5:
        median = decision_prices.median()
        charge_thresh = median - 2.5
        discharge_thresh = median + 2.5

# ============================================================================
# PRICE DYNAMICS CHART
# ============================================================================

st.subheader(f"ERCOT Price Dynamics - {state.selected_node}")

fig_price = go.Figure()

fig_price.add_trace(go.Scatter(
    x=node_data['timestamp'],
    y=node_data['price_mwh_rt'],
    name='Real-Time Price',
    line=dict(color='#0A5F7A', width=2),
    hovertemplate='RT: $%{y:.2f}/MWh<extra></extra>'
))

fig_price.add_trace(go.Scatter(
    x=node_data['timestamp'],
    y=node_data['price_mwh_da'],
    name='Day-Ahead Price',
    line=dict(color='#FF6B35', width=2, dash='dash'),
    hovertemplate='DA: $%{y:.2f}/MWh<extra></extra>'
))

# Highlight negative prices
negative_rt = node_data[node_data['price_mwh_rt'] < 0]
if len(negative_rt) > 0:
    fig_price.add_trace(go.Scatter(
        x=negative_rt['timestamp'],
        y=negative_rt['price_mwh_rt'],
        mode='markers',
        name='Negative RT Prices',
        marker=dict(color='red', size=10, symbol='x'),
        hovertemplate='Negative Price: $%{y:.2f}/MWh<extra></extra>'
    ))

# Add threshold lines for threshold-based strategy
if state.strategy_type == "Threshold-Based":
    fig_price.add_hline(
        y=discharge_thresh,
        line_dash="dot",
        line_color="green",
        annotation_text=f"Discharge Threshold (${discharge_thresh:.2f}/MWh)"
    )
    fig_price.add_hline(
        y=charge_thresh,
        line_dash="dot",
        line_color="orange",
        annotation_text=f"Charge Threshold (${charge_thresh:.2f}/MWh)"
    )

fig_price.add_hline(
    y=0,
    line_dash="solid",
    line_color="gray",
    annotation_text="$0/MWh"
)

fig_price.update_layout(
    title=f"{get_date_range_str(node_data)} - {state.selected_node}",
    xaxis_title="Time",
    yaxis_title="Price ($/MWh)",
    height=500,
    hovermode='x unified'
)

from ui.components.charts import apply_standard_chart_styling
apply_standard_chart_styling(fig_price)

st.plotly_chart(fig_price, width="stretch")

# ============================================================================
# PRICE STATISTICS AND FORECAST ERROR
# ============================================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Price Statistics ({get_date_range_str(node_data)})")

    stats_data = {
        'Metric': [
            'Min DA Price',
            'Max DA Price',
            'Avg DA Price',
            'Min RT Price',
            'Max RT Price',
            'Avg RT Price',
            'Negative RT Hours',
            'Large Spread Hours (>$10)'
        ],
        'Value': [
            f"${node_data['price_mwh_da'].min():.2f}",
            f"${node_data['price_mwh_da'].max():.2f}",
            f"${node_data['price_mwh_da'].mean():.2f}",
            f"${node_data['price_mwh_rt'].min():.2f}",
            f"${node_data['price_mwh_rt'].max():.2f}",
            f"${node_data['price_mwh_rt'].mean():.2f}",
            f"{(node_data['price_mwh_rt'] < 0).sum()} hours",
            f"{node_data['extreme_event'].sum()} hours ({node_data['extreme_event'].sum()/len(node_data)*100:.1f}%)"
        ]
    }
    st.table(pd.DataFrame(stats_data))

with col2:
    st.subheader("Forecast Error Distribution")

    # Calculate optimal bins using Sturges' rule
    forecast_data = node_data['forecast_error'].dropna()
    n = len(forecast_data)
    optimal_bins = int(np.ceil(np.log2(n) + 1))

    # Add reasonable bounds (min 10, max 50)
    optimal_bins = max(10, min(50, optimal_bins))

    # Manually calculate histogram with exact bin count
    counts, bin_edges = np.histogram(forecast_data, bins=optimal_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # Create bar chart with exact bins
    fig_error_hist = go.Figure(data=[go.Bar(
        x=bin_centers,
        y=counts,
        width=bin_width * 0.9,
        marker_color='#0A5F7A',
        hovertemplate='Error: %{x:.2f} $/MWh<br>Count: %{y}<extra></extra>'
    )])

    fig_error_hist.update_layout(
        title=f"DA Forecast Error Distribution ({optimal_bins} bins, Sturges' rule)",
        xaxis_title='Forecast Error ($/MWh)',
        yaxis_title='Count',
        showlegend=False
    )
    st.plotly_chart(fig_error_hist, width="stretch")

    mae = node_data['forecast_error'].abs().mean()
    st.metric("Mean Absolute Error", f"${mae:.2f}/MWh")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("💡 Navigate to other pages in the sidebar to explore battery operations, revenue tracking, and optimization strategies.")

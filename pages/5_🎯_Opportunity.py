"""
Opportunity Analysis Page
Zentus - ERCOT Battery Revenue Dashboard

This page provides sensitivity analysis showing revenue impact
across different forecast improvement levels and strategy comparison.
"""

import streamlit as st
from config.page_config import configure_page
from ui.styles.custom_css import apply_custom_styles
from ui.components.header import render_header
from ui.components.sidebar import render_sidebar
from utils.state import get_state, has_valid_config, get_date_range_str
from core.battery.simulator import BatterySimulator
from core.battery.battery import BatterySpecs
from core.battery.strategies import ThresholdStrategy, RollingWindowStrategy, LinearOptimizationStrategy
from core.data.loaders import DataLoader
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

configure_page("Opportunity Analysis")
apply_custom_styles()

# ============================================================================
# HEADER AND SIDEBAR
# ============================================================================

render_header()
render_sidebar()

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.header("🎯 Revenue Opportunity Analysis")

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

# Check if battery specs are configured
if state.battery_specs is None:
    st.error("⚠️ Battery specifications not configured. Please configure in the sidebar.")
    st.stop()

# ============================================================================
# SENSITIVITY ANALYSIS
# ============================================================================

st.subheader("Impact of Forecast Accuracy on Revenue - Strategy Comparison")

# Reduce points for performance (0, 10, 20... 100)
@st.cache_data(ttl=3600, show_spinner="Running sensitivity analysis...")
def run_sensitivity_analysis(
    node_data: pd.DataFrame,
    battery_specs: BatterySpecs,
    charge_percentile: float,
    discharge_percentile: float,
    window_hours: int
):
    """
    Run sensitivity analysis with caching.

    Runs all three strategies (Threshold, Rolling Window, Linear Optimization)
    across the full range of forecast improvement levels (0-100%).

    Note: Linear Optimization finds the optimal solution for any given forecast
    quality. At 100% improvement (perfect foresight), it achieves theoretical max.
    At lower improvement levels, it optimizes for imperfect forecasts but revenue
    is still calculated at actual RT prices, so it may underperform its potential.
    """
    improvement_range = range(0, 101, 10)  # Reduced resolution for speed
    revenue_threshold = []
    revenue_rolling_window = []
    revenue_linear = []

    simulator = BatterySimulator(battery_specs)

    for imp in improvement_range:
        improvement_factor = imp / 100

        # Threshold strategy
        strategy_threshold = ThresholdStrategy(charge_percentile, discharge_percentile)
        temp_result_threshold = simulator.run(
            node_data,
            strategy_threshold,
            improvement_factor=improvement_factor
        )
        revenue_threshold.append(temp_result_threshold.total_revenue)

        # Rolling window strategy
        strategy_window = RollingWindowStrategy(window_hours)
        temp_result_window = simulator.run(
            node_data,
            strategy_window,
            improvement_factor=improvement_factor
        )
        revenue_rolling_window.append(temp_result_window.total_revenue)

        # Linear Optimization strategy (new instance for each improvement level)
        strategy_linear = LinearOptimizationStrategy()
        temp_result_linear = simulator.run(
            node_data,
            strategy_linear,
            improvement_factor=improvement_factor
        )
        revenue_linear.append(temp_result_linear.total_revenue)

    return list(improvement_range), revenue_threshold, revenue_rolling_window, revenue_linear

# Run analysis
improvement_range, revenue_threshold, revenue_rolling_window, revenue_linear = run_sensitivity_analysis(
    node_data,
    state.battery_specs,
    state.charge_percentile,
    state.discharge_percentile,
    state.window_hours
)

# Create comparison chart
fig_sensitivity = go.Figure()

fig_sensitivity.add_trace(go.Scatter(
    x=list(improvement_range),
    y=revenue_threshold,
    name='Threshold-Based',
    line=dict(color='#6B7280', width=2),
    mode='lines+markers',
    hovertemplate='Threshold<br>Improvement: %{x}%<br>Revenue: $%{y:,.0f}<extra></extra>'
))

fig_sensitivity.add_trace(go.Scatter(
    x=list(improvement_range),
    y=revenue_rolling_window,
    name='Rolling Window',
    line=dict(color='#0A5F7A', width=3),
    mode='lines+markers',
    hovertemplate='Rolling Window<br>Improvement: %{x}%<br>Revenue: $%{y:,.0f}<extra></extra>'
))

# Linear Optimization curve (upper bound at each improvement level)
fig_sensitivity.add_trace(go.Scatter(
    x=list(improvement_range),
    y=revenue_linear,
    name='Linear Optimization',
    line=dict(color='#28a745', width=3),
    mode='lines+markers',
    hovertemplate='Linear Opt<br>Improvement: %{x}%<br>Revenue: $%{y:,.0f}<extra></extra>'
))

fig_sensitivity.update_layout(
    title="Revenue Sensitivity: All Strategies vs Forecast Improvement",
    xaxis_title="Forecast Improvement (%)",
    yaxis_title="Revenue ($)",
    height=500,
    hovermode='x unified',
    legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01)
)

st.plotly_chart(fig_sensitivity, width="stretch")

# ============================================================================
# STRATEGY COMPARISON INSIGHTS
# ============================================================================

st.markdown("### Strategy Comparison Insights")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Threshold-Based Strategy:**")
    # Check if monotonic
    threshold_diffs = [revenue_threshold[i+1] - revenue_threshold[i] for i in range(len(revenue_threshold)-1)]
    negative_changes = sum(1 for d in threshold_diffs if d < 0)

    if negative_changes > 0:
        st.warning(f"⚠️ Non-monotonic: {negative_changes} instances where improvement REDUCED revenue")
    else:
        st.success("✓ Monotonic improvement")

    st.metric("Revenue Range",
             f"${min(revenue_threshold):,.0f} to ${max(revenue_threshold):,.0f}")

with col2:
    st.markdown("**Rolling Window Strategy:**")
    # Check if monotonic
    window_diffs = [revenue_rolling_window[i+1] - revenue_rolling_window[i] for i in range(len(revenue_rolling_window)-1)]
    negative_changes_window = sum(1 for d in window_diffs if d < 0)

    if negative_changes_window > 0:
        st.warning(f"⚠️ Non-monotonic: {negative_changes_window} instances where improvement REDUCED revenue")
    else:
        st.success("✓ Monotonic improvement")

    st.metric("Revenue Range",
             f"${min(revenue_rolling_window):,.0f} to ${max(revenue_rolling_window):,.0f}")

with col3:
    st.markdown("**Linear Optimization:**")
    # Check if monotonic
    linear_diffs = [revenue_linear[i+1] - revenue_linear[i] for i in range(len(revenue_linear)-1)]
    negative_changes_linear = sum(1 for d in linear_diffs if d < 0)

    if negative_changes_linear > 0:
        st.warning(f"⚠️ Non-monotonic: {negative_changes_linear} instances where improvement REDUCED revenue")
    else:
        st.success("✓ Monotonic improvement")

    st.metric("Revenue Range",
             f"${min(revenue_linear):,.0f} to ${max(revenue_linear):,.0f}")

# ============================================================================
# KEY INSIGHTS
# ============================================================================

st.markdown("---")
st.subheader(f"Market Insights - {get_date_range_str(node_data)}")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Price Events")
    st.markdown(f"""
    **Negative Price Events:**
    - {(node_data['price_mwh_rt'] < 0).sum()} hours with negative RT prices
    - Lowest: ${node_data['price_mwh_rt'].min():.2f}/MWh
    - Batteries can charge while getting paid

    **Price Volatility:**
    - {node_data['extreme_event'].sum()} hours with >$10/MWh spread
    - Max spread: ${node_data['price_spread'].max():.2f}/MWh
    - These hours drive significant revenue opportunity

    # Forecast Performance:
    - Mean Absolute Error: ${node_data['forecast_error'].abs().mean():.2f}/MWh
    - {state.forecast_improvement}% improvement = ${(revenue_rolling_window[state.forecast_improvement//10] - revenue_rolling_window[0]):,.0f} gain (Rolling Window)
    """)

with col2:
    st.markdown("#### Price Spread Analysis")
    fig_spread = px.scatter(
        node_data,
        x='price_mwh_da',
        y='price_mwh_rt',
        title="Day-Ahead vs Real-Time Prices",
        labels={
            'price_mwh_da': 'DA Price ($/MWh)',
            'price_mwh_rt': 'RT Price ($/MWh)'
        },
        color='price_spread',
        color_continuous_scale='Reds',
        hover_data=['timestamp']
    )

    # Add diagonal line (perfect forecast)
    min_price = min(node_data['price_mwh_da'].min(), node_data['price_mwh_rt'].min())
    max_price = max(node_data['price_mwh_da'].max(), node_data['price_mwh_rt'].max())
    fig_spread.add_trace(go.Scatter(
        x=[min_price, max_price],
        y=[min_price, max_price],
        mode='lines',
        name='Perfect Forecast',
        line=dict(color='gray', dash='dash'),
        showlegend=True
    ))

    st.plotly_chart(fig_spread, width="stretch")

# ============================================================================
# REVENUE CAPTURE ANALYSIS
# ============================================================================

st.markdown("---")
st.subheader("Revenue Capture Potential")

# Current settings revenue (using Linear Optimization as the true optimal benchmark)
current_idx = state.forecast_improvement // 10
baseline_revenue = revenue_linear[0]  # LP at 0% improvement
current_revenue = revenue_linear[current_idx]  # LP at current improvement
max_revenue = revenue_linear[-1]  # LP at 100% (perfect foresight)

captured = current_revenue - baseline_revenue
remaining = max_revenue - current_revenue
total_opportunity = max_revenue - baseline_revenue

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Baseline Revenue (LP)",
        f"${baseline_revenue:,.0f}",
        help="Linear Optimization revenue with 0% forecast improvement (DA only)"
    )

with col2:
    st.metric(
        f"Current Revenue (+{state.forecast_improvement}%)",
        f"${current_revenue:,.0f}",
        delta=f"+${captured:,.0f}",
        help=f"Linear Optimization revenue with {state.forecast_improvement}% forecast improvement"
    )

with col3:
    st.metric(
        "Perfect Foresight Max",
        f"${max_revenue:,.0f}",
        delta=f"+${remaining:,.0f} remaining",
        help="Linear Optimization at 100% (theoretical maximum)"
    )

with col4:
    capture_pct = (captured / total_opportunity * 100) if total_opportunity != 0 else 0
    st.metric(
        "Capture Rate",
        f"{capture_pct:.1f}%",
        help="Percentage of total forecast improvement opportunity captured"
    )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("💡 Navigate to other pages in the sidebar to explore decision timelines and optimization strategies.")

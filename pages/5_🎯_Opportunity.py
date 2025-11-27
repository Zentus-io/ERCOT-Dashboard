"""
Opportunity Analysis Page
Zentus - ERCOT Battery Revenue Dashboard

This page provides sensitivity analysis showing revenue impact
across different forecast improvement levels and strategy comparison.
"""

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.page_config import configure_page
from core.battery.battery import BatterySpecs
from core.battery.simulator import BatterySimulator
from core.battery.strategies import (
    LinearOptimizationStrategy,
    MPCStrategy,
    RollingWindowStrategy,
    ThresholdStrategy,
)
from ui.components.header import render_header
from ui.components.sidebar import render_sidebar
from ui.styles.custom_css import apply_custom_styles
from utils.state import get_date_range_str, get_state, has_valid_config

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
    st.warning(
        "⚠️ Please configure battery specifications and select a settlement point in the sidebar to begin analysis.")
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
# Load node data
if state.price_data.empty:
    st.warning("⚠️ No price data available. Please check your data source or date range.")
    st.stop()

if 'node' in state.price_data.columns:
    node_col = 'node'
elif 'settlement_point' in state.price_data.columns:
    node_col = 'settlement_point'
elif 'SettlementPoint' in state.price_data.columns:
    node_col = 'SettlementPoint'
else:
    st.error(f"❌ Price data has unexpected column names: {list(state.price_data.columns)}")
    st.stop()

node_data = state.price_data[state.price_data[node_col] == state.selected_node].copy()

# Check if battery specs are configured
if state.battery_specs is None:
    st.error("⚠️ Battery specifications not configured. Please configure in the sidebar.")
    st.stop()

# ============================================================================
# SENSITIVITY ANALYSIS
# ============================================================================

st.subheader("Impact of Forecast Accuracy on Revenue")

st.info("""
**Chart Guide:** The green LP Benchmark line represents the **theoretical maximum** achievable with perfect hindsight.
The gap between your selected strategy and LP shows potential gains from strategy improvements.
""")

# Reduce points for performance (0, 10, 20... 100)


def run_sensitivity_analysis(
    node_data: pd.DataFrame,
    battery_specs: BatterySpecs,
    charge_percentile: float,
    discharge_percentile: float,
    window_hours: int
):
    """
    Run sensitivity analysis with progress bar and parallel strategy execution.
    """
    improvement_range = range(0, 101, 10)  # Reduced resolution for speed
    revenue_threshold = []
    revenue_rolling_window = []
    revenue_mpc = []
    revenue_linear = []

    simulator = BatterySimulator(battery_specs)

    progress_bar = st.progress(0)
    status_text = st.empty()
    total_steps = len(improvement_range)

    def run_strategy(strategy, improvement_factor):
        """Helper function to run a single strategy simulation."""
        return simulator.run(node_data, strategy, improvement_factor=improvement_factor)

    for i, imp in enumerate(improvement_range):
        status_text.text(f"Simulating forecast improvement: {imp}%...")
        improvement_factor = imp / 100

        # Create strategy instances
        strategy_threshold = ThresholdStrategy(charge_percentile, discharge_percentile)
        strategy_window = RollingWindowStrategy(window_hours)
        strategy_mpc = MPCStrategy(horizon_hours=24)
        strategy_linear = LinearOptimizationStrategy()

        # Run all strategies in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all strategy simulations
            future_threshold = executor.submit(run_strategy, strategy_threshold, improvement_factor)
            future_window = executor.submit(run_strategy, strategy_window, improvement_factor)
            future_mpc = executor.submit(run_strategy, strategy_mpc, improvement_factor)
            future_linear = executor.submit(run_strategy, strategy_linear, improvement_factor)

            # Collect results
            result_threshold = future_threshold.result()
            result_window = future_window.result()
            result_mpc = future_mpc.result()
            result_linear = future_linear.result()

        revenue_threshold.append(result_threshold.total_revenue)
        revenue_rolling_window.append(result_window.total_revenue)
        revenue_mpc.append(result_mpc.total_revenue)
        revenue_linear.append(result_linear.total_revenue)

        progress_bar.progress((i + 1) / total_steps)

    status_text.empty()
    progress_bar.empty()

    return list(improvement_range), revenue_threshold, revenue_rolling_window, revenue_mpc, revenue_linear


def run_window_sensitivity_analysis(
    node_data: pd.DataFrame,
    battery_specs: BatterySpecs,
    improvement_factor: float
):
    """
    Run sensitivity analysis for Rolling Window strategy varying window size.
    """
    window_range = range(2, 49, 2)  # 2 to 48 hours
    revenue_window = []

    simulator = BatterySimulator(battery_specs)

    progress_bar = st.progress(0)
    status_text = st.empty()
    total_steps = len(window_range)

    for i, window in enumerate(window_range):
        status_text.text(f"Simulating window size: {window} hours...")
        strategy = RollingWindowStrategy(window_hours=window)
        result = simulator.run(
            node_data,
            strategy,
            improvement_factor=improvement_factor
        )
        revenue_window.append(result.total_revenue)
        progress_bar.progress((i + 1) / total_steps)

    status_text.empty()
    progress_bar.empty()

    return list(window_range), revenue_window


def run_horizon_sensitivity_analysis(
    node_data: pd.DataFrame,
    battery_specs: BatterySpecs,
    improvement_factor: float
):
    """
    Run sensitivity analysis for MPC strategy varying horizon size.
    """
    horizon_range = range(2, 14, 2)  # 2, 4, 6... 24 hours
    revenue_horizon = []

    simulator = BatterySimulator(battery_specs)

    progress_bar = st.progress(0)
    status_text = st.empty()
    total_steps = len(horizon_range)

    for i, horizon in enumerate(horizon_range):
        status_text.text(f"Simulating MPC horizon: {horizon} hours...")
        strategy = MPCStrategy(horizon_hours=horizon)
        result = simulator.run(
            node_data,
            strategy,
            improvement_factor=improvement_factor
        )
        revenue_horizon.append(result.total_revenue)
        progress_bar.progress((i + 1) / total_steps)

    status_text.empty()
    progress_bar.empty()

    return list(horizon_range), revenue_horizon


# Run forecast sensitivity analysis
improvement_range, revenue_threshold, revenue_rolling_window, revenue_mpc, revenue_linear = run_sensitivity_analysis(
    node_data, state.battery_specs, state.charge_percentile, state.discharge_percentile, state.window_hours)

# Create comparison chart
fig_sensitivity = go.Figure()

fig_sensitivity.add_trace(go.Scatter(
    x=list(improvement_range),
    y=revenue_threshold,
    name='Threshold-Based',
    line={"color": '#6B7280', "width": 2},
    mode='lines+markers',
    hovertemplate='Threshold<br>Improvement: %{x}%<br>Revenue: $%{y:,.0f}<extra></extra>'
))

fig_sensitivity.add_trace(go.Scatter(
    x=list(improvement_range),
    y=revenue_rolling_window,
    name='Rolling Window',
    line={"color": '#0A5F7A', "width": 3},
    mode='lines+markers',
    hovertemplate='Rolling Window<br>Improvement: %{x}%<br>Revenue: $%{y:,.0f}<extra></extra>'
))

fig_sensitivity.add_trace(go.Scatter(
    x=list(improvement_range),
    y=revenue_mpc,
    name='MPC (24h Horizon)',
    line={"color": '#8B5CF6', "width": 3},
    mode='lines+markers',
    hovertemplate='MPC<br>Improvement: %{x}%<br>Revenue: $%{y:,.0f}<extra></extra>'
))

# LP Benchmark curve (theoretical upper bound at each improvement level)
fig_sensitivity.add_trace(go.Scatter(
    x=list(improvement_range),
    y=revenue_linear,
    name='LP Benchmark (Theoretical Max)',
    line={"color": '#28a745', "width": 3},
    mode='lines+markers',
    hovertemplate='LP Benchmark<br>Improvement: %{x}%<br>Revenue: $%{y:,.0f}<extra></extra>'
))

fig_sensitivity.update_layout(
    title="Revenue Sensitivity: Practical Strategies vs LP Benchmark",
    xaxis_title="Forecast Improvement (%)",
    yaxis_title="Revenue ($)",
    height=500,
    hovermode='x unified',
    legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
    yaxis={"fixedrange": False},  # Allow Y-axis zooming
    xaxis={"fixedrange": False}
)

st.plotly_chart(fig_sensitivity, width="stretch")

# Window Sensitivity (Only for Rolling Window Strategy)
if state.strategy_type == "Rolling Window Optimization":
    st.markdown("---")
    st.subheader(f"Window Size Sensitivity (at {state.forecast_improvement}% Improvement)")

    window_range, revenue_window_sens = run_window_sensitivity_analysis(
        node_data,
        state.battery_specs,
        state.forecast_improvement / 100.0
    )

    fig_window = go.Figure()

    fig_window.add_trace(go.Scatter(
        x=window_range,
        y=revenue_window_sens,
        name='Revenue',
        line={"color": '#0A5F7A', "width": 3},
        mode='lines+markers',
        hovertemplate='Window: %{x}h<br>Revenue: $%{y:,.0f}<extra></extra>'
    ))

    # Add marker for current selection
    current_rev = revenue_window_sens[window_range.index(
        state.window_hours)] if state.window_hours in window_range else 0
    if state.window_hours in window_range:
        fig_window.add_trace(go.Scatter(
            x=[state.window_hours],
            y=[current_rev],
            mode='markers',
            marker={"color": 'red', "size": 12, "symbol": 'star'},
            name='Current Selection',
            hoverinfo='skip'
        ))

    fig_window.update_layout(
        title="Revenue vs. Lookahead Window Size",
        xaxis_title="Lookahead Window (Hours)",
        yaxis_title="Revenue ($)",
        height=400,
        hovermode='x unified',
        yaxis={"fixedrange": False},
        xaxis={"fixedrange": False}
    )

    st.plotly_chart(fig_window, width="stretch")

    # Find optimal window
    best_window = window_range[revenue_window_sens.index(max(revenue_window_sens))]
    st.info(f"💡 Optimal lookahead window for this scenario appears to be **{best_window} hours**.")

# Horizon Sensitivity (Only for MPC Strategy)
elif state.strategy_type == "MPC (Rolling Horizon)":
    st.markdown("---")
    st.subheader(f"Optimization Horizon Sensitivity (at {state.forecast_improvement}% Improvement)")

    horizon_range, revenue_horizon_sens = run_horizon_sensitivity_analysis(
        node_data,
        state.battery_specs,
        state.forecast_improvement / 100.0
    )

    fig_horizon = go.Figure()

    fig_horizon.add_trace(go.Scatter(
        x=horizon_range,
        y=revenue_horizon_sens,
        name='Revenue',
        line={"color": '#8B5CF6', "width": 3},
        mode='lines+markers',
        hovertemplate='Horizon: %{x}h<br>Revenue: $%{y:,.0f}<extra></extra>'
    ))

    # Add marker for current selection
    current_rev = revenue_horizon_sens[horizon_range.index(
        state.horizon_hours)] if state.horizon_hours in horizon_range else 0
    if state.horizon_hours in horizon_range:
        fig_horizon.add_trace(go.Scatter(
            x=[state.horizon_hours],
            y=[current_rev],
            mode='markers',
            marker={"color": 'red', "size": 12, "symbol": 'star'},
            name='Current Selection',
            hoverinfo='skip'
        ))

    fig_horizon.update_layout(
        title="Revenue vs. Optimization Horizon",
        xaxis_title="Optimization Horizon (Hours)",
        yaxis_title="Revenue ($)",
        height=400,
        hovermode='x unified',
        yaxis={"fixedrange": False},
        xaxis={"fixedrange": False}
    )

    st.plotly_chart(fig_horizon, width="stretch")

    # Find optimal horizon
    best_horizon = horizon_range[revenue_horizon_sens.index(max(revenue_horizon_sens))]
    st.info(f"💡 Optimal horizon for this scenario appears to be **{best_horizon} hours**.")

# ============================================================================
# STRATEGY COMPARISON INSIGHTS
# ============================================================================

st.markdown("### Strategy Comparison Insights")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**Threshold-Based Strategy:**")
    # Check if monotonic
    threshold_diffs = [revenue_threshold[i + 1] - revenue_threshold[i]
                       for i in range(len(revenue_threshold) - 1)]
    negative_changes = sum(1 for d in threshold_diffs if d < 0)

    if negative_changes > 0:
        st.warning(
            f"⚠️ Non-monotonic: {negative_changes} instances where improvement REDUCED revenue")
    else:
        st.success("✓ Monotonic improvement")

    st.metric("Revenue Range",
              f"${min(revenue_threshold):,.0f} to ${max(revenue_threshold):,.0f}")

with col2:
    st.markdown("**Rolling Window Strategy:**")
    # Check if monotonic
    window_diffs = [revenue_rolling_window[i + 1] - revenue_rolling_window[i]
                    for i in range(len(revenue_rolling_window) - 1)]
    negative_changes_window = sum(1 for d in window_diffs if d < 0)

    if negative_changes_window > 0:
        st.warning(
            f"⚠️ Non-monotonic: {negative_changes_window} instances where improvement REDUCED revenue")
    else:
        st.success("✓ Monotonic improvement")

    st.metric("Revenue Range",
              f"${min(revenue_rolling_window):,.0f} to ${max(revenue_rolling_window):,.0f}")

with col3:
    st.markdown("**MPC Strategy (24h):**")
    # Check if monotonic
    mpc_diffs = [revenue_mpc[i + 1] - revenue_mpc[i] for i in range(len(revenue_mpc) - 1)]
    negative_changes_mpc = sum(1 for d in mpc_diffs if d < 0)

    if negative_changes_mpc > 0:
        st.warning(
            f"⚠️ Non-monotonic: {negative_changes_mpc} instances where improvement REDUCED revenue")
    else:
        st.success("✓ Monotonic improvement")

    st.metric("Revenue Range",
              f"${min(revenue_mpc):,.0f} to ${max(revenue_mpc):,.0f}")

with col4:
    st.markdown("**LP Benchmark (Theoretical):**")
    # Check if monotonic
    linear_diffs = [revenue_linear[i + 1] - revenue_linear[i]
                    for i in range(len(revenue_linear) - 1)]
    negative_changes_linear = sum(1 for d in linear_diffs if d < 0)

    if negative_changes_linear > 0:
        st.warning(
            f"⚠️ Non-monotonic: {negative_changes_linear} instances where improvement REDUCED revenue")
    else:
        st.success("✓ Monotonic improvement")

    st.metric("Revenue Range",
              f"${min(revenue_linear):,.0f} to ${max(revenue_linear):,.0f}")
    st.caption("Upper bound for any strategy")

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
    - {state.forecast_improvement}% improvement = ${(revenue_rolling_window[int(state.forecast_improvement // 10)] - revenue_rolling_window[0]):,.0f} gain (Rolling Window)
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
        line={"color": 'gray', "dash": 'dash'},
        showlegend=True
    ))

    st.plotly_chart(fig_spread, width="stretch")

# ============================================================================
# REVENUE CAPTURE ANALYSIS
# ============================================================================

st.markdown("---")
st.subheader("LP Benchmark Analysis")

st.info("""
This section shows how revenue improves along the **LP Benchmark curve** as forecast accuracy increases.
The LP benchmark represents the theoretical maximum at each improvement level.
""")

# Current settings revenue (using LP Benchmark as the true optimal)
current_idx = int(state.forecast_improvement // 10)
baseline_revenue_lp = revenue_linear[0]  # LP at 0% improvement
current_revenue_lp = revenue_linear[current_idx]  # LP at current improvement
max_revenue_lp = revenue_linear[-1]  # LP at 100% (perfect foresight)

captured = current_revenue_lp - baseline_revenue_lp
remaining = max_revenue_lp - current_revenue_lp
total_opportunity = max_revenue_lp - baseline_revenue_lp

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "LP @ 0% (DA Only)",
        f"${baseline_revenue_lp:,.0f}",
        help="LP Benchmark revenue using only day-ahead forecasts"
    )

with col2:
    st.metric(
        f"LP @ {state.forecast_improvement}%",
        f"${current_revenue_lp:,.0f}",
        delta=f"+${captured:,.0f}",
        help=f"LP Benchmark revenue with {state.forecast_improvement}% forecast improvement"
    )

with col3:
    st.metric(
        "LP @ 100% (Perfect)",
        f"${max_revenue_lp:,.0f}",
        delta=f"+${remaining:,.0f} remaining",
        help="LP Benchmark at 100% (absolute theoretical maximum)"
    )

with col4:
    capture_pct = (captured / total_opportunity * 100) if total_opportunity != 0 else 0
    st.metric(
        "LP Capture Rate",
        f"{capture_pct:.1f}%",
        help="Percentage of LP improvement potential captured at current forecast level"
    )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("💡 Navigate to other pages in the sidebar to explore decision timelines and optimization strategies.")

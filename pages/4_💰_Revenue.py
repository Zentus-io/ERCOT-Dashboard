"""
Revenue Comparison Page
Zentus - ERCOT Battery Revenue Dashboard

This page provides cumulative revenue tracking over time
with detailed breakdown of costs and revenues.
"""

import streamlit as st
from config.page_config import configure_page
from ui.styles.custom_css import apply_custom_styles
from ui.components.header import render_header
from ui.components.sidebar import render_sidebar
from utils.state import get_state, has_valid_config
from core.battery.simulator import BatterySimulator
from core.battery.strategies import ThresholdStrategy, RollingWindowStrategy
from core.data.loaders import DataLoader
from pathlib import Path
import plotly.graph_objects as go

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

configure_page("Revenue Comparison")
apply_custom_styles()

# ============================================================================
# HEADER AND SIDEBAR
# ============================================================================

render_header()
render_sidebar()

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.header("💰 Revenue Comparison")

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
# RUN SIMULATIONS
# ============================================================================

with st.spinner('Running battery simulations...'):
    simulator = BatterySimulator(state.battery_specs)

    # Select strategy
    if state.strategy_type == "Rolling Window Optimization":
        strategy_baseline = RollingWindowStrategy(state.window_hours)
        strategy_improved = RollingWindowStrategy(state.window_hours)
        strategy_optimal = RollingWindowStrategy(state.window_hours)
    else:  # Threshold-Based
        strategy_baseline = ThresholdStrategy(state.charge_percentile, state.discharge_percentile)
        strategy_improved = ThresholdStrategy(state.charge_percentile, state.discharge_percentile)
        strategy_optimal = ThresholdStrategy(state.charge_percentile, state.discharge_percentile)

    # Run simulations for each scenario
    baseline_result = simulator.run(node_data, strategy_baseline, improvement_factor=0.0)
    improved_result = simulator.run(node_data, strategy_improved, improvement_factor=state.forecast_improvement/100)
    optimal_result = simulator.run(node_data, strategy_optimal, improvement_factor=1.0)

# ============================================================================
# CUMULATIVE REVENUE CHART
# ============================================================================

st.subheader("Cumulative Revenue Comparison")

fig_revenue = go.Figure()

fig_revenue.add_trace(go.Scatter(
    x=optimal_result.dispatch_df['timestamp'],
    y=optimal_result.dispatch_df['cumulative_revenue'],
    name='Optimal (Perfect Foresight)',
    line=dict(color='#28A745', width=3),
    hovertemplate='$%{y:,.0f}<extra></extra>'
))

fig_revenue.add_trace(go.Scatter(
    x=improved_result.dispatch_df['timestamp'],
    y=improved_result.dispatch_df['cumulative_revenue'],
    name=f'Improved Forecast (+{state.forecast_improvement}%)',
    line=dict(color='#FFC107', width=2.5),
    hovertemplate='$%{y:,.0f}<extra></extra>'
))

fig_revenue.add_trace(go.Scatter(
    x=baseline_result.dispatch_df['timestamp'],
    y=baseline_result.dispatch_df['cumulative_revenue'],
    name='Baseline (Day-Ahead)',
    line=dict(color='#DC3545', width=2, dash='dash'),
    hovertemplate='$%{y:,.0f}<extra></extra>'
))

fig_revenue.update_layout(
    title="Revenue Accumulation Over 24 Hours",
    xaxis_title="Time",
    yaxis_title="Cumulative Revenue ($)",
    height=500,
    hovermode='x unified'
)

st.plotly_chart(fig_revenue, width="stretch")

# ============================================================================
# REVENUE BREAKDOWN
# ============================================================================

st.subheader("Revenue Breakdown")

# Calculate final revenues
baseline_revenue = baseline_result.total_revenue
improved_revenue = improved_result.total_revenue
optimal_revenue = optimal_result.total_revenue

opportunity_vs_baseline = improved_revenue - baseline_revenue
improvement_pct = (opportunity_vs_baseline / abs(baseline_revenue)) * 100 if baseline_revenue != 0 else 0
max_opportunity = optimal_revenue - baseline_revenue

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Baseline (DA only)", f"${baseline_revenue:,.2f}")
    st.caption(f"Charge cost: ${baseline_result.charge_cost:,.0f}")
    st.caption(f"Discharge revenue: ${baseline_result.discharge_revenue:,.0f}")

with col2:
    st.metric(
        f"Improved (+{state.forecast_improvement}%)",
        f"${improved_revenue:,.2f}",
        delta=f"+${opportunity_vs_baseline:,.2f}" if opportunity_vs_baseline >= 0 else f"${opportunity_vs_baseline:,.2f}"
    )
    st.caption(f"Gain: ${opportunity_vs_baseline:,.2f}")
    st.caption(f"Improvement: {improvement_pct:.1f}%")

with col3:
    st.metric(
        "Optimal (Perfect)",
        f"${optimal_revenue:,.2f}",
        delta=f"+${max_opportunity:,.2f}" if max_opportunity >= 0 else f"${max_opportunity:,.2f}"
    )
    st.caption(f"Max opportunity: ${max_opportunity:,.2f}")
    captured_pct = (opportunity_vs_baseline / max_opportunity * 100) if max_opportunity != 0 else 0
    st.caption(f"Captured: {captured_pct:.1f}%")

# ============================================================================
# DETAILED METRICS
# ============================================================================

st.markdown("---")
st.subheader("Detailed Financial Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Baseline Scenario**")
    st.metric("Total Charge Cost", f"${baseline_result.charge_cost:,.0f}")
    st.metric("Total Discharge Revenue", f"${baseline_result.discharge_revenue:,.0f}")
    st.metric("Net Revenue", f"${baseline_revenue:,.0f}")

    # Calculate average prices
    baseline_charge_df = baseline_result.dispatch_df[baseline_result.dispatch_df['dispatch'] == 'charge']
    baseline_discharge_df = baseline_result.dispatch_df[baseline_result.dispatch_df['dispatch'] == 'discharge']

    if len(baseline_charge_df) > 0:
        avg_charge_price = baseline_charge_df['actual_price'].mean()
        st.metric("Avg Charge Price", f"${avg_charge_price:.2f}/MWh")

    if len(baseline_discharge_df) > 0:
        avg_discharge_price = baseline_discharge_df['actual_price'].mean()
        st.metric("Avg Discharge Price", f"${avg_discharge_price:.2f}/MWh")

with col2:
    st.markdown(f"**Improved Scenario (+{state.forecast_improvement}%)**")
    st.metric("Total Charge Cost", f"${improved_result.charge_cost:,.0f}")
    st.metric("Total Discharge Revenue", f"${improved_result.discharge_revenue:,.0f}")
    st.metric("Net Revenue", f"${improved_revenue:,.0f}")

    # Calculate average prices
    improved_charge_df = improved_result.dispatch_df[improved_result.dispatch_df['dispatch'] == 'charge']
    improved_discharge_df = improved_result.dispatch_df[improved_result.dispatch_df['dispatch'] == 'discharge']

    if len(improved_charge_df) > 0:
        avg_charge_price = improved_charge_df['actual_price'].mean()
        st.metric("Avg Charge Price", f"${avg_charge_price:.2f}/MWh")

    if len(improved_discharge_df) > 0:
        avg_discharge_price = improved_discharge_df['actual_price'].mean()
        st.metric("Avg Discharge Price", f"${avg_discharge_price:.2f}/MWh")

with col3:
    st.markdown("**Optimal Scenario (Perfect)**")
    st.metric("Total Charge Cost", f"${optimal_result.charge_cost:,.0f}")
    st.metric("Total Discharge Revenue", f"${optimal_result.discharge_revenue:,.0f}")
    st.metric("Net Revenue", f"${optimal_revenue:,.0f}")

    # Calculate average prices
    optimal_charge_df = optimal_result.dispatch_df[optimal_result.dispatch_df['dispatch'] == 'charge']
    optimal_discharge_df = optimal_result.dispatch_df[optimal_result.dispatch_df['dispatch'] == 'discharge']

    if len(optimal_charge_df) > 0:
        avg_charge_price = optimal_charge_df['actual_price'].mean()
        st.metric("Avg Charge Price", f"${avg_charge_price:.2f}/MWh")

    if len(optimal_discharge_df) > 0:
        avg_discharge_price = optimal_discharge_df['actual_price'].mean()
        st.metric("Avg Discharge Price", f"${avg_discharge_price:.2f}/MWh")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("💡 Navigate to other pages in the sidebar to explore opportunity analysis, decision timelines, and optimization strategies.")

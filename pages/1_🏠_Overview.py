"""
Overview Page - Strategy Performance Comparison
Zentus - ERCOT Battery Revenue Dashboard

This page provides a high-level comparison of dispatch strategies
and key performance metrics.
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

configure_page("Overview")
apply_custom_styles()

# ============================================================================
# HEADER AND SIDEBAR
# ============================================================================

render_header()
render_sidebar()

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.header("📊 Revenue Analysis")

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
# KEY METRICS
# ============================================================================

st.info("""
**How to use this dashboard:**
Each strategy is tested with **3 forecast quality scenarios**:
1. **Baseline (DA Only)** - Uses only day-ahead forecasts (no improvement) ← Never changes
2. **Improved** - Uses day-ahead + forecast improvement ← **Changes when you adjust the slider**
3. **Perfect Foresight** - Uses actual real-time prices (theoretical maximum) ← Never changes

💡 **Tip:** Move the "Forecast Accuracy Improvement" slider to see how better forecasts improve Scenario 2!
""")

col1, col2, col3, col4 = st.columns(4)

naive_revenue = baseline_result.total_revenue
improved_revenue = improved_result.total_revenue
optimal_revenue = optimal_result.total_revenue

opportunity_vs_naive = optimal_revenue - naive_revenue
opportunity_vs_improved = improved_revenue - naive_revenue
improvement_pct = (opportunity_vs_improved / abs(naive_revenue)) * 100 if naive_revenue != 0 else 0

with col1:
    st.metric(
        label="Scenario 1: Baseline (DA Only)",
        value=f"${naive_revenue:,.0f}",
        help="Revenue using only day-ahead forecasts (no improvement)"
    )

with col2:
    st.metric(
        label=f"Scenario 2: Improved (+{state.forecast_improvement}%)",
        value=f"${improved_revenue:,.0f}",
        delta=f"+${opportunity_vs_improved:,.0f}" if opportunity_vs_improved >= 0 else f"${opportunity_vs_improved:,.0f}",
        delta_color="normal",
        help=f"Revenue with {state.forecast_improvement}% forecast accuracy improvement"
    )

with col3:
    st.metric(
        label="Scenario 3: Perfect Foresight",
        value=f"${optimal_revenue:,.0f}",
        delta=f"+${opportunity_vs_naive:,.0f}" if opportunity_vs_naive >= 0 else f"${opportunity_vs_naive:,.0f}",
        delta_color="normal",
        help="Theoretical maximum revenue using perfect real-time price knowledge"
    )

with col4:
    st.metric(
        label="Revenue Improvement",
        value=f"{improvement_pct:.1f}%",
        help="Percentage improvement over baseline"
    )

# ============================================================================
# DISPATCH STATISTICS
# ============================================================================

st.markdown("---")
st.subheader("📋 Battery Dispatch Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Baseline (DA Forecast)**")
    st.write(f"Charge events: {baseline_result.charge_count}")
    st.write(f"Discharge events: {baseline_result.discharge_count}")
    st.write(f"Hold periods: {baseline_result.hold_count}")
    st.write(f"Charge cost: ${baseline_result.charge_cost:,.0f}")
    st.write(f"Discharge revenue: ${baseline_result.discharge_revenue:,.0f}")

with col2:
    st.markdown(f"**Improved (+{state.forecast_improvement}%)**")
    st.write(f"Charge events: {improved_result.charge_count}")
    st.write(f"Discharge events: {improved_result.discharge_count}")
    st.write(f"Hold periods: {improved_result.hold_count}")
    st.write(f"Charge cost: ${improved_result.charge_cost:,.0f}")
    st.write(f"Discharge revenue: ${improved_result.discharge_revenue:,.0f}")

with col3:
    st.markdown("**Optimal (Perfect Forecast)**")
    st.write(f"Charge events: {optimal_result.charge_count}")
    st.write(f"Discharge events: {optimal_result.discharge_count}")
    st.write(f"Hold periods: {optimal_result.hold_count}")
    st.write(f"Charge cost: ${optimal_result.charge_cost:,.0f}")
    st.write(f"Discharge revenue: ${optimal_result.discharge_revenue:,.0f}")

# ============================================================================
# STRATEGY COMPARISON
# ============================================================================

st.markdown("---")
st.subheader("Strategy Performance Comparison")

# Create three columns for side-by-side comparison
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📉 Baseline (DA Only)")
    st.metric("Revenue", f"${naive_revenue:,.0f}")
    st.metric("Charge Events", baseline_result.charge_count)
    st.metric("Discharge Events", baseline_result.discharge_count)
    charge_pct = (baseline_result.charge_count / len(node_data)) * 100
    discharge_pct = (baseline_result.discharge_count / len(node_data)) * 100
    st.progress(charge_pct / 100, text=f"Charging: {charge_pct:.1f}% of time")
    st.progress(discharge_pct / 100, text=f"Discharging: {discharge_pct:.1f}% of time")

with col2:
    st.markdown(f"### 📈 Improved (+{state.forecast_improvement}%)")
    st.metric("Revenue", f"${improved_revenue:,.0f}",
             delta=f"+${opportunity_vs_improved:,.0f}" if opportunity_vs_improved >= 0 else f"${opportunity_vs_improved:,.0f}")
    st.metric("Charge Events", improved_result.charge_count)
    st.metric("Discharge Events", improved_result.discharge_count)
    charge_pct = (improved_result.charge_count / len(node_data)) * 100
    discharge_pct = (improved_result.discharge_count / len(node_data)) * 100
    st.progress(charge_pct / 100, text=f"Charging: {charge_pct:.1f}% of time")
    st.progress(discharge_pct / 100, text=f"Discharging: {discharge_pct:.1f}% of time")

with col3:
    st.markdown("### ⭐ Theoretical Max")
    st.metric("Revenue", f"${optimal_revenue:,.0f}",
             delta=f"+${opportunity_vs_naive:,.0f}" if opportunity_vs_naive >= 0 else f"${opportunity_vs_naive:,.0f}")
    st.metric("Charge Events", optimal_result.charge_count)
    st.metric("Discharge Events", optimal_result.discharge_count)
    charge_pct = (optimal_result.charge_count / len(node_data)) * 100
    discharge_pct = (optimal_result.discharge_count / len(node_data)) * 100
    st.progress(charge_pct / 100, text=f"Charging: {charge_pct:.1f}% of time")
    st.progress(discharge_pct / 100, text=f"Discharging: {discharge_pct:.1f}% of time")

# ============================================================================
# REVENUE COMPARISON BAR CHART
# ============================================================================

st.markdown("---")
st.markdown("### Revenue Comparison")

revenue_data = {
    'Strategy': ['Baseline\n(DA Only)', f'Improved\n(+{state.forecast_improvement}%)', 'Theoretical\nMax'],
    'Revenue': [naive_revenue, improved_revenue, optimal_revenue],
    'Color': ['#6B7280', '#4A9FB8', '#0A5F7A']
}

fig_revenue_bars = go.Figure()
fig_revenue_bars.add_trace(go.Bar(
    x=revenue_data['Strategy'],
    y=revenue_data['Revenue'],
    marker_color=revenue_data['Color'],
    text=[f"${r:,.0f}" for r in revenue_data['Revenue']],
    textposition='outside',
    hovertemplate='%{x}<br>Revenue: $%{y:,.0f}<extra></extra>'
))

fig_revenue_bars.update_layout(
    title=f"Strategy Performance - {state.strategy_type}",
    yaxis_title="Revenue ($)",
    height=400,
    showlegend=False
)

st.plotly_chart(fig_revenue_bars, width="stretch")

# ============================================================================
# STRATEGY INSIGHTS
# ============================================================================

st.markdown("### Strategy Insights")

if state.strategy_type == "Rolling Window Optimization":
    improvement_rate = ((improved_revenue - naive_revenue) / abs(naive_revenue) * 100) if naive_revenue != 0 else 0
    st.info(f"""
    **Rolling Window Strategy** with {state.window_hours}-hour lookahead window:
    - Achieves {improvement_rate:+.1f}% revenue improvement with {state.forecast_improvement}% better forecasts
    - Naturally handles temporal constraints (must charge before discharge)
    - Avoids threshold crossing sensitivity issues
    - Makes decisions based on price ranking within lookahead window
    - **More robust** to forecast errors than threshold-based
    """)
else:  # Threshold-Based
    st.warning(f"""
    **Threshold-Based Strategy** using {int(state.charge_percentile*100)}th/{int(state.discharge_percentile*100)}th percentiles:
    - May show non-monotonic improvement (small forecast gains can reduce revenue)
    - Sensitive to threshold parameter selection
    - Simple and interpretable but suboptimal for arbitrage
    - Consider switching to Rolling Window for more consistent gains
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("💡 Navigate to other pages in the sidebar to explore detailed analysis of prices, battery operations, and optimization strategies.")

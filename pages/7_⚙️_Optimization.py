"""
Optimization Analysis Page
Zentus - ERCOT Battery Revenue Dashboard

This page provides deep dive into the selected dispatch strategy,
showing how it works and comparing decisions between scenarios.
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
import pandas as pd

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

configure_page("Optimization Analysis")
apply_custom_styles()

# ============================================================================
# HEADER AND SIDEBAR
# ============================================================================

render_header()
render_sidebar()

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.header("⚙️ Optimization Analysis")

# Check if configuration is valid
if not has_valid_config():
    st.warning("⚠️ Please configure battery specifications and select a settlement point in the sidebar to begin analysis.")
    st.stop()

# Get state
state = get_state()

# Load node data
loader = DataLoader(Path(__file__).parent.parent / 'data')
node_data = loader.filter_by_node(state.price_data, state.selected_node)

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
# STRATEGY-SPECIFIC ANALYSIS
# ============================================================================

if state.strategy_type == "Rolling Window Optimization":
    # ROLLING WINDOW STRATEGY ANALYSIS
    st.markdown(f"### Rolling Window Strategy (Lookahead: {state.window_hours} hours)")

    st.info(f"""
    **How it works:**
    - At each hour, look ahead {state.window_hours} hours into the future
    - Charge if current price is the MINIMUM in the window (cheap now, might be expensive later)
    - Discharge if current price is the MAXIMUM in the window (expensive now, might be cheap later)
    - Hold otherwise (current price is neither min nor max)

    **Advantages:**
    - No threshold crossing sensitivity issues
    - Better forecast → better price ranking → better decisions
    - Naturally handles temporal constraints
    """)

    # Show example hours where window optimization helped
    st.markdown("---")
    st.markdown("### Example: How Lookahead Window Improves Decisions")

    # Find interesting hours where improved made different decision than baseline
    baseline_df = baseline_result.dispatch_df.copy()
    improved_df = improved_result.dispatch_df.copy()

    # Merge on timestamp to compare decisions
    comparison = baseline_df.merge(
        improved_df,
        on='timestamp',
        suffixes=('_baseline', '_improved')
    )

    different_decisions = comparison[
        comparison['dispatch_baseline'] != comparison['dispatch_improved']
    ]

    if len(different_decisions) > 0:
        example = different_decisions.iloc[0]
        example_timestamp = example['timestamp']

        # Find the index in the original node_data
        example_idx = node_data[node_data['timestamp'] == example_timestamp].index[0]
        window_end = min(example_idx + state.window_hours, len(node_data))

        st.markdown(f"**Example Hour: Improved forecast made DIFFERENT decision than baseline**")
        st.markdown(f"**Time:** {example_timestamp}")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Baseline Decision:**")
            st.markdown(f"- Action: **{example['dispatch_baseline'].upper()}**")
            st.markdown(f"- Price: ${example['actual_price_baseline']:.2f}/MWh")

        with col2:
            st.markdown("**Improved Decision:**")
            st.markdown(f"- Action: **{example['dispatch_improved'].upper()}**")
            st.markdown(f"- Price: ${example['actual_price_improved']:.2f}/MWh")

        # Show window prices
        window_data = node_data.iloc[example_idx:window_end].copy()
        fig_window = go.Figure()

        fig_window.add_trace(go.Scatter(
            x=window_data['timestamp'],
            y=window_data['price_mwh_rt'],
            mode='lines+markers',
            name='RT Price',
            line=dict(color='#0A5F7A')
        ))

        # Add vertical line at decision point
        fig_window.add_shape(
            type="line",
            x0=example_timestamp,
            x1=example_timestamp,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="red", width=2, dash="dash")
        )
        fig_window.add_annotation(
            x=example_timestamp,
            y=1,
            yref="paper",
            text="Decision Point",
            showarrow=False,
            yshift=10
        )

        fig_window.update_layout(
            title=f"{state.window_hours}-Hour Lookahead Window",
            xaxis_title="Time",
            yaxis_title="Price ($/MWh)",
            height=300
        )

        st.plotly_chart(fig_window, width="stretch")

        # Show all different decisions
        st.markdown("---")
        st.markdown("### All Decision Differences")
        st.markdown(f"**Total hours where decisions differed:** {len(different_decisions)}")

        differences_summary = different_decisions[[
            'timestamp',
            'dispatch_baseline',
            'dispatch_improved',
            'actual_price_baseline'
        ]].copy()
        differences_summary.columns = ['Timestamp', 'Baseline Action', 'Improved Action', 'Price ($/MWh)']
        st.dataframe(differences_summary, width="stretch")

    else:
        st.info("Baseline and improved strategies made identical decisions for all hours with current parameters.")

else:
    # THRESHOLD-BASED STRATEGY ANALYSIS
    st.markdown("### Threshold-Based Strategy")
    st.markdown(f"**Charge threshold:** {int(state.charge_percentile*100)}th percentile")
    st.markdown(f"**Discharge threshold:** {int(state.discharge_percentile*100)}th percentile")

    # Calculate thresholds
    decision_prices = node_data['price_mwh_da']
    charge_thresh = decision_prices.quantile(state.charge_percentile)
    discharge_thresh = decision_prices.quantile(state.discharge_percentile)

    # Ensure minimum spread
    if discharge_thresh - charge_thresh < 5:
        median = decision_prices.median()
        charge_thresh = median - 2.5
        discharge_thresh = median + 2.5

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Charge Below", f"${charge_thresh:.2f}/MWh")
    with col2:
        st.metric("Discharge Above", f"${discharge_thresh:.2f}/MWh")

    st.markdown("---")
    st.markdown("### Price Distribution with Trading Thresholds")

    # Show price distribution with thresholds
    fig_price_dist = go.Figure()

    fig_price_dist.add_trace(go.Histogram(
        x=node_data['price_mwh_rt'],
        name='RT Price Distribution',
        nbinsx=20,
        marker_color='#0A5F7A'
    ))

    fig_price_dist.add_vline(
        x=charge_thresh,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Charge < ${charge_thresh:.2f}"
    )

    fig_price_dist.add_vline(
        x=discharge_thresh,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Discharge > ${discharge_thresh:.2f}"
    )

    fig_price_dist.update_layout(
        title="Price Distribution with Trading Thresholds",
        xaxis_title="Price ($/MWh)",
        yaxis_title="Frequency",
        height=400
    )

    st.plotly_chart(fig_price_dist, width="stretch")

    st.markdown("---")
    st.markdown("### Understanding the 3 Scenarios")

    st.info("""
    **How the scenarios differ:**
    - **Scenario 1 (Baseline):** Uses DA forecasts as-is
    - **Scenario 2 (Improved):** Corrects DA forecasts by the slider percentage toward actual RT prices
    - **Scenario 3 (Perfect):** Uses actual RT prices (theoretical maximum)

    Each scenario calculates thresholds from its own forecast distribution, ensuring
    fair comparison and revealing the true impact of forecast quality.
    """)

    # Analyze where perfect foresight made better decisions
    baseline_df = baseline_result.dispatch_df.copy()
    optimal_df = optimal_result.dispatch_df.copy()

    comparison = baseline_df.merge(
        optimal_df,
        on='timestamp',
        suffixes=('_baseline', '_optimal')
    )

    better_decisions = comparison[
        comparison['dispatch_baseline'] != comparison['dispatch_optimal']
    ]

    st.metric(
        "Hours where perfect foresight made different decision than baseline",
        len(better_decisions)
    )

    if len(better_decisions) > 0:
        decision_diff_revenue = optimal_result.total_revenue - improved_result.total_revenue
        st.metric(
            "Revenue gap between improved and perfect",
            f"${decision_diff_revenue:,.0f}"
        )
        st.caption("This gap represents the remaining opportunity from better forecasting")

# ============================================================================
# COMPARATIVE PERFORMANCE METRICS
# ============================================================================

st.markdown("---")
st.subheader("Comparative Performance Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Baseline Performance**")
    st.metric("Revenue", f"${baseline_result.total_revenue:,.0f}")
    st.metric("Total Actions", baseline_result.charge_count + baseline_result.discharge_count)
    efficiency_baseline = (baseline_result.total_revenue / baseline_result.discharge_revenue * 100) if baseline_result.discharge_revenue > 0 else 0
    st.metric("Efficiency", f"{efficiency_baseline:.1f}%")
    st.caption("Revenue / Discharge Revenue")

with col2:
    st.markdown(f"**Improved Performance (+{state.forecast_improvement}%)**")
    st.metric("Revenue", f"${improved_result.total_revenue:,.0f}")
    st.metric("Total Actions", improved_result.charge_count + improved_result.discharge_count)
    efficiency_improved = (improved_result.total_revenue / improved_result.discharge_revenue * 100) if improved_result.discharge_revenue > 0 else 0
    st.metric("Efficiency", f"{efficiency_improved:.1f}%")
    st.caption("Revenue / Discharge Revenue")

with col3:
    st.markdown("**Optimal Performance**")
    st.metric("Revenue", f"${optimal_result.total_revenue:,.0f}")
    st.metric("Total Actions", optimal_result.charge_count + optimal_result.discharge_count)
    efficiency_optimal = (optimal_result.total_revenue / optimal_result.discharge_revenue * 100) if optimal_result.discharge_revenue > 0 else 0
    st.metric("Efficiency", f"{efficiency_optimal:.1f}%")
    st.caption("Revenue / Discharge Revenue")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("💡 Navigate to other pages in the sidebar to explore price analysis, battery operations, and revenue tracking.")

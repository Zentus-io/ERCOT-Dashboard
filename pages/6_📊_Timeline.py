"""
Decision Timeline Page
Zentus - ERCOT Battery Revenue Dashboard

This page provides a gantt-style timeline showing when the battery
charges, discharges, or holds across different scenarios.
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
from plotly.subplots import make_subplots
import pandas as pd

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

configure_page("Decision Timeline")
apply_custom_styles()

# ============================================================================
# HEADER AND SIDEBAR
# ============================================================================

render_header()
render_sidebar()

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.header("📊 Decision Timeline")

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
# DISPATCH TIMELINE CHART
# ============================================================================

st.subheader("Dispatch Schedule Comparison")

# Create figure with subplots for each strategy
fig_timeline = make_subplots(
    rows=3, cols=1,
    subplot_titles=(
        "Baseline (DA Only)",
        f"Improved (+{state.forecast_improvement}%)",
        "Theoretical Maximum"
    ),
    shared_xaxes=True,
    vertical_spacing=0.1,
    row_heights=[0.33, 0.33, 0.33]
)

# Define colors for actions
action_colors = {
    'charge': {'light': 'rgba(40, 167, 69, 0.3)', 'dark': '#28A745'},
    'discharge': {'light': 'rgba(220, 53, 69, 0.3)', 'dark': '#DC3545'},
    'hold': {'light': '#6C757D', 'dark': '#6C757D'}
}

def add_dispatch_bars(fig, dispatch_df, row_num, battery_power_mw):
    """Add dispatch bars with proper stacking to show actual vs constrained power."""
    timestamps = dispatch_df['timestamp'].tolist()

    actual_heights = []
    actual_colors = []
    actual_hovers = []
    constrained_heights = []
    constrained_colors = []

    for _, row in dispatch_df.iterrows():
        action = row['dispatch']

        if action in ['charge', 'discharge']:
            actual_energy = row['energy_mwh']
            # Fraction represents % of hourly power capacity used
            fraction = actual_energy / battery_power_mw

            # Bottom: dark color, height = fraction
            actual_heights.append(fraction)
            actual_colors.append(action_colors[action]['dark'])
            actual_hovers.append(
                f"{action.capitalize()}<br>"
                f"Energy: {actual_energy:.1f} MWh<br>"
                f"% of Power Capacity: {fraction:.1%}"
            )

            # Top: light color, height = remaining
            constrained_heights.append(1.0 - fraction)
            constrained_colors.append(action_colors[action]['light'])

        else:  # hold
            # Bottom: zero height (invisible)
            actual_heights.append(0)
            actual_colors.append('rgba(0,0,0,0)')  # Transparent
            actual_hovers.append('Hold')

            # Top: full height gray
            constrained_heights.append(1.0)
            constrained_colors.append(action_colors['hold']['light'])

    # Add bottom layer trace (actual energy)
    fig.add_trace(go.Bar(
        x=timestamps,
        y=actual_heights,
        marker_color=actual_colors,
        name='Actual Energy' if row_num == 1 else None,
        showlegend=(row_num == 1),
        hovertext=actual_hovers,
        hoverinfo='text'
    ), row=row_num, col=1)

    # Add top layer trace (constrained portion)
    fig.add_trace(go.Bar(
        x=timestamps,
        y=constrained_heights,
        marker_color=constrained_colors,
        name='Constrained' if row_num == 1 else None,
        showlegend=(row_num == 1),
        hoverinfo='skip'
    ), row=row_num, col=1)

# Add data for each strategy
add_dispatch_bars(fig_timeline, baseline_result.dispatch_df, 1, state.battery_specs.power_mw)
add_dispatch_bars(fig_timeline, improved_result.dispatch_df, 2, state.battery_specs.power_mw)
add_dispatch_bars(fig_timeline, optimal_result.dispatch_df, 3, state.battery_specs.power_mw)

fig_timeline.update_layout(
    height=600,
    barmode='stack',
    bargap=0.1,
    title_text=f"Battery Dispatch Timeline - {state.strategy_type}",
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

# Fix y-axis range to [0, 1] for accurate visual proportions
fig_timeline.update_yaxes(range=[0, 1], autorange=False, fixedrange=True, visible=False)
fig_timeline.update_xaxes(title_text="Time", row=3, col=1)

st.plotly_chart(fig_timeline, width="stretch")

# ============================================================================
# PRICE CONTEXT FOR DISPATCH DECISIONS
# ============================================================================

st.markdown("---")
st.markdown("### Price Context for Dispatch Decisions")

fig_price_dispatch = go.Figure()

# Add price line
fig_price_dispatch.add_trace(go.Scatter(
    x=node_data['timestamp'],
    y=node_data['price_mwh_rt'],
    name='Real-Time Price',
    line=dict(color='#0A5F7A', width=2)
))

# Add dispatch markers for improved strategy
charge_times = improved_result.dispatch_df[improved_result.dispatch_df['dispatch'] == 'charge']
discharge_times = improved_result.dispatch_df[improved_result.dispatch_df['dispatch'] == 'discharge']

fig_price_dispatch.add_trace(go.Scatter(
    x=charge_times['timestamp'],
    y=charge_times['actual_price'],
    mode='markers',
    name='Charge',
    marker=dict(color='#28A745', size=12, symbol='triangle-down'),
    hovertemplate='Charge<br>Time: %{x}<br>Price: $%{y:.2f}/MWh<extra></extra>'
))

fig_price_dispatch.add_trace(go.Scatter(
    x=discharge_times['timestamp'],
    y=discharge_times['actual_price'],
    mode='markers',
    name='Discharge',
    marker=dict(color='#DC3545', size=12, symbol='triangle-up'),
    hovertemplate='Discharge<br>Time: %{x}<br>Price: $%{y:.2f}/MWh<extra></extra>'
))

fig_price_dispatch.update_layout(
    title=f"Dispatch Decisions Overlaid on Price - Improved Strategy ({state.forecast_improvement}% improvement)",
    xaxis_title="Time",
    yaxis_title="Price ($/MWh)",
    height=400,
    hovermode='x unified'
)

st.plotly_chart(fig_price_dispatch, width="stretch")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

st.markdown("---")
st.subheader("Decision Summary Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Charge Count", improved_result.charge_count)
    if improved_result.charge_count > 0:
        avg_charge_price = charge_times['actual_price'].mean()
        st.caption(f"Avg charge price: ${avg_charge_price:.2f}/MWh")
    else:
        st.caption("No charges")

with col2:
    st.metric("Discharge Count", improved_result.discharge_count)
    if improved_result.discharge_count > 0:
        avg_discharge_price = discharge_times['actual_price'].mean()
        st.caption(f"Avg discharge price: ${avg_discharge_price:.2f}/MWh")
    else:
        st.caption("No discharges")

with col3:
    if improved_result.charge_count > 0 and improved_result.discharge_count > 0:
        avg_charge_price = charge_times['actual_price'].mean()
        avg_discharge_price = discharge_times['actual_price'].mean()
        spread = avg_discharge_price - avg_charge_price
        st.metric("Avg Price Spread", f"${spread:.2f}/MWh")
        st.caption("Theoretical gain per cycle")
    else:
        st.metric("Avg Price Spread", "N/A")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("💡 Navigate to other pages in the sidebar to explore optimization strategies and detailed analysis.")

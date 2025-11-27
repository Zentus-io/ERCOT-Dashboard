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
# RUN SIMULATIONS
# ============================================================================

# ============================================================================
# RUN SIMULATIONS
# ============================================================================

from utils.simulation_runner import run_or_get_cached_simulation

with st.spinner('Running battery simulations...'):
    baseline_result, improved_result, optimal_result, theoretical_max_result = run_or_get_cached_simulation()

    if not baseline_result or not improved_result or not optimal_result or not theoretical_max_result:
        st.error("⚠️ Failed to run simulations. Please check data availability.")
        st.stop()

# ============================================================================
# DISPATCH TIMELINE CHART
# ============================================================================

# ============================================================================
# DISPATCH TIMELINE CHART
# ============================================================================

st.subheader("Dispatch Schedule Comparison")

# Add view toggle
view_mode = st.radio(
    "View Mode",
    ["Detailed (Hourly)", "Daily Aggregation"],
    horizontal=True
)

# Calculate charge/discharge times for improved strategy (used by both views and summary stats)
charge_times = improved_result.dispatch_df[improved_result.dispatch_df['dispatch'] == 'charge']
discharge_times = improved_result.dispatch_df[improved_result.dispatch_df['dispatch'] == 'discharge']

if view_mode == "Detailed (Hourly)":
    # Create figure with subplots for each strategy PLUS price context
    fig_timeline = make_subplots(
        rows=5, cols=1,
        subplot_titles=(
            "Baseline (DA Only)",
            f"Improved (+{state.forecast_improvement}%)",
            "Strategy Max (100%)",
            "LP Benchmark (Theoretical Max)",
            f"Price Context - Improved Strategy ({state.forecast_improvement}% improvement)"
        ),
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.2, 0.2, 0.2, 0.2, 0.2],
        specs=[[{"secondary_y": False}]] * 5  # All rows have single y-axis
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
    add_dispatch_bars(fig_timeline, theoretical_max_result.dispatch_df, 4, state.battery_specs.power_mw)

    # Add price context to row 5
    fig_timeline.add_trace(go.Scatter(
        x=node_data['timestamp'],
        y=node_data['price_mwh_rt'],
        name='Real-Time Price',
        line=dict(color='#0A5F7A', width=2),
        showlegend=True
    ), row=5, col=1)

    # Add dispatch markers for improved strategy on price chart
    fig_timeline.add_trace(go.Scatter(
        x=charge_times['timestamp'],
        y=charge_times['actual_price'],
        mode='markers',
        name='Charge Decisions',
        marker=dict(color='#28A745', size=10, symbol='triangle-down'),
        hovertemplate='Charge<br>Time: %{x}<br>Price: $%{y:.2f}/MWh<extra></extra>',
        showlegend=True
    ), row=5, col=1)

    fig_timeline.add_trace(go.Scatter(
        x=discharge_times['timestamp'],
        y=discharge_times['actual_price'],
        mode='markers',
        name='Discharge Decisions',
        marker=dict(color='#DC3545', size=10, symbol='triangle-up'),
        hovertemplate='Discharge<br>Time: %{x}<br>Price: $%{y:.2f}/MWh<extra></extra>',
        showlegend=True
    ), row=5, col=1)

    fig_timeline.update_layout(
        height=1000,
        barmode='stack',
        bargap=0.1,
        title_text=f"Battery Dispatch Timeline with Price Context - {state.strategy_type} vs LP Benchmark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1)
    )

    # Fix y-axis range to [0, 1] for dispatch rows (rows 1-4)
    for row in range(1, 5):
        fig_timeline.update_yaxes(range=[0, 1], autorange=False, fixedrange=True, visible=False, row=row, col=1)

    # Configure price chart y-axis (row 5)
    fig_timeline.update_yaxes(title_text="Price ($/MWh)", row=5, col=1)

    # Configure x-axis for the bottom plot only
    fig_timeline.update_xaxes(title_text="Time", row=5, col=1)

else:
    # Daily Aggregation View
    fig_timeline = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            "Baseline (DA Only)",
            f"Improved (+{state.forecast_improvement}%)",
            "Strategy Max (100%)",
            "LP Benchmark (Theoretical Max)"
        ),
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.25, 0.25, 0.25, 0.25]
    )

    def add_daily_bars(fig, dispatch_df, row_num):
        # Resample to daily
        df = dispatch_df.copy()
        df['date'] = df['timestamp'].dt.date
        
        # Calculate daily charge and discharge
        daily_charge = df[df['dispatch'] == 'charge'].groupby('date')['energy_mwh'].sum()
        daily_discharge = df[df['dispatch'] == 'discharge'].groupby('date')['energy_mwh'].sum()
        
        # Align indexes
        all_dates = sorted(list(set(daily_charge.index) | set(daily_discharge.index)))
        
        # Add traces
        fig.add_trace(go.Bar(
            x=all_dates,
            y=[daily_charge.get(d, 0) for d in all_dates],
            name='Total Charge' if row_num == 1 else None,
            marker_color='#28A745',
            showlegend=(row_num == 1)
        ), row=row_num, col=1)
        
        fig.add_trace(go.Bar(
            x=all_dates,
            y=[daily_discharge.get(d, 0) for d in all_dates],
            name='Total Discharge' if row_num == 1 else None,
            marker_color='#DC3545',
            showlegend=(row_num == 1)
        ), row=row_num, col=1)

    add_daily_bars(fig_timeline, baseline_result.dispatch_df, 1)
    add_daily_bars(fig_timeline, improved_result.dispatch_df, 2)
    add_daily_bars(fig_timeline, optimal_result.dispatch_df, 3)
    add_daily_bars(fig_timeline, theoretical_max_result.dispatch_df, 4)

    fig_timeline.update_layout(
        height=800,
        barmode='group',
        title_text=f"Daily Aggregated Dispatch - {state.strategy_type} vs LP Benchmark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_timeline.update_yaxes(title_text="Energy (MWh)")

st.plotly_chart(fig_timeline, width="stretch")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

st.markdown("---")
st.subheader("Decision Summary Statistics")

# Two main sections: Improved Forecast and LP Benchmark
section1, section2 = st.columns(2)

with section1:
    st.markdown("#### Improved Forecast")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Charge Count", improved_result.charge_count)
        if improved_result.charge_count > 0:
            avg_charge_price = charge_times['actual_price'].mean()
            st.caption(f"Avg: ${avg_charge_price:.2f}/MWh")
        else:
            st.caption("No charges")

    with col2:
        st.metric("Discharge Count", improved_result.discharge_count)
        if improved_result.discharge_count > 0:
            avg_discharge_price = discharge_times['actual_price'].mean()
            st.caption(f"Avg: ${avg_discharge_price:.2f}/MWh")
        else:
            st.caption("No discharges")

    with col3:
        if improved_result.charge_count > 0 and improved_result.discharge_count > 0:
            avg_charge_price = charge_times['actual_price'].mean()
            avg_discharge_price = discharge_times['actual_price'].mean()
            spread = avg_discharge_price - avg_charge_price
            st.metric("Price Spread", f"${spread:.2f}/MWh")
        else:
            st.metric("Price Spread", "N/A")

with section2:
    st.markdown("#### LP Benchmark")
    col4, col5, col6 = st.columns(3)

    lp_charge_times = theoretical_max_result.dispatch_df[theoretical_max_result.dispatch_df['dispatch'] == 'charge']
    lp_discharge_times = theoretical_max_result.dispatch_df[theoretical_max_result.dispatch_df['dispatch'] == 'discharge']

    with col4:
        st.metric("Charge Count", theoretical_max_result.charge_count)
        if len(lp_charge_times) > 0:
            lp_avg_charge = lp_charge_times['actual_price'].mean()
            st.caption(f"Avg: ${lp_avg_charge:.2f}/MWh")
        else:
            st.caption("No charges")

    with col5:
        st.metric("Discharge Count", theoretical_max_result.discharge_count)
        if len(lp_discharge_times) > 0:
            lp_avg_discharge = lp_discharge_times['actual_price'].mean()
            st.caption(f"Avg: ${lp_avg_discharge:.2f}/MWh")
        else:
            st.caption("No discharges")

    with col6:
        if len(lp_charge_times) > 0 and len(lp_discharge_times) > 0:
            lp_avg_charge = lp_charge_times['actual_price'].mean()
            lp_avg_discharge = lp_discharge_times['actual_price'].mean()
            lp_spread = lp_avg_discharge - lp_avg_charge
            st.metric("Price Spread", f"${lp_spread:.2f}/MWh")
        else:
            st.metric("Price Spread", "N/A")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("💡 Navigate to other pages in the sidebar to explore optimization strategies and detailed analysis.")

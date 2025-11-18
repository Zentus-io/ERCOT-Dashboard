"""
Sidebar Configuration Component
Zentus - ERCOT Battery Revenue Dashboard

This component renders the shared configuration sidebar
that persists across all pages.
"""

import streamlit as st
from pathlib import Path
import pandas as pd
from core.battery.battery import BatterySpecs
from core.data.loaders import DataLoader
from utils.state import get_state, update_state, clear_simulation_cache
from config.settings import DEFAULT_BATTERY, BATTERY_PRESETS


@st.cache_data
def load_app_data():
    """Load price and battery data with caching."""
    data_dir = Path(__file__).parent.parent.parent / 'data'
    loader = DataLoader(data_dir)

    price_data = loader.load_prices()
    eia_data = loader.load_eia_batteries()
    nodes = loader.get_nodes(price_data)

    return price_data, eia_data, nodes


def render_sidebar():
    """
    Render configuration sidebar.

    This sidebar is shared across all pages and manages:
    - Settlement point selection
    - Battery specifications
    - Dispatch strategy configuration
    - Forecast improvement settings
    """
    state = get_state()

    # Load data if not already loaded
    if state.price_data is None:
        price_data, eia_data, nodes = load_app_data()
        state.price_data = price_data
        state.eia_battery_data = eia_data
        state.available_nodes = nodes

    st.sidebar.header("Analysis Configuration")

    # ========================================================================
    # NODE SELECTION
    # ========================================================================
    selected_node = st.sidebar.selectbox(
        "Select Settlement Point:",
        state.available_nodes or [],
        index=0 if state.selected_node is None or state.available_nodes is None else state.available_nodes.index(state.selected_node),
        help="Choose a wind resource settlement point to analyze"
    )

    if selected_node != state.selected_node:
        update_state(selected_node=selected_node)
        clear_simulation_cache()

    # ========================================================================
    # STRATEGY SELECTION
    # ========================================================================
    st.sidebar.markdown("---")
    st.sidebar.subheader("Dispatch Strategy")

    strategy_type = st.sidebar.radio(
        "Battery Trading Strategy:",
        options=["Threshold-Based", "Rolling Window Optimization"],
        index=0 if state.strategy_type == "Threshold-Based" else 1,
        help="Choose the battery dispatch optimization approach"
    )

    if strategy_type != state.strategy_type:
        update_state(strategy_type=strategy_type)
        clear_simulation_cache()

    # Strategy-specific parameters
    if strategy_type == "Threshold-Based":
        st.sidebar.markdown("**Threshold Parameters:**")

        charge_pct = st.sidebar.slider(
            "Charge Threshold Percentile:",
            min_value=10,
            max_value=40,
            value=int(state.charge_percentile * 100),
            step=5,
            help="Charge when price below this percentile"
        ) / 100

        discharge_pct = st.sidebar.slider(
            "Discharge Threshold Percentile:",
            min_value=60,
            max_value=90,
            value=int(state.discharge_percentile * 100),
            step=5,
            help="Discharge when price above this percentile"
        ) / 100

        if charge_pct != state.charge_percentile or discharge_pct != state.discharge_percentile:
            update_state(charge_percentile=charge_pct, discharge_percentile=discharge_pct)
            clear_simulation_cache()

    else:  # Rolling Window
        st.sidebar.markdown("**Optimization Parameters:**")

        window = st.sidebar.slider(
            "Lookahead Window (hours):",
            min_value=2,
            max_value=12,
            value=state.window_hours,
            step=1,
            help="Number of hours to look ahead for optimization"
        )

        if window != state.window_hours:
            update_state(window_hours=window)
            clear_simulation_cache()

    # ========================================================================
    # BATTERY SPECIFICATIONS
    # ========================================================================
    st.sidebar.markdown("---")
    st.sidebar.subheader("Battery Specifications")

    # Battery presets
    if state.eia_battery_data is not None:
        preset_options = ["Custom"] + [v['name'] for v in BATTERY_PRESETS.values()]

        battery_preset = st.sidebar.selectbox(
            "Battery System Preset:",
            preset_options,
            help="Select a preset based on real Texas battery systems (EIA-860 data)"
        )

        # Set defaults based on preset
        if battery_preset == "Custom":
            default_capacity = DEFAULT_BATTERY['capacity_mwh']
            default_power = DEFAULT_BATTERY['power_mw']
        elif "Small" in battery_preset:
            default_capacity = BATTERY_PRESETS['Small']['capacity_mwh']
            default_power = BATTERY_PRESETS['Small']['power_mw']
        elif "Medium" in battery_preset:
            default_capacity = BATTERY_PRESETS['Medium']['capacity_mwh']
            default_power = BATTERY_PRESETS['Medium']['power_mw']
        elif "Large" in battery_preset:
            default_capacity = int(state.eia_battery_data['Nameplate Energy Capacity (MWh)'].quantile(0.9))
            default_power = int(state.eia_battery_data['Nameplate Capacity (MW)'].quantile(0.9))
        else:  # Very Large
            default_capacity = BATTERY_PRESETS['Very Large']['capacity_mwh']
            default_power = BATTERY_PRESETS['Very Large']['power_mw']
    else:
        battery_preset = "Custom"
        default_capacity = DEFAULT_BATTERY['capacity_mwh']
        default_power = DEFAULT_BATTERY['power_mw']

    capacity = st.sidebar.slider(
        "Energy Capacity (MWh):",
        min_value=10,
        max_value=600,
        value=default_capacity,
        step=5 if default_capacity < 100 else 10,
        help="Total energy storage capacity of the battery",
        disabled=(battery_preset != "Custom" and state.eia_battery_data is not None)
    )

    power = st.sidebar.slider(
        "Power Capacity (MW):",
        min_value=5,
        max_value=300,
        value=default_power,
        step=5,
        help="Maximum charge/discharge rate",
        disabled=(battery_preset != "Custom" and state.eia_battery_data is not None)
    )

    efficiency = st.sidebar.slider(
        "Round-trip Efficiency:",
        min_value=0.7,
        max_value=0.95,
        value=DEFAULT_BATTERY['efficiency'],
        step=0.05,
        help="Energy efficiency for charge/discharge cycle"
    )

    # Update battery specs if changed
    new_specs = BatterySpecs(
        capacity_mwh=capacity,
        power_mw=power,
        efficiency=efficiency
    )

    if state.battery_specs != new_specs:
        update_state(battery_specs=new_specs)
        clear_simulation_cache()

    # ========================================================================
    # FORECAST IMPROVEMENT
    # ========================================================================
    st.sidebar.subheader("Forecast Improvement Scenario")

    forecast_improvement = st.sidebar.slider(
        "Forecast Accuracy Improvement (%):",
        min_value=0,
        max_value=100,
        value=state.forecast_improvement,
        step=5,
        help="% of the forecast error to correct (0% = DA only, 100% = perfect RT knowledge)"
    )

    if forecast_improvement != state.forecast_improvement:
        update_state(forecast_improvement=forecast_improvement)
        clear_simulation_cache()

    # ========================================================================
    # DATA SUMMARY
    # ========================================================================
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Summary")

    if state.selected_node:
        node_data = state.price_data[state.price_data['node'] == state.selected_node]
        st.sidebar.metric("Date", "July 20, 2025")
        st.sidebar.metric("Hours Available", len(node_data))
        st.sidebar.metric("Extreme Events (>$10 spread)", node_data['extreme_event'].sum())

    # ========================================================================
    # EIA BATTERY MARKET CONTEXT
    # ========================================================================
    if state.eia_battery_data is not None:
        st.sidebar.markdown("---")
        with st.sidebar.expander("📊 ERCOT Battery Market Context", expanded=False):
            percentile_energy = (state.eia_battery_data['Nameplate Energy Capacity (MWh)'] < capacity).mean() * 100
            percentile_power = (state.eia_battery_data['Nameplate Capacity (MW)'] < power).mean() * 100

            duration_hours = capacity / power if power > 0 else 0

            st.markdown(f"""
            **Texas Battery Market (EIA-860 2024)**

            **Your System:**
            - {capacity:.0f} MWh / {power:.0f} MW
            - {duration_hours:.1f} hour duration
            - Larger than **{percentile_energy:.0f}%** of TX batteries (by energy)
            - Larger than **{percentile_power:.0f}%** of TX batteries (by power)

            **Market Summary:**
            - **136 operational systems** in Texas
            - **8,060 MW** total installed capacity
            - **54% primarily used for arbitrage** ✓
            - Median: 10 MW / 17 MWh (1h duration)
            - Mean: 59 MW / 85 MWh (1.4h duration)

            **Use Cases (% of systems):**
            - Arbitrage: 54% (your focus!)
            - Ramping Reserve: 46%
            - Frequency Regulation: 35%
            """)

            if percentile_energy < 50:
                st.info("💡 Your system is smaller than average - representative of typical merchant battery operators.")
            elif percentile_energy > 80:
                st.success("💡 Your system is in the top 20% by size - representative of large utility-scale projects.")
            else:
                st.info("💡 Your system is mid-sized - representative of the average Texas battery market.")

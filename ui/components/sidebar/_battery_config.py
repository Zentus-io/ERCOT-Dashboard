"""
Battery Configuration UI
Zentus - ERCOT Battery Revenue Dashboard

This module handles battery specification UI with presets and custom values.
"""

import pandas as pd
import streamlit as st

from config.settings import BATTERY_PRESETS, DEFAULT_BATTERY
from core.battery.battery import BatterySpecs
from utils.state import clear_simulation_cache, get_state, update_state

from ._widget_sync import render_synced_slider_input, update_synced_value


def render_battery_config(eia_data: pd.DataFrame, engie_data: pd.DataFrame) -> BatterySpecs:
    """
    Render battery specification UI with presets and sliders.

    Parameters
    ----------
    eia_data : pd.DataFrame
        EIA-860 battery data for presets
    engie_data : pd.DataFrame
        Engie asset data for "Current Asset" preset

    Returns
    -------
    BatterySpecs
        Battery specifications (may be new or unchanged)
    """
    state = get_state()

    # ========================================================================
    # BATTERY PRESET SELECTOR
    # ========================================================================
    if eia_data is not None and not eia_data.empty:
        preset_options = ["Custom", "Current Asset"] + [v['name'] for v in BATTERY_PRESETS.values()]

        # Initialize battery preset in session state if not present
        if 'battery_preset_value' not in st.session_state:
            st.session_state.battery_preset_value = "Custom"

        # Find the index for the stored preset value
        try:
            preset_index = preset_options.index(st.session_state.battery_preset_value)
        except (ValueError, AttributeError):
            preset_index = 0  # Default to "Custom"
            st.session_state.battery_preset_value = "Custom"

        # Callback to persist selection
        def on_preset_change():
            st.session_state.battery_preset_value = st.session_state.battery_preset_widget

        battery_preset = st.sidebar.selectbox(
            "Battery System Preset:",
            preset_options,
            index=preset_index,
            key="battery_preset_widget",
            on_change=on_preset_change,
            help="Select a preset based on real Texas battery systems (EIA-860 data) or the currently selected asset."
        )

        # Ensure our persistent value is in sync
        st.session_state.battery_preset_value = battery_preset

        # Determine default values based on preset
        default_capacity, default_power = _get_preset_values(
            battery_preset, state, eia_data, engie_data
        )
    else:
        # No EIA data available - force Custom mode
        if 'battery_preset_value' not in st.session_state:
            st.session_state.battery_preset_value = "Custom"
        battery_preset = "Custom"
        default_capacity = DEFAULT_BATTERY['capacity_mwh']
        default_power = DEFAULT_BATTERY['power_mw']

    # Clamp values to valid ranges
    default_capacity = max(1, min(1000, default_capacity))
    default_power = max(1, min(500, default_power))

    # ========================================================================
    # TRACK PRESET CHANGES
    # ========================================================================
    if 'last_battery_preset' not in st.session_state:
        st.session_state.last_battery_preset = st.session_state.get('battery_preset_value', 'Custom')

    # Track node changes for Current Asset updates
    if 'last_selected_node' not in st.session_state:
        st.session_state.last_selected_node = state.selected_node

    preset_changed = (st.session_state.last_battery_preset !=
                      st.session_state.get('battery_preset_value', 'Custom'))
    node_changed = st.session_state.last_selected_node != state.selected_node

    if preset_changed:
        st.session_state.last_battery_preset = st.session_state.get('battery_preset_value', 'Custom')
    if node_changed:
        st.session_state.last_selected_node = state.selected_node

    # Determine if we should force update from preset/defaults
    should_update_defaults = preset_changed or (
        st.session_state.get('battery_preset_value', 'Custom') == "Current Asset" and node_changed
    )

    # Update synced widgets if preset changed
    if should_update_defaults:
        update_synced_value("capacity", float(default_capacity))
        update_synced_value("power", float(default_power))

    # ========================================================================
    # BATTERY CAPACITY SLIDER
    # ========================================================================
    is_disabled = battery_preset != "Custom"
    
    st.sidebar.markdown("**Battery Parameters:**")

    # Determine initial value
    if should_update_defaults:
        init_capacity = float(default_capacity)
    elif state.battery_specs is not None:
        init_capacity = float(state.battery_specs.capacity_mwh)
    else:
        init_capacity = float(default_capacity)

    capacity = render_synced_slider_input(
        label="Energy Capacity (MWh):",
        min_val=5.0,
        max_val=1000.0,
        key_prefix="capacity",
        default_value=init_capacity,
        slider_step=10.0,
        input_step=10.0,
        format_str="%.1f",
        help_text="Total energy storage capacity of the battery",
        disabled=is_disabled
    )

    # ========================================================================
    # BATTERY POWER SLIDER
    # ========================================================================
    # Determine initial value
    if should_update_defaults:
        init_power = float(default_power)
    elif state.battery_specs is not None:
        init_power = float(state.battery_specs.power_mw)
    else:
        init_power = float(default_power)

    power = render_synced_slider_input(
        label="Power Capacity (MW):",
        min_val=5.0,
        max_val=500.0,
        key_prefix="power",
        default_value=init_power,
        slider_step=5.0,
        input_step=5.0,
        format_str="%.1f",
        help_text="Maximum charge/discharge rate",
        disabled=is_disabled
    )

    # ========================================================================
    # BATTERY EFFICIENCY SLIDER
    # ========================================================================
    # Determine initial value (efficiency doesn't change with presets)
    if state.battery_specs is not None:
        init_efficiency = float(state.battery_specs.efficiency)
    else:
        init_efficiency = DEFAULT_BATTERY['efficiency']

    efficiency = render_synced_slider_input(
        label="Round-trip Efficiency:",
        min_val=0.70,
        max_val=1.00,
        key_prefix="efficiency",
        default_value=init_efficiency,
        slider_step=0.05,
        input_step=0.01,
        format_str="%.2f",
        help_text="Energy efficiency for charge/discharge cycle (100% = theoretical perfect efficiency)"
    )

    # ========================================================================
    # CREATE BATTERY SPECS
    # ========================================================================
    new_specs = BatterySpecs(
        capacity_mwh=capacity,
        power_mw=power,
        efficiency=efficiency
    )

    return new_specs


def _get_preset_values(
    battery_preset: str,
    state,
    eia_data: pd.DataFrame,
    engie_data: pd.DataFrame
) -> tuple[int, int]:
    """
    Get capacity and power values based on preset selection.

    Parameters
    ----------
    battery_preset : str
        Selected preset name
    state : AppState
        Application state
    eia_data : pd.DataFrame
        EIA battery data
    engie_data : pd.DataFrame
        Engie asset data

    Returns
    -------
    tuple[int, int]
        (capacity_mwh, power_mw)
    """
    if battery_preset == "Custom":
        return DEFAULT_BATTERY['capacity_mwh'], DEFAULT_BATTERY['power_mw']

    elif battery_preset == "Current Asset":
        # Load Engie assets and match on node
        if not engie_data.empty and state.selected_node:
            asset_match = engie_data[engie_data['settlement_point'] == state.selected_node]

            if not asset_match.empty:
                # Use the first match
                asset = asset_match.iloc[0]
                power = int(asset['nameplate_power_mw']) if pd.notna(
                    asset['nameplate_power_mw']) else DEFAULT_BATTERY['power_mw']
                capacity = int(asset['nameplate_energy_mwh']) if pd.notna(
                    asset['nameplate_energy_mwh']) else DEFAULT_BATTERY['capacity_mwh']

                st.sidebar.success(
                    f"✅ Matched: {asset['plant_name']} ({power} MW / {capacity} MWh)")
                return capacity, power
            else:
                st.sidebar.warning(f"⚠️ No asset found for {state.selected_node}")
        else:
            st.sidebar.warning("⚠️ No asset data available or node not selected")
        
        return DEFAULT_BATTERY['capacity_mwh'], DEFAULT_BATTERY['power_mw']

    elif battery_preset == BATTERY_PRESETS['Small']['name']:
        return BATTERY_PRESETS['Small']['capacity_mwh'], BATTERY_PRESETS['Small']['power_mw']

    elif battery_preset == BATTERY_PRESETS['Medium']['name']:
        return BATTERY_PRESETS['Medium']['capacity_mwh'], BATTERY_PRESETS['Medium']['power_mw']

    elif battery_preset == BATTERY_PRESETS['Large']['name']:
        capacity = int(eia_data['Nameplate Energy Capacity (MWh)'].quantile(0.9))
        power = int(eia_data['Nameplate Capacity (MW)'].quantile(0.9))
        return capacity, power

    elif battery_preset == BATTERY_PRESETS['Very Large']['name']:
        return BATTERY_PRESETS['Very Large']['capacity_mwh'], BATTERY_PRESETS['Very Large']['power_mw']

    else:
        # Fallback for any unrecognized preset
        return DEFAULT_BATTERY['capacity_mwh'], DEFAULT_BATTERY['power_mw']

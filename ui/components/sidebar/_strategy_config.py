"""
Strategy Configuration UI
Zentus - ERCOT Battery Revenue Dashboard

This module handles dispatch strategy selection and parameters.
"""

import streamlit as st

from utils.state import clear_simulation_cache, get_state, update_state

from ._widget_sync import render_synced_slider_input


def render_strategy_config() -> dict:
    """
    Render strategy selection and parameters.

    Returns
    -------
    dict
        Dictionary with keys: 'type' (str), 'params_changed' (bool)
    """
    state = get_state()

    def on_strategy_change():
        """Callback for strategy change"""
        if "strategy_radio" in st.session_state:
            new_strategy = st.session_state.strategy_radio
            update_state(strategy_type=new_strategy)
            clear_simulation_cache()

    strategy_type = st.sidebar.radio(
        "Battery Trading Strategy:",
        options=["Threshold-Based", "Rolling Window Optimization", "MPC (Rolling Horizon)"],
        index={
            "Threshold-Based": 0,
            "Rolling Window Optimization": 1,
            "MPC (Rolling Horizon)": 2
        }.get(state.strategy_type, 2),  # DEFAULT: MPC
        help="Choose the battery dispatch strategy. Linear Programming is used as a theoretical benchmark (see Opportunity page).",
        key="strategy_radio",
        on_change=on_strategy_change
    )

    # Strategy-specific parameters
    if strategy_type == "Threshold-Based":
        st.sidebar.markdown("**Threshold Parameters:**")

        # Callback for threshold changes
        def on_threshold_change():
            clear_simulation_cache()

        # Charge threshold slider
        charge_pct = st.sidebar.slider(
            "Charge Threshold Percentile:",
            min_value=10,
            max_value=40,
            value=int(state.charge_percentile * 100),
            step=5,
            help="Charge when price below this percentile",
            key="charge_slider",
            on_change=lambda: _update_charge_threshold(on_threshold_change)
        )
        
        # Update state if changed
        new_charge = charge_pct / 100
        if abs(new_charge - state.charge_percentile) > 0.001:
            update_state(charge_percentile=new_charge)

        # Discharge threshold slider
        discharge_pct = st.sidebar.slider(
            "Discharge Threshold Percentile:",
            min_value=60,
            max_value=90,
            value=int(state.discharge_percentile * 100),
            step=5,
            help="Discharge when price above this percentile",
            key="discharge_slider",
            on_change=lambda: _update_discharge_threshold(on_threshold_change)
        )
        
        # Update state if changed
        new_discharge = discharge_pct / 100
        if abs(new_discharge - state.discharge_percentile) > 0.001:
            update_state(discharge_percentile=new_discharge)

    elif strategy_type == "Rolling Window Optimization":
        st.sidebar.markdown("**Optimization Parameters:**")

        # Callback for window changes
        def on_window_change():
            new_val = st.session_state.window_slider
            if new_val != state.window_hours:
                update_state(window_hours=new_val)
                clear_simulation_cache()

        # Lookahead window slider
        st.sidebar.slider(
            "Lookahead Window (hours):",
            min_value=2,
            max_value=24,
            value=state.window_hours if hasattr(state, 'window_hours') else 12,
            step=1,
            help="Number of hours to look ahead for optimization",
            key="window_slider",
            on_change=on_window_change
        )

    elif strategy_type == "MPC (Rolling Horizon)":
        st.sidebar.markdown("**MPC Parameters:**")

        # Callback for horizon changes
        def on_horizon_change():
            new_val = st.session_state.mpc_horizon_slider
            if not hasattr(state, 'horizon_hours') or new_val != state.horizon_hours:
                update_state(horizon_hours=new_val)
                clear_simulation_cache()

        # Optimization horizon slider
        st.sidebar.slider(
            "Optimization Horizon (hours):",
            min_value=2,
            max_value=24,
            value=state.horizon_hours if hasattr(state, 'horizon_hours') else 6,
            step=1,
            help="Lookahead horizon for each optimization step.",
            key="mpc_horizon_slider",
            on_change=on_horizon_change
        )

    return {'type': strategy_type, 'params_changed': False}


def _update_charge_threshold(callback):
    """Helper to update charge threshold"""
    state = get_state()
    new_pct = st.session_state.charge_slider / 100
    if abs(new_pct - state.charge_percentile) > 0.001:
        update_state(charge_percentile=new_pct)
        callback()


def _update_discharge_threshold(callback):
    """Helper to update discharge threshold"""
    state = get_state()
    new_pct = st.session_state.discharge_slider / 100
    if abs(new_pct - state.discharge_percentile) > 0.001:
        update_state(discharge_percentile=new_pct)
        callback()


def render_forecast_config() -> float:
    """
    Render forecast improvement slider and input.

    Returns
    -------
    float
        Forecast improvement percentage (0-100)
    """
    st.sidebar.markdown("**Forecast Scenario:**")

    # Use the synced widget utility
    forecast_improvement = render_synced_slider_input(
        label="Forecast Accuracy (%):",
        min_val=0.0,
        max_val=100.0,
        key_prefix="forecast",
        default_value=float(get_state().forecast_improvement),
        slider_step=5.0,
        input_step=0.1,
        format_str="%.1f",
        help_text="% of the forecast error to correct (0% = DA only, 100% = perfect RT knowledge)",
        on_change_callback=lambda: _update_forecast_state()
    )

    return forecast_improvement


def _update_forecast_state():
    """Update forecast improvement in state and clear cache"""
    state = get_state()
    new_val = st.session_state.get('forecast_master', state.forecast_improvement)
    if abs(new_val - state.forecast_improvement) > 0.01:
        update_state(forecast_improvement=new_val)
        clear_simulation_cache()

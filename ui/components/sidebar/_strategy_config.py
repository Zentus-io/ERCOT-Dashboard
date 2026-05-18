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

    # Pass index= matching state AND key=. Streamlit uses session_state[key]
    # when present (so user clicks win) and falls back to index= when the
    # key was GC'd on cross-page navigation (so we restore state on the
    # next page). Force-syncing session_state[key] before render would
    # instead OVERWRITE the user's brand-new click.
    _strategy_options = ["Threshold-Based", "Rolling Window Optimization", "MPC (Rolling Horizon)"]
    _state_strategy = state.strategy_type if state.strategy_type in _strategy_options else "MPC (Rolling Horizon)"
    _state_strategy_index = _strategy_options.index(_state_strategy)
    strategy_type = st.sidebar.radio(
        "Battery Trading Strategy:",
        options=_strategy_options,
        index=_state_strategy_index,
        help="Choose the battery dispatch strategy. Linear Programming is used as a theoretical benchmark (see Opportunity page).",
        key="strategy_radio",
    )
    if strategy_type != state.strategy_type:
        update_state(strategy_type=strategy_type)
        clear_simulation_cache()

    # Strategy-specific parameters
    if strategy_type == "Threshold-Based":
        st.sidebar.markdown("**Threshold Parameters:**")

        # Callback for threshold changes
        def on_threshold_change():
            clear_simulation_cache()

        charge_pct = st.sidebar.slider(
            "Charge Threshold Percentile:",
            min_value=10,
            max_value=40,
            value=int(state.charge_percentile * 100),
            step=5,
            help="Charge when price below this percentile",
            key="charge_slider",
        )
        new_charge = charge_pct / 100
        if abs(new_charge - state.charge_percentile) > 0.001:
            update_state(charge_percentile=new_charge)
            clear_simulation_cache()

        discharge_pct = st.sidebar.slider(
            "Discharge Threshold Percentile:",
            min_value=60,
            max_value=90,
            value=int(state.discharge_percentile * 100),
            step=5,
            help="Discharge when price above this percentile",
            key="discharge_slider",
        )
        
        # Update state if changed
        new_discharge = discharge_pct / 100
        if abs(new_discharge - state.discharge_percentile) > 0.001:
            update_state(discharge_percentile=new_discharge)

    elif strategy_type == "Rolling Window Optimization":
        st.sidebar.markdown("**Optimization Parameters:**")

        window_hours = st.sidebar.slider(
            "Lookahead Window (hours):",
            min_value=2,
            max_value=24,
            value=state.window_hours if hasattr(state, "window_hours") else 12,
            step=1,
            help="Number of hours to look ahead for optimization",
            key="window_slider",
        )
        if window_hours != state.window_hours:
            update_state(window_hours=window_hours)
            clear_simulation_cache()

    elif strategy_type == "MPC (Rolling Horizon)":
        st.sidebar.markdown("**MPC Parameters:**")

        horizon_hours = st.sidebar.slider(
            "Optimization Horizon (hours):",
            min_value=2,
            max_value=24,
            value=state.horizon_hours if hasattr(state, "horizon_hours") else 6,
            step=1,
            help="Lookahead horizon for each optimization step.",
            key="mpc_horizon_slider",
        )
        if horizon_hours != state.horizon_hours:
            update_state(horizon_hours=horizon_hours)
            clear_simulation_cache()

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

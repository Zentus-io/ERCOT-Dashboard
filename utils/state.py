"""
Session State Management
Zentus - ERCOT Battery Revenue Dashboard

This module provides centralized session state management for the Streamlit app.
"""

import streamlit as st
from dataclasses import dataclass, field
from typing import Optional, Dict
from core.battery.battery import BatterySpecs
from core.battery.simulator import SimulationResult


@dataclass
class AppState:
    """
    Centralized application state.

    This dataclass holds all persistent state across page navigation.

    Attributes
    ----------
    battery_specs : BatterySpecs, optional
        Current battery system specifications
    strategy_type : str
        Selected dispatch strategy ("Threshold-Based" or "Rolling Window Optimization")
    charge_percentile : float
        Charge threshold percentile (0 to 1)
    discharge_percentile : float
        Discharge threshold percentile (0 to 1)
    window_hours : int
        Rolling window lookahead hours
    forecast_improvement : int
        Forecast improvement percentage (0 to 100)
    selected_node : str, optional
        Currently selected settlement point
    simulation_results : dict
        Cached simulation results for each scenario
    """
    # Battery configuration
    battery_specs: Optional[BatterySpecs] = None

    # Strategy settings
    strategy_type: str = "Threshold-Based"
    charge_percentile: float = 0.25
    discharge_percentile: float = 0.75
    window_hours: int = 6
    forecast_improvement: int = 10

    # Data selection
    selected_node: Optional[str] = None

    # Simulation results (cached)
    simulation_results: Dict[str, Optional[SimulationResult]] = field(default_factory=lambda: {
        'baseline': None,
        'improved': None,
        'optimal': None
    })

    # Data caches
    price_data: Optional[object] = None
    eia_battery_data: Optional[object] = None
    available_nodes: Optional[list] = None


def init_state():
    """
    Initialize session state on first load.

    This should be called at the start of app.py and each page.
    """
    if 'app_state' not in st.session_state:
        st.session_state.app_state = AppState()


def get_state() -> AppState:
    """
    Get current application state.

    Returns
    -------
    AppState
        Current state object
    """
    if 'app_state' not in st.session_state:
        init_state()
    return st.session_state.app_state


def update_state(**kwargs):
    """
    Update state attributes.

    Parameters
    ----------
    **kwargs
        Attribute name-value pairs to update
    """
    state = get_state()
    for key, value in kwargs.items():
        if hasattr(state, key):
            setattr(state, key, value)


def clear_simulation_cache():
    """
    Clear cached simulation results when configuration changes.

    Call this when battery specs, strategy, or other parameters change.
    """
    state = get_state()
    state.simulation_results = {
        'baseline': None,
        'improved': None,
        'optimal': None
    }


def has_valid_config() -> bool:
    """
    Check if app has valid configuration to run simulations.

    Returns
    -------
    bool
        True if battery_specs and selected_node are configured
    """
    state = get_state()
    return (
        state.battery_specs is not None and
        state.selected_node is not None
    )


def get_node_data():
    """
    Get filtered price data for selected node.

    Returns
    -------
    pd.DataFrame or None
        Filtered price data, or None if not available
    """
    state = get_state()
    if state.price_data is None or state.selected_node is None:
        return None

    return state.price_data[state.price_data['node'] == state.selected_node].copy()

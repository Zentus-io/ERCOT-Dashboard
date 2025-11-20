"""
Session State Management
Zentus - ERCOT Battery Revenue Dashboard

This module provides centralized session state management for the Streamlit app.
"""

import streamlit as st
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from datetime import date, timedelta
from core.battery.battery import BatterySpecs
from core.battery.simulator import SimulationResult
from config.settings import DEFAULT_DATA_SOURCE, DEFAULT_DAYS_BACK, MAX_DAYS_RANGE


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
    start_date : date
        Analysis start date (for database queries)
    end_date : date
        Analysis end date (for database queries)
    data_source : str
        Data source ('csv' or 'database')
    simulation_results : dict
        Cached simulation results for each scenario
    price_data : pd.DataFrame, optional
        Cached price data
    eia_battery_data : pd.DataFrame, optional
        Cached EIA battery market data
    available_nodes : list, optional
        List of available settlement points
    available_date_range : tuple, optional
        Available date range from database (earliest, latest)
    _data_cache_key : str, optional
        Cache key tracking what data is currently loaded
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

    # Date range selection (for database mode)
    start_date: date = field(default_factory=lambda: date.today() - timedelta(days=DEFAULT_DAYS_BACK))
    end_date: date = field(default_factory=date.today)
    data_source: str = DEFAULT_DATA_SOURCE  # 'csv' or 'database'

    # Simulation results (cached)
    simulation_results: Dict[str, Optional[SimulationResult]] = field(default_factory=lambda: {
        'baseline': None,
        'improved': None,
        'optimal': None
    })
    # Data caches
    price_data: Optional[pd.DataFrame] = None
    eia_battery_data: Optional[pd.DataFrame] = None
    available_nodes: Optional[list] = None
    available_date_range: Optional[Tuple[date, date]] = None  # (earliest, latest) from database

    # Cache tracking
    _data_cache_key: Optional[str] = None  # Track what data is currently cached



def init_state():
    """
    Initialize session state on first load.

    This should be called at the start of app.py and each page.
    Handles migration from old state schema to new schema.
    """
    if 'app_state' not in st.session_state:
        st.session_state.app_state = AppState()
    else:
        # Migrate old state to new schema if needed
        state = st.session_state.app_state

        # Add new date range fields if missing (backward compatibility)
        if not hasattr(state, 'start_date'):
            state.start_date = date.today() - timedelta(days=DEFAULT_DAYS_BACK)
        if not hasattr(state, 'end_date'):
            state.end_date = date.today()
        if not hasattr(state, 'data_source'):
            state.data_source = DEFAULT_DATA_SOURCE
        if not hasattr(state, 'available_date_range'):
            state.available_date_range = None
        if not hasattr(state, '_data_cache_key'):
            state._data_cache_key = None


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
    Does NOT clear price_data cache - use clear_data_cache() for that.
    """
    state = get_state()
    state.simulation_results = {
        'baseline': None,
        'improved': None,
        'optimal': None
    }


def clear_data_cache():
    """
    Clear cached data when data source or date range changes.

    Call this when switching data sources or selecting new date range.
    """
    state = get_state()
    state.price_data = None
    state._data_cache_key = None
    clear_simulation_cache()  # Also clear simulations since data changed


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


def get_cache_key(node: Optional[str] = None, start_date: Optional[date] = None,
                  end_date: Optional[date] = None, source: Optional[str] = None) -> str:
    """
    Generate cache key for current data configuration.

    Parameters
    ----------
    node : str, optional
        Settlement point name (uses state if None)
    start_date : date, optional
        Start date (uses state if None)
    end_date : date, optional
        End date (uses state if None)
    source : str, optional
        Data source (uses state if None)

    Returns
    -------
    str
        Cache key string
    """
    state = get_state()
    node = node or state.selected_node or "all"
    start_date = start_date or state.start_date
    end_date = end_date or state.end_date
    source = source or state.data_source

    return f"{source}_{node}_{start_date}_{end_date}"


def needs_data_reload() -> bool:
    """
    Check if data needs reloading based on current state.

    Returns
    -------
    bool
        True if data should be reloaded
    """
    state = get_state()

    # No data cached
    if state.price_data is None:
        return True

    # Cache key doesn't match current configuration
    current_key = get_cache_key()
    if state._data_cache_key != current_key:
        return True

    return False


def update_date_range(start_date: date, end_date: date):
    """
    Update date range and mark cache for reload.

    Parameters
    ----------
    start_date : date
        New start date
    end_date : date
        New end date

    Raises
    ------
    ValueError
        If date range is invalid
    """
    # Validate date range
    if end_date < start_date:
        raise ValueError("End date must be after start date")

    date_diff = (end_date - start_date).days
    if date_diff > MAX_DAYS_RANGE:
        raise ValueError(f"Date range cannot exceed {MAX_DAYS_RANGE} days (selected: {date_diff} days)")

    # Update state
    state = get_state()
    state.start_date = start_date
    state.end_date = end_date

    # Clear data cache if date range actually changed
    if needs_data_reload():
        clear_data_cache()


def get_date_range_str(data: Optional[pd.DataFrame] = None) -> str:
    """
    Get formatted date range string from data or state.

    Parameters
    ----------
    data : pd.DataFrame, optional
        DataFrame with 'timestamp' column. If None, uses state dates.

    Returns
    -------
    str
        Formatted date range string (e.g., "2025-07-20" or "2025-07-13 to 2025-07-20")
    """
    if data is not None and 'timestamp' in data.columns and len(data) > 0:
        start = data['timestamp'].min().date()
        end = data['timestamp'].max().date()
    else:
        state = get_state()
        start = state.start_date
        end = state.end_date

    if start == end:
        return str(start)
    return f"{start} to {end}"

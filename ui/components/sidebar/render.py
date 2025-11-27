"""
Main Sidebar Render Function
Zentus - ERCOT Battery Revenue Dashboard

This module orchestrates all sidebar components into the main render_sidebar() function.
"""

from datetime import date, timedelta

import streamlit as st

from config.settings import DEFAULT_DAYS_BACK, MAX_DAYS_RANGE
from core.data.loaders import SupabaseDataLoader, UploadedFileLoader
from postgrest.exceptions import APIError
from utils.state import (
    clear_simulation_cache,
    get_cache_key,
    get_state,
    needs_data_reload,
    update_date_range,
    update_state,
)

from ._battery_config import render_battery_config
from ._data_loaders import load_eia_data, load_engie_data, load_node_list, load_node_prices
from ._data_source import render_data_source_selector, render_file_upload
from ._date_utils import validate_date_range
from ._node_selection import handle_node_change, render_node_selector
from ._strategy_config import render_forecast_config, render_strategy_config


def render_sidebar():
    """
    Render configuration sidebar.

    This sidebar is shared across all pages and manages:
    - Data source selection (CSV/Database)
    - Date range selection (Database mode)
    - Settlement point selection
    - Battery specifications
    - Dispatch strategy configuration
    - Forecast improvement settings
    """
    state = get_state()

    st.sidebar.header("Analysis Configuration")

    # ========================================================================
    # 1. DATA SOURCE SELECTION
    # ========================================================================
    data_source = render_data_source_selector()

   # Show file upload UI if Local Parquet is selected
    if data_source == 'local_parquet':
        render_file_upload()

    # ========================================================================
    # 2. DATE RANGE SELECTION
    # ========================================================================
    _render_date_range_selector(state, data_source)

    # ========================================================================
    # 3. LOAD NODE LIST
    # ========================================================================
    nodes = load_node_list(source=state.data_source)
    state.available_nodes = nodes

    # ========================================================================
    # 4. NODE SELECTION
    # ========================================================================
    selected_node = render_node_selector(nodes)
    if selected_node != state.selected_node:
        handle_node_change(state.selected_node, selected_node)

    # ========================================================================
    # 5. LOAD PRICE DATA
    # ========================================================================
    if needs_data_reload() or state.price_data is None or state.price_data.empty or (
            state.selected_node and 'node' in state.price_data.columns and state.price_data['node'].iloc[0] != state.selected_node):
        if state.selected_node:
            with st.spinner(f'Loading data for {state.selected_node}...'):
                # Determine actual source to use
                actual_source = state.data_source
                file_sig = None

                if state.data_source == 'local_parquet':
                    actual_source = 'uploaded'
                    # Generate signature to invalidate cache when files change
                    if state.uploaded_dam_file and state.uploaded_rtm_file:
                        dam_name = state.uploaded_dam_file.name  # type: ignore[attr-defined]
                        dam_size = state.uploaded_dam_file.size  # type: ignore[attr-defined]
                        rtm_name = state.uploaded_rtm_file.name  # type: ignore[attr-defined]
                        rtm_size = state.uploaded_rtm_file.size  # type: ignore[attr-defined]
                        file_sig = f"{dam_name}_{dam_size}_{rtm_name}_{rtm_size}"

                price_data = load_node_prices(
                    source=actual_source,
                    node=state.selected_node,
                    start_date=state.start_date,
                    end_date=state.end_date,
                    file_signature=file_sig
                )
                state.price_data = price_data

                # Load EIA data once
                if state.eia_battery_data is None:
                    state.eia_battery_data = load_eia_data()

                state._data_cache_key = get_cache_key()

    # ========================================================================
    # 6. DISPATCH STRATEGY CONFIGURATION
    # ========================================================================
    st.sidebar.markdown("---")
    st.sidebar.subheader("Dispatch Strategy")
    render_strategy_config()
    forecast_improvement = render_forecast_config()

    # Update forecast in state if changed 
    if abs(forecast_improvement - state.forecast_improvement) > 0.01:
        update_state(forecast_improvement=forecast_improvement)

    # ========================================================================
    # 7. BATTERY SPECIFICATIONS
    # ========================================================================
    st.sidebar.markdown("---")
    st.sidebar.subheader("Battery Specifications")
    
    new_specs = render_battery_config(
        eia_data=load_eia_data(),
        engie_data=load_engie_data()
    )

    if state.battery_specs != new_specs:
        update_state(battery_specs=new_specs)
        clear_simulation_cache()


def _render_date_range_selector(state, data_source):
    """
    Render date range selection UI (database or local parquet mode).

    Parameters
    ----------
    state : AppState
        Application state
    data_source : str
        Current data source
    """
    if data_source == 'database':
        st.sidebar.markdown("---")
        st.sidebar.subheader("Date Range")

        # Get available date range from database if not cached
        if state.available_date_range is None:
            try:
                db_loader = SupabaseDataLoader()
                date_range = db_loader.get_date_range()
                # Validate that both dates are not None
                if date_range[0] is not None and date_range[1] is not None:
                    state.available_date_range = (date_range[0], date_range[1])
                else:
                    state.available_date_range = (
                        date.today() - timedelta(days=DEFAULT_DAYS_BACK),
                        date.today()
                    )
            except APIError as e:
                st.sidebar.error(f"⚠️ Cannot fetch date range: {str(e)}")
                state.available_date_range = (
                    date.today() - timedelta(days=DEFAULT_DAYS_BACK),
                    date.today()
                )

        earliest, latest = state.available_date_range or (
            date.today() - timedelta(days=DEFAULT_DAYS_BACK),
            date.today()
        )

        # Ensure state dates are within available range
        if earliest and state.start_date < earliest:
            state.start_date = earliest
        if latest and state.start_date > latest:
            state.start_date = latest
        if earliest and state.end_date < earliest:
            state.end_date = earliest
        if latest and state.end_date > latest:
            state.end_date = latest

        # Date range picker
        date_range_input = st.sidebar.date_input(
            "Select Date Range",
            value=(state.start_date, state.end_date),
            min_value=earliest,
            max_value=latest,
            help=f"Data available from {earliest} to {latest}"
        )

        # Handle range selection
        if len(date_range_input) == 2:
            start_date, end_date = date_range_input
        else:
            # Handle case where user is still selecting (only one date picked)
            start_date, end_date = state.start_date, state.end_date

        # Validate and update date range
        is_valid, error_msg = validate_date_range(start_date, end_date)
        if not is_valid:
            st.sidebar.error(error_msg)
        elif start_date != state.start_date or end_date != state.end_date:
            # User manually changed dates - update and mark as manual selection
            try:
                update_date_range(start_date, end_date)
                state._dates_auto_selected = False  # User manually set dates
            except ValueError as e:
                st.sidebar.error(f"⚠️ {str(e)}")

        # Show date range summary
        days_selected = (state.end_date - state.start_date).days + 1
        auto_indicator = " (auto-selected)" if state._dates_auto_selected else ""
        st.sidebar.caption(f"📅 **{days_selected} days** selected{auto_indicator}")

    elif data_source == 'local_parquet':
        st.sidebar.markdown("---")
        st.sidebar.subheader("Date Range")

        # Get date range from uploaded files
        try:
            loader = UploadedFileLoader(state.uploaded_dam_file, state.uploaded_rtm_file)
            date_range = loader.get_date_range()

            if date_range[0] is not None and date_range[1] is not None:
                parquet_earliest, parquet_latest = date_range
            else:
                parquet_earliest = date(2025, 1, 1)
                parquet_latest = date(2025, 12, 31)
        except BaseException:
            parquet_earliest = date(2025, 1, 1)
            parquet_latest = date(2025, 12, 31)

        st.sidebar.info(f"📌 Data available: {parquet_earliest} to {parquet_latest}")

        # Ensure state dates are within parquet range
        if parquet_earliest and state.start_date < parquet_earliest:
            state.start_date = parquet_earliest
        if parquet_latest and state.start_date > parquet_latest:
            state.start_date = parquet_latest
        if parquet_earliest and state.end_date < parquet_earliest:
            state.end_date = parquet_earliest
        if parquet_latest and state.end_date > parquet_latest:
            state.end_date = parquet_latest

        # Date range picker
        date_range_input = st.sidebar.date_input(
            "Select Date Range",
            value=(state.start_date, state.end_date),
            min_value=parquet_earliest,
            max_value=parquet_latest,
            help="Filter parquet data by date range"
        )

        # Handle range selection
        if len(date_range_input) == 2:
            start_date, end_date = date_range_input
        else:
            start_date, end_date = state.start_date, state.end_date

        # Validate and update date range
        is_valid, error_msg = validate_date_range(start_date, end_date)
        if not is_valid:
            st.sidebar.error(error_msg)
        elif start_date != state.start_date or end_date != state.end_date:
            try:
                update_date_range(start_date, end_date)
            except ValueError as e:
                st.sidebar.error(f"⚠️ {str(e)}")

        # Show date range summary
        days_selected = (state.end_date - state.start_date).days + 1
        st.sidebar.caption(f"📅 **{days_selected} days** selected")

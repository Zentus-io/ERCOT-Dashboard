"""
Sidebar Configuration Component
Zentus - ERCOT Battery Revenue Dashboard

This component renders the shared configuration sidebar
that persists across all pages.
"""

import streamlit as st
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
from core.battery.battery import BatterySpecs
from core.data.loaders import SupabaseDataLoader, UploadedFileLoader, ParquetDataLoader
from utils.state import (
    get_state, update_state, clear_simulation_cache, needs_data_reload,
    get_cache_key, update_date_range, get_date_range_str, clear_data_cache
)
from config.settings import (
    DEFAULT_BATTERY, BATTERY_PRESETS, SUPABASE_URL, SUPABASE_KEY,
    DEFAULT_DAYS_BACK, MAX_DAYS_RANGE, DEFAULT_STRATEGY
)


@st.cache_data(ttl=3600, show_spinner="Loading node list...")
def load_node_list(source: str = 'csv'):
    """
    Load available settlement points.
    """
    data_dir = Path(__file__).parent.parent.parent / 'data'
    
    if source == 'database':
        try:
            db_loader = SupabaseDataLoader()
            nodes = db_loader.get_available_nodes()
            if nodes:
                return nodes
        except Exception as e:
            st.warning(f"⚠️ Database error fetching nodes: {str(e)}")
            
    elif source == 'local_parquet':
        try:
            state = get_state()
            if state.uploaded_dam_file and state.uploaded_rtm_file:
                loader = UploadedFileLoader(state.uploaded_dam_file, state.uploaded_rtm_file)
                nodes = loader.get_available_nodes()
                if nodes:
                    return nodes
            return []
        except Exception as e:
            st.warning(f"⚠️ Error fetching nodes from uploaded files: {str(e)}")

    # Fallback if DB/Parquet failed or empty
    return []


# @st.cache_data(ttl=3600, max_entries=10, show_spinner="Loading price data...")
def load_node_prices(
    source: str,
    node: str,
    start_date: date | None = None,
    end_date: date | None = None,
    file_signature: str | None = None
):
    """
    Load price data for a specific node.
    """
    data_dir = Path(__file__).parent.parent.parent / 'data'

    if source == 'database':
        try:
            db_loader = SupabaseDataLoader()
            return db_loader.load_prices(node=node, start_date=start_date, end_date=end_date)
        except Exception as e:
            st.warning(f"⚠️ Database error: {str(e)}. Falling back to CSV.")
            
    elif source == 'local_parquet':
        try:
            # Import here to avoid circular imports if any, or just use the one from loaders
            # But we need to make sure ParquetDataLoader is available. 
            # It is imported from core.data.loaders in the file header? 
            # No, only DataLoader and SupabaseDataLoader are imported.
            # Need to fix imports too.
            from core.data.loaders import ParquetDataLoader
            loader = ParquetDataLoader(data_dir)
            return loader.load_prices(node=node)
        except Exception as e:
            st.warning(f"⚠️ Parquet error: {str(e)}. Falling back to CSV.")

    elif source == 'uploaded':
        try:
            state = get_state()
            if state.uploaded_dam_file and state.uploaded_rtm_file:
                loader = UploadedFileLoader(state.uploaded_dam_file, state.uploaded_rtm_file)
                # Apply date filtering if specified
                df = loader.load_prices(node=node)
                
                if not df.empty and (start_date or end_date):
                    if start_date:
                        df = df[df['timestamp'].dt.date >= start_date]  # type: ignore
                    if end_date:
                        df = df[df['timestamp'].dt.date <= end_date]  # type: ignore
                return df
            return pd.DataFrame()
        except Exception as e:
            st.warning(f"⚠️ Upload error: {str(e)}")


    # Fallback
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_eia_data():
    try:
        return SupabaseDataLoader().load_eia_batteries()
    except Exception as e:
        st.warning(f"Could not load battery data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_engie_data():
    try:
        return SupabaseDataLoader().load_engie_assets()
    except Exception as e:
        st.warning(f"Could not load Engie asset data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def find_best_date_range(node: str, min_completeness: float = 95.0):
    """
    Find the longest contiguous date range with high data availability.

    Args:
        node: Settlement point name
        min_completeness: Minimum completeness threshold (default 95%)

    Returns:
        tuple: (start_date, end_date) or None if no suitable range found
    """
    try:
        db_loader = SupabaseDataLoader()
        availability_df = db_loader.get_node_availability(node)

        if availability_df.empty:
            return None

        # Filter for high-quality data
        availability_df = availability_df[availability_df['completeness'] >= min_completeness].copy()

        if availability_df.empty:
            return None

        # Sort by date
        availability_df = availability_df.sort_values('date').reset_index(drop=True)

        # Convert date column to datetime.date if needed
        if not isinstance(availability_df['date'].iloc[0], date):
            availability_df['date'] = pd.to_datetime(availability_df['date']).dt.date

        # Find longest contiguous sequence
        max_length = 0
        max_start = None
        max_end = None

        current_start = availability_df['date'].iloc[0]
        current_length = 1

        for i in range(1, len(availability_df)):
            prev_date = availability_df['date'].iloc[i - 1]
            curr_date = availability_df['date'].iloc[i]

            # Check if dates are consecutive
            if (curr_date - prev_date).days == 1:
                current_length += 1
            else:
                # End of contiguous sequence
                if current_length > max_length:
                    max_length = current_length
                    max_start = current_start
                    max_end = prev_date

                # Start new sequence
                current_start = curr_date
                current_length = 1

        # Check last sequence
        if current_length > max_length:
            max_length = current_length
            max_start = current_start
            max_end = availability_df['date'].iloc[-1]

        if max_start and max_end and max_length > 0:
            # Cap the range to MAX_DAYS_RANGE if needed
            if (max_end - max_start).days > MAX_DAYS_RANGE:
                max_end = max_start + timedelta(days=MAX_DAYS_RANGE)

            return (max_start, max_end)

        return None

    except Exception as e:
        st.warning(f"⚠️ Could not auto-select date range: {str(e)}")
        return None


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
    # DATA SOURCE SELECTION
    # ========================================================================
    # ========================================================================
    # DATA SOURCE SELECTION
    # ========================================================================
    
    def on_data_source_change():
        """Callback for data source change"""
        # The new value is already in st.session_state when this runs
        # We just need to ensure our custom state object is updated
        pass

    if SUPABASE_URL and SUPABASE_KEY:
        data_source_ui = st.sidebar.radio(
            "Data Source:",
            options=['Database', 'Local Parquet'],
            index={
                'database': 0,
                'local_parquet': 1
            }.get(state.data_source, 0),
            help="**Database**: Multi-day historical data from Supabase\n**Local Parquet**: Upload your own DAM/RTM parquet files",
            key="data_source_radio",
            on_change=on_data_source_change
        )

        # Map display names to internal values
        if data_source_ui == 'Database':
            data_source_internal = 'database'
        else:
            data_source_internal = 'local_parquet'

        # Logic: Only switch to local_parquet if files are ready
        if data_source_internal == 'local_parquet':
            # Check if files are uploaded
            if state.uploaded_dam_file and state.uploaded_rtm_file:
                if state.data_source != 'local_parquet':
                    state.data_source = 'local_parquet'
                    state.using_uploaded_files = True
                    clear_data_cache()
            else:
                # Files not ready - keep previous source
                # But we still show the upload UI below because data_source_ui is 'Local Parquet'
                pass
        else:
            # For Database or CSV, switch immediately
            if data_source_internal != state.data_source:
                state.data_source = data_source_internal
                state.using_uploaded_files = False
                clear_data_cache()
    else:
        # No database credentials - force Local Parquet mode
        state.data_source = 'local_parquet'
        st.sidebar.info("📌 **Local Mode** - Configure Supabase for historical data")

    # ========================================================================
    # DATE RANGE SELECTION (Database mode only)
    # ========================================================================
    if state.data_source == 'database':
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
            except Exception as e:
                st.sidebar.error(f"⚠️ Cannot fetch date range: {str(e)}")
                state.available_date_range = (
                    date.today() - timedelta(days=DEFAULT_DAYS_BACK),
                    date.today()
                )

        date_range_to_unpack = state.available_date_range
        if not date_range_to_unpack:
            date_range_to_unpack = (
                date.today() - timedelta(days=DEFAULT_DAYS_BACK),
                date.today()
            )
        earliest, latest = date_range_to_unpack

        # Ensure state dates are within available range
        if earliest and state.start_date < earliest:
            state.start_date = earliest
        if latest and state.start_date > latest:
            state.start_date = latest
        if earliest and state.end_date < earliest:
            state.end_date = earliest
        if latest and state.end_date > latest:
            state.end_date = latest

        # Date range picker (Single input for range to save space and show full dates)
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
        if end_date < start_date:
            st.sidebar.error("⚠️ End date must be after start date")
        else:
            date_diff = (end_date - start_date).days
            if date_diff > MAX_DAYS_RANGE:
                st.sidebar.error(f"⚠️ Maximum range: {MAX_DAYS_RANGE} days (selected: {date_diff} days)")
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

    elif state.data_source == 'local_parquet':
        # Only show date picker if we are actually IN parquet mode (files uploaded)
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
        except:
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
            # Handle case where user is still selecting
            start_date, end_date = state.start_date, state.end_date

        # Validate and update date range
        if end_date < start_date:
            st.sidebar.error("⚠️ End date must be after start date")
        else:
            date_diff = (end_date - start_date).days
            if date_diff > MAX_DAYS_RANGE:
                st.sidebar.error(f"⚠️ Maximum range: {MAX_DAYS_RANGE} days (selected: {date_diff} days)")
            elif start_date != state.start_date or end_date != state.end_date:
                try:
                    update_date_range(start_date, end_date)
                except ValueError as e:
                    st.sidebar.error(f"⚠️ {str(e)}")

        # Show date range summary
        days_selected = (state.end_date - state.start_date).days + 1
        st.sidebar.caption(f"📅 **{days_selected} days** selected")

    elif state.data_source == 'csv':
        # Should not happen with new logic, but safe fallback
        st.sidebar.warning("⚠️ CSV Demo Mode is deprecated. Please use Local Parquet.")

    # ========================================================================
    # PARQUET FILE UPLOAD (Show if UI selected Local Parquet)
    # ========================================================================
    # We use data_source_ui (the radio button value) to decide visibility
    # This allows showing the upload UI even if state.data_source is still 'csv'/'database'
    if 'data_source_ui' in locals() and data_source_ui == 'Local Parquet':
        st.sidebar.markdown("---")
        st.sidebar.subheader("📁 Custom Data Upload")
        
        st.sidebar.markdown("**Upload ERCOT Price Files**")

        def on_file_upload():
            """Callback for file uploads to ensure immediate state update"""
            # Access the file uploaders directly from session state if needed, 
            # but st.file_uploader returns the value directly.
            # We update the state object here to ensure it's fresh for the rest of the script.
            state = get_state()
            if st.session_state.get('dam_upload'):
                state.uploaded_dam_file = st.session_state.dam_upload
            if st.session_state.get('rtm_upload'):
                state.uploaded_rtm_file = st.session_state.rtm_upload
            
            # If both files are present, switch to local_parquet immediately
            if state.uploaded_dam_file and state.uploaded_rtm_file:
                state.data_source = 'local_parquet'
                state.using_uploaded_files = True
                clear_data_cache()

        dam_file = st.sidebar.file_uploader(
            "DAM Prices",
            type=['parquet', 'csv'],
            help="Day-Ahead Market price file (.parquet or .csv)",
            key='dam_upload',
            on_change=on_file_upload
        )

        rtm_file = st.sidebar.file_uploader(
            "RTM Prices",
            type=['parquet', 'csv'],
            help="Real-Time Market price file (.parquet or .csv)",
            key='rtm_upload',
            on_change=on_file_upload
        )
        
        # Update state with uploaded files (redundant with callback but safe)
        if dam_file:
            state.uploaded_dam_file = dam_file
        if rtm_file:
            state.uploaded_rtm_file = rtm_file
            
        # File Status Indicators
        if state.uploaded_dam_file:
            st.sidebar.success(f"✅ DAM: {state.uploaded_dam_file.name}")  # type: ignore
        else:
            st.sidebar.warning("⚠️ Waiting for DAM file...")
            
        if state.uploaded_rtm_file:
            st.sidebar.success(f"✅ RTM: {state.uploaded_rtm_file.name}")  # type: ignore
        else:
            st.sidebar.warning("⚠️ Waiting for RTM file...")
            
        # Global status
        if not (state.uploaded_dam_file and state.uploaded_rtm_file):
             st.sidebar.info("ℹ️ Using previous data until both files are uploaded.")

    # ========================================================================
    # LOAD DATA (Two-step process)
    # ========================================================================
    
    # 1. Load Node List
    # Determine actual source for node list
    # If pending upload, we might not have nodes yet. 
    # But if we are in 'local_parquet' mode (files uploaded), we load from files.
    # If we are pending (files missing), state.data_source is still the OLD source (csv/db).
    # So we just load from state.data_source.
    
    nodes = load_node_list(source=state.data_source)
    state.available_nodes = nodes
    
    # ========================================================================
    # NODE SELECTION
    # ========================================================================
    selected_node = st.sidebar.selectbox(
        "Select Settlement Point:",
        state.available_nodes or [],
        index=0 if (
            state.selected_node is None
            or state.available_nodes is None
            or state.selected_node not in state.available_nodes
        ) else state.available_nodes.index(state.selected_node),
        help="Choose a wind resource settlement point to analyze"
    )

    if selected_node != state.selected_node:
        update_state(selected_node=selected_node)
        clear_simulation_cache()

        # Auto-select best date range when node changes (database mode only)
        if state.data_source == 'database' and selected_node:
            # Check if current range is valid for the new node
            db_loader = SupabaseDataLoader()
            current_availability = db_loader.get_node_availability(selected_node)
            
            # Filter for current date range
            if not current_availability.empty:
                mask = (current_availability['date'] >= state.start_date) & \
                       (current_availability['date'] <= state.end_date)
                range_stats = current_availability[mask]
                
                # Calculate average completeness for the current range
                if not range_stats.empty:
                    avg_completeness = range_stats['completeness'].mean()
                else:
                    avg_completeness = 0
            else:
                avg_completeness = 0
            
            # Only auto-select if current range is poor (<95% complete)
            if avg_completeness < 95:
                best_range = find_best_date_range(selected_node)
                if best_range:
                    start_date, end_date = best_range
                    try:
                        update_date_range(start_date, end_date)
                        state._dates_auto_selected = True
                        st.sidebar.success(f"✅ Auto-selected {(end_date - start_date).days + 1} days with complete data")
                    except ValueError:
                        pass  # If update fails, just keep current dates
            else:
                # Current range is good, keep it but mark as NOT auto-selected (user choice persisted)
                # or we can say "Kept current range"
                pass

    # 2. Load Price Data for Selected Node
    if needs_data_reload() or state.price_data is None or state.price_data.empty or (state.selected_node and 'node' in state.price_data.columns and state.price_data['node'].iloc[0] != state.selected_node):
        if state.selected_node:
            with st.spinner(f'Loading data for {state.selected_node}...'):
                # Determine actual source to use
                # For local_parquet, we now ALWAYS use uploaded files
                actual_source = state.data_source
                file_sig = None
                
                if state.data_source == 'local_parquet':
                    actual_source = 'uploaded'
                    # Generate signature to invalidate cache when files change
                    if state.uploaded_dam_file and state.uploaded_rtm_file:
                        dam_sig = f"{state.uploaded_dam_file.name}_{state.uploaded_dam_file.size}"  # type: ignore
                        rtm_sig = f"{state.uploaded_rtm_file.name}_{state.uploaded_rtm_file.size}"  # type: ignore
                        file_sig = f"{dam_sig}_{rtm_sig}"
                    
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
    # STRATEGY SELECTION
    # ========================================================================
    st.sidebar.markdown("---")
    st.sidebar.subheader("Dispatch Strategy")

    def on_strategy_change():
        """Callback for strategy change"""
        # Check if the key exists before accessing it
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
        }.get(state.strategy_type, 2),  # DEFAULT CHANGED: 0 → 2 (MPC)
        help="Choose the battery dispatch strategy. Linear Programming is used as a theoretical benchmark (see Opportunity page).",
        key="strategy_radio",
        on_change=on_strategy_change
    )

    # Strategy-specific parameters
    if strategy_type == "Threshold-Based":
        st.sidebar.markdown("**Threshold Parameters:**")

        # Callback for charge threshold
        def on_charge_change():
            new_pct = st.session_state.charge_slider / 100
            if abs(new_pct - state.charge_percentile) > 0.001:
                update_state(charge_percentile=new_pct)
                clear_simulation_cache()

        # Callback for discharge threshold
        def on_discharge_change():
            new_pct = st.session_state.discharge_slider / 100
            if abs(new_pct - state.discharge_percentile) > 0.001:
                update_state(discharge_percentile=new_pct)
                clear_simulation_cache()

        # Charge threshold slider
        st.sidebar.slider(
            "Charge Threshold Percentile:",
            min_value=10,
            max_value=40,
            value=int(state.charge_percentile * 100),
            step=5,
            help="Charge when price below this percentile",
            key="charge_slider",
            on_change=on_charge_change
        )

        # Discharge threshold slider
        st.sidebar.slider(
            "Discharge Threshold Percentile:",
            min_value=60,
            max_value=90,
            value=int(state.discharge_percentile * 100),
            step=5,
            help="Discharge when price above this percentile",
            key="discharge_slider",
            on_change=on_discharge_change
        )

    elif strategy_type == "Rolling Window Optimization":
        st.sidebar.markdown("**Optimization Parameters:**")

        # Callback for window hours
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

        # Callback for horizon hours
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

    # ========================================================================
    # FORECAST IMPROVEMENT (placed right after strategy parameters)
    # ========================================================================
    st.sidebar.markdown("**Forecast Scenario:**")

    # Initialize master forecast value and widget states
    # Determine what values to use
    if 'forecast_master' not in st.session_state:
        init_forecast = float(state.forecast_improvement)
    else:
        init_forecast = st.session_state.forecast_master

    # Always ensure all three values exist and are synchronized
    st.session_state.forecast_master = init_forecast
    if 'forecast_slider' not in st.session_state:
        st.session_state.forecast_slider = int(init_forecast)
    if 'forecast_input' not in st.session_state:
        st.session_state.forecast_input = init_forecast

    # Callbacks to synchronize both widgets
    def on_slider_change():
        """Slider changed - update master and input to match"""
        new_val = float(st.session_state.forecast_slider)
        st.session_state.forecast_master = new_val
        st.session_state.forecast_input = new_val
        # Move state update INTO callback (performance optimization)
        if abs(new_val - state.forecast_improvement) > 0.01:
            update_state(forecast_improvement=new_val)
            clear_simulation_cache()

    def on_input_change():
        """Input changed - update master and slider to match"""
        new_val = float(st.session_state.forecast_input)
        st.session_state.forecast_master = new_val
        # Round slider to nearest 5
        st.session_state.forecast_slider = int(round(new_val / 5) * 5)
        # Move state update INTO callback (performance optimization)
        if abs(new_val - state.forecast_improvement) > 0.01:
            update_state(forecast_improvement=new_val)
            clear_simulation_cache()

    # Two-column layout: slider + precise input (aligned)
    col_slider, col_input = st.sidebar.columns([2.5, 1])

    with col_slider:
        st.slider(
            "Forecast Accuracy (%):",
            min_value=0,
            max_value=100,
            value=st.session_state.forecast_slider,
            step=5,
            help="% of the forecast error to correct (0% = DA only, 100% = perfect RT knowledge)",
            key="forecast_slider",
            on_change=on_slider_change
        )

    with col_input:
        # Add spacing to align with slider
        st.write("")  # Empty line for vertical alignment
        st.number_input(
            "value",
            min_value=0.0,
            max_value=100.0,
            value=st.session_state.forecast_input,
            step=0.1,
            format="%.1f",
            help="Enter precise value with decimals",
            key="forecast_input",
            on_change=on_input_change,
            label_visibility="collapsed"
        )

    # State updates now handled in callbacks (performance optimization)

    # ========================================================================
    # BATTERY SPECIFICATIONS
    # ========================================================================
    st.sidebar.markdown("---")
    st.sidebar.subheader("Battery Specifications")

    # Battery presets
    if state.eia_battery_data is not None:
        preset_options = ["Custom", "Current Asset"] + [v['name'] for v in BATTERY_PRESETS.values()]

        # Initialize battery preset in session state if not present
        # IMPORTANT: We must initialize BEFORE creating the widget
        if 'battery_preset_value' not in st.session_state:
            st.session_state.battery_preset_value = "Custom"

        # Find the index for the stored preset value
        try:
            preset_index = preset_options.index(st.session_state.battery_preset_value)
        except (ValueError, AttributeError):
            preset_index = 0  # Default to "Custom"
            st.session_state.battery_preset_value = "Custom"

        # Callback to persist selection to our custom state variable
        def on_preset_change():
            # Store the selected value in our persistent state variable
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

        # Set defaults based on preset
        if battery_preset == "Custom":
            default_capacity = DEFAULT_BATTERY['capacity_mwh']
            default_power = DEFAULT_BATTERY['power_mw']
        elif battery_preset == "Current Asset":
            # Load Engie assets
            engie_data = load_engie_data()
            
            # Find asset matching selected node
            if not engie_data.empty and state.selected_node:
                # Match on settlement_point
                asset_match = engie_data[engie_data['settlement_point'] == state.selected_node]
                
                if not asset_match.empty:
                    # Use the first match
                    asset = asset_match.iloc[0]
                    default_power = int(asset['nameplate_power_mw']) if pd.notna(asset['nameplate_power_mw']) else DEFAULT_BATTERY['power_mw']
                    default_capacity = int(asset['nameplate_energy_mwh']) if pd.notna(asset['nameplate_energy_mwh']) else DEFAULT_BATTERY['capacity_mwh']
                    
                    st.sidebar.success(f"✅ Matched: {asset['plant_name']} ({default_power} MW / {default_capacity} MWh)")
                else:
                    st.sidebar.warning(f"⚠️ No asset found for {state.selected_node}")
                    default_capacity = DEFAULT_BATTERY['capacity_mwh']
                    default_power = DEFAULT_BATTERY['power_mw']
            else:
                st.sidebar.warning("⚠️ No asset data available or node not selected")
                default_capacity = DEFAULT_BATTERY['capacity_mwh']
                default_power = DEFAULT_BATTERY['power_mw']
                
        elif battery_preset == BATTERY_PRESETS['Small']['name']:
            default_capacity = BATTERY_PRESETS['Small']['capacity_mwh']
            default_power = BATTERY_PRESETS['Small']['power_mw']
        elif battery_preset == BATTERY_PRESETS['Medium']['name']:
            default_capacity = BATTERY_PRESETS['Medium']['capacity_mwh']
            default_power = BATTERY_PRESETS['Medium']['power_mw']
        elif battery_preset == BATTERY_PRESETS['Large']['name']:
            default_capacity = int(state.eia_battery_data['Nameplate Energy Capacity (MWh)'].quantile(0.9))
            default_power = int(state.eia_battery_data['Nameplate Capacity (MW)'].quantile(0.9))
        elif battery_preset == BATTERY_PRESETS['Very Large']['name']:
            default_capacity = BATTERY_PRESETS['Very Large']['capacity_mwh']
            default_power = BATTERY_PRESETS['Very Large']['power_mw']
        else:
            # Fallback for any unrecognized preset
            default_capacity = DEFAULT_BATTERY['capacity_mwh']
            default_power = DEFAULT_BATTERY['power_mw']
    else:
        # No EIA data available - force Custom mode
        if 'battery_preset_value' not in st.session_state:
            st.session_state.battery_preset_value = "Custom"
        battery_preset = "Custom"
        default_capacity = DEFAULT_BATTERY['capacity_mwh']
        default_power = DEFAULT_BATTERY['power_mw']

    # Clamp values to valid ranges (min 1, max 1000/500)
    default_capacity = max(1, min(1000, default_capacity))
    default_power = max(1, min(500, default_power))

    # Track preset changes - update sliders when preset switches
    if 'last_battery_preset' not in st.session_state:
        st.session_state.last_battery_preset = st.session_state.get('battery_preset_value', 'Custom')

    # Track node changes for Current Asset updates
    if 'last_selected_node' not in st.session_state:
        st.session_state.last_selected_node = state.selected_node

    preset_changed = (st.session_state.last_battery_preset != st.session_state.get('battery_preset_value', 'Custom'))
    node_changed = (st.session_state.last_selected_node != state.selected_node)

    if preset_changed:
        st.session_state.last_battery_preset = st.session_state.get('battery_preset_value', 'Custom')
    if node_changed:
        st.session_state.last_selected_node = state.selected_node

    # Determine if we should force update from preset/defaults
    should_update_defaults = preset_changed or (st.session_state.get('battery_preset_value', 'Custom') == "Current Asset" and node_changed)

    # Initialize session state values - prioritize existing state, then presets, then defaults
    # This ensures consistency when switching between pages

    # Determine what values to use
    if should_update_defaults:
        # Preset changed - use new preset values
        init_capacity = float(default_capacity)
    elif 'capacity_master' not in st.session_state:
        # First initialization - use existing state if available
        if state.battery_specs is not None:
            init_capacity = float(state.battery_specs.capacity_mwh)
        else:
            init_capacity = float(default_capacity)
    else:
        # Already initialized, use existing values
        init_capacity = st.session_state.capacity_master

    # Always ensure all three values exist and are synchronized
    st.session_state.capacity_master = init_capacity
    if 'capacity_slider' not in st.session_state or should_update_defaults:
        st.session_state.capacity_slider = init_capacity
    if 'capacity_input' not in st.session_state or should_update_defaults:
        st.session_state.capacity_input = init_capacity

    # Callbacks - synchronize all three values (master, slider, input)
    def on_capacity_slider_change():
        new_val = float(st.session_state.capacity_slider)
        st.session_state.capacity_master = new_val
        st.session_state.capacity_input = new_val

    def on_capacity_input_change():
        new_val = float(st.session_state.capacity_input)
        st.session_state.capacity_master = new_val
        st.session_state.capacity_slider = new_val

    # Two-column layout
    col1, col2 = st.sidebar.columns([2.25, 1])

    is_disabled = (battery_preset != "Custom")

    with col1:
        st.slider(
            "Energy Capacity (MWh):",
            min_value=5.0,
            max_value=1000.0,
            value=st.session_state.capacity_slider,
            step=10.0,
            help="Total energy storage capacity of the battery",
            disabled=is_disabled,
            key="capacity_slider",
            on_change=on_capacity_slider_change
        )

    with col2:
        st.write("")  # Alignment
        st.number_input(
            "MWh",
            min_value=5.0,
            max_value=1000.0,
            value=st.session_state.capacity_input,
            step=10.0,
            format="%.1f",
            help="Enter precise capacity",
            disabled=is_disabled,
            key="capacity_input",
            on_change=on_capacity_input_change,
            label_visibility="collapsed"
        )

    # Use master value
    capacity = st.session_state.capacity_master

    # Initialize session state for power - prioritize existing state, then presets, then defaults
    # Determine what values to use
    if should_update_defaults:
        # Preset changed - use new preset values
        init_power = float(default_power)
    elif 'power_master' not in st.session_state:
        # First initialization - use existing state if available
        if state.battery_specs is not None:
            init_power = float(state.battery_specs.power_mw)
        else:
            init_power = float(default_power)
    else:
        # Already initialized, use existing values
        init_power = st.session_state.power_master

    # Always ensure all three values exist and are synchronized
    st.session_state.power_master = init_power
    if 'power_slider' not in st.session_state or should_update_defaults:
        st.session_state.power_slider = init_power
    if 'power_input' not in st.session_state or should_update_defaults:
        st.session_state.power_input = init_power

    # Callbacks - synchronize all three values (master, slider, input)
    def on_power_slider_change():
        new_val = float(st.session_state.power_slider)
        st.session_state.power_master = new_val
        st.session_state.power_input = new_val

    def on_power_input_change():
        new_val = float(st.session_state.power_input)
        st.session_state.power_master = new_val
        st.session_state.power_slider = new_val

    # Two-column layout
    col1, col2 = st.sidebar.columns([2.25, 1])

    is_disabled = (battery_preset != "Custom")

    with col1:
        st.slider(
            "Power Capacity (MW):",
            min_value=5.0,
            max_value=500.0,
            value=st.session_state.power_slider,
            step=5.0,
            help="Maximum charge/discharge rate",
            disabled=is_disabled,
            key="power_slider",
            on_change=on_power_slider_change
        )

    with col2:
        st.write("")  # Alignment
        st.number_input(
            "MW",
            min_value=5.0,
            max_value=500.0,
            value=st.session_state.power_input,
            step=5.0,
            format="%.1f",
            help="Enter precise power",
            disabled=is_disabled,
            key="power_input",
            on_change=on_power_input_change,
            label_visibility="collapsed"
        )

    # Use master value
    power = st.session_state.power_master

    # Initialize session state for efficiency - prioritize existing state, then defaults
    # Determine what values to use
    if 'efficiency_master' not in st.session_state:
        # First initialization - use existing state if available
        if state.battery_specs is not None:
            init_efficiency = float(state.battery_specs.efficiency)
        else:
            init_efficiency = DEFAULT_BATTERY['efficiency']
    else:
        # Already initialized, use existing values (efficiency doesn't change with presets)
        init_efficiency = st.session_state.efficiency_master

    # Always ensure all three values exist and are synchronized
    st.session_state.efficiency_master = init_efficiency
    if 'efficiency_slider' not in st.session_state:
        st.session_state.efficiency_slider = init_efficiency
    if 'efficiency_input' not in st.session_state:
        st.session_state.efficiency_input = init_efficiency

    # Callbacks - synchronize all three values (master, slider, input)
    def on_efficiency_slider_change():
        new_val = float(st.session_state.efficiency_slider)
        st.session_state.efficiency_master = new_val
        st.session_state.efficiency_input = new_val

    def on_efficiency_input_change():
        new_val = float(st.session_state.efficiency_input)
        st.session_state.efficiency_master = new_val
        # Round to nearest 0.05 for slider
        st.session_state.efficiency_slider = round(new_val / 0.05) * 0.05

    # Two-column layout
    col1, col2 = st.sidebar.columns([2.25, 1])

    with col1:
        st.slider(
            "Round-trip Efficiency:",
            min_value=0.5,
            max_value=1.0,
            value=st.session_state.efficiency_slider,
            step=0.05,  # Keep current step
            help="Energy efficiency for charge/discharge cycle (100% = theoretical perfect efficiency)",
            key="efficiency_slider",
            on_change=on_efficiency_slider_change
        )

    with col2:
        st.write("")  # Alignment
        st.number_input(
            "eff",
            min_value=0.70,
            max_value=1.00,
            value=st.session_state.efficiency_input,
            step=0.01,  # Finer precision
            format="%.2f",
            help="Enter precise efficiency (0.70-1.00)",
            key="efficiency_input",
            on_change=on_efficiency_input_change,
            label_visibility="collapsed"
        )

    # Use master value
    efficiency = st.session_state.efficiency_master

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
    # DATA SUMMARY & MARKET CONTEXT MOVED TO HOME
    # ========================================================================
    # These sections have been moved to Home.py to improve layout

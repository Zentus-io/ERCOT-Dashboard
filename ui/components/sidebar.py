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
from core.data.loaders import DataLoader, SupabaseDataLoader
from utils.state import (
    get_state, update_state, clear_simulation_cache, needs_data_reload,
    get_cache_key, update_date_range, get_date_range_str, clear_data_cache
)
from config.settings import (
    DEFAULT_BATTERY, BATTERY_PRESETS, SUPABASE_URL, SUPABASE_KEY,
    DEFAULT_DAYS_BACK, MAX_DAYS_RANGE
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
            
    # Fallback to CSV or if DB failed
    csv_loader = DataLoader(data_dir)
    price_data = csv_loader.load_prices()
    return csv_loader.get_nodes(price_data)


@st.cache_data(ttl=3600, max_entries=10, show_spinner="Loading price data...")
def load_node_prices(
    source: str,
    node: str,
    start_date: date | None = None,
    end_date: date | None = None
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

    # CSV fallback (demo mode)
    csv_loader = DataLoader(data_dir)
    price_data = csv_loader.load_prices()
    return csv_loader.filter_by_node(price_data, node)


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
    if SUPABASE_URL and SUPABASE_KEY:
        data_source = st.sidebar.radio(
            "Data Source:",
            options=['Database', 'CSV Demo', 'Local Parquet'],
            index={
                'database': 0,
                'csv': 1,
                'local_parquet': 2
            }.get(state.data_source, 1),
            help="**Database**: Multi-day historical data from Supabase\n**CSV Demo**: Single-day sample data\n**Local Parquet**: Full year 2025 data"
        )

        # Map display names to internal values
        if data_source == 'Database':
            data_source_internal = 'database'
        elif data_source == 'Local Parquet':
            data_source_internal = 'local_parquet'
        else:
            data_source_internal = 'csv'

        if data_source_internal != state.data_source:
            state.data_source = data_source_internal
            clear_data_cache()
    else:
        # No database credentials - force CSV mode
        state.data_source = 'csv'
        st.sidebar.info("📌 **CSV Demo Mode** - Configure Supabase for multi-day data")

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

        earliest, latest = state.available_date_range or (
            date.today() - timedelta(days=DEFAULT_DAYS_BACK),
            date.today()
        )

        # Ensure state dates are within available range
        if state.start_date < earliest:
            state.start_date = earliest
        if state.start_date > latest:
            state.start_date = latest
        if state.end_date < earliest:
            state.end_date = earliest
        if state.end_date > latest:
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
        st.sidebar.markdown("---")
        st.sidebar.info("📌 **Local Parquet Mode** - Full Year 2025")

    else:
        # CSV mode - use fixed date from data
        st.sidebar.markdown("---")
        st.sidebar.info("📌 **Single-day demo data** (July 20, 2025)")

    # ========================================================================
    # LOAD DATA (Two-step process)
    # ========================================================================
    
    # 1. Load Node List
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
            best_range = find_best_date_range(selected_node)
            if best_range:
                start_date, end_date = best_range
                try:
                    update_date_range(start_date, end_date)
                    state._dates_auto_selected = True
                    st.sidebar.success(f"✅ Auto-selected {(end_date - start_date).days + 1} days with complete data")
                except ValueError:
                    pass  # If update fails, just keep current dates

    # 2. Load Price Data for Selected Node
    if needs_data_reload() or state.price_data is None or state.price_data.empty or (state.selected_node and 'node' in state.price_data.columns and state.price_data['node'].iloc[0] != state.selected_node):
        if state.selected_node:
            with st.spinner(f'Loading data for {state.selected_node}...'):
                price_data = load_node_prices(
                    source=state.data_source,
                    node=state.selected_node,
                    start_date=state.start_date if state.data_source == 'database' else None,
                    end_date=state.end_date if state.data_source == 'database' else None
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

    strategy_type = st.sidebar.radio(
        "Battery Trading Strategy:",
        options=["Threshold-Based", "Rolling Window Optimization"],
        index={
            "Threshold-Based": 0,
            "Rolling Window Optimization": 1
        }.get(state.strategy_type, 0),
        help="Choose the battery dispatch strategy. Linear Programming is used as a theoretical benchmark (see Opportunity page)."
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
    # FORECAST IMPROVEMENT (placed right after strategy parameters)
    # ========================================================================
    st.sidebar.markdown("**Forecast Scenario:**")

    forecast_improvement = st.sidebar.slider(
        "Forecast Accuracy Improvement (%):",
        min_value=0,
        max_value=100,
        value=state.forecast_improvement,
        step=10,
        help="% of the forecast error to correct (0% = DA only, 100% = perfect RT knowledge)"
    )

    if forecast_improvement != state.forecast_improvement:
        update_state(forecast_improvement=forecast_improvement)
        clear_simulation_cache()

    # ========================================================================
    # BATTERY SPECIFICATIONS
    # ========================================================================
    st.sidebar.markdown("---")
    st.sidebar.subheader("Battery Specifications")

    # Battery presets
    if state.eia_battery_data is not None:
        preset_options = ["Custom", "Current Asset"] + [v['name'] for v in BATTERY_PRESETS.values()]

        battery_preset = st.sidebar.selectbox(
            "Battery System Preset:",
            preset_options,
            help="Select a preset based on real Texas battery systems (EIA-860 data) or the currently selected asset."
        )

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
        disabled=(battery_preset != "Custom")
    )

    power = st.sidebar.slider(
        "Power Capacity (MW):",
        min_value=5,
        max_value=300,
        value=default_power,
        step=5,
        help="Maximum charge/discharge rate",
        disabled=(battery_preset != "Custom")
    )

    efficiency = st.sidebar.slider(
        "Round-trip Efficiency:",
        min_value=0.7,
        max_value=1.0,
        value=DEFAULT_BATTERY['efficiency'],
        step=0.05,
        help="Energy efficiency for charge/discharge cycle (100% = theoretical perfect efficiency)"
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
    # DATA SUMMARY
    # ========================================================================
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Summary")

    # Defensive check for empty or invalid data
    if state.selected_node and state.price_data is not None and not state.price_data.empty:
        # Check if 'node' column exists, handle both 'node' and 'settlement_point' naming
        if 'node' in state.price_data.columns:
            node_data = state.price_data[state.price_data['node'] == state.selected_node]
        elif 'settlement_point' in state.price_data.columns:
            # Fallback if data hasn't been renamed yet
            node_data = state.price_data[state.price_data['settlement_point'] == state.selected_node]
        else:
            st.sidebar.error("❌ Price data has unexpected column names")
            st.sidebar.caption(f"Available columns: {list(state.price_data.columns)}")
            node_data = None
    else:
        node_data = None

    if node_data is not None and not node_data.empty:

        # Dynamic date range display
        date_range = get_date_range_str(node_data)
        st.sidebar.metric("Date Range", date_range)

        # Hours available
        st.sidebar.metric("Hours Available", len(node_data))

        # Extreme events
        extreme_count = int(node_data['extreme_event'].sum()) if 'extreme_event' in node_data.columns else 0
        st.sidebar.metric("Extreme Events (>$10 spread)", extreme_count)

        # Data completeness (database mode only)
        if state.data_source == 'database' and len(node_data) > 0:
            expected_hours = (state.end_date - state.start_date).days * 24 + 24
            actual_hours = len(node_data)
            completeness = (actual_hours / expected_hours) * 100

            if completeness < 95:
                st.sidebar.warning(f"⚠️ Data coverage: {completeness:.1f}%")
                missing_hours = expected_hours - actual_hours
                st.sidebar.caption(f"Missing {missing_hours} hours")
            elif completeness < 100:
                st.sidebar.info(f"ℹ️ Data coverage: {completeness:.1f}%")
            else:
                st.sidebar.success("✓ Complete data")
    else:
        # Show why data isn't available
        if state.price_data is None:
            st.sidebar.warning("⚠️ No data loaded")
        elif state.price_data.empty:
            st.sidebar.warning("⚠️ Data query returned no results")
            if state.data_source == 'database':
                st.sidebar.caption(f"Check date range: {state.start_date} to {state.end_date}")
        elif not state.selected_node:
            st.sidebar.info("ℹ️ Select a node to view data")

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

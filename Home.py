"""
ERCOT Battery Storage Revenue Opportunity Dashboard
Zentus - Intelligent Forecasting for Renewables

Author: Juan Manuel Boullosa Novo [jmboullosa@zentus.io]
Date: November 2025

Run with: streamlit run Home.py
"""

import pandas as pd
import streamlit as st
from streamlit_calendar import calendar

from config.page_config import configure_page
from config.settings import DEFAULT_BATTERY
from core.data.loaders import SupabaseDataLoader
from ui.components.header import render_header
from ui.components.sidebar import render_sidebar
from ui.styles.custom_css import apply_custom_styles
from utils.state import get_date_range_str, get_state, init_state

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

configure_page()
apply_custom_styles()
init_state()

# ============================================================================
# HEADER AND SIDEBAR
# ============================================================================

render_header()
render_sidebar()

st.markdown("""
## Welcome to the ERCOT Battery Revenue Dashboard

This dashboard demonstrates how improved renewable energy forecasting increases
battery storage revenue in ERCOT markets.
""")

state = get_state()

# ============================================================================
# DATA SUMMARY & MARKET CONTEXT
# ============================================================================

# ============================================================================
# DASHBOARD CONTENT LAYOUT
# ============================================================================

# Create a 2x2 grid layout
row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

# --- ROW 1, COLUMN 1: Data Availability Calendar ---
with row1_col1:
    if state.data_source == 'database' and state.selected_node:
        with st.expander("📅 Data Availability Calendar", expanded=True):
            st.markdown(f"### Data Availability for **{state.selected_node}**")
            try:
                # Fetch availability data
                @st.cache_data(ttl=3600, show_spinner=False)
                def get_availability_events(node, start_date, end_date):
                    db_loader = SupabaseDataLoader()
                    availability_df = db_loader.get_node_availability(
                        node, 
                        start_date=start_date, 
                        end_date=end_date
                    )

                    events = []
                    if not availability_df.empty:
                        for _, row in availability_df.iterrows():
                            completeness = row['completeness']
                            event_date = row['date']

                            if completeness >= 95:
                                color = "#28a745"  # Green
                                title = f"{completeness:.0f}%"
                            elif completeness > 0:
                                color = "#ffc107"  # Yellow
                                title = f"{completeness:.0f}%"
                            else:
                                continue  # Don't show empty days

                            events.append({
                                "title": title,
                                "start": event_date.isoformat(),
                                "end": event_date.isoformat(),
                                "backgroundColor": color,
                                "borderColor": color,
                                "allDay": True
                            })
                    return events

                calendar_events = get_availability_events(
                    state.selected_node, 
                    state.start_date, 
                    state.end_date
                )

                calendar_options = {
                    "headerToolbar": {
                        "left": "prev,next today",
                        "center": "title",
                        "right": "dayGridMonth"
                    },
                    "initialView": "dayGridMonth",
                    # "height": 400, # Reduced height for better fit
                    "selectable": True,
                }

                # Use a stable key to prevent re-mounting issues
                calendar(
                    events=calendar_events,
                    options=calendar_options,
                    key="main_data_availability_calendar"
                )

                st.caption("Green: Complete Data (>95%) | Yellow: Partial Data | Empty: No Data")

            except Exception as e:
                st.error(f"Could not load calendar: {str(e)}")
    else:
        # Placeholder if not in database mode or no node selected
        with st.expander("📅 Data Availability Calendar", expanded=True):
            if state.data_source != 'database':
                st.info("Select 'Database' source to view availability calendar.")
            elif not state.selected_node:
                st.info("Select a settlement point to view availability.")

# --- ROW 1, COLUMN 2: Data Summary ---
with row1_col2:
    with st.expander("📊 Data Summary", expanded=True):
        node_data = None
        # Defensive check for empty or invalid data
        if state.selected_node and state.price_data is not None and not state.price_data.empty:
            # Check if 'node' column exists, handle both 'node' and 'settlement_point' naming
            if 'node' in state.price_data.columns:
                node_data_slice = state.price_data[state.price_data['node'] == state.selected_node]
            elif 'settlement_point' in state.price_data.columns:
                # Fallback if data hasn't been renamed yet
                node_data_slice = state.price_data[state.price_data['settlement_point']
                                                   == state.selected_node]
            else:
                st.error("❌ Price data has unexpected column names")
                node_data_slice = None

            if node_data_slice is not None and not node_data_slice.empty:
                node_data = node_data_slice.copy()
                # Explicitly filter by selected date range to ensure accuracy
                if 'timestamp' in node_data.columns:
                    node_data['timestamp'] = pd.to_datetime(node_data['timestamp'])
                    mask = (node_data['timestamp'].dt.date >= state.start_date) & \
                           (node_data['timestamp'].dt.date <= state.end_date)
                    node_data = node_data[mask]

        if node_data is not None and not node_data.empty:
            # Dynamic date range display
            date_range = get_date_range_str(node_data)
            st.metric("Date Range", date_range)

            # Number of settlements
            st.metric("Number of Settlements", len(node_data))

            # Extreme events
            extreme_count = int(node_data['extreme_event'].sum()
                                ) if 'extreme_event' in node_data.columns else 0
            st.metric("Extreme Events (>$10 spread)", extreme_count)

            # Data completeness (database mode only)
            if state.data_source == 'database' and len(node_data) > 0:
                # Calculate data frequency to determine expected rows
                if len(node_data) > 1:
                    # Infer frequency from the mode of time differences
                    time_diffs = node_data['timestamp'].diff().dropna()
                    # Get the most common time difference in minutes
                    freq_minutes = time_diffs.dt.total_seconds().mode().iloc[0] / 60
                else:
                    # Default to 15 minutes if only one row (safe assumption for RTM)
                    freq_minutes = 15

                # Avoid division by zero
                if freq_minutes <= 0:
                    freq_minutes = 60

                # Calculate actual hours of data available
                actual_hours = len(node_data) * (freq_minutes / 60)

                # Calculate expected hours based on selected date range
                # (End Date - Start Date + 1) * 24 hours
                days_selected = (state.end_date - state.start_date).days + 1
                expected_hours = days_selected * 24

                completeness = (actual_hours / expected_hours) * 100

                # Update metric to show actual hours (calculated)
                st.metric("Hours Available", f"{actual_hours:.1f}")

                if completeness < 95:
                    st.warning(f"⚠️ Data coverage: {completeness:.1f}%")
                    missing_hours = expected_hours - actual_hours
                    st.caption(f"Missing {missing_hours:.1f} hours")
                elif completeness < 100:
                    st.info(f"ℹ️ Data coverage: {completeness:.1f}%")
                else:
                    st.success("✓ Complete data")
        else:
            # Show why data isn't available
            if state.price_data is None:
                st.warning("⚠️ No data loaded")
            elif state.price_data.empty:
                st.warning("⚠️ Data query returned no results")
                if state.data_source == 'database':
                    st.caption(f"Check date range: {state.start_date} to {state.end_date}")
            elif not state.selected_node:
                st.info("ℹ️ Select a node to view data")

# --- ROW 2, COLUMN 1: ERCOT Battery Market Context ---
with row2_col1:
    if state.eia_battery_data is not None:
        with st.expander("💡 ERCOT Battery Market Context", expanded=False):
            # Get current capacity/power from state or defaults
            capacity = (state.battery_specs.capacity_mwh if state.battery_specs
                        else DEFAULT_BATTERY['capacity_mwh'])
            power = (state.battery_specs.power_mw if state.battery_specs
                     else DEFAULT_BATTERY['power_mw'])

            percentile_energy = (
                state.eia_battery_data['Nameplate Energy Capacity (MWh)'] < capacity).mean() * 100
            percentile_power = (
                state.eia_battery_data['Nameplate Capacity (MW)'] < power).mean() * 100

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
            """)

            if percentile_energy < 50:
                st.info(
                    "Your system is smaller than average - representative of typical "
                    "merchant battery operators.")
            elif percentile_energy > 80:
                st.success(
                    "Your system is in the top 20% by size - representative of large "
                    "utility-scale projects.")
            else:
                st.info(
                    "Your system is mid-sized - representative of the average Texas "
                    "battery market.")
    else:
        with st.expander("💡 ERCOT Battery Market Context", expanded=False):
            st.info("EIA-860 data not loaded.")

# --- ROW 2, COLUMN 2: Data File Format Guide ---
with row2_col2:
    with st.expander("📂 Data File Format Guide", expanded=False):
        st.markdown("""
        When uploading custom data files, please ensure they follow these formats:

        **DAM Prices (Day-Ahead Market)**
        - **Format:** CSV or Parquet
        - **Required Columns:**
            - `SettlementPoint` (or `SettlementPointName`): Name of the node
            - `DeliveryDate`: Date of delivery (YYYY-MM-DD)
            - `HourEnding`: Hour ending (1-24)
            - `SettlementPointPrice` (or `Price`, `LMP`): Price in $/MWh

        **RTM Prices (Real-Time Market)**
        - **Format:** CSV or Parquet
        - **Required Columns:**
            - `SettlementPointName` (or `SettlementPoint`): Name of the node
            - `DeliveryDate`: Date of delivery (YYYY-MM-DD)
            - `DeliveryHour`: Hour of delivery (0-23 or 1-24)
            - `DeliveryInterval`: Interval (1-4 for 15-min markets)
            - `SettlementPointPrice` (or `Price`, `LMP`): Price in $/MWh

        **Notes:**
        - Column names are case-insensitive and flexible (e.g., 'Price' works for 'SettlementPointPrice').
        - Timestamps will be automatically constructed from Date + Hour + Interval.
        """)

st.markdown("""
### 🚀 Get Started

1. **Configure your battery system** in the sidebar
2. **Select a settlement point** from available wind resources
3. **Choose a dispatch strategy** (Threshold-Based or Rolling Window)
4. **Navigate to analysis pages** using the top navigation menu
### 📊 Available Analysis Pages

Use the top navigation to explore different aspects of battery revenue optimization:

- **📊 Overview**: Strategy performance comparison and key metrics
- **📈 Price Analysis**: ERCOT price dynamics and forecast error distribution
- **🔋 Operations**: Battery state of charge and dispatch actions
- **💰 Revenue**: Cumulative revenue tracking and breakdown
- **🎯 Opportunity**: Sensitivity analysis and revenue opportunity
- **📅 Timeline**: Dispatch schedule visualization with price context
- **⚙️ Optimization**: Deep-dive into strategy behavior and decision-making

### 💡 Key Features

- **Real ERCOT price data** from July 20, 2025
- **Two dispatch strategies** with configurable parameters
- **Three forecast scenarios** (Baseline, Improved, Theoretical Max)
- **Battery presets** based on 136 real Texas systems (EIA-860 data)
- **Interactive visualizations** with Plotly
- **Revenue sensitivity analysis** to forecast accuracy

### 📚 How It Works

Each strategy is tested with **3 forecast quality scenarios**:

1. **Baseline (DA Only)**: Uses only day-ahead forecasts (no improvement)
2. **Improved**: Uses day-ahead + forecast improvement (adjust with slider)
3. **Perfect Foresight**: Uses actual real-time prices (theoretical maximum)

Move the "Forecast Accuracy Improvement" slider in the sidebar to see how better
forecasts improve Scenario 2!

### 🎯 About This Project

Built for the **Engie Urja AI Challenge 2025**, this dashboard showcases the
value of intelligent forecasting for renewable energy storage operations in
competitive electricity markets.

**Author**: Juan Manuel Boullosa Novo
**Organization**: Zentus - Intelligent Forecasting for Renewables
**Contact**: info@zentus.io

---

Ready to explore? Select a page to begin your analysis! 👈
""")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6C757D;'>
    <p><strong>Zentus</strong> - Intelligent Forecasting for Renewables</p>
    <p>Stanford Doerr School of Sustainability Accelerator Fellow</p>
    <p>Engie Urja AI Challenge 2025</p>
    <p>Contact: info@zentus.io | zentus.io</p>
</div>
""", unsafe_allow_html=True)

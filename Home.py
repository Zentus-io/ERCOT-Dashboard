"""
ERCOT Battery Storage Revenue Opportunity Dashboard
Zentus - Intelligent Forecasting for Renewables

Author: Juan Manuel Boullosa Novo [jmboullosa@zentus.io]
Date: November 2025

Run with: streamlit run Home.py
"""

import streamlit as st
from config.page_config import configure_page
from ui.styles.custom_css import apply_custom_styles
from ui.components.header import render_header
from ui.components.sidebar import render_sidebar
from ui.components.sidebar import render_sidebar
from utils.state import init_state, get_state
from core.data.loaders import SupabaseDataLoader
from streamlit_calendar import calendar
import pandas as pd

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

# ============================================================================
# DATA AVAILABILITY CALENDAR (Database Mode)
# ============================================================================
state = get_state()
if state.data_source == 'database' and state.selected_node:
    with st.expander("📅 Data Availability Calendar", expanded=True):
        st.markdown(f"### Data Availability for **{state.selected_node}**")
        try:
            # Fetch availability data
            @st.cache_data(ttl=3600, show_spinner=False)
            def get_availability_events(node):
                db_loader = SupabaseDataLoader()
                availability_df = db_loader.get_node_availability(node)
                
                events = []
                if not availability_df.empty:
                    for _, row in availability_df.iterrows():
                        completeness = row['completeness']
                        event_date = row['date']
                        
                        if completeness >= 95:
                            color = "#28a745" # Green
                            title = f"{completeness:.0f}%"
                        elif completeness > 0:
                            color = "#ffc107" # Yellow
                            title = f"{completeness:.0f}%"
                        else:
                            continue # Don't show empty days
                            
                        events.append({
                            "title": title,
                            "start": event_date.isoformat(),
                            "end": event_date.isoformat(),
                            "backgroundColor": color,
                            "borderColor": color,
                            "allDay": True,
                            "display": "background" 
                        })
                return events

            calendar_events = get_availability_events(state.selected_node)
            
            calendar_options = {
                "headerToolbar": {
                    "left": "prev,next today",
                    "center": "title",
                    "right": "dayGridMonth,timeGridWeek"
                },
                "initialView": "dayGridMonth",
                "height": 600, # Fixed height for better stability
                "selectable": True,
            }
            
            # Use a stable key to prevent re-mounting issues
            cal = calendar(
                events=calendar_events,
                options=calendar_options,
                key="main_data_availability_calendar"
            )
            
            st.caption("Green: Complete Data (>95%) | Yellow: Partial Data | Empty: No Data")
            
        except Exception as e:
            st.error(f"Could not load calendar: {str(e)}")

# ============================================================================
# HOMEPAGE CONTENT
# ============================================================================

st.markdown("""
## Welcome to the ERCOT Battery Revenue Dashboard

This dashboard demonstrates how improved renewable energy forecasting increases
battery storage revenue in ERCOT markets.
""")

with st.expander("📂 Data File Format Guide"):
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
4. **Navigate to analysis pages** using the sidebar menu
### 📊 Available Analysis Pages

Use the sidebar navigation to explore different aspects of battery revenue optimization:

- **🏠 Overview**: Strategy performance comparison and key metrics
- **📈 Price Analysis**: ERCOT price dynamics and forecast error distribution
- **🔋 Operations**: Battery state of charge and dispatch actions
- **💰 Revenue**: Cumulative revenue tracking and breakdown
- **🎯 Opportunity**: Sensitivity analysis and revenue opportunity
- **📊 Timeline**: Dispatch schedule visualization with price context
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

Ready to explore? Select a page from the sidebar to begin your analysis! 👈
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

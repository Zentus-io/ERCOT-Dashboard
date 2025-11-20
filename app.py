"""
ERCOT Battery Storage Revenue Opportunity Dashboard
Zentus - Intelligent Forecasting for Renewables

Author: Juan Manuel Boullosa Novo [jmboullosa@zentus.io]
Date: November 2025

Run with: streamlit run app.py
"""

import streamlit as st
from config.page_config import configure_page
from ui.styles.custom_css import apply_custom_styles
from ui.components.header import render_header
from ui.components.sidebar import render_sidebar
from utils.state import init_state

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
# HOMEPAGE CONTENT
# ============================================================================

st.markdown("""
## Welcome to the ERCOT Battery Revenue Dashboard

This dashboard demonstrates how improved renewable energy forecasting increases
battery storage revenue in ERCOT markets.

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

### 🎯 About This Demo

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

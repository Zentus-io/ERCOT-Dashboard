"""
ERCOT Battery Storage Revenue Opportunity Dashboard
Zentus - Intelligent Forecasting for Renewables

Author: Juan Manuel Boullosa Novo
Date: November 2025

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from pathlib import Path

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

# Set custom favicon
favicon_path = Path(__file__).parent / 'media' / 'favicon-32x32.png'

st.set_page_config(
    page_title="Zentus - ERCOT Revenue Opportunity",
    page_icon=str(favicon_path) if favicon_path.exists() else "⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        color: #0A5F7A;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0;
    }
    .sub-header {
        color: #1E3A5F;
        font-size: 1.2rem;
        margin-top: 0;
    }
    .metric-card {
        background-color: #f0f8fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #0A5F7A;
    }
    .opportunity-metric {
        font-size: 2rem;
        color: #0A5F7A;
        font-weight: bold;
    }
    .data-note {
        background-color: #FFF3CD;
        padding: 10px;
        border-radius: 5px;
        border-left: 3px solid #FFC107;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING FUNCTIONS (Define functions FIRST, then load data)
# ============================================================================

@st.cache_data
def load_price_data():
    """Load real ERCOT price data from CSV files."""

    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'data'

    # Load DA and RT prices
    da_prices = pd.read_csv(data_dir / 'da_prices.csv')
    rt_prices = pd.read_csv(data_dir / 'rt_prices.csv')

    # Convert timestamps
    da_prices['timestamp'] = pd.to_datetime(da_prices['timestamp'])
    rt_prices['timestamp'] = pd.to_datetime(rt_prices['timestamp'])

    # Merge on timestamp and node
    merged = da_prices.merge(
        rt_prices,
        on=['timestamp', 'node'],
        suffixes=('_da', '_rt')
    )

    # Calculate metrics
    merged['forecast_error'] = merged['price_mwh_rt'] - merged['price_mwh_da']
    merged['price_spread'] = abs(merged['forecast_error'])
    merged['extreme_event'] = merged['price_spread'] > 10  # $10/MWh threshold

    return merged

@st.cache_data
def get_available_nodes(price_data):
    """Get list of available settlement point nodes."""
    return sorted(price_data['node'].unique())

@st.cache_data
def load_eia_battery_data():
    """Load EIA-860 battery storage market data for Texas (ERCOT) from optimized Parquet file."""
    try:
        # Try loading preprocessed Parquet file first (20-30x faster than Excel!)
        parquet_path = Path(__file__).parent / 'data' / 'processed' / 'ercot_batteries.parquet'

        if parquet_path.exists():
            # Load with Polars (fastest option)
            df_polars = pl.read_parquet(parquet_path)
            # Convert to pandas for compatibility with existing dashboard code
            df = df_polars.to_pandas()

            # Rename columns to match expected format
            df = df.rename(columns={
                'nameplate_power_mw': 'Nameplate Capacity (MW)',
                'nameplate_energy_mwh': 'Nameplate Energy Capacity (MWh)',
                'duration_hours': 'Duration (hours)',
                'state': 'State',
                'use_arbitrage': 'Arbitrage',
                'use_frequency_regulation': 'Frequency Regulation',
                'use_ramping_reserve': 'Ramping / Spinning Reserve'
            })

            return df

        else:
            # Fallback to Excel if Parquet not available
            file_path = Path(__file__).parent / 'data' / 'eia8602024' / '3_4_Energy_Storage_Y2024.xlsx'
            df = pd.read_excel(file_path, sheet_name='Operable', header=1)

            # Filter for Texas batteries
            texas = df[df['State'] == 'TX'].copy()

            # Calculate duration
            texas['Duration (hours)'] = (
                texas['Nameplate Energy Capacity (MWh)'] /
                texas['Nameplate Capacity (MW)']
            )

            return texas

    except Exception as e:
        # Return None if file not found (graceful degradation)
        return None

# ============================================================================
# LOAD DATA (After functions are defined)
# ============================================================================

with st.spinner('Loading ERCOT price data...'):
    price_data = load_price_data()
    available_nodes = get_available_nodes(price_data)
    eia_battery_data = load_eia_battery_data()  # Load EIA-860 battery market data

# ============================================================================
# BATTERY SIMULATION
# ============================================================================

def calculate_dynamic_thresholds(price_df, improvement_factor=0.0, use_optimal=False,
                                 charge_percentile=0.25, discharge_percentile=0.75):
    """
    Calculate dynamic charge/discharge thresholds based on price distribution.

    Thresholds are calculated from the forecast type being used to ensure
    fair comparison and meaningful differences between strategies.

    Parameters:
    -----------
    price_df : pd.DataFrame
        Price data with columns: price_mwh_da, price_mwh_rt, forecast_error
    improvement_factor : float
        Fraction of forecast error correction (0 to 1)
    use_optimal : bool
        If True, use RT prices. If False, use DA-based forecasts.
    charge_percentile : float
        Percentile for charge threshold (default: 0.25 = 25th percentile)
    discharge_percentile : float
        Percentile for discharge threshold (default: 0.75 = 75th percentile)

    Returns:
    --------
    tuple: (charge_threshold, discharge_threshold)
    """
    if use_optimal:
        # For optimal strategy, use RT price distribution
        decision_prices = price_df['price_mwh_rt']
    else:
        # For baseline/improved, calculate the forecast prices that will be used
        decision_prices = price_df['price_mwh_da'] + (price_df['forecast_error'] * improvement_factor)

    # Use configurable percentiles for thresholds
    charge_threshold = decision_prices.quantile(charge_percentile)
    discharge_threshold = decision_prices.quantile(discharge_percentile)

    # Ensure minimum spread of $5/MWh for arbitrage opportunity
    if discharge_threshold - charge_threshold < 5:
        median = decision_prices.median()
        charge_threshold = median - 2.5
        discharge_threshold = median + 2.5

    return charge_threshold, discharge_threshold


def simulate_battery_dispatch(price_df, battery_capacity_mwh, battery_power_mw,
                              efficiency, use_optimal=True, improvement_factor=0.0,
                              charge_percentile=0.25, discharge_percentile=0.75):
    """
    Simulate battery dispatch strategy with dynamic thresholds.

    Parameters:
    -----------
    price_df : pd.DataFrame
        Price data with columns: timestamp, price_mwh_da, price_mwh_rt, forecast_error
    battery_capacity_mwh : float
        Battery energy capacity
    battery_power_mw : float
        Battery power capacity (charge/discharge rate)
    efficiency : float
        Round-trip efficiency (0 to 1)
    use_optimal : bool
        If True, use perfect foresight (RT prices). If False, use DA prices.
    improvement_factor : float
        Fraction of forecast error to correct (0 to 1)
    charge_percentile : float
        Percentile for charge threshold (default: 0.25)
    discharge_percentile : float
        Percentile for discharge threshold (default: 0.75)

    Returns:
    --------
    pd.DataFrame: Original dataframe with added columns for dispatch decisions
    """
    df = price_df.copy()

    # Calculate thresholds appropriate for this forecast type
    charge_threshold, discharge_threshold = calculate_dynamic_thresholds(
        df, improvement_factor=improvement_factor, use_optimal=use_optimal,
        charge_percentile=charge_percentile, discharge_percentile=discharge_percentile
    )

    # Start at 50% state of charge
    soc = 0.5 * battery_capacity_mwh
    revenue = 0
    charge_cost = 0
    discharge_revenue = 0

    dispatch = []
    soc_history = []
    revenue_history = []
    power_history = []

    for idx, row in df.iterrows():
        if use_optimal:
            # Perfect foresight - use actual RT prices for decisions
            decision_price = row['price_mwh_rt']
        else:
            # Use day-ahead forecast with potential improvement
            # improvement_factor = 0: uses DA only (baseline)
            # improvement_factor = 1: uses RT (perfect forecast)
            # improvement_factor = 0.1: moves 10% towards RT from DA
            improved_forecast = row['price_mwh_da'] + (row['forecast_error'] * improvement_factor)
            decision_price = improved_forecast

        # Actual price paid/received (always RT price in real market)
        rt_price = row['price_mwh_rt']

        # Dynamic threshold strategy
        # Charge when price is below 25th percentile and battery not full
        if decision_price < charge_threshold and soc < battery_capacity_mwh * 0.95:
            # Charge
            charge_amount = min(battery_power_mw, battery_capacity_mwh * 0.95 - soc)
            soc += charge_amount * efficiency
            cost = charge_amount * rt_price
            revenue -= cost
            charge_cost += cost
            action = 'charge'
            power = -charge_amount

        # Discharge when price is above 75th percentile and battery not empty
        elif decision_price > discharge_threshold and soc > battery_capacity_mwh * 0.05:
            # Discharge
            discharge_amount = min(battery_power_mw, soc - battery_capacity_mwh * 0.05)
            soc -= discharge_amount
            revenue_from_sale = discharge_amount * rt_price
            revenue += revenue_from_sale
            discharge_revenue += revenue_from_sale
            action = 'discharge'
            power = discharge_amount

        else:
            # Hold
            action = 'hold'
            power = 0

        dispatch.append(action)
        soc_history.append(soc)
        revenue_history.append(revenue)
        power_history.append(power)

    df['dispatch'] = dispatch
    df['soc'] = soc_history
    df['cumulative_revenue'] = revenue_history
    df['power'] = power_history

    # Store thresholds and totals for display
    df.attrs['charge_threshold'] = charge_threshold
    df.attrs['discharge_threshold'] = discharge_threshold
    df.attrs['total_charge_cost'] = charge_cost
    df.attrs['total_discharge_revenue'] = discharge_revenue
    df.attrs['charge_count'] = (df['dispatch'] == 'charge').sum()
    df.attrs['discharge_count'] = (df['dispatch'] == 'discharge').sum()
    df.attrs['hold_count'] = (df['dispatch'] == 'hold').sum()

    return df


def simulate_rolling_window_dispatch(price_df, battery_capacity_mwh, battery_power_mw,
                                     efficiency, window_hours=6, improvement_factor=0.0):
    """
    Simulate battery dispatch using rolling window optimization strategy.

    At each hour, looks ahead N hours and makes locally optimal decision:
    - Charge if current price is minimum in lookahead window
    - Discharge if current price is maximum in lookahead window
    - Hold otherwise

    This strategy naturally handles temporal constraints and avoids threshold
    crossing sensitivity issues of percentile-based strategies.

    Parameters:
    -----------
    price_df : pd.DataFrame
        Price data with columns: timestamp, price_mwh_da, price_mwh_rt, forecast_error
    battery_capacity_mwh : float
        Battery energy capacity
    battery_power_mw : float
        Battery power capacity (charge/discharge rate)
    efficiency : float
        Round-trip efficiency (0 to 1)
    window_hours : int
        Number of hours to look ahead for optimization (default: 6)
    improvement_factor : float
        Fraction of forecast error to correct (0 to 1)

    Returns:
    --------
    pd.DataFrame: Original dataframe with added columns for dispatch decisions
    """
    df = price_df.copy()

    # Start at 50% state of charge
    soc = 0.5 * battery_capacity_mwh
    revenue = 0
    charge_cost = 0
    discharge_revenue = 0

    dispatch = []
    soc_history = []
    revenue_history = []
    power_history = []

    for idx in range(len(df)):
        row = df.iloc[idx]

        # Calculate decision price (forecast with potential improvement)
        improved_forecast = row['price_mwh_da'] + (row['forecast_error'] * improvement_factor)
        decision_price = improved_forecast

        # Actual price paid/received (always RT price in real market)
        rt_price = row['price_mwh_rt']

        # Define lookahead window
        window_end = min(idx + window_hours, len(df))
        window_prices = df.iloc[idx:window_end].apply(
            lambda r: r['price_mwh_da'] + (r['forecast_error'] * improvement_factor),
            axis=1
        )

        # Rolling window optimization logic
        # Charge if current price is minimum in window AND battery not full
        if decision_price == window_prices.min() and soc < battery_capacity_mwh * 0.95:
            # Charge: this is the cheapest hour in lookahead window
            charge_amount = min(battery_power_mw, battery_capacity_mwh * 0.95 - soc)
            soc += charge_amount * efficiency
            cost = charge_amount * rt_price
            revenue -= cost
            charge_cost += cost
            action = 'charge'
            power = -charge_amount

        # Discharge if current price is maximum in window AND battery not empty
        elif decision_price == window_prices.max() and soc > battery_capacity_mwh * 0.05:
            # Discharge: this is the most expensive hour in lookahead window
            discharge_amount = min(battery_power_mw, soc - battery_capacity_mwh * 0.05)
            soc -= discharge_amount
            revenue_from_sale = discharge_amount * rt_price
            revenue += revenue_from_sale
            discharge_revenue += revenue_from_sale
            action = 'discharge'
            power = discharge_amount

        else:
            # Hold
            action = 'hold'
            power = 0

        dispatch.append(action)
        soc_history.append(soc)
        revenue_history.append(revenue)
        power_history.append(power)

    df['dispatch'] = dispatch
    df['soc'] = soc_history
    df['cumulative_revenue'] = revenue_history
    df['power'] = power_history

    # Store totals for display
    df.attrs['window_hours'] = window_hours
    df.attrs['total_charge_cost'] = charge_cost
    df.attrs['total_discharge_revenue'] = discharge_revenue
    df.attrs['charge_count'] = (df['dispatch'] == 'charge').sum()
    df.attrs['discharge_count'] = (df['dispatch'] == 'discharge').sum()
    df.attrs['hold_count'] = (df['dispatch'] == 'hold').sum()

    return df


# ============================================================================
# HEADER
# ============================================================================

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown('<p class="main-header">⚡ ERCOT Battery Storage Revenue Opportunity</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Demonstrating Value Through Intelligent Forecasting</p>',
                unsafe_allow_html=True)

with col2:
    # Display Zentus logo and brand text together
    logo_path = Path(__file__).parent / 'media' / 'Logo_Option_5_nobg.png'

    col2a, col2b = st.columns([1, 2])

    with col2a:
        if logo_path.exists():
            st.image(str(logo_path), width=80)

    with col2b:
        st.markdown("""
            <div style='padding-top: 15px;'>
                <span style='font-size: 1.8rem; color: #0A5F7A; font-weight: bold;'>Zentus</span><br>
                <span style='font-size: 0.8rem; color: #6C757D;'>Intelligent Forecasting</span>
            </div>
        """, unsafe_allow_html=True)

# Data availability notice
eia_note = ""
if eia_battery_data is not None:
    eia_note = " Battery system parameters validated against EIA-860 data (136 operational Texas systems)."

st.markdown(f"""
<div class='data-note'>
    <strong>MVP Demo:</strong> Currently showing July 20, 2025 data from ERCOT wind resources.
    This single-day snapshot demonstrates the revenue opportunity concept.{eia_note}
    Additional historical data with extreme price events is being processed.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# LOAD DATA
# ============================================================================

with st.spinner('Loading ERCOT price data...'):
    price_data = load_price_data()
    available_nodes = get_available_nodes(price_data)
    eia_battery_data = load_eia_battery_data()  # Load EIA-860 battery market data

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================

st.sidebar.header("Analysis Configuration")

# Node Selection
selected_node = st.sidebar.selectbox(
    "Select Settlement Point:",
    available_nodes,
    help="Choose a wind resource settlement point to analyze"
)

# Filter data for selected node
node_data = price_data[price_data['node'] == selected_node].copy()
node_data = node_data.sort_values('timestamp').reset_index(drop=True)

# Strategy Selection
st.sidebar.markdown("---")
st.sidebar.subheader("Dispatch Strategy")

strategy_type = st.sidebar.radio(
    "Battery Trading Strategy:",
    options=[
        "Threshold-Based",
        "Rolling Window Optimization"
    ],
    index=0,
    help="Choose the battery dispatch optimization approach. Each strategy shows 3 scenarios: Baseline (DA only), Improved (DA + forecast improvement), and Theoretical Max (perfect RT prices)"
)

# Strategy-specific parameters
if strategy_type == "Threshold-Based":
    st.sidebar.markdown("**Threshold Parameters:**")
    charge_percentile = st.sidebar.slider(
        "Charge Threshold Percentile:",
        min_value=10,
        max_value=40,
        value=25,
        step=5,
        help="Charge when price below this percentile"
    ) / 100  # Convert to decimal

    discharge_percentile = st.sidebar.slider(
        "Discharge Threshold Percentile:",
        min_value=60,
        max_value=90,
        value=75,
        step=5,
        help="Discharge when price above this percentile"
    ) / 100  # Convert to decimal
else:
    # Use defaults for non-threshold strategies
    charge_percentile = 0.25
    discharge_percentile = 0.75

if strategy_type == "Rolling Window Optimization":
    st.sidebar.markdown("**Optimization Parameters:**")
    window_hours = st.sidebar.slider(
        "Lookahead Window (hours):",
        min_value=2,
        max_value=12,
        value=6,
        step=1,
        help="Number of hours to look ahead for optimization"
    )
else:
    window_hours = 6  # Default

# Battery Parameters
st.sidebar.markdown("---")
st.sidebar.subheader("Battery Specifications")

# Battery system presets based on real EIA-860 Texas data
if eia_battery_data is not None:
    preset_options = [
        "Custom",
        "Small (TX Median: 10 MW / 17 MWh)",
        "Medium (TX Mean: 59 MW / 85 MWh)",
        "Large (TX 90th percentile)",
        "Very Large (TX Max: 300 MW)"
    ]

    battery_preset = st.sidebar.selectbox(
        "Battery System Preset:",
        preset_options,
        help="Select a preset based on real Texas battery systems (EIA-860 data) or choose Custom"
    )

    # Set initial values based on preset
    if battery_preset == "Small (TX Median: 10 MW / 17 MWh)":
        default_capacity = 17
        default_power = 10
    elif battery_preset == "Medium (TX Mean: 59 MW / 85 MWh)":
        default_capacity = 85
        default_power = 59
    elif battery_preset == "Large (TX 90th percentile)":
        capacity_90 = int(eia_battery_data['Nameplate Energy Capacity (MWh)'].quantile(0.9))
        power_90 = int(eia_battery_data['Nameplate Capacity (MW)'].quantile(0.9))
        default_capacity = capacity_90
        default_power = power_90
    elif battery_preset == "Very Large (TX Max: 300 MW)":
        default_capacity = 600
        default_power = 300
    else:  # Custom
        default_capacity = 100
        default_power = 50
else:
    # No EIA data available, use defaults
    battery_preset = "Custom"
    default_capacity = 100
    default_power = 50

battery_capacity_mwh = st.sidebar.slider(
    "Energy Capacity (MWh):",
    min_value=10,
    max_value=600,
    value=default_capacity,
    step=5 if default_capacity < 100 else 10,
    help="Total energy storage capacity of the battery",
    disabled=(battery_preset != "Custom" and eia_battery_data is not None)
)

battery_power_mw = st.sidebar.slider(
    "Power Capacity (MW):",
    min_value=5,
    max_value=300,
    value=default_power,
    step=5,
    help="Maximum charge/discharge rate",
    disabled=(battery_preset != "Custom" and eia_battery_data is not None)
)

efficiency = st.sidebar.slider(
    "Round-trip Efficiency:",
    min_value=0.7,
    max_value=0.95,
    value=0.9,
    step=0.05,
    help="Energy efficiency for charge/discharge cycle"
)

# Forecast Improvement Scenario
st.sidebar.subheader("Forecast Improvement Scenario")

forecast_improvement = st.sidebar.slider(
    "Forecast Accuracy Improvement (%):",
    min_value=0,
    max_value=50,
    value=10,
    step=5,
    help="Simulate the impact of improving forecast accuracy during extreme events"
)

# Display data summary
st.sidebar.markdown("---")
st.sidebar.subheader("Data Summary")
st.sidebar.metric("Date", "July 20, 2025")
st.sidebar.metric("Hours Available", len(node_data))
st.sidebar.metric("Extreme Events (>$10 spread)", node_data['extreme_event'].sum())

# EIA-860 Market Context
if eia_battery_data is not None:
    st.sidebar.markdown("---")
    with st.sidebar.expander("📊 ERCOT Battery Market Context", expanded=False):
        # Calculate where user's system fits in market
        percentile_energy = (eia_battery_data['Nameplate Energy Capacity (MWh)'] < battery_capacity_mwh).mean() * 100
        percentile_power = (eia_battery_data['Nameplate Capacity (MW)'] < battery_power_mw).mean() * 100

        duration_hours = battery_capacity_mwh / battery_power_mw if battery_power_mw > 0 else 0

        st.markdown(f"""
        **Texas Battery Market (EIA-860 2024)**

        **Your System:**
        - {battery_capacity_mwh:.0f} MWh / {battery_power_mw:.0f} MW
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

        # Show distribution insight
        if percentile_energy < 50:
            st.info("💡 Your system is smaller than average - representative of typical merchant battery operators.")
        elif percentile_energy > 80:
            st.success("💡 Your system is in the top 20% by size - representative of large utility-scale projects.")
        else:
            st.info("💡 Your system is mid-sized - representative of the average Texas battery market.")

# Calculate dynamic thresholds (always calculate for charts, display in sidebar only for threshold-based)
charge_thresh, discharge_thresh = calculate_dynamic_thresholds(
    node_data, improvement_factor=0.0, use_optimal=False,
    charge_percentile=charge_percentile, discharge_percentile=discharge_percentile
)

# Display thresholds in sidebar only for threshold-based strategy
if strategy_type == "Threshold-Based":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Trading Thresholds")
    st.sidebar.metric("Charge Below", f"${charge_thresh:.2f}/MWh",
                     help=f"Charge when price is below {int(charge_percentile*100)}th percentile")
    st.sidebar.metric("Discharge Above", f"${discharge_thresh:.2f}/MWh",
                     help=f"Discharge when price is above {int(discharge_percentile*100)}th percentile")

# ============================================================================
# RUN SIMULATIONS
# ============================================================================

with st.spinner('Running battery simulations...'):
    # Run simulations based on selected strategy
    if strategy_type == "Rolling Window Optimization":
        # Rolling window optimization
        optimal_dispatch = simulate_rolling_window_dispatch(
            node_data, battery_capacity_mwh, battery_power_mw, efficiency,
            window_hours=window_hours,
            improvement_factor=1.0  # Perfect forecast for theoretical max
        )
        naive_dispatch = simulate_rolling_window_dispatch(
            node_data, battery_capacity_mwh, battery_power_mw, efficiency,
            window_hours=window_hours,
            improvement_factor=0.0  # Baseline (DA only)
        )
        improved_dispatch = simulate_rolling_window_dispatch(
            node_data, battery_capacity_mwh, battery_power_mw, efficiency,
            window_hours=window_hours,
            improvement_factor=forecast_improvement/100
        )

    else:  # Threshold-Based
        # Threshold-based strategy
        optimal_dispatch = simulate_battery_dispatch(
            node_data, battery_capacity_mwh, battery_power_mw, efficiency,
            use_optimal=True,
            charge_percentile=charge_percentile,
            discharge_percentile=discharge_percentile
        )
        naive_dispatch = simulate_battery_dispatch(
            node_data, battery_capacity_mwh, battery_power_mw, efficiency,
            use_optimal=False,
            charge_percentile=charge_percentile,
            discharge_percentile=discharge_percentile
        )
        improved_dispatch = simulate_battery_dispatch(
            node_data, battery_capacity_mwh, battery_power_mw, efficiency,
            use_optimal=False,
            improvement_factor=forecast_improvement/100,
            charge_percentile=charge_percentile,
            discharge_percentile=discharge_percentile
        )

# ============================================================================
# KEY METRICS
# ============================================================================

st.header("📊 Revenue Analysis")

# Explanation box
st.info("""
**How to use this dashboard:**
Each strategy is tested with **3 forecast quality scenarios**:
1. **Baseline (DA Only)** - Uses only day-ahead forecasts (no improvement) ← Never changes
2. **Improved** - Uses day-ahead + forecast improvement ← **Changes when you adjust the slider**
3. **Perfect Foresight** - Uses actual real-time prices (theoretical maximum) ← Never changes

💡 **Tip:** Move the "Forecast Accuracy Improvement" slider to see how better forecasts improve Scenario 2!
""")

col1, col2, col3, col4 = st.columns(4)

naive_revenue = naive_dispatch['cumulative_revenue'].iloc[-1]
improved_revenue = improved_dispatch['cumulative_revenue'].iloc[-1]
optimal_revenue = optimal_dispatch['cumulative_revenue'].iloc[-1]

opportunity_vs_naive = optimal_revenue - naive_revenue
opportunity_vs_improved = improved_revenue - naive_revenue
remaining_opportunity = optimal_revenue - improved_revenue

with col1:
    st.metric(
        label="Scenario 1: Baseline (DA Only)",
        value=f"${naive_revenue:,.0f}",
        help="Revenue using only day-ahead forecasts (no improvement)"
    )

with col2:
    st.metric(
        label=f"Scenario 2: Improved (+{forecast_improvement}%)",
        value=f"${improved_revenue:,.0f}",
        delta=f"+${opportunity_vs_improved:,.0f}" if opportunity_vs_improved >= 0 else f"${opportunity_vs_improved:,.0f}",
        delta_color="normal",
        help=f"Revenue with {forecast_improvement}% forecast accuracy improvement (adjust slider to change)"
    )

with col3:
    st.metric(
        label="Scenario 3: Perfect Foresight",
        value=f"${optimal_revenue:,.0f}",
        delta=f"+${opportunity_vs_naive:,.0f}" if opportunity_vs_naive >= 0 else f"${opportunity_vs_naive:,.0f}",
        delta_color="normal",
        help="Theoretical maximum revenue using perfect real-time price knowledge (impossible in practice)"
    )

with col4:
    improvement_pct = (opportunity_vs_improved / abs(naive_revenue)) * 100 if naive_revenue != 0 else 0
    st.metric(
        label="Revenue Improvement",
        value=f"{improvement_pct:.1f}%",
        help="Percentage improvement over baseline"
    )

st.markdown("---")

# Dispatch statistics
st.subheader("📋 Battery Dispatch Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Baseline (DA Forecast)**")
    st.write(f"Charge events: {naive_dispatch.attrs['charge_count']}")
    st.write(f"Discharge events: {naive_dispatch.attrs['discharge_count']}")
    st.write(f"Hold periods: {naive_dispatch.attrs['hold_count']}")
    st.write(f"Charge cost: ${naive_dispatch.attrs['total_charge_cost']:,.0f}")
    st.write(f"Discharge revenue: ${naive_dispatch.attrs['total_discharge_revenue']:,.0f}")

with col2:
    st.markdown(f"**Improved (+{forecast_improvement}%)**")
    st.write(f"Charge events: {improved_dispatch.attrs['charge_count']}")
    st.write(f"Discharge events: {improved_dispatch.attrs['discharge_count']}")
    st.write(f"Hold periods: {improved_dispatch.attrs['hold_count']}")
    st.write(f"Charge cost: ${improved_dispatch.attrs['total_charge_cost']:,.0f}")
    st.write(f"Discharge revenue: ${improved_dispatch.attrs['total_discharge_revenue']:,.0f}")

with col3:
    st.markdown("**Optimal (Perfect Forecast)**")
    st.write(f"Charge events: {optimal_dispatch.attrs['charge_count']}")
    st.write(f"Discharge events: {optimal_dispatch.attrs['discharge_count']}")
    st.write(f"Hold periods: {optimal_dispatch.attrs['hold_count']}")
    st.write(f"Charge cost: ${optimal_dispatch.attrs['total_charge_cost']:,.0f}")
    st.write(f"Discharge revenue: ${optimal_dispatch.attrs['total_discharge_revenue']:,.0f}")

st.markdown("---")

# ============================================================================
# MAIN VISUALIZATIONS
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "⚡ Strategy Comparison",
    "📈 Price Analysis",
    "🔋 Battery Operations",
    "💰 Revenue Comparison",
    "🎯 Opportunity Breakdown",
    "📊 Decision Timeline",
    "🎯 Optimization Analysis"
])

with tab1:
    st.subheader("Strategy Performance Comparison")

    # Create three columns for side-by-side comparison
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📉 Baseline (DA Only)")
        st.metric("Revenue", f"${naive_revenue:,.0f}")
        st.metric("Charge Events", naive_dispatch.attrs.get('charge_count', 0))
        st.metric("Discharge Events", naive_dispatch.attrs.get('discharge_count', 0))
        charge_pct = (naive_dispatch.attrs.get('charge_count', 0) / len(node_data)) * 100
        discharge_pct = (naive_dispatch.attrs.get('discharge_count', 0) / len(node_data)) * 100
        st.progress(charge_pct / 100, text=f"Charging: {charge_pct:.1f}% of time")
        st.progress(discharge_pct / 100, text=f"Discharging: {discharge_pct:.1f}% of time")

    with col2:
        st.markdown(f"### 📈 Improved (+{forecast_improvement}%)")
        st.metric("Revenue", f"${improved_revenue:,.0f}",
                 delta=f"+${opportunity_vs_improved:,.0f}" if opportunity_vs_improved >= 0 else f"${opportunity_vs_improved:,.0f}")
        st.metric("Charge Events", improved_dispatch.attrs.get('charge_count', 0))
        st.metric("Discharge Events", improved_dispatch.attrs.get('discharge_count', 0))
        charge_pct = (improved_dispatch.attrs.get('charge_count', 0) / len(node_data)) * 100
        discharge_pct = (improved_dispatch.attrs.get('discharge_count', 0) / len(node_data)) * 100
        st.progress(charge_pct / 100, text=f"Charging: {charge_pct:.1f}% of time")
        st.progress(discharge_pct / 100, text=f"Discharging: {discharge_pct:.1f}% of time")

    with col3:
        st.markdown("### ⭐ Theoretical Max")
        st.metric("Revenue", f"${optimal_revenue:,.0f}",
                 delta=f"+${opportunity_vs_naive:,.0f}" if opportunity_vs_naive >= 0 else f"${opportunity_vs_naive:,.0f}")
        st.metric("Charge Events", optimal_dispatch.attrs.get('charge_count', 0))
        st.metric("Discharge Events", optimal_dispatch.attrs.get('discharge_count', 0))
        charge_pct = (optimal_dispatch.attrs.get('charge_count', 0) / len(node_data)) * 100
        discharge_pct = (optimal_dispatch.attrs.get('discharge_count', 0) / len(node_data)) * 100
        st.progress(charge_pct / 100, text=f"Charging: {charge_pct:.1f}% of time")
        st.progress(discharge_pct / 100, text=f"Discharging: {discharge_pct:.1f}% of time")

    st.markdown("---")

    # Revenue comparison bar chart
    st.markdown("### Revenue Comparison")
    revenue_data = {
        'Strategy': ['Baseline\n(DA Only)', f'Improved\n(+{forecast_improvement}%)', 'Theoretical\nMax'],
        'Revenue': [naive_revenue, improved_revenue, optimal_revenue],
        'Color': ['#6B7280', '#4A9FB8', '#0A5F7A']
    }

    fig_revenue_bars = go.Figure()
    fig_revenue_bars.add_trace(go.Bar(
        x=revenue_data['Strategy'],
        y=revenue_data['Revenue'],
        marker_color=revenue_data['Color'],
        text=[f"${r:,.0f}" for r in revenue_data['Revenue']],
        textposition='outside',
        hovertemplate='%{x}<br>Revenue: $%{y:,.0f}<extra></extra>'
    ))

    fig_revenue_bars.update_layout(
        title=f"Strategy Performance - {strategy_type}",
        yaxis_title="Revenue ($)",
        height=400,
        showlegend=False
    )

    st.plotly_chart(fig_revenue_bars, width='stretch')

    # Show strategy-specific insights
    st.markdown("### Strategy Insights")
    if strategy_type == "Rolling Window Optimization":
        improvement_rate = ((improved_revenue - naive_revenue) / abs(naive_revenue) * 100) if naive_revenue != 0 else 0
        st.info(f"""
        **Rolling Window Strategy** with {window_hours}-hour lookahead window:
        - Achieves {improvement_rate:+.1f}% revenue improvement with {forecast_improvement}% better forecasts
        - Naturally handles temporal constraints (must charge before discharge)
        - Avoids threshold crossing sensitivity issues
        - Makes decisions based on price ranking within lookahead window
        - **More robust** to forecast errors than threshold-based
        """)
    else:  # Threshold-Based
        st.warning(f"""
        **Threshold-Based Strategy** using {int(charge_percentile*100)}th/{int(discharge_percentile*100)}th percentiles:
        - May show non-monotonic improvement (small forecast gains can reduce revenue)
        - Sensitive to threshold parameter selection
        - Simple and interpretable but suboptimal for arbitrage
        - Consider switching to Rolling Window for more consistent gains
        """)

with tab2:
    st.subheader(f"ERCOT Price Dynamics - {selected_node}")

    fig_price = go.Figure()

    fig_price.add_trace(go.Scatter(
        x=node_data['timestamp'],
        y=node_data['price_mwh_rt'],
        name='Real-Time Price',
        line=dict(color='#0A5F7A', width=2),
        hovertemplate='RT: $%{y:.2f}/MWh<extra></extra>'
    ))

    fig_price.add_trace(go.Scatter(
        x=node_data['timestamp'],
        y=node_data['price_mwh_da'],
        name='Day-Ahead Price',
        line=dict(color='#FF6B35', width=2, dash='dash'),
        hovertemplate='DA: $%{y:.2f}/MWh<extra></extra>'
    ))

    # Highlight negative prices
    negative_rt = node_data[node_data['price_mwh_rt'] < 0]
    if len(negative_rt) > 0:
        fig_price.add_trace(go.Scatter(
            x=negative_rt['timestamp'],
            y=negative_rt['price_mwh_rt'],
            mode='markers',
            name='Negative RT Prices',
            marker=dict(color='red', size=10, symbol='x'),
            hovertemplate='Negative Price: $%{y:.2f}/MWh<extra></extra>'
        ))

    # Add dynamic threshold lines
    fig_price.add_hline(y=discharge_thresh, line_dash="dot", line_color="green",
                        annotation_text=f"Discharge Threshold (${discharge_thresh:.2f}/MWh)")
    fig_price.add_hline(y=charge_thresh, line_dash="dot", line_color="orange",
                        annotation_text=f"Charge Threshold (${charge_thresh:.2f}/MWh)")
    fig_price.add_hline(y=0, line_dash="solid", line_color="gray",
                        annotation_text="$0/MWh")

    fig_price.update_layout(
        title=f"July 20, 2025 - {selected_node}",
        xaxis_title="Time",
        yaxis_title="Price ($/MWh)",
        height=500,
        hovermode='x unified'
    )

    st.plotly_chart(fig_price, width='stretch')

    # Price statistics and forecast error
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Price Statistics (July 20)")

        stats_data = {
            'Metric': [
                'Min DA Price',
                'Max DA Price',
                'Avg DA Price',
                'Min RT Price',
                'Max RT Price',
                'Avg RT Price',
                'Negative RT Hours',
                'Large Spread Hours (>$10)'
            ],
            'Value': [
                f"${node_data['price_mwh_da'].min():.2f}",
                f"${node_data['price_mwh_da'].max():.2f}",
                f"${node_data['price_mwh_da'].mean():.2f}",
                f"${node_data['price_mwh_rt'].min():.2f}",
                f"${node_data['price_mwh_rt'].max():.2f}",
                f"${node_data['price_mwh_rt'].mean():.2f}",
                f"{(node_data['price_mwh_rt'] < 0).sum()} hours",
                f"{node_data['extreme_event'].sum()} hours ({node_data['extreme_event'].sum()/len(node_data)*100:.1f}%)"
            ]
        }
        st.table(pd.DataFrame(stats_data))

    with col2:
        st.subheader("Forecast Error Distribution")
        fig_error_hist = px.histogram(
            node_data,
            x='forecast_error',
            nbins=20,
            title="DA Forecast Error (RT - DA)",
            labels={'forecast_error': 'Forecast Error ($/MWh)'},
            color_discrete_sequence=['#0A5F7A']
        )
        fig_error_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_error_hist, width='stretch')

        mae = node_data['forecast_error'].abs().mean()
        st.metric("Mean Absolute Error", f"${mae:.2f}/MWh")

with tab3:
    st.subheader("Battery State of Charge")

    fig_soc = go.Figure()

    fig_soc.add_trace(go.Scatter(
        x=optimal_dispatch['timestamp'],
        y=optimal_dispatch['soc'],
        name='Optimal Strategy (Perfect Foresight)',
        line=dict(color='#28A745', width=2.5),
        hovertemplate='SOC: %{y:.1f} MWh<extra></extra>'
    ))

    fig_soc.add_trace(go.Scatter(
        x=improved_dispatch['timestamp'],
        y=improved_dispatch['soc'],
        name=f'Improved Forecast (+{forecast_improvement}%)',
        line=dict(color='#FFC107', width=2),
        hovertemplate='SOC: %{y:.1f} MWh<extra></extra>'
    ))

    fig_soc.add_trace(go.Scatter(
        x=naive_dispatch['timestamp'],
        y=naive_dispatch['soc'],
        name='Baseline (Day-Ahead Only)',
        line=dict(color='#DC3545', width=2, dash='dash'),
        hovertemplate='SOC: %{y:.1f} MWh<extra></extra>'
    ))

    fig_soc.add_hline(y=battery_capacity_mwh, line_dash="dot",
                      annotation_text=f"Max Capacity ({battery_capacity_mwh} MWh)")
    fig_soc.add_hline(y=0, line_dash="dot",
                      annotation_text="Empty")

    fig_soc.update_layout(
        title="Battery State of Charge Over Time",
        xaxis_title="Time",
        yaxis_title="State of Charge (MWh)",
        height=500,
        hovermode='x unified'
    )

    st.plotly_chart(fig_soc, width='stretch')

    # Dispatch actions comparison
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Baseline Dispatch Actions")
        dispatch_counts_naive = naive_dispatch['dispatch'].value_counts()
        fig_dispatch_naive = px.pie(
            values=dispatch_counts_naive.values,
            names=dispatch_counts_naive.index,
            title="Distribution of Actions (Baseline)",
            color_discrete_map={'charge': '#28A745', 'discharge': '#DC3545', 'hold': '#6C757D'}
        )
        st.plotly_chart(fig_dispatch_naive, width='stretch')

    with col2:
        st.subheader("Optimal Dispatch Actions")
        dispatch_counts_opt = optimal_dispatch['dispatch'].value_counts()
        fig_dispatch_opt = px.pie(
            values=dispatch_counts_opt.values,
            names=dispatch_counts_opt.index,
            title="Distribution of Actions (Optimal)",
            color_discrete_map={'charge': '#28A745', 'discharge': '#DC3545', 'hold': '#6C757D'}
        )
        st.plotly_chart(fig_dispatch_opt, width='stretch')

with tab4:
    st.subheader("Cumulative Revenue Comparison")

    fig_revenue = go.Figure()

    fig_revenue.add_trace(go.Scatter(
        x=optimal_dispatch['timestamp'],
        y=optimal_dispatch['cumulative_revenue'],
        name='Optimal (Perfect Foresight)',
        line=dict(color='#28A745', width=3),
        hovertemplate='$%{y:,.0f}<extra></extra>'
    ))

    fig_revenue.add_trace(go.Scatter(
        x=improved_dispatch['timestamp'],
        y=improved_dispatch['cumulative_revenue'],
        name=f'Improved Forecast (+{forecast_improvement}%)',
        line=dict(color='#FFC107', width=2.5),
        hovertemplate='$%{y:,.0f}<extra></extra>'
    ))

    fig_revenue.add_trace(go.Scatter(
        x=naive_dispatch['timestamp'],
        y=naive_dispatch['cumulative_revenue'],
        name='Baseline (Day-Ahead)',
        line=dict(color='#DC3545', width=2, dash='dash'),
        hovertemplate='$%{y:,.0f}<extra></extra>'
    ))

    fig_revenue.update_layout(
        title="Revenue Accumulation Over 24 Hours",
        xaxis_title="Time",
        yaxis_title="Cumulative Revenue ($)",
        height=500,
        hovermode='x unified'
    )

    st.plotly_chart(fig_revenue, width='stretch')

    # Revenue metrics
    st.subheader("Revenue Breakdown")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Baseline (DA only)", f"${naive_revenue:,.2f}")
        charge_revenue_naive = (naive_dispatch[naive_dispatch['dispatch'] == 'charge']['power'] *
                               naive_dispatch[naive_dispatch['dispatch'] == 'charge']['price_mwh_rt']).sum()
        discharge_revenue_naive = (naive_dispatch[naive_dispatch['dispatch'] == 'discharge']['power'] *
                                  naive_dispatch[naive_dispatch['dispatch'] == 'discharge']['price_mwh_rt']).sum()
        st.caption(f"Charge cost: ${abs(charge_revenue_naive):,.0f}")
        st.caption(f"Discharge revenue: ${discharge_revenue_naive:,.0f}")

    with col2:
        st.metric(f"Improved (+{forecast_improvement}%)", f"${improved_revenue:,.2f}")
        improvement = improved_revenue - naive_revenue
        st.caption(f"Gain: ${improvement:,.2f}")
        st.caption(f"Improvement: {improvement_pct:.1f}%")

    with col3:
        st.metric("Optimal (Perfect)", f"${optimal_revenue:,.2f}")
        max_gain = optimal_revenue - naive_revenue
        st.caption(f"Max opportunity: ${max_gain:,.2f}")
        st.caption(f"Captured: {(opportunity_vs_improved/max_gain*100) if max_gain != 0 else 0:.1f}%")

with tab5:
    st.subheader("Revenue Opportunity Analysis")

    # Sensitivity analysis - compare strategies
    st.subheader("Impact of Forecast Accuracy on Revenue - Strategy Comparison")

    improvement_range = range(0, 51, 5)
    revenue_threshold = []
    revenue_rolling_window = []

    with st.spinner("Running sensitivity analysis across forecast improvement range..."):
        for imp in improvement_range:
            # Threshold strategy
            temp_dispatch_threshold = simulate_battery_dispatch(
                node_data, battery_capacity_mwh, battery_power_mw, efficiency,
                use_optimal=False,
                improvement_factor=imp/100,
                charge_percentile=charge_percentile,
                discharge_percentile=discharge_percentile
            )
            revenue_threshold.append(temp_dispatch_threshold['cumulative_revenue'].iloc[-1])

            # Rolling window strategy
            temp_dispatch_window = simulate_rolling_window_dispatch(
                node_data, battery_capacity_mwh, battery_power_mw, efficiency,
                window_hours=window_hours,
                improvement_factor=imp/100
            )
            revenue_rolling_window.append(temp_dispatch_window['cumulative_revenue'].iloc[-1])

    # Create comparison chart
    fig_sensitivity = go.Figure()

    fig_sensitivity.add_trace(go.Scatter(
        x=list(improvement_range),
        y=revenue_threshold,
        name='Threshold-Based',
        line=dict(color='#6B7280', width=2),
        mode='lines+markers',
        hovertemplate='Threshold<br>Improvement: %{x}%<br>Revenue: $%{y:,.0f}<extra></extra>'
    ))

    fig_sensitivity.add_trace(go.Scatter(
        x=list(improvement_range),
        y=revenue_rolling_window,
        name='Rolling Window',
        line=dict(color='#0A5F7A', width=3),
        mode='lines+markers',
        hovertemplate='Rolling Window<br>Improvement: %{x}%<br>Revenue: $%{y:,.0f}<extra></extra>'
    ))

    # Add reference lines
    fig_sensitivity.add_hline(
        y=optimal_revenue,
        line_dash="dash",
        line_color="green",
        annotation_text="Theoretical Maximum"
    )

    fig_sensitivity.update_layout(
        title="Revenue Sensitivity: Threshold vs Rolling Window Strategies",
        xaxis_title="Forecast Improvement (%)",
        yaxis_title="Revenue ($)",
        height=500,
        hovermode='x unified'
    )

    st.plotly_chart(fig_sensitivity, width='stretch')

    # Show insights about monotonicity
    st.markdown("### Strategy Comparison Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Threshold-Based Strategy:**")
        # Check if monotonic
        threshold_diffs = [revenue_threshold[i+1] - revenue_threshold[i] for i in range(len(revenue_threshold)-1)]
        negative_changes = sum(1 for d in threshold_diffs if d < 0)

        if negative_changes > 0:
            st.warning(f"⚠️ Non-monotonic: {negative_changes} instances where improvement REDUCED revenue")
        else:
            st.success("✓ Monotonic improvement")

        st.metric("Revenue Range",
                 f"${min(revenue_threshold):,.0f} to ${max(revenue_threshold):,.0f}")

    with col2:
        st.markdown("**Rolling Window Strategy:**")
        # Check if monotonic
        window_diffs = [revenue_rolling_window[i+1] - revenue_rolling_window[i] for i in range(len(revenue_rolling_window)-1)]
        negative_changes_window = sum(1 for d in window_diffs if d < 0)

        if negative_changes_window > 0:
            st.warning(f"⚠️ Non-monotonic: {negative_changes_window} instances where improvement REDUCED revenue")
        else:
            st.success("✓ Monotonic improvement")

        st.metric("Revenue Range",
                 f"${min(revenue_rolling_window):,.0f} to ${max(revenue_rolling_window):,.0f}")

    # Key insights
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Key Insights - July 20, 2025")
        st.markdown(f"""
        **Negative Price Events:**
        - {(node_data['price_mwh_rt'] < 0).sum()} hours with negative RT prices
        - Lowest: ${node_data['price_mwh_rt'].min():.2f}/MWh
        - Batteries can charge while getting paid

        **Price Volatility:**
        - {node_data['extreme_event'].sum()} hours with >$10/MWh spread
        - Max spread: ${node_data['price_spread'].max():.2f}/MWh
        - These hours drive 25% of revenue opportunity

        **Forecast Performance:**
        - Mean Absolute Error: ${node_data['forecast_error'].abs().mean():.2f}/MWh
        - {forecast_improvement}% improvement = ${opportunity_vs_improved:,.0f} gain
        """)

    with col2:
        st.subheader("Price Spread Analysis")
        fig_spread = px.scatter(
            node_data,
            x='price_mwh_da',
            y='price_mwh_rt',
            title="Day-Ahead vs Real-Time Prices",
            labels={'price_mwh_da': 'DA Price ($/MWh)',
                   'price_mwh_rt': 'RT Price ($/MWh)'},
            color='price_spread',
            color_continuous_scale='Reds',
            hover_data=['timestamp']
        )

        # Add diagonal line (perfect forecast)
        min_price = min(node_data['price_mwh_da'].min(), node_data['price_mwh_rt'].min())
        max_price = max(node_data['price_mwh_da'].max(), node_data['price_mwh_rt'].max())
        fig_spread.add_trace(go.Scatter(
            x=[min_price, max_price],
            y=[min_price, max_price],
            mode='lines',
            name='Perfect Forecast',
            line=dict(color='gray', dash='dash')
        ))

        st.plotly_chart(fig_spread, width='stretch')

with tab6:
    st.subheader("Decision Timeline - Charge/Discharge Schedule")

    # Create gantt-style chart showing when battery is charging, discharging, or holding
    st.markdown("### Dispatch Schedule Comparison")

    # Create figure with subplots for each strategy
    fig_timeline = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Baseline (DA Only)", f"Improved (+{forecast_improvement}%)", "Theoretical Maximum"),
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.33, 0.33, 0.33]
    )

    # Define colors for actions
    action_colors = {'charge': '#28A745', 'discharge': '#DC3545', 'hold': '#6C757D'}

    # Helper function to add dispatch bars
    def add_dispatch_bars(fig, dispatch_data, row_num):
        for action, color in action_colors.items():
            action_data = dispatch_data[dispatch_data['dispatch'] == action].copy()
            if len(action_data) > 0:
                fig.add_trace(go.Bar(
                    x=action_data['timestamp'],
                    y=[1] * len(action_data),
                    name=action.capitalize() if row_num == 1 else None,
                    marker_color=color,
                    showlegend=(row_num == 1),
                    hovertemplate=f'{action.capitalize()}<br>%{{x}}<br>Price: $%{{customdata[0]:.2f}}/MWh<extra></extra>',
                    customdata=action_data[['price_mwh_rt']].values
                ), row=row_num, col=1)

    # Add data for each strategy
    add_dispatch_bars(fig_timeline, naive_dispatch, 1)
    add_dispatch_bars(fig_timeline, improved_dispatch, 2)
    add_dispatch_bars(fig_timeline, optimal_dispatch, 3)

    fig_timeline.update_layout(
        height=600,
        barmode='stack',
        title_text=f"Battery Dispatch Timeline - {strategy_type}",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig_timeline.update_yaxes(visible=False)
    fig_timeline.update_xaxes(title_text="Time", row=3, col=1)

    st.plotly_chart(fig_timeline, width='stretch')

    # Show price overlay
    st.markdown("### Price Context for Dispatch Decisions")

    fig_price_dispatch = go.Figure()

    # Add price line
    fig_price_dispatch.add_trace(go.Scatter(
        x=node_data['timestamp'],
        y=node_data['price_mwh_rt'],
        name='Real-Time Price',
        line=dict(color='#0A5F7A', width=2),
        yaxis='y'
    ))

    # Add dispatch markers for improved strategy
    charge_times = improved_dispatch[improved_dispatch['dispatch'] == 'charge']
    discharge_times = improved_dispatch[improved_dispatch['dispatch'] == 'discharge']

    fig_price_dispatch.add_trace(go.Scatter(
        x=charge_times['timestamp'],
        y=charge_times['price_mwh_rt'],
        mode='markers',
        name='Charge',
        marker=dict(color='#28A745', size=12, symbol='triangle-down'),
        hovertemplate='Charge<br>Time: %{x}<br>Price: $%{y:.2f}/MWh<extra></extra>'
    ))

    fig_price_dispatch.add_trace(go.Scatter(
        x=discharge_times['timestamp'],
        y=discharge_times['price_mwh_rt'],
        mode='markers',
        name='Discharge',
        marker=dict(color='#DC3545', size=12, symbol='triangle-up'),
        hovertemplate='Discharge<br>Time: %{x}<br>Price: $%{y:.2f}/MWh<extra></extra>'
    ))

    fig_price_dispatch.update_layout(
        title=f"Dispatch Decisions Overlaid on Price - Improved Strategy ({forecast_improvement}% improvement)",
        xaxis_title="Time",
        yaxis_title="Price ($/MWh)",
        height=400,
        hovermode='x unified'
    )

    st.plotly_chart(fig_price_dispatch, width='stretch')

    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Charge Count", improved_dispatch.attrs.get('charge_count', 0))
        avg_charge_price = improved_dispatch[improved_dispatch['dispatch'] == 'charge']['price_mwh_rt'].mean()
        st.caption(f"Avg charge price: ${avg_charge_price:.2f}/MWh" if not pd.isna(avg_charge_price) else "No charges")

    with col2:
        st.metric("Discharge Count", improved_dispatch.attrs.get('discharge_count', 0))
        avg_discharge_price = improved_dispatch[improved_dispatch['dispatch'] == 'discharge']['price_mwh_rt'].mean()
        st.caption(f"Avg discharge price: ${avg_discharge_price:.2f}/MWh" if not pd.isna(avg_discharge_price) else "No discharges")

    with col3:
        if not pd.isna(avg_charge_price) and not pd.isna(avg_discharge_price):
            spread = avg_discharge_price - avg_charge_price
            st.metric("Avg Price Spread", f"${spread:.2f}/MWh")
            st.caption(f"Theoretical gain per cycle")
        else:
            st.metric("Avg Price Spread", "N/A")

with tab7:
    st.subheader("Optimization Analysis")

    if strategy_type == "Rolling Window Optimization":
        st.markdown(f"### Rolling Window Strategy (Lookahead: {window_hours} hours)")

        st.info(f"""
        **How it works:**
        - At each hour, look ahead {window_hours} hours into the future
        - Charge if current price is the MINIMUM in the window (cheap now, might be expensive later)
        - Discharge if current price is the MAXIMUM in the window (expensive now, might be cheap later)
        - Hold otherwise (current price is neither min nor max)

        **Advantages:**
        - No threshold crossing sensitivity issues
        - Better forecast → better price ranking → better decisions
        - Naturally handles temporal constraints
        """)

        # Show example hours where window optimization helped
        st.markdown("### Example: How Lookahead Window Improves Decisions")

        # Find an interesting hour (where improved made different decision than baseline)
        different_decisions = improved_dispatch[
            improved_dispatch['dispatch'] != naive_dispatch['dispatch']
        ].copy()

        if len(different_decisions) > 0:
            example_hour = different_decisions.iloc[0]
            example_idx = different_decisions.index[0]
            window_end = min(example_idx + window_hours, len(node_data))

            st.markdown(f"**Hour {example_idx}: Improved forecast made DIFFERENT decision than baseline**")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Baseline Decision:**")
                st.markdown(f"- Action: **{naive_dispatch.iloc[example_idx]['dispatch'].upper()}**")
                st.markdown(f"- Price: ${example_hour['price_mwh_rt']:.2f}/MWh")

            with col2:
                st.markdown("**Improved Decision:**")
                st.markdown(f"- Action: **{improved_dispatch.iloc[example_idx]['dispatch'].upper()}**")
                st.markdown(f"- Price: ${example_hour['price_mwh_rt']:.2f}/MWh")

            # Show window prices
            window_data = node_data.iloc[example_idx:window_end].copy()
            fig_window = go.Figure()

            fig_window.add_trace(go.Scatter(
                x=window_data['timestamp'],
                y=window_data['price_mwh_rt'],
                mode='lines+markers',
                name='RT Price',
                line=dict(color='#0A5F7A')
            ))

            fig_window.add_vline(
                x=example_hour['timestamp'],
                line_dash="dash",
                line_color="red",
                annotation_text="Decision Point"
            )

            fig_window.update_layout(
                title=f"{window_hours}-Hour Lookahead Window from Hour {example_idx}",
                xaxis_title="Time",
                yaxis_title="Price ($/MWh)",
                height=300
            )

            st.plotly_chart(fig_window, width='stretch')

        else:
            st.info("Baseline and improved strategies made identical decisions for all hours with current parameters.")

    elif strategy_type == "Threshold-Based":
        st.markdown(f"### Threshold-Based Strategy")
        st.markdown(f"**Charge threshold:** {int(charge_percentile*100)}th percentile")
        st.markdown(f"**Discharge threshold:** {int(discharge_percentile*100)}th percentile")

        if 'charge_thresh' in locals() and 'discharge_thresh' in locals():
            st.metric("Charge Below", f"${charge_thresh:.2f}/MWh")
            st.metric("Discharge Above", f"${discharge_thresh:.2f}/MWh")

        # Show price distribution with thresholds
        fig_price_dist = go.Figure()

        fig_price_dist.add_trace(go.Histogram(
            x=node_data['price_mwh_rt'],
            name='RT Price Distribution',
            nbinsx=20,
            marker_color='#0A5F7A'
        ))

        if 'charge_thresh' in locals():
            fig_price_dist.add_vline(
                x=charge_thresh,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Charge < ${charge_thresh:.2f}"
            )

        if 'discharge_thresh' in locals():
            fig_price_dist.add_vline(
                x=discharge_thresh,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Discharge > ${discharge_thresh:.2f}"
            )

        fig_price_dist.update_layout(
            title="Price Distribution with Trading Thresholds",
            xaxis_title="Price ($/MWh)",
            yaxis_title="Frequency",
            height=400
        )

        st.plotly_chart(fig_price_dist, width='stretch')

    else:  # Threshold-Based
        st.markdown(f"### Threshold-Based Strategy Analysis")

        st.info("""
        **Understanding the 3 Scenarios:**
        - **Scenario 1 (Baseline):** Uses DA forecasts as-is
        - **Scenario 2 (Improved):** Corrects DA forecasts by the slider percentage toward actual RT prices
        - **Scenario 3 (Perfect):** Uses actual RT prices (theoretical maximum)
        """)

        # Analyze where perfect foresight made better decisions
        better_decisions = optimal_dispatch[
            optimal_dispatch['dispatch'] != improved_dispatch['dispatch']
        ]

        st.metric("Hours where perfect foresight made different decision than improved scenario",
                 len(better_decisions))

        if len(better_decisions) > 0:
            decision_diff_revenue = optimal_revenue - improved_revenue
            st.metric("Revenue gap between improved and perfect",
                     f"${decision_diff_revenue:,.0f}")
            st.caption("This gap represents the remaining opportunity from better forecasting")

# ============================================================================
# FOOTER AND EXPORT
# ============================================================================

st.markdown("---")

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    st.markdown("""
    ### About This Dashboard
    This interactive tool demonstrates how improved renewable energy forecasting
    increases battery storage revenue in ERCOT markets. Built for the Engie Urja
    AI Challenge 2025.

    **Key Features:**
    - Real ERCOT price data analysis
    - Two dispatch strategies: Threshold-Based and Rolling Window Optimization
    - Each strategy tested with 3 forecast quality scenarios (DA only, Improved, Perfect)
    - Battery system presets based on 136 real Texas systems
    - Revenue sensitivity analysis
    """)

with col2:
    st.markdown("""
    ### Data Sources
    **Price Data:** ERCOT Day-Ahead and Real-Time Markets (July 20, 2025)

    **Battery Market Data:** U.S. Energy Information Administration (EIA) Form EIA-860
    (2024 Annual Electric Generator Report)

    **Settlement Points:** Wind resource nodes (BUFF_GAP_ALL, BAIRDWND_ALL,
    CEDROHI_CHW1, SWTWN4_WND45, WH_WIND_ALL)
    """)

with col3:
    if st.button("📥 Export Results", type="primary"):
        export_data = pd.DataFrame({
            'Metric': [
                'Date',
                'Settlement Point',
                'Battery Capacity (MWh)',
                'Battery Power (MW)',
                'Efficiency',
                'Forecast Improvement (%)',
                'Baseline Revenue ($)',
                'Improved Revenue ($)',
                'Optimal Revenue ($)',
                'Revenue Gain ($)',
                'Revenue Improvement (%)',
                'Max Opportunity ($)',
                'Negative Price Hours',
                'Extreme Event Hours'
            ],
            'Value': [
                'July 20, 2025',
                selected_node,
                battery_capacity_mwh,
                battery_power_mw,
                efficiency,
                forecast_improvement,
                f"{naive_revenue:.2f}",
                f"{improved_revenue:.2f}",
                f"{optimal_revenue:.2f}",
                f"{opportunity_vs_improved:.2f}",
                f"{improvement_pct:.1f}",
                f"{opportunity_vs_naive:.2f}",
                (node_data['price_mwh_rt'] < 0).sum(),
                node_data['extreme_event'].sum()
            ]
        })

        csv = export_data.to_csv(index=False)
        st.download_button(
            label="Download Summary CSV",
            data=csv,
            file_name=f"zentus_ercot_analysis_{selected_node}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6C757D;'>
    <p><strong>Zentus</strong> - Intelligent Forecasting for Renewables</p>
    <p>Stanford Doerr School of Sustainability Accelerator Fellow</p>
    <p>Engie Urja AI Challenge 2025</p>
    <p>Contact: info@zentus.io | zentus.io</p>
</div>
""", unsafe_allow_html=True)

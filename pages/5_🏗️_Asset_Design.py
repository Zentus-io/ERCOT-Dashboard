"""
Hybrid Asset Design & Optimization Page
Zentus - ERCOT Battery Revenue Dashboard

This page allows users to:
1. Design hybrid assets (Solar + Storage) using real market data and generation profiles.
2. Compare the currently selected asset against an optimal configuration.
3. Visualize clipping capture and revenue potential.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import numpy as np
import pandas as pd

import altair as alt
import plotly.graph_objects as go
import streamlit as st

from config.page_config import configure_page
from core.battery.battery import BatterySpecs
from core.battery.simulator import BatterySimulator
from core.battery.strategies import (
    ClippingOnlyStrategy,
    ClippingAwareMPCStrategy,
    calculate_dt,
    MPCStrategy,
    RollingWindowStrategy,
    ThresholdStrategy,
)
from core.battery.hybrid_strategy import HybridDispatchStrategy
from core.data.loaders import SupabaseDataLoader
from ui.components.header import render_header
from ui.components.sidebar import render_sidebar
from ui.styles.custom_css import apply_custom_styles
from utils.simulation_runner import run_or_get_cached_simulation
from utils.state import get_date_range_str, get_state, has_valid_config
from utils.data_validation import (
    validate_solar_price_alignment,
    validate_clipped_energy,
    validate_hybrid_configuration,
    display_validation_warnings
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

configure_page("Hybrid Asset Design")
apply_custom_styles()

# ============================================================================
# HEADER AND SIDEBAR
# ============================================================================

render_header()
render_sidebar()

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.header("🏗️ Hybrid Asset Design & Optimization")


# Check if configuration is valid
if not has_valid_config():
    st.warning(
        "⚠️ Please configure battery specifications and select a settlement point in the sidebar to begin analysis.")
    st.stop()

# Get state
state = get_state()

# --- STEP 1: CORE DATA INTEGRATION (GUARDRAILS) ---
if state.price_data is None or state.price_data.empty:
    st.error("⚠️ Price data not loaded. Please refresh the page or check data availability in the sidebar.")
    st.stop()

if state.selected_node is None:
    st.error("⚠️ No settlement point selected. Please select a node in the sidebar.")
    st.stop()

# Filter data for selected node
# Handle different column names for node
node_col = 'node' if 'node' in state.price_data.columns else 'settlement_point'
if node_col not in state.price_data.columns:
    node_col = state.price_data.columns[0] # Fallback

df_prices = state.price_data[state.price_data[node_col] == state.selected_node].copy()

if df_prices.empty:
    st.warning(f"⚠️ No data available for node {state.selected_node}. Please select a different node or date range.")
    st.stop()

# Ensure timestamp index
if 'timestamp' in df_prices.columns:
    df_prices['timestamp'] = pd.to_datetime(df_prices['timestamp'])
    df_prices = df_prices.set_index('timestamp', drop=False).sort_index()

# Alias for compatibility with existing code in Tab 2
node_data = df_prices

# ============================================================================
# HYBRID DESIGN
# ============================================================================

st.subheader(f"Asset Optimization: {state.selected_node}")

with st.expander("📖 How to use this page", expanded=False):
    st.markdown("""
    **Goal:** Design a hybrid Solar + Storage asset and optimize the battery size to capture "clipped" energy (energy lost when solar generation exceeds the grid interconnection limit).

    ### 1. Configure Asset (Top Left)
    - **Solar Capacity:** Total DC capacity of the solar array.
    - **Interconnection Limit:** Max power export to grid (POI limit).
    - **Solar Profile:** Upload custom CSV or use regional potential from database.

    ### 2. Analyze Operational Profile (Top Right)
    - **Yellow:** Available Solar Resource.
    - **Orange:** Exported Energy (capped at limit).
    - **Green:** Clipped Energy (Free fuel for battery!).

    ### 3. Optimize Battery (Bottom)
    - Run a full grid search to find the optimal battery power/duration to capture clipping.
    """)

# --- HELPER FUNCTIONS ---
@st.cache_data(ttl=3600)
def get_smart_solar_profile(_df_prices, _uploaded_file=None, _node_name=None):
    """
    Generates or loads a solar profile aligned with df_prices index.
    Returns a DataFrame with columns ['gen_mw', 'forecast_mw', 'potential_mw'] (normalized 0-1)
    and the source type string.
    # Cache invalidation trigger
    """
    target_index = _df_prices.index
    start_date = target_index.min().date()
    end_date = target_index.max().date()
    
    # Initialize result DataFrame
    result_df = pd.DataFrame(index=target_index)
    result_df['gen_mw'] = 0.0
    result_df['forecast_mw'] = 0.0
    result_df['potential_mw'] = 0.0
    
    # 1. Uploaded File
    if _uploaded_file is not None:
        try:
            df_solar = pd.read_csv(_uploaded_file)
            # Find generation column
            possible_cols = ['Generation', 'generation', 'Solar', 'solar', 'Output', 'output', 'gen_mw']
            target_col = next((c for c in possible_cols if c in df_solar.columns), None)
            
            if not target_col and pd.api.types.is_numeric_dtype(df_solar.iloc[:, 0]):
                target_col = df_solar.columns[0]
                
            if target_col:
                # Normalize
                raw = df_solar[target_col].values
                mx = np.max(raw)
                profile_values = raw / mx if mx > 0 else raw
                
                # Tile/Slice to match target length
                if len(profile_values) >= len(target_index):
                    aligned_values = profile_values[:len(target_index)]
                else:
                    repeats = int(np.ceil(len(target_index) / len(profile_values)))
                    aligned_values = np.tile(profile_values, repeats)[:len(target_index)]
                    
                result_df['gen_mw'] = aligned_values
                # Assume forecast/potential same as actuals for CSV upload
                result_df['forecast_mw'] = aligned_values
                result_df['potential_mw'] = aligned_values
                    
                return result_df, "Uploaded CSV"
        except Exception as e:
            st.error(f"Error parsing CSV: {e}")
    
    # 2. Database Fetch
    if _node_name and state.data_source == 'database':
        try:
            loader = SupabaseDataLoader()
            # Try fetching Solar data
            df_gen = loader.load_generation_data(
                node=_node_name, 
                fuel_type='Solar',
                start_date=start_date,
                end_date=end_date
            )
            
            if not df_gen.empty:
                # Reindex to match target_index (fill missing with 0 or interpolate)
                # The DB data might be hourly, target might be 15-min.
                
                # Ensure both indices are TZ-naive for alignment
                if df_gen.index.tz is not None:
                    df_gen.index = df_gen.index.tz_convert(None)
                
                # Helper to align and norm
                def align_and_norm(col_name):
                    if col_name not in df_gen.columns:
                        return pd.Series(0.0, index=target_index)
                    
                    # Ensure target_index is also naive (it should be, but be safe)
                    safe_target = target_index
                    if safe_target.tz is not None:
                        safe_target = safe_target.tz_convert(None)
                        
                    aligned = df_gen[col_name].reindex(safe_target).interpolate(method='time').fillna(0)
                    mx = aligned.max()
                    if mx > 0:
                        return aligned / mx
                    return aligned

                result_df['gen_mw'] = align_and_norm('gen_mw')
                result_df['forecast_mw'] = align_and_norm('forecast_mw')
                result_df['potential_mw'] = align_and_norm('potential_mw')
                
                # If potential is missing (all 0), use gen_mw
                if result_df['potential_mw'].max() == 0:
                        result_df['potential_mw'] = result_df['gen_mw']
                    
                return result_df, "Database (Regional Forecast)"
        except Exception as e:
            # st.warning(f"DB Fetch Error: {e}")
            pass

    # 3. Fallback: Synthetic Bell Curve
    hours = np.linspace(0, 24, 96)
    daily_profile = np.exp(-((hours - 12)**2) / (2 * 3**2))
    daily_profile[daily_profile < 0.01] = 0
    
    days = int(np.ceil(len(target_index) / 96))
    full_profile = np.tile(daily_profile, days)[:len(target_index)]
    
    result_df['gen_mw'] = full_profile
    result_df['forecast_mw'] = full_profile
    result_df['potential_mw'] = full_profile
    
    return result_df, "Synthetic (Bell Curve)"

# --- LAYOUT START ---
col_top_left, col_top_right = st.columns([1, 3])

with col_top_left:
    st.markdown("### Asset Config")
    
    # Default values from State (Current Asset)
    current_power = state.battery_specs.power_mw
    current_energy = state.battery_specs.capacity_mwh
    
    # Solar Capacity Input
    solar_capacity_mw = st.number_input(
        "Solar Capacity (MW)",
        min_value=0.0,
        value=float(current_power * 1.5), # Default to 1.5x POI
        step=10.0,
        help="Total DC capacity of the solar array."
    )
    
    # Interconnection Limit Input
    interconnection_limit_mw = st.number_input(
        "Interconnection Limit (MW)",
        min_value=0.0,
        value=float(current_power), # Default to Battery Power (POI)
        step=10.0,
        help="Maximum power allowed to be exported to the grid (POI limit)."
    )
    
    st.markdown("---")
    st.markdown("### Solar Profile")
    uploaded_file = st.file_uploader("Upload Solar Profile (CSV)", type=['csv'])
    
    # Fetch Profile
    solar_profile, source_type = get_smart_solar_profile(df_prices, uploaded_file, state.selected_node)

    if "Database" in source_type:
            st.success(f"✅ Using {source_type}")
    elif "Synthetic" in source_type:
            st.caption(f"Using {source_type}. Upload CSV or use Database mode for real data.")
    else:
            st.info(f"Using {source_type}")

    # Validate solar/price data alignment
    is_valid_alignment, alignment_warnings = validate_solar_price_alignment(solar_profile, df_prices)
    if alignment_warnings:
        display_validation_warnings(alignment_warnings, title="Solar/Price Data Alignment")

# --- CALCULATION ENGINE ---

# 1. Calculate Solar Vectors
# Use Potential MW (Resource Availability) for simulation if available, else Actuals
# The profile is already normalized 0-1

# CRITICAL: Remove duplicate timestamps if they exist (can happen on date range changes)
if df_prices.index.duplicated().any():
    st.warning(f"⚠️ Detected {df_prices.index.duplicated().sum()} duplicate timestamps in price data. Removing duplicates...")
    df_prices = df_prices[~df_prices.index.duplicated(keep='first')]

if solar_profile.index.duplicated().any():
    st.warning(f"⚠️ Detected {solar_profile.index.duplicated().sum()} duplicate timestamps in solar profile. Removing duplicates...")
    solar_profile = solar_profile[~solar_profile.index.duplicated(keep='first')]

df_sim = df_prices.copy()
df_sim['Solar_MW'] = solar_profile['potential_mw'] * solar_capacity_mw
df_sim['Export_MW'] = np.minimum(df_sim['Solar_MW'], interconnection_limit_mw)
df_sim['Clipped_MW'] = np.maximum(0, df_sim['Solar_MW'] - interconnection_limit_mw)

# === DEBUG: Data Alignment Check ===
# st.write("### 🔍 DEBUG: Data Analysis")
# debug_info = f"""
# **Data Dimensions:**
# - Price data rows: {len(df_prices)}
# - Solar profile rows: {len(solar_profile)}
# - Simulation data rows: {len(df_sim)}
# - Date range: {df_sim.index.min()} to {df_sim.index.max()}
# - Data frequency: {pd.infer_freq(df_sim.index) or 'Mixed/Unknown'}

# **Solar Profile Statistics:**
# - Potential MW (normalized): min={solar_profile['potential_mw'].min():.3f}, max={solar_profile['potential_mw'].max():.3f}, mean={solar_profile['potential_mw'].mean():.3f}
# - Gen MW (normalized): min={solar_profile['gen_mw'].min():.3f}, max={solar_profile['gen_mw'].max():.3f}, mean={solar_profile['gen_mw'].mean():.3f}
# - Non-zero potential values: {(solar_profile['potential_mw'] > 0.01).sum()} / {len(solar_profile)}

# **Calculated Vectors:**
# - Solar MW: min={df_sim['Solar_MW'].min():.2f}, max={df_sim['Solar_MW'].max():.2f}, mean={df_sim['Solar_MW'].mean():.2f}
# - Export MW: min={df_sim['Export_MW'].min():.2f}, max={df_sim['Export_MW'].max():.2f}, mean={df_sim['Export_MW'].mean():.2f}
# - Clipped MW: min={df_sim['Clipped_MW'].min():.2f}, max={df_sim['Clipped_MW'].max():.2f}, mean={df_sim['Clipped_MW'].mean():.2f}
# - Clipped (non-zero): {(df_sim['Clipped_MW'] > 0.1).sum()} intervals out of {len(df_sim)}
# - Total Clipped Energy: {df_sim['Clipped_MW'].sum() / 4:.1f} MWh

# **Configuration:**
# - Solar Capacity: {solar_capacity_mw:.1f} MW
# - Interconnection Limit: {interconnection_limit_mw:.1f} MW
# - Expected Max Clipping: {max(0, solar_capacity_mw - interconnection_limit_mw):.1f} MW (if solar at 100%)
# """
# st.code(debug_info, language="text")

# Validate clipping calculations
is_valid_clipping, clipping_warnings = validate_clipped_energy(
    df_sim['Clipped_MW'],
    df_sim['Solar_MW'],
    interconnection_limit_mw
)
if clipping_warnings:
    with col_top_left:
        display_validation_warnings(clipping_warnings, title="Clipping Calculation")

# Validate hybrid configuration
current_capacity = state.battery_specs.capacity_mwh
is_valid_config, config_warnings = validate_hybrid_configuration(
    solar_capacity_mw,
    interconnection_limit_mw,
    current_power,
    current_capacity
)
if config_warnings:
    with col_top_left:
        display_validation_warnings(config_warnings, title="Hybrid Configuration")

# --- REVENUE CALCULATION ---

# Calculate time step duration (0.25h for 15-min RTM, 1.0h for hourly DAM)
dt_hours = calculate_dt(df_sim)

# 1. Base Solar Revenue (Same for everyone)
df_sim['Solar_Revenue'] = df_sim['Export_MW'] * df_sim['price_mwh_rt'] * dt_hours
base_solar_revenue = df_sim['Solar_Revenue'].sum()

# Pre-calculate Daily Stats (Used for Heuristic & Sweep)
df_sim['Date'] = df_sim.index.date
daily_stats = df_sim.groupby('Date').agg({
    'Clipped_MW': 'sum', # Sum of MW per interval
    'price_mwh_rt': ['max', 'min']
})
# Flatten columns
daily_stats.columns = ['Daily_Clipped_Sum_MW', 'Daily_Max_Price', 'Daily_Min_Price']
daily_stats['Daily_Clipped_MWh'] = daily_stats['Daily_Clipped_Sum_MW'] * dt_hours

# Ensure battery efficiency is defined
batt_eff = state.battery_specs.efficiency

# Prepare data for simulation (needed for helper function)
df_hybrid = df_sim.copy()
df_hybrid['clipped_mw'] = df_hybrid['Clipped_MW']
improvement_factor = state.forecast_improvement / 100.0


# Define simulation function (Moved up to be used by Current Asset calc)
def simulate_battery_config(power_mw: float, duration_h: float) -> dict:
        """
        Run full battery simulation for a specific configuration using ClippingOnlyStrategy.

        This simulates clipping-only optimization where the battery:
        - Charges ONLY from clipped solar energy (free, mandatory)
        - Uses selected dispatch strategy to optimize discharge timing
        - Never charges from grid (no arbitrage)

        Parameters
        ----------
        power_mw : float
            Battery power rating (MW)
        duration_h : float
            Battery duration (hours)
        debug : bool
            If True, print detailed debug information

        Returns
        -------
        dict
            Simulation results including power, capacity, total_revenue, and metrics
        """
        # Create battery specs
        capacity_mwh = power_mw * duration_h
        specs = BatterySpecs(
            capacity_mwh=capacity_mwh,
            power_mw=power_mw,
            efficiency=batt_eff,
            initial_soc=0.05  # Start at min SOC (effectively empty) for clipping-only optimization
        )

        # Create base strategy (use strategy from sidebar settings)
        if state.strategy_type == "Threshold-Based":
            base_strategy = ThresholdStrategy(
                state.charge_percentile,
                state.discharge_percentile
            )
        elif state.strategy_type == "Rolling Window Optimization":
            base_strategy = RollingWindowStrategy(state.window_hours)
        elif state.strategy_type == "MPC (Rolling Horizon)":
            # Use Clipping-Aware MPC for better optimization
            base_strategy = ClippingAwareMPCStrategy(state.horizon_hours)
        else:
            # Default to Rolling Window
            base_strategy = RollingWindowStrategy(window_hours=6)

        # Wrap with ClippingOnlyStrategy (clipping capture only, no grid arbitrage)
        clipping_strategy = ClippingOnlyStrategy(
            base_strategy=base_strategy
        )

        # Run simulation
        simulator = BatterySimulator(specs)
        result = simulator.run(
            df_hybrid,
            clipping_strategy,
            improvement_factor=improvement_factor
        )

        # Calculate total hybrid revenue (solar + battery)
        total_hybrid_revenue = base_solar_revenue + result.total_revenue

        # Extract metadata from simulation result (NOT from original strategy!)
        # The result.metadata comes from the reconstructed strategy that actually ran
        metadata = result.metadata
        


        return {
            'power_mw': power_mw,
            'duration_h': duration_h,
            'capacity_mwh': capacity_mwh,
            'battery_revenue': result.total_revenue,
            'total_revenue': total_hybrid_revenue,
            'clipped_captured': metadata.get('clipped_energy_captured', 0),
            'curtailed_clipping': metadata.get('curtailed_clipping', 0),
            'grid_arbitrage': metadata.get('grid_arbitrage_revenue', 0)  # Always 0 for clipping-only
        }

# Calculate current asset revenue (using Clipping-Only logic for consistency)
with st.spinner("Calculating current asset revenue..."):
    # Use the same simulation logic as the optimization grid
    # This ensures apples-to-apples comparison (no grid arbitrage in either)
    current_res = simulate_battery_config(current_power, current_energy / current_power if current_power > 0 else 0)
    current_batt_revenue = current_res['battery_revenue']
    current_hybrid_rev = current_res['total_revenue']

# --- OPERATIONAL CHART & METRICS (TOP RIGHT) ---
with col_top_right:
    st.markdown("### ⚡ Operational Profile (Solar + Clipping)")
    plot_df = df_sim.copy()
    
    # Add actuals for comparison if available
    plot_df['Actual_Generation_MW'] = solar_profile['gen_mw'] * solar_capacity_mw
        
    fig_hybrid = go.Figure()
    
    # Plot Potential (Resource)
    fig_hybrid.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df['Solar_MW'],
        mode='lines', name='Solar Potential (Resource)',
        line=dict(color='#FFD700', width=1, dash='dot')
    ))
    
    # Plot Actual Export (Simulated)
    fig_hybrid.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df['Export_MW'],
        mode='lines', name='Simulated Export',
        fill='tozeroy', line=dict(color='#FFA500', width=0), stackgroup='one'
    ))
    
    fig_hybrid.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df['Clipped_MW'],
        mode='lines', name='Clipped Energy',
        fill='tonexty', line=dict(color='#28a745', width=0), stackgroup='one'
    ))
    
    fig_hybrid.add_trace(go.Scatter(
        x=[plot_df.index.min(), plot_df.index.max()],
        y=[interconnection_limit_mw, interconnection_limit_mw],
        mode='lines', name='Interconnection Limit',
        line=dict(color='#dc3545', dash='dash', width=2)
    ))

    # Add Price Trace (Secondary Axis)
    fig_hybrid.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df['price_mwh_rt'],
        mode='lines', name='RTM Price',
        line=dict(color='rgba(100, 100, 100, 0.5)', width=1),
        yaxis='y2'
    ))
    
    fig_hybrid.update_layout(
        title="Hybrid Asset Operation",
        xaxis_title="Time",
        yaxis_title="Power (MW)",
        yaxis2=dict(
            title="Price ($/MWh)",
            overlaying='y',
            side='right',
            showgrid=False
        ),
        height=350, # Slightly reduced height
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    st.plotly_chart(fig_hybrid, width='stretch')

    # --- CURRENT PERFORMANCE METRICS (MOVED HERE) ---
    st.markdown("#### Current Asset Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Battery Power", f"{current_power:.0f} MW")
    m2.metric("Battery Capacity", f"{current_capacity:.1f} MWh")
    m3.metric("Battery Revenue", f"${current_batt_revenue:,.0f}", help="Revenue from clipping capture only (no grid arbitrage)")
    m4.metric("Total Hybrid Revenue", f"${current_hybrid_rev:,.0f}")


# ============================================================================
# ALTAIR CHART HELPER FOR AMR VISUALIZATION
# ============================================================================

def create_optimization_chart(history_df, bounds_dict, tick_positions=None, alternative_configs=None):
    """
    Create multi-layer Altair chart for battery optimization with AMR support.

    Parameters:
    -----------
    history_df : pd.DataFrame
        Dataframe with columns: x1, x2, y1, y2, revenue, power_mw, duration_h, is_skipped
    bounds_dict : dict
        {'min_d': float, 'max_d': float, 'min_p': float, 'max_p': float,
         'current_p': float (optional), 'current_d': float (optional)}
    tick_positions : dict, optional
        {'duration_values': list, 'power_values': list}
        Discrete tick positions for grid centers
    alternative_configs : dict, optional
        {'min_power': (p, d), 'min_capacity': (p, d)}
        Coordinates for alternative battery configurations

    Returns:
    --------
    alt.Chart: Layered Altair chart
    """
    
    if not history_df.empty:
        # Clean data: Ensure revenue is numeric and drop NaNs/Infs
        history_df = history_df.copy()
        history_df['revenue'] = pd.to_numeric(history_df['revenue'], errors='coerce')
        history_df = history_df.dropna(subset=['revenue'])

        min_rev = history_df['revenue'].min()
        max_rev = history_df['revenue'].max()
    else:
        st.warning(f"⚠️ Chart data is empty! No simulation results to display.")
        min_rev, max_rev = 0, 1

    # Prepare axis configuration
    if tick_positions and 'duration_values' in tick_positions:
        duration_axis = alt.Axis(
            values=tick_positions['duration_values'],
            labelAngle=0,
            title='Battery Duration (hours)'
        )
    else:
        duration_axis = alt.Axis(title='Battery Duration (hours)')

    if tick_positions and 'power_values' in tick_positions:
        power_axis = alt.Axis(
            values=tick_positions['power_values'],
            title='Battery Power (MW)'
        )
    else:
        power_axis = alt.Axis(title='Battery Power (MW)')

    # Layer 1: Multi-Level AMR Heatmap
    # Render all grid resolutions simultaneously (coarse cells in back, fine cells on top)
    heatmap_layers = []

    if not history_df.empty and 'p_step' in history_df.columns and 'd_step' in history_df.columns:
        # Group by resolution (p_step, d_step)
        resolution_groups = history_df.groupby(['p_step', 'd_step'])

        # Sort by cell area (coarsest first for z-order)
        sorted_resolutions = sorted(
            resolution_groups.groups.keys(),
            key=lambda res: res[0] * res[1],  # area = p_step * d_step
            reverse=True  # Largest cells first (back layer)
        )

        # Create heatmap layer for each resolution
        for idx, (p_step_val, d_step_val) in enumerate(sorted_resolutions):
            res_df = resolution_groups.get_group((p_step_val, d_step_val))

            # Add visual boundaries: finer resolutions get thinner white borders
            cell_area = p_step_val * d_step_val
            stroke_width = max(0.5, 2.0 / (idx + 1))  # Coarser = thicker stroke

            heatmap_layer = alt.Chart(res_df).mark_rect(
                opacity=0.9,  # Slight transparency to show multi-level structure
                stroke='white',  # White borders to distinguish regions
                strokeWidth=stroke_width
            ).encode(
                x=alt.X(
                    'x1:Q',
                    scale=alt.Scale(
                        domain=[bounds_dict['min_d'], bounds_dict['max_d']],
                        nice=False
                    ),
                    axis=duration_axis
                ),
                x2=alt.X2('x2:Q'),
                y=alt.Y(
                    'y1:Q',
                    scale=alt.Scale(
                        domain=[bounds_dict['min_p'], bounds_dict['max_p']],
                        nice=False
                    ),
                    axis=power_axis
                ),
                y2=alt.Y2('y2:Q'),
                color=alt.Color(
                    'revenue:Q',
                    scale=alt.Scale(scheme='viridis', domain=[min_rev, max_rev]),  # Global scale
                    legend=None
                ),
                tooltip=[
                    alt.Tooltip('power_mw', title='Power (MW)', format='.1f'),
                    alt.Tooltip('duration_h', title='Duration (h)', format='.2f'),
                    alt.Tooltip('revenue', title='Revenue ($)', format=',.0f'),
                    alt.Tooltip('p_step', title='Grid P-Step', format='.1f'),
                    alt.Tooltip('d_step', title='Grid D-Step', format='.2f'),
                    alt.Tooltip('is_skipped', title='Skipped')
                ]
            )
            heatmap_layers.append(heatmap_layer)

    # Fallback: single-layer rendering if no step info
    if not heatmap_layers:
        heatmap_layers = [alt.Chart(history_df).mark_rect().encode(
            x=alt.X(
                'x1:Q',
                scale=alt.Scale(
                    domain=[bounds_dict['min_d'], bounds_dict['max_d']],
                    nice=False
                ),
                axis=duration_axis
            ),
            x2=alt.X2('x2:Q'),
            y=alt.Y(
                'y1:Q',
                scale=alt.Scale(
                    domain=[bounds_dict['min_p'], bounds_dict['max_p']],
                    nice=False
                ),
                axis=power_axis
            ),
            y2=alt.Y2('y2:Q'),
            color=alt.Color(
                'revenue:Q',
                scale=alt.Scale(scheme='viridis', domain=[min_rev, max_rev]),
                legend=None
            ),
            tooltip=[
                alt.Tooltip('power_mw', title='Power (MW)', format='.1f'),
                alt.Tooltip('duration_h', title='Duration (h)', format='.2f'),
                alt.Tooltip('revenue', title='Revenue ($)', format=',.0f'),
                alt.Tooltip('is_skipped', title='Skipped')
            ]
        )]

    # Layer 2: Optimal Point (Diamond)
    if not history_df.empty:
        optimal_row = history_df.loc[history_df['revenue'].idxmax()]
        optimal_point = pd.DataFrame([optimal_row])
        optimal_marker = alt.Chart(optimal_point).mark_point(
            shape='diamond',
            size=200,
            fill='orange',
            stroke='white',
            strokeWidth=2
        ).encode(
            x=alt.X('duration_h:Q', scale=alt.Scale(domain=[bounds_dict['min_d'], bounds_dict['max_d']], nice=False)),
            y=alt.Y('power_mw:Q', scale=alt.Scale(domain=[bounds_dict['min_p'], bounds_dict['max_p']], nice=False)),
            tooltip=[
                alt.Tooltip('power_mw', title='Optimal Power (MW)'),
                alt.Tooltip('duration_h', title='Optimal Duration (h)'),
                alt.Tooltip('revenue', title='Max Revenue ($)', format=',.0f')
            ]
        )
    else:
        optimal_marker = alt.Chart(pd.DataFrame()).mark_point()

    # Layer 3: Skipped Markers
    if not history_df.empty:
        skipped_df = history_df[history_df['is_skipped'] == True].copy()
        if not skipped_df.empty:
            skipped_markers = alt.Chart(skipped_df).mark_point(
                shape='cross',
                size=80,
                fill='white',
                stroke='black',
                strokeWidth=1.5,
                opacity=0.7
            ).encode(
                x=alt.X('duration_h:Q', scale=alt.Scale(domain=[bounds_dict['min_d'], bounds_dict['max_d']], nice=False)),
                y=alt.Y('power_mw:Q', scale=alt.Scale(domain=[bounds_dict['min_p'], bounds_dict['max_p']], nice=False))
            )
        else:
            skipped_markers = alt.Chart(pd.DataFrame()).mark_point()
    else:
        skipped_markers = alt.Chart(pd.DataFrame()).mark_point()

    # Layer 4: Current Asset Marker (Red Star)
    current_p = bounds_dict.get('current_p')
    current_d = bounds_dict.get('current_d')

    current_marker = alt.Chart(pd.DataFrame()).mark_point()
    if current_p is not None and current_d is not None:
        # Create a single point dataframe
        curr_df = pd.DataFrame([{'power_mw': current_p, 'duration_h': current_d}])
        current_marker = alt.Chart(curr_df).mark_point(
            shape='star',
            size=300,
            fill='red',
            stroke='white',
            strokeWidth=2,
            opacity=1.0
        ).encode(
            x=alt.X('duration_h:Q', scale=alt.Scale(domain=[bounds_dict['min_d'], bounds_dict['max_d']], nice=False)),
            y=alt.Y('power_mw:Q', scale=alt.Scale(domain=[bounds_dict['min_p'], bounds_dict['max_p']], nice=False)),
            tooltip=[
                alt.Tooltip('power_mw:Q', title='Current Power (MW)', format='.1f'),
                alt.Tooltip('duration_h:Q', title='Current Duration (h)', format='.1f')
            ]
        )
    else:
        current_marker = alt.Chart(pd.DataFrame()).mark_point()

    # Layer 5: Alternative Configuration Markers (Min Power, Min Capacity)
    min_power_marker = alt.Chart(pd.DataFrame()).mark_point()
    min_capacity_marker = alt.Chart(pd.DataFrame()).mark_point()

    if alternative_configs:
        # Min Power marker (Cyan triangle pointing up)
        if 'min_power' in alternative_configs:
            min_p_p, min_p_d = alternative_configs['min_power']
            min_p_df = pd.DataFrame([{'power_mw': min_p_p, 'duration_h': min_p_d}])
            min_power_marker = alt.Chart(min_p_df).mark_point(
                shape='triangle-up',
                size=250,
                fill='#17a2b8',  # Cyan
                stroke='white',
                strokeWidth=2,
                opacity=1.0
            ).encode(
                x=alt.X('duration_h:Q', scale=alt.Scale(domain=[bounds_dict['min_d'], bounds_dict['max_d']], nice=False)),
                y=alt.Y('power_mw:Q', scale=alt.Scale(domain=[bounds_dict['min_p'], bounds_dict['max_p']], nice=False)),
                tooltip=[
                    alt.Tooltip('power_mw:Q', title='Min Power (MW)', format='.1f'),
                    alt.Tooltip('duration_h:Q', title='Duration (h)', format='.1f')
                ]
            )

        # Min Capacity marker (Yellow triangle pointing down)
        if 'min_capacity' in alternative_configs:
            min_c_p, min_c_d = alternative_configs['min_capacity']
            min_c_df = pd.DataFrame([{'power_mw': min_c_p, 'duration_h': min_c_d}])
            min_capacity_marker = alt.Chart(min_c_df).mark_point(
                shape='triangle-down',
                size=250,
                fill='#ffc107',  # Yellow
                stroke='white',
                strokeWidth=2,
                opacity=1.0
            ).encode(
                x=alt.X('duration_h:Q', scale=alt.Scale(domain=[bounds_dict['min_d'], bounds_dict['max_d']], nice=False)),
                y=alt.Y('power_mw:Q', scale=alt.Scale(domain=[bounds_dict['min_p'], bounds_dict['max_p']], nice=False)),
                tooltip=[
                    alt.Tooltip('power_mw:Q', title='Min Capacity (MW)', format='.1f'),
                    alt.Tooltip('duration_h:Q', title='Duration (h)', format='.1f')
                ]
            )

    # Combine layers using alt.layer() with explicit scale sharing
    all_layers = heatmap_layers + [skipped_markers, optimal_marker, current_marker, min_power_marker, min_capacity_marker]

    chart = alt.layer(*all_layers).resolve_scale(
        x='shared',
        y='shared'
    ).properties(
        width=600,
        height=500,
        title='Battery Sizing Optimization Landscape'
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        gridOpacity=0.3
    ).interactive()  # Enable pan and zoom

    return chart

# ============================================================================
# SESSION STATE INITIALIZATION FOR AMR OPTIMIZATION
# ============================================================================

if 'opt_history' not in st.session_state:
    st.session_state.opt_history = []

if 'opt_bounds' not in st.session_state:
    st.session_state.opt_bounds = {}

if 'opt_running' not in st.session_state:
    st.session_state.opt_running = False

if 'opt_queue' not in st.session_state:
    st.session_state.opt_queue = []

if 'opt_results_cache' not in st.session_state:
    st.session_state.opt_results_cache = {}

if 'opt_iteration' not in st.session_state:
    st.session_state.opt_iteration = 0

# --- FULL SIMULATION GRID OPTIMIZATION (BOTTOM) ---
st.markdown("---")
st.markdown("### 🔋 Battery Sizing Optimization")

# Create 2-column layout for Optimization Section
opt_col_left, opt_col_right = st.columns([1, 3])

with opt_col_left:
    st.markdown("#### ⚙️ Configuration")
    st.markdown("""
    **Clipping-Only Grid Search**: Find the optimal battery configuration by running full simulations.
    """)

    # Grid resolution slider
    grid_resolution = st.select_slider(
        "Grid Resolution",
        options=[
            "5×5 (25 configs, ~12s)",
            "7×7 (49 configs, ~25s)",
            "10×10 (100 configs, ~50s)",
            "12×12 (144 configs, ~1min)",
            "15×15 (225 configs, ~2min)",
            "20×20 (400 configs, ~3min)"
        ],
        value="5×5 (25 configs, ~12s)"  # Start coarse, let smart expansion refine
    )

    # Buttons stacked
    run_optimization = st.button("🚀 Run Optimization", type="primary", width='stretch')
    
    if st.button(
        "🗑️ Clear Cache",
        width='stretch',
        help="Clear cached simulation results to force fresh calculations"
    ):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("✅ Cache cleared! Next run will be fresh.")
        st.rerun()

    # Auto-calculated bounds (will be calculated after clipping analysis)
    # Placeholder - will be set after optimization button is clicked

    st.markdown("---")
    col_smart_1, col_smart_2 = st.columns([4, 1])
    with col_smart_1:
        enable_smart_expansion = st.checkbox("✨ Enable Smart Expansions", value=True, help="Automatically expand search space if optimal is found at the edge")
    with col_smart_2:
        max_expansions = st.number_input("Max Expansions", min_value=1, max_value=10, value=5,
                                        disabled=not enable_smart_expansion, label_visibility="collapsed")

# Logic to run optimization
if run_optimization or st.session_state.get('optimization_running', False):
    # ============================================================================
    # CLIPPING-BASED SEARCH SPACE CALCULATION
    # ============================================================================

    # Analyze clipping patterns
    peak_clipping_mw = df_sim['Clipped_MW'].quantile(0.99)  # 99th percentile (ignore extreme outliers)
    max_clipping_mw = df_sim['Clipped_MW'].max()
    total_clipped_mwh = df_sim['Clipped_MW'].sum() / 4  # Convert to MWh
    clipping_hours = (df_sim['Clipped_MW'] > 0.1).sum()  # Hours with meaningful clipping

    # Calculate daily clipping statistics
    daily_clipped = df_sim.groupby(df_sim.index.date)['Clipped_MW'].sum() / 4  # Daily MWh
    max_daily_clipped_mwh = daily_clipped.max()
    avg_daily_clipped = daily_clipped.mean()

    # Calculate clipping duration patterns (for capacity sizing)
    daily_clipping_hours = df_sim.groupby(df_sim.index.date).apply(
        lambda x: (x['Clipped_MW'] > 0.1).sum()
    )
    avg_clipping_duration = daily_clipping_hours.mean() / 4  # Convert intervals to hours

    # ============================================================================
    # AUTO-CALCULATED BOUNDS (For zero-curtailment coverage)
    # ============================================================================

    # Calculate auto bounds based on clipping profile
    if peak_clipping_mw < 0.5:  # Minimal clipping
        st.warning(f"""
⚠️ **Very Low Clipping Detected**: Peak clipping is only {peak_clipping_mw:.2f} MW.
Battery sizing optimization may not be meaningful with such low clipping.
Consider increasing Solar Capacity or reducing Interconnection Limit.
        """)
        auto_min_power = 1.0
        auto_max_power = 20.0
        auto_min_duration = 0.1
        auto_max_duration = 16.0
        search_rationale = "Minimal clipping (exploratory range)"
    else:
        # Power bounds: 95% to 300% of peak clipping
        # - Min power: Must handle peak clipping (with 5% safety margin)
        # - Max power: High-power/short-duration configs (up to 3x peak)
        auto_min_power = max(1.0, peak_clipping_mw * 0.95)
        auto_max_power = peak_clipping_mw * 3.0

        # Duration bounds: 0.1h to 16h
        # - Short duration: High-power/short-duration configs (0.1h)
        # - Long duration: Low-power/long-duration configs (16h)
        auto_min_duration = 0.1
        auto_max_duration = 16.0

        search_rationale = f"Zero-curtailment coverage: {auto_min_power:.0f}-{auto_max_power:.0f} MW × {auto_min_duration:.1f}-{auto_max_duration:.0f}h"

    # Use auto-calculated bounds
    min_power_mw = auto_min_power
    max_power_mw = auto_max_power
    heuristic_max_d = auto_max_duration

    # Display search rationale in left column
    with opt_col_left:
        # Show recommended bounds in an expander
        with st.expander("📊 Auto-Calculated Search Bounds", expanded=False):
            st.caption(f"""
**Clipping Profile Analysis:**
- Peak clipping: {peak_clipping_mw:.1f} MW (99th percentile)
- Max clipping: {max_clipping_mw:.1f} MW
- Total clipped energy: {total_clipped_mwh:,.0f} MWh

**Recommended Bounds (for zero-curtailment coverage):**
- Power range: {auto_min_power:.1f} - {auto_max_power:.1f} MW
- Duration range: {auto_min_duration:.1f} - {auto_max_duration:.1f}h

These bounds ensure the grid search covers configurations that can achieve
zero curtailment (<0.1%) using full MPC simulation.
            """)

        st.info(f"""
**Active Search Range:**
- **Power:** {min_power_mw:.0f} - {max_power_mw:.0f} MW
- **Duration:** {auto_min_duration:.1f} - {heuristic_max_d:.1f}h
- **Rationale:** {search_rationale}
        """)

    # Generate cache key for invalidation
    cache_inputs = f"{solar_capacity_mw}_{interconnection_limit_mw}_{batt_eff}_{df_prices.index.min()}_{df_prices.index.max()}_{min_power_mw}_{max_power_mw}"
    cache_key = hashlib.md5(cache_inputs.encode()).hexdigest()

    # ============================================================================
    # SMART ITERATIVE SEARCH LOOP
    # ============================================================================

    import time
    from concurrent.futures import as_completed

    # Parse grid resolution from slider
    resolution_map = {
        "5×5 (25 configs, ~12s)": (5, 5),
        "7×7 (49 configs, ~25s)": (7, 7),
        "10×10 (100 configs, ~50s)": (10, 10),
        "12×12 (144 configs, ~1min)": (12, 12),
        "15×15 (225 configs, ~2min)": (15, 15),
        "20×20 (400 configs, ~3min)": (20, 20)
    }
    base_n_power, base_n_duration = resolution_map[grid_resolution]

    # Initialize iterative variables
    current_min_p, current_max_p = min_power_mw, max_power_mw

    # Duration range initialization using auto-calculated bounds
    current_min_d = auto_min_duration  # Use auto-calculated min (0.1h)
    current_max_d = heuristic_max_d
    
    # Calculate fixed steps based on initial resolution
    raw_p_step = (current_max_p - current_min_p) / (base_n_power - 1)
    raw_d_step = (current_max_d - current_min_d) / (base_n_duration - 1)
    
    # Round steps
    p_step = max(1.0, round(raw_p_step, 0)) if raw_p_step > 1 else round(raw_p_step, 1)
    d_step = 0.25 if raw_d_step < 0.25 else round(raw_d_step * 4) / 4 
    
    # Align max to step
    current_max_p = current_min_p + p_step * (base_n_power - 1)
    current_max_d = current_min_d + d_step * (base_n_duration - 1)

    # Local cache for results
    results_cache = {}

    # ============================================================================
    # PRE-SEARCH PHASE: Coarse scan of full bounds
    # ============================================================================
    # This ensures we explore the zero-curtailment region even if it's far from revenue-optimal
    if enable_smart_expansion:
        with opt_col_right:
            presearch_placeholder = st.empty()
            with presearch_placeholder.container():
                st.info("🔍 Pre-search: Scanning full bounds at coarse resolution...")

        # Use coarse grid for pre-search (5×5)
        presearch_n_power = 5
        presearch_n_duration = 5

        presearch_power_range = np.linspace(min_power_mw, max_power_mw, presearch_n_power)
        presearch_duration_range = np.linspace(auto_min_duration, heuristic_max_d, presearch_n_duration)

        presearch_configs = [(p, d) for p in presearch_power_range for d in presearch_duration_range]

        # Calculate pre-search step sizes for AMR visualization
        presearch_p_step = (max_power_mw - min_power_mw) / (presearch_n_power - 1) if presearch_n_power > 1 else 1.0
        presearch_d_step = (heuristic_max_d - auto_min_duration) / (presearch_n_duration - 1) if presearch_n_duration > 1 else 0.25

        # Simulate pre-search configs (will be cached and reused in main search)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(simulate_battery_config, p, d): (p, d)
                       for p, d in presearch_configs}

            for future in as_completed(futures):
                p, d = futures[future]
                result = future.result()
                result['p_step'] = presearch_p_step
                result['d_step'] = presearch_d_step
                results_cache[(p, d)] = result

        presearch_placeholder.success(f"✅ Pre-search complete: {len(presearch_configs)} configs explored")
        time.sleep(1)  # Brief pause to show message
        presearch_placeholder.empty()

    saturated_configs = [] # List of (p, d, revenue) that achieved saturation
    max_revenue_seen = 0  # Track maximum revenue for plateau detection

    # max_expansions = 2 if enable_smart_expansion else 0 # This line is now replaced by the UI input
    expansion_count = 0

    start_time = time.time()

    # UI Placeholders
    with opt_col_right:
        grid_title_placeholder = st.empty()
        heatmap_placeholder = st.empty()
        progress_container = st.empty()  # Changed from st.container() to st.empty()

    # Initial UI Setup - use a container inside the empty placeholder
    with progress_container.container():
        st.markdown(f"""
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 10px;'>
            <h4 style='margin: 0; color: #0A5F7A;'>🔄 Optimization in Progress</h4>
            <p style='margin: 5px 0 0 0; color: #666;'>Step sizes: {p_step:.1f} MW, {d_step:.2f} h</p>
        </div>
        """, unsafe_allow_html=True)
        progress_bar = st.progress(0)
        status_col1, status_col2, status_col3 = st.columns(3)
        with status_col1: status_completed = st.empty()
        with status_col2: status_current = st.empty()
        with status_col3: status_time = st.empty()

    # Helper to update heatmap - AMR multi-level visualization
    def update_live_heatmap(curr_results_cache, curr_p_range, curr_d_range):
        """Build cumulative history with ALL resolutions for AMR visualization"""
        history_data = []

        # Iterate over FULL cache (all resolutions), not just current grid
        for (p_val, d_val), res in curr_results_cache.items():
            # Get stored step sizes from result (fallback to current)
            p_step_val = res.get('p_step', p_step)
            d_step_val = res.get('d_step', d_step)

            history_data.append({
                'x1': d_val - d_step_val/2,
                'x2': d_val + d_step_val/2,
                'y1': p_val - p_step_val/2,
                'y2': p_val + p_step_val/2,
                'power_mw': p_val,
                'duration_h': d_val,
                'revenue': res['total_revenue'],
                'is_skipped': res.get('is_skipped', False),
                'p_step': p_step_val,  # Pass to chart for multi-level rendering
                'd_step': d_step_val
            })

        if history_data:
            history_df = pd.DataFrame(history_data)

            # Bounds span ENTIRE search space (all iterations), not just current grid
            all_powers = [row['power_mw'] for row in history_data]
            all_durations = [row['duration_h'] for row in history_data]
            all_p_steps = [row['p_step'] for row in history_data]
            all_d_steps = [row['d_step'] for row in history_data]

            max_p_step = max(all_p_steps) if all_p_steps else p_step
            max_d_step = max(all_d_steps) if all_d_steps else d_step

            bounds_dict = {
                'min_d': min(all_durations) - max_d_step/2,
                'max_d': max(all_durations) + max_d_step/2,
                'min_p': min(all_powers) - max_p_step/2,
                'max_p': max(all_powers) + max_p_step/2,
                'current_p': current_power,
                'current_d': current_energy / current_power if current_power > 0 else 0
            }

            # Tick positions for current grid only (finest resolution)
            p_list = list(curr_p_range)
            d_list = list(curr_d_range)
            tick_positions = {
                'duration_values': d_list,
                'power_values': p_list
            }

            chart = create_optimization_chart(history_df, bounds_dict, tick_positions)
            heatmap_placeholder.altair_chart(chart, width='stretch')

    # Helper functions for saturation detection
    def _is_saturated_horizontally(p, d):
        """Check if increasing duration at power p doesn't improve revenue (plateau)."""
        for sp, sd, srev in saturated_configs:
            if abs(p - sp) < 0.1 and sd >= d * 0.95:
                return True
        return False

    def _is_saturated_vertically(p, d):
        """Check if increasing power at duration d doesn't improve revenue."""
        for sp, sd, srev in saturated_configs:
            if abs(d - sd) < 0.1 and sp >= p * 0.95:
                return True
        return False

    # Continuous Loop
    while True:
        # 1. Generate Grid for current bounds
        power_range = np.arange(current_min_p, current_max_p + p_step/100, p_step)
        duration_range = np.arange(current_min_d, current_max_d + d_step/100, d_step)
        
        power_range = np.round(power_range, 1)
        duration_range = np.round(duration_range, 2)
        
        total_configs = len(power_range) * len(duration_range)
        grid_title_placeholder.markdown(f"### 🗺️ Simulation Grid: {len(power_range)}×{len(duration_range)} = {total_configs} configurations")

        # 2. Identify NEW configs
        new_configs = []
        for p in power_range:
            for d in duration_range:
                if (p, d) not in results_cache:
                    new_configs.append((p, d))
        
        # Sort by capacity (Smallest first) -> Critical for skipping logic
        new_configs.sort(key=lambda x: x[0] * x[1])

        # Count only configs in CURRENT grid (not cumulative cache)
        current_grid_configs = {(p, d) for p in power_range for d in duration_range}
        total_processed = len(current_grid_configs & set(results_cache.keys()))
        
        # 3. Process individually with real-time updates
        batch_size = 10 
        skipped_in_this_grid = 0  # Track skips for this grid expansion
        
        for i in range(0, len(new_configs), batch_size):
            batch = new_configs[i:i+batch_size]
            batch_to_run = []
            
            # Check skipping for this batch
            for p, d in batch:
                is_dominated = False
                dominated_revenue = 0

                # === REVENUE-BASED PRUNING ===
                # Skip fine-grid points in regions where coarse-grid shows poor revenue
                if results_cache:  # Only if we have prior results
                    current_best_revenue = max(res['total_revenue'] for res in results_cache.values())
                    revenue_threshold = current_best_revenue * 0.70  # 70% of optimal

                    # Check if nearby coarse-grid points suggest this region is unpromising
                    nearby_revenues = []
                    search_radius_p = p_step * 2  # Look within 2 steps
                    search_radius_d = d_step * 2

                    for (cp, cd), cres in results_cache.items():
                        # Only consider coarser or same resolution (don't use finer grid results)
                        if cres.get('p_step', float('inf')) >= p_step and cres.get('d_step', float('inf')) >= d_step:
                            if abs(cp - p) <= search_radius_p and abs(cd - d) <= search_radius_d:
                                nearby_revenues.append(cres['total_revenue'])

                    # If all nearby coarse points have poor revenue, skip this fine point
                    if nearby_revenues and max(nearby_revenues) < revenue_threshold:
                        is_dominated = True
                        dominated_revenue = max(nearby_revenues)  # Use best nearby value

                # Check against ALL known saturated configs (including from previous batches)
                if not is_dominated:
                    for sp, sd, srev in saturated_configs:
                        # HORIZONTAL SKIPPING ONLY
                        # Only skip if we are at the SAME power level (approx) but LARGER duration
                        # Increasing power can still increase revenue (faster discharge) even if energy is saturated
                        is_same_power = abs(p - sp) < 0.1
                        is_larger_duration = d >= sd * 1.01 # 1% margin

                        if is_same_power and is_larger_duration:
                            is_dominated = True
                            dominated_revenue = srev
                            break
                
                if is_dominated:
                    results_cache[(p, d)] = {
                        'power_mw': p, 'duration_h': d,
                        'total_revenue': dominated_revenue,
                        'curtailed_mwh': 0.0, 'is_skipped': True,
                        'p_step': p_step, 'd_step': d_step  # Store grid resolution
                    }
                    # Update progress for skipped items too
                    total_processed += 1
                    skipped_in_this_grid += 1
                else:
                    batch_to_run.append((p, d))
            
            # Run simulations for batch
            if batch_to_run:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {executor.submit(simulate_battery_config, p, d): (p, d) for p, d in batch_to_run}
                    
                    for future in as_completed(futures):
                        p, d = futures[future]
                        result = future.result()
                        result['p_step'] = p_step  # Store grid resolution for AMR visualization
                        result['d_step'] = d_step
                        results_cache[(p, d)] = result
                        
                        # Dynamic Saturation Update
                        curtailed = result.get('curtailed_clipping', float('inf'))
                        revenue = result['total_revenue']
                        
                        # Check 1: Absolute Low Curtailment
                        is_low_curtailment = curtailed < 0.1
                        
                        # Check 2: Horizontal Revenue Plateau
                        # Check if adding duration at this power level yielded negligible revenue gain
                        is_revenue_plateau = False
                        prev_d = round(d - d_step, 2)
                        if (p, prev_d) in results_cache:
                            prev_res = results_cache[(p, prev_d)]
                            prev_rev = prev_res['total_revenue']
                            # If revenue increased by less than 0.05% despite adding duration
                            if prev_rev > 0 and (revenue - prev_rev) / prev_rev < 0.0005:
                                is_revenue_plateau = True
                        
                        if is_low_curtailment or is_revenue_plateau:
                            saturated_configs.append((p, d, revenue))
                            # Notify user when we first detect saturation
                            if len(saturated_configs) == 1:
                                st.toast(f"🎯 Saturation detected at {p:.0f}MW × {d:.1f}h - larger durations will be skipped!", icon="🚀")
                        
                        # Update Progress after EACH simulation
                        total_processed += 1
                        progress_pct = min(1.0, total_processed / total_configs)
                        progress_bar.progress(progress_pct)
                        status_completed.metric("Completed", f"{total_processed}/{total_configs}")
                        status_current.metric("Current", f"{p:.0f}MW × {d:.1f}h")
                        
                        # Update ETA
                        elapsed = time.time() - start_time
                        if total_processed > 0:
                            rate = elapsed / total_processed
                            remaining = total_configs - total_processed
                            status_time.metric("ETA", f"{rate * remaining:.0f}s")
                        
                        # Live Heatmap Update after EACH simulation
                        update_live_heatmap(results_cache, power_range, duration_range)
        
        # Show summary of skips for this grid
        if skipped_in_this_grid > 0:
            st.toast(f"⏩ Skipped {skipped_in_this_grid} saturated configs in this expansion", icon="✨")

        # Force update at end of grid
        update_live_heatmap(results_cache, power_range, duration_range)

        # 4. Per-Dimension Expansion/Zoom Decision Logic
        if total_processed == total_configs and expansion_count < max_expansions and enable_smart_expansion:

            # Find current optimal
            best_config = max(results_cache.items(), key=lambda x: x[1]['total_revenue'])
            opt_p, opt_d = best_config[0]

            # Initialize actions for each dimension
            power_action = None  # 'expand_up', 'expand_down', 'zoom', or None
            duration_action = None

            # === POWER DIMENSION DECISION ===
            at_max_p = opt_p >= current_max_p - p_step * 1.01
            at_min_p = (opt_p <= current_min_p + p_step * 1.01) and (current_min_p > 1.0)

            if at_max_p:
                power_action = 'expand_up'
            elif at_min_p:
                power_action = 'expand_down'
            elif p_step > 0.1 and _is_saturated_vertically(opt_p, opt_d):
                power_action = 'zoom'  # Saturation detected, refine power resolution

            # === DURATION DIMENSION DECISION ===
            at_max_d = opt_d >= current_max_d - d_step * 1.01
            at_min_d = (opt_d <= current_min_d + d_step * 1.01) and (current_min_d > 0.5)

            if at_max_d:
                duration_action = 'expand_up'
            elif at_min_d:
                duration_action = 'expand_down'
            elif d_step > 0.1 and _is_saturated_horizontally(opt_p, opt_d):
                duration_action = 'zoom'  # Saturation detected, refine duration resolution

            # === EXECUTE ACTIONS ===
            actions_taken = []

            # Power dimension actions
            if power_action == 'expand_up':
                current_max_p += p_step * 5
                current_max_p = round(current_max_p / p_step) * p_step
                actions_taken.append("Expand Power ↑")
            elif power_action == 'expand_down':
                current_min_p = max(1.0, current_min_p - p_step * 5)
                current_min_p = round(current_min_p / p_step) * p_step
                actions_taken.append("Expand Power ↓")
            elif power_action == 'zoom':
                # Local refinement: only refine small region around optimal (3×3 cells at finer resolution)
                p_step = max(0.05, round(p_step / 2, 2))  # Halve step size first
                new_span_p = p_step * 2  # Only 3×3 local refinement (smaller than before)
                current_min_p = max(1.0, opt_p - new_span_p)
                current_max_p = min(max_power_mw, opt_p + new_span_p)  # Don't exceed original bounds
                actions_taken.append("Zoom Power 🔍")

            # Duration dimension actions
            if duration_action == 'expand_up':
                current_max_d += d_step * 5
                current_max_d = round(current_max_d / d_step) * d_step
                actions_taken.append("Expand Duration ↑")
            elif duration_action == 'expand_down':
                current_min_d = max(0.5, current_min_d - d_step * 5)
                current_min_d = round(current_min_d / d_step) * d_step
                actions_taken.append("Expand Duration ↓")
            elif duration_action == 'zoom':
                # Local refinement: only refine small region around optimal (3×3 cells at finer resolution)
                d_step = max(0.05, round(d_step / 2, 2))  # Halve step size first
                new_span_d = d_step * 2  # Only 3×3 local refinement (smaller than before)
                current_min_d = max(0.5, opt_d - new_span_d)
                current_max_d = min(heuristic_max_d, opt_d + new_span_d)  # Cap at auto-calculated max
                actions_taken.append("Zoom Duration 🔍")

            # === TERMINATION CHECK ===
            if power_action is None and duration_action is None:
                # No actions needed - we've converged
                if expansion_count >= max_expansions or (p_step <= 0.05 and d_step <= 0.05):
                    st.toast("✅ Optimization Complete (Converged)", icon="🏁")
                    break
            else:
                # Actions were taken - increment counter and notify
                expansion_count += 1
                action_str = " + ".join(actions_taken)
                st.toast(f"🚀 {action_str} (Round {expansion_count}/{max_expansions})", icon="🔭")

                # Reset saturation if we zoomed (finer resolution changes saturation criteria)
                if 'zoom' in power_action.lower() if power_action else False or \
                   'zoom' in duration_action.lower() if duration_action else False:
                    saturated_configs = []
                    max_revenue_seen = 0
        elif total_processed == total_configs:
             st.toast("✅ Optimization Complete", icon="🏁")
             break

    # --- Final Cleanup ---
    elapsed_time = time.time() - start_time
    progress_container.empty()

    # Show completion message in the grid title placeholder instead
    grid_title_placeholder.success(f"✅ Optimization complete! Found optimal in {elapsed_time:.1f}s.")
    
    # Re-calculate final optimal details for display
    # Find best config from final cache
    best_config_item = max(results_cache.items(), key=lambda x: x[1]['total_revenue'])
    optimal_power, optimal_duration = best_config_item[0]
    optimal_result = best_config_item[1]
    optimal_revenue = optimal_result['total_revenue']
    
    optimal_capacity = optimal_power * optimal_duration
    # optimal_result is already retrieved above
    
    # Check final edge status for warning
    is_power_edge = (optimal_power == current_min_p) or (optimal_power == current_max_p)
    is_duration_edge = (optimal_duration == current_min_d) or (optimal_duration == current_max_d)
    
    if is_power_edge or is_duration_edge:
        edge_dims = []
        if is_power_edge: edge_dims.append(f"power ({optimal_power:.0f} MW)")
        if is_duration_edge: edge_dims.append(f"duration ({optimal_duration:.1f}h)")
        st.warning(f"⚠️ **Search Space Limit Reached**: Optimal is at edge of {' and '.join(edge_dims)} range even after expansion.")

    # ============================================================================
    # EXTRACT ZERO-CURTAILMENT FRONTIER FROM GRID SEARCH RESULTS (COMPUTATION ONLY)
    # ============================================================================
    # Perform calculations here so we can pass markers to heatmap
    # UI display happens later in the curtailment elimination section

    # Calculate total clipped energy for curtailment percentage
    # NOTE: This is RAW clipped energy (no efficiency applied yet)
    # Battery efficiency is accounted for in the simulator when calculating captured/curtailed amounts
    total_clipped_mwh = df_sim['Clipped_MW'].sum() / 4

    # Build frontier from grid search results (already have full MPC simulations!)
    frontier_data = []
    for (p, d), res in results_cache.items():
        # Skip configs that were pruned/skipped
        if res.get('is_skipped', False):
            continue

        curtailed_mwh = res.get('curtailed_clipping', 0)
        curtailment_pct = (curtailed_mwh / max(total_clipped_mwh, 1)) * 100

        frontier_data.append({
            'power_mw': p,
            'duration_h': d,
            'capacity_mwh': res['capacity_mwh'],
            'curtailment_mwh': curtailed_mwh,
            'curtailment_pct': curtailment_pct,
            'total_revenue': res['total_revenue'],
            'battery_revenue': res['battery_revenue']
        })

    frontier_df = pd.DataFrame(frontier_data)

    # Filter to zero-curtailment configs (<0.1%)
    valid_configs = frontier_df[frontier_df['curtailment_pct'] < 0.1].copy()

    if not valid_configs.empty:
        # FIND TRUE FRONTIER: For each power level, find the MINIMUM duration that works
        frontier_curve = valid_configs.loc[valid_configs.groupby('power_mw')['duration_h'].idxmin()].copy()
        frontier_curve = frontier_curve.sort_values('power_mw')

        # 1. Minimum Power Configuration
        min_power_row = frontier_curve.iloc[0]
        min_power_p = min_power_row['power_mw']
        min_power_d = min_power_row['duration_h']
        min_power_c = min_power_p * min_power_d

        # 2. Minimum Capacity Configuration
        min_capacity_row = frontier_curve.loc[frontier_curve['capacity_mwh'].idxmin()]
        min_capacity_p = min_capacity_row['power_mw']
        min_capacity_d = min_capacity_row['duration_h']
        min_capacity_c = min_capacity_row['capacity_mwh']

        # Extract from cache (already simulated)
        min_power_res = results_cache.get((min_power_p, min_power_d), {})
        min_capacity_res = results_cache.get((min_capacity_p, min_capacity_d), {})

        min_power_revenue = min_power_res.get('total_revenue', 0)
        min_power_batt_rev = min_power_res.get('battery_revenue', 0)
        min_capacity_revenue = min_capacity_res.get('total_revenue', 0)
        min_capacity_batt_rev = min_capacity_res.get('battery_revenue', 0)

    else:
        # No configs achieved zero curtailment - use best available
        best_row = frontier_df.loc[frontier_df['curtailment_pct'].idxmin()]

        # Use best as both min power and min capacity
        min_power_p = min_capacity_p = best_row['power_mw']
        min_power_d = min_capacity_d = best_row['duration_h']
        min_power_c = min_capacity_c = best_row['capacity_mwh']

        # Extract from cache
        best_res = results_cache.get((min_power_p, min_power_d), {})
        min_power_revenue = min_capacity_revenue = best_res.get('total_revenue', 0)
        min_power_batt_rev = min_capacity_batt_rev = best_res.get('battery_revenue', 0)

        # Create empty frontier curve for plotting
        frontier_curve = pd.DataFrame()

    # Extract curtailment data for all four configurations
    current_curtailed = current_res.get('curtailed_clipping', 0)
    current_curtailment_pct = (current_curtailed / max(total_clipped_mwh, 1)) * 100

    min_power_curtailed = min_power_res.get('curtailed_clipping', 0)
    min_power_curtailment_pct = (min_power_curtailed / max(total_clipped_mwh, 1)) * 100

    min_capacity_curtailed = min_capacity_res.get('curtailed_clipping', 0)
    min_capacity_curtailment_pct = (min_capacity_curtailed / max(total_clipped_mwh, 1)) * 100

    optimal_curtailed = optimal_result.get('curtailed_clipping', 0)
    optimal_curtailment_pct = (optimal_curtailed / max(total_clipped_mwh, 1)) * 100

    # ============================================================================
    # OPTIMAL RESULTS (RIGHT COLUMN)
    # ============================================================================

    with opt_col_right:
        # Clear the live heatmap placeholder so we can render the final polished one (with stars etc)
        if 'heatmap_placeholder' in locals():
            heatmap_placeholder.empty()
            
        # --- 1. Optimal Configuration Metrics ---
        st.markdown("### 🌟 Optimal Battery Configuration")

        # Get current asset values for delta comparison
        curr_p = current_res['power_mw']
        curr_d = current_res['duration_h']
        curr_c = current_res['capacity_mwh']
        curr_b_rev = current_res['battery_revenue']

        # Metrics with Deltas
        col1, col2, col3, col4, col5 = st.columns(5)
        
        col1.metric("Optimal Power", f"{optimal_power:.0f} MW", 
                    delta=f"{optimal_power - curr_p:+.0f} MW")
        
        col2.metric("Optimal Duration", f"{optimal_duration:.1f}h",
                    delta=f"{optimal_duration - curr_d:+.1f}h")
        
        col3.metric("Optimal Capacity", f"{optimal_capacity:.1f} MWh",
                    delta=f"{optimal_capacity - curr_c:+.1f} MWh")
        
        # Optimal Battery Revenue
        opt_batt_rev = optimal_result['battery_revenue']
        col4.metric("Battery Revenue", f"${opt_batt_rev:,.0f}",
                    delta=f"${opt_batt_rev - curr_b_rev:,.0f}")

        # Compare with current asset (Total Revenue)
        revenue_improvement = optimal_revenue - current_hybrid_rev
        col5.metric("Total Revenue", f"${optimal_revenue:,.0f}",
                    delta=f"+${revenue_improvement:,.0f}")

        st.markdown("---")

        # --- 2. Heatmap Visualization (Altair AMR) ---
        st.markdown("### 📊 Revenue Heatmap")

        # Build AMR multi-level data structure from ALL cached results
        final_history_data = []

        # Iterate over FULL cache (all resolutions) for AMR visualization
        for (p_val, d_val), res in results_cache.items():
            # Get stored step sizes from result
            p_step_val = res.get('p_step', p_step)
            d_step_val = res.get('d_step', d_step)

            final_history_data.append({
                'x1': d_val - d_step_val/2,
                'x2': d_val + d_step_val/2,
                'y1': p_val - p_step_val/2,
                'y2': p_val + p_step_val/2,
                'power_mw': p_val,
                'duration_h': d_val,
                'revenue': res['total_revenue'],
                'is_skipped': res.get('is_skipped', False),
                'p_step': p_step_val,  # Include for multi-level rendering
                'd_step': d_step_val
            })

        if final_history_data:
            final_history_df = pd.DataFrame(final_history_data)

            # Bounds span ENTIRE search space (all iterations)
            all_powers = final_history_df['power_mw'].tolist()
            all_durations = final_history_df['duration_h'].tolist()
            all_p_steps = final_history_df['p_step'].tolist()
            all_d_steps = final_history_df['d_step'].tolist()

            max_p_step = max(all_p_steps)
            max_d_step = max(all_d_steps)

            final_bounds_dict = {
                'min_d': min(all_durations) - max_d_step/2,
                'max_d': max(all_durations) + max_d_step/2,
                'min_p': min(all_powers) - max_p_step/2,
                'max_p': max(all_powers) + max_p_step/2,
                'current_p': curr_p,
                'current_d': curr_d
            }

            # Tick positions for finest resolution (final grid)
            p_list = list(power_range)
            d_list = list(duration_range)
            final_tick_positions = {
                'duration_values': d_list,
                'power_values': p_list
            }

            # Pass alternative configurations for markers on heatmap
            alternative_configs = {
                'min_power': (min_power_p, min_power_d),
                'min_capacity': (min_capacity_p, min_capacity_d)
            }

            final_chart = create_optimization_chart(
                final_history_df,
                final_bounds_dict,
                final_tick_positions,
                alternative_configs
            )
            final_chart = final_chart.properties(
                title=f"Full Simulation Grid Search: {len(power_range)}×{len(duration_range)} = {total_configs} Configurations"
            )
            st.altair_chart(final_chart, width='stretch')

        st.caption(f"""
    💡 Heatmap shows true simulated revenue across {total_configs} configurations
    ({len(power_range)} power levels × {len(duration_range)} durations) using full BatterySimulator with
    ClippingOnlyStrategy. Runtime: {elapsed_time:.1f} seconds.
        """)

    # ============================================================================
    # DETAILED BREAKDOWN
    # ============================================================================

    if revenue_improvement > 0:
        st.info(f"💡 **Recommended**: Upgrade to **{optimal_power:.0f} MW / {optimal_capacity:.1f} MWh** battery to capture **${revenue_improvement:,.0f}** additional revenue.")
    else:
        st.success("🎉 Current asset configuration is already optimal!")
    
    # ============================================================================
    # CURTAILMENT ELIMINATION ANALYSIS
    # ============================================================================
    
    st.markdown("---")
    st.markdown("### 🎯 Curtailment Elimination: Minimum Battery Configurations")

    # Note: Frontier extraction calculations were performed earlier (before heatmap creation)
    # Results are available in: min_power_p, min_power_d, min_capacity_p, min_capacity_d, etc.

    # Display warning if no zero-curtailment configs were found
    if frontier_curve.empty:
        best_curtailment = min_power_curtailment_pct  # Already calculated earlier
        st.warning(f"""
⚠️ **No configuration achieved target curtailment (<0.1%)**

Best result: {best_curtailment:.2f}% curtailment at {min_power_p:.1f} MW × {min_power_d:.1f}h

**Recommendations**:
1. Increase Max Power bound (extend search to higher power ratings)
2. Increase Max Duration bound (allow longer-duration batteries)
3. Check if forecast improvement factor is realistic
4. Verify clipping profile data quality
5. Consider that zero curtailment may not be economically optimal
        """)

    # Create comparison DataFrame with 4 configurations
    comparison_configs = pd.DataFrame([
        {
            'Configuration': 'Current Asset',
            'Power (MW)': curr_p,
            'Duration (h)': curr_d,
            'Capacity (MWh)': curr_c,
            'Total Revenue': current_hybrid_rev,
            'Battery Revenue': curr_b_rev,
            'Type': 'current',
            'Order': 1
        },
        {
            'Configuration': 'Min Power (Lowest MW)',
            'Power (MW)': min_power_p,
            'Duration (h)': min_power_d,
            'Capacity (MWh)': min_power_c,
            'Total Revenue': min_power_revenue,
            'Battery Revenue': min_power_batt_rev,
            'Type': 'min_power',
            'Order': 2
        },
        {
            'Configuration': 'Min Capacity (Lowest MWh)',
            'Power (MW)': min_capacity_p,
            'Duration (h)': min_capacity_d,
            'Capacity (MWh)': min_capacity_c,
            'Total Revenue': min_capacity_revenue,
            'Battery Revenue': min_capacity_batt_rev,
            'Type': 'min_capacity',
            'Order': 3
        },
        {
            'Configuration': 'Revenue-Optimal',
            'Power (MW)': optimal_power,
            'Duration (h)': optimal_duration,
            'Capacity (MWh)': optimal_capacity,
            'Total Revenue': optimal_revenue,
            'Battery Revenue': opt_batt_rev,
            'Type': 'optimal',
            'Order': 4
        }
    ])
    
    # Create scatter plot
    fig_comparison = go.Figure()
    
    # Define colors and symbols
    config_styles = {
        'current': {'color': '#dc3545', 'symbol': 'circle', 'name': 'Current Asset', 'size': 15},
        'min_power': {'color': '#17a2b8', 'symbol': 'triangle-up', 'name': 'Min Power (Lowest MW)', 'size': 18},
        'min_capacity': {'color': '#ffc107', 'symbol': 'triangle-down', 'name': 'Min Capacity (Lowest MWh)', 'size': 18},
        'optimal': {'color': '#28a745', 'symbol': 'star', 'name': 'Revenue-Optimal', 'size': 20}
    }
    
    # Plot the iso-curtailment frontier line
    if not frontier_curve.empty:
        fig_comparison.add_trace(go.Scatter(
            x=frontier_curve['duration_h'],
            y=frontier_curve['power_mw'],
            mode='lines',
            line=dict(color='rgba(23, 162, 184, 0.4)', width=3),
            name='Zero-Curtailment Frontier',
            customdata=frontier_curve[['capacity_mwh', 'total_revenue', 'battery_revenue']],
            hovertemplate=(
                '<b>Zero-Curtailment Frontier</b><br>'
                'Power: %{y:.1f} MW<br>'
                'Duration: %{x:.1f}h<br>'
                'Capacity: %{customdata[0]:.1f} MWh<br>'
                'Total Revenue: $%{customdata[1]:,.0f}<br>'
                'Battery Revenue: $%{customdata[2]:,.0f}'
                '<extra></extra>'
            ),
            showlegend=True
        ))
    
    # Add individual configuration points
    for _, row in comparison_configs.iterrows():
        style = config_styles[row['Type']]
        fig_comparison.add_trace(go.Scatter(
            x=[row['Duration (h)']],
            y=[row['Power (MW)']],
            mode='markers',
            marker=dict(
                size=style['size'] + (row['Capacity (MWh)'] / 20),  # Size by capacity
                color=style['color'],
                symbol=style['symbol'],
                line=dict(width=2, color='white'),
                opacity=0.95
            ),
            name=style['name'],
            hovertext=[f"{row['Configuration']}<br>" +
                       f"Power: {row['Power (MW)']:.1f} MW<br>" +
                       f"Duration: {row['Duration (h)']:.1f}h<br>" +
                       f"Capacity: {row['Capacity (MWh)']:.1f} MWh<br>" +
                       f"Total Revenue: ${row['Total Revenue']:,.0f}<br>" +
                       f"Battery Revenue: ${row['Battery Revenue']:,.0f}"],
            hovertemplate='%{hovertext}<extra></extra>'
        ))
    
    fig_comparison.update_layout(
        title="Battery Sizing: Current vs. Minimum Zero-Curtailment vs. Revenue-Optimal",
        xaxis_title="Battery Duration (hours)",
        yaxis_title="Battery Power (MW)",
        height=550,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(255, 255, 255, 0.7)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1
        )
    )
    
    st.plotly_chart(fig_comparison, width='stretch')
    
    st.caption("""
    **💡 Zero-Curtailment Frontier**: The cyan line shows the **minimum** battery configurations needed to achieve ~0% curtailment.
    - **Min Power**: The smallest inverter size (MW) that can handle the peak clipping.
    - **Min Capacity**: The smallest total battery size (MWh) that can store the energy (often requires higher power to discharge faster).
    """)
    
    # Comparison metrics table (4 columns)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("#### 🔴 Current")
        st.metric("Power", f"{curr_p:.0f} MW")
        st.metric("Duration", f"{curr_d:.1f}h")
        st.metric("Capacity", f"{curr_c:.1f} MWh")
        st.metric("Total Revenue", f"${current_hybrid_rev:,.0f}")
        st.metric("Curtailment", f"{current_curtailment_pct:.2f}%",
                  help=f"{current_curtailed:.1f} MWh curtailed")
        st.caption("Baseline asset")
    
    with col2:
        st.markdown("#### 🔷 Min Power")
        st.metric("Power", f"{min_power_p:.0f} MW", delta=f"{min_power_p - curr_p:+.0f} MW")
        st.metric("Duration", f"{min_power_d:.1f}h", delta=f"{min_power_d - curr_d:+.1f}h")
        st.metric("Capacity", f"{min_power_c:.1f} MWh", delta=f"{min_power_c - curr_c:+.1f} MWh")
        st.metric("Total Revenue", f"${min_power_revenue:,.0f}", delta=f"{min_power_revenue - current_hybrid_rev:+,.0f}")
        st.metric("Curtailment", f"{min_power_curtailment_pct:.2f}%",
                  delta=f"{min_power_curtailment_pct - current_curtailment_pct:+.2f}%",
                  delta_color="inverse",
                  help=f"{min_power_curtailed:.1f} MWh curtailed")
        st.caption("Lowest viable MW")
    
    with col3:
        st.markdown("#### 🟨 Min Capacity")
        st.metric("Power", f"{min_capacity_p:.0f} MW", delta=f"{min_capacity_p - curr_p:+.0f} MW")
        st.metric("Duration", f"{min_capacity_d:.1f}h", delta=f"{min_capacity_d - curr_d:+.1f}h")
        st.metric("Capacity", f"{min_capacity_c:.1f} MWh", delta=f"{min_capacity_c - curr_c:+.1f} MWh")
        st.metric("Total Revenue", f"${min_capacity_revenue:,.0f}", delta=f"{min_capacity_revenue - current_hybrid_rev:+,.0f}")
        st.metric("Curtailment", f"{min_capacity_curtailment_pct:.2f}%",
                  delta=f"{min_capacity_curtailment_pct - current_curtailment_pct:+.2f}%",
                  delta_color="inverse",
                  help=f"{min_capacity_curtailed:.1f} MWh curtailed")
        st.caption("Lowest viable MWh")
    
    with col4:
        st.markdown("#### 🟢 Optimal")
        st.metric("Power", f"{optimal_power:.0f} MW", delta=f"{optimal_power - curr_p:+.0f} MW")
        st.metric("Duration", f"{optimal_duration:.1f}h", delta=f"{optimal_duration - curr_d:+.1f}h")
        st.metric("Capacity", f"{optimal_capacity:.1f} MWh", delta=f"{optimal_capacity - curr_c:+.1f} MWh")
        st.metric("Total Revenue", f"${optimal_revenue:,.0f}", delta=f"{revenue_improvement:+,.0f}")
        st.metric("Curtailment", f"{optimal_curtailment_pct:.2f}%",
                  delta=f"{optimal_curtailment_pct - current_curtailment_pct:+.2f}%",
                  delta_color="inverse",
                  help=f"{optimal_curtailed:.1f} MWh curtailed")
        st.caption("Max revenue")


    with st.expander("📊 Detailed Breakdown (Optimal Configuration)"):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Clipped Energy Captured", f"{optimal_result['clipped_captured']:.1f} MWh")
        col2.metric("Curtailed Clipping", f"{optimal_result.get('curtailed_clipping', 0):.1f} MWh")
        col3.metric("Capture Efficiency",
                    f"{100 * optimal_result['clipped_captured'] / max(total_clipped_mwh, 1):.1f}%")
        col4.metric("Battery Revenue", f"${optimal_result['battery_revenue']:,.0f}")

        # ALWAYS show curtailment analysis with optimization suggestions
        curtailed = optimal_result.get('curtailed_clipping', 0)
        curtailment_revenue_lost = curtailed * (optimal_result['battery_revenue'] / max(optimal_result['clipped_captured'], 1))

        if curtailed > 0.01:  # Any curtailment
            # Calculate what duration would capture more
            additional_duration_needed = curtailed / max(optimal_power, 1)
            suggested_duration = optimal_duration + additional_duration_needed
            suggested_capacity = optimal_power * suggested_duration

            severity = "🔴 High" if curtailed > optimal_result['clipped_captured'] * 0.2 else \
                       "🟡 Moderate" if curtailed > optimal_result['clipped_captured'] * 0.1 else \
                       "🟢 Low"

            st.info(f"""
📉 **Curtailment Analysis** ({severity} Impact)

**Lost Opportunity:**
- Curtailed clipping: {curtailed:.1f} MWh ({100 * curtailed / total_clipped_mwh:.1f}% of total clipping)
- Estimated lost revenue: ${curtailment_revenue_lost:,.0f}

**Optimization Suggestions:**
1. **Increase battery duration** from {optimal_duration:.1f}h to {suggested_duration:.1f}h
- New capacity: {suggested_capacity:.1f} MWh (current: {optimal_capacity:.1f} MWh)
- Would capture additional {curtailed:.1f} MWh
- Potential revenue gain: ${curtailment_revenue_lost:,.0f}

2. **Optimize discharge strategy** to free up capacity faster:
- Current strategy: {state.strategy_type}
- Try increasing forecast improvement from {state.forecast_improvement}% to {min(100, state.forecast_improvement + 20)}%
- Or reduce MPC/Rolling Window horizon for more aggressive discharge

3. **Consider power/duration trade-offs:**
- Current: {optimal_power:.0f} MW × {optimal_duration:.1f}h
- Alternative: {optimal_power * 0.8:.0f} MW × {suggested_duration * 0.8:.1f}h (faster cycling)
            """)
        else:
            st.success("""
✅ **No Curtailment**: Battery captures all available clipping!
Current configuration is optimal for clipping capture.
            """)

        st.markdown(f"""
**Revenue Components:**
- Base Solar Export: ${base_solar_revenue:,.0f}
- Battery Operations (Clipping Only): ${optimal_result['battery_revenue']:,.0f}
- Clipped Energy Captured: {optimal_result['clipped_captured']:.1f} MWh
- Clipped Energy Curtailed: {curtailed:.1f} MWh
- Capture Rate: {100 * optimal_result['clipped_captured'] / max(total_clipped_mwh, 1):.1f}%
- **Total Hybrid Revenue**: ${optimal_result['total_revenue']:,.0f}

**Configuration:**
- Power: {optimal_power:.0f} MW ({100 * optimal_power / peak_clipping_mw:.0f}% of peak clipping)
- Duration: {optimal_duration:.1f}h
- Capacity: {optimal_capacity:.1f} MWh
- Strategy: Clipping-Only ({state.strategy_type} for discharge)
- Forecast Improvement: {state.forecast_improvement}%
- Battery Efficiency: {batt_eff*100:.1f}%
        """)

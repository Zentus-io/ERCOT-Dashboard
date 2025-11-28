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

import plotly.graph_objects as go
import streamlit as st

from config.page_config import configure_page
from core.battery.battery import BatterySpecs
from core.battery.simulator import BatterySimulator
from core.battery.strategies import (
    ClippingOnlyStrategy,
    ClippingAwareMPCStrategy,

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

# Create Tabs
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

# 1. Base Solar Revenue (Same for everyone)
df_sim['Solar_Revenue'] = (df_sim['Export_MW'] * df_sim['price_mwh_rt']) / 4
base_solar_revenue = df_sim['Solar_Revenue'].sum()

# Pre-calculate Daily Stats (Used for Heuristic & Sweep)
df_sim['Date'] = df_sim.index.date
daily_stats = df_sim.groupby('Date').agg({
    'Clipped_MW': 'sum', # Sum of MW per interval
    'price_mwh_rt': ['max', 'min']
})
# Flatten columns
daily_stats.columns = ['Daily_Clipped_Sum_MW', 'Daily_Max_Price', 'Daily_Min_Price']
daily_stats['Daily_Clipped_MWh'] = daily_stats['Daily_Clipped_Sum_MW'] / 4

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
        value="10×10 (100 configs, ~50s)"
    )

    # Buttons stacked
    run_optimization = st.button("🚀 Run Optimization", type="primary", width='stretch')
    
    if st.button(
        "🗑️ Clear Cache",
        width='stretch',
        help="Clear cached simulation results to force fresh calculations"
    ):
        st.cache_data.clear()
        st.success("✅ Cache cleared! Next run will be fresh.")
        st.rerun()

    # Manual Search Space Override
    with st.expander("⚙️ Advanced: Manual Search Space", expanded=True):
        use_manual_search = st.checkbox("Override automatic search space")
        manual_min_p = st.number_input("Min Power (MW)", 1.0, 100.0, 1.0)
        manual_max_p = st.number_input("Max Power (MW)", 5.0, 500.0, 20.0)
        manual_min_d = st.number_input("Min Duration (h)", 0.5, 10.0, 0.5, step=0.5)
        manual_max_d = st.number_input("Max Duration (h)", 1.0, 24.0, 10.0, step=0.5)
        
        st.markdown("---")
        enable_smart_expansion = st.checkbox("✨ Enable Smart Expansion", value=True, help="Automatically expand search space if optimal is found at the edge")

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

    # POWER RANGE: Expanded to ensure we catch the optimal
    if use_manual_search:
        min_power_mw = manual_min_p
        max_power_mw = manual_max_p
        search_rationale = "Manual override"
    elif peak_clipping_mw < 0.5:  # Minimal clipping
        st.warning(f"""
⚠️ **Very Low Clipping Detected**: Peak clipping is only {peak_clipping_mw:.2f} MW.
Battery sizing optimization may not be meaningful with such low clipping.
Consider increasing Solar Capacity or reducing Interconnection Limit.
        """)
        min_power_mw = 5
        max_power_mw = 20
        search_rationale = "Minimal clipping (exploratory range)"
    else:
        # --- DATA-DRIVEN HEURISTIC (CAPACITY-AWARE) ---
        # 1. Target Capacity: Max Daily Clipped Energy (MWh)
        target_capacity_mwh = max(max_daily_clipped_mwh * 1.1, 5.0)
        
        # 2. Target Power: Peak Clipping (MW)
        target_power_mw = max(peak_clipping_mw * 1.1, 5.0)
        
        # 3. Max Power Search: Ensure we can capture the daily energy even with reasonable duration (e.g. 2h)
        # If we only search up to Peak MW, we might miss optimal if Duration is constrained.
        # So we ensure Max Power is at least Target Capacity / 2.0
        max_power_mw = max(target_power_mw, target_capacity_mwh / 2.0)
        min_power_mw = max(1.0, peak_clipping_mw * 0.1)
        
        search_rationale = f"Capacity-Aware: Power up to {max_power_mw:.0f} MW (to cover {target_capacity_mwh:.0f} MWh)"
        
        # 4. Max Duration Search: Ensure we can capture energy with lower power
        # Heuristic: Allow duration to reach Target Capacity / Target Power * 1.5
        heuristic_max_d = (target_capacity_mwh / target_power_mw) * 1.5
        
        # Clamp duration to reasonable bounds (4h - 12h) unless data screams for more
        heuristic_max_d = max(4.0, min(heuristic_max_d, 12.0))

    # Display search rationale in left column
    with opt_col_left:
        st.info(f"""
**Search Space Info:**
- **Clipping:** {peak_clipping_mw:.1f} MW peak (99%)
- **Max:** {max_clipping_mw:.1f} MW
- **Energy:** {total_clipped_mwh:,.0f} MWh

**Search Range:**
- **Power:** {min_power_mw:.0f} - {max_power_mw:.0f} MW
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
    current_n_p = base_n_power
    
    # Duration range initialization
    if use_manual_search:
        current_min_d, current_max_d = manual_min_d, manual_max_d
    else:
        current_min_d = 0.25
        current_max_d = heuristic_max_d # Use data-driven heuristic
        search_rationale += f", Duration up to {current_max_d:.1f}h"
    
    current_n_d = base_n_duration

    # Local cache for results: {(power, duration): result_dict}
    results_cache = {}
    
    max_iterations = 3 if enable_smart_expansion else 1
    iteration = 0
    optimal_found = False
    
    start_time = time.time()
    
    # Placeholder for progress UI and heatmap
    with opt_col_right:
        grid_title_placeholder = st.empty()
        heatmap_placeholder = st.empty()  # Single placeholder for ALL iterations
        progress_container = st.container()

    while iteration < max_iterations:
        iteration += 1
        
        # --- 1. Generate Focused Grid ---
        # Power range (Linear)
        power_range = np.linspace(current_min_p, current_max_p, current_n_p)
        power_span = current_max_p - current_min_p
        if power_span > 50:
            power_range = np.round(power_range).astype(int)
            power_range = np.unique(power_range)
        else:
            power_range = np.round(power_range, 1)
            
        # Duration range (Focused / Non-Linear)
        # We want high granularity in the 0.25h - 2.0h range (the "sweet spot")
        # and coarser granularity for the tail.
        if current_max_d > 2.0 and current_n_d >= 5:  # Lower threshold to work with smaller grids
            # Define breakpoints for non-linear grid
            # 0.25 - 2.0h: Very high density (50% of points)
            # 2.0 - 4.0h: Medium density (30% of points)
            # 4.0 - Max: Low density (20% of points)
            
            # Ensure we have enough points for each segment
            n_p1 = max(3, int(current_n_d * 0.5))  # At least 3 points in sweet spot
            n_p2 = max(2, int(current_n_d * 0.3))
            n_p3 = max(current_n_d - n_p1 - n_p2, 1)
            
            if current_max_d > 4.0:
                # Full 3-segment approach
                d1 = np.linspace(current_min_d, 2.0, n_p1)
                d2 = np.linspace(2.0, 4.0, n_p2 + 1)[1:]
                d3 = np.linspace(4.0, current_max_d, n_p3 + 1)[1:]
                duration_range = np.concatenate([d1, d2, d3])
            else:
                # 2-segment: sweet spot + tail
                d1 = np.linspace(current_min_d, 2.0, n_p1)
                d2 = np.linspace(2.0, current_max_d, n_p2 + n_p3 + 1)[1:]
                duration_range = np.concatenate([d1, d2])
            
            duration_range = np.unique(duration_range)
        else:
            duration_range = np.linspace(current_min_d, current_max_d, current_n_d)
            
        duration_range = np.round(duration_range, 2)
        
        total_configs = len(power_range) * len(duration_range)
        
        # --- 2. Update UI ---
        grid_title_placeholder.markdown(f"### 🗺️ Full Simulation Grid: {len(power_range)}×{len(duration_range)} = {total_configs} configurations (Iter {iteration}/{max_iterations})")
        
        with progress_container:
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 10px;'>
                <h4 style='margin: 0; color: #0A5F7A;'>🔄 Running Iteration {iteration}</h4>
                <p style='margin: 5px 0 0 0; color: #666;'>Testing range: {current_min_p:.0f}-{current_max_p:.0f} MW × {current_min_d:.1f}-{current_max_d:.1f}h</p>
            </div>
            """, unsafe_allow_html=True)
            progress_bar = st.progress(0)
            status_col1, status_col2, status_col3 = st.columns(3)
            with status_col1: status_completed = st.empty()
            with status_col2: status_current = st.empty()
            with status_col3: status_time = st.empty()

        # --- 3. Identify New Configs & Apply Saturation Skipping ---
        # Sort by Capacity (Power * Duration) to hit saturation early
        # We want to process smaller batteries first.
        all_configs = []
        for p in power_range:
            for d in duration_range:
                all_configs.append((p, d))
        
        # Sort by capacity
        all_configs.sort(key=lambda x: x[0] * x[1])
        
        configs_to_run = []
        saturated_configs = [] # List of (p, d, revenue) that achieved 0 curtailment
        
        # First pass: Check cache and saturation
        # We can't fully skip yet because we need to run simulations in order.
        # But we can identify what needs to be run.
        
        # Actually, for saturation skipping to work efficiently with parallelism, 
        # we should run in batches or just check saturation against *cached* results first.
        # Simplified approach: We will run all non-cached, but we check if any CACHED result
        # already dominates this config with 0 curtailment.
        
        # Better approach: We just run them. But if we find a result has 0 curtailment,
        # we can potentially skip future larger ones in the same batch if we are careful.
        # For now, let's stick to the user's request: "Skip them to save computation time".
        # This implies we shouldn't even submit them.
        
        # To do this robustly in parallel, we need to know if a smaller config is saturated.
        # But we haven't run the smaller config yet! 
        # So we can only skip based on *already cached* saturated results.
        # OR we run in generations. 
        # Let's stick to: Skip if dominated by a CACHED saturated config.
        # AND: If we run a config and it returns saturated, we add it to cache for next iteration/batch.
        
        # Identify saturated configs in cache
        for (cp, cd), res in results_cache.items():
             # Check if 'curtailed_mwh' is 0 (or very close)
             # We need to ensure 'curtailed_mwh' is in the result. 
             # If not available, we can't skip.
             if res.get('curtailed_mwh', 1.0) < 0.1:
                 saturated_configs.append((cp, cd, res['total_revenue']))

        final_configs_to_run = []
        skipped_count = 0
        
        for p, d in all_configs:
            if (p, d) in results_cache:
                continue
                
            # Check if dominated by any saturated config
            is_dominated = False
            dominated_revenue = 0
            for sp, sd, srev in saturated_configs:
                if p >= sp and d >= sd:
                    is_dominated = True
                    dominated_revenue = srev
                    break
            
            if is_dominated:
                # Skip simulation, artificially populate cache with EXACT revenue
                results_cache[(p, d)] = {
                    'power_mw': p,
                    'duration_h': d,
                    'total_revenue': dominated_revenue,  # EXACT copy
                    'curtailed_mwh': 0.0, # It's saturated
                    'is_skipped': True,  # Explicit flag for visualization
                    'revenue_components': {} # Placeholder
                }
                skipped_count += 1
            else:
                final_configs_to_run.append((p, d))
                
        n_new = len(final_configs_to_run)
        n_cached_or_skipped = total_configs - n_new
        
        # --- 4. Run Simulations & Live Updates ---
        if n_new > 0:
            # Helper to update heatmap live
            def update_live_heatmap(curr_results_cache, curr_power_range, curr_duration_range, iter_num):
                # Build matrices for ALL cached results (cumulative across iterations)
                rev_mat = np.zeros((len(curr_power_range), len(curr_duration_range)))
                skipped_x = []
                skipped_y = []
                
                for i, p_val in enumerate(curr_power_range):
                    for j, d_val in enumerate(curr_duration_range):
                        res = curr_results_cache.get((p_val, d_val))
                        if res:
                            rev_mat[i, j] = res['total_revenue']
                            # Check if explicitly marked as skipped
                            if res.get('is_skipped', False):
                                skipped_x.append(p_val)
                                skipped_y.append(d_val)
                
                # Create figure
                fig = go.Figure()
                
                # Heatmap
                fig.add_trace(go.Heatmap(
                    z=rev_mat.T,
                    x=curr_power_range,
                    y=curr_duration_range,
                    colorscale='Viridis',
                    colorbar=dict(title='Revenue ($)'),
                    hovertemplate='Power: %{x:.0f} MW<br>Duration: %{y:.1f} h<br>Revenue: $%{z:,.0f}<extra></extra>'
                ))
                
                # Skipped Markers (with better visibility)
                if skipped_x:
                    fig.add_trace(go.Scatter(
                        x=skipped_x,
                        y=skipped_y,
                        mode='markers',
                        marker=dict(
                            symbol='x',
                            color='white',
                            size=10,
                            line=dict(color='black', width=1)
                        ),
                        name='Skipped (Saturated)',
                        showlegend=True
                    ))

                fig.update_layout(
                    title=f'Revenue Optimization Landscape (Iter {iter_num})',
                    xaxis_title='Battery Power (MW)',
                    yaxis_title='Battery Duration (hours)',
                    height=500,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                # Use the SINGLE placeholder created outside the loop
                heatmap_placeholder.plotly_chart(fig, use_container_width=True)

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(simulate_battery_config, p, d): (p, d) for p, d in final_configs_to_run}
                
                completed_new = 0
                for future in as_completed(futures):
                    p, d = futures[future]
                    result = future.result()
                    results_cache[(p, d)] = result
                    
                    # Check for saturation to help future skips (in next loop/iter)
                    if result.get('curtailed_mwh', 1.0) < 0.1:
                         saturated_configs.append((p, d, result['total_revenue']))
                    
                    completed_new += 1
                    # Progress update
                    total_completed = n_cached_or_skipped + completed_new
                    progress_pct = total_completed / total_configs
                    progress_bar.progress(progress_pct)
                    status_completed.metric("Completed", f"{total_completed}/{total_configs}")
                    status_current.metric("Current", f"{p:.0f}MW × {d:.1f}h")
                    
                    # ETA Calculation
                    elapsed = time.time() - start_time
                    if completed_new > 0:
                        rate = elapsed / completed_new 
                        remaining_sims = n_new - completed_new
                        eta_seconds = rate * remaining_sims
                        status_time.metric("ETA", f"{eta_seconds:.0f}s")
                    
                    # LIVE UPDATE every 5 simulations
                    if completed_new % 5 == 0:
                        update_live_heatmap(results_cache, power_range, duration_range, iteration)
        else:
             progress_bar.progress(1.0)
             status_completed.metric("Completed", f"{total_configs}/{total_configs}")
             status_time.metric("ETA", "0s")
             
        if skipped_count > 0:
            st.toast(f"⏩ Skipped {skipped_count} saturated configs (Plateau Optimization)", icon="🚀")

        # --- 5. Analyze Results & Check Edges ---
        # Build revenue matrix for current grid
        revenue_matrix = np.zeros((len(power_range), len(duration_range)))
        results_grid = [] # For finding optimal result object later
        
        for i, p in enumerate(power_range):
            for j, d in enumerate(duration_range):
                res = results_cache.get((p, d))
                if res:
                    revenue_matrix[i, j] = res['total_revenue']
                    results_grid.append(res)
        
        # --- FINAL ITERATION UPDATE ---
        # Ensure the final state of this iteration is shown
        # We reuse the helper logic but we need to call it or just let the loop finish.
        # The helper above uses st.session_state placeholders. 
        # Let's just do a final update call.
        # Note: We need to define update_live_heatmap outside the if block if we want to call it here,
        # but it was defined inside. That's a scope issue if n_new=0.
        # However, if n_new=0, we didn't run sims, so we just show the static plot.
        pass # We will rely on the final plot at the end of function or the last live update.

        # --- FIND OPTIMAL (SMART PLATEAU HANDLING) ---
        # Instead of just argmax (which picks the first max), we find ALL configs close to max revenue
        # and pick the one with the SMALLEST CAPACITY.
        max_rev = np.max(revenue_matrix)
        tolerance = max_rev * 0.001 # 0.1% tolerance for "equal" revenue
        
        # Get indices of all configs within tolerance of max
        candidates_idx = np.argwhere(revenue_matrix >= max_rev - tolerance)
        
        best_candidate = None
        min_capacity = float('inf')
        
        for idx in candidates_idx:
            i, j = idx
            p = power_range[i]
            d = duration_range[j]
            cap = p * d
            
            # We want smallest capacity. If tie, smallest power.
            if cap < min_capacity:
                min_capacity = cap
                best_candidate = (i, j)
            elif cap == min_capacity:
                if p < power_range[best_candidate[0]]:
                    best_candidate = (i, j)
        
        optimal_idx_i, optimal_idx_j = best_candidate
        optimal_power = power_range[optimal_idx_i]
        optimal_duration = duration_range[optimal_idx_j]
        optimal_revenue = revenue_matrix[optimal_idx_i, optimal_idx_j]
        
        # Check Edges (of the CHOSEN optimal, not the plateau edge)
        # We only expand if the "Knee" itself is at the edge.
        is_max_power_edge = (optimal_idx_i == len(power_range) - 1)
        is_max_duration_edge = (optimal_idx_j == len(duration_range) - 1)
        
        # Smart Expansion Logic
        expanded = False
        if enable_smart_expansion and iteration < max_iterations:
            if is_max_power_edge:
                current_max_p = current_max_p * 1.5
                current_n_p = int(current_n_p * 1.5)
                expanded = True
                st.toast(f"⚡ Expanding Power Range to {current_max_p:.0f} MW (Iter {iteration})", icon="🔄")
            
            if is_max_duration_edge:
                current_max_d = current_max_d * 1.5
                current_n_d = int(current_n_d * 1.5)
                expanded = True
                st.toast(f"⚡ Expanding Duration Range to {current_max_d:.1f} h (Iter {iteration})", icon="🔄")
        
        if not expanded:
            break # Optimal found within bounds or max iterations reached
            
    # --- Final Cleanup ---
    elapsed_time = time.time() - start_time
    progress_container.empty()
    st.success(f"✅ Optimization complete! Found optimal in {iteration} iteration(s).")
    
    # Re-calculate final optimal details for display
    optimal_capacity = optimal_power * optimal_duration
    optimal_result = next(r for r in results_grid if r['power_mw'] == optimal_power and r['duration_h'] == optimal_duration)
    
    # Check final edge status for warning
    is_power_edge = (optimal_power == current_min_p) or (optimal_power == current_max_p)
    is_duration_edge = (optimal_duration == current_min_d) or (optimal_duration == current_max_d)
    
    if is_power_edge or is_duration_edge:
        edge_dims = []
        if is_power_edge: edge_dims.append(f"power ({optimal_power:.0f} MW)")
        if is_duration_edge: edge_dims.append(f"duration ({optimal_duration:.1f}h)")
        st.warning(f"⚠️ **Search Space Limit Reached**: Optimal is at edge of {' and '.join(edge_dims)} range even after expansion.")

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

        # --- 2. Heatmap Visualization ---
        st.markdown("### 📊 Revenue Heatmap")

        # Create heatmap with Plotly
        fig_heatmap = go.Figure(data=go.Heatmap(
        z=revenue_matrix,
        x=duration_range,
        y=power_range,
        colorscale='Viridis',
        hovertemplate='<b>Configuration</b><br>Power: %{y:.0f} MW<br>Duration: %{x:.1f}h<br>Revenue: $%{z:,.0f}<extra></extra>',
        colorbar=dict(
            title="Revenue ($)",
            tickformat="$,.0f"
        ),
        xgap=2,  # Gap between cells
        ygap=2
    ))

        # Add star marker at optimal configuration
        fig_heatmap.add_trace(go.Scatter(
            x=[optimal_duration],
            y=[optimal_power],
            mode='markers+text',
            marker=dict(
                color='white',
                size=25,
                symbol='star',
                line=dict(width=3, color='red')
            ),
            text=["★"],
            textfont=dict(size=24, color='white'),
            name='Optimal',
            showlegend=True,
            hovertemplate=f'<b>OPTIMAL</b><br>Power: {optimal_power:.0f} MW<br>Duration: {optimal_duration:.1f}h<br>Revenue: ${optimal_revenue:,.0f}<extra></extra>'
        ))

        # Calculate appropriate chart height based on number of power levels
        # More power levels = taller chart to maintain readable cell sizes
        # Reduced multiplier to make cells smaller and chart less overwhelming
        base_height = 400 
        cell_height = 35  # Reduced from 50
        chart_height = max(base_height, len(power_range) * cell_height + 80)

        fig_heatmap.update_layout(
            title=dict(
                text=f"Full Simulation Grid Search: {len(power_range)}×{len(duration_range)} = {total_configs} Configurations",
                font=dict(size=16)
            ),
            xaxis=dict(
                title="Battery Duration (hours)",
                side="bottom",
                tickmode='array',
                tickvals=duration_range,
                ticktext=[f"{d:.1f}h" for d in duration_range]
            ),
            yaxis=dict(
                title="Battery Power (MW)",
                tickmode='array',
                tickvals=power_range,
                ticktext=[f"{p:.0f}" for p in power_range]
            ),
            height=chart_height,
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=80, r=120, t=80, b=80)
        )

        st.plotly_chart(fig_heatmap, width='stretch')

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

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
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.page_config import configure_page
from core.battery.battery import BatterySpecs
from core.battery.simulator import BatterySimulator
from core.battery.strategies import (
    ClippingOnlyStrategy,
    LinearOptimizationStrategy,
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
tab1, tab2 = st.tabs(["☀️ Hybrid Design", "📈 Strategy Analysis"])

# ============================================================================
# TAB 1: HYBRID DESIGN
# ============================================================================
with tab1:
    st.subheader(f"Asset Optimization: {state.selected_node}")

    with st.expander("📖 How to use this page"):
        st.markdown("""
        **Goal:** Design a hybrid Solar + Storage asset and optimize the battery size to capture "clipped" energy (energy lost when solar generation exceeds the grid interconnection limit).

        ### What This Tool Does
        - **Optimization Focus:** Battery sizing for clipping capture ONLY
        - **No Grid Arbitrage:** Battery only charges from clipped solar energy (free)
        - **Discharge Optimization:** Uses your selected strategy (MPC, Rolling Window, Threshold) to optimize when to sell stored clipped energy
        - **Revenue Source:** Value of clipped energy sold at optimal times

        ### How It Works
        1. **Forced Charging:** Battery automatically charges when clipping occurs (up to battery power limit)
        2. **Smart Discharging:** Selected strategy optimizes discharge timing to maximize revenue
        3. **Curtailment Tracking:** Shows clipping that couldn't be captured (when battery is full)

        ### 1. Configure Your Asset (Left Panel)
        *   **Solar Capacity (MW):** The total DC capacity of your solar array.
        *   **Interconnection Limit (MW):** The maximum power you are allowed to export to the grid (POI limit).
        *   **Solar Profile:** Uses real historical **Solar Potential** (Resource Availability) from the database for the selected region.

        ### 2. Understand the Simulation
        *   **Yellow Curve (Resource):** The total solar energy available based on the region's weather.
        *   **Orange Area (Export):** The energy sold to the grid (capped at the Interconnection Limit).
        *   **Green Area (Clipping):** The "wasted" energy that exceeds the limit. This is the opportunity for your battery!

        ### 3. Analyze Performance
        *   **Current Asset:** Shows revenue for your currently selected battery (from the Sidebar).
        *   **Optimal Asset:** The system tests different battery sizes to find the configuration that captures the most clipped energy value.
        """)
    
    # --- CONFIGURATION SECTION ---
    col_left, col_right = st.columns([1, 3])
    
    with col_left:
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
        
        # --- SMART SLICING / DATA FETCHING ---
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

    with col_right:
        # --- SIMULATION ENGINE ---
        
        # 1. Calculate Solar Vectors
        # Use Potential MW (Resource Availability) for simulation if available, else Actuals
        # The profile is already normalized 0-1
        df_sim = df_prices.copy()
        df_sim['Solar_MW'] = solar_profile['potential_mw'] * solar_capacity_mw
        df_sim['Export_MW'] = np.minimum(df_sim['Solar_MW'], interconnection_limit_mw)
        df_sim['Clipped_MW'] = np.maximum(0, df_sim['Solar_MW'] - interconnection_limit_mw)

        # Validate clipping calculations
        is_valid_clipping, clipping_warnings = validate_clipped_energy(
            df_sim['Clipped_MW'],
            df_sim['Solar_MW'],
            interconnection_limit_mw
        )
        if clipping_warnings:
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
            display_validation_warnings(config_warnings, title="Hybrid Configuration")

        # ============================================================================
        # FULL SIMULATION GRID OPTIMIZATION
        # ============================================================================

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
        base_solar_revenue = df_sim['Solar_Revenue'].sum()

        # Calculate current asset revenue (for comparison with optimal)
        current_power = state.battery_specs.power_mw
        current_capacity = state.battery_specs.capacity_mwh

        # Run quick simulation for current asset to get baseline revenue
        with st.spinner("Calculating current asset revenue..."):
            baseline_res, improved_res, _, _ = run_or_get_cached_simulation()
            if improved_res:
                current_batt_revenue = improved_res.total_revenue
            else:
                current_batt_revenue = 0.0

        current_hybrid_rev = base_solar_revenue + current_batt_revenue

        # Display current asset performance
        st.markdown("### 📊 Current Asset Performance")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Battery Power", f"{current_power:.0f} MW")
        col2.metric("Battery Capacity", f"{current_capacity:.1f} MWh")
        col3.metric("Battery Revenue", f"${current_batt_revenue:,.0f}")
        col4.metric("Total Hybrid Revenue", f"${current_hybrid_rev:,.0f}")

        # --- FULL SIMULATION GRID OPTIMIZATION ---
        st.markdown("---")
        st.markdown("### 🔋 Battery Sizing Optimization")

        st.markdown("""
**Clipping-Only Grid Search**: Find the optimal battery configuration by running full simulations
across different power and duration combinations. Battery only charges from clipped solar energy
and uses your selected dispatch strategy to optimize discharge timing for maximum revenue.
        """)

        # Grid resolution slider and buttons
        col_slider, col_run, col_clear = st.columns([3, 1, 1])

        with col_slider:
            grid_resolution = st.select_slider(
                "Grid Resolution (Power × Duration)",
                options=[
                    "5×5 (25 configs, ~12s)",
                    "7×7 (49 configs, ~25s)",
                    "10×10 (100 configs, ~50s)",
                    "12×12 (144 configs, ~1min)",
                    "15×15 (225 configs, ~2min)",
                    "20×20 (400 configs, ~3min)"
                ],
                value="7×7 (49 configs, ~25s)",
                help="Higher resolution = more accurate but slower. 20×20 provides maximum detail."
            )

        with col_run:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            run_optimization = st.button(
                "▶️ Run Optimization",
                type="primary",
                use_container_width=True,
                help="Start the full simulation grid search"
            )

        with col_clear:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            if st.button(
                "🗑️ Clear Cache",
                use_container_width=True,
                help="Clear cached simulation results to force fresh calculations"
            ):
                st.cache_data.clear()
                st.success("✅ Cache cleared! Next run will be fresh.")
                st.rerun()

        if run_optimization or st.session_state.get('optimization_running', False):
            # ============================================================================
            # CLIPPING-BASED SEARCH SPACE CALCULATION
            # ============================================================================

            # Analyze clipping patterns
            peak_clipping_mw = df_sim['Clipped_MW'].quantile(0.95)  # 95th percentile
            total_clipped_mwh = df_sim['Clipped_MW'].sum() / 4  # Convert to MWh
            clipping_hours = (df_sim['Clipped_MW'] > 0.1).sum()  # Hours with meaningful clipping

            # Calculate daily clipping statistics
            daily_clipped = df_sim.groupby(df_sim.index.date)['Clipped_MW'].sum() / 4  # Daily MWh
            avg_daily_clipped = daily_clipped.mean()
            peak_daily_clipped = daily_clipped.quantile(0.95)

            # Calculate clipping duration patterns (for capacity sizing)
            daily_clipping_hours = df_sim.groupby(df_sim.index.date).apply(
                lambda x: (x['Clipped_MW'] > 0.1).sum()
            )
            avg_clipping_duration = daily_clipping_hours.mean() / 4  # Convert intervals to hours

            # POWER RANGE: Based on peak clipping capture capability
            if peak_clipping_mw < 0.5:  # Minimal clipping
                st.warning(f"""
⚠️ **Very Low Clipping Detected**: Peak clipping is only {peak_clipping_mw:.2f} MW.
Battery sizing optimization may not be meaningful with such low clipping.
Consider increasing Solar Capacity or reducing Interconnection Limit.
                """)
                min_power_mw = 5
                max_power_mw = 20
                search_rationale = "Minimal clipping (exploratory range)"
            else:
                # Size battery to capture 25% to 125% of peak clipping
                # 25%: Captures only highest-value clipping periods
                # 100%: Captures all peak clipping
                # 125%: Headroom for variations
                min_power_mw = max(5, peak_clipping_mw * 0.25)
                max_power_mw = peak_clipping_mw * 1.25
                search_rationale = f"Clipping capture optimized (peak: {peak_clipping_mw:.1f} MW)"

            # Display search rationale
            st.info(f"""
📊 **Clipping-Based Search Space:**
- **Clipping Analysis:**
  - Peak clipping: {peak_clipping_mw:.1f} MW (95th percentile)
  - Total clipped energy: {total_clipped_mwh:,.0f} MWh
  - Clipping hours: {clipping_hours} intervals/year ({clipping_hours/4:.0f} hours)
  - Avg daily clipped: {avg_daily_clipped:.1f} MWh
  - Avg clipping duration: {avg_clipping_duration:.1f} hours/day

- **Battery Power Range:**
  - Power: **{min_power_mw:.0f} to {max_power_mw:.0f} MW**
  - Rationale: {search_rationale}
  - Duration range determined by grid resolution (see slider)

- **Optimization Focus:** Clipping capture only (no grid arbitrage)
            """)

            # Generate cache key for invalidation
            cache_inputs = f"{solar_capacity_mw}_{interconnection_limit_mw}_{batt_eff}_{df_prices.index.min()}_{df_prices.index.max()}_{min_power_mw}_{max_power_mw}"
            cache_key = hashlib.md5(cache_inputs.encode()).hexdigest()

            # ============================================================================
            # GRID CONFIGURATION (Based on Slider Selection)
            # ============================================================================

            # Parse grid resolution from slider
            resolution_map = {
                "5×5 (25 configs, ~12s)": (5, 5),
                "7×7 (49 configs, ~25s)": (7, 7),
                "10×10 (100 configs, ~50s)": (10, 10),
                "12×12 (144 configs, ~1min)": (12, 12),
                "15×15 (225 configs, ~2min)": (15, 15),
                "20×20 (400 configs, ~3min)": (20, 20)
            }
            n_power, n_duration = resolution_map[grid_resolution]

            # Duration range: adaptive based on resolution
            # Focus on small durations (0.25h - 2h) with high granularity for typical Engie assets (1-2h or <1h)
            if n_duration == 5:
                # Small range: 0.25h to 2h
                duration_range = np.array([0.25, 0.5, 1.0, 1.5, 2.0])
            elif n_duration == 7:
                # Small-medium range: 0.25h to 3h
                duration_range = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
            elif n_duration == 10:
                # Extended range with granularity: 0.25h to 4h
                duration_range = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0])
            elif n_duration == 12:
                # Full range with high granularity: 0.25h to 6h
                duration_range = np.linspace(0.25, 6.0, 12)
            elif n_duration == 15:
                # Extended range: 0.25h to 8h
                duration_range = np.linspace(0.25, 8.0, 15)
            elif n_duration == 20:
                # Maximum range: 0.25h to 10h (for edge cases with long clipping periods)
                duration_range = np.linspace(0.25, 10.0, 20)
            else:
                # Fallback: default to 0.5h - 6h range
                duration_range = np.linspace(0.5, 6.0, n_duration)

            # Round to 2 decimal places for cleaner display
            duration_range = np.round(duration_range, 2)

            # Power range: n_power levels dynamically spaced
            power_range = np.linspace(min_power_mw, max_power_mw, n_power)

            # Smart rounding: Only round to integers for large ranges (>50 MW span)
            # For small ranges, keep decimal precision to ensure requested resolution
            power_span = max_power_mw - min_power_mw
            if power_span > 50:
                # Large range: round to integers
                power_range = np.round(power_range).astype(int)
                power_range = np.unique(power_range)  # Remove duplicates
            else:
                # Small range: keep 1 decimal place to preserve resolution
                power_range = np.round(power_range, 1)

            # Total configurations
            total_configs = len(power_range) * len(duration_range)

            st.markdown(f"### 🗺️ Full Simulation Grid: {len(power_range)}×{len(duration_range)} = {total_configs} configurations")

            # ============================================================================
            # PREPARE DATA FOR SIMULATION
            # ============================================================================

            # Prepare price data with clipped_mw column for hybrid simulation
            df_hybrid = df_sim.copy()
            df_hybrid['clipped_mw'] = df_hybrid['Clipped_MW']

            # Get improvement factor from state
            improvement_factor = state.forecast_improvement / 100.0

            # ============================================================================
            # SIMULATION FUNCTION
            # ============================================================================

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
                        base_strategy = MPCStrategy(state.horizon_hours)
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

                    # Extract metadata from ClippingOnlyStrategy
                    metadata = clipping_strategy.get_metadata()

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

            # ============================================================================
            # PARALLEL EXECUTION (7×5 Grid = 35 Configurations)
            # ============================================================================

            import time
            from concurrent.futures import as_completed

            start_time = time.time()

            # Enhanced progress visualization
            progress_container = st.container()
            with progress_container:
                st.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 10px;'>
                    <h4 style='margin: 0; color: #0A5F7A;'>🔄 Running Full Simulation Grid</h4>
                    <p style='margin: 5px 0 0 0; color: #666;'>Testing {total_configs} configurations (Power: {min_power_mw:.0f}-{max_power_mw:.0f} MW × Duration: {duration_range[0]:.1f}-{duration_range[-1]:.1f}h)</p>
                </div>
                """, unsafe_allow_html=True)

                progress_bar = st.progress(0)
                status_col1, status_col2, status_col3 = st.columns(3)
                with status_col1:
                    status_completed = st.empty()
                with status_col2:
                    status_current = st.empty()
                with status_col3:
                    status_time = st.empty()

            # Initialize results storage
            results_grid = []
            revenue_matrix = np.zeros((len(power_range), len(duration_range)))

            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit all power × duration combinations
                futures = {}
                for i, power_mw in enumerate(power_range):
                    for j, duration_h in enumerate(duration_range):
                        future = executor.submit(simulate_battery_config, power_mw, duration_h)
                        futures[future] = (i, j, power_mw, duration_h)

                # Collect results as they complete
                completed = 0
                for future in as_completed(futures):
                    result = future.result()
                    i, j, power_mw, duration_h = futures[future]
                    revenue_matrix[i, j] = result['total_revenue']
                    results_grid.append(result)
                    completed += 1

                    # Update progress
                    progress_pct = completed / total_configs
                    progress_bar.progress(progress_pct)
                    status_completed.metric("Completed", f"{completed}/{total_configs}")
                    status_current.metric("Current", f"{power_mw:.0f}MW × {duration_h:.1f}h")
                    elapsed = time.time() - start_time
                    eta = (elapsed / completed) * (total_configs - completed) if completed > 0 else 0
                    status_time.metric("ETA", f"{eta:.0f}s")

            elapsed_time = time.time() - start_time
            progress_container.empty()

            # Check if suspiciously fast (likely cached)
            expected_min_time = total_configs * 0.3  # Minimum ~0.3s per config
            if elapsed_time < expected_min_time:
                st.warning(f"""
⚡ **Very fast completion ({elapsed_time:.1f}s)!** This suggests:
- Results may be **cached** from previous run
- OR data size is very small
- Expected minimum: ~{expected_min_time:.0f}s for {total_configs} fresh simulations
                """)
            else:
                st.success(f"✅ Grid search complete! Simulated {total_configs} configurations in {elapsed_time:.1f} seconds ({elapsed_time/total_configs:.2f}s per config)")

            # ============================================================================
            # FIND OPTIMAL CONFIGURATION
            # ============================================================================

            # Find optimal in revenue matrix
            optimal_idx_flat = np.argmax(revenue_matrix)
            optimal_idx_i, optimal_idx_j = np.unravel_index(optimal_idx_flat, revenue_matrix.shape)
            optimal_power = power_range[optimal_idx_i]
            optimal_duration = duration_range[optimal_idx_j]
            optimal_capacity = optimal_power * optimal_duration
            optimal_revenue = revenue_matrix[optimal_idx_i, optimal_idx_j]

            # Find the corresponding result details
            optimal_result = next(
                r for r in results_grid
                if r['power_mw'] == optimal_power and r['duration_h'] == optimal_duration
            )

            # ============================================================================
            # DISPLAY OPTIMAL CONFIGURATION
            # ============================================================================

            st.markdown("### 🌟 Optimal Battery Configuration")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Optimal Power", f"{optimal_power:.0f} MW")
            col2.metric("Optimal Duration", f"{optimal_duration:.1f}h")
            col3.metric("Optimal Capacity", f"{optimal_capacity:.1f} MWh")

            # Compare with current asset
            revenue_improvement = optimal_revenue - current_hybrid_rev
            col4.metric("Total Revenue", f"${optimal_revenue:,.0f}",
                        delta=f"+${revenue_improvement:,.0f}")

            # Check if optimal is near edge (suggests broader search needed)
            is_power_edge = optimal_idx_i >= len(power_range) - 1  # Last power level
            is_duration_edge = optimal_idx_j >= len(duration_range) - 1  # Last duration level

            if is_power_edge or is_duration_edge:
                edge_dims = []
                if is_power_edge:
                    edge_dims.append(f"power ({optimal_power:.0f} MW)")
                if is_duration_edge:
                    edge_dims.append(f"duration ({optimal_duration:.1f}h)")
                st.warning(f"""
⚠️ **Search Space May Be Too Small**: Optimal is at edge of {' and '.join(edge_dims)} range.
The true optimal might be even larger! Consider:
- Increasing search range manually, or
- Re-running with broader multipliers
- True optimal could be outside current search space
                """)

            # ============================================================================
            # HEATMAP VISUALIZATION
            # ============================================================================

            st.markdown("### 📊 Revenue Heatmap (Power × Duration)")

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
            base_height = 400
            cell_height = 50  # Target ~50px per power level
            chart_height = max(base_height, len(power_range) * cell_height + 200)

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

            st.plotly_chart(fig_heatmap, use_container_width=True)

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

        # 2. Operational Chart (Solar + Clipping Visualization)
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
        
        fig_hybrid.update_layout(
            title="Hybrid Asset Operation",
            xaxis_title="Time",
            yaxis_title="Power (MW)",
            height=400,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_hybrid, width="stretch")


# ============================================================================
# TAB 2: STRATEGY ANALYSIS (Existing Logic)
# ============================================================================
with tab2:
    # ============================================================================
    # SENSITIVITY ANALYSIS
    # ============================================================================

    st.subheader("Impact of Forecast Accuracy on Revenue")

    st.info("""
    **Chart Guide:** The green LP Benchmark line represents the **theoretical maximum** achievable with perfect hindsight.
    The gap between your selected strategy and LP shows potential gains from strategy improvements.
    """)

    # Reduce points for performance (0, 10, 20... 100)


    def run_sensitivity_analysis(
        node_data: pd.DataFrame,
        battery_specs: BatterySpecs,
        charge_percentile: float,
        discharge_percentile: float,
        window_hours: int
    ):
        """
        Run sensitivity analysis with progress bar and parallel strategy execution.
        """
        improvement_range = range(0, 101, 10)  # Reduced resolution for speed
        revenue_threshold = []
        revenue_rolling_window = []
        revenue_mpc = []
        revenue_linear = []

        simulator = BatterySimulator(battery_specs)

        progress_bar = st.progress(0)
        status_text = st.empty()
        total_steps = len(improvement_range)

        def run_strategy(strategy, improvement_factor):
            """Helper function to run a single strategy simulation."""
            return simulator.run(node_data, strategy, improvement_factor=improvement_factor)

        for i, imp in enumerate(improvement_range):
            status_text.text(f"Simulating forecast improvement: {imp}%...")
            improvement_factor = imp / 100

            # Create strategy instances
            strategy_threshold = ThresholdStrategy(charge_percentile, discharge_percentile)
            strategy_window = RollingWindowStrategy(window_hours)
            strategy_mpc = MPCStrategy(horizon_hours=24)
            strategy_linear = LinearOptimizationStrategy()

            # Run all strategies in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit all strategy simulations
                future_threshold = executor.submit(run_strategy, strategy_threshold, improvement_factor)
                future_window = executor.submit(run_strategy, strategy_window, improvement_factor)
                future_mpc = executor.submit(run_strategy, strategy_mpc, improvement_factor)
                future_linear = executor.submit(run_strategy, strategy_linear, improvement_factor)

                # Collect results
                result_threshold = future_threshold.result()
                result_window = future_window.result()
                result_mpc = future_mpc.result()
                result_linear = future_linear.result()

            revenue_threshold.append(result_threshold.total_revenue)
            revenue_rolling_window.append(result_window.total_revenue)
            revenue_mpc.append(result_mpc.total_revenue)
            revenue_linear.append(result_linear.total_revenue)

            progress_bar.progress((i + 1) / total_steps)

        status_text.empty()
        progress_bar.empty()

        return list(improvement_range), revenue_threshold, revenue_rolling_window, revenue_mpc, revenue_linear


    def run_window_sensitivity_analysis(
        node_data: pd.DataFrame,
        battery_specs: BatterySpecs,
        improvement_factor: float
    ):
        """
        Run sensitivity analysis for Rolling Window strategy varying window size.
        """
        window_range = range(2, 49, 2)  # 2 to 48 hours
        revenue_window = []

        simulator = BatterySimulator(battery_specs)

        progress_bar = st.progress(0)
        status_text = st.empty()
        total_steps = len(window_range)

        for i, window in enumerate(window_range):
            status_text.text(f"Simulating window size: {window} hours...")
            strategy = RollingWindowStrategy(window_hours=window)
            result = simulator.run(
                node_data,
                strategy,
                improvement_factor=improvement_factor
            )
            revenue_window.append(result.total_revenue)
            progress_bar.progress((i + 1) / total_steps)

        status_text.empty()
        progress_bar.empty()

        return list(window_range), revenue_window


    def run_horizon_sensitivity_analysis(
        node_data: pd.DataFrame,
        battery_specs: BatterySpecs,
        improvement_factor: float
    ):
        """
        Run sensitivity analysis for MPC strategy varying horizon size.
        """
        horizon_range = range(2, 14, 2)  # 2, 4, 6... 24 hours
        revenue_horizon = []

        simulator = BatterySimulator(battery_specs)

        progress_bar = st.progress(0)
        status_text = st.empty()
        total_steps = len(horizon_range)

        for i, horizon in enumerate(horizon_range):
            status_text.text(f"Simulating MPC horizon: {horizon} hours...")
            strategy = MPCStrategy(horizon_hours=horizon)
            result = simulator.run(
                node_data,
                strategy,
                improvement_factor=improvement_factor
            )
            revenue_horizon.append(result.total_revenue)
            progress_bar.progress((i + 1) / total_steps)

        status_text.empty()
        progress_bar.empty()

        return list(horizon_range), revenue_horizon


    # Run forecast sensitivity analysis
    improvement_range, revenue_threshold, revenue_rolling_window, revenue_mpc, revenue_linear = run_sensitivity_analysis(
        node_data, state.battery_specs, state.charge_percentile, state.discharge_percentile, state.window_hours)

    # Create comparison chart
    fig_sensitivity = go.Figure()

    fig_sensitivity.add_trace(go.Scatter(
        x=list(improvement_range),
        y=revenue_threshold,
        name='Threshold-Based',
        line={"color": '#6B7280', "width": 2},
        mode='lines+markers',
        hovertemplate='Threshold<br>Improvement: %{x}%<br>Revenue: $%{y:,.0f}<extra></extra>'
    ))

    fig_sensitivity.add_trace(go.Scatter(
        x=list(improvement_range),
        y=revenue_rolling_window,
        name='Rolling Window',
        line={"color": '#0A5F7A', "width": 3},
        mode='lines+markers',
        hovertemplate='Rolling Window<br>Improvement: %{x}%<br>Revenue: $%{y:,.0f}<extra></extra>'
    ))

    fig_sensitivity.add_trace(go.Scatter(
        x=list(improvement_range),
        y=revenue_mpc,
        name='MPC (24h Horizon)',
        line={"color": '#8B5CF6', "width": 3},
        mode='lines+markers',
        hovertemplate='MPC<br>Improvement: %{x}%<br>Revenue: $%{y:,.0f}<extra></extra>'
    ))

    # LP Benchmark curve (theoretical upper bound at each improvement level)
    fig_sensitivity.add_trace(go.Scatter(
        x=list(improvement_range),
        y=revenue_linear,
        name='LP Benchmark (Theoretical Max)',
        line={"color": '#28a745', "width": 3},
        mode='lines+markers',
        hovertemplate='LP Benchmark<br>Improvement: %{x}%<br>Revenue: $%{y:,.0f}<extra></extra>'
    ))

    fig_sensitivity.update_layout(
        title="Revenue Sensitivity: Practical Strategies vs LP Benchmark",
        xaxis_title="Forecast Improvement (%)",
        yaxis_title="Revenue ($)",
        height=500,
        hovermode='x unified',
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
        yaxis={"fixedrange": False},  # Allow Y-axis zooming
        xaxis={"fixedrange": False}
    )

    st.plotly_chart(fig_sensitivity, width="stretch")

    # Window Sensitivity (Only for Rolling Window Strategy)
    if state.strategy_type == "Rolling Window Optimization":
        st.markdown("---")
        st.subheader(f"Window Size Sensitivity (at {state.forecast_improvement}% Improvement)")

        window_range, revenue_window_sens = run_window_sensitivity_analysis(
            node_data,
            state.battery_specs,
            state.forecast_improvement / 100.0
        )

        fig_window = go.Figure()

        fig_window.add_trace(go.Scatter(
            x=window_range,
            y=revenue_window_sens,
            name='Revenue',
            line={"color": '#0A5F7A', "width": 3},
            mode='lines+markers',
            hovertemplate='Window: %{x}h<br>Revenue: $%{y:,.0f}<extra></extra>'
        ))

        # Add marker for current selection
        current_rev = revenue_window_sens[window_range.index(
            state.window_hours)] if state.window_hours in window_range else 0
        if state.window_hours in window_range:
            fig_window.add_trace(go.Scatter(
                x=[state.window_hours],
                y=[current_rev],
                mode='markers',
                marker={"color": 'red', "size": 12, "symbol": 'star'},
                name='Current Selection',
                hoverinfo='skip'
            ))

        fig_window.update_layout(
            title="Revenue vs. Lookahead Window Size",
            xaxis_title="Lookahead Window (Hours)",
            yaxis_title="Revenue ($)",
            height=400,
            hovermode='x unified',
            yaxis={"fixedrange": False},
            xaxis={"fixedrange": False}
        )

        st.plotly_chart(fig_window, width="stretch")

        # Find optimal window
        best_window = window_range[revenue_window_sens.index(max(revenue_window_sens))]
        st.info(f"💡 Optimal lookahead window for this scenario appears to be **{best_window} hours**.")

    # Horizon Sensitivity (Only for MPC Strategy)
    elif state.strategy_type == "MPC (Rolling Horizon)":
        st.markdown("---")
        st.subheader(f"Optimization Horizon Sensitivity (at {state.forecast_improvement}% Improvement)")

        horizon_range, revenue_horizon_sens = run_horizon_sensitivity_analysis(
            node_data,
            state.battery_specs,
            state.forecast_improvement / 100.0
        )

        fig_horizon = go.Figure()

        fig_horizon.add_trace(go.Scatter(
            x=horizon_range,
            y=revenue_horizon_sens,
            name='Revenue',
            line={"color": '#8B5CF6', "width": 3},
            mode='lines+markers',
            hovertemplate='Horizon: %{x}h<br>Revenue: $%{y:,.0f}<extra></extra>'
        ))

        # Add marker for current selection
        current_rev = revenue_horizon_sens[horizon_range.index(
            state.horizon_hours)] if state.horizon_hours in horizon_range else 0
        if state.horizon_hours in horizon_range:
            fig_horizon.add_trace(go.Scatter(
                x=[state.horizon_hours],
                y=[current_rev],
                mode='markers',
                marker={"color": 'red', "size": 12, "symbol": 'star'},
                name='Current Selection',
                hoverinfo='skip'
            ))

        fig_horizon.update_layout(
            title="Revenue vs. Optimization Horizon",
            xaxis_title="Optimization Horizon (Hours)",
            yaxis_title="Revenue ($)",
            height=400,
            hovermode='x unified',
            yaxis={"fixedrange": False},
            xaxis={"fixedrange": False}
        )

        st.plotly_chart(fig_horizon, width="stretch")

        # Find optimal horizon
        best_horizon = horizon_range[revenue_horizon_sens.index(max(revenue_horizon_sens))]
        st.info(f"💡 Optimal horizon for this scenario appears to be **{best_horizon} hours**.")

    # ============================================================================
    # STRATEGY COMPARISON INSIGHTS
    # ============================================================================

    st.markdown("### Strategy Comparison Insights")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**Threshold-Based Strategy:**")
        # Check if monotonic
        threshold_diffs = [revenue_threshold[i + 1] - revenue_threshold[i]
                           for i in range(len(revenue_threshold) - 1)]
        negative_changes = sum(1 for d in threshold_diffs if d < 0)

        if negative_changes > 0:
            st.warning(
                f"⚠️ Non-monotonic: {negative_changes} instances where improvement REDUCED revenue")
        else:
            st.success("✓ Monotonic improvement")

        st.metric("Revenue Range",
                  f"${min(revenue_threshold):,.0f} to ${max(revenue_threshold):,.0f}")

    with col2:
        st.markdown("**Rolling Window Strategy:**")
        # Check if monotonic
        window_diffs = [revenue_rolling_window[i + 1] - revenue_rolling_window[i]
                        for i in range(len(revenue_rolling_window) - 1)]
        negative_changes_window = sum(1 for d in window_diffs if d < 0)

        if negative_changes_window > 0:
            st.warning(
                f"⚠️ Non-monotonic: {negative_changes_window} instances where improvement REDUCED revenue")
        else:
            st.success("✓ Monotonic improvement")

        st.metric("Revenue Range",
                  f"${min(revenue_rolling_window):,.0f} to ${max(revenue_rolling_window):,.0f}")

    with col3:
        st.markdown("**MPC Strategy (24h):**")
        # Check if monotonic
        mpc_diffs = [revenue_mpc[i + 1] - revenue_mpc[i] for i in range(len(revenue_mpc) - 1)]
        negative_changes_mpc = sum(1 for d in mpc_diffs if d < 0)

        if negative_changes_mpc > 0:
            st.warning(
                f"⚠️ Non-monotonic: {negative_changes_mpc} instances where improvement REDUCED revenue")
        else:
            st.success("✓ Monotonic improvement")

        st.metric("Revenue Range",
                  f"${min(revenue_mpc):,.0f} to ${max(revenue_mpc):,.0f}")

    with col4:
        st.markdown("**LP Benchmark (Theoretical):**")
        # Check if monotonic
        linear_diffs = [revenue_linear[i + 1] - revenue_linear[i]
                        for i in range(len(revenue_linear) - 1)]
        negative_changes_linear = sum(1 for d in linear_diffs if d < 0)

        if negative_changes_linear > 0:
            st.warning(
                f"⚠️ Non-monotonic: {negative_changes_linear} instances where improvement REDUCED revenue")
        else:
            st.success("✓ Monotonic improvement")

        st.metric("Revenue Range",
                  f"${min(revenue_linear):,.0f} to ${max(revenue_linear):,.0f}")
        st.caption("Upper bound for any strategy")

    # ============================================================================
    # KEY INSIGHTS
    # ============================================================================

    st.markdown("---")
    st.subheader(f"Market Insights - {get_date_range_str(node_data)}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Price Events")
        st.markdown(f"""
        **Negative Price Events:**
        - {(node_data['price_mwh_rt'] < 0).sum()} hours with negative RT prices
        - Lowest: ${node_data['price_mwh_rt'].min():.2f}/MWh
        - Batteries can charge while getting paid

        **Price Volatility:**
        - {node_data['extreme_event'].sum()} hours with >$10/MWh spread
        - Max spread: ${node_data['price_spread'].max():.2f}/MWh
        - These hours drive significant revenue opportunity

        # Forecast Performance:
        - Mean Absolute Error: ${node_data['forecast_error'].abs().mean():.2f}/MWh
        - {state.forecast_improvement}% improvement = ${(revenue_rolling_window[int(state.forecast_improvement // 10)] - revenue_rolling_window[0]):,.0f} gain (Rolling Window)
        """)

    with col2:
        st.markdown("#### Price Spread Analysis")
        fig_spread = px.scatter(
            node_data,
            x='price_mwh_da',
            y='price_mwh_rt',
            title="Day-Ahead vs Real-Time Prices",
            labels={
                'price_mwh_da': 'DA Price ($/MWh)',
                'price_mwh_rt': 'RT Price ($/MWh)'
            },
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
            line={"color": 'gray', "dash": 'dash'},
            showlegend=True
        ))

        st.plotly_chart(fig_spread, width="stretch")

    # ============================================================================
    # REVENUE CAPTURE ANALYSIS
    # ============================================================================

    st.markdown("---")
    st.subheader("LP Benchmark Analysis")

    st.info("""
    This section shows how revenue improves along the **LP Benchmark curve** as forecast accuracy increases.
    The LP benchmark represents the theoretical maximum at each improvement level.
    """)

    # Current settings revenue (using LP Benchmark as the true optimal)
    current_idx = int(state.forecast_improvement // 10)
    baseline_revenue_lp = revenue_linear[0]  # LP at 0% improvement
    current_revenue_lp = revenue_linear[current_idx]  # LP at current improvement
    max_revenue_lp = revenue_linear[-1]  # LP at 100% (perfect foresight)

    captured = current_revenue_lp - baseline_revenue_lp
    remaining = max_revenue_lp - current_revenue_lp
    total_opportunity = max_revenue_lp - baseline_revenue_lp

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "LP @ 0% (DA Only)",
            f"${baseline_revenue_lp:,.0f}",
            help="LP Benchmark revenue using only day-ahead forecasts"
        )

    with col2:
        st.metric(
            f"LP @ {state.forecast_improvement}%",
            f"${current_revenue_lp:,.0f}",
            delta=f"+${captured:,.0f}",
            help=f"LP Benchmark revenue with {state.forecast_improvement}% forecast improvement"
        )

    with col3:
        st.metric(
            "LP @ 100% (Perfect)",
            f"${max_revenue_lp:,.0f}",
            delta=f"+${remaining:,.0f} remaining",
            help="LP Benchmark at 100% (absolute theoretical maximum)"
        )

    with col4:
        capture_pct = (captured / total_opportunity * 100) if total_opportunity != 0 else 0
        st.metric(
            "LP Capture Rate",
            f"{capture_pct:.1f}%",
            help="Percentage of LP improvement potential captured at current forecast level"
        )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("💡 Navigate to other pages in the sidebar to explore decision timelines and optimization strategies.")

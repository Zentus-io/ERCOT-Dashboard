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
        *   **Optimal Asset:** The system simulates battery sizes from 0 to 1.5x Solar Capacity to find the size that maximizes total revenue (Base Solar + Battery Arbitrage of Clipped Energy).
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

    with col_right:
        # --- SIMULATION ENGINE ---
        
        # 1. Calculate Solar Vectors
        # Use Potential MW (Resource Availability) for simulation if available, else Actuals
        # The profile is already normalized 0-1
        df_sim = df_prices.copy()
        df_sim['Solar_MW'] = solar_profile['potential_mw'] * solar_capacity_mw
        df_sim['Export_MW'] = np.minimum(df_sim['Solar_MW'], interconnection_limit_mw)
        df_sim['Clipped_MW'] = np.maximum(0, df_sim['Solar_MW'] - interconnection_limit_mw)

        # ============================================================================
        # STAGE 1: HEATMAP OPTIMIZATION (Vectorized Heuristic)
        # ============================================================================

        @st.cache_data(ttl=3600)
        def run_stage1_heatmap(
            daily_clipped_mwh_arr,
            daily_max_price_arr,
            daily_min_price_arr,
            base_solar_revenue_val,
            battery_efficiency_val,
            max_power_mw,
            cache_key: str  # For cache invalidation
        ):
            """
            Stage 1: Fast 2D heatmap optimization (Power × Duration).

            Uses vectorized numpy calculations on daily aggregated statistics
            to quickly explore the search space. Returns a revenue matrix for
            heatmap visualization and the optimal configuration.

            Parameters
            ----------
            daily_clipped_mwh_arr : np.ndarray
                Daily clipped energy (MWh per day)
            daily_max_price_arr : np.ndarray
                Daily maximum RT price ($/MWh)
            daily_min_price_arr : np.ndarray
                Daily minimum RT price ($/MWh)
            base_solar_revenue_val : float
                Base solar revenue (export only, no battery)
            battery_efficiency_val : float
                Round-trip battery efficiency (0-1)
            max_power_mw : int
                Maximum power to sweep (MW)
            cache_key : str
                Hash for cache invalidation (based on inputs)

            Returns
            -------
            tuple
                (power_range, duration_range, revenue_matrix, optimal_power, optimal_duration)
            """
            # Define search space
            # Power: 20 MW steps from 20 to max_power
            power_range = np.arange(20, max_power_mw + 1, 20)
            if len(power_range) == 0:  # Handle small max_power
                power_range = np.array([max_power_mw])

            # Duration: 1h, 2h, 4h (standard battery configurations)
            duration_range = np.array([1.0, 2.0, 4.0])

            # Initialize revenue matrix (rows=power, cols=duration)
            revenue_matrix = np.zeros((len(power_range), len(duration_range)))

            # Vectorized calculation over power × duration grid
            for i, power_mw in enumerate(power_range):
                for j, duration_h in enumerate(duration_range):
                    # Battery capacity (MWh)
                    capacity_mwh = power_mw * duration_h

                    # HEURISTIC REVENUE CALCULATION (per day)
                    # 1. Clipped Energy Capture (free source)
                    vol_clipped = np.minimum(daily_clipped_mwh_arr, capacity_mwh)
                    rev_clipped = vol_clipped * daily_max_price_arr * battery_efficiency_val

                    # 2. Grid Arbitrage (remaining capacity)
                    vol_grid = np.maximum(0, capacity_mwh - vol_clipped)
                    spread = np.maximum(0, daily_max_price_arr - daily_min_price_arr)
                    rev_grid = vol_grid * spread * battery_efficiency_val

                    # Total battery revenue (sum across all days)
                    total_battery_revenue = (rev_clipped + rev_grid).sum()

                    # Total hybrid revenue (solar + battery)
                    revenue_matrix[i, j] = base_solar_revenue_val + total_battery_revenue

            # Find optimal configuration
            max_idx = np.unravel_index(np.argmax(revenue_matrix), revenue_matrix.shape)
            optimal_power = power_range[max_idx[0]]
            optimal_duration = duration_range[max_idx[1]]

            return power_range, duration_range, revenue_matrix, optimal_power, optimal_duration

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
        
        # 2. Battery Revenue (Current Asset)
        # Toggle for Full Simulation vs Heuristic
        use_full_sim = st.toggle("Run Full Simulation (Slower)", value=False, help="Run detailed dispatch simulation respecting strategy & forecast settings.")
        
        current_batt_revenue = 0.0
        sim_details = ""
        
        if use_full_sim:
            with st.spinner("Running detailed simulation..."):
                # Run simulation using the centralized runner
                # This respects the sidebar settings (Strategy, Forecast, etc.)
                baseline_res, improved_res, optimal_res, theoretical_res = run_or_get_cached_simulation()
                
                if improved_res:
                    current_batt_revenue = improved_res.total_revenue
                    sim_details = f"Strategy: {state.strategy_type} | Forecast: {state.forecast_improvement}%"
                else:
                    st.error("Simulation failed. Check data.")
        else:
            # Heuristic: Simple Arbitrage (Clipped Energy * Daily Peak Price)
            # Uses the pre-calculated daily stats
            
            daily_clipped_mwh = daily_stats['Daily_Clipped_MWh']
            daily_peak_price = daily_stats['Daily_Max_Price']
            
            batt_eff = state.battery_specs.efficiency
            
            # Revenue = Min(Clipped_MWh, Power * 1h) * Peak_Price * Efficiency
            daily_batt_revenue = np.minimum(daily_clipped_mwh, current_power) * daily_peak_price * batt_eff
            current_batt_revenue = daily_batt_revenue.sum()
            sim_details = "Heuristic: Perfect Peak Discharge (1h Duration)"

        current_hybrid_rev = df_sim['Solar_Revenue'].sum() + current_batt_revenue
        
        # --- OPTIMIZATION SWEEPER ---
        # We run this automatically now to compare
        max_batt_mw = int(solar_capacity_mw * 1.5) # Sweep up to 1.5x Solar Capacity
        step_size = 1
        sizes = range(0, max_batt_mw + 1, step_size)
        revenues = []
        
        # Base Solar Revenue (Export only)
        base_solar_revenue = df_sim['Solar_Revenue'].sum()
        
        # Extract arrays for vectorized operations in loop
        clipped_mwh = daily_stats['Daily_Clipped_MWh'].values
        max_prices = daily_stats['Daily_Max_Price'].values
        min_prices = daily_stats['Daily_Min_Price'].values
        
        # Ensure batt_eff is defined for the sweep
        batt_eff = state.battery_specs.efficiency
        
        for size_mw in sizes:
            if size_mw == 0:
                revenues.append(base_solar_revenue)
                continue
                
            # Heuristic: 
            # 1. Discharge Clipped Energy at Peak Price (Free energy source)
            # 2. Arbitrage Grid Energy for remaining capacity (Buy Min, Sell Max)
            
            # Capacity in MWh (assuming 1h duration for power constraint in this simple heuristic)
            # Real battery might have 2h or 4h, but for "Power" sweep we assume 1h or proportional
            # To make it fair with "Current Asset" which might be 2h, we should ideally use duration.
            # But this is a "Power" sweep. Let's stick to 1h for the sweep x-axis meaning.
            
            capacity_mwh = size_mw * 1.0 
            
            # 1. Clipped Capture
            vol_clipped = np.minimum(clipped_mwh, capacity_mwh)
            rev_clipped = vol_clipped * max_prices * batt_eff
            
            # 2. Grid Arbitrage (Remaining capacity)
            # We can fill the rest of the battery from the grid at Min Price
            vol_grid = np.maximum(0, capacity_mwh - vol_clipped)
            spread = max_prices - min_prices
            # Only arbitrage if spread is positive
            spread = np.maximum(0, spread)
            rev_grid = vol_grid * spread * batt_eff
            
            total_daily_rev = rev_clipped + rev_grid
            total_arb = total_daily_rev.sum()
            
            revenues.append(base_solar_revenue + total_arb)
                
        # Find Optimal
        optimal_idx = np.argmax(revenues)
        optimal_size = sizes[optimal_idx]
        optimal_revenue = revenues[optimal_idx]
        
        lost_revenue = optimal_revenue - current_hybrid_rev
            
        # --- METRICS & COMPARISON ---
        st.markdown("### 📊 Performance Comparison")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Asset Size", f"{current_power} MW")
        m2.metric("Current Revenue", f"${current_hybrid_rev:,.0f}")
        m3.metric("Optimal Size", f"{optimal_size} MW")
        m4.metric("Potential Revenue", f"${optimal_revenue:,.0f}", delta=f"+${lost_revenue:,.0f}")
        
        if lost_revenue > 0:
            st.info(f"💡 Increasing battery size to **{optimal_size} MW** could capture **${lost_revenue:,.0f}** in additional revenue from clipped energy.")
        else:
            st.success("✅ Current asset is optimally sized for clipping capture!")

        # --- HEATMAP VIEW (STAGE 1) ---
        st.markdown("---")
        show_heatmap = st.toggle("Show 2D Heatmap (Power × Duration)", value=False,
                                  help="Visualize revenue across power and duration configurations using fast heuristic.")

        if show_heatmap:
            # Generate cache key for invalidation
            cache_inputs = f"{solar_capacity_mw}_{interconnection_limit_mw}_{batt_eff}_{df_prices.index.min()}_{df_prices.index.max()}"
            cache_key = hashlib.md5(cache_inputs.encode()).hexdigest()

            # Run Stage 1 heatmap optimization
            max_power_sweep = int(solar_capacity_mw * 1.5)
            power_range_s1, duration_range_s1, revenue_matrix_s1, optimal_power_s1, optimal_duration_s1 = run_stage1_heatmap(
                daily_clipped_mwh_arr=daily_stats['Daily_Clipped_MWh'].values,
                daily_max_price_arr=daily_stats['Daily_Max_Price'].values,
                daily_min_price_arr=daily_stats['Daily_Min_Price'].values,
                base_solar_revenue_val=base_solar_revenue,
                battery_efficiency_val=batt_eff,
                max_power_mw=max_power_sweep,
                cache_key=cache_key
            )

            # Display Stage 1 optimal configuration
            st.markdown("### 🗺️ Stage 1: Power × Duration Heatmap")
            col_h1, col_h2, col_h3 = st.columns(3)
            col_h1.metric("Optimal Power (Stage 1)", f"{optimal_power_s1} MW")
            col_h2.metric("Optimal Duration (Stage 1)", f"{optimal_duration_s1:.1f}h")
            optimal_capacity_s1 = optimal_power_s1 * optimal_duration_s1
            col_h3.metric("Optimal Capacity (Stage 1)", f"{optimal_capacity_s1:.1f} MWh")

            # Create heatmap visualization
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=revenue_matrix_s1,
                x=duration_range_s1,
                y=power_range_s1,
                colorscale='Viridis',
                colorbar=dict(title="Revenue ($)"),
                hovertemplate='Power: %{y} MW<br>Duration: %{x}h<br>Revenue: $%{z:,.0f}<extra></extra>'
            ))

            # Mark optimal point with a star
            optimal_idx_i = np.where(power_range_s1 == optimal_power_s1)[0][0]
            optimal_idx_j = np.where(duration_range_s1 == optimal_duration_s1)[0][0]

            fig_heatmap.add_trace(go.Scatter(
                x=[optimal_duration_s1],
                y=[optimal_power_s1],
                mode='markers+text',
                marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='white')),
                text=["Optimal"],
                textposition="top center",
                textfont=dict(color='white', size=12),
                name='Stage 1 Optimal',
                hoverinfo='skip'
            ))

            fig_heatmap.update_layout(
                title="Revenue Heatmap: Battery Power × Duration",
                xaxis_title="Duration (hours)",
                yaxis_title="Power (MW)",
                height=500,
                xaxis=dict(type='category'),  # Categorical for discrete durations
            )

            st.plotly_chart(fig_heatmap, use_container_width=True)

            st.caption(f"💡 Heatmap shows revenue across {len(power_range_s1)}×{len(duration_range_s1)} = {len(power_range_s1) * len(duration_range_s1)} configurations using vectorized heuristic (fast).")

            # ============================================================================
            # STAGE 2: FINE REFINEMENT (Full Simulation with Hybrid Strategy)
            # ============================================================================

            st.markdown("---")
            run_stage2 = st.toggle("Run Stage 2: Fine Refinement (Full Simulation)", value=False,
                                    help="Refine around Stage 1 optimal using full BatterySimulator with HybridDispatchStrategy. Uses parallel execution (~2-3s).")

            if run_stage2:
                st.markdown("### 🎯 Stage 2: Fine Refinement with Full Simulation")

                # Define refined search space around Stage 1 optimal
                power_min_s2 = max(10, optimal_power_s1 - 20)  # ±20 MW around optimal
                power_max_s2 = optimal_power_s1 + 20
                power_range_s2 = np.arange(power_min_s2, power_max_s2 + 1, 2)  # 2 MW steps

                # Use optimal duration from Stage 1 (fixed for Stage 2)
                fixed_duration_s2 = optimal_duration_s1

                # Prepare price data with clipped_mw column for hybrid simulation
                df_hybrid = df_sim.copy()
                df_hybrid['clipped_mw'] = df_hybrid['Clipped_MW']

                # Get improvement factor from state
                improvement_factor = state.forecast_improvement / 100.0

                # Simulation function for parallel execution
                def simulate_battery_config(power_mw: float, duration_h: float) -> dict:
                    """
                    Run full battery simulation for a specific configuration using HybridDispatchStrategy.

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
                        initial_soc=0.5
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

                    # Wrap with HybridDispatchStrategy
                    hybrid_strategy = HybridDispatchStrategy(
                        base_strategy=base_strategy,
                        clipped_priority=True
                    )

                    # Run simulation
                    simulator = BatterySimulator(specs)
                    result = simulator.run(
                        df_hybrid,
                        hybrid_strategy,
                        improvement_factor=improvement_factor
                    )

                    # Calculate total hybrid revenue (solar + battery)
                    total_hybrid_revenue = base_solar_revenue + result.total_revenue

                    # Extract metadata
                    metadata = hybrid_strategy.get_metadata()

                    return {
                        'power_mw': power_mw,
                        'duration_h': duration_h,
                        'capacity_mwh': capacity_mwh,
                        'battery_revenue': result.total_revenue,
                        'total_revenue': total_hybrid_revenue,
                        'clipped_captured': metadata.get('clipped_energy_captured', 0),
                        'grid_arbitrage': metadata.get('grid_arbitrage_revenue', 0)
                    }

                # Run simulations in parallel
                st.info(f"🔄 Running {len(power_range_s2)} simulations in parallel (Power: {power_min_s2}-{power_max_s2} MW, Duration: {fixed_duration_s2:.1f}h)...")

                results_s2 = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                with ThreadPoolExecutor(max_workers=4) as executor:
                    # Submit all simulations
                    futures = {
                        executor.submit(simulate_battery_config, p, fixed_duration_s2): p
                        for p in power_range_s2
                    }

                    # Collect results as they complete
                    completed = 0
                    for future in futures:
                        result = future.result()
                        results_s2.append(result)
                        completed += 1
                        progress_bar.progress(completed / len(power_range_s2))
                        status_text.text(f"Completed: {completed}/{len(power_range_s2)} simulations")

                progress_bar.empty()
                status_text.empty()

                # Sort results by power for plotting
                results_s2 = sorted(results_s2, key=lambda x: x['power_mw'])

                # Extract data for visualization
                powers_s2 = [r['power_mw'] for r in results_s2]
                revenues_s2 = [r['total_revenue'] for r in results_s2]

                # Find Stage 2 optimal
                optimal_idx_s2 = np.argmax(revenues_s2)
                optimal_power_s2 = powers_s2[optimal_idx_s2]
                optimal_revenue_s2 = revenues_s2[optimal_idx_s2]
                optimal_capacity_s2 = optimal_power_s2 * fixed_duration_s2

                # Display Stage 2 optimal configuration
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                col_s1.metric("Optimal Power (Stage 2)", f"{optimal_power_s2:.0f} MW")
                col_s2.metric("Optimal Capacity (Stage 2)", f"{optimal_capacity_s2:.1f} MWh")
                col_s3.metric("Optimal Revenue (Stage 2)", f"${optimal_revenue_s2:,.0f}")

                # Compare with current asset
                revenue_improvement_s2 = optimal_revenue_s2 - current_hybrid_rev
                col_s4.metric("vs Current Asset", f"+${revenue_improvement_s2:,.0f}",
                            delta_color="normal" if revenue_improvement_s2 >= 0 else "inverse")

                # Create refinement curve visualization
                fig_refinement = go.Figure()

                # Stage 2 refinement curve
                fig_refinement.add_trace(go.Scatter(
                    x=powers_s2,
                    y=revenues_s2,
                    mode='lines+markers',
                    name='Stage 2 (Full Simulation)',
                    line=dict(color='#0A5F7A', width=3),
                    marker=dict(size=6),
                    hovertemplate='Power: %{x:.0f} MW<br>Revenue: $%{y:,.0f}<extra></extra>'
                ))

                # Current asset marker
                fig_refinement.add_trace(go.Scatter(
                    x=[current_power],
                    y=[current_hybrid_rev],
                    mode='markers+text',
                    marker=dict(color='red', size=15, symbol='x', line=dict(width=2, color='darkred')),
                    text=["Current"],
                    textposition="top center",
                    name='Current Asset',
                    hovertemplate='Current Asset<br>Power: %{x:.0f} MW<br>Revenue: $%{y:,.0f}<extra></extra>'
                ))

                # Stage 1 optimal marker
                fig_refinement.add_trace(go.Scatter(
                    x=[optimal_power_s1],
                    y=[revenue_matrix_s1[optimal_idx_i, optimal_idx_j]],
                    mode='markers+text',
                    marker=dict(color='orange', size=15, symbol='star', line=dict(width=2, color='darkorange')),
                    text=["Stage 1"],
                    textposition="bottom center",
                    name='Stage 1 Optimal',
                    hovertemplate='Stage 1 Optimal<br>Power: %{x:.0f} MW<br>Revenue: $%{y:,.0f}<extra></extra>'
                ))

                # Stage 2 optimal marker
                fig_refinement.add_trace(go.Scatter(
                    x=[optimal_power_s2],
                    y=[optimal_revenue_s2],
                    mode='markers+text',
                    marker=dict(color='green', size=18, symbol='star', line=dict(width=2, color='darkgreen')),
                    text=["Stage 2 Optimal"],
                    textposition="top center",
                    name='Stage 2 Optimal',
                    hovertemplate='Stage 2 Optimal<br>Power: %{x:.0f} MW<br>Revenue: $%{y:,.0f}<extra></extra>'
                ))

                fig_refinement.update_layout(
                    title=f"Stage 2 Refinement: Power Optimization at {fixed_duration_s2:.1f}h Duration",
                    xaxis_title="Battery Power (MW)",
                    yaxis_title="Total Hybrid Revenue ($)",
                    height=500,
                    hovermode='closest',
                    legend=dict(
                        yanchor="bottom",
                        y=0.01,
                        xanchor="right",
                        x=0.99,
                        bgcolor="rgba(255, 255, 255, 0.8)"
                    )
                )

                st.plotly_chart(fig_refinement, use_container_width=True)

                # Display insights
                st.success(f"✅ Stage 2 complete! Evaluated {len(results_s2)} configurations using full hybrid simulation.")

                if revenue_improvement_s2 > 0:
                    st.info(f"💡 **Recommended**: Upgrade to **{optimal_power_s2:.0f} MW / {optimal_capacity_s2:.1f} MWh** battery to capture **${revenue_improvement_s2:,.0f}** additional revenue.")
                else:
                    st.success("🎉 Current asset configuration is already optimal!")

                # Show detailed breakdown for optimal config
                with st.expander("📊 Detailed Breakdown (Stage 2 Optimal)"):
                    optimal_result = results_s2[optimal_idx_s2]

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Clipped Energy Captured", f"{optimal_result['clipped_captured']:.1f} MWh")
                    col2.metric("Grid Arbitrage Revenue", f"${optimal_result['grid_arbitrage']:,.0f}")
                    col3.metric("Battery Revenue", f"${optimal_result['battery_revenue']:,.0f}")

                    st.markdown(f"""
                    **Revenue Components:**
                    - Base Solar Export: ${base_solar_revenue:,.0f}
                    - Battery Operations: ${optimal_result['battery_revenue']:,.0f}
                        - Clipped Energy Capture: {optimal_result['clipped_captured']:.1f} MWh
                        - Grid Arbitrage: ${optimal_result['grid_arbitrage']:,.0f}
                    - **Total Hybrid Revenue**: ${optimal_result['total_revenue']:,.0f}

                    **Configuration:**
                    - Strategy: Hybrid ({state.strategy_type})
                    - Forecast Improvement: {state.forecast_improvement}%
                    - Battery Efficiency: {batt_eff*100:.1f}%
                    """)

        # --- VISUALIZATION ---
        
        # 1. Revenue Curve
        fig_sweep = px.line(
            x=sizes, 
            y=revenues,
            title="Asset Sizing Revenue Optimization",
            labels={'x': 'Battery Power (MW)', 'y': 'Total Revenue ($)'}
        )
        # Add Current Asset Marker
        fig_sweep.add_trace(go.Scatter(
            x=[current_power], y=[current_hybrid_rev],
            mode='markers', marker=dict(color='red', size=12, symbol='x'),
            name='Current Asset'
        ))
        # Add Optimal Marker
        fig_sweep.add_trace(go.Scatter(
            x=[optimal_size], y=[optimal_revenue],
            mode='markers', marker=dict(color='green', size=12, symbol='star'),
            name='Optimal Asset'
        ))
        
        fig_sweep.update_layout(
            legend=dict(
                yanchor="bottom",
                y=0.01,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255, 255, 255, 0.5)"
            )
        )
        
        st.plotly_chart(fig_sweep, width="stretch")
        
        # 2. Operational Chart (Representative Week)
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

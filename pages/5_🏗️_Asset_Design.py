"""
Hybrid Asset Design & Optimization Page
Zentus - ERCOT Battery Revenue Dashboard

This page allows users to:
1. Design hybrid assets (Solar + Storage) using real market data and generation profiles.
2. Compare the currently selected asset against an optimal configuration.
3. Visualize clipping capture and revenue potential.
"""

from concurrent.futures import ThreadPoolExecutor
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
from core.data.loaders import SupabaseDataLoader
from ui.components.header import render_header
from ui.components.sidebar import render_sidebar
from ui.styles.custom_css import apply_custom_styles
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
            Priorities:
            1. Uploaded CSV
            2. Database (ercot_generation table)
            3. Synthetic Bell Curve
            """
            target_index = _df_prices.index
            start_date = target_index.min().date()
            end_date = target_index.max().date()
            
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
                            
                        return pd.Series(aligned_values, index=target_index), "Uploaded CSV"
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
                        aligned = df_gen['gen_mw'].reindex(target_index).interpolate(method='time').fillna(0)
                        
                        # Normalize
                        mx = aligned.max()
                        if mx > 0:
                            aligned = aligned / mx
                            
                        return aligned, "Database (Regional Forecast)"
                except Exception as e:
                    # st.warning(f"DB Fetch Error: {e}")
                    pass

            # 3. Fallback: Synthetic Bell Curve
            hours = np.linspace(0, 24, 96)
            daily_profile = np.exp(-((hours - 12)**2) / (2 * 3**2))
            daily_profile[daily_profile < 0.01] = 0
            
            days = int(np.ceil(len(target_index) / 96))
            full_profile = np.tile(daily_profile, days)[:len(target_index)]
            
            return pd.Series(full_profile, index=target_index), "Synthetic (Bell Curve)"

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
        df_sim = df_prices.copy()
        df_sim['Solar_MW'] = solar_profile * solar_capacity_mw
        df_sim['Export_MW'] = np.minimum(df_sim['Solar_MW'], interconnection_limit_mw)
        df_sim['Clipped_MW'] = np.maximum(0, df_sim['Solar_MW'] - interconnection_limit_mw)
        
        # 2. Base Solar Revenue
        # RTM Price is usually 15-min, so divide by 4 for MWh
        df_sim['Solar_Revenue'] = (df_sim['Export_MW'] * df_sim['price_mwh_rt']) / 4
        
        # 3. Hybrid Arbitrage Simulation (Vectorized)
        # Strategy: Capture clipped energy, discharge at daily peak price
        df_sim['Date'] = df_sim.index.date
        
        daily_stats = df_sim.groupby('Date').agg({
            'Clipped_MW': 'sum', # Sum of MW per interval
            'price_mwh_rt': 'max'
        }).rename(columns={'Clipped_MW': 'Daily_Clipped_Sum_MW', 'price_mwh_rt': 'Daily_Max_Price'})
        
        daily_stats['Daily_Clipped_MWh'] = daily_stats['Daily_Clipped_Sum_MW'] / 4
        
        batt_eff = state.battery_specs.efficiency
        
        # --- OPTIMIZATION SWEEPER ---
        # We run this automatically now to compare
        max_batt_mw = max(200, int(solar_capacity_mw)) # Sweep up to Solar Capacity
        step_size = 10
        sizes = range(0, max_batt_mw + 1, step_size)
        revenues = []
        
        # Pre-calculate for speed
        clipped_mwh = daily_stats['Daily_Clipped_MWh'].values
        max_prices = daily_stats['Daily_Max_Price'].values
        
        for size_mw in sizes:
            # Heuristic: Discharge at Peak Price limited by Power (1 hour duration assumption for peak)
            # Volume = min(Daily_Clipped_MWh, Size_MW * 1h)
            # Revenue = Volume * Max_Price * Efficiency
            daily_rev = np.minimum(clipped_mwh, size_mw * 1.0) * max_prices * batt_eff
            total_arb = daily_rev.sum()
            revenues.append(df_sim['Solar_Revenue'].sum() + total_arb)
            
        # Find Optimal
        optimal_idx = np.argmax(revenues)
        optimal_size = sizes[optimal_idx]
        optimal_revenue = revenues[optimal_idx]
        
        # Calculate Current Asset Performance
        current_daily_rev = np.minimum(clipped_mwh, current_power * 1.0) * max_prices * batt_eff
        current_hybrid_rev = df_sim['Solar_Revenue'].sum() + current_daily_rev.sum()
        
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

        # --- VISUALIZATION ---
        
        # 1. Revenue Curve
        fig_sweep = px.line(
            x=sizes, 
            y=revenues,
            title="Revenue Optimization Curve",
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
        
        st.plotly_chart(fig_sweep, width="stretch")
        
        # 2. Operational Chart (Representative Week)
        st.markdown("### ⚡ Operational Profile (Solar + Clipping)")
        plot_df = df_sim.copy()
        if len(plot_df) > 96 * 7:
            plot_df = plot_df.iloc[:96*7]
            st.caption(f"Visualizing first 7 days.")
            
        fig_hybrid = go.Figure()
        
        fig_hybrid.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df['Export_MW'],
            mode='lines', name='Solar Export',
            fill='tozeroy', line=dict(color='#FFD700', width=0), stackgroup='one'
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

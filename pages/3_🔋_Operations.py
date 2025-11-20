"""
Battery Operations Page
Zentus - ERCOT Battery Revenue Dashboard

This page provides detailed battery operations analysis including
SOC trajectories and dispatch action distributions.
"""

import streamlit as st
from config.page_config import configure_page
from ui.styles.custom_css import apply_custom_styles
from ui.components.header import render_header
from ui.components.sidebar import render_sidebar
from utils.state import get_state, has_valid_config
from core.battery.simulator import BatterySimulator
from core.battery.strategies import ThresholdStrategy, RollingWindowStrategy
from core.data.loaders import DataLoader
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

configure_page("Battery Operations")
apply_custom_styles()

# ============================================================================
# HEADER AND SIDEBAR
# ============================================================================

render_header()
render_sidebar()

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.header("🔋 Battery Operations")

# Check if configuration is valid
if not has_valid_config():
    st.warning("⚠️ Please configure battery specifications and select a settlement point in the sidebar to begin analysis.")
    st.stop()

# Get state
state = get_state()

if state.price_data is None:
    st.error("⚠️ Price data not loaded. Please refresh the page or check data availability.")
    st.stop()

if state.selected_node is None:
    st.error("⚠️ No settlement point selected. Please select a node in the sidebar.")
    st.stop()

# Load node data
loader = DataLoader(Path(__file__).parent.parent / 'data')
node_data = loader.filter_by_node(state.price_data, state.selected_node)

# Check if battery specs are configured
if state.battery_specs is None:
    st.error("⚠️ Battery specifications not configured. Please configure in the sidebar.")
    st.stop()

# ============================================================================
# RUN SIMULATIONS
# ============================================================================

# ============================================================================
# RUN SIMULATIONS
# ============================================================================

from utils.simulation_runner import run_or_get_cached_simulation

with st.spinner('Running battery simulations...'):
    baseline_result, improved_result, optimal_result = run_or_get_cached_simulation()
    
    if baseline_result is None:
        st.error("⚠️ Failed to run simulations. Please check data availability.")
        st.stop()

# ============================================================================
# STATE OF CHARGE CHART
# ============================================================================

st.subheader("Battery State of Charge")

fig_soc = go.Figure()

# Create sorted dataframes for each scenario to ensure chronological plotting
# Optimal strategy
optimal_soc_df = pd.DataFrame({
    'timestamp': optimal_result.soc_timestamps,
    'soc': optimal_result.soc_values
}).sort_values('timestamp')

fig_soc.add_trace(go.Scatter(
    x=optimal_soc_df['timestamp'],
    y=optimal_soc_df['soc'],
    name='Optimal Strategy (Perfect Foresight)',
    line=dict(color='#28A745', width=2.5),
    hovertemplate='SOC: %{y:.1f} MWh<extra></extra>',
    mode='lines'
))

# Improved forecast
improved_soc_df = pd.DataFrame({
    'timestamp': improved_result.soc_timestamps,
    'soc': improved_result.soc_values
}).sort_values('timestamp')

fig_soc.add_trace(go.Scatter(
    x=improved_soc_df['timestamp'],
    y=improved_soc_df['soc'],
    name=f'Improved Forecast (+{state.forecast_improvement}%)',
    line=dict(color='#FFC107', width=2),
    hovertemplate='SOC: %{y:.1f} MWh<extra></extra>',
    mode='lines'
))

# Baseline
baseline_soc_df = pd.DataFrame({
    'timestamp': baseline_result.soc_timestamps,
    'soc': baseline_result.soc_values
}).sort_values('timestamp')

fig_soc.add_trace(go.Scatter(
    x=baseline_soc_df['timestamp'],
    y=baseline_soc_df['soc'],
    name='Baseline (Day-Ahead Only)',
    line=dict(color='#DC3545', width=2, dash='dash'),
    hovertemplate='SOC: %{y:.1f} MWh<extra></extra>',
    mode='lines'
))

fig_soc.add_hline(
    y=state.battery_specs.capacity_mwh,
    line_dash="dot",
    annotation_text=f"Max Capacity ({state.battery_specs.capacity_mwh} MWh)"
)
fig_soc.add_hline(
    y=0,
    line_dash="dot",
    annotation_text="Empty"
)

fig_soc.update_layout(
    title="Battery State of Charge Over Time",
    xaxis_title="Time",
    yaxis_title="State of Charge (MWh)",
    height=500,
    hovermode='x unified'
)

from ui.components.charts import apply_standard_chart_styling
apply_standard_chart_styling(fig_soc)

st.plotly_chart(fig_soc, width="stretch")

# ============================================================================
# DISPATCH ACTIONS COMPARISON
# ============================================================================

st.subheader("Dispatch Action Distribution")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Baseline Dispatch Actions")
    dispatch_counts_baseline = baseline_result.dispatch_df['dispatch'].value_counts()
    fig_dispatch_baseline = px.pie(
        values=dispatch_counts_baseline.values,
        names=dispatch_counts_baseline.index,
        title="Distribution of Actions (Baseline)",
        color_discrete_map={
            'charge': '#28A745',
            'discharge': '#DC3545',
            'hold': '#6C757D'
        }
    )
    st.plotly_chart(fig_dispatch_baseline, width="stretch")

with col2:
    st.markdown("### Optimal Dispatch Actions")
    dispatch_counts_optimal = optimal_result.dispatch_df['dispatch'].value_counts()
    fig_dispatch_optimal = px.pie(
        values=dispatch_counts_optimal.values,
        names=dispatch_counts_optimal.index,
        title="Distribution of Actions (Optimal)",
        color_discrete_map={
            'charge': '#28A745',
            'discharge': '#DC3545',
            'hold': '#6C757D'
        }
    )
    st.plotly_chart(fig_dispatch_optimal, width="stretch")

# ============================================================================
# OPERATIONS SUMMARY
# ============================================================================

st.markdown("---")
st.subheader("Operations Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Baseline (DA Only)**")
    st.metric("Charge Events", baseline_result.charge_count)
    st.metric("Discharge Events", baseline_result.discharge_count)
    st.metric("Hold Periods", baseline_result.hold_count)
    utilization = ((baseline_result.charge_count + baseline_result.discharge_count) / len(node_data)) * 100
    st.metric("Utilization Rate", f"{utilization:.1f}%")

with col2:
    st.markdown(f"**Improved (+{state.forecast_improvement}%)**")
    st.metric("Charge Events", improved_result.charge_count)
    st.metric("Discharge Events", improved_result.discharge_count)
    st.metric("Hold Periods", improved_result.hold_count)
    utilization = ((improved_result.charge_count + improved_result.discharge_count) / len(node_data)) * 100
    st.metric("Utilization Rate", f"{utilization:.1f}%")

with col3:
    st.markdown("**Optimal (Perfect)**")
    st.metric("Charge Events", optimal_result.charge_count)
    st.metric("Discharge Events", optimal_result.discharge_count)
    st.metric("Hold Periods", optimal_result.hold_count)
    utilization = ((optimal_result.charge_count + optimal_result.discharge_count) / len(node_data)) * 100
    st.metric("Utilization Rate", f"{utilization:.1f}%")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("💡 Navigate to other pages in the sidebar to explore revenue tracking, opportunity analysis, and optimization strategies.")

"""
Example: Integrating Curtailment Elimination into Asset Design Page

This shows how to add the second optimization objective to your page.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from core.analytics.curtailment_optimizer import (
    summarize_curtailment_elimination,
    calculate_curtailment_frontier,
    calculate_capex_optimized_specs
)

# ============================================================================
# EXAMPLE INTEGRATION (Add after line 729 in your Asset Design page)
# ============================================================================

def add_curtailment_elimination_mode(df_sim, opt_col_left, opt_col_right):
    """
    Add curtailment elimination optimization mode.
    
    Parameters
    ----------
    df_sim : pd.DataFrame
        Simulation data with columns: Clipped_MW, price_mwh_rt, index as datetime
    opt_col_left : streamlit column
        Left column for controls
    opt_col_right : streamlit column
        Right column for visualization
    """
    
    with opt_col_left:
        st.markdown("#### ⚙️ Configuration")
        st.markdown("""
        **Curtailment Elimination**: Find minimum battery specs to capture 100% of clipped energy.
        """)
        
        # Analysis method selector
        analysis_method = st.selectbox(
            "Analysis Method",
            ["Rolling Balance (Accurate)", "Simple Daily Max (Conservative)"],
            help="Rolling Balance simulates discharge opportunities. Simple uses max daily clipping."
        )
        
        # CAPEX inputs (optional)
        with st.expander("💰 CAPEX Optimization (Optional)", expanded=False):
            enable_capex = st.checkbox("Optimize for minimum cost")
            
            if enable_capex:
                power_cost = st.number_input(
                    "Power Cost ($/MW)", 
                    value=200000, 
                    step=10000,
                    help="Cost per MW of power capacity"
                )
                energy_cost = st.number_input(
                    "Energy Cost ($/MWh)", 
                    value=300000, 
                    step=10000,
                    help="Cost per MWh of energy capacity"
                )
                max_curtailment = st.slider(
                    "Max Acceptable Curtailment (%)",
                    0.0, 10.0, 1.0, 0.5,
                    help="Maximum acceptable curtailment percentage"
                )
        
        # Run analysis button
        run_analysis = st.button("🔍 Analyze Curtailment Elimination", type="primary", width='stretch')
    
    # ========================================================================
    # ANALYSIS EXECUTION
    # ========================================================================
    
    if run_analysis:
        with st.spinner("Analyzing minimum battery specs..."):
            # Get efficiency from state
            efficiency = 0.95  # Or get from state.battery_specs.efficiency
            
            # Run comprehensive analysis
            analysis = summarize_curtailment_elimination(
                df_clipping=df_sim['Clipped_MW'],
                df_prices=df_sim['price_mwh_rt'],
                df_index=df_sim.index,
                efficiency=efficiency
            )
            
            # Display results in right column
            with opt_col_right:
                st.markdown("### 🎯 Minimum Battery Specifications")
                
                # Method-specific results
                if "Simple" in analysis_method:
                    capacity = analysis['min_capacity_simple_mwh']
                    method_note = "Conservative (ignores discharge opportunities)"
                else:
                    capacity = analysis['min_capacity_rolling_mwh']
                    method_note = f"Discharge threshold: ${analysis['discharge_threshold_price']:.1f}/MWh"
                
                power = analysis['min_power_mw']
                duration = capacity / max(power, 1)
                
                # Main metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Min Power", f"{power:.1f} MW", 
                           help="99th percentile clipping rate")
                col2.metric("Min Capacity", f"{capacity:.1f} MWh",
                           help=method_note)
                col3.metric("Min Duration", f"{duration:.1f}h")
                col4.metric("Total Size", f"{power * duration:.0f} MWh")
                
                st.caption(f"📊 Method: {analysis_method} | {method_note}")
                
                # ============================================================
                # POWER/CAPACITY TRADE-OFF VISUALIZATION
                # ============================================================
                
                st.markdown("---")
                st.markdown("### ⚖️ Power vs. Capacity Trade-offs")
                
                # Calculate frontier
                power_range = [power * x for x in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]]
                duration_range = [duration * x for x in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]]
                
                frontier = calculate_curtailment_frontier(
                    df_sim['Clipped_MW'],
                    df_sim['price_mwh_rt'],
                    efficiency,
                    power_range=power_range,
                    duration_range=duration_range
                )
                
                # Create trade-off scatter plot
                fig = go.Figure()
                
                # Color by curtailment percentage
                fig.add_trace(go.Scatter(
                    x=frontier['power_mw'],
                    y=frontier['duration_h'],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=frontier['curtailment_pct'],
                        colorscale='RdYlGn_r',  # Red = high curtailment, Green = low
                        colorbar=dict(title="Curtailment %"),
                        line=dict(width=1, color='white')
                    ),
                    text=[
                        f"Power: {row['power_mw']:.1f} MW<br>" +
                        f"Duration: {row['duration_h']:.1f}h<br>" +
                        f"Capacity: {row['capacity_mwh']:.1f} MWh<br>" +
                        f"Curtailment: {row['curtailment_pct']:.2f}%"
                        for _, row in frontier.iterrows()
                    ],
                    hovertemplate='%{text}<extra></extra>'
                ))
                
                # Highlight minimum specs point
                fig.add_trace(go.Scatter(
                    x=[power],
                    y=[duration],
                    mode='markers',
                    marker=dict(
                        size=20,
                        color='gold',
                        symbol='star',
                        line=dict(width=2, color='white')
                    ),
                    name='Recommended Min',
                    hovertemplate=f'Minimum Specs<br>{power:.1f} MW × {duration:.1f}h<extra></extra>'
                ))
                
                fig.update_layout(
                    title="Battery Sizing for Curtailment Elimination",
                    xaxis_title="Power (MW)",
                    yaxis_title="Duration (hours)",
                    hovermode='closest',
                    height=450
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # ============================================================
                # CAPEX OPTIMIZATION (if enabled)
                # ============================================================
                
                if enable_capex:
                    st.markdown("---")
                    st.markdown("### 💰 CAPEX-Optimized Configuration")
                    
                    optimal = calculate_capex_optimized_specs(
                        frontier,
                        power_cost,
                        energy_cost,
                        max_curtailment
                    )
                    
                    if optimal['power_mw'] is not None:
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Power", f"{optimal['power_mw']:.1f} MW")
                        col2.metric("Duration", f"{optimal['duration_h']:.1f}h")
                        col3.metric("Capacity", f"{optimal['capacity_mwh']:.1f} MWh")
                        col4.metric("Total CAPEX", f"${optimal['capex']/1e6:.2f}M")
                        
                        st.caption(f"✅ Curtailment: {optimal['curtailment_pct']:.2f}% (target: ≤{max_curtailment}%)")
                        
                        # Show cost breakdown
                        power_capex = optimal['power_mw'] * power_cost
                        energy_capex = optimal['capacity_mwh'] * energy_cost
                        
                        st.markdown(f"""
**Cost Breakdown:**
- Power ({optimal['power_mw']:.1f} MW @ ${power_cost/1000:.0f}k/MW): ${power_capex/1e6:.2f}M
- Energy ({optimal['capacity_mwh']:.1f} MWh @ ${energy_cost/1000:.0f}k/MWh): ${energy_capex/1e6:.2f}M
- **Total**: ${optimal['capex']/1e6:.2f}M
                        """)
                    else:
                        st.warning(optimal['error'])
                
                # ============================================================
                # COMPARISON TABLE
                # ============================================================
                
                st.markdown("---")
                st.markdown("### 📊 Configuration Comparison")
                
                comparison_df = pd.DataFrame([
                    {
                        'Configuration': 'Minimum (Zero Curtailment)',
                        'Power (MW)': f"{power:.1f}",
                        'Duration (h)': f"{duration:.1f}",
                        'Capacity (MWh)': f"{capacity:.1f}",
                        'Curtailment': '~0%',
                        'Use Case': 'Maximize clipping capture'
                    },
                    {
                        'Configuration': 'Current Asset',
                        'Power (MW)': f"{st.session_state.get('current_power', 0):.1f}",
                        'Duration (h)': f"{st.session_state.get('current_duration', 0):.1f}",
                        'Capacity (MWh)': f"{st.session_state.get('current_capacity', 0):.1f}",
                        'Curtailment': 'TBD',
                        'Use Case': 'Current setup'
                    }
                ])
                
                if enable_capex and optimal['power_mw'] is not None:
                    comparison_df = pd.concat([comparison_df, pd.DataFrame([{
                        'Configuration': 'CAPEX-Optimized',
                        'Power (MW)': f"{optimal['power_mw']:.1f}",
                        'Duration (h)': f"{optimal['duration_h']:.1f}",
                        'Capacity (MWh)': f"{optimal['capacity_mwh']:.1f}",
                        'Curtailment': f"{optimal['curtailment_pct']:.2f}%",
                        'Use Case': f'Min cost @ ≤{max_curtailment}% curtailment'
                    }])], ignore_index=True)
                
                st.dataframe(comparison_df, width='stretch', hide_index=True)


# ============================================================================
# USAGE IN YOUR ASSET DESIGN PAGE
# ============================================================================

"""
In your pages/5_🏗️_Asset_Design.py, after line 729:

    st.markdown("### 🔋 Battery Sizing Optimization")
    
    # Add objective selector
    optimization_objective = st.radio(
        "Optimization Objective",
        ["Revenue Maximization", "Curtailment Elimination"],
        horizontal=True,
        help="Revenue Max: Find best ROI. Curtailment Elimination: Min specs for zero waste."
    )
    
    # Create columns
    opt_col_left, opt_col_right = st.columns([1, 3])
    
    if optimization_objective == "Revenue Maximization":
        # Your existing grid search code here
        with opt_col_left:
            # ... existing controls ...
        with opt_col_right:
            # ... existing heatmap ...
    
    else:  # Curtailment Elimination
        add_curtailment_elimination_mode(df_sim, opt_col_left, opt_col_right)
"""

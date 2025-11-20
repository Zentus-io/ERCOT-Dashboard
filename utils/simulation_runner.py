import streamlit as st
from utils.state import get_state
from core.battery.simulator import BatterySimulator
from core.battery.strategies import ThresholdStrategy, RollingWindowStrategy

def run_or_get_cached_simulation():
    """
    Run simulations if not cached, or return cached results.
    
    Returns
    -------
    tuple
        (baseline_result, improved_result, optimal_result)
    """
    state = get_state()
    
    # Check if we have valid cached results
    if (state.simulation_results['baseline'] is not None and 
        state.simulation_results['improved'] is not None and 
        state.simulation_results['optimal'] is not None):
        return (
            state.simulation_results['baseline'],
            state.simulation_results['improved'],
            state.simulation_results['optimal']
        )
        
    # If not cached, run simulations
    # We need node_data for this. 
    # Ideally this function should be called where node_data is available or we fetch it here.
    # To avoid circular dependency or complex data fetching here, let's assume the caller 
    # might want to handle data loading, but for convenience we can try to load it if state has it.
    
    if state.price_data is None or state.selected_node is None:
        return None, None, None
        
    # Filter data (logic duplicated from pages, but necessary for centralized runner)
    # We can't easily import DataLoader here without potentially causing issues, 
    # but we can do the filtering manually since it's just a pandas operation
    node_data = state.price_data[state.price_data['node'] == state.selected_node].copy()
    
    if node_data.empty:
        return None, None, None

    simulator = BatterySimulator(state.battery_specs)

    # Select strategy
    if state.strategy_type == "Rolling Window Optimization":
        strategy_baseline = RollingWindowStrategy(state.window_hours)
        strategy_improved = RollingWindowStrategy(state.window_hours)
        strategy_optimal = RollingWindowStrategy(state.window_hours)
    else:  # Threshold-Based
        strategy_baseline = ThresholdStrategy(state.charge_percentile, state.discharge_percentile)
        strategy_improved = ThresholdStrategy(state.charge_percentile, state.discharge_percentile)
        strategy_optimal = ThresholdStrategy(state.charge_percentile, state.discharge_percentile)

    # Run simulations
    baseline_result = simulator.run(node_data, strategy_baseline, improvement_factor=0.0)
    improved_result = simulator.run(node_data, strategy_improved, improvement_factor=state.forecast_improvement/100)
    optimal_result = simulator.run(node_data, strategy_optimal, improvement_factor=1.0)
    
    # Update cache
    state.simulation_results['baseline'] = baseline_result
    state.simulation_results['improved'] = improved_result
    state.simulation_results['optimal'] = optimal_result
    
    return baseline_result, improved_result, optimal_result

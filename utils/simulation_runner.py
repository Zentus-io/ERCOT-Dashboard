import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from utils.state import get_state
from core.battery.simulator import BatterySimulator
from core.battery.strategies import ThresholdStrategy, RollingWindowStrategy, LinearOptimizationStrategy


def _should_rerun_simulation(scenario: str) -> bool:
    """
    Determine if a specific scenario needs to be rerun based on what parameters changed.

    Parameters
    ----------
    scenario : str
        One of: 'baseline', 'improved', 'optimal', 'theoretical_max'

    Returns
    -------
    bool
        True if simulation should be rerun
    """
    state = get_state()

    # If not cached at all, needs to run
    if state.simulation_results.get(scenario) is None:
        return True

    # Theoretical max (LP) only depends on battery specs and data, not strategy params
    if scenario == 'theoretical_max':
        return state.simulation_results['theoretical_max'] is None

    # All other scenarios depend on strategy parameters
    return False


def run_or_get_cached_simulation():
    """
    Run simulations if not cached, or return cached results.

    Returns 4 scenarios:
    - baseline: selected strategy @ 0% forecast improvement (DA only)
    - improved: selected strategy @ X% forecast improvement (slider value)
    - optimal: selected strategy @ 100% forecast improvement (best this strategy can do)
    - theoretical_max: LP @ 100% (absolute theoretical maximum - hindsight benchmark)

    Returns
    -------
    tuple
        (baseline_result, improved_result, optimal_result, theoretical_max_result)
    """
    state = get_state()

    # Check if we have valid cached results
    if (state.simulation_results['baseline'] is not None and
        state.simulation_results['improved'] is not None and
        state.simulation_results['optimal'] is not None and
        state.simulation_results['theoretical_max'] is not None):
        return (
            state.simulation_results['baseline'],
            state.simulation_results['improved'],
            state.simulation_results['optimal'],
            state.simulation_results['theoretical_max']
        )
        
    # If not cached, run simulations
    # We need node_data for this. 
    # Ideally this function should be called where node_data is available or we fetch it here.
    # To avoid circular dependency or complex data fetching here, let's assume the caller 
    # might want to handle data loading, but for convenience we can try to load it if state has it.
    
    if state.price_data is None or state.selected_node is None or state.battery_specs is None:
        return None, None, None, None

    # Filter data (logic duplicated from pages, but necessary for centralized runner)
    # We can't easily import DataLoader here without potentially causing issues,
    # but we can do the filtering manually since it's just a pandas operation
    node_data = state.price_data[state.price_data['node'] == state.selected_node].copy()

    if node_data.empty:
        return None, None, None, None

    simulator = BatterySimulator(state.battery_specs)

    # Determine which simulations need to run
    scenarios_to_run = {}

    # Check each scenario
    if _should_rerun_simulation('baseline'):
        scenarios_to_run['baseline'] = ('baseline', 0.0)
    if _should_rerun_simulation('improved'):
        scenarios_to_run['improved'] = ('improved', state.forecast_improvement/100)
    if _should_rerun_simulation('optimal'):
        scenarios_to_run['optimal'] = ('optimal', 1.0)
    if _should_rerun_simulation('theoretical_max'):
        scenarios_to_run['theoretical_max'] = ('theoretical_max', 1.0)

    # If nothing needs to run, return cached results
    if not scenarios_to_run:
        return (
            state.simulation_results['baseline'],
            state.simulation_results['improved'],
            state.simulation_results['optimal'],
            state.simulation_results['theoretical_max']
        )

    # Define simulation task function
    def run_single_simulation(scenario_name, improvement_factor):
        """Run a single simulation scenario."""
        # Create strategy instance
        if scenario_name == 'theoretical_max':
            strategy = LinearOptimizationStrategy()
        elif state.strategy_type == "Rolling Window Optimization":
            strategy = RollingWindowStrategy(state.window_hours)
        else:  # Threshold-Based
            strategy = ThresholdStrategy(state.charge_percentile, state.discharge_percentile)

        # Run simulation
        return simulator.run(node_data, strategy, improvement_factor=improvement_factor)

    # Run simulations with progress indication
    num_to_run = len(scenarios_to_run)

    if len(scenarios_to_run) > 1:
        # Parallel execution for multiple scenarios
        with st.spinner(f'Running {num_to_run} simulations in parallel...'):
            progress_bar = st.progress(0)
            status_text = st.empty()

            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit all tasks
                futures = {}
                for scenario_name, (_, improvement_factor) in scenarios_to_run.items():
                    future = executor.submit(run_single_simulation, scenario_name, improvement_factor)
                    futures[scenario_name] = future

                # Collect results and update progress
                completed = 0
                for scenario_name, future in futures.items():
                    result = future.result()
                    state.simulation_results[scenario_name] = result
                    completed += 1
                    progress_bar.progress(completed / num_to_run)
                    status_text.text(f"Completed {completed}/{num_to_run} simulations")

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
    else:
        # Single scenario - run directly (no need for parallel)
        scenario_name, (_, improvement_factor) = list(scenarios_to_run.items())[0]
        with st.spinner(f'Running {scenario_name} simulation...'):
            result = run_single_simulation(scenario_name, improvement_factor)
            state.simulation_results[scenario_name] = result

    return (
        state.simulation_results['baseline'],
        state.simulation_results['improved'],
        state.simulation_results['optimal'],
        state.simulation_results['theoretical_max']
    )

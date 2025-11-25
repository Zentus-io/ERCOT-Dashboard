"""
Battery Simulation Engine
Zentus - ERCOT Battery Revenue Dashboard

This module provides the high-level simulation orchestrator.
"""

from dataclasses import dataclass
from typing import List
import pandas as pd
from .battery import Battery, BatterySpecs
from .strategies import DispatchStrategy, DispatchDecision


@dataclass
class SimulationResult:
    """
    Complete simulation results.

    Attributes
    ----------
    dispatch_df : pd.DataFrame
        Timestep-by-timestep results with columns:
        - All original price_df columns
        - dispatch: action taken ('charge', 'discharge', 'hold')
        - power: power level (MW)
        - energy_mwh: energy transferred (MWh)
        - decision_price: price used for decision-making ($/MWh)
        - actual_price: actual RT price paid/received ($/MWh)
        - soc: state of charge at start of timestep (MWh)
        - cumulative_revenue: cumulative revenue at start of timestep ($)
    total_revenue : float
        Net revenue over entire period ($)
    charge_cost : float
        Total cost of charging ($)
    discharge_revenue : float
        Total revenue from discharging ($)
    charge_count : int
        Number of charge events
    discharge_count : int
        Number of discharge events
    hold_count : int
        Number of hold periods
    soc_timestamps : list
        Timestamps for smooth SOC visualization
    soc_values : list
        SOC values for smooth SOC visualization
    metadata : dict
        Strategy-specific metadata
    """
    dispatch_df: pd.DataFrame
    total_revenue: float
    charge_cost: float
    discharge_revenue: float
    charge_count: int
    discharge_count: int
    hold_count: int
    soc_timestamps: list
    soc_values: list
    metadata: dict


import streamlit as st

@st.cache_data(show_spinner=False)
def run_simulation_cached(
    price_df: pd.DataFrame,
    battery_specs: dict,
    strategy_params: dict,
    improvement_factor: float = 0.0
) -> SimulationResult:
    """
    Cached simulation runner.
    Takes simple types (dicts) instead of objects to ensure hashability.
    """
    # Reconstruct objects
    specs = BatterySpecs(**battery_specs)
    
    # Reconstruct strategy
    # Reconstruct strategy
    strategy_type = strategy_params.get('type', 'Threshold-Based')
    
    # Import here to avoid circular imports if any
    from .strategies import ThresholdStrategy, RollingWindowStrategy, LinearOptimizationStrategy, MPCStrategy
    
    if strategy_type == 'Threshold-Based':
        strategy = ThresholdStrategy(
            charge_percentile=strategy_params.get('charge_percentile', 0.25),
            discharge_percentile=strategy_params.get('discharge_percentile', 0.75)
        )
    elif strategy_type == 'Rolling Window Optimization':
        strategy = RollingWindowStrategy(
            window_hours=strategy_params.get('window_hours', 6)
        )
    elif strategy_type == 'Linear Optimization (Perfect Foresight)':
        strategy = LinearOptimizationStrategy()
    elif strategy_type == 'MPC (Rolling Horizon)':
        strategy = MPCStrategy(
            horizon_hours=strategy_params.get('horizon_hours', 24)
        )
    else:
        # Default fallback
        strategy = ThresholdStrategy(0.25, 0.75)

    # Run simulation logic (extracted from original run method)
    if price_df.empty:
        return SimulationResult(
            dispatch_df=pd.DataFrame(),
            total_revenue=0.0,
            charge_cost=0.0,
            discharge_revenue=0.0,
            charge_count=0,
            discharge_count=0,
            hold_count=0,
            soc_timestamps=[],
            soc_values=[],
            metadata={}
        )

    battery = Battery(specs)
    battery.reset()

    decisions = []
    soc_history = []
    revenue_history = []
    soc_timestamps = [price_df.iloc[0]['timestamp']]
    soc_values = [battery.soc]

    total_revenue = 0.0
    charge_cost = 0.0
    discharge_revenue = 0.0

    for idx in range(len(price_df)):
        soc_history.append(battery.soc)
        revenue_history.append(total_revenue)

        decision = strategy.decide(battery, idx, price_df, improvement_factor)
        decisions.append(decision)

        if decision.action == 'charge':
            cost = decision.energy_mwh * decision.actual_price
            total_revenue -= cost
            charge_cost += cost
        elif decision.action == 'discharge':
            revenue = decision.energy_mwh * decision.actual_price
            total_revenue += revenue
            discharge_revenue += revenue

        # Visualization updates
        row = price_df.iloc[idx]
        if decision.action != 'hold':
            if abs(decision.energy_mwh) >= specs.power_mw:
                duration_hours = 1.0
            else:
                duration_hours = abs(decision.energy_mwh) / specs.power_mw
            
            soc_timestamps.append(row['timestamp'] + pd.Timedelta(hours=duration_hours))
            soc_values.append(battery.soc)
            
            if duration_hours < 1.0:
                soc_timestamps.append(row['timestamp'] + pd.Timedelta(hours=1))
                soc_values.append(battery.soc)
        else:
            soc_timestamps.append(row['timestamp'] + pd.Timedelta(hours=1))
            soc_values.append(battery.soc)

    result_df = price_df.copy()
    result_df['dispatch'] = [d.action for d in decisions]
    result_df['power'] = [d.power_mw for d in decisions]
    result_df['energy_mwh'] = [d.energy_mwh for d in decisions]
    result_df['decision_price'] = [d.decision_price for d in decisions]
    result_df['actual_price'] = [d.actual_price for d in decisions]
    result_df['soc'] = soc_history
    result_df['cumulative_revenue'] = revenue_history

    return SimulationResult(
        dispatch_df=result_df,
        total_revenue=total_revenue,
        charge_cost=charge_cost,
        discharge_revenue=discharge_revenue,
        charge_count=sum(1 for d in decisions if d.action == 'charge'),
        discharge_count=sum(1 for d in decisions if d.action == 'discharge'),
        hold_count=sum(1 for d in decisions if d.action == 'hold'),
        soc_timestamps=soc_timestamps,
        soc_values=soc_values,
        metadata=strategy.get_metadata()
    )


class BatterySimulator:
    """
    High-level battery simulation orchestrator.
    Wrapper around cached simulation logic.
    """

    def __init__(self, specs: BatterySpecs):
        self.specs = specs

    def run(self,
            price_df: pd.DataFrame,
            strategy: DispatchStrategy,
            improvement_factor: float = 0.0) -> SimulationResult:
        """
        Run complete simulation (cached).
        """
        # Convert objects to dicts for caching
        battery_specs_dict = {
            'capacity_mwh': self.specs.capacity_mwh,
            'power_mw': self.specs.power_mw,
            'efficiency': self.specs.efficiency,
            'max_soc': self.specs.max_soc,
            'min_soc': self.specs.min_soc,
            'initial_soc': self.specs.initial_soc
        }
        
        # Extract strategy params
        metadata = strategy.get_metadata()
        strategy_params = {'type': metadata.get('strategy', 'Unknown')}
        
        if hasattr(strategy, 'charge_percentile'):
            strategy_params['charge_percentile'] = strategy.charge_percentile
            strategy_params['discharge_percentile'] = strategy.discharge_percentile
        
        if hasattr(strategy, 'window_hours'):
            strategy_params['window_hours'] = strategy.window_hours

        if hasattr(strategy, 'horizon_hours'):
            strategy_params['horizon_hours'] = strategy.horizon_hours
            
        return run_simulation_cached(
            price_df,
            battery_specs_dict,
            strategy_params,
            improvement_factor
        )

"""
Battery Simulation Engine
Zentus - ERCOT Battery Revenue Dashboard

This module provides the high-level simulation orchestrator.
"""

from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

from .battery import Battery, BatterySpecs
from .strategies import (
    ClippingAwareMPCStrategy,
    ClippingOnlyStrategy,
    DispatchStrategy,
    LinearOptimizationStrategy,
    MPCStrategy,
    RollingWindowStrategy,
    ThresholdStrategy,
)


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


@dataclass
class SimulationState:
    """Holds the state of the simulation."""
    decisions: list = field(default_factory=list)
    soc_history: list = field(default_factory=list)
    revenue_history: list = field(default_factory=list)
    soc_timestamps: list = field(default_factory=list)
    soc_values: list = field(default_factory=list)
    total_revenue: float = 0.0
    charge_cost: float = 0.0
    discharge_revenue: float = 0.0


class BatterySimulator:
    """
    High-level battery simulation orchestrator.
    Wrapper around cached simulation logic.
    """

    def __init__(self, specs: BatterySpecs):
        self.specs = specs

    @staticmethod
    def _reconstruct_strategy(strategy_params: dict) -> DispatchStrategy:
        """Reconstructs the strategy object from the strategy parameters."""
        strategy_type = strategy_params.get('type', 'Threshold-Based')
        is_clipping_only = strategy_params.get('is_clipping_only', False)
        is_clipping_aware_mpc = strategy_params.get('is_clipping_aware_mpc', False)

        # Reconstruct base strategy first
        base_strategy = None
        
        if 'Threshold-Based' in strategy_type:
            base_strategy = ThresholdStrategy(
                charge_percentile=strategy_params.get('charge_percentile', 0.25),
                discharge_percentile=strategy_params.get('discharge_percentile', 0.75)
            )
        elif 'Rolling Window' in strategy_type:
            base_strategy = RollingWindowStrategy(
                window_hours=strategy_params.get('window_hours', 6)
            )
        elif 'Linear Optimization' in strategy_type:
            base_strategy = LinearOptimizationStrategy()
        elif 'MPC' in strategy_type:
            if is_clipping_aware_mpc:
                base_strategy = ClippingAwareMPCStrategy(
                    horizon_hours=strategy_params.get('horizon_hours', 24)
                )
            else:
                base_strategy = MPCStrategy(
                    horizon_hours=strategy_params.get('horizon_hours', 24)
                )
        else:
            # Default fallback
            base_strategy = ThresholdStrategy(0.25, 0.75)
        
        # Wrap with ClippingOnlyStrategy if needed
        if is_clipping_only:
            return ClippingOnlyStrategy(base_strategy=base_strategy)
        else:
            return base_strategy

    @staticmethod
    def _initialize_simulation_state(price_df: pd.DataFrame, battery: Battery) -> SimulationState:
        """Initializes the variables used in the simulation."""
        state = SimulationState()
        state.soc_timestamps = [price_df.iloc[0]['timestamp']]
        state.soc_values = [battery.soc]
        return state

    @staticmethod
    def _run_simulation_loop(
        price_df: pd.DataFrame,
        battery: Battery,
        strategy: DispatchStrategy,
        improvement_factor: float,
        state: SimulationState
    ) -> None:
        """Runs the main simulation loop."""
        for idx in range(len(price_df)):
            state.soc_history.append(battery.soc)
            state.revenue_history.append(state.total_revenue)

            decision = strategy.decide(battery, idx, price_df, improvement_factor)
            state.decisions.append(decision)

            if decision.action == 'charge':
                cost = decision.energy_mwh * decision.actual_price
                state.total_revenue -= cost
                state.charge_cost += cost
            elif decision.action == 'discharge':
                revenue = decision.energy_mwh * decision.actual_price
                state.total_revenue += revenue
                state.discharge_revenue += revenue

            # Visualization updates
            row = price_df.iloc[idx]
            if decision.action != 'hold':
                if abs(decision.energy_mwh) >= battery.specs.power_mw:
                    duration_hours = 1.0
                else:
                    duration_hours = abs(decision.energy_mwh) / battery.specs.power_mw

                state.soc_timestamps.append(row['timestamp'] + pd.Timedelta(hours=duration_hours))
                state.soc_values.append(battery.soc)

                if duration_hours < 1.0:
                    state.soc_timestamps.append(row['timestamp'] + pd.Timedelta(hours=1))
                    state.soc_values.append(battery.soc)
            else:
                state.soc_timestamps.append(row['timestamp'] + pd.Timedelta(hours=1))
                state.soc_values.append(battery.soc)

    @staticmethod
    def _create_result_df(
        price_df: pd.DataFrame,
        state: SimulationState
    ) -> pd.DataFrame:
        """Creates the final result dataframe."""
        result_df = price_df.copy()
        result_df['dispatch'] = [d.action for d in state.decisions]
        result_df['power'] = [d.power_mw for d in state.decisions]
        result_df['energy_mwh'] = [d.energy_mwh for d in state.decisions]
        result_df['decision_price'] = [d.decision_price for d in state.decisions]
        result_df['actual_price'] = [d.actual_price for d in state.decisions]
        result_df['soc'] = state.soc_history
        result_df['cumulative_revenue'] = state.revenue_history
        return result_df

    @staticmethod
    @st.cache_resource(show_spinner=False)
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
        strategy = BatterySimulator._reconstruct_strategy(strategy_params)

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

        state = BatterySimulator._initialize_simulation_state(price_df, battery)

        BatterySimulator._run_simulation_loop(
            price_df,
            battery,
            strategy,
            improvement_factor,
            state
        )

        result_df = BatterySimulator._create_result_df(price_df, state)

        return SimulationResult(
            dispatch_df=result_df,
            total_revenue=state.total_revenue,
            charge_cost=state.charge_cost,
            discharge_revenue=state.discharge_revenue,
            charge_count=sum(1 for d in state.decisions if d.action == 'charge'),
            discharge_count=sum(1 for d in state.decisions if d.action == 'discharge'),
            hold_count=sum(1 for d in state.decisions if d.action == 'hold'),
            soc_timestamps=state.soc_timestamps,
            soc_values=state.soc_values,
            metadata=strategy.get_metadata()
        )

    def run(
            self,
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
        
        # Check if this is a ClippingOnlyStrategy wrapper
        is_clipping_only = isinstance(strategy, ClippingOnlyStrategy)
        strategy_params['is_clipping_only'] = is_clipping_only
        
        # If it's a wrapper, extract the base strategy
        base_strategy = strategy.base_strategy if is_clipping_only else strategy
        
        # Check for ClippingAwareMPCStrategy
        is_clipping_aware_mpc = isinstance(base_strategy, ClippingAwareMPCStrategy)
        strategy_params['is_clipping_aware_mpc'] = is_clipping_aware_mpc

        if isinstance(base_strategy, ThresholdStrategy):
            strategy_params['charge_percentile'] = base_strategy.charge_percentile
            strategy_params['discharge_percentile'] = base_strategy.discharge_percentile

        if isinstance(base_strategy, (RollingWindowStrategy, MPCStrategy, ClippingAwareMPCStrategy)):
            strategy_params['window_hours'] = getattr(base_strategy, 'window_hours', None)
            strategy_params['horizon_hours'] = getattr(base_strategy, 'horizon_hours', None)

        return self.run_simulation_cached(
            price_df,
            battery_specs_dict,
            strategy_params,
            improvement_factor
        )

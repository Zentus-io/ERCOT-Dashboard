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


class BatterySimulator:
    """
    High-level battery simulation orchestrator.

    This class runs complete simulations using a battery specification
    and a dispatch strategy.

    Attributes
    ----------
    specs : BatterySpecs
        Battery system specifications
    """

    def __init__(self, specs: BatterySpecs):
        """
        Initialize simulator with battery specifications.

        Parameters
        ----------
        specs : BatterySpecs
            Battery system specifications
        """
        self.specs = specs

    def run(self,
            price_df: pd.DataFrame,
            strategy: DispatchStrategy,
            improvement_factor: float = 0.0) -> SimulationResult:
        """
        Run complete simulation.

        Parameters
        ----------
        price_df : pd.DataFrame
            Price data with columns: timestamp, price_mwh_da, price_mwh_rt, forecast_error
        strategy : DispatchStrategy
            Dispatch strategy to use
        improvement_factor : float, optional
            Forecast improvement factor from 0 (DA only) to 1 (perfect RT) (default: 0.0)

        Returns
        -------
        SimulationResult
            Complete simulation results
        """
        # Validate input data
        if price_df.empty:
            # Return empty result for empty data
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

        # Initialize battery
        battery = Battery(self.specs)
        battery.reset()

        # Track results
        decisions: List[DispatchDecision] = []
        soc_history = []
        revenue_history = []

        # Smooth SOC visualization data
        soc_timestamps = [price_df.iloc[0]['timestamp']]
        soc_values = [battery.soc]

        # Financial tracking
        total_revenue = 0
        charge_cost = 0
        discharge_revenue = 0

        # Main simulation loop
        for idx in range(len(price_df)):
            # Record state at BEGINNING of this hour (before trading)
            soc_history.append(battery.soc)
            revenue_history.append(total_revenue)

            # Make dispatch decision
            decision = strategy.decide(battery, idx, price_df, improvement_factor)
            decisions.append(decision)

            # Update financials based on decision
            if decision.action == 'charge':
                cost = decision.energy_mwh * decision.actual_price
                total_revenue -= cost
                charge_cost += cost

            elif decision.action == 'discharge':
                revenue = decision.energy_mwh * decision.actual_price
                total_revenue += revenue
                discharge_revenue += revenue

            # Update SOC visualization
            row = price_df.iloc[idx]

            # Calculate actual duration based on power capacity
            if decision.action != 'hold':
                if abs(decision.energy_mwh) >= self.specs.power_mw:
                    duration_hours = 1.0
                else:
                    duration_hours = abs(decision.energy_mwh) / self.specs.power_mw

                # Add point at end of ramping (when operation completes)
                soc_timestamps.append(row['timestamp'] + pd.Timedelta(hours=duration_hours))
                soc_values.append(battery.soc)

                # If operation completes before end of hour, add flat section
                if duration_hours < 1.0:
                    soc_timestamps.append(row['timestamp'] + pd.Timedelta(hours=1))
                    soc_values.append(battery.soc)
            else:
                # Hold - SOC stays constant, just add end point
                soc_timestamps.append(row['timestamp'] + pd.Timedelta(hours=1))
                soc_values.append(battery.soc)

        # Build results dataframe
        result_df = price_df.copy()
        result_df['dispatch'] = [d.action for d in decisions]
        result_df['power'] = [d.power_mw for d in decisions]
        result_df['energy_mwh'] = [d.energy_mwh for d in decisions]
        result_df['decision_price'] = [d.decision_price for d in decisions]
        result_df['actual_price'] = [d.actual_price for d in decisions]
        result_df['soc'] = soc_history
        result_df['cumulative_revenue'] = revenue_history

        # Return complete results
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

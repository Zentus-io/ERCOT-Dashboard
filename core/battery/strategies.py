"""
Dispatch Strategy Classes
Zentus - ERCOT Battery Revenue Dashboard

This module implements the Strategy pattern for battery dispatch algorithms.
"""
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from .battery import Battery, BatterySpecs


def calculate_dt(price_df: pd.DataFrame) -> float:
    """
    Calculate time step duration from price data timestamps.

    Parameters
    ----------
    price_df : pd.DataFrame
        Price data with 'timestamp' column

    Returns
    -------
    float
        Time step duration in hours:
        - 1.0 for hourly data (DAM)
        - 0.25 for 15-minute data (RTM)
        - Defaults to 1.0 if cannot determine
    """
    if 'timestamp' in price_df.columns and len(price_df) > 1:
        dt = (price_df['timestamp'].iloc[1] -
              price_df['timestamp'].iloc[0]).total_seconds() / 3600.0
        return dt
    return 1.0


@dataclass
class DispatchDecision:
    """
    Single timestep dispatch decision.

    Attributes
    ----------
    action : {'charge', 'discharge', 'hold'}
        Dispatch action for this timestep
    power_mw : float
        Power level (negative for charge, positive for discharge, zero for hold)
    energy_mwh : float
        Energy transferred (MWh)
    decision_price : float
        Price used for decision-making ($/MWh)
    actual_price : float
        Actual RT price paid/received ($/MWh)
    """
    action: Literal['charge', 'discharge', 'hold']
    power_mw: float
    energy_mwh: float
    decision_price: float
    actual_price: float


class DispatchStrategy(ABC):
    """
    Abstract base class for dispatch strategies.

    All dispatch strategies must implement the decide() method which
    makes dispatch decisions based on current state and price forecasts.
    """

    @abstractmethod
    def decide(self,
               battery: Battery,
               current_idx: int,
               price_df: pd.DataFrame,
               improvement_factor: float) -> DispatchDecision:
        """
        Make dispatch decision for current timestep.

        Parameters
        ----------
        battery : Battery
            Current battery state
        current_idx : int
            Index of current timestep in price_df
        price_df : pd.DataFrame
            Price data with columns: timestamp, price_mwh_da, price_mwh_rt, forecast_error
        improvement_factor : float
            Forecast improvement factor (0 to 1)

        Returns
        -------
        DispatchDecision
            Decision for this timestep
        """
        pass

    @abstractmethod
    def get_metadata(self) -> dict:
        """
        Return strategy-specific metadata.

        Returns
        -------
        dict
            Strategy parameters and calculated thresholds
        """
        pass


class ThresholdStrategy(DispatchStrategy):
    """
    Percentile-based threshold dispatch strategy.

    Charges when forecast price is below charge_percentile threshold.
    Discharges when forecast price is above discharge_percentile threshold.

    Attributes
    ----------
    charge_percentile : float
        Percentile for charge threshold (default: 0.25 = 25th percentile)
    discharge_percentile : float
        Percentile for discharge threshold (default: 0.75 = 75th percentile)
    """

    def __init__(self,
                 charge_percentile: float = 0.25,
                 discharge_percentile: float = 0.75):
        """
        Initialize threshold strategy.

        Parameters
        ----------
        charge_percentile : float, optional
            Charge below this percentile (default: 0.25)
        discharge_percentile : float, optional
            Discharge above this percentile (default: 0.75)
        """
        self.charge_percentile = charge_percentile
        self.discharge_percentile = discharge_percentile
        self._thresholds_calculated = False
        self.charge_threshold: Optional[float] = None
        self.discharge_threshold: Optional[float] = None

    def calculate_thresholds(self, price_df: pd.DataFrame, improvement_factor: float):
        """
        Calculate dynamic thresholds from price distribution.

        Parameters
        ----------
        price_df : pd.DataFrame
            Price data
        improvement_factor : float
            Forecast improvement factor (0 to 1)
        """
        # Calculate decision prices (DA + improvement * error)
        decision_prices = (
            price_df['price_mwh_da'] +
            price_df['forecast_error'] * improvement_factor
        )

        # Calculate percentile thresholds
        self.charge_threshold = decision_prices.quantile(self.charge_percentile)
        self.discharge_threshold = decision_prices.quantile(self.discharge_percentile)

        # Ensure minimum spread for arbitrage opportunity
        if self.discharge_threshold - self.charge_threshold < 5:
            median = decision_prices.median()
            self.charge_threshold = median - 2.5
            self.discharge_threshold = median + 2.5

        self._thresholds_calculated = True

    def decide(self,
               battery: Battery,
               current_idx: int,
               price_df: pd.DataFrame,
               improvement_factor: float) -> DispatchDecision:
        """Make dispatch decision using threshold logic."""
        # Calculate thresholds on first call
        if not self._thresholds_calculated:
            self.calculate_thresholds(price_df, improvement_factor)

        # Get time step duration
        dt = calculate_dt(price_df)

        # Get current row
        row = price_df.iloc[current_idx]

        # Calculate decision price
        decision_price = row['price_mwh_da'] + row['forecast_error'] * improvement_factor
        actual_price = row['price_mwh_rt']

        # Decision logic
        if decision_price < self.charge_threshold and battery.can_charge(battery.specs.power_mw):
            # Charge at max power for this time step
            energy = battery.charge(battery.specs.power_mw, dt)
            return DispatchDecision('charge', -energy, energy, decision_price, actual_price)

        if decision_price > self.discharge_threshold and battery.can_discharge(
                battery.specs.power_mw):
            # Discharge at max power for this time step
            energy = battery.discharge(battery.specs.power_mw, dt)
            return DispatchDecision('discharge', energy, energy, decision_price, actual_price)

        # Hold
        return DispatchDecision('hold', 0, 0, decision_price, actual_price)

    def get_metadata(self) -> dict:
        """Return strategy metadata."""
        return {
            'strategy': 'Threshold-Based',
            'charge_threshold': self.charge_threshold,
            'discharge_threshold': self.discharge_threshold,
            'charge_percentile': self.charge_percentile,
            'discharge_percentile': self.discharge_percentile,
        }


class RollingWindowStrategy(DispatchStrategy):
    """
    Rolling window optimization dispatch strategy.

    Looks ahead N hours and charges if current price is minimum in window,
    discharges if current price is maximum in window.

    Attributes
    ----------
    window_hours : int
        Number of hours to look ahead (default: 6)
    """

    def __init__(self, window_hours: int = 6):
        """
        Initialize rolling window strategy.

        Parameters
        ----------
        window_hours : int, optional
            Lookahead window size in hours (default: 6)
        """
        self.window_hours = window_hours

    def decide(self,
               battery: Battery,
               current_idx: int,
               price_df: pd.DataFrame,
               improvement_factor: float) -> DispatchDecision:
        """Make dispatch decision using rolling window logic."""
        # Get time step duration
        dt = calculate_dt(price_df)

        # Get current row
        row = price_df.iloc[current_idx]

        # Calculate decision price
        decision_price = row['price_mwh_da'] + row['forecast_error'] * improvement_factor
        actual_price = row['price_mwh_rt']

        # Calculate lookahead window (convert hours to steps based on dt)
        window_steps = max(1, int(self.window_hours / dt))
        window_end = min(current_idx + window_steps, len(price_df))
        window_prices = price_df.iloc[current_idx:window_end].apply(
            lambda r: r['price_mwh_da'] + r['forecast_error'] * improvement_factor,
            axis=1
        )

        # Decision logic: charge at minimum, discharge at maximum
        if (np.isclose(decision_price, window_prices.min()) and
                battery.can_charge(battery.specs.power_mw)):
            # Charge: cheapest period in window
            energy = battery.charge(battery.specs.power_mw, dt)
            return DispatchDecision('charge', -energy, energy, decision_price, actual_price)

        if (np.isclose(decision_price, window_prices.max()) and
                battery.can_discharge(battery.specs.power_mw)):
            # Discharge: most expensive period in window
            energy = battery.discharge(battery.specs.power_mw, dt)
            return DispatchDecision('discharge', energy, energy, decision_price, actual_price)

        # Hold
        return DispatchDecision('hold', 0, 0, decision_price, actual_price)

    def get_metadata(self) -> dict:
        """Return strategy metadata."""
        return {
            'strategy': 'Rolling Window Optimization',
            'window_hours': self.window_hours,
        }


def solve_linear_optimization(
    prices: np.ndarray,
    dt: float,
    battery_specs: 'BatterySpecs',
    initial_soc: float
) -> pd.DataFrame:
    """
    Solves the linear optimization problem for a given price series and battery state.
    Parameters
    ----------
    prices : np.ndarray
        Array of decision prices ($/MWh)
    dt : float
        Time step duration in hours
    battery_specs : BatterySpecs
        Battery specifications
    initial_soc : float
        Initial state of charge (MWh)
    Returns
    -------
    pd.DataFrame
        Optimization results with columns: charge_mw, discharge_mw, soc_mwh
    """
    from scipy import sparse  # pylint: disable=import-outside-toplevel
    from scipy.optimize import linprog  # pylint: disable=import-outside-toplevel

    n_steps = len(prices)
    if n_steps == 0:
        return pd.DataFrame({
            'charge_mw': [],
            'discharge_mw': [],
            'soc_mwh': []
        })

    # Variables vector x: [Charge_0...N-1, Discharge_0...N-1, SOC_0...N-1]
    # Total variables = 3 * n_steps

    # 1. Objective Function: Minimize Cost - Revenue
    # Minimize: Sum(Price * C * dt) - Sum(Price * D * dt)

    c_charge = prices * dt
    c_discharge = -prices * dt
    c_soc = np.zeros(n_steps)
    c = np.concatenate([c_charge, c_discharge, c_soc])

    # 2. Bounds
    min_soc_mwh = battery_specs.min_soc * battery_specs.capacity_mwh
    max_soc_mwh = battery_specs.max_soc * battery_specs.capacity_mwh

    bounds_c = [(0, battery_specs.power_mw)] * n_steps
    bounds_d = [(0, battery_specs.power_mw)] * n_steps
    bounds_soc = [(min_soc_mwh, max_soc_mwh)] * n_steps
    bounds = bounds_c + bounds_d + bounds_soc

    # 3. Equality Constraints (Dynamics)
    eff = battery_specs.one_way_efficiency
    coeff_c = -(eff * dt)
    coeff_d = dt / eff

    # Construct A_eq matrix (n_steps rows, 3*n_steps cols)
    rows_soc_t = np.arange(n_steps)
    cols_soc_t = np.arange(2 * n_steps, 3 * n_steps)
    vals_soc_t = np.ones(n_steps)

    rows_soc_prev = np.arange(1, n_steps)
    cols_soc_prev = np.arange(2 * n_steps, 3 * n_steps - 1)
    vals_soc_prev = -np.ones(n_steps - 1)

    rows_c = np.arange(n_steps)
    cols_c = np.arange(n_steps)
    vals_c = np.full(n_steps, coeff_c)

    rows_d = np.arange(n_steps)
    cols_d = np.arange(n_steps, 2 * n_steps)
    vals_d = np.full(n_steps, coeff_d)

    rows = np.concatenate([rows_soc_t, rows_soc_prev, rows_c, rows_d])
    cols = np.concatenate([cols_soc_t, cols_soc_prev, cols_c, cols_d])
    vals = np.concatenate([vals_soc_t, vals_soc_prev, vals_c, vals_d])

    A_eq = sparse.coo_matrix((vals, (rows, cols)), shape=(n_steps, 3 * n_steps))

    # RHS vector b_eq
    b_eq = np.zeros(n_steps)
    b_eq[0] = initial_soc

    # Solve
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if res.success:
        return pd.DataFrame({
            'charge_mw': res.x[:n_steps],
            'discharge_mw': res.x[n_steps:2 * n_steps],
            'soc_mwh': res.x[2 * n_steps:]
        })
    # Fallback: do nothing
    return pd.DataFrame({
        'charge_mw': np.zeros(n_steps),
        'discharge_mw': np.zeros(n_steps),
        'soc_mwh': np.full(n_steps, initial_soc)
    })


class LinearOptimizationStrategy(DispatchStrategy):
    """
    Global linear optimization strategy (Perfect Foresight).
    Uses linear programming to find the theoretically optimal dispatch.
    """

    def __init__(self):
        self._optimization_run = False
        self._dispatch_plan = None
        self._dt = 1.0

    def decide(self,
               battery: Battery,
               current_idx: int,
               price_df: pd.DataFrame,
               improvement_factor: float) -> DispatchDecision:

        if not self._optimization_run:
            # Run optimization for the whole horizon once
            dt = calculate_dt(price_df)
            self._dt = dt

            prices = (price_df['price_mwh_da'] +
                      price_df['forecast_error'] * improvement_factor).to_numpy()

            # DIAGNOSTIC LOGGING
            if os.getenv('MPC_DIAGNOSTICS', 'false').lower() == 'true':
                print("\n=== LP Diagnostics ===")
                print(f"Dataset total: {len(price_df)} timesteps")
                print(f"Sees full dataset: ALL {len(prices)} steps")
                print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
                print(f"Time step (dt): {dt:.2f} hours")
                print(f"Improvement factor: {improvement_factor * 100:.0f}%")

            self._dispatch_plan = solve_linear_optimization(
                prices=prices,
                dt=dt,
                battery_specs=battery.specs,
                initial_soc=battery.soc
            )
            self._optimization_run = True

        # Get planned action
        if self._dispatch_plan is not None and current_idx < len(self._dispatch_plan):
            charge_mw = self._dispatch_plan.iloc[current_idx]['charge_mw']
            discharge_mw = self._dispatch_plan.iloc[current_idx]['discharge_mw']
        else:
            charge_mw = 0
            discharge_mw = 0

        # Get prices
        row = price_df.iloc[current_idx]
        decision_price = row['price_mwh_da'] + row['forecast_error'] * improvement_factor
        actual_price = row['price_mwh_rt']

        if discharge_mw > 0.001:
            energy = battery.discharge(discharge_mw, self._dt)
            return DispatchDecision('discharge', energy, energy, decision_price, actual_price)
        if charge_mw > 0.001:
            energy = battery.charge(charge_mw, self._dt)
            return DispatchDecision('charge', -energy, energy, decision_price, actual_price)
        return DispatchDecision('hold', 0, 0, decision_price, actual_price)

    def get_metadata(self) -> dict:
        return {
            'strategy': 'Linear Optimization (Perfect Foresight)',
            'solver': 'scipy.highs'
        }


class MPCStrategy(DispatchStrategy):
    """
    Model Predictive Control (MPC) strategy.
    Solves a linear optimization problem over a rolling horizon at each time step.
    Implements the first action of the optimal plan and repeats.
    Attributes
    ----------
    horizon_hours : int
        Lookahead horizon in hours (default: 24)
    """

    def __init__(self, horizon_hours: int = 24):
        self.horizon_hours = horizon_hours

    def decide(self,
               battery: Battery,
               current_idx: int,
               price_df: pd.DataFrame,
               improvement_factor: float) -> DispatchDecision:

        # Get time step duration
        dt = calculate_dt(price_df)

        # Determine horizon in steps
        horizon_steps = max(1, int(self.horizon_hours / dt))

        # Slice data for horizon
        # Note: We use the 'decision prices' (forecasts) for optimization
        horizon_end = min(current_idx + horizon_steps, len(price_df))
        horizon_df = price_df.iloc[current_idx:horizon_end]

        if horizon_df.empty:
            return DispatchDecision('hold', 0, 0, 0, 0)

        prices = (horizon_df['price_mwh_da'] +
                  horizon_df['forecast_error'] * improvement_factor).to_numpy()

        # DIAGNOSTIC LOGGING (only on first timestep)
        if current_idx == 0 and os.getenv('MPC_DIAGNOSTICS', 'false').lower() == 'true':
            print(f"\n=== MPC Diagnostics (Horizon={self.horizon_hours}h) ===")
            print(f"Dataset total: {len(price_df)} timesteps")
            print(
                f"Horizon window: [{current_idx}:{horizon_end}] = {
                    horizon_end -
                    current_idx} steps")
            print(f"Expected horizon steps: {horizon_steps}")
            print(
                f"Prices seen: min=${
                    prices.min():.2f}, max=${
                    prices.max():.2f}, mean=${
                    prices.mean():.2f}")
            print(f"Time step (dt): {dt:.2f} hours")
            print(f"Improvement factor: {improvement_factor * 100:.0f}%")

            # Sanity check: Verify horizon slicing is correct
            actual_horizon_length = horizon_end - current_idx
            if actual_horizon_length != min(horizon_steps, len(price_df)):
                print("⚠️  WARNING: Horizon length mismatch!")
                print(f"   Expected: {min(horizon_steps, len(price_df))} steps")
                print(f"   Actual: {actual_horizon_length} steps")

        # Solve LP for this horizon
        plan = solve_linear_optimization(
            prices=prices,
            dt=dt,
            battery_specs=battery.specs,
            initial_soc=battery.soc
        )
        # Take first action
        if not plan.empty:
            charge_mw = plan.iloc[0]['charge_mw']
            discharge_mw = plan.iloc[0]['discharge_mw']
        else:
            charge_mw = 0
            discharge_mw = 0

        # Get prices for return object
        row = price_df.iloc[current_idx]
        decision_price = row['price_mwh_da'] + row['forecast_error'] * improvement_factor
        actual_price = row['price_mwh_rt']

        if discharge_mw > 0.001:
            energy = battery.discharge(discharge_mw, dt)
            return DispatchDecision('discharge', energy, energy, decision_price, actual_price)
        if charge_mw > 0.001:
            energy = battery.charge(charge_mw, dt)
            return DispatchDecision('charge', -energy, energy, decision_price, actual_price)
        return DispatchDecision('hold', 0, 0, decision_price, actual_price)

    def get_metadata(self) -> dict:
        return {
            'strategy': 'MPC (Rolling Horizon)',
            'horizon_hours': self.horizon_hours
        }

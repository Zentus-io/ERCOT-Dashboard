"""
Dispatch Strategy Classes
Zentus - ERCOT Battery Revenue Dashboard

This module implements the Strategy pattern for battery dispatch algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional
import pandas as pd
import numpy as np
from .battery import Battery


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

        # Get current row
        row = price_df.iloc[current_idx]

        # Calculate decision price
        decision_price = row['price_mwh_da'] + row['forecast_error'] * improvement_factor
        actual_price = row['price_mwh_rt']

        # Decision logic
        if decision_price < self.charge_threshold and battery.can_charge(battery.specs.power_mw):
            # Charge
            energy = battery.charge(battery.specs.power_mw)
            return DispatchDecision('charge', -energy, energy, decision_price, actual_price)

        elif decision_price > self.discharge_threshold and battery.can_discharge(battery.specs.power_mw):
            # Discharge
            energy = battery.discharge(battery.specs.power_mw)
            return DispatchDecision('discharge', energy, energy, decision_price, actual_price)

        else:
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
        # Get current row
        row = price_df.iloc[current_idx]

        # Calculate decision price
        decision_price = row['price_mwh_da'] + row['forecast_error'] * improvement_factor
        actual_price = row['price_mwh_rt']

        # Calculate lookahead window
        window_end = min(current_idx + self.window_hours, len(price_df))
        window_prices = price_df.iloc[current_idx:window_end].apply(
            lambda r: r['price_mwh_da'] + r['forecast_error'] * improvement_factor,
            axis=1
        )

        # Decision logic: charge at minimum, discharge at maximum
        if np.isclose(decision_price, window_prices.min()) and battery.can_charge(battery.specs.power_mw):
            # Charge: cheapest hour in window
            energy = battery.charge(battery.specs.power_mw)
            return DispatchDecision('charge', -energy, energy, decision_price, actual_price)

        elif np.isclose(decision_price, window_prices.max()) and battery.can_discharge(battery.specs.power_mw):
            # Discharge: most expensive hour in window
            energy = battery.discharge(battery.specs.power_mw)
            return DispatchDecision('discharge', energy, energy, decision_price, actual_price)

        else:
            # Hold
            return DispatchDecision('hold', 0, 0, decision_price, actual_price)

    def get_metadata(self) -> dict:
        """Return strategy metadata."""
        return {
            'strategy': 'Rolling Window Optimization',
            'window_hours': self.window_hours,
        }

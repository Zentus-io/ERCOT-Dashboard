"""
Hybrid Dispatch Strategy for Solar + Storage
Zentus - ERCOT Battery Revenue Dashboard

This strategy prioritizes charging from clipped solar energy (free) over
grid-based arbitrage (paid).
"""

from typing import Optional
import pandas as pd

from .battery import Battery
from .strategies import DispatchStrategy, DispatchDecision, calculate_dt


class HybridDispatchStrategy(DispatchStrategy):
    """
    Hybrid dispatch strategy for solar + storage co-location.

    Decision Priority:
    1. Charge from clipped energy when available (free energy source)
    2. Discharge to grid during high price periods (arbitrage)
    3. Grid-based arbitrage (buy low, sell high) for remaining capacity

    The strategy wraps an existing base strategy (e.g., RollingWindowStrategy)
    and augments it with clipped energy awareness.

    Attributes
    ----------
    base_strategy : DispatchStrategy
        Underlying strategy for grid-based arbitrage decisions
    clipped_priority : bool
        If True, prioritize clipped energy charging over base strategy
    _clipped_energy_captured : float
        Running total of clipped energy captured (MWh)
    _grid_arbitrage_revenue : float
        Running total of grid arbitrage revenue ($)
    """

    def __init__(
        self,
        base_strategy: DispatchStrategy,
        clipped_priority: bool = True
    ):
        """
        Initialize hybrid strategy.

        Parameters
        ----------
        base_strategy : DispatchStrategy
            Base strategy for grid arbitrage (e.g., RollingWindowStrategy)
        clipped_priority : bool, optional
            Prioritize clipped energy over base strategy (default: True)
        """
        self.base_strategy = base_strategy
        self.clipped_priority = clipped_priority
        self._clipped_energy_captured = 0.0
        self._grid_arbitrage_revenue = 0.0

    def decide(
        self,
        battery: Battery,
        current_idx: int,
        price_df: pd.DataFrame,
        improvement_factor: float
    ) -> DispatchDecision:
        """
        Make hybrid dispatch decision.

        Parameters
        ----------
        battery : Battery
            Current battery state
        current_idx : int
            Index of current timestep in price_df
        price_df : pd.DataFrame
            Price data with columns:
            - timestamp, price_mwh_da, price_mwh_rt, forecast_error
            - clipped_mw (optional): Available clipped energy at this timestep
        improvement_factor : float
            Forecast improvement factor (0 to 1)

        Returns
        -------
        DispatchDecision
            Decision for this timestep, considering clipped energy priority
        """
        # Get time step duration
        dt = calculate_dt(price_df)

        # Get current row
        row = price_df.iloc[current_idx]

        # Calculate decision and actual prices
        decision_price = row['price_mwh_da'] + row['forecast_error'] * improvement_factor
        actual_price = row['price_mwh_rt']

        # Check if clipped energy is available
        clipped_mw = row.get('clipped_mw', 0.0)

        # PRIORITY 1: Charge from clipped energy (free)
        if self.clipped_priority and clipped_mw > 0.01:  # Threshold for numerical stability
            # Calculate how much clipped energy we can capture
            # Limited by: available clipped power, battery power rating, and SOC headroom

            if battery.can_charge(clipped_mw):
                # Determine chargeable power (limited by clipped availability and battery rating)
                chargeable_power = min(clipped_mw, battery.specs.power_mw)

                # Calculate energy captured (considering time step duration)
                # clipped_mw is instantaneous power, need to convert to energy
                # available_clipped_energy = clipped_mw * dt  # MWh available this timestep

                # Attempt to charge (battery.charge handles SOC limits and efficiency)
                energy_from_grid = battery.charge(chargeable_power, dt)

                # For clipped energy, the "grid" is actually free solar
                # So we track this as zero cost
                # Note: We still apply efficiency losses (energy_from_grid > energy_stored in SOC)

                self._clipped_energy_captured += energy_from_grid

                return DispatchDecision(
                    action='charge',
                    power_mw=-energy_from_grid / dt if dt > 0 else 0,
                    energy_mwh=energy_from_grid,
                    decision_price=0.0,  # Clipped energy is free
                    actual_price=0.0     # No grid cost
                )

        # PRIORITY 2: Fall back to base strategy for grid-based arbitrage
        # This handles both discharge decisions and grid-based charging
        base_decision = self.base_strategy.decide(
            battery, current_idx, price_df, improvement_factor
        )

        # Track grid arbitrage separately
        if base_decision.action == 'discharge':
            self._grid_arbitrage_revenue += base_decision.energy_mwh * actual_price
        elif base_decision.action == 'charge':
            self._grid_arbitrage_revenue -= base_decision.energy_mwh * actual_price

        return base_decision

    def get_metadata(self) -> dict:
        """
        Return strategy metadata.

        Returns
        -------
        dict
            Strategy parameters including clipped energy capture metrics
        """
        base_metadata = self.base_strategy.get_metadata()
        return {
            'strategy': f"Hybrid ({base_metadata.get('strategy', 'Unknown')})",
            'base_strategy': base_metadata.get('strategy'),
            'clipped_priority': self.clipped_priority,
            'clipped_energy_captured': self._clipped_energy_captured,
            'grid_arbitrage_revenue': self._grid_arbitrage_revenue,
            **base_metadata  # Include all base strategy metadata
        }

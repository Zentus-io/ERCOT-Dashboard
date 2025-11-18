"""
Battery System Classes
Zentus - ERCOT Battery Revenue Dashboard

This module defines the Battery and BatterySpecs classes that model
battery storage system behavior.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BatterySpecs:
    """
    Immutable battery system specifications.

    Attributes
    ----------
    capacity_mwh : float
        Total energy storage capacity (MWh)
    power_mw : float
        Maximum charge/discharge rate (MW)
    efficiency : float
        Round-trip efficiency (0 to 1)
    min_soc : float, optional
        Minimum state of charge as fraction (default: 0.05 = 5%)
    max_soc : float, optional
        Maximum state of charge as fraction (default: 0.95 = 95%)
    initial_soc : float, optional
        Initial state of charge as fraction (default: 0.5 = 50%)
    """
    capacity_mwh: float
    power_mw: float
    efficiency: float
    min_soc: float = 0.05
    max_soc: float = 0.95
    initial_soc: float = 0.5

    def __post_init__(self):
        """Validate specifications."""
        if self.capacity_mwh <= 0:
            raise ValueError("capacity_mwh must be positive")
        if self.power_mw <= 0:
            raise ValueError("power_mw must be positive")
        if not 0 < self.efficiency <= 1:
            raise ValueError("efficiency must be between 0 and 1")
        if not 0 <= self.min_soc < self.max_soc <= 1:
            raise ValueError("must have 0 <= min_soc < max_soc <= 1")
        if not self.min_soc <= self.initial_soc <= self.max_soc:
            raise ValueError("initial_soc must be between min_soc and max_soc")

    @property
    def duration_hours(self) -> float:
        """Calculate battery duration in hours at full power."""
        return self.capacity_mwh / self.power_mw

    @property
    def usable_capacity_mwh(self) -> float:
        """Calculate usable energy capacity between min and max SOC."""
        return self.capacity_mwh * (self.max_soc - self.min_soc)


class Battery:
    """
    Battery state manager.

    This class manages the state of charge and enforces operational constraints
    during charging and discharging operations.

    Attributes
    ----------
    specs : BatterySpecs
        Battery specifications
    soc : float
        Current state of charge (MWh)
    """

    def __init__(self, specs: BatterySpecs):
        """
        Initialize battery with specifications.

        Parameters
        ----------
        specs : BatterySpecs
            Battery system specifications
        """
        self.specs = specs
        self.soc = specs.initial_soc * specs.capacity_mwh

    @property
    def soc_fraction(self) -> float:
        """Get current SOC as fraction of capacity."""
        return self.soc / self.specs.capacity_mwh

    def can_charge(self, amount_mwh: float) -> bool:
        """
        Check if battery can accept charge.

        Parameters
        ----------
        amount_mwh : float
            Requested charge amount (MWh)

        Returns
        -------
        bool
            True if battery can accept any charge
        """
        return self.soc < self.specs.capacity_mwh * self.specs.max_soc

    def can_discharge(self, amount_mwh: float) -> bool:
        """
        Check if battery can discharge.

        Parameters
        ----------
        amount_mwh : float
            Requested discharge amount (MWh)

        Returns
        -------
        bool
            True if battery can provide any discharge
        """
        return self.soc > self.specs.capacity_mwh * self.specs.min_soc

    def charge(self, amount_mwh: float) -> float:
        """
        Charge battery and return actual energy purchased from grid.

        The actual charge is limited by:
        - Requested amount
        - Power capacity
        - Remaining capacity to max SOC

        Parameters
        ----------
        amount_mwh : float
            Requested charge amount (MWh)

        Returns
        -------
        float
            Actual energy purchased from grid (MWh)
            This is BEFORE efficiency losses.
        """
        # Calculate maximum chargeable amount
        max_charge = min(
            amount_mwh,
            self.specs.power_mw,
            (self.specs.capacity_mwh * self.specs.max_soc - self.soc) / self.specs.efficiency
        )

        # Energy stored in battery (after efficiency losses)
        energy_stored = max_charge * self.specs.efficiency
        self.soc += energy_stored

        # Return energy purchased from grid (before losses)
        return max_charge

    def discharge(self, amount_mwh: float) -> float:
        """
        Discharge battery and return actual energy delivered to grid.

        The actual discharge is limited by:
        - Requested amount
        - Power capacity
        - Available capacity above min SOC

        Parameters
        ----------
        amount_mwh : float
            Requested discharge amount (MWh)

        Returns
        -------
        float
            Actual energy delivered to grid (MWh)
        """
        # Calculate maximum dischargeable amount
        max_discharge = min(
            amount_mwh,
            self.specs.power_mw,
            self.soc - self.specs.capacity_mwh * self.specs.min_soc
        )

        # Remove from battery and return
        self.soc -= max_discharge
        return max_discharge

    def reset(self):
        """Reset battery to initial state of charge."""
        self.soc = self.specs.initial_soc * self.specs.capacity_mwh

    def __repr__(self) -> str:
        """String representation of battery state."""
        return (
            f"Battery(SOC={self.soc:.1f} MWh / {self.specs.capacity_mwh:.1f} MWh "
            f"[{self.soc_fraction:.1%}], Power={self.specs.power_mw} MW)"
        )

"""
Battery System Classes
Zentus - ERCOT Battery Revenue Dashboard

This module defines the Battery and BatterySpecs classes that model
battery storage system behavior.
"""

from dataclasses import dataclass


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
        Round-trip efficiency (0 to 1). Applied as sqrt(efficiency) on both
        charge and discharge to model realistic bidirectional losses.
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

    @property
    def one_way_efficiency(self) -> float:
        """Single-direction efficiency (sqrt of round-trip)."""
        return self.efficiency ** 0.5

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

    Time Step Handling:
    - For DAM (Day-Ahead Market): dt = 1.0 hour
    - For RTM (Real-Time Market): dt = 0.25 hours (15-minute settlements in ERCOT)
    - Energy transferred = Power × dt

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

    def can_charge(self, _power_mw: float) -> bool:
        """
        Check if battery can accept charge.

        Parameters
        ----------
        _power_mw : float
            Requested charge power (MW)

        Returns
        -------
        bool
            True if battery can accept any charge
        """
        return self.soc < self.specs.capacity_mwh * self.specs.max_soc

    def can_discharge(self, _power_mw: float) -> bool:
        """
        Check if battery can discharge.

        Parameters
        ----------
        _power_mw : float
            Requested discharge power (MW)

        Returns
        -------
        bool
            True if battery can provide any discharge
        """
        return self.soc > self.specs.capacity_mwh * self.specs.min_soc

    def charge(self, power_mw: float, dt_hours: float = 1.0) -> float:
        """
        Charge battery at specified power for specified duration.

        The actual charge is limited by:
        - Requested power
        - Maximum power capacity
        - Remaining capacity to max SOC

        Efficiency model: Uses sqrt(round_trip_efficiency) for charging,
        representing realistic bidirectional losses.

        Parameters
        ----------
        power_mw : float
            Requested charge power (MW)
        dt_hours : float, optional
            Time step duration in hours (default: 1.0 for hourly/DAM,
            use 0.25 for 15-minute RTM)

        Returns
        -------
        float
            Actual energy purchased from grid (MWh) = actual_power × dt
            This is BEFORE efficiency losses.
        """
        one_way_eff = self.specs.one_way_efficiency

        # Calculate maximum chargeable power (limited by power rating)
        max_power = min(power_mw, self.specs.power_mw)

        # Calculate maximum energy that can be stored (limited by SOC headroom)
        soc_headroom = self.specs.capacity_mwh * self.specs.max_soc - self.soc
        max_energy_to_store = soc_headroom  # MWh we can add to SOC

        # To store X MWh, we need X/eff MWh from grid
        max_grid_energy_from_soc = max_energy_to_store / one_way_eff

        # Actual grid energy is limited by power×dt and SOC headroom
        grid_energy = min(max_power * dt_hours, max_grid_energy_from_soc)

        # Energy stored in battery (after one-way efficiency loss)
        energy_stored = grid_energy * one_way_eff
        self.soc += energy_stored

        # Return energy purchased from grid (before losses)
        return grid_energy

    def discharge(self, power_mw: float, dt_hours: float = 1.0) -> float:
        """
        Discharge battery at specified power for specified duration.

        The actual discharge is limited by:
        - Requested power
        - Maximum power capacity
        - Available capacity above min SOC

        Efficiency model: Uses sqrt(round_trip_efficiency) for discharging,
        representing realistic bidirectional losses. Energy delivered to grid
        is less than energy drawn from SOC.

        Parameters
        ----------
        power_mw : float
            Requested discharge power (MW) - power to deliver to grid
        dt_hours : float, optional
            Time step duration in hours (default: 1.0 for hourly/DAM,
            use 0.25 for 15-minute RTM)

        Returns
        -------
        float
            Actual energy delivered to grid (MWh) = actual_power × dt
        """
        one_way_eff = self.specs.one_way_efficiency

        # Calculate maximum discharge power (limited by power rating)
        max_power = min(power_mw, self.specs.power_mw)

        # Available energy in SOC above minimum
        available_soc = self.soc - self.specs.capacity_mwh * self.specs.min_soc

        # Maximum grid energy we can deliver (SOC × efficiency)
        max_grid_energy_from_soc = available_soc * one_way_eff

        # Actual grid energy is limited by power×dt and available SOC
        grid_energy = min(max_power * dt_hours, max_grid_energy_from_soc)

        # Energy drawn from SOC (more than delivered due to losses)
        energy_from_soc = grid_energy / one_way_eff
        self.soc -= energy_from_soc

        # Return energy delivered to grid (after losses)
        return grid_energy

    def reset(self):
        """Reset battery to initial state of charge."""
        self.soc = self.specs.initial_soc * self.specs.capacity_mwh

    def __repr__(self) -> str:
        """String representation of battery state."""
        return (
            f"Battery(SOC={self.soc:.1f} MWh / {self.specs.capacity_mwh:.1f} MWh "
            f"[{self.soc_fraction:.1%}], Power={self.specs.power_mw} MW)"
        )

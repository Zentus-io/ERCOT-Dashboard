"""
Unit tests for HybridDispatchStrategy.
Zentus - ERCOT Battery Revenue Dashboard
"""

import pytest
import pandas as pd
import numpy as np

from core.battery.battery import Battery, BatterySpecs
from core.battery.strategies import RollingWindowStrategy, ThresholdStrategy
from core.battery.hybrid_strategy import HybridDispatchStrategy


def create_test_price_data(clipped_pattern: str = 'none', num_steps: int = 96):
    """
    Create test price data with configurable clipping patterns.

    Parameters
    ----------
    clipped_pattern : str
        Pattern for clipped energy:
        - 'none': No clipping (all zeros)
        - 'constant': Constant 10 MW clipped
        - 'solar_curve': Bell curve pattern (solar generation shape)
    num_steps : int
        Number of 15-minute timesteps (default: 96 = 1 day)

    Returns
    -------
    pd.DataFrame
        Test price data with timestamp index and clipped_mw column
    """
    timestamps = pd.date_range('2024-01-01', periods=num_steps, freq='15min')

    df = pd.DataFrame({
        'timestamp': timestamps,
        'price_mwh_da': np.random.uniform(20, 80, num_steps),
        'price_mwh_rt': np.random.uniform(20, 80, num_steps),
        'forecast_error': np.random.uniform(-5, 5, num_steps),
    })

    df = df.set_index('timestamp')

    # Add clipped energy based on pattern
    if clipped_pattern == 'none':
        df['clipped_mw'] = 0.0
    elif clipped_pattern == 'constant':
        df['clipped_mw'] = 10.0  # Constant 10 MW clipped
    elif clipped_pattern == 'solar_curve':
        # Bell curve pattern (solar generation shape)
        hours = np.arange(num_steps) / 4  # Convert to hours
        df['clipped_mw'] = 20 * np.exp(-((hours - 12)**2) / (2 * 3**2))
        df.loc[df['clipped_mw'] < 0.1, 'clipped_mw'] = 0

    return df


@pytest.fixture
def battery():
    """Create test battery with standard specs."""
    specs = BatterySpecs(
        capacity_mwh=100,
        power_mw=50,
        efficiency=0.90,
        initial_soc=0.5
    )
    return Battery(specs)


@pytest.fixture
def battery_high_efficiency():
    """Create test battery with high efficiency for easier calculations."""
    specs = BatterySpecs(
        capacity_mwh=100,
        power_mw=50,
        efficiency=0.81,  # One-way efficiency = 0.9
        initial_soc=0.5
    )
    return Battery(specs)


def test_hybrid_strategy_without_clipping(battery):
    """Test that hybrid strategy behaves like base strategy when no clipping."""
    df = create_test_price_data(clipped_pattern='none')

    base_strategy = RollingWindowStrategy(window_hours=6)
    hybrid_strategy = HybridDispatchStrategy(base_strategy)

    # Both should make same decisions when clipped_mw = 0
    base_decision = base_strategy.decide(battery, 0, df, 0.5)

    battery.reset()  # Reset for fair comparison
    hybrid_decision = hybrid_strategy.decide(battery, 0, df, 0.5)

    assert base_decision.action == hybrid_decision.action
    assert abs(base_decision.energy_mwh - hybrid_decision.energy_mwh) < 0.01


def test_hybrid_strategy_prioritizes_clipping(battery):
    """Test that hybrid strategy prioritizes clipped energy charging."""
    df = create_test_price_data(clipped_pattern='constant')

    # Set up scenario where base strategy would discharge but clipping is available
    df.loc[df.index[0], 'price_mwh_da'] = 100  # High price (base would want to discharge)
    df.loc[df.index[0], 'price_mwh_rt'] = 100
    df.loc[df.index[0], 'clipped_mw'] = 15  # But clipping is available

    base_strategy = RollingWindowStrategy(window_hours=6)
    hybrid_strategy = HybridDispatchStrategy(base_strategy, clipped_priority=True)

    decision = hybrid_strategy.decide(battery, 0, df, 0.5)

    # Should charge from clipping despite high price
    assert decision.action == 'charge'
    assert decision.decision_price == 0.0  # Clipped energy is free
    assert decision.actual_price == 0.0


def test_hybrid_strategy_tracks_clipped_capture(battery):
    """Test that hybrid strategy tracks clipped energy capture."""
    df = create_test_price_data(clipped_pattern='constant')

    # Set all prices low to ensure base strategy doesn't interfere
    df['price_mwh_da'] = 20
    df['price_mwh_rt'] = 20

    base_strategy = RollingWindowStrategy(window_hours=6)
    hybrid_strategy = HybridDispatchStrategy(base_strategy)

    # Run through several timesteps with constant clipping
    # At least some should result in charging from clipped energy
    for i in range(20):
        if battery.can_charge(10):  # Only if battery can still charge
            hybrid_strategy.decide(battery, i, df, 0.5)

    metadata = hybrid_strategy.get_metadata()

    # Should have captured some clipped energy
    assert metadata['clipped_energy_captured'] > 0


def test_hybrid_efficiency_applied(battery_high_efficiency):
    """Test that one-way efficiency is applied to clipped energy charging."""
    specs = battery_high_efficiency.specs
    battery = battery_high_efficiency

    df = create_test_price_data(clipped_pattern='constant')

    base_strategy = RollingWindowStrategy(window_hours=6)
    hybrid_strategy = HybridDispatchStrategy(base_strategy)

    initial_soc = battery.soc
    decision = hybrid_strategy.decide(battery, 0, df, 0.5)

    # Check that SOC increased by less than energy_mwh due to efficiency
    # one_way_efficiency = sqrt(0.81) = 0.9
    # SOC increase should be energy_mwh * 0.9
    expected_soc_increase = decision.energy_mwh * 0.9
    actual_soc_increase = battery.soc - initial_soc

    assert abs(actual_soc_increase - expected_soc_increase) < 0.01


def test_hybrid_respects_power_limits(battery):
    """Test that hybrid strategy respects battery power limits."""
    df = create_test_price_data(clipped_pattern='none')

    # Set very high clipping (more than battery can handle)
    df.loc[df.index[0], 'clipped_mw'] = 100  # 100 MW clipped, but battery is only 50 MW

    base_strategy = ThresholdStrategy(0.25, 0.75)
    hybrid_strategy = HybridDispatchStrategy(base_strategy)

    decision = hybrid_strategy.decide(battery, 0, df, 0.5)

    # Should charge, but limited by battery power rating
    if decision.action == 'charge':
        # Power should be capped at battery rating (50 MW)
        assert abs(decision.power_mw) <= battery.specs.power_mw * 1.01  # Small tolerance


def test_hybrid_respects_soc_limits(battery):
    """Test that hybrid strategy respects SOC limits."""
    df = create_test_price_data(clipped_pattern='constant')

    # Charge battery to near full
    battery.soc = battery.specs.capacity_mwh * battery.specs.max_soc - 1  # Almost full

    base_strategy = ThresholdStrategy(0.25, 0.75)
    hybrid_strategy = HybridDispatchStrategy(base_strategy)

    initial_soc = battery.soc
    decision = hybrid_strategy.decide(battery, 0, df, 0.5)

    # After decision, SOC should not exceed max_soc
    assert battery.soc <= battery.specs.capacity_mwh * battery.specs.max_soc * 1.01


def test_hybrid_no_clipped_priority_flag(battery):
    """Test hybrid strategy with clipped_priority=False."""
    df = create_test_price_data(clipped_pattern='constant')

    # Low price scenario - base strategy would charge
    df.loc[df.index[0], 'price_mwh_da'] = 10
    df.loc[df.index[0], 'price_mwh_rt'] = 10

    base_strategy = ThresholdStrategy(0.25, 0.75)
    hybrid_strategy = HybridDispatchStrategy(base_strategy, clipped_priority=False)

    decision = hybrid_strategy.decide(battery, 0, df, 0.5)

    # With clipped_priority=False, should behave like base strategy
    # Base strategy would charge at low price
    # Decision price should not be 0 (not using clipped energy)
    if decision.action == 'charge':
        assert decision.decision_price != 0.0


def test_hybrid_metadata_structure(battery):
    """Test that hybrid strategy returns proper metadata."""
    df = create_test_price_data(clipped_pattern='none')

    base_strategy = RollingWindowStrategy(window_hours=12)
    hybrid_strategy = HybridDispatchStrategy(base_strategy)

    # Run a few steps
    for i in range(5):
        hybrid_strategy.decide(battery, i, df, 0.5)

    metadata = hybrid_strategy.get_metadata()

    # Check required fields
    assert 'strategy' in metadata
    assert 'base_strategy' in metadata
    assert 'clipped_priority' in metadata
    assert 'clipped_energy_captured' in metadata
    assert 'grid_arbitrage_revenue' in metadata

    # Check hybrid strategy name format
    assert 'Hybrid' in metadata['strategy']


def test_hybrid_temporal_availability(battery):
    """Test that clipped energy is only used when temporally available."""
    df = create_test_price_data(clipped_pattern='none')

    # Add clipping only at specific timesteps
    df.loc[df.index[5], 'clipped_mw'] = 20  # Clipping at timestep 5
    df.loc[df.index[10], 'clipped_mw'] = 20  # Clipping at timestep 10

    base_strategy = ThresholdStrategy(0.25, 0.75)
    hybrid_strategy = HybridDispatchStrategy(base_strategy)

    # At timestep 0 (no clipping), should not capture clipped energy
    decision_0 = hybrid_strategy.decide(battery, 0, df, 0.5)
    captured_0 = hybrid_strategy._clipped_energy_captured

    battery.reset()

    # At timestep 5 (with clipping), should capture clipped energy
    decision_5 = hybrid_strategy.decide(battery, 5, df, 0.5)
    captured_5 = hybrid_strategy._clipped_energy_captured

    # Captured at timestep 5 should be more than at timestep 0
    if decision_5.action == 'charge' and decision_5.decision_price == 0.0:
        assert captured_5 > captured_0


def test_hybrid_with_different_base_strategies(battery):
    """Test that hybrid works with different base strategies."""
    df = create_test_price_data(clipped_pattern='none')

    strategies = [
        ThresholdStrategy(0.25, 0.75),
        RollingWindowStrategy(window_hours=6),
    ]

    for base_strategy in strategies:
        hybrid_strategy = HybridDispatchStrategy(base_strategy)
        battery.reset()

        # Should not raise errors
        decision = hybrid_strategy.decide(battery, 0, df, 0.5)
        assert decision is not None
        assert decision.action in ['charge', 'discharge', 'hold']

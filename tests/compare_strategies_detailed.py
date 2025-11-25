"""
Detailed Strategy Comparison
Zentus - ERCOT Battery Revenue Dashboard

Compares MPC and LP strategies action-by-action to determine:
- Are decisions identical or just revenues identical?
- When/why do strategies diverge?
- Does MPC converge to LP as horizon increases?

This helps diagnose the root cause of flat horizon sensitivity.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, date
from dotenv import load_dotenv
from supabase import create_client

from core.battery.battery import BatterySpecs
from core.battery.strategies import MPCStrategy, LinearOptimizationStrategy
from core.battery.simulator import BatterySimulator

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


def load_ercot_data(
    node: str = "ALVIN_RN",
    start_date: date = date(2025, 11, 13),
    end_date: date = date(2025, 11, 20)
) -> pd.DataFrame:
    """Load ERCOT price data from Supabase."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Supabase credentials not found in .env file")

    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Fetch both DAM and RTM data from unified table
    response = client.table("ercot_prices") \
        .select("timestamp,settlement_point,market,price_mwh") \
        .eq("settlement_point", node) \
        .gte("timestamp", start_date.isoformat()) \
        .lte("timestamp", end_date.isoformat()) \
        .order("timestamp") \
        .limit(20000) \
        .execute()

    if not response.data:
        raise ValueError(f"No data found for {node}")

    # Convert to DataFrame
    df_all = pd.DataFrame(response.data)
    df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])

    # Pivot: separate DAM and RTM into columns
    dam_df = df_all[df_all['market'] == 'DAM'][['timestamp', 'settlement_point', 'price_mwh']].copy()
    rtm_df = df_all[df_all['market'] == 'RTM'][['timestamp', 'settlement_point', 'price_mwh']].copy()

    if dam_df.empty or rtm_df.empty:
        raise ValueError(f"Incomplete data for {node} - missing DAM or RTM prices")

    dam_df = dam_df.rename(columns={'price_mwh': 'price_mwh_da', 'settlement_point': 'node'})
    rtm_df = rtm_df.rename(columns={'price_mwh': 'price_mwh_rt', 'settlement_point': 'node'})

    # Merge using nearest timestamp (DAM is hourly, RTM is 15-min)
    rtm_df['hour'] = rtm_df['timestamp'].dt.floor('h')
    dam_df['hour'] = dam_df['timestamp'].dt.floor('h')

    df = rtm_df.merge(
        dam_df[['hour', 'node', 'price_mwh_da']],
        on=['hour', 'node'],
        how='left'
    ).drop('hour', axis=1)

    # Add forecast_error column (required by strategies)
    df['forecast_error'] = df['price_mwh_rt'] - df['price_mwh_da']

    return df


def compare_two_strategies(
    result1: 'SimulationResult',
    result2: 'SimulationResult',
    name1: str,
    name2: str,
    show_first_n: int = 24
) -> dict:
    """
    Compare two strategy results action-by-action.

    Returns dict with:
    - total_timesteps: number of timesteps
    - identical_actions: number of identical actions
    - identical_power: number of identical power levels
    - revenue_diff: revenue difference
    - first_divergence: first timestep where actions differ
    - comparison_df: DataFrame with side-by-side comparison
    """
    df1 = result1.dispatch_df[['timestamp', 'dispatch', 'power', 'energy_mwh', 'soc']].copy()
    df2 = result2.dispatch_df[['timestamp', 'dispatch', 'power', 'energy_mwh', 'soc']].copy()

    df1.columns = ['timestamp', f'{name1}_action', f'{name1}_power', f'{name1}_energy', f'{name1}_soc']
    df2.columns = ['timestamp', f'{name2}_action', f'{name2}_power', f'{name2}_energy', f'{name2}_soc']

    comparison = df1.merge(df2, on='timestamp')

    # Count identical actions
    identical_actions = (comparison[f'{name1}_action'] == comparison[f'{name2}_action']).sum()

    # Count identical power levels (within 0.1 MW tolerance)
    identical_power = (np.abs(comparison[f'{name1}_power'] - comparison[f'{name2}_power']) < 0.1).sum()

    # Find first divergence
    divergence_mask = comparison[f'{name1}_action'] != comparison[f'{name2}_action']
    if divergence_mask.any():
        first_divergence = comparison[divergence_mask].index[0]
    else:
        first_divergence = None

    # Revenue difference
    revenue_diff = result1.total_revenue - result2.total_revenue

    return {
        'total_timesteps': len(comparison),
        'identical_actions': identical_actions,
        'identical_power': identical_power,
        'revenue_diff': revenue_diff,
        'first_divergence': first_divergence,
        'comparison_df': comparison.head(show_first_n)
    }


def test_mpc_lp_convergence(
    price_df: pd.DataFrame,
    battery_specs: BatterySpecs,
    horizons: list = [2, 4, 8, 12, 24],
    improvement_factor: float = 1.0
):
    """
    Test if MPC converges to LP as horizon increases.

    Critical question: At 100% forecast accuracy, does MPC 24h match LP exactly?
    """
    print(f"\n{'='*60}")
    print(f"Testing MPC→LP Convergence (Improvement={improvement_factor*100:.0f}%)")
    print(f"{'='*60}")

    # Run LP
    lp_strategy = LinearOptimizationStrategy()
    simulator = BatterySimulator(battery_specs)
    lp_result = simulator.run(price_df, lp_strategy, improvement_factor)

    print(f"\nLP Benchmark: ${lp_result.total_revenue:,.0f}")

    # Run MPC at various horizons
    mpc_results = {}
    for h in horizons:
        mpc_strategy = MPCStrategy(horizon_hours=h)
        mpc_result = simulator.run(price_df, mpc_strategy, improvement_factor)
        mpc_results[h] = mpc_result
        print(f"MPC {h:2d}h:      ${mpc_result.total_revenue:,.0f}")

    # Compare each MPC horizon to LP
    print(f"\n{'='*60}")
    print(f"Action-by-Action Comparison vs LP")
    print(f"{'='*60}")

    for h in horizons:
        comparison = compare_two_strategies(
            mpc_results[h],
            lp_result,
            f"MPC_{h}h",
            "LP",
            show_first_n=10
        )

        pct_identical = (comparison['identical_actions'] / comparison['total_timesteps'] * 100)

        print(f"\nMPC {h}h vs LP:")
        print(f"  Identical actions: {comparison['identical_actions']}/{comparison['total_timesteps']} ({pct_identical:.1f}%)")
        print(f"  Identical power:   {comparison['identical_power']}/{comparison['total_timesteps']}")
        print(f"  Revenue diff:      ${comparison['revenue_diff']:,.0f}")
        print(f"  First divergence:  Step {comparison['first_divergence']}" if comparison['first_divergence'] is not None else "  First divergence:  None (100% match)")

        if pct_identical == 100 and abs(comparison['revenue_diff']) < 0.01:
            print(f"  ⚠️  CRITICAL: MPC {h}h is IDENTICAL to LP!")

    # Show detailed comparison for longest horizon
    print(f"\n{'='*60}")
    print(f"First 10 Steps: MPC {horizons[-1]}h vs LP")
    print(f"{'='*60}")
    longest_comparison = compare_two_strategies(
        mpc_results[horizons[-1]],
        lp_result,
        f"MPC_{horizons[-1]}h",
        "LP",
        show_first_n=10
    )
    print(longest_comparison['comparison_df'].to_string(index=False))


def test_mpc_horizon_differences(
    price_df: pd.DataFrame,
    battery_specs: BatterySpecs,
    improvement_factor: float = 0.0
):
    """
    Compare MPC strategies with different horizons to each other.

    If H3 (horizon slicing bug), we expect MPC 2h = MPC 24h.
    """
    print(f"\n{'='*60}")
    print(f"Testing MPC Horizon Differences")
    print(f"{'='*60}")

    horizons = [2, 4, 8, 12, 24]
    simulator = BatterySimulator(battery_specs)

    # Run all horizons
    results = {}
    for h in horizons:
        strategy = MPCStrategy(horizon_hours=h)
        result = simulator.run(price_df, strategy, improvement_factor)
        results[h] = result
        print(f"MPC {h:2d}h: ${result.total_revenue:,.0f}")

    # Compare consecutive horizons
    print(f"\n{'='*60}")
    print(f"Consecutive Horizon Comparisons")
    print(f"{'='*60}")

    for i in range(len(horizons) - 1):
        h1 = horizons[i]
        h2 = horizons[i + 1]

        comparison = compare_two_strategies(
            results[h1],
            results[h2],
            f"MPC_{h1}h",
            f"MPC_{h2}h",
            show_first_n=5
        )

        pct_identical = (comparison['identical_actions'] / comparison['total_timesteps'] * 100)

        print(f"\nMPC {h1}h vs MPC {h2}h:")
        print(f"  Identical actions: {comparison['identical_actions']}/{comparison['total_timesteps']} ({pct_identical:.1f}%)")
        print(f"  Revenue diff:      ${comparison['revenue_diff']:,.0f}")

        if pct_identical > 95:
            print(f"  ⚠️  WARNING: Nearly identical decisions despite {h2-h1}h horizon increase!")

    # Compare 2h vs 24h directly
    print(f"\n{'='*60}")
    print(f"Direct Comparison: MPC 2h vs MPC 24h")
    print(f"{'='*60}")

    comparison_2_24 = compare_two_strategies(
        results[2],
        results[24],
        "MPC_2h",
        "MPC_24h",
        show_first_n=20
    )

    pct_identical = (comparison_2_24['identical_actions'] / comparison_2_24['total_timesteps'] * 100)

    print(f"\nIdentical actions: {comparison_2_24['identical_actions']}/{comparison_2_24['total_timesteps']} ({pct_identical:.1f}%)")
    print(f"Revenue diff:      ${comparison_2_24['revenue_diff']:,.0f}")

    if pct_identical > 90:
        print(f"\n⚠️  CRITICAL: 2h and 24h horizons make nearly identical decisions!")
        print(f"   This suggests either:")
        print(f"   - H3: Horizon slicing bug (both see same data)")
        print(f"   - H1: Price patterns too simple (2h sufficient)")
        print(f"   - H2: Battery constraints limit strategies")

    print(f"\nFirst 20 timesteps comparison:")
    print(comparison_2_24['comparison_df'].to_string(index=False))


def main():
    """Run detailed strategy comparison."""
    print("\n" + "="*60)
    print("DETAILED STRATEGY COMPARISON")
    print("="*60)

    # Load data
    price_df = load_ercot_data(
        node="ALVIN_RN",
        start_date=date(2025, 11, 13),
        end_date=date(2025, 11, 20)
    )

    print(f"\nLoaded {len(price_df)} price points")
    print(f"Period: {price_df['timestamp'].min()} to {price_df['timestamp'].max()}")

    # Define battery specs
    battery_specs = BatterySpecs(
        capacity_mwh=100.0,
        power_mw=50.0,
        efficiency=0.85,
        min_soc=0.05,
        max_soc=0.95,
        initial_soc=0.5
    )

    print(f"\nBattery Specs:")
    print(f"  Capacity: {battery_specs.capacity_mwh} MWh")
    print(f"  Power: {battery_specs.power_mw} MW")
    print(f"  Duration: {battery_specs.duration_hours:.1f} hours")

    # Test 1: MPC horizon differences (baseline forecast)
    test_mpc_horizon_differences(price_df, battery_specs, improvement_factor=0.0)

    # Test 2: MPC→LP convergence (perfect forecast)
    test_mpc_lp_convergence(price_df, battery_specs, improvement_factor=1.0)

    print(f"\n{'='*60}")
    print(f"COMPARISON COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

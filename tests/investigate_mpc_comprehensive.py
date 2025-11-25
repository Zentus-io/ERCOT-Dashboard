"""
Comprehensive MPC Diagnostic Test Suite
Zentus - ERCOT Battery Revenue Dashboard

Tests MPC behavior across multiple horizons with real ERCOT data
to diagnose why horizon sensitivity is flat.

Hypotheses:
- H1: ERCOT prices simple/repetitive enough that 2h = 24h foresight
- H2: Battery constraints (2h duration) limit all strategies equally
- H3: Bug in horizon slicing (MPC sees more than intended)
- H4: Caching bug returns wrong results for different horizons
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from datetime import datetime, date
from dotenv import load_dotenv
from supabase import create_client

from core.battery.battery import BatterySpecs, Battery
from core.battery.strategies import MPCStrategy, LinearOptimizationStrategy
from core.battery.simulator import BatterySimulator

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


def load_real_ercot_data(
    node: str = "ALVIN_RN",
    start_date: date = date(2025, 11, 13),
    end_date: date = date(2025, 11, 20),
    limit: int = 10000
) -> pd.DataFrame:
    """
    Load real ERCOT price data from Supabase.

    Parameters
    ----------
    node : str
        Settlement point name (default: ALVIN_RN - resource node)
    start_date : date
        Start date for data
    end_date : date
        End date for data
    limit : int
        Maximum rows to fetch

    Returns
    -------
    pd.DataFrame
        Price data with columns: timestamp, node, price_mwh_da, price_mwh_rt
    """
    print(f"\n{'='*60}")
    print(f"Loading ERCOT Data from Supabase")
    print(f"{'='*60}")
    print(f"Node: {node}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Limit: {limit} rows\n")

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
        .limit(limit * 2) \
        .execute()

    if not response.data:
        raise ValueError(f"No data found for {node} in period {start_date} to {end_date}")

    # Convert to DataFrame
    df = pd.DataFrame(response.data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Pivot: separate DAM and RTM into columns
    dam_df = df[df['market'] == 'DAM'][['timestamp', 'settlement_point', 'price_mwh']].copy()
    rtm_df = df[df['market'] == 'RTM'][['timestamp', 'settlement_point', 'price_mwh']].copy()

    if dam_df.empty or rtm_df.empty:
        raise ValueError(f"Incomplete data for {node} - missing DAM or RTM prices")

    # Rename columns
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

    print(f"✓ Loaded {len(df)} rows")
    print(f"  Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  DA Price range: ${df['price_mwh_da'].min():.2f} - ${df['price_mwh_da'].max():.2f}")
    print(f"  RT Price range: ${df['price_mwh_rt'].min():.2f} - ${df['price_mwh_rt'].max():.2f}")

    return df


def run_mpc_with_diagnostics(
    price_df: pd.DataFrame,
    battery_specs: BatterySpecs,
    horizon_hours: int,
    improvement_factor: float = 1.0
) -> dict:
    """
    Run MPC strategy with diagnostic logging.

    Returns dict with:
    - result: SimulationResult
    - diagnostics: dict of diagnostic info
    """
    print(f"\n{'='*60}")
    print(f"Running MPC: Horizon={horizon_hours}h, Improvement={improvement_factor*100:.0f}%")
    print(f"{'='*60}")

    strategy = MPCStrategy(horizon_hours=horizon_hours)
    simulator = BatterySimulator(battery_specs)

    # Add diagnostic checks
    diagnostics = {
        'horizon_hours': horizon_hours,
        'total_timesteps': len(price_df),
        'improvement_factor': improvement_factor
    }

    # Run simulation
    result = simulator.run(price_df, strategy, improvement_factor)

    print(f"✓ Completed: Revenue = ${result.total_revenue:,.0f}")
    print(f"  Charge events: {result.charge_count}")
    print(f"  Discharge events: {result.discharge_count}")
    print(f"  Hold periods: {result.hold_count}")

    return {
        'result': result,
        'diagnostics': diagnostics
    }


def run_lp_with_diagnostics(
    price_df: pd.DataFrame,
    battery_specs: BatterySpecs,
    improvement_factor: float = 1.0
) -> dict:
    """
    Run LP strategy with diagnostic logging.
    """
    print(f"\n{'='*60}")
    print(f"Running LP Benchmark: Improvement={improvement_factor*100:.0f}%")
    print(f"{'='*60}")

    strategy = LinearOptimizationStrategy()
    simulator = BatterySimulator(battery_specs)

    diagnostics = {
        'total_timesteps': len(price_df),
        'improvement_factor': improvement_factor
    }

    result = simulator.run(price_df, strategy, improvement_factor)

    print(f"✓ Completed: Revenue = ${result.total_revenue:,.0f}")
    print(f"  Charge events: {result.charge_count}")
    print(f"  Discharge events: {result.discharge_count}")
    print(f"  Hold periods: {result.hold_count}")

    return {
        'result': result,
        'diagnostics': diagnostics
    }


def compare_horizons(
    price_df: pd.DataFrame,
    battery_specs: BatterySpecs,
    horizons: list = [2, 4, 8, 12, 24],
    improvement_factor: float = 1.0
) -> pd.DataFrame:
    """
    Compare MPC performance across different horizons.

    Returns DataFrame with columns:
    - horizon_hours
    - total_revenue
    - charge_count
    - discharge_count
    - hold_count
    - first_10_actions (for action comparison)
    """
    results = []

    for h in horizons:
        mpc_result = run_mpc_with_diagnostics(price_df, battery_specs, h, improvement_factor)

        # Extract first 10 actions for comparison
        first_10 = mpc_result['result'].dispatch_df.iloc[:10][['timestamp', 'dispatch', 'power', 'soc']].to_dict('records')

        results.append({
            'horizon_hours': h,
            'total_revenue': mpc_result['result'].total_revenue,
            'charge_count': mpc_result['result'].charge_count,
            'discharge_count': mpc_result['result'].discharge_count,
            'hold_count': mpc_result['result'].hold_count,
            'first_10_actions': first_10
        })

    return pd.DataFrame(results)


def test_hypothesis_h3_horizon_slicing(
    price_df: pd.DataFrame,
    battery_specs: BatterySpecs
):
    """
    H3: Test if there's a bug in horizon slicing.

    Manually verify that MPC with 2h horizon only sees 2 steps ahead.
    """
    print(f"\n{'='*60}")
    print(f"Testing H3: Horizon Slicing Bug")
    print(f"{'='*60}")

    # Create a battery and strategy
    battery = Battery(battery_specs)
    strategy_2h = MPCStrategy(horizon_hours=2)
    strategy_24h = MPCStrategy(horizon_hours=24)

    # Make decisions at timestep 0
    decision_2h = strategy_2h.decide(battery, 0, price_df, 0.0)

    battery.reset()
    decision_24h = strategy_24h.decide(battery, 0, price_df, 0.0)

    print(f"\nTimestep 0 Decisions:")
    print(f"  2h horizon:  {decision_2h.action} @ {decision_2h.power_mw:.2f} MW")
    print(f"  24h horizon: {decision_24h.action} @ {decision_24h.power_mw:.2f} MW")

    if decision_2h.action == decision_24h.action and abs(decision_2h.power_mw - decision_24h.power_mw) < 0.01:
        print(f"\n⚠️  WARNING: 2h and 24h horizons made IDENTICAL decisions!")
        print(f"    This suggests H3 (horizon slicing bug) OR H1 (prices too simple)")
    else:
        print(f"\n✓ Different decisions - horizon slicing appears correct")


def test_hypothesis_h4_caching(
    price_df: pd.DataFrame,
    battery_specs: BatterySpecs
):
    """
    H4: Test if caching is returning wrong results.

    Run same horizon twice and different horizons to verify cache behavior.
    """
    print(f"\n{'='*60}")
    print(f"Testing H4: Caching Bug")
    print(f"{'='*60}")

    # Run 8h horizon twice
    result_8h_run1 = run_mpc_with_diagnostics(price_df, battery_specs, 8, 0.0)
    result_8h_run2 = run_mpc_with_diagnostics(price_df, battery_specs, 8, 0.0)

    # Run 12h horizon
    result_12h = run_mpc_with_diagnostics(price_df, battery_specs, 12, 0.0)

    print(f"\nCaching Test Results:")
    print(f"  8h Run 1: ${result_8h_run1['result'].total_revenue:,.0f}")
    print(f"  8h Run 2: ${result_8h_run2['result'].total_revenue:,.0f}")
    print(f"  12h:      ${result_12h['result'].total_revenue:,.0f}")

    if abs(result_8h_run1['result'].total_revenue - result_8h_run2['result'].total_revenue) > 0.01:
        print(f"\n⚠️  WARNING: Same horizon gave different results - cache inconsistency!")
    else:
        print(f"\n✓ Same horizon consistent across runs")

    if abs(result_8h_run1['result'].total_revenue - result_12h['result'].total_revenue) < 0.01:
        print(f"⚠️  WARNING: Different horizons gave same revenue - possible cache bug!")
    else:
        print(f"✓ Different horizons gave different results")


def main():
    """Run comprehensive MPC diagnostics."""
    print("\n" + "="*60)
    print("COMPREHENSIVE MPC DIAGNOSTIC TEST SUITE")
    print("="*60)

    # Load real ERCOT data
    price_df = load_real_ercot_data(
        node="ALVIN_RN",
        start_date=date(2025, 11, 13),
        end_date=date(2025, 11, 20)
    )

    # Define battery specs (default 100 MWh / 50 MW = 2h duration)
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
    print(f"  Efficiency: {battery_specs.efficiency*100:.0f}%")

    # Test H3: Horizon slicing
    test_hypothesis_h3_horizon_slicing(price_df, battery_specs)

    # Test H4: Caching
    test_hypothesis_h4_caching(price_df, battery_specs)

    # Compare horizons with baseline forecast (0% improvement)
    print(f"\n{'='*60}")
    print(f"Horizon Comparison: Baseline Forecast (0% improvement)")
    print(f"{'='*60}")
    horizon_comparison = compare_horizons(price_df, battery_specs, [2, 4, 8, 12, 24], 0.0)
    print("\n" + horizon_comparison.to_string(index=False))

    # Run LP benchmark
    lp_result = run_lp_with_diagnostics(price_df, battery_specs, 0.0)
    print(f"\nLP Benchmark Revenue: ${lp_result['result'].total_revenue:,.0f}")

    # Compare MPC horizons vs LP
    print(f"\n{'='*60}")
    print(f"MPC vs LP Comparison")
    print(f"{'='*60}")
    for _, row in horizon_comparison.iterrows():
        gap = lp_result['result'].total_revenue - row['total_revenue']
        pct_of_lp = (row['total_revenue'] / lp_result['result'].total_revenue * 100) if lp_result['result'].total_revenue > 0 else 0
        print(f"MPC {row['horizon_hours']:2d}h: ${row['total_revenue']:10,.0f}  ({pct_of_lp:5.1f}% of LP, Gap: ${gap:,.0f})")

    # Test with perfect foresight (100% improvement)
    print(f"\n{'='*60}")
    print(f"Perfect Foresight Test (100% improvement)")
    print(f"{'='*60}")

    mpc_24h_perfect = run_mpc_with_diagnostics(price_df, battery_specs, 24, 1.0)
    lp_perfect = run_lp_with_diagnostics(price_df, battery_specs, 1.0)

    print(f"\nPerfect Foresight Results:")
    print(f"  MPC 24h: ${mpc_24h_perfect['result'].total_revenue:,.0f}")
    print(f"  LP:      ${lp_perfect['result'].total_revenue:,.0f}")

    if abs(mpc_24h_perfect['result'].total_revenue - lp_perfect['result'].total_revenue) < 0.01:
        print(f"\n⚠️  CRITICAL: MPC 24h matches LP at 100% accuracy!")
        print(f"    This confirms the user's observation.")

    print(f"\n{'='*60}")
    print(f"DIAGNOSTICS COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

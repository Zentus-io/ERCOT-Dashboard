"""
Precompute Demo Defaults
Zentus - ERCOT Battery Revenue Dashboard

Purpose:
    Generate gzipped-pickle artifacts that the dashboard loads on cold-start
    instead of running the full 4-scenario simulator. Cuts ~5-15s off the
    first page hit when the user's configuration matches the canonical demo
    defaults (DEFAULT_NODE + DEFAULT_BATTERY + DEFAULT_STRATEGY + full
    available DB date window).

    Run this script:
      - Once after the initial deploy.
      - Every time DEFAULT_BATTERY, DEFAULT_STRATEGY, DEFAULT_NODE, or the
        SimulationResult dataclass changes.
      - Every time the Supabase data window grows materially (so the date
        range in the manifest matches what state.available_date_range will
        report).

Usage:
    python scripts/precompute_demo_defaults.py
    python scripts/precompute_demo_defaults.py --node CACH_ESS_RN
    python scripts/precompute_demo_defaults.py --start 2025-11-13 --end 2025-11-26

Output:
    data/precomputed/sim_default.pkl.gz   # {scenario_name: SimulationResult}
    data/precomputed/manifest.json        # records exact signature used
"""

import argparse
import gzip
import os
import pickle
import sys
import time
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path so we can import the same modules the app uses.
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import DEFAULT_BATTERY, DEFAULT_NODE, DEFAULT_STRATEGY  # noqa: E402
from core.battery.battery import BatterySpecs  # noqa: E402
from core.battery.simulator import BatterySimulator  # noqa: E402
from core.battery.strategies import (  # noqa: E402
    LinearOptimizationStrategy,
    MPCStrategy,
    RollingWindowStrategy,
    ThresholdStrategy,
)
from core.data.loaders import SupabaseDataLoader  # noqa: E402
from utils.demo_defaults import (  # noqa: E402
    SIM_PICKLE_PATH,
    PRECOMPUTE_DIR,
    write_manifest,
)


SCENARIOS = (
    # (scenario_name, improvement_factor_or_None)
    # None → theoretical_max which always uses LP, regardless of selected strategy.
    ('baseline', 0.0),
    ('improved', None),  # special-cased below: uses DEFAULT_FORECAST_IMPROVEMENT
    ('optimal', 1.0),
    ('theoretical_max', 1.0),
)


def _build_strategy(scenario_name: str, strategy_type: str):
    """Mirror utils/simulation_runner.run_single_simulation strategy selection."""
    if scenario_name == 'theoretical_max':
        return LinearOptimizationStrategy()
    if strategy_type == 'Rolling Window Optimization':
        return RollingWindowStrategy(int(DEFAULT_STRATEGY['window_hours']))
    if strategy_type == 'MPC (Rolling Horizon)':
        return MPCStrategy(int(DEFAULT_STRATEGY['horizon_hours']))
    # default: Threshold-Based
    return ThresholdStrategy(
        float(DEFAULT_STRATEGY['charge_percentile']),
        float(DEFAULT_STRATEGY['discharge_percentile']),
    )


def _build_signature(node, start_date_str, end_date_str, forecast_improvement_pct):
    """Mirror utils/demo_defaults.state_signature() exactly."""

    def r(x):
        return round(float(x), 6)

    return (
        node,
        start_date_str,
        end_date_str,
        r(DEFAULT_BATTERY['capacity_mwh']),
        r(DEFAULT_BATTERY['power_mw']),
        r(DEFAULT_BATTERY['efficiency']),
        r(DEFAULT_BATTERY['min_soc']),
        r(DEFAULT_BATTERY['max_soc']),
        r(DEFAULT_BATTERY['initial_soc']),
        DEFAULT_STRATEGY['type'],
        int(DEFAULT_STRATEGY['horizon_hours']),
        int(DEFAULT_STRATEGY['window_hours']),
        r(DEFAULT_STRATEGY['charge_percentile']),
        r(DEFAULT_STRATEGY['discharge_percentile']),
        r(forecast_improvement_pct),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node', default=DEFAULT_NODE,
                        help=f"Settlement point (default: {DEFAULT_NODE})")
    parser.add_argument('--start', type=str, default=None,
                        help='Start date YYYY-MM-DD (default: DB earliest after intersection)')
    parser.add_argument('--end', type=str, default=None,
                        help='End date YYYY-MM-DD (default: DB latest)')
    parser.add_argument('--forecast-improvement', type=float, default=10.0,
                        help='Slider value used to build the "improved" scenario (default: 10%%)')
    parser.add_argument('--output-dir', default=None,
                        help=f'Output dir (default: {PRECOMPUTE_DIR})')
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv('SUPABASE_URL') or not os.getenv('SUPABASE_KEY'):
        print('ERROR: SUPABASE_URL and SUPABASE_KEY must be set in .env')
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else PRECOMPUTE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    sim_path = output_dir / 'sim_default.pkl.gz' if args.output_dir else SIM_PICKLE_PATH

    print('=' * 80)
    print('Precompute demo defaults')
    print('=' * 80)
    print(f'Node:        {args.node}')
    print(f'Strategy:    {DEFAULT_STRATEGY["type"]} (horizon={DEFAULT_STRATEGY["horizon_hours"]}h)')
    print(f'Battery:     {DEFAULT_BATTERY["capacity_mwh"]} MWh / '
          f'{DEFAULT_BATTERY["power_mw"]} MW @ {DEFAULT_BATTERY["efficiency"]:.0%}')
    print(f'Improvement: {args.forecast_improvement:.1f}%')

    # 1. Resolve date range
    loader = SupabaseDataLoader()
    if args.start and args.end:
        start_date = date.fromisoformat(args.start)
        end_date = date.fromisoformat(args.end)
    else:
        earliest, latest = loader.get_date_range()
        if earliest is None or latest is None:
            print('ERROR: could not fetch DB date range — is the database empty?')
            return 1
        start_date = date.fromisoformat(args.start) if args.start else earliest
        end_date = date.fromisoformat(args.end) if args.end else latest
    print(f'Date range:  {start_date} → {end_date}')

    # 2. Fetch price data
    print('\n[1/3] Fetching price data...')
    t0 = time.time()
    price_df = loader.load_prices(node=args.node, start_date=start_date, end_date=end_date)
    if price_df.empty:
        print(f'ERROR: no price data returned for {args.node} in {start_date}..{end_date}')
        return 1
    print(f'  → {len(price_df):,} rows in {time.time() - t0:.1f}s')

    # 3. Build battery + simulator
    specs = BatterySpecs(**DEFAULT_BATTERY)
    simulator = BatterySimulator(specs)

    # 4. Run all 4 scenarios sequentially (no parallelism — easier to debug)
    print(f'\n[2/3] Running {len(SCENARIOS)} scenarios sequentially...')
    results = {}
    for scenario_name, improvement_factor in SCENARIOS:
        # 'improved' scenario uses the slider value (forecast_improvement / 100)
        if improvement_factor is None:
            improvement_factor = args.forecast_improvement / 100.0

        strategy = _build_strategy(scenario_name, DEFAULT_STRATEGY['type'])
        t0 = time.time()
        result = simulator.run(price_df, strategy, improvement_factor=improvement_factor)
        results[scenario_name] = result
        print(f'  {scenario_name:18s} revenue=${result.total_revenue:>10,.0f}  '
              f'charge={result.charge_count:4d}  discharge={result.discharge_count:4d}  '
              f'({time.time() - t0:.1f}s)')

    # 5. Serialize
    print('\n[3/3] Writing artifacts...')
    with gzip.open(sim_path, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    sim_size_kb = sim_path.stat().st_size / 1024
    print(f'  ✓ {sim_path.relative_to(Path(__file__).parent.parent)}  ({sim_size_kb:.1f} KB)')

    # 6. Manifest
    signature = _build_signature(
        args.node, str(start_date), str(end_date), args.forecast_improvement,
    )
    write_manifest(signature, extra={
        'generated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'price_rows': len(price_df),
        'scenarios': {
            k: {
                'total_revenue': round(float(v.total_revenue), 2),
                'charge_count': int(v.charge_count),
                'discharge_count': int(v.discharge_count),
                'hold_count': int(v.hold_count),
            }
            for k, v in results.items()
        },
    })
    print(f'  ✓ {(output_dir / "manifest.json").relative_to(Path(__file__).parent.parent)}')

    print('\nDone. Reload the dashboard to verify cold-start time drops to <1s.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

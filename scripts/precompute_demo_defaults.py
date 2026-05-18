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
    NODAL_PICKLE_PATH,
    PRECOMPUTE_DIR,
    SIM_PICKLE_PATH,
    STRATEGY_PICKLE_PATH,
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


def _precompute_strategy_analysis(simulator, price_df, args):
    """Run the 3 Strategy Analysis sensitivity sweeps used by page 6.

    Mirrors the loops in pages/6_🔍_Strategy_Analysis.py:
      - forecast sensitivity: 11 improvement levels × 4 strategies
      - horizon sensitivity (MPC only): 6 horizon values
      - window sensitivity (Rolling Window only): SKIPPED — not active when
        default strategy is MPC, so the page never calls it.
    """
    print(f'\n[Strategy] Forecast sensitivity (11 improvements × 4 strategies)...')
    improvement_range = list(range(0, 101, 10))
    revenue_threshold, revenue_rolling_window, revenue_mpc, revenue_linear = [], [], [], []
    t0 = time.time()
    for imp in improvement_range:
        factor = imp / 100.0
        # Strategy instances rebuilt per iteration to mirror page code
        s_threshold = ThresholdStrategy(0.25, 0.75)
        s_window = RollingWindowStrategy(6)
        # NOTE: page hardcodes MPC horizon=24 inside the sweep (line 133 of page 6)
        s_mpc = MPCStrategy(horizon_hours=24)
        s_linear = LinearOptimizationStrategy()
        revenue_threshold.append(simulator.run(price_df, s_threshold, improvement_factor=factor).total_revenue)
        revenue_rolling_window.append(simulator.run(price_df, s_window, improvement_factor=factor).total_revenue)
        revenue_mpc.append(simulator.run(price_df, s_mpc, improvement_factor=factor).total_revenue)
        revenue_linear.append(simulator.run(price_df, s_linear, improvement_factor=factor).total_revenue)
        print(f'  imp={imp:3d}%  threshold=${revenue_threshold[-1]:>10,.0f}  '
              f'window=${revenue_rolling_window[-1]:>10,.0f}  '
              f'mpc=${revenue_mpc[-1]:>10,.0f}  lp=${revenue_linear[-1]:>10,.0f}')
    print(f'  → {time.time() - t0:.1f}s')

    print(f'\n[Strategy] Horizon sensitivity (6 horizons @ {args.forecast_improvement:.0f}%)...')
    horizon_range = list(range(2, 14, 2))
    revenue_horizon = []
    t0 = time.time()
    factor = args.forecast_improvement / 100.0
    for h in horizon_range:
        result = simulator.run(price_df, MPCStrategy(horizon_hours=h), improvement_factor=factor)
        revenue_horizon.append(result.total_revenue)
        print(f'  horizon={h:2d}h  revenue=${revenue_horizon[-1]:>10,.0f}')
    print(f'  → {time.time() - t0:.1f}s')

    return {
        'sensitivity': {
            'improvement_range': improvement_range,
            'revenue_threshold': revenue_threshold,
            'revenue_rolling_window': revenue_rolling_window,
            'revenue_mpc': revenue_mpc,
            'revenue_linear': revenue_linear,
        },
        'horizon_sensitivity': {
            'horizon_range': horizon_range,
            'revenue_horizon': revenue_horizon,
        },
    }


def _precompute_nodal_analysis(start_date, end_date):
    """Run the Nodal Analysis cross-node assessment for the visible date window.

    Mirrors pages/1_🗺️_Nodal_Analysis.py:run_nodal_assessment exactly so the
    intercept can drop these results straight into st.session_state.
    """
    from core.data.loaders import load_data as _load_data  # local: page does the same

    print('\n[Nodal] Cross-node scan...')
    loader = SupabaseDataLoader()
    nodes = loader.get_available_nodes()[:50]
    print(f'  scanning {len(nodes)} nodes for {start_date}..{end_date}')

    results = []
    node_data_cache = {}
    t0 = time.time()
    for i, node in enumerate(nodes, start=1):
        try:
            df = _load_data(source='database', node=node, start_date=start_date, end_date=end_date)
        except Exception as exc:  # match page behavior: skip nodes that fail
            print(f'  [{i:2d}/{len(nodes)}] {node:18s} ERROR: {exc}')
            continue
        if df.empty:
            print(f'  [{i:2d}/{len(nodes)}] {node:18s} (no data)')
            continue
        volatility = df['price_mwh_rt'].std()
        avg_spread = df['price_spread'].mean()
        profitable_spreads = df[df['price_spread'] > 20]['price_spread']
        revenue_score = profitable_spreads.sum()
        results.append({
            'Node': node,
            'Volatility ($/MWh)': round(volatility, 2),
            'Avg Spread ($/MWh)': round(avg_spread, 2),
            'Revenue Score': round(revenue_score, 2),
            'Data Points': len(df),
        })
        if len(node_data_cache) < 10:
            node_data_cache[node] = df
        print(f'  [{i:2d}/{len(nodes)}] {node:18s} vol=${volatility:6.2f}  score=${revenue_score:>10,.0f}')

    import pandas as pd  # local import: avoid touching module-top imports
    results_df = pd.DataFrame(results).sort_values('Revenue Score', ascending=False).reset_index(drop=True)
    results_df.index += 1
    # Make sure the top-10 (used by viz) are all in the cache, like the page does.
    top_nodes = results_df.head(10)['Node'].tolist()
    for node in top_nodes:
        if node not in node_data_cache:
            try:
                node_data_cache[node] = _load_data(
                    source='database', node=node, start_date=start_date, end_date=end_date)
            except Exception:
                pass
    print(f'  → {time.time() - t0:.1f}s, {len(results_df)} nodes ranked')
    return results_df, node_data_cache


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
    parser.add_argument('--scope', choices=['all', 'sim', 'strategy', 'nodal'], default='all',
                        help='Which artifacts to (re)generate (default: all)')
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

    extra_manifest = {
        'generated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'price_rows': len(price_df),
    }

    sim_results = None
    if args.scope in ('all', 'sim'):
        # 4. Run all 4 scenarios sequentially (no parallelism — easier to debug)
        print(f'\n[2/4] Running {len(SCENARIOS)} core scenarios sequentially...')
        sim_results = {}
        for scenario_name, improvement_factor in SCENARIOS:
            if improvement_factor is None:
                improvement_factor = args.forecast_improvement / 100.0
            strategy = _build_strategy(scenario_name, DEFAULT_STRATEGY['type'])
            t0 = time.time()
            result = simulator.run(price_df, strategy, improvement_factor=improvement_factor)
            sim_results[scenario_name] = result
            print(f'  {scenario_name:18s} revenue=${result.total_revenue:>10,.0f}  '
                  f'charge={result.charge_count:4d}  discharge={result.discharge_count:4d}  '
                  f'({time.time() - t0:.1f}s)')
        with gzip.open(sim_path, 'wb') as f:
            pickle.dump(sim_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'  ✓ {sim_path.relative_to(Path(__file__).parent.parent)}  '
              f'({sim_path.stat().st_size / 1024:.1f} KB)')
        extra_manifest['scenarios'] = {
            k: {
                'total_revenue': round(float(v.total_revenue), 2),
                'charge_count': int(v.charge_count),
                'discharge_count': int(v.discharge_count),
                'hold_count': int(v.hold_count),
            }
            for k, v in sim_results.items()
        }

    if args.scope in ('all', 'strategy'):
        print(f'\n[3/4] Strategy Analysis sensitivity sweeps...')
        strategy_results = _precompute_strategy_analysis(simulator, price_df, args)
        strat_path = output_dir / 'strategy_default.pkl.gz' if args.output_dir else STRATEGY_PICKLE_PATH
        with gzip.open(strat_path, 'wb') as f:
            pickle.dump(strategy_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'  ✓ {strat_path.relative_to(Path(__file__).parent.parent)}  '
              f'({strat_path.stat().st_size / 1024:.1f} KB)')
        extra_manifest['strategy_analysis'] = {
            'sensitivity_points': len(strategy_results['sensitivity']['improvement_range']),
            'horizon_points': len(strategy_results['horizon_sensitivity']['horizon_range']),
        }

    if args.scope in ('all', 'nodal'):
        print(f'\n[4/4] Nodal Analysis cross-node scan...')
        nodal_results = _precompute_nodal_analysis(start_date, end_date)
        nodal_path = output_dir / 'nodal_default.pkl.gz' if args.output_dir else NODAL_PICKLE_PATH
        with gzip.open(nodal_path, 'wb') as f:
            pickle.dump(nodal_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'  ✓ {nodal_path.relative_to(Path(__file__).parent.parent)}  '
              f'({nodal_path.stat().st_size / 1024:.1f} KB)')
        results_df, node_data_cache = nodal_results
        extra_manifest['nodal_analysis'] = {
            'nodes_ranked': len(results_df),
            'nodes_cached': len(node_data_cache),
            'top_node': str(results_df.iloc[0]['Node']) if not results_df.empty else None,
        }

    # 5. Manifest (always rewritten — covers whichever scopes ran)
    signature = _build_signature(
        args.node, str(start_date), str(end_date), args.forecast_improvement,
    )
    write_manifest(signature, extra=extra_manifest)
    print(f'\n  ✓ {(output_dir / "manifest.json").relative_to(Path(__file__).parent.parent)}')

    print('\nDone. Reload the dashboard to verify cold-start time drops on covered pages.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

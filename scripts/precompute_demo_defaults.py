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

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Add project root to path so we can import the same modules the app uses.
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import DEFAULT_BATTERY, DEFAULT_NODE, DEFAULT_STRATEGY  # noqa: E402
from core.battery.battery import BatterySpecs  # noqa: E402
from core.battery.simulator import BatterySimulator  # noqa: E402
from core.battery.strategies import (  # noqa: E402
    ClippingAwareMPCStrategy,
    ClippingOnlyStrategy,
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
    SWEEP_PICKLE_PATH,
    write_manifest,
)

# Default Asset Design solar config — mirrors the number_input defaults in
# pages/5_🏗️_Asset_Design.py: solar = 1.5× battery power, POI = 1× battery power.
ASSET_DESIGN_GRID_RESOLUTION = '10×10 (100 configs, ~50s)'  # mirrors page default


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


def _resolve_current_asset_specs(loader, node: str):
    """Mirror ui/components/sidebar/_battery_config._get_preset_values for
    the "Current Asset" preset: capacity/power come from
    engie_storage_assets row for `node`; efficiency / SOC bounds come from
    DEFAULT_BATTERY.

    Returns BatterySpecs ready to be passed to BatterySimulator.
    """
    engie = loader.load_engie_assets()
    capacity_mwh = float(DEFAULT_BATTERY['capacity_mwh'])
    power_mw = float(DEFAULT_BATTERY['power_mw'])
    if not engie.empty:
        match = engie[engie['settlement_point'] == node]
        if not match.empty:
            row = match.iloc[0]
            # Page casts to int() — mirror that so the signature matches exactly.
            if row.get('nameplate_power_mw') is not None:
                power_mw = float(int(row['nameplate_power_mw']))
            if row.get('nameplate_energy_mwh') is not None:
                capacity_mwh = float(int(row['nameplate_energy_mwh']))
    return BatterySpecs(
        capacity_mwh=capacity_mwh,
        power_mw=power_mw,
        efficiency=DEFAULT_BATTERY['efficiency'],
        min_soc=DEFAULT_BATTERY['min_soc'],
        max_soc=DEFAULT_BATTERY['max_soc'],
        initial_soc=DEFAULT_BATTERY['initial_soc'],
    )


def _build_signature(node, start_date_str, end_date_str, forecast_improvement_pct, specs):
    """Mirror utils/demo_defaults.state_signature() exactly."""

    def r(x):
        return round(float(x), 6)

    return (
        node,
        start_date_str,
        end_date_str,
        r(specs.capacity_mwh),
        r(specs.power_mw),
        r(specs.efficiency),
        r(specs.min_soc),
        r(specs.max_soc),
        r(specs.initial_soc),
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


def _build_solar_profile_db(loader, df_prices, node):
    """Mirror pages/5_🏗️_Asset_Design.py:get_smart_solar_profile database path.

    Returns a DataFrame indexed like df_prices with columns
    ['gen_mw', 'forecast_mw', 'potential_mw'] normalized 0-1.

    When the generation table only covers part of the requested window
    (very common — generation is backfilled less aggressively than prices),
    the missing days are filled by tiling the average per-time-of-day
    pattern computed from the available data. This keeps the realistic
    day/night shape across the full window instead of flat-zero gaps.
    """
    target_index = df_prices.index
    start_date = target_index.min().date()
    end_date = target_index.max().date()

    df_gen = loader.load_generation_data(
        node=node, fuel_type='Solar', start_date=start_date, end_date=end_date,
    )
    result = pd.DataFrame(index=target_index, data={'gen_mw': 0.0, 'forecast_mw': 0.0, 'potential_mw': 0.0})
    if df_gen.empty:
        return result

    if df_gen.index.tz is not None:
        df_gen.index = df_gen.index.tz_convert(None)

    for col in ('gen_mw', 'forecast_mw', 'potential_mw'):
        if col not in df_gen.columns:
            continue
        aligned = df_gen[col].reindex(target_index)
        if aligned.notna().any() and aligned.isna().any():
            available = aligned.dropna()
            daily_template = available.groupby(available.index.time).mean()
            missing_mask = aligned.isna()
            time_keys = pd.Series(aligned.index.time, index=aligned.index)[missing_mask]
            fill_values = time_keys.map(daily_template).astype(float)
            aligned = aligned.copy()
            aligned[missing_mask] = fill_values
        aligned = aligned.interpolate(method='time').fillna(0)
        mx = aligned.max()
        result[col] = (aligned / mx) if mx > 0 else aligned
    return result


def _build_df_hybrid(df_prices_indexed, solar_profile, solar_capacity_mw, interconnection_limit_mw):
    """Replicate pages/5_🏗️_Asset_Design.py lines 306-309 + 384-387 (df_hybrid)."""
    df = df_prices_indexed.copy()
    df['Solar_MW'] = solar_profile['potential_mw'] * solar_capacity_mw
    df['Export_MW'] = pd.concat([df['Solar_MW'], pd.Series(interconnection_limit_mw, index=df.index)], axis=1).min(axis=1)
    df['Clipped_MW'] = (df['Solar_MW'] - interconnection_limit_mw).clip(lower=0)
    df['clipped_mw'] = df['Clipped_MW']  # lowercase variant the strategy reads
    return df


def _compute_base_solar_revenue(df_hybrid, dt_hours):
    """Replicate pages/5_🏗️_Asset_Design.py lines 367-369."""
    return float((df_hybrid['Export_MW'] * df_hybrid['price_mwh_rt'] * dt_hours).sum())


def _simulate_clipping_only(df_hybrid, power_mw, duration_h, battery_eff, base_solar_revenue,
                             strategy_type, state_horizon, state_window,
                             state_charge_pct, state_discharge_pct):
    """Replicate pages/5_🏗️_Asset_Design.py:simulate_battery_config (lines 391-469)."""
    capacity_mwh = power_mw * duration_h
    specs = BatterySpecs(
        capacity_mwh=capacity_mwh,
        power_mw=power_mw,
        efficiency=battery_eff,
        initial_soc=0.05,
    )

    if strategy_type == 'Threshold-Based':
        base = ThresholdStrategy(state_charge_pct, state_discharge_pct)
    elif strategy_type == 'Rolling Window Optimization':
        base = RollingWindowStrategy(state_window)
    elif strategy_type == 'MPC (Rolling Horizon)':
        base = ClippingAwareMPCStrategy(state_horizon)
    else:
        base = RollingWindowStrategy(window_hours=6)

    strategy = ClippingOnlyStrategy(base_strategy=base)
    simulator = BatterySimulator(specs)
    result = simulator.run(df_hybrid, strategy, improvement_factor=0.0)
    metadata = result.metadata or {}
    return {
        'power_mw': power_mw,
        'duration_h': duration_h,
        'capacity_mwh': capacity_mwh,
        'battery_revenue': float(result.total_revenue),
        'total_revenue': float(base_solar_revenue + result.total_revenue),
        'clipped_captured': float(metadata.get('clipped_energy_captured', 0)),
        'curtailed_clipping': float(metadata.get('curtailed_clipping', 0)),
        'grid_arbitrage': float(metadata.get('grid_arbitrage_revenue', 0)),
    }


def _precompute_asset_design(loader, df_prices, specs, args):
    """Run current_res + a fixed 10×10 sweep for the demo defaults.

    Saves shape:
      {
        'current_res': dict,         # simulate_battery_config(current_power, current_duration)
        'sweep_results': dict,       # {(p, d): result_dict, ...} keys are tuples
        'config': {                  # what config produced this — used for verification
            'solar_capacity_mw', 'interconnection_limit_mw',
            'grid_resolution', 'base_solar_revenue', 'battery_eff',
            'power_range', 'duration_range',
        },
      }
    """
    print('\n[Asset Design] Building solar profile + df_hybrid...')
    # Page uses drop=False so 'timestamp' stays as both index and column —
    # the simulator's _initialize_simulation_state reads price_df.iloc[0]['timestamp'].
    if 'timestamp' in df_prices.columns and not isinstance(df_prices.index, pd.DatetimeIndex):
        df_prices_indexed = df_prices.set_index('timestamp', drop=False).sort_index()
    else:
        df_prices_indexed = df_prices
    df_prices_indexed = df_prices_indexed[~df_prices_indexed.index.duplicated(keep='first')]

    solar_profile = _build_solar_profile_db(loader, df_prices_indexed, args.node)
    if solar_profile['potential_mw'].sum() == 0:
        print(f'  ⚠ No solar generation data found for {args.node} — skipping Asset Design precompute.')
        return None

    current_power = float(specs.power_mw)
    current_duration = float(specs.capacity_mwh / specs.power_mw) if specs.power_mw > 0 else 0.0
    solar_capacity_mw = current_power * 1.5
    interconnection_limit_mw = current_power

    df_hybrid = _build_df_hybrid(df_prices_indexed, solar_profile, solar_capacity_mw, interconnection_limit_mw)
    dt_hours = float((df_hybrid.index[1] - df_hybrid.index[0]).total_seconds() / 3600.0)
    base_solar_revenue = _compute_base_solar_revenue(df_hybrid, dt_hours)

    print(f'  solar={solar_capacity_mw:.0f} MW, POI={interconnection_limit_mw:.0f} MW, '
          f'base_solar_revenue=${base_solar_revenue:,.0f}, dt={dt_hours:.2f}h')
    print(f'  clipped: peak={df_hybrid["Clipped_MW"].max():.1f} MW, '
          f'total={df_hybrid["Clipped_MW"].sum() * dt_hours:,.0f} MWh')

    print('\n[Asset Design] Current Asset simulation...')
    t0 = time.time()
    current_res = _simulate_clipping_only(
        df_hybrid, current_power, current_duration, specs.efficiency, base_solar_revenue,
        DEFAULT_STRATEGY['type'], int(DEFAULT_STRATEGY['horizon_hours']),
        int(DEFAULT_STRATEGY['window_hours']),
        float(DEFAULT_STRATEGY['charge_percentile']), float(DEFAULT_STRATEGY['discharge_percentile']),
    )
    print(f'  current ({current_power:.0f} MW × {current_duration:.2f}h)  '
          f'battery=${current_res["battery_revenue"]:>9,.0f}  '
          f'total=${current_res["total_revenue"]:>10,.0f}  ({time.time() - t0:.1f}s)')

    # Fixed 10x10 sweep (no adaptive expansion — that's a UI-only feature).
    # Bounds chosen to bracket the heuristic the page would compute: power range
    # 0.5×–2.5× current_power, duration 0.5h–4h.
    n_p, n_d = 10, 10
    p_min, p_max = max(5.0, current_power * 0.3), current_power * 2.0
    d_min, d_max = 0.5, 4.0
    power_range = list(np.linspace(p_min, p_max, n_p).round(1))
    duration_range = list(np.linspace(d_min, d_max, n_d).round(2))
    print(f'\n[Asset Design] Sweep {n_p}×{n_d}={n_p * n_d} configs '
          f'(power {p_min:.0f}-{p_max:.0f} MW, duration {d_min:.1f}-{d_max:.1f}h)...')

    p_step = (p_max - p_min) / (n_p - 1)
    d_step = (d_max - d_min) / (n_d - 1)
    sweep_results = {}
    t0 = time.time()
    total = n_p * n_d
    done = 0
    for p in power_range:
        for d in duration_range:
            done += 1
            res = _simulate_clipping_only(
                df_hybrid, float(p), float(d), specs.efficiency, base_solar_revenue,
                DEFAULT_STRATEGY['type'], int(DEFAULT_STRATEGY['horizon_hours']),
                int(DEFAULT_STRATEGY['window_hours']),
                float(DEFAULT_STRATEGY['charge_percentile']),
                float(DEFAULT_STRATEGY['discharge_percentile']),
            )
            res['p_step'] = p_step
            res['d_step'] = d_step
            res['is_skipped'] = False
            sweep_results[(float(p), float(d))] = res
            if done % 10 == 0 or done == total:
                print(f'  [{done:3d}/{total}] best so far: '
                      f'${max(r["total_revenue"] for r in sweep_results.values()):>10,.0f}')
    elapsed = time.time() - t0
    best_key, best_res = max(sweep_results.items(), key=lambda kv: kv[1]['total_revenue'])
    print(f'  → {elapsed:.1f}s. Optimal: {best_key[0]:.0f} MW × {best_key[1]:.1f}h '
          f'= ${best_res["total_revenue"]:,.0f}')

    # --- Curtailment Elimination: Minimum Battery Configurations ---
    # Replicates the spinner block in pages/5_🏗️_Asset_Design.py around
    # line 1509: builds the curtailment frontier, finds Min Power and Min
    # Capacity configs that hit ~0% curtailment, simulates each.
    print('\n[Asset Design] Curtailment frontier + min-config sims...')
    from core.analytics.curtailment_optimizer import calculate_curtailment_frontier
    t0 = time.time()
    peak_clipping = float(df_hybrid['Clipped_MW'].quantile(0.99))
    power_search = np.linspace(peak_clipping * 0.95, peak_clipping * 3.0, 20)
    duration_search = np.linspace(0.1, 16.0, 160)
    frontier_df = calculate_curtailment_frontier(
        df_hybrid['Clipped_MW'],
        df_hybrid['price_mwh_rt'],
        efficiency=specs.efficiency,
        power_range=power_search,
        duration_range=duration_search,
    )
    print(f'  frontier rows={len(frontier_df)}  built in {time.time() - t0:.1f}s')

    valid_configs = frontier_df[frontier_df['curtailment_pct'] < 0.1].copy()
    curtailment_payload = None
    if not valid_configs.empty:
        frontier_curve = valid_configs.loc[
            valid_configs.groupby('power_mw')['duration_h'].idxmin()
        ].copy().sort_values('power_mw')
        min_power_row = frontier_curve.iloc[0]
        min_power_p = float(min_power_row['power_mw'])
        min_power_d = float(min_power_row['duration_h'])
        min_capacity_row = frontier_curve.loc[frontier_curve['capacity_mwh'].idxmin()]
        min_capacity_p = float(min_capacity_row['power_mw'])
        min_capacity_d = float(min_capacity_row['duration_h'])

        t0 = time.time()
        min_power_res = _simulate_clipping_only(
            df_hybrid, min_power_p, min_power_d, specs.efficiency, base_solar_revenue,
            DEFAULT_STRATEGY['type'], int(DEFAULT_STRATEGY['horizon_hours']),
            int(DEFAULT_STRATEGY['window_hours']),
            float(DEFAULT_STRATEGY['charge_percentile']),
            float(DEFAULT_STRATEGY['discharge_percentile']),
        )
        min_capacity_res = _simulate_clipping_only(
            df_hybrid, min_capacity_p, min_capacity_d, specs.efficiency, base_solar_revenue,
            DEFAULT_STRATEGY['type'], int(DEFAULT_STRATEGY['horizon_hours']),
            int(DEFAULT_STRATEGY['window_hours']),
            float(DEFAULT_STRATEGY['charge_percentile']),
            float(DEFAULT_STRATEGY['discharge_percentile']),
        )
        print(f'  min_power   {min_power_p:.1f} MW × {min_power_d:.2f}h  '
              f'rev=${min_power_res["total_revenue"]:>10,.0f}')
        print(f'  min_capacity {min_capacity_p:.1f} MW × {min_capacity_d:.2f}h  '
              f'rev=${min_capacity_res["total_revenue"]:>10,.0f}  ({time.time() - t0:.1f}s)')

        curtailment_payload = {
            'frontier_curve': frontier_curve.reset_index(drop=True),
            'peak_clipping_mw': peak_clipping,
            'min_power': {'power_mw': min_power_p, 'duration_h': min_power_d,
                          'capacity_mwh': min_power_p * min_power_d, 'res': min_power_res},
            'min_capacity': {'power_mw': min_capacity_p, 'duration_h': min_capacity_d,
                             'capacity_mwh': min_capacity_p * min_capacity_d, 'res': min_capacity_res},
        }

    return {
        'current_res': current_res,
        'sweep_results': sweep_results,
        'curtailment': curtailment_payload,
        'config': {
            'solar_capacity_mw': solar_capacity_mw,
            'interconnection_limit_mw': interconnection_limit_mw,
            'grid_resolution': ASSET_DESIGN_GRID_RESOLUTION,
            'base_solar_revenue': base_solar_revenue,
            'battery_eff': specs.efficiency,
            'power_range': power_range,
            'duration_range': duration_range,
            'p_step': p_step,
            'd_step': d_step,
            'dt_hours': dt_hours,
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
    parser.add_argument('--scope', choices=['all', 'sim', 'strategy', 'nodal', 'asset_design'], default='all',
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

    # 3. Build battery + simulator — use "Current Asset" preset for the demo node
    # so the precompute signature matches what render_battery_config produces.
    specs = _resolve_current_asset_specs(loader, args.node)
    print(f'Battery:     {specs.power_mw:.0f} MW / {specs.capacity_mwh:.0f} MWh @ '
          f'{specs.efficiency:.0%}  (from "Current Asset" preset for {args.node})')
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
        print(f'\n[4/5] Nodal Analysis cross-node scan...')
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

    if args.scope in ('all', 'asset_design'):
        print(f'\n[5/5] Asset Design (current_res + fixed-grid sweep)...')
        ad_results = _precompute_asset_design(loader, price_df, specs, args)
        if ad_results is None:
            print('  ⚠ Skipped — no solar generation data for this node.')
        else:
            sweep_path = output_dir / 'sweep_default.pkl.gz' if args.output_dir else SWEEP_PICKLE_PATH
            with gzip.open(sweep_path, 'wb') as f:
                pickle.dump(ad_results, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'  ✓ {sweep_path.relative_to(Path(__file__).parent.parent)}  '
                  f'({sweep_path.stat().st_size / 1024:.1f} KB)')
            best_key, best_res = max(
                ad_results['sweep_results'].items(),
                key=lambda kv: kv[1]['total_revenue'],
            )
            extra_manifest['asset_design'] = {
                'solar_capacity_mw': ad_results['config']['solar_capacity_mw'],
                'interconnection_limit_mw': ad_results['config']['interconnection_limit_mw'],
                'grid_resolution': ad_results['config']['grid_resolution'],
                'current_total_revenue': round(ad_results['current_res']['total_revenue'], 2),
                'best_config': {
                    'power_mw': best_key[0], 'duration_h': best_key[1],
                    'total_revenue': round(best_res['total_revenue'], 2),
                    'battery_revenue': round(best_res['battery_revenue'], 2),
                },
                'sweep_size': len(ad_results['sweep_results']),
            }

    # 5. Manifest (always rewritten — covers whichever scopes ran)
    signature = _build_signature(
        args.node, str(start_date), str(end_date), args.forecast_improvement, specs,
    )
    write_manifest(signature, extra=extra_manifest)
    print(f'\n  ✓ {(output_dir / "manifest.json").relative_to(Path(__file__).parent.parent)}')

    print('\nDone. Reload the dashboard to verify cold-start time drops on covered pages.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

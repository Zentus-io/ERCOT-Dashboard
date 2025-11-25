"""
ERCOT Price Pattern Analysis
Zentus - ERCOT Battery Revenue Dashboard

Analyzes price patterns to test H1: ERCOT prices are simple/repetitive
enough that short horizons (2-4h) provide same value as long horizons (24h).

Tests:
- Autocorrelation at various lags
- Number of distinct local minima/maxima per day
- Price volatility and predictability
- Optimal arbitrage cycle length
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
from scipy.signal import find_peaks

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

    # Fetch RTM data (higher frequency, more representative)
    rtm_response = client.table("ercot_prices") \
        .select("timestamp,settlement_point,price_mwh") \
        .eq("settlement_point", node) \
        .eq("market", "RTM") \
        .gte("timestamp", start_date.isoformat()) \
        .lte("timestamp", end_date.isoformat()) \
        .order("timestamp") \
        .limit(10000) \
        .execute()

    if not rtm_response.data:
        raise ValueError(f"No RTM data found for {node}")

    df = pd.DataFrame(rtm_response.data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.rename(columns={'settlement_point': 'node'})

    return df


def calculate_autocorrelation(prices: pd.Series, max_lag_hours: int = 48) -> pd.DataFrame:
    """
    Calculate autocorrelation at various lags.

    Returns DataFrame with columns: lag_hours, autocorr
    """
    results = []

    # Assume hourly data (adjust if 15-min)
    freq_per_hour = 1  # Will auto-detect
    if len(prices) > 24:
        time_diff = (prices.index[1] - prices.index[0]).total_seconds() / 3600
        if time_diff < 1:
            freq_per_hour = int(1 / time_diff)  # e.g., 4 for 15-min data

    for lag_hours in [1, 2, 4, 6, 12, 24, 48]:
        if lag_hours <= max_lag_hours:
            lag_steps = lag_hours * freq_per_hour
            if lag_steps < len(prices):
                autocorr = prices.autocorr(lag=lag_steps)
                results.append({
                    'lag_hours': lag_hours,
                    'autocorr': autocorr
                })

    return pd.DataFrame(results)


def find_local_extrema(prices: pd.Series, prominence: float = 5.0) -> dict:
    """
    Find local minima and maxima in price series.

    Parameters
    ----------
    prices : pd.Series
        Price series
    prominence : float
        Minimum prominence ($/MWh) for a peak to be considered significant

    Returns
    -------
    dict
        - n_minima: number of local minima
        - n_maxima: number of local maxima
        - minima_indices: array of indices
        - maxima_indices: array of indices
        - avg_cycles_per_day: estimated charge/discharge opportunities per day
    """
    # Find maxima (discharge opportunities)
    maxima_indices, maxima_props = find_peaks(prices.values, prominence=prominence)

    # Find minima (charge opportunities) by inverting
    minima_indices, minima_props = find_peaks(-prices.values, prominence=prominence)

    # Estimate cycles per day
    total_days = len(prices) / (24 * 4)  # Assume 15-min data, adjust if hourly
    if len(prices) > 100:
        time_diff_hours = (prices.index[-1] - prices.index[0]).total_seconds() / 3600
        total_days = time_diff_hours / 24

    avg_cycles_per_day = min(len(minima_indices), len(maxima_indices)) / max(total_days, 1)

    return {
        'n_minima': len(minima_indices),
        'n_maxima': len(maxima_indices),
        'minima_indices': minima_indices,
        'maxima_indices': maxima_indices,
        'avg_cycles_per_day': avg_cycles_per_day,
        'total_days': total_days
    }


def analyze_price_volatility(prices: pd.Series) -> dict:
    """
    Analyze price volatility and predictability.

    Returns
    -------
    dict
        - mean_price: mean price ($/MWh)
        - std_price: standard deviation
        - cv: coefficient of variation (std/mean)
        - range: max - min
        - iqr: interquartile range
        - daily_volatility: average daily price range
    """
    stats = {
        'mean_price': prices.mean(),
        'std_price': prices.std(),
        'cv': prices.std() / prices.mean() if prices.mean() > 0 else 0,
        'range': prices.max() - prices.min(),
        'iqr': prices.quantile(0.75) - prices.quantile(0.25)
    }

    # Calculate daily volatility
    if 'timestamp' in prices.index.names or isinstance(prices.index, pd.DatetimeIndex):
        daily_ranges = prices.groupby(prices.index.date).agg(lambda x: x.max() - x.min())
        stats['daily_volatility'] = daily_ranges.mean()
    else:
        stats['daily_volatility'] = None

    return stats


def estimate_optimal_cycle_length(prices: pd.Series) -> dict:
    """
    Estimate optimal arbitrage cycle length based on price patterns.

    Logic:
    - Find average time between consecutive peaks
    - Find average time between consecutive troughs
    - Optimal cycle = time from trough to peak + time from peak to next trough
    """
    extrema = find_local_extrema(prices, prominence=5.0)

    minima_idx = extrema['minima_indices']
    maxima_idx = extrema['maxima_indices']

    if len(minima_idx) < 2 or len(maxima_idx) < 2:
        return {'optimal_cycle_hours': None, 'reason': 'Insufficient extrema found'}

    # Calculate average spacing
    minima_spacing = np.diff(minima_idx).mean() if len(minima_idx) > 1 else 0
    maxima_spacing = np.diff(maxima_idx).mean() if len(maxima_idx) > 1 else 0

    # Estimate time resolution (15-min = 0.25h, hourly = 1h)
    time_resolution_hours = 1.0  # Default to hourly
    if isinstance(prices.index, pd.DatetimeIndex) and len(prices) > 1:
        time_resolution_hours = (prices.index[1] - prices.index[0]).total_seconds() / 3600

    # Average cycle length = spacing between peaks (or troughs)
    avg_cycle_steps = (minima_spacing + maxima_spacing) / 2
    optimal_cycle_hours = avg_cycle_steps * time_resolution_hours

    return {
        'optimal_cycle_hours': optimal_cycle_hours,
        'minima_spacing_steps': minima_spacing,
        'maxima_spacing_steps': maxima_spacing,
        'time_resolution_hours': time_resolution_hours
    }


def test_hypothesis_h1(df: pd.DataFrame):
    """
    Test H1: ERCOT prices are simple/repetitive.

    If TRUE, we expect:
    - High autocorrelation at short lags (>0.7 at 4h)
    - Few distinct peaks per day (1-2 charge/discharge cycles)
    - Low volatility relative to mean
    - Optimal cycle length ~4-8 hours (matches battery duration)
    """
    print(f"\n{'='*60}")
    print(f"Testing H1: Price Pattern Simplicity")
    print(f"{'='*60}")

    df = df.set_index('timestamp')
    prices = df['price_mwh']

    # 1. Autocorrelation
    print(f"\n1. Autocorrelation Analysis")
    autocorr_df = calculate_autocorrelation(prices)
    print(autocorr_df.to_string(index=False))

    high_autocorr_4h = autocorr_df[autocorr_df['lag_hours'] == 4]['autocorr'].values
    if len(high_autocorr_4h) > 0 and high_autocorr_4h[0] > 0.7:
        print(f"\n⚠️  High autocorrelation at 4h lag ({high_autocorr_4h[0]:.2f}) - prices are predictable!")
        print(f"    This supports H1: Short horizons may be sufficient.")
    else:
        print(f"\n✓ Autocorrelation decays - longer horizons may help")

    # 2. Local extrema
    print(f"\n2. Local Extrema (Charge/Discharge Opportunities)")
    extrema = find_local_extrema(prices)
    print(f"  Total minima (charge opportunities): {extrema['n_minima']}")
    print(f"  Total maxima (discharge opportunities): {extrema['n_maxima']}")
    print(f"  Total days analyzed: {extrema['total_days']:.1f}")
    print(f"  Avg cycles per day: {extrema['avg_cycles_per_day']:.2f}")

    if extrema['avg_cycles_per_day'] < 2:
        print(f"\n⚠️  Low cycle frequency (<2 per day) - simple price pattern!")
        print(f"    This supports H1: Limited arbitrage opportunities.")
    else:
        print(f"\n✓ Multiple cycles per day - complex patterns exist")

    # 3. Volatility
    print(f"\n3. Price Volatility")
    volatility = analyze_price_volatility(prices)
    print(f"  Mean price: ${volatility['mean_price']:.2f}/MWh")
    print(f"  Std deviation: ${volatility['std_price']:.2f}")
    print(f"  Coefficient of variation: {volatility['cv']:.2f}")
    print(f"  Range: ${volatility['range']:.2f}")
    print(f"  IQR: ${volatility['iqr']:.2f}")
    if volatility['daily_volatility'] is not None:
        print(f"  Avg daily range: ${volatility['daily_volatility']:.2f}")

    # 4. Optimal cycle length
    print(f"\n4. Optimal Cycle Length")
    cycle_info = estimate_optimal_cycle_length(prices)
    if cycle_info['optimal_cycle_hours'] is not None:
        print(f"  Estimated optimal cycle: {cycle_info['optimal_cycle_hours']:.1f} hours")
        print(f"  Time resolution: {cycle_info['time_resolution_hours']:.2f} hours")

        if cycle_info['optimal_cycle_hours'] < 6:
            print(f"\n⚠️  Optimal cycle <6h - matches 2h battery + 4h horizon!")
            print(f"    This supports H1: Longer horizons don't help.")
        else:
            print(f"\n✓ Optimal cycle >{cycle_info['optimal_cycle_hours']:.0f}h - longer horizons beneficial")
    else:
        print(f"  Could not estimate cycle length: {cycle_info['reason']}")

    # Summary
    print(f"\n{'='*60}")
    print(f"H1 Diagnosis Summary")
    print(f"{'='*60}")

    h1_evidence_count = 0
    if len(high_autocorr_4h) > 0 and high_autocorr_4h[0] > 0.7:
        h1_evidence_count += 1
    if extrema['avg_cycles_per_day'] < 2:
        h1_evidence_count += 1
    if cycle_info['optimal_cycle_hours'] is not None and cycle_info['optimal_cycle_hours'] < 6:
        h1_evidence_count += 1

    if h1_evidence_count >= 2:
        print(f"\n✓ H1 LIKELY: Price patterns are simple/repetitive")
        print(f"  Evidence points: {h1_evidence_count}/3")
        print(f"  Conclusion: 4h MPC horizon may be sufficient for this data.")
        print(f"  Longer horizons won't improve performance if prices follow simple daily cycles.")
    else:
        print(f"\n✗ H1 UNLIKELY: Price patterns are complex")
        print(f"  Evidence points: {h1_evidence_count}/3")
        print(f"  Conclusion: Longer horizons SHOULD help - investigate other hypotheses.")


def main():
    """Run price pattern analysis."""
    print("\n" + "="*60)
    print("ERCOT PRICE PATTERN ANALYSIS")
    print("="*60)

    # Load data
    df = load_ercot_data(
        node="ALVIN_RN",
        start_date=date(2025, 11, 13),
        end_date=date(2025, 11, 20)
    )

    print(f"\nLoaded {len(df)} price points")
    print(f"Period: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Test H1
    test_hypothesis_h1(df)

    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

"""
Curtailment Elimination Optimizer

This module provides functions to calculate the minimum battery specifications
needed to eliminate all curtailment (absorb 100% of clipped energy).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def calculate_min_power_requirement(df_clipping: pd.Series, percentile: float = 99) -> float:
    """
    Calculate minimum power requirement to capture clipping.
    
    Parameters
    ----------
    df_clipping : pd.Series
        Time series of clipped power (MW)
    percentile : float
        Percentile to use for peak clipping (default 99 to ignore outliers)
        
    Returns
    -------
    float
        Minimum power required in MW
    """
    if df_clipping.empty or df_clipping.max() == 0:
        return 0.0
    
    return df_clipping.quantile(percentile / 100)


def calculate_min_capacity_simple(df_clipping: pd.Series, df_index: pd.DatetimeIndex) -> Dict[str, float]:
    """
    Calculate minimum capacity using simple daily maximum method (conservative estimate).
    
    Parameters
    ----------
    df_clipping : pd.Series
        Time series of clipped power (MW)
    df_index : pd.DatetimeIndex
        Time index for grouping by date
        
    Returns
    -------
    dict
        Contains 'capacity_mwh', 'max_daily_mwh', 'avg_daily_mwh'
    """
    # Convert MW to MWh for 15-min intervals
    df_clipping_mwh = df_clipping / 4
    
    # Group by date
    df_temp = pd.DataFrame({'date': df_index.date, 'clipped_mwh': df_clipping_mwh})
    daily_clipped = df_temp.groupby('date')['clipped_mwh'].sum()
    
    max_daily = daily_clipped.max() if not daily_clipped.empty else 0.0
    avg_daily = daily_clipped.mean() if not daily_clipped.empty else 0.0
    
    return {
        'capacity_mwh': max_daily,
        'max_daily_mwh': max_daily,
        'avg_daily_mwh': avg_daily,
        'method': 'simple_daily_max'
    }


def calculate_min_capacity_rolling_balance(
    df_clipping: pd.Series,
    df_prices: pd.Series,
    min_power_mw: float,
    efficiency: float = 0.95,
    discharge_percentile: float = 75
) -> Dict[str, float]:
    """
    Calculate minimum capacity using rolling energy balance simulation.
    
    This method simulates a simple charge/discharge strategy:
    - Charge: Absorb all clipping
    - Discharge: When price exceeds threshold
    
    The minimum capacity is the maximum SOC reached during simulation.
    
    Parameters
    ----------
    df_clipping : pd.Series
        Time series of clipped power (MW)
    df_prices : pd.Series
        Time series of prices ($/MWh)
    min_power_mw : float
        Minimum power rating for discharge
    efficiency : float
        Round-trip efficiency
    discharge_percentile : float
        Price percentile above which to discharge
        
    Returns
    -------
    dict
        Contains 'capacity_mwh', 'max_soc_mwh', 'discharge_threshold', 'avg_soc_mwh'
    """
    discharge_threshold = df_prices.quantile(discharge_percentile / 100)
    
    soc_mwh = 0.0
    max_soc = 0.0
    soc_history = []
    
    for idx in df_clipping.index:
        clipped_mw = df_clipping.loc[idx]
        price = df_prices.loc[idx]
        
        # Charge from clipping (0.25h interval for 15-min data)
        charge_energy = clipped_mw * 0.25 * efficiency
        soc_mwh += charge_energy
        
        # Discharge if price is above threshold and we have energy
        if price >= discharge_threshold and soc_mwh > 0:
            # Max discharge in this interval (0.25h)
            max_discharge = min_power_mw * 0.25
            actual_discharge = min(soc_mwh, max_discharge)
            soc_mwh -= actual_discharge
        
        # Track max SOC
        max_soc = max(max_soc, soc_mwh)
        soc_history.append(soc_mwh)
    
    avg_soc = np.mean(soc_history) if soc_history else 0.0
    
    return {
        'capacity_mwh': max_soc,
        'max_soc_mwh': max_soc,
        'avg_soc_mwh': avg_soc,
        'discharge_threshold': discharge_threshold,
        'method': 'rolling_balance'
    }


def calculate_curtailment_frontier(
    df_clipping: pd.Series,
    df_prices: pd.Series,
    efficiency: float = 0.95,
    power_range: np.ndarray = None,
    duration_range: np.ndarray = None
) -> pd.DataFrame:
    """
    Calculate the curtailment frontier: iso-curtailment curves showing
    power/capacity trade-offs for different curtailment levels.
    
    Parameters
    ----------
    df_clipping : pd.Series
        Time series of clipped power (MW)
    df_prices : pd.Series
        Time series of prices ($/MWh)
    efficiency : float
        Round-trip efficiency
    power_range : np.ndarray, optional
        Power values to test (MW)
    duration_range : np.ndarray, optional
        Duration values to test (hours)
        
    Returns
    -------
    pd.DataFrame
        Columns: power_mw, duration_h, capacity_mwh, curtailment_pct, curtailment_mwh
    """
    # Default ranges if not provided
    if power_range is None:
        max_clip = df_clipping.quantile(0.99)
        power_range = np.linspace(max_clip * 0.5, max_clip * 2, 10)
    
    if duration_range is None:
        duration_range = np.linspace(2, 12, 10)
    
    results = []
    total_clipped_mwh = (df_clipping.sum() / 4) * efficiency
    
    for power in power_range:
        for duration in duration_range:
            curtailment = _simulate_curtailment(
                df_clipping, df_prices, power, duration, efficiency
            )
            curtailment_pct = (curtailment / max(total_clipped_mwh, 1)) * 100
            
            results.append({
                'power_mw': power,
                'duration_h': duration,
                'capacity_mwh': power * duration,
                'curtailment_mwh': curtailment,
                'curtailment_pct': curtailment_pct
            })
    
    return pd.DataFrame(results)


def _simulate_curtailment(
    df_clipping: pd.Series,
    df_prices: pd.Series,
    power_mw: float,
    duration_h: float,
    efficiency: float
) -> float:
    """
    Simplified curtailment simulation for frontier calculation.
    
    Returns
    -------
    float
        Total curtailed energy in MWh
    """
    capacity_mwh = power_mw * duration_h
    discharge_threshold = df_prices.quantile(0.75)
    
    soc_mwh = 0.0
    curtailed = 0.0
    
    for idx in df_clipping.index:
        clipped_mw = df_clipping.loc[idx]
        price = df_prices.loc[idx]
        
        # Attempt to charge
        desired_charge = clipped_mw * 0.25 * efficiency
        available_capacity = capacity_mwh - soc_mwh
        
        # Charge limited by power and available capacity
        max_charge = min(power_mw * 0.25, available_capacity, desired_charge)
        soc_mwh += max_charge
        
        # Track curtailment
        curtailed += (desired_charge - max_charge) / efficiency
        
        # Discharge if profitable
        if price >= discharge_threshold and soc_mwh > 0:
            max_discharge = min(power_mw * 0.25, soc_mwh)
            soc_mwh -= max_discharge
    
    return curtailed


def calculate_capex_optimized_specs(
    curtailment_frontier: pd.DataFrame,
    power_cost_per_mw: float,
    energy_cost_per_mwh: float,
    max_curtailment_pct: float = 1.0
) -> Dict[str, float]:
    """
    Find the minimum CAPEX configuration that achieves target curtailment.
    
    Parameters
    ----------
    curtailment_frontier : pd.DataFrame
        Output from calculate_curtailment_frontier
    power_cost_per_mw : float
        Cost per MW of power capacity ($)
    energy_cost_per_mwh : float
        Cost per MWh of energy capacity ($)
    max_curtailment_pct : float
        Maximum acceptable curtailment percentage (default 1%)
        
    Returns
    -------
    dict
        Contains 'power_mw', 'duration_h', 'capacity_mwh', 'capex', 'curtailment_pct'
    """
    # Filter to configs meeting curtailment target
    viable = curtailment_frontier[
        curtailment_frontier['curtailment_pct'] <= max_curtailment_pct
    ].copy()
    
    if viable.empty:
        return {
            'power_mw': None,
            'duration_h': None,
            'capacity_mwh': None,
            'capex': None,
            'curtailment_pct': None,
            'error': 'No configuration meets curtailment target'
        }
    
    # Calculate CAPEX for each
    viable['capex'] = (
        viable['power_mw'] * power_cost_per_mw +
        viable['capacity_mwh'] * energy_cost_per_mwh
    )
    
    # Find minimum CAPEX
    optimal = viable.loc[viable['capex'].idxmin()]
    
    return {
        'power_mw': optimal['power_mw'],
        'duration_h': optimal['duration_h'],
        'capacity_mwh': optimal['capacity_mwh'],
        'capex': optimal['capex'],
        'curtailment_pct': optimal['curtailment_pct']
    }


def summarize_curtailment_elimination(
    df_clipping: pd.Series,
    df_prices: pd.Series,
    df_index: pd.DatetimeIndex,
    efficiency: float = 0.95
) -> Dict:
    """
    Complete curtailment elimination analysis combining all methods.
    
    Returns
    -------
    dict
        Comprehensive analysis with min_power, min_capacity (simple & rolling),
        and recommendations
    """
    # Step 1: Minimum Power
    min_power = calculate_min_power_requirement(df_clipping)
    
    # Step 2: Minimum Capacity (Simple)
    capacity_simple = calculate_min_capacity_simple(df_clipping, df_index)
    
    # Step 3: Minimum Capacity (Rolling Balance)
    capacity_rolling = calculate_min_capacity_rolling_balance(
        df_clipping, df_prices, min_power, efficiency
    )
    
    # Step 4: Recommendations
    # Use rolling balance as primary (more accurate)
    recommended_capacity = capacity_rolling['capacity_mwh']
    recommended_duration = recommended_capacity / max(min_power, 1)
    
    return {
        'min_power_mw': min_power,
        'min_capacity_simple_mwh': capacity_simple['capacity_mwh'],
        'min_capacity_rolling_mwh': capacity_rolling['capacity_mwh'],
        'recommended_power_mw': min_power,
        'recommended_capacity_mwh': recommended_capacity,
        'recommended_duration_h': recommended_duration,
        'discharge_threshold_price': capacity_rolling.get('discharge_threshold', 0),
        'daily_stats': {
            'max_daily_clipped': capacity_simple['max_daily_mwh'],
            'avg_daily_clipped': capacity_simple['avg_daily_mwh']
        }
    }

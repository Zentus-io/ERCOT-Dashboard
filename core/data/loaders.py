"""
Data Loading Utilities
Zentus - ERCOT Battery Revenue Dashboard

This module provides data loading functions for ERCOT price data
and EIA battery market data.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import polars as pl


class DataLoader:
    """
    Centralized data loading with caching.

    This class handles loading of ERCOT price data and EIA battery data
    from CSV and Parquet files.

    Attributes
    ----------
    data_dir : Path
        Path to data directory
    """

    def __init__(self, data_dir: Path):
        """
        Initialize data loader.

        Parameters
        ----------
        data_dir : Path
            Path to directory containing data files
        """
        self.data_dir = Path(data_dir)

    def load_prices(self) -> pd.DataFrame:
        """
        Load and merge day-ahead and real-time prices.

        Returns
        -------
        pd.DataFrame
            Merged price data with columns:
            - timestamp: datetime
            - node: settlement point name
            - price_mwh_da: day-ahead price ($/MWh)
            - price_mwh_rt: real-time price ($/MWh)
            - forecast_error: RT - DA ($/MWh)
            - price_spread: |forecast_error| ($/MWh)
            - extreme_event: bool, True if spread > $10/MWh
        """
        # Load DA and RT prices
        da_prices = pd.read_csv(
            self.data_dir / 'da_prices.csv',
            parse_dates=['timestamp']
        )
        rt_prices = pd.read_csv(
            self.data_dir / 'rt_prices.csv',
            parse_dates=['timestamp']
        )

        # Merge on timestamp and node
        merged = da_prices.merge(
            rt_prices,
            on=['timestamp', 'node'],
            suffixes=('_da', '_rt')
        )

        # Calculate derived metrics
        merged['forecast_error'] = merged['price_mwh_rt'] - merged['price_mwh_da']
        merged['price_spread'] = abs(merged['forecast_error'])
        merged['extreme_event'] = merged['price_spread'] > 10

        return merged

    def load_eia_batteries(self) -> Optional[pd.DataFrame]:
        """
        Load EIA-860 battery storage market data for Texas (ERCOT).

        Tries to load from optimized Parquet file first, falls back to Excel.

        Returns
        -------
        pd.DataFrame or None
            Battery data with columns:
            - Nameplate Capacity (MW)
            - Nameplate Energy Capacity (MWh)
            - Duration (hours)
            - State
            - Arbitrage
            - Frequency Regulation
            - Ramping / Spinning Reserve
            Returns None if data not found.
        """
        try:
            # Try loading preprocessed Parquet file first (20-30x faster than Excel!)
            parquet_path = self.data_dir / 'processed' / 'ercot_batteries.parquet'

            if parquet_path.exists():
                # Load with Polars (fastest option)
                df_polars = pl.read_parquet(parquet_path)
                # Convert to pandas for compatibility
                df = df_polars.to_pandas()

                # Rename columns to match expected format
                df = df.rename(columns={
                    'nameplate_power_mw': 'Nameplate Capacity (MW)',
                    'nameplate_energy_mwh': 'Nameplate Energy Capacity (MWh)',
                    'duration_hours': 'Duration (hours)',
                    'state': 'State',
                    'use_arbitrage': 'Arbitrage',
                    'use_frequency_regulation': 'Frequency Regulation',
                    'use_ramping_reserve': 'Ramping / Spinning Reserve'
                })

                return df

            else:
                # Fallback to Excel if Parquet not available
                file_path = self.data_dir / 'eia8602024' / '3_4_Energy_Storage_Y2024.xlsx'

                if not file_path.exists():
                    return None

                df = pd.read_excel(file_path, sheet_name='Operable', header=1)

                # Filter for Texas batteries
                texas = df[df['State'] == 'TX'].copy()

                # Calculate duration
                texas['Duration (hours)'] = (
                    texas['Nameplate Energy Capacity (MWh)'] /
                    texas['Nameplate Capacity (MW)']
                )

                return texas

        except Exception as e:
            # Graceful degradation if file not found
            print(f"Warning: Could not load EIA battery data: {e}")
            return None

    def get_nodes(self, price_df: pd.DataFrame) -> list:
        """
        Get list of available settlement point nodes.

        Parameters
        ----------
        price_df : pd.DataFrame
            Price dataframe with 'node' column

        Returns
        -------
        list
            Sorted list of unique node names
        """
        return sorted(price_df['node'].unique())

    def filter_by_node(self, price_df: pd.DataFrame, node: str) -> pd.DataFrame:
        """
        Filter price data for a specific settlement node.

        Parameters
        ----------
        price_df : pd.DataFrame
            Full price data
        node : str
            Settlement point name

        Returns
        -------
        pd.DataFrame
            Filtered and sorted data for the specified node
        """
        node_data = price_df[price_df['node'] == node].copy()
        node_data = node_data.sort_values('timestamp').reset_index(drop=True)
        return node_data

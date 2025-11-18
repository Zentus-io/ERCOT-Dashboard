"""
Data Loading Utilities
Zentus - ERCOT Battery Revenue Dashboard

This module provides data loading functions for ERCOT price data
and EIA battery market data from both CSV files and Supabase database.
"""

from pathlib import Path
from typing import Optional, Literal
from datetime import datetime, date, timedelta
import pandas as pd
import polars as pl
from supabase import create_client, Client
from config.settings import (
    SUPABASE_URL,
    SUPABASE_KEY,
    DB_TIMEOUT,
    DEFAULT_DATA_SOURCE,
    DEFAULT_DAYS_BACK
)


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


class SupabaseDataLoader:
    """
    Data loader for Supabase database.

    Provides methods to query ERCOT price data from Supabase with
    date range filtering and caching.

    Attributes
    ----------
    client : Client
        Supabase client instance
    """

    def __init__(self):
        """Initialize Supabase client."""
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError(
                "Supabase credentials not configured. "
                "Set SUPABASE_URL and SUPABASE_KEY in .env file"
            )

        self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    def load_prices(
        self,
        node: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Load price data from Supabase database.

        Parameters
        ----------
        node : str, optional
            Settlement point to filter by. If None, returns all nodes.
        start_date : date, optional
            Start date for query. If None, defaults to DEFAULT_DAYS_BACK ago.
        end_date : date, optional
            End date for query. If None, defaults to today.

        Returns
        -------
        pd.DataFrame
            Price data with columns matching CSV format:
            - timestamp: datetime
            - node: settlement point name
            - price_mwh_da: day-ahead price ($/MWh)
            - price_mwh_rt: real-time price ($/MWh)
            - forecast_error: RT - DA ($/MWh)
            - price_spread: |forecast_error| ($/MWh)
            - extreme_event: bool, True if spread > $10/MWh
        """
        # Set date defaults
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=DEFAULT_DAYS_BACK)

        try:
            # Query the merged view (pre-joined DAM + RTM)
            query = self.client.table("ercot_prices_merged").select("*")

            # Add filters
            if node:
                query = query.eq("node", node)

            query = query.gte("timestamp", start_date.isoformat())
            query = query.lte("timestamp", end_date.isoformat())
            query = query.order("timestamp")

            # Execute query
            response = query.execute()

            if not response.data:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(response.data)

            # Parse timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Ensure expected columns exist
            required_cols = [
                'timestamp', 'node', 'price_mwh_da', 'price_mwh_rt',
                'forecast_error', 'price_spread', 'extreme_event'
            ]

            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in database: {missing_cols}")

            return df

        except Exception as e:
            print(f"Error loading data from Supabase: {e}")
            raise

    def get_available_nodes(self) -> list[str]:
        """
        Get list of available settlement points from database.

        Returns
        -------
        list[str]
            Sorted list of unique node names
        """
        try:
            # Query distinct locations
            response = (
                self.client.table("ercot_prices")
                .select("location")
                .limit(1000)
                .execute()
            )

            if not response.data:
                return []

            # Extract unique locations
            nodes = sorted(set(row['location'] for row in response.data))
            return nodes

        except Exception as e:
            print(f"Error fetching available nodes: {e}")
            return []

    def get_date_range(self) -> tuple[Optional[date], Optional[date]]:
        """
        Get available date range from database.

        Returns
        -------
        tuple[date, date] or (None, None)
            Earliest and latest dates available in database
        """
        try:
            # Get earliest timestamp
            earliest_response = (
                self.client.table("ercot_prices")
                .select("timestamp")
                .order("timestamp", desc=False)
                .limit(1)
                .execute()
            )

            # Get latest timestamp
            latest_response = (
                self.client.table("ercot_prices")
                .select("timestamp")
                .order("timestamp", desc=True)
                .limit(1)
                .execute()
            )

            if not earliest_response.data or not latest_response.data:
                return None, None

            earliest = pd.to_datetime(earliest_response.data[0]['timestamp']).date()
            latest = pd.to_datetime(latest_response.data[0]['timestamp']).date()

            return earliest, latest

        except Exception as e:
            print(f"Error fetching date range: {e}")
            return None, None


def load_data(
    source: Literal['csv', 'database'] = DEFAULT_DATA_SOURCE,
    node: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    data_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Unified data loading function supporting both CSV and database sources.

    Parameters
    ----------
    source : {'csv', 'database'}
        Data source to use
    node : str, optional
        Settlement point to filter by (database only)
    start_date : date, optional
        Start date for query (database only)
    end_date : date, optional
        End date for query (database only)
    data_dir : Path, optional
        Path to CSV data directory (CSV only). Defaults to ../data

    Returns
    -------
    pd.DataFrame
        Price data in standard format

    Raises
    ------
    ValueError
        If source is invalid or required parameters missing
    """
    if source == 'csv':
        if data_dir is None:
            # Default to standard data directory
            data_dir = Path(__file__).parent.parent.parent / 'data'

        loader = DataLoader(data_dir)
        df = loader.load_prices()

        # Apply node filter if specified
        if node:
            df = loader.filter_by_node(df, node)

        return df

    elif source == 'database':
        loader = SupabaseDataLoader()
        return loader.load_prices(node, start_date, end_date)

    else:
        raise ValueError(f"Invalid data source: {source}. Must be 'csv' or 'database'")

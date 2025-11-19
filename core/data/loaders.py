"""
Data Loading Utilities (Optimized for Lean Schema V2)
Zentus - ERCOT Battery Revenue Dashboard
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
    DEFAULT_DATA_SOURCE,
    DEFAULT_DAYS_BACK
)


class DataLoader:
    """
    Centralized data loading from local CSV files.
    """
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def load_prices(self) -> pd.DataFrame:
        """Loads and merges DA and RT prices from local CSV files."""
        da_prices_path = self.data_dir / 'da_prices.csv'
        rt_prices_path = self.data_dir / 'rt_prices.csv'

        if not da_prices_path.exists() or not rt_prices_path.exists():
            return pd.DataFrame()

        da_prices = pd.read_csv(da_prices_path, parse_dates=['timestamp'])
        rt_prices = pd.read_csv(rt_prices_path, parse_dates=['timestamp'])

        # Rename 'node' to 'settlement_point' to match new schema
        da_prices.rename(columns={'node': 'settlement_point'}, inplace=True)
        rt_prices.rename(columns={'node': 'settlement_point'}, inplace=True)
        
        merged = pd.merge(
            da_prices, rt_prices, on=['timestamp', 'settlement_point'], suffixes=('_da', '_rt')
        )

        merged['forecast_error'] = merged['price_mwh_rt'] - merged['price_mwh_da']
        merged['price_spread'] = abs(merged['forecast_error'])
        merged['extreme_event'] = merged['price_spread'] > 10
        return merged.rename(columns={'settlement_point': 'node'}) # Rename back to node for app consistency

    def load_eia_batteries(self) -> Optional[pd.DataFrame]:
        """Loads EIA-860 battery market data for Texas."""
        try:
            parquet_path = self.data_dir / 'processed' / 'ercot_batteries.parquet'
            if not parquet_path.exists(): return None

            df = pl.read_parquet(parquet_path).to_pandas()
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
        except Exception as e:
            print(f"Warning: Could not load EIA battery data: {e}")
            return None

    def get_nodes(self, price_df: pd.DataFrame) -> list:
        return sorted(price_df['node'].unique())

    def filter_by_node(self, price_df: pd.DataFrame, node: str) -> pd.DataFrame:
        node_data = price_df[price_df['node'] == node].copy()
        return node_data.sort_values('timestamp').reset_index(drop=True)


class SupabaseDataLoader:
    """
    Data loader for Supabase database with an optimized schema.
    Performs the join and calculations in Python.
    """
    def __init__(self):
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Supabase credentials not configured in .env file.")
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    def load_prices(
        self,
        node: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Loads and merges DAM and RTM price data from the Supabase 'ercot_prices' table.
        """
        if end_date is None: end_date = date.today()
        if start_date is None: start_date = end_date - timedelta(days=DEFAULT_DAYS_BACK)

        try:
            # Fetch DAM data
            dam_query = self.client.table("ercot_prices").select("timestamp, settlement_point, price_mwh").eq("market", "DAM")
            if node: dam_query = dam_query.eq("settlement_point", node)
            dam_query = dam_query.gte("timestamp", start_date.isoformat()).lte("timestamp", (end_date + timedelta(days=1)).isoformat())
            dam_data = dam_query.execute().data
            
            # Fetch RTM data
            rtm_query = self.client.table("ercot_prices").select("timestamp, settlement_point, price_mwh").eq("market", "RTM")
            if node: rtm_query = rtm_query.eq("settlement_point", node)
            rtm_query = rtm_query.gte("timestamp", start_date.isoformat()).lte("timestamp", (end_date + timedelta(days=1)).isoformat())
            rtm_data = rtm_query.execute().data

            if not dam_data or not rtm_data:
                return pd.DataFrame()

            # Convert to DataFrames
            dam_df = pd.DataFrame(dam_data).rename(columns={'price_mwh': 'price_mwh_da'})
            rtm_df = pd.DataFrame(rtm_data).rename(columns={'price_mwh': 'price_mwh_rt'})
            dam_df['timestamp'] = pd.to_datetime(dam_df['timestamp'])
            rtm_df['timestamp'] = pd.to_datetime(rtm_df['timestamp'])

            # Merge in Python
            merged = pd.merge(dam_df, rtm_df, on=['timestamp', 'settlement_point'])
            
            if merged.empty:
                return pd.DataFrame()

            # Calculate derived metrics
            merged['forecast_error'] = merged['price_mwh_rt'] - merged['price_mwh_da']
            merged['price_spread'] = abs(merged['forecast_error'])
            merged['extreme_event'] = merged['price_spread'] > 10
            
            # Rename for app consistency
            return merged.rename(columns={'settlement_point': 'node'}).sort_values('timestamp').reset_index(drop=True)

        except Exception as e:
            print(f"Error loading data from Supabase: {e}")
            raise

    def get_available_nodes(self) -> list[str]:
        """Gets list of available settlement points from the database."""
        try:
            response = self.client.table("ercot_prices").select("settlement_point").execute()
            if not response.data: return []
            nodes = set(str(row['settlement_point']) for row in response.data if row and 'settlement_point' in row)
            return sorted(list(nodes))
        except Exception as e:
            print(f"Error fetching available nodes: {e}")
            return []

    def get_date_range(self) -> tuple[Optional[date], Optional[date]]:
        """Gets available date range from the database."""
        try:
            earliest_res = self.client.table("ercot_prices").select("timestamp").order("timestamp").limit(1).execute()
            latest_res = self.client.table("ercot_prices").select("timestamp").order("timestamp", desc=True).limit(1).execute()

            if not earliest_res.data or not latest_res.data: return None, None
            earliest = pd.to_datetime(str(earliest_res.data[0]['timestamp'])).date()
            latest = pd.to_datetime(str(latest_res.data[0]['timestamp'])).date()
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
    """Unified data loading function supporting both CSV and database sources."""
    if source == 'csv':
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / 'data'
        loader = DataLoader(data_dir)
        df = loader.load_prices()
        if node:
            df = loader.filter_by_node(df, node)
        return df
    elif source == 'database':
        loader = SupabaseDataLoader()
        # In DB mode, the node filter is applied in the SQL query itself
        return loader.load_prices(node=node, start_date=start_date, end_date=end_date)
    else:
        raise ValueError(f"Invalid data source: {source}. Must be 'csv' or 'database'")
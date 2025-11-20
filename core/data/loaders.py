"""
Data Loading Utilities (Optimized for Lean Schema V2)
Zentus - ERCOT Battery Revenue Dashboard
"""

from pathlib import Path
from typing import Optional, Literal
from datetime import datetime, date, timedelta
import pandas as pd

import streamlit as st
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



    def get_nodes(self, price_df: pd.DataFrame) -> list:
        # Handle both 'node' and 'settlement_point' column names
        if 'node' in price_df.columns:
            return sorted(price_df['node'].unique())
        elif 'settlement_point' in price_df.columns:
            return sorted(price_df['settlement_point'].unique())
        else:
            return []

    def filter_by_node(self, price_df: pd.DataFrame, node: str) -> pd.DataFrame:
        # Handle both 'node' and 'settlement_point' column names
        if price_df.empty:
            return price_df

        if 'node' in price_df.columns:
            node_data = price_df[price_df['node'] == node].copy()
        elif 'settlement_point' in price_df.columns:
            node_data = price_df[price_df['settlement_point'] == node].copy()
        else:
            return pd.DataFrame()

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
        Loads DAM and RTM price data from the unified 'ercot_prices' table.
        Pivots the data so DAM and RTM prices are in separate columns.
        """
        if end_date is None: end_date = date.today()
        if start_date is None: start_date = end_date - timedelta(days=DEFAULT_DAYS_BACK)

        # Optimization: Require node selection for database queries to avoid hitting row limits
        if not node:
            return pd.DataFrame()

        try:
            # Fetch data for specific node
            query = self.client.table("ercot_prices").select("timestamp, settlement_point, market, price_mwh")
            query = query.eq("settlement_point", node)
            query = query.gte("timestamp", start_date.isoformat()).lte("timestamp", (end_date + timedelta(days=1)).isoformat())
            
            # Increase limit to ensure we get all data for the date range
            # 30 days * 24 hours * 4 intervals = ~2880 rows per market type
            # Total ~6000 rows max for one node. Supabase default is 1000.
            data = query.limit(10000).execute().data

            if not data:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Pivot: market values (DAM/RTM) become columns
            # Use RTM as the base (more granular 15-min data)
            rtm_df = df[df['market'] == 'RTM'].copy()
            dam_df = df[df['market'] == 'DAM'].copy()

            if rtm_df.empty:
                return pd.DataFrame()

            # Start with RTM data
            result = rtm_df[['timestamp', 'settlement_point', 'price_mwh']].copy()
            result.rename(columns={'price_mwh': 'price_mwh_rt'}, inplace=True)

            # Merge DAM data using nearest timestamp (DAM is hourly, RTM is 15-min)
            # Round RTM timestamps to the hour to match DAM
            if not dam_df.empty:
                dam_df = dam_df[['timestamp', 'settlement_point', 'price_mwh']].copy()
                dam_df.rename(columns={'price_mwh': 'price_mwh_da'}, inplace=True)

                # For each RTM record, find the matching DAM record at the hour
                result['hour'] = result['timestamp'].dt.floor('h')
                dam_df['hour'] = dam_df['timestamp'].dt.floor('h')

                # Merge on hour and settlement_point
                result = result.merge(
                    dam_df[['hour', 'settlement_point', 'price_mwh_da']],
                    on=['hour', 'settlement_point'],
                    how='left'
                )
                result.drop('hour', axis=1, inplace=True)
            else:
                # No DAM data, fill with NaN
                result['price_mwh_da'] = float('nan')

            # Calculate derived metrics
            result['forecast_error'] = result['price_mwh_rt'] - result['price_mwh_da']
            result['price_spread'] = abs(result['forecast_error'])
            result['extreme_event'] = result['price_spread'] > 10

            # Rename settlement_point to node for app consistency
            result = result.rename(columns={'settlement_point': 'node'}).sort_values('timestamp').reset_index(drop=True)

            return result

        except Exception as e:
            print(f"Error loading data from Supabase: {e}")
            import traceback
            traceback.print_exc()
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

    def get_node_availability(self, node: str) -> pd.DataFrame:
        """
        Fetches daily data availability for a specific node.
        Returns a DataFrame with 'date' and 'completeness' (0-100).
        """
        try:
            # Query to get counts per day
            # Note: This is a simplified approach. For large datasets, 
            # a database view or RPC would be more performant.
            # We fetch only timestamps for the node to minimize data transfer.
            response = self.client.table("ercot_prices") \
                .select("timestamp") \
                .eq("settlement_point", node) \
                .eq("market", "RTM") \
                .execute()

            if not response.data:
                return pd.DataFrame(columns=['date', 'completeness'])

            df = pd.DataFrame(response.data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            
            # Count records per day
            daily_counts = df.groupby('date').size().reset_index(name='count')
            
            # Calculate completeness (assuming 96 intervals per day for RTM)
            # Cap at 100%
            daily_counts['completeness'] = (daily_counts['count'] / 96 * 100).clip(upper=100)
            
            return daily_counts

        except Exception as e:
            print(f"Error fetching node availability: {e}")
            return pd.DataFrame(columns=['date', 'completeness'])

    @st.cache_data(ttl=36000)
    def load_eia_batteries(_self) -> pd.DataFrame:
        """
        Load EIA-860 battery data from Supabase.
        Cached for 10 hours.
        """
        try:
            # Fetch all batteries from Supabase
            response = _self.client.table("eia_batteries").select("*").execute()
            
            if not response.data:
                st.warning("No battery data found in Supabase.")
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(response.data)
            
            # Ensure numeric columns are properly typed
            numeric_cols = [
                'plant_code', 'utility_id', 'nameplate_power_mw', 
                'summer_power_mw', 'winter_power_mw', 'nameplate_energy_mwh',
                'max_charge_mw', 'max_discharge_mw', 'operating_month', 
                'operating_year', 'sector_code', 'duration_hours', 'e_to_p_ratio'
            ]
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Rename columns to match app expectations (legacy parquet format)
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
            st.error(f"Error loading battery data: {e}")
            return pd.DataFrame()


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
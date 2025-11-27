"""
Data Loading Utilities (Optimized for Lean Schema V2)
Zentus - ERCOT Battery Revenue Dashboard
"""

import traceback
from datetime import date, timedelta
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import polars as pl
import streamlit as st
from postgrest.exceptions import APIError
from supabase import Client, create_client

from config.settings import DEFAULT_DATA_SOURCE, DEFAULT_DAYS_BACK, SUPABASE_KEY, SUPABASE_URL


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
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=DEFAULT_DAYS_BACK)

        # Optimization: Require node selection for database queries to avoid hitting row limits
        if not node:
            return pd.DataFrame()

        try:
            # Fetch data for specific node
            query = self.client.table("ercot_prices").select(
                "timestamp, settlement_point, market, price_mwh"
            )
            query = query.eq("settlement_point", node)
            query = query.gte("timestamp", start_date.isoformat()).lte(
                "timestamp", (end_date + timedelta(days=1)).isoformat()
            )

            # Fetch all data using pagination to bypass 1000-row limit
            data = []
            start = 0
            batch_size = 1000
            
            while True:
                response = query.range(start, start + batch_size - 1).execute()
                if not response.data:
                    break
                data.extend(response.data)
                if len(response.data) < batch_size:
                    break
                start += batch_size

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
                # Ensure timestamps are datetimes and silence static type checker for .dt.floor
                result['timestamp'] = pd.to_datetime(result['timestamp'])
                dam_df['timestamp'] = pd.to_datetime(dam_df['timestamp'])
                result['hour'] = result['timestamp'].dt.floor('h')  # type: ignore[attr-defined]
                dam_df['hour'] = dam_df['timestamp'].dt.floor('h')  # type: ignore[attr-defined]

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
            result = result.rename(columns={'settlement_point': 'node'})
            result = result.sort_values('timestamp').reset_index(drop=True)

            return result

        except APIError as e:
            st.error(f"Database error in load_prices: {e}")
            raise
        except Exception as e:
            st.error(f"An unexpected error occurred in load_prices: {e}")
            traceback.print_exc()
            raise

    def get_available_nodes(self) -> list[str]:
        """Gets list of available settlement points from the database."""
        try:
            response = self.client.table("ercot_prices").select("settlement_point").execute()
            if not response.data:
                return []
            nodes = set()
            for row in response.data:
                if isinstance(row, dict) and "settlement_point" in row:
                    nodes.add(str(row["settlement_point"]))

            return sorted(list(nodes))
        except APIError as e:
            st.error(f"Database error fetching available nodes: {e}")
            return []
        except Exception as e:
            st.warning(f"An unexpected error occurred fetching available nodes: {e}")
            return []

    def get_date_range(self) -> tuple[Optional[date], Optional[date]]:
        """Gets available date range from the database."""
        try:
            earliest_res = (
                self.client.table("ercot_prices")
                .select("timestamp")
                .order("timestamp")
                .limit(1)
                .execute()
            )
            latest_res = (
                self.client.table("ercot_prices")
                .select("timestamp")
                .order("timestamp", desc=True)
                .limit(1)
                .execute()
            )

            if not earliest_res.data or not latest_res.data:
                return None, None
            # Convert to date type
            earliest_data = earliest_res.data[0]
            latest_data = latest_res.data[0]
            if isinstance(earliest_data, dict) and isinstance(latest_data, dict):
                earliest = pd.to_datetime(str(earliest_data['timestamp'])).date()
                latest = pd.to_datetime(str(latest_data['timestamp'])).date()
                return earliest, latest
            return None, None
        except APIError as e:
            st.error(f"Database error fetching date range: {e}")
            return None, None
        except Exception as e:
            st.warning(f"An unexpected error occurred fetching date range: {e}")
            return None, None

    def get_node_availability(
        self,
        node: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Fetches daily data availability for a specific node.
        Returns a DataFrame with 'date' and 'completeness' (0-100).
        
        Parameters
        ----------
        node : str
            Settlement point name
        start_date : Optional[date]
            Start date for availability query (inclusive)
        end_date : Optional[date]
            End date for availability query (inclusive)
        """
        try:
            # Query to get counts per day
            # Note: This is a simplified approach. For large datasets,
            # a database view or RPC would be more performant.
            # We fetch only timestamps for the node to minimize data transfer.
            query = self.client.table("ercot_prices") \
                .select("timestamp") \
                .eq("settlement_point", node) \
                .eq("market", "RTM")
            
            # Apply date range filters if provided
            if start_date:
                query = query.gte("timestamp", start_date.isoformat())
            if end_date:
                # Add one day to end_date to include the entire end day
                end_dt = date(end_date.year, end_date.month, end_date.day) + timedelta(days=1)
                query = query.lt("timestamp", end_dt.isoformat())
            
            # Fetch all data using pagination to bypass 1000-row limit
            all_data = []
            start = 0
            batch_size = 1000
            
            while True:
                # Range is inclusive
                response = query.range(start, start + batch_size - 1).execute()
                
                if not response.data:
                    break
                    
                all_data.extend(response.data)
                
                if len(response.data) < batch_size:
                    break
                    
                start += batch_size

            if not all_data:
                return pd.DataFrame(columns=['date', 'completeness'])

            df = pd.DataFrame(all_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date  # type: ignore[attr-defined]

            # Count records per day
            daily_counts = df.groupby('date').size().reset_index(name='count')

            # Calculate completeness (assuming 96 intervals per day for RTM)
            # Cap at 100%
            daily_counts['completeness'] = (daily_counts['count'] / 96 * 100).clip(upper=100)

            return daily_counts

        except APIError as e:
            st.error(f"Database error fetching node availability: {e}")
            return pd.DataFrame(columns=['date', 'completeness'])
        except Exception as e:
            st.warning(f"An unexpected error occurred fetching node availability: {e}")
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

        except APIError as e:
            st.error(f"Database error loading battery data: {e}")
            return pd.DataFrame()
        except Exception as e:
            st.warning(f"Unexpected error loading battery data: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=36000)
    def load_engie_assets(_self) -> pd.DataFrame:
        """
        Load Engie/Broad Reach Power asset data from Supabase.
        Cached for 10 hours.
        """
        try:
            # Fetch all Engie assets from Supabase
            response = _self.client.table("engie_storage_assets").select("*").execute()

            if not response.data:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(response.data)

            # Ensure numeric columns are properly typed
            numeric_cols = [
                'nameplate_power_mw', 'nameplate_energy_mwh',
                'duration_hours', 'operating_year', 'hsl', 'lsl'
            ]

            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            return df

        except APIError as e:
            st.error(f"Database error loading Engie asset data: {e}")
            return pd.DataFrame()
        except Exception as e:
            st.warning(f"Unexpected error loading Engie asset data: {e}")
            return pd.DataFrame()

    def load_generation_data(
        self,
        node: str,
        fuel_type: str = 'Solar',
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Load generation data (Solar/Wind) for a specific node.
        """
        try:
            query = self.client.table("ercot_generation").select("timestamp, gen_mw, forecast_mw, potential_mw")
            query = query.eq("settlement_point", node).eq("fuel_type", fuel_type)
            
            if start_date:
                query = query.gte("timestamp", start_date.isoformat())
            if end_date:
                # Add 1 day to include the end date fully
                query = query.lte("timestamp", (end_date + timedelta(days=1)).isoformat())
                
            response = query.execute()
            
            if not response.data:
                return pd.DataFrame()
                
            df = pd.DataFrame(response.data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Convert to US/Central if TZ-aware, else assume it matches price data (which is usually local-naive in this app)
            # The fetch script inserts as UTC (isoformat). 
            # Price data in this app is usually naive (local time).
            # We need to convert UTC -> Central -> Naive to match.
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_convert('US/Central').dt.tz_localize(None)
                
            df = df.set_index('timestamp').sort_index()
            return df
            
        except APIError as e:
            st.error(f"Database error loading generation data: {e}")
            return pd.DataFrame()
        except Exception as e:
            st.warning(f"Unexpected error loading generation data: {e}")
            return pd.DataFrame()


class ParquetDataLoader:
    """
    Data loader for local Parquet files with specific schema.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        # Hardcoded paths based on user request, but could be dynamic
        self.dam_path = (
            self.data_dir / 'DAM_Prices' /
            'dam_consolidated_2025-01-01_2025-11-23.parquet'
        )
        self.rtm_path = (
            self.data_dir / 'RTM_Prices' /
            'rtm_consolidated_2025-01-01_2025-11-23.parquet'
        )

    def load_prices(self, node: Optional[str] = None) -> pd.DataFrame:
        """
        Loads and merges DA and RT prices from local Parquet files.
        Uses predicate pushdown (filters) to only load data for the selected node.
        """
        if not self.dam_path.exists() or not self.rtm_path.exists():
            st.error(f"Parquet files not found in {self.data_dir}")
            return pd.DataFrame()

        # Load DAM
        try:
            # Use polars lazy loading with filter for efficient I/O
            if node:
                dam_df = pl.scan_parquet(self.dam_path).filter(
                    pl.col('SettlementPoint') == node
                ).collect().to_pandas()
            else:
                dam_df = pl.read_parquet(self.dam_path).to_pandas()

            if dam_df.empty:
                return pd.DataFrame()

            # Ensure numeric types for time calculations
            dam_df['HE_str'] = dam_df['HourEnding'].astype(str)
            dam_df['HE_num'] = pd.to_numeric(
                dam_df['HE_str'].str.split(':').str[0], errors='coerce'
            )
            dam_df = dam_df.dropna(subset=['HE_num'])
            dam_df['HE_num'] = dam_df['HE_num'].astype(int)

            dam_df['date_dt'] = pd.to_datetime(
                dam_df['DeliveryDate'], errors='coerce'
            )
            dam_df = dam_df.dropna(subset=['date_dt'])

            dam_df['timestamp'] = (
                dam_df['date_dt'] +
                pd.to_timedelta(dam_df['HE_num'], unit='h') -
                pd.Timedelta(hours=1)
            )

            dam_df = dam_df.rename(columns={
                'SettlementPoint': 'node',
                'SettlementPointPrice': 'price_mwh_da'
            })
            dam_df = dam_df[['timestamp', 'node', 'price_mwh_da']]
        except Exception as e:
            st.error(f"Error loading DAM parquet: {e}")
            return pd.DataFrame()

        # Load RTM
        try:
            # Use polars lazy loading with filter for efficient I/O
            if node:
                rtm_df = pl.scan_parquet(self.rtm_path).filter(
                    pl.col('SettlementPointName') == node
                ).collect().to_pandas()
            else:
                rtm_df = pl.read_parquet(self.rtm_path).to_pandas()

            if rtm_df.empty:
                return pd.DataFrame()

            rtm_df['DH_num'] = pd.to_numeric(
                rtm_df['DeliveryHour'], errors='coerce'
            )
            rtm_df['DI_num'] = pd.to_numeric(
                rtm_df['DeliveryInterval'], errors='coerce'
            )
            rtm_df = rtm_df.dropna(subset=['DH_num', 'DI_num'])
            rtm_df['DH_num'] = rtm_df['DH_num'].astype(int)
            rtm_df['DI_num'] = rtm_df['DI_num'].astype(int)

            rtm_df['date_dt'] = pd.to_datetime(
                rtm_df['DeliveryDate'], errors='coerce'
            )
            rtm_df = rtm_df.dropna(subset=['date_dt'])

            rtm_df['timestamp'] = (
                rtm_df['date_dt'] +
                pd.to_timedelta(rtm_df['DH_num'], unit='h') -
                pd.Timedelta(hours=1) +
                pd.to_timedelta(rtm_df['DI_num'] * 15, unit='min') -
                pd.Timedelta(minutes=15)
            )

            rtm_df = rtm_df.rename(columns={
                'SettlementPointName': 'node',
                'SettlementPointPrice': 'price_mwh_rt'
            })
            rtm_df = rtm_df[['timestamp', 'node', 'price_mwh_rt']]
        except Exception as e:
            st.error(f"Error loading RTM parquet: {e}")
            return pd.DataFrame()

        # Merge
        rtm_df['hour_start'] = rtm_df['timestamp'].dt.floor('h')  # type: ignore[attr-defined]

        merged = pd.merge(
            rtm_df,
            dam_df,
            left_on=['hour_start', 'node'],
            right_on=['timestamp', 'node'],
            suffixes=('', '_dam_drop'),
            how='left'
        )

        merged = merged.drop(columns=['hour_start', 'timestamp_dam_drop'], errors='ignore')

        merged['forecast_error'] = merged['price_mwh_rt'] - merged['price_mwh_da']
        merged['price_spread'] = abs(merged['forecast_error'])
        merged['extreme_event'] = merged['price_spread'] > 10

        return merged.sort_values(['node', 'timestamp']).reset_index(drop=True)

    def get_available_nodes(self) -> list[str]:
        """Reads unique nodes from the DAM file (faster than RTM)."""
        if not self.dam_path.exists():
            return []
        try:
            # Read only the SettlementPoint column to get unique values
            # Use polars for fast parquet I/O, then convert to pandas
            unique_nodes = (
                pl.scan_parquet(self.dam_path)
                .select('SettlementPoint')
                .unique()
                .collect()
            )
            return sorted(unique_nodes.to_pandas()['SettlementPoint'].tolist())
        except Exception as e:
            st.error(f"Error getting nodes from parquet: {e}")
            return []

    def get_date_range(self) -> tuple[Optional[date], Optional[date]]:
        """Reads min/max date from DAM file."""
        if not self.dam_path.exists():
            return None, None
        try:
            # Use polars for fast parquet I/O
            dates_pl = pl.scan_parquet(self.dam_path).select('DeliveryDate').collect()
            dates = pd.to_datetime(dates_pl.to_pandas()['DeliveryDate'])
            return dates.min().date(), dates.max().date()
        except Exception:
            return None, None


class UploadedFileLoader:
    """
    Data loader for user-uploaded CSV or Parquet files.
    Handles both file formats dynamically based on file extension.
    """

    def __init__(self, dam_file, rtm_file):
        """
        Parameters
        ----------
        dam_file : BytesIO | UploadedFile
            Uploaded DAM file from st.file_uploader
        rtm_file : BytesIO | UploadedFile
            Uploaded RTM file from st.file_uploader
        """
        self.dam_file = dam_file
        self.rtm_file = rtm_file
        self.dam_filename = getattr(dam_file, 'name', 'dam_file')
        self.rtm_filename = getattr(rtm_file, 'name', 'rtm_file')

    def _load_file(self, uploaded_file, filename: str) -> pd.DataFrame:
        """Load file based on extension (CSV or Parquet)"""
        try:
            if hasattr(uploaded_file, 'seek'):
                uploaded_file.seek(0)

            if filename.endswith('.parquet'):
                return pd.read_parquet(uploaded_file)
            if filename.endswith('.csv'):
                return pd.read_csv(uploaded_file)
            # Try to detect by content or default to CSV if unknown
            try:
                return pd.read_parquet(uploaded_file)
            except Exception:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")
            return pd.DataFrame()

    def _normalize_columns(self, df: pd.DataFrame, file_type: str) -> pd.DataFrame:
        """
        Normalize column names to standard expected format.
        file_type: 'dam' or 'rtm'
        """
        df.columns = df.columns.str.strip()

        # Common mappings
        mappings = {
            'Settlement Point': 'SettlementPoint',
            'Settlement Point Name': 'SettlementPointName',
            'Settlement Point Price': 'SettlementPointPrice',
            'Price': 'SettlementPointPrice',
            'LMP': 'SettlementPointPrice',
            'Hour Ending': 'HourEnding',
            'Delivery Date': 'DeliveryDate',
            'Delivery Hour': 'DeliveryHour',
            'Delivery Interval': 'DeliveryInterval',
            'Repeated Hour Flag': 'RepeatedHourFlag'
        }

        # Apply mappings
        df = df.rename(columns=mappings)

        # Specific fix for DAM vs RTM naming confusion
        if file_type == 'dam':
            if 'SettlementPointName' in df.columns and 'SettlementPoint' not in df.columns:
                df = df.rename(columns={'SettlementPointName': 'SettlementPoint'})
        elif file_type == 'rtm':
            if 'SettlementPoint' in df.columns and 'SettlementPointName' not in df.columns:
                df = df.rename(columns={'SettlementPoint': 'SettlementPointName'})

        return df

    def _validate_dam_schema(self, df: pd.DataFrame) -> bool:
        """Validate DAM file has required columns"""
        required = ['SettlementPoint', 'HourEnding', 'DeliveryDate', 'SettlementPointPrice']
        # Check if columns exist (case-insensitive check handled by normalization)
        missing = [col for col in required if col not in df.columns]
        if missing:
            st.error(f"DAM file missing required columns: {', '.join(missing)}")
            st.caption(f"Found columns: {list(df.columns)}")
            return False
        return True

    def _validate_rtm_schema(self, df: pd.DataFrame) -> bool:
        """Validate RTM file has required columns"""
        required = ['SettlementPointName', 'DeliveryHour', 'DeliveryInterval',
                    'DeliveryDate', 'SettlementPointPrice']
        missing = [col for col in required if col not in df.columns]
        if missing:
            st.error(f"RTM file missing required columns: {', '.join(missing)}")
            st.caption(f"Found columns: {list(df.columns)}")
            return False
        return True

    def load_prices(self, node: Optional[str] = None) -> pd.DataFrame:
        """
        Loads and merges uploaded DAM and RTM price files.
        Uses predicate pushdown (filters) for Parquet files to reduce memory usage.
        """
        # Define filters if node is provided
        dam_filters = [('SettlementPoint', '==', node)] if node else None
        rtm_filters = [('SettlementPointName', '==', node)] if node else None

        # Load DAM file
        try:
            if hasattr(self.dam_file, 'seek'):
                self.dam_file.seek(0)

            if self.dam_filename.endswith('.parquet'):
                dam_df = pd.read_parquet(self.dam_file, filters=dam_filters)
                dam_df = self._normalize_columns(dam_df, 'dam')
            else:
                # CSV or other
                dam_df = self._load_file(self.dam_file, self.dam_filename)
                dam_df = self._normalize_columns(dam_df, 'dam')
                if node and 'SettlementPoint' in dam_df.columns:
                    dam_df = dam_df[dam_df['SettlementPoint'] == node]

            if dam_df.empty or not self._validate_dam_schema(dam_df):
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading DAM file: {e}")
            return pd.DataFrame()

        # Load RTM file
        try:
            if hasattr(self.rtm_file, 'seek'):
                self.rtm_file.seek(0)

            if self.rtm_filename.endswith('.parquet'):
                rtm_df = pd.read_parquet(self.rtm_file, filters=rtm_filters)
                rtm_df = self._normalize_columns(rtm_df, 'rtm')
            else:
                # CSV or other
                rtm_df = self._load_file(self.rtm_file, self.rtm_filename)
                rtm_df = self._normalize_columns(rtm_df, 'rtm')
                if node and 'SettlementPointName' in rtm_df.columns:
                    rtm_df = rtm_df[rtm_df['SettlementPointName'] == node]

            if rtm_df.empty or not self._validate_rtm_schema(rtm_df):
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading RTM file: {e}")
            return pd.DataFrame()

        # Process DAM (same logic as ParquetDataLoader)
        try:
            # Note: Filtering already done during load for Parquet, or immediately after for CSV
            dam_df = dam_df.copy()

            dam_df['HE_str'] = dam_df['HourEnding'].astype(str)
            dam_df['HE_num'] = pd.to_numeric(
                dam_df['HE_str'].str.split(':').str[0], errors='coerce'
            )
            dam_df = dam_df.dropna(subset=['HE_num'])
            dam_df['HE_num'] = dam_df['HE_num'].astype(int)

            dam_df['date_dt'] = pd.to_datetime(
                dam_df['DeliveryDate'], errors='coerce'
            )
            dam_df = dam_df.dropna(subset=['date_dt'])

            dam_df['timestamp'] = (
                dam_df['date_dt'] +
                pd.to_timedelta(dam_df['HE_num'], unit='h') -
                pd.Timedelta(hours=1)
            )

            dam_df = dam_df.rename(columns={
                'SettlementPoint': 'node',
                'SettlementPointPrice': 'price_mwh_da'
            })
            dam_df = dam_df[['timestamp', 'node', 'price_mwh_da']]
        except Exception as e:
            st.error(f"Error processing DAM file: {e}")
            return pd.DataFrame()

        # Process RTM (same logic as ParquetDataLoader)
        try:
            # Note: Filtering already done during load for Parquet, or immediately after for CSV
            rtm_df = rtm_df.copy()

            rtm_df['DH_num'] = pd.to_numeric(
                rtm_df['DeliveryHour'], errors='coerce'
            )
            rtm_df['DI_num'] = pd.to_numeric(
                rtm_df['DeliveryInterval'], errors='coerce'
            )
            rtm_df = rtm_df.dropna(subset=['DH_num', 'DI_num'])
            rtm_df['DH_num'] = rtm_df['DH_num'].astype(int)
            rtm_df['DI_num'] = rtm_df['DI_num'].astype(int)

            rtm_df['date_dt'] = pd.to_datetime(
                rtm_df['DeliveryDate'], errors='coerce'
            )
            rtm_df = rtm_df.dropna(subset=['date_dt'])

            rtm_df['timestamp'] = (
                rtm_df['date_dt'] +
                pd.to_timedelta(rtm_df['DH_num'], unit='h') -
                pd.Timedelta(hours=1) +
                pd.to_timedelta(rtm_df['DI_num'] * 15, unit='min') -
                pd.Timedelta(minutes=15)
            )

            rtm_df = rtm_df.rename(columns={
                'SettlementPointName': 'node',
                'SettlementPointPrice': 'price_mwh_rt'
            })
            rtm_df = rtm_df[['timestamp', 'node', 'price_mwh_rt']]
        except Exception as e:
            st.error(f"Error processing RTM file: {e}")
            return pd.DataFrame()

        # Merge DAM and RTM
        rtm_df['hour_start'] = rtm_df['timestamp'].dt.floor('h')  # type: ignore[attr-defined]

        merged = pd.merge(
            rtm_df,
            dam_df,
            left_on=['hour_start', 'node'],
            right_on=['timestamp', 'node'],
            suffixes=('', '_dam_drop'),
            how='left'
        )

        merged = merged.drop(columns=['hour_start', 'timestamp_dam_drop'], errors='ignore')

        merged['forecast_error'] = merged['price_mwh_rt'] - merged['price_mwh_da']
        merged['price_spread'] = abs(merged['forecast_error'])
        merged['extreme_event'] = merged['price_spread'] > 10

        return merged.sort_values(['node', 'timestamp']).reset_index(drop=True)

    def get_available_nodes(self) -> list[str]:
        """Get list of unique nodes from uploaded DAM file"""
        dam_df = self._load_file(self.dam_file, self.dam_filename)
        dam_df = self._normalize_columns(dam_df, 'dam')
        if dam_df.empty or 'SettlementPoint' not in dam_df.columns:
            return []
        try:
            return sorted(dam_df['SettlementPoint'].unique().tolist())
        except Exception as e:
            st.error(f"Error getting nodes from uploaded files: {e}")
            return []

    def get_date_range(self) -> tuple[Optional[date], Optional[date]]:
        """Get min/max date from uploaded DAM file"""
        dam_df = self._load_file(self.dam_file, self.dam_filename)
        dam_df = self._normalize_columns(dam_df, 'dam')
        if dam_df.empty or 'DeliveryDate' not in dam_df.columns:
            return None, None
        try:
            dates = pd.to_datetime(dam_df['DeliveryDate'])
            return dates.min().date(), dates.max().date()
        except Exception:
            return None, None


def load_data(
    source: Literal['database', 'local_parquet'] = DEFAULT_DATA_SOURCE,
    node: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    data_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Unified data loading function supporting database and local parquet sources."""
    if source == 'database':
        loader = SupabaseDataLoader()
        return loader.load_prices(node=node, start_date=start_date, end_date=end_date)
    if source == 'local_parquet':
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / 'data'
        loader = ParquetDataLoader(data_dir)
        df = loader.load_prices(node=node)
        # Filter by date if provided
        if not df.empty and (start_date or end_date):
            if start_date:
                df = df[df['timestamp'].dt.date >= start_date]  # type: ignore[attr-defined]
            if end_date:
                df = df[df['timestamp'].dt.date <= end_date]  # type: ignore[attr-defined]
        return df
    raise ValueError(f"Invalid data source: {source}. Must be 'database' or 'local_parquet'")

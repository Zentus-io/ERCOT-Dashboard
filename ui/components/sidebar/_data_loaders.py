"""
Data Loaders for Sidebar Configuration
Zentus - ERCOT Battery Revenue Dashboard

This module centralizes all data loading functions for the sidebar,
including node lists, price data, EIA battery data, and Engie assets.
"""

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from postgrest.exceptions import APIError

from core.data.loaders import ParquetDataLoader, SupabaseDataLoader, UploadedFileLoader



@st.cache_data(ttl=3600, show_spinner="Loading node list...")
def load_node_list(source: str = 'csv') -> list:
    """
    Load available settlement points.

    Parameters
    ----------
    source : str
        Data source: 'database', 'local_parquet', or 'csv'

    Returns
    -------
    list
        List of available settlement point names
    """
    if source == 'database':
        try:
            db_loader = SupabaseDataLoader()
            nodes = db_loader.get_available_nodes()
            if nodes:
                return nodes
        except APIError as e:
            st.warning(f"⚠️ Database error fetching nodes: {str(e)}")

    elif source == 'local_parquet':
        try:
            from utils.state import get_state
            state = get_state()
            if state.uploaded_dam_file and state.uploaded_rtm_file:
                loader = UploadedFileLoader(state.uploaded_dam_file, state.uploaded_rtm_file)
                nodes = loader.get_available_nodes()
                if nodes:
                    return nodes
            return []
        except Exception as e:  # pylint: disable=broad-except
            st.warning(f"⚠️ Error fetching nodes from uploaded files: {str(e)}")

    # Fallback if DB/Parquet failed or empty
    return []


@st.cache_data(ttl=3600, max_entries=10, show_spinner="Loading price data...")
def load_node_prices(
    source: str,
    node: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    file_signature: Optional[str] = None
) -> pd.DataFrame:
    """
    Load price data for a specific node.

    Parameters
    ----------
    source : str
        Data source: 'database', 'local_parquet', 'uploaded', or 'csv'
    node : str
        Settlement point name
    start_date : date, optional
        Start date for filtering (database mode)
    end_date : date, optional
        End date for filtering (database mode)
    file_signature : str, optional
        Signature for uploaded files to invalidate cache when files change

    Returns
    -------
    pd.DataFrame
        Price data with columns: timestamp, da_price, rt_price, node
    """
    data_dir = Path(__file__).parent.parent.parent / 'data'

    if source == 'database':
        try:
            db_loader = SupabaseDataLoader()
            return db_loader.load_prices(node=node, start_date=start_date, end_date=end_date)
        except APIError as e:
            st.warning(f"⚠️ Database error: {str(e)}. Falling back to CSV.")

    elif source == 'local_parquet':
        try:
            loader = ParquetDataLoader(data_dir)
            return loader.load_prices(node=node)
        except Exception as e:  # pylint: disable=broad-except
            st.warning(f"⚠️ Parquet error: {str(e)}. Falling back to CSV.")

    elif source == 'uploaded':
        try:
            from utils.state import get_state
            state = get_state()
            if state.uploaded_dam_file and state.uploaded_rtm_file:
                loader = UploadedFileLoader(state.uploaded_dam_file, state.uploaded_rtm_file)
                # Apply date filtering if specified
                df = loader.load_prices(node=node)

                if not df.empty and (start_date or end_date):
                    if start_date:
                        df = df[df['timestamp'].dt.date >= start_date]  # type: ignore
                    if end_date:
                        df = df[df['timestamp'].dt.date <= end_date]  # type: ignore
                return df
            return pd.DataFrame()
        except Exception as e:  # pylint: disable=broad-except
            st.warning(f"⚠️ Upload error: {str(e)}")

    # Fallback
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_eia_data() -> pd.DataFrame:
    """
    Load EIA-860 battery data from database.

    Returns
    -------
    pd.DataFrame
        Battery specifications from EIA-860 dataset
    """
    try:
        return SupabaseDataLoader().load_eia_batteries()
    except APIError as e:
        st.warning(f"Could not load battery data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_engie_data() -> pd.DataFrame:
    """
    Load Engie asset data from database.

    Returns
    -------
    pd.DataFrame
        Engie asset specifications with settlement point mappings
    """
    try:
        return SupabaseDataLoader().load_engie_assets()
    except Exception as e:
        st.warning(f"Could not load Engie asset data: {e}")
        return pd.DataFrame()

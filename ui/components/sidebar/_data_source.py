"""
Data Source Selection UI
Zentus - ERCOT Battery Revenue Dashboard

This module handles data source selection and file upload UI components.
"""

import streamlit as st

from config.settings import SUPABASE_KEY, SUPABASE_URL
from utils.state import clear_data_cache, get_state


def render_data_source_selector() -> str:
    """
    Render data source selection radio buttons.

    Returns
    -------
    str
        Selected data source ('database' or 'local_parquet')
    """
    state = get_state()

    def on_data_source_change():
        """Callback for data source change"""
        # The new value is already in st.session_state when this runs
        pass

    if SUPABASE_URL and SUPABASE_KEY:
        data_source_ui = st.sidebar.radio(
            "Data Source:",
            options=['Database', 'Local Parquet'],
            index={
                'database': 0,
                'local_parquet': 1
            }.get(state.data_source, 0),
            help=("**Database**: Multi-day historical data from Supabase\n"
                  "**Local Parquet**: Upload your own DAM/RTM parquet files"),
            key="data_source_radio",
            on_change=on_data_source_change
        )

        # Map display names to internal values
        if data_source_ui == 'Database':
            data_source_internal = 'database'
        else:
            data_source_internal = 'local_parquet'

        # Logic: Only switch to local_parquet if files are ready
        if data_source_internal == 'local_parquet':
            # Check if files are uploaded
            if state.uploaded_dam_file and state.uploaded_rtm_file:
                if state.data_source != 'local_parquet':
                    state.data_source = 'local_parquet'
                    state.using_uploaded_files = True
                    clear_data_cache()
            # else: Files not ready - keep previous source
        else:
            # For Database, switch immediately
            if data_source_internal != state.data_source:
                state.data_source = data_source_internal
                state.using_uploaded_files = False
                clear_data_cache()
    else:
        # No database credentials - force Local Parquet mode
        state.data_source = 'local_parquet'
        st.sidebar.info("📌 **Local Mode** - Configure Supabase for historical data")

    return state.data_source


def render_file_upload() -> None:
    """
    Render file upload UI for local parquet files.

    Updates state with uploaded DAM and RTM files.
    """
    state = get_state()

    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Custom Data Upload")
    st.sidebar.markdown("**Upload ERCOT Price Files**")

    def on_file_upload():
        """Callback for file uploads to ensure immediate state update"""
        state = get_state()
        if st.session_state.get('dam_upload'):
            state.uploaded_dam_file = st.session_state.dam_upload
        if st.session_state.get('rtm_upload'):
            state.uploaded_rtm_file = st.session_state.rtm_upload

        # If both files are present, switch to local_parquet immediately
        if state.uploaded_dam_file and state.uploaded_rtm_file:
            state.data_source = 'local_parquet'
            state.using_uploaded_files = True
            clear_data_cache()

    dam_file = st.sidebar.file_uploader(
        "DAM Prices",
        type=['parquet', 'csv'],
        help="Day-Ahead Market price file (.parquet or .csv)",
        key='dam_upload',
        on_change=on_file_upload
    )

    rtm_file = st.sidebar.file_uploader(
        "RTM Prices",
        type=['parquet', 'csv'],
        help="Real-Time Market price file (.parquet or .csv)",
        key='rtm_upload',
        on_change=on_file_upload
    )

    # Update state with uploaded files (redundant with callback but safe)
    if dam_file:
        state.uploaded_dam_file = dam_file
    if rtm_file:
        state.uploaded_rtm_file = rtm_file

    # File Status Indicators
    if state.uploaded_dam_file:
        st.sidebar.success(f"✅ DAM: {state.uploaded_dam_file.name}")  # type: ignore
    else:
        st.sidebar.warning("⚠️ Waiting for DAM file...")

    if state.uploaded_rtm_file:
        st.sidebar.success(f"✅ RTM: {state.uploaded_rtm_file.name}")  # type: ignore
    else:
        st.sidebar.warning("⚠️ Waiting for RTM file...")

    # Global status
    if not (state.uploaded_dam_file and state.uploaded_rtm_file):
        st.sidebar.info("ℹ️ Using previous data until both files are uploaded.")

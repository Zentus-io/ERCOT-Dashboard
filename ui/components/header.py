"""
Header Component
Zentus - ERCOT Battery Revenue Dashboard
"""

from pathlib import Path

import streamlit as st

from ui.components.navigation import render_top_nav
from utils.state import get_state


def render_header():
    """
    Render application header with branding and data notice.

    This component displays the Zentus logo, app title, and current data availability.
    """
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(
            '<p class="main-header">⚡ ERCOT Battery Storage Revenue Opportunity</p>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<p class="sub-header">Demonstrating Value Through Intelligent Forecasting</p>',
            unsafe_allow_html=True
        )

    with col2:
        # Display Zentus logo and brand text together
        logo_path = Path(__file__).parent.parent.parent / 'media' / 'Logo_Option_5_nobg.png'

        col2a, col2b = st.columns([1, 2])

        with col2a:
            if logo_path.exists():
                st.image(str(logo_path), width=80)

        with col2b:
            st.markdown("""
                <div style='padding-top: 15px;'>
                    <span style='font-size: 1.8rem; color: #0A5F7A; font-weight: bold;'>Zentus</span><br>
                    <span style='font-size: 0.8rem; color: #6C757D;'>Intelligent Forecasting</span>
                </div>
            """, unsafe_allow_html=True)

    # Data availability notice
    state = get_state()

    # Dynamic data note based on source
    if state.data_source == 'database':
        date_range = f"{
            state.start_date.strftime('%b %d, %Y')} - {
            state.end_date.strftime('%b %d, %Y')}"
        note_content = f"<strong>Historical Analysis:</strong> Analyzing Engie assets in ERCOT market for {date_range}."
    elif state.data_source == 'local_parquet':
        date_range = f"{
            state.start_date.strftime('%b %d, %Y')} - {
            state.end_date.strftime('%b %d, %Y')}"
        note_content = f"<strong>Local Data:</strong> Analyzing ERCOT data from uploaded parquet files for {date_range}."
    else:
        note_content = "<strong>Select Data Source:</strong> Please choose a data source from the sidebar."

    st.markdown(f"""
    <div class='data-note'>
        {note_content}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Render top navigation
    render_top_nav()

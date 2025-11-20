"""
Header Component
Zentus - ERCOT Battery Revenue Dashboard
"""

import streamlit as st
from pathlib import Path
from config.settings import DATA_NOTE


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
    from utils.state import get_state
    state = get_state()

    eia_note = ""
    if state.eia_battery_data is not None:
        eia_note = " Battery system parameters validated against EIA-860 data (136 operational Texas systems)."

    # Dynamic data note based on source
    if state.data_source == 'database':
        date_range = f"{state.start_date.strftime('%b %d, %Y')} - {state.end_date.strftime('%b %d, %Y')}"
        note_content = f"<strong>Historical Analysis:</strong> Analyzing ERCOT market data for {date_range}."
    else:
        note_content = "<strong>Demo Mode:</strong> Currently showing July 20, 2025 data from ERCOT wind resources (Single-day snapshot)."

    st.markdown(f"""
    <div class='data-note'>
        {note_content}{eia_note}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

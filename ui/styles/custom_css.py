"""
Custom CSS Styling
Zentus - ERCOT Battery Revenue Dashboard
"""

import streamlit as st
from config.settings import COLORS


def apply_custom_styles():
    """Apply custom CSS styling to the Streamlit app."""
    st.markdown(f"""
        <style>
        .main-header {{
            color: {COLORS['primary']};
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0;
        }}
        .sub-header {{
            color: {COLORS['secondary']};
            font-size: 1.2rem;
            margin-top: 0;
        }}
        .metric-card {{
            background-color: {COLORS['background']};
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid {COLORS['primary']};
        }}
        .opportunity-metric {{
            font-size: 2rem;
            color: {COLORS['primary']};
            font-weight: bold;
        }}
        .data-note {{
            background-color: #FFF3CD;
            padding: 10px;
            border-radius: 5px;
            border-left: 3px solid {COLORS['warning']};
            margin: 10px 0;
        }}
        /* Improve sidebar styling */
        .css-1d391kg {{
            padding-top: 1rem;
        }}
        /* Style metric cards */
        [data-testid="stMetricValue"] {{
            font-size: 2rem;
        }}
        </style>
    """, unsafe_allow_html=True)

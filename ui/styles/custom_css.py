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
        /* --- Layout Adjustments for Fixed Nav --- */

        /* 2. Hide Sidebar Nav (since we have top nav) */
        div[data-testid="stSidebarNav"] {{
            display: none;
        }}

        /* 3. Increase Sidebar Width */
        section[data-testid="stSidebar"] {{
            width: 280px; !important;
            min-width: 200px;
        }}

        /* 4. Compact Sidebar Spacing */
        section[data-testid="stSidebar"] > div {{
            padding-top: 0rem !important;
            padding-bottom: 1rem !important;
        }}

        /* Reduce spacing between sidebar elements */
        section[data-testid="stSidebar"] .block-container {{
            padding-top: 0.25rem !important;
            padding-bottom: 0.25rem !important;
        }}

        /* Compact form elements */
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {{
            gap: 0.3rem !important;
        }}

        /* Reduce header spacing in sidebar */
        section[data-testid="stSidebar"] .stMarkdown h1,
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3 {{
            margin-top: 0.5rem !important;
            margin-bottom: 0.3rem !important;
            padding-top: 0 !important;
        }}

        /* Compact widgets */
        section[data-testid="stSidebar"] .stRadio,
        section[data-testid="stSidebar"] .stSelectbox,
        section[data-testid="stSidebar"] .stSlider,
        section[data-testid="stSidebar"] .stNumberInput {{
            margin-bottom: 0.5rem !important;
        }}

        /* Reduce label spacing */
        section[data-testid="stSidebar"] label {{
            margin-bottom: 0.2rem !important;
            font-size: 0.85rem !important;
        }}

        /* Compact expanders */
        section[data-testid="stSidebar"] .streamlit-expanderHeader {{
            padding: 0.3rem 0.5rem !important;
            font-size: 0.9rem !important;
        }}

        /* Reduce divider spacing */
        section[data-testid="stSidebar"] hr {{
            margin-top: 0.5rem !important;
            margin-bottom: 0.5rem !important;
        }}

        /* Compact captions */
        section[data-testid="stSidebar"] .stCaption {{
            margin-top: 0.1rem !important;
            margin-bottom: 0.3rem !important;
        }}

        /* --- Theme-Aware Components --- */

        /* Typography */
        .main-header {{
            color: {COLORS['primary']};
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}

        /* Metric Cards */
        /* Use 'var(--secondary-background-color)' so it works in Dark Mode */
        .metric-card {{
            background-color: var(--secondary-background-color);
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border-left: 4px solid {COLORS['primary']};
            color: var(--text-color);
        }}

        /* Data Note */
        .data-note {{
            background-color: color-mix(in srgb, {COLORS['accent']}, transparent 90%);
            color: var(--text-color);
            padding: 12px;
            border-radius: 6px;
            border-left: 4px solid {COLORS['accent']};
            font-size: 0.9rem;
            margin-bottom: 15px;
        }}

        /* Plotly Fix */
        .js-plotly-plot .plotly .main-svg {{
            background: transparent !important;
        }}
        </style>
    """, unsafe_allow_html=True)

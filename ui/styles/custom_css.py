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
        /* --- Typography --- */
        .main-header {{
            color: {COLORS['primary']};
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}
        .sub-header {{
            color: {COLORS['secondary']};
            font-size: 1.1rem;
            margin-top: 0;
            margin-bottom: 1.5rem;
        }}
        
        /* --- Layout --- */
        /* Hide standard Streamlit header decoration */
        /* Header visibility handled in navigation.py */
        /* Header visibility handled in navigation.py */
        
        /* Adjust top padding to accommodate custom nav */
        /* Adjust top padding to accommodate custom fixed nav */
        /* Note: Padding is now handled in navigation.py via !important to ensure it takes precedence */
        
        /* Remove old sticky positioning if it interferes */
        div[data-testid="stVerticalBlock"] > div:has(div#nav-container-marker) {{
            /* Styles handled in navigation.py */
        }}
        
        /* --- Sidebar --- */
        [data-testid="stSidebar"] {{
            /* Allow default theme background */
            /* background-color: #f8f9fa; */
            /* border-right: 1px solid #e9ecef; */
        }}
        
        /* Hide default navigation */
        [data-testid="stSidebarNav"] {{
            display: none;
        }}
        
        /* Compact sidebar spacing */
        [data-testid="stSidebar"] .block-container {{
            padding-top: 1rem;
            padding-bottom: 1rem;
        }}
        
        /* Make sidebar elements more compact */
        .stRadio > label, .stSelectbox > label, .stSlider > label {{
            font-size: 0.9rem;
            font-weight: 600;
            /* color: #333; Allow default theme color */
        }}
        
        /* --- Components --- */
        /* Metric Cards */
        .metric-card {{
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border-left: 4px solid {COLORS['primary']};
        }}
        
        /* Data Note */
        .data-note {{
            background-color: #eef7f9;
            color: {COLORS['secondary']};
            padding: 12px;
            border-radius: 6px;
            border-left: 4px solid {COLORS['accent']};
            font-size: 0.9rem;
            margin-bottom: 15px;
        }}
        
        /* --- Metrics --- */
        [data-testid="stMetricValue"] {{
            font-size: 1.6rem;
            color: {COLORS['primary']};
        }}
        [data-testid="stMetricLabel"] {{
            font-size: 0.9rem;
            color: {COLORS['gray']};
        }}
        </style>
    """, unsafe_allow_html=True)

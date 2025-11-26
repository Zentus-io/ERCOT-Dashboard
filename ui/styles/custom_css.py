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
        
        /* 1. Push main content down so it doesn't hide behind the navbar */
        /* Increased to 6rem to provide comfortable breathing room */
        div[data-testid="block-container"] {{
            padding-top: 6rem !important; 
        }}

        /* 2. Hide Sidebar Nav (since we have top nav) */
        div[data-testid="stSidebarNav"] {{
            display: none;
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
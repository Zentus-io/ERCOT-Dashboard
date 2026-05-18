"""
Streamlit Page Configuration
Zentus - ERCOT Battery Revenue Dashboard
"""
from pathlib import Path
from typing import Optional

import streamlit as st

from .settings import APP_ICON, APP_TITLE


def configure_page(page_title: Optional[str] = None):
    """
    Configure Streamlit page settings.

    Parameters
    ----------
    page_title : str, optional
        Specific page title. If None, uses default app title.
    """
    # Get favicon path
    favicon_path = Path(__file__).parent.parent / 'media' / 'favicon-32x32.png'

    # Set page config
    st.set_page_config(
        page_title=f"{APP_TITLE} - {page_title}" if page_title else APP_TITLE,
        page_icon=str(favicon_path) if favicon_path.exists() else APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/Zentus-io/ERCOT-Dashboard',
            'Report a bug': 'https://github.com/Zentus-io/ERCOT-Dashboard/issues',
            'About': """
            # ERCOT Battery Storage Revenue Dashboard

            An interactive engine to size, dispatch, and stress-test battery
            storage assets against ERCOT prices. Supports MPC, LP, threshold,
            and rolling-window strategies plus hybrid solar+storage. Market-
            agnostic core — adaptable to OMIE/SRAD/aFRR with a data-layer swap.

            **Author:** Juan Manuel Boullosa Novo

            **Contact:** jmboullosa@zentus.io
            """
        }
    )

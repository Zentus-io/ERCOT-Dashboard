"""
Top Navigation Component
Zentus - ERCOT Battery Revenue Dashboard
"""

import streamlit as st
from utils.state import get_state

def render_top_nav():
    """
    Render a stylized floating top navigation bar with breadcrumbs.
    """
    state = get_state()
    
    # Define pages structure
    pages = [
        {"name": "Home", "path": "Home.py", "icon": "🏠"},
        {"name": "Overview", "path": "pages/1_🏠_Overview.py", "icon": "📊"},
        {"name": "Nodal Analysis", "path": "pages/1_🗺️_Nodal_Analysis.py", "icon": "🗺️"},
        {"name": "Price Analysis", "path": "pages/2_📈_Price_Analysis.py", "icon": "📈"},
        {"name": "Operations", "path": "pages/3_🔋_Operations.py", "icon": "🔋"},
        {"name": "Revenue", "path": "pages/4_💰_Revenue.py", "icon": "💰"},
        {"name": "Opportunity", "path": "pages/5_🎯_Opportunity.py", "icon": "🎯"},
        {"name": "Timeline", "path": "pages/6_📊_Timeline.py", "icon": "📅"},
        {"name": "Optimization", "path": "pages/7_⚙️_Optimization.py", "icon": "⚙️"},
    ]
    
    # Use columns for navigation
    cols = st.columns(len(pages))
    
    # Place marker in the first column so we can target the parent HorizontalBlock
    with cols[0]:
        st.markdown('<div id="nav-container-marker"></div>', unsafe_allow_html=True)

    for i, page in enumerate(pages):
        with cols[i]:
            st.page_link(
                page["path"],
                label=page["name"],
                icon=page["icon"],
                use_container_width=True
            )

    st.markdown("""
    <style>
        /* Floating Navbar Container */
        /* Target the HorizontalBlock that contains the marker */
        div[data-testid="stHorizontalBlock"]:has(div#nav-container-marker) {
            position: fixed !important;
            top: 1rem !important;
            left: 15vw !important;
            width: 70vw !important;
            height: 3rem !important;
            z-index: 1000002 !important;
            background: rgba(255, 255, 255, 0.40) !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
            border-bottom: 1px solid rgba(0,0,0,0.05) !important;
            display: flex !important;
            flex-direction: row !important;
            align-items: center !important;
            justify-content: flex-start !important; /* Align left */
            padding: 0 1rem !important;
            padding-left: 1rem !important; /* Space for sidebar toggle */
            margin: 0 !important;
            transition: all 0.3s ease !important;
            pointer-events: auto !important;
            gap: 0.25rem !important; /* Minimal gap */
        }

        /* Hide the marker itself */
        div#nav-container-marker {
            display: none;
        }
        
        /* Style the columns inside - FORCE COMPACT WIDTH */
        div[data-testid="stHorizontalBlock"]:has(div#nav-container-marker) [data-testid="column"] {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            height: 100% !important;
            min-width: auto !important;
            width: auto !important; /* Override Streamlit's equal width */
            flex: 0 1 auto !important; /* Don't grow, allow shrink */
            padding: 0 !important;
        }
        
        /* Target the page link buttons */
        div[data-testid="stHorizontalBlock"]:has(div#nav-container-marker) button {
            background: transparent !important;
            border: none !important;
            padding: 0.25rem 0.5rem !important;
            font-size: 0.85rem !important;
            line-height: 1 !important;
            min-height: 0 !important;
            height: auto !important;
            margin: 0 !important; /* Remove any default margins */
            box-shadow: none !important;
        }
        
        /* Fix vertical alignment for all content inside buttons */
        div[data-testid="stHorizontalBlock"]:has(div#nav-container-marker) button > div {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 0.3rem !important;
            padding: 0 !important;
            margin: 0 !important;
            line-height: 1 !important;
        }
        
        /* Target the paragraph inside the button (the label) */
        div[data-testid="stHorizontalBlock"]:has(div#nav-container-marker) button p {
            font-size: 0.85rem !important;
            font-weight: 500 !important;
            margin: 0 !important;
            padding: 0 !important;
            line-height: 1 !important;
            display: inline-block !important;
        }
        
        /* Target the icon specifically if possible to ensure alignment */
        div[data-testid="stHorizontalBlock"]:has(div#nav-container-marker) button span[role="img"] {
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            margin: 0 !important;
            padding: 0 !important;
            line-height: 1 !important;
        }

        /* Restore Streamlit header but make it transparent and click-through */
        header[data-testid="stHeader"] {
            display: block !important;
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100vw !important;
            background: transparent !important;
            z-index: 1000001 !important;
            pointer-events: none;
            height: 3rem !important;
        }
        
        /* Re-enable pointer events ONLY for the buttons inside the header */
        header[data-testid="stHeader"] button, 
        header[data-testid="stHeader"] [role="button"],
        header[data-testid="stHeader"] a {
            pointer-events: auto !important;
            visibility: visible !important;
            z-index: 1000003 !important;
        }
        
        /* Hide the colored decoration line if present */
        header[data-testid="stHeader"]::before {
            display: none !important;
        }
        
        /* Ensure main content is pushed down */
        .main .block-container {
            padding-top: 5rem !important;
        }
    </style>
    """, unsafe_allow_html=True)

    
    # Breadcrumbs (Simple implementation based on active page could be added here)
    # For now, the page title usually serves this purpose.
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
    
    # Marker for CSS targeting
    st.markdown('<div id="nav-container-marker"></div>', unsafe_allow_html=True)
    
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

    # Determine active page
    # current_page gets the file path of the script being run
    import os
    # Note: st.navigation or similar new features might be better, 
    # but sticking to standard components for compatibility.
    
    # HTML for Navbar
    # We use st.markdown with unsafe_allow_html to create a fixed header
    
    st.markdown("""
    <style>
        /* Floating Navbar Container */
        .floating-nav-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw; /* Full viewport width */
            height: 3.5rem; /* Fixed height */
            z-index: 999999; /* Highest priority */
            background: rgba(255, 255, 255, 0.98);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            border-bottom: 1px solid rgba(0,0,0,0.05);
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0 1rem;
            transition: all 0.3s ease;
        }

        /* Target the container Streamlit creates for the markdown */
        div[data-testid="stVerticalBlock"] > div:has(div#nav-container-marker) {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100vw !important;
            z-index: 999999 !important;
            margin: 0 !important;
            padding: 0 !important;
            background: transparent !important; /* Let the inner container handle bg */
            pointer-events: none; /* Let clicks pass through wrapper if needed, but we need clicks on children */
        }
        
        /* Re-enable pointer events for the actual content */
        div[data-testid="stVerticalBlock"] > div:has(div#nav-container-marker) > * {
            pointer-events: auto;
        }

        /* Style the inner content of the nav */
        div[data-testid="stVerticalBlock"] > div:has(div#nav-container-marker) .stHorizontalBlock {
            background: rgba(255, 255, 255, 0.98);
            backdrop-filter: blur(12px);
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            padding: 0.5rem 1rem;
            border-radius: 0 0 10px 10px; /* Rounded bottom corners */
            width: 100%;
            max-width: 100%;
        }
        
        /* Restore Streamlit header but make it transparent and click-through */
        header[data-testid="stHeader"] {
            display: block !important;
            background: transparent !important;
            z-index: 1000000 !important; /* Higher than custom nav */
            pointer-events: none; /* Let clicks pass through to our nav */
        }
        
        /* Re-enable pointer events for the buttons inside the header */
        header[data-testid="stHeader"] > * {
            pointer-events: auto;
        }
        
        /* Hide the colored decoration line if present */
        header[data-testid="stHeader"]::before {
            display: none;
        }
        
        /* Ensure main content is pushed down */
        .main .block-container {
            padding-top: 5rem !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # We can't easily create clickable links that route internally in Streamlit 
    # without using st.page_link.
    # So we will use a container at the top that mimics the navbar using native columns
    # but styled to look integrated.
    
    # Note: Pure CSS floating navbar with Streamlit widgets inside is hard because 
    # Streamlit widgets are rendered where they are in the script flow.
    # We will use a top container.
    
    with st.container():
        # Use columns for navigation
        cols = st.columns(len(pages))
        for i, page in enumerate(pages):
            with cols[i]:
                st.page_link(
                    page["path"],
                    label=page["name"],
                    icon=page["icon"],
                    use_container_width=True
                )
    
    st.markdown("---")
    
    # Breadcrumbs (Simple implementation based on active page could be added here)
    # For now, the page title usually serves this purpose.
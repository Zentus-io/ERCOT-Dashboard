"""
Top Navigation Component
Zentus - ERCOT Battery Revenue Dashboard
"""

import streamlit as st
from utils.state import get_state

def render_top_nav():
    """
    Render a stylized floating top navigation bar.
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
    
    # --- STRUCTURE: BALANCED GHOST COLUMNS ---
    # We use explicit ratios to make the marker columns tiny (0.1) and button columns equal (1).
    # Structure: [Ghost Left (0.1)] + [Buttons (1)...] + [Ghost Right (0.1)]
    col_ratios = [0.1] + ([1] * len(pages)) + [0.1]
    cols = st.columns(col_ratios)
    
    # 1. Place marker in the Left Ghost Column
    with cols[0]:
        st.markdown('<div id="nav-marker"></div>', unsafe_allow_html=True)
        
    # 2. Place pages in the middle columns
    for i, page in enumerate(pages):
        with cols[i + 1]:
            st.page_link(
                page["path"],
                label=page["name"],
                icon=page["icon"],
                use_container_width=True
            )
            
    # 3. The last column (cols[-1]) is left empty for symmetry

    # --- CSS STYLES ---
    st.markdown("""
    <style>
        /* 1. FLOAT THE NAVBAR */
        /* Target the HorizontalBlock that contains the marker */
        div[data-testid="stHorizontalBlock"]:has(div#nav-marker) {
            position: fixed !important;
            top: 1.5rem !important;
            left: 50% !important;
            transform: translateX(-50%) !important;
            z-index: 999999 !important;
            
            /* Sizing & Shape - FIXED WIDTH for consistency */
            width: 700px !important;
            max-width: 90vw !important;
            height: auto !important;
            padding: 0.4rem 0.6rem !important;
            border-radius: 100px !important;
            margin: 0 !important;
            
            /* Glassmorphism Theme Adaptive */
            background-color: color-mix(in srgb, var(--background-color), transparent 10%) !important;
            backdrop-filter: blur(16px) !important;
            -webkit-backdrop-filter: blur(16px) !important;
            border: 1px solid rgba(128, 128, 128, 0.2) !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08) !important;
            
            /* Flex Layout */
            display: flex !important;
            flex-direction: row !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 0.25rem !important;
            pointer-events: auto !important;
        }

        /* 2. HIDE GHOST COLUMNS (Symmetry Fix) */
        div[data-testid="stHorizontalBlock"]:has(div#nav-marker) > div[data-testid="column"]:first-child,
        div[data-testid="stHorizontalBlock"]:has(div#nav-marker) > div[data-testid="column"]:last-child {
            display: none !important;
            width: 0 !important;
            flex: 0 !important;
        }

        div#nav-marker {
            display: none !important;
        }
        
        /* 3. ACCORDION COLUMNS */
        div[data-testid="stHorizontalBlock"]:has(div#nav-marker) [data-testid="column"] {
            width: auto !important;
            flex: 1 !important;
            min-width: 0 !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            
            /* Smooth Transition for Flex Grow */
            transition: flex 0.4s cubic-bezier(0.25, 1, 0.5, 1) !important;
            overflow: hidden !important;
        }
        
        /* HOVER STATE: Triggered by JavaScript class */
        div[data-testid="stHorizontalBlock"]:has(div#nav-marker) [data-testid="column"].nav-hover {
            flex: 2.5 !important;
        }

        /* 4. BUTTON/ANCHOR STYLING */
        div[data-testid="stHorizontalBlock"]:has(div#nav-marker) a[data-testid="stPageLink-NavLink"] {
            background-color: transparent !important;
            border: none !important;
            color: var(--text-color) !important;
            box-shadow: none !important;
            padding: 0.4rem 0 !important;
            border-radius: 50px !important;
            transition: all 0.2s ease !important;
            margin: 0 !important;
            
            /* Fill the column */
            width: 100% !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            
            /* Typography */
            font-size: 0.9rem !important;
            font-weight: 600 !important;
            line-height: 1 !important;
            min-height: 0px !important;
            height: auto !important;
            text-decoration: none !important;
        }

        /* Hover Effect on the Link */
        div[data-testid="stHorizontalBlock"]:has(div#nav-marker) a[data-testid="stPageLink-NavLink"]:hover {
            background-color: rgba(128, 128, 128, 0.15) !important;
        }
        
        /* Active/Focus State */
        div[data-testid="stHorizontalBlock"]:has(div#nav-marker) a[data-testid="stPageLink-NavLink"]:active,
        div[data-testid="stHorizontalBlock"]:has(div#nav-marker) a[data-testid="stPageLink-NavLink"]:focus {
             background-color: rgba(128, 128, 128, 0.15) !important;
             border: none !important;
             outline: none !important;
        }

        /* 5. CONTENT ALIGNMENT (Icon + Text) */
        div[data-testid="stHorizontalBlock"]:has(div#nav-marker) a[data-testid="stPageLink-NavLink"] > div {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 0 !important;
            width: 100% !important;
        }
        
        /* Text Label - Hidden by default */
        div[data-testid="stHorizontalBlock"]:has(div#nav-marker) p {
            max-width: 0;
            opacity: 0;
            overflow: hidden;
            white-space: nowrap !important;
            margin: 0 !important;
            padding: 0 !important;
            
            /* Smooth Animation */
            transition: max-width 0.4s cubic-bezier(0.25, 1, 0.5, 1), 
                        opacity 0.3s ease-in-out, 
                        margin-left 0.3s ease-in-out !important;
        }

        /* REVEAL TEXT when COLUMN has nav-hover class */
        div[data-testid="stHorizontalBlock"]:has(div#nav-marker) [data-testid="column"].nav-hover p {
            max-width: 150px !important;
            opacity: 1;
            margin-left: 0.5rem !important;
        }

        /* 6. MOBILE RESPONSIVENESS */
        @media (max-width: 900px) {
            div[data-testid="stHorizontalBlock"]:has(div#nav-marker) {
                width: 90vw !important;
                max-width: 90vw !important;
                left: 50% !important;
                overflow-x: auto !important;
                justify-content: flex-start !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
                -ms-overflow-style: none;
                scrollbar-width: none;
            }
            
            div[data-testid="stHorizontalBlock"]:has(div#nav-marker)::-webkit-scrollbar {
                display: none;
            }
        }
    </style>
    
    <script>
        // JavaScript-based accordion hover using EVENT DELEGATION
        // This is robust against Streamlit's re-rendering and DOM updates.
        
        function setupNavHoverDelegation() {
            if (window.navHoverInitialized) return;
            
            console.log("Initializing Nav Hover Delegation...");
            
            // Helper to find the correct column
            function getNavColumn(target) {
                // Check if we are hovering a nav link or inside one
                const link = target.closest('a[data-testid="stPageLink-NavLink"]');
                if (!link) return null;
                
                // Verify it's inside our specific nav container
                const navContainer = link.closest('div[data-testid="stHorizontalBlock"]:has(div#nav-marker)');
                if (!navContainer) return null;
                
                return {
                    column: link.closest('[data-testid="column"]'),
                    container: navContainer
                };
            }

            // Mouse Over Delegation
            document.body.addEventListener('mouseover', function(e) {
                const result = getNavColumn(e.target);
                if (result) {
                    const { column, container } = result;
                    
                    // Remove hover class from all other columns in this container
                    container.querySelectorAll('[data-testid="column"]').forEach(col => {
                        if (col !== column) col.classList.remove('nav-hover');
                    });
                    
                    // Add hover class to current column
                    if (!column.classList.contains('nav-hover')) {
                        column.classList.add('nav-hover');
                        // console.log('Hover ON:', column);
                    }
                }
            });

            // Mouse Out Delegation
            document.body.addEventListener('mouseout', function(e) {
                const result = getNavColumn(e.target);
                if (result) {
                    const { column } = result;
                    
                    // Check if we are actually leaving the column (not just moving to a child)
                    // relatedTarget is where the mouse went
                    if (!column.contains(e.relatedTarget)) {
                        column.classList.remove('nav-hover');
                        // console.log('Hover OFF:', column);
                    }
                }
            });
            
            window.navHoverInitialized = true;
            console.log("Nav Hover Delegation Setup Complete.");
        }
        
        // Run immediately
        setupNavHoverDelegation();
        
        // Also run on doc ready just in case
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', setupNavHoverDelegation);
        }
    </script>
    """, unsafe_allow_html=True)
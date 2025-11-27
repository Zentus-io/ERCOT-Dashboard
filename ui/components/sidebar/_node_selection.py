"""
Node Selection UI
Zentus - ERCOT Battery Revenue Dashboard

This module handles settlement point selection and auto-date-range logic.
"""

import streamlit as st

from core.data.loaders import SupabaseDataLoader
from utils.state import clear_simulation_cache, get_state, update_date_range, update_state

from ._date_utils import find_best_date_range


def render_node_selector(available_nodes: list) -> str:
    """
    Render settlement point selector.

    Parameters
    ----------
    available_nodes : list
        List of available settlement point names

    Returns
    -------
    str
        Selected node name
    """
    state = get_state()

    selected_node = st.sidebar.selectbox(
        "Select Settlement Point:",
        available_nodes or [],
        index=0 if (
            state.selected_node is None
            or available_nodes is None
            or state.selected_node not in available_nodes
        ) else available_nodes.index(state.selected_node),
        help="Choose a wind resource settlement point to analyze"
    )

    return selected_node


def handle_node_change(old_node: str, new_node: str) -> None:
    """
    Handle node selection change and auto-select best date range.

    Parameters
    ----------
    old_node : str
        Previously selected node
    new_node : str
        Newly selected node
    """
    state = get_state()

    if new_node != old_node:
        update_state(selected_node=new_node)
        clear_simulation_cache()

        # Auto-select best date range when node changes (database mode only)
        if state.data_source == 'database' and new_node:
            # Check if current range is valid for the new node
            db_loader = SupabaseDataLoader()
            current_availability = db_loader.get_node_availability(new_node)

            # Filter for current date range
            if not current_availability.empty:
                mask = (current_availability['date'] >= state.start_date) & \
                       (current_availability['date'] <= state.end_date)
                range_stats = current_availability[mask]

                # Calculate average completeness for the current range
                if not range_stats.empty:
                    avg_completeness = range_stats['completeness'].mean()
                else:
                    avg_completeness = 0
            else:
                avg_completeness = 0

            # Only auto-select if current range is poor (<95% complete)
            if avg_completeness < 95:
                best_range = find_best_date_range(new_node)
                if best_range:
                    start_date, end_date = best_range
                    try:
                        update_date_range(start_date, end_date)
                        state._dates_auto_selected = True
                        st.sidebar.success(
                            f"✅ Auto-selected {(end_date - start_date).days + 1} days with complete data")
                        st.rerun()
                    except ValueError:
                        pass  # If update fails, just keep current dates

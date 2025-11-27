"""
Widget Synchronization Utilities
Zentus - ERCOT Battery Revenue Dashboard

This module provides reusable components for synchronized slider + number_input pairs.
This eliminates ~200 lines of duplicated code across forecast, capacity, power, and efficiency inputs.
"""

from typing import Optional, Callable

import streamlit as st


def render_synced_slider_input(
    label: str,
    min_val: float,
    max_val: float,
    key_prefix: str,
    default_value: Optional[float] = None,
    slider_step: float = 1.0,
    input_step: float = 0.1,
    format_str: str = "%.1f",
    help_text: str = "",
    disabled: bool = False,
    on_change_callback: Optional[Callable] = None,
    input_label_visible: bool = False,
    slider_label: Optional[str] = None,
    input_label: Optional[str] = None
) -> float:
    """
    Render a synchronized slider + number_input pair.

    This component automatically keeps a slider and number_input in sync via session state,
    eliminating the need for manual synchronization boilerplate.

    Parameters
    ----------
    label : str
        Label for the slider (if slider_label is None)
    min_val : float
        Minimum value
    max_val : float
        Maximum value
    key_prefix : str
        Prefix for session state keys (e.g., "capacity" -> "capacity_master", "capacity_slider", "capacity_input")
    default_value : float, optional
        Default value (if None, uses min_val)
    slider_step : float, default=1.0
        Step size for slider
    input_step : float, default=0.1
        Step size for number input
    format_str : str, default="%.1f"
        Number format string
    help_text : str, default=""
        Help tooltip text
    disabled : bool, default=False
        Whether widgets are disabled
    on_change_callback : callable, optional
        Callback function to execute when value changes (e.g., clear_simulation_cache)
    input_label_visible : bool, default=False
        Whether to show label on number input
    slider_label : str, optional
        Override label for slider (uses label if None)
    input_label : str, optional
        Override label for number input (uses label if None)

    Returns
    -------
    float
        Current synchronized value
    """
    # Session state keys
    master_key = f"{key_prefix}_master"
    slider_key = f"{key_prefix}_slider"
    input_key = f"{key_prefix}_input"

    # Initialize master value if not present
    if master_key not in st.session_state:
        if default_value is not None:
            st.session_state[master_key] = float(default_value)
        else:
            st.session_state[master_key] = float(min_val)

    # Initialize widget values if not present
    if slider_key not in st.session_state:
        st.session_state[slider_key] = st.session_state[master_key]
    if input_key not in st.session_state:
        st.session_state[input_key] = st.session_state[master_key]

    # Callbacks to synchronize values
    def on_slider_change():
        """Slider changed - update master and input"""
        new_val = float(st.session_state[slider_key])
        st.session_state[master_key] = new_val
        st.session_state[input_key] = new_val
        if on_change_callback:
            on_change_callback()

    def on_input_change():
        """Input changed - update master and slider"""
        new_val = float(st.session_state[input_key])
        st.session_state[master_key] = new_val
        # Round slider to nearest step
        st.session_state[slider_key] = round(new_val / slider_step) * slider_step
        if on_change_callback:
            on_change_callback()

    # Two-column layout: slider + precise input
    col_slider, col_input = st.sidebar.columns([2.25, 1])

    with col_slider:
        col_slider.slider(
            slider_label or label,
            min_value=min_val,
            max_value=max_val,
            value=st.session_state[slider_key],
            step=slider_step,
            help=help_text,
            disabled=disabled,
            key=slider_key,
            on_change=on_slider_change
        )

    with col_input:
        if not input_label_visible:
            col_input.write("")  # Alignment spacer
        col_input.number_input(
            input_label or label,
            min_value=min_val,
            max_value=max_val,
            value=st.session_state[input_key],
            step=input_step,
            format=format_str,
            help=f"Enter precise {label.lower()}",
            disabled=disabled,
            key=input_key,
            on_change=on_input_change,
            label_visibility="collapsed" if not input_label_visible else "visible"
        )

    return st.session_state[master_key]


def update_synced_value(key_prefix: str, new_value: float) -> None:
    """
    Programmatically update a synced widget's value.

    Use this when you need to update the value from code (e.g., when preset changes).

    Parameters
    ----------
    key_prefix : str
        Prefix for session state keys
    new_value : float
        New value to set
    """
    master_key = f"{key_prefix}_master"
    slider_key = f"{key_prefix}_slider"
    input_key = f"{key_prefix}_input"

    st.session_state[master_key] = float(new_value)
    st.session_state[slider_key] = float(new_value)
    st.session_state[input_key] = float(new_value)

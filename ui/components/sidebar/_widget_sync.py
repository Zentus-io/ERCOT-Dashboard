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

    # master_key is the persistent source of truth that the caller reads from.
    if master_key not in st.session_state:
        st.session_state[master_key] = (
            float(default_value) if default_value is not None else float(min_val)
        )
    master_val = float(st.session_state[master_key])
    clamped_master = max(float(min_val), min(float(max_val), master_val))

    # Pattern: pass `value=` AND `key=` to both widgets. Streamlit uses
    # session_state[key] when present (so user input wins on a normal
    # rerun) but falls back to `value=` when session_state[key] has been
    # GC'd (which is what happens on cross-page navigation). This breaks
    # the "reset to widget min_value" loop we kept hitting with various
    # other patterns. Streamlit emits a soft warning about the combo —
    # accept it; correctness > silencing the warning.

    # Two-column layout: slider + precise input
    col_slider, col_input = st.sidebar.columns([2.25, 1])

    with col_slider:
        slider_val = col_slider.slider(
            slider_label or label,
            min_value=min_val,
            max_value=max_val,
            value=clamped_master,
            step=slider_step,
            help=help_text,
            disabled=disabled,
            key=slider_key,
        )

    with col_input:
        if not input_label_visible:
            col_input.write("")  # Alignment spacer
        input_val = col_input.number_input(
            input_label or label,
            min_value=min_val,
            max_value=max_val,
            value=clamped_master,
            step=input_step,
            format=format_str,
            help=f"Enter precise {label.lower()}",
            disabled=disabled,
            key=input_key,
            label_visibility="collapsed" if not input_label_visible else "visible"
        )

    # Imperative reconciliation: figure out which widget the user touched
    # (if either) and update the master. Skip when both widgets returned
    # the master's current value (no user change). Optional callback
    # fires only on a real change.
    new_slider = float(slider_val)
    new_input = float(input_val)
    if abs(new_slider - master_val) > 1e-9:
        st.session_state[master_key] = new_slider
        if on_change_callback:
            on_change_callback()
    elif abs(new_input - master_val) > 1e-9:
        st.session_state[master_key] = new_input
        if on_change_callback:
            on_change_callback()

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

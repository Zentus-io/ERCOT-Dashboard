"""
Date Range Utilities for Sidebar
Zentus - ERCOT Battery Revenue Dashboard

This module provides date range validation and auto-selection logic.
"""

from datetime import date, timedelta
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
from postgrest.exceptions import APIError

from config.settings import MAX_DAYS_RANGE
from core.data.loaders import SupabaseDataLoader


@st.cache_data(ttl=3600, show_spinner=False)
def find_best_date_range(node: str, min_completeness: float = 95.0) -> Optional[Tuple[date, date]]:
    """
    Find the longest contiguous date range with high data availability.

    Parameters
    ----------
    node : str
        Settlement point name
    min_completeness : float, default=95.0
        Minimum completeness threshold (percentage)

    Returns
    -------
    tuple[date, date] or None
        (start_date, end_date) or None if no suitable range found
    """
    try:
        db_loader = SupabaseDataLoader()
        availability_df = db_loader.get_node_availability(node)

        if availability_df.empty:
            return None

        # Filter for high-quality data
        availability_df = availability_df[availability_df['completeness']
                                          >= min_completeness].copy()

        if availability_df.empty:
            return None

        # Sort by date
        availability_df = availability_df.sort_values('date').reset_index(drop=True)

        # Convert date column to datetime.date if needed
        if not isinstance(availability_df['date'].iloc[0], date):
            availability_df['date'] = pd.to_datetime(availability_df['date']).dt.date

        # Find longest contiguous sequence
        max_length = 0
        max_start = None
        max_end = None

        current_start = availability_df['date'].iloc[0]
        current_length = 1

        for i in range(1, len(availability_df)):
            prev_date = availability_df['date'].iloc[i - 1]
            curr_date = availability_df['date'].iloc[i]

            # Check if dates are consecutive
            if (curr_date - prev_date).days == 1:
                current_length += 1
            else:
                # End of contiguous sequence
                if current_length > max_length:
                    max_length = current_length
                    max_start = current_start
                    max_end = prev_date

                # Start new sequence
                current_start = curr_date
                current_length = 1

        # Check last sequence
        if current_length > max_length:
            max_length = current_length
            max_start = current_start
            max_end = availability_df['date'].iloc[-1]

        if max_start and max_end and max_length > 0:
            # Cap the range to MAX_DAYS_RANGE if needed
            if (max_end - max_start).days > MAX_DAYS_RANGE:
                max_end = max_start + timedelta(days=MAX_DAYS_RANGE)

            return (max_start, max_end)

        return None

    except APIError as e:
        st.warning(f"⚠️ Could not auto-select date range: {str(e)}")
        return None


def validate_date_range(start: date, end: date) -> Tuple[bool, str]:
    """
    Validate a date range.

    Parameters
    ----------
    start : date
        Start date
    end : date
        End date

    Returns
    -------
    tuple[bool, str]
        (is_valid, error_message) - error_message is empty if valid
    """
    if end < start:
        return False, "⚠️ End date must be after start date"

    date_diff = (end - start).days
    if date_diff > MAX_DAYS_RANGE:
        return False, f"⚠️ Maximum range: {MAX_DAYS_RANGE} days (selected: {date_diff} days)"

    return True, ""

"""
Data Validation Utilities
Zentus - ERCOT Battery Revenue Dashboard

Provides validation functions for solar/price data alignment and clipping calculations.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, List


def validate_solar_price_alignment(
    solar_profile: pd.DataFrame,
    price_data: pd.DataFrame
) -> Tuple[bool, List[str]]:
    """
    Validate alignment between solar profile and price data.

    Checks:
    - Date range coverage (solar should cover price data period)
    - Timestamp gaps and missing data
    - Frequency consistency

    Parameters
    ----------
    solar_profile : pd.DataFrame
        Solar generation profile with DatetimeIndex
    price_data : pd.DataFrame
        Price data with DatetimeIndex

    Returns
    -------
    tuple
        (is_valid, warnings) where warnings is a list of warning messages
    """
    warnings = []
    is_valid = True

    # Check if indices are datetime
    if not isinstance(solar_profile.index, pd.DatetimeIndex):
        warnings.append("⚠️ Solar profile does not have a DatetimeIndex")
        is_valid = False
        return is_valid, warnings

    if not isinstance(price_data.index, pd.DatetimeIndex):
        warnings.append("⚠️ Price data does not have a DatetimeIndex")
        is_valid = False
        return is_valid, warnings

    # Date range check
    solar_start = solar_profile.index.min()
    solar_end = solar_profile.index.max()
    price_start = price_data.index.min()
    price_end = price_data.index.max()

    if solar_start > price_start:
        warnings.append(
            f"⚠️ Solar profile starts after price data ({solar_start.date()} vs {price_start.date()}). "
            "Early price data may be missing solar information."
        )

    if solar_end < price_end:
        warnings.append(
            f"⚠️ Solar profile ends before price data ({solar_end.date()} vs {price_end.date()}). "
            "Late price data may be missing solar information."
        )

    # Frequency inference
    try:
        solar_freq = pd.infer_freq(solar_profile.index)
        price_freq = pd.infer_freq(price_data.index)

        if solar_freq and price_freq and solar_freq != price_freq:
            warnings.append(
                f"⚠️ Frequency mismatch: Solar ({solar_freq}) vs Price ({price_freq}). "
                "Data may need resampling."
            )
    except Exception:
        # Frequency inference can fail for irregular data
        pass

    # Gap detection (check for missing timestamps)
    try:
        # Infer expected frequency from first few timestamps
        time_diffs = price_data.index[1:11] - price_data.index[0:10]
        expected_freq = time_diffs.min()

        # Generate expected timestamps
        expected_timestamps = pd.date_range(
            start=price_start,
            end=price_end,
            freq=expected_freq
        )

        missing = set(expected_timestamps) - set(price_data.index)
        if len(missing) > 0:
            missing_pct = len(missing) / len(expected_timestamps) * 100
            if missing_pct > 5:
                warnings.append(
                    f"⚠️ {len(missing)} missing timestamps ({missing_pct:.1f}% of expected data). "
                    "This may affect simulation accuracy."
                )
                is_valid = False
            elif missing_pct > 1:
                warnings.append(
                    f"ℹ️ {len(missing)} missing timestamps ({missing_pct:.1f}% of data). "
                    "Minor gaps detected."
                )
    except Exception as e:
        # Gap detection is best-effort
        warnings.append(f"ℹ️ Could not check for timestamp gaps: {e}")

    # Length check
    len_diff_pct = abs(len(solar_profile) - len(price_data)) / len(price_data) * 100
    if len_diff_pct > 10:
        warnings.append(
            f"⚠️ Length mismatch: Solar has {len(solar_profile)} rows vs Price {len(price_data)} rows "
            f"({len_diff_pct:.1f}% difference)"
        )

    return is_valid, warnings


def validate_clipped_energy(
    clipped_mw: pd.Series,
    solar_mw: pd.Series,
    interconnection_limit: float
) -> Tuple[bool, List[str]]:
    """
    Validate clipped energy calculations.

    Checks:
    - No negative clipping (calculation error)
    - Clipping doesn't exceed solar generation (calculation error)
    - Clipping percentage is reasonable
    - Warns about very high or very low clipping

    Parameters
    ----------
    clipped_mw : pd.Series
        Clipped power at each timestep (MW)
    solar_mw : pd.Series
        Solar generation at each timestep (MW)
    interconnection_limit : float
        POI limit (MW)

    Returns
    -------
    tuple
        (is_valid, warnings) where warnings is a list of warning messages
    """
    warnings = []
    is_valid = True

    # Check for negative clipping (calculation error)
    if (clipped_mw < -0.001).any():  # Small tolerance for numerical errors
        num_negative = (clipped_mw < -0.001).sum()
        warnings.append(
            f"❌ Negative clipping detected ({num_negative} timesteps). "
            "This indicates a calculation error."
        )
        is_valid = False

    # Check if clipping exceeds solar generation (calculation error)
    if (clipped_mw > solar_mw + 0.001).any():  # Small tolerance
        num_exceed = (clipped_mw > solar_mw + 0.001).sum()
        warnings.append(
            f"❌ Clipping exceeds solar generation ({num_exceed} timesteps). "
            "This indicates a calculation error."
        )
        is_valid = False

    # Calculate clipping statistics
    total_solar = solar_mw.sum()
    total_clipped = clipped_mw.sum()

    if total_solar > 0:
        clipping_pct = (total_clipped / total_solar * 100)

        # High clipping percentage (warning)
        if clipping_pct > 50:
            warnings.append(
                f"⚠️ High clipping percentage ({clipping_pct:.1f}% of total solar generation). "
                f"Consider increasing interconnection limit above {interconnection_limit:.0f} MW."
            )
        elif clipping_pct > 30:
            warnings.append(
                f"ℹ️ Moderate clipping ({clipping_pct:.1f}% of total solar generation). "
                "Battery can capture significant value."
            )

        # Very low clipping (informational)
        if clipping_pct < 1 and total_clipped > 0:
            warnings.append(
                f"ℹ️ Low clipping percentage ({clipping_pct:.2f}%). "
                "Battery may have limited opportunity from clipped energy alone."
            )
        elif total_clipped == 0:
            warnings.append(
                "ℹ️ No clipping detected. Solar generation never exceeds interconnection limit. "
                "Battery will operate purely on grid arbitrage."
            )

    # Check if solar capacity makes sense relative to interconnection limit
    max_solar = solar_mw.max()
    if max_solar > 0:
        ratio = interconnection_limit / max_solar
        if ratio > 1.5:
            warnings.append(
                f"ℹ️ Interconnection limit ({interconnection_limit:.0f} MW) is {ratio:.1f}x "
                f"max solar generation ({max_solar:.1f} MW). "
                "No clipping expected with this configuration."
            )
        elif ratio < 0.5:
            warnings.append(
                f"⚠️ Interconnection limit ({interconnection_limit:.0f} MW) is only {ratio:.1%} "
                f"of max solar generation ({max_solar:.1f} MW). "
                "Very high clipping - this may be intentional for co-location."
            )

    return is_valid, warnings


def display_validation_warnings(warnings: List[str], title: str = "⚠️ Data Validation") -> None:
    """
    Display validation warnings in an expandable Streamlit UI element.

    Parameters
    ----------
    warnings : List[str]
        List of warning messages to display
    title : str, optional
        Title for the expander (default: "⚠️ Data Validation")
    """
    if not warnings:
        return

    # Categorize warnings by severity
    errors = [w for w in warnings if w.startswith("❌")]
    warnings_high = [w for w in warnings if w.startswith("⚠️")]
    warnings_info = [w for w in warnings if w.startswith("ℹ️")]

    # Determine if we should expand by default (if there are errors)
    expanded = len(errors) > 0

    # Choose icon based on severity
    if errors:
        icon = "❌"
    elif warnings_high:
        icon = "⚠️"
    else:
        icon = "ℹ️"

    with st.expander(f"{icon} {title} ({len(warnings)} items)", expanded=expanded):
        if errors:
            st.error("**Critical Issues:**")
            for warning in errors:
                st.markdown(f"- {warning}")

        if warnings_high:
            if errors:
                st.markdown("---")
            st.warning("**Warnings:**")
            for warning in warnings_high:
                st.markdown(f"- {warning}")

        if warnings_info:
            if errors or warnings_high:
                st.markdown("---")
            st.info("**Information:**")
            for warning in warnings_info:
                st.markdown(f"- {warning}")


def validate_hybrid_configuration(
    solar_capacity_mw: float,
    interconnection_limit_mw: float,
    battery_power_mw: float,
    battery_capacity_mwh: float
) -> Tuple[bool, List[str]]:
    """
    Validate overall hybrid system configuration.

    Checks for logical consistency and common configuration issues.

    Parameters
    ----------
    solar_capacity_mw : float
        Solar DC capacity (MW)
    interconnection_limit_mw : float
        POI limit (MW)
    battery_power_mw : float
        Battery power rating (MW)
    battery_capacity_mwh : float
        Battery energy capacity (MWh)

    Returns
    -------
    tuple
        (is_valid, warnings) where warnings is a list of warning messages
    """
    warnings = []
    is_valid = True

    # Solar capacity should be > 0
    if solar_capacity_mw <= 0:
        warnings.append("❌ Solar capacity must be positive")
        is_valid = False

    # Interconnection limit should be > 0
    if interconnection_limit_mw <= 0:
        warnings.append("❌ Interconnection limit must be positive")
        is_valid = False

    # Battery power should be > 0
    if battery_power_mw <= 0:
        warnings.append("❌ Battery power must be positive")
        is_valid = False

    # Battery capacity should be > 0
    if battery_capacity_mwh <= 0:
        warnings.append("❌ Battery capacity must be positive")
        is_valid = False

    # Check battery duration (capacity / power)
    if battery_power_mw > 0 and battery_capacity_mwh > 0:
        duration_h = battery_capacity_mwh / battery_power_mw
        if duration_h < 0.5:
            warnings.append(
                f"⚠️ Battery duration is very short ({duration_h:.2f}h). "
                "Typical durations are 1-4 hours."
            )
        elif duration_h > 8:
            warnings.append(
                f"⚠️ Battery duration is very long ({duration_h:.1f}h). "
                "This is unusual for grid-scale batteries."
            )

    # Check if battery can handle clipped energy
    if interconnection_limit_mw > 0 and solar_capacity_mw > interconnection_limit_mw:
        max_clipping_mw = solar_capacity_mw - interconnection_limit_mw
        if battery_power_mw < max_clipping_mw * 0.5:
            warnings.append(
                f"ℹ️ Battery power ({battery_power_mw:.0f} MW) is smaller than typical clipping "
                f"({max_clipping_mw:.0f} MW max). May not capture all available clipped energy."
            )

    # Check if interconnection limit seems reasonable
    if interconnection_limit_mw > solar_capacity_mw * 2:
        warnings.append(
            f"ℹ️ Interconnection limit ({interconnection_limit_mw:.0f} MW) is much larger "
            f"than solar capacity ({solar_capacity_mw:.0f} MW). "
            "No clipping expected with this configuration."
        )

    return is_valid, warnings

"""
This module provides utilities for generating and styling Plotly charts.
"""
import pandas as pd
import plotly.graph_objects as go


def _get_all_x_data(fig_data):
    """Helper function to collect all x-data from figure traces."""
    all_x = []
    for trace in fig_data:
        x_data = getattr(trace, 'x', None)
        if x_data is not None:
            try:
                s = pd.Series(x_data)
                s = s.dropna()
                if not s.empty:
                    all_x.append(s)
            except TypeError:
                pass
    return all_x


def apply_standard_chart_styling(fig: go.Figure):
    """
    Applies standard styling to Plotly figures including:
    - Range slider and selectors (1d, 1w, 1m, All)
    - Vertical lines at day boundaries (midnight)
    """
    # Add range slider and selectors
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector={
            "buttons": list([
                {"count": 1, "label": "1d", "step": "day", "stepmode": "backward"},
                {"count": 7, "label": "1w", "step": "day", "stepmode": "backward"},
                {"count": 1, "label": "1m", "step": "month", "stepmode": "backward"},
                {"step": "all"}
            ])
        }
    )

    # Add day boundary markers
    if not fig.data:
        return fig

    try:
        all_x = _get_all_x_data(fig.data)

        if not all_x:
            return fig

        combined_x = pd.concat(all_x)
        if combined_x.empty:
            return fig

        # Ensure datetime
        combined_x = pd.to_datetime(combined_x)
        min_date = combined_x.min()
        max_date = combined_x.max()

        # Generate midnight timestamps
        # Start from the next midnight after min_date
        start_date = min_date.normalize() + pd.Timedelta(days=1)

        shapes = []
        current_date = start_date
        while current_date <= max_date:
            shapes.append({
                "type": "line",
                "x0": current_date,
                "y0": 0,
                "x1": current_date,
                "y1": 1,
                "xre": "x",
                "yre": "paper",
                "line": {"color": "rgba(128, 128, 128, 0.5)", "width": 1, "dash": "dot"}
            })
            current_date += pd.Timedelta(days=1)

        # Update layout with shapes, preserving existing shapes if any
        existing_shapes = fig.layout.shapes or []
        fig.update_layout(shapes=list(existing_shapes) + shapes)
    except (TypeError, ValueError):
        # Fail silently if data is not in a convertible format
        pass

    return fig

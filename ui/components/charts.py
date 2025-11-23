import plotly.graph_objects as go
import pandas as pd

def apply_standard_chart_styling(fig: go.Figure):
    """
    Applies standard styling to Plotly figures including:
    - Range slider and selectors (1d, 1w, 1m, All)
    - Vertical lines at day boundaries (midnight)
    """
    # Add range slider and selectors
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    # Add day boundary markers
    if fig.data:
        try:
            # Get all x values to find range
            all_x = []
            for trace in fig.data:
                if hasattr(trace, 'x') and trace.x is not None:
                    # Convert to pandas series to handle various formats
                    try:
                        s = pd.Series(trace.x)
                        # Drop NAs
                        s = s.dropna()
                        if not s.empty:
                            all_x.append(s)
                    except:
                        pass
            
            if all_x:
                combined_x = pd.concat(all_x)
                if not combined_x.empty:
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
                        shapes.append(dict(
                            type="line",
                            x0=current_date,
                            y0=0,
                            x1=current_date,
                            y1=1,
                            xref="x",
                            yref="paper",
                            line=dict(color="rgba(128, 128, 128, 0.5)", width=1, dash="dot")
                        ))
                        current_date += pd.Timedelta(days=1)
                    
                    # Update layout with shapes, preserving existing shapes if any
                    existing_shapes = fig.layout.shapes or []
                    fig.update_layout(shapes=list(existing_shapes) + shapes)
        except Exception as e:
            # Fail silently if x data is not datetime or accessible
            # print(f"Error adding day boundaries: {e}")
            pass

    return fig

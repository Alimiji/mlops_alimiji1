"""Chart components for WeatherPredict Pro."""

from typing import Dict, List, Optional

import plotly.express as px
import plotly.graph_objects as go


def create_metrics_chart(metrics: Dict[str, float], title: str = "Model Performance Metrics") -> go.Figure:
    """Create a grouped bar chart for model metrics.

    Args:
        metrics: Dictionary with metric values
        title: Chart title

    Returns:
        Plotly figure object
    """
    splits = ["Train", "Validation", "Test"]
    metric_types = ["RMSE", "MAE", "R²"]

    # Extract values
    rmse_values = [
        metrics.get("train_rmse", 0),
        metrics.get("valid_rmse", 0),
        metrics.get("test_rmse", 0),
    ]
    mae_values = [
        metrics.get("train_mae", 0),
        metrics.get("valid_mae", 0),
        metrics.get("test_mae", 0),
    ]
    r2_values = [
        metrics.get("train_r2", 0),
        metrics.get("valid_r2", 0),
        metrics.get("test_r2", 0),
    ]

    fig = go.Figure()

    # Add RMSE bars
    fig.add_trace(
        go.Bar(
            name="RMSE",
            x=splits,
            y=rmse_values,
            marker_color="#EF5350",
            text=[f"{v:.3f}" for v in rmse_values],
            textposition="outside",
        )
    )

    # Add MAE bars
    fig.add_trace(
        go.Bar(
            name="MAE",
            x=splits,
            y=mae_values,
            marker_color="#FFA726",
            text=[f"{v:.3f}" for v in mae_values],
            textposition="outside",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Dataset Split",
        yaxis_title="Error Value",
        barmode="group",
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_r2_chart(metrics: Dict[str, float], title: str = "R² Score by Split") -> go.Figure:
    """Create a bar chart for R² scores.

    Args:
        metrics: Dictionary with metric values
        title: Chart title

    Returns:
        Plotly figure object
    """
    splits = ["Train", "Validation", "Test"]
    r2_values = [
        metrics.get("train_r2", 0),
        metrics.get("valid_r2", 0),
        metrics.get("test_r2", 0),
    ]

    # Color based on performance
    colors = []
    for v in r2_values:
        if v >= 0.95:
            colors.append("#66BB6A")  # Green - Excellent
        elif v >= 0.90:
            colors.append("#FFA726")  # Orange - Good
        else:
            colors.append("#EF5350")  # Red - Needs improvement

    fig = go.Figure(
        go.Bar(
            x=splits,
            y=r2_values,
            marker_color=colors,
            text=[f"{v:.4f}" for v in r2_values],
            textposition="outside",
        )
    )

    # Add threshold line
    fig.add_hline(y=0.90, line_dash="dash", line_color="gray", annotation_text="Threshold (0.90)")

    fig.update_layout(
        title=title,
        xaxis_title="Dataset Split",
        yaxis_title="R² Score",
        template="plotly_white",
        height=350,
        yaxis=dict(range=[0, 1.05]),
    )

    return fig


def create_gauge_chart(value: float, title: str = "Model Performance", max_value: float = 1.0) -> go.Figure:
    """Create a gauge chart for a single metric.

    Args:
        value: Current value
        title: Chart title
        max_value: Maximum value for the gauge

    Returns:
        Plotly figure object
    """
    # Determine color based on value (assuming R² or similar 0-1 metric)
    if value >= 0.95:
        bar_color = "#66BB6A"  # Green
    elif value >= 0.90:
        bar_color = "#FFA726"  # Orange
    else:
        bar_color = "#EF5350"  # Red

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title, "font": {"size": 16}},
            delta={"reference": 0.90, "increasing": {"color": "#66BB6A"}},
            gauge={
                "axis": {"range": [0, max_value], "tickwidth": 1},
                "bar": {"color": bar_color},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 0.7], "color": "#FFEBEE"},
                    {"range": [0.7, 0.9], "color": "#FFF3E0"},
                    {"range": [0.9, 1], "color": "#E8F5E9"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 0.90,
                },
            },
        )
    )

    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))

    return fig


def create_comparison_chart(
    predictions: List[float],
    timestamps: Optional[List[str]] = None,
    title: str = "Prediction History",
) -> go.Figure:
    """Create a line chart for prediction history.

    Args:
        predictions: List of predicted values
        timestamps: Optional list of timestamps
        title: Chart title

    Returns:
        Plotly figure object
    """
    if timestamps is None:
        timestamps = [f"#{i+1}" for i in range(len(predictions))]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=predictions,
            mode="lines+markers",
            name="Predicted Temperature",
            line=dict(color="#1E88E5", width=2),
            marker=dict(size=8),
        )
    )

    # Add average line
    avg_temp = sum(predictions) / len(predictions) if predictions else 0
    fig.add_hline(
        y=avg_temp,
        line_dash="dash",
        line_color="#FFA726",
        annotation_text=f"Average: {avg_temp:.1f}°C",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Prediction",
        yaxis_title="Temperature (°C)",
        template="plotly_white",
        height=400,
        hovermode="x unified",
    )

    return fig


def create_feature_importance_chart(features: List[str], title: str = "Model Features") -> go.Figure:
    """Create a horizontal bar chart showing features used by the model.

    Args:
        features: List of feature names
        title: Chart title

    Returns:
        Plotly figure object
    """
    # Assign colors to different feature types
    feature_colors = {
        "min_temp": "#1E88E5",
        "max_temp": "#E53935",
        "global_radiation": "#FFA726",
        "sunshine": "#FFEB3B",
        "cloud_cover": "#78909C",
        "precipitation": "#42A5F5",
        "pressure": "#7E57C2",
        "snow_depth": "#90CAF9",
    }

    colors = [feature_colors.get(f, "#1E88E5") for f in features]

    fig = go.Figure(
        go.Bar(
            y=features,
            x=[1] * len(features),  # Equal weight for visualization
            orientation="h",
            marker_color=colors,
            text=features,
            textposition="inside",
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        template="plotly_white",
        height=300,
        showlegend=False,
    )

    return fig

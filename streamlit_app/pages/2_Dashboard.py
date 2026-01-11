"""
Dashboard Page - View model performance metrics and analytics.
"""

import sys
from pathlib import Path

import streamlit as st

# Add app directory to path
app_dir = Path(__file__).parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from components.charts import (
    create_feature_importance_chart,
    create_gauge_chart,
    create_metrics_chart,
    create_r2_chart,
)
from components.header import render_footer, render_header
from components.sidebar import render_sidebar

from utils.api_client import WeatherAPIClient
from utils.config import APP_CONFIG, WEATHER_ICONS

# Page config
st.set_page_config(
    page_title=f"Dashboard | {APP_CONFIG.app_name}",
    page_icon=WEATHER_ICONS["metrics"],
    layout="wide",
)


def main():
    render_sidebar()
    render_header(subtitle=f"{WEATHER_ICONS['metrics']} Model Performance Dashboard")

    # API Client
    client = WeatherAPIClient(APP_CONFIG.api_url)

    # Check API availability
    health = client.health_check()
    if not health.success:
        st.error(f"{WEATHER_ICONS['error']} API is not available. Please ensure the prediction service is running.")
        return

    # Load metrics
    metrics_response = client.get_metrics()
    model_response = client.get_model_info()

    if not metrics_response.success:
        st.warning(f"{WEATHER_ICONS['degraded']} Unable to load metrics: {metrics_response.error}")
        return

    metrics = metrics_response.data

    # Key Performance Indicators
    st.subheader("Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        test_r2 = metrics.get("test_r2", 0)
        st.metric(
            "Test R² Score",
            f"{test_r2:.4f}",
            delta="Excellent" if test_r2 > 0.95 else ("Good" if test_r2 > 0.90 else "Needs Improvement"),
            delta_color="normal" if test_r2 > 0.90 else "inverse",
        )

    with col2:
        test_rmse = metrics.get("test_rmse", 0)
        st.metric(
            "Test RMSE",
            f"{test_rmse:.3f} °C",
            delta=f"{test_rmse - 1.0:.3f}" if test_rmse != 0 else None,
            delta_color="inverse",
        )

    with col3:
        test_mae = metrics.get("test_mae", 0)
        st.metric(
            "Test MAE",
            f"{test_mae:.3f} °C",
        )

    with col4:
        valid_r2 = metrics.get("valid_r2", 0)
        st.metric(
            "Validation R²",
            f"{valid_r2:.4f}",
        )

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Error Metrics Comparison")
        fig = create_metrics_chart(metrics)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("R² Score by Split")
        fig = create_r2_chart(metrics)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Model Performance Gauge and Info
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Overall Model Performance")
        test_r2 = metrics.get("test_r2", 0)
        fig = create_gauge_chart(test_r2, "Test R² Score")
        st.plotly_chart(fig, use_container_width=True)

        # Performance interpretation
        if test_r2 >= 0.95:
            st.success(f"{WEATHER_ICONS['healthy']} Excellent model performance! The model explains 95%+ of variance.")
        elif test_r2 >= 0.90:
            st.info(f"{WEATHER_ICONS['sun']} Good model performance. The model explains 90%+ of variance.")
        else:
            st.warning(f"{WEATHER_ICONS['degraded']} Model performance below threshold. Consider retraining.")

    with col2:
        st.subheader("Model Information")

        if model_response.success:
            model_info = model_response.data

            st.markdown(
                f"""
                | Property | Value |
                |----------|-------|
                | **Model Type** | {model_info.get('model_type', 'Unknown')} |
                | **Experiment** | {model_info.get('experiment_name', 'Unknown')} |
                | **Run ID** | `{model_info.get('run_id', 'Unknown')[:20]}...` |
                """
            )

            st.markdown("**Hyperparameters:**")
            params = model_info.get("params", {})
            param_cols = st.columns(3)
            param_items = list(params.items())

            for i, (key, value) in enumerate(param_items):
                with param_cols[i % 3]:
                    st.metric(key, str(value) if value is not None else "None")

            # Features chart
            features = model_info.get("features", [])
            if features:
                st.markdown("**Model Features:**")
                fig = create_feature_importance_chart(features)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Unable to load model information.")

    st.markdown("---")

    # Detailed Metrics Table
    st.subheader("Detailed Metrics")

    metrics_table = {
        "Split": ["Train", "Validation", "Test"],
        "RMSE (°C)": [
            f"{metrics.get('train_rmse', 0):.4f}",
            f"{metrics.get('valid_rmse', 0):.4f}",
            f"{metrics.get('test_rmse', 0):.4f}",
        ],
        "MAE (°C)": [
            f"{metrics.get('train_mae', 0):.4f}",
            f"{metrics.get('valid_mae', 0):.4f}",
            f"{metrics.get('test_mae', 0):.4f}",
        ],
        "R² Score": [
            f"{metrics.get('train_r2', 0):.4f}",
            f"{metrics.get('valid_r2', 0):.4f}",
            f"{metrics.get('test_r2', 0):.4f}",
        ],
    }

    import pandas as pd

    df = pd.DataFrame(metrics_table)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Metric explanations
    with st.expander("Metric Explanations"):
        st.markdown(
            """
            - **RMSE (Root Mean Square Error)**: Average prediction error in °C. Lower is better.
            - **MAE (Mean Absolute Error)**: Average absolute prediction error in °C. Lower is better.
            - **R² Score**: Proportion of variance explained by the model. Closer to 1.0 is better.
            - **Train**: Performance on training data (may show overfitting if much better than test).
            - **Validation**: Performance on validation data used for hyperparameter tuning.
            - **Test**: Performance on held-out test data (best indicator of real-world performance).
            """
        )

    render_footer()


if __name__ == "__main__":
    main()

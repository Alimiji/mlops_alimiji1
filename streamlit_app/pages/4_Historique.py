"""
History Page - Track and export prediction history.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# Add app directory to path
app_dir = Path(__file__).parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from components.charts import create_comparison_chart
from components.header import render_footer, render_header
from components.sidebar import render_sidebar

from utils.config import APP_CONFIG, WEATHER_ICONS

# Page config
st.set_page_config(
    page_title=f"History | {APP_CONFIG.app_name}",
    page_icon=WEATHER_ICONS["history"],
    layout="wide",
)

# Initialize session state for history
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


def main():
    render_sidebar()
    render_header(subtitle=f"{WEATHER_ICONS['history']} Prediction History")

    # Check if there's any history
    if not st.session_state.prediction_history:
        st.info(
            f"{WEATHER_ICONS['prediction']} No predictions yet. " "Go to the Predictions page to make some forecasts!"
        )

        # Demo data option
        if st.button("Load Demo Data"):
            # Generate demo history
            demo_data = []
            for i in range(20):
                demo_data.append(
                    {
                        "timestamp": (datetime.now() - timedelta(hours=i)).isoformat(),
                        "prediction": 10.0 + (i % 10) - 5 + (i * 0.1),
                        "model_version": "demo_version_123",
                        "min_temp": 5.0 + (i % 5),
                        "max_temp": 15.0 + (i % 5),
                        "global_radiation": 100.0 + (i * 5),
                        "sunshine": 6.0 + (i % 3),
                        "cloud_cover": 5.0,
                        "precipitation": i % 3,
                        "pressure": 101325.0,
                        "snow_depth": 0.0,
                    }
                )
            st.session_state.prediction_history = demo_data
            st.rerun()

        render_footer()
        return

    # Convert to DataFrame
    df = pd.DataFrame(st.session_state.prediction_history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp", ascending=False)

    # Summary stats
    st.subheader("Summary Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Predictions", len(df))

    with col2:
        st.metric("Average Temperature", f"{df['prediction'].mean():.2f} °C")

    with col3:
        st.metric("Min Prediction", f"{df['prediction'].min():.2f} °C")

    with col4:
        st.metric("Max Prediction", f"{df['prediction'].max():.2f} °C")

    st.markdown("---")

    # Filters
    st.subheader("Filters")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Date range filter
        min_date = df["timestamp"].min().date()
        max_date = df["timestamp"].max().date()

        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

    with col2:
        # Temperature range filter
        temp_range = st.slider(
            "Temperature Range (°C)",
            min_value=float(df["prediction"].min()),
            max_value=float(df["prediction"].max()),
            value=(float(df["prediction"].min()), float(df["prediction"].max())),
        )

    with col3:
        # Model version filter
        versions = df["model_version"].unique().tolist()
        selected_versions = st.multiselect("Model Version", versions, default=versions)

    # Apply filters
    filtered_df = df.copy()

    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df["timestamp"].dt.date >= start_date) & (filtered_df["timestamp"].dt.date <= end_date)
        ]

    filtered_df = filtered_df[
        (filtered_df["prediction"] >= temp_range[0]) & (filtered_df["prediction"] <= temp_range[1])
    ]

    if selected_versions:
        filtered_df = filtered_df[filtered_df["model_version"].isin(selected_versions)]

    st.caption(f"Showing {len(filtered_df)} of {len(df)} predictions")

    st.markdown("---")

    # Visualization
    st.subheader("Prediction Trend")

    if len(filtered_df) > 0:
        # Time series chart
        fig = create_comparison_chart(
            predictions=filtered_df["prediction"].tolist(),
            timestamps=filtered_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist(),
            title="Temperature Predictions Over Time",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Distribution
        col1, col2 = st.columns(2)

        with col1:
            import plotly.express as px

            fig = px.histogram(
                filtered_df,
                x="prediction",
                nbins=20,
                title="Prediction Distribution",
                labels={"prediction": "Predicted Temperature (°C)"},
            )
            fig.update_layout(template="plotly_white", height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(
                filtered_df,
                y="prediction",
                title="Prediction Box Plot",
                labels={"prediction": "Predicted Temperature (°C)"},
            )
            fig.update_layout(template="plotly_white", height=300)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data matches the selected filters.")

    st.markdown("---")

    # Data Table
    st.subheader("Prediction Records")

    # Select columns to display
    display_columns = st.multiselect(
        "Columns to Display",
        options=filtered_df.columns.tolist(),
        default=["timestamp", "prediction", "min_temp", "max_temp", "model_version"],
    )

    if display_columns:
        st.dataframe(
            filtered_df[display_columns].reset_index(drop=True),
            use_container_width=True,
            height=400,
        )

    st.markdown("---")

    # Export Options
    st.subheader("Export Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        # CSV Export
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        # JSON Export
        json_data = filtered_df.to_json(orient="records", date_format="iso")
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )

    with col3:
        # Clear History
        if st.button("Clear History", type="secondary", use_container_width=True):
            st.session_state.prediction_history = []
            st.rerun()

    render_footer()


if __name__ == "__main__":
    main()

"""
Predictions Page - Make single and batch temperature predictions.
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# Add app directory to path
app_dir = Path(__file__).parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from components.header import render_footer, render_header  # noqa: E402
from components.sidebar import render_sidebar  # noqa: E402

from utils.api_client import WeatherAPIClient  # noqa: E402
from utils.config import APP_CONFIG, FEATURE_DESCRIPTIONS, WEATHER_ICONS  # noqa: E402

# Page config
st.set_page_config(
    page_title=f"Predictions | {APP_CONFIG.app_name}",
    page_icon=WEATHER_ICONS["prediction"],
    layout="wide",
)

# Initialize session state for history
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


def add_to_history(features: dict, prediction: float, model_version: str):
    """Add prediction to session history."""
    st.session_state.prediction_history.append(
        {
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction,
            "model_version": model_version,
            **features,
        }
    )


def main():
    render_sidebar()
    render_header(subtitle=f"{WEATHER_ICONS['prediction']} Temperature Predictions")

    # API Client
    client = WeatherAPIClient(APP_CONFIG.api_url)

    # Check API availability
    if not client.is_available():
        st.error(f"{WEATHER_ICONS['error']} API is not available. Please ensure the prediction service is running.")
        st.info(f"Expected API URL: {APP_CONFIG.api_url}")
        return

    # Tabs for single and batch predictions
    tab1, tab2 = st.tabs([f"{WEATHER_ICONS['sun']} Single Prediction", f"{WEATHER_ICONS['cloud']} Batch Prediction"])

    with tab1:
        st.markdown("### Enter Weather Features")
        st.caption("Provide the weather conditions to predict the mean temperature.")

        # Feature input form
        col1, col2 = st.columns(2)

        limits = APP_CONFIG.feature_limits

        with col1:
            min_temp = st.number_input(
                f"{WEATHER_ICONS['snow']} Minimum Temperature ({limits['min_temp']['unit']})",
                min_value=limits["min_temp"]["min"],
                max_value=limits["min_temp"]["max"],
                value=limits["min_temp"]["default"],
                help=FEATURE_DESCRIPTIONS["min_temp"],
            )

            max_temp = st.number_input(
                f"{WEATHER_ICONS['temperature']} Maximum Temperature ({limits['max_temp']['unit']})",
                min_value=limits["max_temp"]["min"],
                max_value=limits["max_temp"]["max"],
                value=limits["max_temp"]["default"],
                help=FEATURE_DESCRIPTIONS["max_temp"],
            )

            global_radiation = st.number_input(
                f"{WEATHER_ICONS['sun']} Global Radiation ({limits['global_radiation']['unit']})",
                min_value=limits["global_radiation"]["min"],
                max_value=limits["global_radiation"]["max"],
                value=limits["global_radiation"]["default"],
                help=FEATURE_DESCRIPTIONS["global_radiation"],
            )

            sunshine = st.slider(
                f"{WEATHER_ICONS['sun']} Sunshine Hours",
                min_value=limits["sunshine"]["min"],
                max_value=limits["sunshine"]["max"],
                value=limits["sunshine"]["default"],
                help=FEATURE_DESCRIPTIONS["sunshine"],
            )

        with col2:
            cloud_cover = st.slider(
                f"{WEATHER_ICONS['cloud']} Cloud Cover (oktas)",
                min_value=limits["cloud_cover"]["min"],
                max_value=limits["cloud_cover"]["max"],
                value=limits["cloud_cover"]["default"],
                step=1.0,
                help=FEATURE_DESCRIPTIONS["cloud_cover"],
            )

            precipitation = st.number_input(
                f"{WEATHER_ICONS['rain']} Precipitation ({limits['precipitation']['unit']})",
                min_value=limits["precipitation"]["min"],
                max_value=limits["precipitation"]["max"],
                value=limits["precipitation"]["default"],
                help=FEATURE_DESCRIPTIONS["precipitation"],
            )

            pressure = st.number_input(
                f"{WEATHER_ICONS['pressure']} Atmospheric Pressure ({limits['pressure']['unit']})",
                min_value=limits["pressure"]["min"],
                max_value=limits["pressure"]["max"],
                value=limits["pressure"]["default"],
                help=FEATURE_DESCRIPTIONS["pressure"],
            )

            snow_depth = st.number_input(
                f"{WEATHER_ICONS['snow']} Snow Depth ({limits['snow_depth']['unit']})",
                min_value=limits["snow_depth"]["min"],
                max_value=limits["snow_depth"]["max"],
                value=limits["snow_depth"]["default"],
                help=FEATURE_DESCRIPTIONS["snow_depth"],
            )

        # Validation
        if min_temp > max_temp:
            st.warning("Minimum temperature should be less than or equal to maximum temperature.")

        # Predict button
        st.markdown("---")

        if st.button("Predict Temperature", type="primary", use_container_width=True):
            features = {
                "min_temp": min_temp,
                "max_temp": max_temp,
                "global_radiation": global_radiation,
                "sunshine": sunshine,
                "cloud_cover": cloud_cover,
                "precipitation": precipitation,
                "pressure": pressure,
                "snow_depth": snow_depth,
            }

            with st.spinner("Making prediction..."):
                response = client.predict(features)

            if response.success:
                prediction = response.data["predicted_mean_temp"]
                model_version = response.data["model_version"]

                # Add to history
                add_to_history(features, prediction, model_version)

                # Display result
                st.success(f"{WEATHER_ICONS['temperature']} Prediction Complete!")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Mean Temperature", f"{prediction} °C")
                with col2:
                    st.metric("Model Version", model_version[:12] + "...")
                with col3:
                    st.metric("Latency", f"{response.latency_ms:.0f} ms")

            else:
                st.error(f"{WEATHER_ICONS['error']} Prediction failed: {response.error}")

    with tab2:
        st.markdown("### Batch Prediction")
        st.caption("Upload a CSV file with weather features to get multiple predictions.")

        # Template download
        template_df = pd.DataFrame(
            {
                "min_temp": [5.0, 10.0],
                "max_temp": [15.0, 20.0],
                "global_radiation": [100.0, 150.0],
                "sunshine": [6.0, 8.0],
                "cloud_cover": [5.0, 3.0],
                "precipitation": [0.0, 2.0],
                "pressure": [101325.0, 101500.0],
                "snow_depth": [0.0, 0.0],
            }
        )

        csv_template = template_df.to_csv(index=False)
        st.download_button(
            label="Download CSV Template",
            data=csv_template,
            file_name="weather_features_template.csv",
            mime="text/csv",
        )

        # File upload
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                # Validate columns
                required_cols = [
                    "min_temp",
                    "max_temp",
                    "global_radiation",
                    "sunshine",
                    "cloud_cover",
                    "precipitation",
                    "pressure",
                    "snow_depth",
                ]
                missing_cols = [col for col in required_cols if col not in df.columns]

                if missing_cols:
                    st.error(f"Missing columns: {', '.join(missing_cols)}")
                else:
                    st.success(f"Loaded {len(df)} records")
                    st.dataframe(df.head(10), use_container_width=True)

                    if len(df) > 1000:
                        st.warning("Maximum 1000 records per batch. Only first 1000 will be processed.")
                        df = df.head(1000)

                    if st.button("Run Batch Prediction", type="primary"):
                        instances = df[required_cols].to_dict("records")

                        with st.spinner(f"Processing {len(instances)} predictions..."):
                            response = client.predict_batch(instances)

                        if response.success:
                            predictions = response.data["predictions"]
                            df["predicted_mean_temp"] = predictions

                            st.success(f"{WEATHER_ICONS['healthy']} Batch prediction complete!")
                            st.metric("Predictions Made", response.data["count"])
                            st.metric("Latency", f"{response.latency_ms:.0f} ms")

                            st.dataframe(df, use_container_width=True)

                            # Download results
                            csv_results = df.to_csv(index=False)
                            st.download_button(
                                label="Download Results CSV",
                                data=csv_results,
                                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                            )
                        else:
                            st.error(f"{WEATHER_ICONS['error']} Batch prediction failed: {response.error}")

            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

    render_footer()


if __name__ == "__main__":
    main()

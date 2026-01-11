"""
WeatherPredict Pro - Professional Weather Temperature Prediction Platform

Main application entry point with multi-page navigation.
"""

import sys
from pathlib import Path

import streamlit as st

# Add app directory to path for imports
app_dir = Path(__file__).parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from components.header import render_footer, render_header
from components.sidebar import render_sidebar

from utils.config import APP_CONFIG, WEATHER_ICONS

# Page configuration
st.set_page_config(
    page_title=APP_CONFIG.app_name,
    page_icon=WEATHER_ICONS["temperature"],
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-repo/mlops_alimiji1",
        "Report a bug": "https://github.com/your-repo/mlops_alimiji1/issues",
        "About": f"{APP_CONFIG.app_name} v{APP_CONFIG.app_version} - {APP_CONFIG.app_description}",
    },
)

# Custom CSS for professional look
st.markdown(
    """
    <style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1E88E5;
    }

    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
    }

    .stButton > button:hover {
        border-color: #1E88E5;
        color: #1E88E5;
    }

    /* Input styling */
    .stNumberInput > div > div > input {
        border-radius: 8px;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #F5F7FA;
    }

    /* Success/Error message styling */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 8px;
    }

    /* Table styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    """Main application function."""
    # Render sidebar
    render_sidebar()

    # Render header
    render_header(subtitle=f"{WEATHER_ICONS['sun']} Welcome to the Weather Prediction Platform")

    # Main content
    st.markdown(
        """
        ### About This Platform

        **WeatherPredict Pro** is a professional-grade machine learning platform for
        temperature prediction. Our model uses advanced Random Forest algorithms trained
        on decades of London weather data to provide accurate mean temperature forecasts.
        """
    )

    # Feature cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>{WEATHER_ICONS['prediction']} Predictions</h3>
                <p>Make single or batch temperature predictions</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>{WEATHER_ICONS['metrics']} Dashboard</h3>
                <p>View model performance metrics and analytics</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>{WEATHER_ICONS['monitoring']} Monitoring</h3>
                <p>Real-time API health and system status</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>{WEATHER_ICONS['history']} History</h3>
                <p>Track and export prediction history</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick stats
    st.subheader("Model Overview")

    from utils.api_client import WeatherAPIClient

    client = WeatherAPIClient(APP_CONFIG.api_url)

    col1, col2 = st.columns(2)

    with col1:
        metrics_response = client.get_metrics()
        if metrics_response.success:
            metrics = metrics_response.data
            st.metric(
                "Test R² Score",
                f"{metrics.get('test_r2', 0):.4f}",
                delta="Excellent" if metrics.get("test_r2", 0) > 0.95 else None,
            )
            st.metric("Test RMSE", f"{metrics.get('test_rmse', 0):.3f} °C")
            st.metric("Test MAE", f"{metrics.get('test_mae', 0):.3f} °C")
        else:
            st.warning("Unable to load metrics. Check API connection.")

    with col2:
        model_response = client.get_model_info()
        if model_response.success:
            model_info = model_response.data
            st.metric("Model Type", model_info.get("model_type", "Unknown"))
            st.metric("Estimators", model_info.get("params", {}).get("n_estimators", "N/A"))
            st.metric("Features", len(model_info.get("features", [])))
        else:
            st.warning("Unable to load model info. Check API connection.")

    # Getting started
    st.markdown("---")
    st.subheader("Getting Started")
    st.markdown(
        """
        1. **Navigate** to the **Predictions** page to make temperature forecasts
        2. **Check** the **Dashboard** for detailed model performance metrics
        3. **Monitor** the **Monitoring** page for API health status
        4. **Review** the **History** page to see and export past predictions

        Use the sidebar to navigate between pages.
        """
    )

    # Render footer
    render_footer()


if __name__ == "__main__":
    main()

"""Sidebar component for WeatherPredict Pro."""

import streamlit as st

from utils.api_client import WeatherAPIClient
from utils.config import APP_CONFIG, WEATHER_ICONS


def render_sidebar():
    """Render the sidebar with API status and configuration."""
    with st.sidebar:
        # Logo and branding
        st.markdown(
            f"""
            <div style="text-align: center; padding: 1rem 0;">
                <h2 style="color: #1E88E5; margin: 0;">
                    {WEATHER_ICONS['temperature']} WeatherPredict
                </h2>
                <p style="color: #666; font-size: 0.9rem;">Pro Edition</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # API Status Section
        st.subheader(f"{WEATHER_ICONS['monitoring']} API Status")

        client = WeatherAPIClient(APP_CONFIG.api_url)
        health_response = client.health_check()

        if health_response.success:
            status = health_response.data.get("status", "unknown")
            model_loaded = health_response.data.get("model_loaded", False)

            if status == "healthy" and model_loaded:
                st.success(f"{WEATHER_ICONS['healthy']} API Healthy")
                st.caption(f"Latency: {health_response.latency_ms:.0f}ms")
            else:
                st.warning(f"{WEATHER_ICONS['degraded']} API Degraded")
                if not model_loaded:
                    st.caption("Model not loaded")
        else:
            st.error(f"{WEATHER_ICONS['error']} API Unavailable")
            st.caption(health_response.error)

        st.markdown("---")

        # Configuration Section
        st.subheader("Configuration")
        st.caption(f"**API URL:** {APP_CONFIG.api_url}")

        # Quick Actions
        st.markdown("---")
        st.subheader("Quick Actions")

        if st.button("Refresh Status", use_container_width=True):
            st.rerun()

        # Navigation hints
        st.markdown("---")
        st.caption("**Navigation**")
        st.markdown(
            f"""
            - {WEATHER_ICONS['prediction']} **Predictions**: Make forecasts
            - {WEATHER_ICONS['metrics']} **Dashboard**: View metrics
            - {WEATHER_ICONS['monitoring']} **Monitoring**: API health
            - {WEATHER_ICONS['history']} **History**: Past predictions
            """
        )

        # Version info at bottom
        st.markdown("---")
        st.caption(f"v{APP_CONFIG.app_version}")


def get_api_status() -> dict:
    """Get current API status for display.

    Returns:
        Dictionary with status information
    """
    client = WeatherAPIClient(APP_CONFIG.api_url)
    health_response = client.health_check()

    if health_response.success:
        return {
            "available": True,
            "status": health_response.data.get("status", "unknown"),
            "model_loaded": health_response.data.get("model_loaded", False),
            "latency_ms": health_response.latency_ms,
        }
    else:
        return {
            "available": False,
            "status": "unavailable",
            "model_loaded": False,
            "latency_ms": health_response.latency_ms,
            "error": health_response.error,
        }

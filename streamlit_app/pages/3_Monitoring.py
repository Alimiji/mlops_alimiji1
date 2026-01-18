"""
Monitoring Page - Real-time API health and system status.
"""

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# Add app directory to path
app_dir = Path(__file__).parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from components.header import render_footer, render_header  # noqa: E402
from components.sidebar import render_sidebar  # noqa: E402

from utils.api_client import WeatherAPIClient  # noqa: E402
from utils.config import APP_CONFIG, WEATHER_ICONS  # noqa: E402

# Page config
st.set_page_config(
    page_title=f"Monitoring | {APP_CONFIG.app_name}",
    page_icon=WEATHER_ICONS["monitoring"],
    layout="wide",
)

# Initialize session state for monitoring
if "health_history" not in st.session_state:
    st.session_state.health_history = []


def main():
    render_sidebar()
    render_header(subtitle=f"{WEATHER_ICONS['monitoring']} System Monitoring")

    # API Client
    client = WeatherAPIClient(APP_CONFIG.api_url)

    # Auto-refresh option
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f"Monitoring API at: `{APP_CONFIG.api_url}`")
    with col2:
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)

    if auto_refresh:
        try:
            from streamlit_autorefresh import st_autorefresh

            st_autorefresh(interval=30000, key="monitoring_refresh")
        except ImportError:
            st.info("Install streamlit-autorefresh for auto-refresh functionality")

    st.markdown("---")

    # Current Status
    st.subheader("Current API Status")

    with st.spinner("Checking API health..."):
        health_response = client.health_check()
        check_time = datetime.now()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if health_response.success:
            status = health_response.data.get("status", "unknown")
            if status == "healthy":
                st.success(f"{WEATHER_ICONS['healthy']} Healthy")
            else:
                st.warning(f"{WEATHER_ICONS['degraded']} Degraded")
        else:
            st.error(f"{WEATHER_ICONS['error']} Unavailable")

    with col2:
        if health_response.success:
            model_loaded = health_response.data.get("model_loaded", False)
            if model_loaded:
                st.success(f"{WEATHER_ICONS['healthy']} Model Loaded")
            else:
                st.warning(f"{WEATHER_ICONS['degraded']} Model Not Loaded")
        else:
            st.error(f"{WEATHER_ICONS['error']} Unknown")

    with col3:
        st.metric("Response Time", f"{health_response.latency_ms:.0f} ms")

    with col4:
        st.metric("Last Check", check_time.strftime("%H:%M:%S"))

    # Add to history
    st.session_state.health_history.append(
        {
            "timestamp": check_time,
            "status": "healthy" if health_response.success else "error",
            "latency_ms": health_response.latency_ms,
        }
    )
    # Keep only last 100 entries
    st.session_state.health_history = st.session_state.health_history[-100:]

    st.markdown("---")

    # Detailed Health Information
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("API Details")

        if health_response.success:
            st.json(health_response.data)
        else:
            st.error(f"Error: {health_response.error}")

        # API Info
        root_response = client.get_root()
        if root_response.success:
            st.markdown("**API Information:**")
            st.json(root_response.data)

    with col2:
        st.subheader("Model Status")

        model_response = client.get_model_info()

        if model_response.success:
            model_info = model_response.data

            st.markdown(f"""
                | Property | Value |
                |----------|-------|
                | **Model Type** | {model_info.get('model_type', 'Unknown')} |
                | **Run ID** | `{model_info.get('run_id', 'Unknown')[:16]}...` |
                | **Experiment** | {model_info.get('experiment_name', 'Unknown')} |
                | **Features** | {len(model_info.get('features', []))} |
                """)

            # Reload button
            st.markdown("---")
            if st.button("Reload Model", type="secondary"):
                with st.spinner("Reloading model..."):
                    reload_response = client.reload_model()

                if reload_response.success:
                    st.success(f"{WEATHER_ICONS['healthy']} Model reloaded successfully!")
                    st.rerun()
                else:
                    st.error(f"{WEATHER_ICONS['error']} Reload failed: {reload_response.error}")
        else:
            st.warning(f"Unable to get model info: {model_response.error}")

    st.markdown("---")

    # Latency History Chart
    st.subheader("Response Time History")

    if len(st.session_state.health_history) > 1:
        import pandas as pd
        import plotly.express as px

        df = pd.DataFrame(st.session_state.health_history)

        fig = px.line(
            df,
            x="timestamp",
            y="latency_ms",
            title="API Response Time",
            labels={"latency_ms": "Latency (ms)", "timestamp": "Time"},
        )
        fig.update_traces(mode="lines+markers")
        fig.update_layout(template="plotly_white", height=300)

        st.plotly_chart(fig, use_container_width=True)

        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Latency", f"{df['latency_ms'].mean():.0f} ms")
        with col2:
            st.metric("Min Latency", f"{df['latency_ms'].min():.0f} ms")
        with col3:
            st.metric("Max Latency", f"{df['latency_ms'].max():.0f} ms")
    else:
        st.info("Refresh the page to collect latency data for the chart.")

    st.markdown("---")

    # Endpoint Tests
    st.subheader("Endpoint Health Tests")

    endpoints = [
        ("GET /", "Root", client.get_root),
        ("GET /health", "Health Check", client.health_check),
        ("GET /metrics", "Metrics", client.get_metrics),
        ("GET /model/info", "Model Info", client.get_model_info),
    ]

    cols = st.columns(len(endpoints))

    for i, (endpoint, name, func) in enumerate(endpoints):
        with cols[i]:
            response = func()
            if response.success:
                st.success(f"{WEATHER_ICONS['healthy']} {name}")
                st.caption(f"{response.latency_ms:.0f}ms")
            else:
                st.error(f"{WEATHER_ICONS['error']} {name}")
                st.caption("Failed")

    # Manual refresh button
    st.markdown("---")
    if st.button("Refresh All", type="primary", use_container_width=True):
        st.rerun()

    render_footer()


if __name__ == "__main__":
    main()

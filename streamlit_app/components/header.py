"""Header component for WeatherPredict Pro."""

import streamlit as st

from utils.config import APP_CONFIG, WEATHER_ICONS


def render_header(subtitle: str = None):
    """Render the application header with branding.

    Args:
        subtitle: Optional subtitle for the current page
    """
    # Custom CSS for header styling
    st.markdown(
        """
        <style>
        .main-header {
            background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%);
            padding: 1.5rem 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .main-header h1 {
            color: white;
            margin: 0;
            font-size: 2.2rem;
            font-weight: 700;
        }
        .main-header p {
            color: rgba(255, 255, 255, 0.9);
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
        }
        .header-badge {
            background: rgba(255, 255, 255, 0.2);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            display: inline-block;
            margin-top: 0.5rem;
        }
        .page-subtitle {
            color: #1E88E5;
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #E3F2FD;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Main header
    header_html = f"""
    <div class="main-header">
        <h1>{WEATHER_ICONS['temperature']} {APP_CONFIG.app_name}</h1>
        <p>{APP_CONFIG.app_description}</p>
        <span class="header-badge">v{APP_CONFIG.app_version} | {APP_CONFIG.company_name}</span>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    # Page subtitle if provided
    if subtitle:
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def render_footer():
    """Render the application footer."""
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption(f"{APP_CONFIG.company_name}")

    with col2:
        st.caption(f"Version {APP_CONFIG.app_version}")

    with col3:
        st.caption("Powered by ML & FastAPI")

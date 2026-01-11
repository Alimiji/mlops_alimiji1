"""Configuration module for WeatherPredict Pro UI."""

import os
from dataclasses import dataclass
from typing import Optional

import streamlit as st


@dataclass
class AppConfig:
    """Application configuration."""

    app_name: str = "WeatherPredict Pro"
    app_version: str = "1.0.0"
    app_description: str = "Professional Weather Temperature Prediction Platform"
    company_name: str = "WeatherPredict Analytics"

    # API Configuration
    api_url: str = "http://localhost:8000"

    # Feature limits for validation
    feature_limits: dict = None

    # Theme colors
    primary_color: str = "#1E88E5"
    secondary_color: str = "#FFA726"
    success_color: str = "#66BB6A"
    warning_color: str = "#FF7043"
    error_color: str = "#EF5350"

    def __post_init__(self):
        if self.feature_limits is None:
            self.feature_limits = {
                "min_temp": {"min": -50.0, "max": 50.0, "default": 5.0, "unit": "°C"},
                "max_temp": {"min": -40.0, "max": 60.0, "default": 15.0, "unit": "°C"},
                "global_radiation": {"min": 0.0, "max": 500.0, "default": 100.0, "unit": "W/m²"},
                "sunshine": {"min": 0.0, "max": 24.0, "default": 6.0, "unit": "hours"},
                "cloud_cover": {"min": 0.0, "max": 10.0, "default": 5.0, "unit": "oktas"},
                "precipitation": {"min": 0.0, "max": 200.0, "default": 0.0, "unit": "mm"},
                "pressure": {"min": 87000.0, "max": 108500.0, "default": 101325.0, "unit": "Pa"},
                "snow_depth": {"min": 0.0, "max": 500.0, "default": 0.0, "unit": "cm"},
            }


def get_api_url() -> str:
    """Get API URL from environment or Streamlit secrets."""
    # Try Streamlit secrets first (for Streamlit Cloud)
    try:
        if hasattr(st, "secrets") and "API_URL" in st.secrets:
            return st.secrets["API_URL"]
    except Exception:
        pass

    # Fall back to environment variable
    return os.getenv("API_URL", "http://localhost:8000")


def get_config() -> AppConfig:
    """Get application configuration."""
    config = AppConfig()
    config.api_url = get_api_url()
    return config


# Global configuration instance
APP_CONFIG = get_config()


# Weather icons mapping
WEATHER_ICONS = {
    "temperature": "🌡️",
    "sun": "☀️",
    "cloud": "☁️",
    "rain": "🌧️",
    "snow": "❄️",
    "wind": "💨",
    "pressure": "📊",
    "radiation": "☢️",
    "healthy": "✅",
    "degraded": "⚠️",
    "error": "❌",
    "prediction": "🎯",
    "metrics": "📈",
    "history": "📋",
    "monitoring": "🔍",
}


# Feature descriptions for UI
FEATURE_DESCRIPTIONS = {
    "min_temp": "Minimum temperature recorded during the day",
    "max_temp": "Maximum temperature recorded during the day",
    "global_radiation": "Total solar radiation received",
    "sunshine": "Hours of bright sunshine",
    "cloud_cover": "Sky coverage by clouds (0=clear, 10=overcast)",
    "precipitation": "Total rainfall or water equivalent",
    "pressure": "Atmospheric pressure at sea level",
    "snow_depth": "Depth of snow on ground",
}

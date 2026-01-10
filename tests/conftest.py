"""Pytest configuration and fixtures."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
import joblib


@pytest.fixture
def sample_weather_data():
    """Generate sample weather data for testing."""
    np.random.seed(42)
    n_samples = 100

    # Generate temperatures with proper ordering (min <= mean <= max)
    min_temp = np.random.uniform(-5, 10, n_samples)
    temp_range = np.random.uniform(5, 15, n_samples)
    max_temp = min_temp + temp_range
    mean_temp = (min_temp + max_temp) / 2 + np.random.uniform(-2, 2, n_samples)
    # Ensure mean is between min and max
    mean_temp = np.clip(mean_temp, min_temp, max_temp)

    data = {
        "date": pd.date_range(start="2020-01-01", periods=n_samples, freq="D"),
        "min_temp": min_temp,
        "max_temp": max_temp,
        "mean_temp": mean_temp,
        "sunshine": np.random.uniform(0, 12, n_samples),
        "global_radiation": np.random.uniform(20, 200, n_samples),
        "precipitation": np.random.uniform(0, 20, n_samples),
        "pressure": np.random.uniform(98000, 103000, n_samples),
        "cloud_cover": np.random.uniform(0, 10, n_samples),
        "snow_depth": np.random.uniform(0, 5, n_samples),
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_features():
    """Generate sample feature arrays for testing."""
    np.random.seed(42)
    n_samples = 50

    X = np.random.randn(n_samples, 8)
    y = np.random.randn(n_samples)

    return X, y


@pytest.fixture
def temp_model_dir(sample_features):
    """Create a temporary directory with a trained model."""
    X, y = sample_features

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "models" / "random_forest" / "Production"
        model_dir.mkdir(parents=True)

        # Train a simple model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Save model
        model_path = model_dir / "model.pkl"
        joblib.dump(model, model_path)

        # Save model info
        model_info = {
            "run_id": "test_run_123",
            "experiment_name": "test_experiment",
            "model_type": "RandomForestRegressor",
            "params": {"n_estimators": 10, "random_state": 42},
        }
        (model_dir / "model_info.json").write_text(json.dumps(model_info))

        yield tmpdir


@pytest.fixture
def sample_prediction_input():
    """Sample input for prediction endpoints."""
    return {
        "min_temp": 5.2,
        "max_temp": 12.8,
        "global_radiation": 45.0,
        "sunshine": 3.5,
        "cloud_cover": 6.0,
        "precipitation": 0.5,
        "pressure": 101325.0,
        "snow_depth": 0.0,
    }


@pytest.fixture
def sample_batch_input(sample_prediction_input):
    """Sample batch input for prediction endpoints."""
    return {"instances": [sample_prediction_input for _ in range(5)]}
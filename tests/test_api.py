"""Tests for the FastAPI application."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient


class TestAPIEndpoints:
    """Tests for API endpoints."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = MagicMock()
        model.predict.return_value = np.array([10.5])
        return model

    @pytest.fixture
    def client(self, mock_model):
        """Create test client with mocked model."""
        from src.api import main

        # Mock the global model
        main.model = mock_model
        main.model_info = {"run_id": "test_run_123"}
        main.metrics = {
            "train": {"rmse": 0.5, "mae": 0.3, "r2": 0.95},
            "valid": {"rmse": 0.8, "mae": 0.5, "r2": 0.90},
            "test": {"rmse": 1.0, "mae": 0.7, "r2": 0.85},
        }

        return TestClient(main.app)

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data

    def test_health_endpoint_with_model(self, client):
        """Test health endpoint when model is loaded."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_predict_endpoint(self, client, sample_prediction_input):
        """Test single prediction endpoint."""
        response = client.post("/predict", json=sample_prediction_input)
        assert response.status_code == 200
        data = response.json()
        assert "predicted_mean_temp" in data
        assert "model_version" in data
        assert isinstance(data["predicted_mean_temp"], (int, float))

    def test_predict_batch_endpoint(self, client, sample_batch_input, mock_model):
        """Test batch prediction endpoint."""
        # Update mock for batch prediction
        mock_model.predict.return_value = np.array([10.5, 11.0, 10.8, 11.2, 10.9])

        response = client.post("/predict/batch", json=sample_batch_input)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "count" in data
        assert len(data["predictions"]) == 5

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "train_rmse" in data
        assert "valid_r2" in data
        assert "test_mae" in data

    def test_model_info_endpoint(self, client):
        """Test model info endpoint."""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "run_id" in data
        assert "features" in data


class TestAPIValidation:
    """Tests for API input validation."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = MagicMock()
        model.predict.return_value = np.array([10.5])
        return model

    @pytest.fixture
    def client(self, mock_model):
        """Create test client with mocked model."""
        from src.api import main

        main.model = mock_model
        main.model_info = {"run_id": "test_run_123"}

        return TestClient(main.app)

    def test_predict_missing_field(self, client):
        """Test prediction with missing required field."""
        incomplete_input = {
            "min_temp": 5.2,
            "max_temp": 12.8,
            # Missing other required fields
        }
        response = client.post("/predict", json=incomplete_input)
        assert response.status_code == 422  # Validation error

    def test_predict_invalid_sunshine_range(self, client):
        """Test prediction with invalid sunshine value."""
        invalid_input = {
            "min_temp": 5.2,
            "max_temp": 12.8,
            "global_radiation": 45.0,
            "sunshine": 30.0,  # Invalid: > 24 hours
            "cloud_cover": 6.0,
            "precipitation": 0.5,
            "pressure": 101325.0,
            "snow_depth": 0.0,
        }
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422

    def test_predict_negative_precipitation(self, client):
        """Test prediction with negative precipitation."""
        invalid_input = {
            "min_temp": 5.2,
            "max_temp": 12.8,
            "global_radiation": 45.0,
            "sunshine": 3.5,
            "cloud_cover": 6.0,
            "precipitation": -5.0,  # Invalid: negative
            "pressure": 101325.0,
            "snow_depth": 0.0,
        }
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422

    def test_batch_predict_empty_list(self, client):
        """Test batch prediction with empty list."""
        response = client.post("/predict/batch", json={"instances": []})
        assert response.status_code == 422


class TestAPIWithoutModel:
    """Tests for API behavior when model is not loaded."""

    @pytest.fixture
    def client_no_model(self):
        """Create test client without model."""
        from src.api import main

        main.model = None
        main.model_info = None
        main.metrics = None

        return TestClient(main.app)

    def test_health_degraded_without_model(self, client_no_model):
        """Test health endpoint returns degraded status."""
        response = client_no_model.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False

    def test_predict_fails_without_model(self, client_no_model, sample_prediction_input):
        """Test prediction fails when model not loaded."""
        response = client_no_model.post("/predict", json=sample_prediction_input)
        assert response.status_code == 503

    def test_metrics_fails_without_data(self, client_no_model):
        """Test metrics endpoint fails when no metrics available."""
        response = client_no_model.get("/metrics")
        assert response.status_code == 404
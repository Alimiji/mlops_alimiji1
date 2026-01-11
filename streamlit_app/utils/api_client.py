"""API Client for WeatherPredict Pro."""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass
class APIResponse:
    """Standard API response wrapper."""

    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    latency_ms: float = 0.0


class WeatherAPIClient:
    """Client for Weather Prediction API."""

    def __init__(self, base_url: str, timeout: int = 30):
        """Initialize API client.

        Args:
            base_url: Base URL of the API (e.g., http://localhost:8000)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _request(self, method: str, endpoint: str, **kwargs) -> APIResponse:
        """Make HTTP request and return wrapped response.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional arguments for requests

        Returns:
            APIResponse with success status, data, and latency
        """
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()

        try:
            response = self.session.request(
                method=method,
                url=url,
                timeout=self.timeout,
                **kwargs,
            )
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code >= 400:
                error_detail = response.json().get("detail", response.text)
                return APIResponse(
                    success=False,
                    error=f"HTTP {response.status_code}: {error_detail}",
                    latency_ms=latency_ms,
                )

            return APIResponse(
                success=True,
                data=response.json(),
                latency_ms=latency_ms,
            )

        except requests.exceptions.ConnectionError:
            latency_ms = (time.time() - start_time) * 1000
            return APIResponse(
                success=False,
                error="Connection failed. Is the API server running?",
                latency_ms=latency_ms,
            )
        except requests.exceptions.Timeout:
            latency_ms = (time.time() - start_time) * 1000
            return APIResponse(
                success=False,
                error=f"Request timed out after {self.timeout} seconds",
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return APIResponse(
                success=False,
                error=f"Unexpected error: {str(e)}",
                latency_ms=latency_ms,
            )

    def health_check(self) -> APIResponse:
        """Check API health status.

        Returns:
            APIResponse with health status data
        """
        return self._request("GET", "/health")

    def get_root(self) -> APIResponse:
        """Get API root information.

        Returns:
            APIResponse with API info
        """
        return self._request("GET", "/")

    def predict(self, features: Dict[str, float]) -> APIResponse:
        """Make single temperature prediction.

        Args:
            features: Dictionary with weather features

        Returns:
            APIResponse with prediction
        """
        return self._request("POST", "/predict", json=features)

    def predict_batch(self, instances: List[Dict[str, float]]) -> APIResponse:
        """Make batch temperature predictions.

        Args:
            instances: List of feature dictionaries

        Returns:
            APIResponse with predictions
        """
        return self._request("POST", "/predict/batch", json={"instances": instances})

    def get_metrics(self) -> APIResponse:
        """Get model performance metrics.

        Returns:
            APIResponse with metrics data
        """
        return self._request("GET", "/metrics")

    def get_model_info(self) -> APIResponse:
        """Get model metadata.

        Returns:
            APIResponse with model info
        """
        return self._request("GET", "/model/info")

    def reload_model(self) -> APIResponse:
        """Reload model from disk.

        Returns:
            APIResponse with reload status
        """
        return self._request("POST", "/model/reload")

    def is_available(self) -> bool:
        """Check if API is available.

        Returns:
            True if API responds, False otherwise
        """
        response = self.health_check()
        return response.success

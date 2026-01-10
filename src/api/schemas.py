"""Pydantic schemas for API request/response validation."""

from typing import List, Optional
from pydantic import BaseModel, Field


class WeatherFeatures(BaseModel):
    """Input features for temperature prediction."""

    min_temp: float = Field(..., description="Minimum temperature (Celsius)")
    max_temp: float = Field(..., description="Maximum temperature (Celsius)")
    global_radiation: float = Field(..., ge=0, description="Global radiation (W/m2)")
    sunshine: float = Field(..., ge=0, le=24, description="Hours of sunshine")
    cloud_cover: float = Field(..., ge=0, le=10, description="Cloud cover (oktas, 0-10)")
    precipitation: float = Field(..., ge=0, description="Precipitation (mm)")
    pressure: float = Field(..., gt=0, description="Atmospheric pressure (Pa)")
    snow_depth: float = Field(..., ge=0, description="Snow depth (cm)")

    class Config:
        json_schema_extra = {
            "example": {
                "min_temp": 5.2,
                "max_temp": 12.8,
                "global_radiation": 45.0,
                "sunshine": 3.5,
                "cloud_cover": 6.0,
                "precipitation": 0.5,
                "pressure": 101325.0,
                "snow_depth": 0.0
            }
        }


class PredictionResponse(BaseModel):
    """Response containing the predicted mean temperature."""

    predicted_mean_temp: float = Field(..., description="Predicted mean temperature (Celsius)")
    model_version: str = Field(..., description="Model version/run ID")

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_mean_temp": 9.5,
                "model_version": "random_forest_v1"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""

    instances: List[WeatherFeatures] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    """Response containing multiple predictions."""

    predictions: List[float] = Field(..., description="List of predicted mean temperatures")
    model_version: str = Field(..., description="Model version/run ID")
    count: int = Field(..., description="Number of predictions made")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_path: Optional[str] = Field(None, description="Path to loaded model")


class ModelMetrics(BaseModel):
    """Model performance metrics."""

    train_rmse: float
    train_mae: float
    train_r2: float
    valid_rmse: float
    valid_mae: float
    valid_r2: float
    test_rmse: float
    test_mae: float
    test_r2: float
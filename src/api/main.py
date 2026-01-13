"""FastAPI application for weather temperature prediction."""

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelMetrics,
    PredictionResponse,
    WeatherFeatures,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model storage
model = None
model_info = None
metrics = None

# Paths
ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT / "models" / "random_forest" / "Production" / "model.pkl"
MODEL_INFO_PATH = ROOT / "models" / "random_forest" / "Production" / "model_info.json"
METRICS_PATH = ROOT / "metrics.json"

# Feature order (must match training)
FEATURE_ORDER = [
    "min_temp",
    "max_temp",
    "global_radiation",
    "sunshine",
    "cloud_cover",
    "precipitation",
    "pressure",
    "snow_depth",
]


def download_model_from_dagshub():
    """Download model from DagsHub via HTTP."""
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    dagshub_user = os.getenv("DAGSHUB_USER", "Alimiji")

    if not dagshub_token:
        logger.warning("DAGSHUB_TOKEN not set, skipping model download")
        return False

    # DVC file hashes from dvc.lock
    files_to_download = {
        "models/random_forest/Production/model.pkl": "44bebd223b998cf7e177aed1e73de3a6",
        "models/random_forest/Production/model_info.json": "006851e7426c173879e57b2b91201121",
    }

    base_url = "https://dagshub.com/Alimiji/mlops_alimiji1.dvc/files/md5"

    try:
        for file_path, md5_hash in files_to_download.items():
            full_path = ROOT / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # DagsHub DVC storage URL format: /files/md5/{first2chars}/{remaining}
            url = f"{base_url}/{md5_hash[:2]}/{md5_hash[2:]}"
            logger.info(f"Downloading {file_path} from DagsHub...")

            response = requests.get(
                url,
                auth=(dagshub_user, dagshub_token),
                stream=True,
                timeout=300,
            )
            response.raise_for_status()

            with open(full_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Downloaded {file_path} successfully")

        return True

    except requests.RequestException as e:
        logger.error(f"Failed to download model: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading model: {e}")
        return False


def load_model():
    """Load the trained model and metadata."""
    global model, model_info, metrics

    # Try to download model if not present
    if not MODEL_PATH.exists():
        logger.info("Model not found locally, downloading from DagsHub...")
        download_model_from_dagshub()

    if not MODEL_PATH.exists():
        logger.error(f"Model not found at {MODEL_PATH}")
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    logger.info(f"Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    if MODEL_INFO_PATH.exists():
        model_info = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))
        logger.info(f"Model info loaded: run_id={model_info.get('run_id', 'unknown')}")
    else:
        model_info = {"run_id": "unknown"}

    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        logger.info("Metrics loaded successfully")

    logger.info("Model loaded successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    try:
        load_model()
    except FileNotFoundError as e:
        logger.warning(f"Model not loaded at startup: {e}")
    yield
    # Shutdown
    logger.info("Shutting down API")


# Create FastAPI app
app = FastAPI(
    title="Weather Temperature Prediction API",
    description="API for predicting mean temperature based on weather features using Random Forest model",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Weather Temperature Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_path=str(MODEL_PATH) if model is not None else None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(features: WeatherFeatures):
    """
    Predict mean temperature from weather features.

    Returns the predicted mean temperature in Celsius.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert features to numpy array in correct order
    feature_values = [getattr(features, name) for name in FEATURE_ORDER]
    X = np.array([feature_values])

    # Make prediction
    prediction = model.predict(X)[0]

    return PredictionResponse(
        predicted_mean_temp=round(float(prediction), 2),
        model_version=model_info.get("run_id", "unknown") if model_info else "unknown",
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction for multiple weather instances.

    Accepts up to 1000 instances per request.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert all instances to numpy array
    X = np.array([[getattr(instance, name) for name in FEATURE_ORDER] for instance in request.instances])

    # Make predictions
    predictions = model.predict(X)

    return BatchPredictionResponse(
        predictions=[round(float(p), 2) for p in predictions],
        model_version=model_info.get("run_id", "unknown") if model_info else "unknown",
        count=len(predictions),
    )


@app.get("/metrics", response_model=ModelMetrics, tags=["Model Info"])
async def get_metrics():
    """Get model performance metrics from the last evaluation."""
    if metrics is None:
        raise HTTPException(status_code=404, detail="Metrics not available")

    return ModelMetrics(
        train_rmse=metrics["train"]["rmse"],
        train_mae=metrics["train"]["mae"],
        train_r2=metrics["train"]["r2"],
        valid_rmse=metrics["valid"]["rmse"],
        valid_mae=metrics["valid"]["mae"],
        valid_r2=metrics["valid"]["r2"],
        test_rmse=metrics["test"]["rmse"],
        test_mae=metrics["test"]["mae"],
        test_r2=metrics["test"]["r2"],
    )


@app.get("/model/info", tags=["Model Info"])
async def get_model_info():
    """Get information about the loaded model."""
    if model_info is None:
        raise HTTPException(status_code=404, detail="Model info not available")

    return {
        "run_id": model_info.get("run_id"),
        "experiment_name": model_info.get("experiment_name"),
        "model_type": model_info.get("model_type"),
        "params": model_info.get("params"),
        "features": FEATURE_ORDER,
    }


@app.post("/model/reload", tags=["Model Info"])
async def reload_model():
    """Reload the model from disk."""
    try:
        load_model()
        return {"status": "success", "message": "Model reloaded successfully"}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

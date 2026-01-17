---
title: Weather Prediction API
emoji: 🌡️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# Weather Temperature Prediction API

FastAPI-based machine learning API for predicting mean temperature based on weather features.

## Features

- **Single Prediction**: POST `/predict` with weather features
- **Batch Prediction**: POST `/predict/batch` for multiple predictions
- **Model Metrics**: GET `/metrics` for performance metrics
- **Health Check**: GET `/health` for API status

## API Documentation

Once running, access the interactive API docs at `/docs`

## Model

- **Algorithm**: Random Forest Regressor
- **Features**: min_temp, max_temp, global_radiation, sunshine, cloud_cover, precipitation, pressure, snow_depth
- **Target**: Mean temperature (°C)
- **Performance**: R² = 0.96, RMSE = 1.10°C

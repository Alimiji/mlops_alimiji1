---
title: Weather Prediction API
emoji: 🌤️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# Weather Temperature Prediction API

A FastAPI-based ML API for predicting mean temperature using Random Forest.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/metrics` | GET | Model metrics |
| `/docs` | GET | Swagger UI |

## Usage Example

```python
import requests

response = requests.post(
    "https://YOUR-SPACE.hf.space/predict",
    json={
        "min_temp": 5.0,
        "max_temp": 15.0,
        "global_radiation": 100.0,
        "sunshine": 6.0,
        "cloud_cover": 5.0,
        "precipitation": 0.0,
        "pressure": 101325.0,
        "snow_depth": 0.0
    }
)
print(response.json())
```

## Model Info

- **Algorithm**: Random Forest Regressor
- **Target**: Mean Temperature (°C)
- **Features**: 8 weather variables
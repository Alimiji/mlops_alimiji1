# WeatherPredict Pro - MLOps Pipeline

A production-ready MLOps platform for weather temperature prediction using Machine Learning, with automated CI/CD deployment to Render (API) and Streamlit Cloud (UI).

---

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Architecture](#architecture)
- [MLOps Workflow](#mlops-workflow)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Deployment](#deployment)
- [GitHub Secrets Configuration](#github-secrets-configuration)
- [API Documentation](#api-documentation)
- [Development](#development)
- [CI/CD Pipelines](#cicd-pipelines)

---

## Project Overview

WeatherPredict Pro predicts mean temperature based on 8 weather features using a Random Forest regression model trained on 40+ years of London weather data (1979-2020).

### Model Performance

| Split | RMSE | MAE | R2 Score |
|-------|------|-----|----------|
| Train | 0.334 | 0.250 | 0.997 |
| Valid | 0.959 | 0.745 | 0.975 |
| Test | 1.105 | 0.719 | 0.960 |

### Features Used

| Feature | Description | Unit |
|---------|-------------|------|
| `min_temp` | Minimum daily temperature | C |
| `max_temp` | Maximum daily temperature | C |
| `global_radiation` | Solar radiation received | W/m2 |
| `sunshine` | Hours of bright sunshine | hours |
| `cloud_cover` | Sky coverage (0=clear, 10=overcast) | oktas |
| `precipitation` | Total rainfall | mm |
| `pressure` | Atmospheric pressure | Pa |
| `snow_depth` | Snow depth on ground | cm |

---

## Technologies Used

### Machine Learning & Data

| Technology | Purpose | Why We Use It |
|------------|---------|---------------|
| **scikit-learn** | ML model training | Industry-standard library for Random Forest and model evaluation |
| **pandas** | Data manipulation | Efficient DataFrame operations for data cleaning and transformation |
| **NumPy** | Numerical computing | Fast array operations for feature matrices |
| **DVC** | Data versioning | Git for data - tracks datasets and model files without storing them in Git |
| **MLflow** | Experiment tracking | Logs parameters, metrics, and models for reproducibility |

### API & Backend

| Technology | Purpose | Why We Use It |
|------------|---------|---------------|
| **FastAPI** | REST API framework | Modern, fast Python API with automatic OpenAPI documentation |
| **Pydantic** | Data validation | Type-safe request/response models with automatic validation |
| **Uvicorn** | ASGI server | High-performance async server for production deployment |

### Frontend (UI)

| Technology | Purpose | Why We Use It |
|------------|---------|---------------|
| **Streamlit** | Web interface | Rapid development of interactive ML dashboards |
| **Requests** | HTTP client | API communication from UI to backend |

### DevOps & Deployment

| Technology | Purpose | Why We Use It |
|------------|---------|---------------|
| **Docker** | Containerization | Consistent environment across dev, test, and production |
| **GitHub Actions** | CI/CD automation | Automated testing, building, and deployment pipelines |
| **Render** | API hosting | Free tier available, easy deployment from GitHub |
| **Streamlit Cloud** | UI hosting | Free hosting for Streamlit apps with GitHub integration |

### Code Quality

| Technology | Purpose | Why We Use It |
|------------|---------|---------------|
| **pytest** | Testing framework | Unit and integration testing with coverage reports |
| **Black** | Code formatter | Consistent Python code style |
| **isort** | Import sorter | Organized imports |
| **flake8** | Linter | Code quality and style checking |

---

## Architecture

```
                                    [GitHub Repository]
                                           |
                    +----------------------+----------------------+
                    |                      |                      |
                    v                      v                      v
            [GitHub Actions]        [GitHub Actions]       [GitHub Actions]
               CI Pipeline          DVC Pipeline           CD Pipeline
                    |                      |                      |
                    v                      v                      v
              Run Tests            Train Model            Build Docker
              Lint Code            Track Metrics          Push to GHCR
                    |                      |                      |
                    +----------------------+----------------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
            [Render.com]                                [Streamlit Cloud]
          FastAPI Backend                               Streamlit Frontend
       weather-prediction-api                          weatherpredict-pro
                    |                                             |
                    +--------------------+------------------------+
                                         |
                                         v
                                    [End Users]
                              Make predictions via UI
                              or directly via API
```

### Data Flow

```
[Raw Data] --> [DVC] --> [Feature Engineering] --> [Model Training] --> [MLflow]
                                                          |
                                                          v
                                                   [Saved Model]
                                                          |
                                                          v
                                                   [FastAPI Server]
                                                          |
                                +-------------------------+-------------------------+
                                |                                                   |
                                v                                                   v
                         [REST API]                                        [Streamlit UI]
                    /predict endpoint                               User-friendly interface
```

---

## MLOps Workflow

### 1. Data Versioning (DVC)

DVC tracks large data files and models without storing them in Git:

```yaml
# dvc.yaml - Pipeline definition
stages:
  make_dataset:
    cmd: python src/data/make_dataset.py
    deps: [data/raw/london_weather.csv]
    outs: [data/interim, data/processed]

  build_features:
    cmd: python src/features/build_features.py
    deps: [data/processed]
    outs: [data/features]

  train:
    cmd: python src/models/train_model.py
    deps: [data/features, params.yaml]
    outs: [models/random_forest/Production/model.pkl]

  evaluate:
    cmd: python src/models/evaluate_model.py
    deps: [models/random_forest/Production/model.pkl]
    metrics: [metrics.json]
```

**Commands:**
```bash
dvc pull          # Download data from remote (Google Drive)
dvc repro         # Reproduce entire pipeline
dvc push          # Upload data to remote
```

### 2. Experiment Tracking (MLflow)

MLflow logs all experiments for reproducibility:

```python
# Example from train_model.py
import mlflow

with mlflow.start_run():
    mlflow.log_params({"n_estimators": 300, "max_depth": None})
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_metrics({"rmse": 1.105, "r2": 0.960})
```

**View experiments:**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

### 3. Model Serving (FastAPI)

The API serves predictions via REST endpoints:

```python
# src/api/main.py
@app.post("/predict")
async def predict(features: WeatherFeatures):
    prediction = model.predict([[
        features.min_temp,
        features.max_temp,
        features.global_radiation,
        # ...
    ]])
    return {"predicted_mean_temp": prediction[0]}
```

### 4. CI/CD (GitHub Actions)

Automated pipelines ensure code quality and deployment:

| Workflow | Trigger | Actions |
|----------|---------|---------|
| `ci.yml` | Push/PR | Lint, test, coverage |
| `dvc_pipeline.yml` | Model changes | Run DVC pipeline, compare metrics |
| `mlflow.yml` | Model changes | Train, evaluate, validate thresholds |
| `model-validation-gate.yml` | PR | Check RMSE < 1.5, R2 > 0.90 |
| `cd_api.yml` | Push to master | Build Docker, deploy to Render |
| `streamlit.yml` | UI changes | Test and validate Streamlit app |

### 5. Monitoring

The Streamlit UI provides real-time monitoring:
- API health checks
- Response latency tracking
- Prediction history
- Model metrics dashboard

---

## Project Structure

```
mlops_alimiji1/
|-- .github/
|   |-- workflows/
|       |-- ci.yml                 # Lint, test, coverage
|       |-- cd_api.yml             # Build & deploy API to Render
|       |-- dvc_pipeline.yml       # Run DVC pipeline
|       |-- mlflow.yml             # MLflow training
|       |-- model-validation-gate.yml  # Model quality gate
|       |-- streamlit.yml          # Streamlit CI
|
|-- data/
|   |-- raw/                       # Original dataset (DVC tracked)
|   |-- interim/                   # Cleaned data
|   |-- processed/                 # Train/valid/test splits
|   |-- features/                  # Feature matrices (.npy)
|
|-- src/
|   |-- api/
|   |   |-- main.py                # FastAPI application
|   |   |-- schemas.py             # Pydantic validation models
|   |-- data/
|   |   |-- make_dataset.py        # Data loading & cleaning
|   |-- features/
|   |   |-- build_features.py      # Feature engineering
|   |-- models/
|   |   |-- train_model.py         # Model training with MLflow
|   |   |-- evaluate_model.py      # Model evaluation
|   |-- utils/
|       |-- config.py              # Configuration management
|
|-- streamlit_app/
|   |-- app.py                     # Main Streamlit entry point
|   |-- pages/
|   |   |-- 1_Predictions.py       # Single/batch predictions
|   |   |-- 2_Dashboard.py         # Model metrics
|   |   |-- 3_Monitoring.py        # API health monitoring
|   |   |-- 4_Historique.py        # Prediction history
|   |-- components/                # Reusable UI components
|   |-- utils/
|       |-- api_client.py          # HTTP client for API
|       |-- config.py              # UI configuration
|
|-- models/                        # Trained model artifacts
|-- tests/                         # Unit and integration tests
|-- monitoring/                    # Prometheus configuration
|
|-- Dockerfile                     # API container
|-- docker-compose.yml             # Multi-service orchestration
|-- render.yaml                    # Render deployment config
|-- railway.json                   # Railway deployment config
|-- dvc.yaml                       # DVC pipeline definition
|-- params.yaml                    # Model hyperparameters
|-- requirements.txt               # Python dependencies
|-- Makefile                       # Development commands
|-- DEPLOYMENT.md                  # Deployment guide
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Git
- Docker (optional)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/your-username/mlops_alimiji1.git
cd mlops_alimiji1

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull data with DVC
dvc pull
```

### Run Pipeline

```bash
# Execute complete ML pipeline
dvc repro

# Or use Makefile
make pipeline
```

### Start API

```bash
# Local
make run-api
# or
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Docker
docker-compose up -d api
```

### Start UI

```bash
# Local
make streamlit
# or
cd streamlit_app && streamlit run app.py

# Docker
docker-compose up -d streamlit
```

---

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete deployment instructions.

### Quick Deploy

#### API on Render

1. Go to [render.com](https://render.com) > New > Web Service
2. Connect your GitHub repo
3. Render will auto-detect `render.yaml`
4. Deploy!

#### UI on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. New app > Select repo > `streamlit_app/app.py`
3. Add secret: `API_URL = "https://your-api.onrender.com"`
4. Deploy!

---

## GitHub Secrets Configuration

Configure these secrets in your GitHub repository for CI/CD automation:

**Settings > Secrets and variables > Actions > New repository secret**

### Required Secrets

| Secret Name | Description | How to Get It |
|-------------|-------------|---------------|
| `RENDER_API_KEY` | Render API key for deployment | Render Dashboard > Account Settings > API Keys |
| `RENDER_SERVICE_ID` | Your Render service ID | Render Dashboard > Service > Settings (starts with `srv-`) |

### Optional Secrets

| Secret Name | Description | How to Get It |
|-------------|-------------|---------------|
| `GDRIVE_CREDENTIALS_DATA` | Google Drive credentials for DVC | Google Cloud Console > Service Account > JSON key |
| `CODECOV_TOKEN` | Code coverage reporting | [codecov.io](https://codecov.io) > Your repo > Settings |

### Step-by-Step: Adding Secrets

1. **Go to your GitHub repository**
2. **Click "Settings"** (tab at the top)
3. **Click "Secrets and variables"** (left sidebar)
4. **Click "Actions"**
5. **Click "New repository secret"**
6. **Enter the secret name and value**
7. **Click "Add secret"**

### Getting Render Credentials

```
1. Log in to https://dashboard.render.com
2. Go to Account Settings (click your avatar)
3. Go to "API Keys"
4. Click "Create API Key"
5. Copy the key (starts with 'rnd_')

For Service ID:
1. Go to your deployed service
2. Go to Settings
3. Copy the "Service ID" (starts with 'srv-')
```

---

## API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions (1-1000) |
| `/metrics` | GET | Model performance metrics |
| `/model/info` | GET | Model metadata |
| `/model/reload` | POST | Reload model from disk |

### Example Requests

#### Single Prediction

```bash
curl -X POST "https://your-api.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "min_temp": 5.2,
    "max_temp": 12.8,
    "global_radiation": 45.0,
    "sunshine": 3.5,
    "cloud_cover": 6.0,
    "precipitation": 0.5,
    "pressure": 101325.0,
    "snow_depth": 0.0
  }'
```

**Response:**
```json
{
  "predicted_mean_temp": 8.95,
  "model_version": "Production",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Health Check

```bash
curl https://your-api.onrender.com/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

---

## Development

### Run Tests

```bash
# All tests with coverage
pytest tests/ -v --cov=src

# Specific test file
pytest tests/test_api.py -v
```

### Code Quality

```bash
# Format code
make format
# or
black src/ tests/ && isort src/ tests/

# Lint
make lint
# or
flake8 src/ tests/
```

### Makefile Commands

```bash
make help           # Show all commands
make install        # Install dependencies
make test           # Run tests
make lint           # Run linter
make format         # Format code
make run-api        # Start API locally
make streamlit      # Start Streamlit UI
make docker-build   # Build Docker image
make docker-up      # Start all services
make pipeline       # Run DVC pipeline
```

---

## CI/CD Pipelines

### Pipeline Flow

```
[Push to master]
       |
       v
[CI Pipeline] -----> [Tests Pass?] -----> No ----> [Stop]
       |                   |
       |                  Yes
       v                   |
[DVC Pipeline]             |
       |                   |
       v                   v
[Model Validation] --> [CD Pipeline]
       |                   |
       v                   v
[Metrics OK?]         [Build Docker]
       |                   |
      Yes                  v
       |              [Push to GHCR]
       v                   |
[MLflow Log]               v
                    [Deploy to Render]
```

### Workflow Files

| File | Purpose |
|------|---------|
| `.github/workflows/ci.yml` | Linting (Black, isort, flake8), Testing (pytest), Coverage |
| `.github/workflows/cd_api.yml` | Build Docker, Push to GHCR, Deploy to Render |
| `.github/workflows/dvc_pipeline.yml` | Pull data, Run pipeline, Compare metrics |
| `.github/workflows/mlflow.yml` | Train model, Log to MLflow, Validate thresholds |
| `.github/workflows/model-validation-gate.yml` | Check RMSE <= 1.5, R2 >= 0.90 |
| `.github/workflows/streamlit.yml` | Test Streamlit app imports and config |

---

## Configuration Files

### params.yaml (Model Hyperparameters)

```yaml
random_forest:
  n_estimators: 300
  max_depth: null
  min_samples_split: 2
  min_samples_leaf: 1
  random_state: 42
  n_jobs: -1
```

### render.yaml (Render Deployment)

```yaml
services:
  - type: web
    name: weather-prediction-api
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
    healthCheckPath: /health
    autoDeploy: true
```

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Contact

For questions or support, open an issue on GitHub.# Test MLOps workflow - 20260117_055913

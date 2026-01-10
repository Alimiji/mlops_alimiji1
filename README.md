# Weather Temperature Prediction - MLOps Project

A complete MLOps pipeline for predicting mean temperature based on weather features using Random Forest regression.

## Project Overview

This project demonstrates a production-ready MLOps workflow including:
- **Data versioning** with DVC
- **Experiment tracking** with MLflow
- **Model serving** with FastAPI
- **CI/CD** with GitHub Actions
- **Containerization** with Docker

### Model Performance

| Split | RMSE | MAE | R2 |
|-------|------|-----|-----|
| Train | 0.334 | 0.250 | 0.997 |
| Valid | 0.959 | 0.745 | 0.975 |
| Test | 1.105 | 0.719 | 0.960 |

## Project Structure

```
mlops_alimiji1/
├── data/
│   ├── raw/                 # Original dataset (DVC tracked)
│   ├── interim/             # Cleaned data
│   ├── processed/           # Train/valid/test splits
│   └── features/            # Feature matrices (.npy)
├── src/
│   ├── data/
│   │   └── make_dataset.py  # Data loading & cleaning
│   ├── features/
│   │   └── build_features.py # Feature engineering
│   ├── models/
│   │   ├── train_model.py   # Model training
│   │   └── evaluate_model.py # Model evaluation
│   └── api/
│       ├── main.py          # FastAPI application
│       └── schemas.py       # Pydantic models
├── models/                  # Trained models
├── tests/                   # Unit and integration tests
├── .github/workflows/       # CI/CD pipelines
├── dvc.yaml                 # DVC pipeline definition
├── params.yaml              # Model hyperparameters
├── Dockerfile               # Container configuration
└── docker-compose.yml       # Multi-service deployment
```

## Quick Start

### Prerequisites

- Python 3.10+
- pip
- Docker (optional)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mlops_alimiji1
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Pull data with DVC:
```bash
dvc pull
```

### Run the Pipeline

Execute the complete ML pipeline:
```bash
dvc repro
```

Or run individual stages:
```bash
# Data preparation
python src/data/make_dataset.py

# Feature engineering
python src/features/build_features.py

# Training
python src/models/train_model.py

# Evaluation
python src/models/evaluate_model.py --model-path models/random_forest/Production/model.pkl
```

### Start the API

#### Local
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

#### Docker
```bash
docker-compose up -d api
```

API documentation available at: http://localhost:8000/docs

### API Usage

#### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
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

#### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"min_temp": 5.2, "max_temp": 12.8, ...}]}'
```

## Development

### Run Tests
```bash
pytest tests/ -v --cov=src
```

### Code Quality
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
flake8 src/ tests/
```

### MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## CI/CD Pipelines

| Workflow | Trigger | Description |
|----------|---------|-------------|
| `ci.yml` | Push/PR | Linting, testing, coverage |
| `dvc_pipeline.yml` | Push to main | Run DVC pipeline |
| `mlflow.yml` | Model changes | Train and evaluate |
| `cd_api.yml` | API changes | Build and deploy Docker |

## Configuration

### Model Parameters (`params.yaml`)
```yaml
random_forest:
  n_estimators: 300
  max_depth: null
  min_samples_split: 2
  min_samples_leaf: 1
  random_state: 42
  n_jobs: -1
```

### DVC Remote
Configured with Google Drive. Set credentials:
```bash
export GDRIVE_CREDENTIALS_DATA='<service-account-json>'
```

## Data

**Dataset**: London Weather Data (1979-2020)
- **Features**: min_temp, max_temp, global_radiation, sunshine, cloud_cover, precipitation, pressure, snow_depth
- **Target**: mean_temp
- **Split**: Chronological (Train: 1979-2016, Valid: 2017-2018, Test: 2019-2020)

## License

MIT License
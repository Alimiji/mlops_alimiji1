.PHONY: install test lint format clean run-api docker-build docker-up docker-down pipeline mlflow streamlit streamlit-install

# Python environment
PYTHON := python
PIP := pip
PYTEST := pytest
UVICORN := uvicorn

# Project paths
SRC := src
TESTS := tests

# Installation
install:
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install pytest pytest-cov black isort flake8

# Testing
test:
	$(PYTEST) $(TESTS)/ -v

test-cov:
	$(PYTEST) $(TESTS)/ -v --cov=$(SRC) --cov-report=html --cov-report=term-missing

# Code quality
lint:
	flake8 $(SRC)/ $(TESTS)/ --max-line-length=120

format:
	black $(SRC)/ $(TESTS)/
	isort $(SRC)/ $(TESTS)/

check-format:
	black --check $(SRC)/ $(TESTS)/
	isort --check-only $(SRC)/ $(TESTS)/

# Run API locally
run-api:
	$(UVICORN) src.api.main:app --reload --host 0.0.0.0 --port 8000

# Docker commands
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d api

docker-up-full:
	docker-compose --profile full up -d

docker-up-monitoring:
	docker-compose --profile monitoring up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# DVC Pipeline
pipeline:
	dvc repro

pipeline-force:
	dvc repro --force

dvc-pull:
	dvc pull

dvc-push:
	dvc push

# MLflow
mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Streamlit UI
streamlit-install:
	$(PIP) install -r streamlit_app/requirements.txt

streamlit:
	cd streamlit_app && streamlit run app.py --server.port 8501

docker-up-ui:
	docker-compose up -d api streamlit

docker-up-all:
	docker-compose up -d api streamlit && docker-compose --profile full --profile monitoring up -d

# Data pipeline steps
make-dataset:
	$(PYTHON) src/data/make_dataset.py

build-features:
	$(PYTHON) src/features/build_features.py

train:
	$(PYTHON) src/models/train_model.py

evaluate:
	$(PYTHON) src/models/evaluate_model.py --model-path models/random_forest/Production/model.pkl

# Validation
validate:
	$(PYTHON) src/validation/data_validator.py

# Clean
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/ .mypy_cache/

# Help
help:
	@echo "Available commands:"
	@echo ""
	@echo "  Installation:"
	@echo "    install          - Install production dependencies"
	@echo "    install-dev      - Install development dependencies"
	@echo "    streamlit-install - Install Streamlit UI dependencies"
	@echo ""
	@echo "  Testing & Quality:"
	@echo "    test             - Run tests"
	@echo "    test-cov         - Run tests with coverage report"
	@echo "    lint             - Run linter"
	@echo "    format           - Format code with black and isort"
	@echo ""
	@echo "  Local Development:"
	@echo "    run-api          - Run API server locally (port 8000)"
	@echo "    streamlit        - Run Streamlit UI locally (port 8501)"
	@echo "    mlflow           - Start MLflow UI (port 5000)"
	@echo ""
	@echo "  Docker:"
	@echo "    docker-build     - Build Docker images"
	@echo "    docker-up        - Start API container only"
	@echo "    docker-up-ui     - Start API + Streamlit UI"
	@echo "    docker-up-all    - Start all services (API, UI, MLflow, monitoring)"
	@echo "    docker-down      - Stop all containers"
	@echo "    docker-logs      - View container logs"
	@echo ""
	@echo "  ML Pipeline:"
	@echo "    pipeline         - Run DVC pipeline"
	@echo "    train            - Train model"
	@echo "    evaluate         - Evaluate model"
	@echo "    validate         - Validate data"
	@echo ""
	@echo "  Maintenance:"
	@echo "    clean            - Clean temporary files"
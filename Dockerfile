# Multi-stage build for Weather Temperature Prediction API

# Stage 1: Build stage
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies + DVC
RUN pip install --no-cache-dir --user -r requirements.txt dvc

# Stage 2: Production stage
FROM python:3.10-slim as production

WORKDIR /app

# Install git (required by DVC at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Make sure scripts in .local are usable
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy application code
COPY src/ ./src/
COPY params.yaml ./

# Copy DVC configuration files (needed for runtime pull)
COPY .dvc/ ./.dvc/
COPY dvc.yaml dvc.lock ./

# Create models directory
RUN mkdir -p models/random_forest/Production

# Initialize git (required by DVC)
RUN git init

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the API
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
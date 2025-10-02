# Multi-stage Dockerfile for IoT Predictive Maintenance System
# Optimized for production deployment with minimal image size

# Stage 1: Builder
FROM python:3.10-slim as builder

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

# Set labels
LABEL maintainer="IoT Predictive Maintenance Team"
LABEL description="IoT Predictive Maintenance System with ML-based anomaly detection and forecasting"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    TZ=UTC

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 iotuser && \
    mkdir -p /app /var/lib/iot-system /var/log/iot-system /var/cache/iot-system && \
    chown -R iotuser:iotuser /app /var/lib/iot-system /var/log/iot-system /var/cache/iot-system

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --chown=iotuser:iotuser . /app/

# Set PATH to use virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Create necessary directories
RUN mkdir -p \
    data/raw \
    data/processed \
    data/models \
    logs \
    cache \
    models

# Switch to non-root user
USER iotuser

# Expose dashboard port
EXPOSE 8050

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8050/health || exit 1

# Default command (can be overridden)
CMD ["python", "run_full_dashboard.py"]

# 🚀 CI/CD DEPLOYMENT GUIDE

## IoT Predictive Maintenance System - Complete Deployment Guide

**Last Updated:** 2025-10-02
**Status:** Production Ready ✅
**Version:** 1.0.0

---

## 📋 TABLE OF CONTENTS

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Local Testing](#local-testing)
4. [CI/CD Pipeline](#cicd-pipeline)
5. [Docker Deployment](#docker-deployment)
6. [Kubernetes Deployment](#kubernetes-deployment)
7. [Troubleshooting](#troubleshooting)
8. [Monitoring](#monitoring)

---

## 🎯 QUICK START

### Deploy Locally with Docker Compose (Fastest)

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/IoT-Predictive-Maintenance-System.git
cd IoT-Predictive-Maintenance-System

# 2. Start all services
docker-compose up -d

# 3. Access dashboard
open http://localhost:8050

# 4. Check health
curl http://localhost:8050/health
```

**That's it!** Dashboard with all 12 NASA sensors running in < 2 minutes.

---

## ✅ PREREQUISITES

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Docker | 20.10+ | Container runtime |
| Docker Compose | 2.0+ | Multi-container orchestration |
| Git | 2.30+ | Version control |
| Python | 3.10+ | Local development (optional) |

### Optional (for Kubernetes deployment)

| Software | Version | Purpose |
|----------|---------|---------|
| kubectl | 1.28+ | Kubernetes CLI |
| Helm | 3.12+ | Kubernetes package manager |

### GitHub Repository Secrets (for CI/CD)

Configure these in: **Settings → Secrets and Variables → Actions**

**Required:**
- `GITHUB_TOKEN` - Auto-provided by GitHub

**Optional (for Kubernetes):**
- `KUBE_CONFIG_STAGING` - Base64 kubeconfig for staging
- `KUBE_CONFIG_PRODUCTION` - Base64 kubeconfig for production
- `SLACK_WEBHOOK` - Slack notifications
- `PAGERDUTY_INTEGRATION_KEY` - PagerDuty alerts

---

## 🧪 LOCAL TESTING

### Step 1: Verify Tests Pass

```bash
# Install dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/test_basic.py -v

# Expected: All tests pass ✅
# ============================= 17 passed in XX.XXs ==============================
```

### Step 2: Test Dashboard Locally

```bash
# Install requirements
pip install -r requirements.txt

# Launch dashboard
python launch_complete_dashboard.py

# Access: http://127.0.0.1:8050
# Health check: http://127.0.0.1:8050/health
```

**Expected Output:**
```
IOT PREDICTIVE MAINTENANCE - COMPLETE DASHBOARD (ALL SESSIONS)
================================================================================
  ✓ Overview - System health and architecture
  ✓ Monitoring - Real-time NASA SMAP/MSL sensor data
  ✓ Anomaly Monitor - Real-time anomaly detection
  ...
✓ Loaded 12 NASA sensors
✓ Health check endpoints enabled
✓ Dashboard ready with ALL SESSION 9 features!

🌐 URL: http://127.0.0.1:8050
```

### Step 3: Test Docker Build

```bash
# Build image
docker build -t iot-dashboard:test .

# Check image size (should be < 1GB)
docker images iot-dashboard:test

# Run container
docker run -d -p 8050:8050 --name iot-test iot-dashboard:test

# Test health endpoint
curl http://localhost:8050/health

# Expected response:
# {
#   "status": "healthy",
#   "version": "1.0.0",
#   "timestamp": "2025-10-02T...",
#   "checks": {
#     "nasa_data": true,
#     "services": {"anomaly_service": true, "forecasting_service": true}
#   }
# }

# Check logs
docker logs iot-test

# Stop and remove
docker stop iot-test && docker rm iot-test
```

---

## 🔄 CI/CD PIPELINE

### GitHub Actions Workflow

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Manual workflow dispatch

**Pipeline Stages:**

#### 1. Lint (~ 2 min)
- ✅ Black (code formatting)
- ✅ isort (import sorting)
- ✅ Flake8 (linting)
- ✅ MyPy (type checking)
- ✅ Pylint (code quality)

#### 2. Security (~ 2 min)
- ✅ Safety (dependency vulnerabilities)
- ✅ Bandit (code security analysis)

#### 3. Test (~ 5 min)
- ✅ Unit tests on Python 3.9, 3.10, 3.11
- ✅ Coverage reporting (target: 70%+)
- ✅ Upload to Codecov

#### 4. Integration (~ 3 min)
- ✅ PostgreSQL + TimescaleDB integration
- ✅ Redis integration
- ✅ Integration test suite

#### 5. Build (~ 5 min)
- ✅ Docker image build
- ✅ Multi-stage optimization
- ✅ Layer caching

#### 6. Documentation (~ 1 min)
- ✅ Sphinx documentation build

**Total CI Time:** ~18 minutes

### Monitoring Pipeline Status

```bash
# View workflow runs
https://github.com/YOUR_USERNAME/IoT-Predictive-Maintenance-System/actions

# Check specific run
gh run view [RUN_ID]

# Watch latest run
gh run watch
```

### Manual Trigger

```bash
# Trigger workflow manually
gh workflow run ci.yml

# Or via GitHub UI:
# Actions → CI Pipeline → Run workflow
```

---

## 🐳 DOCKER DEPLOYMENT

### Option A: Single Container (Simple)

```bash
# Pull latest image
docker pull ghcr.io/YOUR_USERNAME/iot-predictive-maintenance-system:latest

# Run dashboard
docker run -d \
  -p 8050:8050 \
  --name iot-dashboard \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -e ENVIRONMENT=production \
  ghcr.io/YOUR_USERNAME/iot-predictive-maintenance-system:latest

# Access dashboard
open http://localhost:8050
```

### Option B: Docker Compose Stack (Recommended)

**Full stack includes:**
- Dashboard (Python/Dash)
- PostgreSQL with TimescaleDB
- Redis cache
- Kafka + Zookeeper
- MLflow tracking server
- Prometheus monitoring
- Grafana dashboards
- Nginx reverse proxy

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f dashboard

# Check status
docker-compose ps

# Scale dashboard (3 replicas)
docker-compose up -d --scale dashboard=3

# Access services:
# Dashboard:    http://localhost:8050
# MLflow:       http://localhost:5000
# Grafana:      http://localhost:3000 (admin/admin)
# Prometheus:   http://localhost:9090

# Stop all services
docker-compose down
```

### Docker Compose Production Setup

```bash
# Set environment variables
export DB_PASSWORD="your_secure_password"
export REDIS_PASSWORD="your_redis_password"

# Start with production config
docker-compose -f docker-compose.yml up -d

# Monitor resources
docker stats

# View all containers
docker ps
```

### Health Checks

```bash
# Dashboard health
curl http://localhost:8050/health

# Readiness (Kubernetes)
curl http://localhost:8050/health/ready

# Liveness (Kubernetes)
curl http://localhost:8050/health/live

# Prometheus metrics
curl http://localhost:8050/metrics
```

---

## ☸️ KUBERNETES DEPLOYMENT

### Prerequisites

```bash
# Verify kubectl access
kubectl cluster-info

# Create namespace
kubectl create namespace iot-system
```

### Deploy All Resources

```bash
# Apply all manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/persistent-volumes.yaml
kubectl apply -f k8s/postgres-deployment.yaml
kubectl apply -f k8s/redis-deployment.yaml
kubectl apply -f k8s/mlflow-deployment.yaml
kubectl apply -f k8s/dashboard-deployment.yaml
kubectl apply -f k8s/ingress.yaml

# Or apply all at once
kubectl apply -f k8s/
```

### Verify Deployment

```bash
# Check all resources
kubectl get all -n iot-system

# Check pods
kubectl get pods -n iot-system

# Check services
kubectl get svc -n iot-system

# Check persistent volumes
kubectl get pvc -n iot-system

# Check deployments
kubectl get deployments -n iot-system
```

### Access Dashboard

**Option 1: Port Forward (Development)**
```bash
kubectl port-forward svc/dashboard-service 8050:80 -n iot-system
open http://localhost:8050
```

**Option 2: Ingress (Production)**
```bash
# Configure DNS to point to your ingress IP
# Access via: https://iot-dashboard.example.com
```

### Scaling

```bash
# Manual scaling
kubectl scale deployment/iot-dashboard --replicas=5 -n iot-system

# Horizontal Pod Autoscaler (already configured)
kubectl get hpa -n iot-system

# Auto-scales between 3-10 replicas based on:
# - CPU usage: 70%
# - Memory usage: 80%
```

### Monitoring

```bash
# View logs
kubectl logs -f deployment/iot-dashboard -n iot-system

# Exec into pod
kubectl exec -it deployment/iot-dashboard -n iot-system -- /bin/bash

# Check metrics
kubectl top pods -n iot-system
kubectl top nodes
```

### Rolling Updates

```bash
# Update image
kubectl set image deployment/iot-dashboard \
  dashboard=ghcr.io/YOUR_USERNAME/iot-predictive-maintenance-system:v2.0.0 \
  -n iot-system

# Check rollout status
kubectl rollout status deployment/iot-dashboard -n iot-system

# Rollback if needed
kubectl rollout undo deployment/iot-dashboard -n iot-system

# View rollout history
kubectl rollout history deployment/iot-dashboard -n iot-system
```

---

## 🔧 TROUBLESHOOTING

### Issue: CI Pipeline Fails on Lint

**Symptom:** Black or flake8 errors

**Solution:**
```bash
# Format code locally
black src/ tests/
isort src/ tests/

# Check before committing
black --check src/
flake8 src/ --max-line-length=120
```

### Issue: Tests Fail in CI

**Symptom:** Unit tests pass locally but fail in CI

**Solution:**
```bash
# Run tests exactly as CI does
pytest tests/ \
  --cov=src \
  --cov-report=xml \
  --cov-report=term \
  -n auto \
  --maxfail=5

# Check test isolation
pytest tests/test_basic.py --verbose
```

### Issue: Docker Build Fails

**Symptom:** Dockerfile build errors

**Solution:**
```bash
# Build with no cache
docker build --no-cache -t iot-dashboard:debug .

# Check build logs
docker build -t iot-dashboard:debug . 2>&1 | tee build.log

# Verify requirements.txt
pip install -r requirements.txt
```

### Issue: Health Check Fails

**Symptom:** Container starts but health check fails

**Solution:**
```bash
# Check if app is running
docker exec iot-dashboard ps aux | grep python

# Check logs
docker logs iot-dashboard

# Test health endpoint manually
docker exec iot-dashboard curl -f http://localhost:8050/health

# Verify port binding
docker port iot-dashboard
```

### Issue: Dashboard Shows Empty Charts

**Symptom:** Dashboard loads but no data visible

**Solution:**
1. Check NASA data loaded:
```python
from src.infrastructure.data.nasa_data_loader import NASADataLoader
loader = NASADataLoader()
print(f"Data loaded: {loader.is_loaded}")
```

2. Verify data files exist:
```bash
ls -la data/raw/smap/
ls -la data/raw/msl/
```

3. Check logs for fallback warnings:
```bash
docker logs iot-dashboard | grep -i "fallback"
```

### Issue: Out of Memory

**Symptom:** Container crashes with OOM error

**Solution:**
```bash
# Increase container memory
docker run -d -p 8050:8050 -m 4g iot-dashboard

# Or in docker-compose.yml:
services:
  dashboard:
    mem_limit: 4g
    mem_reservation: 2g
```

---

## 📊 MONITORING

### Prometheus Metrics

Dashboard exposes metrics at `/metrics`:

```bash
curl http://localhost:8050/metrics
```

**Available Metrics:**
- `iot_dashboard_health` - Overall health status (1=healthy)
- `iot_nasa_data_available` - Data availability (1=available)
- `iot_service_available{service="anomaly_detection"}` - Service status
- `iot_service_available{service="forecasting"}` - Service status

### Grafana Dashboards

1. Access Grafana: http://localhost:3000
2. Login: `admin` / `admin`
3. Add Prometheus datasource: http://prometheus:9090
4. Import dashboards from `k8s/grafana-dashboards/`

### Application Logs

```bash
# Docker Compose
docker-compose logs -f dashboard

# Kubernetes
kubectl logs -f deployment/iot-dashboard -n iot-system

# Container
docker logs -f iot-dashboard
```

### Resource Usage

```bash
# Docker stats
docker stats iot-dashboard

# Kubernetes
kubectl top pods -n iot-system
kubectl top nodes
```

---

## 🎉 SUCCESS CRITERIA

### ✅ CI/CD Pipeline

- All lint checks pass
- All security scans pass (no critical vulnerabilities)
- All tests pass on Python 3.9, 3.10, 3.11
- Test coverage > 70%
- Docker image builds successfully
- Image size < 1GB

### ✅ Deployment

- Container starts within 60 seconds
- Health check returns 200 OK
- Dashboard accessible at port 8050
- All 10 tabs load without errors
- NASA data loaded (12 sensors)
- Charts display data (not empty)

### ✅ Production Readiness

- Auto-scaling configured (3-10 replicas)
- Health checks pass
- Monitoring enabled (Prometheus + Grafana)
- Logs aggregated
- Backups configured
- SSL/TLS enabled (Ingress)

---

## 📚 ADDITIONAL RESOURCES

### Documentation
- [Complete Architecture](FINAL_STATUS_REPORT.md)
- [Session Documentation](SESSION_9_UI_FINAL_INTEGRATION_COMPLETE.md)
- [User Guide](USER_GUIDE.md)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)

### GitHub Actions
- [CI Workflow](.github/workflows/ci.yml)
- [CD Workflow](.github/workflows/cd.yml)

### Docker
- [Dockerfile](Dockerfile)
- [Docker Compose](docker-compose.yml)

### Kubernetes
- [All Manifests](k8s/)

---

## 🤝 SUPPORT

### Issues
- GitHub Issues: https://github.com/YOUR_USERNAME/IoT-Predictive-Maintenance-System/issues
- CI/CD Failures: Check GitHub Actions logs

### Quick Commands Reference

```bash
# Local development
python launch_complete_dashboard.py

# Docker
docker-compose up -d
docker-compose logs -f dashboard

# Kubernetes
kubectl get all -n iot-system
kubectl logs -f deployment/iot-dashboard -n iot-system

# Health checks
curl http://localhost:8050/health

# Tests
pytest tests/ -v
```

---

**Last Updated:** 2025-10-02
**Version:** 1.0.0
**Status:** ✅ Production Ready

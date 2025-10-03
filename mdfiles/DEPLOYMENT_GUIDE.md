# IoT Predictive Maintenance System - Deployment Guide

Complete guide for deploying the IoT Predictive Maintenance System across different environments.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Configuration](#configuration)
3. [Local Development](#local-development)
4. [Docker Deployment](#docker-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [CI/CD Pipeline](#cicd-pipeline)
7. [Monitoring & Observability](#monitoring--observability)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

- **Python**: 3.9, 3.10, or 3.11
- **Docker**: 20.10+ (for containerization)
- **Docker Compose**: 2.0+ (for local multi-container deployment)
- **Kubernetes**: 1.24+ (for production deployment)
- **kubectl**: 1.24+ (Kubernetes CLI)
- **Git**: 2.30+

### Optional Tools

- **Helm**: 3.0+ (Kubernetes package manager)
- **k9s**: Terminal UI for Kubernetes
- **Lens**: Kubernetes IDE
- **Terraform**: Infrastructure as Code (IaC)

---

## Configuration

### Environment-Specific Configurations

The system supports three environments:

1. **Development** (`config.development.yaml`)
   - SQLite database
   - Debug mode enabled
   - Fast refresh, minimal caching
   - Mock data support

2. **Staging** (`config.staging.yaml`)
   - PostgreSQL database
   - Mirrors production
   - Reduced resources
   - Testing environment

3. **Production** (`config.production.yaml`)
   - PostgreSQL with TimescaleDB
   - Redis caching enabled
   - Kafka streaming enabled
   - Full monitoring

### Configuration Manager

Use the centralized configuration manager:

```python
from config.config_manager import load_config, get_config

# Load configuration
config = load_config('config/config.yaml', env='production')

# Access values
db_uri = config.get('data_ingestion.database.postgresql.uri')
mlflow_uri = config.get('mlflow.tracking_uri')

# Get sections
dashboard_config = config.get_section('dashboard')
```

### Environment Variables

Override configuration with environment variables:

```bash
export ENVIRONMENT=production
export DATABASE_URI=postgresql://user:pass@host:5432/db
export MLFLOW_TRACKING_URI=http://mlflow:5000
export REDIS_HOST=redis-service
export KAFKA_BOOTSTRAP_SERVERS=kafka:9092
export LOG_LEVEL=INFO
```

---

## Local Development

### 1. Clone Repository

```bash
git clone https://github.com/your-org/IoT-Predictive-Maintenance-System.git
cd IoT-Predictive-Maintenance-System
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Development Environment

```bash
export ENVIRONMENT=development
```

### 4. Run Dashboard

```bash
python run_full_dashboard.py
```

Access at: http://localhost:8050

### 5. Development with Docker Compose

```bash
docker-compose -f docker-compose.dev.yml up
```

Features:
- Hot reload on code changes
- Debugger support (port 5678)
- Local SQLite database
- Optional MLflow, PostgreSQL, Redis

---

## Docker Deployment

### Build Docker Image

```bash
# Production image
docker build -t iot-predictive-maintenance:latest .

# Development image
docker build -f Dockerfile.dev -t iot-predictive-maintenance:dev .
```

### Run with Docker Compose

```bash
# Full production stack
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f dashboard

# Stop all services
docker-compose down
```

### Services Included

- **Dashboard**: Port 8050
- **PostgreSQL (TimescaleDB)**: Port 5432
- **Redis**: Port 6379
- **Kafka + Zookeeper**: Port 9092
- **MLflow**: Port 5000
- **Prometheus**: Port 9090
- **Grafana**: Port 3000
- **Nginx**: Ports 80, 443

### Environment Variables

Create `.env` file:

```env
# Database
DB_PASSWORD=secure_password_here

# Redis
REDIS_PASSWORD=redis_secure_password

# Grafana
GRAFANA_PASSWORD=grafana_password

# Application
ENVIRONMENT=production
LOG_LEVEL=INFO
```

### Data Persistence

Volumes:
- `iot-postgres-data`: Database storage
- `iot-redis-data`: Redis persistence
- `iot-kafka-data`: Kafka logs
- `iot-mlflow-artifacts`: Model artifacts
- `iot-prometheus-data`: Metrics
- `iot-grafana-data`: Dashboards

---

## Kubernetes Deployment

### 1. Prerequisites

- Kubernetes cluster (EKS, GKE, AKS, or local with Minikube/Kind)
- kubectl configured
- Container registry access

### 2. Build and Push Image

```bash
# Build
docker build -t ghcr.io/your-org/iot-predictive-maintenance:v1.0.0 .

# Login to registry
echo $CR_PAT | docker login ghcr.io -u USERNAME --password-stdin

# Push
docker push ghcr.io/your-org/iot-predictive-maintenance:v1.0.0
```

### 3. Update Configuration

Edit `k8s/secrets.yaml` with your credentials:

```bash
# Encode secrets
echo -n "your_password" | base64
```

### 4. Deploy to Kubernetes

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Apply configurations
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml

# Create persistent volumes
kubectl apply -f k8s/persistent-volumes.yaml

# Deploy services
kubectl apply -f k8s/postgres-deployment.yaml
kubectl apply -f k8s/redis-deployment.yaml
kubectl apply -f k8s/mlflow-deployment.yaml
kubectl apply -f k8s/dashboard-deployment.yaml

# Setup ingress (optional)
kubectl apply -f k8s/ingress.yaml
```

### 5. Verify Deployment

```bash
# Check pods
kubectl get pods -n iot-system

# Check services
kubectl get svc -n iot-system

# View logs
kubectl logs -f deployment/iot-dashboard -n iot-system

# Get dashboard URL
kubectl get ingress -n iot-system
```

### 6. Scaling

```bash
# Manual scaling
kubectl scale deployment/iot-dashboard --replicas=5 -n iot-system

# Auto-scaling (HPA already configured)
kubectl get hpa -n iot-system
```

### 7. Updates and Rollbacks

```bash
# Update image
kubectl set image deployment/iot-dashboard \
  dashboard=ghcr.io/your-org/iot-predictive-maintenance:v1.1.0 \
  -n iot-system

# Check rollout status
kubectl rollout status deployment/iot-dashboard -n iot-system

# Rollback if needed
kubectl rollout undo deployment/iot-dashboard -n iot-system

# View rollout history
kubectl rollout history deployment/iot-dashboard -n iot-system
```

---

## CI/CD Pipeline

### GitHub Actions

Workflows in `.github/workflows/`:

1. **CI Pipeline** (`ci.yml`)
   - Runs on: Push to main/develop, Pull Requests
   - Jobs:
     - Lint (Black, isort, Flake8, MyPy)
     - Security (Safety, Bandit)
     - Unit Tests (Python 3.9, 3.10, 3.11)
     - Integration Tests
     - Docker Build
     - Documentation

2. **CD Pipeline** (`cd.yml`)
   - Runs on: Push to main, Tags
   - Jobs:
     - Build and push Docker image
     - Deploy to Staging (automatic)
     - Deploy to Production (manual for tags)
     - Database migrations
     - Performance testing

### GitLab CI/CD

Pipeline in `.gitlab-ci.yml`:

**Stages**:
1. Lint
2. Test
3. Security (SAST, dependency scan, container scan)
4. Build
5. Deploy to Staging
6. Deploy to Production

### Required Secrets

#### GitHub Secrets
- `KUBE_CONFIG_STAGING`: Kubeconfig for staging (base64)
- `KUBE_CONFIG_PRODUCTION`: Kubeconfig for production (base64)
- `SLACK_WEBHOOK`: Slack notifications
- `PAGERDUTY_INTEGRATION_KEY`: PagerDuty alerts

#### GitLab Variables
- `KUBE_CONFIG_STAGING`: Kubeconfig for staging
- `KUBE_CONFIG_PRODUCTION`: Kubeconfig for production
- `CI_REGISTRY_USER`: Container registry username
- `CI_REGISTRY_PASSWORD`: Container registry password

### Triggering Deployments

#### Staging Deployment
```bash
# GitHub: Automatic on push to main
git push origin main

# GitLab: Automatic on push to main
git push origin main
```

#### Production Deployment
```bash
# Create and push tag
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# GitHub: Manual approval required
# GitLab: Manual trigger required
```

---

## Monitoring & Observability

### Prometheus Metrics

Access at: http://localhost:9090 (Docker) or via Ingress (K8s)

**Key Metrics**:
- `iot_anomaly_detection_total`: Total anomalies detected
- `iot_forecast_requests_total`: Forecast requests
- `iot_model_inference_duration`: Model inference time
- `iot_http_requests_total`: HTTP requests to dashboard

### Grafana Dashboards

Access at: http://localhost:3000 (Docker) or via Ingress (K8s)

**Default Credentials**:
- Username: `admin`
- Password: Set in `.env` or `GRAFANA_PASSWORD`

**Pre-configured Dashboards**:
1. System Overview
2. Anomaly Detection Metrics
3. Forecasting Performance
4. Infrastructure Metrics

### Logs

#### Docker
```bash
# View all logs
docker-compose logs -f

# Specific service
docker-compose logs -f dashboard

# Last 100 lines
docker-compose logs --tail=100 dashboard
```

#### Kubernetes
```bash
# Stream logs
kubectl logs -f deployment/iot-dashboard -n iot-system

# All pods
kubectl logs -f -l app=iot-predictive-maintenance -n iot-system

# Previous pod (after crash)
kubectl logs --previous deployment/iot-dashboard -n iot-system
```

### Health Checks

```bash
# Dashboard health
curl http://localhost:8050/health

# MLflow health
curl http://localhost:5000/health
```

---

## Troubleshooting

### Common Issues

#### 1. Dashboard Won't Start

**Symptom**: Dashboard fails to start or crashes immediately

**Solutions**:
```bash
# Check logs
docker-compose logs dashboard

# Verify database connection
docker-compose exec dashboard python -c "from config.config_manager import load_config; print(load_config())"

# Reset database
docker-compose down -v
docker-compose up -d postgres
# Wait 30 seconds
docker-compose up -d dashboard
```

#### 2. Database Connection Issues

**Symptom**: `psycopg2.OperationalError` or connection refused

**Solutions**:
```bash
# Verify PostgreSQL is running
docker-compose ps postgres
kubectl get pods -n iot-system | grep postgres

# Check credentials
echo $DATABASE_URI

# Test connection
docker-compose exec postgres psql -U iot_user -d iot_telemetry -c "SELECT 1;"
```

#### 3. MLflow Not Accessible

**Symptom**: MLflow UI not loading or models not tracking

**Solutions**:
```bash
# Check MLflow logs
docker-compose logs mlflow

# Verify backend store
docker-compose exec mlflow env | grep BACKEND

# Restart MLflow
docker-compose restart mlflow
```

#### 4. Out of Memory (OOM)

**Symptom**: Pods/containers killed, high memory usage

**Solutions**:
```bash
# Check memory usage
docker stats
kubectl top pods -n iot-system

# Reduce batch sizes in config
# Edit config.production.yaml:
# performance.batch.max_batch_size: 1000 -> 500

# Increase resources in K8s
kubectl edit deployment/iot-dashboard -n iot-system
# Update resources.limits.memory
```

#### 5. Slow Dashboard Performance

**Symptom**: Dashboard loads slowly or times out

**Solutions**:
```bash
# Enable caching
# Edit config:
# dashboard.performance.cache.enabled: true

# Reduce data points
# dashboard.performance.limits.max_data_points: 5000

# Scale horizontally
kubectl scale deployment/iot-dashboard --replicas=5 -n iot-system
```

### Debug Mode

Enable debug mode for detailed logs:

```bash
# Environment variable
export LOG_LEVEL=DEBUG
export ENVIRONMENT=development

# Or in config
system:
  debug: true
  log_level: "DEBUG"
```

### Support

For additional support:
- GitHub Issues: https://github.com/your-org/IoT-Predictive-Maintenance-System/issues
- Documentation: https://docs.example.com
- Email: support@example.com

---

## Security Best Practices

1. **Secrets Management**
   - Use external secret managers (Vault, AWS Secrets Manager)
   - Never commit secrets to Git
   - Rotate credentials regularly

2. **Network Security**
   - Use TLS/SSL for all external connections
   - Configure network policies in Kubernetes
   - Use private container registries

3. **Access Control**
   - Implement RBAC in Kubernetes
   - Use service accounts with minimal permissions
   - Enable authentication on dashboard

4. **Monitoring**
   - Set up alerts for security events
   - Monitor unusual access patterns
   - Regular security audits

---

## Maintenance

### Backup Strategy

#### Database Backups
```bash
# Automated daily backups
docker-compose exec postgres pg_dump -U iot_user iot_telemetry > backup_$(date +%Y%m%d).sql

# Kubernetes CronJob for backups (create separate manifest)
```

#### Model Backups
```bash
# Backup MLflow artifacts
tar -czf mlflow_artifacts_$(date +%Y%m%d).tar.gz /mlflow/artifacts/
```

### Updates

```bash
# Update dependencies
pip install --upgrade -r requirements.txt

# Rebuild Docker image
docker build -t iot-predictive-maintenance:latest .

# Update Kubernetes deployment
kubectl set image deployment/iot-dashboard dashboard=iot-predictive-maintenance:latest -n iot-system
```

---

**Deployment Guide Version**: 1.0.0
**Last Updated**: 2025-10-02

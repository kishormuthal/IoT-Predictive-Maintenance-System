# 🚀 Production Deployment Guide

## IoT Predictive Maintenance System - Complete Deployment Instructions

This guide provides step-by-step instructions to deploy the IoT Predictive Maintenance System on a high-performance company server without Claude Code assistance.

---

## 📋 Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [System Requirements](#system-requirements)
3. [Quick Start (5 Minutes)](#quick-start-5-minutes)
4. [Detailed Installation](#detailed-installation)
5. [Running the Dashboard](#running-the-dashboard)
6. [Validation & Testing](#validation--testing)
7. [Troubleshooting](#troubleshooting)
8. [Production Configuration](#production-configuration)

---

## Pre-Deployment Checklist

Before starting deployment, ensure you have:

- [ ] Python 3.8+ installed
- [ ] 4GB+ RAM available
- [ ] 2GB+ disk space available
- [ ] Network access to install packages
- [ ] Terminal/Command Prompt access
- [ ] This complete codebase copied to the server

---

## System Requirements

### Minimum Requirements
- **OS**: Windows 10+, Ubuntu 18.04+, macOS 10.14+
- **Python**: 3.8 or higher (3.11 recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **Disk**: 2GB for application + data
- **CPU**: 2+ cores recommended

### Required Software
- Python 3.8+
- pip (Python package manager)
- Git (optional, for version control)

---

## Quick Start (5 Minutes)

### Option 1: One-Command Setup (Recommended)

```bash
# 1. Navigate to project directory
cd "IOT Predictive Maintenece System"

# 2. Run pre-flight check (creates directories, checks dependencies)
python preflight_check.py

# 3. Install dependencies (if not already installed)
pip install -r requirements.txt

# 4. Start the dashboard
python start_dashboard.py
```

### Option 2: Step-by-Step

```bash
# Step 1: Check Python version (must be 3.8+)
python --version

# Step 2: Create virtual environment (recommended)
python -m venv venv

# Step 3: Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Step 4: Install dependencies
pip install -r requirements.txt

# Step 5: Run validation
python validate_startup.py

# Step 6: Start dashboard
python start_dashboard.py
```

The dashboard will be available at: **http://127.0.0.1:8050**

---

## Detailed Installation

### Step 1: Environment Setup

#### Create Virtual Environment (Recommended)

```bash
# Navigate to project root
cd "IOT Predictive Maintenece System"

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

#### Verify Python Version

```bash
python --version
# Should show: Python 3.8.x or higher
```

### Step 2: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# This installs:
# - dash (web framework)
# - plotly (visualizations)
# - pandas, numpy (data processing)
# - tensorflow (ML models)
# - scikit-learn (ML utilities)
# - and more...
```

### Step 3: Verify Installation

```bash
# Run comprehensive validation
python validate_startup.py

# Expected output:
# ✓ Python version: 3.x
# ✓ Dependencies: All packages installed
# ✓ Directory structure: All directories present
# ✓ Configuration files: All present
# ✓ Dashboard files: All present
# ✓ NASA data files: All present
```

### Step 4: Optional - Train Models (If Not Present)

If the validation script warns about missing models:

```bash
# Quick training (10-15 minutes, for testing)
python train_forecasting_models.py --quick

# Full training (1-2 hours, for production)
python train_forecasting_models.py

# Train both anomaly detection and forecasting
python setup_models.py
```

---

## Running the Dashboard

### Development Mode (With Debug Info)

```bash
python start_dashboard.py

# Output:
# ========================================================
# IoT PREDICTIVE MAINTENANCE DASHBOARD
# Clean Launch with All Trained Models
# ========================================================
# [STATUS] All 12 forecasting models trained and ready
# [STATUS] NASA SMAP/MSL data loaded for 12 sensors
# [STATUS] Anti-hanging architecture enabled
# --------------------------------------------------------
# [INFO] Creating dashboard application...
# [URL] Dashboard starting at: http://127.0.0.1:8050
# [FEATURES] Overview | Monitoring | Anomalies | Forecasting | Maintenance | Work Orders | Performance
# [CONTROL] Press Ctrl+C to stop the server
```

### Production Mode (Optimized)

For production deployment with Gunicorn (Linux/Mac):

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn app:server --bind 0.0.0.0:8050 --workers 4 --timeout 120
```

For Windows production (using waitress):

```bash
# Install waitress
pip install waitress

# Create production launcher script
# See PRODUCTION_CONFIG.md for details
```

### Custom Port

To run on a different port:

```bash
# Edit start_dashboard.py line 38:
# Change: port=8050
# To: port=YOUR_PORT

# Or set environment variable:
export DASHBOARD_PORT=8080
python start_dashboard.py
```

---

## Validation & Testing

### Pre-Flight Check (Quick)

```bash
python preflight_check.py

# Checks and auto-fixes:
# ✓ Creates missing directories
# ✓ Validates dependencies
# ✓ Checks data files
# ✓ Verifies dashboard files
# ✓ Checks trained models
```

### Full Validation (Comprehensive)

```bash
python validate_startup.py

# Comprehensive checks:
# ✓ Python version
# ✓ All dependencies with versions
# ✓ Directory structure (14+ directories)
# ✓ Configuration files
# ✓ Dashboard files
# ✓ NASA data files (SMAP/MSL)
# ✓ Trained model files
# ✓ Core module imports
# ✓ Port availability
# ✓ Disk space
```

### Deployment Verification

```bash
python verify_deployment.py

# Tests:
# ✓ Dashboard loads successfully
# ✓ All tabs accessible
# ✓ Services initialize
# ✓ Data can be loaded
# ✓ Models can run inference
```

### Manual Testing Checklist

After starting the dashboard, verify:

1. [ ] Dashboard loads at http://127.0.0.1:8050
2. [ ] **Overview tab** shows system metrics
3. [ ] **Monitoring tab** displays sensor data
4. [ ] **Anomalies tab** shows anomaly detection
5. [ ] **Forecasting tab** displays predictions
6. [ ] **Maintenance tab** shows equipment status
7. [ ] **Work Orders tab** displays tasks
8. [ ] **Performance tab** shows system health
9. [ ] No error messages in console
10. [ ] All graphs render correctly

---

## Troubleshooting

### Common Issues and Fixes

#### Issue 1: Dashboard Won't Start

**Error:** `ImportError: No module named 'dash'`

**Fix:**
```bash
# Ensure virtual environment is activated
# Install dependencies again
pip install -r requirements.txt
```

#### Issue 2: Port Already in Use

**Error:** `OSError: [Errno 48] Address already in use`

**Fix:**
```bash
# Find process using port 8050
# On Windows:
netstat -ano | findstr :8050
taskkill /PID <PID> /F

# On Linux/Mac:
lsof -ti:8050 | xargs kill -9

# Or use different port (edit start_dashboard.py)
```

#### Issue 3: Missing Dependencies

**Error:** `ModuleNotFoundError: No module named 'X'`

**Fix:**
```bash
# Install missing package
pip install <package-name>

# Or reinstall all
pip install -r requirements.txt --force-reinstall
```

#### Issue 4: TensorFlow Errors

**Error:** `Cannot import TensorFlow`

**Fix:**
```bash
# Try installing specific TensorFlow version
pip install tensorflow==2.15.0

# Or use CPU version
pip install tensorflow-cpu==2.15.0
```

#### Issue 5: Data Files Missing

**Error:** `NASA data files missing`

**Fix:**
- Data files should be in `data/raw/smap/` and `data/raw/msl/`
- Dashboard will use mock data if files missing
- Copy data files from original system if available

#### Issue 6: Models Not Found

**Warning:** `No models found`

**Fix:**
```bash
# Train models quickly (10-15 min)
python train_forecasting_models.py --quick

# Dashboard works without models, but forecasting won't
```

#### Issue 7: Memory Errors

**Error:** `MemoryError` or system freezes

**Fix:**
- Close other applications
- Use `--quick` flag for model training
- Reduce batch size in `config/config.yaml`

#### Issue 8: Import Errors

**Error:** `ImportError: cannot import name 'EnhancedIoTDashboard'`

**Fix:**
```bash
# This should not happen with fixed codebase
# Verify these files exist:
# - src/presentation/dashboard/enhanced_app.py
# - src/presentation/dashboard/enhanced_app_optimized.py
# - src/presentation/dashboard/enhanced_callbacks_simplified.py

# Run validation to check:
python validate_startup.py
```

---

## Production Configuration

### For High-Performance Servers

#### 1. Optimize for Production

Edit `config/config.yaml`:

```yaml
environment: production

system:
  debug: false
  log_level: "WARNING"  # Reduce logging

dashboard:
  server:
    debug: false
    threaded: true
```

#### 2. Use Production Server

Install gunicorn (Linux/Mac):

```bash
pip install gunicorn
gunicorn app:server --bind 0.0.0.0:8050 --workers 4 --timeout 120
```

Install waitress (Windows):

```bash
pip install waitress
waitress-serve --port=8050 app:server
```

#### 3. Enable Monitoring

```yaml
monitoring:
  enabled: true
  track_latency: true
  track_throughput: true
```

#### 4. Configure Logging

```yaml
logging:
  file:
    enabled: true
    path: "./logs/production.log"
    max_size: 10485760  # 10 MB
    backup_count: 10
```

### Security Recommendations

1. **Change default ports** - Don't use 8050 in production
2. **Enable authentication** - Add login system
3. **Use HTTPS** - Configure SSL/TLS
4. **Restrict access** - Use firewall rules
5. **Regular backups** - Backup models and data

---

## System Architecture

### Directory Structure

```
IOT Predictive Maintenece System/
├── config/                      # Configuration files
│   ├── config.yaml             # Main configuration ✅
│   ├── equipment_config.py     # Equipment definitions ✅
│   └── settings.py             # Settings module ✅
├── data/                        # Data directory
│   ├── raw/                    # Raw NASA data
│   │   ├── smap/              # SMAP satellite data ✅
│   │   └── msl/               # MSL rover data ✅
│   ├── models/                # Trained models
│   │   ├── transformer/       # Forecasting models
│   │   └── registry/          # Model registry
│   └── processed/             # Processed data
├── src/                        # Source code
│   ├── core/                  # Core domain logic ✅
│   ├── application/           # Use cases ✅
│   ├── infrastructure/        # Data access ✅
│   └── presentation/          # Dashboard UI ✅
│       └── dashboard/
│           ├── enhanced_app.py                      # ✅ NEW
│           ├── enhanced_app_optimized.py            # ✅ FIXED
│           ├── enhanced_callbacks_simplified.py     # ✅ NEW
│           ├── app.py                               # ✅
│           ├── layouts/                             # ✅
│           └── components/                          # ✅
├── logs/                       # Application logs
├── requirements.txt            # Dependencies ✅
├── start_dashboard.py          # Dashboard launcher ✅
├── preflight_check.py          # Pre-flight checks ✅ NEW
├── validate_startup.py         # Full validation ✅ NEW
├── verify_deployment.py        # Deployment test ✅ NEW
└── PRODUCTION_DEPLOYMENT.md    # This file ✅ NEW
```

### Key Features

✅ **Clean Architecture** - Proper separation of concerns
✅ **NASA Data Integration** - Real SMAP/MSL telemetry
✅ **Transformer Forecasting** - 219K parameter models
✅ **Anomaly Detection** - NASA Telemanom algorithm
✅ **Model Registry** - Version control & management
✅ **Real-time Monitoring** - Performance tracking
✅ **Alert System** - Intelligent notifications
✅ **Anti-Hang Protection** - Graceful degradation
✅ **Production Ready** - Comprehensive error handling

---

## Quick Reference Commands

```bash
# Check everything
python validate_startup.py

# Quick check + auto-fix
python preflight_check.py

# Start dashboard
python start_dashboard.py

# Train models (quick)
python train_forecasting_models.py --quick

# Train models (full)
python train_forecasting_models.py

# Verify deployment
python verify_deployment.py

# Check Python version
python --version

# Check installed packages
pip list

# Check disk space
df -h  # Linux/Mac
dir    # Windows
```

---

## Support & Contact

For issues or questions:

1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Review logs in `logs/` directory
3. Run `python validate_startup.py` for diagnostics
4. Check system requirements

---

## Summary

### Minimum Steps to Deploy:

1. ✅ Copy codebase to server
2. ✅ Install Python 3.8+
3. ✅ Run `pip install -r requirements.txt`
4. ✅ Run `python preflight_check.py`
5. ✅ Run `python start_dashboard.py`
6. ✅ Access http://127.0.0.1:8050

### Expected Result:

- **Dashboard loads in browser**
- **All 7 tabs accessible**
- **Real-time data visualization**
- **No errors in console**

**Total Setup Time: 5-10 minutes** (excluding model training)

**Model Training Time:**
- Quick mode: 10-15 minutes
- Full mode: 1-2 hours
- Optional: Dashboard works without models

---

## ✅ Production-Ready Checklist

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Validation passed (`python validate_startup.py`)
- [ ] Dashboard starts (`python start_dashboard.py`)
- [ ] Accessible at http://127.0.0.1:8050
- [ ] All tabs load successfully
- [ ] No errors in console
- [ ] Models available (optional, for forecasting)
- [ ] Logs directory writable
- [ ] Sufficient disk space (2GB+)

---

**🎉 Your IoT Predictive Maintenance System is now ready for production!**
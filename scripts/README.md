# Scripts Directory

This directory contains training, testing, and validation scripts for the IoT Predictive Maintenance System.

## 📋 Quick Reference

### Training Scripts

#### `train_forecasting_models.py`
**Purpose:** Train Transformer-based forecasting models for all 12 sensors

**Usage:**
```bash
# Quick training (10-15 minutes)
python scripts/train_forecasting_models.py --quick

# Full training (1-2 hours)
python scripts/train_forecasting_models.py

# Train specific sensors
python scripts/train_forecasting_models.py --sensors SMAP-PWR-001 MSL-MOB-001
```

**What it does:**
- Trains Transformer models (219K parameters each)
- Saves models to `data/models/transformer/`
- Registers models in model registry
- Generates training summary JSON

---

#### `train_anomaly_models.py`
**Purpose:** Train NASA Telemanom anomaly detection models

**Usage:**
```bash
# Quick training
python scripts/train_anomaly_models.py --quick

# Full training
python scripts/train_anomaly_models.py

# Train specific sensors
python scripts/train_anomaly_models.py --sensors SMAP-PWR-001
```

**What it does:**
- Trains Telemanom LSTM models
- Saves models to `data/models/nasa_equipment_models/`
- Creates training summaries

---

#### `setup_models.py`
**Purpose:** One-command setup for both anomaly detection and forecasting models

**Usage:**
```bash
# Quick setup (both models, fast)
python scripts/setup_models.py --quick

# Full setup
python scripts/setup_models.py
```

**What it does:**
- Trains both anomaly and forecasting models
- Sequential execution
- Complete model setup

---

### Validation Scripts

#### `validate_startup.py`
**Purpose:** Comprehensive validation of entire system

**Usage:**
```bash
python scripts/validate_startup.py
```

**What it checks:**
- Python version (3.8+ required)
- All dependencies installed
- Directory structure complete
- Configuration files present
- Dashboard files present
- NASA data files available
- Trained models available
- Core imports working
- Port availability (8050)
- Disk space

**Output:** Detailed validation report with pass/fail/warning for each check

---

#### `verify_deployment.py`
**Purpose:** Test deployment after installation

**Usage:**
```bash
python scripts/verify_deployment.py
```

**What it tests:**
- Critical imports
- Dashboard creation
- Configuration loading
- Data access
- Model availability
- Startup scripts

**Output:** Deployment verification report

---

#### `preflight_check.py`
**Purpose:** Quick validation with auto-fix

**Usage:**
```bash
python scripts/preflight_check.py
```

**What it does:**
- Creates missing directories automatically
- Checks dependencies
- Verifies data files
- Checks dashboard files
- Validates models
- Fast execution (<30 seconds)

**Note:** Has Windows encoding issues with Unicode characters. Use `../quick_start.py` instead for Windows.

---

### Testing Scripts

#### `test_training_pipeline.py`
**Purpose:** Test the training pipeline

**Usage:**
```bash
python scripts/test_training_pipeline.py
```

**What it tests:**
- Training workflow
- Model creation
- Model saving
- Registry integration

---

#### `test_optimized_server.py`
**Purpose:** Test optimized dashboard server

**Usage:**
```bash
python scripts/test_optimized_server.py
```

**What it tests:**
- Dashboard initialization
- Server startup
- Performance

---

### Alternative Launchers

#### `launch_unified_dashboard.py`
**Purpose:** Alternative dashboard launcher

**Usage:**
```bash
python scripts/launch_unified_dashboard.py
```

**Note:** Use `../start_dashboard.py` instead (main launcher in root directory)

---

## 🚀 Typical Workflow

### First Time Setup

```bash
# Step 1: Validate system
python scripts/validate_startup.py

# Step 2: Train models (optional, dashboard works without)
python scripts/setup_models.py --quick

# Step 3: Start dashboard (from root)
cd ..
python start_dashboard.py
```

### Regular Usage

```bash
# Just start the dashboard (from root)
python start_dashboard.py

# Dashboard available at: http://127.0.0.1:8050
```

### Model Retraining

```bash
# Retrain forecasting only
python scripts/train_forecasting_models.py --quick

# Retrain anomaly detection only
python scripts/train_anomaly_models.py --quick

# Retrain everything
python scripts/setup_models.py
```

---

## 📊 Script Dependencies

### Required for Dashboard to Run:
- **NONE** - Dashboard has built-in fallbacks

### Required for Forecasting:
- `train_forecasting_models.py` - Must run once

### Required for Anomaly Detection:
- `train_anomaly_models.py` - Optional, has fallbacks

---

## ⚙️ Configuration

All scripts use configuration from:
- `../config/config.yaml` - Main configuration
- `../config/equipment_config.py` - Equipment definitions
- `../config/settings.py` - Settings module

---

## 🔍 Troubleshooting

### Script Won't Run

**Issue:** `ModuleNotFoundError`

**Fix:**
```bash
# Ensure you're in project root or scripts/ directory
cd "IOT Predictive Maintenece System"
python scripts/script_name.py

# Or from scripts/ directory
cd scripts
python script_name.py
```

### Import Errors

**Issue:** `ImportError: cannot import name 'X'`

**Fix:**
```bash
# Scripts should work from either root or scripts/ directory
# If issues persist, ensure project root is in Python path
```

### Training Fails

**Issue:** Training scripts fail or hang

**Fix:**
- Use `--quick` flag for faster training
- Check available memory (needs 2GB+)
- Check disk space (needs 1GB+ free)
- Review `../logs/training.log`

---

## 📚 Additional Documentation

- **Main README:** `../README.md`
- **Deployment Guide:** `../docs/PRODUCTION_DEPLOYMENT.md`
- **Troubleshooting:** `../docs/TROUBLESHOOTING.md`
- **Completion Summary:** `../docs/DEPLOYMENT_COMPLETE.md`

---

## 💡 Tips

1. **Always use `--quick` flag first** for faster testing
2. **Dashboard works without trained models** - Uses mock data
3. **Training takes time** - Quick: 10-15 min, Full: 1-2 hours
4. **Check logs** in `../logs/` if scripts fail
5. **Run validation** before reporting issues

---

## 🎯 Quick Commands

```bash
# From project root:

# Validate everything
python scripts/validate_startup.py

# Quick training
python scripts/setup_models.py --quick

# Start dashboard
python start_dashboard.py

# Check logs
cat logs/training.log          # Linux/Mac
type logs\training.log          # Windows
```
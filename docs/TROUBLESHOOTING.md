# 🔧 Troubleshooting Guide

## IoT Predictive Maintenance System - Complete Troubleshooting Reference

This guide helps you diagnose and fix common issues when deploying or running the dashboard.

---

## 📋 Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Installation Issues](#installation-issues)
3. [Dashboard Startup Issues](#dashboard-startup-issues)
4. [Runtime Errors](#runtime-errors)
5. [Performance Issues](#performance-issues)
6. [Data & Model Issues](#data--model-issues)
7. [Network & Port Issues](#network--port-issues)
8. [Advanced Debugging](#advanced-debugging)

---

## Quick Diagnostics

### Run Automated Checks

```bash
# Quick check with auto-fix
python preflight_check.py

# Comprehensive validation
python validate_startup.py

# Deployment verification
python verify_deployment.py
```

### Check System Health

```bash
# Python version
python --version

# Installed packages
pip list

# Disk space
df -h  # Linux/Mac
dir    # Windows

# Memory
free -h  # Linux
vm_stat  # Mac
```

---

## Installation Issues

### Issue 1.1: Python Version Too Old

**Symptom:**
```
ERROR: Python 3.7 is not supported
```

**Diagnosis:**
```bash
python --version
```

**Solution:**
```bash
# Install Python 3.8 or higher
# Download from: https://www.python.org/downloads/

# Or use conda:
conda install python=3.11

# Verify:
python --version  # Should show 3.8+
```

---

### Issue 1.2: pip Not Found

**Symptom:**
```
'pip' is not recognized as an internal or external command
```

**Solution:**
```bash
# Windows:
python -m ensurepip --upgrade
python -m pip install --upgrade pip

# Linux/Mac:
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```

---

### Issue 1.3: Permission Denied During Installation

**Symptom:**
```
ERROR: Could not install packages due to an EnvironmentError: [Errno 13] Permission denied
```

**Solution:**
```bash
# Option 1: Use virtual environment (recommended)
python -m venv venv
# Activate and install
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Option 2: User install
pip install --user -r requirements.txt

# Option 3: Admin (not recommended)
# Run terminal as administrator
```

---

### Issue 1.4: Dependency Conflicts

**Symptom:**
```
ERROR: package-a requires package-b<2.0, but you have package-b 2.1
```

**Solution:**
```bash
# Clean install
pip uninstall -y -r requirements.txt
pip cache purge
pip install -r requirements.txt

# Or use fresh virtual environment
rm -rf venv  # Linux/Mac
rmdir /s venv  # Windows
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

---

## Dashboard Startup Issues

### Issue 2.1: ImportError - Module Not Found

**Symptom:**
```
ImportError: No module named 'dash'
```

**Diagnosis:**
```bash
pip list | grep dash
```

**Solution:**
```bash
# Install missing package
pip install dash

# Or reinstall all dependencies
pip install -r requirements.txt

# Verify installation
python -c "import dash; print(dash.__version__)"
```

---

### Issue 2.2: ImportError - Cannot Import Name

**Symptom:**
```
ImportError: cannot import name 'EnhancedIoTDashboard' from 'src.presentation.dashboard.enhanced_app'
```

**Diagnosis:**
```bash
# Check if file exists
ls src/presentation/dashboard/enhanced_app.py

# Check file contents
head -20 src/presentation/dashboard/enhanced_app.py
```

**Solution:**
This should NOT happen with the fixed codebase. If it does:

```bash
# Verify critical files exist:
python validate_startup.py

# Files should be present:
# ✓ src/presentation/dashboard/enhanced_app.py
# ✓ src/presentation/dashboard/enhanced_app_optimized.py
# ✓ src/presentation/dashboard/enhanced_callbacks_simplified.py

# If missing, restore from backup or re-download codebase
```

---

### Issue 2.3: Dashboard Hangs on Startup

**Symptom:**
- Dashboard starts but hangs
- No error message
- Browser shows "Loading..."

**Diagnosis:**
```bash
# Check if services are hanging
# Look for timeout messages in console
```

**Solution:**
The dashboard has anti-hanging protection built-in. If it hangs:

1. **Wait 30 seconds** - Services have timeout protection
2. **Check logs:**
   ```bash
   cat logs/iot_system.log | tail -50
   ```
3. **Restart with debug:**
   ```bash
   # Edit start_dashboard.py
   # Change: debug=False
   # To: debug=True
   ```
4. **Use mock services:**
   - Dashboard automatically falls back to mock services if real ones timeout
   - Check console for "Using mock service" messages

---

### Issue 2.4: ModuleNotFoundError for TensorFlow

**Symptom:**
```
ModuleNotFoundError: No module named 'tensorflow'
```

**Solution:**
```bash
# Install TensorFlow
pip install tensorflow==2.15.0

# If installation fails, try CPU version:
pip install tensorflow-cpu==2.15.0

# If still fails (M1 Mac), use:
pip install tensorflow-macos==2.15.0
pip install tensorflow-metal==1.1.0

# Verify:
python -c "import tensorflow as tf; print(tf.__version__)"
```

**Note:** Dashboard works without TensorFlow, but forecasting won't work.

---

## Runtime Errors

### Issue 3.1: Dashboard Crashes After Starting

**Symptom:**
- Dashboard starts successfully
- Crashes when accessing certain tabs
- Error in console

**Diagnosis:**
1. Check which tab causes crash
2. Check console error message
3. Check logs: `cat logs/dashboard.log`

**Solution:**
```bash
# 1. Clear cache
rm -rf cache/*

# 2. Restart with debug mode
python start_dashboard.py

# 3. Check specific tab error in logs

# 4. If forecasting tab crashes:
#    - Check if models exist
#    - Train models if needed:
python train_forecasting_models.py --quick
```

---

### Issue 3.2: Graphs Not Displaying

**Symptom:**
- Dashboard loads
- Graphs show as blank/empty
- No data visible

**Solution:**
```bash
# 1. Check data files
ls -lh data/raw/smap/*.npy
ls -lh data/raw/msl/*.npy

# 2. Check browser console (F12)
#    - Look for JavaScript errors
#    - Check network tab for failed requests

# 3. Clear browser cache
#    - Hard refresh: Ctrl+Shift+R (Windows/Linux)
#    - Hard refresh: Cmd+Shift+R (Mac)

# 4. Try different browser
#    - Chrome (recommended)
#    - Firefox
#    - Edge
```

---

### Issue 3.3: Callbacks Not Working

**Symptom:**
- Buttons don't respond
- Dropdowns don't update
- Interactive features broken

**Solution:**
```bash
# 1. Check browser console (F12) for errors

# 2. Verify callback registration
#    Look for "callbacks registered" in logs

# 3. Restart dashboard
#    Ctrl+C to stop
#    python start_dashboard.py

# 4. Check if enhanced_callbacks_simplified.py exists
ls src/presentation/dashboard/enhanced_callbacks_simplified.py
```

---

## Performance Issues

### Issue 4.1: Dashboard Loading Slowly

**Symptom:**
- Dashboard takes >30 seconds to load
- Pages are sluggish

**Solution:**
```bash
# 1. Check system resources
top  # Linux
htop  # Linux (better)
Activity Monitor  # Mac
Task Manager  # Windows

# 2. Optimize configuration
# Edit config/config.yaml:
dashboard:
  features:
    update_interval: 10000  # Increase from 5000ms

# 3. Reduce batch size
preprocessing:
  window:
    size: 50  # Reduce from 100

# 4. Close other applications

# 5. Use production mode
# Edit start_dashboard.py:
# Change: debug=True
# To: debug=False
```

---

### Issue 4.2: High Memory Usage

**Symptom:**
- System runs out of memory
- Dashboard crashes with MemoryError

**Solution:**
```bash
# 1. Monitor memory
# Check memory usage:
free -h  # Linux
vm_stat  # Mac

# 2. Reduce memory footprint
# Edit config/config.yaml:
anomaly_detection:
  telemanom:
    batch_size: 32  # Reduce from 70

forecasting:
  transformer:
    training:
      batch_size: 16  # Reduce from 32

# 3. Train models with --quick flag
python train_forecasting_models.py --quick

# 4. Restart services periodically
#    Stop and restart dashboard every few hours
```

---

### Issue 4.3: High CPU Usage

**Symptom:**
- CPU at 100%
- System overheating
- Dashboard slow

**Solution:**
```bash
# 1. Check what's using CPU
top -o cpu  # Sort by CPU

# 2. Reduce update frequency
# Edit config/config.yaml:
dashboard:
  features:
    real_time_updates: false
    update_interval: 30000  # 30 seconds

# 3. Disable monitoring
monitoring:
  enabled: false

# 4. Use fewer workers (production mode)
gunicorn app:server --workers 2  # Instead of 4
```

---

## Data & Model Issues

### Issue 5.1: NASA Data Files Missing

**Symptom:**
```
ERROR: NASA data files missing
```

**Solution:**
Dashboard works with mock data if NASA files are missing:

```bash
# 1. Verify data directory structure
python preflight_check.py

# 2. Dashboard will use mock data automatically
#    Look for: "Using mock data" in console

# 3. To use real data:
#    - Copy NASA .npy files to:
#      data/raw/smap/train.npy
#      data/raw/smap/test.npy
#      data/raw/msl/train.npy
#      data/raw/msl/test.npy

# 4. Restart dashboard
```

---

### Issue 5.2: Models Not Found

**Symptom:**
```
WARNING: No models found - forecasting disabled
```

**Solution:**
```bash
# Option 1: Quick training (10-15 min)
python train_forecasting_models.py --quick

# Option 2: Full training (1-2 hours)
python train_forecasting_models.py

# Option 3: Continue without forecasting
#    Dashboard works without models
#    Forecasting tab will show placeholder

# Check model files:
ls -lh data/models/transformer/
```

---

### Issue 5.3: Model Loading Errors

**Symptom:**
```
ERROR: Cannot load model for sensor X
```

**Solution:**
```bash
# 1. Check model files
ls data/models/transformer/

# 2. Check TensorFlow installation
python -c "import tensorflow as tf; print(tf.__version__)"

# 3. Retrain specific model
python train_forecasting_models.py --sensors SMAP-PWR-001

# 4. Check model registry
ls data/models/registry/

# 5. Clear corrupted models
rm -rf data/models/transformer/*
python train_forecasting_models.py --quick
```

---

## Network & Port Issues

### Issue 6.1: Port Already in Use

**Symptom:**
```
OSError: [Errno 48] Address already in use
ERROR: Port 8050 is already in use
```

**Solution:**
```bash
# Option 1: Kill process using port
# Windows:
netstat -ano | findstr :8050
taskkill /PID <PID> /F

# Linux/Mac:
lsof -ti:8050 | xargs kill -9

# Option 2: Use different port
# Edit start_dashboard.py line 38:
port=8051  # Change from 8050

# Option 3: Set environment variable
export DASHBOARD_PORT=8051
python start_dashboard.py
```

---

### Issue 6.2: Cannot Access Dashboard from Browser

**Symptom:**
- Dashboard says "Running on http://127.0.0.1:8050"
- Browser shows "Connection refused"

**Solution:**
```bash
# 1. Check dashboard is actually running
#    Look for "Dash is running on" message

# 2. Try different URL:
http://localhost:8050
http://127.0.0.1:8050
http://0.0.0.0:8050

# 3. Check firewall
#    Temporarily disable firewall
#    Add exception for port 8050

# 4. Check if port is listening
# Windows:
netstat -an | findstr :8050

# Linux/Mac:
netstat -an | grep :8050
lsof -i :8050

# 5. Use 0.0.0.0 instead of 127.0.0.1
# Edit start_dashboard.py:
host='0.0.0.0'  # Instead of 127.0.0.1
```

---

### Issue 6.3: Slow Network Requests

**Symptom:**
- Dashboard loads but very slow
- Network tab shows slow requests

**Solution:**
```bash
# 1. Check network connectivity
ping google.com

# 2. Use local mode (no external resources)
# Dashboard already uses local assets

# 3. Check browser network tab (F12)
#    Look for slow/failed requests

# 4. Clear browser cache
#    Ctrl+Shift+Delete

# 5. Disable real-time updates
# Edit config/config.yaml:
dashboard:
  features:
    real_time_updates: false
```

---

## Advanced Debugging

### Enable Debug Mode

```python
# Edit start_dashboard.py
app.run_server(
    host='127.0.0.1',
    port=8050,
    debug=True,  # Enable debug mode
    dev_tools_hot_reload=True
)
```

### Check Logs

```bash
# View all logs
ls -lh logs/

# Latest errors
tail -50 logs/iot_system.log

# Follow live logs
tail -f logs/iot_system.log

# Search for errors
grep -i error logs/*.log

# Dashboard specific logs
cat logs/dashboard.log
```

### Python Debugging

```python
# Add debugging to any file
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add debug statements
logger.debug("Variable value: %s", variable)
logger.info("Function called")
logger.error("Error occurred: %s", error)
```

### Test Individual Components

```bash
# Test data loader
python -c "
from src.infrastructure.data.nasa_data_loader import NASADataLoader
loader = NASADataLoader('data/raw')
data = loader.get_sensor_data('SMAP-PWR-001', hours_back=24)
print(f'Loaded {len(data[\"values\"])} data points')
"

# Test anomaly service
python -c "
from src.core.services.anomaly_service import AnomalyDetectionService
service = AnomalyDetectionService()
print('Anomaly service initialized')
"

# Test forecasting service
python -c "
from src.core.services.forecasting_service import ForecastingService
service = ForecastingService()
print('Forecasting service initialized')
"
```

### Check Environment

```bash
# Print Python path
python -c "import sys; print('\n'.join(sys.path))"

# Check imports
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

try:
    from src.presentation.dashboard.enhanced_app import EnhancedIoTDashboard
    print('✓ enhanced_app imports successfully')
except ImportError as e:
    print(f'✗ enhanced_app import failed: {e}')

try:
    from src.presentation.dashboard.enhanced_callbacks_simplified import register_enhanced_callbacks
    print('✓ enhanced_callbacks_simplified imports successfully')
except ImportError as e:
    print(f'✗ enhanced_callbacks_simplified import failed: {e}')
"
```

---

## Common Error Messages & Solutions

### Error: "Dashboard initialization timeout"

**Cause:** Service initialization taking too long

**Solution:**
- Dashboard has 30-second timeout protection
- Will automatically use mock services
- Check logs for slow service
- Increase timeout in code if needed

---

### Error: "Failed to load model registry"

**Cause:** Model registry directory missing or corrupted

**Solution:**
```bash
# Create directory
mkdir -p data/models/registry

# Restart dashboard
python start_dashboard.py
```

---

### Error: "Config file not found"

**Cause:** config.yaml missing

**Solution:**
```bash
# Verify config exists
ls config/config.yaml

# If missing, run validation
python validate_startup.py

# Config should exist - restore from backup if needed
```

---

### Error: "Cannot connect to database"

**Cause:** Database configuration issue

**Solution:**
```bash
# Dashboard uses SQLite by default (no setup needed)
# Check config/config.yaml:

data_ingestion:
  database:
    type: "sqlite"  # Should be sqlite, not postgresql
    sqlite:
      path: "./data/iot_telemetry.db"

# Create database directory
mkdir -p data
```

---

## Getting Additional Help

### 1. Check Documentation
- `README.md` - Overview
- `PRODUCTION_DEPLOYMENT.md` - Deployment guide
- `TROUBLESHOOTING.md` - This file

### 2. Run Diagnostics
```bash
python validate_startup.py
python verify_deployment.py
```

### 3. Check Logs
```bash
cat logs/iot_system.log
cat logs/dashboard.log
```

### 4. Review Configuration
```bash
cat config/config.yaml
```

### 5. System Information
```bash
python --version
pip list
df -h
free -h
```

---

## Quick Reference - Common Commands

```bash
# Start dashboard
python start_dashboard.py

# Run checks
python preflight_check.py
python validate_startup.py
python verify_deployment.py

# Train models
python train_forecasting_models.py --quick

# View logs
tail -f logs/iot_system.log

# Check port
netstat -an | grep :8050  # Linux/Mac
netstat -ano | findstr :8050  # Windows

# Kill process on port
lsof -ti:8050 | xargs kill -9  # Linux/Mac
taskkill /PID <PID> /F  # Windows (after netstat)

# Clear cache
rm -rf cache/*
rm -rf __pycache__
find . -type d -name __pycache__ -exec rm -rf {} +

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Fresh start
rm -rf venv
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

---

**Still having issues?**

1. Run full diagnostic: `python validate_startup.py`
2. Check system requirements
3. Review error messages carefully
4. Search this guide for specific error messages
5. Check logs in `logs/` directory

**Remember:** The dashboard is designed to work even with missing components. It will use mock data and services when needed.
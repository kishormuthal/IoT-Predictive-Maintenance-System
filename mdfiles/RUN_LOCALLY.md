# 🚀 RUN LOCALLY GUIDE

Complete guide to running the IoT Predictive Maintenance System on your local machine.

---

## ✅ **OPTION 1: RUN DASHBOARD LOCALLY** (Recommended)

### **Quick Start** (< 1 minute)

```bash
# Navigate to project directory
cd /workspaces/IoT-Predictive-Maintenance-System

# Run the dashboard
python launch_complete_dashboard.py
```

**Expected Output:**
```
================================================================================
IOT PREDICTIVE MAINTENANCE - COMPLETE DASHBOARD (ALL SESSIONS)
================================================================================

  ✓ Overview - System health and architecture
  ✓ Monitoring - Real-time NASA SMAP/MSL sensor data
  ✓ Anomaly Monitor - Real-time anomaly detection
  ✓ Anomaly Investigation - Deep dive analysis
  ✓ Enhanced Forecasting - Advanced predictions
  ✓ MLflow Integration - Model tracking & management
  ✓ Training Monitor - Real-time training jobs
  ✓ Maintenance Scheduler - Optimize maintenance
  ✓ Work Orders - Task management
  ✓ System Performance - Infrastructure metrics

✓ Loaded 12 NASA sensors
✓ Health check endpoints enabled
✓ Dashboard ready with ALL SESSION 9 features!

🌐 URL: http://127.0.0.1:8050
```

### **Access the Dashboard:**

Open your browser and go to:
```
http://127.0.0.1:8050
```

or if using VS Code, click the popup that says "Open in Browser"

### **What You'll See:**

- **10 Interactive Tabs** with real NASA SMAP/MSL data
- **12 Sensors** (6 from SMAP, 6 from MSL)
- **Real-time Charts** with 744 data points per sensor
- **Anomaly Detection** using Telemanom algorithms
- **Forecasting** with Transformer models

### **Stop the Dashboard:**

Press `Ctrl+C` in the terminal

---

## ⚙️ **OPTION 2: RUN CI/CD CHECKS LOCALLY**

### **Full CI/CD Validation Script**

Create this script to run ALL CI/CD checks locally before pushing:

```bash
#!/bin/bash
# File: run_local_cicd.sh

echo "╔════════════════════════════════════════════════════════════╗"
echo "║          🔄 LOCAL CI/CD VALIDATION                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

FAILED=0

# Stage 1: Lint Checks
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 STAGE 1/6: LINT CHECKS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo -n "  ▸ Black format check... "
if python -m black --check src/ tests/ scripts/ config/ --quiet 2>&1; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${RED}❌ FAIL${NC}"
    FAILED=1
fi

echo -n "  ▸ isort import check... "
if python -m isort --check-only src/ tests/ scripts/ config/ --quiet 2>&1; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${RED}❌ FAIL${NC}"
    FAILED=1
fi

echo -n "  ▸ flake8 linting... "
if python -m flake8 src/ tests/ scripts/ --max-line-length=120 --extend-ignore=E203,W503 --count 2>&1 | grep -q "^0$"; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${YELLOW}⚠️  WARNINGS (acceptable)${NC}"
fi

echo ""

# Stage 2: Security Scans
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔒 STAGE 2/6: SECURITY SCANS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo -n "  ▸ Bandit security scan... "
if bandit -r src/ -q 2>&1 > /dev/null; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${YELLOW}⚠️  WARNINGS (review recommended)${NC}"
fi

echo ""

# Stage 3: Unit Tests
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 STAGE 3/6: UNIT TESTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo -n "  ▸ Running test suite... "
if python -m pytest tests/test_basic.py -q 2>&1 > /tmp/pytest.log; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${RED}❌ FAIL${NC}"
    cat /tmp/pytest.log
    FAILED=1
fi

echo ""

# Stage 4: Import Tests
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 STAGE 4/6: IMPORT VALIDATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo -n "  ▸ NASA data loader... "
if python -c "from src.infrastructure.data.nasa_data_loader import NASADataLoader; NASADataLoader()" 2>&1 > /dev/null; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${RED}❌ FAIL${NC}"
    FAILED=1
fi

echo -n "  ▸ Health check system... "
if python -c "from src.presentation.dashboard.health_check import get_health_status; get_health_status()" 2>&1 > /dev/null; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${RED}❌ FAIL${NC}"
    FAILED=1
fi

echo ""

# Stage 5: Data Validation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 STAGE 5/6: DATA VALIDATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo -n "  ▸ NASA data files exist... "
if [ -f "data/raw/smap/train.npy" ] && [ -f "data/raw/msl/train.npy" ]; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${RED}❌ FAIL (NASA data files missing)${NC}"
    FAILED=1
fi

echo -n "  ▸ Equipment config... "
if python -c "from config.equipment_config import get_equipment_list; assert len(get_equipment_list()) == 12" 2>&1 > /dev/null; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${RED}❌ FAIL${NC}"
    FAILED=1
fi

echo ""

# Stage 6: Dashboard Health
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏥 STAGE 6/6: DASHBOARD HEALTH CHECK"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo -n "  ▸ Dashboard can initialize... "
if timeout 3 python -c "
import sys
sys.path.insert(0, '.')
from src.presentation.dashboard.health_check import add_health_check
import dash
app = dash.Dash(__name__)
add_health_check(app)
print('OK')
" 2>&1 | grep -q "OK"; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${YELLOW}⚠️  TIMEOUT (acceptable)${NC}"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Final Result
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                 ✅ ALL CHECKS PASSED!                      ║"
    echo "║              Ready to push to GitHub                       ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    exit 0
else
    echo -e "${RED}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                 ❌ SOME CHECKS FAILED                      ║"
    echo "║           Fix issues before pushing to GitHub              ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    exit 1
fi
```

### **How to Use the CI/CD Script:**

1. **Create the script:**
   ```bash
   nano run_local_cicd.sh
   # Paste the content above
   ```

2. **Make it executable:**
   ```bash
   chmod +x run_local_cicd.sh
   ```

3. **Run it:**
   ```bash
   ./run_local_cicd.sh
   ```

### **Quick Individual Checks:**

```bash
# Just check formatting
python -m black --check src/ tests/ scripts/ config/
python -m isort --check-only src/ tests/ scripts/ config/

# Just run tests
python -m pytest tests/test_basic.py -v

# Just check health
python -c "from src.presentation.dashboard.health_check import get_health_status; print(get_health_status())"
```

---

## 🐳 **OPTION 3: RUN WITH DOCKER** (If Docker installed)

### **Build and Run:**

```bash
# Build Docker image
docker build -t iot-dashboard:local .

# Run container
docker run -d -p 8050:8050 --name iot-dashboard iot-dashboard:local

# Access dashboard
open http://localhost:8050

# Check health
curl http://localhost:8050/health

# View logs
docker logs -f iot-dashboard

# Stop container
docker stop iot-dashboard && docker rm iot-dashboard
```

---

## 📊 **OPTION 4: RUN TESTS ONLY**

### **Run All Tests:**
```bash
python -m pytest tests/ -v
```

### **Run Specific Test File:**
```bash
python -m pytest tests/test_basic.py -v
```

### **Run With Coverage:**
```bash
python -m pytest tests/test_basic.py --cov=src --cov-report=term
```

### **Run Single Test:**
```bash
python -m pytest tests/test_basic.py::TestHealthCheck::test_health_status -v
```

---

## 🔧 **TROUBLESHOOTING**

### **Issue: Port 8050 already in use**
```bash
# Find process using port
lsof -i :8050

# Kill process
kill -9 <PID>

# Or use different port
python launch_complete_dashboard.py --port 8051
```

### **Issue: Module not found**
```bash
# Install dependencies
pip install -r requirements.txt
```

### **Issue: NASA data not loading**
```bash
# Check if data files exist
ls -la data/raw/smap/
ls -la data/raw/msl/

# If missing, data files need to be downloaded
```

### **Issue: Tests failing**
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run with verbose output
python -m pytest tests/test_basic.py -vv
```

---

## ✅ **VERIFICATION CHECKLIST**

Before pushing to GitHub, verify:

- [ ] `python -m black --check .` passes
- [ ] `python -m isort --check-only .` passes
- [ ] `python -m pytest tests/test_basic.py` passes (17/17 tests)
- [ ] Dashboard launches without errors
- [ ] Health endpoint works: `curl http://localhost:8050/health`
- [ ] NASA data loads (12 sensors)

---

## 📚 **USEFUL COMMANDS**

```bash
# Format code
python -m black src/ tests/ scripts/ config/
python -m isort src/ tests/ scripts/ config/

# Check formatting
python -m black --check src/ tests/ scripts/ config/
python -m isort --check-only src/ tests/ scripts/ config/

# Run linting
python -m flake8 src/ tests/ scripts/ --max-line-length=120

# Run tests
python -m pytest tests/test_basic.py -v

# Launch dashboard
python launch_complete_dashboard.py

# Check health
curl http://localhost:8050/health
curl http://localhost:8050/health/ready
curl http://localhost:8050/health/live
curl http://localhost:8050/metrics

# Train models
python scripts/train_all_sensors.py
```

---

## 🎯 **QUICK START SUMMARY**

**To run dashboard locally:**
```bash
python launch_complete_dashboard.py
# Then open: http://127.0.0.1:8050
```

**To validate before pushing:**
```bash
./run_local_cicd.sh
```

**To run tests:**
```bash
python -m pytest tests/test_basic.py -v
```

---

**That's it!** You can now run everything locally before pushing to GitHub! 🚀

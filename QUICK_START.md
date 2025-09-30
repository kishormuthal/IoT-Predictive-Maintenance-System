# IoT Predictive Maintenance Dashboard - Quick Start Guide

## ✅ Dashboard Status: ONLINE

**URL:** http://127.0.0.1:8050
**Status:** Running (PID 25748)
**Uptime:** Stable

---

## How to Access

1. **Open your web browser**
2. **Navigate to:** http://127.0.0.1:8050
3. **Start exploring!** Click on any of the 7 tabs

---

## Available Features

### 🏠 Overview Tab
- System health status
- Total sensors (12 NASA sensors)
- Model availability
- Active alerts
- System architecture info

### 📊 Monitoring Tab
- Real-time sensor data
- Live monitoring of 12 NASA SMAP/MSL sensors
- Time series visualization

### ⚠️ Anomalies Tab
- NASA Telemanom anomaly detection
- 12 pre-trained models ready
- Real-time anomaly alerts

### 📈 Forecasting Tab
- Transformer-based predictions
- 219K parameter models
- Future sensor value predictions

### 🔧 Maintenance Tab
- Predictive maintenance scheduling
- Remaining useful life (RUL) predictions
- Maintenance task tracking

### 📋 Work Orders Tab
- Work order management
- Task creation and tracking
- Maintenance workflow

### ⚙️ System Tab
- Training hub
- Model registry
- System administration
- Performance monitoring

---

## Starting the Dashboard (if not running)

```bash
# Navigate to project directory
cd /workspaces/IoT-Predictive-Maintenance-System

# Start the dashboard
python launch_dashboard_simple.py
```

**Expected output:**
```
============================================================
IoT PREDICTIVE MAINTENANCE DASHBOARD
Simple Launch Mode - Fast Startup
============================================================
[INFO] Importing dashboard components...
[INFO] Creating Dash application...
[INFO] Dashboard configuration complete
[URL] Starting server at: http://127.0.0.1:8050
[INFO] Press Ctrl+C to stop
------------------------------------------------------------
Dash is running on http://127.0.0.1:8050/
```

---

## Stopping the Dashboard

**Method 1: Using Ctrl+C**
```bash
# If running in foreground, press:
Ctrl+C
```

**Method 2: Kill by PID**
```bash
# Find the process
ps aux | grep launch_dashboard

# Kill it
kill <PID>
```

**Method 3: Kill by name**
```bash
pkill -f launch_dashboard_simple
```

---

## Checking Dashboard Status

```bash
# Check if running
ps aux | grep launch_dashboard_simple | grep -v grep

# Check port
ss -tlnp | grep 8050

# Test HTTP response
curl -s -o /dev/null -w "Status: %{http_code}\n" http://127.0.0.1:8050
```

---

## Troubleshooting

### Dashboard not loading?

1. **Check if running:**
   ```bash
   ps aux | grep launch_dashboard
   ```

2. **Check logs:**
   ```bash
   tail -50 /tmp/dashboard_running.log
   ```

3. **Restart:**
   ```bash
   pkill -f launch_dashboard
   python launch_dashboard_simple.py
   ```

### Port 8050 already in use?

```bash
# Find what's using port 8050
ss -tlnp | grep 8050

# Kill the process
kill <PID>
```

### Missing dependencies?

```bash
# Install all requirements
pip install -r requirements.txt
```

---

## Technical Details

### System Configuration
- **Sensors:** 12 (6 SMAP + 6 MSL)
- **Models:** 12 pre-trained Telemanom models
- **Architecture:** Clean Architecture (4 layers)
- **Framework:** Dash 2.16.1 + Bootstrap

### Resource Usage
- **Memory:** ~42 MB
- **CPU:** < 1%
- **Startup Time:** < 5 seconds

### Requirements
- Python 3.11+
- TensorFlow 2.15.0
- Dash 2.16.1
- See `requirements.txt` for full list

---

## File Locations

### Important Files
- **Dashboard Launcher:** `launch_dashboard_simple.py`
- **Main Dashboard:** `src/presentation/dashboard/unified_dashboard.py`
- **Layouts:** `src/presentation/dashboard/layouts/`
- **Configuration:** `config/equipment_config.py`
- **Data:** `data/raw/smap/` and `data/raw/msl/`
- **Models:** `data/models/nasa_equipment_models/`

### Logs
- **Runtime Logs:** `/tmp/dashboard_running.log`
- **Startup Logs:** `/tmp/dashboard_startup.log`

---

## Next Steps

1. ✅ **Dashboard is running** - Access it at http://127.0.0.1:8050
2. 🔍 **Explore all 7 tabs** - Click through each feature
3. 📊 **View sensor data** - Check the Monitoring tab
4. ⚠️ **Test anomaly detection** - Explore the Anomalies tab
5. 📈 **See predictions** - Try the Forecasting tab

---

## Support Resources

- **Launch Report:** `DASHBOARD_LAUNCH_REPORT.md` (detailed technical report)
- **Configuration:** `config/equipment_config.py` (sensor setup)
- **Documentation:** `docs/` directory

---

## Quick Commands

```bash
# Start dashboard
python launch_dashboard_simple.py

# Check status
ps aux | grep launch_dashboard

# View logs
tail -f /tmp/dashboard_running.log

# Stop dashboard
pkill -f launch_dashboard_simple

# Restart dashboard
pkill -f launch_dashboard_simple && python launch_dashboard_simple.py
```

---

**🎉 You're all set! Enjoy exploring your IoT Predictive Maintenance Dashboard!**

**Access URL:** http://127.0.0.1:8050
# IoT Predictive Maintenance System - User Guide

**Version**: 1.0.0
**Last Updated**: 2025-10-02

---

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Dashboard Overview](#dashboard-overview)
4. [Feature Guides](#feature-guides)
5. [Common Tasks](#common-tasks)
6. [Troubleshooting](#troubleshooting)
7. [FAQ](#faq)

---

## Introduction

### What is IoT Predictive Maintenance?

The IoT Predictive Maintenance System is an advanced ML-powered platform that:
- **Monitors** sensor data from IoT equipment in real-time
- **Detects** anomalies using state-of-the-art deep learning models
- **Forecasts** future sensor values with uncertainty quantification
- **Predicts** equipment failures before they occur
- **Optimizes** maintenance scheduling to minimize downtime
- **Tracks** model performance and training jobs

### Key Benefits

- **Reduce Downtime**: Predict failures 24-168 hours in advance
- **Optimize Costs**: Reduce emergency maintenance by up to 50%
- **Improve Safety**: Early warning system for critical failures
- **Data-Driven Decisions**: ML-powered insights and recommendations

---

## Quick Start

### Accessing the Dashboard

1. **Local Development**:
   ```bash
   python run_full_dashboard.py
   ```
   Access at: http://localhost:8050

2. **Production**:
   Navigate to: https://your-domain.com

### First Login

1. Enter your credentials (if authentication enabled)
2. You'll see the **Overview** page with system health

### Navigation

- **Sidebar**: Access all dashboard pages
- **Top Bar**: System status, alerts, user menu
- **Search**: Find sensors, equipment, or anomalies

---

## Dashboard Overview

### Overview Page

**Purpose**: System health at a glance

**Key Metrics**:
- **Total Sensors**: Number of active sensors
- **Active Anomalies**: Current anomalous conditions
- **Predicted Failures**: Upcoming equipment failures
- **System Health**: Overall system status (0-100%)

**Components**:
- Real-time sensor map
- Critical alerts feed
- System health trends
- Recent anomalies

**Usage**:
1. Check system health score (should be >90%)
2. Review critical alerts (red badges)
3. Click sensor markers on map for details
4. Use time range selector (1h, 6h, 24h, 7d, 30d)

### Anomaly Monitor

**Purpose**: Real-time anomaly detection and alerts

**Features**:
- Live anomaly feed with severity badges
- Interactive timeline chart
- Sensor-specific anomaly counts
- Alert acknowledgment

**Usage**:
1. **View Anomalies**:
   - Critical (red): Immediate action required
   - High (orange): Action required within 4 hours
   - Medium (yellow): Monitor closely
   - Low (blue): Informational

2. **Investigate Anomaly**:
   - Click anomaly row to expand details
   - View sensor readings before/after anomaly
   - Check correlation with other sensors
   - See recommended actions

3. **Acknowledge Alert**:
   - Click "Acknowledge" button
   - Add notes about actions taken
   - Alert moves to "Acknowledged" section

### Advanced Anomaly Investigation

**Purpose**: Deep dive into anomalies with root cause analysis

**Features**:
- Multi-sensor correlation analysis
- Root cause factor identification
- Historical similar anomalies
- Pattern clustering
- Export investigation reports

**Usage**:

**Step 1: Select Anomaly**
- Use filters (sensor, severity, time, type)
- Click anomaly point on timeline

**Step 2: Analyze**
- **Sensor Analysis Tab**: View time series with statistical context
- **Root Cause Tab**: See contributing factors (ranked by importance)
- **Correlation Tab**: Find related sensor anomalies
- **Similar Anomalies Tab**: Review historical matches
- **Recommendations Tab**: View suggested actions

**Step 3: Take Action**
- Create work order directly from recommendations
- Enable enhanced monitoring for correlated sensors
- Configure predictive alerts

**Example Workflow**:
```
1. Filter by Sensor: "SMAP - Power System"
2. Select time range: "Last 24 Hours"
3. Click anomaly spike at 14:30
4. Review Root Cause tab → "Sensor Drift" (85% contribution)
5. Click "Create Work Order" → Schedule calibration
```

### Enhanced Forecasting

**Purpose**: Advanced time series forecasting with uncertainty quantification

**Features**:
- Multi-model forecasting (LSTM, Transformer)
- Confidence intervals (80%, 90%, 95%)
- Quantile predictions
- Ensemble predictions
- Scenario analysis

**Usage**:

**Basic Forecasting**:
1. Select sensor from dropdown
2. Choose forecast horizon (1h to 168h)
3. Select model (Transformer recommended for long horizons)
4. View forecast with confidence bands

**Advanced Features**:
- **Uncertainty Quantification**: See prediction intervals
- **Quantile Forecasting**: View 10th, 25th, 50th, 75th, 90th percentiles
- **Ensemble Predictions**: Combine multiple model forecasts
- **Contributing Factors**: Understand what drives predictions

**Interpreting Results**:
- **Blue Line**: Point forecast (most likely value)
- **Light Blue Band**: 80% confidence interval
- **Lighter Blue Band**: 95% confidence interval
- **Narrow bands**: High confidence
- **Wide bands**: High uncertainty (be cautious)

### MLflow Integration

**Purpose**: Model tracking, comparison, and deployment management

**Features**:
- Embedded MLflow UI
- Experiment comparison
- Model registry
- Deployment status
- Performance trends

**Usage**:

**View Experiments**:
1. Go to "Experiment Comparison" tab
2. Select experiments to compare
3. Choose metric (accuracy, precision, RMSE, etc.)
4. View side-by-side comparison

**Manage Models**:
1. Go to "Model Registry" tab
2. Filter by type or stage
3. View model versions and metadata
4. Promote model to Production

**Monitor Deployments**:
1. Go to "Deployment Status" tab
2. View production model health
3. Check request counts and uptime
4. Review deployment timeline

### Training Monitor

**Purpose**: Real-time model training job monitoring

**Features**:
- Active training jobs with progress bars
- Training queue management
- Resource utilization (CPU, GPU, memory)
- Training history with logs
- Start new training jobs

**Usage**:

**Monitor Active Jobs**:
1. View real-time progress (epoch, loss, ETA)
2. Check loss curves
3. View/tail logs
4. Stop job if needed

**Start New Training**:
1. Click "New Training Job" button
2. Select model type and dataset
3. Configure hyperparameters:
   - Epochs (default: 50)
   - Batch size (16, 32, 64, 128)
   - Learning rate (0.0001 - 0.1)
4. Enable options:
   - Early stopping (recommended)
   - MLflow tracking (recommended)
   - GPU acceleration (if available)
   - Save checkpoints
5. Click "Start Training"

**Review History**:
1. Filter by status, model type, or time range
2. View job outcomes and metrics
3. Access detailed logs
4. Compare training runs

### Maintenance Scheduler

**Purpose**: Optimize maintenance scheduling with cost-benefit analysis

**Features**:
- Automated scheduling based on failure predictions
- Resource constraint management
- Priority-based task allocation
- Maintenance window configuration
- Work order generation

**Usage**:

**View Schedule**:
1. Calendar view shows scheduled maintenance
2. Color-coded by priority:
   - Red: Critical (within 24h)
   - Orange: High (within week)
   - Yellow: Medium (within month)
   - Green: Low (routine)

**Schedule Maintenance**:
1. Click "Add Maintenance Task"
2. Select equipment/sensor
3. Set priority and estimated duration
4. System recommends optimal time slot
5. Assign technicians
6. Generate work order

**Optimize Schedule**:
1. Click "Optimize" button
2. System re-arranges tasks to:
   - Minimize downtime
   - Respect resource constraints
   - Prioritize critical tasks
3. Review optimization suggestions
4. Accept or modify

### Work Orders

**Purpose**: Track and manage maintenance work orders

**Features**:
- Create, assign, and track work orders
- Integration with anomaly detection
- Status tracking (Open, In Progress, Completed)
- Notes and attachments
- Completion verification

**Usage**:

**Create Work Order**:
1. Click "New Work Order"
2. Fill in details:
   - Title and description
   - Related sensor/equipment
   - Priority
   - Due date
   - Assigned technician
3. Add attachments (photos, docs)
4. Submit

**Track Progress**:
1. View all work orders in table
2. Filter by status or priority
3. Click row to view details
4. Update status as work progresses
5. Add completion notes

---

## Feature Guides

### Setting Up Alerts

1. Navigate to **Settings** > **Alerts**
2. Click "Add Alert Rule"
3. Configure:
   - **Sensor**: Which sensor to monitor
   - **Condition**: Threshold or anomaly detection
   - **Severity**: Critical, High, Medium, Low
   - **Notification**: Email, SMS, Slack
   - **Recipients**: Who to notify
4. Test alert (recommended)
5. Save

**Alert Types**:
- **Threshold**: Trigger when value crosses limit
- **Anomaly**: Trigger on anomaly detection
- **Forecast**: Trigger on predicted failure
- **Degradation**: Trigger on performance decline

### Exporting Data

**Export Sensor Data**:
1. Go to sensor detail page
2. Select time range
3. Click "Export" button
4. Choose format (CSV, JSON, Excel)
5. Download

**Export Reports**:
1. Navigate to report page (Anomaly, Forecast, etc.)
2. Click "Export Report" button
3. Choose format (PDF, HTML, PNG)
4. Customize (optional):
   - Include charts
   - Include recommendations
   - Add custom notes
5. Download or email

### Configuring Dashboard

**Customize Layout**:
1. Go to **Settings** > **Dashboard**
2. Drag and drop widgets
3. Resize cards
4. Show/hide components
5. Save layout

**Set Defaults**:
1. **Default Time Range**: 1h, 6h, 24h, etc.
2. **Refresh Interval**: 5s, 30s, 1min, etc.
3. **Theme**: Light or Dark mode
4. **Default Sensors**: Pre-select sensors to display

---

## Common Tasks

### Task 1: Investigate High Priority Anomaly

```
1. Dashboard shows "3 Critical Alerts" badge
2. Click badge → Navigate to Anomaly Monitor
3. See SMAP_PWR_01 with severity: Critical
4. Click row to expand details
5. Review sensor readings → Voltage spike detected
6. Click "Investigate" → Opens Advanced Investigation
7. Review Root Cause tab → Power fluctuation (72% contribution)
8. Check Correlation tab → Thermal system also affected
9. Go to Recommendations tab
10. Click "Create Work Order" → Schedule inspection within 4 hours
11. Click "Enable Enhanced Monitoring" for correlated sensors
12. Acknowledge alert with notes
```

### Task 2: Compare Model Performance

```
1. Navigate to MLflow Integration
2. Go to "Experiment Comparison" tab
3. Select experiments:
   - lstm_predictor_exp_001
   - lstm_predictor_exp_002
   - transformer_exp_001
4. Select metric: "accuracy"
5. View comparison chart
6. Click best performing experiment → transformer_exp_001
7. Go to "Model Registry" tab
8. Find Transformer model
9. Click "Promote to Production"
10. Confirm deployment
```

### Task 3: Schedule Preventive Maintenance

```
1. Navigate to Forecasting page
2. Select sensor: MSL_MOB_F_05
3. Set horizon: 168 hours (7 days)
4. Review forecast → Predicted degradation in 5 days
5. Go to Maintenance Scheduler
6. Click "Add Maintenance Task"
7. Select equipment: MSL Mobility Front
8. Set priority: High
9. Set due date: 4 days (before predicted issue)
10. Assign technician from dropdown
11. System suggests optimal time slot
12. Accept and generate work order
```

### Task 4: Train New Anomaly Detection Model

```
1. Navigate to Training Monitor
2. Click "New Training Job"
3. Configure:
   - Model: LSTM Autoencoder
   - Dataset: SMAP
   - Epochs: 50
   - Batch size: 32
   - Learning rate: 0.001
   - Enable: Early stopping, MLflow tracking
4. Click "Start Training"
5. Monitor progress in Active Jobs tab
6. View real-time loss curve
7. Check resource utilization
8. When complete, go to MLflow Integration
9. Review experiment results
10. Register model if performance is good
```

---

## Troubleshooting

### Issue: Dashboard Not Loading

**Symptoms**: Blank page or "Cannot connect" error

**Solutions**:
1. Check if server is running:
   ```bash
   # Check process
   ps aux | grep python

   # Check port
   lsof -i :8050
   ```

2. Verify configuration:
   ```bash
   # Check config
   cat config/config.yaml
   ```

3. Check logs:
   ```bash
   tail -f logs/iot_system.log
   ```

4. Restart server:
   ```bash
   # Stop
   pkill -f run_full_dashboard

   # Start
   python run_full_dashboard.py
   ```

### Issue: No Sensor Data Displayed

**Symptoms**: Empty charts, "No data available"

**Solutions**:
1. Check sensor connectivity
2. Verify data ingestion:
   ```bash
   # Check database
   sqlite3 data/iot_telemetry.db "SELECT COUNT(*) FROM sensor_readings;"
   ```

3. Check time range selection
4. Refresh page (Ctrl+F5)

### Issue: Anomaly Detection Not Working

**Symptoms**: No anomalies detected despite obvious issues

**Solutions**:
1. Check model status:
   - Navigate to MLflow Integration
   - Verify models are in "Production" stage

2. Check thresholds:
   - Go to Settings > Anomaly Detection
   - Verify threshold values are appropriate

3. Retrain models if needed:
   - Use Training Monitor to start new training job

### Issue: Slow Performance

**Symptoms**: Dashboard loads slowly, charts lag

**Solutions**:
1. Reduce time range (e.g., 24h instead of 30d)
2. Limit number of sensors displayed
3. Enable caching in config:
   ```yaml
   dashboard:
     performance:
       cache:
         enabled: true
   ```

4. Check resource usage:
   ```bash
   # CPU and memory
   top

   # Disk space
   df -h
   ```

### Issue: MLflow Not Accessible

**Symptoms**: MLflow UI shows connection error

**Solutions**:
1. Check if MLflow server is running:
   ```bash
   ps aux | grep mlflow
   ```

2. Start MLflow server:
   ```bash
   mlflow server --host 0.0.0.0 --port 5000
   ```

3. Verify URI in config:
   ```yaml
   mlflow:
     tracking_uri: "http://localhost:5000"
   ```

---

## FAQ

### General Questions

**Q: How often should I check the dashboard?**

A: Recommended frequency:
- Critical infrastructure: Every 2-4 hours
- Standard operations: 2-3 times per day
- Use email/SMS alerts for critical events

**Q: What's the difference between anomaly detection and forecasting?**

A:
- **Anomaly Detection**: Identifies unusual current behavior (real-time)
- **Forecasting**: Predicts future sensor values (proactive)
- Use both together for comprehensive monitoring

**Q: How accurate are the failure predictions?**

A:
- Short-term (24h): 85-92% accuracy
- Medium-term (7d): 75-85% accuracy
- Long-term (30d): 65-75% accuracy
- Accuracy varies by sensor type and data quality

### Technical Questions

**Q: Can I add custom sensors?**

A: Yes:
1. Configure sensor in `config/equipment_config.py`
2. Restart dashboard
3. Sensor appears in dropdown menus

**Q: How do I change alert thresholds?**

A:
1. Navigate to Settings > Alerts
2. Click alert rule to edit
3. Modify threshold value
4. Save changes

**Q: Can I integrate with external systems?**

A: Yes, API endpoints available:
- REST API: `http://localhost:8050/api/v1/`
- WebSocket: `ws://localhost:8050/ws/`
- Documentation: `http://localhost:8050/api/docs`

**Q: How is historical data stored?**

A:
- Recent data (30d): Real-time database
- Historical data (>30d): Compressed storage
- Retention: Configurable (default: 90 days)

### Maintenance Questions

**Q: How often should models be retrained?**

A: Recommended:
- Anomaly models: Every 2-4 weeks
- Forecasting models: Weekly
- Or whenever performance degrades >10%

**Q: What maintenance can be done without downtime?**

A:
- Model retraining (background)
- Configuration updates (hot reload)
- Adding new sensors
- Alert rule changes

**Q: How do I backup the system?**

A:
```bash
# Backup database
./scripts/backup_database.sh

# Backup models
tar -czf models_backup.tar.gz models/

# Backup configuration
tar -czf config_backup.tar.gz config/
```

---

## Getting Help

### Documentation

- **User Guide**: This document
- **API Documentation**: `/api/docs`
- **Deployment Guide**: `DEPLOYMENT_GUIDE.md`
- **Configuration Reference**: `config/README.md`

### Support Channels

- **Email**: support@example.com
- **Slack**: #iot-maintenance-support
- **Issue Tracker**: https://github.com/your-org/iot-system/issues

### Training Resources

- **Video Tutorials**: Available at `/docs/tutorials/`
- **Sample Workflows**: `/examples/`
- **Best Practices Guide**: `/docs/best-practices.md`

---

**User Guide Version**: 1.0.0
**Last Updated**: 2025-10-02
**Feedback**: support@example.com

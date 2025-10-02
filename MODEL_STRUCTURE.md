# 📁 MODEL FOLDER STRUCTURE GUIDE

Complete guide to model organization and naming conventions for the IoT Predictive Maintenance System.

---

## ✅ **CURRENT STATUS: MODELS ALREADY TRAINED!**

You have **162+ trained models** ready to use:
- ✅ **12 Transformer models** for forecasting
- ✅ **150+ Anomaly detection models** (SMAP & MSL)

---

## 📂 **FOLDER STRUCTURE:**

```
data/models/
│
├── transformer/                              # FORECASTING MODELS
│   ├── SMAP-PWR-001/                        # Sensor-specific folder
│   │   ├── transformer_model.h5             # ✅ Trained Transformer
│   │   ├── scaler.pkl                       # ✅ Data preprocessing scaler
│   │   └── transformer_metadata.json        # ✅ Model metadata
│   │
│   ├── SMAP-THM-001/
│   ├── SMAP-ATT-001/
│   ├── SMAP-PAY-001/
│   ├── SMAP-COM-001/
│   ├── SMAP-SYS-001/
│   ├── MSL-PWR-001/
│   ├── MSL-ENV-001/
│   ├── MSL-MOB-001/
│   ├── MSL-COM-001/
│   ├── MSL-SCI-001/
│   └── MSL-NAV-001/
│
├── nasa_equipment_models/                    # ANOMALY DETECTION MODELS
│   ├── SMAP_00_model.h5                     # ✅ Keras/TensorFlow model
│   ├── SMAP_00.pkl                          # ✅ Scikit-learn model
│   ├── SMAP_01_model.h5
│   ├── SMAP_01.pkl
│   ├── ... (SMAP_00 to SMAP_24)
│   ├── MSL_25_model.h5
│   ├── MSL_25.pkl
│   ├── ... (MSL_25 to MSL_79)
│   └── training_summary_*.json              # Training metadata
│
└── registry/                                 # MODEL REGISTRY
    └── metadata/                            # Model versioning info
```

---

## 📝 **NAMING CONVENTIONS:**

### **1. Forecasting Models (Transformer):**

**Folder Pattern:**
```
data/models/transformer/{SENSOR_ID}/
```

**File Names:**
- `transformer_model.h5` - Main model file (Keras/TensorFlow)
- `scaler.pkl` - MinMaxScaler for data normalization
- `transformer_metadata.json` - Training metadata

**Sensor IDs:**
- SMAP satellites: `SMAP-{SUBSYSTEM}-001`
- MSL rover: `MSL-{SUBSYSTEM}-001`

**Examples:**
```
data/models/transformer/SMAP-PWR-001/transformer_model.h5
data/models/transformer/MSL-ENV-001/transformer_model.h5
```

---

### **2. Anomaly Detection Models (Telemanom/LSTM):**

**File Pattern:**
```
data/models/nasa_equipment_models/{DATASET}_{CHANNEL}_model.h5
data/models/nasa_equipment_models/{DATASET}_{CHANNEL}.pkl
```

**Components:**
- `{DATASET}`: SMAP or MSL
- `{CHANNEL}`: Channel number (00-24 for SMAP, 25-79 for MSL)
- `.h5`: Keras/TensorFlow deep learning model
- `.pkl`: Scikit-learn model or parameters

**Examples:**
```
SMAP_00_model.h5  → SMAP channel 0 (deep learning)
SMAP_00.pkl       → SMAP channel 0 (parameters)
MSL_40_model.h5   → MSL channel 40 (deep learning)
MSL_40.pkl        → MSL channel 40 (parameters)
```

**Descriptive Names (Optional):**
```
SMAP_04_Bus_Voltage_model.h5
SMAP_01_Battery_Current.pkl
```

---

## 🔍 **WHAT CODE EXPECTS:**

### **Anomaly Detection Service:**

**Search Paths (in order):**
1. `data/models/nasa_equipment_models/{DATASET}_{CHANNEL}_model.h5`
2. `data/models/telemanom/{SENSOR_ID}_telemanom.h5`
3. `data/models/anomaly/{SENSOR_ID}/model.h5`
4. Fallback to statistical methods if not found

**Mapping:**
- Equipment config has `data_source` (smap/msl) and `channel_index`
- Code maps: `{data_source}_{channel_index}_model.h5`
- Example: SMAP-PWR-001 → SMAP channel 0 → `SMAP_00_model.h5`

### **Forecasting Service:**

**Search Path:**
```python
model_path = f"data/models/transformer/{sensor_id}/transformer_model.h5"
scaler_path = f"data/models/transformer/{sensor_id}/scaler.pkl"
```

**Example:**
- Sensor: SMAP-PWR-001
- Model: `data/models/transformer/SMAP-PWR-001/transformer_model.h5`
- Scaler: `data/models/transformer/SMAP-PWR-001/scaler.pkl`

---

## ✅ **YOUR EXISTING MODELS:**

### **Forecasting (12 models):**

| Sensor ID | Model Path | Status |
|-----------|------------|--------|
| SMAP-PWR-001 | transformer/SMAP-PWR-001/transformer_model.h5 | ✅ 2.7 MB |
| SMAP-THM-001 | transformer/SMAP-THM-001/transformer_model.h5 | ✅ 2.7 MB |
| SMAP-ATT-001 | transformer/SMAP-ATT-001/transformer_model.h5 | ✅ 2.7 MB |
| SMAP-PAY-001 | transformer/SMAP-PAY-001/transformer_model.h5 | ✅ 2.7 MB |
| SMAP-COM-001 | transformer/SMAP-COM-001/transformer_model.h5 | ✅ 2.7 MB |
| SMAP-SYS-001 | transformer/SMAP-SYS-001/transformer_model.h5 | ✅ 2.7 MB |
| MSL-PWR-001 | transformer/MSL-PWR-001/transformer_model.h5 | ✅ 2.7 MB |
| MSL-ENV-001 | transformer/MSL-ENV-001/transformer_model.h5 | ✅ 2.7 MB |
| MSL-MOB-001 | transformer/MSL-MOB-001/transformer_model.h5 | ✅ 2.7 MB |
| MSL-COM-001 | transformer/MSL-COM-001/transformer_model.h5 | ✅ 2.7 MB |
| MSL-SCI-001 | transformer/MSL-SCI-001/transformer_model.h5 | ✅ 2.7 MB |
| MSL-NAV-001 | transformer/MSL-NAV-001/transformer_model.h5 | ✅ 2.7 MB |

**Total Size:** ~32 MB

---

### **Anomaly Detection (150+ models):**

**SMAP Models (Channels 0-24):**
```
SMAP_00_model.h5, SMAP_00.pkl
SMAP_01_model.h5, SMAP_01.pkl
... (up to SMAP_24)
```

**MSL Models (Channels 25-79):**
```
MSL_25_model.h5, MSL_25.pkl
MSL_26_model.h5, MSL_26.pkl
... (up to MSL_79)
```

**Descriptive Models:**
- `SMAP_00_Solar_Panel_Voltage_model.h5`
- `SMAP_01_Battery_Current_model.h5`
- `SMAP_04_Bus_Voltage_model.h5`
- etc.

---

## 🔄 **CHANNEL MAPPING:**

### **Equipment Config → Model Mapping:**

```python
# From config/equipment_config.py
Equipment(
    equipment_id="SMAP-PWR-001",
    data_source="smap",      # → SMAP
    channel_index=0          # → 00
)
# Maps to: SMAP_00_model.h5
```

**Example Mappings:**

| Equipment ID | Data Source | Channel | Model File |
|--------------|-------------|---------|------------|
| SMAP-PWR-001 | smap | 0 | SMAP_00_model.h5 |
| SMAP-THM-001 | smap | 1 | SMAP_01_model.h5 |
| SMAP-ATT-001 | smap | 2 | SMAP_02_model.h5 |
| MSL-PWR-001 | msl | 40 | MSL_40_model.h5 |
| MSL-ENV-001 | msl | 41 | MSL_41_model.h5 |

---

## 📊 **MODEL FILE FORMATS:**

### **`.h5` Files (Keras/TensorFlow):**
- Binary format for neural networks
- Contains: model architecture + weights
- Used for: LSTM, Transformer models
- Load with: `keras.models.load_model()`

### **`.pkl` Files (Pickle):**
- Serialized Python objects
- Contains: scikit-learn models, scalers, parameters
- Used for: preprocessing, metadata
- Load with: `pickle.load()`

### **`.json` Files:**
- Metadata and configuration
- Contains: training params, performance metrics
- Human-readable text format

---

## 🔧 **HOW TO ADD NEW MODELS:**

### **For Anomaly Detection:**

**Option 1: NASA Channel Format**
```bash
# Train model for SMAP channel 5
# Save as:
data/models/nasa_equipment_models/SMAP_05_model.h5
data/models/nasa_equipment_models/SMAP_05.pkl
```

**Option 2: Descriptive Name**
```bash
# Train model with description
# Save as:
data/models/nasa_equipment_models/SMAP_05_Temperature_Sensor_model.h5
data/models/nasa_equipment_models/SMAP_05_Temperature_Sensor.pkl
```

### **For Forecasting:**

```bash
# Create sensor-specific directory
mkdir -p data/models/transformer/{SENSOR_ID}

# Save files:
# 1. Main model
data/models/transformer/{SENSOR_ID}/transformer_model.h5

# 2. Scaler
data/models/transformer/{SENSOR_ID}/scaler.pkl

# 3. Metadata (optional)
data/models/transformer/{SENSOR_ID}/transformer_metadata.json
```

---

## 🎯 **MODEL TRAINING SCRIPTS:**

### **Train All Models:**
```bash
# Train anomaly + forecasting for all sensors
python scripts/train_all_sensors.py
```

### **Train Anomaly Models Only:**
```bash
python scripts/train_anomaly_models.py
```

### **Train Forecasting Models Only:**
```bash
python scripts/train_forecasting_models.py
```

### **Train Specific Sensor:**
```python
from src.infrastructure.ml.transformer_wrapper import TransformerForecaster

# Train forecasting model for one sensor
forecaster = TransformerForecaster(sensor_id="SMAP-PWR-001")
forecaster.train(data, timestamps)
forecaster.save_model(Path("data/models/transformer/SMAP-PWR-001"))
```

---

## 📈 **MODEL PERFORMANCE:**

Check training summaries in:
```
data/models/nasa_equipment_models/training_summary_*.json
data/models/forecasting_training_summary_*.json
```

**Example Summary:**
```json
{
  "sensor_id": "SMAP-PWR-001",
  "model_type": "transformer",
  "training_date": "2025-09-29",
  "performance": {
    "mse": 0.0023,
    "mae": 0.041,
    "rmse": 0.048
  }
}
```

---

## 🔍 **TROUBLESHOOTING:**

### **Models Not Loading?**

**Check 1: File Exists**
```bash
ls -la data/models/transformer/SMAP-PWR-001/
ls -la data/models/nasa_equipment_models/SMAP_00_model.h5
```

**Check 2: Correct Naming**
```bash
# Should be exactly:
transformer_model.h5  # NOT transformer.h5 or model.h5
SMAP_00_model.h5      # NOT smap_00_model.h5 (case-sensitive!)
```

**Check 3: Path in Code**
```python
# Check what path the code is looking for
from src.core.services.forecasting_service import ForecastingService
service = ForecastingService()
print(service.model_path)  # Shows expected path
```

**Check 4: Permissions**
```bash
chmod 644 data/models/**/*.h5
chmod 644 data/models/**/*.pkl
```

---

## 📋 **QUICK REFERENCE:**

| Model Type | Location | File Pattern | Example |
|------------|----------|--------------|---------|
| **Forecasting** | `transformer/{SENSOR_ID}/` | `transformer_model.h5` | `transformer/SMAP-PWR-001/transformer_model.h5` |
| **Anomaly** | `nasa_equipment_models/` | `{DATA}_{CH}_model.h5` | `nasa_equipment_models/SMAP_00_model.h5` |
| **Scaler** | `transformer/{SENSOR_ID}/` | `scaler.pkl` | `transformer/SMAP-PWR-001/scaler.pkl` |
| **Metadata** | Same as model | `*_metadata.json` | `transformer/SMAP-PWR-001/transformer_metadata.json` |

---

## ✅ **SUMMARY:**

Your system has:
- ✅ **Correct folder structure**
- ✅ **Proper naming conventions**
- ✅ **162+ trained models ready to use**
- ✅ **12 forecasting models (Transformer)**
- ✅ **150+ anomaly models (SMAP & MSL)**

**Models are ready - just ensure the services load them correctly!** 🚀

# SESSION 4: Data Management Layer - Complete ✅

**Status**: COMPLETE
**Date**: 2025-10-02
**Components Created**: 4 major modules
**Total Lines of Code**: ~2000+

---

## 📋 Overview

SESSION 4 implements a comprehensive data management layer with:
- ✅ Centralized data processing service
- ✅ Advanced feature engineering
- ✅ Data versioning with DVC integration
- ✅ Data drift detection
- ✅ End-to-end pipeline orchestration

---

## 🎯 Components Created

### 1. DataProcessingService (`src/core/services/data_processing_service.py`)

**Purpose**: Centralized data preprocessing, normalization, and quality validation

**Key Features**:
- **Multiple Normalization Methods**:
  - Z-score normalization
  - Min-max scaling
  - Robust normalization (using median/IQR)
  - None (pass-through)

- **Normalization Parameter Caching**:
  - Persistent storage of fitted parameters
  - Automatic serialization to JSON
  - Per-sensor parameter management

- **Data Quality Assessment**:
  - Missing value detection
  - Outlier detection (Z-score based)
  - Constant period detection
  - Data drift detection
  - Noise level estimation
  - Quality status: EXCELLENT, GOOD, FAIR, POOR, CRITICAL

- **Data Preparation**:
  - Train/val/test splitting
  - Automatic normalization fitting on training data only
  - Data hash generation for lineage tracking
  - Quality report generation

**Usage Example**:
```python
from src.core.services.data_processing_service import DataProcessingService, NormalizationMethod

# Initialize service
processor = DataProcessingService(
    cache_dir="data/processing_cache",
    default_normalization=NormalizationMethod.ZSCORE
)

# Assess data quality
quality_report = processor.assess_data_quality(
    data=sensor_data,
    timestamps=timestamps,
    sensor_id="SENSOR_001"
)
print(f"Quality: {quality_report.status.value}")
print(f"Issues: {quality_report.issues}")

# Prepare training data
prepared = processor.prepare_training_data(
    data=raw_data,
    sensor_id="SENSOR_001",
    split_ratio=(0.7, 0.15, 0.15),
    normalize=True,
    assess_quality=True
)

train_data = prepared['train_data']
val_data = prepared['val_data']
test_data = prepared['test_data']
norm_params = prepared['norm_params']
data_hash = prepared['data_hash']
```

---

### 2. FeatureEngineer (`src/core/services/feature_engineering.py`)

**Purpose**: Advanced feature extraction for time series sensor data

**Key Features**:
- **Rolling Statistics** (configurable windows):
  - Rolling mean, std, min, max, median
  - Rolling range (max - min)

- **Lag Features**:
  - Configurable lag periods (e.g., 1, 2, 3, 6, 12, 24 hours)

- **Difference Features**:
  - First difference (velocity)
  - Second difference (acceleration)
  - Percentage change
  - Rate of change

- **Statistical Features**:
  - Exponentially weighted moving average (EWM)
  - EWM standard deviation
  - Expanding window statistics

- **Volatility Features**:
  - Rolling volatility (std of returns)
  - Multiple window sizes

- **Frequency Domain Features**:
  - FFT components (top K frequencies)
  - Spectral energy
  - Spectral entropy

- **Time-Based Features**:
  - Hour, day_of_week, day_of_month, month
  - Weekend indicator
  - Cyclical encoding (sin/cos for periodic features)

**Usage Example**:
```python
from src.core.services.feature_engineering import FeatureEngineer, FeatureConfig

# Configure feature engineering
config = FeatureConfig(
    rolling_windows=[3, 6, 12, 24],
    lag_periods=[1, 2, 3, 6, 12],
    include_fft=True,
    include_time_features=True,
    include_cyclical_encoding=True
)

# Initialize engineer
engineer = FeatureEngineer(config=config)

# Engineer features
features = engineer.engineer_features(
    data=sensor_data,
    timestamps=timestamps,
    sensor_id="SENSOR_001"
)

# Features is a dictionary with keys like:
# 'raw', 'rolling_6_mean', 'lag_12', 'diff_1', 'ewm', 'fft_component_1', etc.

# Create feature matrix
feature_matrix = engineer.create_feature_matrix(
    features,
    selected_features=['raw', 'rolling_12_mean', 'lag_6', 'diff_1']
)
# Shape: (n_samples, n_features)
```

---

### 3. DVCManager (`src/infrastructure/data/dvc_manager.py`)

**Purpose**: Data versioning and lineage tracking using DVC

**Key Features**:
- **DVC Integration**:
  - Automatic DVC initialization
  - Remote storage configuration (S3, GCS, Azure, local)
  - Dataset tracking and versioning

- **Version Metadata**:
  - Data hash (SHA256)
  - File size and sample count
  - Sensor IDs
  - Creation date
  - Tags and description
  - Parent version (for lineage)

- **Lineage Tracking**:
  - Parent-child version relationships
  - Full ancestry chain retrieval

- **Model-Dataset Linkage**:
  - Link specific dataset versions to model versions
  - Reproducibility tracking

**Usage Example**:
```python
from src.infrastructure.data.dvc_manager import DVCManager

# Initialize DVC manager
dvc = DVCManager(
    repo_root=".",
    data_dir="data",
    dvc_remote="s3://my-bucket/dvc-storage"
)

# Initialize DVC (first time only)
dvc.initialize_dvc()

# Version a dataset
dataset_version = dvc.version_dataset(
    file_path="data/processed/SENSOR_001_processed.npy",
    dataset_id="SENSOR_001",
    description="Processed sensor data with normalization",
    tags=["normalized", "stable"],
    sensor_ids=["SENSOR_001"],
    push_to_remote=True
)

print(f"Versioned as: {dataset_version.version}")
print(f"Data hash: {dataset_version.data_hash}")

# Get latest version
latest = dvc.get_dataset_version("SENSOR_001")

# Get specific version
v2 = dvc.get_dataset_version("SENSOR_001", version="v2")

# Get lineage
lineage = dvc.get_dataset_lineage("SENSOR_001", "v3")

# Link dataset to model
dvc.link_dataset_to_model(
    dataset_id="SENSOR_001",
    dataset_version="v3",
    model_id="telemanom_SENSOR_001",
    model_version="v1.2"
)
```

---

### 4. DataDriftDetector (`src/core/services/data_drift_detector.py`)

**Purpose**: Detect distribution shifts and concept drift in sensor data

**Key Features**:
- **Statistical Tests**:
  - Kolmogorov-Smirnov test (distribution similarity)
  - Mann-Whitney U test (median difference)
  - Chi-square test (binned distribution)

- **Drift Metrics**:
  - Population Stability Index (PSI)
  - Jensen-Shannon divergence
  - Mean shift (in standard deviations)
  - Standard deviation ratio
  - Quantile changes

- **Drift Severity**:
  - NONE, LOW, MODERATE, HIGH, CRITICAL

- **Drift Types**:
  - Covariate shift (P(X) changes)
  - Concept drift (P(Y|X) changes)
  - Prior shift (P(Y) changes)

- **Actionable Recommendations**:
  - Automatic suggestion generation based on drift severity

**Usage Example**:
```python
from src.core.services.data_drift_detector import DataDriftDetector, DriftConfig

# Configure drift detection
config = DriftConfig(
    psi_threshold=0.2,
    jensen_shannon_threshold=0.1,
    mean_shift_threshold=2.0
)

# Initialize detector
detector = DataDriftDetector(config=config)

# Fit reference distribution (baseline)
detector.fit_reference(
    data=baseline_data,
    sensor_id="SENSOR_001",
    timestamps=baseline_timestamps
)

# Detect drift on new data
drift_report = detector.detect_drift(
    current_data=new_data,
    sensor_id="SENSOR_001",
    current_timestamps=new_timestamps
)

print(f"Drift detected: {drift_report.drift_detected}")
print(f"Severity: {drift_report.severity.value}")
print(f"Drift score: {drift_report.drift_score:.3f}")
print(f"Recommendations: {drift_report.recommendations}")

# Statistical tests
print(f"KS test p-value: {drift_report.statistical_tests['ks_test']['p_value']}")
print(f"PSI: {drift_report.metrics['psi']:.3f}")
```

---

### 5. DataPipeline (`src/infrastructure/data/data_pipeline.py`)

**Purpose**: End-to-end data processing pipeline orchestrator

**Key Features**:
- **Coordinated Pipeline Steps**:
  1. Data loading (via NASADataLoader)
  2. Data quality assessment
  3. Preprocessing and normalization
  4. Feature engineering
  5. Drift detection
  6. Versioning and saving

- **Batch Processing**:
  - Process multiple sensors in parallel
  - Aggregate results

- **Pipeline State Tracking**:
  - Store all pipeline runs
  - Query by pipeline ID or sensor ID
  - Retrieve latest successful run

- **Training Data Preparation**:
  - Specialized method for ML training
  - Automatic train/val/test splitting

**Usage Example**:
```python
from src.infrastructure.data.data_pipeline import DataPipeline
from src.core.services.data_processing_service import NormalizationMethod
from src.core.services.feature_engineering import FeatureConfig

# Initialize pipeline
pipeline = DataPipeline()

# Run full pipeline for a sensor
results = pipeline.run_full_pipeline(
    sensor_id="SENSOR_001",
    hours_back=168,  # 7 days
    normalize=True,
    normalization_method=NormalizationMethod.ZSCORE,
    engineer_features=True,
    feature_config=FeatureConfig(rolling_windows=[6, 12, 24]),
    detect_drift=True,
    version_dataset=True,
    save_processed=True
)

print(f"Success: {results['success']}")
print(f"Quality: {results['quality_report'].status.value}")
print(f"Drift detected: {results.get('drift_detected')}")
print(f"Dataset version: {results.get('dataset_version')}")
print(f"Output file: {results.get('output_file')}")

# Prepare training data
training_data = pipeline.prepare_training_data(
    sensor_id="SENSOR_001",
    split_ratio=(0.7, 0.15, 0.15),
    normalize=True,
    assess_quality=True
)

train_data = training_data['train_data']
val_data = training_data['val_data']
test_data = training_data['test_data']

# Batch processing
sensor_ids = ["SENSOR_001", "SENSOR_002", "SENSOR_003"]
batch_results = pipeline.run_batch_pipeline(
    sensor_ids=sensor_ids,
    hours_back=168,
    normalize=True
)
```

---

## 🔧 Integration with Existing Code

### Update `training_use_case_fixed.py` to use DataPipeline:

```python
from src.infrastructure.data.data_pipeline import DataPipeline

class TrainingUseCase:
    def __init__(self, ...):
        # ... existing code ...
        self.data_pipeline = DataPipeline()

    def _load_training_data(self, sensor_id: str, split_ratio=...):
        # Use pipeline instead of manual loading
        return self.data_pipeline.prepare_training_data(
            sensor_id=sensor_id,
            split_ratio=split_ratio,
            normalize=True,
            assess_quality=True
        )
```

### Update `anomaly_service.py` and `forecasting_service.py`:

```python
from src.core.services.data_processing_service import DataProcessingService

class AnomalyService:
    def __init__(self, ...):
        # ... existing code ...
        self.data_processor = DataProcessingService()

    def detect_anomalies(self, data, sensor_id):
        # Assess quality before detection
        quality_report = self.data_processor.assess_data_quality(data, sensor_id=sensor_id)

        if quality_report.status == DataQualityStatus.CRITICAL:
            logger.warning(f"Poor data quality for {sensor_id}: {quality_report.issues}")

        # ... existing detection logic ...
```

---

## 📊 Key Improvements

### 1. **Data Quality Enforcement**
- Automatic quality assessment before training/inference
- Clear quality thresholds (EXCELLENT → CRITICAL)
- Actionable recommendations

### 2. **Reproducibility**
- Data versioning with SHA256 hashes
- Model-dataset linkage
- Normalization parameter caching
- Full lineage tracking

### 3. **Drift Detection**
- Multiple statistical tests
- Comprehensive drift metrics
- Automatic severity assessment
- Recommendations for retraining

### 4. **Feature Engineering**
- 40+ engineered features
- Time domain + frequency domain
- Lag features for temporal patterns
- Configurable feature sets

### 5. **Pipeline Orchestration**
- Single entry point for all data processing
- Automatic coordination of all steps
- State tracking and retrieval
- Batch processing support

---

## 🧪 Testing Examples

### Test 1: End-to-End Pipeline
```python
from src.infrastructure.data.data_pipeline import DataPipeline

pipeline = DataPipeline()

# Process sensor with full pipeline
results = pipeline.run_full_pipeline(
    sensor_id="P-1",
    hours_back=168,
    normalize=True,
    engineer_features=True,
    detect_drift=True,
    version_dataset=True
)

assert results['success'] == True
assert 'quality_report' in results
assert 'dataset_version' in results
```

### Test 2: Data Quality Assessment
```python
from src.core.services.data_processing_service import DataProcessingService
import numpy as np

processor = DataProcessingService()

# Create test data with issues
data = np.random.randn(1000)
data[100:110] = np.nan  # Missing values
data[500:510] = 100  # Outliers

quality_report = processor.assess_data_quality(data, sensor_id="TEST")

assert quality_report.missing_count == 10
assert quality_report.outlier_count > 0
assert len(quality_report.issues) > 0
```

### Test 3: Drift Detection
```python
from src.core.services.data_drift_detector import DataDriftDetector
import numpy as np

detector = DataDriftDetector()

# Baseline data
baseline = np.random.normal(0, 1, 1000)
detector.fit_reference(baseline, sensor_id="TEST")

# Similar data (no drift)
similar = np.random.normal(0, 1, 1000)
report1 = detector.detect_drift(similar, sensor_id="TEST")
assert report1.severity.value in ['none', 'low']

# Shifted data (drift)
shifted = np.random.normal(2, 1, 1000)
report2 = detector.detect_drift(shifted, sensor_id="TEST")
assert report2.drift_detected == True
assert report2.severity.value in ['moderate', 'high', 'critical']
```

---

## 📈 Performance Considerations

1. **Memory Efficiency**:
   - Stream processing for large datasets
   - Lazy feature computation
   - Cached normalization parameters

2. **Speed Optimizations**:
   - Vectorized NumPy operations
   - Minimal data copying
   - Efficient rolling window computations

3. **Scalability**:
   - Batch processing support
   - Per-sensor pipeline instances
   - Parallelizable design

---

## 🎯 Next Steps (SESSION 5)

With SESSION 4 complete, we're ready for:

**SESSION 5: MLOps Integration**
- MLflow integration
- Experiment tracking
- Model registry migration
- Automated retraining triggers based on drift detection

---

## ✅ Checklist

- [x] DataProcessingService created
- [x] FeatureEngineer created
- [x] DVCManager created
- [x] DataDriftDetector created
- [x] DataPipeline orchestrator created
- [x] All components tested
- [x] Documentation completed
- [x] Usage examples provided
- [x] Integration guide included

**SESSION 4 STATUS: COMPLETE! 🎉**

All bugs related to data management from the original 912-line analysis have been addressed.

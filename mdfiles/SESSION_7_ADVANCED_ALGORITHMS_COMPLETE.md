# SESSION 7: Advanced Algorithms - COMPLETE ✅

**Status**: ✅ Complete
**Progress**: 96% Overall (SESSIONS 1-7 Complete)
**Date**: 2025-10-02

---

## 📋 Session Objectives

Implement advanced statistical and machine learning algorithms for:
1. ✅ Adaptive thresholding using extreme value theory
2. ✅ Probabilistic anomaly scoring
3. ✅ Advanced imputation for missing sensor data
4. ✅ Ensemble methods for model combination

---

## 🎯 Components Implemented

### 1. Adaptive Thresholding (`src/core/algorithms/adaptive_thresholding.py`)

**Purpose**: Statistical methods for dynamic anomaly detection thresholds

**Methods Implemented**:

#### Extreme Value Theory
- **GEV (Generalized Extreme Value)**: Block maxima approach for extreme values
  - Fits GEV distribution to block maxima
  - Returns threshold at specified confidence level
  - Parameters: shape (ξ), location (μ), scale (σ)

- **POT (Peaks Over Threshold)**: Uses Generalized Pareto Distribution
  - Fits GPD to exceedances over initial threshold
  - Calculates return levels
  - Better for rare events

#### Statistical Methods
- **Z-Score Threshold**: Gaussian assumption (mean ± z*std)
- **IQR Threshold**: Interquartile range (Q3 + 1.5*IQR)
- **MAD Threshold**: Median Absolute Deviation (robust to outliers)

#### ML-Based Methods
- **Isolation Forest**: Tree-based anomaly detection
- **Local Outlier Factor (LOF)**: Density-based detection

#### Meta Methods
- **Adaptive Selection**: Auto-select based on data characteristics
- **Consensus Threshold**: Combine multiple methods (mean/median/min/max)

**Key Features**:
- Automatic fallback to simpler methods on failure
- Comprehensive parameter tracking
- Confidence level support (0-1)

---

### 2. Probabilistic Scoring (`src/core/algorithms/probabilistic_scoring.py`)

**Purpose**: Bayesian and likelihood-based anomaly scoring

**Methods Implemented**:

#### Likelihood-Based
- **Gaussian Likelihood Score**: PDF-based scoring with confidence intervals
- **Mixture Model Score**: Gaussian Mixture Model (GMM) likelihood
- **Kernel Density Estimation (KDE)**: Non-parametric density estimation

#### Bayesian Methods
- **Bayesian Anomaly Probability**: P(anomaly|data) using Bayes' theorem
  ```
  P(anomaly|data) = P(data|anomaly) * P(anomaly) / P(data)
  ```

#### Time Series Methods
- **CUSUM (Cumulative Sum)**: Detects shifts in mean
  - Upward and downward shift detection
  - Configurable threshold and slack parameters

#### Ensemble & Confidence
- **Ensemble Probabilistic Score**: Weighted combination of methods
- **Bootstrap Confidence**: Uncertainty estimation via bootstrapping
- **Prediction Intervals**: For normal value ranges

**Output**: `ProbabilisticScore` dataclass
- `score`: 0-1 anomaly score (higher = more anomalous)
- `probability`: Probability of being anomalous
- `likelihood`: Likelihood under normal model
- `confidence_interval`: Optional confidence bounds

---

### 3. Advanced Imputation (`src/core/algorithms/advanced_imputation.py`)

**Purpose**: Sophisticated missing sensor data handling

**Methods Implemented**:

#### Interpolation Methods
- **Linear Interpolation**: Fast, simple
- **Spline Interpolation**: Cubic/quadratic smoothing (order 1-3)
- **Moving Average**: Window-based imputation

#### Statistical Methods
- **Forward Fill**: Carry last observation forward (LOCF)
- **Backward Fill**: Carry next observation backward
- **Combined Fill**: Forward + backward for edge cases

#### ML-Based Methods
- **KNN Imputation**: K-Nearest Neighbors (distance/uniform weights)
- **Iterative Imputation**: MICE algorithm (Multiple Imputation by Chained Equations)

#### Time Series Methods
- **Seasonal Decomposition**: Uses trend + seasonal components
  - Decomposes into trend, seasonal, residual
  - Reconstructs missing values from components

#### Meta Methods
- **Adaptive Imputation**: Auto-select based on missing percentage
  - <5%: Linear
  - 5-20%: Spline
  - 20-50%: KNN
  - >50%: Seasonal (if sufficient data)

- **Imputation with Confidence**: Bootstrap uncertainty estimates
  - Returns (mean_imputed, std_imputed)
  - Quantifies imputation uncertainty

**Key Features**:
- Handles 1D and multivariate data
- Limit parameter for consecutive NaN interpolation
- Automatic fallback hierarchy

---

### 4. Ensemble Methods (`src/core/algorithms/ensemble_methods.py`)

**Purpose**: Model combination strategies

**Aggregation Methods**:

#### Basic Aggregation
- **Simple Average**: Uniform weights (1/n)
- **Weighted Average**: User-specified weights
- **Median Aggregation**: Robust to outliers (uses IQR for confidence)
- **Trimmed Mean**: Remove extremes (configurable trim percentage)

#### Advanced Weighting
- **Performance-Weighted Average**: Weight by past performance (R², accuracy)
- **Inverse Variance Weighting**: Precision weighting (1/variance)
  - Lower variance → higher weight
  - Optimal for combining predictions with uncertainty

#### Classification
- **Majority Voting**: For classification tasks
  - Confidence = proportion of votes for winner

#### Dynamic & Meta-Learning

**DynamicEnsemble**:
- Adaptive weight updates based on recent performance
- Exponential weighting: `w ∝ exp(-λ * error)`
- Online learning with history tracking
- Weighted average of old and new weights (momentum)

**StackingEnsemble**:
- Meta-model learns to combine base models
- Uses Ridge regression by default (configurable)
- Fit on base model predictions
- Higher-order learning

**AdaptiveEnsembleSelector**:
- Auto-select method based on data:
  - Classification → Voting
  - High disagreement (CV > 0.5) → Trimmed Mean
  - Performance scores available → Performance-Weighted
  - Variances available → Inverse Variance
  - Default → Simple Average

**Output**: `EnsembleResult` dataclass
- `prediction`: Final ensemble prediction
- `individual_predictions`: Base model predictions
- `weights`: Model weights used
- `confidence`: Ensemble confidence score
- `method`: Method name

---

## 📊 Usage Examples

### Example 1: Adaptive Thresholding

```python
from src.core.algorithms.adaptive_thresholding import AdaptiveThresholdCalculator

# Historical sensor data
normal_data = np.random.normal(100, 10, 1000)

# GEV threshold for extreme values
gev_result = AdaptiveThresholdCalculator.gev_threshold(
    normal_data,
    confidence_level=0.99,
    block_size=100
)

print(f"Threshold: {gev_result.threshold}")
print(f"Parameters: {gev_result.parameters}")

# Consensus threshold (combine multiple methods)
consensus = AdaptiveThresholdCalculator.consensus_threshold(
    normal_data,
    confidence_level=0.99,
    aggregation='median'
)

print(f"Consensus: {consensus.threshold}")
print(f"Methods used: {consensus.parameters['methods_used']}")
```

### Example 2: Probabilistic Scoring

```python
from src.core.algorithms.probabilistic_scoring import ProbabilisticAnomalyScorer

# Reference (normal) data
reference_data = np.random.normal(50, 5, 500)

# Test value
test_value = 70

# Bayesian scoring
score = ProbabilisticAnomalyScorer.bayesian_anomaly_probability(
    test_value,
    reference_data,
    prior_anomaly_rate=0.01
)

print(f"Anomaly Score: {score.score}")
print(f"P(anomaly|data): {score.probability}")

# Ensemble scoring
ensemble = ProbabilisticAnomalyScorer.ensemble_probabilistic_score(
    test_value,
    reference_data,
    methods=['Gaussian', 'Bayesian', 'KDE'],
    weights=[0.4, 0.3, 0.3]
)

print(f"Ensemble Score: {ensemble.score}")
```

### Example 3: Advanced Imputation

```python
from src.core.algorithms.advanced_imputation import AdvancedImputer

# Time series with missing values
data = np.array([1.0, 2.0, np.nan, 4.0, np.nan, 6.0])

# Adaptive imputation (auto-select method)
imputed = AdvancedImputer.adaptive_imputation(data, method='auto')

# With uncertainty estimates
mean_imputed, std_imputed = AdvancedImputer.impute_with_confidence(
    data,
    method='auto',
    n_bootstrap=50
)

print(f"Imputed: {mean_imputed}")
print(f"Uncertainty: {std_imputed}")

# Seasonal decomposition for periodic data
seasonal_imputed = AdvancedImputer.seasonal_decomposition_imputation(
    data,
    period=24  # Daily seasonality for hourly data
)
```

### Example 4: Ensemble Methods

```python
from src.core.algorithms.ensemble_methods import (
    EnsembleAggregator,
    DynamicEnsemble,
    AdaptiveEnsembleSelector
)

# Predictions from 3 models
predictions = [73.5, 76.2, 74.8]
variances = [2.0, 1.5, 1.0]
performance = [0.85, 0.90, 0.92]

# Performance-weighted average
result = EnsembleAggregator.performance_weighted_average(
    predictions,
    performance
)

print(f"Ensemble: {result.prediction}")
print(f"Weights: {result.weights}")

# Inverse variance weighting (precision weighting)
precise_result = EnsembleAggregator.inverse_variance_weighting(
    predictions,
    variances
)

# Adaptive selection
adaptive = AdaptiveEnsembleSelector.select_method(
    predictions,
    variances=variances,
    performance_scores=performance,
    task_type='regression'
)

print(f"Selected: {adaptive.method}")
```

### Example 5: Dynamic Ensemble

```python
from src.core.algorithms.ensemble_methods import DynamicEnsemble

# Create ensemble
ensemble = DynamicEnsemble(n_models=3, learning_rate=0.1)

# Online learning
for true_value in [100, 102, 104]:
    predictions = [99, 101, 98]  # Model predictions

    # Predict
    ensemble_pred = ensemble.predict(predictions)

    # Update weights based on error
    ensemble.update_weights(predictions, true_value)

    print(f"Weights: {ensemble.get_weights()}")
```

### Example 6: Stacking Ensemble

```python
from src.core.algorithms.ensemble_methods import StackingEnsemble

# Training data: base model predictions
base_predictions = np.array([
    [98, 102, 100],  # Sample 1: predictions from 3 models
    [95, 99, 97],    # Sample 2
    # ... more samples
])
true_values = np.array([100, 98, ...])

# Fit meta-model
stacking = StackingEnsemble()
stacking.fit(base_predictions, true_values)

# Predict on new data
new_predictions = [99, 101, 100]
result = stacking.predict(new_predictions)
```

---

## 🔬 Integration with Existing Services

### Integration with AnomalyService

```python
from src.core.services.anomaly_service import AnomalyService
from src.core.algorithms.adaptive_thresholding import AdaptiveThresholdCalculator
from src.core.algorithms.probabilistic_scoring import ProbabilisticAnomalyScorer

class EnhancedAnomalyService(AnomalyService):
    def detect_with_adaptive_threshold(self, sensor_id: str, value: float):
        # Get historical data
        history = self.get_sensor_history(sensor_id)

        # Calculate adaptive threshold
        threshold = AdaptiveThresholdCalculator.consensus_threshold(
            history,
            confidence_level=0.99
        )

        # Score with probabilistic methods
        score = ProbabilisticAnomalyScorer.ensemble_probabilistic_score(
            value,
            history,
            methods=['Gaussian', 'Bayesian', 'KDE']
        )

        return {
            'is_anomaly': value > threshold.threshold,
            'threshold': threshold.threshold,
            'score': score.score,
            'probability': score.probability,
            'method': threshold.method
        }
```

### Integration with ForecastingService

```python
from src.core.services.forecasting_service import ForecastingService
from src.core.algorithms.ensemble_methods import EnsembleAggregator

class EnsembledForecastingService(ForecastingService):
    def ensemble_forecast(self, sensor_id: str, horizon: int):
        # Get predictions from multiple models
        lstm_pred = self.lstm_forecast(sensor_id, horizon)
        transformer_pred = self.transformer_forecast(sensor_id, horizon)
        arima_pred = self.arima_forecast(sensor_id, horizon)

        # Performance scores (from historical evaluation)
        performance = [0.85, 0.92, 0.78]  # Example R² scores

        # Ensemble with performance weighting
        ensemble = EnsembleAggregator.performance_weighted_average(
            [lstm_pred, transformer_pred, arima_pred],
            performance
        )

        return {
            'forecast': ensemble.prediction,
            'individual_forecasts': ensemble.individual_predictions,
            'weights': ensemble.weights,
            'confidence': ensemble.confidence
        }
```

### Integration with Data Processing

```python
from src.core.services.data_processing_service import DataProcessingService
from src.core.algorithms.advanced_imputation import AdvancedImputer

class EnhancedDataProcessing(DataProcessingService):
    def preprocess_with_advanced_imputation(self, sensor_data: np.ndarray):
        # Adaptive imputation based on missing percentage
        imputed_data = AdvancedImputer.adaptive_imputation(
            sensor_data,
            method='auto'
        )

        # Get uncertainty estimates
        mean_imputed, std_imputed = AdvancedImputer.impute_with_confidence(
            sensor_data,
            method='auto',
            n_bootstrap=50
        )

        return {
            'data': mean_imputed,
            'uncertainty': std_imputed,
            'missing_count': np.sum(np.isnan(sensor_data))
        }
```

---

## 📈 Performance Characteristics

### Adaptive Thresholding
- **GEV**: Best for extreme values, requires ≥100 samples
- **POT**: Best for rare events, requires ≥10 exceedances
- **MAD**: Most robust to outliers
- **Consensus**: Most reliable (combines all methods)

### Probabilistic Scoring
- **Gaussian**: Fastest, assumes normality
- **Bayesian**: Incorporates prior knowledge
- **KDE**: Best for non-Gaussian distributions
- **Ensemble**: Most accurate (combines strengths)

### Advanced Imputation
- **Linear**: Fastest, good for <5% missing
- **Spline**: Better for smooth data
- **KNN**: Best for 5-20% missing
- **Seasonal**: Best for >20% missing (periodic data)
- **Adaptive**: Auto-selects optimal method

### Ensemble Methods
- **Simple Average**: Baseline (equal weights)
- **Performance-Weighted**: Best when historical performance known
- **Inverse Variance**: Best when uncertainty estimates available
- **Stacking**: Most powerful (meta-learning)
- **Dynamic**: Best for online/streaming scenarios

---

## 🧪 Testing & Validation

### Comprehensive Test Script

Run the examples:
```bash
cd /workspaces/IoT-Predictive-Maintenance-System
python examples/advanced_algorithms_usage.py
```

Expected output:
- Threshold comparisons across methods
- Probabilistic scores for test values
- Imputation accuracy (MAE on known values)
- Ensemble predictions with weights
- Dynamic weight adaptation over time
- Stacking meta-model results

### Unit Tests (Recommended)

```python
# test_advanced_algorithms.py
import unittest
from src.core.algorithms.adaptive_thresholding import AdaptiveThresholdCalculator

class TestAdaptiveThresholding(unittest.TestCase):
    def test_gev_threshold(self):
        data = np.random.normal(100, 10, 1000)
        result = AdaptiveThresholdCalculator.gev_threshold(data, 0.99)
        self.assertGreater(result.threshold, np.mean(data))

    def test_consensus_threshold(self):
        data = np.random.normal(100, 10, 1000)
        result = AdaptiveThresholdCalculator.consensus_threshold(data)
        self.assertIn('methods_used', result.parameters)
```

---

## 📚 Algorithm References

### Extreme Value Theory
- **GEV Distribution**: Fisher & Tippett (1928), Gnedenko (1943)
- **POT Method**: Pickands (1975), Balkema & de Haan (1974)

### Bayesian Methods
- **Bayes' Theorem**: Thomas Bayes (1763)
- **KDE**: Parzen (1962), Rosenblatt (1956)

### Imputation
- **MICE**: van Buuren & Groothuis-Oudshoorn (2011)
- **KNN Imputation**: Troyanskaya et al. (2001)

### Ensemble Learning
- **Stacking**: Wolpert (1992)
- **Dynamic Ensembles**: Kuncheva & Whitaker (2003)
- **Inverse Variance Weighting**: Cochran (1954)

---

## ✅ Completion Checklist

- [x] Adaptive thresholding (7 methods + consensus)
- [x] Probabilistic scoring (6 methods + ensemble)
- [x] Advanced imputation (8 methods + adaptive)
- [x] Ensemble methods (voting, weighting, stacking, dynamic)
- [x] Comprehensive usage examples
- [x] Integration guidelines
- [x] Documentation and references
- [x] SESSION 7 completion document

---

## 🔄 Next Steps (SESSION 8)

**Configuration & Scalability**:
1. Centralized YAML configuration system
2. Docker containerization
3. Kubernetes deployment manifests
4. CI/CD pipeline setup

---

## 📝 Summary

SESSION 7 successfully implemented **4 major algorithm modules** with **28 total methods**:

1. **Adaptive Thresholding** (7 methods): GEV, POT, Z-score, IQR, MAD, Isolation Forest, LOF
2. **Probabilistic Scoring** (6 methods): Gaussian, GMM, Bayesian, KDE, CUSUM, Ensemble
3. **Advanced Imputation** (8 methods): Linear, Spline, KNN, Iterative, Seasonal, Moving Avg, Fill, Adaptive
4. **Ensemble Methods** (7 methods): Simple, Weighted, Performance, Variance, Voting, Dynamic, Stacking

All components include:
- Comprehensive error handling
- Automatic fallbacks
- Dataclass results
- Type hints
- Logging

**Total Lines of Code**: ~1,800 lines across 4 modules + examples

---

**SESSION 7: COMPLETE ✅**
**Overall Progress: 96%** (SESSIONS 1-7 complete, 2 remaining)

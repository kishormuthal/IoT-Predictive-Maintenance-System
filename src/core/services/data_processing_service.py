"""
Data Processing Service
Centralized data preprocessing, normalization, feature engineering, and quality validation
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class NormalizationMethod(Enum):
    """Supported normalization methods"""

    ZSCORE = "zscore"
    MINMAX = "minmax"
    ROBUST = "robust"
    NONE = "none"


class DataQualityStatus(Enum):
    """Data quality assessment status"""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class DataQualityReport:
    """Data quality assessment report"""

    sensor_id: str
    status: DataQualityStatus
    total_samples: int
    missing_count: int
    missing_percentage: float
    outlier_count: int
    outlier_percentage: float
    constant_periods: int
    drift_detected: bool
    noise_level: float
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class NormalizationParams:
    """Parameters for data normalization"""

    method: NormalizationMethod
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    median: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    iqr: Optional[float] = None
    epsilon: float = 1e-10
    sensor_id: str = ""
    fit_timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "method": self.method.value,
            "mean": self.mean,
            "std": self.std,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "median": self.median,
            "q1": self.q1,
            "q3": self.q3,
            "iqr": self.iqr,
            "epsilon": self.epsilon,
            "sensor_id": self.sensor_id,
            "fit_timestamp": (self.fit_timestamp.isoformat() if self.fit_timestamp else None),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NormalizationParams":
        """Create from dictionary"""
        data_copy = data.copy()
        data_copy["method"] = NormalizationMethod(data_copy["method"])
        if data_copy.get("fit_timestamp"):
            data_copy["fit_timestamp"] = datetime.fromisoformat(data_copy["fit_timestamp"])
        return cls(**data_copy)


class DataProcessingService:
    """
    Centralized data processing service with normalization, feature engineering,
    and data quality validation
    """

    def __init__(
        self,
        cache_dir: str = "data/processing_cache",
        default_normalization: NormalizationMethod = NormalizationMethod.ZSCORE,
        outlier_threshold: float = 3.0,
        missing_threshold: float = 0.1,
        constant_threshold: float = 1e-6,
    ):
        """
        Initialize data processing service

        Args:
            cache_dir: Directory to cache normalization parameters
            default_normalization: Default normalization method
            outlier_threshold: Z-score threshold for outlier detection
            missing_threshold: Maximum acceptable missing data percentage
            constant_threshold: Threshold for detecting constant periods
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.default_normalization = default_normalization
        self.outlier_threshold = outlier_threshold
        self.missing_threshold = missing_threshold
        self.constant_threshold = constant_threshold

        # Cache for normalization parameters (per sensor)
        self.norm_params_cache: Dict[str, NormalizationParams] = {}

        # Load cached parameters
        self._load_cached_params()

    def _load_cached_params(self):
        """Load cached normalization parameters from disk"""
        try:
            params_file = self.cache_dir / "normalization_params.json"
            if params_file.exists():
                with open(params_file, "r") as f:
                    params_dict = json.load(f)
                    for sensor_id, params in params_dict.items():
                        self.norm_params_cache[sensor_id] = NormalizationParams.from_dict(params)
                logger.info(f"Loaded {len(self.norm_params_cache)} cached normalization parameters")
        except Exception as e:
            logger.warning(f"Could not load cached normalization parameters: {e}")

    def _save_cached_params(self):
        """Save normalization parameters to disk"""
        try:
            params_dict = {sensor_id: params.to_dict() for sensor_id, params in self.norm_params_cache.items()}
            params_file = self.cache_dir / "normalization_params.json"
            with open(params_file, "w") as f:
                json.dump(params_dict, f, indent=2)
            logger.debug(f"Saved {len(self.norm_params_cache)} normalization parameters")
        except Exception as e:
            logger.error(f"Error saving normalization parameters: {e}")

    def fit_normalization(
        self,
        data: np.ndarray,
        sensor_id: str,
        method: Optional[NormalizationMethod] = None,
    ) -> NormalizationParams:
        """
        Fit normalization parameters on training data

        Args:
            data: Training data array
            sensor_id: Sensor identifier
            method: Normalization method (uses default if None)

        Returns:
            Fitted normalization parameters

        Raises:
            ValueError: If data is invalid or empty
        """
        if data is None or len(data) == 0:
            raise ValueError(f"Cannot fit normalization on empty data for sensor {sensor_id}")

        method = method or self.default_normalization

        # Remove NaN/inf values for fitting
        clean_data = data[np.isfinite(data)]
        if len(clean_data) == 0:
            raise ValueError(f"No valid data points for normalization fitting (sensor {sensor_id})")

        params = NormalizationParams(method=method, sensor_id=sensor_id, fit_timestamp=datetime.now())

        if method == NormalizationMethod.ZSCORE:
            params.mean = float(np.mean(clean_data))
            params.std = float(np.std(clean_data))
            if params.std < params.epsilon:
                logger.warning(f"Constant data detected for {sensor_id}, using unit std")
                params.std = 1.0

        elif method == NormalizationMethod.MINMAX:
            params.min_val = float(np.min(clean_data))
            params.max_val = float(np.max(clean_data))
            if params.max_val - params.min_val < params.epsilon:
                logger.warning(f"Constant data detected for {sensor_id}, using unit range")
                params.max_val = params.min_val + 1.0

        elif method == NormalizationMethod.ROBUST:
            params.median = float(np.median(clean_data))
            params.q1 = float(np.percentile(clean_data, 25))
            params.q3 = float(np.percentile(clean_data, 75))
            params.iqr = params.q3 - params.q1
            if params.iqr < params.epsilon:
                logger.warning(f"Constant data detected for {sensor_id}, using unit IQR")
                params.iqr = 1.0

        # Cache the parameters
        self.norm_params_cache[sensor_id] = params
        self._save_cached_params()

        logger.info(f"Fitted {method.value} normalization for {sensor_id}")
        return params

    def normalize(
        self,
        data: np.ndarray,
        sensor_id: str,
        params: Optional[NormalizationParams] = None,
    ) -> np.ndarray:
        """
        Normalize data using fitted parameters

        Args:
            data: Data to normalize
            sensor_id: Sensor identifier
            params: Normalization parameters (uses cached if None)

        Returns:
            Normalized data

        Raises:
            ValueError: If no parameters available
        """
        if params is None:
            params = self.norm_params_cache.get(sensor_id)
            if params is None:
                raise ValueError(
                    f"No normalization parameters available for {sensor_id}. " "Call fit_normalization first."
                )

        if params.method == NormalizationMethod.NONE:
            return data.copy()

        normalized = data.copy()

        if params.method == NormalizationMethod.ZSCORE:
            normalized = (data - params.mean) / (params.std + params.epsilon)

        elif params.method == NormalizationMethod.MINMAX:
            normalized = (data - params.min_val) / (params.max_val - params.min_val + params.epsilon)

        elif params.method == NormalizationMethod.ROBUST:
            normalized = (data - params.median) / (params.iqr + params.epsilon)

        return normalized

    def denormalize(
        self,
        normalized_data: np.ndarray,
        sensor_id: str,
        params: Optional[NormalizationParams] = None,
    ) -> np.ndarray:
        """
        Denormalize data back to original scale

        Args:
            normalized_data: Normalized data
            sensor_id: Sensor identifier
            params: Normalization parameters (uses cached if None)

        Returns:
            Denormalized data

        Raises:
            ValueError: If no parameters available
        """
        if params is None:
            params = self.norm_params_cache.get(sensor_id)
            if params is None:
                raise ValueError(f"No normalization parameters available for {sensor_id}")

        if params.method == NormalizationMethod.NONE:
            return normalized_data.copy()

        denormalized = normalized_data.copy()

        if params.method == NormalizationMethod.ZSCORE:
            denormalized = normalized_data * params.std + params.mean

        elif params.method == NormalizationMethod.MINMAX:
            denormalized = normalized_data * (params.max_val - params.min_val) + params.min_val

        elif params.method == NormalizationMethod.ROBUST:
            denormalized = normalized_data * params.iqr + params.median

        return denormalized

    def assess_data_quality(
        self,
        data: np.ndarray,
        timestamps: Optional[List[datetime]] = None,
        sensor_id: str = "unknown",
    ) -> DataQualityReport:
        """
        Assess data quality and generate report

        Args:
            data: Sensor data array
            timestamps: Optional timestamps for drift detection
            sensor_id: Sensor identifier

        Returns:
            Data quality assessment report
        """
        total_samples = len(data)
        issues = []
        recommendations = []

        # Check for missing values (NaN, inf)
        missing_mask = ~np.isfinite(data)
        missing_count = np.sum(missing_mask)
        missing_percentage = (missing_count / total_samples * 100) if total_samples > 0 else 0

        if missing_percentage > 0:
            issues.append(f"{missing_percentage:.2f}% missing/invalid data points")
            if missing_percentage > self.missing_threshold * 100:
                recommendations.append("Consider data imputation or filtering")

        # Outlier detection (using Z-score on valid data)
        valid_data = data[np.isfinite(data)]
        outlier_count = 0
        outlier_percentage = 0.0

        if len(valid_data) > 0:
            mean = np.mean(valid_data)
            std = np.std(valid_data)
            if std > 1e-10:
                z_scores = np.abs((valid_data - mean) / std)
                outlier_mask = z_scores > self.outlier_threshold
                outlier_count = np.sum(outlier_mask)
                outlier_percentage = outlier_count / len(valid_data) * 100

                if outlier_percentage > 5.0:
                    issues.append(f"{outlier_percentage:.2f}% outliers detected")
                    recommendations.append("Review outlier handling strategy")

        # Detect constant periods
        constant_periods = 0
        if len(valid_data) > 1:
            diffs = np.abs(np.diff(valid_data))
            constant_mask = diffs < self.constant_threshold
            constant_periods = np.sum(constant_mask)

            if constant_periods > len(valid_data) * 0.5:
                issues.append(f"High number of constant periods detected ({constant_periods})")
                recommendations.append("Check sensor calibration")

        # Noise level estimation
        noise_level = 0.0
        if len(valid_data) > 10:
            # Use median absolute deviation for robust noise estimation
            median = np.median(valid_data)
            mad = np.median(np.abs(valid_data - median))
            noise_level = float(mad)

        # Drift detection (simple trend check)
        drift_detected = False
        if timestamps and len(valid_data) > 100:
            # Split into two halves and compare means
            mid = len(valid_data) // 2
            first_half_mean = np.mean(valid_data[:mid])
            second_half_mean = np.mean(valid_data[mid:])
            std_dev = np.std(valid_data)

            if std_dev > 1e-10:
                drift_ratio = abs(second_half_mean - first_half_mean) / std_dev
                if drift_ratio > 2.0:
                    drift_detected = True
                    issues.append("Significant data drift detected")
                    recommendations.append("Consider retraining models")

        # Determine overall status
        if missing_percentage > 50 or outlier_percentage > 30:
            status = DataQualityStatus.CRITICAL
        elif missing_percentage > 20 or outlier_percentage > 15 or drift_detected:
            status = DataQualityStatus.POOR
        elif missing_percentage > 10 or outlier_percentage > 10:
            status = DataQualityStatus.FAIR
        elif missing_percentage > 5 or outlier_percentage > 5:
            status = DataQualityStatus.GOOD
        else:
            status = DataQualityStatus.EXCELLENT

        report = DataQualityReport(
            sensor_id=sensor_id,
            status=status,
            total_samples=total_samples,
            missing_count=missing_count,
            missing_percentage=missing_percentage,
            outlier_count=outlier_count,
            outlier_percentage=outlier_percentage,
            constant_periods=constant_periods,
            drift_detected=drift_detected,
            noise_level=noise_level,
            issues=issues,
            recommendations=recommendations,
        )

        logger.info(f"Data quality for {sensor_id}: {status.value} ({len(issues)} issues)")
        return report

    def engineer_features(
        self,
        data: np.ndarray,
        timestamps: Optional[List[datetime]] = None,
        window_sizes: List[int] = [3, 6, 12, 24],
    ) -> Dict[str, np.ndarray]:
        """
        Engineer features from raw sensor data

        Args:
            data: Raw sensor data
            timestamps: Optional timestamps
            window_sizes: Window sizes for rolling statistics

        Returns:
            Dictionary of engineered features
        """
        features = {}

        # Original data
        features["raw"] = data

        # Rolling statistics
        for window in window_sizes:
            if len(data) >= window:
                # Rolling mean
                features[f"rolling_mean_{window}"] = self._rolling_stat(data, window, np.mean)

                # Rolling std
                features[f"rolling_std_{window}"] = self._rolling_stat(data, window, np.std)

                # Rolling min/max
                features[f"rolling_min_{window}"] = self._rolling_stat(data, window, np.min)
                features[f"rolling_max_{window}"] = self._rolling_stat(data, window, np.max)

        # Differences and rates of change
        if len(data) > 1:
            features["diff_1"] = np.concatenate([[0], np.diff(data)])

            if len(data) > 2:
                features["diff_2"] = np.concatenate([[0, 0], np.diff(data, n=2)])

        # Time-based features (if timestamps available)
        if timestamps and len(timestamps) == len(data):
            features["hour"] = np.array([t.hour for t in timestamps])
            features["day_of_week"] = np.array([t.weekday() for t in timestamps])
            features["is_weekend"] = np.array([1 if t.weekday() >= 5 else 0 for t in timestamps])

        logger.debug(f"Engineered {len(features)} feature sets")
        return features

    def _rolling_stat(self, data: np.ndarray, window: int, stat_func) -> np.ndarray:
        """
        Compute rolling statistic with proper padding

        Args:
            data: Input data
            window: Window size
            stat_func: Statistic function (np.mean, np.std, etc.)

        Returns:
            Rolling statistic array (same length as data)
        """
        result = np.zeros(len(data))

        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            window_data = data[start_idx : i + 1]
            result[i] = stat_func(window_data)

        return result

    def prepare_training_data(
        self,
        data: np.ndarray,
        sensor_id: str,
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        normalize: bool = True,
        assess_quality: bool = True,
    ) -> Dict[str, Any]:
        """
        Prepare data for model training with all preprocessing steps

        Args:
            data: Raw sensor data
            sensor_id: Sensor identifier
            split_ratio: (train, val, test) split ratios
            normalize: Whether to normalize data
            assess_quality: Whether to assess data quality

        Returns:
            Dictionary containing processed data and metadata
        """
        result = {}

        # Data quality assessment
        if assess_quality:
            quality_report = self.assess_data_quality(data, sensor_id=sensor_id)
            result["quality_report"] = quality_report

            if quality_report.status == DataQualityStatus.CRITICAL:
                logger.warning(f"Critical data quality issues for {sensor_id}: {quality_report.issues}")

        # Split data
        total_len = len(data)
        train_size = int(total_len * split_ratio[0])
        val_size = int(total_len * split_ratio[1])

        train_data = data[:train_size]
        val_data = data[train_size : train_size + val_size]
        test_data = data[train_size + val_size :]

        # Fit normalization on training data only
        norm_params = None
        if normalize:
            norm_params = self.fit_normalization(train_data, sensor_id)
            train_data = self.normalize(train_data, sensor_id, norm_params)
            val_data = self.normalize(val_data, sensor_id, norm_params)
            test_data = self.normalize(test_data, sensor_id, norm_params)

        # Compute data hash for lineage
        data_hash = hashlib.sha256(data.tobytes()).hexdigest()[:16]

        result.update(
            {
                "train_data": train_data,
                "val_data": val_data,
                "test_data": test_data,
                "norm_params": norm_params,
                "data_hash": data_hash,
                "metadata": {
                    "sensor_id": sensor_id,
                    "total_samples": total_len,
                    "train_samples": len(train_data),
                    "val_samples": len(val_data),
                    "test_samples": len(test_data),
                    "normalized": normalize,
                    "split_ratio": split_ratio,
                },
            }
        )

        logger.info(
            f"Prepared training data for {sensor_id}: "
            f"train={len(train_data)}, val={len(val_data)}, test={len(test_data)}"
        )

        return result

    def get_cached_params(self, sensor_id: str) -> Optional[NormalizationParams]:
        """Get cached normalization parameters for a sensor"""
        return self.norm_params_cache.get(sensor_id)

    def clear_cache(self, sensor_id: Optional[str] = None):
        """
        Clear normalization parameter cache

        Args:
            sensor_id: Specific sensor to clear (clears all if None)
        """
        if sensor_id:
            if sensor_id in self.norm_params_cache:
                del self.norm_params_cache[sensor_id]
                logger.info(f"Cleared cache for {sensor_id}")
        else:
            self.norm_params_cache.clear()
            logger.info("Cleared all normalization caches")

        self._save_cached_params()

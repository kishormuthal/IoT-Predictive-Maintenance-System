"""
Data Drift Detection Module
Detect distribution shifts and concept drift in sensor data
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DriftSeverity(Enum):
    """Drift severity levels"""

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class DriftType(Enum):
    """Types of drift"""

    COVARIATE_SHIFT = "covariate_shift"  # P(X) changes
    CONCEPT_DRIFT = "concept_drift"  # P(Y|X) changes
    PRIOR_SHIFT = "prior_shift"  # P(Y) changes


@dataclass
class DriftReport:
    """Drift detection report"""

    sensor_id: str
    drift_detected: bool
    severity: DriftSeverity
    drift_types: List[DriftType]
    drift_score: float  # 0-1 score
    statistical_tests: Dict[str, Any]
    metrics: Dict[str, float]
    timestamp: datetime
    reference_period: Tuple[datetime, datetime]
    current_period: Tuple[datetime, datetime]
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "sensor_id": self.sensor_id,
            "drift_detected": self.drift_detected,
            "severity": self.severity.value,
            "drift_types": [dt.value for dt in self.drift_types],
            "drift_score": self.drift_score,
            "statistical_tests": self.statistical_tests,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
            "reference_period": [
                self.reference_period[0].isoformat(),
                self.reference_period[1].isoformat(),
            ],
            "current_period": [
                self.current_period[0].isoformat(),
                self.current_period[1].isoformat(),
            ],
            "recommendations": self.recommendations,
        }


@dataclass
class DriftConfig:
    """Configuration for drift detection"""

    # Statistical test thresholds
    ks_test_threshold: float = 0.05  # Kolmogorov-Smirnov test p-value
    mw_test_threshold: float = 0.05  # Mann-Whitney U test p-value
    chi2_test_threshold: float = 0.05  # Chi-square test p-value

    # Distribution-based thresholds
    psi_threshold: float = 0.2  # Population Stability Index
    jensen_shannon_threshold: float = 0.1  # Jensen-Shannon divergence

    # Statistical drift thresholds
    mean_shift_threshold: float = 2.0  # Standard deviations
    std_ratio_threshold: float = 2.0  # Ratio threshold

    # Window sizes
    reference_window_hours: int = 168  # 7 days
    current_window_hours: int = 24  # 1 day
    min_samples: int = 100

    # Severity thresholds
    low_drift_threshold: float = 0.3
    moderate_drift_threshold: float = 0.5
    high_drift_threshold: float = 0.7


class DataDriftDetector:
    """
    Detect data drift using multiple statistical methods
    """

    def __init__(self, config: Optional[DriftConfig] = None):
        """
        Initialize drift detector

        Args:
            config: Drift detection configuration
        """
        self.config = config or DriftConfig()

        # Store reference distributions for each sensor
        self.reference_distributions: Dict[str, Dict[str, Any]] = {}

    def fit_reference(
        self,
        data: np.ndarray,
        sensor_id: str,
        timestamps: Optional[List[datetime]] = None,
    ):
        """
        Fit reference distribution on baseline data

        Args:
            data: Reference data
            sensor_id: Sensor identifier
            timestamps: Optional timestamps
        """
        if len(data) < self.config.min_samples:
            logger.warning(f"Insufficient reference data for {sensor_id}: " f"{len(data)} < {self.config.min_samples}")
            return

        # Store reference statistics
        reference = {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "median": float(np.median(data)),
            "q1": float(np.percentile(data, 25)),
            "q3": float(np.percentile(data, 75)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "n_samples": len(data),
            "data": data.copy(),  # Store for statistical tests
            "fitted_at": datetime.now(),
        }

        if timestamps:
            reference["start_time"] = timestamps[0]
            reference["end_time"] = timestamps[-1]

        self.reference_distributions[sensor_id] = reference

        logger.info(f"Fitted reference distribution for {sensor_id} ({len(data)} samples)")

    def detect_drift(
        self,
        current_data: np.ndarray,
        sensor_id: str,
        current_timestamps: Optional[List[datetime]] = None,
        reference_data: Optional[np.ndarray] = None,
    ) -> DriftReport:
        """
        Detect drift between reference and current data

        Args:
            current_data: Current data to test
            sensor_id: Sensor identifier
            current_timestamps: Optional timestamps for current data
            reference_data: Optional reference data (uses fitted if None)

        Returns:
            DriftReport
        """
        # Get reference data
        if reference_data is None:
            if sensor_id not in self.reference_distributions:
                raise ValueError(
                    f"No reference distribution for {sensor_id}. " "Call fit_reference first or provide reference_data."
                )
            ref_data = self.reference_distributions[sensor_id]["data"]
            ref_period = (
                self.reference_distributions[sensor_id].get("start_time", datetime.now() - timedelta(days=7)),
                self.reference_distributions[sensor_id].get("end_time", datetime.now()),
            )
        else:
            ref_data = reference_data
            ref_period = (
                datetime.now() - timedelta(hours=self.config.reference_window_hours),
                datetime.now(),
            )

        # Current period
        if current_timestamps:
            cur_period = (current_timestamps[0], current_timestamps[-1])
        else:
            cur_period = (
                datetime.now() - timedelta(hours=self.config.current_window_hours),
                datetime.now(),
            )

        # Check minimum samples
        if len(current_data) < self.config.min_samples:
            logger.warning(
                f"Insufficient current data for drift detection: " f"{len(current_data)} < {self.config.min_samples}"
            )

        # Run statistical tests
        statistical_tests = self._run_statistical_tests(ref_data, current_data)

        # Compute drift metrics
        metrics = self._compute_drift_metrics(ref_data, current_data)

        # Determine drift severity
        drift_score = self._compute_drift_score(statistical_tests, metrics)
        severity = self._determine_severity(drift_score)

        # Identify drift types
        drift_types = self._identify_drift_types(statistical_tests, metrics)

        # Generate recommendations
        recommendations = self._generate_recommendations(severity, drift_types, statistical_tests, metrics)

        # Create report
        report = DriftReport(
            sensor_id=sensor_id,
            drift_detected=drift_score > self.config.low_drift_threshold,
            severity=severity,
            drift_types=drift_types,
            drift_score=drift_score,
            statistical_tests=statistical_tests,
            metrics=metrics,
            timestamp=datetime.now(),
            reference_period=ref_period,
            current_period=cur_period,
            recommendations=recommendations,
        )

        logger.info(f"Drift detection for {sensor_id}: " f"severity={severity.value}, score={drift_score:.3f}")

        return report

    def _run_statistical_tests(self, reference: np.ndarray, current: np.ndarray) -> Dict[str, Any]:
        """Run statistical hypothesis tests"""
        tests = {}

        # Kolmogorov-Smirnov test (distribution similarity)
        try:
            ks_stat, ks_pvalue = stats.ks_2samp(reference, current)
            tests["ks_test"] = {
                "statistic": float(ks_stat),
                "p_value": float(ks_pvalue),
                "drift_detected": ks_pvalue < self.config.ks_test_threshold,
            }
        except Exception as e:
            logger.warning(f"KS test failed: {e}")
            tests["ks_test"] = {"error": str(e)}

        # Mann-Whitney U test (median difference)
        try:
            mw_stat, mw_pvalue = stats.mannwhitneyu(reference, current, alternative="two-sided")
            tests["mann_whitney"] = {
                "statistic": float(mw_stat),
                "p_value": float(mw_pvalue),
                "drift_detected": mw_pvalue < self.config.mw_test_threshold,
            }
        except Exception as e:
            logger.warning(f"Mann-Whitney test failed: {e}")
            tests["mann_whitney"] = {"error": str(e)}

        # Chi-square test (binned distribution)
        try:
            # Bin the data
            bins = np.linspace(
                min(reference.min(), current.min()),
                max(reference.max(), current.max()),
                20,
            )
            ref_hist, _ = np.histogram(reference, bins=bins)
            cur_hist, _ = np.histogram(current, bins=bins)

            # Add small constant to avoid division by zero
            ref_hist = ref_hist + 1
            cur_hist = cur_hist + 1

            chi2_stat, chi2_pvalue = stats.chisquare(cur_hist, ref_hist)
            tests["chi_square"] = {
                "statistic": float(chi2_stat),
                "p_value": float(chi2_pvalue),
                "drift_detected": chi2_pvalue < self.config.chi2_test_threshold,
            }
        except Exception as e:
            logger.warning(f"Chi-square test failed: {e}")
            tests["chi_square"] = {"error": str(e)}

        return tests

    def _compute_drift_metrics(self, reference: np.ndarray, current: np.ndarray) -> Dict[str, float]:
        """Compute drift metrics"""
        metrics = {}

        # Mean shift (in standard deviations)
        ref_mean = np.mean(reference)
        ref_std = np.std(reference)
        cur_mean = np.mean(current)

        if ref_std > 1e-10:
            mean_shift = abs(cur_mean - ref_mean) / ref_std
        else:
            mean_shift = 0.0
        metrics["mean_shift_std"] = float(mean_shift)

        # Standard deviation ratio
        cur_std = np.std(current)
        if ref_std > 1e-10:
            std_ratio = cur_std / ref_std
        else:
            std_ratio = 1.0
        metrics["std_ratio"] = float(std_ratio)

        # Population Stability Index (PSI)
        psi = self._compute_psi(reference, current)
        metrics["psi"] = float(psi)

        # Jensen-Shannon divergence
        js_div = self._compute_jensen_shannon(reference, current)
        metrics["jensen_shannon_divergence"] = float(js_div)

        # Relative change in quantiles
        ref_q25, ref_q50, ref_q75 = np.percentile(reference, [25, 50, 75])
        cur_q25, cur_q50, cur_q75 = np.percentile(current, [25, 50, 75])

        if ref_q50 > 1e-10:
            metrics["median_change_pct"] = abs(cur_q50 - ref_q50) / ref_q50 * 100
        else:
            metrics["median_change_pct"] = 0.0

        if ref_q75 - ref_q25 > 1e-10:
            metrics["iqr_change_pct"] = abs((cur_q75 - cur_q25) - (ref_q75 - ref_q25)) / (ref_q75 - ref_q25) * 100
        else:
            metrics["iqr_change_pct"] = 0.0

        return metrics

    def _compute_psi(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """
        Compute Population Stability Index (PSI)

        PSI < 0.1: No significant change
        PSI < 0.2: Moderate change
        PSI >= 0.2: Significant change
        """
        # Create bins based on reference data
        bin_edges = np.percentile(reference, np.linspace(0, 100, bins + 1))
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        # Compute distributions
        ref_dist, _ = np.histogram(reference, bins=bin_edges)
        cur_dist, _ = np.histogram(current, bins=bin_edges)

        # Normalize to get percentages
        ref_pct = ref_dist / len(reference)
        cur_pct = cur_dist / len(current)

        # Avoid log(0)
        ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
        cur_pct = np.where(cur_pct == 0, 0.0001, cur_pct)

        # PSI formula
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

        return psi

    def _compute_jensen_shannon(self, reference: np.ndarray, current: np.ndarray, bins: int = 20) -> float:
        """
        Compute Jensen-Shannon divergence

        0 = identical distributions, 1 = completely different
        """
        # Create common bins
        bin_edges = np.linspace(
            min(reference.min(), current.min()),
            max(reference.max(), current.max()),
            bins + 1,
        )

        # Compute distributions
        ref_hist, _ = np.histogram(reference, bins=bin_edges)
        cur_hist, _ = np.histogram(current, bins=bin_edges)

        # Normalize to probabilities
        ref_prob = ref_hist / (len(reference) + 1e-10)
        cur_prob = cur_hist / (len(current) + 1e-10)

        # Avoid log(0)
        ref_prob = np.where(ref_prob == 0, 1e-10, ref_prob)
        cur_prob = np.where(cur_prob == 0, 1e-10, cur_prob)

        # Jensen-Shannon divergence
        m = 0.5 * (ref_prob + cur_prob)
        js_div = 0.5 * (np.sum(ref_prob * np.log(ref_prob / m)) + np.sum(cur_prob * np.log(cur_prob / m)))

        return js_div

    def _compute_drift_score(self, statistical_tests: Dict[str, Any], metrics: Dict[str, float]) -> float:
        """
        Compute overall drift score (0-1)

        Combines multiple signals into single score
        """
        scores = []

        # Statistical test signals
        if "ks_test" in statistical_tests and "p_value" in statistical_tests["ks_test"]:
            # Convert p-value to drift signal (lower p-value = higher drift)
            ks_signal = 1.0 - statistical_tests["ks_test"]["p_value"]
            scores.append(ks_signal)

        if "mann_whitney" in statistical_tests and "p_value" in statistical_tests["mann_whitney"]:
            mw_signal = 1.0 - statistical_tests["mann_whitney"]["p_value"]
            scores.append(mw_signal)

        # Metric-based signals
        if "psi" in metrics:
            # PSI: 0.2+ is significant
            psi_signal = min(metrics["psi"] / 0.3, 1.0)
            scores.append(psi_signal)

        if "jensen_shannon_divergence" in metrics:
            # JS: 0.1+ is significant
            js_signal = min(metrics["jensen_shannon_divergence"] / 0.2, 1.0)
            scores.append(js_signal)

        if "mean_shift_std" in metrics:
            # Mean shift: 2+ std is significant
            mean_signal = min(metrics["mean_shift_std"] / 3.0, 1.0)
            scores.append(mean_signal)

        if "std_ratio" in metrics:
            # Std ratio: 2x is significant
            std_signal = min(abs(metrics["std_ratio"] - 1.0) / 1.0, 1.0)
            scores.append(std_signal)

        # Average all signals
        if scores:
            return float(np.mean(scores))
        else:
            return 0.0

    def _determine_severity(self, drift_score: float) -> DriftSeverity:
        """Determine drift severity from score"""
        if drift_score < self.config.low_drift_threshold:
            return DriftSeverity.NONE
        elif drift_score < self.config.moderate_drift_threshold:
            return DriftSeverity.LOW
        elif drift_score < self.config.high_drift_threshold:
            return DriftSeverity.MODERATE
        elif drift_score < 0.85:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL

    def _identify_drift_types(self, statistical_tests: Dict[str, Any], metrics: Dict[str, float]) -> List[DriftType]:
        """Identify types of drift present"""
        drift_types = []

        # Covariate shift (distribution of inputs changed)
        if metrics.get("psi", 0) > self.config.psi_threshold:
            drift_types.append(DriftType.COVARIATE_SHIFT)

        if metrics.get("jensen_shannon_divergence", 0) > self.config.jensen_shannon_threshold:
            if DriftType.COVARIATE_SHIFT not in drift_types:
                drift_types.append(DriftType.COVARIATE_SHIFT)

        # For sensor data, we primarily detect covariate shift
        # Concept drift would require labels (prediction targets)

        return drift_types if drift_types else [DriftType.COVARIATE_SHIFT]

    def _generate_recommendations(
        self,
        severity: DriftSeverity,
        drift_types: List[DriftType],
        statistical_tests: Dict[str, Any],
        metrics: Dict[str, float],
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        if severity == DriftSeverity.NONE:
            recommendations.append("No action required - data distribution is stable")
            return recommendations

        if severity in [
            DriftSeverity.MODERATE,
            DriftSeverity.HIGH,
            DriftSeverity.CRITICAL,
        ]:
            recommendations.append("⚠️ Consider retraining models with recent data")

        if severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
            recommendations.append("🔴 Urgent: Review sensor calibration and data pipeline")

        if DriftType.COVARIATE_SHIFT in drift_types:
            recommendations.append("Data distribution has shifted - update preprocessing parameters")

        if metrics.get("mean_shift_std", 0) > self.config.mean_shift_threshold:
            recommendations.append("Significant mean shift detected - verify sensor readings")

        if metrics.get("std_ratio", 1.0) > self.config.std_ratio_threshold:
            recommendations.append("Variance has changed significantly - check for anomalies")

        if metrics.get("psi", 0) > 0.25:
            recommendations.append("High PSI indicates substantial distributional change")

        return recommendations

    def clear_reference(self, sensor_id: Optional[str] = None):
        """
        Clear reference distributions

        Args:
            sensor_id: Specific sensor to clear (clears all if None)
        """
        if sensor_id:
            if sensor_id in self.reference_distributions:
                del self.reference_distributions[sensor_id]
                logger.info(f"Cleared reference distribution for {sensor_id}")
        else:
            self.reference_distributions.clear()
            logger.info("Cleared all reference distributions")

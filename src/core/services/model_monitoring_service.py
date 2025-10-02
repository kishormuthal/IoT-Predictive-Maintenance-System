"""
Model Monitoring Service
Comprehensive model performance monitoring, degradation detection, and alerting
"""

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics"""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    ANOMALY_DETECTION = "anomaly_detection"
    FORECASTING = "forecasting"


class AlertSeverity(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PerformanceMetrics:
    """Container for model performance metrics"""

    timestamp: datetime
    sensor_id: str
    model_type: str
    metric_type: MetricType

    # Classification metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    true_positives: Optional[int] = None
    true_negatives: Optional[int] = None
    false_positives: Optional[int] = None
    false_negatives: Optional[int] = None

    # Regression metrics
    mae: Optional[float] = None  # Mean Absolute Error
    rmse: Optional[float] = None  # Root Mean Squared Error
    mape: Optional[float] = None  # Mean Absolute Percentage Error
    r2_score: Optional[float] = None  # R-squared

    # Anomaly detection specific
    num_anomalies_detected: Optional[int] = None
    anomaly_rate: Optional[float] = None
    mean_anomaly_score: Optional[float] = None

    # Forecasting specific
    forecast_horizon: Optional[int] = None
    forecast_accuracy_by_step: Optional[List[float]] = None

    # Additional metadata
    sample_size: int = 0
    inference_time_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "timestamp": self.timestamp.isoformat(),
            "sensor_id": self.sensor_id,
            "model_type": self.model_type,
            "metric_type": self.metric_type.value,
            "sample_size": self.sample_size,
        }

        # Add non-None metrics
        for key, value in self.__dict__.items():
            if key not in [
                "timestamp",
                "sensor_id",
                "model_type",
                "metric_type",
                "sample_size",
            ]:
                if value is not None:
                    if isinstance(value, list):
                        result[key] = value
                    else:
                        result[key] = float(value) if isinstance(value, (int, float)) else value

        return result


@dataclass
class PerformanceAlert:
    """Performance degradation alert"""

    timestamp: datetime
    sensor_id: str
    model_type: str
    severity: AlertSeverity
    metric_name: str
    current_value: float
    baseline_value: float
    degradation_pct: float
    message: str
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "sensor_id": self.sensor_id,
            "model_type": self.model_type,
            "severity": self.severity.value,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "baseline_value": self.baseline_value,
            "degradation_pct": self.degradation_pct,
            "message": self.message,
            "recommendations": self.recommendations,
        }


class ModelMonitoringService:
    """
    Comprehensive model monitoring service

    Features:
    - Performance metric tracking
    - Degradation detection
    - Alerting system
    - Metric visualization
    - Historical analysis
    """

    def __init__(
        self,
        metrics_storage_path: str = "data/monitoring/metrics",
        alerts_storage_path: str = "data/monitoring/alerts",
        baseline_window_days: int = 7,
        degradation_threshold: float = 0.10,  # 10% degradation
        critical_threshold: float = 0.25,  # 25% degradation
        window_size: int = 100,  # Rolling window for recent metrics
    ):
        """
        Initialize model monitoring service

        Args:
            metrics_storage_path: Path to store metrics
            alerts_storage_path: Path to store alerts
            baseline_window_days: Days for baseline calculation
            degradation_threshold: Threshold for warning alerts (10% default)
            critical_threshold: Threshold for critical alerts (25% default)
            window_size: Size of rolling window for recent metrics
        """
        self.metrics_path = Path(metrics_storage_path)
        self.alerts_path = Path(alerts_storage_path)
        self.metrics_path.mkdir(parents=True, exist_ok=True)
        self.alerts_path.mkdir(parents=True, exist_ok=True)

        self.baseline_window_days = baseline_window_days
        self.degradation_threshold = degradation_threshold
        self.critical_threshold = critical_threshold

        # In-memory storage for recent metrics (per sensor/model)
        self.recent_metrics: Dict[str, deque] = {}
        self.window_size = window_size

        # Baseline metrics (per sensor/model)
        self.baselines: Dict[str, Dict[str, float]] = {}

        # Alert history
        self.alerts: List[PerformanceAlert] = []

        # Load existing data
        self._load_baselines()
        self._load_alerts()

        logger.info("Model monitoring service initialized")

    def _get_key(self, sensor_id: str, model_type: str) -> str:
        """Get cache key for sensor/model combination"""
        return f"{sensor_id}_{model_type}"

    def _load_baselines(self):
        """Load baseline metrics from storage"""
        try:
            baseline_file = self.metrics_path / "baselines.json"
            if baseline_file.exists():
                with open(baseline_file, "r") as f:
                    self.baselines = json.load(f)
                logger.info(f"Loaded {len(self.baselines)} baseline metrics")
        except Exception as e:
            logger.warning(f"Could not load baselines: {e}")

    def _save_baselines(self):
        """Save baseline metrics to storage"""
        try:
            baseline_file = self.metrics_path / "baselines.json"
            with open(baseline_file, "w") as f:
                json.dump(self.baselines, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving baselines: {e}")

    def _load_alerts(self):
        """Load alert history from storage"""
        try:
            alerts_file = self.alerts_path / "alerts_history.json"
            if alerts_file.exists():
                with open(alerts_file, "r") as f:
                    alerts_data = json.load(f)
                    for alert_dict in alerts_data:
                        alert = PerformanceAlert(
                            timestamp=datetime.fromisoformat(alert_dict["timestamp"]),
                            sensor_id=alert_dict["sensor_id"],
                            model_type=alert_dict["model_type"],
                            severity=AlertSeverity(alert_dict["severity"]),
                            metric_name=alert_dict["metric_name"],
                            current_value=alert_dict["current_value"],
                            baseline_value=alert_dict["baseline_value"],
                            degradation_pct=alert_dict["degradation_pct"],
                            message=alert_dict["message"],
                            recommendations=alert_dict.get("recommendations", []),
                        )
                        self.alerts.append(alert)
                logger.info(f"Loaded {len(self.alerts)} alerts")
        except Exception as e:
            logger.warning(f"Could not load alerts: {e}")

    def _save_alerts(self):
        """Save alert history to storage"""
        try:
            alerts_file = self.alerts_path / "alerts_history.json"
            alerts_data = [alert.to_dict() for alert in self.alerts]
            with open(alerts_file, "w") as f:
                json.dump(alerts_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving alerts: {e}")

    def log_metrics(self, metrics: PerformanceMetrics):
        """
        Log performance metrics

        Args:
            metrics: Performance metrics to log
        """
        key = self._get_key(metrics.sensor_id, metrics.model_type)

        # Add to recent metrics
        if key not in self.recent_metrics:
            self.recent_metrics[key] = deque(maxlen=self.window_size)
        self.recent_metrics[key].append(metrics)

        # Save to disk
        self._save_metrics_to_disk(metrics)

        # Check for degradation
        self._check_degradation(metrics)

        logger.info(
            f"Logged metrics for {metrics.sensor_id} ({metrics.model_type}): " f"type={metrics.metric_type.value}"
        )

    def _save_metrics_to_disk(self, metrics: PerformanceMetrics):
        """Save metrics to disk"""
        try:
            # Organize by sensor and date
            date_str = metrics.timestamp.strftime("%Y%m%d")
            sensor_dir = self.metrics_path / metrics.sensor_id
            sensor_dir.mkdir(exist_ok=True)

            metrics_file = sensor_dir / f"{metrics.model_type}_{date_str}.jsonl"

            # Append to JSONL file
            with open(metrics_file, "a") as f:
                f.write(json.dumps(metrics.to_dict()) + "\n")

        except Exception as e:
            logger.error(f"Error saving metrics to disk: {e}")

    def set_baseline(self, sensor_id: str, model_type: str, baseline_metrics: Dict[str, float]):
        """
        Set baseline metrics for a model

        Args:
            sensor_id: Sensor identifier
            model_type: Model type
            baseline_metrics: Dictionary of baseline metric values
        """
        key = self._get_key(sensor_id, model_type)
        self.baselines[key] = baseline_metrics
        self._save_baselines()

        logger.info(f"Set baseline for {sensor_id} ({model_type}): " f"{list(baseline_metrics.keys())}")

    def compute_baseline_from_history(
        self, sensor_id: str, model_type: str, days_back: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute baseline from historical metrics

        Args:
            sensor_id: Sensor identifier
            model_type: Model type
            days_back: Number of days to look back (uses baseline_window_days if None)

        Returns:
            Dictionary of baseline metrics
        """
        days_back = days_back or self.baseline_window_days
        key = self._get_key(sensor_id, model_type)

        # Get recent metrics
        if key in self.recent_metrics and len(self.recent_metrics[key]) > 0:
            recent = list(self.recent_metrics[key])

            # Filter by time window
            cutoff = datetime.now() - timedelta(days=days_back)
            recent = [m for m in recent if m.timestamp >= cutoff]

            if not recent:
                logger.warning(f"No recent metrics for baseline calculation: {key}")
                return {}

            # Compute averages
            baseline = {}

            # Helper to safely compute mean
            def safe_mean(values):
                values = [v for v in values if v is not None]
                return float(np.mean(values)) if values else None

            # Classification metrics
            accuracies = [m.accuracy for m in recent if m.accuracy is not None]
            if accuracies:
                baseline["accuracy"] = safe_mean(accuracies)

            precisions = [m.precision for m in recent if m.precision is not None]
            if precisions:
                baseline["precision"] = safe_mean(precisions)

            recalls = [m.recall for m in recent if m.recall is not None]
            if recalls:
                baseline["recall"] = safe_mean(recalls)

            f1_scores = [m.f1_score for m in recent if m.f1_score is not None]
            if f1_scores:
                baseline["f1_score"] = safe_mean(f1_scores)

            # Regression metrics
            maes = [m.mae for m in recent if m.mae is not None]
            if maes:
                baseline["mae"] = safe_mean(maes)

            rmses = [m.rmse for m in recent if m.rmse is not None]
            if rmses:
                baseline["rmse"] = safe_mean(rmses)

            mapes = [m.mape for m in recent if m.mape is not None]
            if mapes:
                baseline["mape"] = safe_mean(mapes)

            r2_scores = [m.r2_score for m in recent if m.r2_score is not None]
            if r2_scores:
                baseline["r2_score"] = safe_mean(r2_scores)

            # Set as baseline
            if baseline:
                self.set_baseline(sensor_id, model_type, baseline)

            return baseline

        return {}

    def _check_degradation(self, metrics: PerformanceMetrics):
        """Check for performance degradation and create alerts"""
        key = self._get_key(metrics.sensor_id, metrics.model_type)

        # Get baseline
        baseline = self.baselines.get(key)
        if not baseline:
            logger.debug(f"No baseline set for {key}, skipping degradation check")
            return

        # Check each metric
        metrics_to_check = {
            "accuracy": (metrics.accuracy, True),  # (value, higher_is_better)
            "precision": (metrics.precision, True),
            "recall": (metrics.recall, True),
            "f1_score": (metrics.f1_score, True),
            "mae": (metrics.mae, False),
            "rmse": (metrics.rmse, False),
            "mape": (metrics.mape, False),
            "r2_score": (metrics.r2_score, True),
        }

        for metric_name, (current_value, higher_is_better) in metrics_to_check.items():
            if current_value is None or metric_name not in baseline:
                continue

            baseline_value = baseline[metric_name]

            # Calculate degradation
            if higher_is_better:
                degradation = (baseline_value - current_value) / (baseline_value + 1e-10)
            else:
                # For error metrics (lower is better), degradation is increase
                degradation = (current_value - baseline_value) / (baseline_value + 1e-10)

            # Check thresholds
            if degradation >= self.critical_threshold:
                self._create_alert(
                    metrics=metrics,
                    metric_name=metric_name,
                    current_value=current_value,
                    baseline_value=baseline_value,
                    degradation_pct=degradation * 100,
                    severity=AlertSeverity.CRITICAL,
                )
            elif degradation >= self.degradation_threshold:
                self._create_alert(
                    metrics=metrics,
                    metric_name=metric_name,
                    current_value=current_value,
                    baseline_value=baseline_value,
                    degradation_pct=degradation * 100,
                    severity=AlertSeverity.WARNING,
                )

    def _create_alert(
        self,
        metrics: PerformanceMetrics,
        metric_name: str,
        current_value: float,
        baseline_value: float,
        degradation_pct: float,
        severity: AlertSeverity,
    ):
        """Create performance degradation alert"""
        message = (
            f"{severity.value.upper()}: {metric_name} degraded by {degradation_pct:.1f}% "
            f"for {metrics.sensor_id} ({metrics.model_type}). "
            f"Current: {current_value:.3f}, Baseline: {baseline_value:.3f}"
        )

        # Generate recommendations
        recommendations = []
        if degradation_pct >= 25:
            recommendations.append("🔴 CRITICAL: Immediate retraining recommended")
            recommendations.append("Review recent data for quality issues")
            recommendations.append("Check for data drift")
        elif degradation_pct >= 15:
            recommendations.append("⚠️ WARNING: Schedule retraining soon")
            recommendations.append("Monitor closely for further degradation")
        else:
            recommendations.append("Monitor performance trends")

        alert = PerformanceAlert(
            timestamp=datetime.now(),
            sensor_id=metrics.sensor_id,
            model_type=metrics.model_type,
            severity=severity,
            metric_name=metric_name,
            current_value=current_value,
            baseline_value=baseline_value,
            degradation_pct=degradation_pct,
            message=message,
            recommendations=recommendations,
        )

        self.alerts.append(alert)
        self._save_alerts()

        logger.warning(message)

    def get_recent_metrics(self, sensor_id: str, model_type: str, limit: int = 10) -> List[PerformanceMetrics]:
        """Get recent metrics for a model"""
        key = self._get_key(sensor_id, model_type)

        if key in self.recent_metrics:
            recent = list(self.recent_metrics[key])
            return recent[-limit:] if len(recent) > limit else recent

        return []

    def get_alerts(
        self,
        sensor_id: Optional[str] = None,
        model_type: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        days_back: int = 7,
    ) -> List[PerformanceAlert]:
        """
        Get alerts filtered by criteria

        Args:
            sensor_id: Filter by sensor ID
            model_type: Filter by model type
            severity: Filter by severity
            days_back: Number of days to look back

        Returns:
            List of matching alerts
        """
        cutoff = datetime.now() - timedelta(days=days_back)

        filtered = [a for a in self.alerts if a.timestamp >= cutoff]

        if sensor_id:
            filtered = [a for a in filtered if a.sensor_id == sensor_id]

        if model_type:
            filtered = [a for a in filtered if a.model_type == model_type]

        if severity:
            filtered = [a for a in filtered if a.severity == severity]

        return filtered

    def get_metric_trend(
        self, sensor_id: str, model_type: str, metric_name: str, days_back: int = 30
    ) -> Tuple[List[datetime], List[float]]:
        """
        Get trend for a specific metric

        Args:
            sensor_id: Sensor identifier
            model_type: Model type
            metric_name: Metric name
            days_back: Number of days to look back

        Returns:
            (timestamps, values) tuple
        """
        key = self._get_key(sensor_id, model_type)

        if key not in self.recent_metrics:
            return ([], [])

        cutoff = datetime.now() - timedelta(days=days_back)
        recent = [m for m in self.recent_metrics[key] if m.timestamp >= cutoff]

        timestamps = []
        values = []

        for metrics in recent:
            value = getattr(metrics, metric_name, None)
            if value is not None:
                timestamps.append(metrics.timestamp)
                values.append(value)

        return (timestamps, values)

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get overall monitoring summary"""
        recent_alerts = self.get_alerts(days_back=7)

        critical_alerts = [a for a in recent_alerts if a.severity == AlertSeverity.CRITICAL]
        warning_alerts = [a for a in recent_alerts if a.severity == AlertSeverity.WARNING]

        # Count models being monitored
        monitored_models = len(self.recent_metrics)

        # Count models with baselines
        models_with_baselines = len(self.baselines)

        return {
            "monitored_models": monitored_models,
            "models_with_baselines": models_with_baselines,
            "total_alerts_7d": len(recent_alerts),
            "critical_alerts_7d": len(critical_alerts),
            "warning_alerts_7d": len(warning_alerts),
            "degradation_threshold": self.degradation_threshold * 100,
            "critical_threshold": self.critical_threshold * 100,
            "baseline_window_days": self.baseline_window_days,
        }

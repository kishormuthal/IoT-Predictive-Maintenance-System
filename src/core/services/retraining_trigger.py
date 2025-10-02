"""
Automated Retraining Trigger System
Monitors model performance and data drift to trigger automated retraining
"""

import numpy as np
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from pathlib import Path

from ..services.data_drift_detector import DataDriftDetector, DriftReport, DriftSeverity
from ...infrastructure.ml.mlflow_tracker import MLflowTracker, ModelStage

logger = logging.getLogger(__name__)


class RetrainingReason(Enum):
    """Reasons for triggering retraining"""
    DRIFT_DETECTED = "drift_detected"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SCHEDULE = "scheduled_retraining"
    MANUAL = "manual_trigger"
    DATA_QUALITY = "data_quality_issues"


@dataclass
class RetrainingTrigger:
    """Retraining trigger event"""
    sensor_id: str
    model_type: str
    reason: RetrainingReason
    severity: str  # LOW, MODERATE, HIGH, CRITICAL
    timestamp: datetime
    metrics: Dict[str, float]
    details: Dict[str, Any] = field(default_factory=dict)
    triggered: bool = False


@dataclass
class RetrainingPolicy:
    """Policy configuration for automated retraining"""
    # Drift-based triggers
    enable_drift_trigger: bool = True
    drift_severity_threshold: DriftSeverity = DriftSeverity.MODERATE
    drift_score_threshold: float = 0.5

    # Performance-based triggers
    enable_performance_trigger: bool = True
    performance_degradation_threshold: float = 0.15  # 15% drop
    min_performance_threshold: float = 0.70  # Absolute minimum

    # Scheduled retraining
    enable_scheduled_retraining: bool = True
    retraining_interval_days: int = 30

    # Data quality triggers
    enable_quality_trigger: bool = True
    min_data_quality_score: float = 0.7

    # Cooldown period (prevent too frequent retraining)
    cooldown_hours: int = 24

    # Auto-promotion to production
    auto_promote_to_production: bool = False
    min_improvement_for_promotion: float = 0.05  # 5% improvement


class RetrainingTriggerSystem:
    """
    Automated retraining trigger system

    Monitors:
    - Data drift
    - Model performance degradation
    - Data quality issues
    - Scheduled intervals

    Triggers retraining when thresholds exceeded
    """

    def __init__(
        self,
        policy: Optional[RetrainingPolicy] = None,
        drift_detector: Optional[DataDriftDetector] = None,
        mlflow_tracker: Optional[MLflowTracker] = None,
        trigger_log_path: str = "data/retraining_triggers.json"
    ):
        """
        Initialize retraining trigger system

        Args:
            policy: Retraining policy configuration
            drift_detector: Data drift detector instance
            mlflow_tracker: MLflow tracker for model metadata
            trigger_log_path: Path to store trigger history
        """
        self.policy = policy or RetrainingPolicy()
        self.drift_detector = drift_detector or DataDriftDetector()
        self.mlflow_tracker = mlflow_tracker
        self.trigger_log_path = Path(trigger_log_path)

        # Trigger history
        self.triggers: List[RetrainingTrigger] = []

        # Last retraining timestamp per sensor
        self.last_retrained: Dict[str, datetime] = {}

        # Load trigger history
        self._load_trigger_history()

        logger.info("Retraining trigger system initialized")

    def _load_trigger_history(self):
        """Load trigger history from disk"""
        try:
            if self.trigger_log_path.exists():
                with open(self.trigger_log_path, 'r') as f:
                    history = json.load(f)

                    for trigger_data in history:
                        # Reconstruct trigger
                        trigger = RetrainingTrigger(
                            sensor_id=trigger_data['sensor_id'],
                            model_type=trigger_data['model_type'],
                            reason=RetrainingReason(trigger_data['reason']),
                            severity=trigger_data['severity'],
                            timestamp=datetime.fromisoformat(trigger_data['timestamp']),
                            metrics=trigger_data.get('metrics', {}),
                            details=trigger_data.get('details', {}),
                            triggered=trigger_data.get('triggered', False)
                        )
                        self.triggers.append(trigger)

                        # Update last_retrained
                        if trigger.triggered:
                            sensor_key = f"{trigger.sensor_id}_{trigger.model_type}"
                            if sensor_key not in self.last_retrained or trigger.timestamp > self.last_retrained[sensor_key]:
                                self.last_retrained[sensor_key] = trigger.timestamp

                logger.info(f"Loaded {len(self.triggers)} trigger events from history")

        except Exception as e:
            logger.warning(f"Could not load trigger history: {e}")

    def _save_trigger_history(self):
        """Save trigger history to disk"""
        try:
            self.trigger_log_path.parent.mkdir(parents=True, exist_ok=True)

            history = []
            for trigger in self.triggers:
                history.append({
                    'sensor_id': trigger.sensor_id,
                    'model_type': trigger.model_type,
                    'reason': trigger.reason.value,
                    'severity': trigger.severity,
                    'timestamp': trigger.timestamp.isoformat(),
                    'metrics': trigger.metrics,
                    'details': trigger.details,
                    'triggered': trigger.triggered
                })

            with open(self.trigger_log_path, 'w') as f:
                json.dump(history, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving trigger history: {e}")

    def check_drift_trigger(
        self,
        sensor_id: str,
        model_type: str,
        drift_report: DriftReport
    ) -> Optional[RetrainingTrigger]:
        """
        Check if drift warrants retraining

        Args:
            sensor_id: Sensor identifier
            model_type: Model type
            drift_report: Drift detection report

        Returns:
            RetrainingTrigger if triggered, None otherwise
        """
        if not self.policy.enable_drift_trigger:
            return None

        # Check severity threshold
        severity_levels = [
            DriftSeverity.NONE,
            DriftSeverity.LOW,
            DriftSeverity.MODERATE,
            DriftSeverity.HIGH,
            DriftSeverity.CRITICAL
        ]

        drift_severity_index = severity_levels.index(drift_report.severity)
        threshold_index = severity_levels.index(self.policy.drift_severity_threshold)

        if drift_severity_index >= threshold_index or drift_report.drift_score >= self.policy.drift_score_threshold:
            trigger = RetrainingTrigger(
                sensor_id=sensor_id,
                model_type=model_type,
                reason=RetrainingReason.DRIFT_DETECTED,
                severity=drift_report.severity.value,
                timestamp=datetime.now(),
                metrics={
                    'drift_score': drift_report.drift_score,
                    'psi': drift_report.metrics.get('psi', 0.0),
                    'mean_shift': drift_report.metrics.get('mean_shift_std', 0.0)
                },
                details={
                    'drift_types': [dt.value for dt in drift_report.drift_types],
                    'recommendations': drift_report.recommendations,
                    'statistical_tests': drift_report.statistical_tests
                }
            )

            logger.warning(
                f"Drift trigger for {sensor_id} ({model_type}): "
                f"severity={drift_report.severity.value}, score={drift_report.drift_score:.3f}"
            )

            return trigger

        return None

    def check_performance_trigger(
        self,
        sensor_id: str,
        model_type: str,
        current_performance: float,
        baseline_performance: float,
        metric_name: str = "accuracy"
    ) -> Optional[RetrainingTrigger]:
        """
        Check if performance degradation warrants retraining

        Args:
            sensor_id: Sensor identifier
            model_type: Model type
            current_performance: Current model performance
            baseline_performance: Baseline (training) performance
            metric_name: Name of performance metric

        Returns:
            RetrainingTrigger if triggered, None otherwise
        """
        if not self.policy.enable_performance_trigger:
            return None

        # Calculate degradation
        degradation = (baseline_performance - current_performance) / (baseline_performance + 1e-10)

        # Check thresholds
        if degradation >= self.policy.performance_degradation_threshold or \
           current_performance < self.policy.min_performance_threshold:

            trigger = RetrainingTrigger(
                sensor_id=sensor_id,
                model_type=model_type,
                reason=RetrainingReason.PERFORMANCE_DEGRADATION,
                severity="HIGH" if degradation >= 0.25 else "MODERATE",
                timestamp=datetime.now(),
                metrics={
                    'current_performance': current_performance,
                    'baseline_performance': baseline_performance,
                    'degradation_pct': degradation * 100,
                    'metric_name': metric_name
                }
            )

            logger.warning(
                f"Performance trigger for {sensor_id} ({model_type}): "
                f"{metric_name} dropped from {baseline_performance:.3f} to {current_performance:.3f} "
                f"({degradation*100:.1f}% degradation)"
            )

            return trigger

        return None

    def check_scheduled_trigger(
        self,
        sensor_id: str,
        model_type: str,
        last_training_date: datetime
    ) -> Optional[RetrainingTrigger]:
        """
        Check if scheduled retraining is due

        Args:
            sensor_id: Sensor identifier
            model_type: Model type
            last_training_date: Date of last training

        Returns:
            RetrainingTrigger if triggered, None otherwise
        """
        if not self.policy.enable_scheduled_retraining:
            return None

        days_since_training = (datetime.now() - last_training_date).days

        if days_since_training >= self.policy.retraining_interval_days:
            trigger = RetrainingTrigger(
                sensor_id=sensor_id,
                model_type=model_type,
                reason=RetrainingReason.SCHEDULE,
                severity="LOW",
                timestamp=datetime.now(),
                metrics={
                    'days_since_training': days_since_training,
                    'interval_days': self.policy.retraining_interval_days
                }
            )

            logger.info(
                f"Scheduled trigger for {sensor_id} ({model_type}): "
                f"{days_since_training} days since last training"
            )

            return trigger

        return None

    def check_quality_trigger(
        self,
        sensor_id: str,
        model_type: str,
        quality_score: float,
        quality_issues: List[str]
    ) -> Optional[RetrainingTrigger]:
        """
        Check if data quality issues warrant retraining

        Args:
            sensor_id: Sensor identifier
            model_type: Model type
            quality_score: Data quality score (0-1)
            quality_issues: List of quality issues

        Returns:
            RetrainingTrigger if triggered, None otherwise
        """
        if not self.policy.enable_quality_trigger:
            return None

        if quality_score < self.policy.min_data_quality_score:
            trigger = RetrainingTrigger(
                sensor_id=sensor_id,
                model_type=model_type,
                reason=RetrainingReason.DATA_QUALITY,
                severity="MODERATE",
                timestamp=datetime.now(),
                metrics={
                    'quality_score': quality_score,
                    'min_threshold': self.policy.min_data_quality_score
                },
                details={
                    'issues': quality_issues
                }
            )

            logger.warning(
                f"Quality trigger for {sensor_id} ({model_type}): "
                f"quality_score={quality_score:.3f}, issues={quality_issues}"
            )

            return trigger

        return None

    def should_retrain(
        self,
        sensor_id: str,
        model_type: str,
        trigger: RetrainingTrigger
    ) -> bool:
        """
        Determine if retraining should proceed based on cooldown period

        Args:
            sensor_id: Sensor identifier
            model_type: Model type
            trigger: Retraining trigger

        Returns:
            True if retraining should proceed
        """
        sensor_key = f"{sensor_id}_{model_type}"

        # Check cooldown period
        if sensor_key in self.last_retrained:
            time_since_last = datetime.now() - self.last_retrained[sensor_key]
            hours_since_last = time_since_last.total_seconds() / 3600

            if hours_since_last < self.policy.cooldown_hours:
                logger.info(
                    f"Cooldown period active for {sensor_id} ({model_type}): "
                    f"{hours_since_last:.1f}h since last retraining "
                    f"(cooldown: {self.policy.cooldown_hours}h)"
                )
                return False

        return True

    def register_trigger(
        self,
        trigger: RetrainingTrigger,
        execute_retraining: bool = False
    ):
        """
        Register a retraining trigger

        Args:
            trigger: Retraining trigger to register
            execute_retraining: Whether retraining was actually executed
        """
        trigger.triggered = execute_retraining

        self.triggers.append(trigger)

        if execute_retraining:
            sensor_key = f"{trigger.sensor_id}_{trigger.model_type}"
            self.last_retrained[sensor_key] = trigger.timestamp

        self._save_trigger_history()

        logger.info(
            f"Registered trigger for {trigger.sensor_id} ({trigger.model_type}): "
            f"reason={trigger.reason.value}, triggered={execute_retraining}"
        )

    def get_trigger_history(
        self,
        sensor_id: Optional[str] = None,
        model_type: Optional[str] = None,
        days_back: int = 30
    ) -> List[RetrainingTrigger]:
        """
        Get trigger history

        Args:
            sensor_id: Filter by sensor ID
            model_type: Filter by model type
            days_back: Number of days to look back

        Returns:
            List of triggers
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)

        filtered = [
            t for t in self.triggers
            if t.timestamp >= cutoff_date
        ]

        if sensor_id:
            filtered = [t for t in filtered if t.sensor_id == sensor_id]

        if model_type:
            filtered = [t for t in filtered if t.model_type == model_type]

        return filtered

    def get_trigger_summary(self) -> Dict[str, Any]:
        """Get summary of trigger history"""
        if not self.triggers:
            return {'total_triggers': 0}

        # Count by reason
        reason_counts = {}
        for trigger in self.triggers:
            reason = trigger.reason.value
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        # Count triggered vs not triggered
        triggered_count = sum(1 for t in self.triggers if t.triggered)

        return {
            'total_triggers': len(self.triggers),
            'triggered_count': triggered_count,
            'not_triggered_count': len(self.triggers) - triggered_count,
            'triggers_by_reason': reason_counts,
            'last_trigger': self.triggers[-1].timestamp.isoformat() if self.triggers else None
        }

    def auto_promote_model(
        self,
        model_name: str,
        new_version: str,
        new_performance: float,
        production_performance: float
    ) -> bool:
        """
        Automatically promote model to production if improvement threshold met

        Args:
            model_name: Registered model name
            new_version: New model version
            new_performance: Performance of new model
            production_performance: Performance of current production model

        Returns:
            True if promoted
        """
        if not self.policy.auto_promote_to_production or not self.mlflow_tracker:
            return False

        improvement = (new_performance - production_performance) / (production_performance + 1e-10)

        if improvement >= self.policy.min_improvement_for_promotion:
            try:
                self.mlflow_tracker.transition_model_stage(
                    model_name=model_name,
                    version=new_version,
                    stage=ModelStage.PRODUCTION,
                    archive_existing=True
                )

                logger.info(
                    f"Auto-promoted model '{model_name}' v{new_version} to PRODUCTION: "
                    f"{improvement*100:.1f}% improvement"
                )

                return True

            except Exception as e:
                logger.error(f"Error auto-promoting model: {e}")
                return False

        logger.info(
            f"Model '{model_name}' v{new_version} not promoted: "
            f"improvement {improvement*100:.1f}% below threshold {self.policy.min_improvement_for_promotion*100:.1f}%"
        )

        return False

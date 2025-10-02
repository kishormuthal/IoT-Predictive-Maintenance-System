"""
Anomaly Detection Service
Clean service layer for Telemanom-based anomaly detection
Enhanced with model registry integration
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ...infrastructure.ml.model_registry import ModelRegistry
from ...infrastructure.ml.telemanom_wrapper import (
    ModelNotTrainedError,
    NASATelemanom,
    Telemanom_Config,
)
from ..interfaces.detector_interface import AnomalyDetectorInterface
from ..models.anomaly import (
    AnomalyDetection,
    AnomalyDetectionResult,
    AnomalySeverity,
    AnomalySummary,
    AnomalyType,
)
from ..models.sensor_data import SensorDataBatch

logger = logging.getLogger(__name__)


class AnomalyDetectionService(AnomalyDetectorInterface):
    """
    Service for anomaly detection using NASA Telemanom algorithm
    """

    def __init__(
        self,
        registry_path: str = "data/models/registry",
        detection_history_size: int = 1000,
        fallback_threshold: float = 3.0,
    ):
        """
        Initialize anomaly detection service

        Args:
            registry_path: Path to model registry (single source of truth for models)
            detection_history_size: Maximum number of recent detections to keep per sensor
            fallback_threshold: Z-score threshold for fallback statistical detection
        """
        # Initialize model registry as single source of truth
        self.model_registry = ModelRegistry(registry_path)

        # Cache for loaded models
        self._models: Dict[str, NASATelemanom] = {}

        # Detection history for summary generation
        self._detection_history: Dict[str, List[AnomalyDetection]] = {}
        self.detection_history_size = detection_history_size
        self.fallback_threshold = fallback_threshold

        logger.info(
            f"Anomaly Detection Service initialized with registry at {registry_path}, "
            f"history size: {detection_history_size}"
        )

    def _get_model(self, sensor_id: str) -> NASATelemanom:
        """Get or create Telemanom model for sensor

        Uses ModelRegistry as single source of truth for model loading.

        Args:
            sensor_id: Unique sensor identifier

        Returns:
            NASATelemanom model instance (may or may not be trained)
        """
        if sensor_id not in self._models:
            # Create new model instance
            config = Telemanom_Config()
            model = NASATelemanom(sensor_id, config)

            # Load model from registry (single source of truth)
            active_version = self.model_registry.get_active_model_version(
                sensor_id, "telemanom"
            )

            if active_version:
                metadata = self.model_registry.get_model_metadata(active_version)
                if metadata:
                    # Get model path from registry metadata
                    registry_model_path = (
                        Path(metadata.model_path)
                        if hasattr(metadata, "model_path")
                        else None
                    )

                    if registry_model_path and registry_model_path.exists():
                        load_success = model.load_model(registry_model_path.parent)

                        # Enforce is_trained check after loading
                        if load_success and model.is_trained:
                            logger.info(
                                f"Loaded trained model {active_version} for sensor {sensor_id} "
                                f"from {registry_model_path}"
                            )
                        else:
                            logger.warning(
                                f"Model {active_version} loaded but is_trained={model.is_trained} "
                                f"for sensor {sensor_id}"
                            )
                            model.is_trained = False  # Ensure consistency
                    else:
                        logger.warning(
                            f"Model path not found in metadata for version {active_version}, "
                            f"sensor {sensor_id}"
                        )
                else:
                    logger.warning(
                        f"Model metadata not found for version {active_version}, sensor {sensor_id}"
                    )
            else:
                # No model in registry - model will be marked as untrained
                logger.info(
                    f"No registered model found for sensor {sensor_id} - model untrained"
                )

            self._models[sensor_id] = model

        return self._models[sensor_id]

    def _calculate_severity(self, score: float, threshold: float) -> AnomalySeverity:
        """Calculate anomaly severity based on score and threshold

        Args:
            score: Anomaly score from detection algorithm
            threshold: Detection threshold

        Returns:
            AnomalySeverity: Severity level classification
        """
        # Add epsilon for numerical stability
        epsilon = 1e-10
        severity_ratio = score / (threshold + epsilon)

        if severity_ratio >= 3.0:
            return AnomalySeverity.CRITICAL
        elif severity_ratio >= 2.0:
            return AnomalySeverity.HIGH
        elif severity_ratio >= 1.5:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW

    def detect_anomalies(
        self, sensor_id: str, data: np.ndarray, timestamps: List[datetime]
    ) -> Dict[str, Any]:
        """
        Detect anomalies in sensor data using Telemanom

        Args:
            sensor_id: Unique sensor identifier
            data: Time series data to analyze
            timestamps: Corresponding timestamps

        Returns:
            Dictionary containing anomaly detection results
        """
        start_time = datetime.now()

        try:
            # Get model for sensor
            model = self._get_model(sensor_id)

            if not model.is_trained:
                logger.warning(
                    f"Model not trained for sensor {sensor_id}, using fallback detection"
                )
                return self._fallback_detection(sensor_id, data, timestamps)

            # Run Telemanom detection
            detection_result = model.detect_anomalies(data)

            # Convert to structured anomaly objects
            anomalies = []
            anomaly_indices = detection_result["anomalies"]
            scores = detection_result["scores"]  # Full scores array for all data points
            threshold = detection_result["threshold"]

            for idx in anomaly_indices:
                if idx < len(timestamps):
                    # Correct indexing: scores is a full array matching data length
                    # idx refers to the position in the original data
                    score = float(scores[idx]) if idx < len(scores) else 0.0

                    severity = self._calculate_severity(score, threshold)

                    # Calculate severity_ratio instead of "confidence"
                    # This more accurately reflects what we're measuring
                    epsilon = 1e-10
                    severity_magnitude = score / (threshold + epsilon)

                    anomaly = AnomalyDetection(
                        sensor_id=sensor_id,
                        timestamp=timestamps[idx],
                        value=float(data[idx]) if idx < len(data) else 0.0,
                        score=float(score),
                        severity=severity,
                        anomaly_type=AnomalyType.POINT,
                        confidence=min(
                            1.0, severity_magnitude
                        ),  # Capped severity ratio
                        description=f"Telemanom detected anomaly (score: {score:.3f}, severity: {severity.value})",
                    )
                    anomalies.append(anomaly)

            # Store in history
            if sensor_id not in self._detection_history:
                self._detection_history[sensor_id] = []
            self._detection_history[sensor_id].extend(anomalies)

            # Keep only recent detections (configurable size)
            self._detection_history[sensor_id] = self._detection_history[sensor_id][
                -self.detection_history_size :
            ]

            processing_time = (datetime.now() - start_time).total_seconds()

            # Create result object
            result = AnomalyDetectionResult(
                sensor_id=sensor_id,
                data_processed=len(data),
                anomalies_detected=anomalies,
                processing_time=processing_time,
                model_version="Telemanom-1.0",
                detection_timestamp=datetime.now(),
                statistics={
                    "total_points": len(data),
                    "anomaly_count": len(anomalies),
                    "anomaly_rate": len(anomalies) / len(data) if len(data) > 0 else 0,
                    "threshold": threshold,
                    "max_score": max(scores) if scores else 0,
                    "mean_score": np.mean(scores) if scores else 0,
                },
            )

            logger.info(f"Detected {len(anomalies)} anomalies for sensor {sensor_id}")

            return {
                "sensor_id": sensor_id,
                "anomalies": [self._anomaly_to_dict(a) for a in anomalies],
                "statistics": result.statistics,
                "processing_time": processing_time,
                "model_status": "trained",
            }

        except Exception as e:
            logger.error(f"Error detecting anomalies for sensor {sensor_id}: {e}")
            return self._fallback_detection(sensor_id, data, timestamps)

    def _fallback_detection(
        self, sensor_id: str, data: np.ndarray, timestamps: List[datetime]
    ) -> Dict[str, Any]:
        """Fallback anomaly detection when model is not available

        Uses statistical z-score based detection with configurable threshold.
        Handles edge cases like constant data gracefully.

        Args:
            sensor_id: Sensor identifier
            data: Time series data
            timestamps: Corresponding timestamps

        Returns:
            dict: Detection results in consistent format with main detection
        """
        try:
            # Simple statistical anomaly detection
            if len(data) < 10:
                return {
                    "sensor_id": sensor_id,
                    "anomalies": [],
                    "statistics": {"total_points": len(data), "anomaly_count": 0},
                    "processing_time": 0.0,
                    "model_status": "fallback_insufficient_data",
                }

            # Use z-score based detection with configurable threshold
            mean_val = np.mean(data)
            std_val = np.std(data)

            anomalies = []

            # Handle constant data case
            if std_val == 0:
                logger.warning(
                    f"Constant data detected for sensor {sensor_id} (std=0). "
                    f"No anomalies detected via z-score method."
                )
                # All values are identical - no anomalies can be detected via z-score
                return {
                    "sensor_id": sensor_id,
                    "anomalies": [],
                    "statistics": {
                        "total_points": len(data),
                        "anomaly_count": 0,
                        "anomaly_rate": 0.0,
                        "threshold": self.fallback_threshold,
                        "mean_value": float(mean_val),
                        "std_value": 0.0,
                        "note": "Constant data - no variance",
                    },
                    "processing_time": 0.001,
                    "model_status": "fallback_constant_data",
                }

            # Calculate z-scores
            for i, (value, timestamp) in enumerate(zip(data, timestamps)):
                z_score = abs(value - mean_val) / std_val

                if z_score > self.fallback_threshold:
                    # Severity based on z-score magnitude
                    if z_score > self.fallback_threshold * 2:
                        severity = AnomalySeverity.HIGH
                    elif z_score > self.fallback_threshold * 1.5:
                        severity = AnomalySeverity.MEDIUM
                    else:
                        severity = AnomalySeverity.LOW

                    anomaly = AnomalyDetection(
                        sensor_id=sensor_id,
                        timestamp=timestamp,
                        value=float(value),
                        score=z_score / self.fallback_threshold,
                        severity=severity,
                        anomaly_type=AnomalyType.POINT,
                        confidence=0.6,  # Lower confidence for fallback detection
                        description=f"Statistical anomaly (z-score: {z_score:.2f})",
                    )
                    anomalies.append(anomaly)

            return {
                "sensor_id": sensor_id,
                "anomalies": [self._anomaly_to_dict(a) for a in anomalies],
                "statistics": {
                    "total_points": len(data),
                    "anomaly_count": len(anomalies),
                    "anomaly_rate": len(anomalies) / len(data),
                    "threshold": self.fallback_threshold,
                    "mean_value": float(mean_val),
                    "std_value": float(std_val),
                },
                "processing_time": 0.001,
                "model_status": "fallback",
            }

        except Exception as e:
            logger.error(f"Fallback detection failed for sensor {sensor_id}: {e}")
            return {
                "sensor_id": sensor_id,
                "anomalies": [],
                "statistics": {"total_points": len(data), "anomaly_count": 0},
                "processing_time": 0.0,
                "model_status": "error",
            }

    def _anomaly_to_dict(self, anomaly: AnomalyDetection) -> Dict[str, Any]:
        """Convert anomaly object to dictionary"""
        return {
            "sensor_id": anomaly.sensor_id,
            "timestamp": anomaly.timestamp,
            "value": anomaly.value,
            "score": anomaly.score,
            "severity": anomaly.severity.value,
            "type": anomaly.anomaly_type.value,
            "confidence": anomaly.confidence,
            "description": anomaly.description,
        }

    def is_model_trained(self, sensor_id: str) -> bool:
        """Check if model is trained for given sensor"""
        try:
            model = self._get_model(sensor_id)
            return model.is_trained
        except Exception:
            return False

    def get_detection_summary(
        self, sensor_id: str = None, hours_back: int = 24
    ) -> Dict[str, Any]:
        """Get summary of recent detections"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)

            if sensor_id:
                # Summary for specific sensor
                sensor_anomalies = self._detection_history.get(sensor_id, [])
                recent_anomalies = [
                    a for a in sensor_anomalies if a.timestamp >= cutoff_time
                ]

                severity_counts = {}
                for anomaly in recent_anomalies:
                    severity = anomaly.severity.value
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1

                return {
                    "sensor_id": sensor_id,
                    "total_anomalies": len(recent_anomalies),
                    "severity_breakdown": severity_counts,
                    "recent_anomalies": [
                        self._anomaly_to_dict(a) for a in recent_anomalies[-10:]
                    ],
                    "time_range_hours": hours_back,
                }
            else:
                # Summary across all sensors
                all_anomalies = []
                sensor_stats = {}

                for sid, anomalies in self._detection_history.items():
                    recent_anomalies = [
                        a for a in anomalies if a.timestamp >= cutoff_time
                    ]
                    all_anomalies.extend(recent_anomalies)

                    # Fix: Safely access latest_detection
                    latest_detection_time = None
                    if recent_anomalies:
                        # Sort by timestamp to get the truly latest
                        sorted_anomalies = sorted(
                            recent_anomalies, key=lambda x: x.timestamp, reverse=True
                        )
                        latest_detection_time = sorted_anomalies[
                            0
                        ].timestamp.isoformat()

                    sensor_stats[sid] = {
                        "anomaly_count": len(recent_anomalies),
                        "latest_detection": latest_detection_time,
                        "model_trained": self.is_model_trained(sid),
                    }

                severity_breakdown = {}
                for anomaly in all_anomalies:
                    severity = anomaly.severity.value
                    severity_breakdown[severity] = (
                        severity_breakdown.get(severity, 0) + 1
                    )

                # Sort by timestamp for recent anomalies
                all_anomalies.sort(key=lambda x: x.timestamp, reverse=True)

                return {
                    "total_anomalies": len(all_anomalies),
                    "severity_breakdown": severity_breakdown,
                    "recent_anomalies": [
                        self._anomaly_to_dict(a) for a in all_anomalies[:20]
                    ],
                    "sensor_stats": sensor_stats,
                    "time_range_hours": hours_back,
                    "generated_at": datetime.now(),
                }

        except Exception as e:
            logger.error(f"Error generating detection summary: {e}")
            return {
                "total_anomalies": 0,
                "severity_breakdown": {},
                "recent_anomalies": [],
                "sensor_stats": {},
                "error": str(e),
            }

    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all loaded models"""
        status = {}
        for sensor_id, model in self._models.items():
            # Get registry information
            active_version = self.model_registry.get_active_model_version(
                sensor_id, "telemanom"
            )
            metadata = None
            if active_version:
                metadata = self.model_registry.get_model_metadata(active_version)

            status[sensor_id] = {
                "is_trained": model.is_trained,
                "error_threshold": getattr(model, "error_threshold", None),
                "model_parameters": model.model.count_params() if model.model else 0,
                "last_used": datetime.now(),
                "registry_info": {
                    "active_version": active_version,
                    "performance_score": metadata.performance_score if metadata else 0,
                    "last_trained": metadata.created_at if metadata else None,
                    "model_size_mb": (
                        metadata.model_size_bytes / (1024 * 1024) if metadata else 0
                    ),
                },
            }
        return status

    def get_training_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for model training"""
        try:
            from config.equipment_config import get_equipment_list

            equipment_list = get_equipment_list()

            recommendations = {
                "sensors_needing_training": [],
                "sensors_needing_retraining": [],
                "well_performing_sensors": [],
                "total_sensors": len(equipment_list),
            }

            for equipment in equipment_list:
                sensor_id = equipment.equipment_id

                # Check if model exists and its performance
                active_version = self.model_registry.get_active_model_version(
                    sensor_id, "telemanom"
                )

                if not active_version:
                    recommendations["sensors_needing_training"].append(
                        {
                            "sensor_id": sensor_id,
                            "equipment_type": equipment.equipment_type.value,
                            "criticality": equipment.criticality.value,
                            "reason": "No trained model found",
                        }
                    )
                else:
                    metadata = self.model_registry.get_model_metadata(active_version)
                    if metadata:
                        if metadata.performance_score < 0.5:
                            recommendations["sensors_needing_retraining"].append(
                                {
                                    "sensor_id": sensor_id,
                                    "equipment_type": equipment.equipment_type.value,
                                    "criticality": equipment.criticality.value,
                                    "performance_score": metadata.performance_score,
                                    "reason": "Low performance score",
                                }
                            )
                        else:
                            recommendations["well_performing_sensors"].append(
                                {
                                    "sensor_id": sensor_id,
                                    "equipment_type": equipment.equipment_type.value,
                                    "performance_score": metadata.performance_score,
                                }
                            )

            return recommendations

        except Exception as e:
            logger.error(f"Error generating training recommendations: {e}")
            return {
                "error": str(e),
                "sensors_needing_training": [],
                "sensors_needing_retraining": [],
                "well_performing_sensors": [],
            }

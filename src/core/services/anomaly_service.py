"""
Anomaly Detection Service
Clean service layer for Telemanom-based anomaly detection
Enhanced with model registry integration
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

from ..interfaces.detector_interface import AnomalyDetectorInterface
from ..models.anomaly import AnomalyDetection, AnomalyDetectionResult, AnomalySummary, AnomalySeverity, AnomalyType
from ..models.sensor_data import SensorDataBatch
from ...infrastructure.ml.telemanom_wrapper import NASATelemanom, Telemanom_Config
from ...infrastructure.ml.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class AnomalyDetectionService(AnomalyDetectorInterface):
    """
    Service for anomaly detection using NASA Telemanom algorithm
    """

    def __init__(self, model_path: str = "data/models/nasa_equipment_models", registry_path: str = "data/models"):
        """
        Initialize anomaly detection service

        Args:
            model_path: Path to store/load trained models
            registry_path: Path to model registry
        """
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)

        # Initialize model registry
        self.model_registry = ModelRegistry(registry_path)

        # Cache for loaded models
        self._models: Dict[str, NASATelemanom] = {}

        # Detection history for summary generation
        self._detection_history: Dict[str, List[AnomalyDetection]] = {}

        logger.info("Anomaly Detection Service initialized with model registry")

    def _get_model(self, sensor_id: str) -> NASATelemanom:
        """Get or create Telemanom model for sensor"""
        if sensor_id not in self._models:
            # Create new model instance
            config = Telemanom_Config()
            model = NASATelemanom(sensor_id, config)

            # Try to load model from registry first
            active_version = self.model_registry.get_active_model_version(sensor_id, 'telemanom')
            if active_version:
                metadata = self.model_registry.get_model_metadata(active_version)
                if metadata:
                    registry_model_path = Path(metadata.model_path if hasattr(metadata, 'model_path') else self.model_path / sensor_id)
                    if model.load_model(registry_model_path):
                        logger.info(f"Loaded registered model {active_version} for sensor {sensor_id}")
                    else:
                        logger.warning(f"Failed to load registered model {active_version} for sensor {sensor_id}")
                else:
                    logger.warning(f"Model metadata not found for version {active_version}")
            else:
                # Fallback to legacy model loading
                if not model.load_model(self.model_path):
                    logger.warning(f"No trained model found for sensor {sensor_id}")

            self._models[sensor_id] = model

        return self._models[sensor_id]

    def _calculate_severity(self, score: float, threshold: float) -> AnomalySeverity:
        """Calculate anomaly severity based on score and threshold"""
        severity_ratio = score / threshold if threshold > 0 else 1.0

        if severity_ratio >= 3.0:
            return AnomalySeverity.CRITICAL
        elif severity_ratio >= 2.0:
            return AnomalySeverity.HIGH
        elif severity_ratio >= 1.5:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW

    def detect_anomalies(self, sensor_id: str, data: np.ndarray, timestamps: List[datetime]) -> Dict[str, Any]:
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
                logger.warning(f"Model not trained for sensor {sensor_id}, using fallback detection")
                return self._fallback_detection(sensor_id, data, timestamps)

            # Run Telemanom detection
            detection_result = model.detect_anomalies(data)

            # Convert to structured anomaly objects
            anomalies = []
            anomaly_indices = detection_result['anomalies']
            scores = detection_result['scores']
            threshold = detection_result['threshold']

            for idx in anomaly_indices:
                if idx < len(timestamps) and idx < len(scores):
                    score = scores[idx] if isinstance(scores, list) else scores[min(idx, len(scores)-1)]
                    severity = self._calculate_severity(score, threshold)

                    anomaly = AnomalyDetection(
                        sensor_id=sensor_id,
                        timestamp=timestamps[idx],
                        value=float(data[idx]) if idx < len(data) else 0.0,
                        score=float(score),
                        severity=severity,
                        anomaly_type=AnomalyType.POINT,
                        confidence=min(1.0, score / threshold) if threshold > 0 else 0.5,
                        description=f"Telemanom detected anomaly (score: {score:.3f})"
                    )
                    anomalies.append(anomaly)

            # Store in history
            if sensor_id not in self._detection_history:
                self._detection_history[sensor_id] = []
            self._detection_history[sensor_id].extend(anomalies)

            # Keep only recent detections (last 1000)
            self._detection_history[sensor_id] = self._detection_history[sensor_id][-1000:]

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
                    'total_points': len(data),
                    'anomaly_count': len(anomalies),
                    'anomaly_rate': len(anomalies) / len(data) if len(data) > 0 else 0,
                    'threshold': threshold,
                    'max_score': max(scores) if scores else 0,
                    'mean_score': np.mean(scores) if scores else 0
                }
            )

            logger.info(f"Detected {len(anomalies)} anomalies for sensor {sensor_id}")

            return {
                'sensor_id': sensor_id,
                'anomalies': [self._anomaly_to_dict(a) for a in anomalies],
                'statistics': result.statistics,
                'processing_time': processing_time,
                'model_status': 'trained'
            }

        except Exception as e:
            logger.error(f"Error detecting anomalies for sensor {sensor_id}: {e}")
            return self._fallback_detection(sensor_id, data, timestamps)

    def _fallback_detection(self, sensor_id: str, data: np.ndarray, timestamps: List[datetime]) -> Dict[str, Any]:
        """Fallback anomaly detection when model is not available"""
        try:
            # Simple statistical anomaly detection
            if len(data) < 10:
                return {
                    'sensor_id': sensor_id,
                    'anomalies': [],
                    'statistics': {'total_points': len(data), 'anomaly_count': 0},
                    'processing_time': 0.0,
                    'model_status': 'fallback'
                }

            # Use z-score based detection
            mean_val = np.mean(data)
            std_val = np.std(data)
            threshold = 3.0  # 3-sigma rule

            anomalies = []
            for i, (value, timestamp) in enumerate(zip(data, timestamps)):
                z_score = abs(value - mean_val) / std_val if std_val > 0 else 0

                if z_score > threshold:
                    severity = AnomalySeverity.HIGH if z_score > 4 else AnomalySeverity.MEDIUM

                    anomaly = AnomalyDetection(
                        sensor_id=sensor_id,
                        timestamp=timestamp,
                        value=float(value),
                        score=z_score / threshold,
                        severity=severity,
                        anomaly_type=AnomalyType.POINT,
                        confidence=0.7,
                        description=f"Statistical anomaly (z-score: {z_score:.2f})"
                    )
                    anomalies.append(anomaly)

            return {
                'sensor_id': sensor_id,
                'anomalies': [self._anomaly_to_dict(a) for a in anomalies],
                'statistics': {
                    'total_points': len(data),
                    'anomaly_count': len(anomalies),
                    'anomaly_rate': len(anomalies) / len(data),
                    'threshold': threshold,
                    'mean_value': mean_val,
                    'std_value': std_val
                },
                'processing_time': 0.001,
                'model_status': 'fallback'
            }

        except Exception as e:
            logger.error(f"Fallback detection failed for sensor {sensor_id}: {e}")
            return {
                'sensor_id': sensor_id,
                'anomalies': [],
                'statistics': {'total_points': len(data), 'anomaly_count': 0},
                'processing_time': 0.0,
                'model_status': 'error'
            }

    def _anomaly_to_dict(self, anomaly: AnomalyDetection) -> Dict[str, Any]:
        """Convert anomaly object to dictionary"""
        return {
            'sensor_id': anomaly.sensor_id,
            'timestamp': anomaly.timestamp,
            'value': anomaly.value,
            'score': anomaly.score,
            'severity': anomaly.severity.value,
            'type': anomaly.anomaly_type.value,
            'confidence': anomaly.confidence,
            'description': anomaly.description
        }

    def is_model_trained(self, sensor_id: str) -> bool:
        """Check if model is trained for given sensor"""
        try:
            model = self._get_model(sensor_id)
            return model.is_trained
        except Exception:
            return False

    def get_detection_summary(self, sensor_id: str = None, hours_back: int = 24) -> Dict[str, Any]:
        """Get summary of recent detections"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)

            if sensor_id:
                # Summary for specific sensor
                sensor_anomalies = self._detection_history.get(sensor_id, [])
                recent_anomalies = [a for a in sensor_anomalies if a.timestamp >= cutoff_time]

                severity_counts = {}
                for anomaly in recent_anomalies:
                    severity = anomaly.severity.value
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1

                return {
                    'sensor_id': sensor_id,
                    'total_anomalies': len(recent_anomalies),
                    'severity_breakdown': severity_counts,
                    'recent_anomalies': [self._anomaly_to_dict(a) for a in recent_anomalies[-10:]],
                    'time_range_hours': hours_back
                }
            else:
                # Summary across all sensors
                all_anomalies = []
                sensor_stats = {}

                for sid, anomalies in self._detection_history.items():
                    recent_anomalies = [a for a in anomalies if a.timestamp >= cutoff_time]
                    all_anomalies.extend(recent_anomalies)

                    sensor_stats[sid] = {
                        'anomaly_count': len(recent_anomalies),
                        'latest_detection': recent_anomalies[-1].timestamp if recent_anomalies else None,
                        'model_trained': self.is_model_trained(sid)
                    }

                severity_breakdown = {}
                for anomaly in all_anomalies:
                    severity = anomaly.severity.value
                    severity_breakdown[severity] = severity_breakdown.get(severity, 0) + 1

                # Sort by timestamp for recent anomalies
                all_anomalies.sort(key=lambda x: x.timestamp, reverse=True)

                return {
                    'total_anomalies': len(all_anomalies),
                    'severity_breakdown': severity_breakdown,
                    'recent_anomalies': [self._anomaly_to_dict(a) for a in all_anomalies[:20]],
                    'sensor_stats': sensor_stats,
                    'time_range_hours': hours_back,
                    'generated_at': datetime.now()
                }

        except Exception as e:
            logger.error(f"Error generating detection summary: {e}")
            return {
                'total_anomalies': 0,
                'severity_breakdown': {},
                'recent_anomalies': [],
                'sensor_stats': {},
                'error': str(e)
            }

    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all loaded models"""
        status = {}
        for sensor_id, model in self._models.items():
            # Get registry information
            active_version = self.model_registry.get_active_model_version(sensor_id, 'telemanom')
            metadata = None
            if active_version:
                metadata = self.model_registry.get_model_metadata(active_version)

            status[sensor_id] = {
                'is_trained': model.is_trained,
                'error_threshold': getattr(model, 'error_threshold', None),
                'model_parameters': model.model.count_params() if model.model else 0,
                'last_used': datetime.now(),
                'registry_info': {
                    'active_version': active_version,
                    'performance_score': metadata.performance_score if metadata else 0,
                    'last_trained': metadata.created_at if metadata else None,
                    'model_size_mb': metadata.model_size_bytes / (1024 * 1024) if metadata else 0
                }
            }
        return status

    def get_training_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for model training"""
        try:
            from config.equipment_config import get_equipment_list
            equipment_list = get_equipment_list()

            recommendations = {
                'sensors_needing_training': [],
                'sensors_needing_retraining': [],
                'well_performing_sensors': [],
                'total_sensors': len(equipment_list)
            }

            for equipment in equipment_list:
                sensor_id = equipment.equipment_id

                # Check if model exists and its performance
                active_version = self.model_registry.get_active_model_version(sensor_id, 'telemanom')

                if not active_version:
                    recommendations['sensors_needing_training'].append({
                        'sensor_id': sensor_id,
                        'equipment_type': equipment.equipment_type.value,
                        'criticality': equipment.criticality.value,
                        'reason': 'No trained model found'
                    })
                else:
                    metadata = self.model_registry.get_model_metadata(active_version)
                    if metadata:
                        if metadata.performance_score < 0.5:
                            recommendations['sensors_needing_retraining'].append({
                                'sensor_id': sensor_id,
                                'equipment_type': equipment.equipment_type.value,
                                'criticality': equipment.criticality.value,
                                'performance_score': metadata.performance_score,
                                'reason': 'Low performance score'
                            })
                        else:
                            recommendations['well_performing_sensors'].append({
                                'sensor_id': sensor_id,
                                'equipment_type': equipment.equipment_type.value,
                                'performance_score': metadata.performance_score
                            })

            return recommendations

        except Exception as e:
            logger.error(f"Error generating training recommendations: {e}")
            return {
                'error': str(e),
                'sensors_needing_training': [],
                'sensors_needing_retraining': [],
                'well_performing_sensors': []
            }
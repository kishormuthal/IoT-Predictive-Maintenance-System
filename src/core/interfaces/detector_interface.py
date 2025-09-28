"""
Abstract interface for anomaly detectors
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from datetime import datetime


class AnomalyDetectorInterface(ABC):
    """Abstract base class for all anomaly detectors"""

    @abstractmethod
    def detect_anomalies(self, sensor_id: str, data: np.ndarray, timestamps: List[datetime]) -> Dict[str, Any]:
        """
        Detect anomalies in sensor data

        Args:
            sensor_id: Unique sensor identifier
            data: Time series data to analyze
            timestamps: Corresponding timestamps

        Returns:
            Dictionary containing anomaly detection results
        """
        pass

    @abstractmethod
    def is_model_trained(self, sensor_id: str) -> bool:
        """Check if model is trained for given sensor"""
        pass

    @abstractmethod
    def get_detection_summary(self, sensor_id: str) -> Dict[str, Any]:
        """Get summary of recent detections for sensor"""
        pass
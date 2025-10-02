"""
Data models for anomaly detection
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class AnomalySeverity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AnomalyType(Enum):
    POINT = "POINT"  # Single point anomaly
    CONTEXTUAL = "CONTEXTUAL"  # Context-dependent anomaly
    COLLECTIVE = "COLLECTIVE"  # Pattern anomaly


@dataclass
class AnomalyDetection:
    """Single anomaly detection result"""

    sensor_id: str
    timestamp: datetime
    value: float
    score: float  # Anomaly score (0-1)
    severity: AnomalySeverity
    anomaly_type: AnomalyType
    confidence: float = 0.0
    description: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection for a sensor"""

    sensor_id: str
    data_processed: int
    anomalies_detected: List[AnomalyDetection]
    processing_time: float
    model_version: str
    detection_timestamp: datetime
    statistics: Dict[str, Any]


@dataclass
class AnomalySummary:
    """Summary of anomaly detection across all sensors"""

    total_anomalies: int
    severity_breakdown: Dict[str, int]
    recent_anomalies: List[AnomalyDetection]
    sensor_stats: Dict[str, Dict[str, Any]]
    generated_at: datetime

"""
Data models for sensor data
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class SensorStatus(Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class CriticalityLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class SensorReading:
    """Single sensor reading"""

    sensor_id: str
    timestamp: datetime
    value: float
    status: SensorStatus = SensorStatus.NORMAL
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SensorInfo:
    """Sensor metadata and configuration"""

    sensor_id: str
    name: str
    equipment_id: str
    sensor_type: str
    unit: str
    criticality: CriticalityLevel
    location: str
    description: str
    normal_range: tuple
    warning_threshold: float
    critical_threshold: float
    data_source: str  # 'smap' or 'msl'
    channel_index: int


@dataclass
class SensorDataBatch:
    """Batch of sensor data with metadata"""

    sensor_id: str
    timestamps: List[datetime]
    values: np.ndarray
    sensor_info: SensorInfo
    statistics: Dict[str, Any]
    quality_score: float = 1.0

"""
Data models for forecasting
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
from enum import Enum


class ForecastConfidence(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class ForecastPoint:
    """Single forecast point"""
    timestamp: datetime
    predicted_value: float
    confidence_lower: float
    confidence_upper: float
    confidence_level: ForecastConfidence
    risk_level: RiskLevel


@dataclass
class ForecastResult:
    """Forecast result for a sensor"""
    sensor_id: str
    forecast_horizon_hours: int
    historical_timestamps: List[datetime]
    historical_values: np.ndarray
    forecast_timestamps: List[datetime]
    forecast_values: np.ndarray
    confidence_intervals: Dict[str, np.ndarray]  # upper, lower
    accuracy_metrics: Dict[str, float]
    model_version: str
    generated_at: datetime
    risk_assessment: Dict[str, Any]


@dataclass
class ForecastSummary:
    """Summary of forecasts across all sensors"""
    total_sensors_forecasted: int
    average_confidence: float
    risk_distribution: Dict[str, int]
    recent_forecasts: List[ForecastResult]
    model_performance: Dict[str, Dict[str, float]]
    generated_at: datetime
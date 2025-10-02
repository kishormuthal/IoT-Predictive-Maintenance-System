"""
Abstract interface for forecasters
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ForecasterInterface(ABC):
    """Abstract base class for all forecasters"""

    @abstractmethod
    def generate_forecast(self, sensor_id: str, data: np.ndarray, horizon_hours: int = 24) -> Dict[str, Any]:
        """
        Generate forecast for sensor data

        Args:
            sensor_id: Unique sensor identifier
            data: Historical time series data
            horizon_hours: Forecast horizon in hours

        Returns:
            Dictionary containing forecast results
        """
        pass

    @abstractmethod
    def is_model_trained(self, sensor_id: str) -> bool:
        """Check if forecasting model is trained for given sensor"""
        pass

    @abstractmethod
    def get_forecast_accuracy(self, sensor_id: str) -> Dict[str, float]:
        """Get forecast accuracy metrics for sensor"""
        pass

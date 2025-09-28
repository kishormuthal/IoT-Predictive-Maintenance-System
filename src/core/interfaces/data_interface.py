"""
Abstract interface for data sources
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime


class DataSourceInterface(ABC):
    """Abstract base class for data sources"""

    @abstractmethod
    def get_sensor_data(self, sensor_id: str, hours_back: int = 24) -> Dict[str, Any]:
        """
        Get sensor data for specified time period

        Args:
            sensor_id: Unique sensor identifier
            hours_back: Hours of historical data to retrieve

        Returns:
            Dictionary containing sensor data and metadata
        """
        pass

    @abstractmethod
    def get_sensor_list(self) -> List[Dict[str, Any]]:
        """Get list of available sensors"""
        pass

    @abstractmethod
    def get_latest_value(self, sensor_id: str) -> Dict[str, Any]:
        """Get latest sensor reading"""
        pass
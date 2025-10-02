"""
NASA Sensor Test Fixtures - Session 1
Comprehensive test data for 12 NASA sensors (6 SMAP + 6 MSL)
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from src.core.models.anomaly import AnomalyDetection, AnomalySeverity, AnomalyType
from src.core.models.forecast import ForecastConfidence, ForecastResult, RiskLevel
from src.core.models.sensor_data import (
    CriticalityLevel,
    SensorDataBatch,
    SensorInfo,
    SensorReading,
)

# Test markers
pytestmark = [pytest.mark.session1, pytest.mark.data]


@pytest.fixture(scope="session")
def nasa_sensor_configs():
    """Fixture providing configuration for all 12 NASA sensors"""
    return {
        # SMAP Sensors (Soil Moisture Active Passive)
        "SMAP-ATT-001": {
            "name": "Attitude Control System",
            "equipment_id": "ATT-001",
            "sensor_type": "attitude",
            "unit": "degrees",
            "criticality": CriticalityLevel.HIGH,
            "location": "spacecraft_body",
            "description": "Primary attitude control sensor for SMAP mission",
            "normal_range": (0.0, 360.0),
            "warning_threshold": 300.0,
            "critical_threshold": 340.0,
            "data_source": "smap",
            "channel_index": 0,
        },
        "SMAP-COM-001": {
            "name": "Communication System",
            "equipment_id": "COM-001",
            "sensor_type": "communication",
            "unit": "dBm",
            "criticality": CriticalityLevel.MEDIUM,
            "location": "antenna_assembly",
            "description": "Communication signal strength sensor",
            "normal_range": (-80.0, -20.0),
            "warning_threshold": -70.0,
            "critical_threshold": -75.0,
            "data_source": "smap",
            "channel_index": 1,
        },
        "SMAP-PAY-001": {
            "name": "Payload Temperature",
            "equipment_id": "PAY-001",
            "sensor_type": "temperature",
            "unit": "celsius",
            "criticality": CriticalityLevel.HIGH,
            "location": "payload_compartment",
            "description": "Payload temperature monitoring sensor",
            "normal_range": (-10.0, 50.0),
            "warning_threshold": 40.0,
            "critical_threshold": 45.0,
            "data_source": "smap",
            "channel_index": 2,
        },
        "SMAP-PWR-001": {
            "name": "Power System Voltage",
            "equipment_id": "PWR-001",
            "sensor_type": "voltage",
            "unit": "volts",
            "criticality": CriticalityLevel.CRITICAL,
            "location": "power_module",
            "description": "Main power system voltage sensor",
            "normal_range": (24.0, 30.0),
            "warning_threshold": 25.0,
            "critical_threshold": 24.5,
            "data_source": "smap",
            "channel_index": 3,
        },
        "SMAP-THM-001": {
            "name": "Thermal Management",
            "equipment_id": "THM-001",
            "sensor_type": "temperature",
            "unit": "celsius",
            "criticality": CriticalityLevel.MEDIUM,
            "location": "thermal_radiator",
            "description": "Thermal management system sensor",
            "normal_range": (-20.0, 60.0),
            "warning_threshold": 50.0,
            "critical_threshold": 55.0,
            "data_source": "smap",
            "channel_index": 4,
        },
        "SMAP-SOI-001": {
            "name": "Solar Panel Current",
            "equipment_id": "SOI-001",
            "sensor_type": "current",
            "unit": "amperes",
            "criticality": CriticalityLevel.HIGH,
            "location": "solar_array",
            "description": "Solar panel current generation sensor",
            "normal_range": (0.0, 15.0),
            "warning_threshold": 12.0,
            "critical_threshold": 14.0,
            "data_source": "smap",
            "channel_index": 5,
        },
        # MSL Sensors (Mars Science Laboratory)
        "MSL-COM-001": {
            "name": "Communication Array",
            "equipment_id": "COM-001",
            "sensor_type": "communication",
            "unit": "dBm",
            "criticality": CriticalityLevel.HIGH,
            "location": "communication_mast",
            "description": "Mars-Earth communication signal sensor",
            "normal_range": (-90.0, -30.0),
            "warning_threshold": -80.0,
            "critical_threshold": -85.0,
            "data_source": "msl",
            "channel_index": 0,
        },
        "MSL-ENV-001": {
            "name": "Environmental Monitor",
            "equipment_id": "ENV-001",
            "sensor_type": "environmental",
            "unit": "celsius",
            "criticality": CriticalityLevel.MEDIUM,
            "location": "rover_deck",
            "description": "Mars environmental conditions sensor",
            "normal_range": (-80.0, 20.0),
            "warning_threshold": -70.0,
            "critical_threshold": -75.0,
            "data_source": "msl",
            "channel_index": 1,
        },
        "MSL-MOB-001": {
            "name": "Mobility System",
            "equipment_id": "MOB-001",
            "sensor_type": "mechanical",
            "unit": "rpm",
            "criticality": CriticalityLevel.HIGH,
            "location": "wheel_assembly",
            "description": "Rover mobility system sensor",
            "normal_range": (0.0, 100.0),
            "warning_threshold": 80.0,
            "critical_threshold": 90.0,
            "data_source": "msl",
            "channel_index": 2,
        },
        "MSL-NAV-001": {
            "name": "Navigation System",
            "equipment_id": "NAV-001",
            "sensor_type": "navigation",
            "unit": "degrees",
            "criticality": CriticalityLevel.CRITICAL,
            "location": "navigation_module",
            "description": "Rover navigation and positioning sensor",
            "normal_range": (0.0, 360.0),
            "warning_threshold": 300.0,
            "critical_threshold": 340.0,
            "data_source": "msl",
            "channel_index": 3,
        },
        "MSL-PWR-001": {
            "name": "RTG Power System",
            "equipment_id": "PWR-001",
            "sensor_type": "power",
            "unit": "watts",
            "criticality": CriticalityLevel.CRITICAL,
            "location": "rtg_module",
            "description": "Radioisotope Thermoelectric Generator power sensor",
            "normal_range": (100.0, 125.0),
            "warning_threshold": 105.0,
            "critical_threshold": 102.0,
            "data_source": "msl",
            "channel_index": 4,
        },
        "MSL-SCI-001": {
            "name": "Science Instruments",
            "equipment_id": "SCI-001",
            "sensor_type": "scientific",
            "unit": "counts",
            "criticality": CriticalityLevel.MEDIUM,
            "location": "science_deck",
            "description": "Science instrument operational sensor",
            "normal_range": (0.0, 1000.0),
            "warning_threshold": 800.0,
            "critical_threshold": 900.0,
            "data_source": "msl",
            "channel_index": 5,
        },
    }


@pytest.fixture(scope="session")
def nasa_sensor_info_objects(nasa_sensor_configs):
    """Fixture providing SensorInfo objects for all 12 NASA sensors"""
    sensor_infos = {}

    for sensor_id, config in nasa_sensor_configs.items():
        sensor_info = SensorInfo(
            sensor_id=sensor_id,
            name=config["name"],
            equipment_id=config["equipment_id"],
            sensor_type=config["sensor_type"],
            unit=config["unit"],
            criticality=config["criticality"],
            location=config["location"],
            description=config["description"],
            normal_range=config["normal_range"],
            warning_threshold=config["warning_threshold"],
            critical_threshold=config["critical_threshold"],
            data_source=config["data_source"],
            channel_index=config["channel_index"],
        )
        sensor_infos[sensor_id] = sensor_info

    return sensor_infos


@pytest.fixture
def sample_time_series_data():
    """Fixture providing sample time series data for testing"""
    base_time = datetime.now() - timedelta(hours=24)
    timestamps = [base_time + timedelta(minutes=i) for i in range(1440)]  # 24 hours of minute data

    return {
        "timestamps": timestamps,
        "normal_pattern": np.sin(np.linspace(0, 4 * np.pi, 1440)) * 10 + 50,  # Sinusoidal pattern
        "trending_up": np.linspace(40, 60, 1440) + np.random.normal(0, 2, 1440),  # Upward trend
        "trending_down": np.linspace(60, 40, 1440) + np.random.normal(0, 2, 1440),  # Downward trend
        "with_anomalies": np.sin(np.linspace(0, 4 * np.pi, 1440)) * 10
        + 50
        + np.where(np.random.random(1440) < 0.02, np.random.normal(0, 20, 1440), 0),  # Normal pattern with 2% anomalies
        "stable": np.random.normal(50, 1, 1440),  # Stable pattern
    }


@pytest.fixture
def nasa_historical_data(nasa_sensor_configs, sample_time_series_data):
    """Fixture providing historical data for all 12 NASA sensors"""
    historical_data = {}

    for sensor_id, config in nasa_sensor_configs.items():
        # Generate realistic data based on sensor type and normal range
        min_val, max_val = config["normal_range"]
        center_val = (min_val + max_val) / 2
        range_val = (max_val - min_val) / 4

        # Base pattern
        base_pattern = sample_time_series_data["normal_pattern"]
        # Scale to sensor's normal range
        scaled_pattern = (base_pattern - 50) * (range_val / 10) + center_val

        # Add sensor-specific characteristics
        if config["sensor_type"] == "temperature":
            # Temperature sensors might have daily cycles
            daily_cycle = np.sin(np.linspace(0, 2 * np.pi, 1440)) * (range_val * 0.3)
            scaled_pattern += daily_cycle
        elif config["sensor_type"] == "power":
            # Power sensors might have more stability
            scaled_pattern = np.random.normal(center_val, range_val * 0.1, 1440)
        elif config["sensor_type"] == "communication":
            # Communication might have more variation
            scaled_pattern += np.random.normal(0, range_val * 0.2, 1440)

        # Ensure values stay within normal range
        scaled_pattern = np.clip(scaled_pattern, min_val, max_val)

        historical_data[sensor_id] = {
            "timestamps": sample_time_series_data["timestamps"],
            "values": scaled_pattern,
            "sensor_config": config,
        }

    return historical_data


@pytest.fixture
def nasa_anomaly_examples(nasa_sensor_configs):
    """Fixture providing example anomalies for NASA sensors"""
    base_time = datetime.now()
    anomalies = []

    # Create realistic anomalies for each sensor type
    for i, (sensor_id, config) in enumerate(nasa_sensor_configs.items()):
        # Create 1-3 anomalies per sensor
        for j in range(1, np.random.randint(2, 4)):
            timestamp = base_time - timedelta(hours=i * 2 + j)

            # Generate anomaly value outside normal range
            min_val, max_val = config["normal_range"]
            if np.random.random() > 0.5:
                # High anomaly
                anomaly_value = max_val + (max_val - min_val) * 0.1
                severity = (
                    AnomalySeverity.HIGH if anomaly_value < config["critical_threshold"] else AnomalySeverity.CRITICAL
                )
            else:
                # Low anomaly
                anomaly_value = min_val - (max_val - min_val) * 0.1
                severity = AnomalySeverity.MEDIUM

            anomaly = AnomalyDetection(
                sensor_id=sensor_id,
                timestamp=timestamp,
                value=anomaly_value,
                score=np.random.uniform(0.7, 0.95),
                severity=severity,
                anomaly_type=AnomalyType.POINT,
                confidence=np.random.uniform(0.8, 0.95),
                description=f"Anomaly detected in {config['name']} - value outside normal range",
                metadata={
                    "mission": config["data_source"].upper(),
                    "sensor_type": config["sensor_type"],
                    "detection_algorithm": "telemanom",
                },
            )
            anomalies.append(anomaly)

    return anomalies


@pytest.fixture
def nasa_forecast_examples(nasa_sensor_configs):
    """Fixture providing example forecasts for NASA sensors"""
    base_time = datetime.now()
    forecasts = {}

    for sensor_id, config in nasa_sensor_configs.items():
        # Generate 12-hour forecast
        forecast_horizon = 12
        min_val, max_val = config["normal_range"]
        center_val = (min_val + max_val) / 2
        range_val = (max_val - min_val) / 6

        # Historical data (last 24 hours)
        historical_timestamps = [base_time - timedelta(hours=i) for i in range(24, 0, -1)]
        historical_values = np.random.normal(center_val, range_val, 24)

        # Forecast data (next 12 hours)
        forecast_timestamps = [base_time + timedelta(hours=i) for i in range(1, forecast_horizon + 1)]
        forecast_values = np.random.normal(center_val, range_val * 0.8, forecast_horizon)

        # Confidence intervals
        confidence_width = range_val * 0.3
        confidence_intervals = {
            "upper": forecast_values + confidence_width,
            "lower": forecast_values - confidence_width,
        }

        # Accuracy metrics
        accuracy_metrics = {
            "mae": np.random.uniform(1.0, 3.0),
            "rmse": np.random.uniform(1.5, 4.0),
            "mape": np.random.uniform(2.0, 8.0),
            "r2": np.random.uniform(0.85, 0.98),
        }

        # Risk assessment
        risk_level = "LOW"
        if any(forecast_values > config["warning_threshold"]):
            risk_level = "MEDIUM"
        if any(forecast_values > config["critical_threshold"]):
            risk_level = "HIGH"

        risk_assessment = {
            "overall_risk": risk_level,
            "mission": config["data_source"].upper(),
            "risk_factors": (["normal_operation"] if risk_level == "LOW" else ["threshold_approaching"]),
            "recommendations": (["continue_monitoring"] if risk_level == "LOW" else ["increase_monitoring_frequency"]),
        }

        forecast = ForecastResult(
            sensor_id=sensor_id,
            forecast_horizon_hours=forecast_horizon,
            historical_timestamps=historical_timestamps,
            historical_values=historical_values,
            forecast_timestamps=forecast_timestamps,
            forecast_values=forecast_values,
            confidence_intervals=confidence_intervals,
            accuracy_metrics=accuracy_metrics,
            model_version="nasa_transformer_v1.0",
            generated_at=base_time,
            risk_assessment=risk_assessment,
        )

        forecasts[sensor_id] = forecast

    return forecasts


@pytest.fixture
def nasa_sensor_readings_batch(nasa_sensor_info_objects, nasa_historical_data):
    """Fixture providing SensorReading batches for all NASA sensors"""
    batches = {}

    for sensor_id, sensor_info in nasa_sensor_info_objects.items():
        historical = nasa_historical_data[sensor_id]

        # Create individual sensor readings
        readings = []
        for timestamp, value in zip(historical["timestamps"][:100], historical["values"][:100]):  # First 100 points
            reading = SensorReading(
                sensor_id=sensor_id,
                timestamp=timestamp,
                value=float(value),
                status=sensor_info.criticality.name,
                metadata={
                    "data_source": sensor_info.data_source,
                    "channel": sensor_info.channel_index,
                },
            )
            readings.append(reading)

        # Create sensor data batch
        batch = SensorDataBatch(
            sensor_id=sensor_id,
            timestamps=historical["timestamps"][:100],
            values=historical["values"][:100],
            sensor_info=sensor_info,
            statistics={
                "count": 100,
                "mean": float(np.mean(historical["values"][:100])),
                "std": float(np.std(historical["values"][:100])),
                "min": float(np.min(historical["values"][:100])),
                "max": float(np.max(historical["values"][:100])),
            },
            quality_score=np.random.uniform(0.85, 0.98),
        )

        batches[sensor_id] = {"readings": readings, "batch": batch}

    return batches


@pytest.fixture
def nasa_test_scenarios():
    """Fixture providing various test scenarios for NASA sensor testing"""
    return {
        "normal_operation": {
            "description": "All sensors operating within normal parameters",
            "sensor_status": "normal",
            "expected_anomalies": 0,
            "expected_risk_level": "LOW",
        },
        "communication_degradation": {
            "description": "Communication sensors showing signal degradation",
            "affected_sensors": ["SMAP-COM-001", "MSL-COM-001"],
            "sensor_status": "warning",
            "expected_anomalies": 2,
            "expected_risk_level": "MEDIUM",
        },
        "power_system_alert": {
            "description": "Power systems approaching critical thresholds",
            "affected_sensors": ["SMAP-PWR-001", "MSL-PWR-001"],
            "sensor_status": "critical",
            "expected_anomalies": 2,
            "expected_risk_level": "HIGH",
        },
        "thermal_management_issue": {
            "description": "Thermal management systems showing elevated temperatures",
            "affected_sensors": ["SMAP-THM-001", "SMAP-PAY-001", "MSL-ENV-001"],
            "sensor_status": "warning",
            "expected_anomalies": 3,
            "expected_risk_level": "MEDIUM",
        },
        "mission_critical_failure": {
            "description": "Critical system failure scenario",
            "affected_sensors": ["SMAP-PWR-001", "MSL-NAV-001", "MSL-PWR-001"],
            "sensor_status": "critical",
            "expected_anomalies": 5,
            "expected_risk_level": "CRITICAL",
        },
    }


@pytest.fixture
def create_test_data_file():
    """Fixture to create and save test data files"""

    def _create_file(sensor_id: str, data: Dict[str, Any], filename: str = None):
        """Create a test data file for a specific sensor"""
        if filename is None:
            filename = f"test_data_{sensor_id.lower().replace('-', '_')}.json"

        filepath = f"tests/fixtures/{filename}"

        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                serializable_data[key] = value.tolist()
            elif isinstance(value, datetime):
                serializable_data[key] = value.isoformat()
            elif isinstance(value, list) and value and isinstance(value[0], datetime):
                serializable_data[key] = [dt.isoformat() for dt in value]
            else:
                serializable_data[key] = value

        with open(filepath, "w") as f:
            json.dump(serializable_data, f, indent=2)

        return filepath

    return _create_file

"""
Unit tests for core sensor data models
Session 1: Foundation & Environment Setup
"""

from dataclasses import asdict
from datetime import datetime, timedelta

import numpy as np
import pytest

from src.core.models.sensor_data import (
    CriticalityLevel,
    SensorDataBatch,
    SensorInfo,
    SensorReading,
    SensorStatus,
)

# Test markers
pytestmark = [pytest.mark.session1, pytest.mark.unit, pytest.mark.core]


class TestSensorStatus:
    """Test SensorStatus enum"""

    def test_sensor_status_values(self):
        """Test that all sensor status values are defined correctly"""
        assert SensorStatus.NORMAL.value == "NORMAL"
        assert SensorStatus.WARNING.value == "WARNING"
        assert SensorStatus.CRITICAL.value == "CRITICAL"
        assert SensorStatus.UNKNOWN.value == "UNKNOWN"

    def test_sensor_status_count(self):
        """Test that we have exactly 4 status values"""
        assert len(SensorStatus) == 4


class TestCriticalityLevel:
    """Test CriticalityLevel enum"""

    def test_criticality_level_values(self):
        """Test that all criticality levels are defined correctly"""
        assert CriticalityLevel.LOW.value == "LOW"
        assert CriticalityLevel.MEDIUM.value == "MEDIUM"
        assert CriticalityLevel.HIGH.value == "HIGH"
        assert CriticalityLevel.CRITICAL.value == "CRITICAL"

    def test_criticality_level_count(self):
        """Test that we have exactly 4 criticality levels"""
        assert len(CriticalityLevel) == 4


class TestSensorReading:
    """Test SensorReading dataclass"""

    def test_sensor_reading_creation(self):
        """Test basic sensor reading creation"""
        timestamp = datetime.now()
        reading = SensorReading(sensor_id="SMAP-ATT-001", timestamp=timestamp, value=42.5)

        assert reading.sensor_id == "SMAP-ATT-001"
        assert reading.timestamp == timestamp
        assert reading.value == 42.5
        assert reading.status == SensorStatus.NORMAL
        assert reading.metadata == {}

    def test_sensor_reading_with_status(self):
        """Test sensor reading with specific status"""
        reading = SensorReading(
            sensor_id="SMAP-ATT-001",
            timestamp=datetime.now(),
            value=85.0,
            status=SensorStatus.WARNING,
        )

        assert reading.status == SensorStatus.WARNING

    def test_sensor_reading_with_metadata(self):
        """Test sensor reading with metadata"""
        metadata = {"quality": "high", "source": "primary"}
        reading = SensorReading(
            sensor_id="SMAP-ATT-001",
            timestamp=datetime.now(),
            value=42.5,
            metadata=metadata,
        )

        assert reading.metadata == metadata

    def test_sensor_reading_metadata_initialization(self):
        """Test that metadata is initialized to empty dict when None"""
        reading = SensorReading(
            sensor_id="SMAP-ATT-001",
            timestamp=datetime.now(),
            value=42.5,
            metadata=None,
        )

        assert reading.metadata == {}

    def test_sensor_reading_serialization(self):
        """Test that sensor reading can be converted to dict"""
        timestamp = datetime.now()
        reading = SensorReading(sensor_id="SMAP-ATT-001", timestamp=timestamp, value=42.5)

        reading_dict = asdict(reading)
        assert reading_dict["sensor_id"] == "SMAP-ATT-001"
        assert reading_dict["timestamp"] == timestamp
        assert reading_dict["value"] == 42.5


class TestSensorInfo:
    """Test SensorInfo dataclass"""

    def test_sensor_info_creation(self):
        """Test basic sensor info creation"""
        sensor_info = SensorInfo(
            sensor_id="SMAP-ATT-001",
            name="Attitude Control System",
            equipment_id="ATT-001",
            sensor_type="attitude",
            unit="degrees",
            criticality=CriticalityLevel.HIGH,
            location="spacecraft_body",
            description="Primary attitude sensor",
            normal_range=(0.0, 100.0),
            warning_threshold=80.0,
            critical_threshold=95.0,
            data_source="smap",
            channel_index=0,
        )

        assert sensor_info.sensor_id == "SMAP-ATT-001"
        assert sensor_info.name == "Attitude Control System"
        assert sensor_info.criticality == CriticalityLevel.HIGH
        assert sensor_info.normal_range == (0.0, 100.0)
        assert sensor_info.data_source == "smap"
        assert sensor_info.channel_index == 0

    def test_sensor_info_criticality_types(self):
        """Test different criticality levels"""
        for criticality in CriticalityLevel:
            sensor_info = SensorInfo(
                sensor_id="TEST-001",
                name="Test Sensor",
                equipment_id="TEST-001",
                sensor_type="test",
                unit="units",
                criticality=criticality,
                location="test",
                description="Test sensor",
                normal_range=(0.0, 100.0),
                warning_threshold=80.0,
                critical_threshold=95.0,
                data_source="test",
                channel_index=0,
            )
            assert sensor_info.criticality == criticality

    def test_sensor_info_data_sources(self):
        """Test different data sources"""
        for data_source in ["smap", "msl"]:
            sensor_info = SensorInfo(
                sensor_id=f"{data_source.upper()}-TEST-001",
                name="Test Sensor",
                equipment_id="TEST-001",
                sensor_type="test",
                unit="units",
                criticality=CriticalityLevel.MEDIUM,
                location="test",
                description="Test sensor",
                normal_range=(0.0, 100.0),
                warning_threshold=80.0,
                critical_threshold=95.0,
                data_source=data_source,
                channel_index=0,
            )
            assert sensor_info.data_source == data_source


class TestSensorDataBatch:
    """Test SensorDataBatch dataclass"""

    def test_sensor_data_batch_creation(self, sample_sensor_data):
        """Test basic sensor data batch creation"""
        timestamps = [datetime.now() - timedelta(minutes=i) for i in range(10, 0, -1)]
        values = np.random.normal(50, 10, 10)

        sensor_info = SensorInfo(
            sensor_id="SMAP-ATT-001",
            name="Attitude Control System",
            equipment_id="ATT-001",
            sensor_type="attitude",
            unit="degrees",
            criticality=CriticalityLevel.HIGH,
            location="spacecraft_body",
            description="Primary attitude sensor",
            normal_range=(0.0, 100.0),
            warning_threshold=80.0,
            critical_threshold=95.0,
            data_source="smap",
            channel_index=0,
        )

        statistics = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

        batch = SensorDataBatch(
            sensor_id="SMAP-ATT-001",
            timestamps=timestamps,
            values=values,
            sensor_info=sensor_info,
            statistics=statistics,
        )

        assert batch.sensor_id == "SMAP-ATT-001"
        assert len(batch.timestamps) == 10
        assert len(batch.values) == 10
        assert batch.sensor_info.sensor_id == "SMAP-ATT-001"
        assert batch.quality_score == 1.0
        assert "mean" in batch.statistics

    def test_sensor_data_batch_quality_score(self):
        """Test sensor data batch with custom quality score"""
        timestamps = [datetime.now()]
        values = np.array([50.0])

        sensor_info = SensorInfo(
            sensor_id="TEST-001",
            name="Test Sensor",
            equipment_id="TEST-001",
            sensor_type="test",
            unit="units",
            criticality=CriticalityLevel.LOW,
            location="test",
            description="Test sensor",
            normal_range=(0.0, 100.0),
            warning_threshold=80.0,
            critical_threshold=95.0,
            data_source="test",
            channel_index=0,
        )

        batch = SensorDataBatch(
            sensor_id="TEST-001",
            timestamps=timestamps,
            values=values,
            sensor_info=sensor_info,
            statistics={},
            quality_score=0.85,
        )

        assert batch.quality_score == 0.85

    def test_sensor_data_batch_statistics_calculation(self):
        """Test that statistics are correctly calculated"""
        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        expected_mean = 30.0
        expected_std = np.std(values)

        sensor_info = SensorInfo(
            sensor_id="TEST-001",
            name="Test Sensor",
            equipment_id="TEST-001",
            sensor_type="test",
            unit="units",
            criticality=CriticalityLevel.LOW,
            location="test",
            description="Test sensor",
            normal_range=(0.0, 100.0),
            warning_threshold=80.0,
            critical_threshold=95.0,
            data_source="test",
            channel_index=0,
        )

        statistics = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

        batch = SensorDataBatch(
            sensor_id="TEST-001",
            timestamps=[datetime.now()] * 5,
            values=values,
            sensor_info=sensor_info,
            statistics=statistics,
        )

        assert batch.statistics["mean"] == expected_mean
        assert batch.statistics["min"] == 10.0
        assert batch.statistics["max"] == 50.0
        assert abs(batch.statistics["std"] - expected_std) < 1e-10


class TestSensorDataIntegration:
    """Integration tests for sensor data models"""

    def test_sensor_reading_to_batch_conversion(self):
        """Test converting multiple sensor readings to a batch"""
        readings = []
        timestamps = []
        values = []

        for i in range(5):
            timestamp = datetime.now() - timedelta(minutes=i)
            value = 50.0 + i * 2.0

            reading = SensorReading(sensor_id="SMAP-ATT-001", timestamp=timestamp, value=value)

            readings.append(reading)
            timestamps.append(timestamp)
            values.append(value)

        # Create batch from readings
        sensor_info = SensorInfo(
            sensor_id="SMAP-ATT-001",
            name="Attitude Control System",
            equipment_id="ATT-001",
            sensor_type="attitude",
            unit="degrees",
            criticality=CriticalityLevel.HIGH,
            location="spacecraft_body",
            description="Primary attitude sensor",
            normal_range=(0.0, 100.0),
            warning_threshold=80.0,
            critical_threshold=95.0,
            data_source="smap",
            channel_index=0,
        )

        batch = SensorDataBatch(
            sensor_id="SMAP-ATT-001",
            timestamps=timestamps,
            values=np.array(values),
            sensor_info=sensor_info,
            statistics={"count": len(readings)},
        )

        assert len(batch.timestamps) == len(readings)
        assert len(batch.values) == len(readings)
        assert batch.statistics["count"] == 5

    def test_nasa_sensor_configurations(self, sample_sensors):
        """Test creating sensor info for NASA sensors"""
        for sensor_id in sample_sensors[:6]:  # Test first 6 SMAP sensors
            data_source = "smap" if sensor_id.startswith("SMAP") else "msl"
            channel_index = int(sensor_id.split("-")[-1]) if sensor_id.split("-")[-1].isdigit() else 0

            sensor_info = SensorInfo(
                sensor_id=sensor_id,
                name=f"{sensor_id} Sensor",
                equipment_id=sensor_id,
                sensor_type="telemetry",
                unit="units",
                criticality=CriticalityLevel.HIGH,
                location="spacecraft",
                description=f"NASA {data_source.upper()} sensor",
                normal_range=(0.0, 100.0),
                warning_threshold=80.0,
                critical_threshold=95.0,
                data_source=data_source,
                channel_index=channel_index,
            )

            assert sensor_info.sensor_id == sensor_id
            assert sensor_info.data_source == data_source
            assert sensor_info.description.startswith("NASA")

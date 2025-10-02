"""
Unit tests for core anomaly detection models
Session 1: Foundation & Environment Setup
"""

from dataclasses import asdict
from datetime import datetime, timedelta

import numpy as np
import pytest

from src.core.models.anomaly import (
    AnomalyDetection,
    AnomalyDetectionResult,
    AnomalySeverity,
    AnomalySummary,
    AnomalyType,
)

# Test markers
pytestmark = [pytest.mark.session1, pytest.mark.unit, pytest.mark.core]


class TestAnomalySeverity:
    """Test AnomalySeverity enum"""

    def test_anomaly_severity_values(self):
        """Test that all anomaly severity values are defined correctly"""
        assert AnomalySeverity.LOW.value == "LOW"
        assert AnomalySeverity.MEDIUM.value == "MEDIUM"
        assert AnomalySeverity.HIGH.value == "HIGH"
        assert AnomalySeverity.CRITICAL.value == "CRITICAL"

    def test_anomaly_severity_count(self):
        """Test that we have exactly 4 severity levels"""
        assert len(AnomalySeverity) == 4


class TestAnomalyType:
    """Test AnomalyType enum"""

    def test_anomaly_type_values(self):
        """Test that all anomaly types are defined correctly"""
        assert AnomalyType.POINT.value == "POINT"
        assert AnomalyType.CONTEXTUAL.value == "CONTEXTUAL"
        assert AnomalyType.COLLECTIVE.value == "COLLECTIVE"

    def test_anomaly_type_count(self):
        """Test that we have exactly 3 anomaly types"""
        assert len(AnomalyType) == 3


class TestAnomalyDetection:
    """Test AnomalyDetection dataclass"""

    def test_anomaly_detection_creation(self):
        """Test basic anomaly detection creation"""
        timestamp = datetime.now()
        anomaly = AnomalyDetection(
            sensor_id="SMAP-ATT-001",
            timestamp=timestamp,
            value=95.5,
            score=0.85,
            severity=AnomalySeverity.HIGH,
            anomaly_type=AnomalyType.POINT,
        )

        assert anomaly.sensor_id == "SMAP-ATT-001"
        assert anomaly.timestamp == timestamp
        assert anomaly.value == 95.5
        assert anomaly.score == 0.85
        assert anomaly.severity == AnomalySeverity.HIGH
        assert anomaly.anomaly_type == AnomalyType.POINT
        assert anomaly.confidence == 0.0
        assert anomaly.description == ""
        assert anomaly.metadata == {}

    def test_anomaly_detection_with_confidence(self):
        """Test anomaly detection with confidence value"""
        anomaly = AnomalyDetection(
            sensor_id="SMAP-ATT-001",
            timestamp=datetime.now(),
            value=95.5,
            score=0.85,
            severity=AnomalySeverity.HIGH,
            anomaly_type=AnomalyType.POINT,
            confidence=0.92,
        )

        assert anomaly.confidence == 0.92

    def test_anomaly_detection_with_description(self):
        """Test anomaly detection with description"""
        description = "Sudden spike in attitude sensor reading"
        anomaly = AnomalyDetection(
            sensor_id="SMAP-ATT-001",
            timestamp=datetime.now(),
            value=95.5,
            score=0.85,
            severity=AnomalySeverity.HIGH,
            anomaly_type=AnomalyType.POINT,
            description=description,
        )

        assert anomaly.description == description

    def test_anomaly_detection_with_metadata(self):
        """Test anomaly detection with metadata"""
        metadata = {"algorithm": "telemanom", "threshold": 0.8}
        anomaly = AnomalyDetection(
            sensor_id="SMAP-ATT-001",
            timestamp=datetime.now(),
            value=95.5,
            score=0.85,
            severity=AnomalySeverity.HIGH,
            anomaly_type=AnomalyType.POINT,
            metadata=metadata,
        )

        assert anomaly.metadata == metadata

    def test_anomaly_detection_metadata_initialization(self):
        """Test that metadata is initialized to empty dict when None"""
        anomaly = AnomalyDetection(
            sensor_id="SMAP-ATT-001",
            timestamp=datetime.now(),
            value=95.5,
            score=0.85,
            severity=AnomalySeverity.HIGH,
            anomaly_type=AnomalyType.POINT,
            metadata=None,
        )

        assert anomaly.metadata == {}

    def test_anomaly_detection_score_validation(self):
        """Test anomaly detection with various score values"""
        for score in [0.0, 0.5, 0.8, 1.0]:
            anomaly = AnomalyDetection(
                sensor_id="SMAP-ATT-001",
                timestamp=datetime.now(),
                value=95.5,
                score=score,
                severity=AnomalySeverity.HIGH,
                anomaly_type=AnomalyType.POINT,
            )
            assert anomaly.score == score

    def test_anomaly_detection_all_severity_levels(self):
        """Test anomaly detection with all severity levels"""
        for severity in AnomalySeverity:
            anomaly = AnomalyDetection(
                sensor_id="SMAP-ATT-001",
                timestamp=datetime.now(),
                value=95.5,
                score=0.85,
                severity=severity,
                anomaly_type=AnomalyType.POINT,
            )
            assert anomaly.severity == severity

    def test_anomaly_detection_all_types(self):
        """Test anomaly detection with all anomaly types"""
        for anomaly_type in AnomalyType:
            anomaly = AnomalyDetection(
                sensor_id="SMAP-ATT-001",
                timestamp=datetime.now(),
                value=95.5,
                score=0.85,
                severity=AnomalySeverity.HIGH,
                anomaly_type=anomaly_type,
            )
            assert anomaly.anomaly_type == anomaly_type


class TestAnomalyDetectionResult:
    """Test AnomalyDetectionResult dataclass"""

    def test_anomaly_detection_result_creation(self):
        """Test basic anomaly detection result creation"""
        timestamp = datetime.now()
        anomalies = [
            AnomalyDetection(
                sensor_id="SMAP-ATT-001",
                timestamp=timestamp,
                value=95.5,
                score=0.85,
                severity=AnomalySeverity.HIGH,
                anomaly_type=AnomalyType.POINT,
            )
        ]

        statistics = {"total_points": 1000, "anomaly_rate": 0.001, "avg_score": 0.85}

        result = AnomalyDetectionResult(
            sensor_id="SMAP-ATT-001",
            data_processed=1000,
            anomalies_detected=anomalies,
            processing_time=2.5,
            model_version="telemanom_v1.0",
            detection_timestamp=timestamp,
            statistics=statistics,
        )

        assert result.sensor_id == "SMAP-ATT-001"
        assert result.data_processed == 1000
        assert len(result.anomalies_detected) == 1
        assert result.processing_time == 2.5
        assert result.model_version == "telemanom_v1.0"
        assert result.statistics["total_points"] == 1000

    def test_anomaly_detection_result_empty_anomalies(self):
        """Test anomaly detection result with no anomalies"""
        result = AnomalyDetectionResult(
            sensor_id="SMAP-ATT-001",
            data_processed=1000,
            anomalies_detected=[],
            processing_time=1.2,
            model_version="telemanom_v1.0",
            detection_timestamp=datetime.now(),
            statistics={"total_points": 1000},
        )

        assert len(result.anomalies_detected) == 0
        assert result.data_processed == 1000

    def test_anomaly_detection_result_multiple_anomalies(self):
        """Test anomaly detection result with multiple anomalies"""
        timestamp = datetime.now()
        anomalies = []

        for i in range(5):
            anomaly = AnomalyDetection(
                sensor_id="SMAP-ATT-001",
                timestamp=timestamp + timedelta(minutes=i),
                value=90.0 + i,
                score=0.8 + i * 0.02,
                severity=AnomalySeverity.HIGH,
                anomaly_type=AnomalyType.POINT,
            )
            anomalies.append(anomaly)

        result = AnomalyDetectionResult(
            sensor_id="SMAP-ATT-001",
            data_processed=1000,
            anomalies_detected=anomalies,
            processing_time=3.0,
            model_version="telemanom_v1.0",
            detection_timestamp=timestamp,
            statistics={"total_points": 1000, "anomaly_count": 5},
        )

        assert len(result.anomalies_detected) == 5
        assert result.statistics["anomaly_count"] == 5


class TestAnomalySummary:
    """Test AnomalySummary dataclass"""

    def test_anomaly_summary_creation(self):
        """Test basic anomaly summary creation"""
        timestamp = datetime.now()

        recent_anomalies = [
            AnomalyDetection(
                sensor_id="SMAP-ATT-001",
                timestamp=timestamp,
                value=95.5,
                score=0.85,
                severity=AnomalySeverity.HIGH,
                anomaly_type=AnomalyType.POINT,
            ),
            AnomalyDetection(
                sensor_id="MSL-COM-001",
                timestamp=timestamp,
                value=88.0,
                score=0.75,
                severity=AnomalySeverity.MEDIUM,
                anomaly_type=AnomalyType.CONTEXTUAL,
            ),
        ]

        severity_breakdown = {"LOW": 2, "MEDIUM": 3, "HIGH": 4, "CRITICAL": 1}

        sensor_stats = {
            "SMAP-ATT-001": {"anomaly_count": 5, "avg_score": 0.82},
            "MSL-COM-001": {"anomaly_count": 3, "avg_score": 0.76},
        }

        summary = AnomalySummary(
            total_anomalies=10,
            severity_breakdown=severity_breakdown,
            recent_anomalies=recent_anomalies,
            sensor_stats=sensor_stats,
            generated_at=timestamp,
        )

        assert summary.total_anomalies == 10
        assert summary.severity_breakdown["HIGH"] == 4
        assert len(summary.recent_anomalies) == 2
        assert "SMAP-ATT-001" in summary.sensor_stats
        assert summary.generated_at == timestamp

    def test_anomaly_summary_severity_breakdown_validation(self):
        """Test that severity breakdown contains correct keys"""
        severity_breakdown = {"LOW": 2, "MEDIUM": 3, "HIGH": 4, "CRITICAL": 1}

        summary = AnomalySummary(
            total_anomalies=10,
            severity_breakdown=severity_breakdown,
            recent_anomalies=[],
            sensor_stats={},
            generated_at=datetime.now(),
        )

        expected_keys = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        assert set(summary.severity_breakdown.keys()) == expected_keys

    def test_anomaly_summary_empty_recent_anomalies(self):
        """Test anomaly summary with no recent anomalies"""
        summary = AnomalySummary(
            total_anomalies=0,
            severity_breakdown={"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0},
            recent_anomalies=[],
            sensor_stats={},
            generated_at=datetime.now(),
        )

        assert summary.total_anomalies == 0
        assert len(summary.recent_anomalies) == 0
        assert len(summary.sensor_stats) == 0


class TestAnomalyModelIntegration:
    """Integration tests for anomaly models"""

    def test_anomaly_detection_to_summary_aggregation(self, sample_sensors):
        """Test aggregating anomaly detections into a summary"""
        timestamp = datetime.now()
        all_anomalies = []
        sensor_stats = {}

        # Create anomalies for first 3 sensors
        for i, sensor_id in enumerate(sample_sensors[:3]):
            anomaly_count = i + 1
            sensor_anomalies = []

            for j in range(anomaly_count):
                anomaly = AnomalyDetection(
                    sensor_id=sensor_id,
                    timestamp=timestamp + timedelta(minutes=j),
                    value=90.0 + j,
                    score=0.8 + j * 0.05,
                    severity=(
                        AnomalySeverity.HIGH if j % 2 == 0 else AnomalySeverity.MEDIUM
                    ),
                    anomaly_type=AnomalyType.POINT,
                )
                sensor_anomalies.append(anomaly)
                all_anomalies.append(anomaly)

            sensor_stats[sensor_id] = {
                "anomaly_count": anomaly_count,
                "avg_score": np.mean([a.score for a in sensor_anomalies]),
            }

        # Calculate severity breakdown
        severity_breakdown = {
            "LOW": 0,
            "MEDIUM": sum(
                1 for a in all_anomalies if a.severity == AnomalySeverity.MEDIUM
            ),
            "HIGH": sum(1 for a in all_anomalies if a.severity == AnomalySeverity.HIGH),
            "CRITICAL": 0,
        }

        summary = AnomalySummary(
            total_anomalies=len(all_anomalies),
            severity_breakdown=severity_breakdown,
            recent_anomalies=all_anomalies[-5:],  # Last 5 anomalies
            sensor_stats=sensor_stats,
            generated_at=timestamp,
        )

        assert summary.total_anomalies == 6  # 1 + 2 + 3
        assert len(summary.sensor_stats) == 3
        assert (
            summary.severity_breakdown["MEDIUM"] + summary.severity_breakdown["HIGH"]
            == 6
        )

    def test_nasa_sensor_anomaly_detection(self, sample_sensors):
        """Test anomaly detection for NASA sensors"""
        timestamp = datetime.now()

        for sensor_id in sample_sensors[:6]:  # Test SMAP sensors
            anomaly = AnomalyDetection(
                sensor_id=sensor_id,
                timestamp=timestamp,
                value=95.5,
                score=0.85,
                severity=AnomalySeverity.HIGH,
                anomaly_type=AnomalyType.POINT,
                description=f"Anomaly detected in {sensor_id}",
                metadata={"mission": "SMAP" if "SMAP" in sensor_id else "MSL"},
            )

            assert anomaly.sensor_id == sensor_id
            assert (
                "SMAP" in anomaly.metadata["mission"]
                or "MSL" in anomaly.metadata["mission"]
            )
            assert anomaly.description.endswith(sensor_id)

    def test_anomaly_severity_escalation(self):
        """Test scenarios where anomaly severity might escalate"""
        base_anomaly = AnomalyDetection(
            sensor_id="SMAP-ATT-001",
            timestamp=datetime.now(),
            value=95.5,
            score=0.85,
            severity=AnomalySeverity.MEDIUM,
            anomaly_type=AnomalyType.POINT,
        )

        # Simulate escalation by creating new anomaly with higher severity
        escalated_anomaly = AnomalyDetection(
            sensor_id=base_anomaly.sensor_id,
            timestamp=base_anomaly.timestamp + timedelta(minutes=5),
            value=98.0,
            score=0.95,
            severity=AnomalySeverity.CRITICAL,
            anomaly_type=AnomalyType.POINT,
            description="Escalated from previous medium severity anomaly",
        )

        # Check that severity escalated (CRITICAL > MEDIUM)
        severity_order = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        assert (
            severity_order[escalated_anomaly.severity.value]
            > severity_order[base_anomaly.severity.value]
        )
        assert escalated_anomaly.score > base_anomaly.score

"""
Unit tests for core forecast models
Session 1: Foundation & Environment Setup
"""

from dataclasses import asdict
from datetime import datetime, timedelta

import numpy as np
import pytest

from src.core.models.forecast import (
    ForecastConfidence,
    ForecastPoint,
    ForecastResult,
    ForecastSummary,
    RiskLevel,
)

# Test markers
pytestmark = [pytest.mark.session1, pytest.mark.unit, pytest.mark.core]


class TestForecastConfidence:
    """Test ForecastConfidence enum"""

    def test_forecast_confidence_values(self):
        """Test that all forecast confidence values are defined correctly"""
        assert ForecastConfidence.LOW.value == "LOW"
        assert ForecastConfidence.MEDIUM.value == "MEDIUM"
        assert ForecastConfidence.HIGH.value == "HIGH"

    def test_forecast_confidence_count(self):
        """Test that we have exactly 3 confidence levels"""
        assert len(ForecastConfidence) == 3


class TestRiskLevel:
    """Test RiskLevel enum"""

    def test_risk_level_values(self):
        """Test that all risk level values are defined correctly"""
        assert RiskLevel.LOW.value == "LOW"
        assert RiskLevel.MEDIUM.value == "MEDIUM"
        assert RiskLevel.HIGH.value == "HIGH"
        assert RiskLevel.CRITICAL.value == "CRITICAL"

    def test_risk_level_count(self):
        """Test that we have exactly 4 risk levels"""
        assert len(RiskLevel) == 4


class TestForecastPoint:
    """Test ForecastPoint dataclass"""

    def test_forecast_point_creation(self):
        """Test basic forecast point creation"""
        timestamp = datetime.now() + timedelta(hours=1)
        point = ForecastPoint(
            timestamp=timestamp,
            predicted_value=55.2,
            confidence_lower=50.1,
            confidence_upper=60.3,
            confidence_level=ForecastConfidence.HIGH,
            risk_level=RiskLevel.LOW,
        )

        assert point.timestamp == timestamp
        assert point.predicted_value == 55.2
        assert point.confidence_lower == 50.1
        assert point.confidence_upper == 60.3
        assert point.confidence_level == ForecastConfidence.HIGH
        assert point.risk_level == RiskLevel.LOW

    def test_forecast_point_confidence_interval(self):
        """Test that confidence interval is valid"""
        point = ForecastPoint(
            timestamp=datetime.now() + timedelta(hours=1),
            predicted_value=55.2,
            confidence_lower=50.1,
            confidence_upper=60.3,
            confidence_level=ForecastConfidence.HIGH,
            risk_level=RiskLevel.LOW,
        )

        # Confidence interval should be valid
        assert point.confidence_lower < point.predicted_value
        assert point.predicted_value < point.confidence_upper
        assert point.confidence_lower < point.confidence_upper

    def test_forecast_point_all_confidence_levels(self):
        """Test forecast point with all confidence levels"""
        for confidence in ForecastConfidence:
            point = ForecastPoint(
                timestamp=datetime.now() + timedelta(hours=1),
                predicted_value=55.2,
                confidence_lower=50.1,
                confidence_upper=60.3,
                confidence_level=confidence,
                risk_level=RiskLevel.LOW,
            )
            assert point.confidence_level == confidence

    def test_forecast_point_all_risk_levels(self):
        """Test forecast point with all risk levels"""
        for risk in RiskLevel:
            point = ForecastPoint(
                timestamp=datetime.now() + timedelta(hours=1),
                predicted_value=55.2,
                confidence_lower=50.1,
                confidence_upper=60.3,
                confidence_level=ForecastConfidence.HIGH,
                risk_level=risk,
            )
            assert point.risk_level == risk


class TestForecastResult:
    """Test ForecastResult dataclass"""

    def test_forecast_result_creation(self):
        """Test basic forecast result creation"""
        now = datetime.now()

        # Historical data
        historical_timestamps = [now - timedelta(hours=i) for i in range(24, 0, -1)]
        historical_values = np.random.normal(50, 10, 24)

        # Forecast data
        forecast_timestamps = [now + timedelta(hours=i) for i in range(1, 13)]
        forecast_values = np.random.normal(55, 8, 12)

        confidence_intervals = {
            "upper": forecast_values + 5,
            "lower": forecast_values - 5,
        }

        accuracy_metrics = {"mae": 2.5, "rmse": 3.2, "mape": 5.1}

        risk_assessment = {
            "overall_risk": "LOW",
            "risk_factors": ["minor_drift"],
            "recommendations": ["continue_monitoring"],
        }

        result = ForecastResult(
            sensor_id="SMAP-ATT-001",
            forecast_horizon_hours=12,
            historical_timestamps=historical_timestamps,
            historical_values=historical_values,
            forecast_timestamps=forecast_timestamps,
            forecast_values=forecast_values,
            confidence_intervals=confidence_intervals,
            accuracy_metrics=accuracy_metrics,
            model_version="transformer_v1.0",
            generated_at=now,
            risk_assessment=risk_assessment,
        )

        assert result.sensor_id == "SMAP-ATT-001"
        assert result.forecast_horizon_hours == 12
        assert len(result.historical_timestamps) == 24
        assert len(result.forecast_timestamps) == 12
        assert "upper" in result.confidence_intervals
        assert "lower" in result.confidence_intervals
        assert result.accuracy_metrics["mae"] == 2.5
        assert result.model_version == "transformer_v1.0"

    def test_forecast_result_data_consistency(self):
        """Test that forecast result data arrays are consistent"""
        now = datetime.now()

        historical_timestamps = [now - timedelta(hours=i) for i in range(10, 0, -1)]
        historical_values = np.random.normal(50, 10, 10)
        forecast_timestamps = [now + timedelta(hours=i) for i in range(1, 6)]
        forecast_values = np.random.normal(55, 8, 5)

        result = ForecastResult(
            sensor_id="SMAP-ATT-001",
            forecast_horizon_hours=5,
            historical_timestamps=historical_timestamps,
            historical_values=historical_values,
            forecast_timestamps=forecast_timestamps,
            forecast_values=forecast_values,
            confidence_intervals={
                "upper": forecast_values + 2,
                "lower": forecast_values - 2,
            },
            accuracy_metrics={"mae": 1.0},
            model_version="test_v1.0",
            generated_at=now,
            risk_assessment={},
        )

        # Check data consistency
        assert len(result.historical_timestamps) == len(result.historical_values)
        assert len(result.forecast_timestamps) == len(result.forecast_values)
        assert result.forecast_horizon_hours == len(result.forecast_timestamps)

    def test_forecast_result_confidence_intervals(self):
        """Test forecast result confidence intervals"""
        now = datetime.now()
        forecast_values = np.array([50.0, 52.0, 54.0, 56.0, 58.0])

        confidence_intervals = {
            "upper": forecast_values + 3,
            "lower": forecast_values - 3,
        }

        result = ForecastResult(
            sensor_id="SMAP-ATT-001",
            forecast_horizon_hours=5,
            historical_timestamps=[now - timedelta(hours=1)],
            historical_values=np.array([48.0]),
            forecast_timestamps=[now + timedelta(hours=i) for i in range(1, 6)],
            forecast_values=forecast_values,
            confidence_intervals=confidence_intervals,
            accuracy_metrics={},
            model_version="test_v1.0",
            generated_at=now,
            risk_assessment={},
        )

        # Validate confidence intervals
        upper = result.confidence_intervals["upper"]
        lower = result.confidence_intervals["lower"]

        for i in range(len(forecast_values)):
            assert lower[i] < forecast_values[i] < upper[i]


class TestForecastSummary:
    """Test ForecastSummary dataclass"""

    def test_forecast_summary_creation(self):
        """Test basic forecast summary creation"""
        now = datetime.now()

        # Create sample forecast results
        recent_forecasts = []
        for i, sensor_id in enumerate(["SMAP-ATT-001", "MSL-COM-001"]):
            forecast = ForecastResult(
                sensor_id=sensor_id,
                forecast_horizon_hours=12,
                historical_timestamps=[
                    now - timedelta(hours=j) for j in range(24, 0, -1)
                ],
                historical_values=np.random.normal(50, 10, 24),
                forecast_timestamps=[now + timedelta(hours=j) for j in range(1, 13)],
                forecast_values=np.random.normal(55, 8, 12),
                confidence_intervals={
                    "upper": np.ones(12) * 60,
                    "lower": np.ones(12) * 50,
                },
                accuracy_metrics={"mae": 2.0 + i, "rmse": 3.0 + i},
                model_version="transformer_v1.0",
                generated_at=now,
                risk_assessment={"overall_risk": "LOW"},
            )
            recent_forecasts.append(forecast)

        risk_distribution = {"LOW": 8, "MEDIUM": 3, "HIGH": 1, "CRITICAL": 0}

        model_performance = {
            "transformer_v1.0": {"mae": 2.1, "rmse": 3.1, "accuracy": 0.95},
            "lstm_v1.0": {"mae": 2.8, "rmse": 3.8, "accuracy": 0.92},
        }

        summary = ForecastSummary(
            total_sensors_forecasted=12,
            average_confidence=0.85,
            risk_distribution=risk_distribution,
            recent_forecasts=recent_forecasts,
            model_performance=model_performance,
            generated_at=now,
        )

        assert summary.total_sensors_forecasted == 12
        assert summary.average_confidence == 0.85
        assert summary.risk_distribution["LOW"] == 8
        assert len(summary.recent_forecasts) == 2
        assert "transformer_v1.0" in summary.model_performance
        assert summary.generated_at == now

    def test_forecast_summary_risk_distribution_validation(self):
        """Test that risk distribution contains correct keys"""
        risk_distribution = {"LOW": 8, "MEDIUM": 3, "HIGH": 1, "CRITICAL": 0}

        summary = ForecastSummary(
            total_sensors_forecasted=12,
            average_confidence=0.85,
            risk_distribution=risk_distribution,
            recent_forecasts=[],
            model_performance={},
            generated_at=datetime.now(),
        )

        expected_keys = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        assert set(summary.risk_distribution.keys()) == expected_keys

    def test_forecast_summary_model_performance_metrics(self):
        """Test model performance metrics in forecast summary"""
        model_performance = {
            "transformer_v1.0": {
                "mae": 2.1,
                "rmse": 3.1,
                "accuracy": 0.95,
                "training_time": 120.5,
            },
            "lstm_v1.0": {
                "mae": 2.8,
                "rmse": 3.8,
                "accuracy": 0.92,
                "training_time": 85.2,
            },
        }

        summary = ForecastSummary(
            total_sensors_forecasted=12,
            average_confidence=0.85,
            risk_distribution={"LOW": 12, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0},
            recent_forecasts=[],
            model_performance=model_performance,
            generated_at=datetime.now(),
        )

        # Validate model performance metrics
        for model_name, metrics in summary.model_performance.items():
            assert "mae" in metrics
            assert "rmse" in metrics
            assert "accuracy" in metrics
            assert 0.0 <= metrics["accuracy"] <= 1.0


class TestForecastModelIntegration:
    """Integration tests for forecast models"""

    def test_forecast_points_to_result_aggregation(self):
        """Test aggregating forecast points into a forecast result"""
        now = datetime.now()

        # Create forecast points
        forecast_points = []
        forecast_timestamps = []
        forecast_values = []

        for i in range(1, 13):  # 12-hour forecast
            timestamp = now + timedelta(hours=i)
            value = 50.0 + i * 0.5 + np.random.normal(0, 1)

            point = ForecastPoint(
                timestamp=timestamp,
                predicted_value=value,
                confidence_lower=value - 2,
                confidence_upper=value + 2,
                confidence_level=ForecastConfidence.MEDIUM,
                risk_level=RiskLevel.LOW if value < 55 else RiskLevel.MEDIUM,
            )

            forecast_points.append(point)
            forecast_timestamps.append(timestamp)
            forecast_values.append(value)

        # Create forecast result from points
        result = ForecastResult(
            sensor_id="SMAP-ATT-001",
            forecast_horizon_hours=12,
            historical_timestamps=[now - timedelta(hours=i) for i in range(24, 0, -1)],
            historical_values=np.random.normal(48, 5, 24),
            forecast_timestamps=forecast_timestamps,
            forecast_values=np.array(forecast_values),
            confidence_intervals={
                "upper": np.array([p.confidence_upper for p in forecast_points]),
                "lower": np.array([p.confidence_lower for p in forecast_points]),
            },
            accuracy_metrics={"mae": 1.5, "rmse": 2.1},
            model_version="transformer_v1.0",
            generated_at=now,
            risk_assessment={"overall_risk": "LOW"},
        )

        assert len(result.forecast_timestamps) == len(forecast_points)
        assert len(result.forecast_values) == len(forecast_points)
        assert len(result.confidence_intervals["upper"]) == len(forecast_points)

    def test_nasa_sensor_forecasting(self, sample_sensors):
        """Test forecasting for NASA sensors"""
        now = datetime.now()

        for sensor_id in sample_sensors[:6]:  # Test first 6 sensors
            result = ForecastResult(
                sensor_id=sensor_id,
                forecast_horizon_hours=24,
                historical_timestamps=[
                    now - timedelta(hours=i) for i in range(48, 0, -1)
                ],
                historical_values=np.random.normal(50, 10, 48),
                forecast_timestamps=[now + timedelta(hours=i) for i in range(1, 25)],
                forecast_values=np.random.normal(52, 8, 24),
                confidence_intervals={
                    "upper": np.random.normal(57, 5, 24),
                    "lower": np.random.normal(47, 5, 24),
                },
                accuracy_metrics={"mae": 2.0, "rmse": 3.0, "mape": 4.0},
                model_version="nasa_transformer_v1.0",
                generated_at=now,
                risk_assessment={
                    "mission": "SMAP" if "SMAP" in sensor_id else "MSL",
                    "overall_risk": "LOW",
                },
            )

            assert result.sensor_id == sensor_id
            assert result.forecast_horizon_hours == 24
            mission = result.risk_assessment["mission"]
            assert mission in ["SMAP", "MSL"]

    def test_forecast_accuracy_validation(self):
        """Test forecast accuracy metrics validation"""
        now = datetime.now()

        # Create forecast with known accuracy metrics
        accuracy_metrics = {
            "mae": 1.5,  # Mean Absolute Error
            "rmse": 2.1,  # Root Mean Square Error
            "mape": 3.2,  # Mean Absolute Percentage Error
            "r2": 0.95,  # R-squared
        }

        result = ForecastResult(
            sensor_id="SMAP-ATT-001",
            forecast_horizon_hours=12,
            historical_timestamps=[now - timedelta(hours=i) for i in range(24, 0, -1)],
            historical_values=np.random.normal(50, 10, 24),
            forecast_timestamps=[now + timedelta(hours=i) for i in range(1, 13)],
            forecast_values=np.random.normal(52, 8, 12),
            confidence_intervals={"upper": np.ones(12) * 60, "lower": np.ones(12) * 44},
            accuracy_metrics=accuracy_metrics,
            model_version="transformer_v1.0",
            generated_at=now,
            risk_assessment={},
        )

        # Validate accuracy metrics
        assert result.accuracy_metrics["mae"] > 0
        assert result.accuracy_metrics["rmse"] >= result.accuracy_metrics["mae"]
        assert 0 <= result.accuracy_metrics["r2"] <= 1

    def test_risk_level_correlation_with_confidence(self):
        """Test that risk levels correlate appropriately with confidence"""
        now = datetime.now()

        # High confidence should generally correlate with lower risk
        high_confidence_point = ForecastPoint(
            timestamp=now + timedelta(hours=1),
            predicted_value=50.0,
            confidence_lower=49.5,
            confidence_upper=50.5,
            confidence_level=ForecastConfidence.HIGH,
            risk_level=RiskLevel.LOW,
        )

        # Low confidence might correlate with higher risk
        low_confidence_point = ForecastPoint(
            timestamp=now + timedelta(hours=1),
            predicted_value=50.0,
            confidence_lower=45.0,
            confidence_upper=55.0,
            confidence_level=ForecastConfidence.LOW,
            risk_level=RiskLevel.HIGH,
        )

        # Validate confidence intervals width
        high_conf_width = (
            high_confidence_point.confidence_upper
            - high_confidence_point.confidence_lower
        )
        low_conf_width = (
            low_confidence_point.confidence_upper
            - low_confidence_point.confidence_lower
        )

        assert high_conf_width < low_conf_width  # High confidence = narrower interval
        assert (
            high_confidence_point.confidence_level
            != low_confidence_point.confidence_level
        )

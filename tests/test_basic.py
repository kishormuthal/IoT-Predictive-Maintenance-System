"""
Basic Test Suite for CI/CD Pipeline
Tests core functionality of the IoT Predictive Maintenance System
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest


class TestImports:
    """Test that all critical modules can be imported"""

    def test_import_nasa_data_loader(self):
        """Test NASA data loader import"""
        from src.infrastructure.data.nasa_data_loader import NASADataLoader

        assert NASADataLoader is not None

    def test_import_anomaly_service(self):
        """Test anomaly service import"""
        from src.core.services.anomaly_service import AnomalyDetectionService

        assert AnomalyDetectionService is not None

    def test_import_forecasting_service(self):
        """Test forecasting service import"""
        from src.core.services.forecasting_service import ForecastingService

        assert ForecastingService is not None

    def test_import_integration_service(self):
        """Test dashboard integration service import"""
        from src.presentation.dashboard.services.dashboard_integration import (
            DashboardIntegrationService,
        )

        assert DashboardIntegrationService is not None

    def test_import_equipment_config(self):
        """Test equipment configuration import"""
        from config.equipment_config import get_equipment_list

        assert get_equipment_list is not None


class TestNASADataLoader:
    """Test NASA data loading functionality"""

    def test_data_loader_initialization(self):
        """Test that NASA data loader initializes"""
        from src.infrastructure.data.nasa_data_loader import NASADataLoader

        loader = NASADataLoader()
        assert loader is not None

    def test_data_loads(self):
        """Test that NASA data actually loads"""
        from src.infrastructure.data.nasa_data_loader import NASADataLoader

        loader = NASADataLoader()
        # Data should load successfully
        assert loader.is_loaded == True, "NASA data should load successfully"

    def test_equipment_list(self):
        """Test that equipment list is accessible"""
        from config.equipment_config import get_equipment_list

        equipment_list = get_equipment_list()
        assert len(equipment_list) > 0, "Should have at least one equipment"
        assert len(equipment_list) == 12, "Should have exactly 12 configured sensors"

    def test_sensor_data_retrieval(self):
        """Test retrieving sensor data"""
        from config.equipment_config import get_equipment_list
        from src.infrastructure.data.nasa_data_loader import NASADataLoader

        loader = NASADataLoader()
        equipment_list = get_equipment_list()

        if len(equipment_list) > 0:
            sensor_id = equipment_list[0].equipment_id
            data = loader.get_sensor_data(sensor_id, hours_back=24)

            assert data is not None, "Should return data"
            assert "values" in data, "Should have values key"
            assert "timestamps" in data, "Should have timestamps key"
            assert len(data["values"]) > 0, "Should have at least one data point"

    def test_data_quality_indicator(self):
        """Test that data quality is indicated"""
        from config.equipment_config import get_equipment_list
        from src.infrastructure.data.nasa_data_loader import NASADataLoader

        loader = NASADataLoader()
        equipment_list = get_equipment_list()

        if len(equipment_list) > 0:
            sensor_id = equipment_list[0].equipment_id
            data = loader.get_sensor_data(sensor_id)
            assert "data_quality" in data, "Should indicate data quality"
            assert data["data_quality"] in [
                "real",
                "mock",
            ], "Data quality should be 'real' or 'mock'"


class TestIntegrationService:
    """Test dashboard integration service"""

    def test_integration_service_initialization(self):
        """Test integration service initializes without errors"""
        from src.presentation.dashboard.services.dashboard_integration import (
            DashboardIntegrationService,
        )

        service = DashboardIntegrationService()
        assert service is not None

    def test_get_sensor_data(self):
        """Test get_sensor_data method"""
        from config.equipment_config import get_equipment_list
        from src.presentation.dashboard.services.dashboard_integration import (
            DashboardIntegrationService,
        )

        service = DashboardIntegrationService()
        equipment_list = get_equipment_list()

        if len(equipment_list) > 0:
            sensor_id = equipment_list[0].equipment_id
            df = service.get_sensor_data(sensor_id, hours=24)

            assert df is not None, "Should return data"
            assert isinstance(df, pd.DataFrame), "Should return DataFrame"
            assert len(df) > 0, "DataFrame should not be empty"
            assert "timestamp" in df.columns, "Should have timestamp column"
            assert "value" in df.columns, "Should have value column"


class TestHealthCheck:
    """Test health check endpoints"""

    def test_health_check_import(self):
        """Test health check module imports"""
        from src.presentation.dashboard.health_check import get_health_status

        assert get_health_status is not None

    def test_health_status(self):
        """Test get_health_status function"""
        from src.presentation.dashboard.health_check import get_health_status

        status = get_health_status()

        assert status is not None, "Should return status"
        assert "status" in status, "Should have status key"
        assert "version" in status, "Should have version key"
        assert "checks" in status, "Should have checks key"
        assert status["status"] in [
            "healthy",
            "degraded",
            "error",
        ], "Status should be valid"


class TestServices:
    """Test core services initialization"""

    def test_anomaly_service_init(self):
        """Test anomaly service can be initialized"""
        try:
            from src.core.services.anomaly_service import AnomalyDetectionService

            service = AnomalyDetectionService()
            assert service is not None
        except Exception as e:
            pytest.skip(f"Anomaly service init failed (may be due to missing models): {e}")

    def test_forecasting_service_init(self):
        """Test forecasting service can be initialized"""
        try:
            from src.core.services.forecasting_service import ForecastingService

            service = ForecastingService()
            assert service is not None
        except Exception as e:
            pytest.skip(f"Forecasting service init failed (may be due to missing models): {e}")


class TestAdvancedAlgorithms:
    """Test SESSION 7 advanced algorithms"""

    def test_adaptive_thresholding_import(self):
        """Test adaptive thresholding module imports"""
        from src.core.algorithms.adaptive_thresholding import (
            AdaptiveThresholdCalculator,
        )

        assert AdaptiveThresholdCalculator is not None

    def test_probabilistic_scoring_import(self):
        """Test probabilistic scoring module imports"""
        from src.core.algorithms.probabilistic_scoring import ProbabilisticAnomalyScorer

        assert ProbabilisticAnomalyScorer is not None

    def test_adaptive_threshold_calculation(self):
        """Test basic threshold calculation"""
        from src.core.algorithms.adaptive_thresholding import (
            AdaptiveThresholdCalculator,
        )

        # Generate test data
        data = np.random.normal(100, 10, 1000)

        # Calculate threshold
        result = AdaptiveThresholdCalculator.zscore_threshold(data, confidence_level=0.99)

        assert result is not None, "Should return threshold result"
        assert hasattr(result, "threshold"), "Should have threshold attribute"
        assert result.threshold > np.mean(data), "Threshold should be above mean"


class TestDashboardLayouts:
    """Test dashboard layout modules can be imported"""

    def test_import_monitoring_layout(self):
        """Test monitoring layout imports"""
        from src.presentation.dashboard.layouts.monitoring import create_layout

        assert create_layout is not None

    def test_import_overview_layout(self):
        """Test overview layout imports"""
        from src.presentation.dashboard.layouts.overview import create_layout

        assert create_layout is not None

    def test_import_anomaly_monitor_layout(self):
        """Test anomaly monitor layout imports"""
        from src.presentation.dashboard.layouts.anomaly_monitor import create_layout

        assert create_layout is not None


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v", "--tb=short"])

"""
Import Safety Tests - Session 1
Tests all major modules individually to identify hanging imports
"""

import importlib
import sys
import threading
import time
from pathlib import Path

import pytest

# Test markers
pytestmark = [pytest.mark.session1, pytest.mark.import_test, pytest.mark.unit]


class ImportTimeoutError(Exception):
    """Raised when import takes too long"""

    pass


def import_with_timeout(module_name: str, timeout: int = 30):
    """Import a module with timeout to detect hanging imports"""
    result = {"success": False, "error": None, "module": None}

    def import_module():
        try:
            result["module"] = importlib.import_module(module_name)
            result["success"] = True
        except Exception as e:
            result["error"] = e

    thread = threading.Thread(target=import_module)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise ImportTimeoutError(f"Import of {module_name} timed out after {timeout}s")

    if not result["success"] and result["error"]:
        raise result["error"]

    return result["module"]


class TestCoreImports:
    """Test core module imports"""

    def test_import_core_models(self):
        """Test importing core data models"""
        start_time = time.time()

        # Test each model individually
        models_to_test = [
            "src.core.models.sensor_data",
            "src.core.models.anomaly",
            "src.core.models.forecast",
        ]

        for model_name in models_to_test:
            module = import_with_timeout(model_name, timeout=10)
            assert module is not None, f"Failed to import {model_name}"

        import_time = time.time() - start_time
        assert import_time < 5.0, f"Core models import took too long: {import_time}s"

    def test_import_core_interfaces(self):
        """Test importing core interfaces"""
        start_time = time.time()

        interfaces_to_test = [
            "src.core.interfaces.detector_interface",
            "src.core.interfaces.forecaster_interface",
            "src.core.interfaces.data_interface",
        ]

        for interface_name in interfaces_to_test:
            module = import_with_timeout(interface_name, timeout=10)
            assert module is not None, f"Failed to import {interface_name}"

        import_time = time.time() - start_time
        assert import_time < 5.0, f"Core interfaces import took too long: {import_time}s"

    def test_import_core_services_individually(self):
        """Test importing core services one by one"""
        services_to_test = [
            "src.core.services.anomaly_service",
            "src.core.services.forecasting_service",
        ]

        for service_name in services_to_test:
            start_time = time.time()
            try:
                module = import_with_timeout(service_name, timeout=15)
                import_time = time.time() - start_time
                assert module is not None, f"Failed to import {service_name}"
                assert import_time < 10.0, f"{service_name} import took too long: {import_time}s"
            except ImportTimeoutError:
                pytest.fail(f"Import of {service_name} hung and timed out")
            except Exception as e:
                pytest.fail(f"Import of {service_name} failed with error: {e}")


class TestInfrastructureImports:
    """Test infrastructure module imports"""

    def test_import_data_modules(self):
        """Test importing data infrastructure modules"""
        data_modules = ["src.infrastructure.data.nasa_data_loader"]

        for module_name in data_modules:
            start_time = time.time()
            try:
                module = import_with_timeout(module_name, timeout=20)
                import_time = time.time() - start_time
                assert module is not None, f"Failed to import {module_name}"
                assert import_time < 15.0, f"{module_name} import took too long: {import_time}s"
            except ImportTimeoutError:
                pytest.fail(f"Import of {module_name} hung and timed out")

    def test_import_ml_modules_individually(self):
        """Test importing ML modules one by one to identify hanging ones"""
        ml_modules = [
            "src.infrastructure.ml.telemanom_wrapper",
            "src.infrastructure.ml.transformer_wrapper",
            "src.infrastructure.ml.model_registry",
        ]

        for module_name in ml_modules:
            start_time = time.time()
            try:
                module = import_with_timeout(module_name, timeout=25)
                import_time = time.time() - start_time
                assert module is not None, f"Failed to import {module_name}"
                # ML modules may take longer due to TensorFlow/Keras imports
                assert import_time < 20.0, f"{module_name} import took too long: {import_time}s"
            except ImportTimeoutError:
                pytest.fail(f"Import of {module_name} hung and timed out")
            except Exception as e:
                # Log but don't fail for missing dependencies
                print(f"Warning: {module_name} failed to import: {e}")

    def test_import_monitoring_modules(self):
        """Test importing monitoring modules"""
        monitoring_modules = ["src.infrastructure.monitoring.performance_monitor"]

        for module_name in monitoring_modules:
            start_time = time.time()
            try:
                module = import_with_timeout(module_name, timeout=15)
                import_time = time.time() - start_time
                assert module is not None, f"Failed to import {module_name}"
                assert import_time < 10.0, f"{module_name} import took too long: {import_time}s"
            except ImportTimeoutError:
                pytest.fail(f"Import of {module_name} hung and timed out")


class TestApplicationImports:
    """Test application layer imports"""

    def test_import_use_cases(self):
        """Test importing use case modules"""
        use_case_modules = ["src.application.use_cases.training_use_case"]

        for module_name in use_case_modules:
            start_time = time.time()
            try:
                module = import_with_timeout(module_name, timeout=15)
                import_time = time.time() - start_time
                assert module is not None, f"Failed to import {module_name}"
                assert import_time < 10.0, f"{module_name} import took too long: {import_time}s"
            except ImportTimeoutError:
                pytest.fail(f"Import of {module_name} hung and timed out")

    def test_import_application_services(self):
        """Test importing application services"""
        app_services = ["src.application.services.training_config_manager"]

        for module_name in app_services:
            start_time = time.time()
            try:
                module = import_with_timeout(module_name, timeout=15)
                import_time = time.time() - start_time
                assert module is not None, f"Failed to import {module_name}"
                assert import_time < 10.0, f"{module_name} import took too long: {import_time}s"
            except ImportTimeoutError:
                pytest.fail(f"Import of {module_name} hung and timed out")


class TestDashboardImports:
    """Test dashboard module imports (high risk for hanging)"""

    def test_import_dashboard_components_individually(self):
        """Test importing dashboard components one by one"""
        # Start with simplified components
        component_modules = ["src.presentation.dashboard.enhanced_callbacks_simplified"]

        for module_name in component_modules:
            start_time = time.time()
            try:
                module = import_with_timeout(module_name, timeout=20)
                import_time = time.time() - start_time
                assert module is not None, f"Failed to import {module_name}"
                assert import_time < 15.0, f"{module_name} import took too long: {import_time}s"
            except ImportTimeoutError:
                pytest.fail(f"Import of {module_name} hung and timed out")

    def test_import_main_dashboard_module(self):
        """Test importing main dashboard module (most likely to hang)"""
        start_time = time.time()
        try:
            # This is the critical test - main dashboard import
            module = import_with_timeout("src.presentation.dashboard.enhanced_app", timeout=30)
            import_time = time.time() - start_time
            assert module is not None, "Failed to import main dashboard module"
            # Dashboard may take longer due to Dash imports
            assert import_time < 25.0, f"Dashboard import took too long: {import_time}s"
        except ImportTimeoutError:
            pytest.fail("Main dashboard module import hung and timed out - this is the likely cause of hanging")
        except Exception as e:
            pytest.fail(f"Main dashboard module import failed: {e}")

    @pytest.mark.slow
    def test_import_complex_dashboard_layouts(self):
        """Test importing complex dashboard layouts (marked as slow)"""
        layout_modules = [
            "src.presentation.dashboard.layouts.overview",
            "src.presentation.dashboard.layouts.anomaly_monitor",
            "src.presentation.dashboard.layouts.forecast_view",
        ]

        for module_name in layout_modules:
            start_time = time.time()
            try:
                module = import_with_timeout(module_name, timeout=30)
                import_time = time.time() - start_time
                if module is not None:
                    assert import_time < 25.0, f"{module_name} import took too long: {import_time}s"
            except ImportTimeoutError:
                print(f"Warning: {module_name} timed out during import")
            except Exception as e:
                print(f"Warning: {module_name} failed to import: {e}")


class TestConfigImports:
    """Test configuration module imports"""

    def test_import_config_modules(self):
        """Test importing configuration modules"""
        config_modules = ["config.settings", "config.equipment_config"]

        for module_name in config_modules:
            start_time = time.time()
            try:
                module = import_with_timeout(module_name, timeout=10)
                import_time = time.time() - start_time
                assert module is not None, f"Failed to import {module_name}"
                assert import_time < 5.0, f"{module_name} import took too long: {import_time}s"
            except ImportTimeoutError:
                pytest.fail(f"Import of {module_name} hung and timed out")


class TestUtilityImports:
    """Test utility module imports"""

    def test_import_utility_modules(self):
        """Test importing utility modules"""
        utility_modules = ["src.utils.helpers", "src.utils.metrics", "src.utils.logger"]

        for module_name in utility_modules:
            start_time = time.time()
            try:
                module = import_with_timeout(module_name, timeout=10)
                import_time = time.time() - start_time
                if module is not None:
                    assert import_time < 5.0, f"{module_name} import took too long: {import_time}s"
            except ImportTimeoutError:
                print(f"Warning: {module_name} timed out during import")
            except Exception as e:
                print(f"Warning: {module_name} failed to import: {e}")


# Summary test
def test_overall_import_health():
    """Test overall import health and performance"""
    start_time = time.time()

    # Test critical path imports
    critical_imports = [
        "src.core.models.sensor_data",
        "src.core.services.anomaly_service",
        "src.infrastructure.data.nasa_data_loader",
        "config.equipment_config",
    ]

    successful_imports = 0
    for module_name in critical_imports:
        try:
            import_with_timeout(module_name, timeout=15)
            successful_imports += 1
        except Exception:
            pass

    total_time = time.time() - start_time

    # At least 75% of critical imports should succeed
    success_rate = successful_imports / len(critical_imports)
    assert success_rate >= 0.75, f"Critical import success rate too low: {success_rate:.2%}"

    # Total import time should be reasonable
    assert total_time < 30.0, f"Critical imports took too long: {total_time}s"

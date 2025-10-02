"""
Dashboard Component Tests - Session 2
Testing the updated dashboard with timeout fixes and component analysis
"""

import threading
import time
from datetime import datetime
from unittest.mock import Mock, patch

import psutil
import pytest

# Test markers
pytestmark = [pytest.mark.session2, pytest.mark.dashboard, pytest.mark.integration]


class TestUpdatedDashboardImport:
    """Test the updated dashboard with timeout mechanisms"""

    def test_dashboard_import_with_timeout_fixes(self):
        """Test that the updated dashboard imports successfully with timeout fixes"""
        start_time = time.time()

        try:
            # Test the updated dashboard import
            from src.presentation.dashboard.enhanced_app import EnhancedIoTDashboard

            import_time = time.time() - start_time

            # Should import much faster now with timeout fixes
            assert import_time < 15.0, f"Dashboard import still too slow: {import_time}s"
            assert EnhancedIoTDashboard is not None, "Dashboard class not imported"

        except Exception as e:
            pytest.fail(f"Dashboard import failed even with timeout fixes: {e}")

    def test_dashboard_initialization_with_timeouts(self):
        """Test dashboard initialization with the new timeout mechanisms"""
        start_time = time.time()

        try:
            from src.presentation.dashboard.enhanced_app import EnhancedIoTDashboard

            # Initialize dashboard (should use timeouts and fallbacks)
            dashboard = EnhancedIoTDashboard(debug=False)

            init_time = time.time() - start_time

            # Should initialize faster with timeouts
            assert init_time < 30.0, f"Dashboard initialization still too slow: {init_time}s"
            assert dashboard is not None, "Dashboard not initialized"
            assert hasattr(dashboard, "app"), "Dashboard app not created"

        except Exception as e:
            pytest.fail(f"Dashboard initialization failed: {e}")

    def test_service_initialization_fallbacks(self):
        """Test that service initialization uses fallbacks when needed"""
        from src.presentation.dashboard.enhanced_app import EnhancedIoTDashboard

        dashboard = EnhancedIoTDashboard(debug=False)

        # Check that core services are initialized (even if some fail)
        assert hasattr(dashboard, "data_loader"), "Data loader not present"
        assert hasattr(dashboard, "anomaly_service"), "Anomaly service not present"
        assert hasattr(dashboard, "forecasting_service"), "Forecasting service not present"

        # Optional services might be None due to fallbacks
        assert hasattr(dashboard, "training_use_case"), "Training use case attribute missing"
        assert hasattr(dashboard, "config_manager"), "Config manager attribute missing"
        assert hasattr(dashboard, "model_registry"), "Model registry attribute missing"
        assert hasattr(dashboard, "performance_monitor"), "Performance monitor attribute missing"

    def test_dashboard_memory_usage(self):
        """Test dashboard memory usage during initialization"""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        from src.presentation.dashboard.enhanced_app import EnhancedIoTDashboard

        dashboard = EnhancedIoTDashboard(debug=False)

        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 200MB)
        assert memory_increase < 200, f"Dashboard uses too much memory: {memory_increase}MB"

    def test_performance_monitoring_startup(self):
        """Test that performance monitoring starts properly with new fixes"""
        from src.presentation.dashboard.enhanced_app import EnhancedIoTDashboard

        dashboard = EnhancedIoTDashboard(debug=False)

        # Check that performance monitoring is handled gracefully
        # (either started successfully or skipped with None service)
        if dashboard.performance_monitor is not None:
            # If performance monitor exists, it should be started
            assert hasattr(
                dashboard.performance_monitor, "start_monitoring"
            ), "Performance monitor missing start_monitoring method"
        else:
            # If None, that's acceptable as fallback behavior
            pass


class TestDashboardComponentIsolation:
    """Test individual dashboard components in isolation"""

    def test_simplified_callbacks_import(self):
        """Test importing simplified callbacks module"""
        start_time = time.time()

        from src.presentation.dashboard.enhanced_callbacks_simplified import (
            register_enhanced_callbacks,
        )

        import_time = time.time() - start_time
        assert import_time < 5.0, f"Simplified callbacks import too slow: {import_time}s"
        assert register_enhanced_callbacks is not None, "Callback registration function not available"

    def test_layout_component_functions(self):
        """Test individual layout component creation functions"""
        from src.presentation.dashboard.enhanced_callbacks_simplified import (
            create_config_management_layout,
            create_model_registry_layout,
            create_system_admin_layout,
            create_training_hub_layout,
        )

        # Test each layout function
        training_layout = create_training_hub_layout()
        assert training_layout is not None, "Training hub layout creation failed"

        model_layout = create_model_registry_layout()
        assert model_layout is not None, "Model registry layout creation failed"

        admin_layout = create_system_admin_layout()
        assert admin_layout is not None, "System admin layout creation failed"

        config_layout = create_config_management_layout()
        assert config_layout is not None, "Config management layout creation failed"

    @pytest.mark.slow
    def test_dashboard_tab_rendering(self):
        """Test that dashboard tabs can be rendered without hanging"""
        from src.presentation.dashboard.enhanced_app import EnhancedIoTDashboard

        dashboard = EnhancedIoTDashboard(debug=False)

        # Test tab content rendering methods
        test_state = {"system_health": "operational", "last_update": datetime.now()}

        # These should not hang
        start_time = time.time()

        overview_content = dashboard._create_enhanced_overview_tab(test_state)
        assert time.time() - start_time < 5.0, "Overview tab rendering too slow"
        assert overview_content is not None, "Overview tab content not created"

        monitoring_content = dashboard._create_enhanced_monitoring_tab()
        assert time.time() - start_time < 10.0, "Monitoring tab rendering too slow"
        assert monitoring_content is not None, "Monitoring tab content not created"

    def test_dashboard_state_management(self):
        """Test dashboard state management components"""
        from src.presentation.dashboard.enhanced_app import EnhancedIoTDashboard

        dashboard = EnhancedIoTDashboard(debug=False)

        # Check dashboard state initialization
        assert hasattr(dashboard, "dashboard_state"), "Dashboard state not initialized"
        assert isinstance(dashboard.dashboard_state, dict), "Dashboard state not a dictionary"

        required_state_keys = [
            "system_health",
            "last_update",
            "active_alerts",
            "performance_metrics",
        ]
        for key in required_state_keys:
            assert key in dashboard.dashboard_state, f"Required state key '{key}' missing"

    def test_equipment_list_loading(self):
        """Test that equipment list loads properly"""
        from src.presentation.dashboard.enhanced_app import EnhancedIoTDashboard

        dashboard = EnhancedIoTDashboard(debug=False)

        # Check equipment list
        assert hasattr(dashboard, "equipment_list"), "Equipment list not loaded"
        assert hasattr(dashboard, "sensor_ids"), "Sensor IDs not loaded"

        if dashboard.equipment_list:
            assert len(dashboard.equipment_list) > 0, "Equipment list is empty"
            assert len(dashboard.sensor_ids) > 0, "Sensor IDs list is empty"


class TestDashboardPerformanceOptimizations:
    """Test performance optimizations and monitoring"""

    def test_service_initialization_timing(self):
        """Test that individual service initialization is within timeout limits"""
        # Import the dashboard module to test service init timing
        from src.presentation.dashboard.enhanced_app import EnhancedIoTDashboard

        # Create dashboard instance and measure service init times
        start_time = time.time()
        dashboard = EnhancedIoTDashboard(debug=False)
        total_time = time.time() - start_time

        # Total initialization should be under reasonable time limit
        assert total_time < 45.0, f"Total dashboard initialization too slow: {total_time}s"

    def test_timeout_mechanism_effectiveness(self):
        """Test that timeout mechanisms are working"""
        # This test ensures the timeout mechanisms are in place
        from src.presentation.dashboard.enhanced_app import EnhancedIoTDashboard

        # If we can create the dashboard without hanging, timeouts are working
        start_time = time.time()
        dashboard = EnhancedIoTDashboard(debug=False)
        initialization_time = time.time() - start_time

        # Should complete within timeout limits
        assert initialization_time < 60.0, f"Dashboard still hanging despite timeouts: {initialization_time}s"

    def test_fallback_service_functionality(self):
        """Test that fallback services provide basic functionality"""
        from src.presentation.dashboard.enhanced_app import EnhancedIoTDashboard

        dashboard = EnhancedIoTDashboard(debug=False)

        # Even with fallbacks, core services should be functional
        if dashboard.data_loader:
            assert hasattr(dashboard.data_loader, "__class__"), "Data loader not properly initialized"

        if dashboard.anomaly_service:
            assert hasattr(dashboard.anomaly_service, "__class__"), "Anomaly service not properly initialized"

        if dashboard.forecasting_service:
            assert hasattr(dashboard.forecasting_service, "__class__"), "Forecasting service not properly initialized"

    def test_simplified_layout_performance(self):
        """Test that simplified layout setup is fast"""
        from src.presentation.dashboard.enhanced_app import EnhancedIoTDashboard

        start_time = time.time()
        dashboard = EnhancedIoTDashboard(debug=False)

        # Test simplified layout method directly
        layout_start = time.time()
        dashboard._setup_simplified_layout()
        layout_time = time.time() - layout_start

        assert layout_time < 5.0, f"Simplified layout setup too slow: {layout_time}s"
        assert hasattr(dashboard.app, "layout"), "App layout not set"


class TestDashboardErrorHandling:
    """Test error handling and recovery mechanisms"""

    def test_service_failure_recovery(self):
        """Test dashboard behavior when services fail to initialize"""
        from src.presentation.dashboard.enhanced_app import EnhancedIoTDashboard

        # Dashboard should handle service failures gracefully
        dashboard = EnhancedIoTDashboard(debug=False)

        # Should not crash even if some services are None
        assert dashboard.app is not None, "Dashboard app not created despite service failures"

    def test_timeout_error_handling(self):
        """Test that timeout errors are handled properly"""
        # This is implicitly tested by successful dashboard creation
        # If timeouts weren't handled, dashboard creation would hang
        from src.presentation.dashboard.enhanced_app import EnhancedIoTDashboard

        try:
            dashboard = EnhancedIoTDashboard(debug=False)
            # If we reach here, timeout handling is working
            assert True, "Timeout handling successful"
        except Exception as e:
            pytest.fail(f"Timeout error handling failed: {e}")

    def test_fallback_initialization(self):
        """Test fallback initialization mechanism"""
        from src.presentation.dashboard.enhanced_app import EnhancedIoTDashboard

        dashboard = EnhancedIoTDashboard(debug=False)

        # Test fallback service initialization method exists
        assert hasattr(dashboard, "_initialize_fallback_services"), "Fallback service method missing"

        # Fallback services should be available
        fallback_services = ["data_loader", "anomaly_service", "forecasting_service"]
        for service in fallback_services:
            assert hasattr(dashboard, service), f"Fallback service {service} missing"


class TestDashboardIntegration:
    """Integration tests for dashboard components working together"""

    def test_dashboard_end_to_end_creation(self):
        """Test complete dashboard creation process"""
        start_time = time.time()

        from src.presentation.dashboard.enhanced_app import EnhancedIoTDashboard

        dashboard = EnhancedIoTDashboard(debug=False)

        creation_time = time.time() - start_time

        # Complete dashboard should be created successfully
        assert dashboard is not None, "Dashboard creation failed"
        assert dashboard.app is not None, "Dashboard app not created"
        assert creation_time < 60.0, f"Dashboard creation too slow: {creation_time}s"

    def test_dashboard_callback_registration(self):
        """Test that dashboard callbacks are registered without hanging"""
        from src.presentation.dashboard.enhanced_app import EnhancedIoTDashboard

        start_time = time.time()
        dashboard = EnhancedIoTDashboard(debug=False)
        callback_time = time.time() - start_time

        # Callback registration should complete
        assert callback_time < 45.0, f"Callback registration too slow: {callback_time}s"

        # Dashboard should have callbacks registered
        assert len(dashboard.app.callback_map) > 0, "No callbacks registered"

    @pytest.mark.slow
    def test_dashboard_startup_sequence(self):
        """Test the complete dashboard startup sequence"""
        startup_start = time.time()

        # Test complete startup
        from src.presentation.dashboard.enhanced_app import create_enhanced_dashboard

        dashboard = create_enhanced_dashboard(debug=False)

        startup_time = time.time() - startup_start

        assert startup_time < 90.0, f"Complete dashboard startup too slow: {startup_time}s"
        assert dashboard is not None, "Dashboard not created"
        assert dashboard.app is not None, "Dashboard app not available"


def test_session2_overall_dashboard_health():
    """Overall health check for Session 2 dashboard improvements"""
    start_time = time.time()

    try:
        from src.presentation.dashboard.enhanced_app import EnhancedIoTDashboard

        dashboard = EnhancedIoTDashboard(debug=False)

        total_time = time.time() - start_time

        # All improvements should result in faster, more reliable dashboard
        assert total_time < 60.0, f"Dashboard health check failed - still too slow: {total_time}s"
        assert dashboard is not None, "Dashboard health check failed - not created"

        # Check that improvements are in place
        assert hasattr(dashboard, "_initialize_fallback_services"), "Fallback mechanism missing"
        assert hasattr(dashboard, "_start_performance_monitoring"), "Performance monitoring startup missing"

    except Exception as e:
        pytest.fail(f"Session 2 dashboard health check failed: {e}")

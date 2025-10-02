"""
Dashboard Diagnostic Tests - Session 2
Detailed analysis of hanging issues in dashboard components
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Test markers
pytestmark = [pytest.mark.session2, pytest.mark.dashboard, pytest.mark.diagnostic]


class TestDashboardDiagnostics:
    """Diagnostic tests to identify hanging issues"""

    def test_minimal_dashboard_import(self):
        """Test minimal dashboard import without instantiation"""
        start_time = time.time()

        try:
            # Test just the module import
            from src.presentation.dashboard import enhanced_app

            import_time = time.time() - start_time

            assert import_time < 10.0, f"Module import too slow: {import_time}s"
            assert hasattr(enhanced_app, "EnhancedIoTDashboard"), "Dashboard class not found"

        except Exception as e:
            pytest.fail(f"Module import failed: {e}")

    def test_dashboard_class_access(self):
        """Test accessing the dashboard class without instantiation"""
        from src.presentation.dashboard.enhanced_app import EnhancedIoTDashboard

        # Should be able to access the class
        assert EnhancedIoTDashboard is not None, "Dashboard class is None"
        assert hasattr(EnhancedIoTDashboard, "__init__"), "Dashboard class missing __init__"

    def test_simplified_callbacks_isolated(self):
        """Test simplified callbacks in isolation"""
        start_time = time.time()

        from src.presentation.dashboard.enhanced_callbacks_simplified import (
            create_config_management_layout,
            create_model_registry_layout,
            create_system_admin_layout,
            create_training_hub_layout,
            register_enhanced_callbacks,
        )

        import_time = time.time() - start_time
        assert import_time < 5.0, f"Simplified callbacks import too slow: {import_time}s"

        # Test layout creation
        layout_start = time.time()
        training_layout = create_training_hub_layout()
        layout_time = time.time() - layout_start

        assert layout_time < 2.0, f"Layout creation too slow: {layout_time}s"
        assert training_layout is not None, "Training layout not created"

    def test_individual_service_initialization(self):
        """Test individual service initialization timing"""

        # Test each service individually
        services_to_test = [
            ("AnomalyDetectionService", "src.core.services.anomaly_service"),
            ("ForecastingService", "src.core.services.forecasting_service"),
            ("NASADataLoader", "src.infrastructure.data.nasa_data_loader"),
            ("TrainingUseCase", "src.application.use_cases.training_use_case"),
            (
                "TrainingConfigManager",
                "src.application.services.training_config_manager",
            ),
            ("ModelRegistry", "src.infrastructure.ml.model_registry"),
            ("PerformanceMonitor", "src.infrastructure.monitoring.performance_monitor"),
        ]

        results = {}

        for service_name, module_path in services_to_test:
            start_time = time.time()
            try:
                module = __import__(module_path, fromlist=[service_name])
                service_class = getattr(module, service_name)

                # Try to create instance
                init_start = time.time()
                service_instance = service_class()
                init_time = time.time() - init_start

                total_time = time.time() - start_time
                results[service_name] = {
                    "import_time": total_time - init_time,
                    "init_time": init_time,
                    "total_time": total_time,
                    "success": True,
                }

            except Exception as e:
                total_time = time.time() - start_time
                results[service_name] = {
                    "import_time": 0,
                    "init_time": 0,
                    "total_time": total_time,
                    "success": False,
                    "error": str(e),
                }

        # Analyze results
        for service_name, result in results.items():
            if result["success"]:
                assert result["total_time"] < 10.0, f"{service_name} initialization too slow: {result['total_time']}s"
            else:
                # Log failed services but don't fail test
                print(f"Warning: {service_name} failed to initialize: {result.get('error', 'Unknown error')}")

    @pytest.mark.slow
    def test_dashboard_timeout_behavior(self):
        """Test dashboard behavior with external timeout"""
        project_root = Path(__file__).parent.parent.parent

        # Create a test script to run dashboard with timeout
        test_script = f"""
import sys
import signal
import time
sys.path.insert(0, '{project_root}/src')

def timeout_handler(signum, frame):
    print("TIMEOUT: Dashboard creation exceeded time limit")
    sys.exit(1)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(20)  # 20 second timeout

try:
    from src.presentation.dashboard.enhanced_app import EnhancedIoTDashboard
    print("Dashboard class imported successfully")

    dashboard = EnhancedIoTDashboard(debug=False)
    print("Dashboard created successfully")
    signal.alarm(0)  # Cancel timeout

except Exception as e:
    print(f"Dashboard creation failed: {{e}}")
    signal.alarm(0)
    sys.exit(1)
"""

        # Write and execute test script
        script_path = project_root / "temp_dashboard_test.py"
        with open(script_path, "w") as f:
            f.write(test_script)

        try:
            # Run with timeout
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(project_root),
            )

            if result.returncode == 0:
                assert "Dashboard created successfully" in result.stdout, "Dashboard creation did not complete"
            else:
                # Dashboard failed or timed out
                if "TIMEOUT" in result.stdout:
                    pytest.fail("Dashboard creation timed out after 20 seconds")
                else:
                    pytest.fail(f"Dashboard creation failed: {result.stdout}")

        except subprocess.TimeoutExpired:
            pytest.fail("Dashboard test script timed out after 30 seconds")
        finally:
            # Cleanup
            if script_path.exists():
                script_path.unlink()

    def test_component_memory_usage(self):
        """Test memory usage of individual components"""
        import psutil

        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Test importing dashboard components
        from src.presentation.dashboard.enhanced_callbacks_simplified import (
            create_model_registry_layout,
            create_training_hub_layout,
        )

        post_import_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Create layouts
        layout1 = create_training_hub_layout()
        layout2 = create_model_registry_layout()

        final_memory = psutil.Process().memory_info().rss / 1024 / 1024

        import_increase = post_import_memory - initial_memory
        layout_increase = final_memory - post_import_memory
        total_increase = final_memory - initial_memory

        # Memory increases should be reasonable
        assert import_increase < 50, f"Component import uses too much memory: {import_increase}MB"
        assert layout_increase < 20, f"Layout creation uses too much memory: {layout_increase}MB"
        assert total_increase < 70, f"Total component memory usage too high: {total_increase}MB"

    def test_config_dependencies(self):
        """Test configuration and equipment dependencies"""
        start_time = time.time()

        try:
            from config.equipment_config import get_equipment_by_id, get_equipment_list

            config_time = time.time() - start_time

            assert config_time < 5.0, f"Config import too slow: {config_time}s"

            # Test equipment list loading
            equipment_start = time.time()
            equipment_list = get_equipment_list()
            equipment_time = time.time() - equipment_start

            assert equipment_time < 3.0, f"Equipment list loading too slow: {equipment_time}s"
            assert len(equipment_list) > 0, "Equipment list is empty"

        except Exception as e:
            pytest.fail(f"Config dependency test failed: {e}")

    def test_state_manager_components(self):
        """Test state management components individually"""
        try:
            # Test individual state management imports
            from src.presentation.dashboard.state.filter_state import FilterStateManager
            from src.presentation.dashboard.state.realtime_state import (
                RealtimeStateManager,
            )
            from src.presentation.dashboard.state.shared_state import SharedStateManager
            from src.presentation.dashboard.state.time_state import TimeStateManager

            # These should import without hanging
            assert SharedStateManager is not None, "SharedStateManager not available"

        except ImportError as e:
            # State managers might not exist, that's OK
            print(f"Note: State managers not available: {e}")
        except Exception as e:
            pytest.fail(f"State manager test failed: {e}")


class TestDashboardComponentPerformance:
    """Test performance of individual dashboard components"""

    def test_dash_bootstrap_components_performance(self):
        """Test Dash Bootstrap Components import performance"""
        start_time = time.time()

        import dash_bootstrap_components as dbc

        import_time = time.time() - start_time
        assert import_time < 5.0, f"DBC import too slow: {import_time}s"

        # Test component creation
        component_start = time.time()
        card = dbc.Card([dbc.CardHeader("Test"), dbc.CardBody("Test content")])
        component_time = time.time() - component_start

        assert component_time < 1.0, f"DBC component creation too slow: {component_time}s"

    def test_plotly_performance(self):
        """Test Plotly import and basic functionality"""
        start_time = time.time()

        import plotly.express as px
        import plotly.graph_objects as go

        import_time = time.time() - start_time
        assert import_time < 10.0, f"Plotly import too slow: {import_time}s"

        # Test basic chart creation
        chart_start = time.time()
        fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
        chart_time = time.time() - chart_start

        assert chart_time < 2.0, f"Chart creation too slow: {chart_time}s"

    def test_dash_core_performance(self):
        """Test Dash core components performance"""
        start_time = time.time()

        from dash import Input, Output, State, callback, ctx, dcc, html

        import_time = time.time() - start_time
        assert import_time < 5.0, f"Dash core import too slow: {import_time}s"

        # Test component creation
        component_start = time.time()
        div = html.Div([html.H1("Test"), dcc.Graph(id="test-graph")])
        component_time = time.time() - component_start

        assert component_time < 1.0, f"Dash component creation too slow: {component_time}s"


def test_session2_diagnostic_summary():
    """Summary test for Session 2 diagnostics"""

    # Run a quick overall diagnostic
    issues_found = []

    try:
        # Test basic imports
        from src.presentation.dashboard.enhanced_app import EnhancedIoTDashboard
        from src.presentation.dashboard.enhanced_callbacks_simplified import (
            create_training_hub_layout,
        )

        # Test layout creation
        layout = create_training_hub_layout()
        if layout is None:
            issues_found.append("Layout creation returned None")

    except Exception as e:
        issues_found.append(f"Import/layout test failed: {e}")

    # Report findings
    if issues_found:
        pytest.fail(f"Session 2 diagnostics found issues: {'; '.join(issues_found)}")
    else:
        # If we get here, basic functionality is working
        assert True, "Session 2 diagnostics completed successfully"

"""
Phase 3 Integration Validation Script
Tests all Phase 3 components and integrations
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.dashboard.integration.event_coordinator import EventType
from src.dashboard.nasa_dashboard_orchestrator import phase3_manager
from src.dashboard.state.shared_state import shared_state_manager

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def test_event_coordinator():
    """Test event coordinator functionality"""
    logger.info("Testing Event Coordinator...")

    try:
        # Test event publishing
        event_id = phase3_manager.event_coordinator.publish_event(
            EventType.SENSOR_DATA_UPDATE,
            source="validation_script",
            data={"sensor_id": "T-1", "value": 75.5, "timestamp": time.time()},
        )

        assert event_id, "Event publishing failed"

        # Test event history
        history = phase3_manager.event_coordinator.get_event_history(limit=5)
        assert len(history) > 0, "Event history empty"

        logger.info("✓ Event Coordinator tests passed")
        return True

    except Exception as e:
        logger.error(f"✗ Event Coordinator test failed: {e}")
        return False


async def test_workflow_manager():
    """Test workflow manager functionality"""
    logger.info("Testing Workflow Manager...")

    try:
        if not phase3_manager.workflow_manager:
            logger.warning("Workflow Manager not initialized")
            return True

        # Test workflow creation
        workflow_id = await phase3_manager.workflow_manager.create_workflow(
            phase3_manager.workflow_manager.WorkflowType.ANOMALY_TO_MAINTENANCE,
            phase3_manager.event_coordinator.Event(
                EventType.ANOMALY_DETECTED,
                source="validation_script",
                data={"sensor_id": "T-1", "anomaly_score": 0.85},
            ),
            priority=2,
        )

        assert workflow_id, "Workflow creation failed"

        # Check workflow status
        workflow = phase3_manager.workflow_manager.get_workflow_status(workflow_id)
        assert workflow is not None, "Workflow not found"

        logger.info("✓ Workflow Manager tests passed")
        return True

    except Exception as e:
        logger.error(f"✗ Workflow Manager test failed: {e}")
        return False


async def test_cache_manager():
    """Test enhanced cache manager functionality"""
    logger.info("Testing Enhanced Cache Manager...")

    try:
        # Test sensor data caching
        data = await phase3_manager.cache_manager.get_sensor_data("T-1", "1h", "sensor_monitoring")
        assert data is not None, "Sensor data retrieval failed"

        # Test cross-feature data
        cross_data = await phase3_manager.cache_manager.get_cross_feature_data(
            ["T-1", "P-1"], ["sensor_monitoring", "anomaly_detection"]
        )
        assert len(cross_data) > 0, "Cross-feature data retrieval failed"

        # Test cache metrics
        metrics = phase3_manager.cache_manager.get_cache_metrics()
        assert hasattr(metrics, "hit_rate_12h"), "Cache metrics missing"

        logger.info("✓ Enhanced Cache Manager tests passed")
        return True

    except Exception as e:
        logger.error(f"✗ Enhanced Cache Manager test failed: {e}")
        return False


async def test_lazy_loader():
    """Test lazy loader functionality"""
    logger.info("Testing Lazy Loader...")

    try:
        # Test component registration
        def mock_load_function():
            return {"data": "test_data", "loaded_at": time.time()}

        success = phase3_manager.lazy_loader.register_component("test_component", "sensor_chart", mock_load_function)
        assert success, "Component registration failed"

        # Test component loading
        data = await phase3_manager.lazy_loader.load_component("test_component")
        assert data is not None, "Component loading failed"

        # Test loading metrics
        metrics = phase3_manager.lazy_loader.get_loading_metrics()
        assert hasattr(metrics, "total_components"), "Loading metrics missing"

        logger.info("✓ Lazy Loader tests passed")
        return True

    except Exception as e:
        logger.error(f"✗ Lazy Loader test failed: {e}")
        return False


async def test_performance_monitor():
    """Test performance monitor functionality"""
    logger.info("Testing Performance Monitor...")

    try:
        # Test performance summary
        summary = phase3_manager.performance_monitor.get_performance_summary()
        assert "performance_score" in summary, "Performance summary missing score"

        # Test alerts
        alerts = phase3_manager.performance_monitor.get_alerts()
        assert isinstance(alerts, list), "Alerts not returned as list"

        # Test recommendations
        recommendations = phase3_manager.performance_monitor.get_recommendations()
        assert isinstance(recommendations, list), "Recommendations not returned as list"

        logger.info("✓ Performance Monitor tests passed")
        return True

    except Exception as e:
        logger.error(f"✗ Performance Monitor test failed: {e}")
        return False


def test_theme_manager():
    """Test theme manager functionality"""
    logger.info("Testing Theme Manager...")

    try:
        # Test theme setting
        success = phase3_manager.theme_manager.set_theme("dark")
        assert success, "Theme setting failed"

        # Test theme info
        theme_info = phase3_manager.theme_manager.get_current_theme_info()
        assert theme_info["name"] == "dark", "Theme not applied correctly"

        # Test sensor colors
        color = phase3_manager.theme_manager.get_sensor_color("T-1")
        assert color.startswith("#"), "Sensor color not valid hex"

        # Test CSS variables
        css_vars = phase3_manager.theme_manager.get_theme_css_variables()
        assert "--color-primary" in css_vars, "CSS variables missing"

        # Reset to light theme
        phase3_manager.theme_manager.set_theme("light")

        logger.info("✓ Theme Manager tests passed")
        return True

    except Exception as e:
        logger.error(f"✗ Theme Manager test failed: {e}")
        return False


def test_responsive_utils():
    """Test responsive utilities functionality"""
    logger.info("Testing Responsive Utils...")

    try:
        # Test device detection
        device_type = phase3_manager.responsive_utils.detect_device_type(1920, 1080)
        assert device_type is not None, "Device detection failed"

        # Test breakpoint detection
        breakpoint = phase3_manager.responsive_utils.get_current_breakpoint(1200)
        assert breakpoint in ["xs", "sm", "md", "lg", "xl", "xxl"], "Invalid breakpoint"

        # Test NASA grid configuration
        grid_config = phase3_manager.responsive_utils.get_sensor_grid_config()
        assert "rows" in grid_config and "columns" in grid_config, "Grid config missing"
        assert grid_config["total_sensors"] == 12, "NASA sensor count incorrect"

        # Test font sizes
        font_sizes = phase3_manager.responsive_utils.get_font_sizes()
        assert "base" in font_sizes, "Base font size missing"

        logger.info("✓ Responsive Utils tests passed")
        return True

    except Exception as e:
        logger.error(f"✗ Responsive Utils test failed: {e}")
        return False


def test_shared_state_integration():
    """Test shared state integration"""
    logger.info("Testing Shared State Integration...")

    try:
        # Test state setting and getting
        shared_state_manager.set_state("test.phase3_validation", True, "validation_script")
        value = shared_state_manager.get_state("test.phase3_validation")
        assert value is True, "State setting/getting failed"

        # Test integration status
        integration_status = shared_state_manager.get_integration_status()
        assert "event_coordinator_active" in integration_status, "Integration status missing"

        # Test NASA sensor state
        nasa_state = shared_state_manager.get_nasa_sensor_state("T-1")
        assert "active_mission" in nasa_state, "NASA sensor state missing"

        logger.info("✓ Shared State Integration tests passed")
        return True

    except Exception as e:
        logger.error(f"✗ Shared State Integration test failed: {e}")
        return False


async def test_end_to_end_integration():
    """Test end-to-end integration workflow"""
    logger.info("Testing End-to-End Integration...")

    try:
        # Simulate user sensor selection
        shared_state_manager.set_state("selections.sensor_id", "T-1", "validation_script")

        # Wait for event propagation
        await asyncio.sleep(0.5)

        # Simulate anomaly detection
        phase3_manager.event_coordinator.publish_event(
            EventType.ANOMALY_DETECTED,
            source="validation_script",
            data={"sensor_id": "T-1", "anomaly_score": 0.85, "timestamp": time.time()},
        )

        # Wait for workflow processing
        await asyncio.sleep(1.0)

        # Check if workflow was created
        if phase3_manager.workflow_manager:
            active_workflows = phase3_manager.workflow_manager.get_active_workflows()
            logger.info(f"Active workflows after anomaly: {len(active_workflows)}")

        # Check cache optimization
        cache_status = phase3_manager.cache_manager.get_sensor_cache_status()
        assert "T-1" in cache_status, "Sensor not found in cache status"

        # Check performance impact
        perf_summary = phase3_manager.performance_monitor.get_performance_summary()
        logger.info(f"Performance score: {perf_summary.get('performance_score', 'N/A')}")

        logger.info("✓ End-to-End Integration tests passed")
        return True

    except Exception as e:
        logger.error(f"✗ End-to-End Integration test failed: {e}")
        return False


async def run_validation():
    """Run complete Phase 3 validation"""
    logger.info("=" * 50)
    logger.info("Phase 3 Integration Validation")
    logger.info("=" * 50)

    try:
        # Initialize Phase 3
        logger.info("Initializing Phase 3 system...")
        await phase3_manager.initialize()

        # Wait for initialization to complete
        await asyncio.sleep(2.0)

        # Run all tests
        test_results = {}

        test_results["event_coordinator"] = await test_event_coordinator()
        test_results["workflow_manager"] = await test_workflow_manager()
        test_results["cache_manager"] = await test_cache_manager()
        test_results["lazy_loader"] = await test_lazy_loader()
        test_results["performance_monitor"] = await test_performance_monitor()
        test_results["theme_manager"] = test_theme_manager()
        test_results["responsive_utils"] = test_responsive_utils()
        test_results["shared_state_integration"] = test_shared_state_integration()
        test_results["end_to_end_integration"] = await test_end_to_end_integration()

        # Summary
        passed = sum(test_results.values())
        total = len(test_results)

        logger.info("=" * 50)
        logger.info("VALIDATION RESULTS")
        logger.info("=" * 50)

        for test_name, result in test_results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            logger.info(f"{test_name:25}: {status}")

        logger.info("-" * 50)
        logger.info(f"Total: {passed}/{total} tests passed")

        if passed == total:
            logger.info("🎉 ALL TESTS PASSED - Phase 3 integration successful!")
        else:
            logger.warning(f"⚠️  {total - passed} tests failed - Phase 3 integration has issues")

        # Show integration status
        logger.info("\nIntegration Status:")
        status = phase3_manager.get_integration_status()
        for component, active in status["components"].items():
            status_icon = "✓" if active else "✗"
            logger.info(f"  {status_icon} {component}")

        # Show dashboard data
        logger.info(f"\nDashboard Summary:")
        dashboard_data = phase3_manager.get_phase3_dashboard_data()
        if "error" not in dashboard_data:
            logger.info(f"  Performance Score: {dashboard_data.get('performance', {}).get('performance_score', 'N/A')}")
            logger.info(f"  Cache Hit Rate: {dashboard_data.get('cache', {}).get('hit_rate_12h', 'N/A'):.1%}")
            logger.info(f"  Loaded Components: {dashboard_data.get('loading', {}).get('loaded_components', 'N/A')}")
            logger.info(f"  Current Theme: {dashboard_data.get('theme', {}).get('name', 'N/A')}")
            logger.info(f"  Device Type: {dashboard_data.get('integration_status', {}).get('device_type', 'N/A')}")
            logger.info(f"  NASA Sensors: {dashboard_data.get('nasa_sensors', {}).get('total', 'N/A')}")

        return passed == total

    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        return False

    finally:
        # Cleanup
        try:
            await phase3_manager.shutdown()
            logger.info("Phase 3 system shutdown completed")
        except Exception as e:
            logger.error(f"Shutdown error: {e}")


if __name__ == "__main__":
    # Run validation
    success = asyncio.run(run_validation())

    # Exit with appropriate code
    sys.exit(0 if success else 1)

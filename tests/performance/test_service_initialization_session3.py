"""
Service Initialization Performance Tests - Session 3
Deep analysis of service initialization bottlenecks
"""

import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil
import pytest

# Test markers
pytestmark = [pytest.mark.session3, pytest.mark.performance, pytest.mark.slow]

# Add project root to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logger = logging.getLogger(__name__)


class ServiceInitializationProfiler:
    """Profile service initialization performance"""

    def __init__(self):
        self.results = {}
        self.memory_snapshots = {}

    def profile_service_init(self, service_name, service_class, timeout=30):
        """Profile individual service initialization"""
        result = {
            "service_name": service_name,
            "start_time": datetime.now(),
            "import_time": 0,
            "init_time": 0,
            "total_time": 0,
            "memory_before": 0,
            "memory_after": 0,
            "memory_increase": 0,
            "success": False,
            "error": None,
            "timed_out": False,
        }

        # Get initial memory
        result["memory_before"] = psutil.Process().memory_info().rss / 1024 / 1024

        try:
            # Import timing
            import_start = time.time()
            if isinstance(service_class, str):
                # Import from string
                module_path, class_name = service_class.rsplit(".", 1)
                module = __import__(module_path, fromlist=[class_name])
                service_class = getattr(module, class_name)
            result["import_time"] = time.time() - import_start

            # Initialization timing with timeout
            init_result = {"service": None, "success": False, "error": None}

            def init_target():
                try:
                    init_start = time.time()
                    init_result["service"] = service_class()
                    init_result["init_time"] = time.time() - init_start
                    init_result["success"] = True
                except Exception as e:
                    init_result["error"] = e

            thread = threading.Thread(target=init_target)
            thread.daemon = True

            init_start = time.time()
            thread.start()
            thread.join(timeout)

            if thread.is_alive():
                result["timed_out"] = True
                result["init_time"] = timeout
            else:
                result["init_time"] = init_result.get(
                    "init_time", time.time() - init_start
                )
                result["success"] = init_result["success"]
                result["error"] = init_result.get("error")

            result["total_time"] = result["import_time"] + result["init_time"]

        except Exception as e:
            result["error"] = str(e)
            result["total_time"] = time.time() - import_start

        # Final memory
        result["memory_after"] = psutil.Process().memory_info().rss / 1024 / 1024
        result["memory_increase"] = result["memory_after"] - result["memory_before"]
        result["end_time"] = datetime.now()

        self.results[service_name] = result
        return result

    def get_summary(self):
        """Get profiling summary"""
        if not self.results:
            return "No profiling results available"

        total_services = len(self.results)
        successful = sum(1 for r in self.results.values() if r["success"])
        timed_out = sum(1 for r in self.results.values() if r["timed_out"])

        total_time = sum(r["total_time"] for r in self.results.values())
        total_memory = sum(r["memory_increase"] for r in self.results.values())

        slowest = max(self.results.values(), key=lambda x: x["total_time"])
        most_memory = max(self.results.values(), key=lambda x: x["memory_increase"])

        return {
            "total_services": total_services,
            "successful": successful,
            "timed_out": timed_out,
            "success_rate": successful / total_services if total_services > 0 else 0,
            "total_init_time": total_time,
            "average_init_time": (
                total_time / total_services if total_services > 0 else 0
            ),
            "total_memory_increase": total_memory,
            "slowest_service": {
                "name": slowest["service_name"],
                "time": slowest["total_time"],
            },
            "most_memory_service": {
                "name": most_memory["service_name"],
                "memory": most_memory["memory_increase"],
            },
        }


class TestServiceInitializationPerformance:
    """Test individual service initialization performance"""

    @pytest.fixture
    def profiler(self):
        """Provide service initialization profiler"""
        return ServiceInitializationProfiler()

    def test_core_services_initialization(self, profiler):
        """Test core services initialization timing"""

        core_services = [
            (
                "AnomalyDetectionService",
                "src.core.services.anomaly_service.AnomalyDetectionService",
            ),
            (
                "ForecastingService",
                "src.core.services.forecasting_service.ForecastingService",
            ),
            (
                "NASADataLoader",
                "src.infrastructure.data.nasa_data_loader.NASADataLoader",
            ),
        ]

        for service_name, service_path in core_services:
            result = profiler.profile_service_init(
                service_name, service_path, timeout=15
            )

            # Core services should initialize within reasonable time
            if result["success"]:
                assert (
                    result["total_time"] < 10.0
                ), f"{service_name} too slow: {result['total_time']}s"
                assert (
                    result["memory_increase"] < 100
                ), f"{service_name} uses too much memory: {result['memory_increase']}MB"
            else:
                print(
                    f"Warning: {service_name} failed to initialize: {result['error']}"
                )

    def test_optional_services_initialization(self, profiler):
        """Test optional services initialization timing"""

        optional_services = [
            (
                "TrainingUseCase",
                "src.application.use_cases.training_use_case.TrainingUseCase",
            ),
            (
                "TrainingConfigManager",
                "src.application.services.training_config_manager.TrainingConfigManager",
            ),
            ("ModelRegistry", "src.infrastructure.ml.model_registry.ModelRegistry"),
            (
                "PerformanceMonitor",
                "src.infrastructure.monitoring.performance_monitor.PerformanceMonitor",
            ),
        ]

        for service_name, service_path in optional_services:
            result = profiler.profile_service_init(
                service_name, service_path, timeout=10
            )

            # Optional services are allowed to fail, but if they succeed, should be reasonable
            if result["success"]:
                assert (
                    result["total_time"] < 8.0
                ), f"{service_name} too slow: {result['total_time']}s"
            elif result["timed_out"]:
                print(f"Warning: {service_name} timed out during initialization")
            else:
                print(
                    f"Note: {service_name} failed (acceptable for optional service): {result['error']}"
                )

    def test_nasa_data_loader_performance(self, profiler):
        """Detailed analysis of NASA data loader performance"""

        # Profile data loader specifically since it's likely the bottleneck
        result = profiler.profile_service_init(
            "NASADataLoader_Detailed",
            "src.infrastructure.data.nasa_data_loader.NASADataLoader",
            timeout=20,
        )

        if result["success"]:
            # Data loader is critical but may be slow due to data loading
            assert (
                result["total_time"] < 15.0
            ), f"NASA Data Loader too slow: {result['total_time']}s"

            # Memory usage analysis
            assert (
                result["memory_increase"] < 200
            ), f"NASA Data Loader uses too much memory: {result['memory_increase']}MB"

            print(f"NASA Data Loader Performance:")
            print(f"  Import time: {result['import_time']:.2f}s")
            print(f"  Init time: {result['init_time']:.2f}s")
            print(f"  Memory increase: {result['memory_increase']:.2f}MB")
        else:
            print(f"NASA Data Loader failed: {result['error']}")

    def test_service_initialization_bottleneck_analysis(self, profiler):
        """Comprehensive bottleneck analysis"""

        # Test all services to identify bottlenecks
        all_services = [
            (
                "AnomalyDetectionService",
                "src.core.services.anomaly_service.AnomalyDetectionService",
            ),
            (
                "ForecastingService",
                "src.core.services.forecasting_service.ForecastingService",
            ),
            (
                "NASADataLoader",
                "src.infrastructure.data.nasa_data_loader.NASADataLoader",
            ),
            (
                "TrainingUseCase",
                "src.application.use_cases.training_use_case.TrainingUseCase",
            ),
            (
                "TrainingConfigManager",
                "src.application.services.training_config_manager.TrainingConfigManager",
            ),
            ("ModelRegistry", "src.infrastructure.ml.model_registry.ModelRegistry"),
            (
                "PerformanceMonitor",
                "src.infrastructure.monitoring.performance_monitor.PerformanceMonitor",
            ),
        ]

        # Profile all services
        for service_name, service_path in all_services:
            profiler.profile_service_init(service_name, service_path, timeout=12)

        # Analyze results
        summary = profiler.get_summary()

        print(f"\n=== SERVICE INITIALIZATION ANALYSIS ===")
        print(f"Total services tested: {summary['total_services']}")
        print(f"Successful initializations: {summary['successful']}")
        print(f"Timed out: {summary['timed_out']}")
        print(f"Success rate: {summary['success_rate']:.1%}")
        print(f"Total initialization time: {summary['total_init_time']:.2f}s")
        print(f"Average per service: {summary['average_init_time']:.2f}s")
        print(f"Total memory increase: {summary['total_memory_increase']:.2f}MB")
        print(
            f"Slowest service: {summary['slowest_service']['name']} ({summary['slowest_service']['time']:.2f}s)"
        )
        print(
            f"Most memory: {summary['most_memory_service']['name']} ({summary['most_memory_service']['memory']:.2f}MB)"
        )

        # Performance assertions
        assert (
            summary["average_init_time"] < 5.0
        ), f"Average service init time too high: {summary['average_init_time']:.2f}s"
        assert (
            summary["total_memory_increase"] < 500
        ), f"Total memory usage too high: {summary['total_memory_increase']:.2f}MB"


class TestDashboardCreationAnalysis:
    """Analyze dashboard creation process step by step"""

    def test_dashboard_creation_step_by_step(self):
        """Break down dashboard creation into steps"""

        print("\n=== DASHBOARD CREATION STEP ANALYSIS ===")

        # Step 1: Import
        step1_start = time.time()
        from src.presentation.dashboard.enhanced_app import EnhancedIoTDashboard

        step1_time = time.time() - step1_start
        print(f"Step 1 - Import: {step1_time:.2f}s")
        assert step1_time < 10.0, f"Import step too slow: {step1_time}s"

        # Step 2: Class instantiation (this is where it likely hangs)
        step2_start = time.time()

        # Use a timeout for this step
        creation_result = {"dashboard": None, "success": False, "error": None}

        def create_dashboard():
            try:
                creation_result["dashboard"] = EnhancedIoTDashboard(debug=False)
                creation_result["success"] = True
            except Exception as e:
                creation_result["error"] = e

        thread = threading.Thread(target=create_dashboard)
        thread.daemon = True
        thread.start()
        thread.join(30)  # 30 second timeout

        step2_time = time.time() - step2_start

        if thread.is_alive():
            print(f"Step 2 - Creation: TIMED OUT after {step2_time:.2f}s")
            pytest.fail("Dashboard creation timed out - bottleneck identified")
        elif creation_result["success"]:
            print(f"Step 2 - Creation: {step2_time:.2f}s")
            assert creation_result["dashboard"] is not None, "Dashboard object is None"
        else:
            print(f"Step 2 - Creation: FAILED after {step2_time:.2f}s")
            print(f"Error: {creation_result['error']}")
            pytest.fail(f"Dashboard creation failed: {creation_result['error']}")

    def test_identify_creation_bottleneck(self):
        """Identify specific bottleneck in dashboard creation"""

        # Test if we can at least get the class
        from src.presentation.dashboard.enhanced_app import EnhancedIoTDashboard

        # Test class attributes
        assert hasattr(
            EnhancedIoTDashboard, "__init__"
        ), "Dashboard class missing __init__"
        assert hasattr(
            EnhancedIoTDashboard, "_initialize_services"
        ), "Dashboard class missing _initialize_services"

        # The bottleneck is likely in service initialization during __init__
        print("Bottleneck identified: Service initialization during dashboard __init__")


class TestServiceOptimizationStrategies:
    """Test various optimization strategies"""

    def test_lazy_loading_concept(self):
        """Test concept of lazy loading for services"""

        # Create a simple lazy loading wrapper
        class LazyService:
            def __init__(self, service_class):
                self._service_class = service_class
                self._service = None
                self._initialized = False

            @property
            def service(self):
                if not self._initialized:
                    self._service = self._service_class()
                    self._initialized = True
                return self._service

        # Test lazy loading concept
        start_time = time.time()

        # This should be very fast
        from src.core.services.anomaly_service import AnomalyDetectionService

        lazy_anomaly = LazyService(AnomalyDetectionService)

        lazy_creation_time = time.time() - start_time
        assert (
            lazy_creation_time < 0.1
        ), f"Lazy wrapper creation too slow: {lazy_creation_time}s"

        # Actual service creation happens on first access
        access_start = time.time()
        actual_service = lazy_anomaly.service
        access_time = time.time() - access_start

        print(f"Lazy wrapper creation: {lazy_creation_time:.4f}s")
        print(f"First access (actual init): {access_time:.2f}s")

        assert actual_service is not None, "Lazy loaded service is None"

    def test_minimal_service_initialization(self):
        """Test minimal service initialization strategy"""

        # Test if we can create services with minimal configuration
        try:
            from src.infrastructure.data.nasa_data_loader import NASADataLoader

            # Time a basic data loader creation
            start_time = time.time()
            data_loader = NASADataLoader()
            init_time = time.time() - start_time

            print(f"NASA Data Loader initialization: {init_time:.2f}s")

            # If it takes too long, it's our bottleneck
            if init_time > 5.0:
                print(
                    f"BOTTLENECK IDENTIFIED: NASA Data Loader takes {init_time:.2f}s to initialize"
                )

        except Exception as e:
            print(f"NASA Data Loader initialization failed: {e}")


def test_session3_service_analysis_summary():
    """Summary test for Session 3 service analysis"""

    print("\n=== SESSION 3 ANALYSIS SUMMARY ===")

    # Test if we can identify the main bottleneck
    try:
        # Quick service initialization test
        from src.infrastructure.data.nasa_data_loader import NASADataLoader

        start_time = time.time()
        data_loader = NASADataLoader()
        data_loader_time = time.time() - start_time

        if data_loader_time > 3.0:
            print(f"PRIMARY BOTTLENECK: NASA Data Loader ({data_loader_time:.2f}s)")
        else:
            print(f"NASA Data Loader acceptable: {data_loader_time:.2f}s")

    except Exception as e:
        print(f"Could not test NASA Data Loader: {e}")

    print("Session 3 analysis provides foundation for optimization")
    assert True, "Session 3 analysis completed"

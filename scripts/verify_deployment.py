#!/usr/bin/env python3
"""
Deployment Verification Script
Test deployment after installation
"""

import os
import sys
import threading
import time
from pathlib import Path

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


class TestResult:
    """Track test results"""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.tests = []

    def add_test(
        self, name: str, passed: bool, message: str = "", warning: bool = False
    ):
        """Add test result"""
        self.tests.append(
            {"name": name, "passed": passed, "warning": warning, "message": message}
        )

        if warning:
            self.warnings += 1
        elif passed:
            self.passed += 1
        else:
            self.failed += 1

    def print_summary(self):
        """Print test summary"""
        print(f"\n{BOLD}{'='*70}{RESET}")
        print(f"{BOLD}DEPLOYMENT VERIFICATION SUMMARY{RESET}")
        print(f"{BOLD}{'='*70}{RESET}\n")

        for test in self.tests:
            if test["warning"]:
                icon = f"{YELLOW}⚠{RESET}"
                status = "WARNING"
            elif test["passed"]:
                icon = f"{GREEN}✓{RESET}"
                status = "PASSED"
            else:
                icon = f"{RED}✗{RESET}"
                status = "FAILED"

            print(f"{icon} {test['name']}: {status}")
            if test["message"]:
                print(f"   {test['message']}")

        print(f"\n{BOLD}Results:{RESET}")
        print(f"  {GREEN}Passed: {self.passed}{RESET}")
        if self.warnings > 0:
            print(f"  {YELLOW}Warnings: {self.warnings}{RESET}")
        if self.failed > 0:
            print(f"  {RED}Failed: {self.failed}{RESET}")

        print(f"\n{BOLD}{'='*70}{RESET}\n")

        if self.failed > 0:
            print(f"{RED}{BOLD}❌ DEPLOYMENT VERIFICATION FAILED{RESET}")
            print(f"{YELLOW}Please fix the errors above{RESET}\n")
            return False
        elif self.warnings > 0:
            print(f"{YELLOW}{BOLD}⚠️  DEPLOYMENT VERIFIED WITH WARNINGS{RESET}")
            print(f"{YELLOW}Some features may not work optimally{RESET}\n")
            return True
        else:
            print(f"{GREEN}{BOLD}✅ DEPLOYMENT VERIFIED SUCCESSFULLY{RESET}")
            print(f"{GREEN}System is production-ready!{RESET}\n")
            return True


def test_imports(result: TestResult):
    """Test critical imports"""
    print(f"{BOLD}Testing imports...{RESET}")

    sys.path.insert(0, str(Path(__file__).parent))

    # Test dashboard imports
    try:
        from src.presentation.dashboard.enhanced_app import EnhancedIoTDashboard

        result.add_test("Import EnhancedIoTDashboard", True)
    except ImportError as e:
        result.add_test("Import EnhancedIoTDashboard", False, str(e))

    try:
        from src.presentation.dashboard.enhanced_app_optimized import (
            OptimizedIoTDashboard,
            create_app,
        )

        result.add_test("Import OptimizedIoTDashboard", True)
    except ImportError as e:
        result.add_test("Import OptimizedIoTDashboard", False, str(e))

    try:
        from src.presentation.dashboard.enhanced_callbacks_simplified import (
            create_training_hub_layout,
            register_enhanced_callbacks,
        )

        result.add_test("Import callbacks module", True)
    except ImportError as e:
        result.add_test("Import callbacks module", False, str(e))

    # Test core services
    try:
        from src.core.services.anomaly_service import AnomalyDetectionService
        from src.core.services.forecasting_service import ForecastingService

        result.add_test("Import core services", True)
    except ImportError as e:
        result.add_test("Import core services", False, str(e))

    # Test infrastructure
    try:
        from src.infrastructure.data.nasa_data_loader import NASADataLoader
        from src.infrastructure.ml.model_registry import ModelRegistry

        result.add_test("Import infrastructure", True)
    except ImportError as e:
        result.add_test("Import infrastructure", False, str(e))


def test_dashboard_creation(result: TestResult):
    """Test dashboard can be created"""
    print(f"\n{BOLD}Testing dashboard creation...{RESET}")

    sys.path.insert(0, str(Path(__file__).parent))

    try:
        from src.presentation.dashboard.enhanced_app_optimized import create_app

        # Create dashboard with timeout
        creation_result = {"success": False, "error": None}

        def create_dashboard():
            try:
                app = create_app(debug=False)
                creation_result["success"] = True
                creation_result["app"] = app
            except Exception as e:
                creation_result["error"] = str(e)

        thread = threading.Thread(target=create_dashboard)
        thread.daemon = True
        thread.start()
        thread.join(timeout=30)  # 30 second timeout

        if thread.is_alive():
            result.add_test("Dashboard creation", False, "Timeout after 30 seconds")
        elif creation_result["success"]:
            result.add_test(
                "Dashboard creation", True, "Dashboard created successfully"
            )
        else:
            result.add_test(
                "Dashboard creation", False, f"Error: {creation_result['error']}"
            )

    except Exception as e:
        result.add_test("Dashboard creation", False, str(e))


def test_configuration(result: TestResult):
    """Test configuration loading"""
    print(f"\n{BOLD}Testing configuration...{RESET}")

    sys.path.insert(0, str(Path(__file__).parent))

    try:
        from config.settings import settings

        result.add_test("Load settings", True)
    except Exception as e:
        result.add_test("Load settings", False, str(e))

    try:
        from config.equipment_config import EQUIPMENT_REGISTRY, get_equipment_list

        equipment_list = get_equipment_list()
        if len(equipment_list) == 12:
            result.add_test("Equipment config", True, f"12 equipment units configured")
        else:
            result.add_test(
                "Equipment config",
                True,
                f"{len(equipment_list)} equipment units",
                warning=True,
            )
    except Exception as e:
        result.add_test("Equipment config", False, str(e))


def test_data_access(result: TestResult):
    """Test data can be loaded"""
    print(f"\n{BOLD}Testing data access...{RESET}")

    sys.path.insert(0, str(Path(__file__).parent))

    try:
        from src.infrastructure.data.nasa_data_loader import NASADataLoader

        data_loader = NASADataLoader("data/raw")

        # Try loading SMAP data
        try:
            sensor_data = data_loader.get_sensor_data("SMAP-PWR-001", hours_back=24)
            if sensor_data and sensor_data.get("values"):
                result.add_test(
                    "Load SMAP data", True, f"{len(sensor_data['values'])} data points"
                )
            else:
                result.add_test(
                    "Load SMAP data", True, "No data (will use mock)", warning=True
                )
        except Exception as e:
            result.add_test(
                "Load SMAP data",
                True,
                "Data loading error (will use mock)",
                warning=True,
            )

    except Exception as e:
        result.add_test("Data access", False, str(e))


def test_model_availability(result: TestResult):
    """Test model availability"""
    print(f"\n{BOLD}Testing model availability...{RESET}")

    project_root = Path(__file__).parent
    models_dir = project_root / "data" / "models"

    if not models_dir.exists():
        result.add_test(
            "Model availability", True, "No models (forecasting disabled)", warning=True
        )
        return

    model_files = list(models_dir.glob("**/*.h5"))

    if model_files:
        result.add_test("Model availability", True, f"{len(model_files)} models found")
    else:
        result.add_test(
            "Model availability", True, "No models (forecasting disabled)", warning=True
        )


def test_startup_scripts(result: TestResult):
    """Test startup scripts exist"""
    print(f"\n{BOLD}Testing startup scripts...{RESET}")

    project_root = Path(__file__).parent

    scripts = [
        ("start_dashboard.py", "Dashboard launcher"),
        ("preflight_check.py", "Pre-flight check"),
        ("validate_startup.py", "Startup validation"),
        ("verify_deployment.py", "This script"),
    ]

    all_present = True
    for script, description in scripts:
        script_path = project_root / script
        if script_path.exists():
            pass  # Don't report individual scripts
        else:
            all_present = False
            result.add_test(f"Script: {description}", False, f"{script} missing")

    if all_present:
        result.add_test("Startup scripts", True, "All scripts present")


def main():
    """Run deployment verification"""
    print(f"\n{BOLD}{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}{BLUE}DEPLOYMENT VERIFICATION{RESET}")
    print(f"{BOLD}{BLUE}Testing complete installation{RESET}")
    print(f"{BOLD}{BLUE}{'='*70}{RESET}\n")

    result = TestResult()

    # Run all tests
    test_imports(result)
    test_configuration(result)
    test_data_access(result)
    test_model_availability(result)
    test_dashboard_creation(result)
    test_startup_scripts(result)

    # Print summary
    success = result.print_summary()

    if success:
        print(f"{BOLD}Next steps:{RESET}")
        print(f"  1. Run: {BLUE}python start_dashboard.py{RESET}")
        print(f"  2. Open: {BLUE}http://127.0.0.1:8050{RESET}")
        print(f"  3. Test all dashboard tabs\n")
        return 0
    else:
        print(f"{BOLD}Troubleshooting:{RESET}")
        print(f"  1. Check: {BLUE}TROUBLESHOOTING.md{RESET}")
        print(f"  2. Run: {BLUE}python validate_startup.py{RESET}")
        print(f"  3. Review logs in {BLUE}logs/{RESET} directory\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

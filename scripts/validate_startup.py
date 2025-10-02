#!/usr/bin/env python3
"""
Startup Validation Script
Comprehensive checks before starting the dashboard
"""

import importlib
import os
import socket
import sys
from pathlib import Path

# ANSI color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


class ValidationResult:
    """Track validation results"""

    def __init__(self):
        self.passed = []
        self.warnings = []
        self.failed = []

    def add_pass(self, check: str, message: str = ""):
        self.passed.append((check, message))

    def add_warning(self, check: str, message: str):
        self.warnings.append((check, message))

    def add_fail(self, check: str, message: str):
        self.failed.append((check, message))

    def print_summary(self):
        """Print validation summary"""
        print(f"\n{BOLD}{'='*70}{RESET}")
        print(f"{BOLD}VALIDATION SUMMARY{RESET}")
        print(f"{BOLD}{'='*70}{RESET}\n")

        print(f"{GREEN}✓ Passed: {len(self.passed)}{RESET}")
        for check, msg in self.passed:
            print(f"  {GREEN}✓{RESET} {check}")
            if msg:
                print(f"    {msg}")

        if self.warnings:
            print(f"\n{YELLOW}⚠ Warnings: {len(self.warnings)}{RESET}")
            for check, msg in self.warnings:
                print(f"  {YELLOW}⚠{RESET} {check}")
                print(f"    {msg}")

        if self.failed:
            print(f"\n{RED}✗ Failed: {len(self.failed)}{RESET}")
            for check, msg in self.failed:
                print(f"  {RED}✗{RESET} {check}")
                print(f"    {msg}")

        print(f"\n{BOLD}{'='*70}{RESET}\n")

        if self.failed:
            print(f"{RED}{BOLD}❌ VALIDATION FAILED{RESET}")
            print(
                f"{YELLOW}Please fix the errors above before starting the dashboard{RESET}\n"
            )
            return False
        elif self.warnings:
            print(f"{YELLOW}{BOLD}⚠️  VALIDATION PASSED WITH WARNINGS{RESET}")
            print(
                f"{YELLOW}Dashboard can start, but some features may not work optimally{RESET}\n"
            )
            return True
        else:
            print(f"{GREEN}{BOLD}✅ ALL CHECKS PASSED{RESET}")
            print(f"{GREEN}Dashboard is ready to start!{RESET}\n")
            return True


def check_python_version(result: ValidationResult):
    """Check Python version"""
    major, minor = sys.version_info[:2]
    if major == 3 and minor >= 8:
        result.add_pass("Python version", f"Python {major}.{minor}")
    else:
        result.add_fail(
            "Python version", f"Python 3.8+ required, found {major}.{minor}"
        )


def check_dependencies(result: ValidationResult):
    """Check required Python packages"""
    required_packages = [
        "dash",
        "dash_bootstrap_components",
        "plotly",
        "pandas",
        "numpy",
        "tensorflow",
        "sklearn",
        "yaml",
        "joblib",
    ]

    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package.replace("-", "_"))
        except ImportError:
            missing.append(package)

    if not missing:
        result.add_pass(
            "Dependencies", f"All {len(required_packages)} packages installed"
        )
    else:
        result.add_fail("Dependencies", f"Missing packages: {', '.join(missing)}")


def check_directory_structure(result: ValidationResult):
    """Check required directories exist"""
    project_root = Path(__file__).parent
    required_dirs = [
        "data",
        "data/raw",
        "data/raw/smap",
        "data/raw/msl",
        "data/models",
        "logs",
        "config",
        "src",
        "src/core",
        "src/application",
        "src/infrastructure",
        "src/presentation",
    ]

    missing = []
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            missing.append(dir_path)

    if not missing:
        result.add_pass(
            "Directory structure", f"All {len(required_dirs)} directories present"
        )
    else:
        result.add_fail(
            "Directory structure", f"Missing directories: {', '.join(missing)}"
        )


def check_nasa_data_files(result: ValidationResult):
    """Check NASA data files"""
    project_root = Path(__file__).parent
    smap_data = project_root / "data" / "raw" / "smap"
    msl_data = project_root / "data" / "raw" / "msl"

    required_files = [
        (smap_data / "train.npy", "SMAP training data"),
        (smap_data / "test.npy", "SMAP test data"),
        (msl_data / "train.npy", "MSL training data"),
        (msl_data / "test.npy", "MSL test data"),
    ]

    missing = []
    for file_path, description in required_files:
        if not file_path.exists():
            missing.append(description)

    if not missing:
        result.add_pass("NASA data files", "All data files present")
    else:
        result.add_fail("NASA data files", f"Missing: {', '.join(missing)}")


def check_model_files(result: ValidationResult):
    """Check model files"""
    project_root = Path(__file__).parent
    models_dir = project_root / "data" / "models"

    if not models_dir.exists():
        result.add_warning(
            "Model files", "Models directory missing - training may be required"
        )
        return

    # Count model files
    model_files = list(models_dir.glob("**/*.h5")) + list(models_dir.glob("**/*.pkl"))

    if model_files:
        result.add_pass("Model files", f"Found {len(model_files)} model files")
    else:
        result.add_warning(
            "Model files", "No models found - training required before forecasting"
        )


def check_config_files(result: ValidationResult):
    """Check configuration files"""
    project_root = Path(__file__).parent
    config_files = [
        ("config/config.yaml", "Main configuration"),
        ("config/equipment_config.py", "Equipment configuration"),
        ("config/settings.py", "Settings module"),
    ]

    missing = []
    for file_path, description in config_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing.append(f"{description} ({file_path})")

    if not missing:
        result.add_pass("Configuration files", "All config files present")
    else:
        result.add_fail("Configuration files", f"Missing: {', '.join(missing)}")


def check_dashboard_files(result: ValidationResult):
    """Check dashboard files"""
    project_root = Path(__file__).parent
    dashboard_files = [
        "src/presentation/dashboard/enhanced_app.py",
        "src/presentation/dashboard/enhanced_app_optimized.py",
        "src/presentation/dashboard/enhanced_callbacks_simplified.py",
        "src/presentation/dashboard/app.py",
    ]

    missing = []
    for file_path in dashboard_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing.append(file_path)

    if not missing:
        result.add_pass("Dashboard files", "All dashboard files present")
    else:
        result.add_fail("Dashboard files", f"Missing: {', '.join(missing)}")


def check_port_availability(result: ValidationResult, port=8050):
    """Check if dashboard port is available"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
        sock.close()
        result.add_pass("Port availability", f"Port {port} is available")
    except OSError:
        result.add_warning(
            "Port availability", f"Port {port} is in use - dashboard may fail to start"
        )


def check_disk_space(result: ValidationResult):
    """Check available disk space"""
    try:
        import shutil

        project_root = Path(__file__).parent
        stat = shutil.disk_usage(project_root)
        free_gb = stat.free / (1024**3)

        if free_gb > 1.0:
            result.add_pass("Disk space", f"{free_gb:.1f} GB available")
        else:
            result.add_warning("Disk space", f"Low disk space: {free_gb:.1f} GB")
    except Exception as e:
        result.add_warning("Disk space", f"Could not check disk space: {e}")


def check_core_imports(result: ValidationResult):
    """Check if core modules can be imported"""
    sys.path.insert(0, str(Path(__file__).parent))

    core_modules = [
        "config.settings",
        "config.equipment_config",
        "src.core.services.anomaly_service",
        "src.core.services.forecasting_service",
        "src.infrastructure.data.nasa_data_loader",
        "src.infrastructure.ml.model_registry",
    ]

    failed_imports = []
    for module in core_modules:
        try:
            importlib.import_module(module)
        except ImportError as e:
            failed_imports.append(f"{module}: {str(e)}")

    if not failed_imports:
        result.add_pass(
            "Core imports", f"All {len(core_modules)} core modules importable"
        )
    else:
        result.add_fail(
            "Core imports", f"Failed imports:\n    " + "\n    ".join(failed_imports)
        )


def main():
    """Run all validation checks"""
    print(f"\n{BOLD}{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}{BLUE}IoT PREDICTIVE MAINTENANCE SYSTEM{RESET}")
    print(f"{BOLD}{BLUE}Startup Validation{RESET}")
    print(f"{BOLD}{BLUE}{'='*70}{RESET}\n")

    result = ValidationResult()

    print(f"{BOLD}Running validation checks...{RESET}\n")

    # Run all checks
    check_python_version(result)
    check_dependencies(result)
    check_directory_structure(result)
    check_config_files(result)
    check_dashboard_files(result)
    check_nasa_data_files(result)
    check_model_files(result)
    check_core_imports(result)
    check_port_availability(result)
    check_disk_space(result)

    # Print summary
    success = result.print_summary()

    if not success:
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-Flight Check Script
Quick validation and automatic fixes before dashboard launch
"""

import os
import subprocess
import sys
from pathlib import Path

# Set UTF-8 encoding for Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"

# ASCII-safe check marks for Windows compatibility
CHECK = "[OK]"
CROSS = "[X]"
WARN = "[!]"


def print_header():
    """Print header"""
    print(f"\n{BOLD}{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}{BLUE}PRE-FLIGHT CHECK{RESET}")
    print(f"{BOLD}{BLUE}Quick validation and auto-fix{RESET}")
    print(f"{BOLD}{BLUE}{'='*70}{RESET}\n")


def check_and_fix_directories():
    """Create missing directories"""
    project_root = Path(__file__).parent
    required_dirs = [
        "data",
        "data/raw",
        "data/raw/smap",
        "data/raw/msl",
        "data/models",
        "data/models/registry",
        "data/models/transformer",
        "data/processed",
        "logs",
        "cache",
    ]

    print(f"{BOLD}Checking directories...{RESET}")
    created = []
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            full_path.mkdir(parents=True, exist_ok=True)
            created.append(dir_path)
            print(f"  {GREEN}{CHECK}{RESET} Created: {dir_path}")
        else:
            print(f"  {GREEN}{CHECK}{RESET} Exists: {dir_path}")

    if created:
        print(f"\n{YELLOW}Created {len(created)} missing directories{RESET}")
    print()


def check_dependencies():
    """Check Python dependencies"""
    print(f"{BOLD}Checking dependencies...{RESET}")

    try:
        import dash
        import dash_bootstrap_components as dbc
        import numpy as np
        import pandas as pd
        import plotly

        print(f"  {GREEN}✓{RESET} dash: {dash.__version__}")
        print(f"  {GREEN}✓{RESET} dash-bootstrap-components: {dbc.__version__}")
        print(f"  {GREEN}✓{RESET} plotly: {plotly.__version__}")
        print(f"  {GREEN}✓{RESET} pandas: {pd.__version__}")
        print(f"  {GREEN}✓{RESET} numpy: {np.__version__}")

        try:
            import tensorflow as tf

            print(f"  {GREEN}✓{RESET} tensorflow: {tf.__version__}")
        except ImportError:
            print(f"  {YELLOW}⚠{RESET} tensorflow: Not available (forecasting may not work)")

        try:
            import sklearn

            print(f"  {GREEN}✓{RESET} scikit-learn: {sklearn.__version__}")
        except ImportError:
            print(f"  {YELLOW}⚠{RESET} scikit-learn: Not available")

        print(f"\n{GREEN}Core dependencies OK{RESET}\n")
        return True

    except ImportError as e:
        print(f"\n{RED}Missing dependency: {e}{RESET}")
        print(f"{YELLOW}Run: pip install -r requirements.txt{RESET}\n")
        return False


def check_data_files():
    """Check NASA data files"""
    print(f"{BOLD}Checking NASA data files...{RESET}")
    project_root = Path(__file__).parent

    smap_train = project_root / "data" / "raw" / "smap" / "train.npy"
    smap_test = project_root / "data" / "raw" / "smap" / "test.npy"
    msl_train = project_root / "data" / "raw" / "msl" / "train.npy"
    msl_test = project_root / "data" / "raw" / "msl" / "test.npy"

    all_present = True
    if smap_train.exists():
        size_mb = smap_train.stat().st_size / (1024 * 1024)
        print(f"  {GREEN}✓{RESET} SMAP train: {size_mb:.1f} MB")
    else:
        print(f"  {RED}✗{RESET} SMAP train: Missing")
        all_present = False

    if smap_test.exists():
        size_mb = smap_test.stat().st_size / (1024 * 1024)
        print(f"  {GREEN}✓{RESET} SMAP test: {size_mb:.1f} MB")
    else:
        print(f"  {RED}✗{RESET} SMAP test: Missing")
        all_present = False

    if msl_train.exists():
        size_mb = msl_train.stat().st_size / (1024 * 1024)
        print(f"  {GREEN}✓{RESET} MSL train: {size_mb:.1f} MB")
    else:
        print(f"  {RED}✗{RESET} MSL train: Missing")
        all_present = False

    if msl_test.exists():
        size_mb = msl_test.stat().st_size / (1024 * 1024)
        print(f"  {GREEN}✓{RESET} MSL test: {size_mb:.1f} MB")
    else:
        print(f"  {RED}✗{RESET} MSL test: Missing")
        all_present = False

    if all_present:
        print(f"\n{GREEN}All data files present{RESET}\n")
    else:
        print(f"\n{RED}Some data files missing{RESET}")
        print(f"{YELLOW}Dashboard will use mock data{RESET}\n")

    return all_present


def check_dashboard_files():
    """Check dashboard files"""
    print(f"{BOLD}Checking dashboard files...{RESET}")
    project_root = Path(__file__).parent

    required_files = {
        "src/presentation/dashboard/enhanced_app.py": "Enhanced app wrapper",
        "src/presentation/dashboard/enhanced_app_optimized.py": "Optimized dashboard",
        "src/presentation/dashboard/enhanced_callbacks_simplified.py": "Callbacks module",
        "start_dashboard.py": "Dashboard launcher",
    }

    all_present = True
    for file_path, description in required_files.items():
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  {GREEN}✓{RESET} {description}")
        else:
            print(f"  {RED}✗{RESET} {description}: Missing")
            all_present = False

    if all_present:
        print(f"\n{GREEN}All dashboard files present{RESET}\n")
    else:
        print(f"\n{RED}Some dashboard files missing{RESET}\n")

    return all_present


def check_models():
    """Check trained models"""
    print(f"{BOLD}Checking trained models...{RESET}")
    project_root = Path(__file__).parent
    models_dir = project_root / "data" / "models"

    if not models_dir.exists():
        print(f"  {YELLOW}⚠{RESET} Models directory missing")
        print(f"\n{YELLOW}No models found - forecasting will not work{RESET}")
        print(f"{YELLOW}Run: python train_forecasting_models.py --quick{RESET}\n")
        return False

    model_files = list(models_dir.glob("**/*.h5"))
    pkl_files = list(models_dir.glob("**/*.pkl"))

    print(f"  {GREEN}✓{RESET} Found {len(model_files)} .h5 model files")
    print(f"  {GREEN}✓{RESET} Found {len(pkl_files)} .pkl scaler files")

    if model_files:
        print(f"\n{GREEN}Models available{RESET}\n")
        return True
    else:
        print(f"\n{YELLOW}No models found - forecasting will not work{RESET}")
        print(f"{YELLOW}Run: python train_forecasting_models.py --quick{RESET}\n")
        return False


def print_summary(checks):
    """Print summary"""
    print(f"{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}PRE-FLIGHT CHECK SUMMARY{RESET}")
    print(f"{BOLD}{'='*70}{RESET}\n")

    all_passed = all(checks.values())

    if all_passed:
        print(f"{GREEN}{BOLD}✅ ALL CHECKS PASSED{RESET}")
        print(f"{GREEN}System is ready to launch!{RESET}\n")
        print(f"{BOLD}To start the dashboard:{RESET}")
        print(f"  {BLUE}python start_dashboard.py{RESET}\n")
    else:
        failed = [name for name, passed in checks.items() if not passed]
        print(f"{YELLOW}{BOLD}⚠️  CHECKS COMPLETED WITH WARNINGS{RESET}")
        print(f"{YELLOW}Some components not available: {', '.join(failed)}{RESET}\n")
        print(f"{BOLD}Dashboard can still start, but some features may not work{RESET}\n")
        print(f"{BOLD}To start anyway:{RESET}")
        print(f"  {BLUE}python start_dashboard.py{RESET}\n")


def main():
    """Run pre-flight checks"""
    print_header()

    # Run checks
    check_and_fix_directories()
    deps_ok = check_dependencies()
    data_ok = check_data_files()
    dashboard_ok = check_dashboard_files()
    models_ok = check_models()

    checks = {
        "Dependencies": deps_ok,
        "Data Files": data_ok,
        "Dashboard Files": dashboard_ok,
        "Models": models_ok,
    }

    print_summary(checks)

    # Exit with error only if critical components missing
    if not deps_ok or not dashboard_ok:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

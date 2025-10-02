#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Start Script - Windows Compatible
Simple validation and launch script
"""

import os
import sys
from pathlib import Path


def main():
    """Simple quick start"""
    print("\n" + "=" * 70)
    print("IOT PREDICTIVE MAINTENANCE SYSTEM - QUICK START")
    print("=" * 70 + "\n")

    project_root = Path(__file__).parent

    # Step 1: Create directories
    print("[1/5] Creating required directories...")
    directories = [
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
    for dir_path in directories:
        (project_root / dir_path).mkdir(parents=True, exist_ok=True)
    print("      OK - Directories ready\n")

    # Step 2: Check Python
    print("[2/5] Checking Python version...")
    major, minor = sys.version_info[:2]
    if major == 3 and minor >= 8:
        print(f"      OK - Python {major}.{minor}\n")
    else:
        print(f"      ERROR - Python 3.8+ required, found {major}.{minor}")
        return 1

    # Step 3: Check dependencies
    print("[3/5] Checking dependencies...")
    try:
        import dash
        import numpy
        import pandas
        import plotly

        print("      OK - Core packages installed\n")
    except ImportError as e:
        print(f"      ERROR - Missing: {e}")
        print("      Run: pip install -r requirements.txt\n")
        return 1

    # Step 4: Check dashboard files
    print("[4/5] Checking dashboard files...")
    required_files = [
        "src/presentation/dashboard/enhanced_app.py",
        "src/presentation/dashboard/enhanced_app_optimized.py",
        "src/presentation/dashboard/enhanced_callbacks_simplified.py",
        "start_dashboard.py",
    ]
    missing = []
    for file_path in required_files:
        if not (project_root / file_path).exists():
            missing.append(file_path)

    if missing:
        print(f"      ERROR - Missing files:")
        for f in missing:
            print(f"              {f}")
        return 1
    print("      OK - All dashboard files present\n")

    # Step 5: Data check
    print("[5/5] Checking data files...")
    smap_train = project_root / "data" / "raw" / "smap" / "train.npy"
    if smap_train.exists():
        print("      OK - NASA data available\n")
    else:
        print("      WARNING - NASA data missing (will use mock data)\n")

    # Success message
    print("=" * 70)
    print("SYSTEM READY!")
    print("=" * 70)
    print("\nTo start the dashboard:")
    print("  python start_dashboard.py")
    print("\nDashboard will be available at:")
    print("  http://127.0.0.1:8050")
    print("\n" + "=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

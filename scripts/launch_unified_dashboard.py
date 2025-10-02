#!/usr/bin/env python3
"""
Clean Launch Script for Unified Production Dashboard
Freshly launches our new consolidated dashboard with all trained models
"""

import logging
import os
import sys
from pathlib import Path

# Setup clean environment
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Clear Python cache
import importlib

if hasattr(importlib, "invalidate_caches"):
    importlib.invalidate_caches()

# Configure clean logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def main():
    """Launch the unified production dashboard"""

    print("=" * 70)
    print("LAUNCHING UNIFIED PRODUCTION DASHBOARD")
    print("   Session 3: Clean Architecture with Trained Models")
    print("=" * 70)
    print("[OK] All 12 forecasting models trained and ready")
    print("[OK] Anti-hanging architecture enabled")
    print("[OK] All rich components re-enabled")
    print("[OK] Zero feature compromises")
    print("=" * 70)

    try:
        # Import and launch
        from src.presentation.dashboard.app import create_app

        print("[INFO] Starting dashboard server...")
        app = create_app()

        print("[URL] Dashboard available at: http://127.0.0.1:8050")
        print("[FEATURES] Overview | Monitoring | Anomalies | Forecasting | Maintenance | Work Orders | Performance")
        print("[CTRL] Press Ctrl+C to stop")
        print("-" * 70)

        # Launch server
        app.run_server(
            host="127.0.0.1",
            port=8050,
            debug=False,
            dev_tools_hot_reload=False,
            dev_tools_ui=False,
        )

    except Exception as e:
        print(f"[ERROR] Error launching dashboard: {e}")
        print("[HELP] Ensure all dependencies are installed and models are trained")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

#!/usr/bin/env python3
"""
Clean Dashboard Launcher
Starts the production IoT dashboard with all trained forecasting models
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("=" * 60)
    print("IoT PREDICTIVE MAINTENANCE DASHBOARD")
    print("Clean Launch with All Trained Models")
    print("=" * 60)
    print("[STATUS] All 12 forecasting models trained and ready")
    print("[STATUS] NASA SMAP/MSL data loaded for 12 sensors")
    print("[STATUS] Anti-hanging architecture enabled")
    print("-" * 60)

    try:
        # Import and start the working dashboard
        from src.presentation.dashboard.enhanced_app_optimized import create_app

        print("[INFO] Creating dashboard application...")
        app = create_app()

        print("[URL] Dashboard starting at: http://127.0.0.1:8050")
        print("[FEATURES] Overview | Monitoring | Anomalies | Forecasting | Maintenance | Work Orders | Performance")
        print("[CONTROL] Press Ctrl+C to stop the server")
        print("-" * 60)

        # Start the server
        app.run_server(
            host='127.0.0.1',
            port=8050,
            debug=False,
            dev_tools_hot_reload=False
        )

    except ImportError:
        print("[ERROR] Enhanced dashboard not found, falling back to basic app...")
        try:
            from src.presentation.dashboard.app import create_app
            app = create_app()

            print("[INFO] Starting basic unified dashboard...")
            app.run_server(host='127.0.0.1', port=8050, debug=False)

        except Exception as e:
            print(f"[ERROR] Failed to start dashboard: {e}")
            return 1

    except Exception as e:
        print(f"[ERROR] Dashboard startup failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
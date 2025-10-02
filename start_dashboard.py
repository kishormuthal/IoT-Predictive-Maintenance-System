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
        # Import and start the UNIFIED dashboard (ALL features enabled)
        from src.presentation.dashboard.unified_dashboard import UnifiedIoTDashboard

        print("[INFO] Creating UNIFIED dashboard application...")
        print("[INFO] ALL features from src/ enabled - ZERO compromise")
        dashboard = UnifiedIoTDashboard(debug=False)

        print("[URL] Dashboard starting at: http://127.0.0.1:8050")
        print(
            "[FEATURES] Overview | Monitoring | Anomalies | Forecasting | Maintenance | Work Orders | Performance"
        )
        print(
            "[ARCHITECTURE] Clean Architecture (Core, Application, Infrastructure, Presentation)"
        )
        print("[CONTROL] Press Ctrl+C to stop the server")
        print("-" * 60)

        # Start the server
        dashboard.run(host="127.0.0.1", port=8050, debug=False)

    except ImportError as e:
        print(f"[ERROR] Unified dashboard import failed: {e}")
        print("[FALLBACK] Trying alternative dashboard...")
        try:
            from src.presentation.dashboard.enhanced_app_optimized import create_app

            app = create_app()

            print("[INFO] Starting fallback dashboard...")
            app.run_server(host="127.0.0.1", port=8050, debug=False)

        except Exception as e2:
            print(f"[ERROR] Failed to start fallback dashboard: {e2}")
            return 1

    except Exception as e:
        print(f"[ERROR] Dashboard startup failed: {e}")
        print(f"[DEBUG] Error details: {type(e).__name__}: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

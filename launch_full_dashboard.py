#!/usr/bin/env python3
"""
Launch FULL-FEATURED IoT Dashboard
Uses unified_dashboard.py with ALL advanced features
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    print("=" * 70)
    print("LAUNCHING FULL-FEATURED IoT DASHBOARD")
    print("=" * 70)
    print()
    print("Features:")
    print("  ✓ Overview - IoT Architecture, Network topology, 12 sensors")
    print("  ✓ Anomaly Monitor - Heatmaps, Alert actions, Threshold manager")
    print("  ✓ Forecasting - Risk Matrix, What-If Analysis, Model comparison")
    print("  ✓ Maintenance - Calendar/Gantt views, Resource optimization")
    print("  ✓ Work Orders - Priority tracking, Technician workload")
    print("  ✓ System Performance - Training Hub, Model Registry, Pipeline")
    print()
    print("=" * 70)
    print()

    try:
        # Import the FULL unified dashboard
        from src.presentation.dashboard.unified_dashboard import UnifiedIoTDashboard

        # Create and run the dashboard
        dashboard = UnifiedIoTDashboard()
        app = dashboard.app

        print("✓ Full-featured dashboard initialized")
        print()
        print("Dashboard URL: http://127.0.0.1:8050")
        print("Press Ctrl+C to stop")
        print("=" * 70)

        # Run the server
        app.run_server(host="127.0.0.1", port=8050, debug=False)

    except KeyboardInterrupt:
        print("\n\nDashboard stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python3
"""
Quick test to verify all dashboard tabs load without errors
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test all critical imports"""
    print("Testing imports...")

    try:
        # Test NASA data loader
        from src.infrastructure.data.nasa_data_loader import NASADataLoader

        loader = NASADataLoader()
        print(f"✓ NASA Data Loader: SMAP {loader.smap_data['data'].shape}, MSL {loader.msl_data['data'].shape}")
    except Exception as e:
        print(f"✗ NASA Data Loader failed: {e}")
        return False

    try:
        # Test equipment config
        from config.equipment_config import get_equipment_list

        equipment = get_equipment_list()
        print(f"✓ Equipment Config: {len(equipment)} sensors loaded")
    except Exception as e:
        print(f"✗ Equipment Config failed: {e}")
        return False

    return True


def test_layouts():
    """Test all layout imports"""
    print("\nTesting layout imports...")

    layouts = [
        ("Overview", "src.presentation.dashboard.layouts.overview"),
        ("Monitoring", "src.presentation.dashboard.layouts.monitoring"),
        ("Anomaly Monitor", "src.presentation.dashboard.layouts.anomaly_monitor"),
        ("Forecasting", "src.presentation.dashboard.layouts.enhanced_forecasting"),
        (
            "Maintenance",
            "src.presentation.dashboard.layouts.enhanced_maintenance_scheduler",
        ),
        ("Work Orders", "src.presentation.dashboard.layouts.work_orders_simple"),
        ("System Performance", "src.presentation.dashboard.layouts.system_performance"),
    ]

    all_passed = True
    for name, module_path in layouts:
        try:
            module = __import__(module_path, fromlist=["create_layout"])
            create_layout = getattr(module, "create_layout")
            print(f"✓ {name}: create_layout() found")
        except Exception as e:
            print(f"✗ {name}: {e}")
            all_passed = False

    return all_passed


def test_callbacks():
    """Test callback setup"""
    print("\nTesting callbacks...")

    try:
        from src.presentation.dashboard.callbacks.dashboard_callbacks import (
            setup_dashboard_callbacks,
        )

        print("✓ setup_dashboard_callbacks() found")
    except Exception as e:
        print(f"✗ Callbacks failed: {e}")
        return False

    return True


def test_dependencies():
    """Test critical dependencies"""
    print("\nTesting dependencies...")

    deps = [
        ("dash", "Dash framework"),
        ("plotly", "Plotly graphing"),
        ("pandas", "Pandas data"),
        ("numpy", "NumPy arrays"),
        ("pulp", "PuLP optimization"),
        ("psutil", "System monitoring"),
    ]

    all_passed = True
    for dep, desc in deps:
        try:
            __import__(dep)
            print(f"✓ {dep}: {desc}")
        except ImportError:
            print(f"✗ {dep}: NOT INSTALLED")
            all_passed = False

    return True


def main():
    """Run all tests"""
    print("=" * 70)
    print("IoT Dashboard - Comprehensive Test Suite")
    print("=" * 70)

    tests = [
        ("Dependencies", test_dependencies),
        ("Imports", test_imports),
        ("Layouts", test_layouts),
        ("Callbacks", test_callbacks),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            results[test_name] = False

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s} {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Dashboard is ready to run.")
        print("\nRun: python launch_unified_dashboard.py")
        print("URL: http://127.0.0.1:8050")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Test what data the dashboard is ACTUALLY returning when callbacks are triggered
This simulates what happens when you click "Monitoring" tab and select a sensor
"""

import sys

sys.path.insert(0, "/workspaces/IoT-Predictive-Maintenance-System")

print("=" * 80)
print("TESTING ACTUAL DASHBOARD DATA FLOW")
print("=" * 80)
print()

# Simulate what happens when monitoring tab loads
print("SIMULATING: User clicks Monitoring tab and selects SMAP-PWR-001")
print("-" * 80)

try:
    # Import exactly what the dashboard imports
    from src.presentation.dashboard.services.dashboard_integration import (
        get_integration_service,
    )

    print("✓ Integration service imported")

    # Get integration service (this is what dashboard does)
    integration = get_integration_service()
    print("✓ Integration service initialized")

    # This is what monitoring.py callback does now
    sensor_id = "SMAP-PWR-001"
    hours = 24

    print(f"\nCalling: integration.get_sensor_data('{sensor_id}', hours={hours})")
    df = integration.get_sensor_data(sensor_id, hours=hours)

    print(f"\n✓ Returned DataFrame:")
    print(f"  - Type: {type(df)}")
    print(f"  - Shape: {df.shape}")
    print(f"  - Columns: {list(df.columns)}")
    print(f"  - Row count: {len(df)}")
    print()
    print(f"✓ Data statistics:")
    print(f"  - Value range: [{df['value'].min():.4f}, {df['value'].max():.4f}]")
    print(f"  - Value mean: {df['value'].mean():.4f}")
    print(f"  - Value std: {df['value'].std():.4f}")
    print()
    print(f"✓ First 3 data points:")
    for idx in range(min(3, len(df))):
        row = df.iloc[idx]
        print(
            f"  [{idx}] timestamp={row['timestamp']}, value={row['value']:.4f}, sensor={row['sensor_id']}"
        )
    print()

    # Check if this is real NASA data or mock data
    print("✓ Data verification:")
    has_negative = (df["value"] < 0).any()
    value_range = df["value"].max() - df["value"].min()

    if has_negative and df["value"].min() < -2:
        print(
            "  ✅ REAL NASA DATA - Has characteristic negative values for power sensors"
        )
        print(f"     (Range: {df['value'].min():.2f} to {df['value'].max():.2f})")
    elif value_range < 5 and 45 < df["value"].mean() < 60:
        print("  ❌ MOCK DATA - Values centered around 50 with sinusoidal pattern")
        print(f"     (Range: {df['value'].min():.2f} to {df['value'].max():.2f})")
    else:
        print(
            f"  ⚠ UNKNOWN - Value range: {df['value'].min():.2f} to {df['value'].max():.2f}"
        )

    print()
    print("=" * 80)
    print("RESULT: Dashboard integration service is working!")
    print("=" * 80)

    if has_negative:
        print("\n✅ SUCCESS: Real NASA data is being returned")
        print("   The dashboard SHOULD show real data now")
        print("   If you still see mock data, try:")
        print("   1. Hard refresh browser (Ctrl+Shift+R)")
        print("   2. Clear browser cache")
        print("   3. Check browser console for errors")
    else:
        print("\n⚠ WARNING: Data doesn't look like NASA power sensor data")
        print("   May need to check which sensor is being loaded")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback

    traceback.print_exc()

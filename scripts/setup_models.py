#!/usr/bin/env python3
"""
Complete Model Training Setup for GitHub Codespaces
Trains both anomaly detection and forecasting models for all 12 sensors
"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def print_banner():
    """Print setup banner"""
    print("=" * 80)
    print("🤖 IoT PREDICTIVE MAINTENANCE - MODEL TRAINING SETUP")
    print("   Complete Training Pipeline for GitHub Codespaces")
    print("=" * 80)
    print("🎯 Training both Anomaly Detection + Forecasting models")
    print("📊 12 NASA SMAP/MSL sensors")
    print("⚡ Optimized for cloud environments")
    print("-" * 80)


def check_environment():
    """Check if running in suitable environment"""
    print("🔍 Checking environment...")

    # Check available memory (rough estimate)
    try:
        import psutil

        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"✅ Available RAM: {memory_gb:.1f}GB")
        if memory_gb < 2:
            print("⚠️  Warning: Low memory detected. Consider using --quick mode")
    except ImportError:
        print("ℹ️  Memory check skipped (psutil not available)")

    # Check if models already exist
    models_dir = Path("data/models")
    if models_dir.exists():
        anomaly_models = list(models_dir.glob("**/telemanom_*.h5"))
        forecast_models = list(models_dir.glob("**/transformer_model.h5"))
        print(f"📁 Found {len(anomaly_models)} anomaly models")
        print(f"📁 Found {len(forecast_models)} forecasting models")

    print()


def train_anomaly_models(quick_mode=False):
    """Train anomaly detection models"""
    print("🔥 TRAINING ANOMALY DETECTION MODELS")
    print("-" * 40)

    start_time = time.time()

    try:
        # Import and run anomaly training
        print("🚀 Starting anomaly model training...")

        if quick_mode:
            result = os.system("python train_anomaly_models.py --quick")
        else:
            result = os.system("python train_anomaly_models.py")

        if result == 0:
            elapsed = time.time() - start_time
            print(f"✅ Anomaly models trained successfully in {elapsed:.1f}s")
            return True
        else:
            print("❌ Anomaly training failed")
            return False

    except Exception as e:
        print(f"❌ Error training anomaly models: {e}")
        return False


def train_forecasting_models(quick_mode=False):
    """Train forecasting models"""
    print("🔮 TRAINING FORECASTING MODELS")
    print("-" * 40)

    start_time = time.time()

    try:
        # Import and run forecasting training
        print("🚀 Starting forecasting model training...")

        if quick_mode:
            result = os.system("python train_forecasting_models.py --quick")
        else:
            result = os.system("python train_forecasting_models.py")

        if result == 0:
            elapsed = time.time() - start_time
            print(f"✅ Forecasting models trained successfully in {elapsed:.1f}s")
            return True
        else:
            print("❌ Forecasting training failed")
            return False

    except Exception as e:
        print(f"❌ Error training forecasting models: {e}")
        return False


def main():
    """Main setup function"""
    import argparse

    parser = argparse.ArgumentParser(description="Complete Model Training Setup")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick training mode (faster, smaller models)",
    )
    parser.add_argument(
        "--anomaly-only",
        action="store_true",
        help="Train only anomaly detection models",
    )
    parser.add_argument("--forecasting-only", action="store_true", help="Train only forecasting models")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip training if models already exist",
    )

    args = parser.parse_args()

    print_banner()
    check_environment()

    # Check if models already exist and should skip
    if args.skip_existing:
        models_dir = Path("data/models")
        if models_dir.exists():
            existing_forecast = list(models_dir.glob("**/transformer_model.h5"))
            existing_anomaly = list(models_dir.glob("**/telemanom_*.h5"))

            if len(existing_forecast) >= 12 and len(existing_anomaly) >= 12:
                print("✅ Models already exist, skipping training")
                print("🚀 Ready to launch dashboard!")
                return 0

    total_start = time.time()
    success_count = 0
    total_tasks = 0

    # Train anomaly models
    if not args.forecasting_only:
        total_tasks += 1
        print("⏳ Step 1/2: Anomaly Detection Training")
        if train_anomaly_models(args.quick):
            success_count += 1
        print()

    # Train forecasting models
    if not args.anomaly_only:
        total_tasks += 1
        step_num = 2 if not args.forecasting_only else 1
        total_steps = 2 if not args.forecasting_only else 1
        print(f"⏳ Step {step_num}/{total_steps}: Forecasting Training")
        if train_forecasting_models(args.quick):
            success_count += 1
        print()

    # Summary
    total_elapsed = time.time() - total_start
    print("=" * 80)
    print("📊 TRAINING COMPLETE!")
    print("=" * 80)
    print(f"✅ Successful: {success_count}/{total_tasks} training tasks")
    print(f"⏱️  Total time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")

    if success_count == total_tasks:
        print("🎉 ALL MODELS TRAINED SUCCESSFULLY!")
        print()
        print("🚀 NEXT STEPS:")
        print("   1. Launch dashboard: python start_dashboard.py")
        print("   2. Access at port 8050 (auto-forwarded in Codespaces)")
        print("   3. Explore all 7 tabs with your trained models!")
        return 0
    else:
        print("⚠️  Some training tasks failed. Check logs above.")
        print("💡 Try running with --quick flag for faster training")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

"""
Standalone NASA Telemanom Anomaly Detection Training Script

This script trains anomaly detection models for all 12 sensors independently,
using NASA Telemanom LSTM architecture. Models are saved to the registry
for dashboard inference consumption.

Usage:
    python train_anomaly_models.py
    python train_anomaly_models.py --sensors SMAP-PWR-001 MSL-MOB-001
    python train_anomaly_models.py --quick  # Fast training for testing
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("train_anomaly_models.log")],
)
logger = logging.getLogger(__name__)

# Imports after path setup
from config.equipment_config import EQUIPMENT_REGISTRY, get_equipment_by_id
from src.core.services.anomaly_service import AnomalyDetectionService
from src.infrastructure.data.nasa_data_loader import NASADataLoader
from src.infrastructure.ml.model_registry import ModelRegistry
from src.infrastructure.ml.telemanom_wrapper import (
    NASATelemanom,
    Telemanom_Config,
    _load_tensorflow,
)


class AnomalyModelTrainer:
    """Standalone trainer for NASA Telemanom anomaly detection models"""

    def __init__(self, data_root: str = "data/raw", models_dir: str = "data/models"):
        """
        Initialize anomaly model trainer

        Args:
            data_root: Path to NASA data files
            models_dir: Path to save trained models
        """
        self.data_root = Path(data_root)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.data_loader = NASADataLoader(str(self.data_root))
        self.model_registry = ModelRegistry(str(self.models_dir / "registry"))

        # Training session info
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.training_results = {}

        logger.info(f"Anomaly Model Trainer initialized (Session: {self.session_id})")
        logger.info(f"Data root: {self.data_root}")
        logger.info(f"Models directory: {self.models_dir}")

    def create_training_config(self, quick_mode: bool = False) -> Telemanom_Config:
        """Create training configuration"""
        if quick_mode:
            return Telemanom_Config(
                sequence_length=50,  # Reduced for speed
                lstm_units=[40, 40],  # Smaller model
                epochs=5,  # Fast training
                batch_size=32,
                validation_split=0.1,
            )
        else:
            return Telemanom_Config(
                sequence_length=250,  # NASA standard
                lstm_units=[80, 80],  # NASA standard
                epochs=35,  # NASA standard
                batch_size=70,
                validation_split=0.2,
            )

    def prepare_training_data(self, sensor_id: str) -> Optional[Dict[str, Any]]:
        """
        Prepare training data for specific sensor

        Args:
            sensor_id: Equipment ID from EQUIPMENT_REGISTRY

        Returns:
            Dictionary with training data and metadata
        """
        try:
            logger.info(f"Preparing training data for {sensor_id}")

            # Get equipment configuration
            equipment = get_equipment_by_id(sensor_id)
            if not equipment:
                logger.error(f"Equipment {sensor_id} not found in registry")
                return None

            # Get sensor data (extended history for training)
            sensor_data = self.data_loader.get_sensor_data(
                sensor_id, hours_back=8760
            )  # 1 year

            if not sensor_data["values"] or len(sensor_data["values"]) < 100:
                logger.warning(f"Insufficient data for {sensor_id}, using mock data")
                return None

            # Convert to numpy array
            import numpy as np

            values = np.array(sensor_data["values"]).reshape(-1, 1)

            # Split data: 80% train, 20% validation
            split_point = int(len(values) * 0.8)
            train_data = values[:split_point]
            validation_data = values[split_point:]

            logger.info(f"Data prepared for {sensor_id}:")
            logger.info(f"  Training samples: {len(train_data)}")
            logger.info(f"  Validation samples: {len(validation_data)}")
            logger.info(f"  Data quality: {sensor_data['data_quality']}")

            return {
                "sensor_id": sensor_id,
                "equipment": equipment,
                "train_data": train_data,
                "validation_data": validation_data,
                "sensor_info": sensor_data["sensor_info"],
                "data_quality": sensor_data["data_quality"],
            }

        except Exception as e:
            logger.error(f"Error preparing training data for {sensor_id}: {e}")
            return None

    def train_sensor_model(
        self, sensor_id: str, config: Telemanom_Config
    ) -> Optional[Dict[str, Any]]:
        """
        Train Telemanom model for specific sensor

        Args:
            sensor_id: Equipment ID
            config: Training configuration

        Returns:
            Training results dictionary
        """
        try:
            logger.info(f"Starting training for {sensor_id}")
            start_time = time.time()

            # Prepare training data
            data_prep = self.prepare_training_data(sensor_id)
            if not data_prep:
                return None

            # Initialize Telemanom model
            telemanom = NASATelemanom(sensor_id, config)

            # Train model
            training_metrics = telemanom.train(data_prep["train_data"])

            # Validate model
            validation_metrics = self.validate_model(
                telemanom, data_prep["validation_data"]
            )

            # Calculate training time
            training_time = time.time() - start_time

            # Save model to filesystem
            model_path = self.models_dir / "telemanom" / sensor_id
            telemanom.save_model(self.models_dir / "telemanom")

            # Register model in registry
            version_id = self.model_registry.register_model(
                sensor_id=sensor_id,
                model_type="telemanom",
                model_path=model_path,
                training_config=config.__dict__,
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                training_time_seconds=training_time,
                description=f"NASA Telemanom LSTM for {data_prep['equipment'].name}",
                tags=[
                    "nasa",
                    "telemanom",
                    "lstm",
                    "anomaly_detection",
                    data_prep["equipment"].equipment_type.value.lower(),
                ],
            )

            result = {
                "sensor_id": sensor_id,
                "version_id": version_id,
                "training_time_seconds": training_time,
                "training_metrics": training_metrics,
                "validation_metrics": validation_metrics,
                "model_path": str(model_path),
                "data_quality": data_prep["data_quality"],
                "equipment_info": {
                    "name": data_prep["equipment"].name,
                    "type": data_prep["equipment"].equipment_type.value,
                    "criticality": data_prep["equipment"].criticality.value,
                    "data_source": data_prep["equipment"].data_source,
                },
            }

            logger.info(f"Training completed for {sensor_id}:")
            logger.info(f"  Version ID: {version_id}")
            logger.info(f"  Training time: {training_time:.1f}s")
            logger.info(
                f"  Error threshold: {training_metrics.get('error_threshold', 'N/A')}"
            )
            logger.info(
                f"  Model parameters: {training_metrics.get('model_parameters', 'N/A')}"
            )

            return result

        except Exception as e:
            logger.error(f"Error training model for {sensor_id}: {e}")
            return None

    def validate_model(
        self, model: NASATelemanom, validation_data: Any
    ) -> Dict[str, Any]:
        """Validate trained model"""
        try:
            # Run anomaly detection on validation data
            results = model.detect_anomalies(validation_data)

            # Calculate validation metrics
            anomaly_rate = (
                results["anomaly_count"] / results["total_points"]
                if results["total_points"] > 0
                else 0
            )

            validation_metrics = {
                "validation_performed": True,
                "validation_samples": results["total_points"],
                "anomalies_detected": results["anomaly_count"],
                "anomaly_rate": anomaly_rate,
                "error_threshold": results["threshold"],
                "validation_quality": (
                    "good" if 0.01 <= anomaly_rate <= 0.1 else "needs_review"
                ),
            }

            return validation_metrics

        except Exception as e:
            logger.error(f"Error during validation: {e}")
            return {"validation_performed": False, "error": str(e)}

    def train_all_sensors(
        self, sensor_list: List[str], config: Telemanom_Config
    ) -> Dict[str, Any]:
        """
        Train models for all specified sensors

        Args:
            sensor_list: List of sensor IDs to train
            config: Training configuration

        Returns:
            Training summary
        """
        logger.info(f"Starting batch training for {len(sensor_list)} sensors")
        logger.info(f"Configuration: {config.__dict__}")

        # Load TensorFlow once for all training
        tf_available = _load_tensorflow()
        if not tf_available:
            logger.warning("TensorFlow not available, using mock implementation")

        successful_trains = 0
        failed_trains = 0
        total_start_time = time.time()

        for i, sensor_id in enumerate(sensor_list, 1):
            logger.info(f"Training sensor {i}/{len(sensor_list)}: {sensor_id}")

            result = self.train_sensor_model(sensor_id, config)
            if result:
                self.training_results[sensor_id] = result
                successful_trains += 1
            else:
                failed_trains += 1
                self.training_results[sensor_id] = {
                    "sensor_id": sensor_id,
                    "status": "failed",
                    "error": "Training failed",
                }

        total_time = time.time() - total_start_time

        # Create training summary
        summary = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "total_sensors": len(sensor_list),
            "successful_trains": successful_trains,
            "failed_trains": failed_trains,
            "total_time_seconds": total_time,
            "average_time_per_sensor": (
                total_time / len(sensor_list) if sensor_list else 0
            ),
            "tensorflow_available": tf_available,
            "configuration": config.__dict__,
            "results": self.training_results,
        }

        # Save training summary
        summary_file = (
            self.models_dir / f"anomaly_training_summary_{self.session_id}.json"
        )
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Batch training completed:")
        logger.info(f"  Successful: {successful_trains}/{len(sensor_list)}")
        logger.info(f"  Failed: {failed_trains}/{len(sensor_list)}")
        logger.info(f"  Total time: {total_time:.1f}s")
        logger.info(f"  Summary saved: {summary_file}")

        return summary


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(
        description="Train NASA Telemanom Anomaly Detection Models"
    )
    parser.add_argument(
        "--sensors", nargs="+", help="Specific sensors to train (default: all)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick training mode (faster, less accurate)",
    )
    parser.add_argument("--data-root", default="data/raw", help="Data root directory")
    parser.add_argument(
        "--models-dir", default="data/models", help="Models output directory"
    )

    args = parser.parse_args()

    # Print banner
    print("=" * 70)
    print("NASA TELEMANOM ANOMALY DETECTION MODEL TRAINER")
    print("Session 1: Standalone Training Script")
    print("=" * 70)
    print(f"Started at: {datetime.now()}")
    print(f"Quick mode: {'ON' if args.quick else 'OFF'}")
    print()

    try:
        # Initialize trainer
        trainer = AnomalyModelTrainer(args.data_root, args.models_dir)

        # Determine sensors to train
        if args.sensors:
            sensor_list = args.sensors
            logger.info(f"Training specific sensors: {sensor_list}")
        else:
            sensor_list = list(EQUIPMENT_REGISTRY.keys())
            logger.info(f"Training all sensors: {len(sensor_list)} total")

        # Validate sensors
        valid_sensors = []
        for sensor_id in sensor_list:
            if sensor_id in EQUIPMENT_REGISTRY:
                valid_sensors.append(sensor_id)
            else:
                logger.warning(f"Unknown sensor: {sensor_id}")

        if not valid_sensors:
            logger.error("No valid sensors found")
            return 1

        # Create training configuration
        config = trainer.create_training_config(args.quick)

        # Train models
        summary = trainer.train_all_sensors(valid_sensors, config)

        # Print results
        print("\n" + "=" * 70)
        print("TRAINING RESULTS")
        print("=" * 70)
        print(f"Session ID: {summary['session_id']}")
        print(f"Total sensors: {summary['total_sensors']}")
        print(f"Successful: {summary['successful_trains']}")
        print(f"Failed: {summary['failed_trains']}")
        print(f"Total time: {summary['total_time_seconds']:.1f}s")
        print(f"Average per sensor: {summary['average_time_per_sensor']:.1f}s")
        print()

        # Print individual results
        for sensor_id, result in summary["results"].items():
            if "version_id" in result:
                print(
                    f"✓ {sensor_id}: Version {result['version_id']} "
                    f"({result['training_time_seconds']:.1f}s)"
                )
            else:
                print(f"✗ {sensor_id}: {result.get('error', 'Failed')}")

        print("\n" + "=" * 70)
        print("MODELS TRAINED AND REGISTERED FOR DASHBOARD INFERENCE")
        print("Run dashboard to see anomaly detection in action!")
        print("=" * 70)

        return 0 if summary["failed_trains"] == 0 else 1

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\nTraining failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

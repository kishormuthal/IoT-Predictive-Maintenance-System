"""
Standalone Transformer Forecasting Model Training Script

This script trains Transformer-based forecasting models for all 12 sensors
independently. Models are saved to the registry for dashboard inference consumption.

Usage:
    python train_forecasting_models.py
    python train_forecasting_models.py --sensors SMAP-PWR-001 MSL-MOB-001
    python train_forecasting_models.py --quick  # Fast training for testing
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

import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train_forecasting_models.log"),
    ],
)
logger = logging.getLogger(__name__)

# Imports after path setup
from config.equipment_config import EQUIPMENT_REGISTRY, get_equipment_by_id
from src.infrastructure.data.nasa_data_loader import NASADataLoader
from src.infrastructure.ml.model_registry import ModelRegistry
from src.infrastructure.ml.transformer_wrapper import (
    TENSORFLOW_AVAILABLE,
    TransformerConfig,
    TransformerForecaster,
)


class ForecastingModelTrainer:
    """Standalone trainer for Transformer forecasting models"""

    def __init__(self, data_root: str = "data/raw", models_dir: str = "data/models"):
        """
        Initialize forecasting model trainer

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

        logger.info(f"Forecasting Model Trainer initialized (Session: {self.session_id})")
        logger.info(f"Data root: {self.data_root}")
        logger.info(f"Models directory: {self.models_dir}")
        logger.info(f"TensorFlow available: {TENSORFLOW_AVAILABLE}")

    def create_training_config(self, quick_mode: bool = False) -> TransformerConfig:
        """Create training configuration"""
        if quick_mode:
            return TransformerConfig(
                sequence_length=48,  # 2 days of hourly data
                forecast_horizon=12,  # 12 hours forecast
                d_model=64,  # Smaller model
                num_heads=4,
                num_layers=2,
                dff=256,
                epochs=10,  # Fast training
                batch_size=16,
                validation_split=0.1,
                patience=5,
            )
        else:
            return TransformerConfig(
                sequence_length=168,  # 1 week of hourly data
                forecast_horizon=24,  # 24 hours forecast
                d_model=128,  # Standard model
                num_heads=8,
                num_layers=4,
                dff=512,
                epochs=100,  # Full training
                batch_size=32,
                validation_split=0.2,
                patience=15,
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
            logger.info(f"Preparing forecasting training data for {sensor_id}")

            # Get equipment configuration
            equipment = get_equipment_by_id(sensor_id)
            if not equipment:
                logger.error(f"Equipment {sensor_id} not found in registry")
                return None

            # Get sensor data (extended history for training)
            sensor_data = self.data_loader.get_sensor_data(sensor_id, hours_back=8760)  # 1 year

            if not sensor_data["values"] or len(sensor_data["values"]) < 300:
                logger.warning(f"Insufficient data for forecasting training: {sensor_id}")
                return None

            # Convert to numpy array
            values = np.array(sensor_data["values"]).reshape(-1, 1)

            # Split data: 80% train, 20% validation
            split_point = int(len(values) * 0.8)
            train_data = values[:split_point]
            validation_data = values[split_point:]

            logger.info(f"Forecasting data prepared for {sensor_id}:")
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
            logger.error(f"Error preparing forecasting training data for {sensor_id}: {e}")
            return None

    def validate_forecasting_model(
        self,
        model: TransformerForecaster,
        validation_data: np.ndarray,
        config: TransformerConfig,
    ) -> Dict[str, Any]:
        """Validate trained forecasting model"""
        try:
            # Use validation data for forecasting evaluation
            sequence_length = config.sequence_length
            forecast_horizon = config.forecast_horizon

            if len(validation_data) < sequence_length + forecast_horizon:
                logger.warning("Insufficient validation data for proper forecasting validation")
                return {"validation_performed": False, "insufficient_data": True}

            # Create test sequences from validation data
            test_sequences = []
            actual_targets = []

            for i in range(
                0,
                len(validation_data) - sequence_length - forecast_horizon,
                forecast_horizon,
            ):
                input_seq = validation_data[i : i + sequence_length]
                actual_forecast = validation_data[i + sequence_length : i + sequence_length + forecast_horizon]

                test_sequences.append(input_seq.flatten())
                actual_targets.append(actual_forecast.flatten())

            if not test_sequences:
                return {"validation_performed": False, "insufficient_sequences": True}

            # Generate forecasts for test sequences
            predictions = []
            for seq in test_sequences[:5]:  # Test on first 5 sequences
                seq_reshaped = seq.reshape(-1, 1)
                forecast_result = model.predict(seq_reshaped, forecast_horizon)
                predictions.append(forecast_result["forecast_values"][:forecast_horizon])

            # Calculate accuracy metrics
            if predictions and actual_targets:
                predictions_array = np.array(predictions)
                actuals_array = np.array(actual_targets[: len(predictions)])

                # Calculate R² score
                predictions_flat = predictions_array.flatten()
                actuals_flat = actuals_array.flatten()

                # Ensure same length
                min_length = min(len(predictions_flat), len(actuals_flat))
                predictions_flat = predictions_flat[:min_length]
                actuals_flat = actuals_flat[:min_length]

                # Calculate metrics
                mae = np.mean(np.abs(actuals_flat - predictions_flat))
                mse = np.mean((actuals_flat - predictions_flat) ** 2)
                rmse = np.sqrt(mse)

                # MAPE (handling division by zero)
                non_zero_actuals = actuals_flat != 0
                if np.any(non_zero_actuals):
                    mape = (
                        np.mean(
                            np.abs(
                                (actuals_flat[non_zero_actuals] - predictions_flat[non_zero_actuals])
                                / actuals_flat[non_zero_actuals]
                            )
                        )
                        * 100
                    )
                else:
                    mape = 100.0

                # R² score
                ss_res = np.sum((actuals_flat - predictions_flat) ** 2)
                ss_tot = np.sum((actuals_flat - np.mean(actuals_flat)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                validation_metrics = {
                    "validation_performed": True,
                    "validation_sequences": len(predictions),
                    "mae": float(mae),
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "mape": float(mape),
                    "r2_score": float(r2),
                    "forecast_horizon_tested": forecast_horizon,
                    "validation_quality": ("excellent" if r2 > 0.8 else "good" if r2 > 0.5 else "needs_improvement"),
                }

                return validation_metrics

            return {
                "validation_performed": True,
                "no_valid_predictions": True,
                "mae": 0.0,
                "mse": 0.0,
                "rmse": 0.0,
                "mape": 100.0,
                "r2_score": 0.0,
            }

        except Exception as e:
            logger.error(f"Error during forecasting validation: {e}")
            return {"validation_performed": False, "error": str(e)}

    def train_sensor_model(self, sensor_id: str, config: TransformerConfig) -> Optional[Dict[str, Any]]:
        """
        Train Transformer forecasting model for specific sensor

        Args:
            sensor_id: Equipment ID
            config: Training configuration

        Returns:
            Training results dictionary
        """
        try:
            logger.info(f"Starting forecasting training for {sensor_id}")
            start_time = time.time()

            # Prepare training data
            data_prep = self.prepare_training_data(sensor_id)
            if not data_prep:
                return None

            # Initialize Transformer model
            transformer = TransformerForecaster(sensor_id, config)

            # Train model
            training_metrics = transformer.train(data_prep["train_data"])

            # Validate model
            validation_metrics = self.validate_forecasting_model(transformer, data_prep["validation_data"], config)

            # Calculate training time
            training_time = time.time() - start_time

            # Save model to filesystem
            model_path = self.models_dir / "transformer" / sensor_id
            transformer.save_model(self.models_dir / "transformer")

            # Register model in registry
            version_id = self.model_registry.register_model(
                sensor_id=sensor_id,
                model_type="transformer",
                model_path=model_path,
                training_config=config.__dict__,
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                training_time_seconds=training_time,
                description=f"Transformer forecaster for {data_prep['equipment'].name}",
                tags=[
                    "transformer",
                    "forecasting",
                    "time_series",
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

            logger.info(f"Forecasting training completed for {sensor_id}:")
            logger.info(f"  Version ID: {version_id}")
            logger.info(f"  Training time: {training_time:.1f}s")
            logger.info(f"  R² Score: {validation_metrics.get('r2_score', 'N/A')}")
            logger.info(f"  Model parameters: {training_metrics.get('model_parameters', 'N/A')}")

            return result

        except Exception as e:
            logger.error(f"Error training forecasting model for {sensor_id}: {e}")
            return None

    def train_all_sensors(self, sensor_list: List[str], config: TransformerConfig) -> Dict[str, Any]:
        """
        Train forecasting models for all specified sensors

        Args:
            sensor_list: List of sensor IDs to train
            config: Training configuration

        Returns:
            Training summary
        """
        logger.info(f"Starting batch forecasting training for {len(sensor_list)} sensors")
        logger.info(f"Configuration: {config.__dict__}")

        successful_trains = 0
        failed_trains = 0
        total_start_time = time.time()

        for i, sensor_id in enumerate(sensor_list, 1):
            logger.info(f"Training forecasting model {i}/{len(sensor_list)}: {sensor_id}")

            result = self.train_sensor_model(sensor_id, config)
            if result:
                self.training_results[sensor_id] = result
                successful_trains += 1
            else:
                failed_trains += 1
                self.training_results[sensor_id] = {
                    "sensor_id": sensor_id,
                    "status": "failed",
                    "error": "Forecasting training failed",
                }

        total_time = time.time() - total_start_time

        # Create training summary
        summary = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "model_type": "transformer_forecasting",
            "total_sensors": len(sensor_list),
            "successful_trains": successful_trains,
            "failed_trains": failed_trains,
            "total_time_seconds": total_time,
            "average_time_per_sensor": (total_time / len(sensor_list) if sensor_list else 0),
            "tensorflow_available": TENSORFLOW_AVAILABLE,
            "configuration": config.__dict__,
            "results": self.training_results,
        }

        # Save training summary
        summary_file = self.models_dir / f"forecasting_training_summary_{self.session_id}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Batch forecasting training completed:")
        logger.info(f"  Successful: {successful_trains}/{len(sensor_list)}")
        logger.info(f"  Failed: {failed_trains}/{len(sensor_list)}")
        logger.info(f"  Total time: {total_time:.1f}s")
        logger.info(f"  Summary saved: {summary_file}")

        return summary


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description="Train Transformer Forecasting Models")
    parser.add_argument("--sensors", nargs="+", help="Specific sensors to train (default: all)")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick training mode (faster, less accurate)",
    )
    parser.add_argument("--data-root", default="data/raw", help="Data root directory")
    parser.add_argument("--models-dir", default="data/models", help="Models output directory")

    args = parser.parse_args()

    # Print banner
    print("=" * 70)
    print("TRANSFORMER FORECASTING MODEL TRAINER")
    print("Session 1: Standalone Training Script")
    print("=" * 70)
    print(f"Started at: {datetime.now()}")
    print(f"Quick mode: {'ON' if args.quick else 'OFF'}")
    print(f"TensorFlow available: {TENSORFLOW_AVAILABLE}")
    print()

    try:
        # Initialize trainer
        trainer = ForecastingModelTrainer(args.data_root, args.models_dir)

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
        print("FORECASTING TRAINING RESULTS")
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
                validation_quality = result.get("validation_metrics", {}).get("validation_quality", "unknown")
                r2_score = result.get("validation_metrics", {}).get("r2_score", 0.0)
                print(
                    f"✓ {sensor_id}: Version {result['version_id']} "
                    f"(R²={r2_score:.3f}, {validation_quality}, {result['training_time_seconds']:.1f}s)"
                )
            else:
                print(f"✗ {sensor_id}: {result.get('error', 'Failed')}")

        print("\n" + "=" * 70)
        print("FORECASTING MODELS TRAINED AND REGISTERED FOR DASHBOARD")
        print("Run dashboard to see time series forecasting in action!")
        print("=" * 70)

        return 0 if summary["failed_trains"] == 0 else 1

    except KeyboardInterrupt:
        print("\nForecasting training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Forecasting training failed: {e}")
        print(f"\nForecasting training failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

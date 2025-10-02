"""
Training Use Cases - FIXED VERSION
Application layer for training operations with all 12 critical fixes applied
"""

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.equipment_config import get_equipment_by_id, get_equipment_list
from src.infrastructure.ml.model_registry_sqlite import ModelRegistrySQLite
from src.infrastructure.ml.telemanom_wrapper import (
    InsufficientDataError,
    ModelNotTrainedError,
    NASATelemanom,
    Telemanom_Config,
)
from src.infrastructure.ml.transformer_wrapper import (
    TransformerConfig,
    TransformerForecaster,
)

logger = logging.getLogger(__name__)


class TrainingUseCase:
    """
    Use case for training operations with comprehensive fixes
    Coordinates training pipelines and model registry
    """

    def __init__(
        self,
        config_path: str = None,
        registry_path: str = None,
        model_base_path: str = None,
        data_loader=None,
    ):
        """
        Initialize training use case

        Args:
            config_path: Path to training configuration
            registry_path: Path to model registry (externalized, configurable)
            model_base_path: Base path for saving models
            data_loader: Data loader instance for loading training data
        """
        self.config_path = config_path or "training/config/training_config.yaml"

        # FIX 1: Externalized registry_path configuration
        self.registry_path = registry_path or "./models/registry_sqlite"
        self.model_registry = ModelRegistrySQLite(self.registry_path)

        self.model_base_path = Path(model_base_path or "./data/models")
        self.model_base_path.mkdir(parents=True, exist_ok=True)

        # Data loader for loading training data
        self.data_loader = data_loader

        # FIX 2: Per-sensor pipeline instances (no shared state)
        # Pipelines are created per-sensor, not shared
        # This ensures thread safety and prevents concurrent training issues

        logger.info(
            f"Training use case initialized with registry: {self.registry_path}, "
            f"model path: {self.model_base_path}"
        )

    def _load_training_data(
        self,
        sensor_id: str,
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    ) -> Dict[str, Any]:
        """
        Load and split training data for a sensor

        FIX 4: Data loading workflow - TrainingUseCase loads data, passes to wrappers

        Args:
            sensor_id: Sensor identifier
            split_ratio: (train, validation, test) split ratio

        Returns:
            Dict with train_data, val_data, test_data, data_hash, metadata
        """
        try:
            if self.data_loader is None:
                raise ValueError("Data loader not configured")

            # Load data using configured data loader
            data = self.data_loader.load_sensor_data(sensor_id)

            if data is None or len(data) == 0:
                raise InsufficientDataError(f"No data available for sensor {sensor_id}")

            # Calculate data hash for lineage tracking
            # FIX 12: Pass data_hash from actual training data
            data_hash = hashlib.sha256(data.tobytes()).hexdigest()[:16]

            # Split data
            train_size = int(len(data) * split_ratio[0])
            val_size = int(len(data) * split_ratio[1])

            train_data = data[:train_size]
            val_data = data[train_size : train_size + val_size]
            test_data = data[train_size + val_size :]

            return {
                "train_data": train_data,
                "val_data": val_data,
                "test_data": test_data,
                "data_hash": data_hash,
                "metadata": {
                    "total_samples": len(data),
                    "train_samples": len(train_data),
                    "val_samples": len(val_data),
                    "test_samples": len(test_data),
                    "data_start": self.data_loader.get_data_start_date(sensor_id),
                    "data_end": self.data_loader.get_data_end_date(sensor_id),
                },
            }

        except FileNotFoundError as e:
            logger.error(f"Data file not found for sensor {sensor_id}: {e}")
            raise
        except InsufficientDataError:
            raise
        except Exception as e:
            logger.error(f"Error loading training data for {sensor_id}: {e}")
            raise

    def _validate_model(
        self, model: Any, val_data: np.ndarray, model_type: str
    ) -> Dict[str, Any]:
        """
        Validate model on held-out validation set

        FIX 6: Implement proper validation step with held-out dataset
        FIX 10: Implement validate_model method

        Args:
            model: Trained model instance
            val_data: Validation dataset
            model_type: Type of model ('telemanom' or 'transformer')

        Returns:
            Validation metrics with validation_performed flag
        """
        try:
            validation_metrics = {"validation_performed": True}

            if model_type == "telemanom":
                # Create sequences for validation
                if hasattr(model, "_create_sequences"):
                    X_val, y_val = model._create_sequences(
                        model.scaler.transform(val_data.reshape(-1, 1))
                    )

                    if len(X_val) > 0:
                        # Get predictions
                        y_pred = model.model.predict(X_val, verbose=0)

                        # Calculate validation errors
                        val_errors = model._calculate_prediction_errors(y_val, y_pred)

                        # Calculate metrics
                        validation_metrics.update(
                            {
                                "mean_error": float(np.mean(val_errors)),
                                "std_error": float(np.std(val_errors)),
                                "max_error": float(np.max(val_errors)),
                                "anomaly_rate": float(
                                    np.sum(val_errors > model.error_threshold)
                                    / len(val_errors)
                                ),
                            }
                        )

            elif model_type == "transformer":
                # Forecast validation
                if len(val_data) >= model.config.sequence_length:
                    # Use part of validation data for input
                    input_data = val_data[: model.config.sequence_length]
                    actual_future = val_data[
                        model.config.sequence_length : model.config.sequence_length
                        + model.config.forecast_horizon
                    ]

                    # Generate forecast
                    forecast_result = model.predict(input_data)
                    forecast_values = np.array(forecast_result["forecast_values"])

                    # Calculate metrics if we have actual future values
                    if len(actual_future) > 0:
                        min_len = min(len(forecast_values), len(actual_future))
                        forecast_values = forecast_values[:min_len]
                        actual_future = actual_future[:min_len]

                        # Calculate validation metrics
                        mae = np.mean(np.abs(actual_future - forecast_values))
                        mse = np.mean((actual_future - forecast_values) ** 2)
                        rmse = np.sqrt(mse)

                        # R² score
                        ss_res = np.sum((actual_future - forecast_values) ** 2)
                        ss_tot = np.sum((actual_future - np.mean(actual_future)) ** 2)
                        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                        # MAPE
                        epsilon = 1e-10
                        mape = (
                            np.mean(
                                np.abs(
                                    (actual_future - forecast_values)
                                    / (actual_future + epsilon)
                                )
                            )
                            * 100
                        )

                        validation_metrics.update(
                            {
                                "mae": float(mae),
                                "mse": float(mse),
                                "rmse": float(rmse),
                                "r2_score": float(r2),
                                "mape": float(mape),
                            }
                        )

            return validation_metrics

        except Exception as e:
            logger.error(f"Error during validation: {e}")
            # Return minimal valid metrics on error
            return {"validation_performed": False, "error": str(e)}

    def train_sensor_anomaly_detection(
        self, sensor_id: str, config: Telemanom_Config = None
    ) -> Dict[str, Any]:
        """
        Train anomaly detection model for a sensor

        FIX 3: Align training interfaces - consistent method signatures

        Args:
            sensor_id: Equipment sensor ID
            config: Optional custom configuration

        Returns:
            Training results with registry information
        """
        start_time = datetime.now()

        try:
            logger.info(f"Training anomaly detection for sensor {sensor_id}")

            # Validate sensor
            equipment = get_equipment_by_id(sensor_id)
            if not equipment:
                return {"success": False, "error": f"Invalid sensor ID: {sensor_id}"}

            # Load training data
            # FIX 4: TrainingUseCase loads data, passes to wrappers
            data_split = self._load_training_data(sensor_id)

            # FIX 2: Create per-sensor pipeline instance (not shared)
            model_config = config or Telemanom_Config()
            model = NASATelemanom(sensor_id, model_config)

            # Train model
            training_result = model.train(data_split["train_data"])

            # FIX 6: Perform proper validation on held-out data
            validation_metrics = self._validate_model(
                model, data_split["val_data"], "telemanom"
            )

            # FIX 5: Correct model_path retrieval from save operation
            model_path = self.model_base_path / "telemanom" / sensor_id
            model_path.mkdir(parents=True, exist_ok=True)
            model.save_model(model_path.parent)

            training_time = (datetime.now() - start_time).total_seconds()

            # FIX 12: Pass data_hash from training data
            version_id = self.model_registry.register_model(
                sensor_id=sensor_id,
                model_type="telemanom",
                model_path=model_path,
                training_config=model_config.__dict__,
                training_metrics=training_result,
                validation_metrics=validation_metrics,
                training_time_seconds=training_time,
                data_hash=data_split["data_hash"],
                data_source=(
                    self.data_loader.get_data_source() if self.data_loader else None
                ),
                data_start_date=data_split["metadata"].get("data_start"),
                data_end_date=data_split["metadata"].get("data_end"),
                num_samples=data_split["metadata"]["train_samples"],
                description=f"Telemanom anomaly detection model for {equipment.equipment_type.value}",
                tags=[
                    "anomaly_detection",
                    equipment.equipment_type.value,
                    equipment.criticality.value,
                ],
            )

            result = {
                "success": True,
                "sensor_id": sensor_id,
                "training_result": training_result,
                "validation_metrics": validation_metrics,
                "training_time_seconds": training_time,
                "data_metadata": data_split["metadata"],
                "registry": {
                    "version_id": version_id,
                    "model_registered": True,
                    "model_path": str(model_path),
                },
            }

            logger.info(
                f"Anomaly detection training completed for {sensor_id}, version: {version_id}"
            )
            return result

        except FileNotFoundError as e:
            # FIX 7: Specific exception handling
            logger.error(f"Data file not found for sensor {sensor_id}: {e}")
            return {
                "success": False,
                "error": f"Data file not found: {str(e)}",
                "sensor_id": sensor_id,
            }
        except InsufficientDataError as e:
            logger.error(f"Insufficient data for sensor {sensor_id}: {e}")
            return {
                "success": False,
                "error": f"Insufficient data: {str(e)}",
                "sensor_id": sensor_id,
            }
        except ValueError as e:
            logger.error(f"Validation error for sensor {sensor_id}: {e}")
            return {
                "success": False,
                "error": f"Validation error: {str(e)}",
                "sensor_id": sensor_id,
            }
        except Exception as e:
            logger.error(
                f"Unexpected error training anomaly detection for {sensor_id}: {e}"
            )
            return {"success": False, "error": str(e), "sensor_id": sensor_id}

    def train_sensor_forecasting(
        self, sensor_id: str, config: TransformerConfig = None
    ) -> Dict[str, Any]:
        """
        Train forecasting model for a sensor

        FIX 3: Align training interfaces - consistent method signatures

        Args:
            sensor_id: Equipment sensor ID
            config: Optional custom configuration

        Returns:
            Training results with registry information
        """
        start_time = datetime.now()

        try:
            logger.info(f"Training forecasting for sensor {sensor_id}")

            # Validate sensor
            equipment = get_equipment_by_id(sensor_id)
            if not equipment:
                return {"success": False, "error": f"Invalid sensor ID: {sensor_id}"}

            # Load training data
            data_split = self._load_training_data(sensor_id)

            # FIX 2: Create per-sensor pipeline instance
            model_config = config or TransformerConfig()
            model = TransformerForecaster(sensor_id, model_config)

            # Train model
            training_result = model.train(data_split["train_data"])

            # FIX 6: Perform proper validation
            validation_metrics = self._validate_model(
                model, data_split["val_data"], "transformer"
            )

            # FIX 5: Correct model_path retrieval
            model_path = self.model_base_path / "transformer" / sensor_id
            model_path.mkdir(parents=True, exist_ok=True)
            model.save_model(model_path.parent)

            training_time = (datetime.now() - start_time).total_seconds()

            # FIX 12: Pass data_hash from training data
            version_id = self.model_registry.register_model(
                sensor_id=sensor_id,
                model_type="transformer",
                model_path=model_path,
                training_config=model_config.__dict__,
                training_metrics=training_result,
                validation_metrics=validation_metrics,
                training_time_seconds=training_time,
                data_hash=data_split["data_hash"],
                data_source=(
                    self.data_loader.get_data_source() if self.data_loader else None
                ),
                data_start_date=data_split["metadata"].get("data_start"),
                data_end_date=data_split["metadata"].get("data_end"),
                num_samples=data_split["metadata"]["train_samples"],
                description=f"Transformer forecasting model for {equipment.equipment_type.value}",
                tags=[
                    "forecasting",
                    equipment.equipment_type.value,
                    equipment.criticality.value,
                ],
            )

            result = {
                "success": True,
                "sensor_id": sensor_id,
                "training_result": training_result,
                "validation_metrics": validation_metrics,
                "training_time_seconds": training_time,
                "data_metadata": data_split["metadata"],
                "registry": {
                    "version_id": version_id,
                    "model_registered": True,
                    "model_path": str(model_path),
                },
            }

            logger.info(
                f"Forecasting training completed for {sensor_id}, version: {version_id}"
            )
            return result

        except FileNotFoundError as e:
            logger.error(f"Data file not found for sensor {sensor_id}: {e}")
            return {
                "success": False,
                "error": f"Data file not found: {str(e)}",
                "sensor_id": sensor_id,
            }
        except InsufficientDataError as e:
            logger.error(f"Insufficient data for sensor {sensor_id}: {e}")
            return {
                "success": False,
                "error": f"Insufficient data: {str(e)}",
                "sensor_id": sensor_id,
            }
        except ValueError as e:
            logger.error(f"Validation error for sensor {sensor_id}: {e}")
            return {
                "success": False,
                "error": f"Validation error: {str(e)}",
                "sensor_id": sensor_id,
            }
        except Exception as e:
            logger.error(f"Unexpected error training forecasting for {sensor_id}: {e}")
            return {"success": False, "error": str(e), "sensor_id": sensor_id}

    def train_all_sensors(self, model_types: List[str] = None) -> Dict[str, Any]:
        """
        Train models for all sensors

        FIX 8: Refactor train_all_sensors - iterate in TrainingUseCase, not wrappers

        Args:
            model_types: List of model types to train ('telemanom', 'transformer')

        Returns:
            Batch training results
        """
        try:
            if model_types is None:
                model_types = ["telemanom", "transformer"]

            logger.info(f"Training all sensors for model types: {model_types}")

            results = {
                "batch_training": True,
                "model_types": model_types,
                "start_time": datetime.now().isoformat(),
                "results": {},
                "summary": {},
            }

            equipment_list = get_equipment_list()

            # FIX 8: Iterate in TrainingUseCase, not in wrappers
            # This allows proper per-sensor pipeline creation and error handling

            # Train Telemanom models
            if "telemanom" in model_types:
                logger.info("Training Telemanom models for all sensors")

                telemanom_results = {
                    "individual_results": {},
                    "sensors_successful": 0,
                    "sensors_failed": 0,
                    "models_registered": 0,
                }

                for equipment in equipment_list:
                    sensor_id = equipment.equipment_id
                    result = self.train_sensor_anomaly_detection(sensor_id)

                    telemanom_results["individual_results"][sensor_id] = result

                    if result.get("success", False):
                        telemanom_results["sensors_successful"] += 1
                        if result.get("registry", {}).get("model_registered", False):
                            telemanom_results["models_registered"] += 1
                    else:
                        telemanom_results["sensors_failed"] += 1

                results["results"]["telemanom"] = telemanom_results

            # Train Transformer models
            if "transformer" in model_types:
                logger.info("Training Transformer models for all sensors")

                transformer_results = {
                    "individual_results": {},
                    "sensors_successful": 0,
                    "sensors_failed": 0,
                    "models_registered": 0,
                }

                for equipment in equipment_list:
                    sensor_id = equipment.equipment_id
                    result = self.train_sensor_forecasting(sensor_id)

                    transformer_results["individual_results"][sensor_id] = result

                    if result.get("success", False):
                        transformer_results["sensors_successful"] += 1
                        if result.get("registry", {}).get("model_registered", False):
                            transformer_results["models_registered"] += 1
                    else:
                        transformer_results["sensors_failed"] += 1

                results["results"]["transformer"] = transformer_results

            # Calculate summary
            total_successful = 0
            total_failed = 0
            total_registered = 0

            for model_type, model_results in results["results"].items():
                total_successful += model_results.get("sensors_successful", 0)
                total_failed += model_results.get("sensors_failed", 0)
                total_registered += model_results.get("models_registered", 0)

            results["summary"] = {
                "total_successful": total_successful,
                "total_failed": total_failed,
                "total_registered": total_registered,
                "success_rate": (
                    total_successful / (total_successful + total_failed)
                    if (total_successful + total_failed) > 0
                    else 0
                ),
            }

            results["end_time"] = datetime.now().isoformat()

            logger.info(
                f"Batch training completed: {total_successful} successful, {total_registered} registered"
            )
            return results

        except Exception as e:
            logger.error(f"Error in batch training: {e}")
            return {"batch_training": True, "success": False, "error": str(e)}

    def get_training_status(self) -> Dict[str, Any]:
        """
        Get current training status for all equipment

        FIX 9: Defensive metadata access - handle None values

        Returns:
            Training status summary
        """
        try:
            equipment_list = get_equipment_list()
            registry_stats = self.model_registry.get_registry_stats()

            status = {
                "total_equipment": len(equipment_list),
                "registry_stats": registry_stats,
                "equipment_status": {},
                "generated_at": datetime.now().isoformat(),
            }

            # Check status for each equipment
            for equipment in equipment_list:
                sensor_id = equipment.equipment_id

                # FIX 9: Defensive metadata access with None handling
                telemanom_version = self.model_registry.get_active_model_version(
                    sensor_id, "telemanom"
                )
                telemanom_metadata = None
                if telemanom_version:
                    telemanom_metadata = self.model_registry.get_model_metadata(
                        telemanom_version
                    )

                transformer_version = self.model_registry.get_active_model_version(
                    sensor_id, "transformer"
                )
                transformer_metadata = None
                if transformer_version:
                    transformer_metadata = self.model_registry.get_model_metadata(
                        transformer_version
                    )

                status["equipment_status"][sensor_id] = {
                    "equipment_type": equipment.equipment_type.value,
                    "criticality": equipment.criticality.value,
                    "anomaly_detection": {
                        "trained": telemanom_version is not None,
                        "version": telemanom_version,
                        "performance_score": (
                            telemanom_metadata.performance_score
                            if telemanom_metadata
                            else 0.0
                        ),
                        "last_trained": (
                            telemanom_metadata.created_at
                            if telemanom_metadata
                            else None
                        ),
                    },
                    "forecasting": {
                        "trained": transformer_version is not None,
                        "version": transformer_version,
                        "performance_score": (
                            transformer_metadata.performance_score
                            if transformer_metadata
                            else 0.0
                        ),
                        "last_trained": (
                            transformer_metadata.created_at
                            if transformer_metadata
                            else None
                        ),
                    },
                }

            return status

        except Exception as e:
            logger.error(f"Error getting training status: {e}")
            return {"error": str(e), "generated_at": datetime.now().isoformat()}

    def validate_models_batch(
        self, sensor_ids: List[str] = None, model_type: str = None
    ) -> Dict[str, Any]:
        """
        Batch validate trained models

        FIX 11: Implement batch validation

        Args:
            sensor_ids: Specific sensors to validate (all if None)
            model_type: Specific model type to validate (all if None)

        Returns:
            Batch validation results
        """
        try:
            logger.info(f"Batch validating models - type: {model_type}")

            equipment_list = get_equipment_list()

            if sensor_ids:
                # Filter to specific sensors
                equipment_list = [
                    eq for eq in equipment_list if eq.equipment_id in sensor_ids
                ]

            validation_results = {
                "batch_validation": True,
                "sensors_validated": 0,
                "sensors_failed": 0,
                "results": {},
                "validated_at": datetime.now().isoformat(),
            }

            for equipment in equipment_list:
                sensor_id = equipment.equipment_id
                sensor_results = {}

                try:
                    # Load validation data
                    data_split = self._load_training_data(sensor_id)

                    # Validate Telemanom if requested
                    if model_type is None or model_type == "telemanom":
                        version_id = self.model_registry.get_active_model_version(
                            sensor_id, "telemanom"
                        )
                        if version_id:
                            # Create model and load
                            model = NASATelemanom(sensor_id)
                            metadata = self.model_registry.get_model_metadata(
                                version_id
                            )

                            if metadata and Path(metadata.model_path).exists():
                                model.load_model(Path(metadata.model_path).parent)

                                val_metrics = self._validate_model(
                                    model, data_split["val_data"], "telemanom"
                                )
                                sensor_results["telemanom"] = val_metrics

                    # Validate Transformer if requested
                    if model_type is None or model_type == "transformer":
                        version_id = self.model_registry.get_active_model_version(
                            sensor_id, "transformer"
                        )
                        if version_id:
                            model = TransformerForecaster(sensor_id)
                            metadata = self.model_registry.get_model_metadata(
                                version_id
                            )

                            if metadata and Path(metadata.model_path).exists():
                                model.load_model(Path(metadata.model_path).parent)

                                val_metrics = self._validate_model(
                                    model, data_split["val_data"], "transformer"
                                )
                                sensor_results["transformer"] = val_metrics

                    validation_results["results"][sensor_id] = {
                        "success": True,
                        "validation_results": sensor_results,
                    }
                    validation_results["sensors_validated"] += 1

                except Exception as e:
                    logger.error(f"Error validating sensor {sensor_id}: {e}")
                    validation_results["results"][sensor_id] = {
                        "success": False,
                        "error": str(e),
                    }
                    validation_results["sensors_failed"] += 1

            return validation_results

        except Exception as e:
            logger.error(f"Error in batch validation: {e}")
            return {
                "batch_validation": True,
                "error": str(e),
                "validated_at": datetime.now().isoformat(),
            }

    def manage_model_versions(self, action: str, **kwargs) -> Dict[str, Any]:
        """
        Manage model versions

        Args:
            action: Action to perform ('list', 'promote', 'delete', 'cleanup')
            **kwargs: Action-specific parameters

        Returns:
            Action results
        """
        try:
            if action == "list":
                sensor_id = kwargs.get("sensor_id")
                model_type = kwargs.get("model_type")

                if sensor_id and model_type:
                    versions = self.model_registry.list_versions(sensor_id, model_type)
                    return {
                        "action": "list",
                        "sensor_id": sensor_id,
                        "model_type": model_type,
                        "versions": versions,
                    }
                else:
                    models = self.model_registry.list_models(model_type)
                    return {"action": "list", "models": models}

            elif action == "promote":
                version_id = kwargs.get("version_id")
                if not version_id:
                    return {"error": "version_id required for promote action"}

                success = self.model_registry.promote_version(version_id)
                return {
                    "action": "promote",
                    "version_id": version_id,
                    "success": success,
                }

            elif action == "delete":
                version_id = kwargs.get("version_id")
                force = kwargs.get("force", False)
                delete_artifacts = kwargs.get("delete_artifacts", True)

                if not version_id:
                    return {"error": "version_id required for delete action"}

                success = self.model_registry.delete_version(
                    version_id, force=force, delete_artifacts=delete_artifacts
                )
                return {
                    "action": "delete",
                    "version_id": version_id,
                    "success": success,
                }

            elif action == "cleanup":
                keep_last_n = kwargs.get("keep_last_n", 3)
                cleaned = self.model_registry.cleanup_old_versions(keep_last_n)
                return {"action": "cleanup", "versions_cleaned": cleaned}

            else:
                return {"error": f"Unknown action: {action}"}

        except Exception as e:
            logger.error(f"Error managing model versions: {e}")
            return {"error": str(e), "action": action}

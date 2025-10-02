"""
MLflow Training Orchestrator
Integrated training orchestrator with MLflow experiment tracking and model registry
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.services.data_drift_detector import DataDriftDetector
from src.core.services.retraining_trigger import (
    RetrainingPolicy,
    RetrainingReason,
    RetrainingTrigger,
    RetrainingTriggerSystem,
)
from src.infrastructure.data.data_pipeline import DataPipeline
from src.infrastructure.ml.mlflow_tracker import MLflowTracker, ModelStage
from src.infrastructure.ml.telemanom_wrapper import NASATelemanom, Telemanom_Config
from src.infrastructure.ml.transformer_wrapper import (
    TransformerConfig,
    TransformerForecaster,
)

logger = logging.getLogger(__name__)


class MLflowTrainingOrchestrator:
    """
    Integrated training orchestrator with MLflow tracking

    Features:
    - Experiment tracking
    - Model versioning
    - Automated retraining triggers
    - Model staging and promotion
    - Comprehensive logging
    """

    def __init__(
        self,
        mlflow_tracker: Optional[MLflowTracker] = None,
        data_pipeline: Optional[DataPipeline] = None,
        retraining_system: Optional[RetrainingTriggerSystem] = None,
        experiment_name: str = "iot-predictive-maintenance",
        tracking_uri: str = "file:./mlruns",
    ):
        """
        Initialize MLflow training orchestrator

        Args:
            mlflow_tracker: MLflow tracker instance
            data_pipeline: Data pipeline for data preparation
            retraining_system: Automated retraining trigger system
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking server URI
        """
        self.mlflow_tracker = mlflow_tracker or MLflowTracker(
            experiment_name=experiment_name, tracking_uri=tracking_uri
        )

        self.data_pipeline = data_pipeline or DataPipeline()

        self.retraining_system = retraining_system or RetrainingTriggerSystem(mlflow_tracker=self.mlflow_tracker)

        logger.info("MLflow training orchestrator initialized")

    def train_anomaly_detection_model(
        self,
        sensor_id: str,
        config: Optional[Telemanom_Config] = None,
        hours_back: int = 168,
        auto_promote: bool = False,
        run_tags: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Train anomaly detection model with MLflow tracking

        Args:
            sensor_id: Sensor identifier
            config: Model configuration
            hours_back: Hours of historical data to use
            auto_promote: Automatically promote to production if better
            run_tags: Additional tags for MLflow run

        Returns:
            Training results with MLflow run info
        """
        model_type = "telemanom"
        config = config or Telemanom_Config()

        logger.info(f"Training {model_type} for {sensor_id}")

        try:
            # Prepare training data
            data_prepared = self.data_pipeline.prepare_training_data(
                sensor_id=sensor_id,
                split_ratio=(0.7, 0.15, 0.15),
                hours_back=hours_back,
                normalize=True,
                assess_quality=True,
            )

            train_data = data_prepared["train_data"]
            val_data = data_prepared["val_data"]
            test_data = data_prepared["test_data"]
            data_hash = data_prepared["data_hash"]
            quality_report = data_prepared.get("quality_report")

            # Check data quality trigger
            if quality_report:
                quality_trigger = self.retraining_system.check_quality_trigger(
                    sensor_id=sensor_id,
                    model_type=model_type,
                    quality_score=self._quality_status_to_score(quality_report.status),
                    quality_issues=quality_report.issues,
                )
                if quality_trigger:
                    logger.warning(f"Data quality issues detected: {quality_report.issues}")

            # Create model
            model = NASATelemanom(sensor_id=sensor_id, config=config)

            # Prepare MLflow parameters
            params = {
                **config.__dict__,
                "sensor_id": sensor_id,
                "data_hash": data_hash,
                "train_samples": len(train_data),
                "val_samples": len(val_data),
                "test_samples": len(test_data),
                "quality_status": (quality_report.status.value if quality_report else "unknown"),
            }

            # Prepare tags
            tags = {
                "sensor_id": sensor_id,
                "model_type": model_type,
                "data_quality": (quality_report.status.value if quality_report else "unknown"),
            }
            if run_tags:
                tags.update(run_tags)

            # Start MLflow run
            run_name = f"{model_type}_{sensor_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            with self.mlflow_tracker.start_run(run_name=run_name, tags=tags) as run:
                # Log parameters
                for key, value in params.items():
                    try:
                        self.mlflow_tracker.client.log_param(run.info.run_id, key, str(value))
                    except Exception as e:
                        logger.warning(f"Could not log param {key}: {e}")

                # Train model
                logger.info(f"Training {model_type} model...")
                training_result = model.train(train_data)

                # Validate on held-out data
                logger.info("Validating on held-out dataset...")
                val_result = model.detect_anomalies(val_data, sensor_id=sensor_id)

                # Compute validation metrics
                val_metrics = self._compute_validation_metrics(val_result)

                # Log training metrics
                training_metrics = training_result.get("training_metrics", {})
                for key, value in training_metrics.items():
                    try:
                        if isinstance(value, (int, float)) and not np.isnan(value):
                            self.mlflow_tracker.client.log_metric(run.info.run_id, f"train_{key}", float(value))
                    except Exception as e:
                        logger.warning(f"Could not log training metric {key}: {e}")

                # Log validation metrics
                for key, value in val_metrics.items():
                    try:
                        if isinstance(value, (int, float)) and not np.isnan(value):
                            self.mlflow_tracker.client.log_metric(run.info.run_id, f"val_{key}", float(value))
                    except Exception as e:
                        logger.warning(f"Could not log validation metric {key}: {e}")

                # Save model artifact
                model_path = Path(f"models/{sensor_id}_{model_type}_model.pkl")
                model_path.parent.mkdir(parents=True, exist_ok=True)
                # model.save(str(model_path))  # Assuming save method exists

                # Log model artifact
                try:
                    self.mlflow_tracker.client.log_artifact(run.info.run_id, str(model_path))
                except Exception as e:
                    logger.warning(f"Could not log model artifact: {e}")

                # Register model
                model_name = f"{model_type}_{sensor_id}"
                try:
                    version = self.mlflow_tracker.register_model(
                        run_id=run.info.run_id,
                        model_name=model_name,
                        artifact_path="model",
                    )

                    # Promote to staging
                    self.mlflow_tracker.transition_model_stage(
                        model_name=model_name,
                        version=version,
                        stage=ModelStage.STAGING,
                        archive_existing=False,
                    )

                    # Auto-promote to production if enabled
                    if auto_promote:
                        production_version = self.mlflow_tracker.get_model_version(
                            model_name=model_name, stage=ModelStage.PRODUCTION
                        )

                        if production_version:
                            # Get production performance
                            prod_run = self.mlflow_tracker.client.get_run(production_version.run_id)
                            prod_performance = prod_run.data.metrics.get("val_accuracy", 0.0)
                            new_performance = val_metrics.get("accuracy", 0.0)

                            self.retraining_system.auto_promote_model(
                                model_name=model_name,
                                new_version=version,
                                new_performance=new_performance,
                                production_performance=prod_performance,
                            )
                        else:
                            # No production model, promote this one
                            self.mlflow_tracker.transition_model_stage(
                                model_name=model_name,
                                version=version,
                                stage=ModelStage.PRODUCTION,
                                archive_existing=False,
                            )

                except Exception as e:
                    logger.error(f"Error registering/promoting model: {e}")

                logger.info(f"Training complete for {sensor_id}")

                return {
                    "success": True,
                    "sensor_id": sensor_id,
                    "model_type": model_type,
                    "run_id": run.info.run_id,
                    "model_name": model_name,
                    "training_metrics": training_metrics,
                    "validation_metrics": val_metrics,
                    "data_hash": data_hash,
                    "quality_status": (quality_report.status.value if quality_report else "unknown"),
                }

        except Exception as e:
            logger.error(f"Error training {model_type} for {sensor_id}: {e}", exc_info=True)
            return {
                "success": False,
                "sensor_id": sensor_id,
                "model_type": model_type,
                "error": str(e),
            }

    def train_forecasting_model(
        self,
        sensor_id: str,
        config: Optional[TransformerConfig] = None,
        hours_back: int = 168,
        auto_promote: bool = False,
        run_tags: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Train forecasting model with MLflow tracking

        Args:
            sensor_id: Sensor identifier
            config: Model configuration
            hours_back: Hours of historical data to use
            auto_promote: Automatically promote to production if better
            run_tags: Additional tags for MLflow run

        Returns:
            Training results with MLflow run info
        """
        model_type = "transformer"
        config = config or TransformerConfig()

        logger.info(f"Training {model_type} for {sensor_id}")

        try:
            # Prepare training data
            data_prepared = self.data_pipeline.prepare_training_data(
                sensor_id=sensor_id,
                split_ratio=(0.7, 0.15, 0.15),
                hours_back=hours_back,
                normalize=True,
                assess_quality=True,
            )

            train_data = data_prepared["train_data"]
            val_data = data_prepared["val_data"]
            test_data = data_prepared["test_data"]
            data_hash = data_prepared["data_hash"]
            quality_report = data_prepared.get("quality_report")

            # Create model
            model = TransformerForecaster(sensor_id=sensor_id, config=config)

            # Prepare MLflow parameters
            params = {
                **config.__dict__,
                "sensor_id": sensor_id,
                "data_hash": data_hash,
                "train_samples": len(train_data),
                "val_samples": len(val_data),
                "test_samples": len(test_data),
                "quality_status": (quality_report.status.value if quality_report else "unknown"),
            }

            # Prepare tags
            tags = {
                "sensor_id": sensor_id,
                "model_type": model_type,
                "data_quality": (quality_report.status.value if quality_report else "unknown"),
            }
            if run_tags:
                tags.update(run_tags)

            # Start MLflow run
            run_name = f"{model_type}_{sensor_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            with self.mlflow_tracker.start_run(run_name=run_name, tags=tags) as run:
                # Log parameters
                for key, value in params.items():
                    try:
                        self.mlflow_tracker.client.log_param(run.info.run_id, key, str(value))
                    except Exception as e:
                        logger.warning(f"Could not log param {key}: {e}")

                # Train model
                logger.info(f"Training {model_type} model...")
                training_result = model.train(train_data)

                # Validate on held-out data
                logger.info("Validating on held-out dataset...")
                val_result = model.forecast(
                    historical_data=val_data[: config.lookback_window],
                    horizon=config.forecast_horizon,
                    sensor_id=sensor_id,
                )

                # Compute validation metrics
                val_metrics = self._compute_forecast_metrics(
                    val_result["forecast_values"],
                    val_data[config.lookback_window : config.lookback_window + config.forecast_horizon],
                )

                # Log training metrics
                training_metrics = training_result.get("training_metrics", {})
                for key, value in training_metrics.items():
                    try:
                        if isinstance(value, (int, float)) and not np.isnan(value):
                            self.mlflow_tracker.client.log_metric(run.info.run_id, f"train_{key}", float(value))
                    except Exception as e:
                        logger.warning(f"Could not log training metric {key}: {e}")

                # Log validation metrics
                for key, value in val_metrics.items():
                    try:
                        if isinstance(value, (int, float)) and not np.isnan(value):
                            self.mlflow_tracker.client.log_metric(run.info.run_id, f"val_{key}", float(value))
                    except Exception as e:
                        logger.warning(f"Could not log validation metric {key}: {e}")

                # Register model
                model_name = f"{model_type}_{sensor_id}"
                try:
                    version = self.mlflow_tracker.register_model(
                        run_id=run.info.run_id,
                        model_name=model_name,
                        artifact_path="model",
                    )

                    # Promote to staging
                    self.mlflow_tracker.transition_model_stage(
                        model_name=model_name,
                        version=version,
                        stage=ModelStage.STAGING,
                        archive_existing=False,
                    )

                except Exception as e:
                    logger.error(f"Error registering model: {e}")

                logger.info(f"Training complete for {sensor_id}")

                return {
                    "success": True,
                    "sensor_id": sensor_id,
                    "model_type": model_type,
                    "run_id": run.info.run_id,
                    "model_name": model_name,
                    "training_metrics": training_metrics,
                    "validation_metrics": val_metrics,
                    "data_hash": data_hash,
                }

        except Exception as e:
            logger.error(f"Error training {model_type} for {sensor_id}: {e}", exc_info=True)
            return {
                "success": False,
                "sensor_id": sensor_id,
                "model_type": model_type,
                "error": str(e),
            }

    def check_and_retrain(
        self,
        sensor_id: str,
        model_type: str,
        current_data: np.ndarray,
        reference_data: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Check if retraining is needed and trigger if necessary

        Args:
            sensor_id: Sensor identifier
            model_type: Model type
            current_data: Current sensor data
            reference_data: Reference data for drift detection

        Returns:
            Results of check and retraining (if triggered)
        """
        logger.info(f"Checking retraining triggers for {sensor_id} ({model_type})")

        results = {
            "sensor_id": sensor_id,
            "model_type": model_type,
            "retraining_triggered": False,
            "triggers": [],
        }

        # Check drift
        if reference_data is not None:
            drift_report = self.retraining_system.drift_detector.detect_drift(
                current_data=current_data,
                sensor_id=sensor_id,
                reference_data=reference_data,
            )

            drift_trigger = self.retraining_system.check_drift_trigger(
                sensor_id=sensor_id, model_type=model_type, drift_report=drift_report
            )

            if drift_trigger:
                results["triggers"].append(drift_trigger)

        # If any triggers, check if retraining should proceed
        if results["triggers"]:
            for trigger in results["triggers"]:
                should_retrain = self.retraining_system.should_retrain(
                    sensor_id=sensor_id, model_type=model_type, trigger=trigger
                )

                if should_retrain:
                    # Execute retraining
                    logger.info(f"Executing retraining for {sensor_id} ({model_type})")

                    if model_type == "telemanom":
                        training_result = self.train_anomaly_detection_model(
                            sensor_id=sensor_id,
                            auto_promote=True,
                            run_tags={"trigger_reason": trigger.reason.value},
                        )
                    elif model_type == "transformer":
                        training_result = self.train_forecasting_model(
                            sensor_id=sensor_id,
                            auto_promote=True,
                            run_tags={"trigger_reason": trigger.reason.value},
                        )
                    else:
                        training_result = {
                            "success": False,
                            "error": f"Unknown model type: {model_type}",
                        }

                    # Register trigger
                    self.retraining_system.register_trigger(
                        trigger=trigger,
                        execute_retraining=training_result.get("success", False),
                    )

                    results["retraining_triggered"] = True
                    results["training_result"] = training_result

                    break  # Only retrain once

        return results

    def _compute_validation_metrics(self, detection_result: Dict[str, Any]) -> Dict[str, float]:
        """Compute validation metrics for anomaly detection"""
        # Extract anomaly predictions
        anomalies = detection_result.get("anomalies", [])
        anomaly_scores = detection_result.get("anomaly_scores", [])

        if not anomaly_scores:
            return {}

        return {
            "num_anomalies": len(anomalies),
            "mean_score": float(np.mean(anomaly_scores)),
            "max_score": float(np.max(anomaly_scores)),
            "std_score": float(np.std(anomaly_scores)),
        }

    def _compute_forecast_metrics(self, forecast: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
        """Compute validation metrics for forecasting"""
        if len(forecast) == 0 or len(actual) == 0:
            return {}

        # Ensure same length
        min_len = min(len(forecast), len(actual))
        forecast = forecast[:min_len]
        actual = actual[:min_len]

        # MAE
        mae = float(np.mean(np.abs(forecast - actual)))

        # RMSE
        rmse = float(np.sqrt(np.mean((forecast - actual) ** 2)))

        # MAPE
        epsilon = 1e-10
        mape = float(np.mean(np.abs((actual - forecast) / (actual + epsilon))) * 100)

        return {"mae": mae, "rmse": rmse, "mape": mape}

    def _quality_status_to_score(self, status) -> float:
        """Convert quality status to numeric score"""
        from src.core.services.data_processing_service import DataQualityStatus

        quality_mapping = {
            DataQualityStatus.EXCELLENT: 1.0,
            DataQualityStatus.GOOD: 0.8,
            DataQualityStatus.FAIR: 0.6,
            DataQualityStatus.POOR: 0.4,
            DataQualityStatus.CRITICAL: 0.2,
        }

        return quality_mapping.get(status, 0.5)

"""
MLflow Experiment Tracking Wrapper
Comprehensive MLOps integration for model training, versioning, and lifecycle management
"""

import json
import logging
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import numpy as np
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """Model lifecycle stages"""

    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


@dataclass
class ExperimentConfig:
    """Configuration for MLflow experiments"""

    experiment_name: str
    tracking_uri: str = "file:./mlruns"
    artifact_location: Optional[str] = None
    tags: Dict[str, str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


class MLflowTracker:
    """
    MLflow experiment tracking and model registry integration

    Provides comprehensive MLOps capabilities:
    - Experiment tracking
    - Parameter and metric logging
    - Artifact management
    - Model versioning and staging
    - Model lifecycle management
    """

    def __init__(
        self,
        experiment_name: str = "iot-predictive-maintenance",
        tracking_uri: str = "file:./mlruns",
        artifact_location: Optional[str] = None,
        registry_uri: Optional[str] = None,
    ):
        """
        Initialize MLflow tracker

        Args:
            experiment_name: Name of MLflow experiment
            tracking_uri: MLflow tracking server URI
            artifact_location: Custom artifact storage location
            registry_uri: Model registry URI (uses tracking_uri if None)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.artifact_location = artifact_location
        self.registry_uri = registry_uri or tracking_uri

        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)

        # Set registry URI
        if self.registry_uri:
            mlflow.set_registry_uri(self.registry_uri)

        # Initialize or get experiment
        self.experiment = self._get_or_create_experiment()

        # MLflow client for advanced operations
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

        logger.info(
            f"MLflow tracker initialized: experiment='{self.experiment_name}', "
            f"tracking_uri='{self.tracking_uri}'"
        )

    def _get_or_create_experiment(self) -> mlflow.entities.Experiment:
        """Get existing experiment or create new one"""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)

            if experiment is None:
                # Create new experiment
                experiment_id = mlflow.create_experiment(
                    name=self.experiment_name, artifact_location=self.artifact_location
                )
                experiment = mlflow.get_experiment(experiment_id)
                logger.info(f"Created new experiment: {self.experiment_name}")
            else:
                logger.info(f"Using existing experiment: {self.experiment_name}")

            return experiment

        except Exception as e:
            logger.error(f"Error managing experiment: {e}")
            raise

    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
    ):
        """
        Context manager for MLflow runs

        Usage:
            with tracker.start_run(run_name="training_run") as run:
                mlflow.log_param("lr", 0.01)
                mlflow.log_metric("accuracy", 0.95)

        Args:
            run_name: Name for the run
            tags: Additional tags for the run
            nested: Whether this is a nested run

        Yields:
            Active MLflow run
        """
        run = mlflow.start_run(
            experiment_id=self.experiment.experiment_id,
            run_name=run_name,
            tags=tags,
            nested=nested,
        )

        try:
            yield run
        finally:
            mlflow.end_run()

    def log_training_run(
        self,
        sensor_id: str,
        model_type: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts: Optional[Dict[str, str]] = None,
        model: Optional[Any] = None,
        model_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Log a complete training run

        Args:
            sensor_id: Sensor identifier
            model_type: Type of model (telemanom, transformer, etc.)
            params: Training parameters
            metrics: Training and validation metrics
            artifacts: Dictionary of artifact paths to log
            model: Trained model object (for model registry)
            model_name: Registered model name
            tags: Additional run tags

        Returns:
            Run ID
        """
        run_name = (
            f"{model_type}_{sensor_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Prepare tags
        run_tags = {
            "sensor_id": sensor_id,
            "model_type": model_type,
            "training_date": datetime.now().isoformat(),
        }
        if tags:
            run_tags.update(tags)

        with self.start_run(run_name=run_name, tags=run_tags) as run:
            # Log parameters
            for key, value in params.items():
                try:
                    # MLflow params must be strings
                    mlflow.log_param(key, str(value))
                except Exception as e:
                    logger.warning(f"Could not log param {key}: {e}")

            # Log metrics
            for key, value in metrics.items():
                try:
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        mlflow.log_metric(key, float(value))
                except Exception as e:
                    logger.warning(f"Could not log metric {key}: {e}")

            # Log artifacts
            if artifacts:
                for artifact_name, artifact_path in artifacts.items():
                    try:
                        if Path(artifact_path).exists():
                            mlflow.log_artifact(artifact_path)
                        else:
                            logger.warning(f"Artifact not found: {artifact_path}")
                    except Exception as e:
                        logger.warning(f"Could not log artifact {artifact_name}: {e}")

            # Log model to registry
            if model is not None and model_name is not None:
                try:
                    self._log_model_to_registry(
                        model=model, model_name=model_name, model_type=model_type
                    )
                except Exception as e:
                    logger.error(f"Could not log model to registry: {e}")

            logger.info(f"Logged training run: {run_name} (ID: {run.info.run_id})")
            return run.info.run_id

    def _log_model_to_registry(self, model: Any, model_name: str, model_type: str):
        """Log model to MLflow model registry"""
        try:
            if model_type == "sklearn" or hasattr(model, "predict"):
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=model_name,
                )
            elif model_type == "tensorflow":
                mlflow.tensorflow.log_model(
                    model=model, artifact_path="model", registered_model_name=model_name
                )
            else:
                # Generic Python model
                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=model,
                    registered_model_name=model_name,
                )

            logger.info(f"Logged model '{model_name}' to registry")

        except Exception as e:
            logger.error(f"Error logging model: {e}")
            raise

    def register_model(
        self, run_id: str, model_name: str, artifact_path: str = "model"
    ) -> str:
        """
        Register a model from a completed run

        Args:
            run_id: MLflow run ID
            model_name: Name for registered model
            artifact_path: Path to model artifact in run

        Returns:
            Model version number
        """
        try:
            model_uri = f"runs:/{run_id}/{artifact_path}"

            result = mlflow.register_model(model_uri=model_uri, name=model_name)

            logger.info(
                f"Registered model '{model_name}' version {result.version} "
                f"from run {run_id}"
            )

            return result.version

        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise

    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: ModelStage,
        archive_existing: bool = True,
    ):
        """
        Transition model to a different stage

        Args:
            model_name: Registered model name
            version: Model version number
            stage: Target stage (STAGING, PRODUCTION, ARCHIVED)
            archive_existing: Archive existing models in target stage
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage.value,
                archive_existing_versions=archive_existing,
            )

            logger.info(
                f"Transitioned model '{model_name}' v{version} to {stage.value}"
            )

        except Exception as e:
            logger.error(f"Error transitioning model stage: {e}")
            raise

    def get_model_version(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[ModelStage] = None,
    ) -> Optional[Any]:
        """
        Get a specific model version

        Args:
            model_name: Registered model name
            version: Specific version (optional)
            stage: Get latest from stage (STAGING or PRODUCTION)

        Returns:
            Model version metadata
        """
        try:
            if stage:
                # Get latest version in stage
                versions = self.client.get_latest_versions(
                    name=model_name, stages=[stage.value]
                )
                if versions:
                    return versions[0]
                else:
                    return None

            elif version:
                # Get specific version
                return self.client.get_model_version(name=model_name, version=version)

            else:
                # Get latest version overall
                versions = self.client.search_model_versions(
                    filter_string=f"name='{model_name}'"
                )
                if versions:
                    return max(versions, key=lambda v: int(v.version))
                else:
                    return None

        except MlflowException as e:
            logger.warning(f"Model version not found: {e}")
            return None

    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[ModelStage] = None,
    ) -> Any:
        """
        Load a model from registry

        Args:
            model_name: Registered model name
            version: Specific version number
            stage: Load from stage (STAGING or PRODUCTION)

        Returns:
            Loaded model
        """
        try:
            if stage:
                model_uri = f"models:/{model_name}/{stage.value}"
            elif version:
                model_uri = f"models:/{model_name}/{version}"
            else:
                model_uri = f"models:/{model_name}/latest"

            model = mlflow.pyfunc.load_model(model_uri)

            logger.info(f"Loaded model from {model_uri}")
            return model

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def search_runs(
        self,
        filter_string: Optional[str] = None,
        max_results: int = 100,
        order_by: Optional[List[str]] = None,
    ) -> List[mlflow.entities.Run]:
        """
        Search for runs in the experiment

        Args:
            filter_string: Filter expression (e.g., "params.lr > 0.01")
            max_results: Maximum number of results
            order_by: List of columns to order by

        Returns:
            List of matching runs
        """
        try:
            runs = self.client.search_runs(
                experiment_ids=[self.experiment.experiment_id],
                filter_string=filter_string,
                max_results=max_results,
                order_by=order_by,
            )

            return runs

        except Exception as e:
            logger.error(f"Error searching runs: {e}")
            return []

    def get_best_run(
        self,
        metric_name: str,
        maximize: bool = True,
        filter_string: Optional[str] = None,
    ) -> Optional[mlflow.entities.Run]:
        """
        Get the best run based on a metric

        Args:
            metric_name: Metric to optimize
            maximize: Whether to maximize (True) or minimize (False)
            filter_string: Optional filter for runs

        Returns:
            Best run or None
        """
        order_direction = "DESC" if maximize else "ASC"
        order_by = [f"metrics.{metric_name} {order_direction}"]

        runs = self.search_runs(
            filter_string=filter_string, max_results=1, order_by=order_by
        )

        return runs[0] if runs else None

    def compare_runs(
        self, run_ids: List[str], metrics: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare metrics across multiple runs

        Args:
            run_ids: List of run IDs to compare
            metrics: Specific metrics to compare (all if None)

        Returns:
            Dictionary mapping run_id to metrics
        """
        comparison = {}

        for run_id in run_ids:
            try:
                run = self.client.get_run(run_id)

                if metrics:
                    run_metrics = {
                        k: v for k, v in run.data.metrics.items() if k in metrics
                    }
                else:
                    run_metrics = run.data.metrics

                comparison[run_id] = {
                    "metrics": run_metrics,
                    "params": run.data.params,
                    "tags": run.data.tags,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                }

            except Exception as e:
                logger.warning(f"Could not get run {run_id}: {e}")

        return comparison

    def delete_runs(self, run_ids: List[str], permanently: bool = False):
        """
        Delete runs

        Args:
            run_ids: List of run IDs to delete
            permanently: If True, permanently delete (else mark as deleted)
        """
        for run_id in run_ids:
            try:
                if permanently:
                    self.client.delete_run(run_id)
                    logger.info(f"Permanently deleted run {run_id}")
                else:
                    # MLflow soft delete
                    self.client.delete_run(run_id)
                    logger.info(f"Deleted run {run_id}")

            except Exception as e:
                logger.warning(f"Could not delete run {run_id}: {e}")

    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the experiment"""
        try:
            runs = self.search_runs(max_results=1000)

            if not runs:
                return {"experiment_name": self.experiment_name, "total_runs": 0}

            # Extract metrics
            all_metrics = {}
            for run in runs:
                for metric_name, metric_value in run.data.metrics.items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(metric_value)

            # Compute statistics
            metric_stats = {}
            for metric_name, values in all_metrics.items():
                metric_stats[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values),
                }

            return {
                "experiment_name": self.experiment_name,
                "total_runs": len(runs),
                "metric_statistics": metric_stats,
                "first_run": runs[-1].info.start_time if runs else None,
                "last_run": runs[0].info.start_time if runs else None,
            }

        except Exception as e:
            logger.error(f"Error generating experiment summary: {e}")
            return {"error": str(e)}

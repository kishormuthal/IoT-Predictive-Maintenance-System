"""
Data Pipeline Orchestrator
End-to-end data processing pipeline with versioning, validation, and drift detection
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...core.services.data_drift_detector import (
    DataDriftDetector,
    DriftConfig,
    DriftReport,
)
from ...core.services.data_processing_service import (
    DataProcessingService,
    DataQualityReport,
    DataQualityStatus,
    NormalizationMethod,
)
from ...core.services.feature_engineering import FeatureConfig, FeatureEngineer
from .dvc_manager import DatasetVersion, DVCManager
from .nasa_data_loader import NASADataLoader

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    End-to-end data processing pipeline orchestrator

    Coordinates:
    - Data loading
    - Quality assessment
    - Preprocessing/normalization
    - Feature engineering
    - Drift detection
    - Versioning
    """

    def __init__(
        self,
        data_loader: Optional[NASADataLoader] = None,
        processing_service: Optional[DataProcessingService] = None,
        feature_engineer: Optional[FeatureEngineer] = None,
        drift_detector: Optional[DataDriftDetector] = None,
        dvc_manager: Optional[DVCManager] = None,
        pipeline_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize data pipeline

        Args:
            data_loader: Data loading service
            processing_service: Data processing service
            feature_engineer: Feature engineering service
            drift_detector: Drift detection service
            dvc_manager: DVC version control manager
            pipeline_config: Pipeline configuration
        """
        # Initialize components
        self.data_loader = data_loader or NASADataLoader()
        self.processing_service = processing_service or DataProcessingService()
        self.feature_engineer = feature_engineer or FeatureEngineer()
        self.drift_detector = drift_detector or DataDriftDetector()
        self.dvc_manager = dvc_manager or DVCManager()

        # Configuration
        self.config = pipeline_config or {}

        # Pipeline state
        self.pipeline_runs: Dict[str, Dict[str, Any]] = {}
        self.sensor_metadata: Dict[str, Dict[str, Any]] = {}

        # Output directory
        self.output_dir = Path(self.config.get("output_dir", "data/processed"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_full_pipeline(
        self,
        sensor_id: str,
        hours_back: int = 168,  # 7 days
        normalize: bool = True,
        normalization_method: Optional[NormalizationMethod] = None,
        engineer_features: bool = True,
        feature_config: Optional[FeatureConfig] = None,
        detect_drift: bool = True,
        version_dataset: bool = True,
        save_processed: bool = True,
    ) -> Dict[str, Any]:
        """
        Run full data processing pipeline for a sensor

        Args:
            sensor_id: Sensor identifier
            hours_back: Hours of historical data to process
            normalize: Whether to normalize data
            normalization_method: Normalization method to use
            engineer_features: Whether to engineer features
            feature_config: Feature engineering configuration
            detect_drift: Whether to detect drift
            version_dataset: Whether to version processed dataset
            save_processed: Whether to save processed data to disk

        Returns:
            Dictionary containing pipeline results
        """
        pipeline_id = f"{sensor_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting pipeline run {pipeline_id}")

        results = {
            "pipeline_id": pipeline_id,
            "sensor_id": sensor_id,
            "start_time": datetime.now(),
            "success": False,
            "errors": [],
        }

        try:
            # Step 1: Load raw data
            logger.info(f"[1/6] Loading data for {sensor_id}")
            raw_data_dict = self.data_loader.get_sensor_data(sensor_id, hours_back)

            if not raw_data_dict or not raw_data_dict.get("values"):
                raise ValueError(f"No data available for sensor {sensor_id}")

            raw_data = np.array(raw_data_dict["values"])
            timestamps = raw_data_dict.get("timestamps", [])

            results["raw_data_shape"] = raw_data.shape
            results["timestamps_count"] = len(timestamps)

            # Step 2: Data quality assessment
            logger.info(f"[2/6] Assessing data quality for {sensor_id}")
            quality_report = self.processing_service.assess_data_quality(raw_data, timestamps, sensor_id)

            results["quality_report"] = quality_report

            if quality_report.status == DataQualityStatus.CRITICAL:
                logger.error(f"Critical data quality issues: {quality_report.issues}")
                results["errors"].append(f"Critical data quality: {quality_report.issues}")
                # Continue anyway, but flag it

            # Step 3: Preprocessing and normalization
            logger.info(f"[3/6] Preprocessing data for {sensor_id}")
            processed_data = raw_data.copy()

            # Handle missing values (simple forward fill)
            missing_mask = ~np.isfinite(processed_data)
            if np.any(missing_mask):
                logger.warning(f"Imputing {np.sum(missing_mask)} missing values")
                processed_data = self._impute_missing(processed_data)

            # Normalize if requested
            norm_params = None
            if normalize:
                method = normalization_method or self.processing_service.default_normalization

                # Check if we have existing parameters
                existing_params = self.processing_service.get_cached_params(sensor_id)

                if existing_params:
                    logger.info(f"Using cached normalization parameters for {sensor_id}")
                    norm_params = existing_params
                    normalized_data = self.processing_service.normalize(processed_data, sensor_id, norm_params)
                else:
                    # Fit new parameters on this data
                    logger.info(f"Fitting new normalization parameters for {sensor_id}")
                    norm_params = self.processing_service.fit_normalization(processed_data, sensor_id, method)
                    normalized_data = self.processing_service.normalize(processed_data, sensor_id, norm_params)

                processed_data = normalized_data

            results["normalized"] = normalize
            results["norm_params"] = norm_params.to_dict() if norm_params else None

            # Step 4: Feature engineering
            features = None
            if engineer_features:
                logger.info(f"[4/6] Engineering features for {sensor_id}")
                feature_config = feature_config or FeatureConfig()
                self.feature_engineer.config = feature_config

                features = self.feature_engineer.engineer_features(processed_data, timestamps, sensor_id)

                results["engineered_features"] = list(features.keys())
                results["feature_count"] = len(features)
            else:
                logger.info(f"[4/6] Skipping feature engineering")
                results["engineered_features"] = []

            # Step 5: Drift detection
            drift_report = None
            if detect_drift:
                logger.info(f"[5/6] Detecting drift for {sensor_id}")

                # Check if we have reference distribution
                if sensor_id in self.drift_detector.reference_distributions:
                    drift_report = self.drift_detector.detect_drift(processed_data, sensor_id, timestamps)
                    results["drift_report"] = drift_report.to_dict()
                    results["drift_detected"] = drift_report.drift_detected
                else:
                    # Fit reference distribution
                    logger.info(f"Fitting reference distribution for {sensor_id}")
                    self.drift_detector.fit_reference(processed_data, sensor_id, timestamps)
                    results["drift_report"] = "Reference distribution fitted (no drift check)"
                    results["drift_detected"] = False
            else:
                logger.info(f"[5/6] Skipping drift detection")

            # Step 6: Versioning and saving
            dataset_version = None
            if save_processed or version_dataset:
                logger.info(f"[6/6] Saving and versioning processed data")

                # Save processed data
                output_file = self.output_dir / f"{sensor_id}_processed.npy"
                np.save(output_file, processed_data)

                # Save features if engineered
                if features:
                    features_file = self.output_dir / f"{sensor_id}_features.npz"
                    np.savez(features_file, **features)

                # Save metadata
                metadata = {
                    "sensor_id": sensor_id,
                    "pipeline_id": pipeline_id,
                    "raw_shape": raw_data.shape,
                    "processed_shape": processed_data.shape,
                    "normalized": normalize,
                    "features_engineered": engineer_features,
                    "quality_status": quality_report.status.value,
                    "drift_detected": results.get("drift_detected", False),
                    "processing_timestamp": datetime.now().isoformat(),
                }

                metadata_file = self.output_dir / f"{sensor_id}_metadata.json"
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)

                # Version with DVC
                if version_dataset:
                    dataset_version = self.dvc_manager.version_dataset(
                        file_path=str(output_file),
                        dataset_id=sensor_id,
                        description=f"Processed sensor data from pipeline {pipeline_id}",
                        tags=[
                            quality_report.status.value,
                            "normalized" if normalize else "raw",
                            "drift" if results.get("drift_detected") else "stable",
                        ],
                        sensor_ids=[sensor_id],
                        push_to_remote=False,
                    )

                    if dataset_version:
                        results["dataset_version"] = dataset_version.version
                        results["data_hash"] = dataset_version.data_hash
                    else:
                        results["errors"].append("Dataset versioning failed")

                results["output_file"] = str(output_file)
            else:
                logger.info(f"[6/6] Skipping save/version")

            # Mark success
            results["success"] = True
            results["end_time"] = datetime.now()
            results["duration_seconds"] = (results["end_time"] - results["start_time"]).total_seconds()

            # Store pipeline run
            self.pipeline_runs[pipeline_id] = results

            logger.info(f"Pipeline {pipeline_id} completed successfully in " f"{results['duration_seconds']:.2f}s")

            return results

        except Exception as e:
            logger.error(f"Pipeline {pipeline_id} failed: {e}", exc_info=True)
            results["success"] = False
            results["errors"].append(str(e))
            results["end_time"] = datetime.now()
            return results

    def prepare_training_data(
        self,
        sensor_id: str,
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        hours_back: int = 168,
        normalize: bool = True,
        assess_quality: bool = True,
    ) -> Dict[str, Any]:
        """
        Prepare data specifically for model training

        Args:
            sensor_id: Sensor identifier
            split_ratio: (train, val, test) split ratios
            hours_back: Hours of historical data
            normalize: Whether to normalize data
            assess_quality: Whether to assess data quality

        Returns:
            Dictionary containing train/val/test splits and metadata
        """
        logger.info(f"Preparing training data for {sensor_id}")

        # Load data
        raw_data_dict = self.data_loader.get_sensor_data(sensor_id, hours_back)
        if not raw_data_dict or not raw_data_dict.get("values"):
            raise ValueError(f"No data available for sensor {sensor_id}")

        raw_data = np.array(raw_data_dict["values"])

        # Handle missing values
        raw_data = self._impute_missing(raw_data)

        # Use processing service to prepare training data
        prepared_data = self.processing_service.prepare_training_data(
            raw_data, sensor_id, split_ratio, normalize, assess_quality
        )

        logger.info(
            f"Training data prepared for {sensor_id}: "
            f"train={len(prepared_data['train_data'])}, "
            f"val={len(prepared_data['val_data'])}, "
            f"test={len(prepared_data['test_data'])}"
        )

        return prepared_data

    def run_batch_pipeline(self, sensor_ids: List[str], **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Run pipeline for multiple sensors

        Args:
            sensor_ids: List of sensor identifiers
            **kwargs: Arguments passed to run_full_pipeline

        Returns:
            Dictionary mapping sensor_id to pipeline results
        """
        results = {}

        for sensor_id in sensor_ids:
            logger.info(f"Processing sensor {sensor_id}")
            try:
                sensor_results = self.run_full_pipeline(sensor_id, **kwargs)
                results[sensor_id] = sensor_results
            except Exception as e:
                logger.error(f"Failed to process sensor {sensor_id}: {e}")
                results[sensor_id] = {"success": False, "errors": [str(e)]}

        # Summary
        successful = sum(1 for r in results.values() if r.get("success"))
        logger.info(f"Batch pipeline complete: {successful}/{len(sensor_ids)} successful")

        return results

    def _impute_missing(self, data: np.ndarray) -> np.ndarray:
        """
        Impute missing values using forward fill

        Args:
            data: Data with potential missing values

        Returns:
            Data with imputed values
        """
        imputed = data.copy()
        missing_mask = ~np.isfinite(imputed)

        if not np.any(missing_mask):
            return imputed

        # Forward fill
        last_valid = None
        for i in range(len(imputed)):
            if missing_mask[i]:
                if last_valid is not None:
                    imputed[i] = last_valid
                else:
                    # No valid value yet, use 0
                    imputed[i] = 0.0
            else:
                last_valid = imputed[i]

        return imputed

    def get_pipeline_run(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get results from a specific pipeline run"""
        return self.pipeline_runs.get(pipeline_id)

    def list_pipeline_runs(self, sensor_id: Optional[str] = None) -> List[str]:
        """
        List all pipeline runs

        Args:
            sensor_id: Filter by sensor ID (all if None)

        Returns:
            List of pipeline IDs
        """
        if sensor_id:
            return [pid for pid, results in self.pipeline_runs.items() if results.get("sensor_id") == sensor_id]
        else:
            return list(self.pipeline_runs.keys())

    def get_sensor_latest_pipeline(self, sensor_id: str) -> Optional[Dict[str, Any]]:
        """Get latest successful pipeline run for a sensor"""
        sensor_runs = [
            results
            for results in self.pipeline_runs.values()
            if results.get("sensor_id") == sensor_id and results.get("success")
        ]

        if not sensor_runs:
            return None

        # Sort by start_time and get latest
        sensor_runs.sort(key=lambda x: x.get("start_time", datetime.min), reverse=True)
        return sensor_runs[0]

    def clear_cache(self):
        """Clear all cached data (normalization params, reference distributions)"""
        self.processing_service.clear_cache()
        self.drift_detector.clear_reference()
        logger.info("Cleared all pipeline caches")

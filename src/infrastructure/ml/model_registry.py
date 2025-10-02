"""
Model Registry and Versioning System
Manages trained models, versions, and metadata
"""

import hashlib
import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Model metadata structure"""

    model_id: str
    sensor_id: str
    model_type: str  # 'telemanom' or 'transformer'
    version: str
    created_at: str
    training_config: Dict[str, Any]
    training_metrics: Dict[str, Any]
    validation_metrics: Dict[str, Any]
    model_size_bytes: int
    training_time_seconds: float
    data_hash: str
    model_hash: str
    description: str = ""
    tags: List[str] = None
    is_active: bool = True
    performance_score: float = 0.0

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ModelRegistry:
    """
    Central registry for managing trained models and their versions
    """

    def __init__(self, registry_path: str = "./models/registry"):
        """
        Initialize model registry

        Args:
            registry_path: Path to registry directory
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        # Registry files
        self.models_index_file = self.registry_path / "models_index.json"
        self.versions_index_file = self.registry_path / "versions_index.json"
        self.metadata_dir = self.registry_path / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)

        # Initialize indices
        self._initialize_indices()

        logger.info(f"Model registry initialized at {self.registry_path}")

    def _initialize_indices(self):
        """Initialize or load registry indices"""
        # Models index: {model_id: {sensor_id, model_type, active_version, versions}}
        if self.models_index_file.exists():
            with open(self.models_index_file, "r") as f:
                self.models_index = json.load(f)
        else:
            self.models_index = {}

        # Versions index: {version_id: metadata_path}
        if self.versions_index_file.exists():
            with open(self.versions_index_file, "r") as f:
                self.versions_index = json.load(f)
        else:
            self.versions_index = {}

    def _save_indices(self):
        """Save registry indices to disk"""
        with open(self.models_index_file, "w") as f:
            json.dump(self.models_index, f, indent=2)

        with open(self.versions_index_file, "w") as f:
            json.dump(self.versions_index, f, indent=2)

    def _generate_model_id(self, sensor_id: str, model_type: str) -> str:
        """Generate unique model ID"""
        return f"{model_type}_{sensor_id}"

    def _generate_version_id(self, model_id: str, timestamp: str) -> str:
        """Generate unique version ID"""
        hash_input = f"{model_id}_{timestamp}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:8]

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate hash of a file"""
        try:
            with open(file_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""

    def _calculate_model_size(self, model_dir: Path) -> int:
        """Calculate total size of model files"""
        total_size = 0
        try:
            for file_path in model_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception:
            pass
        return total_size

    def register_model(
        self,
        sensor_id: str,
        model_type: str,
        model_path: Path,
        training_config: Dict[str, Any],
        training_metrics: Dict[str, Any],
        validation_metrics: Dict[str, Any],
        training_time_seconds: float,
        description: str = "",
        tags: List[str] = None,
    ) -> str:
        """
        Register a new model version

        Args:
            sensor_id: Equipment sensor ID
            model_type: Type of model ('telemanom' or 'transformer')
            model_path: Path to trained model directory
            training_config: Configuration used for training
            training_metrics: Training performance metrics
            validation_metrics: Validation performance metrics
            training_time_seconds: Time taken for training
            description: Model description
            tags: Model tags

        Returns:
            Version ID of registered model
        """
        try:
            # Generate model ID and metadata
            model_id = self._generate_model_id(sensor_id, model_type)
            timestamp = datetime.now().isoformat()
            version_id = self._generate_version_id(model_id, timestamp)

            # Calculate model hash and size
            model_hash = ""
            model_size = 0
            if model_path.exists():
                model_size = self._calculate_model_size(model_path)
                # Use main model file for hash (e.g., model.h5 or model.pkl)
                main_files = list(model_path.glob("*.h5")) + list(model_path.glob("*.pkl"))
                if main_files:
                    model_hash = self._calculate_file_hash(main_files[0])

            # Calculate data hash (placeholder - should be based on training data)
            data_hash = hashlib.md5(f"{sensor_id}_{timestamp}".encode()).hexdigest()[:8]

            # Calculate performance score
            performance_score = self._calculate_performance_score(validation_metrics)

            # Create metadata
            metadata = ModelMetadata(
                model_id=model_id,
                sensor_id=sensor_id,
                model_type=model_type,
                version=version_id,
                created_at=timestamp,
                training_config=training_config,
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                model_size_bytes=model_size,
                training_time_seconds=training_time_seconds,
                data_hash=data_hash,
                model_hash=model_hash,
                description=description,
                tags=tags or [],
                performance_score=performance_score,
            )

            # Save metadata
            metadata_file = self.metadata_dir / f"{version_id}.json"
            with open(metadata_file, "w") as f:
                json.dump(asdict(metadata), f, indent=2)

            # Update models index
            if model_id not in self.models_index:
                self.models_index[model_id] = {
                    "sensor_id": sensor_id,
                    "model_type": model_type,
                    "active_version": version_id,
                    "versions": [],
                }

            self.models_index[model_id]["versions"].append(
                {
                    "version_id": version_id,
                    "created_at": timestamp,
                    "performance_score": performance_score,
                    "is_active": True,
                }
            )

            # Update active version if this one is better
            current_active = self.models_index[model_id]["active_version"]
            if current_active != version_id:
                current_metadata = self.get_model_metadata(current_active)
                if current_metadata and performance_score > current_metadata.performance_score:
                    self.models_index[model_id]["active_version"] = version_id
                    # Mark old version as inactive
                    self._set_version_active_status(current_active, False)

            # Update versions index
            self.versions_index[version_id] = str(metadata_file)

            # Save indices
            self._save_indices()

            logger.info(f"Model registered: {model_id} v{version_id} (score: {performance_score:.4f})")
            return version_id

        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise

    def _calculate_performance_score(self, validation_metrics: Dict[str, Any]) -> float:
        """Calculate overall performance score"""
        try:
            if not validation_metrics or not validation_metrics.get("validation_performed", False):
                return 0.0

            # For anomaly detection (telemanom)
            if "anomaly_rate" in validation_metrics:
                anomaly_rate = validation_metrics.get("anomaly_rate", 0.5)
                # Good score for reasonable anomaly rate (not too high, not zero)
                if 0.01 <= anomaly_rate <= 0.1:
                    return 0.8 + (0.1 - anomaly_rate) * 2  # 0.8-1.0 range
                elif anomaly_rate == 0:
                    return 0.3  # Too conservative
                else:
                    return max(0.1, 1.0 - anomaly_rate)  # Too sensitive

            # For forecasting (transformer)
            elif "r2_score" in validation_metrics:
                r2 = validation_metrics.get("r2_score", 0)
                mape = validation_metrics.get("mape", 100)

                # Combine R² and MAPE for score
                r2_score = max(0, min(1, r2))  # Clip to [0, 1]
                mape_score = max(0, 1 - (mape / 100))  # Convert MAPE to score

                return r2_score * 0.7 + mape_score * 0.3  # Weighted combination

            return 0.5  # Default score

        except Exception:
            return 0.0

    def _set_version_active_status(self, version_id: str, is_active: bool):
        """Set active status for a version"""
        try:
            metadata = self.get_model_metadata(version_id)
            if metadata:
                metadata.is_active = is_active
                metadata_file = self.metadata_dir / f"{version_id}.json"
                with open(metadata_file, "w") as f:
                    json.dump(asdict(metadata), f, indent=2)
        except Exception as e:
            logger.error(f"Error setting version active status: {e}")

    def get_model_metadata(self, version_id: str) -> Optional[ModelMetadata]:
        """Get metadata for specific model version"""
        try:
            if version_id not in self.versions_index:
                return None

            metadata_file = Path(self.versions_index[version_id])
            if not metadata_file.exists():
                return None

            with open(metadata_file, "r") as f:
                data = json.load(f)

            return ModelMetadata(**data)

        except Exception as e:
            logger.error(f"Error loading model metadata: {e}")
            return None

    def get_active_model_version(self, sensor_id: str, model_type: str) -> Optional[str]:
        """Get active version for a model"""
        model_id = self._generate_model_id(sensor_id, model_type)
        return self.models_index.get(model_id, {}).get("active_version")

    def list_models(self, model_type: str = None) -> List[Dict[str, Any]]:
        """List all registered models"""
        models = []
        for model_id, info in self.models_index.items():
            if model_type is None or info["model_type"] == model_type:
                models.append(
                    {
                        "model_id": model_id,
                        "sensor_id": info["sensor_id"],
                        "model_type": info["model_type"],
                        "active_version": info["active_version"],
                        "total_versions": len(info["versions"]),
                    }
                )
        return models

    def list_versions(self, sensor_id: str, model_type: str) -> List[Dict[str, Any]]:
        """List all versions for a specific model"""
        model_id = self._generate_model_id(sensor_id, model_type)
        if model_id not in self.models_index:
            return []

        versions = []
        for version_info in self.models_index[model_id]["versions"]:
            metadata = self.get_model_metadata(version_info["version_id"])
            if metadata:
                versions.append(
                    {
                        "version_id": version_info["version_id"],
                        "created_at": version_info["created_at"],
                        "performance_score": version_info["performance_score"],
                        "is_active": metadata.is_active,
                        "model_size_mb": metadata.model_size_bytes / (1024 * 1024),
                        "training_time_seconds": metadata.training_time_seconds,
                    }
                )

        return sorted(versions, key=lambda x: x["created_at"], reverse=True)

    def promote_version(self, version_id: str) -> bool:
        """Promote a version to be the active version"""
        try:
            metadata = self.get_model_metadata(version_id)
            if not metadata:
                return False

            model_id = metadata.model_id

            # Update active version
            old_active = self.models_index[model_id]["active_version"]
            self.models_index[model_id]["active_version"] = version_id

            # Update active status
            self._set_version_active_status(old_active, False)
            self._set_version_active_status(version_id, True)

            # Save indices
            self._save_indices()

            logger.info(f"Promoted version {version_id} to active for model {model_id}")
            return True

        except Exception as e:
            logger.error(f"Error promoting version: {e}")
            return False

    def delete_version(self, version_id: str, force: bool = False) -> bool:
        """Delete a model version"""
        try:
            metadata = self.get_model_metadata(version_id)
            if not metadata:
                return False

            # Don't delete active version unless forced
            if metadata.is_active and not force:
                logger.warning(f"Cannot delete active version {version_id} without force=True")
                return False

            # Remove from indices
            model_id = metadata.model_id
            if model_id in self.models_index:
                self.models_index[model_id]["versions"] = [
                    v for v in self.models_index[model_id]["versions"] if v["version_id"] != version_id
                ]

            if version_id in self.versions_index:
                del self.versions_index[version_id]

            # Delete metadata file
            metadata_file = self.metadata_dir / f"{version_id}.json"
            if metadata_file.exists():
                metadata_file.unlink()

            # Save indices
            self._save_indices()

            logger.info(f"Deleted version {version_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting version: {e}")
            return False

    def cleanup_old_versions(self, keep_last_n: int = 3) -> int:
        """Clean up old model versions, keeping only the last N versions"""
        cleaned = 0
        try:
            for model_id, info in self.models_index.items():
                versions = sorted(info["versions"], key=lambda x: x["created_at"], reverse=True)

                # Keep active version and last N versions
                active_version = info["active_version"]
                versions_to_keep = set([active_version])

                # Add last N versions
                for version in versions[:keep_last_n]:
                    versions_to_keep.add(version["version_id"])

                # Delete older versions
                for version in versions[keep_last_n:]:
                    if version["version_id"] not in versions_to_keep:
                        if self.delete_version(version["version_id"], force=False):
                            cleaned += 1

            logger.info(f"Cleaned up {cleaned} old model versions")
            return cleaned

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return cleaned

    def is_model_available(self, sensor_id: str, model_type: str) -> bool:
        """Check if a model is available for given sensor and type"""
        try:
            active_version = self.get_active_model_version(sensor_id, model_type)
            if not active_version:
                return False

            # Check if metadata exists
            metadata = self.get_model_metadata(active_version)
            return metadata is not None and metadata.is_active
        except Exception as e:
            logger.error(f"Error checking model availability for {sensor_id} ({model_type}): {e}")
            return False

    def get_model_availability_report(self) -> Dict[str, Any]:
        """Get comprehensive model availability report"""
        try:
            from config.equipment_config import EQUIPMENT_REGISTRY

            availability_report = {
                "telemanom_models": {},
                "transformer_models": {},
                "availability_summary": {
                    "total_sensors": len(EQUIPMENT_REGISTRY),
                    "telemanom_available": 0,
                    "transformer_available": 0,
                    "both_available": 0,
                    "none_available": 0,
                },
            }

            for sensor_id in EQUIPMENT_REGISTRY.keys():
                # Check Telemanom availability
                telemanom_available = self.is_model_available(sensor_id, "telemanom")
                transformer_available = self.is_model_available(sensor_id, "transformer")

                availability_report["telemanom_models"][sensor_id] = {
                    "available": telemanom_available,
                    "active_version": (
                        self.get_active_model_version(sensor_id, "telemanom") if telemanom_available else None
                    ),
                }

                availability_report["transformer_models"][sensor_id] = {
                    "available": transformer_available,
                    "active_version": (
                        self.get_active_model_version(sensor_id, "transformer") if transformer_available else None
                    ),
                }

                # Update summary counts
                if telemanom_available:
                    availability_report["availability_summary"]["telemanom_available"] += 1
                if transformer_available:
                    availability_report["availability_summary"]["transformer_available"] += 1
                if telemanom_available and transformer_available:
                    availability_report["availability_summary"]["both_available"] += 1
                if not telemanom_available and not transformer_available:
                    availability_report["availability_summary"]["none_available"] += 1

            availability_report["availability_summary"]["coverage_percentage"] = (
                (availability_report["availability_summary"]["both_available"] / len(EQUIPMENT_REGISTRY)) * 100
                if EQUIPMENT_REGISTRY
                else 0
            )

            return availability_report

        except Exception as e:
            logger.error(f"Error generating model availability report: {e}")
            return {
                "error": str(e),
                "availability_summary": {
                    "total_sensors": 0,
                    "telemanom_available": 0,
                    "transformer_available": 0,
                    "both_available": 0,
                    "none_available": 0,
                    "coverage_percentage": 0,
                },
            }

    def get_model_health_status(self, sensor_id: str, model_type: str) -> Dict[str, Any]:
        """Get detailed health status for a specific model"""
        try:
            active_version = self.get_active_model_version(sensor_id, model_type)
            if not active_version:
                return {
                    "status": "not_available",
                    "message": f"No {model_type} model available for sensor {sensor_id}",
                }

            metadata = self.get_model_metadata(active_version)
            if not metadata:
                return {
                    "status": "metadata_missing",
                    "message": f"Model metadata missing for {sensor_id} {model_type}",
                }

            # Check model file existence
            model_path = None
            if model_type == "telemanom":
                model_path = Path(f"data/models/telemanom/{sensor_id}")
            elif model_type == "transformer":
                model_path = Path(f"data/models/transformer/{sensor_id}")

            model_files_exist = False
            if model_path and model_path.exists():
                if model_type == "telemanom":
                    model_files_exist = (model_path / "metadata.json").exists() and (model_path / "scaler.pkl").exists()
                elif model_type == "transformer":
                    model_files_exist = (model_path / "transformer_metadata.json").exists() and (
                        model_path / "scaler.pkl"
                    ).exists()

            # Determine overall health
            if metadata.is_active and model_files_exist and metadata.performance_score > 0.5:
                status = "healthy"
                message = f"Model is healthy and ready for inference"
            elif metadata.is_active and model_files_exist:
                status = "available_low_performance"
                message = f"Model available but performance score is {metadata.performance_score:.3f}"
            elif metadata.is_active and not model_files_exist:
                status = "metadata_only"
                message = f"Model registered but files missing"
            else:
                status = "inactive"
                message = f"Model is inactive or has issues"

            return {
                "status": status,
                "message": message,
                "version_id": active_version,
                "performance_score": metadata.performance_score,
                "model_size_mb": metadata.model_size_bytes / (1024 * 1024),
                "created_at": metadata.created_at,
                "files_exist": model_files_exist,
                "is_active": metadata.is_active,
            }

        except Exception as e:
            logger.error(f"Error checking model health for {sensor_id} ({model_type}): {e}")
            return {"status": "error", "message": f"Error checking model health: {e}"}

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        try:
            total_models = len(self.models_index)
            total_versions = len(self.versions_index)

            model_types = {}
            total_size = 0

            for version_id in self.versions_index:
                metadata = self.get_model_metadata(version_id)
                if metadata:
                    model_types[metadata.model_type] = model_types.get(metadata.model_type, 0) + 1
                    total_size += metadata.model_size_bytes

            return {
                "total_models": total_models,
                "total_versions": total_versions,
                "model_types": model_types,
                "total_size_mb": total_size / (1024 * 1024),
                "registry_path": str(self.registry_path),
            }

        except Exception as e:
            logger.error(f"Error getting registry stats: {e}")
            return {}

"""
DVC Manager
Data versioning and lineage tracking using DVC (Data Version Control)
"""

import subprocess
import hashlib
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import logging
import shutil

logger = logging.getLogger(__name__)


@dataclass
class DatasetVersion:
    """Dataset version metadata"""
    dataset_id: str
    version: str
    data_hash: str
    file_path: str
    creation_date: datetime
    size_bytes: int
    num_samples: int
    sensor_ids: List[str]
    description: str
    tags: List[str]
    parent_version: Optional[str] = None
    dvc_tracked: bool = False
    remote_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['creation_date'] = self.creation_date.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetVersion':
        """Create from dictionary"""
        data_copy = data.copy()
        data_copy['creation_date'] = datetime.fromisoformat(data_copy['creation_date'])
        return cls(**data_copy)


class DVCManager:
    """
    DVC integration for data versioning and lineage tracking
    """

    def __init__(
        self,
        repo_root: str = ".",
        data_dir: str = "data",
        dvc_remote: Optional[str] = None
    ):
        """
        Initialize DVC manager

        Args:
            repo_root: Git repository root
            data_dir: Data directory (relative to repo root)
            dvc_remote: DVC remote name (e.g., 's3://bucket/path')
        """
        self.repo_root = Path(repo_root)
        self.data_dir = self.repo_root / data_dir
        self.dvc_remote = dvc_remote

        # Metadata directory
        self.metadata_dir = self.data_dir / ".dvc_metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Version registry
        self.registry_file = self.metadata_dir / "dataset_registry.json"
        self.versions: Dict[str, List[DatasetVersion]] = {}

        # Check DVC availability
        self.dvc_available = self._check_dvc_installed()

        # Load existing registry
        self._load_registry()

    def _check_dvc_installed(self) -> bool:
        """Check if DVC is installed and initialized"""
        try:
            result = subprocess.run(
                ['dvc', 'version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"DVC is available: {result.stdout.strip()}")
                return True
            else:
                logger.warning("DVC is not installed or not in PATH")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"DVC not available: {e}")
            return False

    def initialize_dvc(self) -> bool:
        """
        Initialize DVC in the repository

        Returns:
            True if successful, False otherwise
        """
        if not self.dvc_available:
            logger.error("DVC is not installed. Install with: pip install dvc")
            return False

        try:
            # Check if already initialized
            dvc_dir = self.repo_root / ".dvc"
            if dvc_dir.exists():
                logger.info("DVC already initialized")
                return True

            # Initialize DVC
            result = subprocess.run(
                ['dvc', 'init'],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                logger.info("DVC initialized successfully")

                # Configure remote if provided
                if self.dvc_remote:
                    self.configure_remote(self.dvc_remote)

                return True
            else:
                logger.error(f"DVC init failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error initializing DVC: {e}")
            return False

    def configure_remote(self, remote_url: str, remote_name: str = "storage") -> bool:
        """
        Configure DVC remote storage

        Args:
            remote_url: Remote storage URL (s3://, gs://, azure://, or local path)
            remote_name: Remote name

        Returns:
            True if successful
        """
        if not self.dvc_available:
            logger.error("DVC is not available")
            return False

        try:
            # Add remote
            subprocess.run(
                ['dvc', 'remote', 'add', '-f', remote_name, remote_url],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=10
            )

            # Set as default
            subprocess.run(
                ['dvc', 'remote', 'default', remote_name],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=10
            )

            logger.info(f"DVC remote '{remote_name}' configured: {remote_url}")
            self.dvc_remote = remote_url
            return True

        except Exception as e:
            logger.error(f"Error configuring DVC remote: {e}")
            return False

    def _load_registry(self):
        """Load dataset version registry"""
        try:
            if self.registry_file.exists():
                with open(self.registry_file, 'r') as f:
                    registry_data = json.load(f)
                    for dataset_id, versions in registry_data.items():
                        self.versions[dataset_id] = [
                            DatasetVersion.from_dict(v) for v in versions
                        ]
                logger.info(f"Loaded {len(self.versions)} dataset registries")
        except Exception as e:
            logger.warning(f"Could not load dataset registry: {e}")

    def _save_registry(self):
        """Save dataset version registry"""
        try:
            registry_data = {
                dataset_id: [v.to_dict() for v in versions]
                for dataset_id, versions in self.versions.items()
            }
            with open(self.registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
            logger.debug("Saved dataset registry")
        except Exception as e:
            logger.error(f"Error saving dataset registry: {e}")

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def version_dataset(
        self,
        file_path: str,
        dataset_id: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        sensor_ids: Optional[List[str]] = None,
        parent_version: Optional[str] = None,
        push_to_remote: bool = False
    ) -> Optional[DatasetVersion]:
        """
        Version a dataset file using DVC

        Args:
            file_path: Path to dataset file
            dataset_id: Unique dataset identifier
            description: Dataset description
            tags: Optional tags
            sensor_ids: List of sensor IDs in dataset
            parent_version: Parent version ID for lineage
            push_to_remote: Whether to push to DVC remote

        Returns:
            DatasetVersion object or None if failed
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"Dataset file not found: {file_path}")
            return None

        try:
            # Compute file hash
            data_hash = self._compute_file_hash(file_path)

            # Check if this exact version already exists
            if dataset_id in self.versions:
                existing = [v for v in self.versions[dataset_id] if v.data_hash == data_hash]
                if existing:
                    logger.info(f"Dataset {dataset_id} with hash {data_hash[:8]} already versioned")
                    return existing[0]

            # Get file stats
            file_stats = file_path.stat()
            size_bytes = file_stats.st_size

            # Count samples (assuming .npy or .csv)
            num_samples = self._count_samples(file_path)

            # Generate version number
            version = self._generate_version_number(dataset_id)

            # Track with DVC if available
            dvc_tracked = False
            if self.dvc_available:
                dvc_tracked = self._dvc_add(file_path, push_to_remote)

            # Create version metadata
            dataset_version = DatasetVersion(
                dataset_id=dataset_id,
                version=version,
                data_hash=data_hash,
                file_path=str(file_path.relative_to(self.repo_root)),
                creation_date=datetime.now(),
                size_bytes=size_bytes,
                num_samples=num_samples,
                sensor_ids=sensor_ids or [],
                description=description,
                tags=tags or [],
                parent_version=parent_version,
                dvc_tracked=dvc_tracked,
                remote_url=self.dvc_remote if dvc_tracked else None
            )

            # Add to registry
            if dataset_id not in self.versions:
                self.versions[dataset_id] = []
            self.versions[dataset_id].append(dataset_version)

            # Save registry
            self._save_registry()

            logger.info(
                f"Versioned dataset {dataset_id} v{version} "
                f"({size_bytes} bytes, {num_samples} samples, hash: {data_hash[:8]})"
            )

            return dataset_version

        except Exception as e:
            logger.error(f"Error versioning dataset: {e}")
            return None

    def _dvc_add(self, file_path: Path, push_to_remote: bool = False) -> bool:
        """
        Add file to DVC tracking

        Args:
            file_path: File to track
            push_to_remote: Whether to push to remote

        Returns:
            True if successful
        """
        try:
            # Add file to DVC
            result = subprocess.run(
                ['dvc', 'add', str(file_path)],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                logger.warning(f"DVC add failed: {result.stderr}")
                return False

            logger.info(f"Added {file_path} to DVC tracking")

            # Push to remote if requested
            if push_to_remote and self.dvc_remote:
                push_result = subprocess.run(
                    ['dvc', 'push', str(file_path) + '.dvc'],
                    cwd=self.repo_root,
                    capture_output=True,
                    text=True,
                    timeout=300
                )

                if push_result.returncode == 0:
                    logger.info(f"Pushed {file_path} to DVC remote")
                else:
                    logger.warning(f"DVC push failed: {push_result.stderr}")

            return True

        except Exception as e:
            logger.error(f"Error in DVC add: {e}")
            return False

    def _count_samples(self, file_path: Path) -> int:
        """Count number of samples in dataset file"""
        try:
            import numpy as np

            if file_path.suffix == '.npy':
                data = np.load(file_path)
                return len(data)
            elif file_path.suffix == '.csv':
                import pandas as pd
                df = pd.read_csv(file_path)
                return len(df)
            else:
                return 0
        except Exception as e:
            logger.warning(f"Could not count samples in {file_path}: {e}")
            return 0

    def _generate_version_number(self, dataset_id: str) -> str:
        """Generate next version number for dataset"""
        if dataset_id not in self.versions or not self.versions[dataset_id]:
            return "v1"

        # Get latest version number
        latest_version = self.versions[dataset_id][-1].version
        try:
            # Extract number from vN format
            version_num = int(latest_version.replace('v', ''))
            return f"v{version_num + 1}"
        except ValueError:
            return f"v{len(self.versions[dataset_id]) + 1}"

    def get_dataset_version(
        self,
        dataset_id: str,
        version: Optional[str] = None
    ) -> Optional[DatasetVersion]:
        """
        Get specific dataset version (latest if version not specified)

        Args:
            dataset_id: Dataset identifier
            version: Version number (e.g., 'v1', 'v2')

        Returns:
            DatasetVersion or None
        """
        if dataset_id not in self.versions:
            return None

        if version is None:
            # Return latest version
            return self.versions[dataset_id][-1]

        # Find specific version
        for v in self.versions[dataset_id]:
            if v.version == version:
                return v

        return None

    def get_dataset_lineage(self, dataset_id: str, version: str) -> List[DatasetVersion]:
        """
        Get full lineage (ancestry chain) of a dataset version

        Args:
            dataset_id: Dataset identifier
            version: Version number

        Returns:
            List of DatasetVersion objects in chronological order
        """
        lineage = []

        current = self.get_dataset_version(dataset_id, version)
        while current:
            lineage.insert(0, current)

            if current.parent_version:
                current = self.get_dataset_version(dataset_id, current.parent_version)
            else:
                break

        return lineage

    def list_datasets(self) -> List[str]:
        """Get list of all tracked dataset IDs"""
        return list(self.versions.keys())

    def list_versions(self, dataset_id: str) -> List[DatasetVersion]:
        """Get all versions of a dataset"""
        return self.versions.get(dataset_id, [])

    def pull_dataset(self, dataset_id: str, version: Optional[str] = None) -> bool:
        """
        Pull dataset from DVC remote

        Args:
            dataset_id: Dataset identifier
            version: Version to pull (latest if None)

        Returns:
            True if successful
        """
        dataset_version = self.get_dataset_version(dataset_id, version)
        if not dataset_version:
            logger.error(f"Dataset {dataset_id} version {version} not found")
            return False

        if not dataset_version.dvc_tracked:
            logger.warning(f"Dataset {dataset_id} v{dataset_version.version} is not DVC tracked")
            return False

        if not self.dvc_available:
            logger.error("DVC is not available")
            return False

        try:
            dvc_file = Path(dataset_version.file_path).with_suffix('.dvc')

            result = subprocess.run(
                ['dvc', 'pull', str(dvc_file)],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                logger.info(f"Pulled {dataset_id} v{dataset_version.version}")
                return True
            else:
                logger.error(f"DVC pull failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error pulling dataset: {e}")
            return False

    def link_dataset_to_model(
        self,
        dataset_id: str,
        dataset_version: str,
        model_id: str,
        model_version: str
    ):
        """
        Create linkage between dataset version and model version

        Args:
            dataset_id: Dataset identifier
            dataset_version: Dataset version
            model_id: Model identifier
            model_version: Model version
        """
        # Store linkage in metadata
        linkage_file = self.metadata_dir / f"linkage_{model_id}_{model_version}.json"

        linkage = {
            'model_id': model_id,
            'model_version': model_version,
            'dataset_id': dataset_id,
            'dataset_version': dataset_version,
            'created_at': datetime.now().isoformat()
        }

        with open(linkage_file, 'w') as f:
            json.dump(linkage, f, indent=2)

        logger.info(
            f"Linked model {model_id} v{model_version} to "
            f"dataset {dataset_id} v{dataset_version}"
        )

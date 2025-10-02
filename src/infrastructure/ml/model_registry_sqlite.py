"""
Model Registry with SQLite Backend
Provides ACID properties, concurrency support, and robust data management
"""

import sqlite3
import json
import shutil
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from threading import Lock

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
    data_hash: str  # Hash of actual training data
    model_hash: str
    model_path: str  # Centralized path management
    description: str = ""
    tags: str = ""  # JSON string of tags
    is_active: bool = True
    performance_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with JSON serialization for nested fields"""
        data = asdict(self)
        # Convert dict fields to JSON strings for SQLite storage
        data['training_config'] = json.dumps(self.training_config)
        data['training_metrics'] = json.dumps(self.training_metrics)
        data['validation_metrics'] = json.dumps(self.validation_metrics)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary with JSON deserialization"""
        # Parse JSON strings back to dicts
        data['training_config'] = json.loads(data['training_config']) if isinstance(data['training_config'], str) else data['training_config']
        data['training_metrics'] = json.loads(data['training_metrics']) if isinstance(data['training_metrics'], str) else data['training_metrics']
        data['validation_metrics'] = json.loads(data['validation_metrics']) if isinstance(data['validation_metrics'], str) else data['validation_metrics']
        return cls(**data)


class ModelRegistrySQLite:
    """
    SQLite-based model registry with ACID properties and concurrency support
    """

    def __init__(
        self,
        registry_path: str = "./models/registry",
        performance_score_weights: Dict[str, float] = None
    ):
        """
        Initialize SQLite model registry

        Args:
            registry_path: Path to registry directory
            performance_score_weights: Custom weights for performance scoring
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        # SQLite database file
        self.db_path = self.registry_path / "model_registry.db"

        # Thread safety
        self._lock = Lock()

        # Performance scoring configuration
        self.performance_score_weights = performance_score_weights or {
            'r2_weight': 0.7,
            'mape_weight': 0.3,
            'anomaly_min': 0.01,
            'anomaly_max': 0.1
        }

        # Initialize database
        self._initialize_database()

        logger.info(f"SQLite Model Registry initialized at {self.registry_path}")

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections with thread safety"""
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            try:
                yield conn
            finally:
                conn.close()

    def _initialize_database(self):
        """Initialize SQLite database schema"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Models table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    sensor_id TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    active_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(sensor_id, model_type)
                )
            """)

            # Versions table with all metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS versions (
                    version_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    sensor_id TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    training_config TEXT NOT NULL,
                    training_metrics TEXT NOT NULL,
                    validation_metrics TEXT NOT NULL,
                    model_size_bytes INTEGER NOT NULL,
                    training_time_seconds REAL NOT NULL,
                    data_hash TEXT NOT NULL,
                    model_hash TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    tags TEXT DEFAULT '',
                    is_active INTEGER DEFAULT 1,
                    performance_score REAL DEFAULT 0.0,
                    FOREIGN KEY (model_id) REFERENCES models(model_id) ON DELETE CASCADE
                )
            """)

            # Create indices for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_versions_model_id
                ON versions(model_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_versions_sensor_id
                ON versions(sensor_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_versions_is_active
                ON versions(is_active)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_versions_performance
                ON versions(performance_score)
            """)

            # Data lineage table for tracking training data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_lineage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_id TEXT NOT NULL,
                    data_hash TEXT NOT NULL,
                    data_source TEXT,
                    data_start_date TEXT,
                    data_end_date TEXT,
                    num_samples INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (version_id) REFERENCES versions(version_id) ON DELETE CASCADE
                )
            """)

            conn.commit()
            logger.info("SQLite database schema initialized")

    def _generate_model_id(self, sensor_id: str, model_type: str) -> str:
        """Generate unique model ID"""
        return f"{model_type}_{sensor_id}"

    def _generate_version_id(self, model_id: str, timestamp: str) -> str:
        """Generate unique version ID"""
        hash_input = f"{model_id}_{timestamp}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate hash of a file"""
        try:
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    hasher.update(chunk)
            return hasher.hexdigest()[:16]
        except FileNotFoundError:
            logger.warning(f"File not found for hashing: {file_path}")
            return ""
        except Exception as e:
            logger.error(f"Error calculating file hash: {e}")
            return ""

    def _calculate_model_size(self, model_dir: Path) -> int:
        """Calculate total size of model files"""
        total_size = 0
        try:
            for file_path in model_dir.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            logger.error(f"Error calculating model size: {e}")
        return total_size

    def _calculate_performance_score(self, validation_metrics: Dict[str, Any]) -> float:
        """Calculate overall performance score with configurable weights

        Args:
            validation_metrics: Validation metrics dictionary

        Returns:
            float: Performance score between 0 and 1
        """
        try:
            if not validation_metrics:
                return 0.0

            # Check if validation was actually performed
            if not validation_metrics.get('validation_performed', False):
                logger.warning("Validation not performed - returning score 0.0")
                return 0.0

            # For anomaly detection (telemanom)
            if 'anomaly_rate' in validation_metrics:
                anomaly_rate = validation_metrics.get('anomaly_rate', 0.5)
                anomaly_min = self.performance_score_weights.get('anomaly_min', 0.01)
                anomaly_max = self.performance_score_weights.get('anomaly_max', 0.1)

                # Good score for reasonable anomaly rate
                if anomaly_min <= anomaly_rate <= anomaly_max:
                    return 0.8 + (anomaly_max - anomaly_rate) / (anomaly_max - anomaly_min) * 0.2
                elif anomaly_rate == 0:
                    return 0.3  # Too conservative
                else:
                    return max(0.1, 1.0 - anomaly_rate)  # Too sensitive

            # For forecasting (transformer)
            elif 'r2_score' in validation_metrics:
                r2 = validation_metrics.get('r2_score', 0)
                mape = validation_metrics.get('mape', 100)

                # Combine R² and MAPE with configurable weights
                r2_score = max(0, min(1, r2))  # Clip to [0, 1]
                mape_score = max(0, 1 - (mape / 100))  # Convert MAPE to score

                r2_weight = self.performance_score_weights.get('r2_weight', 0.7)
                mape_weight = self.performance_score_weights.get('mape_weight', 0.3)

                return (r2_score * r2_weight + mape_score * mape_weight)

            return 0.5  # Default score

        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 0.0

    def register_model(
        self,
        sensor_id: str,
        model_type: str,
        model_path: Path,
        training_config: Dict[str, Any],
        training_metrics: Dict[str, Any],
        validation_metrics: Dict[str, Any],
        training_time_seconds: float,
        data_hash: str = None,  # Accept from training pipeline
        data_source: str = None,
        data_start_date: str = None,
        data_end_date: str = None,
        num_samples: int = None,
        description: str = "",
        tags: List[str] = None
    ) -> str:
        """
        Register a new model version with proper data lineage

        Args:
            sensor_id: Equipment sensor ID
            model_type: Type of model ('telemanom' or 'transformer')
            model_path: Path to trained model directory
            training_config: Configuration used for training
            training_metrics: Training performance metrics
            validation_metrics: Validation performance metrics (must include validation_performed=True)
            training_time_seconds: Time taken for training
            data_hash: Hash of training data (from training pipeline)
            data_source: Source of training data
            data_start_date: Start date of training data
            data_end_date: End date of training data
            num_samples: Number of training samples
            description: Model description
            tags: Model tags

        Returns:
            Version ID of registered model

        Raises:
            ValueError: If validation metrics don't include validation_performed flag
        """
        try:
            # Enforce proper validation metrics
            if not validation_metrics.get('validation_performed', False):
                raise ValueError(
                    "validation_metrics must include 'validation_performed': True. "
                    "Ensure model is validated on held-out data before registration."
                )

            # Generate IDs
            model_id = self._generate_model_id(sensor_id, model_type)
            timestamp = datetime.now().isoformat()
            version_id = self._generate_version_id(model_id, timestamp)

            # Calculate model hash and size
            model_hash = ""
            model_size = 0
            if model_path.exists():
                model_size = self._calculate_model_size(model_path)
                # Use main model file for hash
                main_files = list(model_path.glob("*.h5")) + list(model_path.glob("*.pkl"))
                if main_files:
                    model_hash = self._calculate_file_hash(main_files[0])

            # Use provided data_hash or generate placeholder
            if data_hash is None:
                logger.warning(
                    f"No data_hash provided for {sensor_id}. "
                    f"Consider passing data_hash from training pipeline for proper lineage tracking."
                )
                data_hash = hashlib.sha256(f"{sensor_id}_{timestamp}".encode()).hexdigest()[:16]

            # Calculate performance score
            performance_score = self._calculate_performance_score(validation_metrics)

            # Centralized model path
            model_path_str = str(model_path.absolute())

            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Insert or update model entry
                cursor.execute("""
                    INSERT INTO models (model_id, sensor_id, model_type, active_version, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(model_id) DO UPDATE SET
                        updated_at = excluded.updated_at
                """, (model_id, sensor_id, model_type, version_id, timestamp))

                # Insert version
                cursor.execute("""
                    INSERT INTO versions (
                        version_id, model_id, sensor_id, model_type, created_at,
                        training_config, training_metrics, validation_metrics,
                        model_size_bytes, training_time_seconds, data_hash, model_hash,
                        model_path, description, tags, is_active, performance_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    version_id, model_id, sensor_id, model_type, timestamp,
                    json.dumps(training_config), json.dumps(training_metrics),
                    json.dumps(validation_metrics), model_size, training_time_seconds,
                    data_hash, model_hash, model_path_str, description,
                    json.dumps(tags or []), 1, performance_score
                ))

                # Insert data lineage if provided
                if any([data_source, data_start_date, data_end_date, num_samples]):
                    cursor.execute("""
                        INSERT INTO data_lineage (
                            version_id, data_hash, data_source, data_start_date,
                            data_end_date, num_samples
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        version_id, data_hash, data_source, data_start_date,
                        data_end_date, num_samples
                    ))

                # Check if this should be the new active version
                # Compare against ALL previous versions, not just current active
                cursor.execute("""
                    SELECT version_id, performance_score
                    FROM versions
                    WHERE model_id = ? AND version_id != ?
                    ORDER BY performance_score DESC
                    LIMIT 1
                """, (model_id, version_id))

                best_previous = cursor.fetchone()

                if best_previous and performance_score > best_previous['performance_score']:
                    # New version is better - make it active
                    cursor.execute("""
                        UPDATE versions SET is_active = 0 WHERE model_id = ? AND version_id != ?
                    """, (model_id, version_id))

                    cursor.execute("""
                        UPDATE models SET active_version = ? WHERE model_id = ?
                    """, (version_id, model_id))

                    logger.info(
                        f"New active version: {version_id} (score: {performance_score:.4f}) "
                        f"replaced {best_previous['version_id']} (score: {best_previous['performance_score']:.4f})"
                    )
                elif not best_previous:
                    # First version - make it active
                    cursor.execute("""
                        UPDATE models SET active_version = ? WHERE model_id = ?
                    """, (version_id, model_id))

                conn.commit()

            logger.info(
                f"Model registered: {model_id} v{version_id} (score: {performance_score:.4f}, "
                f"data_hash: {data_hash[:8]}...)"
            )
            return version_id

        except ValueError as e:
            logger.error(f"Validation error registering model: {e}")
            raise
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise

    def get_model_metadata(self, version_id: str) -> Optional[ModelMetadata]:
        """Get metadata for specific model version"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM versions WHERE version_id = ?
                """, (version_id,))

                row = cursor.fetchone()
                if not row:
                    return None

                return ModelMetadata.from_dict(dict(row))

        except Exception as e:
            logger.error(f"Error loading model metadata for {version_id}: {e}")
            return None

    def get_active_model_version(self, sensor_id: str, model_type: str) -> Optional[str]:
        """Get active version for a model"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                model_id = self._generate_model_id(sensor_id, model_type)

                cursor.execute("""
                    SELECT active_version FROM models WHERE model_id = ?
                """, (model_id,))

                row = cursor.fetchone()
                return row['active_version'] if row else None

        except Exception as e:
            logger.error(f"Error getting active version for {sensor_id} ({model_type}): {e}")
            return None

    def list_models(self, model_type: str = None) -> List[Dict[str, Any]]:
        """List all registered models"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                if model_type:
                    cursor.execute("""
                        SELECT m.*, COUNT(v.version_id) as total_versions
                        FROM models m
                        LEFT JOIN versions v ON m.model_id = v.model_id
                        WHERE m.model_type = ?
                        GROUP BY m.model_id
                    """, (model_type,))
                else:
                    cursor.execute("""
                        SELECT m.*, COUNT(v.version_id) as total_versions
                        FROM models m
                        LEFT JOIN versions v ON m.model_id = v.model_id
                        GROUP BY m.model_id
                    """)

                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def list_versions(self, sensor_id: str, model_type: str) -> List[Dict[str, Any]]:
        """List all versions for a specific model"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                model_id = self._generate_model_id(sensor_id, model_type)

                cursor.execute("""
                    SELECT
                        version_id, created_at, performance_score, is_active,
                        model_size_bytes, training_time_seconds
                    FROM versions
                    WHERE model_id = ?
                    ORDER BY created_at DESC
                """, (model_id,))

                versions = []
                for row in cursor.fetchall():
                    versions.append({
                        'version_id': row['version_id'],
                        'created_at': row['created_at'],
                        'performance_score': row['performance_score'],
                        'is_active': bool(row['is_active']),
                        'model_size_mb': row['model_size_bytes'] / (1024 * 1024),
                        'training_time_seconds': row['training_time_seconds']
                    })

                return versions

        except Exception as e:
            logger.error(f"Error listing versions for {sensor_id} ({model_type}): {e}")
            return []

    def promote_version(self, version_id: str) -> bool:
        """Promote a version to be the active version"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Get model_id for this version
                cursor.execute("""
                    SELECT model_id FROM versions WHERE version_id = ?
                """, (version_id,))

                row = cursor.fetchone()
                if not row:
                    logger.warning(f"Version {version_id} not found")
                    return False

                model_id = row['model_id']

                # Deactivate all versions for this model
                cursor.execute("""
                    UPDATE versions SET is_active = 0 WHERE model_id = ?
                """, (model_id,))

                # Activate the selected version
                cursor.execute("""
                    UPDATE versions SET is_active = 1 WHERE version_id = ?
                """, (version_id,))

                # Update active version in models table
                cursor.execute("""
                    UPDATE models SET active_version = ?, updated_at = ?
                    WHERE model_id = ?
                """, (version_id, datetime.now().isoformat(), model_id))

                conn.commit()
                logger.info(f"Promoted version {version_id} to active for model {model_id}")
                return True

        except Exception as e:
            logger.error(f"Error promoting version {version_id}: {e}")
            return False

    def delete_version(self, version_id: str, force: bool = False, delete_artifacts: bool = True) -> bool:
        """Delete a model version and optionally its artifacts

        Args:
            version_id: Version to delete
            force: Allow deletion of active version
            delete_artifacts: Also delete model files from disk

        Returns:
            bool: Success status
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Get version metadata
                cursor.execute("""
                    SELECT is_active, model_path FROM versions WHERE version_id = ?
                """, (version_id,))

                row = cursor.fetchone()
                if not row:
                    logger.warning(f"Version {version_id} not found")
                    return False

                # Check if active
                if row['is_active'] and not force:
                    logger.warning(f"Cannot delete active version {version_id} without force=True")
                    return False

                # Delete model artifacts from disk if requested
                if delete_artifacts and row['model_path']:
                    model_path = Path(row['model_path'])
                    if model_path.exists():
                        try:
                            shutil.rmtree(model_path)
                            logger.info(f"Deleted model artifacts at {model_path}")
                        except Exception as e:
                            logger.error(f"Error deleting model artifacts: {e}")

                # Delete from database (cascades to data_lineage)
                cursor.execute("""
                    DELETE FROM versions WHERE version_id = ?
                """, (version_id,))

                conn.commit()
                logger.info(f"Deleted version {version_id}")
                return True

        except Exception as e:
            logger.error(f"Error deleting version {version_id}: {e}")
            return False

    def cleanup_old_versions(self, keep_last_n: int = 3) -> int:
        """Clean up old model versions, keeping only the last N versions"""
        cleaned = 0
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Get all models
                cursor.execute("SELECT model_id, active_version FROM models")
                models = cursor.fetchall()

                for model in models:
                    model_id = model['model_id']
                    active_version = model['active_version']

                    # Get all versions sorted by date
                    cursor.execute("""
                        SELECT version_id, created_at
                        FROM versions
                        WHERE model_id = ?
                        ORDER BY created_at DESC
                    """, (model_id,))

                    versions = cursor.fetchall()

                    # Keep active + last N versions
                    versions_to_keep = set([active_version])
                    for i, version in enumerate(versions):
                        if i < keep_last_n:
                            versions_to_keep.add(version['version_id'])

                    # Delete older versions
                    for version in versions[keep_last_n:]:
                        if version['version_id'] not in versions_to_keep:
                            if self.delete_version(version['version_id'], force=False, delete_artifacts=True):
                                cleaned += 1

            logger.info(f"Cleaned up {cleaned} old model versions")
            return cleaned

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return cleaned

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Total models
                cursor.execute("SELECT COUNT(*) as count FROM models")
                total_models = cursor.fetchone()['count']

                # Total versions
                cursor.execute("SELECT COUNT(*) as count FROM versions")
                total_versions = cursor.fetchone()['count']

                # Model types distribution
                cursor.execute("""
                    SELECT model_type, COUNT(*) as count
                    FROM models
                    GROUP BY model_type
                """)
                model_types = {row['model_type']: row['count'] for row in cursor.fetchall()}

                # Total size
                cursor.execute("SELECT SUM(model_size_bytes) as total FROM versions")
                total_size = cursor.fetchone()['total'] or 0

                return {
                    'total_models': total_models,
                    'total_versions': total_versions,
                    'model_types': model_types,
                    'total_size_mb': total_size / (1024 * 1024),
                    'registry_path': str(self.registry_path),
                    'database_path': str(self.db_path)
                }

        except Exception as e:
            logger.error(f"Error getting registry stats: {e}")
            return {}

    def get_data_lineage(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Get data lineage information for a model version"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT * FROM data_lineage WHERE version_id = ?
                """, (version_id,))

                row = cursor.fetchone()
                return dict(row) if row else None

        except Exception as e:
            logger.error(f"Error getting data lineage for {version_id}: {e}")
            return None

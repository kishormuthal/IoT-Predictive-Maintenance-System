#!/usr/bin/env python3
"""
Migration Script: JSON Model Registry to SQLite
Migrates existing JSON-based model registry to SQLite backend
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.infrastructure.ml.model_registry_sqlite import (
    ModelMetadata,
    ModelRegistrySQLite,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RegistryMigration:
    """Handles migration from JSON to SQLite registry"""

    def __init__(self, old_registry_path: str, new_registry_path: str):
        """
        Initialize migration

        Args:
            old_registry_path: Path to old JSON registry
            new_registry_path: Path for new SQLite registry
        """
        self.old_path = Path(old_registry_path)
        self.new_path = Path(new_registry_path)

        # Create backup
        self.backup_path = self.old_path.parent / f"registry_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def backup_old_registry(self):
        """Create backup of old registry"""
        try:
            import shutil

            if self.old_path.exists():
                shutil.copytree(self.old_path, self.backup_path)
                logger.info(f"Created backup at {self.backup_path}")
                return True
            else:
                logger.warning(f"Old registry not found at {self.old_path}")
                return False

        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False

    def load_json_indices(self):
        """Load old JSON indices"""
        try:
            models_index_file = self.old_path / "models_index.json"
            versions_index_file = self.old_path / "versions_index.json"
            metadata_dir = self.old_path / "metadata"

            models_index = {}
            versions_index = {}

            if models_index_file.exists():
                with open(models_index_file, "r") as f:
                    models_index = json.load(f)
                logger.info(f"Loaded {len(models_index)} models from JSON")

            if versions_index_file.exists():
                with open(versions_index_file, "r") as f:
                    versions_index = json.load(f)
                logger.info(f"Loaded {len(versions_index)} versions from JSON")

            return models_index, versions_index, metadata_dir

        except Exception as e:
            logger.error(f"Error loading JSON indices: {e}")
            return {}, {}, None

    def migrate_to_sqlite(self, dry_run: bool = False):
        """
        Perform migration to SQLite

        Args:
            dry_run: If True, only report what would be migrated without making changes
        """
        logger.info("=" * 80)
        logger.info("STARTING MIGRATION: JSON Registry -> SQLite")
        logger.info("=" * 80)

        # Step 1: Backup
        logger.info("\n[Step 1/4] Creating backup...")
        if not dry_run:
            if not self.backup_old_registry():
                logger.error("Backup failed. Aborting migration.")
                return False

        # Step 2: Load JSON data
        logger.info("\n[Step 2/4] Loading JSON data...")
        models_index, versions_index, metadata_dir = self.load_json_indices()

        if not models_index:
            logger.warning("No models found in JSON registry. Nothing to migrate.")
            return True

        # Step 3: Initialize SQLite registry
        logger.info("\n[Step 3/4] Initializing SQLite registry...")
        if dry_run:
            logger.info("[DRY RUN] Would create SQLite registry at: {self.new_path}")
        else:
            sqlite_registry = ModelRegistrySQLite(registry_path=str(self.new_path))

        # Step 4: Migrate data
        logger.info("\n[Step 4/4] Migrating data...")
        migrated_count = 0
        failed_count = 0

        for model_id, model_info in models_index.items():
            logger.info(f"\nMigrating model: {model_id}")

            for version_info in model_info.get("versions", []):
                version_id = version_info["version_id"]

                try:
                    # Load version metadata
                    if metadata_dir:
                        metadata_file = metadata_dir / f"{version_id}.json"

                        if metadata_file.exists():
                            with open(metadata_file, "r") as f:
                                metadata = json.load(f)

                            if dry_run:
                                logger.info(f"  [DRY RUN] Would migrate version: {version_id}")
                                logger.info(f"    - Sensor: {metadata.get('sensor_id')}")
                                logger.info(f"    - Type: {metadata.get('model_type')}")
                                logger.info(f"    - Score: {metadata.get('performance_score', 0):.4f}")
                                migrated_count += 1
                            else:
                                # Add validation_performed flag if missing
                                validation_metrics = metadata.get("validation_metrics", {})
                                if "validation_performed" not in validation_metrics:
                                    logger.warning(
                                        f"    Adding 'validation_performed': False to {version_id} "
                                        f"(original metadata missing validation flag)"
                                    )
                                    validation_metrics["validation_performed"] = False

                                # Register in SQLite
                                sqlite_registry.register_model(
                                    sensor_id=metadata["sensor_id"],
                                    model_type=metadata["model_type"],
                                    model_path=Path(
                                        metadata.get(
                                            "model_path",
                                            f"data/models/{metadata['model_type']}/{metadata['sensor_id']}",
                                        )
                                    ),
                                    training_config=metadata.get("training_config", {}),
                                    training_metrics=metadata.get("training_metrics", {}),
                                    validation_metrics=validation_metrics,
                                    training_time_seconds=metadata.get("training_time_seconds", 0.0),
                                    data_hash=metadata.get("data_hash", ""),
                                    description=metadata.get("description", ""),
                                    tags=metadata.get("tags", []),
                                )

                                logger.info(f"  ✓ Migrated version: {version_id}")
                                migrated_count += 1

                        else:
                            logger.warning(f"  ✗ Metadata file not found: {metadata_file}")
                            failed_count += 1

                except Exception as e:
                    logger.error(f"  ✗ Error migrating version {version_id}: {e}")
                    failed_count += 1

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("MIGRATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total models: {len(models_index)}")
        logger.info(f"Versions migrated: {migrated_count}")
        logger.info(f"Versions failed: {failed_count}")

        if not dry_run:
            logger.info(f"Backup location: {self.backup_path}")
            logger.info(f"New SQLite registry: {self.new_path / 'model_registry.db'}")

            if failed_count == 0:
                logger.info("\n✓ Migration completed successfully!")
            else:
                logger.warning(f"\n⚠ Migration completed with {failed_count} failures")

        return True


def main():
    """Main migration script"""
    import argparse

    parser = argparse.ArgumentParser(description="Migrate JSON model registry to SQLite")
    parser.add_argument(
        "--old-path",
        type=str,
        default="data/models/registry",
        help="Path to old JSON registry (default: data/models/registry)",
    )
    parser.add_argument(
        "--new-path",
        type=str,
        default="data/models/registry_sqlite",
        help="Path for new SQLite registry (default: data/models/registry_sqlite)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run without making changes")

    args = parser.parse_args()

    # Run migration
    migration = RegistryMigration(args.old_path, args.new_path)

    if args.dry_run:
        logger.info("=" * 80)
        logger.info("DRY RUN MODE - No changes will be made")
        logger.info("=" * 80)

    success = migration.migrate_to_sqlite(dry_run=args.dry_run)

    if success:
        if not args.dry_run:
            logger.info("\n" + "=" * 80)
            logger.info("NEXT STEPS:")
            logger.info("=" * 80)
            logger.info("1. Verify the new SQLite registry works correctly")
            logger.info("2. Update your services to use ModelRegistrySQLite instead of ModelRegistry")
            logger.info("3. Test thoroughly before deleting the backup")
            logger.info(f"4. Backup location: {migration.backup_path}")
        sys.exit(0)
    else:
        logger.error("\nMigration failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

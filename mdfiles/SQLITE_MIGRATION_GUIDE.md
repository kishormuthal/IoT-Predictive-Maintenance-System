# SQLite Model Registry Migration Guide

## Overview

This guide explains how to migrate from the JSON-based `ModelRegistry` to the new SQLite-based `ModelRegistrySQLite` implementation.

## Why SQLite?

The new SQLite implementation provides:

✅ **ACID Properties** - Atomic, Consistent, Isolated, Durable transactions
✅ **Concurrency Support** - Thread-safe operations with proper locking
✅ **Better Performance** - Indexed queries for fast lookups
✅ **Data Integrity** - Foreign key constraints and referential integrity
✅ **Data Lineage Tracking** - Track training data provenance
✅ **Centralized Path Management** - Model paths stored in metadata
✅ **Proper Validation** - Enforces validation_performed flag

## What's Fixed

### 8 Critical Fixes from Analysis:

1. ✅ **SQLite Database** - Replaced JSON files with proper database (ACID, concurrency)
2. ✅ **Proper data_hash** - Accepts from training pipeline, not generated placeholder
3. ✅ **Fixed Active Version Logic** - Compares against ALL versions, not just current
4. ✅ **Specific Exceptions** - Proper exception handling (FileNotFoundError, ValueError)
5. ✅ **Configurable Performance Scoring** - Weights can be customized
6. ✅ **Enforce Validation Metrics** - Requires validation_performed=True flag
7. ✅ **Delete Model Artifacts** - delete_version() removes files from disk
8. ✅ **Centralized Path Management** - model_path stored in metadata

## Migration Steps

### Step 1: Dry Run (Recommended)

First, run a dry run to see what will be migrated:

```bash
python scripts/migrate_registry_to_sqlite.py --dry-run
```

This will show you:
- How many models/versions will be migrated
- Any potential issues
- No changes made to your data

### Step 2: Actual Migration

Run the migration:

```bash
python scripts/migrate_registry_to_sqlite.py \
  --old-path data/models/registry \
  --new-path data/models/registry_sqlite
```

This will:
1. **Create backup** of your old registry
2. **Load JSON data** from old registry
3. **Create SQLite database** with proper schema
4. **Migrate all models** and versions
5. **Preserve metadata** including performance scores

### Step 3: Update Service Imports

Update your services to use the new SQLite registry:

#### Before:
```python
from src.infrastructure.ml.model_registry import ModelRegistry

registry = ModelRegistry("data/models/registry")
```

#### After:
```python
from src.infrastructure.ml.model_registry_sqlite import ModelRegistrySQLite

registry = ModelRegistrySQLite("data/models/registry_sqlite")
```

### Step 4: Update Training Pipeline

The new registry requires proper validation:

#### Before:
```python
registry.register_model(
    sensor_id="sensor_001",
    model_type="telemanom",
    model_path=model_path,
    training_config=config,
    training_metrics=train_metrics,
    validation_metrics={},  # ❌ Empty or missing validation
    training_time_seconds=100.0
)
```

#### After:
```python
# Ensure you perform actual validation on held-out data
validation_metrics = {
    'validation_performed': True,  # ✅ Required flag
    'mae': 0.15,
    'r2_score': 0.85,
    # ... other metrics
}

# Pass data_hash from training pipeline
data_hash = hashlib.sha256(training_data.tobytes()).hexdigest()[:16]

registry.register_model(
    sensor_id="sensor_001",
    model_type="telemanom",
    model_path=model_path,
    training_config=config,
    training_metrics=train_metrics,
    validation_metrics=validation_metrics,  # ✅ Proper validation
    training_time_seconds=100.0,
    data_hash=data_hash,  # ✅ From training data
    data_source="nasa_turbofan",
    data_start_date="2024-01-01",
    data_end_date="2024-03-01",
    num_samples=10000
)
```

### Step 5: Verify Migration

Check the new registry:

```python
from src.infrastructure.ml.model_registry_sqlite import ModelRegistrySQLite

registry = ModelRegistrySQLite("data/models/registry_sqlite")

# Get stats
stats = registry.get_registry_stats()
print(f"Total models: {stats['total_models']}")
print(f"Total versions: {stats['total_versions']}")

# Check a specific model
metadata = registry.get_model_metadata("your_version_id")
if metadata:
    print(f"Performance score: {metadata.performance_score}")
    print(f"Model path: {metadata.model_path}")

# Get data lineage
lineage = registry.get_data_lineage("your_version_id")
if lineage:
    print(f"Data source: {lineage['data_source']}")
    print(f"Training samples: {lineage['num_samples']}")
```

## New Features

### 1. Data Lineage Tracking

Track the provenance of training data:

```python
registry.register_model(
    # ... other params ...
    data_hash=data_hash,  # Hash of actual training data
    data_source="nasa_turbofan_dataset",
    data_start_date="2024-01-01T00:00:00",
    data_end_date="2024-03-01T23:59:59",
    num_samples=15000
)

# Later, retrieve lineage
lineage = registry.get_data_lineage(version_id)
```

### 2. Configurable Performance Scoring

Customize performance score calculation:

```python
registry = ModelRegistrySQLite(
    registry_path="data/models/registry_sqlite",
    performance_score_weights={
        'r2_weight': 0.6,        # Custom R² weight
        'mape_weight': 0.4,      # Custom MAPE weight
        'anomaly_min': 0.02,     # Adjust anomaly rate range
        'anomaly_max': 0.08
    }
)
```

### 3. Better Active Version Management

Automatically promotes best version:

```python
# New version automatically becomes active if score is better
version_id = registry.register_model(...)

# Or manually promote
registry.promote_version("specific_version_id")
```

### 4. Artifact Deletion

Delete model files when removing versions:

```python
# Delete version and its files
registry.delete_version(
    version_id="old_version",
    force=False,  # Prevent deleting active version
    delete_artifacts=True  # Also delete model files from disk
)
```

## Database Schema

The SQLite database has 3 main tables:

### 1. `models` Table
- `model_id` (PRIMARY KEY)
- `sensor_id`
- `model_type`
- `active_version`
- `created_at`, `updated_at`

### 2. `versions` Table
- `version_id` (PRIMARY KEY)
- `model_id` (FOREIGN KEY)
- All metadata fields
- Indexed on: model_id, sensor_id, is_active, performance_score

### 3. `data_lineage` Table
- `id` (AUTO INCREMENT)
- `version_id` (FOREIGN KEY)
- `data_hash`, `data_source`
- `data_start_date`, `data_end_date`
- `num_samples`

## Performance Benefits

| Operation | JSON Registry | SQLite Registry |
|-----------|---------------|-----------------|
| Get active version | O(n) file read | O(1) indexed query |
| List versions | O(n) file reads | Single indexed query |
| Filter by score | Load all + filter | Direct WHERE clause |
| Concurrent writes | ❌ Race conditions | ✅ ACID transactions |
| Data integrity | ❌ Manual | ✅ Foreign keys |

## Troubleshooting

### Issue: "validation_metrics must include 'validation_performed': True"

**Solution**: Ensure you perform actual validation before registering:

```python
# Perform validation on held-out data
val_predictions = model.predict(validation_data)
validation_metrics = calculate_metrics(val_predictions, val_actuals)
validation_metrics['validation_performed'] = True
```

### Issue: Migration fails with missing metadata

**Solution**: Check that all metadata files exist in `data/models/registry/metadata/`

### Issue: Model files not found

**Solution**: Ensure model_path in registration points to actual model directory

## Rollback Plan

If you need to rollback:

1. Your original registry was backed up to: `data/models/registry_backup_YYYYMMDD_HHMMSS/`
2. Restore from backup:
   ```bash
   mv data/models/registry_backup_YYYYMMDD_HHMMSS data/models/registry
   ```
3. Revert code changes to use `ModelRegistry`

## Testing Checklist

Before deploying to production:

- [ ] Dry run migration completed successfully
- [ ] All models migrated (check counts)
- [ ] Active versions preserved correctly
- [ ] Services updated to use ModelRegistrySQLite
- [ ] Training pipeline includes validation_performed flag
- [ ] Data hash passed from training pipeline
- [ ] Model paths accessible and correct
- [ ] Backup created and verified
- [ ] Test model loading from registry
- [ ] Test model registration with new pipeline

## Support

For issues or questions:
1. Check migration logs in console output
2. Verify backup was created
3. Review this guide's troubleshooting section
4. Check SQLite database directly: `sqlite3 data/models/registry_sqlite/model_registry.db`

## Next Steps

After successful migration:

1. ✅ Update anomaly_service.py to use ModelRegistrySQLite
2. ✅ Update forecasting_service.py to use ModelRegistrySQLite
3. ✅ Update training_use_case.py to:
   - Pass data_hash from training data
   - Include validation_performed=True in metrics
   - Pass data lineage information
4. ✅ Consider MLflow integration (SESSION 5) for even more advanced features

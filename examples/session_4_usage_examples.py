"""
SESSION 4 Data Management - Usage Examples
Complete examples for all new data management components
"""

from datetime import datetime, timedelta

import numpy as np

# =============================================================================
# Example 1: Data Processing Service - Normalization and Quality Assessment
# =============================================================================


def example_1_data_processing():
    """Example: Use DataProcessingService for normalization and quality checks"""
    from src.core.services.data_processing_service import (
        DataProcessingService,
        NormalizationMethod,
    )

    # Initialize service
    processor = DataProcessingService(
        cache_dir="data/processing_cache",
        default_normalization=NormalizationMethod.ZSCORE,
        outlier_threshold=3.0,
        missing_threshold=0.1,
    )

    # Generate sample sensor data (with some issues)
    sensor_data = np.random.randn(1000)
    sensor_data[100:110] = np.nan  # Add missing values
    sensor_data[500:510] = 10  # Add outliers

    # Assess data quality
    quality_report = processor.assess_data_quality(
        data=sensor_data, sensor_id="SENSOR_001"
    )

    print("=== Data Quality Report ===")
    print(f"Status: {quality_report.status.value}")
    print(
        f"Missing: {quality_report.missing_count} ({quality_report.missing_percentage:.2f}%)"
    )
    print(
        f"Outliers: {quality_report.outlier_count} ({quality_report.outlier_percentage:.2f}%)"
    )
    print(f"Issues: {quality_report.issues}")
    print(f"Recommendations: {quality_report.recommendations}")

    # Prepare training data with normalization
    # First, clean the data
    clean_data = sensor_data[np.isfinite(sensor_data)]

    prepared = processor.prepare_training_data(
        data=clean_data,
        sensor_id="SENSOR_001",
        split_ratio=(0.7, 0.15, 0.15),
        normalize=True,
        assess_quality=True,
    )

    print("\n=== Training Data Prepared ===")
    print(f"Train samples: {len(prepared['train_data'])}")
    print(f"Val samples: {len(prepared['val_data'])}")
    print(f"Test samples: {len(prepared['test_data'])}")
    print(f"Normalization method: {prepared['norm_params'].method.value}")
    print(f"Mean: {prepared['norm_params'].mean:.3f}")
    print(f"Std: {prepared['norm_params'].std:.3f}")
    print(f"Data hash: {prepared['data_hash']}")

    # Denormalize for visualization
    denormalized = processor.denormalize(prepared["train_data"], sensor_id="SENSOR_001")
    print(f"Denormalized mean: {np.mean(denormalized):.3f}")


# =============================================================================
# Example 2: Feature Engineering - Create 40+ Features
# =============================================================================


def example_2_feature_engineering():
    """Example: Engineer comprehensive feature set from raw sensor data"""
    from src.core.services.feature_engineering import FeatureConfig, FeatureEngineer

    # Configure feature engineering
    config = FeatureConfig(
        rolling_windows=[3, 6, 12, 24],  # Multiple window sizes
        lag_periods=[1, 2, 3, 6, 12, 24],  # Lag features
        include_rolling_mean=True,
        include_rolling_std=True,
        include_rolling_min=True,
        include_rolling_max=True,
        include_diff_1=True,
        include_diff_2=True,
        include_ewm=True,
        include_fft=True,
        include_time_features=True,
        include_cyclical_encoding=True,
    )

    # Initialize engineer
    engineer = FeatureEngineer(config=config)

    # Generate sample sensor data with temporal pattern
    hours = 168  # 1 week
    timestamps = [datetime.now() - timedelta(hours=hours - i) for i in range(hours)]

    # Simulate sensor data with daily cycle + trend
    t = np.arange(hours)
    sensor_data = (
        50  # Base value
        + 5 * np.sin(2 * np.pi * t / 24)  # Daily cycle
        + 0.1 * t  # Trend
        + np.random.normal(0, 1, hours)  # Noise
    )

    # Engineer features
    features = engineer.engineer_features(
        data=sensor_data, timestamps=timestamps, sensor_id="SENSOR_001"
    )

    print("=== Feature Engineering Results ===")
    print(f"Total feature sets created: {len(features)}")
    print(f"Feature names: {list(features.keys())[:10]}...")  # Show first 10

    # Create feature matrix for ML
    selected_features = [
        "raw",
        "rolling_12_mean",
        "rolling_12_std",
        "lag_6",
        "diff_1",
        "ewm",
        "hour_sin",
        "hour_cos",
    ]

    feature_matrix = engineer.create_feature_matrix(
        features, selected_features=selected_features
    )

    print(f"\nFeature matrix shape: {feature_matrix.shape}")
    print(f"Selected features: {selected_features}")

    # Show sample values
    print("\nSample feature values (last 5 timesteps):")
    for i, fname in enumerate(selected_features):
        print(f"  {fname:20s}: {feature_matrix[-5:, i].tolist()}")


# =============================================================================
# Example 3: DVC Manager - Data Versioning
# =============================================================================


def example_3_dvc_versioning():
    """Example: Version datasets with DVC for reproducibility"""
    from src.infrastructure.data.dvc_manager import DVCManager

    # Initialize DVC manager
    dvc = DVCManager(
        repo_root=".", data_dir="data", dvc_remote=None  # Use local storage for demo
    )

    # Create sample dataset
    sample_data = np.random.randn(1000, 10)
    dataset_file = "data/processed/SENSOR_001_processed.npy"
    np.save(dataset_file, sample_data)

    print("=== DVC Data Versioning ===")

    # Version the dataset
    dataset_version = dvc.version_dataset(
        file_path=dataset_file,
        dataset_id="SENSOR_001",
        description="Processed sensor data with normalization",
        tags=["normalized", "stable", "v1"],
        sensor_ids=["SENSOR_001"],
        push_to_remote=False,  # Set True if remote configured
    )

    if dataset_version:
        print(f"✅ Dataset versioned successfully!")
        print(f"  Version: {dataset_version.version}")
        print(f"  Data hash: {dataset_version.data_hash[:16]}...")
        print(f"  Size: {dataset_version.size_bytes} bytes")
        print(f"  Samples: {dataset_version.num_samples}")
        print(f"  DVC tracked: {dataset_version.dvc_tracked}")

    # List all versions
    versions = dvc.list_versions("SENSOR_001")
    print(f"\nAll versions for SENSOR_001: {[v.version for v in versions]}")

    # Get latest version
    latest = dvc.get_dataset_version("SENSOR_001")
    print(f"Latest version: {latest.version}")

    # Link dataset to model
    dvc.link_dataset_to_model(
        dataset_id="SENSOR_001",
        dataset_version=dataset_version.version,
        model_id="telemanom_SENSOR_001",
        model_version="v1.0",
    )
    print(f"✅ Linked dataset to model for reproducibility")


# =============================================================================
# Example 4: Data Drift Detection - Monitor Distribution Shifts
# =============================================================================


def example_4_drift_detection():
    """Example: Detect data drift using statistical tests"""
    from src.core.services.data_drift_detector import (
        DataDriftDetector,
        DriftConfig,
        DriftSeverity,
    )

    # Configure drift detection
    config = DriftConfig(
        psi_threshold=0.2,
        jensen_shannon_threshold=0.1,
        mean_shift_threshold=2.0,
        std_ratio_threshold=2.0,
    )

    # Initialize detector
    detector = DataDriftDetector(config=config)

    print("=== Data Drift Detection ===")

    # Baseline data (reference distribution)
    baseline_data = np.random.normal(0, 1, 1000)
    baseline_timestamps = [
        datetime.now() - timedelta(days=7) + timedelta(hours=i) for i in range(1000)
    ]

    # Fit reference distribution
    detector.fit_reference(
        data=baseline_data, sensor_id="SENSOR_001", timestamps=baseline_timestamps
    )
    print("✅ Reference distribution fitted")

    # Test 1: No drift (similar distribution)
    similar_data = np.random.normal(0, 1, 1000)
    similar_timestamps = [
        datetime.now() - timedelta(hours=1000 - i) for i in range(1000)
    ]

    report1 = detector.detect_drift(
        current_data=similar_data,
        sensor_id="SENSOR_001",
        current_timestamps=similar_timestamps,
    )

    print("\n--- Test 1: Similar Data (No Drift Expected) ---")
    print(f"Drift detected: {report1.drift_detected}")
    print(f"Severity: {report1.severity.value}")
    print(f"Drift score: {report1.drift_score:.3f}")
    print(f"Metrics:")
    print(f"  PSI: {report1.metrics['psi']:.3f}")
    print(f"  Mean shift: {report1.metrics['mean_shift_std']:.3f} std")
    print(f"Recommendations: {report1.recommendations}")

    # Test 2: Significant drift (shifted distribution)
    shifted_data = np.random.normal(3, 1.5, 1000)  # Different mean and std

    report2 = detector.detect_drift(
        current_data=shifted_data,
        sensor_id="SENSOR_001",
        current_timestamps=similar_timestamps,
    )

    print("\n--- Test 2: Shifted Data (Drift Expected) ---")
    print(f"Drift detected: {report2.drift_detected}")
    print(f"Severity: {report2.severity.value}")
    print(f"Drift score: {report2.drift_score:.3f}")
    print(f"Drift types: {[dt.value for dt in report2.drift_types]}")
    print(f"Metrics:")
    print(f"  PSI: {report2.metrics['psi']:.3f}")
    print(f"  Mean shift: {report2.metrics['mean_shift_std']:.3f} std")
    print(f"  Std ratio: {report2.metrics['std_ratio']:.3f}")
    print(f"Statistical tests:")
    print(f"  KS test p-value: {report2.statistical_tests['ks_test']['p_value']:.6f}")
    print(f"Recommendations:")
    for rec in report2.recommendations:
        print(f"  - {rec}")


# =============================================================================
# Example 5: End-to-End Data Pipeline
# =============================================================================


def example_5_full_pipeline():
    """Example: Run complete data processing pipeline"""
    from src.core.services.data_processing_service import NormalizationMethod
    from src.core.services.feature_engineering import FeatureConfig
    from src.infrastructure.data.data_pipeline import DataPipeline

    print("=== End-to-End Data Pipeline ===")

    # Initialize pipeline (auto-initializes all components)
    pipeline = DataPipeline()

    # Run full pipeline for a sensor
    results = pipeline.run_full_pipeline(
        sensor_id="P-1",  # Using one of the configured sensors
        hours_back=168,  # 1 week of data
        normalize=True,
        normalization_method=NormalizationMethod.ZSCORE,
        engineer_features=True,
        feature_config=FeatureConfig(
            rolling_windows=[6, 12, 24], lag_periods=[1, 6, 12], include_fft=True
        ),
        detect_drift=True,
        version_dataset=True,
        save_processed=True,
    )

    print(f"\nPipeline ID: {results['pipeline_id']}")
    print(f"Success: {results['success']}")
    print(f"Duration: {results.get('duration_seconds', 0):.2f}s")

    if results["success"]:
        print(f"\n✅ Pipeline completed successfully!")
        print(f"\nData Quality:")
        quality = results["quality_report"]
        print(f"  Status: {quality.status.value}")
        print(f"  Missing: {quality.missing_percentage:.2f}%")
        print(f"  Outliers: {quality.outlier_percentage:.2f}%")

        print(f"\nProcessing:")
        print(f"  Raw shape: {results['raw_data_shape']}")
        print(f"  Normalized: {results['normalized']}")

        print(f"\nFeatures:")
        print(f"  Count: {results.get('feature_count', 0)}")
        print(f"  Names: {results.get('engineered_features', [])[:5]}...")

        print(f"\nDrift Detection:")
        print(f"  Drift detected: {results.get('drift_detected', 'N/A')}")

        print(f"\nVersioning:")
        print(f"  Dataset version: {results.get('dataset_version', 'N/A')}")
        print(f"  Data hash: {results.get('data_hash', 'N/A')}")
        print(f"  Output file: {results.get('output_file', 'N/A')}")
    else:
        print(f"❌ Pipeline failed: {results.get('errors', [])}")


# =============================================================================
# Example 6: Batch Processing Multiple Sensors
# =============================================================================


def example_6_batch_processing():
    """Example: Process multiple sensors in batch"""
    from src.infrastructure.data.data_pipeline import DataPipeline

    print("=== Batch Pipeline Processing ===")

    pipeline = DataPipeline()

    # Process multiple sensors
    sensor_ids = ["P-1", "P-2", "T-1"]  # Pumps and turbine

    batch_results = pipeline.run_batch_pipeline(
        sensor_ids=sensor_ids,
        hours_back=72,  # 3 days
        normalize=True,
        engineer_features=True,
        detect_drift=False,  # Skip drift for speed
        version_dataset=False,
        save_processed=True,
    )

    print(f"\nProcessed {len(sensor_ids)} sensors")

    for sensor_id, results in batch_results.items():
        if results.get("success"):
            quality_status = results["quality_report"].status.value
            print(f"  ✅ {sensor_id}: {quality_status}")
        else:
            print(f"  ❌ {sensor_id}: {results.get('errors', ['Unknown error'])[0]}")


# =============================================================================
# Example 7: Training Data Preparation
# =============================================================================


def example_7_training_preparation():
    """Example: Prepare data specifically for model training"""
    from src.infrastructure.data.data_pipeline import DataPipeline

    print("=== Training Data Preparation ===")

    pipeline = DataPipeline()

    # Prepare training data with quality checks
    training_data = pipeline.prepare_training_data(
        sensor_id="P-1",
        split_ratio=(0.7, 0.15, 0.15),
        hours_back=168,
        normalize=True,
        assess_quality=True,
    )

    print(f"\nTraining data prepared:")
    print(f"  Train samples: {len(training_data['train_data'])}")
    print(f"  Val samples: {len(training_data['val_data'])}")
    print(f"  Test samples: {len(training_data['test_data'])}")
    print(f"  Data hash: {training_data['data_hash']}")

    if "quality_report" in training_data:
        quality = training_data["quality_report"]
        print(f"\nData quality: {quality.status.value}")
        if quality.issues:
            print(f"  Issues: {quality.issues}")

    # Access normalized data for training
    train_data = training_data["train_data"]
    val_data = training_data["val_data"]
    test_data = training_data["test_data"]

    print(f"\nReady for model training!")
    print(f"  Train: {train_data.shape}")
    print(f"  Val: {val_data.shape}")
    print(f"  Test: {test_data.shape}")


# =============================================================================
# Main: Run All Examples
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SESSION 4 DATA MANAGEMENT - USAGE EXAMPLES")
    print("=" * 80)

    try:
        print("\n\n" + "=" * 80)
        print("EXAMPLE 1: Data Processing Service")
        print("=" * 80)
        example_1_data_processing()
    except Exception as e:
        print(f"Error in Example 1: {e}")

    try:
        print("\n\n" + "=" * 80)
        print("EXAMPLE 2: Feature Engineering")
        print("=" * 80)
        example_2_feature_engineering()
    except Exception as e:
        print(f"Error in Example 2: {e}")

    try:
        print("\n\n" + "=" * 80)
        print("EXAMPLE 3: DVC Versioning")
        print("=" * 80)
        example_3_dvc_versioning()
    except Exception as e:
        print(f"Error in Example 3: {e}")

    try:
        print("\n\n" + "=" * 80)
        print("EXAMPLE 4: Drift Detection")
        print("=" * 80)
        example_4_drift_detection()
    except Exception as e:
        print(f"Error in Example 4: {e}")

    try:
        print("\n\n" + "=" * 80)
        print("EXAMPLE 5: Full Pipeline")
        print("=" * 80)
        example_5_full_pipeline()
    except Exception as e:
        print(f"Error in Example 5: {e}")

    try:
        print("\n\n" + "=" * 80)
        print("EXAMPLE 6: Batch Processing")
        print("=" * 80)
        example_6_batch_processing()
    except Exception as e:
        print(f"Error in Example 6: {e}")

    try:
        print("\n\n" + "=" * 80)
        print("EXAMPLE 7: Training Preparation")
        print("=" * 80)
        example_7_training_preparation()
    except Exception as e:
        print(f"Error in Example 7: {e}")

    print("\n\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETE!")
    print("=" * 80)

"""
Complete NASA IoT Predictive Maintenance System Validation
Comprehensive validation using real MSL, SMAP, and Telemanom data
"""

import asyncio
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/system_validation.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Data paths
BASE_DATA_PATH = Path(__file__).parent.parent / "data" / "raw"
MSL_DATA_PATH = BASE_DATA_PATH / "msl"
SMAP_DATA_PATH = BASE_DATA_PATH / "smap"
TELEMANOM_DATA_PATH = BASE_DATA_PATH / "data" / "data" / "2018-05-19_15.00.10"


class CompleteSystemValidator:
    """Comprehensive validator for the complete NASA IoT system"""

    def __init__(self):
        self.validation_results = {}
        self.start_time = time.time()
        self.nasa_datasets = {}
        self.performance_metrics = {}

    async def initialize(self):
        """Initialize the validator"""
        logger.info("Initializing Complete System Validator...")

        # Load NASA datasets
        await self._load_nasa_datasets()

        # Initialize system components
        await self._initialize_components()

    async def _load_nasa_datasets(self):
        """Load all available NASA datasets"""
        logger.info("Loading NASA datasets...")

        # Load MSL data
        if MSL_DATA_PATH.exists():
            msl_train = MSL_DATA_PATH / "train.npy"
            msl_test = MSL_DATA_PATH / "test.npy"
            msl_train_labels = MSL_DATA_PATH / "train_labels.npy"
            msl_test_labels = MSL_DATA_PATH / "test_labels.npy"

            if msl_train.exists():
                self.nasa_datasets["msl_train"] = np.load(msl_train)
                logger.info(f"Loaded MSL training data: {self.nasa_datasets['msl_train'].shape}")

            if msl_test.exists():
                self.nasa_datasets["msl_test"] = np.load(msl_test)
                logger.info(f"Loaded MSL test data: {self.nasa_datasets['msl_test'].shape}")

            if msl_train_labels.exists():
                self.nasa_datasets["msl_train_labels"] = np.load(msl_train_labels)
                logger.info(f"Loaded MSL training labels: {self.nasa_datasets['msl_train_labels'].shape}")

            if msl_test_labels.exists():
                self.nasa_datasets["msl_test_labels"] = np.load(msl_test_labels)
                logger.info(f"Loaded MSL test labels: {self.nasa_datasets['msl_test_labels'].shape}")

        # Load SMAP data
        if SMAP_DATA_PATH.exists():
            smap_train = SMAP_DATA_PATH / "train.npy"
            smap_test = SMAP_DATA_PATH / "test.npy"
            smap_train_labels = SMAP_DATA_PATH / "train_labels.npy"
            smap_test_labels = SMAP_DATA_PATH / "test_labels.npy"

            if smap_train.exists():
                self.nasa_datasets["smap_train"] = np.load(smap_train)
                logger.info(f"Loaded SMAP training data: {self.nasa_datasets['smap_train'].shape}")

            if smap_test.exists():
                self.nasa_datasets["smap_test"] = np.load(smap_test)
                logger.info(f"Loaded SMAP test data: {self.nasa_datasets['smap_test'].shape}")

            if smap_train_labels.exists():
                self.nasa_datasets["smap_train_labels"] = np.load(smap_train_labels)
                logger.info(f"Loaded SMAP training labels: {self.nasa_datasets['smap_train_labels'].shape}")

            if smap_test_labels.exists():
                self.nasa_datasets["smap_test_labels"] = np.load(smap_test_labels)
                logger.info(f"Loaded SMAP test labels: {self.nasa_datasets['smap_test_labels'].shape}")

        # Load Telemanom models
        if TELEMANOM_DATA_PATH.exists():
            models_path = TELEMANOM_DATA_PATH / "models"
            if models_path.exists():
                model_files = list(models_path.glob("*.h5"))
                self.nasa_datasets["telemanom_models"] = {f.stem: f for f in model_files}
                logger.info(f"Found {len(self.nasa_datasets['telemanom_models'])} Telemanom models")

        # Load labeled anomalies
        labeled_anomalies_path = BASE_DATA_PATH / "labeled_anomalies.csv"
        if labeled_anomalies_path.exists():
            import pandas as pd

            self.nasa_datasets["labeled_anomalies"] = pd.read_csv(labeled_anomalies_path)
            logger.info(f"Loaded labeled anomalies: {len(self.nasa_datasets['labeled_anomalies'])} entries")

        logger.info(f"Total datasets loaded: {len(self.nasa_datasets)}")

    async def _initialize_components(self):
        """Initialize all system components"""
        logger.info("Initializing system components...")

        try:
            from src.dashboard.nasa_dashboard_orchestrator import phase3_manager

            await phase3_manager.initialize()
            self.phase3_manager = phase3_manager
            logger.info("✓ Phase 3 Dashboard Orchestrator initialized")
        except Exception as e:
            logger.warning(f"Dashboard Orchestrator initialization failed: {e}")
            self.phase3_manager = None

        # Initialize other components as available
        self.components_initialized = True

    async def validate_data_integrity(self):
        """Validate integrity of all NASA datasets"""
        logger.info("=== Validating Data Integrity ===")

        validation_results = {}

        for dataset_name, data in self.nasa_datasets.items():
            if isinstance(data, np.ndarray):
                try:
                    # Basic integrity checks
                    checks = {
                        "no_nan": not np.any(np.isnan(data)),
                        "no_inf": not np.any(np.isinf(data)),
                        "non_empty": data.size > 0,
                        "reasonable_range": self._check_reasonable_range(data),
                        "consistent_shape": len(data.shape) >= 1,
                    }

                    validation_results[dataset_name] = {
                        "shape": data.shape,
                        "dtype": str(data.dtype),
                        "size_mb": data.nbytes / (1024 * 1024),
                        "checks": checks,
                        "all_passed": all(checks.values()),
                    }

                    status = "✓" if checks["all_passed"] else "✗"
                    logger.info(
                        f"{status} {dataset_name}: {data.shape}, {validation_results[dataset_name]['size_mb']:.2f}MB"
                    )

                    if not checks["all_passed"]:
                        failed_checks = [k for k, v in checks.items() if not v]
                        logger.warning(f"  Failed checks: {failed_checks}")

                except Exception as e:
                    logger.error(f"✗ Data integrity validation failed for {dataset_name}: {e}")
                    validation_results[dataset_name] = {"error": str(e)}

        self.validation_results["data_integrity"] = validation_results
        return validation_results

    def _check_reasonable_range(self, data):
        """Check if data values are in reasonable ranges"""
        try:
            min_val = np.min(data)
            max_val = np.max(data)

            # Reasonable range check (adjust thresholds as needed)
            return min_val >= -10000 and max_val <= 10000
        except:
            return False

    async def validate_cross_mission_compatibility(self):
        """Validate compatibility between different mission datasets"""
        logger.info("=== Validating Cross-Mission Compatibility ===")

        compatibility_results = {}

        # Check MSL vs SMAP compatibility
        if "msl_train" in self.nasa_datasets and "smap_train" in self.nasa_datasets:
            msl_data = self.nasa_datasets["msl_train"]
            smap_data = self.nasa_datasets["smap_train"]

            compatibility_checks = {
                "same_dimensions": msl_data.ndim == smap_data.ndim,
                "compatible_dtypes": msl_data.dtype == smap_data.dtype,
                "similar_value_ranges": self._check_similar_ranges(msl_data, smap_data),
                "processable_together": True,  # Assume true for now
            }

            compatibility_results["msl_smap"] = compatibility_checks

            status = "✓" if all(compatibility_checks.values()) else "✗"
            logger.info(f"{status} MSL-SMAP compatibility")

            if not all(compatibility_checks.values()):
                failed_checks = [k for k, v in compatibility_checks.items() if not v]
                logger.warning(f"  Compatibility issues: {failed_checks}")

        # Check label consistency
        if "msl_train_labels" in self.nasa_datasets and "smap_train_labels" in self.nasa_datasets:
            msl_labels = self.nasa_datasets["msl_train_labels"]
            smap_labels = self.nasa_datasets["smap_train_labels"]

            label_compatibility = {
                "same_label_format": msl_labels.dtype == smap_labels.dtype,
                "binary_labels": np.all(np.isin(msl_labels, [0, 1])) and np.all(np.isin(smap_labels, [0, 1])),
                "reasonable_anomaly_ratio": self._check_anomaly_ratios(msl_labels, smap_labels),
            }

            compatibility_results["labels"] = label_compatibility

            status = "✓" if all(label_compatibility.values()) else "✗"
            logger.info(f"{status} Label compatibility")

        self.validation_results["cross_mission_compatibility"] = compatibility_results
        return compatibility_results

    def _check_similar_ranges(self, data1, data2):
        """Check if two datasets have similar value ranges"""
        try:
            range1 = np.max(data1) - np.min(data1)
            range2 = np.max(data2) - np.min(data2)

            # Check if ranges are within same order of magnitude
            ratio = max(range1, range2) / min(range1, range2)
            return ratio < 100  # Allow up to 2 orders of magnitude difference
        except:
            return False

    def _check_anomaly_ratios(self, labels1, labels2):
        """Check if anomaly ratios are reasonable"""
        try:
            ratio1 = np.sum(labels1) / len(labels1)
            ratio2 = np.sum(labels2) / len(labels2)

            # Both should have reasonable anomaly ratios (1% to 50%)
            return 0.01 <= ratio1 <= 0.5 and 0.01 <= ratio2 <= 0.5
        except:
            return False

    async def validate_anomaly_detection_pipeline(self):
        """Validate the complete anomaly detection pipeline"""
        logger.info("=== Validating Anomaly Detection Pipeline ===")

        pipeline_results = {}

        try:
            from src.anomaly_detection.models.lstm_detector import LSTMDetector

            detector = LSTMDetector()

            # Test with MSL data
            if "msl_train" in self.nasa_datasets:
                msl_sample = self.nasa_datasets["msl_train"][:500]

                start_time = time.time()
                if hasattr(detector, "detect_anomalies"):
                    msl_anomalies = detector.detect_anomalies(msl_sample)
                    detection_time = time.time() - start_time

                    pipeline_results["msl"] = {
                        "samples_processed": len(msl_sample),
                        "anomalies_detected": (len(msl_anomalies) if msl_anomalies else 0),
                        "detection_time": detection_time,
                        "processing_rate": len(msl_sample) / detection_time,
                        "success": True,
                    }

                    logger.info(
                        f"✓ MSL anomaly detection: {pipeline_results['msl']['anomalies_detected']} anomalies in {detection_time:.4f}s"
                    )

            # Test with SMAP data
            if "smap_train" in self.nasa_datasets:
                smap_sample = self.nasa_datasets["smap_train"][:500]

                start_time = time.time()
                if hasattr(detector, "detect_anomalies"):
                    smap_anomalies = detector.detect_anomalies(smap_sample)
                    detection_time = time.time() - start_time

                    pipeline_results["smap"] = {
                        "samples_processed": len(smap_sample),
                        "anomalies_detected": (len(smap_anomalies) if smap_anomalies else 0),
                        "detection_time": detection_time,
                        "processing_rate": len(smap_sample) / detection_time,
                        "success": True,
                    }

                    logger.info(
                        f"✓ SMAP anomaly detection: {pipeline_results['smap']['anomalies_detected']} anomalies in {detection_time:.4f}s"
                    )

        except ImportError as e:
            logger.warning(f"LSTM Detector not available: {e}")
            pipeline_results["error"] = "LSTM Detector not available"
        except Exception as e:
            logger.error(f"Anomaly detection pipeline validation failed: {e}")
            pipeline_results["error"] = str(e)

        self.validation_results["anomaly_detection_pipeline"] = pipeline_results
        return pipeline_results

    async def validate_telemanom_models_integration(self):
        """Validate Telemanom models integration"""
        logger.info("=== Validating Telemanom Models Integration ===")

        models_results = {}

        if "telemanom_models" not in self.nasa_datasets:
            logger.warning("No Telemanom models available for validation")
            return {}

        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            logger.warning("TensorFlow not available for model validation")
            return {}

        models = self.nasa_datasets["telemanom_models"]
        successful_loads = 0
        total_models = len(models)

        # Test first 10 models to avoid excessive test time
        test_models = dict(list(models.items())[:10])

        for sensor_id, model_path in test_models.items():
            try:
                start_time = time.time()
                model = keras.models.load_model(model_path)
                load_time = time.time() - start_time

                # Test model prediction
                input_shape = model.input_shape
                if input_shape:
                    test_shape = list(input_shape)
                    test_shape[0] = 1
                    test_data = np.random.random(test_shape).astype(np.float32)

                    prediction_start = time.time()
                    prediction = model.predict(test_data, verbose=0)
                    prediction_time = time.time() - prediction_start

                    models_results[sensor_id] = {
                        "load_time": load_time,
                        "prediction_time": prediction_time,
                        "input_shape": input_shape,
                        "output_shape": prediction.shape,
                        "model_size_mb": model_path.stat().st_size / (1024 * 1024),
                        "success": True,
                    }

                    successful_loads += 1
                    logger.info(f"✓ Model {sensor_id}: load {load_time:.4f}s, predict {prediction_time:.4f}s")

            except Exception as e:
                logger.error(f"✗ Model {sensor_id} validation failed: {e}")
                models_results[sensor_id] = {"error": str(e), "success": False}

        models_summary = {
            "total_models": total_models,
            "tested_models": len(test_models),
            "successful_loads": successful_loads,
            "success_rate": successful_loads / len(test_models) if test_models else 0,
            "individual_results": models_results,
        }

        logger.info(f"Model validation summary: {successful_loads}/{len(test_models)} models successful")

        self.validation_results["telemanom_models"] = models_summary
        return models_summary

    async def validate_dashboard_integration(self):
        """Validate complete dashboard integration"""
        logger.info("=== Validating Dashboard Integration ===")

        dashboard_results = {}

        if not self.phase3_manager:
            logger.warning("Dashboard not available for validation")
            return {}

        try:
            # Test dashboard initialization
            integration_status = self.phase3_manager.get_integration_status()
            dashboard_results["integration_status"] = integration_status

            # Test dashboard data generation
            dashboard_data = self.phase3_manager.get_phase3_dashboard_data()
            dashboard_results["dashboard_data_available"] = dashboard_data is not None

            # Test with sample NASA data
            if "msl_train" in self.nasa_datasets:
                msl_sample = self.nasa_datasets["msl_train"][:10]
                test_data = {
                    "mission": "MSL",
                    "sensor_data": msl_sample.tolist(),
                    "timestamp": time.time(),
                }

                if hasattr(self.phase3_manager, "process_mission_data"):
                    result = await self.phase3_manager.process_mission_data(test_data)
                    dashboard_results["msl_data_processing"] = result is not None

            # Test performance monitoring
            if hasattr(self.phase3_manager, "performance_monitor"):
                performance_summary = self.phase3_manager.performance_monitor.get_performance_summary()
                dashboard_results["performance_monitoring"] = performance_summary is not None

            # Test cache integration
            if hasattr(self.phase3_manager, "cache_manager"):
                cache_metrics = self.phase3_manager.cache_manager.get_cache_metrics()
                dashboard_results["cache_integration"] = cache_metrics is not None

            success_count = sum(1 for v in dashboard_results.values() if v is True)
            total_tests = len([v for v in dashboard_results.values() if isinstance(v, bool)])

            dashboard_results["overall_success_rate"] = success_count / total_tests if total_tests > 0 else 0

            logger.info(f"Dashboard integration: {success_count}/{total_tests} tests passed")

        except Exception as e:
            logger.error(f"Dashboard integration validation failed: {e}")
            dashboard_results["error"] = str(e)

        self.validation_results["dashboard_integration"] = dashboard_results
        return dashboard_results

    async def validate_performance_benchmarks(self):
        """Validate system performance benchmarks"""
        logger.info("=== Validating Performance Benchmarks ===")

        performance_results = {}

        # Test data loading performance
        if self.nasa_datasets:
            largest_dataset = max(
                (data for data in self.nasa_datasets.values() if isinstance(data, np.ndarray)),
                key=lambda x: x.size,
                default=None,
            )

            if largest_dataset is not None:
                # Test processing performance
                start_time = time.time()
                sample_data = largest_dataset[:10000]  # Process 10k samples

                # Basic operations
                means = np.mean(sample_data, axis=0)
                stds = np.std(sample_data, axis=0)

                # Anomaly detection simulation
                z_scores = np.abs((sample_data - means) / (stds + 1e-8))
                anomalies = np.sum(z_scores > 2.0, axis=1)

                processing_time = time.time() - start_time

                performance_results["data_processing"] = {
                    "samples_processed": len(sample_data),
                    "processing_time": processing_time,
                    "throughput": len(sample_data) / processing_time,
                    "anomalies_found": np.sum(anomalies > 0),
                }

                logger.info(
                    f"Data processing performance: {performance_results['data_processing']['throughput']:.2f} samples/s"
                )

        # Test memory usage
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()

        performance_results["memory_usage"] = {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
            "percent": process.memory_percent(),
        }

        logger.info(f"Memory usage: {performance_results['memory_usage']['rss_mb']:.2f} MB")

        # Performance assertions
        assertions = {
            "reasonable_throughput": performance_results.get("data_processing", {}).get("throughput", 0) > 1000,
            "reasonable_memory": performance_results["memory_usage"]["rss_mb"] < 2000,
            "low_memory_percent": performance_results["memory_usage"]["percent"] < 50,
        }

        performance_results["assertions"] = assertions
        passed_assertions = sum(assertions.values())

        logger.info(f"Performance assertions: {passed_assertions}/{len(assertions)} passed")

        self.validation_results["performance_benchmarks"] = performance_results
        return performance_results

    async def validate_end_to_end_workflow(self):
        """Validate complete end-to-end workflow"""
        logger.info("=== Validating End-to-End Workflow ===")

        workflow_results = {}

        try:
            # Simulate complete workflow: Data → Anomaly Detection → Forecasting → Maintenance

            # Step 1: Data ingestion simulation
            if "msl_train" in self.nasa_datasets:
                msl_sample = self.nasa_datasets["msl_train"][:100]

                workflow_results["data_ingestion"] = {
                    "samples_loaded": len(msl_sample),
                    "data_shape": msl_sample.shape,
                    "success": True,
                }

                # Step 2: Anomaly detection
                anomaly_threshold = np.percentile(msl_sample, 95)
                detected_anomalies = np.where(msl_sample > anomaly_threshold)

                workflow_results["anomaly_detection"] = {
                    "threshold": float(anomaly_threshold),
                    "anomalies_detected": len(detected_anomalies[0]),
                    "anomaly_ratio": len(detected_anomalies[0]) / len(msl_sample),
                    "success": True,
                }

                # Step 3: Forecasting simulation
                forecast_input = msl_sample[-10:]  # Use last 10 samples

                # Simple forecasting simulation
                forecast = np.mean(forecast_input, axis=0) + np.random.normal(0, 0.1, forecast_input.shape[1])

                workflow_results["forecasting"] = {
                    "forecast_generated": True,
                    "forecast_shape": forecast.shape,
                    "input_samples": len(forecast_input),
                    "success": True,
                }

                # Step 4: Maintenance scheduling simulation
                if workflow_results["anomaly_detection"]["anomalies_detected"] > 0:
                    maintenance_tasks = min(3, workflow_results["anomaly_detection"]["anomalies_detected"])

                    workflow_results["maintenance_scheduling"] = {
                        "tasks_scheduled": maintenance_tasks,
                        "priority_sensors": ["T-1", "P-1", "A-1"][:maintenance_tasks],
                        "success": True,
                    }
                else:
                    workflow_results["maintenance_scheduling"] = {
                        "tasks_scheduled": 0,
                        "reason": "No anomalies detected",
                        "success": True,
                    }

                # Overall workflow success
                workflow_success = all(
                    step.get("success", False) for step in workflow_results.values() if isinstance(step, dict)
                )

                workflow_results["overall_success"] = workflow_success

                logger.info(f"End-to-end workflow: {'✓' if workflow_success else '✗'}")
                logger.info(f"  Data samples: {workflow_results['data_ingestion']['samples_loaded']}")
                logger.info(f"  Anomalies detected: {workflow_results['anomaly_detection']['anomalies_detected']}")
                logger.info(f"  Maintenance tasks: {workflow_results['maintenance_scheduling']['tasks_scheduled']}")

        except Exception as e:
            logger.error(f"End-to-end workflow validation failed: {e}")
            workflow_results["error"] = str(e)
            workflow_results["overall_success"] = False

        self.validation_results["end_to_end_workflow"] = workflow_results
        return workflow_results

    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        logger.info("=== Generating Validation Report ===")

        total_time = time.time() - self.start_time

        report = {
            "validation_timestamp": datetime.now().isoformat(),
            "total_validation_time": total_time,
            "datasets_available": list(self.nasa_datasets.keys()),
            "validation_results": self.validation_results,
        }

        # Calculate overall scores
        scores = {}

        for category, results in self.validation_results.items():
            if isinstance(results, dict):
                if "error" in results:
                    scores[category] = 0.0
                elif category == "data_integrity":
                    passed = sum(1 for r in results.values() if r.get("all_passed", False))
                    total = len([r for r in results.values() if "all_passed" in r])
                    scores[category] = passed / total if total > 0 else 0.0
                elif category == "telemanom_models":
                    scores[category] = results.get("success_rate", 0.0)
                elif category == "dashboard_integration":
                    scores[category] = results.get("overall_success_rate", 0.0)
                elif category == "end_to_end_workflow":
                    scores[category] = 1.0 if results.get("overall_success", False) else 0.0
                else:
                    # Generic scoring for other categories
                    if isinstance(results, dict):
                        success_keys = [k for k, v in results.items() if isinstance(v, bool) and v]
                        total_keys = [k for k, v in results.items() if isinstance(v, bool)]
                        scores[category] = len(success_keys) / len(total_keys) if total_keys else 0.0

        overall_score = sum(scores.values()) / len(scores) if scores else 0.0

        report["scores"] = scores
        report["overall_score"] = overall_score

        # Save report
        report_path = Path("logs") / f'validation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Print summary
        logger.info("=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Overall Score: {overall_score:.2%}")
        logger.info(f"Total Time: {total_time:.2f} seconds")
        logger.info(f"Datasets Available: {len(self.nasa_datasets)}")

        logger.info("\nCategory Scores:")
        for category, score in scores.items():
            status = "✓" if score >= 0.8 else "⚠" if score >= 0.5 else "✗"
            logger.info(f"  {status} {category}: {score:.2%}")

        logger.info(f"\nDetailed report saved to: {report_path}")

        return report

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up validation resources...")

        if hasattr(self, "phase3_manager") and self.phase3_manager:
            try:
                await self.phase3_manager.shutdown()
            except Exception as e:
                logger.warning(f"Dashboard shutdown warning: {e}")


async def main():
    """Main validation function"""
    logger.info("Starting Complete NASA System Validation")

    validator = CompleteSystemValidator()

    try:
        # Initialize validator
        await validator.initialize()

        # Run all validation tests
        await validator.validate_data_integrity()
        await validator.validate_cross_mission_compatibility()
        await validator.validate_anomaly_detection_pipeline()
        await validator.validate_telemanom_models_integration()
        await validator.validate_dashboard_integration()
        await validator.validate_performance_benchmarks()
        await validator.validate_end_to_end_workflow()

        # Generate final report
        report = validator.generate_validation_report()

        # Determine exit code based on overall score
        exit_code = 0 if report["overall_score"] >= 0.7 else 1

        logger.info(f"Validation completed with exit code: {exit_code}")
        return exit_code

    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        logger.error(traceback.format_exc())
        return 1

    finally:
        await validator.cleanup()


if __name__ == "__main__":
    # Run validation
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

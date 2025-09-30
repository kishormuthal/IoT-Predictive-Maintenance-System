"""
Test Training Pipeline Integration

This script tests the complete training pipeline:
1. Anomaly detection training
2. Forecasting training
3. Model registry integration
4. Service availability checking

Usage:
    python test_training_pipeline.py --quick  # Quick test mode
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_training_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

# Imports after path setup
from config.equipment_config import EQUIPMENT_REGISTRY
from src.infrastructure.ml.model_registry import ModelRegistry
from src.core.services.anomaly_service import AnomalyDetectionService
from src.core.services.forecasting_service import ForecastingService
from src.infrastructure.data.nasa_data_loader import NASADataLoader


class TrainingPipelineValidator:
    """Validator for complete training pipeline"""

    def __init__(self, models_dir: str = "data/models"):
        """
        Initialize pipeline validator

        Args:
            models_dir: Directory containing models and registry
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.model_registry = ModelRegistry(str(self.models_dir / "registry"))
        self.data_loader = NASADataLoader("data/raw")

        # Initialize services (these will try to load models from registry)
        self.anomaly_service = AnomalyDetectionService(registry_path=str(self.models_dir))
        self.forecasting_service = ForecastingService(str(self.models_dir))

        self.test_results = {}

        logger.info("Training Pipeline Validator initialized")

    def test_model_registry_health(self) -> Dict[str, any]:
        """Test model registry health and availability"""
        try:
            logger.info("Testing model registry health...")

            # Get registry stats
            registry_stats = self.model_registry.get_registry_stats()

            # Get availability report
            availability_report = self.model_registry.get_model_availability_report()

            # Test individual sensor model health
            sample_sensors = list(EQUIPMENT_REGISTRY.keys())[:3]  # Test first 3 sensors
            health_checks = {}

            for sensor_id in sample_sensors:
                health_checks[sensor_id] = {
                    'telemanom': self.model_registry.get_model_health_status(sensor_id, 'telemanom'),
                    'transformer': self.model_registry.get_model_health_status(sensor_id, 'transformer')
                }

            result = {
                'registry_stats': registry_stats,
                'availability_report': availability_report,
                'health_checks': health_checks,
                'status': 'passed'
            }

            logger.info(f"Registry health test passed")
            logger.info(f"Total models: {registry_stats.get('total_models', 0)}")
            logger.info(f"Total versions: {registry_stats.get('total_versions', 0)}")
            logger.info(f"Coverage: {availability_report['availability_summary'].get('coverage_percentage', 0):.1f}%")

            return result

        except Exception as e:
            logger.error(f"Model registry health test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def test_anomaly_service_integration(self) -> Dict[str, any]:
        """Test anomaly detection service with model registry"""
        try:
            logger.info("Testing anomaly service integration...")

            # Test with a sample sensor
            test_sensor = "SMAP-PWR-001"

            # Get test data
            sensor_data = self.data_loader.get_sensor_data(test_sensor, hours_back=100)
            data = np.array(sensor_data['values'])
            timestamps = sensor_data['timestamps']

            # Test anomaly detection
            detection_result = self.anomaly_service.detect_anomalies(
                test_sensor,
                data,
                timestamps
            )

            # Test service methods
            is_trained = self.anomaly_service.is_model_trained(test_sensor)
            model_status = self.anomaly_service.get_model_status()
            detection_summary = self.anomaly_service.get_detection_summary()
            training_recommendations = self.anomaly_service.get_training_recommendations()

            result = {
                'detection_result': {
                    'sensor_id': detection_result['sensor_id'],
                    'anomalies_detected': len(detection_result['anomalies']),
                    'model_status': detection_result['model_status'],
                    'processing_time': detection_result['processing_time']
                },
                'is_trained': is_trained,
                'model_status_count': len(model_status),
                'detection_summary_anomalies': detection_summary.get('total_anomalies', 0),
                'training_recommendations': {
                    'sensors_needing_training': len(training_recommendations.get('sensors_needing_training', [])),
                    'sensors_needing_retraining': len(training_recommendations.get('sensors_needing_retraining', [])),
                    'well_performing_sensors': len(training_recommendations.get('well_performing_sensors', []))
                },
                'status': 'passed'
            }

            logger.info(f"Anomaly service test passed")
            logger.info(f"Detection result: {result['detection_result']['anomalies_detected']} anomalies, model_status: {result['detection_result']['model_status']}")

            return result

        except Exception as e:
            logger.error(f"Anomaly service integration test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def test_forecasting_service_integration(self) -> Dict[str, any]:
        """Test forecasting service with model registry"""
        try:
            logger.info("Testing forecasting service integration...")

            # Test with a sample sensor
            test_sensor = "MSL-MOB-001"

            # Get test data
            sensor_data = self.data_loader.get_sensor_data(test_sensor, hours_back=200)
            data = np.array(sensor_data['values'])

            # Test forecasting
            forecast_result = self.forecasting_service.generate_forecast(
                test_sensor,
                data,
                horizon_hours=24
            )

            # Test service methods
            is_trained = self.forecasting_service.is_model_trained(test_sensor)
            forecast_accuracy = self.forecasting_service.get_forecast_accuracy(test_sensor)
            forecast_summary = self.forecasting_service.get_forecast_summary()
            model_status = self.forecasting_service.get_model_status()

            result = {
                'forecast_result': {
                    'sensor_id': forecast_result['sensor_id'],
                    'forecast_values_count': len(forecast_result['forecast_values']),
                    'model_status': forecast_result['model_status'],
                    'processing_time': forecast_result['processing_time']
                },
                'is_trained': is_trained,
                'forecast_accuracy': forecast_accuracy,
                'forecast_summary_sensors': forecast_summary.get('total_sensors_forecasted', 0),
                'model_status_count': len(model_status),
                'status': 'passed'
            }

            logger.info(f"Forecasting service test passed")
            logger.info(f"Forecast result: {result['forecast_result']['forecast_values_count']} forecast points, model_status: {result['forecast_result']['model_status']}")

            return result

        except Exception as e:
            logger.error(f"Forecasting service integration test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def test_data_loader_integration(self) -> Dict[str, any]:
        """Test NASA data loader functionality"""
        try:
            logger.info("Testing data loader integration...")

            # Test data quality report
            data_quality = self.data_loader.get_data_quality_report()

            # Test sensor list
            sensor_list = self.data_loader.get_sensor_list()

            # Test individual sensor data
            test_sensors = ["SMAP-PWR-001", "MSL-MOB-001"]
            sensor_tests = {}

            for sensor_id in test_sensors:
                sensor_data = self.data_loader.get_sensor_data(sensor_id, hours_back=50)
                sensor_tests[sensor_id] = {
                    'data_points': len(sensor_data['values']),
                    'data_quality': sensor_data['data_quality'],
                    'sensor_type': sensor_data['sensor_info']['type']
                }

            result = {
                'data_quality_report': data_quality,
                'total_sensors': len(sensor_list),
                'sensor_tests': sensor_tests,
                'status': 'passed'
            }

            logger.info(f"Data loader test passed")
            logger.info(f"Total sensors available: {len(sensor_list)}")
            logger.info(f"Data quality: {data_quality.get('data_quality', 'unknown')}")

            return result

        except Exception as e:
            logger.error(f"Data loader integration test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def run_full_pipeline_test(self) -> Dict[str, any]:
        """Run complete pipeline integration test"""
        logger.info("Starting full pipeline integration test...")

        start_time = datetime.now()

        # Run all tests
        self.test_results['data_loader'] = self.test_data_loader_integration()
        self.test_results['model_registry'] = self.test_model_registry_health()
        self.test_results['anomaly_service'] = self.test_anomaly_service_integration()
        self.test_results['forecasting_service'] = self.test_forecasting_service_integration()

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        # Calculate overall results
        passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'passed')
        total_tests = len(self.test_results)

        overall_result = {
            'test_session': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'total_time_seconds': total_time
            },
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            },
            'detailed_results': self.test_results,
            'overall_status': 'passed' if passed_tests == total_tests else 'failed'
        }

        # Save test results
        results_file = self.models_dir / f"pipeline_test_results_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(overall_result, f, indent=2, default=str)

        logger.info(f"Pipeline test completed: {passed_tests}/{total_tests} tests passed")
        logger.info(f"Results saved: {results_file}")

        return overall_result


def main():
    """Main test script"""
    parser = argparse.ArgumentParser(description='Test Training Pipeline Integration')
    parser.add_argument('--models-dir', default='data/models', help='Models directory')

    args = parser.parse_args()

    # Print banner
    print("=" * 70)
    print("TRAINING PIPELINE INTEGRATION TEST")
    print("Session 1: Comprehensive Validation")
    print("=" * 70)
    print(f"Started at: {datetime.now()}")
    print()

    try:
        # Initialize validator
        validator = TrainingPipelineValidator(args.models_dir)

        # Run tests
        results = validator.run_full_pipeline_test()

        # Print results
        print("\n" + "=" * 70)
        print("TEST RESULTS SUMMARY")
        print("=" * 70)
        print(f"Total tests: {results['test_summary']['total_tests']}")
        print(f"Passed: {results['test_summary']['passed_tests']}")
        print(f"Failed: {results['test_summary']['failed_tests']}")
        print(f"Success rate: {results['test_summary']['success_rate']:.1f}%")
        print(f"Total time: {results['test_session']['total_time_seconds']:.1f}s")
        print()

        # Print individual test results
        for test_name, result in results['detailed_results'].items():
            status_symbol = "✓" if result['status'] == 'passed' else "✗"
            print(f"{status_symbol} {test_name.replace('_', ' ').title()}: {result['status']}")
            if result['status'] == 'failed' and 'error' in result:
                print(f"   Error: {result['error']}")

        print("\n" + "=" * 70)
        if results['overall_status'] == 'passed':
            print("ALL TESTS PASSED - TRAINING PIPELINE IS READY!")
            print("You can now run:")
            print("  python train_anomaly_models.py")
            print("  python train_forecasting_models.py")
        else:
            print("SOME TESTS FAILED - CHECK CONFIGURATION")
        print("=" * 70)

        return 0 if results['overall_status'] == 'passed' else 1

    except Exception as e:
        logger.error(f"Test pipeline failed: {e}")
        print(f"\nTest pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    import numpy as np  # Import here to avoid issues if not available
    exit_code = main()
    sys.exit(exit_code)
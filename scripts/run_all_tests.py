"""
Comprehensive Test Runner for NASA IoT Predictive Maintenance System
Runs all unit, integration, e2e, and performance tests with real NASA data
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


class TestRunner:
    """Comprehensive test runner for the NASA IoT system"""

    def __init__(self, verbose=False, fast_mode=False):
        self.verbose = verbose
        self.fast_mode = fast_mode
        self.results = {}
        self.start_time = time.time()

    def run_test_suite(self, suite_name, test_path, description):
        """Run a specific test suite"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {suite_name}: {description}")
        logger.info(f"{'='*60}")

        start_time = time.time()

        # Build pytest command
        cmd = [
            sys.executable, '-m', 'pytest',
            str(test_path),
            '-v' if self.verbose else '--tb=short',
            '--color=yes',
            '-x' if self.fast_mode else '',  # Stop on first failure in fast mode
        ]

        # Filter out empty strings
        cmd = [c for c in cmd if c]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT
            )

            duration = time.time() - start_time

            self.results[suite_name] = {
                'success': result.returncode == 0,
                'duration': duration,
                'output': result.stdout,
                'errors': result.stderr,
                'return_code': result.returncode
            }

            status = "✓ PASSED" if result.returncode == 0 else "✗ FAILED"
            logger.info(f"{status} {suite_name} ({duration:.2f}s)")

            if result.returncode != 0:
                logger.error(f"Error output: {result.stderr}")
                if self.verbose:
                    logger.error(f"Full output: {result.stdout}")

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"✗ FAILED {suite_name} ({duration:.2f}s) - Exception: {e}")

            self.results[suite_name] = {
                'success': False,
                'duration': duration,
                'output': '',
                'errors': str(e),
                'return_code': -1
            }

    def run_unit_tests(self):
        """Run all unit tests"""
        unit_tests = [
            ('NASA Dashboard Orchestrator', 'tests/unit/test_nasa_dashboard_orchestrator.py'),
            ('Event Coordinator', 'tests/unit/test_event_coordinator.py'),
            ('Enhanced Cache Manager', 'tests/unit/test_enhanced_cache_manager.py'),
        ]

        for test_name, test_path in unit_tests:
            if (PROJECT_ROOT / test_path).exists():
                self.run_test_suite(f"Unit: {test_name}", test_path, f"Unit tests for {test_name}")
            else:
                logger.warning(f"Test file not found: {test_path}")

    def run_integration_tests(self):
        """Run all integration tests with real NASA data"""
        integration_tests = [
            ('MSL Data Integration', 'tests/integration/test_msl_data_integration.py'),
            ('SMAP Data Integration', 'tests/integration/test_smap_data_integration.py'),
            ('Telemanom Models', 'tests/integration/test_telemanom_trained_models.py'),
        ]

        for test_name, test_path in integration_tests:
            if (PROJECT_ROOT / test_path).exists():
                self.run_test_suite(f"Integration: {test_name}", test_path, f"Integration tests with real {test_name}")
            else:
                logger.warning(f"Test file not found: {test_path}")

    def run_e2e_tests(self):
        """Run end-to-end tests"""
        e2e_tests = [
            ('NASA Data Pipeline E2E', 'tests/e2e/test_nasa_data_pipeline_e2e.py'),
        ]

        for test_name, test_path in e2e_tests:
            if (PROJECT_ROOT / test_path).exists():
                self.run_test_suite(f"E2E: {test_name}", test_path, f"End-to-end {test_name}")
            else:
                logger.warning(f"Test file not found: {test_path}")

    def run_performance_tests(self):
        """Run performance tests"""
        performance_tests = [
            ('NASA Data Performance', 'tests/performance/test_nasa_data_performance.py'),
        ]

        for test_name, test_path in performance_tests:
            if (PROJECT_ROOT / test_path).exists():
                self.run_test_suite(f"Performance: {test_name}", test_path, f"Performance tests with {test_name}")
            else:
                logger.warning(f"Test file not found: {test_path}")

    def run_system_validation(self):
        """Run complete system validation"""
        validation_script = PROJECT_ROOT / 'scripts' / 'validate_complete_nasa_system.py'

        if validation_script.exists():
            logger.info(f"\n{'='*60}")
            logger.info("Running Complete System Validation")
            logger.info(f"{'='*60}")

            start_time = time.time()

            try:
                result = subprocess.run(
                    [sys.executable, str(validation_script)],
                    capture_output=True,
                    text=True,
                    cwd=PROJECT_ROOT
                )

                duration = time.time() - start_time

                self.results['System Validation'] = {
                    'success': result.returncode == 0,
                    'duration': duration,
                    'output': result.stdout,
                    'errors': result.stderr,
                    'return_code': result.returncode
                }

                status = "✓ PASSED" if result.returncode == 0 else "✗ FAILED"
                logger.info(f"{status} System Validation ({duration:.2f}s)")

                if self.verbose or result.returncode != 0:
                    logger.info(f"Validation output:\n{result.stdout}")

                if result.returncode != 0:
                    logger.error(f"Validation errors:\n{result.stderr}")

            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"✗ System Validation failed with exception: {e}")

                self.results['System Validation'] = {
                    'success': False,
                    'duration': duration,
                    'output': '',
                    'errors': str(e),
                    'return_code': -1
                }
        else:
            logger.warning("System validation script not found")

    def generate_summary(self):
        """Generate test summary"""
        total_time = time.time() - self.start_time

        logger.info(f"\n{'='*60}")
        logger.info("TEST EXECUTION SUMMARY")
        logger.info(f"{'='*60}")

        # Count results
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r['success'])
        failed_tests = total_tests - passed_tests

        logger.info(f"Total Test Suites: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success Rate: {passed_tests/total_tests*100:.1f}%" if total_tests > 0 else "N/A")
        logger.info(f"Total Execution Time: {total_time:.2f} seconds")

        logger.info(f"\nDetailed Results:")
        for test_name, result in self.results.items():
            status = "✓" if result['success'] else "✗"
            logger.info(f"  {status} {test_name}: {result['duration']:.2f}s")

        # Show failed tests details
        if failed_tests > 0:
            logger.info(f"\nFailed Test Details:")
            for test_name, result in self.results.items():
                if not result['success']:
                    logger.error(f"\n{test_name}:")
                    logger.error(f"  Return Code: {result['return_code']}")
                    logger.error(f"  Error: {result['errors']}")

        return passed_tests == total_tests

    def run_all_tests(self, include_performance=True, include_validation=True):
        """Run all test suites"""
        logger.info("Starting Comprehensive NASA IoT System Testing")
        logger.info(f"Fast Mode: {self.fast_mode}")
        logger.info(f"Verbose: {self.verbose}")

        # Run test suites
        self.run_unit_tests()
        self.run_integration_tests()
        self.run_e2e_tests()

        if include_performance and not self.fast_mode:
            self.run_performance_tests()

        if include_validation:
            self.run_system_validation()

        # Generate summary
        all_passed = self.generate_summary()

        return all_passed


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='NASA IoT System Test Runner')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('-f', '--fast', action='store_true', help='Fast mode (skip performance tests)')
    parser.add_argument('--unit-only', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration-only', action='store_true', help='Run only integration tests')
    parser.add_argument('--e2e-only', action='store_true', help='Run only e2e tests')
    parser.add_argument('--performance-only', action='store_true', help='Run only performance tests')
    parser.add_argument('--validation-only', action='store_true', help='Run only system validation')
    parser.add_argument('--no-validation', action='store_true', help='Skip system validation')

    args = parser.parse_args()

    runner = TestRunner(verbose=args.verbose, fast_mode=args.fast)

    try:
        # Run specific test types based on arguments
        if args.unit_only:
            runner.run_unit_tests()
        elif args.integration_only:
            runner.run_integration_tests()
        elif args.e2e_only:
            runner.run_e2e_tests()
        elif args.performance_only:
            runner.run_performance_tests()
        elif args.validation_only:
            runner.run_system_validation()
        else:
            # Run all tests
            include_performance = not args.fast
            include_validation = not args.no_validation
            runner.run_all_tests(include_performance, include_validation)

        # Generate summary
        all_passed = runner.generate_summary()

        # Exit with appropriate code
        sys.exit(0 if all_passed else 1)

    except KeyboardInterrupt:
        logger.error("\nTest execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Test runner failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
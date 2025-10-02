"""
Test Utility Functions - Session 1
Helper functions for testing the IoT Predictive Maintenance System
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import psutil

# Test markers
import pytest

pytestmark = [pytest.mark.session1, pytest.mark.utils]

logger = logging.getLogger(__name__)


class ImportTimer:
    """Context manager for timing imports and detecting hanging"""

    def __init__(self, module_name: str, timeout: int = 30):
        self.module_name = module_name
        self.timeout = timeout
        self.start_time = None
        self.end_time = None
        self.import_time = None
        self.timed_out = False

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.import_time = self.end_time - self.start_time

        if self.import_time > self.timeout:
            self.timed_out = True
            logger.warning(f"Import of {self.module_name} took {self.import_time:.2f}s (timeout: {self.timeout}s)")

    def is_slow(self, threshold: float = 5.0) -> bool:
        """Check if import was slower than threshold"""
        return self.import_time and self.import_time > threshold

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the import"""
        return {
            "module_name": self.module_name,
            "import_time": self.import_time,
            "timed_out": self.timed_out,
            "timeout_threshold": self.timeout,
        }


class MemoryMonitor:
    """Monitor memory usage during tests"""

    def __init__(self):
        self.initial_memory = None
        self.peak_memory = None
        self.final_memory = None
        self.memory_samples = []

    def start(self):
        """Start memory monitoring"""
        self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.initial_memory
        self.memory_samples = [self.initial_memory]

    def sample(self):
        """Take a memory sample"""
        if self.initial_memory is not None:
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            self.memory_samples.append(current_memory)
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory

    def stop(self):
        """Stop memory monitoring"""
        self.final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        return {
            "initial_mb": self.initial_memory,
            "peak_mb": self.peak_memory,
            "final_mb": self.final_memory,
            "increase_mb": (
                self.final_memory - self.initial_memory if self.final_memory and self.initial_memory else 0
            ),
            "samples_count": len(self.memory_samples),
        }

    def has_memory_leak(self, threshold_mb: float = 50.0) -> bool:
        """Check if there's a potential memory leak"""
        usage = self.get_memory_usage()
        return usage["increase_mb"] > threshold_mb


class TestDataValidator:
    """Validate test data for correctness and consistency"""

    @staticmethod
    def validate_sensor_data(data: Dict[str, Any]) -> Dict[str, bool]:
        """Validate sensor data structure and values"""
        results = {}

        # Check required fields
        required_fields = ["timestamps", "values", "sensor_id"]
        results["has_required_fields"] = all(field in data for field in required_fields)

        # Check data consistency
        if "timestamps" in data and "values" in data:
            results["timestamps_values_match"] = len(data["timestamps"]) == len(data["values"])
        else:
            results["timestamps_values_match"] = False

        # Check value types
        if "values" in data:
            values = data["values"]
            if hasattr(values, "__iter__"):
                results["values_are_numeric"] = all(isinstance(v, (int, float, np.number)) for v in values)
            else:
                results["values_are_numeric"] = False
        else:
            results["values_are_numeric"] = False

        # Check timestamp ordering
        if "timestamps" in data and hasattr(data["timestamps"], "__iter__"):
            timestamps = data["timestamps"]
            if len(timestamps) > 1:
                results["timestamps_ordered"] = all(t1 <= t2 for t1, t2 in zip(timestamps[:-1], timestamps[1:]))
            else:
                results["timestamps_ordered"] = True
        else:
            results["timestamps_ordered"] = False

        return results

    @staticmethod
    def validate_anomaly_data(anomalies: List[Dict[str, Any]]) -> Dict[str, bool]:
        """Validate anomaly detection data"""
        results = {}

        # Check if data is provided
        results["has_anomalies"] = len(anomalies) > 0

        if not results["has_anomalies"]:
            return results

        # Check required fields for each anomaly
        required_fields = ["sensor_id", "timestamp", "score", "severity"]
        results["all_have_required_fields"] = all(
            all(field in anomaly for field in required_fields) for anomaly in anomalies
        )

        # Check score ranges (0-1)
        results["scores_in_range"] = all(0.0 <= anomaly.get("score", -1) <= 1.0 for anomaly in anomalies)

        # Check severity values
        valid_severities = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        results["valid_severities"] = all(anomaly.get("severity") in valid_severities for anomaly in anomalies)

        return results

    @staticmethod
    def validate_forecast_data(forecast: Dict[str, Any]) -> Dict[str, bool]:
        """Validate forecast data structure"""
        results = {}

        # Check required fields
        required_fields = ["sensor_id", "forecast_timestamps", "forecast_values"]
        results["has_required_fields"] = all(field in forecast for field in required_fields)

        # Check forecast consistency
        if "forecast_timestamps" in forecast and "forecast_values" in forecast:
            results["forecast_length_match"] = len(forecast["forecast_timestamps"]) == len(forecast["forecast_values"])
        else:
            results["forecast_length_match"] = False

        # Check confidence intervals
        if "confidence_intervals" in forecast:
            ci = forecast["confidence_intervals"]
            if isinstance(ci, dict) and "upper" in ci and "lower" in ci:
                results["has_confidence_intervals"] = True
                if "forecast_values" in forecast:
                    forecast_len = len(forecast["forecast_values"])
                    results["confidence_intervals_match_length"] = (
                        len(ci["upper"]) == forecast_len and len(ci["lower"]) == forecast_len
                    )
                else:
                    results["confidence_intervals_match_length"] = False
            else:
                results["has_confidence_intervals"] = False
                results["confidence_intervals_match_length"] = False
        else:
            results["has_confidence_intervals"] = False
            results["confidence_intervals_match_length"] = False

        return results


class PerformanceBenchmark:
    """Benchmark performance of various operations"""

    def __init__(self):
        self.benchmarks = {}

    def time_operation(self, operation_name: str, operation: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Time a specific operation"""
        memory_monitor = MemoryMonitor()
        memory_monitor.start()

        start_time = time.time()
        try:
            result = operation(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        end_time = time.time()

        memory_monitor.stop()

        benchmark_result = {
            "operation_name": operation_name,
            "execution_time": end_time - start_time,
            "success": success,
            "error": error,
            "memory_usage": memory_monitor.get_memory_usage(),
            "timestamp": datetime.now(),
        }

        self.benchmarks[operation_name] = benchmark_result
        return benchmark_result

    def compare_operations(self, operation1: str, operation2: str) -> Dict[str, Any]:
        """Compare performance of two operations"""
        if operation1 not in self.benchmarks or operation2 not in self.benchmarks:
            return {"error": "One or both operations not found in benchmarks"}

        bench1 = self.benchmarks[operation1]
        bench2 = self.benchmarks[operation2]

        return {
            "operation1": operation1,
            "operation2": operation2,
            "time_difference": bench2["execution_time"] - bench1["execution_time"],
            "faster_operation": (operation1 if bench1["execution_time"] < bench2["execution_time"] else operation2),
            "performance_ratio": (
                bench2["execution_time"] / bench1["execution_time"] if bench1["execution_time"] > 0 else float("inf")
            ),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmarks"""
        if not self.benchmarks:
            return {"message": "No benchmarks recorded"}

        execution_times = [b["execution_time"] for b in self.benchmarks.values() if b["success"]]
        memory_increases = [b["memory_usage"]["increase_mb"] for b in self.benchmarks.values() if b["success"]]

        return {
            "total_operations": len(self.benchmarks),
            "successful_operations": sum(1 for b in self.benchmarks.values() if b["success"]),
            "average_execution_time": (np.mean(execution_times) if execution_times else 0),
            "max_execution_time": max(execution_times) if execution_times else 0,
            "average_memory_increase": (np.mean(memory_increases) if memory_increases else 0),
            "max_memory_increase": max(memory_increases) if memory_increases else 0,
        }


class TestSessionTracker:
    """Track test session progress and results"""

    def __init__(self, session_id: int, session_name: str):
        self.session_id = session_id
        self.session_name = session_name
        self.start_time = datetime.now()
        self.end_time = None
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.import_tests = []
        self.unit_tests = []
        self.integration_tests = []
        self.issues_found = []

    def add_test_result(
        self,
        test_name: str,
        test_type: str,
        passed: bool,
        execution_time: float = 0.0,
        error: str = None,
    ):
        """Add a test result to the tracker"""
        test_result = {
            "test_name": test_name,
            "test_type": test_type,
            "passed": passed,
            "execution_time": execution_time,
            "error": error,
            "timestamp": datetime.now(),
        }

        self.tests_run += 1
        if passed:
            self.tests_passed += 1
        else:
            self.tests_failed += 1

        if test_type == "import":
            self.import_tests.append(test_result)
        elif test_type == "unit":
            self.unit_tests.append(test_result)
        elif test_type == "integration":
            self.integration_tests.append(test_result)

        if not passed and error:
            self.issues_found.append(
                {
                    "test_name": test_name,
                    "test_type": test_type,
                    "error": error,
                    "timestamp": datetime.now(),
                }
            )

    def add_issue(self, issue_type: str, description: str, severity: str = "medium"):
        """Add an issue found during testing"""
        issue = {
            "issue_type": issue_type,
            "description": description,
            "severity": severity,
            "timestamp": datetime.now(),
        }
        self.issues_found.append(issue)

    def finish_session(self):
        """Mark the session as finished"""
        self.end_time = datetime.now()

    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary"""
        duration = (self.end_time or datetime.now()) - self.start_time

        return {
            "session_id": self.session_id,
            "session_name": self.session_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": duration.total_seconds(),
            "tests_summary": {
                "total_tests": self.tests_run,
                "passed": self.tests_passed,
                "failed": self.tests_failed,
                "success_rate": (self.tests_passed / self.tests_run if self.tests_run > 0 else 0),
            },
            "test_breakdown": {
                "import_tests": len(self.import_tests),
                "unit_tests": len(self.unit_tests),
                "integration_tests": len(self.integration_tests),
            },
            "issues_found": len(self.issues_found),
            "critical_issues": len([i for i in self.issues_found if i.get("severity") == "critical"]),
            "session_status": "completed" if self.end_time else "in_progress",
        }

    def generate_report(self) -> str:
        """Generate a detailed text report"""
        summary = self.get_session_summary()

        report = f"""
IoT Predictive Maintenance System - Test Session Report
Session {self.session_id}: {self.session_name}

=== SUMMARY ===
Duration: {summary['duration_seconds']:.1f} seconds
Tests Run: {summary['tests_summary']['total_tests']}
Success Rate: {summary['tests_summary']['success_rate']:.1%}

=== TEST BREAKDOWN ===
Import Tests: {summary['test_breakdown']['import_tests']}
Unit Tests: {summary['test_breakdown']['unit_tests']}
Integration Tests: {summary['test_breakdown']['integration_tests']}

=== ISSUES FOUND ===
Total Issues: {summary['issues_found']}
Critical Issues: {summary['critical_issues']}

"""

        if self.issues_found:
            report += "\n=== DETAILED ISSUES ===\n"
            for i, issue in enumerate(self.issues_found, 1):
                report += f"{i}. [{issue.get('severity', 'unknown').upper()}] {issue.get('issue_type', 'unknown')}\n"
                report += f"   Description: {issue.get('description', 'No description')}\n"
                if "error" in issue:
                    report += f"   Error: {issue['error']}\n"
                report += "\n"

        return report


def assert_within_tolerance(actual: float, expected: float, tolerance: float = 0.01, message: str = ""):
    """Assert that actual value is within tolerance of expected value"""
    diff = abs(actual - expected)
    tolerance_value = abs(expected * tolerance)
    assert (
        diff <= tolerance_value
    ), f"{message} - Expected: {expected}, Actual: {actual}, Tolerance: {tolerance_value}, Diff: {diff}"


def assert_sensor_data_valid(sensor_data: Dict[str, Any], sensor_id: str):
    """Assert that sensor data is valid for a given sensor"""
    validator = TestDataValidator()
    validation_results = validator.validate_sensor_data(sensor_data)

    assert validation_results["has_required_fields"], f"Sensor {sensor_id} missing required fields"
    assert validation_results["timestamps_values_match"], f"Sensor {sensor_id} timestamps and values length mismatch"
    assert validation_results["values_are_numeric"], f"Sensor {sensor_id} has non-numeric values"
    assert validation_results["timestamps_ordered"], f"Sensor {sensor_id} timestamps not in order"


def create_mock_sensor_data(
    sensor_id: str,
    num_points: int = 100,
    base_value: float = 50.0,
    noise_level: float = 5.0,
) -> Dict[str, Any]:
    """Create mock sensor data for testing"""
    base_time = datetime.now() - timedelta(hours=num_points // 60)
    timestamps = [base_time + timedelta(minutes=i) for i in range(num_points)]
    values = np.random.normal(base_value, noise_level, num_points)

    return {
        "sensor_id": sensor_id,
        "timestamps": timestamps,
        "values": values,
        "metadata": {
            "generated": True,
            "num_points": num_points,
            "base_value": base_value,
            "noise_level": noise_level,
        },
    }

"""
Performance Monitoring and Metrics
Comprehensive monitoring system for training and inference performance
"""

import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric"""

    timestamp: str
    metric_name: str
    value: float
    unit: str
    tags: Dict[str, str] = None


@dataclass
class SystemMetrics:
    """System resource metrics"""

    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float


@dataclass
class TrainingMetrics:
    """Training-specific metrics"""

    timestamp: str
    sensor_id: str
    model_type: str
    training_time_seconds: float
    memory_peak_mb: float
    cpu_avg_percent: float
    training_loss: float = 0.0
    validation_loss: float = 0.0
    model_size_mb: float = 0.0
    data_samples: int = 0


@dataclass
class InferenceMetrics:
    """Inference-specific metrics"""

    timestamp: str
    sensor_id: str
    model_type: str
    inference_time_ms: float
    memory_used_mb: float
    batch_size: int
    throughput_samples_per_sec: float
    accuracy: float = 0.0
    confidence: float = 0.0


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system
    """

    def __init__(self, metrics_dir: str = "./metrics"):
        """
        Initialize performance monitor

        Args:
            metrics_dir: Directory to store metrics
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Metrics storage
        self.system_metrics: deque = deque(maxlen=1000)
        self.training_metrics: deque = deque(maxlen=500)
        self.inference_metrics: deque = deque(maxlen=1000)
        self.custom_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = 5  # seconds

        # Performance context tracking
        self.active_contexts: Dict[str, Dict[str, Any]] = {}

        logger.info(
            f"Performance monitor initialized - metrics dir: {self.metrics_dir}"
        )

    def start_monitoring(self, interval: int = 5):
        """Start continuous system monitoring"""
        try:
            if self.monitoring_active:
                logger.warning("Monitoring already active")
                return

            self.monitoring_interval = interval
            self.monitoring_active = True

            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop, daemon=True
            )
            self.monitoring_thread.start()

            logger.info(f"Started performance monitoring (interval: {interval}s)")

        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")

    def stop_monitoring(self):
        """Stop continuous monitoring"""
        try:
            if not self.monitoring_active:
                return

            self.monitoring_active = False

            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=10)

            logger.info("Stopped performance monitoring")

        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()

                # Sleep for interval
                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)

    def _collect_system_metrics(self):
        """Collect current system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_mb = memory.available / (1024 * 1024)

            # Disk metrics
            disk = psutil.disk_usage("/")
            disk_usage_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024 * 1024 * 1024)

            # Create metrics object
            metrics = SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available_mb=memory_available_mb,
                disk_usage_percent=disk_usage_percent,
                disk_free_gb=disk_free_gb,
            )

            self.system_metrics.append(metrics)

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    def start_training_context(self, sensor_id: str, model_type: str) -> str:
        """
        Start a training performance context

        Args:
            sensor_id: Equipment sensor ID
            model_type: Model type being trained

        Returns:
            Context ID for tracking
        """
        try:
            context_id = f"training_{sensor_id}_{model_type}_{int(time.time())}"

            # Get initial system state
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

            self.active_contexts[context_id] = {
                "type": "training",
                "sensor_id": sensor_id,
                "model_type": model_type,
                "start_time": time.time(),
                "start_timestamp": datetime.now().isoformat(),
                "initial_memory_mb": initial_memory,
                "peak_memory_mb": initial_memory,
                "cpu_samples": [],
                "memory_samples": [],
            }

            logger.info(f"Started training context: {context_id}")
            return context_id

        except Exception as e:
            logger.error(f"Error starting training context: {e}")
            return ""

    def update_training_context(self, context_id: str, metrics: Dict[str, Any]):
        """Update training context with metrics"""
        try:
            if context_id not in self.active_contexts:
                logger.warning(f"Training context not found: {context_id}")
                return

            context = self.active_contexts[context_id]

            # Update with provided metrics
            context.update(metrics)

            # Collect current system metrics
            try:
                process = psutil.Process()
                current_memory = process.memory_info().rss / (1024 * 1024)
                current_cpu = psutil.cpu_percent()

                context["peak_memory_mb"] = max(
                    context["peak_memory_mb"], current_memory
                )
                context["cpu_samples"].append(current_cpu)
                context["memory_samples"].append(current_memory)

                # Keep only recent samples
                context["cpu_samples"] = context["cpu_samples"][-100:]
                context["memory_samples"] = context["memory_samples"][-100:]

            except Exception:
                pass  # Continue without system metrics if not available

        except Exception as e:
            logger.error(f"Error updating training context: {e}")

    def end_training_context(self, context_id: str) -> Optional[TrainingMetrics]:
        """
        End training context and record metrics

        Args:
            context_id: Context ID from start_training_context

        Returns:
            Training metrics object
        """
        try:
            if context_id not in self.active_contexts:
                logger.warning(f"Training context not found: {context_id}")
                return None

            context = self.active_contexts[context_id]
            end_time = time.time()
            training_time = end_time - context["start_time"]

            # Calculate averages
            avg_cpu = (
                sum(context["cpu_samples"]) / len(context["cpu_samples"])
                if context["cpu_samples"]
                else 0
            )
            peak_memory = context["peak_memory_mb"]

            # Create training metrics
            metrics = TrainingMetrics(
                timestamp=datetime.now().isoformat(),
                sensor_id=context["sensor_id"],
                model_type=context["model_type"],
                training_time_seconds=training_time,
                memory_peak_mb=peak_memory,
                cpu_avg_percent=avg_cpu,
                training_loss=context.get("training_loss", 0.0),
                validation_loss=context.get("validation_loss", 0.0),
                model_size_mb=context.get("model_size_mb", 0.0),
                data_samples=context.get("data_samples", 0),
            )

            # Store metrics
            self.training_metrics.append(metrics)

            # Remove context
            del self.active_contexts[context_id]

            logger.info(
                f"Ended training context: {context_id} (time: {training_time:.2f}s)"
            )
            return metrics

        except Exception as e:
            logger.error(f"Error ending training context: {e}")
            return None

    def record_inference_metrics(
        self,
        sensor_id: str,
        model_type: str,
        inference_time_ms: float,
        batch_size: int = 1,
        accuracy: float = 0.0,
        confidence: float = 0.0,
    ):
        """Record inference performance metrics"""
        try:
            # Get current memory usage
            try:
                process = psutil.Process()
                memory_used = process.memory_info().rss / (1024 * 1024)
            except Exception:
                memory_used = 0.0

            # Calculate throughput
            throughput = (
                (batch_size * 1000) / inference_time_ms if inference_time_ms > 0 else 0
            )

            metrics = InferenceMetrics(
                timestamp=datetime.now().isoformat(),
                sensor_id=sensor_id,
                model_type=model_type,
                inference_time_ms=inference_time_ms,
                memory_used_mb=memory_used,
                batch_size=batch_size,
                throughput_samples_per_sec=throughput,
                accuracy=accuracy,
                confidence=confidence,
            )

            self.inference_metrics.append(metrics)

        except Exception as e:
            logger.error(f"Error recording inference metrics: {e}")

    def record_custom_metric(
        self, name: str, value: float, unit: str = "", tags: Dict[str, str] = None
    ):
        """Record custom metric"""
        try:
            metric = PerformanceMetric(
                timestamp=datetime.now().isoformat(),
                metric_name=name,
                value=value,
                unit=unit,
                tags=tags or {},
            )

            self.custom_metrics[name].append(metric)

        except Exception as e:
            logger.error(f"Error recording custom metric: {e}")

    def get_system_metrics_summary(self, hours_back: int = 1) -> Dict[str, Any]:
        """Get system metrics summary"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)

            recent_metrics = [
                m
                for m in self.system_metrics
                if datetime.fromisoformat(m.timestamp) >= cutoff_time
            ]

            if not recent_metrics:
                return {"error": "No recent system metrics available"}

            # Calculate statistics
            cpu_values = [m.cpu_percent for m in recent_metrics]
            memory_values = [m.memory_percent for m in recent_metrics]

            return {
                "time_range_hours": hours_back,
                "samples_count": len(recent_metrics),
                "cpu": {
                    "current": cpu_values[-1] if cpu_values else 0,
                    "average": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                    "max": max(cpu_values) if cpu_values else 0,
                    "min": min(cpu_values) if cpu_values else 0,
                },
                "memory": {
                    "current_percent": memory_values[-1] if memory_values else 0,
                    "average_percent": (
                        sum(memory_values) / len(memory_values) if memory_values else 0
                    ),
                    "max_percent": max(memory_values) if memory_values else 0,
                    "available_mb": (
                        recent_metrics[-1].memory_available_mb if recent_metrics else 0
                    ),
                },
                "disk": {
                    "usage_percent": (
                        recent_metrics[-1].disk_usage_percent if recent_metrics else 0
                    ),
                    "free_gb": recent_metrics[-1].disk_free_gb if recent_metrics else 0,
                },
            }

        except Exception as e:
            logger.error(f"Error getting system metrics summary: {e}")
            return {"error": str(e)}

    def get_training_metrics_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get training metrics summary"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)

            recent_metrics = [
                m
                for m in self.training_metrics
                if datetime.fromisoformat(m.timestamp) >= cutoff_time
            ]

            if not recent_metrics:
                return {"training_sessions": 0, "time_range_hours": hours_back}

            # Group by model type
            by_model_type = defaultdict(list)
            for metric in recent_metrics:
                by_model_type[metric.model_type].append(metric)

            summary = {
                "time_range_hours": hours_back,
                "total_training_sessions": len(recent_metrics),
                "by_model_type": {},
            }

            for model_type, metrics in by_model_type.items():
                training_times = [m.training_time_seconds for m in metrics]
                memory_peaks = [m.memory_peak_mb for m in metrics]

                summary["by_model_type"][model_type] = {
                    "sessions": len(metrics),
                    "avg_training_time_seconds": sum(training_times)
                    / len(training_times),
                    "total_training_time_seconds": sum(training_times),
                    "avg_memory_peak_mb": sum(memory_peaks) / len(memory_peaks),
                    "max_memory_peak_mb": max(memory_peaks),
                    "sensors_trained": len(set(m.sensor_id for m in metrics)),
                }

            return summary

        except Exception as e:
            logger.error(f"Error getting training metrics summary: {e}")
            return {"error": str(e)}

    def get_inference_metrics_summary(self, hours_back: int = 1) -> Dict[str, Any]:
        """Get inference metrics summary"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)

            recent_metrics = [
                m
                for m in self.inference_metrics
                if datetime.fromisoformat(m.timestamp) >= cutoff_time
            ]

            if not recent_metrics:
                return {"inference_requests": 0, "time_range_hours": hours_back}

            # Group by model type
            by_model_type = defaultdict(list)
            for metric in recent_metrics:
                by_model_type[metric.model_type].append(metric)

            summary = {
                "time_range_hours": hours_back,
                "total_inference_requests": len(recent_metrics),
                "by_model_type": {},
            }

            for model_type, metrics in by_model_type.items():
                inference_times = [m.inference_time_ms for m in metrics]
                throughputs = [m.throughput_samples_per_sec for m in metrics]
                accuracies = [m.accuracy for m in metrics if m.accuracy > 0]

                summary["by_model_type"][model_type] = {
                    "requests": len(metrics),
                    "avg_inference_time_ms": sum(inference_times)
                    / len(inference_times),
                    "max_inference_time_ms": max(inference_times),
                    "min_inference_time_ms": min(inference_times),
                    "avg_throughput": sum(throughputs) / len(throughputs),
                    "avg_accuracy": (
                        sum(accuracies) / len(accuracies) if accuracies else 0
                    ),
                    "unique_sensors": len(set(m.sensor_id for m in metrics)),
                }

            return summary

        except Exception as e:
            logger.error(f"Error getting inference metrics summary: {e}")
            return {"error": str(e)}

    def get_performance_alerts(self) -> List[Dict[str, Any]]:
        """Get performance alerts based on thresholds"""
        alerts = []

        try:
            # Check recent system metrics
            if self.system_metrics:
                latest_system = self.system_metrics[-1]

                # CPU alert
                if latest_system.cpu_percent > 90:
                    alerts.append(
                        {
                            "type": "HIGH_CPU",
                            "severity": "critical",
                            "message": f"High CPU usage: {latest_system.cpu_percent:.1f}%",
                            "timestamp": latest_system.timestamp,
                        }
                    )

                # Memory alert
                if latest_system.memory_percent > 85:
                    alerts.append(
                        {
                            "type": "HIGH_MEMORY",
                            "severity": "warning",
                            "message": f"High memory usage: {latest_system.memory_percent:.1f}%",
                            "timestamp": latest_system.timestamp,
                        }
                    )

                # Disk space alert
                if latest_system.disk_usage_percent > 90:
                    alerts.append(
                        {
                            "type": "LOW_DISK_SPACE",
                            "severity": "critical",
                            "message": f"Low disk space: {latest_system.disk_free_gb:.1f}GB free",
                            "timestamp": latest_system.timestamp,
                        }
                    )

            # Check recent training performance
            recent_training = (
                list(self.training_metrics)[-10:] if self.training_metrics else []
            )
            slow_training = [
                m for m in recent_training if m.training_time_seconds > 300
            ]  # 5 minutes

            if len(slow_training) > 3:
                alerts.append(
                    {
                        "type": "SLOW_TRAINING",
                        "severity": "warning",
                        "message": f"{len(slow_training)} recent training sessions took over 5 minutes",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            # Check recent inference performance
            recent_inference = (
                list(self.inference_metrics)[-50:] if self.inference_metrics else []
            )
            slow_inference = [
                m for m in recent_inference if m.inference_time_ms > 1000
            ]  # 1 second

            if len(slow_inference) > 10:
                alerts.append(
                    {
                        "type": "SLOW_INFERENCE",
                        "severity": "warning",
                        "message": f"{len(slow_inference)} recent inference requests took over 1 second",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        except Exception as e:
            logger.error(f"Error getting performance alerts: {e}")

        return alerts

    def export_metrics(self, output_file: str = None, format: str = "json") -> str:
        """Export all metrics to file"""
        try:
            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = self.metrics_dir / f"metrics_export_{timestamp}.{format}"

            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "system_metrics": [asdict(m) for m in self.system_metrics],
                "training_metrics": [asdict(m) for m in self.training_metrics],
                "inference_metrics": [asdict(m) for m in self.inference_metrics],
                "custom_metrics": {
                    name: [asdict(m) for m in metrics]
                    for name, metrics in self.custom_metrics.items()
                },
            }

            if format == "json":
                with open(output_file, "w") as f:
                    json.dump(export_data, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Metrics exported to {output_file}")
            return str(output_file)

        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            raise

    def clear_old_metrics(self, days_back: int = 7):
        """Clear metrics older than specified days"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days_back)

            # Clear old system metrics
            self.system_metrics = deque(
                [
                    m
                    for m in self.system_metrics
                    if datetime.fromisoformat(m.timestamp) >= cutoff_time
                ],
                maxlen=self.system_metrics.maxlen,
            )

            # Clear old training metrics
            self.training_metrics = deque(
                [
                    m
                    for m in self.training_metrics
                    if datetime.fromisoformat(m.timestamp) >= cutoff_time
                ],
                maxlen=self.training_metrics.maxlen,
            )

            # Clear old inference metrics
            self.inference_metrics = deque(
                [
                    m
                    for m in self.inference_metrics
                    if datetime.fromisoformat(m.timestamp) >= cutoff_time
                ],
                maxlen=self.inference_metrics.maxlen,
            )

            # Clear old custom metrics
            for name, metrics in self.custom_metrics.items():
                self.custom_metrics[name] = deque(
                    [
                        m
                        for m in metrics
                        if datetime.fromisoformat(m.timestamp) >= cutoff_time
                    ],
                    maxlen=metrics.maxlen,
                )

            logger.info(f"Cleared metrics older than {days_back} days")

        except Exception as e:
            logger.error(f"Error clearing old metrics: {e}")

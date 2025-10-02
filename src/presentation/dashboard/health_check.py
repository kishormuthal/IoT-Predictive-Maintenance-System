"""
Health Check Endpoint for Dashboard
Provides health status for Docker/Kubernetes health checks
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def check_nasa_data() -> bool:
    """Check if NASA data is available"""
    try:
        from src.infrastructure.data.nasa_data_loader import NASADataLoader

        loader = NASADataLoader()
        return loader.is_loaded
    except Exception as e:
        logger.warning(f"NASA data check failed: {e}")
        return False


def check_services() -> Dict[str, bool]:
    """Check availability of core services"""
    services_status = {}

    try:
        from src.core.services.anomaly_service import AnomalyDetectionService

        services_status["anomaly_service"] = True
    except Exception:
        services_status["anomaly_service"] = False

    try:
        from src.core.services.forecasting_service import ForecastingService

        services_status["forecasting_service"] = True
    except Exception:
        services_status["forecasting_service"] = False

    return services_status


def get_health_status() -> Dict[str, Any]:
    """
    Get comprehensive health status

    Returns:
        Dictionary with health information
    """
    nasa_data_available = check_nasa_data()
    services = check_services()

    # Overall health is OK if NASA data loads and at least one service works
    healthy = nasa_data_available and any(services.values())

    return {
        "status": "healthy" if healthy else "degraded",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "checks": {"nasa_data": nasa_data_available, "services": services},
    }


def add_health_check(app):
    """
    Add health check endpoint to Dash app

    Args:
        app: Dash application instance
    """

    @app.server.route("/health")
    def health():
        """Health check endpoint for container orchestration"""
        try:
            status = get_health_status()

            if status["status"] == "healthy":
                return status, 200
            else:
                return status, 503  # Service Unavailable

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }, 500

    @app.server.route("/health/ready")
    def readiness():
        """Readiness check for Kubernetes"""
        try:
            # Check if app is ready to serve requests
            status = get_health_status()
            if status["checks"]["nasa_data"]:
                return {"status": "ready"}, 200
            else:
                return {"status": "not_ready", "reason": "nasa_data_not_loaded"}, 503
        except Exception as e:
            return {"status": "error", "error": str(e)}, 500

    @app.server.route("/health/live")
    def liveness():
        """Liveness check for Kubernetes"""
        # Simple check - if this endpoint responds, app is alive
        return {"status": "alive", "timestamp": datetime.now().isoformat()}, 200

    logger.info("✓ Health check endpoints registered: /health, /health/ready, /health/live")


def add_metrics_endpoint(app):
    """
    Add Prometheus-compatible metrics endpoint (optional)

    Args:
        app: Dash application instance
    """

    @app.server.route("/metrics")
    def metrics():
        """Prometheus metrics endpoint"""
        try:
            status = get_health_status()

            # Simple metrics in Prometheus format
            metrics_text = f"""# HELP iot_dashboard_health Dashboard health status (1=healthy, 0=unhealthy)
# TYPE iot_dashboard_health gauge
iot_dashboard_health{{version="1.0.0"}} {1 if status['status'] == 'healthy' else 0}

# HELP iot_nasa_data_available NASA data availability (1=available, 0=unavailable)
# TYPE iot_nasa_data_available gauge
iot_nasa_data_available {1 if status['checks']['nasa_data'] else 0}

# HELP iot_service_available Service availability (1=available, 0=unavailable)
# TYPE iot_service_available gauge
iot_service_available{{service="anomaly_detection"}} {1 if status['checks']['services'].get('anomaly_service', False) else 0}
iot_service_available{{service="forecasting"}} {1 if status['checks']['services'].get('forecasting_service', False) else 0}
"""

            return metrics_text, 200, {"Content-Type": "text/plain; charset=utf-8"}

        except Exception as e:
            logger.error(f"Metrics endpoint failed: {e}")
            return f"# Error generating metrics: {e}", 500

    logger.info("✓ Metrics endpoint registered: /metrics")

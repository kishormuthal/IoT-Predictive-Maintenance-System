"""
Alert System and Real-time Notifications
Enhanced alert management and notification system for the dashboard
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback_context, dcc, html

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertCategory(Enum):
    """Alert categories"""

    SYSTEM = "system"
    TRAINING = "training"
    ANOMALY = "anomaly"
    PERFORMANCE = "performance"
    MODEL = "model"


@dataclass
class Alert:
    """Alert data structure"""

    id: str
    title: str
    message: str
    severity: AlertSeverity
    category: AlertCategory
    timestamp: str
    source: str
    acknowledged: bool = False
    auto_dismiss: bool = True
    dismiss_after_seconds: int = 10
    actions: List[Dict[str, str]] = None

    def __post_init__(self):
        if self.actions is None:
            self.actions = []


class AlertManager:
    """
    Centralized alert management system
    """

    def __init__(self):
        self.alerts: List[Alert] = []
        self.max_alerts = 100
        self.alert_counter = 0

    def create_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        category: AlertCategory,
        source: str,
        auto_dismiss: bool = True,
        dismiss_after_seconds: int = 10,
        actions: List[Dict[str, str]] = None,
    ) -> str:
        """Create a new alert"""
        self.alert_counter += 1
        alert_id = f"alert_{self.alert_counter}_{int(datetime.now().timestamp())}"

        alert = Alert(
            id=alert_id,
            title=title,
            message=message,
            severity=severity,
            category=category,
            timestamp=datetime.now().isoformat(),
            source=source,
            auto_dismiss=auto_dismiss,
            dismiss_after_seconds=dismiss_after_seconds,
            actions=actions or [],
        )

        self.alerts.insert(0, alert)  # Add to beginning

        # Keep only max_alerts
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[: self.max_alerts]

        logger.info(f"Created alert: {alert_id} - {title}")
        return alert_id

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                logger.info(f"Acknowledged alert: {alert_id}")
                return True
        return False

    def dismiss_alert(self, alert_id: str) -> bool:
        """Dismiss an alert"""
        for i, alert in enumerate(self.alerts):
            if alert.id == alert_id:
                del self.alerts[i]
                logger.info(f"Dismissed alert: {alert_id}")
                return True
        return False

    def get_active_alerts(
        self, category: Optional[AlertCategory] = None
    ) -> List[Alert]:
        """Get active alerts, optionally filtered by category"""
        alerts = [alert for alert in self.alerts if not alert.acknowledged]

        if category:
            alerts = [alert for alert in alerts if alert.category == category]

        return alerts

    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get alerts by severity level"""
        return [alert for alert in self.alerts if alert.severity == severity]

    def clear_old_alerts(self, hours_old: int = 24):
        """Clear alerts older than specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours_old)

        self.alerts = [
            alert
            for alert in self.alerts
            if datetime.fromisoformat(alert.timestamp) >= cutoff_time
        ]


# Global alert manager instance
alert_manager = AlertManager()


def create_alert_system_components():
    """Create alert system UI components"""
    return html.Div(
        [
            # Alert notifications area (top-right)
            html.Div(
                id="alert-notifications-area",
                className="position-fixed",
                style={
                    "top": "80px",
                    "right": "20px",
                    "z-index": "9999",
                    "max-width": "400px",
                    "max-height": "80vh",
                    "overflow-y": "auto",
                },
            ),
            # Alert management modal
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Alert Management")),
                    dbc.ModalBody([html.Div(id="alert-management-content")]),
                    dbc.ModalFooter(
                        [
                            dbc.Button(
                                "Clear All",
                                id="clear-all-alerts-btn",
                                color="warning",
                                className="me-2",
                            ),
                            dbc.Button(
                                "Close", id="close-alert-modal", className="ms-auto"
                            ),
                        ]
                    ),
                ],
                id="alert-management-modal",
                size="lg",
            ),
            # Alert summary badge (can be placed in header)
            html.Div(
                [
                    dbc.Button(
                        [
                            html.I(className="fas fa-bell me-2"),
                            dbc.Badge(
                                id="alert-count-badge", color="danger", className="ms-1"
                            ),
                        ],
                        id="alert-summary-btn",
                        color="outline-secondary",
                        size="sm",
                    )
                ],
                id="alert-summary-component",
            ),
            # Hidden stores for alert data
            dcc.Store(id="alert-store", data=[]),
            dcc.Store(
                id="alert-config-store",
                data={
                    "show_notifications": True,
                    "auto_dismiss": True,
                    "sound_enabled": False,
                },
            ),
            # Interval for alert updates
            dcc.Interval(id="alert-update-interval", interval=2000, n_intervals=0),
        ]
    )


def register_alert_system_callbacks(
    app, performance_monitor, training_use_case, anomaly_service
):
    """Register alert system callbacks"""

    @app.callback(
        [Output("alert-store", "data"), Output("alert-count-badge", "children")],
        [
            Input("alert-update-interval", "n_intervals"),
            Input("global-refresh", "n_intervals"),
        ],
    )
    def update_alert_store(alert_intervals, global_intervals):
        """Update alert store with new alerts"""
        try:
            # Generate alerts based on system status
            _generate_system_alerts(
                performance_monitor, training_use_case, anomaly_service
            )

            # Get current alerts
            active_alerts = alert_manager.get_active_alerts()
            alert_data = [asdict(alert) for alert in active_alerts]

            # Count by severity
            alert_count = len(active_alerts)
            critical_count = len(
                [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
            )

            # Display count (show critical count if any, otherwise total)
            display_count = (
                str(critical_count)
                if critical_count > 0
                else str(alert_count) if alert_count > 0 else ""
            )

            return alert_data, display_count

        except Exception as e:
            logger.error(f"Error updating alert store: {e}")
            return [], ""

    @app.callback(
        Output("alert-notifications-area", "children"),
        [Input("alert-store", "data")],
        [State("alert-config-store", "data")],
    )
    def update_alert_notifications(alert_data, alert_config):
        """Update alert notification display"""
        try:
            if not alert_config.get("show_notifications", True):
                return []

            # Show only the most recent 5 alerts
            recent_alerts = alert_data[:5]
            notifications = []

            for alert_dict in recent_alerts:
                alert = Alert(**alert_dict)

                if alert.acknowledged:
                    continue

                # Map severity to toast props
                severity_map = {
                    AlertSeverity.INFO: {"icon": "info", "header_color": "info"},
                    AlertSeverity.WARNING: {
                        "icon": "warning",
                        "header_color": "warning",
                    },
                    AlertSeverity.ERROR: {"icon": "danger", "header_color": "danger"},
                    AlertSeverity.CRITICAL: {
                        "icon": "danger",
                        "header_color": "danger",
                    },
                }

                props = severity_map.get(
                    alert.severity, {"icon": "secondary", "header_color": "secondary"}
                )

                # Create action buttons
                action_buttons = []
                for action in alert.actions:
                    action_buttons.append(
                        dbc.Button(
                            action.get("label", "Action"),
                            id=f"alert-action-{alert.id}-{action.get('id', 'default')}",
                            size="sm",
                            color="outline-primary",
                            className="me-1",
                        )
                    )

                # Create toast notification
                toast = dbc.Toast(
                    [
                        html.P(alert.message, className="mb-2"),
                        html.Div(
                            [
                                html.Small(
                                    f"Source: {alert.source}",
                                    className="text-muted me-3",
                                ),
                                html.Small(
                                    f"Category: {alert.category.value.title()}",
                                    className="text-muted",
                                ),
                            ],
                            className="mb-2",
                        ),
                        html.Div(action_buttons) if action_buttons else None,
                        html.Div(
                            [
                                dbc.Button(
                                    "Acknowledge",
                                    id=f"ack-alert-{alert.id}",
                                    size="sm",
                                    color="success",
                                    className="me-1",
                                ),
                                dbc.Button(
                                    "Dismiss",
                                    id=f"dismiss-alert-{alert.id}",
                                    size="sm",
                                    color="secondary",
                                ),
                            ],
                            className="mt-2",
                        ),
                    ],
                    id=f"alert-toast-{alert.id}",
                    header=alert.title,
                    icon=props["icon"],
                    dismissable=True,
                    is_open=True,
                    duration=(
                        alert.dismiss_after_seconds * 1000
                        if alert.auto_dismiss
                        else False
                    ),
                    style={"margin-bottom": "10px", "max-width": "380px"},
                )

                notifications.append(toast)

            return notifications

        except Exception as e:
            logger.error(f"Error updating alert notifications: {e}")
            return []

    @app.callback(
        Output("alert-management-modal", "is_open"),
        [
            Input("alert-summary-btn", "n_clicks"),
            Input("close-alert-modal", "n_clicks"),
        ],
        [State("alert-management-modal", "is_open")],
    )
    def toggle_alert_modal(summary_clicks, close_clicks, is_open):
        """Toggle alert management modal"""
        if summary_clicks or close_clicks:
            return not is_open
        return is_open

    @app.callback(
        Output("alert-management-content", "children"),
        [Input("alert-management-modal", "is_open")],
        [State("alert-store", "data")],
    )
    def update_alert_management_content(is_open, alert_data):
        """Update alert management modal content"""
        try:
            if not is_open:
                return []

            if not alert_data:
                return dbc.Alert("No active alerts", color="success")

            # Create alert management table
            alert_items = []
            for alert_dict in alert_data:
                alert = Alert(**alert_dict)

                # Severity badge
                severity_colors = {
                    AlertSeverity.INFO: "info",
                    AlertSeverity.WARNING: "warning",
                    AlertSeverity.ERROR: "danger",
                    AlertSeverity.CRITICAL: "danger",
                }

                # Category badge
                category_colors = {
                    AlertCategory.SYSTEM: "primary",
                    AlertCategory.TRAINING: "success",
                    AlertCategory.ANOMALY: "warning",
                    AlertCategory.PERFORMANCE: "info",
                    AlertCategory.MODEL: "secondary",
                }

                alert_item = dbc.ListGroupItem(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.H6(alert.title, className="mb-1"),
                                        html.P(alert.message, className="mb-2"),
                                        html.Div(
                                            [
                                                dbc.Badge(
                                                    alert.severity.value.upper(),
                                                    color=severity_colors.get(
                                                        alert.severity, "secondary"
                                                    ),
                                                    className="me-2",
                                                ),
                                                dbc.Badge(
                                                    alert.category.value.upper(),
                                                    color=category_colors.get(
                                                        alert.category, "secondary"
                                                    ),
                                                    className="me-2",
                                                ),
                                                html.Small(
                                                    f"Source: {alert.source}",
                                                    className="text-muted",
                                                ),
                                            ]
                                        ),
                                    ],
                                    width=8,
                                ),
                                dbc.Col(
                                    [
                                        html.Small(
                                            alert.timestamp, className="text-muted"
                                        ),
                                        html.Br(),
                                        dbc.ButtonGroup(
                                            [
                                                dbc.Button(
                                                    "Ack",
                                                    id=f"modal-ack-{alert.id}",
                                                    size="sm",
                                                    color="success",
                                                ),
                                                dbc.Button(
                                                    "Dismiss",
                                                    id=f"modal-dismiss-{alert.id}",
                                                    size="sm",
                                                    color="secondary",
                                                ),
                                            ],
                                            size="sm",
                                            className="mt-2",
                                        ),
                                    ],
                                    width=4,
                                ),
                            ]
                        )
                    ],
                    color="light" if alert.acknowledged else None,
                )

                alert_items.append(alert_item)

            return dbc.ListGroup(alert_items, flush=True)

        except Exception as e:
            logger.error(f"Error updating alert management content: {e}")
            return dbc.Alert("Error loading alerts", color="danger")

    # Dynamic callbacks for alert actions
    @app.callback(
        Output("alert-store", "data", allow_duplicate=True),
        [Input({"type": "alert-action", "index": dash.dependencies.ALL}, "n_clicks")],
        [State("alert-store", "data")],
        prevent_initial_call=True,
    )
    def handle_alert_actions(action_clicks, alert_data):
        """Handle alert acknowledgment and dismissal actions"""
        try:
            ctx = callback_context
            if not ctx.triggered:
                return dash.no_update

            # Parse the triggered component
            triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

            if "ack-alert-" in triggered_id:
                alert_id = triggered_id.replace("ack-alert-", "")
                alert_manager.acknowledge_alert(alert_id)
            elif "dismiss-alert-" in triggered_id:
                alert_id = triggered_id.replace("dismiss-alert-", "")
                alert_manager.dismiss_alert(alert_id)

            # Return updated alert data
            active_alerts = alert_manager.get_active_alerts()
            return [asdict(alert) for alert in active_alerts]

        except Exception as e:
            logger.error(f"Error handling alert actions: {e}")
            return dash.no_update

    return app


def _generate_system_alerts(performance_monitor, training_use_case, anomaly_service):
    """Generate alerts based on system status"""
    try:
        # Performance alerts
        performance_alerts = performance_monitor.get_performance_alerts()
        for perf_alert in performance_alerts:
            severity_map = {
                "critical": AlertSeverity.CRITICAL,
                "warning": AlertSeverity.WARNING,
                "info": AlertSeverity.INFO,
            }

            alert_manager.create_alert(
                title=f"Performance Alert: {perf_alert.get('type', 'UNKNOWN')}",
                message=perf_alert.get("message", "Performance issue detected"),
                severity=severity_map.get(
                    perf_alert.get("severity", "info"), AlertSeverity.INFO
                ),
                category=AlertCategory.PERFORMANCE,
                source="Performance Monitor",
                auto_dismiss=False,
            )

        # Training status alerts
        training_status = training_use_case.get_training_status()
        equipment_status = training_status.get("equipment_status", {})

        untrained_count = sum(
            1
            for status in equipment_status.values()
            if not status.get("anomaly_detection", {}).get("trained")
            and not status.get("forecasting", {}).get("trained")
        )

        if untrained_count > 0:
            alert_manager.create_alert(
                title="Training Required",
                message=f"{untrained_count} sensors need model training",
                severity=AlertSeverity.WARNING,
                category=AlertCategory.TRAINING,
                source="Training Manager",
                actions=[{"id": "train_all", "label": "Train All Models"}],
            )

        # Anomaly alerts
        anomaly_summary = anomaly_service.get_detection_summary()
        critical_anomalies = [
            a
            for a in anomaly_summary.get("recent_anomalies", [])
            if a.get("severity") == "CRITICAL"
        ]

        for anomaly in critical_anomalies[:3]:  # Limit to 3 most recent
            alert_manager.create_alert(
                title=f"Critical Anomaly: {anomaly.get('sensor_id')}",
                message=f"Critical anomaly detected with score {anomaly.get('score', 0):.3f}",
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.ANOMALY,
                source="Anomaly Detection",
                actions=[
                    {"id": "investigate", "label": "Investigate"},
                    {"id": "acknowledge", "label": "Acknowledge"},
                ],
            )

    except Exception as e:
        logger.error(f"Error generating system alerts: {e}")


# Utility functions for creating specific alert types
def create_training_alert(message: str, sensor_id: str = None):
    """Create a training-related alert"""
    title = f"Training Alert: {sensor_id}" if sensor_id else "Training Alert"
    return alert_manager.create_alert(
        title=title,
        message=message,
        severity=AlertSeverity.INFO,
        category=AlertCategory.TRAINING,
        source="Training System",
    )


def create_anomaly_alert(sensor_id: str, score: float, severity: str = "warning"):
    """Create an anomaly detection alert"""
    severity_map = {
        "low": AlertSeverity.INFO,
        "medium": AlertSeverity.WARNING,
        "high": AlertSeverity.ERROR,
        "critical": AlertSeverity.CRITICAL,
    }

    return alert_manager.create_alert(
        title=f"Anomaly Detected: {sensor_id}",
        message=f"Anomaly detected with score {score:.3f}",
        severity=severity_map.get(severity.lower(), AlertSeverity.WARNING),
        category=AlertCategory.ANOMALY,
        source="Anomaly Detection System",
        actions=[
            {"id": "investigate", "label": "Investigate"},
            {"id": "acknowledge", "label": "Acknowledge"},
        ],
    )


def create_model_alert(sensor_id: str, model_type: str, message: str):
    """Create a model-related alert"""
    return alert_manager.create_alert(
        title=f"Model Alert: {sensor_id}",
        message=f"{model_type} model: {message}",
        severity=AlertSeverity.WARNING,
        category=AlertCategory.MODEL,
        source="Model Registry",
    )

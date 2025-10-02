"""
System Administration Dashboard Component
System administration and configuration interface
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dcc, html

logger = logging.getLogger(__name__)


def create_system_admin_layout():
    """Create the system administration layout"""
    return dbc.Container(
        [
            # System Admin Header
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H4(
                                                [
                                                    html.I(
                                                        className="fas fa-cogs me-3 text-secondary"
                                                    ),
                                                    "System Administration",
                                                ],
                                                className="mb-3",
                                            ),
                                            html.P(
                                                "Configure system settings, manage components, and perform maintenance tasks.",
                                                className="text-muted mb-0",
                                            ),
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ],
                className="mb-4",
            ),
            # Quick System Actions
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.I(className="fas fa-tools me-2"),
                                            "Quick Actions",
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dbc.Button(
                                                                [
                                                                    html.I(
                                                                        className="fas fa-sync me-2"
                                                                    ),
                                                                    "System Health Check",
                                                                ],
                                                                id="health-check-btn",
                                                                color="primary",
                                                                size="lg",
                                                                className="w-100 mb-2",
                                                            ),
                                                            html.Small(
                                                                "Run comprehensive system diagnostics",
                                                                className="text-muted",
                                                            ),
                                                        ],
                                                        width=3,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Button(
                                                                [
                                                                    html.I(
                                                                        className="fas fa-broom me-2"
                                                                    ),
                                                                    "Cleanup & Maintenance",
                                                                ],
                                                                id="cleanup-btn",
                                                                color="warning",
                                                                size="lg",
                                                                className="w-100 mb-2",
                                                            ),
                                                            html.Small(
                                                                "Clean logs, cache, and temporary files",
                                                                className="text-muted",
                                                            ),
                                                        ],
                                                        width=3,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Button(
                                                                [
                                                                    html.I(
                                                                        className="fas fa-download me-2"
                                                                    ),
                                                                    "Backup System",
                                                                ],
                                                                id="backup-btn",
                                                                color="success",
                                                                size="lg",
                                                                className="w-100 mb-2",
                                                            ),
                                                            html.Small(
                                                                "Create system backup including models",
                                                                className="text-muted",
                                                            ),
                                                        ],
                                                        width=3,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Button(
                                                                [
                                                                    html.I(
                                                                        className="fas fa-chart-bar me-2"
                                                                    ),
                                                                    "Generate Report",
                                                                ],
                                                                id="report-btn",
                                                                color="info",
                                                                size="lg",
                                                                className="w-100 mb-2",
                                                            ),
                                                            html.Small(
                                                                "Generate comprehensive system report",
                                                                className="text-muted",
                                                            ),
                                                        ],
                                                        width=3,
                                                    ),
                                                ]
                                            )
                                        ]
                                    ),
                                ]
                            )
                        ]
                    )
                ],
                className="mb-4",
            ),
            # System Status Dashboard
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.I(className="fas fa-heartbeat me-2"),
                                            "System Health Dashboard",
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="system-health-dashboard")]
                                    ),
                                ]
                            )
                        ],
                        width=8,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.I(className="fas fa-list-ul me-2"),
                                            "System Information",
                                        ]
                                    ),
                                    dbc.CardBody([html.Div(id="system-info-panel")]),
                                ]
                            )
                        ],
                        width=4,
                    ),
                ],
                className="mb-4",
            ),
            # Configuration Management
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.I(className="fas fa-sliders-h me-2"),
                                            "Configuration Management",
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dbc.Tabs(
                                                [
                                                    dbc.Tab(
                                                        label="📋 General Settings",
                                                        tab_id="general-config",
                                                    ),
                                                    dbc.Tab(
                                                        label="🤖 Training Config",
                                                        tab_id="training-config",
                                                    ),
                                                    dbc.Tab(
                                                        label="🔧 Equipment Settings",
                                                        tab_id="equipment-config",
                                                    ),
                                                    dbc.Tab(
                                                        label="📊 Monitoring Config",
                                                        tab_id="monitoring-config",
                                                    ),
                                                ],
                                                id="config-tabs",
                                                active_tab="general-config",
                                            ),
                                            html.Div(
                                                id="config-content", className="mt-3"
                                            ),
                                        ]
                                    ),
                                ]
                            )
                        ]
                    )
                ],
                className="mb-4",
            ),
            # Logs and Diagnostics
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.I(className="fas fa-file-alt me-2"),
                                            "System Logs",
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Log Level:",
                                                                className="fw-bold",
                                                            ),
                                                            dcc.Dropdown(
                                                                id="log-level-filter",
                                                                options=[
                                                                    {
                                                                        "label": "All Levels",
                                                                        "value": "all",
                                                                    },
                                                                    {
                                                                        "label": "Error",
                                                                        "value": "error",
                                                                    },
                                                                    {
                                                                        "label": "Warning",
                                                                        "value": "warning",
                                                                    },
                                                                    {
                                                                        "label": "Info",
                                                                        "value": "info",
                                                                    },
                                                                    {
                                                                        "label": "Debug",
                                                                        "value": "debug",
                                                                    },
                                                                ],
                                                                value="all",
                                                            ),
                                                        ],
                                                        width=3,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Component:",
                                                                className="fw-bold",
                                                            ),
                                                            dcc.Dropdown(
                                                                id="log-component-filter",
                                                                options=[
                                                                    {
                                                                        "label": "All Components",
                                                                        "value": "all",
                                                                    },
                                                                    {
                                                                        "label": "Training Pipeline",
                                                                        "value": "training",
                                                                    },
                                                                    {
                                                                        "label": "Anomaly Detection",
                                                                        "value": "anomaly",
                                                                    },
                                                                    {
                                                                        "label": "Forecasting",
                                                                        "value": "forecasting",
                                                                    },
                                                                    {
                                                                        "label": "Dashboard",
                                                                        "value": "dashboard",
                                                                    },
                                                                ],
                                                                value="all",
                                                            ),
                                                        ],
                                                        width=3,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Time Range:",
                                                                className="fw-bold",
                                                            ),
                                                            dcc.Dropdown(
                                                                id="log-time-filter",
                                                                options=[
                                                                    {
                                                                        "label": "Last Hour",
                                                                        "value": 1,
                                                                    },
                                                                    {
                                                                        "label": "Last 6 Hours",
                                                                        "value": 6,
                                                                    },
                                                                    {
                                                                        "label": "Last 24 Hours",
                                                                        "value": 24,
                                                                    },
                                                                    {
                                                                        "label": "Last 7 Days",
                                                                        "value": 168,
                                                                    },
                                                                ],
                                                                value=24,
                                                            ),
                                                        ],
                                                        width=3,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Button(
                                                                [
                                                                    html.I(
                                                                        className="fas fa-sync me-2"
                                                                    ),
                                                                    "Refresh Logs",
                                                                ],
                                                                id="refresh-logs-btn",
                                                                color="outline-primary",
                                                                className="mt-4",
                                                            )
                                                        ],
                                                        width=3,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            html.Div(
                                                id="system-logs-display",
                                                style={
                                                    "height": "400px",
                                                    "overflow-y": "auto",
                                                },
                                            ),
                                        ]
                                    ),
                                ]
                            )
                        ],
                        width=8,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.I(className="fas fa-stethoscope me-2"),
                                            "Diagnostics",
                                        ]
                                    ),
                                    dbc.CardBody([html.Div(id="diagnostics-panel")]),
                                ]
                            )
                        ],
                        width=4,
                    ),
                ],
                className="mb-4",
            ),
            # Maintenance and Updates
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.I(className="fas fa-wrench me-2"),
                                            "Maintenance & Updates",
                                        ]
                                    ),
                                    dbc.CardBody([html.Div(id="maintenance-panel")]),
                                ]
                            )
                        ]
                    )
                ]
            ),
            # Hidden components for admin operations
            dcc.Store(id="admin-operation-status", data={}),
            dcc.Store(id="system-health-data", data={}),
            dcc.Interval(id="admin-refresh-interval", interval=10000, n_intervals=0),
        ]
    )


def register_system_admin_callbacks(
    app, performance_monitor, config_manager, model_registry, training_use_case
):
    """Register callbacks for system administration functionality"""

    @app.callback(
        [
            Output("system-health-dashboard", "children"),
            Output("system-health-data", "data"),
        ],
        Input("admin-refresh-interval", "n_intervals"),
    )
    def update_system_health_dashboard(n):
        """Update system health dashboard"""
        try:
            # Get system metrics
            system_summary = performance_monitor.get_system_metrics_summary(
                hours_back=1
            )
            alerts = performance_monitor.get_performance_alerts()
            training_status = training_use_case.get_training_status()
            registry_stats = model_registry.get_registry_stats()

            # Calculate health scores
            cpu_health = 100 - system_summary.get("cpu", {}).get("current", 0)
            memory_health = 100 - system_summary.get("memory", {}).get(
                "current_percent", 0
            )
            disk_health = 100 - system_summary.get("disk", {}).get("usage_percent", 0)
            alert_health = max(0, 100 - len(alerts) * 10)  # Reduce by 10 for each alert

            overall_health = (
                cpu_health + memory_health + disk_health + alert_health
            ) / 4

            # Create health dashboard
            health_components = dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5(
                                                "Overall Health",
                                                className="text-center",
                                            ),
                                            dcc.Graph(
                                                figure=create_health_gauge(
                                                    overall_health,
                                                    "Overall System Health",
                                                ),
                                                style={"height": "200px"},
                                            ),
                                        ]
                                    )
                                ]
                            )
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5(
                                                "Component Status", className="mb-3"
                                            ),
                                            dbc.ListGroup(
                                                [
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.I(
                                                                        className="fas fa-microchip me-2"
                                                                    ),
                                                                    "CPU",
                                                                    dbc.Badge(
                                                                        (
                                                                            "Healthy"
                                                                            if cpu_health
                                                                            > 70
                                                                            else (
                                                                                "Warning"
                                                                                if cpu_health
                                                                                > 50
                                                                                else "Critical"
                                                                            )
                                                                        ),
                                                                        color=(
                                                                            "success"
                                                                            if cpu_health
                                                                            > 70
                                                                            else (
                                                                                "warning"
                                                                                if cpu_health
                                                                                > 50
                                                                                else "danger"
                                                                            )
                                                                        ),
                                                                        className="ms-auto",
                                                                    ),
                                                                ],
                                                                className="d-flex justify-content-between align-items-center",
                                                            )
                                                        ]
                                                    ),
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.I(
                                                                        className="fas fa-memory me-2"
                                                                    ),
                                                                    "Memory",
                                                                    dbc.Badge(
                                                                        (
                                                                            "Healthy"
                                                                            if memory_health
                                                                            > 70
                                                                            else (
                                                                                "Warning"
                                                                                if memory_health
                                                                                > 50
                                                                                else "Critical"
                                                                            )
                                                                        ),
                                                                        color=(
                                                                            "success"
                                                                            if memory_health
                                                                            > 70
                                                                            else (
                                                                                "warning"
                                                                                if memory_health
                                                                                > 50
                                                                                else "danger"
                                                                            )
                                                                        ),
                                                                        className="ms-auto",
                                                                    ),
                                                                ],
                                                                className="d-flex justify-content-between align-items-center",
                                                            )
                                                        ]
                                                    ),
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.I(
                                                                        className="fas fa-hdd me-2"
                                                                    ),
                                                                    "Storage",
                                                                    dbc.Badge(
                                                                        (
                                                                            "Healthy"
                                                                            if disk_health
                                                                            > 70
                                                                            else (
                                                                                "Warning"
                                                                                if disk_health
                                                                                > 50
                                                                                else "Critical"
                                                                            )
                                                                        ),
                                                                        color=(
                                                                            "success"
                                                                            if disk_health
                                                                            > 70
                                                                            else (
                                                                                "warning"
                                                                                if disk_health
                                                                                > 50
                                                                                else "danger"
                                                                            )
                                                                        ),
                                                                        className="ms-auto",
                                                                    ),
                                                                ],
                                                                className="d-flex justify-content-between align-items-center",
                                                            )
                                                        ]
                                                    ),
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.I(
                                                                        className="fas fa-database me-2"
                                                                    ),
                                                                    "Models",
                                                                    dbc.Badge(
                                                                        f"{registry_stats.get('total_models', 0)} Active",
                                                                        color="info",
                                                                        className="ms-auto",
                                                                    ),
                                                                ],
                                                                className="d-flex justify-content-between align-items-center",
                                                            )
                                                        ]
                                                    ),
                                                ],
                                                flush=True,
                                            ),
                                        ]
                                    )
                                ]
                            )
                        ],
                        width=6,
                    ),
                ]
            )

            health_data = {
                "overall_health": overall_health,
                "cpu_health": cpu_health,
                "memory_health": memory_health,
                "disk_health": disk_health,
                "alert_count": len(alerts),
            }

            return health_components, health_data

        except Exception as e:
            logger.error(f"Error updating system health dashboard: {e}")
            return dbc.Alert("Error loading system health", color="danger"), {}

    @app.callback(
        Output("system-info-panel", "children"),
        Input("admin-refresh-interval", "n_intervals"),
    )
    def update_system_info(n):
        """Update system information panel"""
        try:
            import platform
            from pathlib import Path

            import psutil

            # Get system information
            system_info = dbc.ListGroup(
                [
                    dbc.ListGroupItem(
                        [
                            html.Strong("System: "),
                            f"{platform.system()} {platform.release()}",
                        ]
                    ),
                    dbc.ListGroupItem(
                        [html.Strong("Python: "), platform.python_version()]
                    ),
                    dbc.ListGroupItem(
                        [html.Strong("CPU Cores: "), str(psutil.cpu_count())]
                    ),
                    dbc.ListGroupItem(
                        [
                            html.Strong("Total Memory: "),
                            f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
                        ]
                    ),
                    dbc.ListGroupItem(
                        [
                            html.Strong("Uptime: "),
                            f"{datetime.now() - datetime.fromtimestamp(psutil.boot_time())}",
                        ]
                    ),
                    dbc.ListGroupItem(
                        [html.Strong("Dashboard Version: "), "3.0.0-enhanced"]
                    ),
                    dbc.ListGroupItem(
                        [
                            html.Strong("Last Updated: "),
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        ]
                    ),
                ],
                flush=True,
            )

            return system_info

        except Exception as e:
            logger.error(f"Error updating system info: {e}")
            return dbc.Alert("Error loading system info", color="warning")

    @app.callback(
        Output("config-content", "children"), Input("config-tabs", "active_tab")
    )
    def update_config_content(active_tab):
        """Update configuration content based on active tab"""
        try:
            if active_tab == "general-config":
                return _create_general_config_content()
            elif active_tab == "training-config":
                return _create_training_config_content()
            elif active_tab == "equipment-config":
                return _create_equipment_config_content()
            elif active_tab == "monitoring-config":
                return _create_monitoring_config_content()
            else:
                return dbc.Alert("Configuration section not available", color="warning")

        except Exception as e:
            logger.error(f"Error updating config content: {e}")
            return dbc.Alert(f"Error: {str(e)}", color="danger")

    def _create_general_config_content():
        """Create general configuration content"""
        return dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H6("System Settings", className="mb-3"),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label("Log Level:"),
                                                        dcc.Dropdown(
                                                            id="system-log-level",
                                                            options=[
                                                                {
                                                                    "label": "DEBUG",
                                                                    "value": "DEBUG",
                                                                },
                                                                {
                                                                    "label": "INFO",
                                                                    "value": "INFO",
                                                                },
                                                                {
                                                                    "label": "WARNING",
                                                                    "value": "WARNING",
                                                                },
                                                                {
                                                                    "label": "ERROR",
                                                                    "value": "ERROR",
                                                                },
                                                            ],
                                                            value="INFO",
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            "Refresh Interval (seconds):"
                                                        ),
                                                        dbc.Input(
                                                            id="refresh-interval-input",
                                                            type="number",
                                                            value=15,
                                                            min=5,
                                                            max=300,
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                            ],
                                            className="mb-3",
                                        ),
                                        dbc.Button(
                                            "Save Settings",
                                            id="save-general-config",
                                            color="primary",
                                        ),
                                    ]
                                )
                            ]
                        )
                    ]
                )
            ]
        )

    def _create_training_config_content():
        """Create training configuration content"""
        return dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H6(
                                            "Training Parameters", className="mb-3"
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            "Parallel Training:"
                                                        ),
                                                        dbc.Switch(
                                                            id="parallel-training-switch",
                                                            value=False,
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Max Workers:"),
                                                        dbc.Input(
                                                            id="max-workers-input",
                                                            type="number",
                                                            value=4,
                                                            min=1,
                                                            max=16,
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                            ],
                                            className="mb-3",
                                        ),
                                        dbc.Button(
                                            "Update Training Config",
                                            id="save-training-config",
                                            color="primary",
                                        ),
                                    ]
                                )
                            ]
                        )
                    ]
                )
            ]
        )

    def _create_equipment_config_content():
        """Create equipment configuration content"""
        return dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H6(
                                            "Equipment Management", className="mb-3"
                                        ),
                                        html.Div(id="equipment-config-table"),
                                        dbc.Button(
                                            "Refresh Equipment",
                                            id="refresh-equipment-btn",
                                            color="outline-primary",
                                            className="mt-3",
                                        ),
                                    ]
                                )
                            ]
                        )
                    ]
                )
            ]
        )

    def _create_monitoring_config_content():
        """Create monitoring configuration content"""
        return dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H6(
                                            "Monitoring Settings", className="mb-3"
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            "Performance Monitoring:"
                                                        ),
                                                        dbc.Switch(
                                                            id="performance-monitoring-switch",
                                                            value=True,
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            "Alert Threshold (CPU %):"
                                                        ),
                                                        dbc.Input(
                                                            id="cpu-alert-threshold",
                                                            type="number",
                                                            value=80,
                                                            min=50,
                                                            max=95,
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                            ],
                                            className="mb-3",
                                        ),
                                        dbc.Button(
                                            "Update Monitoring Config",
                                            id="save-monitoring-config",
                                            color="primary",
                                        ),
                                    ]
                                )
                            ]
                        )
                    ]
                )
            ]
        )

    @app.callback(
        Output("system-logs-display", "children"),
        [
            Input("refresh-logs-btn", "n_clicks"),
            Input("log-level-filter", "value"),
            Input("log-component-filter", "value"),
            Input("log-time-filter", "value"),
        ],
    )
    def update_system_logs(refresh_clicks, log_level, component, time_range):
        """Update system logs display"""
        try:
            # Mock system logs for demonstration
            log_entries = []
            current_time = datetime.now()

            for i in range(50):  # Generate 50 mock log entries
                timestamp = current_time - timedelta(minutes=i * 10)
                levels = ["INFO", "WARNING", "ERROR", "DEBUG"]
                components = ["training", "anomaly", "forecasting", "dashboard"]

                level = levels[i % len(levels)]
                comp = components[i % len(components)]

                messages = {
                    "training": f"Training completed for sensor SMAP-PWR-00{i%3+1}",
                    "anomaly": f"Anomaly detected in sensor MSL-TMP-00{i%3+1}",
                    "forecasting": f"Forecast generated for sensor SMAP-VIB-00{i%3+1}",
                    "dashboard": f"Dashboard refreshed - {i} components updated",
                }

                message = messages.get(comp, "System operation completed")

                # Apply filters
                if log_level != "all" and level.lower() != log_level:
                    continue
                if component != "all" and comp != component:
                    continue

                # Color coding for log levels
                color_map = {
                    "ERROR": "danger",
                    "WARNING": "warning",
                    "INFO": "info",
                    "DEBUG": "secondary",
                }

                log_entry = dbc.ListGroupItem(
                    [
                        html.Div(
                            [
                                html.Small(
                                    timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                                    className="text-muted me-3",
                                ),
                                dbc.Badge(
                                    level,
                                    color=color_map.get(level, "secondary"),
                                    className="me-2",
                                ),
                                dbc.Badge(
                                    comp.upper(),
                                    color="outline-primary",
                                    className="me-2",
                                ),
                                html.Span(message),
                            ]
                        )
                    ],
                    className="py-2",
                )

                log_entries.append(log_entry)

            if not log_entries:
                return dbc.Alert(
                    "No logs found matching the selected filters", color="info"
                )

            return dbc.ListGroup(log_entries[:20], flush=True)  # Show only first 20

        except Exception as e:
            logger.error(f"Error updating system logs: {e}")
            return dbc.Alert("Error loading system logs", color="danger")

    @app.callback(
        Output("diagnostics-panel", "children"),
        Input("admin-refresh-interval", "n_intervals"),
    )
    def update_diagnostics_panel(n):
        """Update diagnostics panel"""
        try:
            # Run system diagnostics
            diagnostics = dbc.ListGroup(
                [
                    dbc.ListGroupItem(
                        [
                            html.Div(
                                [
                                    html.I(
                                        className="fas fa-check-circle text-success me-2"
                                    ),
                                    "Database Connection",
                                    dbc.Badge(
                                        "OK", color="success", className="ms-auto"
                                    ),
                                ],
                                className="d-flex justify-content-between align-items-center",
                            )
                        ]
                    ),
                    dbc.ListGroupItem(
                        [
                            html.Div(
                                [
                                    html.I(
                                        className="fas fa-check-circle text-success me-2"
                                    ),
                                    "Model Registry",
                                    dbc.Badge(
                                        "OK", color="success", className="ms-auto"
                                    ),
                                ],
                                className="d-flex justify-content-between align-items-center",
                            )
                        ]
                    ),
                    dbc.ListGroupItem(
                        [
                            html.Div(
                                [
                                    html.I(
                                        className="fas fa-check-circle text-success me-2"
                                    ),
                                    "Data Loader",
                                    dbc.Badge(
                                        "OK", color="success", className="ms-auto"
                                    ),
                                ],
                                className="d-flex justify-content-between align-items-center",
                            )
                        ]
                    ),
                    dbc.ListGroupItem(
                        [
                            html.Div(
                                [
                                    html.I(
                                        className="fas fa-exclamation-triangle text-warning me-2"
                                    ),
                                    "Performance Monitor",
                                    dbc.Badge(
                                        "WARNING", color="warning", className="ms-auto"
                                    ),
                                ],
                                className="d-flex justify-content-between align-items-center",
                            )
                        ]
                    ),
                ],
                flush=True,
            )

            return diagnostics

        except Exception as e:
            logger.error(f"Error updating diagnostics: {e}")
            return dbc.Alert("Error running diagnostics", color="danger")

    @app.callback(
        Output("maintenance-panel", "children"),
        Input("admin-refresh-interval", "n_intervals"),
    )
    def update_maintenance_panel(n):
        """Update maintenance panel"""
        try:
            maintenance_info = dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H6(
                                                "System Maintenance", className="mb-3"
                                            ),
                                            dbc.ListGroup(
                                                [
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Div(
                                                                [
                                                                    "Last Backup:",
                                                                    html.Span(
                                                                        "2024-01-15 14:30:00",
                                                                        className="ms-auto text-muted",
                                                                    ),
                                                                ],
                                                                className="d-flex justify-content-between",
                                                            )
                                                        ]
                                                    ),
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Div(
                                                                [
                                                                    "Log Cleanup:",
                                                                    html.Span(
                                                                        "2024-01-16 02:00:00",
                                                                        className="ms-auto text-muted",
                                                                    ),
                                                                ],
                                                                className="d-flex justify-content-between",
                                                            )
                                                        ]
                                                    ),
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Div(
                                                                [
                                                                    "Model Cleanup:",
                                                                    html.Span(
                                                                        "2024-01-14 18:45:00",
                                                                        className="ms-auto text-muted",
                                                                    ),
                                                                ],
                                                                className="d-flex justify-content-between",
                                                            )
                                                        ]
                                                    ),
                                                ],
                                                flush=True,
                                            ),
                                        ]
                                    )
                                ]
                            )
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H6(
                                                "Scheduled Tasks", className="mb-3"
                                            ),
                                            dbc.ListGroup(
                                                [
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.I(
                                                                        className="fas fa-clock me-2"
                                                                    ),
                                                                    "Daily Backup",
                                                                    dbc.Badge(
                                                                        "Enabled",
                                                                        color="success",
                                                                        className="ms-auto",
                                                                    ),
                                                                ],
                                                                className="d-flex justify-content-between align-items-center",
                                                            )
                                                        ]
                                                    ),
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.I(
                                                                        className="fas fa-broom me-2"
                                                                    ),
                                                                    "Weekly Cleanup",
                                                                    dbc.Badge(
                                                                        "Enabled",
                                                                        color="success",
                                                                        className="ms-auto",
                                                                    ),
                                                                ],
                                                                className="d-flex justify-content-between align-items-center",
                                                            )
                                                        ]
                                                    ),
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.I(
                                                                        className="fas fa-sync me-2"
                                                                    ),
                                                                    "Model Validation",
                                                                    dbc.Badge(
                                                                        "Disabled",
                                                                        color="secondary",
                                                                        className="ms-auto",
                                                                    ),
                                                                ],
                                                                className="d-flex justify-content-between align-items-center",
                                                            )
                                                        ]
                                                    ),
                                                ],
                                                flush=True,
                                            ),
                                        ]
                                    )
                                ]
                            )
                        ],
                        width=6,
                    ),
                ]
            )

            return maintenance_info

        except Exception as e:
            logger.error(f"Error updating maintenance panel: {e}")
            return dbc.Alert("Error loading maintenance info", color="warning")

    return app


def create_health_gauge(value, title):
    """Create a health gauge chart"""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=value,
            title={"text": title},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": "darkgreen"},
                "steps": [
                    {"range": [0, 50], "color": "red"},
                    {"range": [50, 80], "color": "yellow"},
                    {"range": [80, 100], "color": "green"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 90,
                },
            },
        )
    )
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig

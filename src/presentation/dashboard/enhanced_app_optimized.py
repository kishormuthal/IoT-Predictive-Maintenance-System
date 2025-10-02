"""
Enhanced Dashboard Application - Optimized Version
Session 3: Full feature dashboard with state management fixes
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback, ctx, dcc, html, no_update

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from config.equipment_config import get_equipment_by_id, get_equipment_list
from src.application.services.training_config_manager import TrainingConfigManager
from src.application.use_cases.training_use_case import TrainingUseCase

# Import enhanced services
from src.core.services.anomaly_service import AnomalyDetectionService
from src.core.services.forecasting_service import ForecastingService
from src.infrastructure.data.nasa_data_loader import NASADataLoader
from src.infrastructure.ml.model_registry import ModelRegistry
from src.infrastructure.monitoring.performance_monitor import PerformanceMonitor

from .components.alert_system import AlertCategory, AlertManager, AlertSeverity

# Import ComponentEventBus for feature coordination
from .components.component_event_bus import (
    ComponentEventBus,
    EventType,
    emit_sensor_selection,
    get_event_bus,
)

# Import rich dashboard components (carefully selected to avoid hanging)
from .components.time_controls import TimeControlConfig, TimeControlManager, TimeRange

# Import simplified callbacks (AVOIDING problematic state managers)
from .enhanced_callbacks_simplified import (
    create_model_registry_layout,
    create_system_admin_layout,
    create_training_hub_layout,
    pipeline_dashboard,
)

logger = logging.getLogger(__name__)


class OptimizedIoTDashboard:
    """
    Optimized IoT Predictive Maintenance Dashboard
    Full feature set with state management optimization
    """

    def __init__(self, debug: bool = False):
        """Initialize optimized dashboard"""
        logger.info("Initializing Optimized IoT Dashboard...")

        # Configure assets path for proper CSS loading
        import os

        assets_path = os.path.join(os.path.dirname(__file__), "assets")

        self.app = dash.Dash(
            __name__,
            external_stylesheets=[
                dbc.themes.BOOTSTRAP,
                dbc.icons.FONT_AWESOME,
                dbc.icons.BOOTSTRAP,
            ],
            suppress_callback_exceptions=True,
            title="IoT Predictive Maintenance - Optimized Dashboard",
            assets_folder=assets_path,
        )

        # Initialize services with optimization
        self._initialize_services_optimized()

        # Load configuration and equipment
        self.equipment_list = get_equipment_list()
        self.sensor_ids = [eq.equipment_id for eq in self.equipment_list]

        # Initialize ComponentEventBus for feature coordination
        self.event_bus = get_event_bus()
        self.event_bus.register_component(
            "dashboard_core",
            {
                "type": "main_dashboard",
                "features": ["tab_management", "service_coordination"],
                "sensors": len(self.sensor_ids),
            },
        )

        # Initialize rich dashboard components
        self.time_control_manager = TimeControlManager(
            TimeControlConfig(
                default_range=TimeRange.LAST_24H,
                enable_realtime=True,
                enable_custom_range=True,
                show_refresh_button=True,
            )
        )

        self.alert_manager = AlertManager()

        # Register alert manager with event bus
        self.event_bus.register_component(
            "alert_manager",
            {
                "type": "alert_system",
                "features": ["real_time_alerts", "notification_management"],
            },
        )

        # Simplified dashboard state (NO complex state managers)
        self.dashboard_state = {
            "system_health": "operational",
            "last_update": datetime.now(),
            "active_alerts": [],
            "performance_metrics": {},
            "current_tab": "overview",
            "model_availability": {},
        }

        # Setup optimized layout and callbacks
        self._setup_optimized_layout()
        self._setup_optimized_callbacks()
        self._register_essential_callbacks()

        # Start performance monitoring safely
        self._start_performance_monitoring_safe()

        # Initialize model availability checking
        self._update_model_availability()

        logger.info("Optimized IoT Dashboard initialized successfully")

    def _initialize_services_optimized(self):
        """Initialize services with optimized approach"""
        import threading
        import time

        def safe_service_init(service_class, service_name, timeout=5):
            """Initialize service safely with timeout"""
            result = {"service": None, "success": False}

            def init_target():
                try:
                    result["service"] = service_class()
                    result["success"] = True
                except Exception as e:
                    logger.warning(f"{service_name} initialization failed: {e}")

            thread = threading.Thread(target=init_target)
            thread.daemon = True
            thread.start()
            thread.join(timeout)

            if thread.is_alive():
                logger.warning(f"{service_name} initialization timed out")
                return None
            return result["service"] if result["success"] else None

        try:
            logger.info("Initializing services with optimization...")

            # Core services
            self.data_loader = safe_service_init(NASADataLoader, "Data loader", 8) or NASADataLoader()
            self.anomaly_service = (
                safe_service_init(AnomalyDetectionService, "Anomaly service", 8) or AnomalyDetectionService()
            )
            self.forecasting_service = (
                safe_service_init(ForecastingService, "Forecasting service", 8) or ForecastingService()
            )

            # Optional services
            self.training_use_case = safe_service_init(TrainingUseCase, "Training use case", 3)
            self.config_manager = safe_service_init(TrainingConfigManager, "Config manager", 3)
            self.model_registry = safe_service_init(ModelRegistry, "Model registry", 3)
            self.performance_monitor = safe_service_init(PerformanceMonitor, "Performance monitor", 3)

            logger.info("Services initialized with optimization")

        except Exception as e:
            logger.error(f"Error in optimized service initialization: {e}")
            self._initialize_fallback_services()

    def _initialize_fallback_services(self):
        """Initialize minimal fallback services"""
        try:
            self.data_loader = NASADataLoader()
            self.anomaly_service = AnomalyDetectionService()
            self.forecasting_service = ForecastingService()
            self.training_use_case = None
            self.config_manager = None
            self.model_registry = None
            self.performance_monitor = None
            logger.info("Fallback services initialized")
        except Exception as e:
            logger.error(f"Failed to initialize fallback services: {e}")

    def _start_performance_monitoring_safe(self):
        """Start performance monitoring safely"""
        try:
            if self.performance_monitor:
                self.performance_monitor.start_monitoring(interval=10)  # Longer interval
                logger.info("Performance monitoring started safely")
        except Exception as e:
            logger.warning(f"Performance monitoring start failed: {e}")

    def _update_model_availability(self):
        """Update model availability status for all sensors"""
        try:
            if not self.model_registry:
                logger.warning("Model registry not available for availability check")
                return

            # Get availability report from model registry
            availability_report = self.model_registry.get_model_availability_report()
            self.dashboard_state["model_availability"] = availability_report

            # Emit event about model availability update
            self.event_bus.emit(
                EventType.PERFORMANCE_UPDATED,
                "dashboard_core",
                {
                    "type": "model_availability",
                    "report": availability_report,
                    "total_sensors": len(self.sensor_ids),
                    "telemanom_available": availability_report["availability_summary"].get("telemanom_available", 0),
                    "transformer_available": availability_report["availability_summary"].get(
                        "transformer_available", 0
                    ),
                    "coverage_percentage": availability_report["availability_summary"].get("coverage_percentage", 0),
                },
            )

            logger.info(
                f"Model availability updated: {availability_report['availability_summary'].get('coverage_percentage', 0):.1f}% coverage"
            )

        except Exception as e:
            logger.error(f"Error updating model availability: {e}")
            self.dashboard_state["model_availability"] = {
                "availability_summary": {
                    "total_sensors": len(self.sensor_ids),
                    "telemanom_available": 0,
                    "transformer_available": 0,
                    "both_available": 0,
                    "none_available": len(self.sensor_ids),
                    "coverage_percentage": 0,
                }
            }

    def _setup_optimized_layout(self):
        """Setup optimized dashboard layout"""
        # Enhanced header
        header = dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H1(
                                    [
                                        html.I(className="fas fa-cogs me-3 text-primary"),
                                        "IoT Predictive Maintenance",
                                    ],
                                    className="mb-1",
                                ),
                                html.P(
                                    [
                                        html.Span(
                                            "Optimized Mode",
                                            className="badge bg-success me-2",
                                        ),
                                        html.Span(
                                            "Full Features",
                                            className="badge bg-primary me-2",
                                        ),
                                        html.Span(
                                            "NASA Telemanom + Transformer",
                                            className="badge bg-info",
                                        ),
                                    ],
                                    className="text-muted mb-0",
                                ),
                            ],
                            width=8,
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.Div(id="alert-notifications", className="mb-2"),
                                        html.Div(id="optimized-status-panel"),
                                    ],
                                    className="text-end",
                                )
                            ],
                            width=4,
                        ),
                    ]
                )
            ],
            fluid=True,
            className="bg-light py-3 mb-4 shadow-sm",
        )

        # Full navigation - 7 tabs with Bootstrap Nav implementation
        nav_tabs = html.Div(
            [
                html.Ul(
                    [
                        html.Li(
                            [
                                html.A(
                                    "Overview",
                                    href="#",
                                    id="tab-overview",
                                    className="nav-link active",
                                    **{"data-tab": "overview"},
                                )
                            ],
                            className="nav-item",
                        ),
                        html.Li(
                            [
                                html.A(
                                    "Monitoring",
                                    href="#",
                                    id="tab-monitoring",
                                    className="nav-link",
                                    **{"data-tab": "monitoring"},
                                )
                            ],
                            className="nav-item",
                        ),
                        html.Li(
                            [
                                html.A(
                                    "Anomalies",
                                    href="#",
                                    id="tab-anomalies",
                                    className="nav-link",
                                    **{"data-tab": "anomalies"},
                                )
                            ],
                            className="nav-item",
                        ),
                        html.Li(
                            [
                                html.A(
                                    "Forecasting",
                                    href="#",
                                    id="tab-forecasting",
                                    className="nav-link",
                                    **{"data-tab": "forecasting"},
                                )
                            ],
                            className="nav-item",
                        ),
                        html.Li(
                            [
                                html.A(
                                    "Maintenance",
                                    href="#",
                                    id="tab-maintenance",
                                    className="nav-link",
                                    **{"data-tab": "maintenance"},
                                )
                            ],
                            className="nav-item",
                        ),
                        html.Li(
                            [
                                html.A(
                                    "Work Orders",
                                    href="#",
                                    id="tab-work-orders",
                                    className="nav-link",
                                    **{"data-tab": "work_orders"},
                                )
                            ],
                            className="nav-item",
                        ),
                        html.Li(
                            [
                                html.A(
                                    "System Performance",
                                    href="#",
                                    id="tab-system-performance",
                                    className="nav-link",
                                    **{"data-tab": "system_performance"},
                                )
                            ],
                            className="nav-item",
                        ),
                    ],
                    className="nav nav-tabs",
                    id="optimized-tabs",
                ),
                html.Div(
                    id="active-tab-indicator",
                    style={"display": "none"},
                    children="overview",
                ),
            ],
            className="mb-3",
        )

        # Main layout with optimized refresh
        main_layout = dbc.Container(
            [
                header,
                nav_tabs,
                html.Div(id="optimized-tab-content", className="mt-4"),
                dcc.Interval(id="optimized-refresh", interval=60 * 1000, n_intervals=0),  # 1 minute refresh
                dcc.Interval(id="global-refresh", interval=30 * 1000, n_intervals=0),  # Global refresh for callbacks
            ],
            fluid=True,
        )

        self.app.layout = main_layout

    def _setup_optimized_callbacks(self):
        """Setup optimized dashboard callbacks"""

        @self.app.callback(
            Output("optimized-status-panel", "children"),
            Input("optimized-refresh", "n_intervals"),
            prevent_initial_call=True,
        )
        def update_status_panel(n):
            """Update status panel"""
            try:
                return [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H6("System Status", className="mb-1"),
                                    html.P(
                                        "Operational",
                                        className="mb-0 text-success fw-bold",
                                    ),
                                    html.Small(f"Last update: {datetime.now().strftime('%H:%M:%S')}"),
                                ]
                            )
                        ],
                        className="mb-2",
                    ),
                    dbc.Badge(
                        [
                            html.I(className="fas fa-server me-1"),
                            f"{len(self.equipment_list)} Sensors",
                        ],
                        color="primary",
                    ),
                ]
            except Exception as e:
                logger.error(f"Error updating status: {e}")
                return [dbc.Alert("Status update error", color="warning", className="p-2")]

        # Single callback to handle all tab clicks
        @self.app.callback(
            [
                Output("active-tab-indicator", "children"),
                Output("optimized-tab-content", "children"),
                Output("tab-overview", "className"),
                Output("tab-monitoring", "className"),
                Output("tab-anomalies", "className"),
                Output("tab-forecasting", "className"),
                Output("tab-maintenance", "className"),
                Output("tab-work-orders", "className"),
                Output("tab-system-performance", "className"),
            ],
            [
                Input("tab-overview", "n_clicks"),
                Input("tab-monitoring", "n_clicks"),
                Input("tab-anomalies", "n_clicks"),
                Input("tab-forecasting", "n_clicks"),
                Input("tab-maintenance", "n_clicks"),
                Input("tab-work-orders", "n_clicks"),
                Input("tab-system-performance", "n_clicks"),
            ],
            prevent_initial_call=False,
        )
        def handle_tab_navigation(
            overview_clicks,
            monitoring_clicks,
            anomalies_clicks,
            forecasting_clicks,
            maintenance_clicks,
            work_orders_clicks,
            system_performance_clicks,
        ):
            """Handle tab navigation"""
            try:
                # Determine which tab was clicked
                ctx_triggered = ctx.triggered
                if not ctx_triggered:
                    # Initial load - show overview
                    active_tab = "overview"
                else:
                    button_id = ctx_triggered[0]["prop_id"].split(".")[0]

                    # Handle both named tabs (tab-overview) and numbered tabs (tab-0, tab-1, etc.)
                    named_tab_mapping = {
                        "tab-overview": "overview",
                        "tab-monitoring": "monitoring",
                        "tab-anomalies": "anomalies",
                        "tab-forecasting": "forecasting",
                        "tab-maintenance": "maintenance",
                        "tab-work-orders": "work_orders",
                        "tab-system-performance": "system_performance",
                    }

                    # Handle numbered tabs from DBC Tabs component
                    numbered_tab_mapping = {
                        "tab-0": "overview",
                        "tab-1": "monitoring",
                        "tab-2": "anomalies",
                        "tab-3": "forecasting",
                        "tab-4": "maintenance",
                        "tab-5": "work_orders",
                        "tab-6": "system_performance",
                    }

                    # Try named mapping first, then numbered mapping
                    active_tab = named_tab_mapping.get(button_id) or numbered_tab_mapping.get(button_id, "overview")

                    # Log the tab navigation for debugging
                    logger.info(f"Tab navigation: {button_id} -> {active_tab}")

                # Update dashboard state
                self.dashboard_state["current_tab"] = active_tab
                self.dashboard_state["last_update"] = datetime.now()

                # Get content for active tab
                if active_tab == "overview":
                    content = self._create_overview_tab()
                elif active_tab == "monitoring":
                    content = self._create_monitoring_tab()
                elif active_tab == "anomalies":
                    content = self._create_anomalies_tab()
                elif active_tab == "forecasting":
                    content = self._create_forecasting_tab()
                elif active_tab == "maintenance":
                    content = self._create_maintenance_tab()
                elif active_tab == "work_orders":
                    content = self._create_work_orders_tab()
                elif active_tab == "system_performance":
                    content = self._create_system_performance_tab()
                else:
                    content = self._create_overview_tab()

                # Set tab classes (active/inactive)
                tab_classes = []
                for tab_name in [
                    "overview",
                    "monitoring",
                    "anomalies",
                    "forecasting",
                    "maintenance",
                    "work_orders",
                    "system_performance",
                ]:
                    if tab_name == active_tab:
                        tab_classes.append("nav-link active")
                    else:
                        tab_classes.append("nav-link")

                return [active_tab, content] + tab_classes

            except Exception as e:
                logger.error(f"Error in tab navigation: {e}")
                return (
                    ["overview", dbc.Alert(f"Error: {str(e)}", color="danger")] + ["nav-link active"] + ["nav-link"] * 6
                )

        # Additional callback to handle DBC Tab clicks (numbered tabs)
        @self.app.callback(
            [
                Output("active-tab-indicator", "children", allow_duplicate=True),
                Output("optimized-tab-content", "children", allow_duplicate=True),
            ],
            [
                Input("tab-0", "n_clicks"),
                Input("tab-1", "n_clicks"),
                Input("tab-2", "n_clicks"),
                Input("tab-3", "n_clicks"),
                Input("tab-4", "n_clicks"),
                Input("tab-5", "n_clicks"),
                Input("tab-6", "n_clicks"),
            ],
            prevent_initial_call=True,
        )
        def handle_numbered_tab_clicks(tab0, tab1, tab2, tab3, tab4, tab5, tab6):
            """Handle DBC numbered tab clicks"""
            try:
                ctx_triggered = ctx.triggered
                if not ctx_triggered:
                    return no_update, no_update

                button_id = ctx_triggered[0]["prop_id"].split(".")[0]

                # Map numbered tabs to content
                numbered_tab_mapping = {
                    "tab-0": "overview",
                    "tab-1": "monitoring",
                    "tab-2": "anomalies",
                    "tab-3": "forecasting",
                    "tab-4": "maintenance",
                    "tab-5": "work_orders",
                    "tab-6": "system_performance",
                }

                active_tab = numbered_tab_mapping.get(button_id, "overview")
                logger.info(f"Numbered tab navigation: {button_id} -> {active_tab}")

                # Update dashboard state
                self.dashboard_state["current_tab"] = active_tab
                self.dashboard_state["last_update"] = datetime.now()

                # Get content for active tab
                if active_tab == "overview":
                    content = self._create_overview_tab()
                elif active_tab == "monitoring":
                    content = self._create_monitoring_tab()
                elif active_tab == "anomalies":
                    content = self._create_anomalies_tab()
                elif active_tab == "forecasting":
                    content = self._create_forecasting_tab()
                elif active_tab == "maintenance":
                    content = self._create_maintenance_tab()
                elif active_tab == "work_orders":
                    content = self._create_work_orders_tab()
                elif active_tab == "system_performance":
                    content = self._create_system_performance_tab()
                else:
                    content = self._create_overview_tab()

                return active_tab, content

            except Exception as e:
                logger.error(f"Error in numbered tab navigation: {e}")
                return "overview", dbc.Alert(f"Tab Error: {str(e)}", color="warning")

    def _register_essential_callbacks(self):
        """Register essential callbacks only (NO complex state management)"""
        try:
            # Simplified callback registration
            from .enhanced_callbacks_simplified import register_enhanced_callbacks

            # Prepare services dict
            services = {
                "anomaly_service": self.anomaly_service,
                "forecasting_service": self.forecasting_service,
                "data_loader": self.data_loader,
                "training_use_case": self.training_use_case,
                "config_manager": self.config_manager,
                "model_registry": self.model_registry,
                "performance_monitor": self.performance_monitor,
                "equipment_list": self.equipment_list,
            }

            # Register ONLY essential callbacks
            register_enhanced_callbacks(self.app, services)

            # Register rich component callbacks
            self._register_time_control_callbacks()
            self._register_alert_callbacks()

            logger.info("Essential callbacks registered successfully")

        except Exception as e:
            logger.warning(f"Callback registration had issues: {e}")

    def _register_time_control_callbacks(self):
        """Register enhanced time control callbacks"""

        @callback(
            Output("overview-refresh-btn", "children"),
            Input("overview-refresh-btn", "n_clicks"),
            Input("overview-time-range", "value"),
            prevent_initial_call=True,
        )
        def handle_overview_time_controls(refresh_clicks, time_range):
            if ctx.triggered_id == "overview-refresh-btn":
                # Update model availability when refresh is clicked
                self._update_model_availability()
                # Emit event for other components
                self.event_bus.emit(
                    EventType.TIME_RANGE_CHANGED,
                    "overview_controls",
                    {"time_range": time_range, "action": "refresh"},
                )
                return [html.I(className="fas fa-check me-1 text-success"), "Refreshed"]
            elif ctx.triggered_id == "overview-time-range":
                # Update time range across components
                self.time_control_manager.global_time_state["current_range"] = time_range
                self.event_bus.emit(
                    EventType.TIME_RANGE_CHANGED,
                    "overview_controls",
                    {"time_range": time_range, "action": "time_change"},
                )
                return [html.I(className="fas fa-sync-alt me-1"), "Refresh"]

            return no_update

    def _register_alert_callbacks(self):
        """Register alert system callbacks"""

        @callback(
            Output("alert-notifications", "children"),
            Input("overview-refresh-btn", "n_clicks"),
            Input("overview-time-range", "value"),
            prevent_initial_call=True,
        )
        def update_alert_notifications(refresh_clicks, time_range):
            """Update alert notifications display"""
            try:
                # Get recent alerts
                recent_alerts = self.alert_manager.get_active_alerts()[:3]

                if not recent_alerts:
                    return []

                alert_components = []
                for alert in recent_alerts:
                    color_map = {
                        "info": "info",
                        "warning": "warning",
                        "error": "danger",
                        "critical": "danger",
                    }

                    alert_badge = dbc.Alert(
                        [
                            html.Strong(alert.title),
                            html.Br(),
                            html.Small(alert.message),
                        ],
                        color=color_map.get(alert.severity.value, "info"),
                        dismissable=True,
                        is_open=True,
                        className="mb-1 p-2",
                    )
                    alert_components.append(alert_badge)

                return alert_components

            except Exception as e:
                logger.error(f"Error updating alerts: {e}")
                return []

        # Create some sample alerts for demonstration
        self._create_sample_alerts()

        # Subscribe to events for alert coordination
        self._subscribe_to_events()

    def _subscribe_to_events(self):
        """Subscribe to component events for coordination"""

        def handle_anomaly_detection(event):
            """Handle anomaly detection events"""
            try:
                sensor_id = event.data.get("sensor_id", "Unknown")
                severity = event.data.get("severity", "info")
                score = event.data.get("score", 0)

                self.alert_manager.create_alert(
                    f"Anomaly Detected: {sensor_id}",
                    f"Anomaly score: {score:.2f} - Severity: {severity}",
                    (AlertSeverity.WARNING if severity == "medium" else AlertSeverity.ERROR),
                    AlertCategory.ANOMALY,
                    event.source_component,
                )
                logger.info(f"Created anomaly alert for {sensor_id}")
            except Exception as e:
                logger.error(f"Error handling anomaly event: {e}")

        def handle_forecast_generation(event):
            """Handle forecast generation events"""
            try:
                sensor_id = event.data.get("sensor_id", "Unknown")
                accuracy = event.data.get("accuracy", 0)

                if accuracy < 0.7:  # Low accuracy threshold
                    self.alert_manager.create_alert(
                        f"Low Forecast Accuracy: {sensor_id}",
                        f"Forecast accuracy: {accuracy:.1%} - Consider model retraining",
                        AlertSeverity.WARNING,
                        AlertCategory.MODEL,
                        event.source_component,
                    )
            except Exception as e:
                logger.error(f"Error handling forecast event: {e}")

        def handle_sensor_selection(event):
            """Handle sensor selection events"""
            try:
                sensor_id = event.data.get("sensor_id", "Unknown")
                self.alert_manager.create_alert(
                    f"Sensor Selected: {sensor_id}",
                    f"Now monitoring sensor {sensor_id}",
                    AlertSeverity.INFO,
                    AlertCategory.SYSTEM,
                    event.source_component,
                    auto_dismiss=True,
                    dismiss_after_seconds=5,
                )
            except Exception as e:
                logger.error(f"Error handling sensor selection event: {e}")

        # Subscribe to events
        self.event_bus.subscribe(EventType.ANOMALY_DETECTED, handle_anomaly_detection, "alert_manager")
        self.event_bus.subscribe(EventType.FORECAST_GENERATED, handle_forecast_generation, "alert_manager")
        self.event_bus.subscribe(EventType.SENSOR_SELECTED, handle_sensor_selection, "alert_manager")

        logger.info("Alert manager subscribed to component events")

    def _create_sample_alerts(self):
        """Create sample alerts for demonstration"""
        try:
            # Create some sample alerts
            self.alert_manager.create_alert(
                "System Online",
                "Dashboard successfully initialized with all features",
                AlertSeverity.INFO,
                AlertCategory.SYSTEM,
                "dashboard_core",
            )

            if (
                self.dashboard_state.get("model_availability", {})
                .get("availability_summary", {})
                .get("coverage_percentage", 0)
                < 50
            ):
                self.alert_manager.create_alert(
                    "Low Model Coverage",
                    f"Only {self.dashboard_state.get('model_availability', {}).get('availability_summary', {}).get('coverage_percentage', 0):.0f}% model coverage detected",
                    AlertSeverity.WARNING,
                    AlertCategory.MODEL,
                    "model_registry",
                )

        except Exception as e:
            logger.error(f"Error creating sample alerts: {e}")

    def _create_overview_tab(self):
        """Create comprehensive overview tab with model availability"""
        # Get current model availability
        availability = self.dashboard_state.get("model_availability", {})
        availability_summary = availability.get("availability_summary", {})

        telemanom_available = availability_summary.get("telemanom_available", 0)
        transformer_available = availability_summary.get("transformer_available", 0)
        coverage_percentage = availability_summary.get("coverage_percentage", 0)

        return dbc.Container(
            [
                # Enhanced Time Controls Header
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H4(
                                    [
                                        html.I(className="fas fa-chart-line me-2"),
                                        "System Overview",
                                    ],
                                    className="mb-0",
                                )
                            ],
                            width=8,
                        ),
                        dbc.Col(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dcc.Dropdown(
                                                    id="overview-time-range",
                                                    options=[
                                                        {
                                                            "label": "Real-time",
                                                            "value": "real_time",
                                                        },
                                                        {
                                                            "label": "Last 15 minutes",
                                                            "value": "15m",
                                                        },
                                                        {
                                                            "label": "Last Hour",
                                                            "value": "1h",
                                                        },
                                                        {
                                                            "label": "Last 6 Hours",
                                                            "value": "6h",
                                                        },
                                                        {
                                                            "label": "Last 24 Hours",
                                                            "value": "24h",
                                                        },
                                                        {
                                                            "label": "Last 7 Days",
                                                            "value": "7d",
                                                        },
                                                        {
                                                            "label": "Last 30 Days",
                                                            "value": "30d",
                                                        },
                                                    ],
                                                    value="24h",
                                                    clearable=False,
                                                    className="mb-0",
                                                )
                                            ],
                                            width=8,
                                        ),
                                        dbc.Col(
                                            [
                                                dbc.Button(
                                                    [
                                                        html.I(className="fas fa-sync-alt me-1"),
                                                        "Refresh",
                                                    ],
                                                    id="overview-refresh-btn",
                                                    color="outline-primary",
                                                    size="sm",
                                                )
                                            ],
                                            width=4,
                                        ),
                                    ],
                                    className="g-1",
                                )
                            ],
                            width=4,
                        ),
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            [
                                                html.I(className="fas fa-tachometer-alt me-2"),
                                                "System Overview",
                                            ]
                                        ),
                                        dbc.CardBody(
                                            [
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.H3(
                                                                    len(self.equipment_list),
                                                                    className="text-primary",
                                                                ),
                                                                html.P(
                                                                    "Total Sensors",
                                                                    className="text-muted mb-0",
                                                                ),
                                                            ],
                                                            width=3,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                html.H3(
                                                                    f"{telemanom_available}",
                                                                    className="text-success",
                                                                ),
                                                                html.P(
                                                                    "Anomaly Models",
                                                                    className="text-muted mb-0",
                                                                ),
                                                            ],
                                                            width=3,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                html.H3(
                                                                    f"{transformer_available}",
                                                                    className="text-info",
                                                                ),
                                                                html.P(
                                                                    "Forecast Models",
                                                                    className="text-muted mb-0",
                                                                ),
                                                            ],
                                                            width=3,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                html.H3(
                                                                    f"{coverage_percentage:.0f}%",
                                                                    className="text-warning",
                                                                ),
                                                                html.P(
                                                                    "Model Coverage",
                                                                    className="text-muted mb-0",
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
                            ],
                            width=12,
                        )
                    ],
                    className="mb-4",
                ),
                # Model Availability Status Row
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            [
                                                html.I(className="fas fa-brain me-2"),
                                                "Model Registry Status",
                                            ]
                                        ),
                                        dbc.CardBody(
                                            [
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Alert(
                                                                    [
                                                                        html.H6(
                                                                            "Telemanom (Anomaly Detection)",
                                                                            className="mb-1",
                                                                        ),
                                                                        html.P(
                                                                            f"{telemanom_available}/{len(self.equipment_list)} sensors trained",
                                                                            className="mb-0",
                                                                        ),
                                                                    ],
                                                                    color=(
                                                                        "success"
                                                                        if telemanom_available > 0
                                                                        else "warning"
                                                                    ),
                                                                )
                                                            ],
                                                            width=6,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dbc.Alert(
                                                                    [
                                                                        html.H6(
                                                                            "Transformer (Forecasting)",
                                                                            className="mb-1",
                                                                        ),
                                                                        html.P(
                                                                            f"{transformer_available}/{len(self.equipment_list)} sensors trained",
                                                                            className="mb-0",
                                                                        ),
                                                                    ],
                                                                    color=(
                                                                        "info"
                                                                        if transformer_available > 0
                                                                        else "warning"
                                                                    ),
                                                                )
                                                            ],
                                                            width=6,
                                                        ),
                                                    ]
                                                )
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=12,
                        )
                    ],
                    className="mb-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("SMAP Mission Sensors"),
                                        dbc.CardBody(
                                            [
                                                html.Div(
                                                    [
                                                        dbc.Badge(
                                                            f"{eq.equipment_id}",
                                                            color="primary",
                                                            className="me-2 mb-2",
                                                        )
                                                        for eq in self.equipment_list
                                                        if eq.equipment_id.startswith("SMAP")
                                                    ]
                                                )
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("MSL Mission Sensors"),
                                        dbc.CardBody(
                                            [
                                                html.Div(
                                                    [
                                                        dbc.Badge(
                                                            f"{eq.equipment_id}",
                                                            color="success",
                                                            className="me-2 mb-2",
                                                        )
                                                        for eq in self.equipment_list
                                                        if eq.equipment_id.startswith("MSL")
                                                    ]
                                                )
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=6,
                        ),
                    ],
                    className="mb-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Service Status"),
                                        dbc.CardBody(
                                            [
                                                html.Div(
                                                    [
                                                        dbc.ListGroupItem(
                                                            [
                                                                html.I(
                                                                    className="fas fa-check-circle text-success me-2"
                                                                ),
                                                                f"Data Loader: {'Active' if self.data_loader else 'Inactive'}",
                                                            ]
                                                        ),
                                                        dbc.ListGroupItem(
                                                            [
                                                                html.I(
                                                                    className="fas fa-check-circle text-success me-2"
                                                                ),
                                                                f"Anomaly Detection: {'Active' if self.anomaly_service else 'Inactive'}",
                                                            ]
                                                        ),
                                                        dbc.ListGroupItem(
                                                            [
                                                                html.I(
                                                                    className="fas fa-check-circle text-success me-2"
                                                                ),
                                                                f"Forecasting: {'Active' if self.forecasting_service else 'Inactive'}",
                                                            ]
                                                        ),
                                                        dbc.ListGroupItem(
                                                            [
                                                                html.I(className="fas fa-cog text-info me-2"),
                                                                f"Training System: {'Available' if self.training_use_case else 'Unavailable'}",
                                                            ]
                                                        ),
                                                    ]
                                                )
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=12,
                        )
                    ]
                ),
            ]
        )

    def _create_monitoring_tab(self):
        """Create monitoring tab"""
        return dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Real-time Sensor Monitoring"),
                                        dbc.CardBody(
                                            [
                                                html.H5("NASA Telemanom Algorithm"),
                                                html.P(
                                                    "Monitoring 12 sensors (6 SMAP + 6 MSL) using NASA's advanced telemetry anomaly detection"
                                                ),
                                                dbc.Progress(
                                                    value=85,
                                                    label="System Performance: 85%",
                                                    className="mb-3",
                                                ),
                                                dbc.ButtonGroup(
                                                    [
                                                        dbc.Button(
                                                            "View Real-time Data",
                                                            color="primary",
                                                            size="sm",
                                                        ),
                                                        dbc.Button(
                                                            "Historical Analysis",
                                                            color="secondary",
                                                            size="sm",
                                                        ),
                                                        dbc.Button(
                                                            "Export Data",
                                                            color="info",
                                                            size="sm",
                                                        ),
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=12,
                        )
                    ]
                )
            ]
        )

    def _create_anomalies_tab(self):
        """Create anomalies detection tab"""
        return dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("NASA Telemanom Anomaly Detection"),
                                        dbc.CardBody(
                                            [
                                                html.H5("Advanced Anomaly Detection System"),
                                                html.P(
                                                    "Using NASA's Telemanom LSTM-based algorithm for spacecraft telemetry anomaly detection"
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.H4(
                                                                    "0",
                                                                    className="text-success",
                                                                ),
                                                                html.P("Current Anomalies"),
                                                            ],
                                                            width=4,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                html.H4(
                                                                    "12",
                                                                    className="text-info",
                                                                ),
                                                                html.P("Sensors Monitored"),
                                                            ],
                                                            width=4,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                html.H4(
                                                                    "99.8%",
                                                                    className="text-primary",
                                                                ),
                                                                html.P("Detection Accuracy"),
                                                            ],
                                                            width=4,
                                                        ),
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=12,
                        )
                    ]
                )
            ]
        )

    def _create_forecasting_tab(self):
        """Create forecasting tab"""
        return dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Transformer-based Forecasting"),
                                        dbc.CardBody(
                                            [
                                                html.H5("Predictive Analytics"),
                                                html.P(
                                                    "Advanced forecasting using Transformer models for predictive maintenance"
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.H4(
                                                                    "24h",
                                                                    className="text-primary",
                                                                ),
                                                                html.P("Forecast Horizon"),
                                                            ],
                                                            width=4,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                html.H4(
                                                                    "95%",
                                                                    className="text-success",
                                                                ),
                                                                html.P("Prediction Accuracy"),
                                                            ],
                                                            width=4,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                html.H4(
                                                                    "Low",
                                                                    className="text-info",
                                                                ),
                                                                html.P("Risk Level"),
                                                            ],
                                                            width=4,
                                                        ),
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=12,
                        )
                    ]
                )
            ]
        )

    def _create_maintenance_tab(self):
        """Create maintenance tab"""
        return dbc.Container(
            [
                dbc.Card(
                    [
                        dbc.CardHeader("Predictive Maintenance Scheduler"),
                        dbc.CardBody(
                            [
                                html.H5("Maintenance Planning"),
                                html.P("Automated maintenance scheduling based on sensor data and predictions"),
                                dbc.Alert("No pending maintenance tasks", color="success"),
                                dbc.Button("Schedule Maintenance", color="primary"),
                            ]
                        ),
                    ]
                )
            ]
        )

    def _create_work_orders_tab(self):
        """Create work orders tab"""
        return dbc.Container(
            [
                dbc.Card(
                    [
                        dbc.CardHeader("Work Order Management"),
                        dbc.CardBody(
                            [
                                html.H5("Active Work Orders"),
                                html.P("Track and manage maintenance work orders"),
                                dbc.Alert("No active work orders", color="info"),
                                dbc.Button("Create Work Order", color="success"),
                            ]
                        ),
                    ]
                )
            ]
        )

    def _create_system_performance_tab(self):
        """Create comprehensive system performance tab"""
        return dbc.Container(
            [
                dbc.Card(
                    [
                        dbc.CardHeader("System Performance & Management"),
                        dbc.CardBody(
                            [
                                html.H5("Consolidated System Management"),
                                html.P("Training Hub, Model Registry, ML Pipeline, and System Administration"),
                                html.Div(
                                    [
                                        create_training_hub_layout(),
                                        html.Hr(),
                                        create_model_registry_layout(),
                                        html.Hr(),
                                        create_system_admin_layout(),
                                        html.Hr(),
                                        pipeline_dashboard(),
                                    ]
                                ),
                            ]
                        ),
                    ]
                )
            ]
        )

    def run(self, host: str = "127.0.0.1", port: int = 8050, debug: bool = False):
        """Run the optimized dashboard"""
        logger.info(f"Starting Optimized IoT Dashboard at http://{host}:{port}")
        logger.info("Full feature dashboard with optimized state management")
        try:
            self.app.run(host=host, port=port, debug=debug)
        finally:
            # Cleanup
            if hasattr(self, "performance_monitor") and self.performance_monitor:
                try:
                    self.performance_monitor.stop_monitoring()
                except:
                    pass


def create_optimized_dashboard(debug: bool = False) -> OptimizedIoTDashboard:
    """Create and return optimized dashboard instance"""
    return OptimizedIoTDashboard(debug=debug)


def create_app(debug: bool = False):
    """
    Create and return Dash app instance for use with start_dashboard.py

    Args:
        debug: Enable debug mode

    Returns:
        Dash app instance
    """
    dashboard = OptimizedIoTDashboard(debug=debug)
    return dashboard.app


if __name__ == "__main__":
    dashboard = create_optimized_dashboard(debug=True)
    dashboard.run(debug=True)

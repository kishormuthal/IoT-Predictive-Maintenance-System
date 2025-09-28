"""
Enhanced Dashboard Application - Batch 3
Complete IoT Predictive Maintenance Dashboard with Training Management
"""

import dash
from dash import dcc, html, Input, Output, callback, State, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import enhanced services
from src.core.services.anomaly_service import AnomalyDetectionService
from src.core.services.forecasting_service import ForecastingService
from src.infrastructure.data.nasa_data_loader import NASADataLoader
from src.application.use_cases.training_use_case import TrainingUseCase
from src.application.services.training_config_manager import TrainingConfigManager
from src.infrastructure.ml.model_registry import ModelRegistry
from src.infrastructure.monitoring.performance_monitor import PerformanceMonitor
from config.equipment_config import get_equipment_list, get_equipment_by_id

# Import dashboard components
from src.presentation.dashboard.components.config_manager import create_config_management_layout, register_config_callbacks

logger = logging.getLogger(__name__)


class EnhancedIoTDashboard:
    """
    Enhanced IoT Predictive Maintenance Dashboard with Training Management
    """

    def __init__(self, debug: bool = False):
        """Initialize enhanced dashboard"""
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[
                dbc.themes.BOOTSTRAP,
                dbc.icons.FONT_AWESOME,
                dbc.icons.BOOTSTRAP
            ],
            suppress_callback_exceptions=True,
            title="IoT Predictive Maintenance - Enhanced Dashboard"
        )

        # Initialize services
        self._initialize_services()

        # Load configuration and equipment
        self.equipment_list = get_equipment_list()
        self.sensor_ids = [eq.equipment_id for eq in self.equipment_list]

        # Dashboard state
        self.dashboard_state = {
            'system_health': 'initializing',
            'last_update': datetime.now(),
            'active_alerts': [],
            'performance_metrics': {}
        }

        # Setup enhanced layout and callbacks
        self._setup_layout()
        self._setup_callbacks()
        self._register_enhanced_callbacks()

        logger.info("Enhanced IoT Dashboard initialized successfully")

    def _initialize_services(self):
        """Initialize all services with proper integration"""
        try:
            # Core services
            self.data_loader = NASADataLoader()
            self.anomaly_service = AnomalyDetectionService()
            self.forecasting_service = ForecastingService()

            # Training and management services
            self.training_use_case = TrainingUseCase()
            self.config_manager = TrainingConfigManager()
            self.model_registry = ModelRegistry()
            self.performance_monitor = PerformanceMonitor()

            # Start performance monitoring
            self.performance_monitor.start_monitoring(interval=5)

            logger.info("All services initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing services: {e}")
            # Initialize minimal services for basic functionality
            self.data_loader = NASADataLoader()
            self.anomaly_service = AnomalyDetectionService()
            self.forecasting_service = ForecastingService()

    def _setup_layout(self):
        """Setup enhanced dashboard layout"""
        # Enhanced header with system status
        header = self._create_header()

        # Enhanced navigation with new tabs
        nav_tabs = dbc.Tabs([
            dbc.Tab(label="🏠 Overview", tab_id="overview", active_label_style={"color": "#0d6efd"}),
            dbc.Tab(label="📊 Monitoring", tab_id="monitoring"),
            dbc.Tab(label="🚨 Anomalies", tab_id="anomalies"),
            dbc.Tab(label="📈 Forecasting", tab_id="forecasting"),
            dbc.Tab(label="🤖 Training Hub", tab_id="training"),
            dbc.Tab(label="📋 Models", tab_id="models"),
            dbc.Tab(label="🔧 Configuration", tab_id="configuration"),
            dbc.Tab(label="⚙️ System Admin", tab_id="admin")
        ], id="main-tabs", active_tab="overview", className="custom-tabs")

        # Main content area
        main_content = dbc.Container([
            # Global refresh interval
            dcc.Interval(id='global-refresh', interval=15*1000, n_intervals=0),

            # Store components for state management
            dcc.Store(id='dashboard-state', data=self.dashboard_state),
            dcc.Store(id='training-state', data={}),
            dcc.Store(id='alert-state', data=[]),

            nav_tabs,
            html.Div(id="tab-content", className="mt-4"),

            # Global notification area
            html.Div(id="global-notifications", className="position-fixed",
                    style={"top": "20px", "right": "20px", "z-index": "9999"})
        ], fluid=True)

        # Custom CSS
        custom_css = html.Style("""
            .custom-tabs .nav-link {
                font-weight: 500;
                border-radius: 8px 8px 0 0;
            }
            .custom-tabs .nav-link.active {
                background-color: #0d6efd;
                color: white;
                border-color: #0d6efd;
            }
            .metric-card {
                transition: transform 0.2s;
            }
            .metric-card:hover {
                transform: translateY(-2px);
            }
            .alert-pulse {
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.05); }
                100% { transform: scale(1); }
            }
        """)

        self.app.layout = html.Div([custom_css, header, main_content])

    def _register_enhanced_callbacks(self):
        """Register all enhanced dashboard callbacks"""
        try:
            from .enhanced_callbacks import register_enhanced_callbacks

            # Prepare services dict
            services = {
                'anomaly_service': self.anomaly_service,
                'forecasting_service': self.forecasting_service,
                'data_loader': self.data_loader,
                'training_use_case': self.training_use_case,
                'config_manager': self.config_manager,
                'model_registry': self.model_registry,
                'performance_monitor': self.performance_monitor,
                'equipment_list': self.equipment_list
            }

            # Register all enhanced callbacks
            register_enhanced_callbacks(self.app, services)

            # Register configuration management callbacks
            register_config_callbacks(self.app)

            logger.info("Enhanced callbacks registered successfully")

        except Exception as e:
            logger.error(f"Error registering enhanced callbacks: {e}")
            # Continue with basic functionality

    def _create_header(self):
        """Create enhanced header with real-time status"""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1([
                        html.I(className="fas fa-cogs me-3 text-primary"),
                        "IoT Predictive Maintenance"
                    ], className="mb-1"),
                    html.P([
                        html.Span("Enhanced Dashboard", className="badge bg-primary me-2"),
                        html.Span("Training + Inference", className="badge bg-success me-2"),
                        html.Span("NASA Telemanom + Transformer", className="badge bg-info")
                    ], className="text-muted mb-0")
                ], width=8),
                dbc.Col([
                    html.Div(id="header-status-panel", className="text-end")
                ], width=4)
            ])
        ], fluid=True, className="bg-light py-3 mb-4 shadow-sm")

    def _setup_callbacks(self):
        """Setup all dashboard callbacks"""

        @self.app.callback(
            [Output('header-status-panel', 'children'),
             Output('dashboard-state', 'data'),
             Output('alert-state', 'data')],
            [Input('global-refresh', 'n_intervals')],
            [State('dashboard-state', 'data')]
        )
        def update_header_status(n, current_state):
            """Update header status panel and dashboard state"""
            try:
                # Get current system metrics
                anomaly_summary = self.anomaly_service.get_detection_summary()
                training_status = self.training_use_case.get_training_status()
                performance_alerts = self.performance_monitor.get_performance_alerts()

                # Calculate system health
                total_anomalies = anomaly_summary.get('total_anomalies', 0)
                equipment_count = training_status.get('total_equipment', 0)
                critical_alerts = [a for a in performance_alerts if a.get('severity') == 'critical']

                # Determine system health status
                if len(critical_alerts) > 0:
                    health_status = 'critical'
                    health_color = 'danger'
                    health_icon = 'fas fa-exclamation-triangle'
                elif total_anomalies > equipment_count * 0.3:  # 30% threshold
                    health_status = 'warning'
                    health_color = 'warning'
                    health_icon = 'fas fa-exclamation-circle'
                else:
                    health_status = 'healthy'
                    health_color = 'success'
                    health_icon = 'fas fa-check-circle'

                # Create status panel
                status_panel = [
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.I(className=f"{health_icon} fa-2x text-{health_color}")
                                ], width=3),
                                dbc.Col([
                                    html.H6("System Health", className="mb-1"),
                                    html.P(health_status.title(), className=f"mb-0 text-{health_color} fw-bold")
                                ], width=9)
                            ])
                        ])
                    ], className="mb-2"),

                    dbc.Row([
                        dbc.Col([
                            dbc.Badge([
                                html.I(className="fas fa-server me-1"),
                                f"{equipment_count} Sensors"
                            ], color="primary", className="me-1")
                        ], width=12),
                    ], className="mb-2"),

                    dbc.Row([
                        dbc.Col([
                            dbc.Badge([
                                html.I(className="fas fa-exclamation-triangle me-1"),
                                f"{total_anomalies} Anomalies"
                            ], color="danger" if total_anomalies > 0 else "success", className="me-1")
                        ], width=12),
                    ], className="mb-2"),

                    dbc.Row([
                        dbc.Col([
                            dbc.Badge([
                                html.I(className="fas fa-bell me-1"),
                                f"{len(performance_alerts)} Alerts"
                            ], color="warning" if performance_alerts else "success")
                        ], width=12)
                    ])
                ]

                # Update dashboard state
                new_state = current_state.copy()
                new_state.update({
                    'system_health': health_status,
                    'last_update': datetime.now().isoformat(),
                    'total_anomalies': total_anomalies,
                    'equipment_count': equipment_count,
                    'alert_count': len(performance_alerts)
                })

                return status_panel, new_state, performance_alerts

            except Exception as e:
                logger.error(f"Error updating header status: {e}")
                error_panel = [
                    dbc.Alert([
                        html.I(className="fas fa-exclamation-triangle me-2"),
                        "System Loading..."
                    ], color="secondary", className="mb-0")
                ]
                return error_panel, current_state, []

        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('main-tabs', 'active_tab')],
            [State('dashboard-state', 'data')]
        )
        def render_tab_content(active_tab, dashboard_state):
            """Render content based on active tab"""
            try:
                if active_tab == "overview":
                    return self._create_enhanced_overview_tab(dashboard_state)
                elif active_tab == "monitoring":
                    return self._create_enhanced_monitoring_tab()
                elif active_tab == "anomalies":
                    return self._create_enhanced_anomalies_tab()
                elif active_tab == "forecasting":
                    return self._create_enhanced_forecasting_tab()
                elif active_tab == "training":
                    return self._create_training_hub_tab()
                elif active_tab == "models":
                    return self._create_models_tab()
                elif active_tab == "configuration":
                    return self._create_configuration_tab()
                elif active_tab == "admin":
                    return self._create_admin_tab()
                else:
                    return dbc.Alert("Tab not found", color="warning")
            except Exception as e:
                logger.error(f"Error rendering tab {active_tab}: {e}")
                return dbc.Alert(f"Error loading {active_tab}: {str(e)}", color="danger")

        @self.app.callback(
            Output('global-notifications', 'children'),
            [Input('alert-state', 'data')]
        )
        def update_notifications(alerts):
            """Update global notification toasts"""
            try:
                if not alerts:
                    return []

                # Show only critical alerts as notifications
                critical_alerts = [a for a in alerts if a.get('severity') == 'critical']

                notifications = []
                for i, alert in enumerate(critical_alerts[:3]):  # Limit to 3 notifications
                    notification = dbc.Toast([
                        html.P([
                            html.I(className="fas fa-exclamation-triangle me-2"),
                            alert.get('message', 'System Alert')
                        ], className="mb-0")
                    ],
                    id=f"alert-toast-{i}",
                    header="Critical System Alert",
                    icon="danger",
                    dismissable=True,
                    is_open=True,
                    duration=10000,  # 10 seconds
                    style={"margin-bottom": "10px"}
                    )
                    notifications.append(notification)

                return notifications

            except Exception as e:
                logger.error(f"Error updating notifications: {e}")
                return []

    def _create_enhanced_overview_tab(self, dashboard_state):
        """Create enhanced overview tab with comprehensive metrics"""
        return dbc.Container([
            # Key metrics row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-server fa-2x text-primary mb-2"),
                                html.H3(f"{dashboard_state.get('equipment_count', 0)}", className="mb-1"),
                                html.P("Equipment Units", className="text-muted mb-0")
                            ], className="text-center")
                        ])
                    ], className="metric-card h-100")
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-exclamation-triangle fa-2x text-danger mb-2"),
                                html.H3(f"{dashboard_state.get('total_anomalies', 0)}", className="mb-1"),
                                html.P("Active Anomalies", className="text-muted mb-0")
                            ], className="text-center")
                        ])
                    ], className="metric-card h-100")
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-robot fa-2x text-success mb-2"),
                                html.H3(id="trained-models-count", className="mb-1"),
                                html.P("Trained Models", className="text-muted mb-0")
                            ], className="text-center")
                        ])
                    ], className="metric-card h-100")
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-bell fa-2x text-warning mb-2"),
                                html.H3(f"{dashboard_state.get('alert_count', 0)}", className="mb-1"),
                                html.P("System Alerts", className="text-muted mb-0")
                            ], className="text-center")
                        ])
                    ], className="metric-card h-100")
                ], width=3)
            ], className="mb-4"),

            # Equipment status grid
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-list-ul me-2"),
                            "Equipment Status Overview"
                        ]),
                        dbc.CardBody([
                            html.Div(id="equipment-status-grid")
                        ])
                    ])
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-chart-pie me-2"),
                            "System Health"
                        ]),
                        dbc.CardBody([
                            dcc.Graph(id="system-health-chart")
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),

            # Recent activity
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-clock me-2"),
                            "Recent Activity"
                        ]),
                        dbc.CardBody([
                            html.Div(id="recent-activity-feed")
                        ])
                    ])
                ])
            ])
        ])

    def _create_enhanced_monitoring_tab(self):
        """Create enhanced monitoring tab with real-time features"""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📈 Real-time Sensor Monitoring"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Select Sensor:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id='monitoring-sensor-selector',
                                        options=[
                                            {'label': f"{eq.equipment_id} - {eq.name}", 'value': eq.equipment_id}
                                            for eq in self.equipment_list
                                        ],
                                        value=self.equipment_list[0].equipment_id if self.equipment_list else None,
                                        placeholder="Select a sensor to monitor"
                                    )
                                ], width=6),
                                dbc.Col([
                                    html.Label("Time Range:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id='monitoring-time-range',
                                        options=[
                                            {'label': 'Last Hour', 'value': 1},
                                            {'label': 'Last 6 Hours', 'value': 6},
                                            {'label': 'Last 24 Hours', 'value': 24},
                                            {'label': 'Last 48 Hours', 'value': 48}
                                        ],
                                        value=24
                                    )
                                ], width=6)
                            ])
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 Sensor Data Visualization"),
                        dbc.CardBody([
                            dcc.Graph(id="enhanced-realtime-chart", style={'height': '500px'})
                        ])
                    ])
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("ℹ️ Sensor Details"),
                        dbc.CardBody([
                            html.Div(id="enhanced-sensor-info")
                        ])
                    ], className="mb-3"),
                    dbc.Card([
                        dbc.CardHeader("📈 Live Statistics"),
                        dbc.CardBody([
                            html.Div(id="live-sensor-stats")
                        ])
                    ])
                ], width=4)
            ])
        ])

    def _create_enhanced_anomalies_tab(self):
        """Create enhanced anomalies tab with detailed analysis"""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-exclamation-triangle me-2"),
                            "Anomaly Detection Dashboard - NASA Telemanom"
                        ]),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Div(id="anomaly-summary-cards")
                                ], width=4),
                                dbc.Col([
                                    dcc.Graph(id="anomaly-timeline-chart")
                                ], width=8)
                            ])
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🚨 Recent Anomalies"),
                        dbc.CardBody([
                            html.Div(id="enhanced-anomaly-list")
                        ])
                    ])
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 Anomaly Analysis"),
                        dbc.CardBody([
                            html.Div(id="anomaly-analysis")
                        ])
                    ])
                ], width=4)
            ])
        ])

    def _create_enhanced_forecasting_tab(self):
        """Create enhanced forecasting tab with advanced features"""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🔮 Predictive Forecasting - Transformer Model"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Select Sensor:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id='forecast-sensor-selector',
                                        options=[
                                            {'label': f"{eq.equipment_id} - {eq.name}", 'value': eq.equipment_id}
                                            for eq in self.equipment_list
                                        ],
                                        value=self.equipment_list[0].equipment_id if self.equipment_list else None
                                    )
                                ], width=4),
                                dbc.Col([
                                    html.Label("Forecast Horizon (hours):", className="fw-bold"),
                                    dcc.Dropdown(
                                        id='forecast-horizon',
                                        options=[
                                            {'label': '6 Hours', 'value': 6},
                                            {'label': '12 Hours', 'value': 12},
                                            {'label': '24 Hours', 'value': 24},
                                            {'label': '48 Hours', 'value': 48}
                                        ],
                                        value=24
                                    )
                                ], width=4),
                                dbc.Col([
                                    html.Label("Model Mode:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id='forecast-mode',
                                        options=[
                                            {'label': 'Standard', 'value': 'standard'},
                                            {'label': 'High Precision', 'value': 'precision'},
                                            {'label': 'Fast', 'value': 'fast'}
                                        ],
                                        value='standard'
                                    )
                                ], width=4)
                            ])
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📈 Forecast Visualization"),
                        dbc.CardBody([
                            dcc.Graph(id="enhanced-forecast-chart", style={'height': '500px'})
                        ])
                    ])
                ], width=9),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 Forecast Metrics"),
                        dbc.CardBody([
                            html.Div(id="forecast-metrics")
                        ])
                    ])
                ], width=3)
            ])
        ])

    def _create_training_hub_tab(self):
        """Create training management hub"""
        return html.Div(id="training-hub-content")

    def _create_models_tab(self):
        """Create model management tab"""
        return html.Div(id="models-tab-content")

    def _create_configuration_tab(self):
        """Create configuration management tab"""
        return create_config_management_layout()

    def _create_admin_tab(self):
        """Create system administration tab"""
        return html.Div(id="admin-tab-content")

    def run(self, host: str = '127.0.0.1', port: int = 8050, debug: bool = False):
        """Run the enhanced dashboard"""
        logger.info(f"Starting Enhanced IoT Dashboard at http://{host}:{port}")
        logger.info("Features: Training Management | Model Registry | Performance Monitoring")
        try:
            self.app.run(host=host, port=port, debug=debug)
        finally:
            # Cleanup on shutdown
            if hasattr(self, 'performance_monitor'):
                self.performance_monitor.stop_monitoring()


def create_enhanced_dashboard(debug: bool = False) -> EnhancedIoTDashboard:
    """Create and return enhanced dashboard instance"""
    return EnhancedIoTDashboard(debug=debug)


if __name__ == "__main__":
    dashboard = create_enhanced_dashboard(debug=True)
    dashboard.run(debug=True)
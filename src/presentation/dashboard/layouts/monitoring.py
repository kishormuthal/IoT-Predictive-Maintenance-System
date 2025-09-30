"""
Real-time Monitoring Dashboard for IoT System
Live sensor data visualization and monitoring
"""

from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import dash_daq as daq
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

# Import Clean Architecture services
try:
    from src.infrastructure.data.nasa_data_loader import NASADataLoader
    from src.core.services.anomaly_service import AnomalyDetectionService
    from config.equipment_config import get_equipment_list, get_equipment_by_id
    SERVICES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Services not fully available: {e}")
    SERVICES_AVAILABLE = False

logger = logging.getLogger(__name__)


class MonitoringDashboard:
    """Real-time monitoring dashboard component"""

    def __init__(self):
        """Initialize monitoring dashboard"""
        self.equipment_list = []
        self.sensor_ids = []

        if SERVICES_AVAILABLE:
            try:
                self.equipment_list = get_equipment_list()
                self.sensor_ids = [eq.equipment_id for eq in self.equipment_list]
                self.data_loader = NASADataLoader()
                logger.info(f"Monitoring dashboard initialized with {len(self.sensor_ids)} sensors")
            except Exception as e:
                logger.warning(f"Service initialization failed: {e}")
                self.data_loader = None
        else:
            self.data_loader = None

    def create_layout(self) -> html.Div:
        """Create monitoring dashboard layout

        Returns:
            Dashboard layout with real-time monitoring components
        """
        return dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H3([
                        html.I(className="fas fa-tv me-3 text-primary"),
                        "Real-time Sensor Monitoring"
                    ]),
                    html.P("Live monitoring of all IoT sensors with real-time data updates",
                           className="text-muted")
                ])
            ], className="mb-4"),

            # Control Panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Label("Select Equipment:", className="fw-bold mb-2"),
                            dcc.Dropdown(
                                id='monitoring-equipment-selector',
                                options=[
                                    {'label': eq.name if hasattr(eq, 'name') else eq.equipment_id,
                                     'value': eq.equipment_id}
                                    for eq in self.equipment_list
                                ] if self.equipment_list else [
                                    {'label': f'Sensor {i+1}', 'value': f'sensor_{i+1}'}
                                    for i in range(12)
                                ],
                                value=self.sensor_ids[0] if self.sensor_ids else 'sensor_1',
                                clearable=False,
                                className="mb-3"
                            ),

                            html.Label("Time Range:", className="fw-bold mb-2"),
                            dcc.Dropdown(
                                id='monitoring-time-range',
                                options=[
                                    {'label': 'Last Hour', 'value': '1h'},
                                    {'label': 'Last 6 Hours', 'value': '6h'},
                                    {'label': 'Last 24 Hours', 'value': '24h'},
                                    {'label': 'Last 7 Days', 'value': '7d'}
                                ],
                                value='24h',
                                clearable=False,
                                className="mb-3"
                            ),

                            dbc.Button([
                                html.I(className="fas fa-sync-alt me-2"),
                                "Refresh Data"
                            ], id='monitoring-refresh-btn', color="primary", className="w-100")
                        ])
                    ])
                ], width=3),

                # Status Cards
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            self._create_status_card(
                                "Total Sensors",
                                str(len(self.sensor_ids)) if self.sensor_ids else "12",
                                "fas fa-broadcast-tower",
                                "primary",
                                "monitoring-total-sensors"
                            )
                        ], width=3),
                        dbc.Col([
                            self._create_status_card(
                                "Online",
                                str(len(self.sensor_ids)) if self.sensor_ids else "12",
                                "fas fa-check-circle",
                                "success",
                                "monitoring-online-count"
                            )
                        ], width=3),
                        dbc.Col([
                            self._create_status_card(
                                "Warnings",
                                "0",
                                "fas fa-exclamation-triangle",
                                "warning",
                                "monitoring-warning-count"
                            )
                        ], width=3),
                        dbc.Col([
                            self._create_status_card(
                                "Critical",
                                "0",
                                "fas fa-exclamation-circle",
                                "danger",
                                "monitoring-critical-count"
                            )
                        ], width=3)
                    ])
                ], width=9)
            ], className="mb-4"),

            # Main Monitoring Charts
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-chart-line me-2"),
                                "Sensor Time Series"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dcc.Loading(
                                dcc.Graph(id='monitoring-timeseries-chart', config={'displayModeBar': False}),
                                type="circle"
                            )
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),

            # Real-time Indicators
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Current Reading"),
                        dbc.CardBody([
                            html.Div([
                                daq.Gauge(
                                    id='monitoring-current-gauge',
                                    value=0,
                                    min=0,
                                    max=100,
                                    showCurrentValue=True,
                                    units="units",
                                    color="#00cc00",
                                    size=180
                                )
                            ], className="text-center")
                        ])
                    ])
                ], width=4),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Statistics"),
                        dbc.CardBody([
                            html.Div(id='monitoring-stats-panel', children=[
                                self._create_stat_row("Mean", "0.00"),
                                self._create_stat_row("Std Dev", "0.00"),
                                self._create_stat_row("Min", "0.00"),
                                self._create_stat_row("Max", "0.00"),
                                self._create_stat_row("Last Update", datetime.now().strftime("%H:%M:%S"))
                            ])
                        ])
                    ])
                ], width=4),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Sensor Health"),
                        dbc.CardBody([
                            html.Div([
                                self._create_health_indicator("Connection", True),
                                self._create_health_indicator("Data Quality", True),
                                self._create_health_indicator("Latency", True),
                                self._create_health_indicator("Model Active", True)
                            ])
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),

            # Refresh interval
            dcc.Interval(id='monitoring-refresh-interval', interval=10000, n_intervals=0)

        ], fluid=True, className="mt-2")

    def _create_status_card(self, title: str, value: str, icon: str,
                           color: str, card_id: str) -> dbc.Card:
        """Create status card"""
        return dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.I(className=f"{icon} fa-2x text-{color} mb-2"),
                    html.H3(value, className="mb-0", id=card_id),
                    html.P(title, className="text-muted small mb-0")
                ], className="text-center")
            ])
        ])

    def _create_stat_row(self, label: str, value: str) -> html.Div:
        """Create statistics row"""
        return html.Div([
            html.Span(label, className="fw-bold me-2"),
            html.Span(value, className="text-muted")
        ], className="mb-2")

    def _create_health_indicator(self, label: str, healthy: bool) -> html.Div:
        """Create health status indicator"""
        return html.Div([
            html.Div([
                daq.Indicator(
                    value=healthy,
                    color="#00cc00" if healthy else "#cc0000",
                    size=12,
                    className="me-2"
                ),
                html.Small(label)
            ], className="d-flex align-items-center mb-2")
        ])


def create_layout():
    """Create monitoring page layout for dashboard routing"""
    dashboard = MonitoringDashboard()
    return dashboard.create_layout()


# Register callbacks
@callback(
    Output('monitoring-timeseries-chart', 'figure'),
    [Input('monitoring-equipment-selector', 'value'),
     Input('monitoring-time-range', 'value'),
     Input('monitoring-refresh-btn', 'n_clicks'),
     Input('monitoring-refresh-interval', 'n_intervals')]
)
def update_timeseries_chart(equipment_id, time_range, n_clicks, n_intervals):
    """Update time series chart with sensor data"""
    try:
        # Generate sample data for demonstration
        hours = {'1h': 1, '6h': 6, '24h': 24, '7d': 168}.get(time_range, 24)
        timestamps = pd.date_range(end=datetime.now(), periods=hours*12, freq='5min')

        # Create sample sensor data
        np.random.seed(42)
        values = 50 + 10 * np.sin(np.linspace(0, 4*np.pi, len(timestamps))) + np.random.randn(len(timestamps)) * 2

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode='lines',
            name=equipment_id or 'Sensor',
            line=dict(color='#00cc00', width=2)
        ))

        fig.update_layout(
            title=f"Sensor Data: {equipment_id or 'Unknown'}",
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified',
            height=400,
            margin=dict(l=50, r=20, t=40, b=40)
        )

        return fig

    except Exception as e:
        logger.error(f"Error updating chart: {e}")
        # Return empty figure on error
        return go.Figure()


@callback(
    Output('monitoring-current-gauge', 'value'),
    [Input('monitoring-refresh-interval', 'n_intervals')]
)
def update_current_gauge(n_intervals):
    """Update current reading gauge"""
    # Generate sample current reading
    return 45 + 15 * np.sin(n_intervals * 0.1) + np.random.randn() * 2


def register_callbacks(app, data_service=None):
    """Register callbacks for monitoring dashboard (placeholder for compatibility)"""
    # Callbacks are auto-registered via @callback decorators
    logger.info("Monitoring dashboard callbacks are auto-registered via @callback decorators")
    return True
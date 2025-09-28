"""
Main Dashboard Application
Clean, restructured IoT Predictive Maintenance Dashboard
"""

import dash
from dash import dcc, html, Input, Output, callback, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import our clean services
from src.core.services.anomaly_service import AnomalyDetectionService
from src.core.services.forecasting_service import ForecastingService
from src.infrastructure.data.nasa_data_loader import NASADataLoader
from config.equipment_config import EQUIPMENT_REGISTRY, get_equipment_list

logger = logging.getLogger(__name__)


class IoTPredictiveDashboard:
    """
    Clean IoT Predictive Maintenance Dashboard
    Using restructured services with Telemanom and Transformer models only
    """

    def __init__(self, debug: bool = False):
        """Initialize the dashboard with clean architecture"""
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
            suppress_callback_exceptions=True,
            title="IoT Predictive Maintenance - Clean Architecture"
        )

        # Initialize clean services
        self.anomaly_service = AnomalyDetectionService()
        self.forecasting_service = ForecastingService()
        self.data_loader = NASADataLoader()

        # Load equipment configuration (12 sensors)
        self.equipment_list = get_equipment_list()
        self.sensor_ids = [eq.equipment_id for eq in self.equipment_list]

        # Setup dashboard
        self._setup_layout()
        self._setup_callbacks()

        logger.info("Clean IoT Dashboard initialized successfully")

    def _setup_layout(self):
        """Setup the main dashboard layout"""
        # Header
        header = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("IoT Predictive Maintenance", className="text-primary mb-0"),
                    html.P("Clean Architecture - Telemanom + Transformer", className="text-muted")
                ], width=8),
                dbc.Col([
                    html.Div(id="system-status-badges")
                ], width=4, className="text-end")
            ])
        ], fluid=True, className="bg-light py-3 mb-4")

        # Navigation tabs
        nav_tabs = dbc.Tabs([
            dbc.Tab(label="📊 System Overview", tab_id="overview"),
            dbc.Tab(label="📈 Real-time Monitoring", tab_id="monitoring"),
            dbc.Tab(label="🚨 Anomaly Detection", tab_id="anomalies"),
            dbc.Tab(label="🔮 Forecasting", tab_id="forecasting"),
            dbc.Tab(label="🔧 Maintenance", tab_id="maintenance")
        ], id="main-tabs", active_tab="overview")

        # Main content
        main_content = dbc.Container([
            dcc.Interval(id='refresh-interval', interval=30*1000, n_intervals=0),
            nav_tabs,
            html.Div(id="tab-content", className="mt-4")
        ], fluid=True)

        self.app.layout = html.Div([header, main_content])

    def _setup_callbacks(self):
        """Setup dashboard callbacks"""

        @self.app.callback(
            Output('system-status-badges', 'children'),
            [Input('refresh-interval', 'n_intervals')]
        )
        def update_system_status(n):
            """Update system status badges"""
            try:
                # Get system statistics
                anomaly_summary = self.anomaly_service.get_detection_summary()
                forecast_summary = self.forecasting_service.get_forecast_summary()

                total_anomalies = anomaly_summary.get('total_anomalies', 0)
                total_forecasts = forecast_summary.get('total_sensors_forecasted', 0)

                badges = [
                    dbc.Badge(f"{len(self.sensor_ids)} Sensors", color="success", className="me-2"),
                    dbc.Badge(
                        f"{total_anomalies} Anomalies",
                        color="danger" if total_anomalies > 0 else "success",
                        className="me-2"
                    ),
                    dbc.Badge(f"{total_forecasts} Forecasts", color="info")
                ]

                return badges

            except Exception as e:
                logger.error(f"Error updating system status: {e}")
                return [dbc.Badge("System Loading...", color="secondary")]

        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('main-tabs', 'active_tab')]
        )
        def render_tab_content(active_tab):
            """Render content based on active tab"""
            try:
                if active_tab == "overview":
                    return self._create_overview_tab()
                elif active_tab == "monitoring":
                    return self._create_monitoring_tab()
                elif active_tab == "anomalies":
                    return self._create_anomalies_tab()
                elif active_tab == "forecasting":
                    return self._create_forecasting_tab()
                elif active_tab == "maintenance":
                    return self._create_maintenance_tab()
                else:
                    return html.Div("Loading...")
            except Exception as e:
                logger.error(f"Error rendering tab {active_tab}: {e}")
                return dbc.Alert(f"Error loading {active_tab}: {str(e)}", color="danger")

    def _create_overview_tab(self):
        """Create system overview tab"""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🏭 Equipment Overview"),
                        dbc.CardBody([
                            html.Div(id="equipment-overview")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 System Health"),
                        dbc.CardBody([
                            html.Div(id="system-health")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📈 Recent Activity"),
                        dbc.CardBody([
                            html.Div(id="recent-activity")
                        ])
                    ])
                ])
            ])
        ])

    def _create_monitoring_tab(self):
        """Create real-time monitoring tab"""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📈 Sensor Selection"),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='sensor-selector',
                                options=[
                                    {'label': f"{eq.equipment_id} - {eq.name}", 'value': eq.equipment_id}
                                    for eq in self.equipment_list
                                ],
                                value=self.equipment_list[0].equipment_id if self.equipment_list else None,
                                placeholder="Select a sensor to monitor"
                            )
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 Real-time Data"),
                        dbc.CardBody([
                            dcc.Graph(id="realtime-chart")
                        ])
                    ])
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("ℹ️ Sensor Information"),
                        dbc.CardBody([
                            html.Div(id="sensor-info")
                        ])
                    ])
                ], width=4)
            ])
        ])

    def _create_anomalies_tab(self):
        """Create anomaly detection tab"""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🚨 Anomaly Detection - Telemanom Algorithm"),
                        dbc.CardBody([
                            html.Div(id="anomaly-results")
                        ])
                    ])
                ])
            ])
        ])

    def _create_forecasting_tab(self):
        """Create forecasting tab"""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🔮 Predictive Forecasting - Transformer Model"),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='forecast-sensor-selector',
                                options=[
                                    {'label': f"{eq.equipment_id} - {eq.name}", 'value': eq.equipment_id}
                                    for eq in self.equipment_list
                                ],
                                value=self.equipment_list[0].equipment_id if self.equipment_list else None,
                                placeholder="Select sensor for forecasting",
                                className="mb-3"
                            ),
                            dcc.Graph(id="forecast-chart")
                        ])
                    ])
                ])
            ])
        ])

    def _create_maintenance_tab(self):
        """Create maintenance tab"""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🔧 Maintenance Dashboard"),
                        dbc.CardBody([
                            html.Div([
                                html.H5("Predictive Maintenance System"),
                                html.P("Maintenance scheduling based on anomaly detection and forecasting results."),
                                dbc.Alert("Maintenance module will be enhanced in the next development phase.", color="info")
                            ])
                        ])
                    ])
                ])
            ])
        ])

        # Add specific callbacks for monitoring tab
        @self.app.callback(
            [Output('realtime-chart', 'figure'),
             Output('sensor-info', 'children')],
            [Input('sensor-selector', 'value'),
             Input('refresh-interval', 'n_intervals')]
        )
        def update_monitoring(sensor_id, n):
            """Update real-time monitoring chart and sensor info"""
            if not sensor_id:
                return {}, "Select a sensor to view data"

            try:
                # Get sensor data (mock for now)
                data = self._get_sensor_data(sensor_id)

                # Create chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data['timestamps'],
                    y=data['values'],
                    mode='lines+markers',
                    name=sensor_id,
                    line=dict(color='blue', width=2)
                ))

                fig.update_layout(
                    title=f"Real-time Data: {sensor_id}",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    height=400,
                    showlegend=False
                )

                # Sensor info
                equipment = next((eq for eq in self.equipment_list if eq.equipment_id == sensor_id), None)
                if equipment:
                    info = dbc.ListGroup([
                        dbc.ListGroupItem([
                            html.Strong("Equipment: "), equipment.name
                        ]),
                        dbc.ListGroupItem([
                            html.Strong("Type: "), equipment.equipment_type.value
                        ]),
                        dbc.ListGroupItem([
                            html.Strong("Location: "), equipment.location
                        ]),
                        dbc.ListGroupItem([
                            html.Strong("Criticality: "), equipment.criticality.value
                        ]),
                        dbc.ListGroupItem([
                            html.Strong("Unit: "), equipment.unit
                        ])
                    ])
                else:
                    info = "Sensor information not found"

                return fig, info

            except Exception as e:
                logger.error(f"Error updating monitoring for {sensor_id}: {e}")
                return {}, f"Error: {str(e)}"

        @self.app.callback(
            Output('anomaly-results', 'children'),
            [Input('refresh-interval', 'n_intervals')]
        )
        def update_anomaly_results(n):
            """Update anomaly detection results"""
            try:
                summary = self.anomaly_service.get_detection_summary()

                if summary.get('total_anomalies', 0) == 0:
                    return dbc.Alert("✅ No anomalies detected", color="success")

                # Create anomaly cards
                anomaly_cards = []
                for anomaly in summary.get('recent_anomalies', [])[:10]:
                    severity = anomaly.get('severity', 'MEDIUM')
                    color_map = {
                        'CRITICAL': 'danger',
                        'HIGH': 'warning',
                        'MEDIUM': 'info',
                        'LOW': 'light'
                    }
                    color = color_map.get(severity, 'info')

                    card = dbc.Card([
                        dbc.CardBody([
                            html.H6(f"🚨 {anomaly.get('sensor_id', 'Unknown')}", className="card-title"),
                            html.P([
                                html.Strong("Severity: "),
                                dbc.Badge(severity, color=color, className="me-2")
                            ]),
                            html.P(f"Score: {anomaly.get('score', 0):.3f}"),
                            html.P(f"Value: {anomaly.get('value', 0):.2f}"),
                            html.Small(f"Time: {anomaly.get('timestamp', 'Unknown')}")
                        ])
                    ], color=color, outline=True, className="mb-2")

                    anomaly_cards.append(card)

                return anomaly_cards

            except Exception as e:
                logger.error(f"Error updating anomaly results: {e}")
                return dbc.Alert(f"Error loading anomalies: {str(e)}", color="danger")

        @self.app.callback(
            Output('forecast-chart', 'figure'),
            [Input('forecast-sensor-selector', 'value')]
        )
        def update_forecast_chart(sensor_id):
            """Update forecasting chart"""
            if not sensor_id:
                return {}

            try:
                # Get historical data and generate forecast
                historical_data = self._get_sensor_data(sensor_id)
                forecast_result = self.forecasting_service.generate_forecast(
                    sensor_id,
                    np.array(historical_data['values']),
                    horizon_hours=24
                )

                fig = go.Figure()

                # Historical data
                fig.add_trace(go.Scatter(
                    x=historical_data['timestamps'],
                    y=historical_data['values'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue')
                ))

                # Forecast data
                if forecast_result.get('forecast_timestamps'):
                    fig.add_trace(go.Scatter(
                        x=[datetime.fromisoformat(ts) for ts in forecast_result['forecast_timestamps']],
                        y=forecast_result['forecast_values'],
                        mode='lines',
                        name='Forecast',
                        line=dict(color='red', dash='dash')
                    ))

                    # Confidence intervals
                    if forecast_result.get('confidence_upper') and forecast_result.get('confidence_lower'):
                        fig.add_trace(go.Scatter(
                            x=[datetime.fromisoformat(ts) for ts in forecast_result['forecast_timestamps']],
                            y=forecast_result['confidence_upper'],
                            fill=None,
                            mode='lines',
                            line_color='rgba(0,0,0,0)',
                            showlegend=False
                        ))

                        fig.add_trace(go.Scatter(
                            x=[datetime.fromisoformat(ts) for ts in forecast_result['forecast_timestamps']],
                            y=forecast_result['confidence_lower'],
                            fill='tonexty',
                            mode='lines',
                            line_color='rgba(0,0,0,0)',
                            name='Confidence Interval',
                            fillcolor='rgba(255,0,0,0.2)'
                        ))

                fig.update_layout(
                    title=f"24-hour Forecast: {sensor_id}",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    height=400
                )

                return fig

            except Exception as e:
                logger.error(f"Error generating forecast for {sensor_id}: {e}")
                return {}

    def _get_sensor_data(self, sensor_id: str, hours_back: int = 48) -> Dict:
        """Get sensor data (mock implementation)"""
        try:
            # Create mock time series data
            now = datetime.now()
            timestamps = [now - timedelta(hours=hours_back-i) for i in range(hours_back)]

            # Generate realistic sensor data based on equipment type
            equipment = next((eq for eq in self.equipment_list if eq.equipment_id == sensor_id), None)
            if equipment:
                base_value = (equipment.normal_range[0] + equipment.normal_range[1]) / 2
                noise_level = (equipment.normal_range[1] - equipment.normal_range[0]) * 0.1
            else:
                base_value = 50.0
                noise_level = 5.0

            # Generate values with some trend and noise
            values = []
            for i in range(hours_back):
                trend = np.sin(i * 0.1) * noise_level * 0.5
                noise = np.random.normal(0, noise_level * 0.3)
                value = base_value + trend + noise
                values.append(value)

            return {
                'timestamps': timestamps,
                'values': values,
                'sensor_info': equipment.__dict__ if equipment else {}
            }

        except Exception as e:
            logger.error(f"Error getting sensor data for {sensor_id}: {e}")
            return {
                'timestamps': [],
                'values': [],
                'sensor_info': {}
            }

    def run(self, host: str = '127.0.0.1', port: int = 8050, debug: bool = False):
        """Run the dashboard"""
        logger.info(f"Starting Clean IoT Dashboard at http://{host}:{port}")
        logger.info("Architecture: Clean Services | Telemanom + Transformer | 12 Sensors")
        self.app.run(host=host, port=port, debug=debug)


# Factory function
def create_dashboard(debug: bool = False) -> IoTPredictiveDashboard:
    """Create and return dashboard instance"""
    return IoTPredictiveDashboard(debug=debug)


if __name__ == "__main__":
    dashboard = create_dashboard(debug=True)
    dashboard.run(debug=True)
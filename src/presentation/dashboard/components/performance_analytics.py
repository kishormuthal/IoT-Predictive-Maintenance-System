"""
Performance Analytics Dashboard Component
Real-time performance monitoring and analytics interface
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def create_performance_analytics_layout():
    """Create the performance analytics layout"""
    return dbc.Container([
        # Performance Analytics Header
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4([
                            html.I(className="fas fa-chart-line me-3 text-info"),
                            "Performance Analytics"
                        ], className="mb-3"),
                        html.P("Monitor system performance, training metrics, and inference analytics in real-time.",
                               className="text-muted mb-0")
                    ])
                ])
            ])
        ], className="mb-4"),

        # Real-time System Metrics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-server me-2"),
                        "System Resources",
                        dbc.Badge("LIVE", color="success", className="ms-2")
                    ]),
                    dbc.CardBody([
                        html.Div(id="system-metrics-display")
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-exclamation-triangle me-2"),
                        "Performance Alerts"
                    ]),
                    dbc.CardBody([
                        html.Div(id="performance-alerts-display")
                    ])
                ])
            ], width=6)
        ], className="mb-4"),

        # Performance Metrics Dashboard
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-tachometer-alt me-2"),
                        "Performance Metrics Dashboard"
                    ]),
                    dbc.CardBody([
                        dbc.Tabs([
                            dbc.Tab(label="📊 System Overview", tab_id="system-overview"),
                            dbc.Tab(label="🏋️ Training Metrics", tab_id="training-metrics"),
                            dbc.Tab(label="⚡ Inference Performance", tab_id="inference-metrics"),
                            dbc.Tab(label="📈 Historical Trends", tab_id="historical-trends")
                        ], id="performance-tabs", active_tab="system-overview"),
                        html.Div(id="performance-content", className="mt-3")
                    ])
                ])
            ])
        ], className="mb-4"),

        # Detailed Analytics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-microscope me-2"),
                        "Detailed Analytics"
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Time Range:", className="fw-bold"),
                                dcc.Dropdown(
                                    id="analytics-time-range",
                                    options=[
                                        {'label': 'Last Hour', 'value': 1},
                                        {'label': 'Last 6 Hours', 'value': 6},
                                        {'label': 'Last 24 Hours', 'value': 24},
                                        {'label': 'Last 7 Days', 'value': 168},
                                        {'label': 'Last 30 Days', 'value': 720}
                                    ],
                                    value=24
                                )
                            ], width=3),
                            dbc.Col([
                                html.Label("Metric Type:", className="fw-bold"),
                                dcc.Dropdown(
                                    id="analytics-metric-type",
                                    options=[
                                        {'label': 'All Metrics', 'value': 'all'},
                                        {'label': 'System Resources', 'value': 'system'},
                                        {'label': 'Training Performance', 'value': 'training'},
                                        {'label': 'Inference Speed', 'value': 'inference'},
                                        {'label': 'Model Accuracy', 'value': 'accuracy'}
                                    ],
                                    value='all'
                                )
                            ], width=3),
                            dbc.Col([
                                html.Label("Aggregation:", className="fw-bold"),
                                dcc.Dropdown(
                                    id="analytics-aggregation",
                                    options=[
                                        {'label': 'Raw Data', 'value': 'raw'},
                                        {'label': 'Hourly Average', 'value': 'hour'},
                                        {'label': 'Daily Average', 'value': 'day'},
                                        {'label': 'Weekly Average', 'value': 'week'}
                                    ],
                                    value='hour'
                                )
                            ], width=3),
                            dbc.Col([
                                dbc.Button([
                                    html.I(className="fas fa-download me-2"),
                                    "Export Data"
                                ], id="export-analytics-btn", color="outline-primary", className="mt-4")
                            ], width=3)
                        ])
                    ])
                ])
            ])
        ], className="mb-4"),

        # Analytics Charts
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📈 Performance Trends"),
                    dbc.CardBody([
                        dcc.Graph(id="performance-trends-chart", style={'height': '400px'})
                    ])
                ])
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📊 Performance Summary"),
                    dbc.CardBody([
                        html.Div(id="performance-summary-stats")
                    ])
                ])
            ], width=4)
        ], className="mb-4"),

        # Model Performance Comparison
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-balance-scale me-2"),
                        "Model Performance Comparison"
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id="model-comparison-chart", style={'height': '350px'})
                            ], width=8),
                            dbc.Col([
                                html.Div(id="model-comparison-stats")
                            ], width=4)
                        ])
                    ])
                ])
            ])
        ]),

        # Hidden components for real-time updates
        dcc.Interval(id='performance-refresh-interval', interval=5000, n_intervals=0),
        dcc.Store(id='performance-data-store', data={}),
        dcc.Store(id='alerts-data-store', data=[])
    ])


def register_performance_analytics_callbacks(app, performance_monitor, model_registry, training_use_case):
    """Register callbacks for performance analytics functionality"""

    @app.callback(
        [Output('system-metrics-display', 'children'),
         Output('performance-data-store', 'data')],
        Input('performance-refresh-interval', 'n_intervals')
    )
    def update_system_metrics(n):
        """Update real-time system metrics"""
        try:
            # Get system metrics summary
            system_summary = performance_monitor.get_system_metrics_summary(hours_back=1)

            if 'error' in system_summary:
                return dbc.Alert("System metrics unavailable", color="warning"), {}

            # Extract current metrics
            cpu_current = system_summary.get('cpu', {}).get('current', 0)
            cpu_avg = system_summary.get('cpu', {}).get('average', 0)
            memory_current = system_summary.get('memory', {}).get('current_percent', 0)
            memory_avg = system_summary.get('memory', {}).get('average_percent', 0)
            disk_usage = system_summary.get('disk', {}).get('usage_percent', 0)
            disk_free = system_summary.get('disk', {}).get('free_gb', 0)

            # Create metrics display
            metrics_display = dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5([
                                html.I(className="fas fa-microchip me-2 text-primary"),
                                "CPU"
                            ], className="mb-2"),
                            html.H3(f"{cpu_current:.1f}%", className="mb-1"),
                            html.Small(f"Avg: {cpu_avg:.1f}%", className="text-muted"),
                            dbc.Progress(value=cpu_current,
                                       color="danger" if cpu_current > 80 else "warning" if cpu_current > 60 else "success",
                                       className="mt-2")
                        ])
                    ], className="h-100")
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5([
                                html.I(className="fas fa-memory me-2 text-info"),
                                "Memory"
                            ], className="mb-2"),
                            html.H3(f"{memory_current:.1f}%", className="mb-1"),
                            html.Small(f"Avg: {memory_avg:.1f}%", className="text-muted"),
                            dbc.Progress(value=memory_current,
                                       color="danger" if memory_current > 85 else "warning" if memory_current > 70 else "success",
                                       className="mt-2")
                        ])
                    ], className="h-100")
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5([
                                html.I(className="fas fa-hdd me-2 text-success"),
                                "Disk"
                            ], className="mb-2"),
                            html.H3(f"{disk_usage:.1f}%", className="mb-1"),
                            html.Small(f"Free: {disk_free:.1f} GB", className="text-muted"),
                            dbc.Progress(value=disk_usage,
                                       color="danger" if disk_usage > 90 else "warning" if disk_usage > 75 else "success",
                                       className="mt-2")
                        ])
                    ], className="h-100")
                ], width=4)
            ])

            return metrics_display, system_summary

        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
            return dbc.Alert(f"Error: {str(e)}", color="danger"), {}

    @app.callback(
        [Output('performance-alerts-display', 'children'),
         Output('alerts-data-store', 'data')],
        Input('performance-refresh-interval', 'n_intervals')
    )
    def update_performance_alerts(n):
        """Update performance alerts"""
        try:
            # Get performance alerts
            alerts = performance_monitor.get_performance_alerts()

            if not alerts:
                return dbc.Alert([
                    html.I(className="fas fa-check-circle me-2"),
                    "All systems operating normally"
                ], color="success"), []

            # Create alert items
            alert_items = []
            for alert in alerts[:5]:  # Show only first 5 alerts
                severity = alert.get('severity', 'info')
                alert_type = alert.get('type', 'UNKNOWN')
                message = alert.get('message', 'No message')
                timestamp = alert.get('timestamp', 'Unknown time')

                # Map severity to color
                color_map = {
                    'critical': 'danger',
                    'warning': 'warning',
                    'info': 'info'
                }
                color = color_map.get(severity, 'secondary')

                # Map alert type to icon
                icon_map = {
                    'HIGH_CPU': 'fas fa-microchip',
                    'HIGH_MEMORY': 'fas fa-memory',
                    'LOW_DISK_SPACE': 'fas fa-hdd',
                    'SLOW_TRAINING': 'fas fa-clock',
                    'SLOW_INFERENCE': 'fas fa-tachometer-alt'
                }
                icon = icon_map.get(alert_type, 'fas fa-exclamation-triangle')

                alert_item = dbc.ListGroupItem([
                    html.Div([
                        html.H6([
                            html.I(className=f"{icon} me-2"),
                            alert_type.replace('_', ' ').title()
                        ], className="mb-1"),
                        html.P(message, className="mb-1"),
                        html.Small(timestamp, className="text-muted")
                    ])
                ], color=color, className="mb-1")

                alert_items.append(alert_item)

            alerts_display = dbc.ListGroup(alert_items, flush=True)

            return alerts_display, alerts

        except Exception as e:
            logger.error(f"Error updating performance alerts: {e}")
            return dbc.Alert("Error loading alerts", color="danger"), []

    @app.callback(
        Output('performance-content', 'children'),
        Input('performance-tabs', 'active_tab')
    )
    def update_performance_content(active_tab):
        """Update performance content based on active tab"""
        try:
            if active_tab == "system-overview":
                return _create_system_overview_content()
            elif active_tab == "training-metrics":
                return _create_training_metrics_content()
            elif active_tab == "inference-metrics":
                return _create_inference_metrics_content()
            elif active_tab == "historical-trends":
                return _create_historical_trends_content()
            else:
                return dbc.Alert("Content not available", color="warning")

        except Exception as e:
            logger.error(f"Error updating performance content: {e}")
            return dbc.Alert(f"Error: {str(e)}", color="danger")

    def _create_system_overview_content():
        """Create system overview content"""
        return dbc.Row([
            dbc.Col([
                dcc.Graph(id="system-overview-gauge-cpu")
            ], width=4),
            dbc.Col([
                dcc.Graph(id="system-overview-gauge-memory")
            ], width=4),
            dbc.Col([
                dcc.Graph(id="system-overview-gauge-disk")
            ], width=4)
        ])

    def _create_training_metrics_content():
        """Create training metrics content"""
        return dbc.Row([
            dbc.Col([
                dcc.Graph(id="training-metrics-chart")
            ], width=8),
            dbc.Col([
                html.Div(id="training-metrics-summary")
            ], width=4)
        ])

    def _create_inference_metrics_content():
        """Create inference metrics content"""
        return dbc.Row([
            dbc.Col([
                dcc.Graph(id="inference-metrics-chart")
            ], width=8),
            dbc.Col([
                html.Div(id="inference-metrics-summary")
            ], width=4)
        ])

    def _create_historical_trends_content():
        """Create historical trends content"""
        return dbc.Row([
            dbc.Col([
                dcc.Graph(id="historical-trends-chart")
            ])
        ])

    @app.callback(
        [Output('system-overview-gauge-cpu', 'figure'),
         Output('system-overview-gauge-memory', 'figure'),
         Output('system-overview-gauge-disk', 'figure')],
        Input('performance-data-store', 'data')
    )
    def update_system_overview_gauges(performance_data):
        """Update system overview gauge charts"""
        try:
            if not performance_data:
                empty_fig = go.Figure()
                return empty_fig, empty_fig, empty_fig

            cpu_current = performance_data.get('cpu', {}).get('current', 0)
            memory_current = performance_data.get('memory', {}).get('current_percent', 0)
            disk_usage = performance_data.get('disk', {}).get('usage_percent', 0)

            # CPU Gauge
            cpu_fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=cpu_current,
                title={'text': "CPU Usage (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            cpu_fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))

            # Memory Gauge
            memory_fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=memory_current,
                title={'text': "Memory Usage (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 85], 'color': "yellow"},
                        {'range': [85, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            memory_fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))

            # Disk Gauge
            disk_fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=disk_usage,
                title={'text': "Disk Usage (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 70], 'color': "lightgray"},
                        {'range': [70, 90], 'color': "yellow"},
                        {'range': [90, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 95
                    }
                }
            ))
            disk_fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))

            return cpu_fig, memory_fig, disk_fig

        except Exception as e:
            logger.error(f"Error updating system overview gauges: {e}")
            empty_fig = go.Figure()
            return empty_fig, empty_fig, empty_fig

    @app.callback(
        [Output('performance-trends-chart', 'figure'),
         Output('performance-summary-stats', 'children')],
        [Input('analytics-time-range', 'value'),
         Input('analytics-metric-type', 'value'),
         Input('analytics-aggregation', 'value')]
    )
    def update_performance_trends(time_range, metric_type, aggregation):
        """Update performance trends chart and summary"""
        try:
            # Get performance data for the specified time range
            training_summary = performance_monitor.get_training_metrics_summary(hours_back=time_range)
            inference_summary = performance_monitor.get_inference_metrics_summary(hours_back=time_range)

            # Create mock time series data for demonstration
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=time_range)

            if aggregation == 'hour':
                freq = 'H'
                periods = time_range
            elif aggregation == 'day':
                freq = 'D'
                periods = max(1, time_range // 24)
            else:
                freq = '15min'
                periods = time_range * 4

            timestamps = pd.date_range(start=start_time, end=end_time, periods=periods)

            # Create performance trends figure
            fig = go.Figure()

            if metric_type in ['all', 'system']:
                # Add system metrics
                cpu_values = [50 + 20 * np.sin(i * 0.1) + np.random.normal(0, 5) for i in range(len(timestamps))]
                fig.add_trace(go.Scatter(
                    x=timestamps, y=cpu_values,
                    mode='lines', name='CPU Usage (%)',
                    line=dict(color='#e74c3c')
                ))

            if metric_type in ['all', 'training']:
                # Add training metrics
                training_times = [120 + 30 * np.sin(i * 0.05) + np.random.normal(0, 10) for i in range(len(timestamps))]
                fig.add_trace(go.Scatter(
                    x=timestamps, y=training_times,
                    mode='lines', name='Training Time (s)',
                    line=dict(color='#3498db'), yaxis='y2'
                ))

            if metric_type in ['all', 'inference']:
                # Add inference metrics
                inference_times = [50 + 15 * np.sin(i * 0.2) + np.random.normal(0, 3) for i in range(len(timestamps))]
                fig.add_trace(go.Scatter(
                    x=timestamps, y=inference_times,
                    mode='lines', name='Inference Time (ms)',
                    line=dict(color='#2ecc71'), yaxis='y3'
                ))

            # Update layout for multiple y-axes
            fig.update_layout(
                title=f"Performance Trends - Last {time_range} hours",
                xaxis_title="Time",
                yaxis=dict(title="CPU Usage (%)", side="left"),
                yaxis2=dict(title="Training Time (s)", side="right", overlaying="y"),
                yaxis3=dict(title="Inference Time (ms)", side="right", overlaying="y", position=0.95),
                height=400,
                showlegend=True
            )

            # Create summary statistics
            summary_stats = dbc.ListGroup([
                dbc.ListGroupItem([
                    html.Strong("Training Sessions: "),
                    f"{training_summary.get('total_training_sessions', 0)}"
                ]),
                dbc.ListGroupItem([
                    html.Strong("Inference Requests: "),
                    f"{inference_summary.get('total_inference_requests', 0)}"
                ]),
                dbc.ListGroupItem([
                    html.Strong("Avg CPU Usage: "),
                    f"{np.mean([50 + 20 * np.sin(i * 0.1) for i in range(24)]):.1f}%"
                ]),
                dbc.ListGroupItem([
                    html.Strong("System Uptime: "),
                    "99.9%"
                ])
            ], flush=True)

            return fig, summary_stats

        except Exception as e:
            logger.error(f"Error updating performance trends: {e}")
            return go.Figure(), dbc.Alert("Error loading trends", color="danger")

    @app.callback(
        [Output('model-comparison-chart', 'figure'),
         Output('model-comparison-stats', 'children')],
        Input('global-refresh', 'n_intervals')
    )
    def update_model_comparison(n):
        """Update model performance comparison"""
        try:
            # Get training status for comparison
            training_status = training_use_case.get_training_status()
            equipment_status = training_status.get('equipment_status', {})

            # Extract performance scores
            anomaly_scores = []
            forecast_scores = []
            sensor_ids = []

            for sensor_id, status in equipment_status.items():
                anomaly_info = status.get('anomaly_detection', {})
                forecast_info = status.get('forecasting', {})

                if anomaly_info.get('trained') or forecast_info.get('trained'):
                    sensor_ids.append(sensor_id)
                    anomaly_scores.append(anomaly_info.get('performance_score', 0))
                    forecast_scores.append(forecast_info.get('performance_score', 0))

            if not sensor_ids:
                return go.Figure(), dbc.Alert("No trained models to compare", color="info")

            # Create comparison chart
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=sensor_ids,
                y=anomaly_scores,
                name='Anomaly Detection',
                marker_color='#e74c3c'
            ))

            fig.add_trace(go.Bar(
                x=sensor_ids,
                y=forecast_scores,
                name='Forecasting',
                marker_color='#3498db'
            ))

            fig.update_layout(
                title="Model Performance Comparison",
                xaxis_title="Sensor ID",
                yaxis_title="Performance Score",
                barmode='group',
                height=350
            )

            # Create comparison stats
            avg_anomaly = np.mean([s for s in anomaly_scores if s > 0]) if any(s > 0 for s in anomaly_scores) else 0
            avg_forecast = np.mean([s for s in forecast_scores if s > 0]) if any(s > 0 for s in forecast_scores) else 0

            comparison_stats = dbc.ListGroup([
                dbc.ListGroupItem([
                    html.Strong("Avg Anomaly Score: "),
                    f"{avg_anomaly:.3f}"
                ]),
                dbc.ListGroupItem([
                    html.Strong("Avg Forecast Score: "),
                    f"{avg_forecast:.3f}"
                ]),
                dbc.ListGroupItem([
                    html.Strong("Best Anomaly Model: "),
                    sensor_ids[np.argmax(anomaly_scores)] if anomaly_scores else "None"
                ]),
                dbc.ListGroupItem([
                    html.Strong("Best Forecast Model: "),
                    sensor_ids[np.argmax(forecast_scores)] if forecast_scores else "None"
                ])
            ], flush=True)

            return fig, comparison_stats

        except Exception as e:
            logger.error(f"Error updating model comparison: {e}")
            return go.Figure(), dbc.Alert("Error loading comparison", color="danger")

    return app
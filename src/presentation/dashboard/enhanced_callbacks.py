"""
Enhanced Dashboard Callbacks Integration
Complete callback integration for all enhanced dashboard components
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Any

from .components.training_hub import register_training_hub_callbacks, create_training_hub_layout
from .components.model_registry import register_model_registry_callbacks, create_model_registry_layout
from .components.performance_analytics import register_performance_analytics_callbacks, create_performance_analytics_layout
from .components.system_admin import register_system_admin_callbacks, create_system_admin_layout

logger = logging.getLogger(__name__)


def register_enhanced_callbacks(app, services):
    """Register all enhanced dashboard callbacks"""

    # Extract services
    anomaly_service = services['anomaly_service']
    forecasting_service = services['forecasting_service']
    data_loader = services['data_loader']
    training_use_case = services['training_use_case']
    config_manager = services['config_manager']
    model_registry = services['model_registry']
    performance_monitor = services['performance_monitor']
    equipment_list = services['equipment_list']

    # Register component-specific callbacks
    register_training_hub_callbacks(app, training_use_case, config_manager, performance_monitor)
    register_model_registry_callbacks(app, model_registry, training_use_case, config_manager)
    register_performance_analytics_callbacks(app, performance_monitor, model_registry, training_use_case)
    register_system_admin_callbacks(app, performance_monitor, config_manager, model_registry, training_use_case)

    # Enhanced tab content callbacks
    @app.callback(
        Output('training-hub-content', 'children'),
        Input('main-tabs', 'active_tab')
    )
    def render_training_hub_content(active_tab):
        """Render training hub content when tab is active"""
        if active_tab == 'training':
            return create_training_hub_layout()
        return dash.no_update

    @app.callback(
        Output('models-tab-content', 'children'),
        Input('main-tabs', 'active_tab')
    )
    def render_models_tab_content(active_tab):
        """Render models tab content when tab is active"""
        if active_tab == 'models':
            return create_model_registry_layout()
        return dash.no_update

    @app.callback(
        Output('admin-tab-content', 'children'),
        Input('main-tabs', 'active_tab')
    )
    def render_admin_tab_content(active_tab):
        """Render admin tab content when tab is active"""
        if active_tab == 'admin':
            return create_system_admin_layout()
        return dash.no_update

    # Enhanced monitoring callbacks
    @app.callback(
        Output('trained-models-count', 'children'),
        Input('global-refresh', 'n_intervals')
    )
    def update_trained_models_count(n):
        """Update trained models count in overview"""
        try:
            registry_stats = model_registry.get_registry_stats()
            return str(registry_stats.get('total_models', 0))
        except Exception as e:
            logger.error(f"Error updating trained models count: {e}")
            return "0"

    @app.callback(
        Output('equipment-status-grid', 'children'),
        Input('global-refresh', 'n_intervals')
    )
    def update_equipment_status_grid(n):
        """Update equipment status grid in overview"""
        try:
            training_status = training_use_case.get_training_status()
            equipment_status = training_status.get('equipment_status', {})

            # Create equipment status cards
            status_cards = []
            for sensor_id, status in equipment_status.items():
                # Get equipment info
                equipment = next((eq for eq in equipment_list if eq.equipment_id == sensor_id), None)
                if not equipment:
                    continue

                anomaly_info = status.get('anomaly_detection', {})
                forecast_info = status.get('forecasting', {})

                # Status indicators
                anomaly_status = "✓" if anomaly_info.get('trained') else "✗"
                forecast_status = "✓" if forecast_info.get('trained') else "✗"

                # Performance scores
                anomaly_score = anomaly_info.get('performance_score', 0)
                forecast_score = forecast_info.get('performance_score', 0)

                # Criticality color
                criticality_colors = {
                    'CRITICAL': 'danger',
                    'HIGH': 'warning',
                    'MEDIUM': 'info',
                    'LOW': 'secondary'
                }
                criticality_color = criticality_colors.get(equipment.criticality.value, 'secondary')

                card = dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.Small(sensor_id, className="text-muted"),
                            dbc.Badge(equipment.criticality.value, color=criticality_color, className="ms-auto")
                        ], className="py-2"),
                        dbc.CardBody([
                            html.H6(equipment.name, className="mb-2"),
                            html.P(equipment.equipment_type.value, className="text-muted small mb-2"),
                            dbc.Row([
                                dbc.Col([
                                    html.Small("Anomaly:", className="fw-bold"),
                                    html.Span(f" {anomaly_status}",
                                             className=f"ms-1 {'text-success' if anomaly_status == '✓' else 'text-danger'}")
                                ], width=6),
                                dbc.Col([
                                    html.Small("Forecast:", className="fw-bold"),
                                    html.Span(f" {forecast_status}",
                                             className=f"ms-1 {'text-success' if forecast_status == '✓' else 'text-danger'}")
                                ], width=6)
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Progress(value=anomaly_score*100, size="sm", color="danger", className="mt-1"),
                                    html.Small(f"{anomaly_score:.2f}", className="text-muted")
                                ], width=6),
                                dbc.Col([
                                    dbc.Progress(value=forecast_score*100, size="sm", color="success", className="mt-1"),
                                    html.Small(f"{forecast_score:.2f}", className="text-muted")
                                ], width=6)
                            ])
                        ], className="py-2")
                    ], className="h-100")
                ], width=6, lg=4, className="mb-3")

                status_cards.append(card)

            if not status_cards:
                return dbc.Alert("No equipment status available", color="info")

            return dbc.Row(status_cards)

        except Exception as e:
            logger.error(f"Error updating equipment status grid: {e}")
            return dbc.Alert("Error loading equipment status", color="danger")

    @app.callback(
        Output('system-health-chart', 'figure'),
        Input('global-refresh', 'n_intervals')
    )
    def update_system_health_chart(n):
        """Update system health pie chart"""
        try:
            # Get system health data
            training_status = training_use_case.get_training_status()
            equipment_status = training_status.get('equipment_status', {})

            # Calculate health categories
            fully_trained = sum(1 for status in equipment_status.values()
                              if status.get('anomaly_detection', {}).get('trained') and
                                 status.get('forecasting', {}).get('trained'))

            partially_trained = sum(1 for status in equipment_status.values()
                                  if (status.get('anomaly_detection', {}).get('trained') or
                                      status.get('forecasting', {}).get('trained')) and
                                     not (status.get('anomaly_detection', {}).get('trained') and
                                          status.get('forecasting', {}).get('trained')))

            not_trained = len(equipment_status) - fully_trained - partially_trained

            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=['Fully Trained', 'Partially Trained', 'Not Trained'],
                values=[fully_trained, partially_trained, not_trained],
                hole=0.4,
                marker_colors=['#28a745', '#ffc107', '#dc3545']
            )])

            fig.update_layout(
                title="System Health Overview",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )

            return fig

        except Exception as e:
            logger.error(f"Error updating system health chart: {e}")
            return go.Figure()

    @app.callback(
        Output('recent-activity-feed', 'children'),
        Input('global-refresh', 'n_intervals')
    )
    def update_recent_activity_feed(n):
        """Update recent activity feed"""
        try:
            # Get recent activities (mock data for now)
            activities = [
                {
                    'time': datetime.now() - timedelta(minutes=5),
                    'type': 'training',
                    'message': 'Model training completed for SMAP-PWR-001',
                    'icon': 'fas fa-graduation-cap',
                    'color': 'success'
                },
                {
                    'time': datetime.now() - timedelta(minutes=12),
                    'type': 'anomaly',
                    'message': 'Anomaly detected in MSL-TMP-002',
                    'icon': 'fas fa-exclamation-triangle',
                    'color': 'warning'
                },
                {
                    'time': datetime.now() - timedelta(minutes=28),
                    'type': 'forecast',
                    'message': 'Forecast generated for SMAP-VIB-001',
                    'icon': 'fas fa-chart-line',
                    'color': 'info'
                },
                {
                    'time': datetime.now() - timedelta(hours=1),
                    'type': 'system',
                    'message': 'System health check completed',
                    'icon': 'fas fa-heartbeat',
                    'color': 'primary'
                }
            ]

            activity_items = []
            for activity in activities:
                time_ago = datetime.now() - activity['time']
                if time_ago.seconds < 3600:
                    time_str = f"{time_ago.seconds // 60} minutes ago"
                else:
                    time_str = f"{time_ago.seconds // 3600} hours ago"

                activity_item = dbc.ListGroupItem([
                    html.Div([
                        html.I(className=f"{activity['icon']} text-{activity['color']} me-3"),
                        html.Div([
                            html.P(activity['message'], className="mb-1"),
                            html.Small(time_str, className="text-muted")
                        ])
                    ], className="d-flex align-items-start")
                ])

                activity_items.append(activity_item)

            return dbc.ListGroup(activity_items, flush=True)

        except Exception as e:
            logger.error(f"Error updating recent activity feed: {e}")
            return dbc.Alert("Error loading recent activities", color="warning")

    # Enhanced monitoring tab callbacks
    @app.callback(
        [Output('enhanced-realtime-chart', 'figure'),
         Output('enhanced-sensor-info', 'children'),
         Output('live-sensor-stats', 'children')],
        [Input('monitoring-sensor-selector', 'value'),
         Input('monitoring-time-range', 'value'),
         Input('global-refresh', 'n_intervals')]
    )
    def update_enhanced_monitoring(sensor_id, time_range, n):
        """Update enhanced real-time monitoring"""
        try:
            if not sensor_id:
                return {}, "Select a sensor to view data", "No sensor selected"

            # Get sensor data
            data_response = data_loader.get_sensor_data(sensor_id, hours_back=time_range)

            if not data_response['values']:
                return {}, "No data available", "No data"

            values = data_response['values']
            timestamps = [datetime.now() - timedelta(hours=time_range-i) for i in range(len(values))]

            # Create enhanced chart with anomaly detection overlay
            fig = go.Figure()

            # Main data line
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=values,
                mode='lines',
                name='Sensor Data',
                line=dict(color='#2c3e50', width=2)
            ))

            # Add anomaly detection overlay (mock)
            try:
                anomaly_result = anomaly_service.detect_anomalies(sensor_id, np.array(values), timestamps)
                anomalies = anomaly_result.get('anomalies', [])

                if anomalies:
                    anomaly_times = [datetime.fromisoformat(a['timestamp']) for a in anomalies]
                    anomaly_values = [a['value'] for a in anomalies]

                    fig.add_trace(go.Scatter(
                        x=anomaly_times,
                        y=anomaly_values,
                        mode='markers',
                        name='Anomalies',
                        marker=dict(color='red', size=10, symbol='x')
                    ))
            except Exception:
                pass  # Continue without anomaly overlay

            # Add forecast if available
            try:
                forecast_result = forecasting_service.generate_forecast(sensor_id, np.array(values[-24:]), horizon_hours=6)
                if forecast_result.get('forecast_values'):
                    forecast_times = [timestamps[-1] + timedelta(hours=i) for i in range(1, 7)]
                    forecast_values = forecast_result['forecast_values']

                    fig.add_trace(go.Scatter(
                        x=forecast_times,
                        y=forecast_values,
                        mode='lines',
                        name='Forecast',
                        line=dict(color='red', dash='dash')
                    ))
            except Exception:
                pass  # Continue without forecast

            fig.update_layout(
                title=f"Enhanced Monitoring: {sensor_id}",
                xaxis_title="Time",
                yaxis_title="Value",
                height=500,
                showlegend=True,
                hovermode='x unified'
            )

            # Enhanced sensor info
            equipment = next((eq for eq in equipment_list if eq.equipment_id == sensor_id), None)
            if equipment:
                sensor_info = dbc.Card([
                    dbc.CardBody([
                        html.H6(equipment.name, className="mb-3"),
                        dbc.ListGroup([
                            dbc.ListGroupItem([html.Strong("Type: "), equipment.equipment_type.value]),
                            dbc.ListGroupItem([html.Strong("Location: "), equipment.location]),
                            dbc.ListGroupItem([html.Strong("Criticality: "), equipment.criticality.value]),
                            dbc.ListGroupItem([html.Strong("Unit: "), equipment.unit]),
                            dbc.ListGroupItem([html.Strong("Range: "), f"{equipment.normal_range[0]:.1f} - {equipment.normal_range[1]:.1f}"]),
                            dbc.ListGroupItem([html.Strong("Status: "), dbc.Badge("Online", color="success")])
                        ], flush=True)
                    ])
                ])
            else:
                sensor_info = dbc.Alert("Sensor information not found", color="warning")

            # Live statistics
            current_value = values[-1] if values else 0
            avg_value = np.mean(values) if values else 0
            min_value = np.min(values) if values else 0
            max_value = np.max(values) if values else 0
            std_value = np.std(values) if values else 0

            live_stats = dbc.Card([
                dbc.CardBody([
                    html.H6("Live Statistics", className="mb-3"),
                    dbc.ListGroup([
                        dbc.ListGroupItem([html.Strong("Current: "), f"{current_value:.2f}"]),
                        dbc.ListGroupItem([html.Strong("Average: "), f"{avg_value:.2f}"]),
                        dbc.ListGroupItem([html.Strong("Minimum: "), f"{min_value:.2f}"]),
                        dbc.ListGroupItem([html.Strong("Maximum: "), f"{max_value:.2f}"]),
                        dbc.ListGroupItem([html.Strong("Std Dev: "), f"{std_value:.2f}"]),
                        dbc.ListGroupItem([html.Strong("Data Points: "), str(len(values))])
                    ], flush=True)
                ])
            ])

            return fig, sensor_info, live_stats

        except Exception as e:
            logger.error(f"Error updating enhanced monitoring: {e}")
            return {}, dbc.Alert(f"Error: {str(e)}", color="danger"), "Error"

    # Enhanced anomalies tab callbacks
    @app.callback(
        [Output('anomaly-summary-cards', 'children'),
         Output('anomaly-timeline-chart', 'figure'),
         Output('enhanced-anomaly-list', 'children'),
         Output('anomaly-analysis', 'children')],
        Input('global-refresh', 'n_intervals')
    )
    def update_enhanced_anomalies(n):
        """Update enhanced anomaly detection interface"""
        try:
            # Get anomaly summary
            summary = anomaly_service.get_detection_summary()

            # Summary cards
            total_anomalies = summary.get('total_anomalies', 0)
            severity_breakdown = summary.get('severity_breakdown', {})

            summary_cards = dbc.Stack([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(str(total_anomalies), className="text-danger mb-1"),
                        html.P("Total Anomalies", className="text-muted mb-0"),
                        html.Small("Last 24 hours", className="text-muted")
                    ])
                ], className="text-center"),
                dbc.Card([
                    dbc.CardBody([
                        html.H3(str(severity_breakdown.get('CRITICAL', 0)), className="text-danger mb-1"),
                        html.P("Critical", className="text-muted mb-0"),
                        html.Small("Immediate action required", className="text-muted")
                    ])
                ], className="text-center"),
                dbc.Card([
                    dbc.CardBody([
                        html.H3(str(severity_breakdown.get('HIGH', 0)), className="text-warning mb-1"),
                        html.P("High Priority", className="text-muted mb-0"),
                        html.Small("Attention needed", className="text-muted")
                    ])
                ], className="text-center")
            ])

            # Timeline chart (mock data)
            hours = list(range(24))
            anomaly_counts = [np.random.poisson(2) for _ in hours]

            timeline_fig = go.Figure()
            timeline_fig.add_trace(go.Bar(
                x=hours,
                y=anomaly_counts,
                name='Anomalies per Hour',
                marker_color='#e74c3c'
            ))

            timeline_fig.update_layout(
                title="Anomaly Timeline - Last 24 Hours",
                xaxis_title="Hour",
                yaxis_title="Anomaly Count",
                height=300
            )

            # Enhanced anomaly list
            recent_anomalies = summary.get('recent_anomalies', [])[:10]
            anomaly_list = []

            for anomaly in recent_anomalies:
                severity = anomaly.get('severity', 'MEDIUM')
                color_map = {'CRITICAL': 'danger', 'HIGH': 'warning', 'MEDIUM': 'info', 'LOW': 'light'}

                anomaly_item = dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6(f"🚨 {anomaly.get('sensor_id', 'Unknown')}", className="mb-1"),
                                html.P(anomaly.get('description', 'No description'), className="mb-2"),
                                html.Small(f"Score: {anomaly.get('score', 0):.3f} | Value: {anomaly.get('value', 0):.2f}")
                            ], width=8),
                            dbc.Col([
                                dbc.Badge(severity, color=color_map.get(severity, 'secondary'), className="mb-2"),
                                html.Br(),
                                html.Small(anomaly.get('timestamp', 'Unknown time'), className="text-muted")
                            ], width=4)
                        ])
                    ])
                ], outline=True, color=color_map.get(severity, 'secondary'), className="mb-2")

                anomaly_list.append(anomaly_item)

            if not anomaly_list:
                anomaly_list = [dbc.Alert("✅ No recent anomalies detected", color="success")]

            # Anomaly analysis
            analysis = dbc.Card([
                dbc.CardBody([
                    html.H6("Analysis Summary", className="mb-3"),
                    dbc.ListGroup([
                        dbc.ListGroupItem([
                            html.Strong("Detection Rate: "),
                            f"{(total_anomalies / max(len(equipment_list), 1)):.1f} anomalies per sensor"
                        ]),
                        dbc.ListGroupItem([
                            html.Strong("Most Affected: "),
                            "MSL-TMP-002" if total_anomalies > 0 else "None"
                        ]),
                        dbc.ListGroupItem([
                            html.Strong("Peak Time: "),
                            "14:00-16:00" if total_anomalies > 0 else "N/A"
                        ]),
                        dbc.ListGroupItem([
                            html.Strong("Trend: "),
                            "Stable" if total_anomalies < 10 else "Increasing"
                        ])
                    ], flush=True)
                ])
            ])

            return summary_cards, timeline_fig, anomaly_list, analysis

        except Exception as e:
            logger.error(f"Error updating enhanced anomalies: {e}")
            return "Error", {}, [dbc.Alert("Error loading anomalies", color="danger")], "Error"

    # Enhanced forecasting tab callback
    @app.callback(
        [Output('enhanced-forecast-chart', 'figure'),
         Output('forecast-metrics', 'children')],
        [Input('forecast-sensor-selector', 'value'),
         Input('forecast-horizon', 'value'),
         Input('forecast-mode', 'value')]
    )
    def update_enhanced_forecast(sensor_id, horizon, mode):
        """Update enhanced forecasting interface"""
        try:
            if not sensor_id:
                return {}, "Select a sensor for forecasting"

            # Get historical data
            data_response = data_loader.get_sensor_data(sensor_id, hours_back=48)
            if not data_response['values']:
                return {}, "No data available for forecasting"

            historical_data = np.array(data_response['values'])
            historical_times = [datetime.now() - timedelta(hours=48-i) for i in range(len(historical_data))]

            # Generate forecast
            forecast_result = forecasting_service.generate_forecast(
                sensor_id, historical_data, horizon_hours=horizon
            )

            fig = go.Figure()

            # Historical data
            fig.add_trace(go.Scatter(
                x=historical_times,
                y=historical_data,
                mode='lines',
                name='Historical Data',
                line=dict(color='#2c3e50', width=2)
            ))

            if forecast_result.get('forecast_values'):
                forecast_times = [historical_times[-1] + timedelta(hours=i) for i in range(1, horizon + 1)]
                forecast_values = forecast_result['forecast_values']

                # Forecast line
                fig.add_trace(go.Scatter(
                    x=forecast_times,
                    y=forecast_values,
                    mode='lines',
                    name='Forecast',
                    line=dict(color='#e74c3c', dash='dash', width=2)
                ))

                # Confidence intervals
                if forecast_result.get('confidence_upper') and forecast_result.get('confidence_lower'):
                    fig.add_trace(go.Scatter(
                        x=forecast_times,
                        y=forecast_result['confidence_upper'],
                        fill=None,
                        mode='lines',
                        line_color='rgba(0,0,0,0)',
                        showlegend=False
                    ))

                    fig.add_trace(go.Scatter(
                        x=forecast_times,
                        y=forecast_result['confidence_lower'],
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(0,0,0,0)',
                        name='Confidence Interval',
                        fillcolor='rgba(231, 76, 60, 0.2)'
                    ))

            fig.update_layout(
                title=f"Enhanced Forecast: {sensor_id} ({horizon}h ahead)",
                xaxis_title="Time",
                yaxis_title="Value",
                height=500,
                showlegend=True,
                hovermode='x unified'
            )

            # Forecast metrics
            if forecast_result.get('forecast_values'):
                forecast_quality = forecast_result.get('forecast_quality', 'unknown')
                confidence = forecast_result.get('confidence', 0)

                metrics = dbc.Card([
                    dbc.CardBody([
                        html.H6("Forecast Metrics", className="mb-3"),
                        dbc.ListGroup([
                            dbc.ListGroupItem([
                                html.Strong("Quality: "),
                                dbc.Badge(forecast_quality.title(), color="success" if forecast_quality == "good" else "warning")
                            ]),
                            dbc.ListGroupItem([html.Strong("Confidence: "), f"{confidence:.1%}"]),
                            dbc.ListGroupItem([html.Strong("Horizon: "), f"{horizon} hours"]),
                            dbc.ListGroupItem([html.Strong("Mode: "), mode.title()]),
                            dbc.ListGroupItem([html.Strong("Model: "), "Transformer"]),
                            dbc.ListGroupItem([html.Strong("Generated: "), datetime.now().strftime("%H:%M:%S")])
                        ], flush=True)
                    ])
                ])
            else:
                metrics = dbc.Alert("Forecast generation failed", color="warning")

            return fig, metrics

        except Exception as e:
            logger.error(f"Error updating enhanced forecast: {e}")
            return {}, dbc.Alert(f"Error: {str(e)}", color="danger")

    return app
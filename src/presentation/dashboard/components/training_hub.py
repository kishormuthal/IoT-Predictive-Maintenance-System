"""
Training Hub Dashboard Component
Training management interface for the enhanced dashboard
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


def create_training_hub_layout():
    """Create the training hub layout"""
    return dbc.Container(
        [
            # Training Hub Header
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
                                                        className="fas fa-graduation-cap me-3 text-primary"
                                                    ),
                                                    "Training Hub",
                                                ],
                                                className="mb-3",
                                            ),
                                            html.P(
                                                "Manage model training, monitor progress, and analyze training performance.",
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
            # Quick Actions Row
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.I(className="fas fa-bolt me-2"),
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
                                                                        className="fas fa-play me-2"
                                                                    ),
                                                                    "Train All Models",
                                                                ],
                                                                id="train-all-btn",
                                                                color="primary",
                                                                size="lg",
                                                                className="w-100 mb-2",
                                                            ),
                                                            html.Small(
                                                                "Train both anomaly detection and forecasting models for all sensors",
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
                                                                        className="fas fa-search me-2"
                                                                    ),
                                                                    "Validate Models",
                                                                ],
                                                                id="validate-all-btn",
                                                                color="success",
                                                                size="lg",
                                                                className="w-100 mb-2",
                                                            ),
                                                            html.Small(
                                                                "Run validation tests on all trained models",
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
                                                                        className="fas fa-chart-line me-2"
                                                                    ),
                                                                    "Training Report",
                                                                ],
                                                                id="training-report-btn",
                                                                color="info",
                                                                size="lg",
                                                                className="w-100 mb-2",
                                                            ),
                                                            html.Small(
                                                                "Generate comprehensive training performance report",
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
                                                                        className="fas fa-cog me-2"
                                                                    ),
                                                                    "Configuration",
                                                                ],
                                                                id="config-btn",
                                                                color="secondary",
                                                                size="lg",
                                                                className="w-100 mb-2",
                                                            ),
                                                            html.Small(
                                                                "Manage training configurations and parameters",
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
            # Training Status Overview
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.I(
                                                className="fas fa-tachometer-alt me-2"
                                            ),
                                            "Training Status Overview",
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="training-status-overview")]
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
                                            html.I(
                                                className="fas fa-exclamation-circle me-2"
                                            ),
                                            "Training Recommendations",
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="training-recommendations")]
                                    ),
                                ]
                            )
                        ],
                        width=4,
                    ),
                ],
                className="mb-4",
            ),
            # Individual Sensor Training
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.I(className="fas fa-microchip me-2"),
                                            "Individual Sensor Training",
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Select Sensor:",
                                                                className="fw-bold",
                                                            ),
                                                            dcc.Dropdown(
                                                                id="training-sensor-selector",
                                                                placeholder="Choose a sensor for individual training",
                                                            ),
                                                        ],
                                                        width=4,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Model Type:",
                                                                className="fw-bold",
                                                            ),
                                                            dcc.Dropdown(
                                                                id="training-model-type",
                                                                options=[
                                                                    {
                                                                        "label": "Anomaly Detection (Telemanom)",
                                                                        "value": "telemanom",
                                                                    },
                                                                    {
                                                                        "label": "Forecasting (Transformer)",
                                                                        "value": "transformer",
                                                                    },
                                                                    {
                                                                        "label": "Both Models",
                                                                        "value": "both",
                                                                    },
                                                                ],
                                                                value="both",
                                                            ),
                                                        ],
                                                        width=4,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Training Mode:",
                                                                className="fw-bold",
                                                            ),
                                                            dcc.Dropdown(
                                                                id="training-mode",
                                                                options=[
                                                                    {
                                                                        "label": "Standard Training",
                                                                        "value": "standard",
                                                                    },
                                                                    {
                                                                        "label": "Quick Training",
                                                                        "value": "quick",
                                                                    },
                                                                    {
                                                                        "label": "Extended Training",
                                                                        "value": "extended",
                                                                    },
                                                                ],
                                                                value="standard",
                                                            ),
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
                                                            dbc.Button(
                                                                [
                                                                    html.I(
                                                                        className="fas fa-play me-2"
                                                                    ),
                                                                    "Start Training",
                                                                ],
                                                                id="start-individual-training",
                                                                color="primary",
                                                                className="me-2",
                                                            ),
                                                            dbc.Button(
                                                                [
                                                                    html.I(
                                                                        className="fas fa-stop me-2"
                                                                    ),
                                                                    "Stop Training",
                                                                ],
                                                                id="stop-training",
                                                                color="danger",
                                                                disabled=True,
                                                            ),
                                                        ]
                                                    )
                                                ]
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
            # Training Progress and Logs
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.I(className="fas fa-tasks me-2"),
                                            "Training Progress",
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="training-progress-display")]
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
                                    dbc.CardHeader(
                                        [
                                            html.I(className="fas fa-file-alt me-2"),
                                            "Training Logs",
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                id="training-logs-display",
                                                style={
                                                    "height": "300px",
                                                    "overflow-y": "auto",
                                                },
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
            # Training History and Performance
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.I(className="fas fa-history me-2"),
                                            "Training History",
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [dcc.Graph(id="training-history-chart")]
                                    ),
                                ]
                            )
                        ]
                    )
                ]
            ),
            # Hidden components for state management
            dcc.Store(id="training-progress-store", data={}),
            dcc.Store(id="training-logs-store", data=[]),
            dcc.Interval(
                id="training-progress-interval",
                interval=2000,
                n_intervals=0,
                disabled=True,
            ),
        ]
    )


def register_training_hub_callbacks(
    app, training_use_case, config_manager, performance_monitor
):
    """Register callbacks for training hub functionality"""

    @app.callback(
        Output("training-sensor-selector", "options"),
        Input("training-hub-content", "children"),
    )
    def populate_sensor_options(children):
        """Populate sensor options for training"""
        try:
            from config.equipment_config import get_equipment_list

            equipment_list = get_equipment_list()

            return [
                {
                    "label": f"{eq.equipment_id} - {eq.name} ({eq.equipment_type.value})",
                    "value": eq.equipment_id,
                }
                for eq in equipment_list
            ]
        except Exception as e:
            logger.error(f"Error populating sensor options: {e}")
            return []

    @app.callback(
        Output("training-status-overview", "children"),
        Input("global-refresh", "n_intervals"),
    )
    def update_training_status_overview(n):
        """Update training status overview"""
        try:
            # Get training status
            training_status = training_use_case.get_training_status()

            # Extract metrics
            total_equipment = training_status.get("total_equipment", 0)
            equipment_status = training_status.get("equipment_status", {})

            # Calculate training coverage
            anomaly_trained = sum(
                1
                for status in equipment_status.values()
                if status.get("anomaly_detection", {}).get("trained", False)
            )
            forecast_trained = sum(
                1
                for status in equipment_status.values()
                if status.get("forecasting", {}).get("trained", False)
            )

            # Calculate average performance scores
            anomaly_scores = [
                status.get("anomaly_detection", {}).get("performance_score", 0)
                for status in equipment_status.values()
                if status.get("anomaly_detection", {}).get("trained", False)
            ]
            forecast_scores = [
                status.get("forecasting", {}).get("performance_score", 0)
                for status in equipment_status.values()
                if status.get("forecasting", {}).get("trained", False)
            ]

            avg_anomaly_score = (
                sum(anomaly_scores) / len(anomaly_scores) if anomaly_scores else 0
            )
            avg_forecast_score = (
                sum(forecast_scores) / len(forecast_scores) if forecast_scores else 0
            )

            # Create overview cards
            overview_cards = dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H4(
                                                f"{anomaly_trained}/{total_equipment}",
                                                className="text-success mb-1",
                                            ),
                                            html.P(
                                                "Anomaly Models",
                                                className="text-muted mb-0",
                                            ),
                                            dbc.Progress(
                                                value=(
                                                    (anomaly_trained / total_equipment)
                                                    * 100
                                                    if total_equipment > 0
                                                    else 0
                                                ),
                                                color="success",
                                                className="mt-2",
                                            ),
                                        ]
                                    )
                                ],
                                className="text-center",
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H4(
                                                f"{forecast_trained}/{total_equipment}",
                                                className="text-info mb-1",
                                            ),
                                            html.P(
                                                "Forecast Models",
                                                className="text-muted mb-0",
                                            ),
                                            dbc.Progress(
                                                value=(
                                                    (forecast_trained / total_equipment)
                                                    * 100
                                                    if total_equipment > 0
                                                    else 0
                                                ),
                                                color="info",
                                                className="mt-2",
                                            ),
                                        ]
                                    )
                                ],
                                className="text-center",
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H4(
                                                f"{avg_anomaly_score:.2f}",
                                                className="text-primary mb-1",
                                            ),
                                            html.P(
                                                "Avg Anomaly Score",
                                                className="text-muted mb-0",
                                            ),
                                            dbc.Progress(
                                                value=avg_anomaly_score * 100,
                                                color="primary",
                                                className="mt-2",
                                            ),
                                        ]
                                    )
                                ],
                                className="text-center",
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H4(
                                                f"{avg_forecast_score:.2f}",
                                                className="text-warning mb-1",
                                            ),
                                            html.P(
                                                "Avg Forecast Score",
                                                className="text-muted mb-0",
                                            ),
                                            dbc.Progress(
                                                value=avg_forecast_score * 100,
                                                color="warning",
                                                className="mt-2",
                                            ),
                                        ]
                                    )
                                ],
                                className="text-center",
                            )
                        ],
                        width=3,
                    ),
                ]
            )

            return overview_cards

        except Exception as e:
            logger.error(f"Error updating training status overview: {e}")
            return dbc.Alert(f"Error loading training status: {str(e)}", color="danger")

    @app.callback(
        Output("training-recommendations", "children"),
        Input("global-refresh", "n_intervals"),
    )
    def update_training_recommendations(n):
        """Update training recommendations"""
        try:
            # Get recommendations from anomaly service
            from src.core.services.anomaly_service import AnomalyDetectionService

            anomaly_service = AnomalyDetectionService()
            recommendations = anomaly_service.get_training_recommendations()

            # Create recommendation items
            recommendation_items = []

            # Sensors needing training
            sensors_needing_training = recommendations.get(
                "sensors_needing_training", []
            )
            if sensors_needing_training:
                recommendation_items.append(
                    dbc.ListGroupItem(
                        [
                            html.H6(
                                [
                                    html.I(
                                        className="fas fa-exclamation-triangle text-danger me-2"
                                    ),
                                    f"{len(sensors_needing_training)} sensors need initial training",
                                ]
                            ),
                            html.P(
                                [
                                    eq["sensor_id"]
                                    for eq in sensors_needing_training[:3]
                                ],
                                className="text-muted mb-0",
                            ),
                        ],
                        color="danger",
                        className="mb-2",
                    )
                )

            # Sensors needing retraining
            sensors_needing_retraining = recommendations.get(
                "sensors_needing_retraining", []
            )
            if sensors_needing_retraining:
                recommendation_items.append(
                    dbc.ListGroupItem(
                        [
                            html.H6(
                                [
                                    html.I(
                                        className="fas fa-sync-alt text-warning me-2"
                                    ),
                                    f"{len(sensors_needing_retraining)} sensors need retraining",
                                ]
                            ),
                            html.P(
                                [
                                    eq["sensor_id"]
                                    for eq in sensors_needing_retraining[:3]
                                ],
                                className="text-muted mb-0",
                            ),
                        ],
                        color="warning",
                        className="mb-2",
                    )
                )

            # Well performing sensors
            well_performing = recommendations.get("well_performing_sensors", [])
            if well_performing:
                recommendation_items.append(
                    dbc.ListGroupItem(
                        [
                            html.H6(
                                [
                                    html.I(
                                        className="fas fa-check-circle text-success me-2"
                                    ),
                                    f"{len(well_performing)} sensors performing well",
                                ]
                            )
                        ],
                        color="success",
                    )
                )

            if not recommendation_items:
                return dbc.Alert("All models are up to date!", color="success")

            return dbc.ListGroup(recommendation_items, flush=True)

        except Exception as e:
            logger.error(f"Error updating training recommendations: {e}")
            return dbc.Alert("Unable to load recommendations", color="warning")

    @app.callback(
        [
            Output("training-progress-display", "children"),
            Output("training-logs-display", "children"),
            Output("training-progress-interval", "disabled"),
        ],
        [
            Input("start-individual-training", "n_clicks"),
            Input("train-all-btn", "n_clicks"),
            Input("training-progress-interval", "n_intervals"),
        ],
        [
            State("training-sensor-selector", "value"),
            State("training-model-type", "value"),
            State("training-mode", "value"),
            State("training-progress-store", "data"),
            State("training-logs-store", "data"),
        ],
    )
    def handle_training_operations(
        individual_clicks,
        all_clicks,
        interval_n,
        sensor_id,
        model_type,
        mode,
        progress_data,
        logs_data,
    ):
        """Handle training operations and progress updates"""
        try:
            ctx = callback_context
            if not ctx.triggered:
                return "No training in progress", [], True

            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

            # Start individual training
            if trigger_id == "start-individual-training" and individual_clicks:
                if not sensor_id or not model_type:
                    return (
                        dbc.Alert(
                            "Please select sensor and model type", color="warning"
                        ),
                        [],
                        True,
                    )

                # Start training (mock implementation)
                progress = dbc.Progress(
                    [dbc.Progress(value=0, id="training-progress-bar", label="0%")],
                    className="mb-3",
                )

                logs = [
                    html.P(
                        f"[{datetime.now().strftime('%H:%M:%S')}] Starting training for {sensor_id}"
                    ),
                    html.P(
                        f"[{datetime.now().strftime('%H:%M:%S')}] Model type: {model_type}"
                    ),
                    html.P(
                        f"[{datetime.now().strftime('%H:%M:%S')}] Training mode: {mode}"
                    ),
                ]

                return progress, logs, False  # Enable interval

            # Start batch training
            elif trigger_id == "train-all-btn" and all_clicks:
                progress = dbc.Progress(
                    [
                        dbc.Progress(
                            value=0,
                            id="batch-training-progress-bar",
                            label="Starting batch training...",
                        )
                    ],
                    className="mb-3",
                )

                logs = [
                    html.P(
                        f"[{datetime.now().strftime('%H:%M:%S')}] Starting batch training for all sensors"
                    ),
                    html.P(
                        f"[{datetime.now().strftime('%H:%M:%S')}] Training both anomaly detection and forecasting models"
                    ),
                ]

                return progress, logs, False  # Enable interval

            # Update progress
            elif trigger_id == "training-progress-interval":
                # Mock progress update
                current_progress = progress_data.get("progress", 0)
                new_progress = min(current_progress + 10, 100)

                progress = dbc.Progress(
                    [dbc.Progress(value=new_progress, label=f"{new_progress}%")],
                    className="mb-3",
                )

                new_logs = logs_data + [
                    html.P(
                        f"[{datetime.now().strftime('%H:%M:%S')}] Training progress: {new_progress}%"
                    )
                ]

                # Stop when complete
                interval_disabled = new_progress >= 100
                if interval_disabled:
                    new_logs.append(
                        html.P(
                            f"[{datetime.now().strftime('%H:%M:%S')}] Training completed successfully!",
                            className="text-success",
                        )
                    )

                return progress, new_logs[-10:], interval_disabled  # Keep last 10 logs

            return "No training in progress", [], True

        except Exception as e:
            logger.error(f"Error handling training operations: {e}")
            return dbc.Alert(f"Training error: {str(e)}", color="danger"), [], True

    @app.callback(
        Output("training-history-chart", "figure"),
        Input("global-refresh", "n_intervals"),
    )
    def update_training_history_chart(n):
        """Update training history chart"""
        try:
            # Get training metrics from performance monitor
            training_summary = performance_monitor.get_training_metrics_summary(
                hours_back=168
            )  # 1 week

            # Create mock data for demonstration
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=7), end=datetime.now(), freq="D"
            )

            # Mock training sessions per day
            anomaly_sessions = [2, 1, 3, 0, 2, 1, 4]
            forecast_sessions = [1, 2, 2, 1, 1, 3, 2]

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=dates,
                    y=anomaly_sessions,
                    name="Anomaly Detection",
                    marker_color="#e74c3c",
                )
            )

            fig.add_trace(
                go.Bar(
                    x=dates,
                    y=forecast_sessions,
                    name="Forecasting",
                    marker_color="#3498db",
                )
            )

            fig.update_layout(
                title="Training Sessions - Last 7 Days",
                xaxis_title="Date",
                yaxis_title="Training Sessions",
                barmode="group",
                height=350,
                showlegend=True,
            )

            return fig

        except Exception as e:
            logger.error(f"Error updating training history chart: {e}")
            return go.Figure().add_annotation(text="Error loading training history")

    return app

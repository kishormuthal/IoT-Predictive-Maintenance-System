"""
MLflow Integration Dashboard
Embedded MLflow UI and model management interface
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

logger = logging.getLogger(__name__)


def create_mlflow_layout() -> html.Div:
    """
    Create MLflow integration dashboard layout

    Features:
    - Embedded MLflow UI iframe
    - Experiment tracking
    - Model comparison
    - Deployment status
    - Performance metrics over time
    """

    return html.Div(
        [
            # Header
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H2(
                                [
                                    html.I(className="fas fa-project-diagram me-2"),
                                    "MLflow Model Tracking & Management",
                                ],
                                className="mb-0",
                            ),
                            html.P(
                                "Track experiments, compare models, and manage deployments",
                                className="text-muted mb-0",
                            ),
                        ],
                        width=8,
                    ),
                    dbc.Col(
                        [
                            dbc.ButtonGroup(
                                [
                                    dbc.Button(
                                        [
                                            html.I(className="fas fa-sync me-2"),
                                            "Refresh",
                                        ],
                                        id="mlflow-refresh-btn",
                                        color="primary",
                                        size="sm",
                                    ),
                                    dbc.Button(
                                        [
                                            html.I(className="fas fa-external-link-alt me-2"),
                                            "Open MLflow UI",
                                        ],
                                        id="mlflow-open-ui-btn",
                                        color="info",
                                        size="sm",
                                        href="http://localhost:5000",
                                        target="_blank",
                                    ),
                                ],
                                className="float-end",
                            )
                        ],
                        width=4,
                    ),
                ],
                className="mb-4",
            ),
            # Status Cards
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                [
                                                    html.I(className="fas fa-flask fa-2x text-primary mb-2"),
                                                    html.H3(
                                                        id="mlflow-total-experiments",
                                                        children="0",
                                                    ),
                                                    html.P(
                                                        "Total Experiments",
                                                        className="text-muted mb-0",
                                                    ),
                                                ],
                                                className="text-center",
                                            )
                                        ]
                                    )
                                ]
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
                                            html.Div(
                                                [
                                                    html.I(className="fas fa-play fa-2x text-success mb-2"),
                                                    html.H3(
                                                        id="mlflow-total-runs",
                                                        children="0",
                                                    ),
                                                    html.P(
                                                        "Total Runs",
                                                        className="text-muted mb-0",
                                                    ),
                                                ],
                                                className="text-center",
                                            )
                                        ]
                                    )
                                ]
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
                                            html.Div(
                                                [
                                                    html.I(className="fas fa-cube fa-2x text-info mb-2"),
                                                    html.H3(
                                                        id="mlflow-registered-models",
                                                        children="0",
                                                    ),
                                                    html.P(
                                                        "Registered Models",
                                                        className="text-muted mb-0",
                                                    ),
                                                ],
                                                className="text-center",
                                            )
                                        ]
                                    )
                                ]
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
                                            html.Div(
                                                [
                                                    html.I(className="fas fa-rocket fa-2x text-warning mb-2"),
                                                    html.H3(
                                                        id="mlflow-deployed-models",
                                                        children="0",
                                                    ),
                                                    html.P(
                                                        "Deployed Models",
                                                        className="text-muted mb-0",
                                                    ),
                                                ],
                                                className="text-center",
                                            )
                                        ]
                                    )
                                ]
                            )
                        ],
                        width=3,
                    ),
                ],
                className="mb-4",
            ),
            # Main Content Tabs
            dbc.Tabs(
                [
                    # Tab 1: MLflow UI Embed
                    dbc.Tab(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                [
                                                    html.P(
                                                        [
                                                            html.I(className="fas fa-info-circle me-2"),
                                                            "MLflow UI is embedded below. You can also ",
                                                            html.A(
                                                                "open it in a new tab",
                                                                href="http://localhost:5000",
                                                                target="_blank",
                                                            ),
                                                            " for full functionality.",
                                                        ],
                                                        className="mb-3",
                                                    ),
                                                    # Connection status
                                                    html.Div(
                                                        id="mlflow-connection-status",
                                                        children=[
                                                            dbc.Alert(
                                                                [
                                                                    html.I(className="fas fa-spinner fa-spin me-2"),
                                                                    "Checking MLflow connection...",
                                                                ],
                                                                color="info",
                                                            )
                                                        ],
                                                        className="mb-3",
                                                    ),
                                                    # Embedded iframe
                                                    html.Iframe(
                                                        id="mlflow-iframe",
                                                        src="http://localhost:5000",
                                                        style={
                                                            "width": "100%",
                                                            "height": "800px",
                                                            "border": "1px solid #dee2e6",
                                                            "borderRadius": "4px",
                                                        },
                                                    ),
                                                ]
                                            )
                                        ]
                                    )
                                ]
                            )
                        ],
                        label="MLflow UI",
                        tab_id="mlflow-ui",
                    ),
                    # Tab 2: Experiment Comparison
                    dbc.Tab(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label("Select Experiments to Compare:"),
                                                            dcc.Dropdown(
                                                                id="mlflow-experiment-selector",
                                                                multi=True,
                                                                placeholder="Select experiments...",
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label("Metric to Compare:"),
                                                            dcc.Dropdown(
                                                                id="mlflow-metric-selector",
                                                                options=[
                                                                    {
                                                                        "label": "Accuracy",
                                                                        "value": "accuracy",
                                                                    },
                                                                    {
                                                                        "label": "Precision",
                                                                        "value": "precision",
                                                                    },
                                                                    {
                                                                        "label": "Recall",
                                                                        "value": "recall",
                                                                    },
                                                                    {
                                                                        "label": "F1 Score",
                                                                        "value": "f1_score",
                                                                    },
                                                                    {
                                                                        "label": "MAE",
                                                                        "value": "mae",
                                                                    },
                                                                    {
                                                                        "label": "RMSE",
                                                                        "value": "rmse",
                                                                    },
                                                                    {
                                                                        "label": "R²",
                                                                        "value": "r2",
                                                                    },
                                                                ],
                                                                value="accuracy",
                                                                clearable=False,
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                ],
                                                className="mb-4",
                                            ),
                                            # Comparison Chart
                                            dcc.Graph(id="mlflow-experiment-comparison-chart"),
                                            # Detailed Comparison Table
                                            html.Div(id="mlflow-experiment-comparison-table"),
                                        ]
                                    )
                                ]
                            )
                        ],
                        label="Experiment Comparison",
                        tab_id="experiment-comparison",
                    ),
                    # Tab 3: Model Registry
                    dbc.Tab(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label("Filter by Model Type:"),
                                                            dcc.Dropdown(
                                                                id="mlflow-model-type-filter",
                                                                options=[
                                                                    {
                                                                        "label": "All Models",
                                                                        "value": "all",
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
                                                                        "label": "Classification",
                                                                        "value": "classification",
                                                                    },
                                                                ],
                                                                value="all",
                                                                clearable=False,
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label("Filter by Stage:"),
                                                            dcc.Dropdown(
                                                                id="mlflow-model-stage-filter",
                                                                options=[
                                                                    {
                                                                        "label": "All Stages",
                                                                        "value": "all",
                                                                    },
                                                                    {
                                                                        "label": "None",
                                                                        "value": "None",
                                                                    },
                                                                    {
                                                                        "label": "Staging",
                                                                        "value": "Staging",
                                                                    },
                                                                    {
                                                                        "label": "Production",
                                                                        "value": "Production",
                                                                    },
                                                                    {
                                                                        "label": "Archived",
                                                                        "value": "Archived",
                                                                    },
                                                                ],
                                                                value="all",
                                                                clearable=False,
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                ],
                                                className="mb-4",
                                            ),
                                            # Model Registry Table
                                            html.Div(id="mlflow-model-registry-table"),
                                        ]
                                    )
                                ]
                            )
                        ],
                        label="Model Registry",
                        tab_id="model-registry",
                    ),
                    # Tab 4: Deployment Status
                    dbc.Tab(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5(
                                                "Production Model Deployment Status",
                                                className="mb-3",
                                            ),
                                            # Deployment status cards
                                            html.Div(id="mlflow-deployment-status-cards"),
                                            html.Hr(),
                                            # Deployment timeline
                                            html.H5("Deployment Timeline", className="mb-3"),
                                            dcc.Graph(id="mlflow-deployment-timeline"),
                                        ]
                                    )
                                ]
                            )
                        ],
                        label="Deployment Status",
                        tab_id="deployment-status",
                    ),
                    # Tab 5: Model Performance Trends
                    dbc.Tab(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label("Select Model:"),
                                                            dcc.Dropdown(
                                                                id="mlflow-model-selector",
                                                                placeholder="Select a model...",
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label("Time Range:"),
                                                            dcc.Dropdown(
                                                                id="mlflow-time-range",
                                                                options=[
                                                                    {
                                                                        "label": "Last 7 Days",
                                                                        "value": "7d",
                                                                    },
                                                                    {
                                                                        "label": "Last 30 Days",
                                                                        "value": "30d",
                                                                    },
                                                                    {
                                                                        "label": "Last 90 Days",
                                                                        "value": "90d",
                                                                    },
                                                                    {
                                                                        "label": "All Time",
                                                                        "value": "all",
                                                                    },
                                                                ],
                                                                value="30d",
                                                                clearable=False,
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                ],
                                                className="mb-4",
                                            ),
                                            # Performance trends
                                            dcc.Graph(id="mlflow-performance-trends"),
                                            # Parameter importance
                                            html.H5(
                                                "Hyperparameter Importance",
                                                className="mt-4 mb-3",
                                            ),
                                            dcc.Graph(id="mlflow-parameter-importance"),
                                        ]
                                    )
                                ]
                            )
                        ],
                        label="Performance Trends",
                        tab_id="performance-trends",
                    ),
                ],
                id="mlflow-tabs",
                active_tab="mlflow-ui",
            ),
            # Auto-refresh interval
            dcc.Interval(
                id="mlflow-refresh-interval",
                interval=30000,  # 30 seconds
                n_intervals=0,
            ),
            # Store for MLflow data
            dcc.Store(id="mlflow-data-store"),
        ],
        className="p-4",
    )


def register_mlflow_callbacks(app: dash.Dash):
    """Register callbacks for MLflow integration"""

    @callback(
        [
            Output("mlflow-connection-status", "children"),
            Output("mlflow-total-experiments", "children"),
            Output("mlflow-total-runs", "children"),
            Output("mlflow-registered-models", "children"),
            Output("mlflow-deployed-models", "children"),
            Output("mlflow-data-store", "data"),
        ],
        [
            Input("mlflow-refresh-interval", "n_intervals"),
            Input("mlflow-refresh-btn", "n_clicks"),
        ],
    )
    def update_mlflow_stats(n_intervals, n_clicks):
        """Update MLflow statistics and connection status"""
        try:
            # Import MLflow (optional dependency)
            try:
                import mlflow
                from mlflow.tracking import MlflowClient

                client = MlflowClient()

                # Get experiments
                experiments = client.search_experiments()
                total_experiments = len(experiments)

                # Get all runs
                all_runs = []
                for exp in experiments:
                    runs = client.search_runs(exp.experiment_id)
                    all_runs.extend(runs)
                total_runs = len(all_runs)

                # Get registered models
                registered_models = client.search_registered_models()
                total_registered = len(registered_models)

                # Count deployed models (production stage)
                deployed_count = 0
                for model in registered_models:
                    versions = client.search_model_versions(f"name='{model.name}'")
                    for version in versions:
                        if version.current_stage == "Production":
                            deployed_count += 1

                # Connection success
                status = dbc.Alert(
                    [
                        html.I(className="fas fa-check-circle me-2"),
                        f"Connected to MLflow (last updated: {datetime.now().strftime('%H:%M:%S')})",
                    ],
                    color="success",
                )

                # Store data
                data = {
                    "experiments": [{"id": exp.experiment_id, "name": exp.name} for exp in experiments],
                    "runs": [{"id": run.info.run_id, "name": run.info.run_name} for run in all_runs],
                    "models": [{"name": model.name} for model in registered_models],
                    "last_updated": datetime.now().isoformat(),
                }

                return (
                    status,
                    total_experiments,
                    total_runs,
                    total_registered,
                    deployed_count,
                    data,
                )

            except ImportError:
                # MLflow not installed
                status = dbc.Alert(
                    [
                        html.I(className="fas fa-exclamation-triangle me-2"),
                        "MLflow is not installed. Install with: pip install mlflow",
                    ],
                    color="warning",
                )
                return status, "N/A", "N/A", "N/A", "N/A", {}

            except Exception as e:
                # Connection error
                status = dbc.Alert(
                    [
                        html.I(className="fas fa-times-circle me-2"),
                        f"Cannot connect to MLflow: {str(e)}",
                    ],
                    color="danger",
                )
                return status, "N/A", "N/A", "N/A", "N/A", {}

        except Exception as e:
            logger.error(f"Error updating MLflow stats: {e}")
            status = dbc.Alert(
                [
                    html.I(className="fas fa-exclamation-circle me-2"),
                    f"Error: {str(e)}",
                ],
                color="danger",
            )
            return status, "0", "0", "0", "0", {}

    @callback(
        Output("mlflow-experiment-selector", "options"),
        Input("mlflow-data-store", "data"),
    )
    def update_experiment_selector(data):
        """Update experiment selector options"""
        if not data or "experiments" not in data:
            return []

        return [{"label": exp["name"], "value": exp["id"]} for exp in data["experiments"]]

    @callback(Output("mlflow-model-selector", "options"), Input("mlflow-data-store", "data"))
    def update_model_selector(data):
        """Update model selector options"""
        if not data or "models" not in data:
            return []

        return [{"label": model["name"], "value": model["name"]} for model in data["models"]]

    @callback(
        Output("mlflow-experiment-comparison-chart", "figure"),
        [
            Input("mlflow-experiment-selector", "value"),
            Input("mlflow-metric-selector", "value"),
        ],
    )
    def update_experiment_comparison(selected_experiments, metric):
        """Update experiment comparison chart"""
        if not selected_experiments:
            # Empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="Select experiments to compare",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color="gray"),
            )
            fig.update_layout(height=400, xaxis=dict(visible=False), yaxis=dict(visible=False))
            return fig

        # Placeholder - would fetch actual data from MLflow
        fig = go.Figure()

        # Sample data
        for exp_id in selected_experiments[:5]:  # Limit to 5
            fig.add_trace(
                go.Bar(
                    name=f"Experiment {exp_id}",
                    x=[metric],
                    y=[0.85 + (hash(exp_id) % 100) / 1000],  # Mock data
                )
            )

        fig.update_layout(
            title=f"{metric.upper()} Comparison Across Experiments",
            xaxis_title="Metric",
            yaxis_title="Score",
            barmode="group",
            height=400,
        )

        return fig

    @callback(
        Output("mlflow-model-registry-table", "children"),
        [
            Input("mlflow-model-type-filter", "value"),
            Input("mlflow-model-stage-filter", "value"),
        ],
    )
    def update_model_registry_table(model_type, stage):
        """Update model registry table"""
        # Placeholder table
        table_header = [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Model Name"),
                        html.Th("Version"),
                        html.Th("Stage"),
                        html.Th("Created"),
                        html.Th("Last Modified"),
                        html.Th("Actions"),
                    ]
                )
            )
        ]

        table_body = [
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td("lstm_anomaly_detector"),
                            html.Td("v1.2.0"),
                            html.Td(dbc.Badge("Production", color="success")),
                            html.Td("2025-09-15"),
                            html.Td("2025-10-01"),
                            html.Td(
                                [
                                    dbc.ButtonGroup(
                                        [
                                            dbc.Button(
                                                "View",
                                                size="sm",
                                                color="info",
                                                outline=True,
                                            ),
                                            dbc.Button(
                                                "Deploy",
                                                size="sm",
                                                color="success",
                                                outline=True,
                                            ),
                                        ]
                                    )
                                ]
                            ),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td("transformer_forecaster"),
                            html.Td("v2.0.1"),
                            html.Td(dbc.Badge("Staging", color="warning")),
                            html.Td("2025-09-20"),
                            html.Td("2025-10-02"),
                            html.Td(
                                [
                                    dbc.ButtonGroup(
                                        [
                                            dbc.Button(
                                                "View",
                                                size="sm",
                                                color="info",
                                                outline=True,
                                            ),
                                            dbc.Button(
                                                "Promote",
                                                size="sm",
                                                color="primary",
                                                outline=True,
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

        return dbc.Table(
            table_header + table_body,
            bordered=True,
            hover=True,
            responsive=True,
            striped=True,
            className="mt-3",
        )

    @callback(
        Output("mlflow-deployment-status-cards", "children"),
        Input("mlflow-data-store", "data"),
    )
    def update_deployment_status(data):
        """Update deployment status cards"""
        deployments = [
            {
                "name": "LSTM Anomaly Detector",
                "version": "v1.2.0",
                "status": "Running",
                "health": "Healthy",
                "uptime": "15d 4h",
                "requests": "1.2M",
            },
            {
                "name": "Transformer Forecaster",
                "version": "v2.0.1",
                "status": "Running",
                "health": "Healthy",
                "uptime": "2d 12h",
                "requests": "450K",
            },
            {
                "name": "LSTM VAE",
                "version": "v1.0.5",
                "status": "Stopped",
                "health": "N/A",
                "uptime": "N/A",
                "requests": "0",
            },
        ]

        cards = []
        for dep in deployments:
            color = "success" if dep["status"] == "Running" else "secondary"
            cards.append(
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(dep["name"]),
                                dbc.CardBody(
                                    [
                                        html.P([html.Strong("Version: "), dep["version"]]),
                                        html.P(
                                            [
                                                html.Strong("Status: "),
                                                dbc.Badge(dep["status"], color=color),
                                            ]
                                        ),
                                        html.P([html.Strong("Health: "), dep["health"]]),
                                        html.P([html.Strong("Uptime: "), dep["uptime"]]),
                                        html.P(
                                            [
                                                html.Strong("Requests: "),
                                                dep["requests"],
                                            ],
                                            className="mb-0",
                                        ),
                                    ]
                                ),
                            ],
                            className="h-100",
                        )
                    ],
                    width=4,
                )
            )

        return dbc.Row(cards, className="mb-4")

    @callback(
        Output("mlflow-performance-trends", "figure"),
        [
            Input("mlflow-model-selector", "value"),
            Input("mlflow-time-range", "value"),
        ],
    )
    def update_performance_trends(model_name, time_range):
        """Update performance trends chart"""
        if not model_name:
            fig = go.Figure()
            fig.add_annotation(
                text="Select a model to view performance trends",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color="gray"),
            )
            fig.update_layout(height=400, xaxis=dict(visible=False), yaxis=dict(visible=False))
            return fig

        # Generate sample trend data
        days = {"7d": 7, "30d": 30, "90d": 90, "all": 180}.get(time_range, 30)
        dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

        fig = go.Figure()

        # Add traces for different metrics
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=[0.85 + (i % 10) / 100 for i in range(days)],
                name="Accuracy",
                mode="lines+markers",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=[0.82 + (i % 8) / 100 for i in range(days)],
                name="Precision",
                mode="lines+markers",
            )
        )

        fig.update_layout(
            title=f"Performance Trends - {model_name}",
            xaxis_title="Date",
            yaxis_title="Score",
            hovermode="x unified",
            height=400,
        )

        return fig

    @callback(
        Output("mlflow-parameter-importance", "figure"),
        Input("mlflow-model-selector", "value"),
    )
    def update_parameter_importance(model_name):
        """Update hyperparameter importance chart"""
        if not model_name:
            fig = go.Figure()
            fig.add_annotation(
                text="Select a model to view parameter importance",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color="gray"),
            )
            fig.update_layout(height=300, xaxis=dict(visible=False), yaxis=dict(visible=False))
            return fig

        # Sample parameter importance data
        params = ["learning_rate", "batch_size", "dropout", "hidden_units", "epochs"]
        importance = [0.85, 0.72, 0.65, 0.58, 0.45]

        fig = go.Figure(
            go.Bar(
                x=importance,
                y=params,
                orientation="h",
                marker=dict(color=importance, colorscale="Viridis"),
            )
        )

        fig.update_layout(
            title="Hyperparameter Importance",
            xaxis_title="Importance Score",
            yaxis_title="Parameter",
            height=300,
        )

        return fig

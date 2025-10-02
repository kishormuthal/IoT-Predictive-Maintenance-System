"""
Model Registry Dashboard Component
Model management and versioning interface for the enhanced dashboard
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
from dash import Input, Output, State, callback_context, dash_table, dcc, html

logger = logging.getLogger(__name__)


def create_model_registry_layout():
    """Create the model registry layout"""
    return dbc.Container(
        [
            # Model Registry Header
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
                                                    html.I(className="fas fa-database me-3 text-success"),
                                                    "Model Registry",
                                                ],
                                                className="mb-3",
                                            ),
                                            html.P(
                                                "Browse, manage, and analyze trained models across all equipment sensors.",
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
            # Registry Statistics
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.I(className="fas fa-chart-bar me-2"),
                                            "Registry Statistics",
                                        ]
                                    ),
                                    dbc.CardBody([html.Div(id="registry-stats-cards")]),
                                ]
                            )
                        ]
                    )
                ],
                className="mb-4",
            ),
            # Model Filter and Search
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.I(className="fas fa-filter me-2"),
                                            "Filter & Search Models",
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Model Type:",
                                                                className="fw-bold",
                                                            ),
                                                            dcc.Dropdown(
                                                                id="model-type-filter",
                                                                options=[
                                                                    {
                                                                        "label": "All Models",
                                                                        "value": "all",
                                                                    },
                                                                    {
                                                                        "label": "Anomaly Detection (Telemanom)",
                                                                        "value": "telemanom",
                                                                    },
                                                                    {
                                                                        "label": "Forecasting (Transformer)",
                                                                        "value": "transformer",
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
                                                                "Performance Grade:",
                                                                className="fw-bold",
                                                            ),
                                                            dcc.Dropdown(
                                                                id="performance-grade-filter",
                                                                options=[
                                                                    {
                                                                        "label": "All Grades",
                                                                        "value": "all",
                                                                    },
                                                                    {
                                                                        "label": "A (Excellent)",
                                                                        "value": "A",
                                                                    },
                                                                    {
                                                                        "label": "B (Good)",
                                                                        "value": "B",
                                                                    },
                                                                    {
                                                                        "label": "C (Fair)",
                                                                        "value": "C",
                                                                    },
                                                                    {
                                                                        "label": "D (Poor)",
                                                                        "value": "D",
                                                                    },
                                                                    {
                                                                        "label": "F (Failed)",
                                                                        "value": "F",
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
                                                                "Equipment Type:",
                                                                className="fw-bold",
                                                            ),
                                                            dcc.Dropdown(
                                                                id="equipment-type-filter",
                                                                placeholder="Select equipment type",
                                                            ),
                                                        ],
                                                        width=3,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Search Sensor:",
                                                                className="fw-bold",
                                                            ),
                                                            dcc.Dropdown(
                                                                id="sensor-search",
                                                                placeholder="Search by sensor ID",
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
            # Model Overview Table
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.I(className="fas fa-table me-2"),
                                            "Model Overview",
                                            dbc.ButtonGroup(
                                                [
                                                    dbc.Button(
                                                        [
                                                            html.I(className="fas fa-sync me-1"),
                                                            "Refresh",
                                                        ],
                                                        id="refresh-models-btn",
                                                        color="outline-primary",
                                                        size="sm",
                                                    ),
                                                    dbc.Button(
                                                        [
                                                            html.I(className="fas fa-download me-1"),
                                                            "Export",
                                                        ],
                                                        id="export-models-btn",
                                                        color="outline-success",
                                                        size="sm",
                                                    ),
                                                ],
                                                className="ms-auto",
                                            ),
                                        ]
                                    ),
                                    dbc.CardBody([html.Div(id="models-table")]),
                                ]
                            )
                        ]
                    )
                ],
                className="mb-4",
            ),
            # Model Details and Actions
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.I(className="fas fa-info-circle me-2"),
                                            "Model Details",
                                        ]
                                    ),
                                    dbc.CardBody([html.Div(id="model-details-panel")]),
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
                                            html.I(className="fas fa-tools me-2"),
                                            "Model Actions",
                                        ]
                                    ),
                                    dbc.CardBody([html.Div(id="model-actions-panel")]),
                                ]
                            )
                        ],
                        width=4,
                    ),
                ],
                className="mb-4",
            ),
            # Model Performance Analytics
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.I(className="fas fa-chart-line me-2"),
                                            "Performance Analytics",
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dbc.Tabs(
                                                [
                                                    dbc.Tab(
                                                        label="Performance Distribution",
                                                        tab_id="perf-dist",
                                                    ),
                                                    dbc.Tab(
                                                        label="Model Versions Timeline",
                                                        tab_id="versions-timeline",
                                                    ),
                                                    dbc.Tab(
                                                        label="Training Metrics",
                                                        tab_id="training-metrics",
                                                    ),
                                                ],
                                                id="analytics-tabs",
                                                active_tab="perf-dist",
                                            ),
                                            html.Div(id="analytics-content", className="mt-3"),
                                        ]
                                    ),
                                ]
                            )
                        ]
                    )
                ]
            ),
            # Hidden components for state management
            dcc.Store(id="selected-model-store", data={}),
            dcc.Store(id="models-data-store", data=[]),
            dcc.Store(id="registry-stats-store", data={}),
        ]
    )


def register_model_registry_callbacks(app, model_registry, training_use_case, config_manager):
    """Register callbacks for model registry functionality"""

    @app.callback(
        [
            Output("equipment-type-filter", "options"),
            Output("sensor-search", "options"),
        ],
        Input("models-tab-content", "children"),
    )
    def populate_filter_options(children):
        """Populate filter dropdown options"""
        try:
            from config.equipment_config import get_equipment_list

            equipment_list = get_equipment_list()

            # Equipment type options
            equipment_types = list(set([eq.equipment_type.value for eq in equipment_list]))
            equipment_options = [{"label": "All Types", "value": "all"}] + [
                {"label": eq_type, "value": eq_type} for eq_type in sorted(equipment_types)
            ]

            # Sensor options
            sensor_options = [
                {"label": f"{eq.equipment_id} - {eq.name}", "value": eq.equipment_id} for eq in equipment_list
            ]

            return equipment_options, sensor_options

        except Exception as e:
            logger.error(f"Error populating filter options: {e}")
            return [], []

    @app.callback(
        [
            Output("registry-stats-cards", "children"),
            Output("registry-stats-store", "data"),
        ],
        Input("global-refresh", "n_intervals"),
    )
    def update_registry_statistics(n):
        """Update registry statistics"""
        try:
            # Get registry stats
            registry_stats = model_registry.get_registry_stats()
            training_status = training_use_case.get_training_status()

            # Calculate additional metrics
            total_models = registry_stats.get("total_models", 0)
            total_versions = registry_stats.get("total_versions", 0)
            total_size_mb = registry_stats.get("total_size_mb", 0)
            model_types = registry_stats.get("model_types", {})

            equipment_status = training_status.get("equipment_status", {})
            total_equipment = len(equipment_status)

            # Calculate coverage
            anomaly_coverage = sum(
                1 for s in equipment_status.values() if s.get("anomaly_detection", {}).get("trained", False)
            )
            forecast_coverage = sum(
                1 for s in equipment_status.values() if s.get("forecasting", {}).get("trained", False)
            )

            # Create statistics cards
            stats_cards = dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H3(
                                                str(total_models),
                                                className="text-primary mb-1",
                                            ),
                                            html.P(
                                                "Total Models",
                                                className="text-muted mb-0",
                                            ),
                                            html.Small(
                                                f"{total_versions} versions",
                                                className="text-muted",
                                            ),
                                        ]
                                    )
                                ],
                                className="text-center h-100",
                            )
                        ],
                        width=2,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H3(
                                                f"{total_size_mb:.1f} MB",
                                                className="text-info mb-1",
                                            ),
                                            html.P(
                                                "Storage Used",
                                                className="text-muted mb-0",
                                            ),
                                            html.Small(
                                                f"Avg: {total_size_mb/max(total_models, 1):.1f} MB/model",
                                                className="text-muted",
                                            ),
                                        ]
                                    )
                                ],
                                className="text-center h-100",
                            )
                        ],
                        width=2,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H3(
                                                f"{anomaly_coverage}/{total_equipment}",
                                                className="text-danger mb-1",
                                            ),
                                            html.P(
                                                "Anomaly Models",
                                                className="text-muted mb-0",
                                            ),
                                            dbc.Progress(
                                                value=(anomaly_coverage / max(total_equipment, 1)) * 100,
                                                color="danger",
                                                size="sm",
                                            ),
                                        ]
                                    )
                                ],
                                className="text-center h-100",
                            )
                        ],
                        width=2,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H3(
                                                f"{forecast_coverage}/{total_equipment}",
                                                className="text-success mb-1",
                                            ),
                                            html.P(
                                                "Forecast Models",
                                                className="text-muted mb-0",
                                            ),
                                            dbc.Progress(
                                                value=(forecast_coverage / max(total_equipment, 1)) * 100,
                                                color="success",
                                                size="sm",
                                            ),
                                        ]
                                    )
                                ],
                                className="text-center h-100",
                            )
                        ],
                        width=2,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H3(
                                                f"{model_types.get('telemanom', 0)}",
                                                className="text-warning mb-1",
                                            ),
                                            html.P("Telemanom", className="text-muted mb-0"),
                                            html.Small(
                                                "Anomaly Detection",
                                                className="text-muted",
                                            ),
                                        ]
                                    )
                                ],
                                className="text-center h-100",
                            )
                        ],
                        width=2,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H3(
                                                f"{model_types.get('transformer', 0)}",
                                                className="text-secondary mb-1",
                                            ),
                                            html.P(
                                                "Transformer",
                                                className="text-muted mb-0",
                                            ),
                                            html.Small("Forecasting", className="text-muted"),
                                        ]
                                    )
                                ],
                                className="text-center h-100",
                            )
                        ],
                        width=2,
                    ),
                ]
            )

            return stats_cards, registry_stats

        except Exception as e:
            logger.error(f"Error updating registry statistics: {e}")
            error_card = dbc.Alert("Error loading registry statistics", color="danger")
            return error_card, {}

    @app.callback(
        [Output("models-table", "children"), Output("models-data-store", "data")],
        [
            Input("refresh-models-btn", "n_clicks"),
            Input("model-type-filter", "value"),
            Input("performance-grade-filter", "value"),
            Input("equipment-type-filter", "value"),
            Input("sensor-search", "value"),
            Input("global-refresh", "n_intervals"),
        ],
    )
    def update_models_table(
        refresh_clicks,
        model_type_filter,
        grade_filter,
        equipment_filter,
        sensor_filter,
        n,
    ):
        """Update models table with filtering"""
        try:
            # Get all models
            models_data = []

            # Get models from registry
            telemanom_models = model_registry.list_models("telemanom")
            transformer_models = model_registry.list_models("transformer")

            all_models = telemanom_models + transformer_models

            # Get equipment info for enrichment
            from config.equipment_config import get_equipment_list

            equipment_list = get_equipment_list()
            equipment_dict = {eq.equipment_id: eq for eq in equipment_list}

            # Enrich model data
            for model in all_models:
                sensor_id = model["sensor_id"]
                equipment = equipment_dict.get(sensor_id)

                # Get active version details
                active_version = model_registry.get_active_model_version(sensor_id, model["model_type"])
                metadata = None
                if active_version:
                    metadata = model_registry.get_model_metadata(active_version)

                model_data = {
                    "sensor_id": sensor_id,
                    "model_type": model["model_type"],
                    "equipment_type": (equipment.equipment_type.value if equipment else "Unknown"),
                    "equipment_name": equipment.name if equipment else "Unknown",
                    "criticality": (equipment.criticality.value if equipment else "Unknown"),
                    "total_versions": model["total_versions"],
                    "active_version": active_version or "None",
                    "performance_score": metadata.performance_score if metadata else 0,
                    "model_size_mb": ((metadata.model_size_bytes / (1024 * 1024)) if metadata else 0),
                    "last_trained": metadata.created_at if metadata else "Never",
                    "performance_grade": _calculate_grade(metadata.performance_score if metadata else 0),
                }

                # Apply filters
                if model_type_filter != "all" and model_data["model_type"] != model_type_filter:
                    continue
                if grade_filter != "all" and model_data["performance_grade"] != grade_filter:
                    continue
                if equipment_filter != "all" and model_data["equipment_type"] != equipment_filter:
                    continue
                if sensor_filter and model_data["sensor_id"] != sensor_filter:
                    continue

                models_data.append(model_data)

            # Create table
            if not models_data:
                return (
                    dbc.Alert("No models found matching the selected filters.", color="info"),
                    [],
                )

            # Create DataFrame for table
            df = pd.DataFrame(models_data)

            table = dash_table.DataTable(
                id="models-datatable",
                columns=[
                    {"name": "Sensor ID", "id": "sensor_id"},
                    {"name": "Model Type", "id": "model_type"},
                    {"name": "Equipment", "id": "equipment_name"},
                    {"name": "Type", "id": "equipment_type"},
                    {"name": "Criticality", "id": "criticality"},
                    {"name": "Versions", "id": "total_versions", "type": "numeric"},
                    {
                        "name": "Performance",
                        "id": "performance_score",
                        "type": "numeric",
                        "format": {"specifier": ".3f"},
                    },
                    {"name": "Grade", "id": "performance_grade"},
                    {
                        "name": "Size (MB)",
                        "id": "model_size_mb",
                        "type": "numeric",
                        "format": {"specifier": ".1f"},
                    },
                    {"name": "Last Trained", "id": "last_trained"},
                ],
                data=df.to_dict("records"),
                sort_action="native",
                filter_action="native",
                row_selectable="single",
                selected_rows=[],
                style_cell={
                    "textAlign": "left",
                    "padding": "10px",
                    "fontFamily": "Arial",
                },
                style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold"},
                style_data_conditional=[
                    {
                        "if": {"filter_query": "{performance_grade} = A"},
                        "backgroundColor": "#d4edda",
                        "color": "black",
                    },
                    {
                        "if": {"filter_query": "{performance_grade} = F"},
                        "backgroundColor": "#f8d7da",
                        "color": "black",
                    },
                ],
                page_size=10,
                style_table={"overflowX": "auto"},
            )

            return table, models_data

        except Exception as e:
            logger.error(f"Error updating models table: {e}")
            return dbc.Alert(f"Error loading models: {str(e)}", color="danger"), []

    @app.callback(
        [
            Output("model-details-panel", "children"),
            Output("model-actions-panel", "children"),
        ],
        [Input("models-datatable", "selected_rows")],
        [State("models-data-store", "data")],
    )
    def update_model_details(selected_rows, models_data):
        """Update model details and actions panel"""
        try:
            if not selected_rows or not models_data:
                return (
                    "Select a model to view details",
                    "Select a model to see available actions",
                )

            selected_model = models_data[selected_rows[0]]
            sensor_id = selected_model["sensor_id"]
            model_type = selected_model["model_type"]

            # Get detailed model information
            active_version = model_registry.get_active_model_version(sensor_id, model_type)
            if not active_version:
                return "Model details not available", "No actions available"

            metadata = model_registry.get_model_metadata(active_version)
            if not metadata:
                return "Model metadata not available", "No actions available"

            # Create details panel
            details = dbc.ListGroup(
                [
                    dbc.ListGroupItem([html.Strong("Model ID: "), metadata.model_id]),
                    dbc.ListGroupItem([html.Strong("Version ID: "), metadata.version]),
                    dbc.ListGroupItem([html.Strong("Created: "), metadata.created_at]),
                    dbc.ListGroupItem(
                        [
                            html.Strong("Training Time: "),
                            f"{metadata.training_time_seconds:.1f} seconds",
                        ]
                    ),
                    dbc.ListGroupItem(
                        [
                            html.Strong("Model Size: "),
                            f"{metadata.model_size_bytes / (1024*1024):.1f} MB",
                        ]
                    ),
                    dbc.ListGroupItem(
                        [
                            html.Strong("Performance Score: "),
                            f"{metadata.performance_score:.3f}",
                        ]
                    ),
                    dbc.ListGroupItem(
                        [
                            html.Strong("Description: "),
                            metadata.description or "No description",
                        ]
                    ),
                ],
                flush=True,
            )

            # Create actions panel
            actions = dbc.Stack(
                [
                    dbc.Button(
                        [html.I(className="fas fa-eye me-2"), "View Versions"],
                        id="view-versions-btn",
                        color="primary",
                        className="mb-2",
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-check me-2"), "Validate Model"],
                        id="validate-model-btn",
                        color="success",
                        className="mb-2",
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-sync me-2"), "Retrain Model"],
                        id="retrain-model-btn",
                        color="warning",
                        className="mb-2",
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-download me-2"), "Export Model"],
                        id="export-model-btn",
                        color="info",
                        className="mb-2",
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-trash me-2"), "Delete Version"],
                        id="delete-version-btn",
                        color="danger",
                    ),
                ]
            )

            return details, actions

        except Exception as e:
            logger.error(f"Error updating model details: {e}")
            return "Error loading model details", "Error loading actions"

    @app.callback(
        Output("analytics-content", "children"),
        [Input("analytics-tabs", "active_tab")],
        [State("models-data-store", "data")],
    )
    def update_analytics_content(active_tab, models_data):
        """Update analytics content based on selected tab"""
        try:
            if not models_data:
                return dbc.Alert("No model data available for analytics", color="info")

            df = pd.DataFrame(models_data)

            if active_tab == "perf-dist":
                # Performance distribution chart
                fig = px.histogram(
                    df,
                    x="performance_score",
                    color="model_type",
                    title="Model Performance Score Distribution",
                    nbins=20,
                    barmode="overlay",
                )
                fig.update_layout(height=400)
                return dcc.Graph(figure=fig)

            elif active_tab == "versions-timeline":
                # Model versions timeline
                fig = px.scatter(
                    df,
                    x="last_trained",
                    y="performance_score",
                    color="model_type",
                    size="model_size_mb",
                    hover_data=["sensor_id", "total_versions"],
                    title="Model Performance vs Training Date",
                )
                fig.update_layout(height=400)
                return dcc.Graph(figure=fig)

            elif active_tab == "training-metrics":
                # Training metrics comparison
                fig = px.box(
                    df,
                    x="model_type",
                    y="performance_score",
                    title="Performance Score Distribution by Model Type",
                )
                fig.update_layout(height=400)
                return dcc.Graph(figure=fig)

            return dbc.Alert("Analytics view not implemented", color="warning")

        except Exception as e:
            logger.error(f"Error updating analytics content: {e}")
            return dbc.Alert(f"Error loading analytics: {str(e)}", color="danger")

    return app


def _calculate_grade(score: float) -> str:
    """Calculate performance grade from score"""
    if score >= 0.9:
        return "A"
    elif score >= 0.8:
        return "B"
    elif score >= 0.7:
        return "C"
    elif score >= 0.6:
        return "D"
    else:
        return "F"

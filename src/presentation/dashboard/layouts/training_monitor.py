"""
Training Job Monitoring Dashboard
Real-time monitoring of model training jobs with progress tracking and logs
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, callback, dcc, html

logger = logging.getLogger(__name__)


def create_training_monitor_layout() -> html.Div:
    """
    Create training job monitoring dashboard

    Features:
    - Active training jobs with real-time progress
    - Training history and logs
    - Resource utilization (CPU, GPU, memory)
    - Loss curves and metric tracking
    - Training job queue management
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
                                    html.I(className="fas fa-brain me-2"),
                                    "Model Training Monitor",
                                ],
                                className="mb-0",
                            ),
                            html.P(
                                "Monitor active training jobs and view training history",
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
                                            html.I(className="fas fa-plus me-2"),
                                            "New Training Job",
                                        ],
                                        id="new-training-job-btn",
                                        color="success",
                                        size="sm",
                                    ),
                                    dbc.Button(
                                        [
                                            html.I(className="fas fa-sync me-2"),
                                            "Refresh",
                                        ],
                                        id="training-refresh-btn",
                                        color="primary",
                                        size="sm",
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
            # Status Summary Cards
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
                                                    html.I(
                                                        className="fas fa-running fa-2x text-success mb-2"
                                                    ),
                                                    html.H3(
                                                        id="training-active-jobs",
                                                        children="0",
                                                    ),
                                                    html.P(
                                                        "Active Jobs",
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
                                                    html.I(
                                                        className="fas fa-hourglass-half fa-2x text-warning mb-2"
                                                    ),
                                                    html.H3(
                                                        id="training-queued-jobs",
                                                        children="0",
                                                    ),
                                                    html.P(
                                                        "Queued Jobs",
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
                                                    html.I(
                                                        className="fas fa-check-circle fa-2x text-info mb-2"
                                                    ),
                                                    html.H3(
                                                        id="training-completed-jobs",
                                                        children="0",
                                                    ),
                                                    html.P(
                                                        "Completed (24h)",
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
                                                    html.I(
                                                        className="fas fa-times-circle fa-2x text-danger mb-2"
                                                    ),
                                                    html.H3(
                                                        id="training-failed-jobs",
                                                        children="0",
                                                    ),
                                                    html.P(
                                                        "Failed (24h)",
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
            # Main Content
            dbc.Tabs(
                [
                    # Tab 1: Active Training Jobs
                    dbc.Tab(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [html.Div(id="active-training-jobs-container")]
                                    )
                                ]
                            )
                        ],
                        label="Active Jobs",
                        tab_id="active-jobs",
                    ),
                    # Tab 2: Training Queue
                    dbc.Tab(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5(
                                                "Training Job Queue", className="mb-3"
                                            ),
                                            html.Div(id="training-queue-container"),
                                        ]
                                    )
                                ]
                            )
                        ],
                        label="Queue",
                        tab_id="queue",
                    ),
                    # Tab 3: Training History
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
                                                            html.Label(
                                                                "Filter by Status:"
                                                            ),
                                                            dcc.Dropdown(
                                                                id="training-status-filter",
                                                                options=[
                                                                    {
                                                                        "label": "All",
                                                                        "value": "all",
                                                                    },
                                                                    {
                                                                        "label": "Completed",
                                                                        "value": "completed",
                                                                    },
                                                                    {
                                                                        "label": "Failed",
                                                                        "value": "failed",
                                                                    },
                                                                    {
                                                                        "label": "Cancelled",
                                                                        "value": "cancelled",
                                                                    },
                                                                ],
                                                                value="all",
                                                                clearable=False,
                                                            ),
                                                        ],
                                                        width=4,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Filter by Model Type:"
                                                            ),
                                                            dcc.Dropdown(
                                                                id="training-model-filter",
                                                                options=[
                                                                    {
                                                                        "label": "All Models",
                                                                        "value": "all",
                                                                    },
                                                                    {
                                                                        "label": "LSTM Predictor",
                                                                        "value": "lstm_predictor",
                                                                    },
                                                                    {
                                                                        "label": "LSTM Autoencoder",
                                                                        "value": "lstm_autoencoder",
                                                                    },
                                                                    {
                                                                        "label": "LSTM VAE",
                                                                        "value": "lstm_vae",
                                                                    },
                                                                    {
                                                                        "label": "Transformer",
                                                                        "value": "transformer",
                                                                    },
                                                                ],
                                                                value="all",
                                                                clearable=False,
                                                            ),
                                                        ],
                                                        width=4,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label("Time Range:"),
                                                            dcc.Dropdown(
                                                                id="training-time-range",
                                                                options=[
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
                                                                    {
                                                                        "label": "All Time",
                                                                        "value": "all",
                                                                    },
                                                                ],
                                                                value="7d",
                                                                clearable=False,
                                                            ),
                                                        ],
                                                        width=4,
                                                    ),
                                                ],
                                                className="mb-4",
                                            ),
                                            html.Div(id="training-history-table"),
                                        ]
                                    )
                                ]
                            )
                        ],
                        label="History",
                        tab_id="history",
                    ),
                    # Tab 4: Resource Utilization
                    dbc.Tab(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5(
                                                "Resource Utilization", className="mb-3"
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dcc.Graph(
                                                                id="training-cpu-usage-chart"
                                                            )
                                                        ],
                                                        width=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dcc.Graph(
                                                                id="training-gpu-usage-chart"
                                                            )
                                                        ],
                                                        width=6,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dcc.Graph(
                                                                id="training-memory-usage-chart"
                                                            )
                                                        ],
                                                        width=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dcc.Graph(
                                                                id="training-disk-usage-chart"
                                                            )
                                                        ],
                                                        width=6,
                                                    ),
                                                ]
                                            ),
                                        ]
                                    )
                                ]
                            )
                        ],
                        label="Resources",
                        tab_id="resources",
                    ),
                ],
                id="training-tabs",
                active_tab="active-jobs",
            ),
            # New Training Job Modal
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Start New Training Job")),
                    dbc.ModalBody(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Label(
                                                "Model Type:", className="fw-bold"
                                            ),
                                            dcc.Dropdown(
                                                id="new-training-model-type",
                                                options=[
                                                    {
                                                        "label": "LSTM Predictor",
                                                        "value": "lstm_predictor",
                                                    },
                                                    {
                                                        "label": "LSTM Autoencoder",
                                                        "value": "lstm_autoencoder",
                                                    },
                                                    {
                                                        "label": "LSTM VAE",
                                                        "value": "lstm_vae",
                                                    },
                                                    {
                                                        "label": "Transformer",
                                                        "value": "transformer",
                                                    },
                                                ],
                                                placeholder="Select model type...",
                                            ),
                                        ],
                                        width=12,
                                    )
                                ],
                                className="mb-3",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Label("Dataset:", className="fw-bold"),
                                            dcc.Dropdown(
                                                id="new-training-dataset",
                                                options=[
                                                    {
                                                        "label": "SMAP (Soil Moisture Active Passive)",
                                                        "value": "smap",
                                                    },
                                                    {
                                                        "label": "MSL (Mars Science Laboratory)",
                                                        "value": "msl",
                                                    },
                                                    {
                                                        "label": "Combined Dataset",
                                                        "value": "combined",
                                                    },
                                                ],
                                                placeholder="Select dataset...",
                                            ),
                                        ],
                                        width=12,
                                    )
                                ],
                                className="mb-3",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Label("Epochs:", className="fw-bold"),
                                            dbc.Input(
                                                id="new-training-epochs",
                                                type="number",
                                                value=50,
                                                min=1,
                                                max=1000,
                                            ),
                                        ],
                                        width=6,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Label(
                                                "Batch Size:", className="fw-bold"
                                            ),
                                            dcc.Dropdown(
                                                id="new-training-batch-size",
                                                options=[
                                                    {"label": "16", "value": 16},
                                                    {"label": "32", "value": 32},
                                                    {"label": "64", "value": 64},
                                                    {"label": "128", "value": 128},
                                                ],
                                                value=32,
                                            ),
                                        ],
                                        width=6,
                                    ),
                                ],
                                className="mb-3",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Label(
                                                "Learning Rate:", className="fw-bold"
                                            ),
                                            dbc.Input(
                                                id="new-training-learning-rate",
                                                type="number",
                                                value=0.001,
                                                step=0.0001,
                                                min=0.00001,
                                                max=0.1,
                                            ),
                                        ],
                                        width=6,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Label(
                                                "Validation Split:", className="fw-bold"
                                            ),
                                            dbc.Input(
                                                id="new-training-val-split",
                                                type="number",
                                                value=0.2,
                                                step=0.05,
                                                min=0.1,
                                                max=0.5,
                                            ),
                                        ],
                                        width=6,
                                    ),
                                ],
                                className="mb-3",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Checklist(
                                                id="new-training-options",
                                                options=[
                                                    {
                                                        "label": "Enable early stopping",
                                                        "value": "early_stopping",
                                                    },
                                                    {
                                                        "label": "Enable MLflow tracking",
                                                        "value": "mlflow",
                                                    },
                                                    {
                                                        "label": "Use GPU acceleration",
                                                        "value": "gpu",
                                                    },
                                                    {
                                                        "label": "Save checkpoints",
                                                        "value": "checkpoints",
                                                    },
                                                ],
                                                value=[
                                                    "early_stopping",
                                                    "mlflow",
                                                    "checkpoints",
                                                ],
                                                inline=False,
                                            )
                                        ],
                                        width=12,
                                    )
                                ]
                            ),
                        ]
                    ),
                    dbc.ModalFooter(
                        [
                            dbc.Button(
                                "Cancel",
                                id="new-training-cancel-btn",
                                color="secondary",
                            ),
                            dbc.Button(
                                "Start Training",
                                id="new-training-start-btn",
                                color="primary",
                            ),
                        ]
                    ),
                ],
                id="new-training-modal",
                size="lg",
                is_open=False,
            ),
            # Auto-refresh interval
            dcc.Interval(
                id="training-refresh-interval",
                interval=5000,  # 5 seconds
                n_intervals=0,
            ),
            # Store for training job data
            dcc.Store(id="training-jobs-store"),
        ],
        className="p-4",
    )


def register_training_monitor_callbacks(app: dash.Dash):
    """Register callbacks for training monitor"""

    @callback(
        Output("new-training-modal", "is_open"),
        [
            Input("new-training-job-btn", "n_clicks"),
            Input("new-training-cancel-btn", "n_clicks"),
            Input("new-training-start-btn", "n_clicks"),
        ],
        State("new-training-modal", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_training_modal(open_click, cancel_click, start_click, is_open):
        """Toggle new training job modal"""
        return not is_open

    @callback(
        [
            Output("training-active-jobs", "children"),
            Output("training-queued-jobs", "children"),
            Output("training-completed-jobs", "children"),
            Output("training-failed-jobs", "children"),
            Output("training-jobs-store", "data"),
        ],
        [
            Input("training-refresh-interval", "n_intervals"),
            Input("training-refresh-btn", "n_clicks"),
        ],
    )
    def update_training_stats(n_intervals, n_clicks):
        """Update training job statistics"""
        # Mock data - replace with actual training job queries
        active_jobs = [
            {
                "id": "job_001",
                "model": "LSTM Predictor",
                "status": "running",
                "progress": 65,
                "epoch": 65,
                "total_epochs": 100,
                "current_loss": 0.0234,
                "started": (datetime.now() - timedelta(minutes=45)).isoformat(),
                "eta": "35 min",
            },
            {
                "id": "job_002",
                "model": "Transformer",
                "status": "running",
                "progress": 30,
                "epoch": 15,
                "total_epochs": 50,
                "current_loss": 0.0567,
                "started": (datetime.now() - timedelta(minutes=20)).isoformat(),
                "eta": "55 min",
            },
        ]

        queued_jobs = [
            {
                "id": "job_003",
                "model": "LSTM VAE",
                "status": "queued",
                "position": 1,
                "submitted": (datetime.now() - timedelta(minutes=5)).isoformat(),
            }
        ]

        # Count jobs from last 24 hours
        completed_count = 5
        failed_count = 1

        jobs_data = {
            "active": active_jobs,
            "queued": queued_jobs,
            "completed_24h": completed_count,
            "failed_24h": failed_count,
            "last_updated": datetime.now().isoformat(),
        }

        return (
            len(active_jobs),
            len(queued_jobs),
            completed_count,
            failed_count,
            jobs_data,
        )

    @callback(
        Output("active-training-jobs-container", "children"),
        Input("training-jobs-store", "data"),
    )
    def update_active_jobs(jobs_data):
        """Update active training jobs display"""
        if not jobs_data or not jobs_data.get("active"):
            return dbc.Alert(
                [
                    html.I(className="fas fa-info-circle me-2"),
                    "No active training jobs",
                ],
                color="info",
            )

        jobs = []
        for job in jobs_data["active"]:
            job_card = dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H5(
                                                [
                                                    html.I(
                                                        className="fas fa-brain me-2"
                                                    ),
                                                    job["model"],
                                                ],
                                                className="mb-0",
                                            )
                                        ],
                                        width=6,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Badge(
                                                f"Epoch {job['epoch']}/{job['total_epochs']}",
                                                color="info",
                                                className="me-2",
                                            ),
                                            dbc.Badge(
                                                f"ETA: {job['eta']}", color="secondary"
                                            ),
                                        ],
                                        width=6,
                                        className="text-end",
                                    ),
                                ]
                            )
                        ]
                    ),
                    dbc.CardBody(
                        [
                            # Progress bar
                            html.Div(
                                [
                                    html.Label(
                                        f"Progress: {job['progress']}%",
                                        className="mb-2",
                                    ),
                                    dbc.Progress(
                                        value=job["progress"],
                                        striped=True,
                                        animated=True,
                                        color=(
                                            "success"
                                            if job["progress"] > 66
                                            else (
                                                "warning"
                                                if job["progress"] > 33
                                                else "info"
                                            )
                                        ),
                                        className="mb-3",
                                    ),
                                ]
                            ),
                            # Metrics
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.P(
                                                [
                                                    html.Strong("Job ID: "),
                                                    html.Code(job["id"]),
                                                ],
                                                className="mb-1",
                                            ),
                                            html.P(
                                                [
                                                    html.Strong("Current Loss: "),
                                                    f"{job['current_loss']:.4f}",
                                                ],
                                                className="mb-1",
                                            ),
                                        ],
                                        width=6,
                                    ),
                                    dbc.Col(
                                        [
                                            html.P(
                                                [
                                                    html.Strong("Started: "),
                                                    datetime.fromisoformat(
                                                        job["started"]
                                                    ).strftime("%H:%M:%S"),
                                                ],
                                                className="mb-1",
                                            ),
                                            html.P(
                                                [
                                                    html.Strong("Status: "),
                                                    dbc.Badge(
                                                        "Running", color="success"
                                                    ),
                                                ],
                                                className="mb-1",
                                            ),
                                        ],
                                        width=6,
                                    ),
                                ]
                            ),
                            # Mini loss curve
                            dcc.Graph(
                                id={"type": "job-loss-curve", "index": job["id"]},
                                figure=create_mini_loss_curve(job["id"]),
                                config={"displayModeBar": False},
                                style={"height": "150px"},
                            ),
                            # Actions
                            dbc.ButtonGroup(
                                [
                                    dbc.Button(
                                        [
                                            html.I(className="fas fa-eye me-2"),
                                            "View Logs",
                                        ],
                                        size="sm",
                                        color="info",
                                        outline=True,
                                    ),
                                    dbc.Button(
                                        [html.I(className="fas fa-stop me-2"), "Stop"],
                                        size="sm",
                                        color="danger",
                                        outline=True,
                                    ),
                                ],
                                className="mt-2",
                            ),
                        ]
                    ),
                ],
                className="mb-3",
            )

            jobs.append(job_card)

        return html.Div(jobs)

    @callback(
        Output("training-queue-container", "children"),
        Input("training-jobs-store", "data"),
    )
    def update_training_queue(jobs_data):
        """Update training queue display"""
        if not jobs_data or not jobs_data.get("queued"):
            return dbc.Alert(
                [html.I(className="fas fa-info-circle me-2"), "No jobs in queue"],
                color="info",
            )

        queue_items = []
        for job in jobs_data["queued"]:
            queue_items.append(
                dbc.ListGroupItem(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Strong(
                                            f"#{job['position']}: {job['model']}"
                                        )
                                    ],
                                    width=6,
                                ),
                                dbc.Col(
                                    [
                                        html.Span(
                                            f"Submitted: {datetime.fromisoformat(job['submitted']).strftime('%H:%M:%S')}",
                                            className="text-muted",
                                        )
                                    ],
                                    width=4,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            "Cancel",
                                            size="sm",
                                            color="danger",
                                            outline=True,
                                        )
                                    ],
                                    width=2,
                                    className="text-end",
                                ),
                            ]
                        )
                    ]
                )
            )

        return dbc.ListGroup(queue_items)

    @callback(
        Output("training-history-table", "children"),
        [
            Input("training-status-filter", "value"),
            Input("training-model-filter", "value"),
            Input("training-time-range", "value"),
        ],
    )
    def update_training_history(status_filter, model_filter, time_range):
        """Update training history table"""
        # Mock history data
        history_data = [
            {
                "job_id": "job_20251002_001",
                "model": "LSTM Predictor",
                "status": "Completed",
                "started": "2025-10-02 08:30",
                "duration": "1h 25m",
                "final_loss": "0.0185",
                "best_metric": "0.9234",
            },
            {
                "job_id": "job_20251002_002",
                "model": "Transformer",
                "status": "Completed",
                "started": "2025-10-02 10:15",
                "duration": "2h 10m",
                "final_loss": "0.0142",
                "best_metric": "0.9456",
            },
            {
                "job_id": "job_20251001_015",
                "model": "LSTM VAE",
                "status": "Failed",
                "started": "2025-10-01 14:20",
                "duration": "15m",
                "final_loss": "N/A",
                "best_metric": "N/A",
            },
        ]

        table_header = [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Job ID"),
                        html.Th("Model"),
                        html.Th("Status"),
                        html.Th("Started"),
                        html.Th("Duration"),
                        html.Th("Final Loss"),
                        html.Th("Best Metric"),
                        html.Th("Actions"),
                    ]
                )
            )
        ]

        rows = []
        for job in history_data:
            status_color = {
                "Completed": "success",
                "Failed": "danger",
                "Cancelled": "secondary",
            }.get(job["status"], "info")

            rows.append(
                html.Tr(
                    [
                        html.Td(html.Code(job["job_id"])),
                        html.Td(job["model"]),
                        html.Td(dbc.Badge(job["status"], color=status_color)),
                        html.Td(job["started"]),
                        html.Td(job["duration"]),
                        html.Td(job["final_loss"]),
                        html.Td(job["best_metric"]),
                        html.Td(
                            [
                                dbc.ButtonGroup(
                                    [
                                        dbc.Button(
                                            "Details",
                                            size="sm",
                                            color="info",
                                            outline=True,
                                        ),
                                        dbc.Button(
                                            "Logs",
                                            size="sm",
                                            color="secondary",
                                            outline=True,
                                        ),
                                    ]
                                )
                            ]
                        ),
                    ]
                )
            )

        table_body = [html.Tbody(rows)]

        return dbc.Table(
            table_header + table_body,
            bordered=True,
            hover=True,
            responsive=True,
            striped=True,
            className="mt-3",
        )

    @callback(
        [
            Output("training-cpu-usage-chart", "figure"),
            Output("training-gpu-usage-chart", "figure"),
            Output("training-memory-usage-chart", "figure"),
            Output("training-disk-usage-chart", "figure"),
        ],
        Input("training-refresh-interval", "n_intervals"),
    )
    def update_resource_charts(n_intervals):
        """Update resource utilization charts"""
        # Generate mock time-series data
        times = pd.date_range(end=datetime.now(), periods=60, freq="10S")

        # CPU Usage
        cpu_fig = go.Figure()
        cpu_fig.add_trace(
            go.Scatter(
                x=times,
                y=[40 + (i % 20) for i in range(60)],
                name="CPU Usage",
                fill="tozeroy",
                line=dict(color="#3498db"),
            )
        )
        cpu_fig.update_layout(
            title="CPU Usage (%)",
            yaxis=dict(range=[0, 100]),
            height=250,
            margin=dict(l=40, r=20, t=40, b=30),
        )

        # GPU Usage
        gpu_fig = go.Figure()
        gpu_fig.add_trace(
            go.Scatter(
                x=times,
                y=[75 + (i % 15) for i in range(60)],
                name="GPU Usage",
                fill="tozeroy",
                line=dict(color="#e74c3c"),
            )
        )
        gpu_fig.update_layout(
            title="GPU Usage (%)",
            yaxis=dict(range=[0, 100]),
            height=250,
            margin=dict(l=40, r=20, t=40, b=30),
        )

        # Memory Usage
        mem_fig = go.Figure()
        mem_fig.add_trace(
            go.Scatter(
                x=times,
                y=[60 + (i % 10) for i in range(60)],
                name="Memory Usage",
                fill="tozeroy",
                line=dict(color="#2ecc71"),
            )
        )
        mem_fig.update_layout(
            title="Memory Usage (%)",
            yaxis=dict(range=[0, 100]),
            height=250,
            margin=dict(l=40, r=20, t=40, b=30),
        )

        # Disk I/O
        disk_fig = go.Figure()
        disk_fig.add_trace(
            go.Scatter(
                x=times,
                y=[30 + (i % 25) for i in range(60)],
                name="Disk I/O",
                fill="tozeroy",
                line=dict(color="#f39c12"),
            )
        )
        disk_fig.update_layout(
            title="Disk I/O (MB/s)", height=250, margin=dict(l=40, r=20, t=40, b=30)
        )

        return cpu_fig, gpu_fig, mem_fig, disk_fig


def create_mini_loss_curve(job_id: str) -> go.Figure:
    """Create mini loss curve for training job"""
    # Generate sample loss curve
    epochs = list(range(1, 66))
    losses = [0.5 * (0.95**i) + 0.02 for i in epochs]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=losses,
            mode="lines",
            line=dict(color="#3498db", width=2),
            name="Training Loss",
        )
    )

    fig.update_layout(
        title="Training Loss Curve",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        showlegend=False,
        margin=dict(l=40, r=20, t=40, b=30),
        hovermode="x unified",
    )

    return fig

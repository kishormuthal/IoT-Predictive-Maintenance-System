"""
Advanced Anomaly Investigation Dashboard
Deep dive into detected anomalies with root cause analysis and correlation
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback, ctx, dcc, html
from plotly.subplots import make_subplots

# FIXED: Import integration service for real data
try:
    from src.presentation.dashboard.services.dashboard_integration import (
        get_integration_service,
    )

    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False


logger = logging.getLogger(__name__)


def create_anomaly_investigation_layout() -> html.Div:
    """
    Create advanced anomaly investigation dashboard

    Features:
    - Anomaly timeline with interactive selection
    - Multi-sensor correlation analysis
    - Root cause analysis with contributing factors
    - Anomaly pattern clustering
    - Historical similar anomalies
    - Export investigation reports
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
                                    html.I(className="fas fa-search me-2"),
                                    "Advanced Anomaly Investigation",
                                ],
                                className="mb-0",
                            ),
                            html.P(
                                "Deep dive into anomalies with root cause analysis and correlation",
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
                                            html.I(className="fas fa-download me-2"),
                                            "Export Report",
                                        ],
                                        id="export-investigation-btn",
                                        color="info",
                                        size="sm",
                                    ),
                                    dbc.Button(
                                        [
                                            html.I(className="fas fa-sync me-2"),
                                            "Refresh",
                                        ],
                                        id="investigation-refresh-btn",
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
            # Search and Filter Section
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Label(
                                                "Equipment/Sensor:", className="fw-bold"
                                            ),
                                            dcc.Dropdown(
                                                id="investigation-sensor-selector",
                                                options=[
                                                    {
                                                        "label": "All Sensors",
                                                        "value": "all",
                                                    },
                                                    {
                                                        "label": "SMAP - Power System",
                                                        "value": "smap_power",
                                                    },
                                                    {
                                                        "label": "SMAP - Communication",
                                                        "value": "smap_comm",
                                                    },
                                                    {
                                                        "label": "MSL - Mobility Front",
                                                        "value": "msl_mobility_f",
                                                    },
                                                    {
                                                        "label": "MSL - Power System",
                                                        "value": "msl_power",
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
                                                "Severity:", className="fw-bold"
                                            ),
                                            dcc.Dropdown(
                                                id="investigation-severity-filter",
                                                options=[
                                                    {
                                                        "label": "All Severities",
                                                        "value": "all",
                                                    },
                                                    {
                                                        "label": "Critical",
                                                        "value": "critical",
                                                    },
                                                    {"label": "High", "value": "high"},
                                                    {
                                                        "label": "Medium",
                                                        "value": "medium",
                                                    },
                                                    {"label": "Low", "value": "low"},
                                                ],
                                                value="all",
                                            ),
                                        ],
                                        width=3,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Label(
                                                "Time Range:", className="fw-bold"
                                            ),
                                            dcc.Dropdown(
                                                id="investigation-time-range",
                                                options=[
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
                                            ),
                                        ],
                                        width=3,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Label(
                                                "Anomaly Type:", className="fw-bold"
                                            ),
                                            dcc.Dropdown(
                                                id="investigation-type-filter",
                                                options=[
                                                    {
                                                        "label": "All Types",
                                                        "value": "all",
                                                    },
                                                    {
                                                        "label": "Point Anomaly",
                                                        "value": "point",
                                                    },
                                                    {
                                                        "label": "Contextual Anomaly",
                                                        "value": "contextual",
                                                    },
                                                    {
                                                        "label": "Collective Anomaly",
                                                        "value": "collective",
                                                    },
                                                ],
                                                value="all",
                                            ),
                                        ],
                                        width=3,
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
                                                        className="fas fa-filter me-2"
                                                    ),
                                                    "Apply Filters",
                                                ],
                                                id="apply-investigation-filters-btn",
                                                color="primary",
                                                className="me-2",
                                            ),
                                            dbc.Button(
                                                [
                                                    html.I(
                                                        className="fas fa-undo me-2"
                                                    ),
                                                    "Reset",
                                                ],
                                                id="reset-investigation-filters-btn",
                                                color="secondary",
                                                outline=True,
                                            ),
                                        ]
                                    )
                                ]
                            ),
                        ]
                    )
                ],
                className="mb-4",
            ),
            # Anomaly Timeline
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            html.H5(
                                [
                                    html.I(className="fas fa-chart-line me-2"),
                                    "Anomaly Timeline (Click to Investigate)",
                                ],
                                className="mb-0",
                            )
                        ]
                    ),
                    dbc.CardBody(
                        [
                            dcc.Graph(
                                id="anomaly-timeline-chart",
                                config={"displayModeBar": True},
                            )
                        ]
                    ),
                ],
                className="mb-4",
            ),
            # Selected Anomaly Details
            html.Div(
                id="selected-anomaly-container",
                children=[
                    dbc.Alert(
                        [
                            html.I(className="fas fa-info-circle me-2"),
                            "Click on an anomaly in the timeline above to start investigation",
                        ],
                        color="info",
                    )
                ],
            ),
            # Auto-refresh interval
            dcc.Interval(
                id="investigation-refresh-interval",
                interval=30000,  # 30 seconds
                n_intervals=0,
            ),
            # Store for anomaly data
            dcc.Store(id="investigation-data-store"),
            dcc.Store(id="selected-anomaly-store"),
        ],
        className="p-4",
    )


def create_anomaly_detail_view(anomaly_data: Dict) -> html.Div:
    """Create detailed view for selected anomaly"""

    return html.Div(
        [
            # Header with anomaly info
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H4(
                                                [
                                                    html.I(
                                                        className="fas fa-exclamation-triangle me-2"
                                                    ),
                                                    f"Anomaly Investigation: {anomaly_data['id']}",
                                                ],
                                                className="mb-0",
                                            )
                                        ],
                                        width=8,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Badge(
                                                anomaly_data["severity"].upper(),
                                                color=get_severity_color(
                                                    anomaly_data["severity"]
                                                ),
                                                className="me-2",
                                            ),
                                            dbc.Badge(
                                                anomaly_data["type"], color="info"
                                            ),
                                        ],
                                        width=4,
                                        className="text-end",
                                    ),
                                ]
                            )
                        ]
                    ),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.P(
                                                [
                                                    html.Strong("Sensor: "),
                                                    anomaly_data["sensor"],
                                                ],
                                                className="mb-1",
                                            ),
                                            html.P(
                                                [
                                                    html.Strong("Detected: "),
                                                    anomaly_data["timestamp"],
                                                ],
                                                className="mb-1",
                                            ),
                                        ],
                                        width=4,
                                    ),
                                    dbc.Col(
                                        [
                                            html.P(
                                                [
                                                    html.Strong("Anomaly Score: "),
                                                    f"{anomaly_data['score']:.3f}",
                                                ],
                                                className="mb-1",
                                            ),
                                            html.P(
                                                [
                                                    html.Strong("Confidence: "),
                                                    f"{anomaly_data['confidence']*100:.1f}%",
                                                ],
                                                className="mb-1",
                                            ),
                                        ],
                                        width=4,
                                    ),
                                    dbc.Col(
                                        [
                                            html.P(
                                                [
                                                    html.Strong("Expected Range: "),
                                                    f"{anomaly_data['expected_min']:.2f} - {anomaly_data['expected_max']:.2f}",
                                                ],
                                                className="mb-1",
                                            ),
                                            html.P(
                                                [
                                                    html.Strong("Actual Value: "),
                                                    f"{anomaly_data['actual_value']:.2f}",
                                                ],
                                                className="mb-1",
                                            ),
                                        ],
                                        width=4,
                                    ),
                                ]
                            )
                        ]
                    ),
                ],
                className="mb-4",
            ),
            # Tabs for detailed analysis
            dbc.Tabs(
                [
                    # Tab 1: Sensor Data Analysis
                    dbc.Tab(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    # Time series with context
                                                    dbc.Col(
                                                        [
                                                            dcc.Graph(
                                                                id="anomaly-detail-timeseries",
                                                                figure=create_anomaly_timeseries(
                                                                    anomaly_data
                                                                ),
                                                            )
                                                        ],
                                                        width=12,
                                                    )
                                                ],
                                                className="mb-3",
                                            ),
                                            dbc.Row(
                                                [
                                                    # Statistical breakdown
                                                    dbc.Col(
                                                        [
                                                            html.H6(
                                                                "Statistical Analysis",
                                                                className="mb-3",
                                                            ),
                                                            dcc.Graph(
                                                                id="anomaly-statistical-breakdown",
                                                                figure=create_statistical_breakdown(
                                                                    anomaly_data
                                                                ),
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                    # Probability distribution
                                                    dbc.Col(
                                                        [
                                                            html.H6(
                                                                "Probability Distribution",
                                                                className="mb-3",
                                                            ),
                                                            dcc.Graph(
                                                                id="anomaly-probability-dist",
                                                                figure=create_probability_distribution(
                                                                    anomaly_data
                                                                ),
                                                            ),
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
                        label="Sensor Analysis",
                        tab_id="sensor-analysis",
                    ),
                    # Tab 2: Root Cause Analysis
                    dbc.Tab(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5(
                                                "Contributing Factors", className="mb-3"
                                            ),
                                            # Factor importance chart
                                            dcc.Graph(
                                                id="root-cause-factors",
                                                figure=create_root_cause_chart(
                                                    anomaly_data
                                                ),
                                            ),
                                            html.Hr(),
                                            # Detailed factors
                                            html.H6(
                                                "Identified Root Causes:",
                                                className="mb-3",
                                            ),
                                            html.Div(
                                                id="root-cause-details",
                                                children=create_root_cause_details(
                                                    anomaly_data
                                                ),
                                            ),
                                        ]
                                    )
                                ]
                            )
                        ],
                        label="Root Cause Analysis",
                        tab_id="root-cause",
                    ),
                    # Tab 3: Correlation Analysis
                    dbc.Tab(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5(
                                                "Multi-Sensor Correlation",
                                                className="mb-3",
                                            ),
                                            # Correlation heatmap
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dcc.Graph(
                                                                id="correlation-heatmap",
                                                                figure=create_correlation_heatmap(
                                                                    anomaly_data
                                                                ),
                                                            )
                                                        ],
                                                        width=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.H6(
                                                                "Correlated Sensors",
                                                                className="mb-3",
                                                            ),
                                                            html.Div(
                                                                id="correlated-sensors",
                                                                children=create_correlated_sensors_list(
                                                                    anomaly_data
                                                                ),
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            html.Hr(),
                                            # Time-lagged correlation
                                            html.H6(
                                                "Time-Lagged Cross-Correlation",
                                                className="mb-3",
                                            ),
                                            dcc.Graph(
                                                id="cross-correlation-chart",
                                                figure=create_cross_correlation_chart(
                                                    anomaly_data
                                                ),
                                            ),
                                        ]
                                    )
                                ]
                            )
                        ],
                        label="Correlation Analysis",
                        tab_id="correlation",
                    ),
                    # Tab 4: Similar Anomalies
                    dbc.Tab(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5(
                                                "Historical Similar Anomalies",
                                                className="mb-3",
                                            ),
                                            # Similarity scores
                                            html.Div(
                                                id="similar-anomalies",
                                                children=create_similar_anomalies_list(
                                                    anomaly_data
                                                ),
                                            ),
                                        ]
                                    )
                                ]
                            )
                        ],
                        label="Similar Anomalies",
                        tab_id="similar",
                    ),
                    # Tab 5: Recommendations
                    dbc.Tab(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5(
                                                "Recommended Actions", className="mb-3"
                                            ),
                                            dbc.ListGroup(
                                                [
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.I(
                                                                        className="fas fa-wrench fa-2x text-warning mb-2"
                                                                    ),
                                                                    html.H6(
                                                                        "Immediate Action Required",
                                                                        className="mt-2",
                                                                    ),
                                                                    html.P(
                                                                        "Schedule maintenance inspection for power system within 4 hours"
                                                                    ),
                                                                    dbc.Button(
                                                                        "Create Work Order",
                                                                        color="warning",
                                                                        size="sm",
                                                                    ),
                                                                ]
                                                            )
                                                        ]
                                                    ),
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.I(
                                                                        className="fas fa-eye fa-2x text-info mb-2"
                                                                    ),
                                                                    html.H6(
                                                                        "Monitor Closely",
                                                                        className="mt-2",
                                                                    ),
                                                                    html.P(
                                                                        "Set up enhanced monitoring for correlated sensors (Communication, Thermal)"
                                                                    ),
                                                                    dbc.Button(
                                                                        "Enable Enhanced Monitoring",
                                                                        color="info",
                                                                        size="sm",
                                                                        outline=True,
                                                                    ),
                                                                ]
                                                            )
                                                        ]
                                                    ),
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.I(
                                                                        className="fas fa-bell fa-2x text-primary mb-2"
                                                                    ),
                                                                    html.H6(
                                                                        "Alert Configuration",
                                                                        className="mt-2",
                                                                    ),
                                                                    html.P(
                                                                        "Configure predictive alerts for similar patterns"
                                                                    ),
                                                                    dbc.Button(
                                                                        "Configure Alerts",
                                                                        color="primary",
                                                                        size="sm",
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
                            )
                        ],
                        label="Recommendations",
                        tab_id="recommendations",
                    ),
                ],
                id="anomaly-detail-tabs",
                active_tab="sensor-analysis",
            ),
        ]
    )


def register_anomaly_investigation_callbacks(app: dash.Dash):
    """Register callbacks for anomaly investigation"""

    @callback(
        [
            Output("investigation-data-store", "data"),
            Output("anomaly-timeline-chart", "figure"),
        ],
        [
            Input("investigation-refresh-interval", "n_intervals"),
            Input("apply-investigation-filters-btn", "n_clicks"),
        ],
        [
            State("investigation-sensor-selector", "value"),
            State("investigation-severity-filter", "value"),
            State("investigation-time-range", "value"),
            State("investigation-type-filter", "value"),
        ],
    )
    def update_anomaly_timeline(
        n_intervals, n_clicks, sensor, severity, time_range, anom_type
    ):
        """Update anomaly timeline based on filters"""

        # Generate mock anomaly data
        hours = {"1h": 1, "6h": 6, "24h": 24, "7d": 168, "30d": 720}.get(time_range, 24)
        timestamps = pd.date_range(end=datetime.now(), periods=hours * 12, freq="5min")

        # Random anomalies
        np.random.seed(42)
        anomaly_indices = np.random.choice(
            len(timestamps), size=int(len(timestamps) * 0.05), replace=False
        )

        anomalies = []
        for idx in sorted(anomaly_indices):
            anomalies.append(
                {
                    "id": f"ANO_{timestamps[idx].strftime('%Y%m%d_%H%M%S')}",
                    "timestamp": timestamps[idx].isoformat(),
                    "sensor": np.random.choice(
                        ["SMAP_PWR_01", "SMAP_COM_03", "MSL_MOB_F_05", "MSL_PWR_02"]
                    ),
                    "severity": np.random.choice(
                        ["critical", "high", "medium", "low"], p=[0.1, 0.3, 0.4, 0.2]
                    ),
                    "score": np.random.uniform(0.7, 1.0),
                    "type": np.random.choice(
                        ["point", "contextual", "collective"], p=[0.6, 0.3, 0.1]
                    ),
                    "value": np.random.uniform(50, 150),
                }
            )

        # Create timeline figure
        fig = go.Figure()

        # Add trace for each severity
        for sev in ["critical", "high", "medium", "low"]:
            sev_anomalies = [a for a in anomalies if a["severity"] == sev]
            if sev_anomalies:
                fig.add_trace(
                    go.Scatter(
                        x=[a["timestamp"] for a in sev_anomalies],
                        y=[a["score"] for a in sev_anomalies],
                        mode="markers",
                        name=sev.capitalize(),
                        marker=dict(
                            size=10,
                            color=get_severity_color(sev),
                            line=dict(width=1, color="white"),
                        ),
                        text=[
                            f"ID: {a['id']}<br>Sensor: {a['sensor']}<br>Score: {a['score']:.3f}"
                            for a in sev_anomalies
                        ],
                        hovertemplate="%{text}<extra></extra>",
                        customdata=[a["id"] for a in sev_anomalies],
                    )
                )

        fig.update_layout(
            title="Anomaly Timeline (Click points to investigate)",
            xaxis_title="Time",
            yaxis_title="Anomaly Score",
            hovermode="closest",
            height=400,
            clickmode="event+select",
        )

        return {"anomalies": anomalies}, fig

    @callback(
        [
            Output("selected-anomaly-container", "children"),
            Output("selected-anomaly-store", "data"),
        ],
        Input("anomaly-timeline-chart", "clickData"),
        State("investigation-data-store", "data"),
    )
    def display_anomaly_details(clickData, data):
        """Display detailed analysis when anomaly is clicked"""
        if not clickData or not data:
            return (
                dbc.Alert(
                    [
                        html.I(className="fas fa-info-circle me-2"),
                        "Click on an anomaly in the timeline above to start investigation",
                    ],
                    color="info",
                ),
                None,
            )

        # Get clicked anomaly ID
        anomaly_id = clickData["points"][0]["customdata"]

        # Find anomaly in data
        anomaly = next((a for a in data["anomalies"] if a["id"] == anomaly_id), None)

        if not anomaly:
            return dbc.Alert("Anomaly not found", color="warning"), None

        # Enrich anomaly data with investigation details
        enriched_anomaly = {
            **anomaly,
            "confidence": np.random.uniform(0.8, 0.99),
            "expected_min": anomaly["value"] * 0.7,
            "expected_max": anomaly["value"] * 0.9,
            "actual_value": anomaly["value"],
        }

        return create_anomaly_detail_view(enriched_anomaly), enriched_anomaly


def get_severity_color(severity: str) -> str:
    """Get color for severity level"""
    colors = {
        "critical": "danger",
        "high": "warning",
        "medium": "info",
        "low": "secondary",
    }
    return colors.get(severity, "secondary")


def create_anomaly_timeseries(anomaly_data: Dict) -> go.Figure:
    """Create time series chart with anomaly context"""
    # Generate context data around anomaly
    times = pd.date_range(
        end=pd.to_datetime(anomaly_data["timestamp"]), periods=100, freq="1min"
    )

    # Normal data with anomaly spike
    np.random.seed(42)
    values = np.random.normal(50, 5, 100)
    values[-10] = anomaly_data["actual_value"]  # Insert anomaly

    fig = go.Figure()

    # Normal data
    fig.add_trace(
        go.Scatter(
            x=times,
            y=values,
            mode="lines",
            name="Sensor Reading",
            line=dict(color="#3498db", width=2),
        )
    )

    # Expected range
    fig.add_trace(
        go.Scatter(
            x=times,
            y=[anomaly_data["expected_max"]] * len(times),
            mode="lines",
            name="Upper Bound",
            line=dict(color="#2ecc71", width=1, dash="dash"),
            showlegend=True,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=[anomaly_data["expected_min"]] * len(times),
            mode="lines",
            name="Lower Bound",
            line=dict(color="#2ecc71", width=1, dash="dash"),
            fill="tonexty",
            fillcolor="rgba(46, 204, 113, 0.1)",
            showlegend=True,
        )
    )

    # Highlight anomaly
    fig.add_trace(
        go.Scatter(
            x=[times[-10]],
            y=[values[-10]],
            mode="markers",
            name="Anomaly",
            marker=dict(size=15, color="#e74c3c", symbol="x"),
        )
    )

    fig.update_layout(
        title="Sensor Reading with Context",
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode="x unified",
        height=400,
    )

    return fig


def create_statistical_breakdown(anomaly_data: Dict) -> go.Figure:
    """Create statistical breakdown chart"""
    categories = ["Mean", "Median", "Std Dev", "Min", "Max"]
    normal_values = [50, 49, 5, 40, 60]
    anomaly_window = [
        anomaly_data["actual_value"],
        anomaly_data["actual_value"],
        8,
        45,
        anomaly_data["actual_value"],
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Normal Period", x=categories, y=normal_values, marker_color="#3498db"
        )
    )
    fig.add_trace(
        go.Bar(
            name="Anomaly Window",
            x=categories,
            y=anomaly_window,
            marker_color="#e74c3c",
        )
    )

    fig.update_layout(title="Statistical Comparison", barmode="group", height=300)

    return fig


def create_probability_distribution(anomaly_data: Dict) -> go.Figure:
    """Create probability distribution chart"""
    x = np.linspace(20, 100, 200)
    normal_dist = np.exp(-0.5 * ((x - 50) / 5) ** 2)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=normal_dist,
            fill="tozeroy",
            name="Normal Distribution",
            line=dict(color="#3498db"),
        )
    )

    # Mark anomaly
    fig.add_vline(
        x=anomaly_data["actual_value"],
        line_dash="dash",
        line_color="#e74c3c",
        annotation_text="Anomaly",
        annotation_position="top",
    )

    fig.update_layout(
        title="Probability Distribution",
        xaxis_title="Sensor Value",
        yaxis_title="Probability Density",
        showlegend=False,
        height=300,
    )

    return fig


def create_root_cause_chart(anomaly_data: Dict) -> go.Figure:
    """Create root cause contributing factors chart"""
    factors = [
        "Sensor Drift",
        "External Temperature",
        "Power Fluctuation",
        "Communication Loss",
        "Equipment Age",
    ]
    contributions = [0.85, 0.72, 0.65, 0.45, 0.32]

    fig = go.Figure(
        go.Bar(
            x=contributions,
            y=factors,
            orientation="h",
            marker=dict(
                color=contributions,
                colorscale="RdYlGn_r",
                showscale=True,
                colorbar=dict(title="Contribution"),
            ),
        )
    )

    fig.update_layout(
        title="Root Cause Contributing Factors",
        xaxis_title="Contribution Score",
        height=350,
    )

    return fig


def create_root_cause_details(anomaly_data: Dict) -> html.Div:
    """Create detailed root cause analysis"""
    causes = [
        {
            "factor": "Sensor Drift",
            "contribution": 0.85,
            "description": "Sensor calibration has drifted 12% from baseline over past 14 days",
            "evidence": "Gradual increase in baseline readings, correlation with temperature changes",
            "action": "Recalibrate sensor during next maintenance window",
        },
        {
            "factor": "External Temperature Spike",
            "contribution": 0.72,
            "description": "Ambient temperature increased by 15°C in 30 minutes",
            "evidence": "Correlated with environmental sensor readings",
            "action": "Verify cooling system performance",
        },
        {
            "factor": "Power Fluctuation",
            "contribution": 0.65,
            "description": "Voltage drop detected 2 minutes before anomaly",
            "evidence": "Power system logs show 8% voltage reduction",
            "action": "Inspect power distribution unit",
        },
    ]

    cards = []
    for cause in causes:
        cards.append(
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            html.H6(
                                [
                                    dbc.Badge(
                                        f"{cause['contribution']*100:.0f}%",
                                        color="warning",
                                        className="me-2",
                                    ),
                                    cause["factor"],
                                ]
                            ),
                            html.P(cause["description"], className="mb-2"),
                            html.P(
                                [html.Strong("Evidence: "), cause["evidence"]],
                                className="mb-2 small",
                            ),
                            html.P(
                                [html.Strong("Recommended Action: "), cause["action"]],
                                className="mb-0 small text-primary",
                            ),
                        ]
                    )
                ],
                className="mb-2",
            )
        )

    return html.Div(cards)


def create_correlation_heatmap(anomaly_data: Dict) -> go.Figure:
    """Create sensor correlation heatmap"""
    sensors = ["PWR_01", "COM_03", "THM_02", "MOB_05", "NAV_01"]
    correlation_matrix = np.random.rand(5, 5)
    np.fill_diagonal(correlation_matrix, 1.0)
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2

    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix,
            x=sensors,
            y=sensors,
            colorscale="RdBu",
            zmid=0,
            text=correlation_matrix,
            texttemplate="%{text:.2f}",
            textfont={"size": 10},
        )
    )

    fig.update_layout(title="Sensor Correlation Matrix", height=400)

    return fig


def create_correlated_sensors_list(anomaly_data: Dict) -> html.Div:
    """Create list of correlated sensors"""
    sensors = [
        {"name": "COM_03", "correlation": 0.87, "lag": "2 min"},
        {"name": "THM_02", "correlation": 0.75, "lag": "5 min"},
        {"name": "MOB_05", "correlation": 0.62, "lag": "1 min"},
    ]

    items = []
    for sensor in sensors:
        items.append(
            dbc.ListGroupItem(
                [
                    html.Div(
                        [
                            html.Strong(sensor["name"]),
                            dbc.Badge(
                                f"r = {sensor['correlation']:.2f}",
                                color="info",
                                className="float-end",
                            ),
                        ]
                    ),
                    html.Small(f"Time lag: {sensor['lag']}", className="text-muted"),
                ]
            )
        )

    return dbc.ListGroup(items)


def create_cross_correlation_chart(anomaly_data: Dict) -> go.Figure:
    """Create cross-correlation chart"""
    lags = np.arange(-20, 21)
    xcorr = np.exp(-0.5 * (lags / 5) ** 2)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=lags,
            y=xcorr,
            marker_color=["#e74c3c" if abs(x) < 3 else "#3498db" for x in lags],
        )
    )

    fig.update_layout(
        title="Cross-Correlation with COM_03 Sensor",
        xaxis_title="Time Lag (minutes)",
        yaxis_title="Correlation Coefficient",
        height=300,
    )

    return fig


def create_similar_anomalies_list(anomaly_data: Dict) -> html.Div:
    """Create list of similar historical anomalies"""
    similar = [
        {
            "id": "ANO_20250925_143022",
            "date": "2025-09-25 14:30",
            "similarity": 0.92,
            "resolution": "Sensor recalibration",
            "outcome": "Resolved - No recurrence",
        },
        {
            "id": "ANO_20250918_091544",
            "date": "2025-09-18 09:15",
            "similarity": 0.88,
            "resolution": "Cooling system maintenance",
            "outcome": "Resolved - Recurred once",
        },
        {
            "id": "ANO_20250910_165533",
            "date": "2025-09-10 16:55",
            "similarity": 0.85,
            "resolution": "Power supply replacement",
            "outcome": "Resolved - No recurrence",
        },
    ]

    cards = []
    for anom in similar:
        cards.append(
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.P(
                                                [
                                                    html.Strong("ID: "),
                                                    html.Code(anom["id"]),
                                                ],
                                                className="mb-1",
                                            ),
                                            html.P(
                                                [html.Strong("Date: "), anom["date"]],
                                                className="mb-1",
                                            ),
                                        ],
                                        width=6,
                                    ),
                                    dbc.Col(
                                        [
                                            html.P(
                                                [
                                                    html.Strong("Similarity: "),
                                                    dbc.Badge(
                                                        f"{anom['similarity']*100:.0f}%",
                                                        color="success",
                                                    ),
                                                ],
                                                className="mb-1",
                                            ),
                                            dbc.Button(
                                                "View Details",
                                                size="sm",
                                                color="info",
                                                outline=True,
                                            ),
                                        ],
                                        width=6,
                                        className="text-end",
                                    ),
                                ]
                            ),
                            html.Hr(className="my-2"),
                            html.P(
                                [html.Strong("Resolution: "), anom["resolution"]],
                                className="mb-1 small",
                            ),
                            html.P(
                                [html.Strong("Outcome: "), anom["outcome"]],
                                className="mb-0 small text-success",
                            ),
                        ]
                    )
                ],
                className="mb-2",
            )
        )

    return html.Div(cards)

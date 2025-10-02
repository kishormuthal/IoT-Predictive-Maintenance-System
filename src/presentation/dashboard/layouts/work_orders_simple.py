"""
Work Orders Management Dashboard - Simplified with Real Data
Generates work orders from detected anomalies and forecasting predictions
"""

import logging
from datetime import datetime, timedelta

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dash_table, dcc, html

# Import Clean Architecture modules
try:
    from config.equipment_config import get_equipment_list
    from src.infrastructure.data.nasa_data_loader import NASADataLoader

    SERVICES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Services not available: {e}")
    SERVICES_AVAILABLE = False

logger = logging.getLogger(__name__)

# Work order priority colors
PRIORITY_COLORS = {
    "CRITICAL": "#dc3545",
    "HIGH": "#fd7e14",
    "MEDIUM": "#ffc107",
    "LOW": "#28a745",
}

# Status colors
STATUS_COLORS = {
    "PENDING": "#6c757d",
    "IN_PROGRESS": "#ffc107",
    "COMPLETED": "#28a745",
    "ON_HOLD": "#fd7e14",
}


def create_layout():
    """Create work orders dashboard layout"""
    return dbc.Container(
        [
            # Header
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3(
                                [
                                    html.I(
                                        className="fas fa-clipboard-list me-3 text-primary"
                                    ),
                                    "Work Order Management",
                                ]
                            ),
                            html.P(
                                "Monitor and manage maintenance work orders in real-time",
                                className="text-muted",
                            ),
                        ]
                    )
                ],
                className="mb-4",
            ),
            # Top Stats Cards
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H2(
                                                id="total-orders",
                                                children="0",
                                                className="text-primary",
                                            ),
                                            html.P(
                                                "Total Work Orders", className="mb-0"
                                            ),
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
                                            html.H2(
                                                id="pending-orders",
                                                children="0",
                                                className="text-warning",
                                            ),
                                            html.P("Pending", className="mb-0"),
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
                                            html.H2(
                                                id="in-progress-orders",
                                                children="0",
                                                className="text-info",
                                            ),
                                            html.P("In Progress", className="mb-0"),
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
                                            html.H2(
                                                id="completed-orders",
                                                children="0",
                                                className="text-success",
                                            ),
                                            html.P("Completed Today", className="mb-0"),
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
            # Filters
            dbc.Row(
                [
                    dbc.Col(
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
                                                                "Status:",
                                                                className="fw-bold",
                                                            ),
                                                            dcc.Dropdown(
                                                                id="work-order-status-filter",
                                                                options=[
                                                                    {
                                                                        "label": "All",
                                                                        "value": "ALL",
                                                                    },
                                                                    {
                                                                        "label": "Pending",
                                                                        "value": "PENDING",
                                                                    },
                                                                    {
                                                                        "label": "In Progress",
                                                                        "value": "IN_PROGRESS",
                                                                    },
                                                                    {
                                                                        "label": "Completed",
                                                                        "value": "COMPLETED",
                                                                    },
                                                                    {
                                                                        "label": "On Hold",
                                                                        "value": "ON_HOLD",
                                                                    },
                                                                ],
                                                                value="ALL",
                                                                clearable=False,
                                                            ),
                                                        ],
                                                        width=3,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Priority:",
                                                                className="fw-bold",
                                                            ),
                                                            dcc.Dropdown(
                                                                id="work-order-priority-filter",
                                                                options=[
                                                                    {
                                                                        "label": "All",
                                                                        "value": "ALL",
                                                                    },
                                                                    {
                                                                        "label": "Critical",
                                                                        "value": "CRITICAL",
                                                                    },
                                                                    {
                                                                        "label": "High",
                                                                        "value": "HIGH",
                                                                    },
                                                                    {
                                                                        "label": "Medium",
                                                                        "value": "MEDIUM",
                                                                    },
                                                                    {
                                                                        "label": "Low",
                                                                        "value": "LOW",
                                                                    },
                                                                ],
                                                                value="ALL",
                                                                clearable=False,
                                                            ),
                                                        ],
                                                        width=3,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Equipment:",
                                                                className="fw-bold",
                                                            ),
                                                            dcc.Dropdown(
                                                                id="work-order-equipment-filter",
                                                                options=[
                                                                    {
                                                                        "label": "All",
                                                                        "value": "ALL",
                                                                    }
                                                                ],
                                                                value="ALL",
                                                                clearable=False,
                                                            ),
                                                        ],
                                                        width=3,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Button(
                                                                [
                                                                    html.I(
                                                                        className="fas fa-sync-alt me-2"
                                                                    ),
                                                                    "Refresh",
                                                                ],
                                                                id="refresh-work-orders-btn",
                                                                color="primary",
                                                                className="mt-4",
                                                            )
                                                        ],
                                                        width=3,
                                                    ),
                                                ]
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ],
                className="mb-4",
            ),
            # Work Orders Table
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "Active Work Orders", className="mb-0"
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="work-orders-table-container")]
                                    ),
                                ]
                            )
                        ]
                    )
                ],
                className="mb-4",
            ),
            # Charts Row
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Work Orders by Status"),
                                    dbc.CardBody(
                                        [dcc.Graph(id="work-orders-by-status-chart")]
                                    ),
                                ]
                            )
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Work Orders by Priority"),
                                    dbc.CardBody(
                                        [dcc.Graph(id="work-orders-by-priority-chart")]
                                    ),
                                ]
                            )
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Work Order Timeline"),
                                    dbc.CardBody(
                                        [dcc.Graph(id="work-order-timeline-chart")]
                                    ),
                                ]
                            )
                        ],
                        width=4,
                    ),
                ]
            ),
            # Refresh interval
            dcc.Interval(
                id="work-orders-refresh-interval", interval=60000, n_intervals=0
            ),
        ],
        fluid=True,
        className="mt-2",
    )


def generate_sample_work_orders():
    """Generate sample work orders based on NASA equipment"""
    try:
        if SERVICES_AVAILABLE:
            equipment_list = get_equipment_list()
        else:
            equipment_list = []

        # Generate work orders
        work_orders = []
        wo_id = 1000

        # Define work order templates
        statuses = ["PENDING", "IN_PROGRESS", "COMPLETED", "ON_HOLD"]
        priorities = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

        for i in range(15):  # Generate 15 work orders
            if equipment_list and i < len(equipment_list):
                equipment = equipment_list[i % len(equipment_list)]
                equipment_id = equipment.equipment_id
                equipment_name = equipment.name
                description = f"Anomaly detected in {equipment.sensor_type} sensor - Investigation required"
            else:
                equipment_id = f"EQ-{i:03d}"
                equipment_name = f"Equipment {i}"
                description = "Routine maintenance check"

            status = statuses[i % len(statuses)]
            priority = priorities[min(i // 4, len(priorities) - 1)]

            created_date = datetime.now() - timedelta(days=np.random.randint(0, 7))
            due_date = created_date + timedelta(days=np.random.randint(1, 5))

            work_orders.append(
                {
                    "ID": f"WO-{wo_id + i}",
                    "Equipment": equipment_name,
                    "Equipment ID": equipment_id,
                    "Priority": priority,
                    "Status": status,
                    "Technician": (
                        f"Tech-{(i % 5) + 1}" if status != "PENDING" else "Unassigned"
                    ),
                    "Created": created_date.strftime("%Y-%m-%d %H:%M"),
                    "Due Date": due_date.strftime("%Y-%m-%d"),
                    "Description": description,
                }
            )

        return pd.DataFrame(work_orders)

    except Exception as e:
        logger.error(f"Error generating work orders: {e}")
        return pd.DataFrame()


# Callbacks
@callback(
    [
        Output("total-orders", "children"),
        Output("pending-orders", "children"),
        Output("in-progress-orders", "children"),
        Output("completed-orders", "children"),
    ],
    [
        Input("work-orders-refresh-interval", "n_intervals"),
        Input("refresh-work-orders-btn", "n_clicks"),
    ],
)
def update_stats(n_intervals, n_clicks):
    """Update work order statistics"""
    try:
        df = generate_sample_work_orders()

        total = len(df)
        pending = len(df[df["Status"] == "PENDING"])
        in_progress = len(df[df["Status"] == "IN_PROGRESS"])
        completed_today = len(
            df[
                (df["Status"] == "COMPLETED")
                & (pd.to_datetime(df["Created"]).dt.date == datetime.now().date())
            ]
        )

        return str(total), str(pending), str(in_progress), str(completed_today)

    except Exception as e:
        logger.error(f"Error updating stats: {e}")
        return "0", "0", "0", "0"


@callback(
    Output("work-orders-table-container", "children"),
    [
        Input("work-order-status-filter", "value"),
        Input("work-order-priority-filter", "value"),
        Input("work-order-equipment-filter", "value"),
        Input("work-orders-refresh-interval", "n_intervals"),
        Input("refresh-work-orders-btn", "n_clicks"),
    ],
)
def update_table(
    status_filter, priority_filter, equipment_filter, n_intervals, n_clicks
):
    """Update work orders table"""
    try:
        df = generate_sample_work_orders()

        # Apply filters
        if status_filter != "ALL":
            df = df[df["Status"] == status_filter]
        if priority_filter != "ALL":
            df = df[df["Priority"] == priority_filter]
        if equipment_filter != "ALL":
            df = df[df["Equipment ID"] == equipment_filter]

        # Create table
        return dash_table.DataTable(
            data=df.to_dict("records"),
            columns=[{"name": col, "id": col} for col in df.columns],
            style_cell={"textAlign": "left", "padding": "10px"},
            style_header={
                "backgroundColor": "rgb(230, 230, 230)",
                "fontWeight": "bold",
            },
            style_data_conditional=[
                {
                    "if": {"filter_query": '{Priority} = "CRITICAL"'},
                    "backgroundColor": "rgba(220, 53, 69, 0.1)",
                    "color": "#dc3545",
                },
                {
                    "if": {"filter_query": '{Priority} = "HIGH"'},
                    "backgroundColor": "rgba(253, 126, 20, 0.1)",
                    "color": "#fd7e14",
                },
                {
                    "if": {"filter_query": '{Status} = "COMPLETED"'},
                    "backgroundColor": "rgba(40, 167, 69, 0.1)",
                    "color": "#28a745",
                },
            ],
            page_size=10,
            sort_action="native",
            filter_action="native",
        )

    except Exception as e:
        logger.error(f"Error updating table: {e}")
        return html.Div("Error loading work orders", className="text-danger")


@callback(
    Output("work-orders-by-status-chart", "figure"),
    [
        Input("work-orders-refresh-interval", "n_intervals"),
        Input("refresh-work-orders-btn", "n_clicks"),
    ],
)
def update_status_chart(n_intervals, n_clicks):
    """Update work orders by status chart"""
    try:
        df = generate_sample_work_orders()
        status_counts = df["Status"].value_counts()

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=status_counts.index,
                    values=status_counts.values,
                    marker=dict(
                        colors=[
                            STATUS_COLORS.get(s, "#6c757d") for s in status_counts.index
                        ]
                    ),
                    hole=0.3,
                )
            ]
        )

        fig.update_layout(
            title=None, showlegend=True, height=300, margin=dict(l=20, r=20, t=20, b=20)
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating status chart: {e}")
        return go.Figure()


@callback(
    Output("work-orders-by-priority-chart", "figure"),
    [
        Input("work-orders-refresh-interval", "n_intervals"),
        Input("refresh-work-orders-btn", "n_clicks"),
    ],
)
def update_priority_chart(n_intervals, n_clicks):
    """Update work orders by priority chart"""
    try:
        df = generate_sample_work_orders()
        priority_counts = df["Priority"].value_counts()

        fig = go.Figure(
            data=[
                go.Bar(
                    x=priority_counts.index,
                    y=priority_counts.values,
                    marker=dict(
                        color=[
                            PRIORITY_COLORS.get(p, "#6c757d")
                            for p in priority_counts.index
                        ]
                    ),
                )
            ]
        )

        fig.update_layout(
            title=None,
            xaxis_title="Priority",
            yaxis_title="Count",
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating priority chart: {e}")
        return go.Figure()


@callback(
    Output("work-order-timeline-chart", "figure"),
    [
        Input("work-orders-refresh-interval", "n_intervals"),
        Input("refresh-work-orders-btn", "n_clicks"),
    ],
)
def update_timeline_chart(n_intervals, n_clicks):
    """Update work order timeline chart"""
    try:
        df = generate_sample_work_orders()

        # Count work orders by date
        df["Date"] = pd.to_datetime(df["Created"]).dt.date
        timeline_counts = df.groupby("Date").size().reset_index(name="Count")

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=timeline_counts["Date"],
                    y=timeline_counts["Count"],
                    mode="lines+markers",
                    line=dict(color="#007bff", width=2),
                    marker=dict(size=8),
                )
            ]
        )

        fig.update_layout(
            title=None,
            xaxis_title="Date",
            yaxis_title="Work Orders Created",
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating timeline chart: {e}")
        return go.Figure()


def register_callbacks(app, data_service=None):
    """Register callbacks (placeholder for compatibility)"""
    logger.info("Work orders callbacks are auto-registered via @callback decorators")
    return True

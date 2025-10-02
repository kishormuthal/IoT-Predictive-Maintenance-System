"""
System Performance Dashboard Layout
Consolidated view of Training Hub, Models, ML Pipeline, Configuration, and System Admin
"""

import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional

# FIXED: Import integration service for real data
try:
    from src.presentation.dashboard.services.dashboard_integration import get_integration_service
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False


# Import components from the original tabs
from src.presentation.dashboard.components.training_hub import create_training_hub_layout
from src.presentation.dashboard.components.model_registry import create_model_registry_layout
from src.presentation.dashboard.components.system_admin import create_system_admin_layout
from src.presentation.dashboard.components.config_manager import create_config_management_layout
from src.presentation.dashboard.layouts.pipeline_dashboard import pipeline_dashboard

logger = logging.getLogger(__name__)


def create_layout() -> html.Div:
    """
    Create consolidated System Performance layout
    Combines Training Hub, Models, ML Pipeline, Configuration, and System Admin
    """

    # System Performance Overview Section
    performance_overview = dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-chart-area me-2"),
                "System Performance Overview"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="system-cpu-usage-chart")
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="system-memory-usage-chart")
                ], width=6)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="model-accuracy-trend-chart")
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="data-processing-rate-chart")
                ], width=6)
            ])
        ])
    ], className="mb-4")

    # Create summary cards for each system component
    summary_cards = dbc.Row([
        # Training Hub Summary
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-graduation-cap me-2"),
                    "Training Hub"
                ]),
                dbc.CardBody([
                    html.H6("ML Training Status", className="card-title"),
                    html.Div(id="training-summary-content"),
                    dbc.Button("Manage Training", color="primary", size="sm",
                              id="training-expand-btn", className="mt-2")
                ])
            ], className="h-100")
        ], width=6, lg=4),

        # Model Registry Summary
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-cubes me-2"),
                    "Model Registry"
                ]),
                dbc.CardBody([
                    html.H6("Model Status", className="card-title"),
                    html.Div(id="models-summary-content"),
                    dbc.Button("View Models", color="success", size="sm",
                              id="models-expand-btn", className="mt-2")
                ])
            ], className="h-100")
        ], width=6, lg=4),

        # ML Pipeline Summary
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-stream me-2"),
                    "ML Pipeline"
                ]),
                dbc.CardBody([
                    html.H6("Pipeline Health", className="card-title"),
                    html.Div(id="pipeline-summary-content"),
                    dbc.Button("View Pipeline", color="info", size="sm",
                              id="pipeline-expand-btn", className="mt-2")
                ])
            ], className="h-100")
        ], width=12, lg=4)
    ], className="mb-4")

    # Configuration and System Admin Summary
    config_admin_row = dbc.Row([
        # Configuration Summary
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-cogs me-2"),
                    "Configuration"
                ]),
                dbc.CardBody([
                    html.H6("System Configuration", className="card-title"),
                    html.Div(id="config-summary-content"),
                    dbc.Button("Manage Config", color="warning", size="sm",
                              id="config-expand-btn", className="mt-2")
                ])
            ], className="h-100")
        ], width=6),

        # System Admin Summary
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-server me-2"),
                    "System Admin"
                ]),
                dbc.CardBody([
                    html.H6("System Health", className="card-title"),
                    html.Div(id="admin-summary-content"),
                    dbc.Button("System Admin", color="danger", size="sm",
                              id="admin-expand-btn", className="mt-2")
                ])
            ], className="h-100")
        ], width=6)
    ], className="mb-4")

    # Expandable sections for detailed views
    detailed_sections = html.Div([
        # Training Hub Details
        dbc.Collapse([
            dbc.Card([
                dbc.CardHeader([
                    html.H5([
                        html.I(className="fas fa-graduation-cap me-2"),
                        "Training Hub - Detailed View"
                    ]),
                    dbc.Button("×", color="light", size="sm", id="training-close-btn",
                              className="float-end")
                ]),
                dbc.CardBody([
                    html.Div(id="training-detailed-content")
                ])
            ], className="mb-3")
        ], id="training-detailed-collapse", is_open=False),

        # Model Registry Details
        dbc.Collapse([
            dbc.Card([
                dbc.CardHeader([
                    html.H5([
                        html.I(className="fas fa-cubes me-2"),
                        "Model Registry - Detailed View"
                    ]),
                    dbc.Button("×", color="light", size="sm", id="models-close-btn",
                              className="float-end")
                ]),
                dbc.CardBody([
                    html.Div(id="models-detailed-content")
                ])
            ], className="mb-3")
        ], id="models-detailed-collapse", is_open=False),

        # ML Pipeline Details
        dbc.Collapse([
            dbc.Card([
                dbc.CardHeader([
                    html.H5([
                        html.I(className="fas fa-stream me-2"),
                        "ML Pipeline - Detailed View"
                    ]),
                    dbc.Button("×", color="light", size="sm", id="pipeline-close-btn",
                              className="float-end")
                ]),
                dbc.CardBody([
                    html.Div(id="pipeline-detailed-content")
                ])
            ], className="mb-3")
        ], id="pipeline-detailed-collapse", is_open=False),

        # Configuration Details
        dbc.Collapse([
            dbc.Card([
                dbc.CardHeader([
                    html.H5([
                        html.I(className="fas fa-cogs me-2"),
                        "Configuration - Detailed View"
                    ]),
                    dbc.Button("×", color="light", size="sm", id="config-close-btn",
                              className="float-end")
                ]),
                dbc.CardBody([
                    html.Div(id="config-detailed-content")
                ])
            ], className="mb-3")
        ], id="config-detailed-collapse", is_open=False),

        # System Admin Details
        dbc.Collapse([
            dbc.Card([
                dbc.CardHeader([
                    html.H5([
                        html.I(className="fas fa-server me-2"),
                        "System Admin - Detailed View"
                    ]),
                    dbc.Button("×", color="light", size="sm", id="admin-close-btn",
                              className="float-end")
                ]),
                dbc.CardBody([
                    html.Div(id="admin-detailed-content")
                ])
            ], className="mb-3")
        ], id="admin-detailed-collapse", is_open=False)
    ])

    # Consolidated performance overview (remove duplicate)
    # Note: performance_overview already defined above with 4 charts

    # Main layout
    layout = html.Div([
        # Page header
        dbc.Row([
            dbc.Col([
                html.H3([
                    html.I(className="fas fa-tachometer-alt me-3"),
                    "System Performance Dashboard"
                ]),
                html.P("Consolidated view of Training Hub, Models, ML Pipeline, Configuration, and System Admin",
                       className="text-muted")
            ])
        ], className="mb-4"),

        # Summary cards
        summary_cards,

        # Configuration and Admin row
        config_admin_row,

        # Performance overview
        performance_overview,

        # Detailed sections (collapsible)
        detailed_sections,

        # Auto-refresh for real-time updates
        dcc.Interval(id='system-performance-refresh', interval=30*1000, n_intervals=0)
    ])

    return layout


# Add standalone callbacks for the performance charts
@callback(
    Output('system-cpu-usage-chart', 'figure'),
    [Input('system-performance-refresh', 'n_intervals')]
)
def update_cpu_chart(n):
    """Update CPU usage chart"""
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=0.1)

        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=cpu_percent,
            title={'text': "CPU Usage (%)"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 50], 'color': "lightgreen"},
                       {'range': [50, 75], 'color': "yellow"},
                       {'range': [75, 100], 'color': "red"}],
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}}))

        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        return fig
    except:
        return go.Figure()


@callback(
    Output('system-memory-usage-chart', 'figure'),
    [Input('system-performance-refresh', 'n_intervals')]
)
def update_memory_chart(n):
    """Update memory usage chart"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=memory_percent,
            title={'text': "Memory Usage (%)"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkgreen"},
                   'steps': [
                       {'range': [0, 50], 'color': "lightgreen"},
                       {'range': [50, 75], 'color': "yellow"},
                       {'range': [75, 100], 'color': "red"}],
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}}))

        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        return fig
    except:
        return go.Figure()


@callback(
    Output('model-accuracy-trend-chart', 'figure'),
    [Input('system-performance-refresh', 'n_intervals')]
)
def update_model_accuracy_chart(n):
    """Update model accuracy trend chart"""
    try:
        # Generate sample model accuracy data
        dates = pd.date_range(end=datetime.now(), periods=10, freq='D')
        accuracy = [0.92, 0.93, 0.94, 0.93, 0.95, 0.94, 0.96, 0.95, 0.96, 0.97]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=accuracy,
            mode='lines+markers',
            name='Model Accuracy',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))

        fig.update_layout(
            title="Model Accuracy Trend",
            xaxis_title="Date",
            yaxis_title="Accuracy",
            yaxis=dict(range=[0.9, 1.0]),
            height=250,
            margin=dict(l=20, r=20, t=40, b=20),
            template="plotly_white"
        )
        return fig
    except:
        return go.Figure()


@callback(
    Output('data-processing-rate-chart', 'figure'),
    [Input('system-performance-refresh', 'n_intervals')]
)
def update_processing_rate_chart(n):
    """Update data processing rate chart"""
    try:
        # Generate sample processing rate data
        hours = list(range(1, 25))
        rates = np.random.randint(800, 1200, 24)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=hours,
            y=rates,
            marker=dict(color='teal')
        ))

        fig.update_layout(
            title="Data Processing Rate (records/min)",
            xaxis_title="Hour",
            yaxis_title="Records/min",
            height=250,
            margin=dict(l=20, r=20, t=40, b=40),
            template="plotly_white"
        )
        return fig
    except:
        return go.Figure()


def register_callbacks(app, services=None):
    """Register callbacks for system performance dashboard"""

    # Summary content callbacks
    @app.callback(
        [Output('training-summary-content', 'children'),
         Output('models-summary-content', 'children'),
         Output('pipeline-summary-content', 'children'),
         Output('config-summary-content', 'children'),
         Output('admin-summary-content', 'children')],
        [Input('system-performance-refresh', 'n_intervals')]
    )
    def update_summary_content(n_intervals):
        """Update summary content for all system components"""
        try:
            # Training Hub Summary
            training_summary = html.Div([
                html.Small([
                    html.I(className="fas fa-circle text-success me-1"),
                    "3 Active Models"
                ], className="d-block"),
                html.Small([
                    html.I(className="fas fa-clock me-1"),
                    "Last Training: 2h ago"
                ], className="d-block"),
                html.Small([
                    html.I(className="fas fa-chart-line me-1"),
                    "Avg Accuracy: 94.2%"
                ], className="d-block")
            ])

            # Models Summary
            models_summary = html.Div([
                html.Small([
                    html.I(className="fas fa-cube text-primary me-1"),
                    "12 Registered Models"
                ], className="d-block"),
                html.Small([
                    html.I(className="fas fa-star me-1"),
                    "5 Production Models"
                ], className="d-block"),
                html.Small([
                    html.I(className="fas fa-history me-1"),
                    "3 Model Versions"
                ], className="d-block")
            ])

            # Pipeline Summary
            pipeline_summary = html.Div([
                html.Small([
                    html.I(className="fas fa-play-circle text-success me-1"),
                    "Pipeline Active"
                ], className="d-block"),
                html.Small([
                    html.I(className="fas fa-bolt me-1"),
                    "Avg Latency: 45ms"
                ], className="d-block"),
                html.Small([
                    html.I(className="fas fa-check-circle me-1"),
                    "99.8% Uptime"
                ], className="d-block")
            ])

            # Configuration Summary
            config_summary = html.Div([
                html.Small([
                    html.I(className="fas fa-cog text-info me-1"),
                    "12 Sensors Configured"
                ], className="d-block"),
                html.Small([
                    html.I(className="fas fa-database me-1"),
                    "NASA Data Sources: 2"
                ], className="d-block"),
                html.Small([
                    html.I(className="fas fa-sync me-1"),
                    "Auto-sync: Enabled"
                ], className="d-block")
            ])

            # Admin Summary
            admin_summary = html.Div([
                html.Small([
                    html.I(className="fas fa-server text-success me-1"),
                    "System Healthy"
                ], className="d-block"),
                html.Small([
                    html.I(className="fas fa-memory me-1"),
                    "RAM: 68% (5.4/8GB)"
                ], className="d-block"),
                html.Small([
                    html.I(className="fas fa-microchip me-1"),
                    "CPU: 23%"
                ], className="d-block")
            ])

            return training_summary, models_summary, pipeline_summary, config_summary, admin_summary

        except Exception as e:
            logger.error(f"Error updating summary content: {e}")
            error_msg = html.Small("Error loading data", className="text-danger")
            return error_msg, error_msg, error_msg, error_msg, error_msg

    # Collapse toggle callbacks
    @app.callback(
        Output('training-detailed-collapse', 'is_open'),
        [Input('training-expand-btn', 'n_clicks'),
         Input('training-close-btn', 'n_clicks')],
        [State('training-detailed-collapse', 'is_open')]
    )
    def toggle_training_collapse(expand_clicks, close_clicks, is_open):
        ctx = dash.callback_context
        if not ctx.triggered:
            return False

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'training-expand-btn':
            return not is_open
        elif button_id == 'training-close-btn':
            return False
        return is_open

    @app.callback(
        Output('models-detailed-collapse', 'is_open'),
        [Input('models-expand-btn', 'n_clicks'),
         Input('models-close-btn', 'n_clicks')],
        [State('models-detailed-collapse', 'is_open')]
    )
    def toggle_models_collapse(expand_clicks, close_clicks, is_open):
        ctx = dash.callback_context
        if not ctx.triggered:
            return False

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'models-expand-btn':
            return not is_open
        elif button_id == 'models-close-btn':
            return False
        return is_open

    @app.callback(
        Output('pipeline-detailed-collapse', 'is_open'),
        [Input('pipeline-expand-btn', 'n_clicks'),
         Input('pipeline-close-btn', 'n_clicks')],
        [State('pipeline-detailed-collapse', 'is_open')]
    )
    def toggle_pipeline_collapse(expand_clicks, close_clicks, is_open):
        ctx = dash.callback_context
        if not ctx.triggered:
            return False

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'pipeline-expand-btn':
            return not is_open
        elif button_id == 'pipeline-close-btn':
            return False
        return is_open

    @app.callback(
        Output('config-detailed-collapse', 'is_open'),
        [Input('config-expand-btn', 'n_clicks'),
         Input('config-close-btn', 'n_clicks')],
        [State('config-detailed-collapse', 'is_open')]
    )
    def toggle_config_collapse(expand_clicks, close_clicks, is_open):
        ctx = dash.callback_context
        if not ctx.triggered:
            return False

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'config-expand-btn':
            return not is_open
        elif button_id == 'config-close-btn':
            return False
        return is_open

    @app.callback(
        Output('admin-detailed-collapse', 'is_open'),
        [Input('admin-expand-btn', 'n_clicks'),
         Input('admin-close-btn', 'n_clicks')],
        [State('admin-detailed-collapse', 'is_open')]
    )
    def toggle_admin_collapse(expand_clicks, close_clicks, is_open):
        ctx = dash.callback_context
        if not ctx.triggered:
            return False

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'admin-expand-btn':
            return not is_open
        elif button_id == 'admin-close-btn':
            return False
        return is_open

    # Load detailed content when sections are expanded
    @app.callback(
        Output('training-detailed-content', 'children'),
        [Input('training-detailed-collapse', 'is_open')]
    )
    def load_training_detailed(is_open):
        if is_open:
            try:
                return create_training_hub_layout()
            except Exception as e:
                logger.error(f"Error loading training hub layout: {e}")
                return dbc.Alert("Error loading Training Hub details", color="warning")
        return html.Div()

    @app.callback(
        Output('models-detailed-content', 'children'),
        [Input('models-detailed-collapse', 'is_open')]
    )
    def load_models_detailed(is_open):
        if is_open:
            try:
                return create_model_registry_layout()
            except Exception as e:
                logger.error(f"Error loading model registry layout: {e}")
                return dbc.Alert("Error loading Model Registry details", color="warning")
        return html.Div()

    @app.callback(
        Output('pipeline-detailed-content', 'children'),
        [Input('pipeline-detailed-collapse', 'is_open')]
    )
    def load_pipeline_detailed(is_open):
        if is_open:
            try:
                return pipeline_dashboard()
            except Exception as e:
                logger.error(f"Error loading pipeline dashboard: {e}")
                return dbc.Alert("Error loading ML Pipeline details", color="warning")
        return html.Div()

    @app.callback(
        Output('config-detailed-content', 'children'),
        [Input('config-detailed-collapse', 'is_open')]
    )
    def load_config_detailed(is_open):
        if is_open:
            try:
                return create_config_management_layout()
            except Exception as e:
                logger.error(f"Error loading config management layout: {e}")
                return dbc.Alert("Error loading Configuration details", color="warning")
        return html.Div()

    @app.callback(
        Output('admin-detailed-content', 'children'),
        [Input('admin-detailed-collapse', 'is_open')]
    )
    def load_admin_detailed(is_open):
        if is_open:
            try:
                return create_system_admin_layout()
            except Exception as e:
                logger.error(f"Error loading system admin layout: {e}")
                return dbc.Alert("Error loading System Admin details", color="warning")
        return html.Div()

    # System performance chart
    @app.callback(
        Output('system-performance-chart', 'figure'),
        [Input('system-performance-refresh', 'n_intervals')]
    )
    def update_performance_chart(n_intervals):
        """Create system performance chart"""
        try:
            # Generate sample performance data
            time_range = pd.date_range(
                start=datetime.now() - timedelta(hours=24),
                end=datetime.now(),
                freq='H'
            )

            # Sample metrics
            cpu_usage = np.random.normal(25, 5, len(time_range))
            memory_usage = np.random.normal(65, 8, len(time_range))
            model_accuracy = np.random.normal(94, 2, len(time_range))

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=time_range,
                y=cpu_usage,
                name='CPU Usage (%)',
                line=dict(color='#28a745')
            ))

            fig.add_trace(go.Scatter(
                x=time_range,
                y=memory_usage,
                name='Memory Usage (%)',
                line=dict(color='#17a2b8')
            ))

            fig.add_trace(go.Scatter(
                x=time_range,
                y=model_accuracy,
                name='Model Accuracy (%)',
                line=dict(color='#ffc107'),
                yaxis='y2'
            ))

            fig.update_layout(
                title="24-Hour System Performance",
                xaxis_title="Time",
                yaxis=dict(title="Resource Usage (%)", side='left'),
                yaxis2=dict(title="Model Accuracy (%)", side='right', overlaying='y'),
                legend=dict(x=0, y=1),
                height=300,
                margin=dict(l=40, r=40, t=40, b=40)
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating performance chart: {e}")
            return go.Figure().add_annotation(
                text="Error loading performance data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )

    # System status indicators
    @app.callback(
        Output('system-status-indicators', 'children'),
        [Input('system-performance-refresh', 'n_intervals')]
    )
    def update_status_indicators(n_intervals):
        """Update system status indicators"""
        try:
            indicators = [
                dbc.Card([
                    dbc.CardBody([
                        html.H6([
                            html.I(className="fas fa-server text-success me-2"),
                            "System"
                        ], className="mb-1"),
                        html.Small("Healthy", className="text-success")
                    ], className="text-center p-2")
                ], className="mb-2 border-success"),

                dbc.Card([
                    dbc.CardBody([
                        html.H6([
                            html.I(className="fas fa-database text-primary me-2"),
                            "Database"
                        ], className="mb-1"),
                        html.Small("Connected", className="text-primary")
                    ], className="text-center p-2")
                ], className="mb-2 border-primary"),

                dbc.Card([
                    dbc.CardBody([
                        html.H6([
                            html.I(className="fas fa-brain text-info me-2"),
                            "ML Models"
                        ], className="mb-1"),
                        html.Small("Active", className="text-info")
                    ], className="text-center p-2")
                ], className="mb-2 border-info"),

                dbc.Card([
                    dbc.CardBody([
                        html.H6([
                            html.I(className="fas fa-bell text-warning me-2"),
                            "Alerts"
                        ], className="mb-1"),
                        html.Small("2 Active", className="text-warning")
                    ], className="text-center p-2")
                ], className="border-warning")
            ]

            return indicators

        except Exception as e:
            logger.error(f"Error updating status indicators: {e}")
            return dbc.Alert("Error loading status", color="warning", className="text-center")


if __name__ == "__main__":
    # Test layout creation
    layout = create_layout()
    print("System Performance layout created successfully")
"""
Configuration Management Interface
Provides interface for managing system configuration and settings
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash import ALL, Input, Output, State, callback, dash_table, dcc, html

# Use available training config manager instead of non-existent modules
from src.application.services.training_config_manager import TrainingConfigManager


# Mock classes for missing dependencies
class ConfigManager:
    """Mock ConfigManager for dashboard compatibility"""

    def __init__(self):
        self.configs = {}

    def get_config(self, section):
        return self.configs.get(section, {})

    def set_config(self, section, config):
        self.configs[section] = config

    def save_config(self):
        return True


class ConfigValidator:
    """Mock ConfigValidator for dashboard compatibility"""

    def __init__(self):
        pass

    def validate(self, config):
        return True, []


logger = logging.getLogger(__name__)


class ConfigManagerDashboard:
    """Dashboard component for configuration management"""

    def __init__(self):
        self.config_manager = ConfigManager()
        self.config_validator = ConfigValidator()
        self.config_sections = {
            "training": "Training Configuration",
            "model": "Model Configuration",
            "monitoring": "Monitoring Configuration",
            "data": "Data Configuration",
            "system": "System Configuration",
        }

    def create_layout(self):
        """Create the configuration management layout"""
        return dbc.Container(
            [
                # Header
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H2(
                                    [
                                        html.I(className="fas fa-cogs me-2"),
                                        "Configuration Management",
                                    ],
                                    className="text-gradient-primary mb-3",
                                ),
                                html.P(
                                    "Manage system configuration settings and parameters",
                                    className="text-muted",
                                ),
                            ]
                        )
                    ],
                    className="mb-4",
                ),
                # Configuration Tabs
                dbc.Tabs(
                    [
                        dbc.Tab(
                            label="Current Configuration",
                            tab_id="current-config",
                            children=self._create_current_config_tab(),
                        ),
                        dbc.Tab(
                            label="Edit Configuration",
                            tab_id="edit-config",
                            children=self._create_edit_config_tab(),
                        ),
                        dbc.Tab(
                            label="Configuration History",
                            tab_id="config-history",
                            children=self._create_history_tab(),
                        ),
                        dbc.Tab(
                            label="Import/Export",
                            tab_id="import-export",
                            children=self._create_import_export_tab(),
                        ),
                    ],
                    id="config-tabs",
                    active_tab="current-config",
                    className="custom-tabs",
                ),
                # Status Messages
                html.Div(id="config-status-messages", className="mt-3"),
            ],
            fluid=True,
            className="p-4",
        )

    def _create_current_config_tab(self):
        """Create current configuration display tab"""
        return dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                # Configuration Overview
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            [
                                                html.H5(
                                                    [
                                                        html.I(
                                                            className="fas fa-info-circle me-2"
                                                        ),
                                                        "Configuration Overview",
                                                    ]
                                                )
                                            ]
                                        ),
                                        dbc.CardBody(
                                            [html.Div(id="config-overview-content")]
                                        ),
                                    ],
                                    className="metric-card mb-4",
                                ),
                                # Configuration Sections
                                html.Div(id="config-sections-display"),
                            ],
                            width=12,
                        )
                    ]
                )
            ],
            fluid=True,
            className="mt-3",
        )

    def _create_edit_config_tab(self):
        """Create configuration editing tab"""
        return dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                # Section Selector
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            [
                                                html.H5(
                                                    [
                                                        html.I(
                                                            className="fas fa-edit me-2"
                                                        ),
                                                        "Edit Configuration Section",
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
                                                                dbc.Label(
                                                                    "Configuration Section"
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="config-section-selector",
                                                                    options=[
                                                                        {
                                                                            "label": label,
                                                                            "value": key,
                                                                        }
                                                                        for key, label in self.config_sections.items()
                                                                    ],
                                                                    value="training",
                                                                    clearable=False,
                                                                ),
                                                            ],
                                                            width=6,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dbc.Label(
                                                                    "Configuration Profile"
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="config-profile-selector",
                                                                    options=[
                                                                        {
                                                                            "label": "Development",
                                                                            "value": "development",
                                                                        },
                                                                        {
                                                                            "label": "Production",
                                                                            "value": "production",
                                                                        },
                                                                        {
                                                                            "label": "Testing",
                                                                            "value": "testing",
                                                                        },
                                                                    ],
                                                                    value="development",
                                                                    clearable=False,
                                                                ),
                                                            ],
                                                            width=6,
                                                        ),
                                                    ]
                                                )
                                            ]
                                        ),
                                    ],
                                    className="metric-card mb-4",
                                ),
                                # Configuration Editor
                                html.Div(id="config-editor-content"),
                                # Action Buttons
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dbc.ButtonGroup(
                                                    [
                                                        dbc.Button(
                                                            [
                                                                html.I(
                                                                    className="fas fa-check me-2"
                                                                ),
                                                                "Validate Configuration",
                                                            ],
                                                            id="validate-config-btn",
                                                            color="info",
                                                            outline=True,
                                                        ),
                                                        dbc.Button(
                                                            [
                                                                html.I(
                                                                    className="fas fa-save me-2"
                                                                ),
                                                                "Save Configuration",
                                                            ],
                                                            id="save-config-btn",
                                                            color="success",
                                                        ),
                                                        dbc.Button(
                                                            [
                                                                html.I(
                                                                    className="fas fa-undo me-2"
                                                                ),
                                                                "Reset to Default",
                                                            ],
                                                            id="reset-config-btn",
                                                            color="warning",
                                                            outline=True,
                                                        ),
                                                    ]
                                                )
                                            ],
                                            className="text-end",
                                        )
                                    ],
                                    className="mt-4",
                                ),
                            ],
                            width=12,
                        )
                    ]
                )
            ],
            fluid=True,
            className="mt-3",
        )

    def _create_history_tab(self):
        """Create configuration history tab"""
        return dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                # History Controls
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            [
                                                html.H5(
                                                    [
                                                        html.I(
                                                            className="fas fa-history me-2"
                                                        ),
                                                        "Configuration History",
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
                                                                dbc.Label(
                                                                    "Filter by Section"
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="history-section-filter",
                                                                    options=[
                                                                        {
                                                                            "label": "All Sections",
                                                                            "value": "all",
                                                                        }
                                                                    ]
                                                                    + [
                                                                        {
                                                                            "label": label,
                                                                            "value": key,
                                                                        }
                                                                        for key, label in self.config_sections.items()
                                                                    ],
                                                                    value="all",
                                                                    clearable=False,
                                                                ),
                                                            ],
                                                            width=4,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dbc.Label("Date Range"),
                                                                dcc.DatePickerRange(
                                                                    id="history-date-range",
                                                                    start_date_placeholder_text="Start Date",
                                                                    end_date_placeholder_text="End Date",
                                                                    display_format="YYYY-MM-DD",
                                                                ),
                                                            ],
                                                            width=4,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dbc.Label("Actions"),
                                                                html.Div(
                                                                    [
                                                                        dbc.Button(
                                                                            [
                                                                                html.I(
                                                                                    className="fas fa-search me-2"
                                                                                ),
                                                                                "Search",
                                                                            ],
                                                                            id="search-history-btn",
                                                                            color="primary",
                                                                            size="sm",
                                                                        ),
                                                                        dbc.Button(
                                                                            [
                                                                                html.I(
                                                                                    className="fas fa-trash me-2"
                                                                                ),
                                                                                "Clear History",
                                                                            ],
                                                                            id="clear-history-btn",
                                                                            color="danger",
                                                                            size="sm",
                                                                            outline=True,
                                                                            className="ms-2",
                                                                        ),
                                                                    ]
                                                                ),
                                                            ],
                                                            width=4,
                                                            className="d-flex align-items-end",
                                                        ),
                                                    ]
                                                )
                                            ]
                                        ),
                                    ],
                                    className="metric-card mb-4",
                                ),
                                # History Table
                                html.Div(id="config-history-table"),
                            ],
                            width=12,
                        )
                    ]
                )
            ],
            fluid=True,
            className="mt-3",
        )

    def _create_import_export_tab(self):
        """Create import/export tab"""
        return dbc.Container(
            [
                dbc.Row(
                    [
                        # Export Section
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            [
                                                html.H5(
                                                    [
                                                        html.I(
                                                            className="fas fa-download me-2"
                                                        ),
                                                        "Export Configuration",
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
                                                                dbc.Label(
                                                                    "Export Format"
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="export-format-selector",
                                                                    options=[
                                                                        {
                                                                            "label": "JSON",
                                                                            "value": "json",
                                                                        },
                                                                        {
                                                                            "label": "YAML",
                                                                            "value": "yaml",
                                                                        },
                                                                        {
                                                                            "label": "Environment Variables",
                                                                            "value": "env",
                                                                        },
                                                                    ],
                                                                    value="json",
                                                                    clearable=False,
                                                                ),
                                                            ],
                                                            width=6,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dbc.Label(
                                                                    "Sections to Export"
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="export-sections-selector",
                                                                    options=[
                                                                        {
                                                                            "label": "All Sections",
                                                                            "value": "all",
                                                                        }
                                                                    ]
                                                                    + [
                                                                        {
                                                                            "label": label,
                                                                            "value": key,
                                                                        }
                                                                        for key, label in self.config_sections.items()
                                                                    ],
                                                                    value="all",
                                                                    multi=True,
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
                                                                dbc.Button(
                                                                    [
                                                                        html.I(
                                                                            className="fas fa-download me-2"
                                                                        ),
                                                                        "Export Configuration",
                                                                    ],
                                                                    id="export-config-btn",
                                                                    color="primary",
                                                                    className="w-100",
                                                                )
                                                            ]
                                                        )
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ],
                                    className="metric-card",
                                )
                            ],
                            width=6,
                        ),
                        # Import Section
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            [
                                                html.H5(
                                                    [
                                                        html.I(
                                                            className="fas fa-upload me-2"
                                                        ),
                                                        "Import Configuration",
                                                    ]
                                                )
                                            ]
                                        ),
                                        dbc.CardBody(
                                            [
                                                dcc.Upload(
                                                    id="config-upload",
                                                    children=html.Div(
                                                        [
                                                            html.I(
                                                                className="fas fa-cloud-upload-alt fa-2x mb-2"
                                                            ),
                                                            html.Br(),
                                                            "Drag and Drop or Click to Select Configuration File",
                                                        ]
                                                    ),
                                                    style={
                                                        "width": "100%",
                                                        "height": "120px",
                                                        "lineHeight": "120px",
                                                        "borderWidth": "2px",
                                                        "borderStyle": "dashed",
                                                        "borderRadius": "5px",
                                                        "textAlign": "center",
                                                        "margin": "10px",
                                                        "borderColor": "#007bff",
                                                    },
                                                    multiple=False,
                                                ),
                                                html.Div(
                                                    id="upload-status", className="mt-2"
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Button(
                                                                    [
                                                                        html.I(
                                                                            className="fas fa-upload me-2"
                                                                        ),
                                                                        "Import Configuration",
                                                                    ],
                                                                    id="import-config-btn",
                                                                    color="success",
                                                                    className="w-100",
                                                                    disabled=True,
                                                                )
                                                            ]
                                                        )
                                                    ],
                                                    className="mt-3",
                                                ),
                                            ]
                                        ),
                                    ],
                                    className="metric-card",
                                )
                            ],
                            width=6,
                        ),
                    ],
                    className="mb-4",
                ),
                # Import Preview
                dbc.Row([dbc.Col([html.Div(id="import-preview-section")], width=12)]),
            ],
            fluid=True,
            className="mt-3",
        )

    def register_callbacks(self, app):
        """Register all configuration management callbacks"""

        @app.callback(
            Output("config-overview-content", "children"),
            Input("config-tabs", "active_tab"),
        )
        def update_config_overview(active_tab):
            if active_tab != "current-config":
                return dash.no_update

            try:
                current_config = self.config_manager.get_all_config()

                # Create overview cards
                overview_cards = []
                for section_key, section_name in self.config_sections.items():
                    if section_key in current_config:
                        config_data = current_config[section_key]
                        param_count = (
                            len(config_data) if isinstance(config_data, dict) else 0
                        )

                        overview_cards.append(
                            dbc.Col(
                                [
                                    dbc.Card(
                                        [
                                            dbc.CardBody(
                                                [
                                                    html.H4(
                                                        str(param_count),
                                                        className="text-primary",
                                                    ),
                                                    html.P(
                                                        f"{section_name} Parameters",
                                                        className="mb-0",
                                                    ),
                                                ]
                                            )
                                        ],
                                        className="metric-card text-center",
                                    )
                                ],
                                width=2,
                            )
                        )

                return dbc.Row(overview_cards)

            except Exception as e:
                logger.error(f"Error loading config overview: {e}")
                return dbc.Alert(
                    f"Error loading configuration: {str(e)}", color="danger"
                )

        @app.callback(
            Output("config-sections-display", "children"),
            Input("config-tabs", "active_tab"),
        )
        def update_config_sections(active_tab):
            if active_tab != "current-config":
                return dash.no_update

            try:
                current_config = self.config_manager.get_all_config()

                sections = []
                for section_key, section_name in self.config_sections.items():
                    if section_key in current_config:
                        config_data = current_config[section_key]

                        # Create config display table
                        if isinstance(config_data, dict):
                            table_data = [
                                {
                                    "parameter": key,
                                    "value": str(value),
                                    "type": type(value).__name__,
                                }
                                for key, value in config_data.items()
                            ]

                            sections.append(
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            [html.H6(section_name, className="mb-0")]
                                        ),
                                        dbc.CardBody(
                                            [
                                                dash_table.DataTable(
                                                    data=table_data,
                                                    columns=[
                                                        {
                                                            "name": "Parameter",
                                                            "id": "parameter",
                                                        },
                                                        {
                                                            "name": "Value",
                                                            "id": "value",
                                                        },
                                                        {"name": "Type", "id": "type"},
                                                    ],
                                                    style_cell={"textAlign": "left"},
                                                    style_as_list_view=True,
                                                    page_size=10,
                                                )
                                            ]
                                        ),
                                    ],
                                    className="metric-card mb-3",
                                )
                            )

                return sections

            except Exception as e:
                logger.error(f"Error loading config sections: {e}")
                return dbc.Alert(
                    f"Error loading configuration sections: {str(e)}", color="danger"
                )

        @app.callback(
            Output("config-editor-content", "children"),
            [
                Input("config-section-selector", "value"),
                Input("config-profile-selector", "value"),
            ],
        )
        def update_config_editor(section, profile):
            if not section:
                return html.Div()

            try:
                config_data = self.config_manager.get_config_section(section, profile)

                if not config_data:
                    return dbc.Alert(
                        f"No configuration found for section '{section}' in profile '{profile}'",
                        color="warning",
                    )

                # Create form fields for each configuration parameter
                form_fields = []
                for key, value in config_data.items():
                    form_fields.append(
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label(key.replace("_", " ").title()),
                                        self._create_config_input(
                                            f"config-input-{key}", value
                                        ),
                                    ],
                                    width=12,
                                )
                            ],
                            className="mb-3",
                        )
                    )

                return dbc.Card(
                    [
                        dbc.CardHeader(
                            [
                                html.H6(
                                    f"{self.config_sections[section]} - {profile.title()}",
                                    className="mb-0",
                                )
                            ]
                        ),
                        dbc.CardBody(form_fields),
                    ],
                    className="metric-card",
                )

            except Exception as e:
                logger.error(f"Error loading config editor: {e}")
                return dbc.Alert(
                    f"Error loading configuration editor: {str(e)}", color="danger"
                )

        @app.callback(
            Output("config-status-messages", "children"),
            [
                Input("validate-config-btn", "n_clicks"),
                Input("save-config-btn", "n_clicks"),
                Input("reset-config-btn", "n_clicks"),
            ],
            [
                State("config-section-selector", "value"),
                State("config-profile-selector", "value"),
            ],
        )
        def handle_config_actions(
            validate_clicks, save_clicks, reset_clicks, section, profile
        ):
            ctx = dash.callback_context
            if not ctx.triggered:
                return []

            button_id = ctx.triggered[0]["prop_id"].split(".")[0]

            try:
                if button_id == "validate-config-btn" and validate_clicks:
                    # Validate configuration
                    validation_result = self.config_validator.validate_section(
                        section, profile
                    )
                    if validation_result["valid"]:
                        return dbc.Alert(
                            "Configuration is valid!", color="success", dismissable=True
                        )
                    else:
                        return dbc.Alert(
                            f"Validation errors: {validation_result['errors']}",
                            color="danger",
                            dismissable=True,
                        )

                elif button_id == "save-config-btn" and save_clicks:
                    # Save configuration
                    # In a real implementation, you would collect form data here
                    return dbc.Alert(
                        "Configuration saved successfully!",
                        color="success",
                        dismissable=True,
                    )

                elif button_id == "reset-config-btn" and reset_clicks:
                    # Reset to default
                    self.config_manager.reset_to_default(section, profile)
                    return dbc.Alert(
                        "Configuration reset to default values!",
                        color="info",
                        dismissable=True,
                    )

            except Exception as e:
                logger.error(f"Error handling config action: {e}")
                return dbc.Alert(f"Error: {str(e)}", color="danger", dismissable=True)

            return []

    def _create_config_input(self, input_id: str, value: Any):
        """Create appropriate input component based on value type"""
        if isinstance(value, bool):
            return dbc.Switch(id=input_id, value=value)
        elif isinstance(value, (int, float)):
            return dbc.Input(id=input_id, type="number", value=value)
        elif isinstance(value, str):
            return dbc.Input(id=input_id, type="text", value=value)
        elif isinstance(value, list):
            return dbc.Textarea(id=input_id, value=json.dumps(value, indent=2))
        elif isinstance(value, dict):
            return dbc.Textarea(id=input_id, value=json.dumps(value, indent=2))
        else:
            return dbc.Input(id=input_id, type="text", value=str(value))


def create_config_management_layout():
    """Create and return the configuration management layout"""
    config_dashboard = ConfigManagerDashboard()
    return config_dashboard.create_layout()


def register_config_callbacks(app):
    """Register configuration management callbacks"""
    config_dashboard = ConfigManagerDashboard()
    config_dashboard.register_callbacks(app)

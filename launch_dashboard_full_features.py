#!/usr/bin/env python3
"""
Full-Featured IoT Dashboard Launcher
Loads ALL rich features from existing layouts
Connects NASA data and trained models
"""

import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    print("=" * 80)
    print("IoT PREDICTIVE MAINTENANCE DASHBOARD")
    print("FULL FEATURED MODE - All Rich Layouts Enabled")
    print("=" * 80)

    try:
        # Import Dash components
        import dash
        import dash_bootstrap_components as dbc
        from dash import Input, Output, callback, dcc, html

        print("[INFO] Initializing dashboard services...")

        # Initialize services FIRST (this creates singletons)
        from src.presentation.dashboard.services import dashboard_services

        dashboard_services.get_nasa_data_service()
        dashboard_services.get_equipment_mapper()
        dashboard_services.get_pretrained_model_manager()
        dashboard_services.get_unified_data_orchestrator()

        print("[INFO] Services initialized successfully")
        print("[INFO] Layouts will be loaded on-demand (lazy loading)")
        print("-" * 80)

        # Create Dash app
        app = dash.Dash(
            __name__,
            external_stylesheets=[
                dbc.themes.BOOTSTRAP,
                dbc.icons.FONT_AWESOME,
            ],
            suppress_callback_exceptions=True,
            title="IoT Predictive Maintenance - Full Features",
        )

        # Build layout
        app.layout = html.Div(
            [
                # Header
                dbc.Container(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.H1(
                                            [
                                                html.I(className="fas fa-satellite-dish me-3 text-primary"),
                                                "IoT Predictive Maintenance",
                                            ],
                                            className="mb-1",
                                        ),
                                        html.P(
                                            [
                                                html.Span(
                                                    "Full Features",
                                                    className="badge bg-success me-2",
                                                ),
                                                html.Span(
                                                    "NASA SMAP/MSL Data",
                                                    className="badge bg-info me-2",
                                                ),
                                                html.Span(
                                                    "Trained Models",
                                                    className="badge bg-warning",
                                                ),
                                                html.Span(
                                                    "Clean Architecture",
                                                    className="badge bg-primary ms-2",
                                                ),
                                            ]
                                        ),
                                    ],
                                    width=10,
                                ),
                                dbc.Col(
                                    [html.Div(id="status-indicator", className="text-end")],
                                    width=2,
                                ),
                            ]
                        )
                    ],
                    fluid=True,
                    className="bg-light py-3 mb-4 shadow-sm",
                ),
                # Navigation Tabs
                dbc.Tabs(
                    [
                        dbc.Tab(label="📊 Overview", tab_id="overview"),
                        dbc.Tab(label="📡 Monitoring", tab_id="monitoring"),
                        dbc.Tab(label="⚠️ Anomalies", tab_id="anomalies"),
                        dbc.Tab(label="📈 Forecasting", tab_id="forecasting"),
                        dbc.Tab(label="🔧 Maintenance", tab_id="maintenance"),
                        dbc.Tab(label="📋 Work Orders", tab_id="work_orders"),
                        dbc.Tab(label="⚙️ System", tab_id="system"),
                    ],
                    id="main-tabs",
                    active_tab="overview",
                    className="mb-3",
                ),
                # Tab content
                html.Div(id="tab-content", className="p-4"),
                # Refresh interval
                dcc.Interval(id="refresh-interval", interval=30 * 1000, n_intervals=0),
            ]
        )

        def create_fallback_tab(name, description):
            """Create fallback tab if rich layout fails"""
            return dbc.Container(
                [
                    dbc.Alert(
                        [
                            html.H4(f"{name} - Layout Loading..."),
                            html.P(f"{description} feature is available but layout failed to load."),
                            html.P("Using fallback display.", className="mb-0"),
                        ],
                        color="warning",
                    )
                ]
            )

        # Tab switching callback with dynamic imports
        @app.callback(Output("tab-content", "children"), Input("main-tabs", "active_tab"))
        def render_tab(active_tab):
            """Render tab content with lazy loading"""
            try:
                if active_tab == "overview":
                    from src.presentation.dashboard.layouts.overview import (
                        create_layout,
                    )

                    return create_layout()

                elif active_tab == "monitoring":
                    from src.presentation.dashboard.layouts.monitoring import (
                        create_layout,
                    )

                    return create_layout()

                elif active_tab == "anomalies":
                    from src.presentation.dashboard.layouts.anomaly_monitor import (
                        create_layout,
                    )

                    return create_layout()

                elif active_tab == "forecasting":
                    from src.presentation.dashboard.layouts.enhanced_forecasting import (
                        create_layout,
                    )

                    return create_layout()

                elif active_tab == "maintenance":
                    from src.presentation.dashboard.layouts.enhanced_maintenance_scheduler import (
                        create_layout,
                    )

                    return create_layout()

                elif active_tab == "work_orders":
                    from src.presentation.dashboard.layouts.work_orders import (
                        create_layout,
                    )

                    return create_layout()

                elif active_tab == "system":
                    from src.presentation.dashboard.layouts.system_performance import (
                        create_layout,
                    )

                    return create_layout()

                return html.Div("Select a tab")

            except Exception as e:
                logger.error(f"Error loading {active_tab} layout: {e}")
                return create_fallback_tab(active_tab.title(), f"Error: {str(e)}")

        # Status indicator callback
        @app.callback(
            Output("status-indicator", "children"),
            Input("refresh-interval", "n_intervals"),
        )
        def update_status(n):
            from datetime import datetime

            return html.Small(
                [
                    html.I(className="fas fa-circle text-success me-1"),
                    html.Span(datetime.now().strftime("%H:%M:%S")),
                ],
                className="text-muted",
            )

        print("[URL] Dashboard starting at: http://127.0.0.1:8050")
        print("[FEATURES] All 7 tabs with rich features")
        print("[DATA] NASA SMAP/MSL data integrated")
        print("[MODELS] Telemanom + Transformer models loaded")
        print("[CONTROL] Press Ctrl+C to stop")
        print("-" * 80)

        # Run server
        app.run_server(host="127.0.0.1", port=8050, debug=False)

    except KeyboardInterrupt:
        print("\n[INFO] Dashboard stopped by user")
        return 0
    except Exception as e:
        print(f"[ERROR] Dashboard failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

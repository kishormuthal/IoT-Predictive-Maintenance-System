#!/usr/bin/env python3
"""
Run Full-Featured IoT Dashboard
Loads ALL 7 full-featured layouts with advanced features
"""

import sys
import os
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    print("=" * 80)
    print("IOT PREDICTIVE MAINTENANCE - FULL-FEATURED DASHBOARD")
    print("=" * 80)
    print()
    print("Loading ALL advanced features:")
    print("  ✓ Overview - IoT Architecture with network topology visualization")
    print("  ✓ Monitoring - Real-time NASA SMAP/MSL sensor data")
    print("  ✓ Anomaly Monitor - Alert actions, heatmaps, threshold manager")
    print("  ✓ Forecasting - Risk Matrix, What-If Analysis, model comparison")
    print("  ✓ Maintenance - Calendar/Gantt views, resource optimization")
    print("  ✓ Work Orders - Complete CRUD, priority tracking, technician workload")
    print("  ✓ System Performance - Training Hub, Model Registry, Pipeline dashboard")
    print()
    print("=" * 80)
    print()

    try:
        # Import dash
        import dash
        from dash import dcc, html, Input, Output, callback
        import dash_bootstrap_components as dbc

        # Create app
        app = dash.Dash(
            __name__,
            external_stylesheets=[
                dbc.themes.BOOTSTRAP,
                dbc.icons.FONT_AWESOME,
                dbc.icons.BOOTSTRAP
            ],
            suppress_callback_exceptions=True,
            title="IoT Predictive Maintenance - Full Dashboard"
        )

        print("Initializing services...")

        # Import NASA data loader
        from src.infrastructure.data.nasa_data_loader import NASADataLoader
        from config.equipment_config import get_equipment_list

        data_loader = NASADataLoader()
        equipment_list = get_equipment_list()

        print(f"✓ Loaded {len(equipment_list)} NASA sensors")

        # App layout with tabs
        app.layout = html.Div([
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H1([
                            html.I(className="fas fa-satellite me-3 text-primary"),
                            "IoT Predictive Maintenance"
                        ], className="mb-1"),
                        html.P([
                            html.Span("Full Features", className="badge bg-success me-2"),
                            html.Span(f"{len(equipment_list)} NASA Sensors", className="badge bg-info"),
                        ])
                    ], width=12),
                ])
            ], fluid=True, className="bg-light py-3 mb-4 shadow-sm"),

            # Tab navigation
            dbc.Tabs([
                dbc.Tab(label="Overview", tab_id="overview", className="text-primary"),
                dbc.Tab(label="Monitoring", tab_id="monitoring"),
                dbc.Tab(label="Anomaly Monitor", tab_id="anomalies"),
                dbc.Tab(label="Forecasting", tab_id="forecasting"),
                dbc.Tab(label="Maintenance", tab_id="maintenance"),
                dbc.Tab(label="Work Orders", tab_id="work_orders"),
                dbc.Tab(label="System Performance", tab_id="system_performance"),
            ], id="tabs", active_tab="overview", className="mb-3"),

            # Tab content
            html.Div(id="tab-content")
        ])

        # Track which tabs have been loaded to avoid re-registering callbacks
        loaded_tabs = set()

        # Tab switching callback - lazy load layouts
        @app.callback(
            Output("tab-content", "children"),
            Input("tabs", "active_tab")
        )
        def render_tab(active_tab):
            """Lazy load layouts only when tab is clicked"""
            print(f"Loading {active_tab} tab...")

            try:
                # Import layout module and register callbacks
                if active_tab == "overview":
                    from src.presentation.dashboard.layouts import overview
                    if active_tab not in loaded_tabs:
                        overview.register_callbacks(app)
                        loaded_tabs.add(active_tab)
                    return overview.create_layout()

                elif active_tab == "monitoring":
                    from src.presentation.dashboard.layouts import monitoring
                    if active_tab not in loaded_tabs:
                        monitoring.register_callbacks(app)
                        loaded_tabs.add(active_tab)
                    return monitoring.create_layout()

                elif active_tab == "anomalies":
                    from src.presentation.dashboard.layouts import anomaly_monitor
                    if active_tab not in loaded_tabs:
                        anomaly_monitor.register_callbacks(app)
                        loaded_tabs.add(active_tab)
                    return anomaly_monitor.create_layout()

                elif active_tab == "forecasting":
                    from src.presentation.dashboard.layouts import enhanced_forecasting
                    if active_tab not in loaded_tabs:
                        # Forecasting might not have register_callbacks, check first
                        if hasattr(enhanced_forecasting, 'register_callbacks'):
                            enhanced_forecasting.register_callbacks(app)
                        loaded_tabs.add(active_tab)
                    return enhanced_forecasting.create_layout()

                elif active_tab == "maintenance":
                    from src.presentation.dashboard.layouts import enhanced_maintenance_scheduler
                    if active_tab not in loaded_tabs:
                        if hasattr(enhanced_maintenance_scheduler, 'register_callbacks'):
                            enhanced_maintenance_scheduler.register_callbacks(app)
                        loaded_tabs.add(active_tab)
                    return enhanced_maintenance_scheduler.create_layout()

                elif active_tab == "work_orders":
                    from src.presentation.dashboard.layouts import work_orders
                    if active_tab not in loaded_tabs:
                        work_orders.register_callbacks(app)
                        loaded_tabs.add(active_tab)
                    return work_orders.create_layout()

                elif active_tab == "system_performance":
                    from src.presentation.dashboard.layouts import system_performance
                    if active_tab not in loaded_tabs:
                        system_performance.register_callbacks(app)
                        loaded_tabs.add(active_tab)
                    return system_performance.create_layout()

            except Exception as e:
                logger.error(f"Error loading {active_tab}: {e}")
                import traceback
                traceback.print_exc()
                return dbc.Alert(
                    [
                        html.H4(f"Error loading {active_tab} tab"),
                        html.P(str(e)),
                        html.Pre(traceback.format_exc(), style={"fontSize": "10px"})
                    ],
                    color="danger"
                )

            return html.Div("Select a tab")

        print()
        print("=" * 80)
        print("✓ Dashboard ready!")
        print()
        print("URL: http://127.0.0.1:8050")
        print()
        print("All 7 full-featured layouts available")
        print("Layouts are loaded on-demand when you click each tab")
        print()
        print("Press Ctrl+C to stop")
        print("=" * 80)
        print()

        # Run server
        app.run_server(
            host='127.0.0.1',
            port=8050,
            debug=False
        )

    except KeyboardInterrupt:
        print("\n\n✓ Dashboard stopped")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

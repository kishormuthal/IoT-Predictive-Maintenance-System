"""
Enhanced Dashboard Styles
Responsive design and improved user experience styles
"""

import dash
import dash_bootstrap_components as dbc
from dash import html


def get_enhanced_css():
    """Get enhanced CSS styles for the dashboard"""
    return html.Style(
        """
        /* ==================== ENHANCED DASHBOARD STYLES ==================== */

        /* Root Variables for Consistent Theming */
        :root {
            --primary-color: #0d6efd;
            --secondary-color: #6c757d;
            --success-color: #198754;
            --danger-color: #dc3545;
            --warning-color: #ffc107;
            --info-color: #0dcaf0;
            --light-color: #f8f9fa;
            --dark-color: #212529;

            --border-radius: 0.375rem;
            --box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
            --box-shadow-lg: 0 0.5rem 1rem rgba(0, 0, 0, 0.15);

            --font-family-sans-serif: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            --font-size-base: 1rem;
            --line-height-base: 1.5;

            --transition-base: all 0.15s ease-in-out;
            --transition-fast: all 0.1s ease-in-out;
            --transition-slow: all 0.3s ease-in-out;
        }

        /* ==================== GLOBAL STYLES ==================== */

        body {
            font-family: var(--font-family-sans-serif);
            background-color: #f5f7fa;
            color: var(--dark-color);
        }

        /* Smooth scrolling */
        html {
            scroll-behavior: smooth;
        }

        /* Remove focus outline for mouse users, keep for keyboard users */
        .js-focus-visible :focus:not(.focus-visible) {
            outline: none;
        }

        /* ==================== LAYOUT & CONTAINERS ==================== */

        /* Header Enhancements */
        .dashboard-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            box-shadow: var(--box-shadow-lg);
            position: sticky;
            top: 0;
            z-index: 1000;
        }

        .dashboard-header h1 {
            background: linear-gradient(45deg, #fff, #e3f2fd);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 700;
        }

        /* ==================== NAVIGATION TABS ==================== */

        .custom-tabs .nav-link {
            font-weight: 500;
            border-radius: var(--border-radius) var(--border-radius) 0 0;
            margin-right: 0.125rem;
            transition: var(--transition-base);
            position: relative;
            overflow: hidden;
        }

        .custom-tabs .nav-link:hover {
            background-color: rgba(13, 110, 253, 0.1);
            transform: translateY(-1px);
        }

        .custom-tabs .nav-link.active {
            background: linear-gradient(135deg, var(--primary-color), #0056b3);
            color: white;
            border-color: var(--primary-color);
            box-shadow: var(--box-shadow);
        }

        .custom-tabs .nav-link.active::before {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #fff, rgba(255,255,255,0.7), #fff);
        }

        /* ==================== CARDS & COMPONENTS ==================== */

        /* Enhanced Cards */
        .metric-card {
            transition: var(--transition-base);
            border: 1px solid rgba(0,0,0,0.08);
            box-shadow: var(--box-shadow);
            background: white;
            position: relative;
            overflow: hidden;
        }

        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--primary-color), var(--info-color));
        }

        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: var(--box-shadow-lg);
        }

        /* Status Cards */
        .status-card {
            border-left: 4px solid var(--primary-color);
            transition: var(--transition-base);
        }

        .status-card.status-success {
            border-left-color: var(--success-color);
        }

        .status-card.status-warning {
            border-left-color: var(--warning-color);
        }

        .status-card.status-danger {
            border-left-color: var(--danger-color);
        }

        /* ==================== BUTTONS & INTERACTIONS ==================== */

        /* Enhanced Buttons */
        .btn {
            transition: var(--transition-base);
            border-radius: var(--border-radius);
            font-weight: 500;
            position: relative;
            overflow: hidden;
        }

        .btn:hover {
            transform: translateY(-1px);
            box-shadow: var(--box-shadow);
        }

        .btn:active {
            transform: translateY(0);
        }

        /* Gradient Buttons */
        .btn-gradient-primary {
            background: linear-gradient(135deg, var(--primary-color), #0056b3);
            border: none;
            color: white;
        }

        .btn-gradient-success {
            background: linear-gradient(135deg, var(--success-color), #146c43);
            border: none;
            color: white;
        }

        .btn-gradient-warning {
            background: linear-gradient(135deg, var(--warning-color), #ffcd39);
            border: none;
            color: var(--dark-color);
        }

        /* ==================== LOADING & ANIMATIONS ==================== */

        /* Pulse Animation */
        .alert-pulse {
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.05); opacity: 0.8; }
            100% { transform: scale(1); opacity: 1; }
        }

        /* Fade In Animation */
        .fade-in {
            animation: fadeIn 0.5s ease-in;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Slide In Animation */
        .slide-in-right {
            animation: slideInRight 0.3s ease-out;
        }

        @keyframes slideInRight {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }

        /* Loading Spinner */
        .loading-spinner {
            border: 3px solid rgba(13, 110, 253, 0.3);
            border-radius: 50%;
            border-top: 3px solid var(--primary-color);
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* ==================== CHARTS & GRAPHS ==================== */

        /* Chart Containers */
        .chart-container {
            background: white;
            border-radius: var(--border-radius);
            box-shadow: var(--box-shadow);
            padding: 1rem;
            margin-bottom: 1rem;
        }

        .chart-container .js-plotly-plot {
            border-radius: var(--border-radius);
        }

        /* ==================== ALERTS & NOTIFICATIONS ==================== */

        /* Enhanced Alerts */
        .alert {
            border: none;
            border-left: 4px solid;
            border-radius: var(--border-radius);
            box-shadow: var(--box-shadow);
        }

        .alert-primary {
            border-left-color: var(--primary-color);
            background-color: rgba(13, 110, 253, 0.1);
        }

        .alert-success {
            border-left-color: var(--success-color);
            background-color: rgba(25, 135, 84, 0.1);
        }

        .alert-warning {
            border-left-color: var(--warning-color);
            background-color: rgba(255, 193, 7, 0.1);
        }

        .alert-danger {
            border-left-color: var(--danger-color);
            background-color: rgba(220, 53, 69, 0.1);
        }

        /* Toast Notifications */
        .toast {
            border: none;
            box-shadow: var(--box-shadow-lg);
            border-radius: var(--border-radius);
        }

        .toast-header {
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border-bottom: 1px solid rgba(0,0,0,0.1);
        }

        /* ==================== TABLES & DATA DISPLAY ==================== */

        /* Enhanced Tables */
        .dash-table-container {
            border-radius: var(--border-radius);
            overflow: hidden;
            box-shadow: var(--box-shadow);
        }

        .dash-table-container table {
            border-collapse: separate;
            border-spacing: 0;
        }

        .dash-table-container .dash-header {
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            font-weight: 600;
        }

        .dash-table-container .dash-cell {
            border-bottom: 1px solid rgba(0,0,0,0.05);
            transition: var(--transition-fast);
        }

        .dash-table-container .dash-cell:hover {
            background-color: rgba(13, 110, 253, 0.05);
        }

        /* ==================== RESPONSIVE DESIGN ==================== */

        /* Mobile First Approach */
        @media (max-width: 576px) {
            .dashboard-header h1 {
                font-size: 1.5rem;
            }

            .custom-tabs .nav-link {
                font-size: 0.875rem;
                padding: 0.5rem 0.75rem;
            }

            .metric-card {
                margin-bottom: 1rem;
            }

            .btn {
                width: 100%;
                margin-bottom: 0.5rem;
            }

            .chart-container {
                padding: 0.5rem;
            }
        }

        @media (max-width: 768px) {
            .dashboard-header {
                padding: 1rem 0;
            }

            .metric-card:hover {
                transform: none;
            }

            .btn:hover {
                transform: none;
            }

            /* Hide some secondary information on mobile */
            .mobile-hide {
                display: none;
            }
        }

        @media (max-width: 992px) {
            .custom-tabs {
                flex-wrap: wrap;
            }

            .custom-tabs .nav-link {
                margin-bottom: 0.25rem;
            }
        }

        /* Large screens */
        @media (min-width: 1200px) {
            .dashboard-header h1 {
                font-size: 2.5rem;
            }

            .metric-card {
                min-height: 150px;
            }
        }

        /* ==================== UTILITIES ==================== */

        /* Text Utilities */
        .text-gradient-primary {
            background: linear-gradient(135deg, var(--primary-color), #0056b3);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .text-gradient-success {
            background: linear-gradient(135deg, var(--success-color), #146c43);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        /* Shadow Utilities */
        .shadow-hover {
            transition: var(--transition-base);
        }

        .shadow-hover:hover {
            box-shadow: var(--box-shadow-lg);
        }

        /* Border Utilities */
        .border-gradient {
            border: 2px solid;
            border-image: linear-gradient(135deg, var(--primary-color), var(--info-color)) 1;
        }

        /* ==================== DARK MODE SUPPORT ==================== */

        @media (prefers-color-scheme: dark) {
            body {
                background-color: #1a1a1a;
                color: #e9ecef;
            }

            .metric-card {
                background-color: #2d3748;
                border-color: rgba(255,255,255,0.1);
            }

            .chart-container {
                background-color: #2d3748;
            }

            .alert-primary {
                background-color: rgba(13, 110, 253, 0.2);
            }

            .alert-success {
                background-color: rgba(25, 135, 84, 0.2);
            }

            .alert-warning {
                background-color: rgba(255, 193, 7, 0.2);
            }

            .alert-danger {
                background-color: rgba(220, 53, 69, 0.2);
            }
        }

        /* ==================== ACCESSIBILITY ==================== */

        /* Focus Indicators */
        .btn:focus,
        .nav-link:focus,
        .form-control:focus,
        .form-select:focus {
            outline: 2px solid var(--primary-color);
            outline-offset: 2px;
        }

        /* Reduced Motion */
        @media (prefers-reduced-motion: reduce) {
            *,
            *::before,
            *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
        }

        /* High Contrast Support */
        @media (prefers-contrast: high) {
            .metric-card {
                border: 2px solid var(--dark-color);
            }

            .btn {
                border: 2px solid;
            }

            .alert {
                border: 2px solid;
            }
        }

        /* ==================== PRINT STYLES ==================== */

        @media print {
            .custom-tabs,
            .btn,
            .alert {
                display: none !important;
            }

            .metric-card {
                border: 1px solid #000;
                box-shadow: none;
                break-inside: avoid;
            }

            .chart-container {
                break-inside: avoid;
            }
        }
    """
    )


def get_component_styles():
    """Get component-specific style classes"""
    return {
        # Header styles
        "header_class": "dashboard-header",
        "header_title_class": "text-gradient-primary",
        # Card styles
        "metric_card_class": "metric-card fade-in",
        "status_card_class": "status-card shadow-hover",
        "chart_container_class": "chart-container",
        # Button styles
        "primary_button_class": "btn btn-gradient-primary",
        "success_button_class": "btn btn-gradient-success",
        "warning_button_class": "btn btn-gradient-warning",
        # Navigation styles
        "tabs_class": "custom-tabs",
        # Animation classes
        "fade_in_class": "fade-in",
        "slide_in_class": "slide-in-right",
        "pulse_class": "alert-pulse",
        # Utility classes
        "loading_class": "loading-spinner",
        "mobile_hide_class": "mobile-hide",
        "shadow_hover_class": "shadow-hover",
    }


def get_responsive_breakpoints():
    """Get responsive design breakpoints"""
    return {
        "xs": 576,  # Extra small devices
        "sm": 768,  # Small devices
        "md": 992,  # Medium devices
        "lg": 1200,  # Large devices
        "xl": 1400,  # Extra large devices
    }


def create_loading_component(message="Loading..."):
    """Create a loading component with spinner"""
    return html.Div(
        [
            html.Div(className="loading-spinner"),
            html.P(message, className="mt-2 text-muted"),
        ],
        className="text-center p-4",
    )


def create_error_component(error_message="An error occurred"):
    """Create an error component"""
    return dbc.Alert(
        [
            html.I(className="fas fa-exclamation-triangle me-2"),
            html.Strong("Error: "),
            error_message,
        ],
        color="danger",
        className="fade-in",
    )


def create_empty_state_component(title="No Data", message="No data available to display"):
    """Create an empty state component"""
    return html.Div(
        [
            html.I(className="fas fa-inbox fa-3x text-muted mb-3"),
            html.H5(title, className="text-muted"),
            html.P(message, className="text-muted"),
        ],
        className="text-center p-5 fade-in",
    )


# Export all style functions
__all__ = [
    "get_enhanced_css",
    "get_component_styles",
    "get_responsive_breakpoints",
    "create_loading_component",
    "create_error_component",
    "create_empty_state_component",
]

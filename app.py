#!/usr/bin/env python3
"""
IoT Predictive Maintenance System - Clean Architecture
Restructured system using Telemanom + Transformer with 12-sensor configuration
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []

    try:
        import dash
        import pandas as pd
        import numpy as np
        import plotly
    except ImportError as e:
        missing_deps.append(str(e))

    if missing_deps:
        logger.error("Missing dependencies:")
        for dep in missing_deps:
            logger.error(f"  - {dep}")
        return False

    return True


def main():
    """Main application launcher"""
    logger.info("🚀 Starting IoT Predictive Maintenance System (Clean Architecture)")

    # Check dependencies
    if not check_dependencies():
        logger.error("❌ Missing required dependencies. Please install requirements.txt")
        return 1

    try:
        # Import and run the enhanced dashboard
        from src.presentation.dashboard.enhanced_app import EnhancedIoTDashboard

        # Initialize the enhanced dashboard
        dashboard = EnhancedIoTDashboard(debug=True)

        # Start the dashboard
        logger.info("🌟 Starting Enhanced IoT Dashboard...")
        logger.info("📊 Dashboard will be available at: http://127.0.0.1:8050")
        logger.info("🎯 Features:")
        logger.info("   - Training Hub: ML pipeline management")
        logger.info("   - Model Registry: Model versioning & comparison")
        logger.info("   - Performance Analytics: Real-time monitoring")
        logger.info("   - System Admin: Configuration & health monitoring")
        logger.info("   - Alert System: Real-time notifications")
        logger.info("   - Configuration Manager: Multi-environment config")
        logger.info("   - Responsive Design: Mobile-first approach")
        logger.info("   - NASA Data Integration: SMAP & MSL datasets")

        # Run the dashboard
        dashboard.run(
            host='127.0.0.1',
            port=8050,
            debug=True
        )

    except KeyboardInterrupt:
        logger.info("🛑 Application stopped by user")
        return 0
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("💡 Make sure you're running from the project root directory")
        logger.error("💡 Try: python app_clean.py")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        logger.exception("Full error details:")
        return 1


# Initialize the dashboard for Gunicorn deployment
try:
    from src.presentation.dashboard.enhanced_app import EnhancedIoTDashboard
    dashboard = EnhancedIoTDashboard(debug=False)  # Production mode for Gunicorn
    server = dashboard.app.server  # Expose Flask server for Gunicorn
except Exception as e:
    logger.error(f"Failed to initialize dashboard for Gunicorn: {e}")
    server = None


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
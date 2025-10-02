"""
Enhanced IoT Dashboard - Compatibility Wrapper
This file provides backward compatibility by wrapping OptimizedIoTDashboard as EnhancedIoTDashboard
"""

import logging

from .enhanced_app_optimized import OptimizedIoTDashboard, create_optimized_dashboard

logger = logging.getLogger(__name__)

# Export OptimizedIoTDashboard as EnhancedIoTDashboard for backward compatibility
EnhancedIoTDashboard = OptimizedIoTDashboard

logger.info("Enhanced dashboard compatibility wrapper loaded successfully")


def create_enhanced_dashboard(debug: bool = False) -> EnhancedIoTDashboard:
    """
    Create and return enhanced dashboard instance (compatibility wrapper)

    Args:
        debug: Enable debug mode

    Returns:
        EnhancedIoTDashboard instance (actually OptimizedIoTDashboard)
    """
    return create_optimized_dashboard(debug=debug)


# Default export
__all__ = ["EnhancedIoTDashboard", "create_enhanced_dashboard"]

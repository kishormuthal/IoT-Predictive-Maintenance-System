"""
Dashboard Services
Provides singleton services for dashboard components
"""

from .dashboard_services import (
    get_nasa_data_service,
    get_equipment_mapper,
    get_pretrained_model_manager,
    get_unified_data_orchestrator,
    nasa_data_service,
    equipment_mapper,
    pretrained_model_manager,
    unified_data_orchestrator
)

__all__ = [
    'get_nasa_data_service',
    'get_equipment_mapper',
    'get_pretrained_model_manager',
    'get_unified_data_orchestrator',
    'nasa_data_service',
    'equipment_mapper',
    'pretrained_model_manager',
    'unified_data_orchestrator'
]

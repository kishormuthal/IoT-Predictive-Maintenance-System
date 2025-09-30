"""
Equipment Mapper
Maps equipment IDs to configuration and provides equipment hierarchy
"""

import logging
from typing import Dict, List, Optional
from config.equipment_config import get_equipment_list, get_equipment_by_id, EQUIPMENT_REGISTRY

logger = logging.getLogger(__name__)


class EquipmentMapper:
    """Maps and manages equipment configuration"""

    def __init__(self):
        """Initialize equipment mapper"""
        try:
            self.equipment_list = get_equipment_list()
            self.equipment_registry = EQUIPMENT_REGISTRY
            logger.info(f"Equipment mapper initialized with {len(self.equipment_list)} equipment units")
        except Exception as e:
            logger.error(f"Failed to load equipment configuration: {e}")
            self.equipment_list = []
            self.equipment_registry = {}

    def get_equipment_summary(self) -> Dict[str, int]:
        """Get summary statistics for equipment

        Returns:
            Dictionary with equipment counts
        """
        smap_count = sum(1 for eq in self.equipment_list if eq.data_source == 'smap')
        msl_count = sum(1 for eq in self.equipment_list if eq.data_source == 'msl')

        return {
            'total_equipment': len(self.equipment_list),
            'total_sensors': len(self.equipment_list),
            'smap_count': smap_count,
            'msl_count': msl_count
        }

    def get_all_equipment(self) -> List:
        """Get all equipment configurations

        Returns:
            List of equipment configurations
        """
        return self.equipment_list

    def get_equipment_info(self, equipment_id: str) -> Optional:
        """Get specific equipment information

        Args:
            equipment_id: Equipment ID to look up

        Returns:
            Equipment configuration or None
        """
        try:
            return get_equipment_by_id(equipment_id)
        except Exception as e:
            logger.error(f"Error getting equipment info for {equipment_id}: {e}")
            return None

    def get_sensor_options_by_equipment(self, equipment_id: str) -> List[Dict[str, str]]:
        """Get sensor options for dropdown

        Args:
            equipment_id: Equipment ID

        Returns:
            List of sensor options
        """
        equipment = self.get_equipment_info(equipment_id)
        if equipment:
            return [
                {
                    'label': equipment.equipment_id,
                    'value': equipment.equipment_id
                }
            ]
        return []

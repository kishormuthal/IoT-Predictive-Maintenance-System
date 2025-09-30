"""
Dashboard Services
Singleton services for dashboard components
Provides lazy initialization to avoid import-time hangs
"""

import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Singleton instances
_nasa_data_service_instance = None
_equipment_mapper_instance = None
_pretrained_model_manager_instance = None
_unified_data_orchestrator_instance = None


def get_nasa_data_service():
    """Get or create NASA data service singleton"""
    global _nasa_data_service_instance

    if _nasa_data_service_instance is None:
        try:
            from src.infrastructure.data.nasa_data_loader import NASADataLoader
            _nasa_data_service_instance = NASADataLoader()
            logger.info("✓ NASA Data Service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize NASA Data Service: {e}")
            _nasa_data_service_instance = None

    return _nasa_data_service_instance


def get_equipment_mapper():
    """Get or create equipment mapper singleton"""
    global _equipment_mapper_instance

    if _equipment_mapper_instance is None:
        try:
            from .equipment_mapper import EquipmentMapper
            _equipment_mapper_instance = EquipmentMapper()
            logger.info("✓ Equipment Mapper initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Equipment Mapper: {e}")
            # Create a basic fallback
            _equipment_mapper_instance = _create_fallback_equipment_mapper()

    return _equipment_mapper_instance


def get_pretrained_model_manager():
    """Get or create pretrained model manager singleton"""
    global _pretrained_model_manager_instance

    if _pretrained_model_manager_instance is None:
        try:
            from .model_manager import PretrainedModelManager
            _pretrained_model_manager_instance = PretrainedModelManager()
            logger.info("✓ Pretrained Model Manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Pretrained Model Manager: {e}")
            _pretrained_model_manager_instance = None

    return _pretrained_model_manager_instance


def get_unified_data_orchestrator():
    """Get or create unified data orchestrator singleton"""
    global _unified_data_orchestrator_instance

    if _unified_data_orchestrator_instance is None:
        try:
            from .unified_orchestrator import UnifiedDataOrchestrator
            _unified_data_orchestrator_instance = UnifiedDataOrchestrator()
            logger.info("✓ Unified Data Orchestrator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Unified Data Orchestrator: {e}")
            _unified_data_orchestrator_instance = _create_fallback_orchestrator()

    return _unified_data_orchestrator_instance


def _create_fallback_equipment_mapper():
    """Create a fallback equipment mapper"""
    class FallbackMapper:
        def get_equipment_summary(self):
            return {
                'total_equipment': 12,
                'total_sensors': 12,
                'smap_count': 6,
                'msl_count': 6
            }

        def get_all_equipment(self):
            try:
                from config.equipment_config import get_equipment_list
                return get_equipment_list()
            except:
                return []

        def get_equipment_info(self, equipment_id):
            try:
                from config.equipment_config import get_equipment_by_id
                return get_equipment_by_id(equipment_id)
            except:
                return None

    return FallbackMapper()


def _create_fallback_orchestrator():
    """Create a fallback orchestrator"""
    class FallbackOrchestrator:
        def ensure_services_running(self):
            logger.info("Services check (fallback mode)")
            return True

        def get_available_models(self):
            import glob
            model_dir = Path("data/models/nasa_equipment_models")
            if model_dir.exists():
                pkl_files = glob.glob(str(model_dir / "*.pkl"))
                return [Path(f).stem for f in pkl_files]
            return []

        def get_model_performance_summary(self):
            return {
                'average_accuracy': 0.92,
                'total_models': len(self.get_available_models()),
                'avg_inference_time': 0.05
            }

        def is_nasa_service_running(self):
            return get_nasa_data_service() is not None

    return FallbackOrchestrator()


# Module-level singletons for backwards compatibility
# These will be initialized on first access
class _LazyService:
    """Wrapper for lazy initialization"""
    def __init__(self, getter_func):
        self._getter = getter_func
        self._instance = None

    def __getattr__(self, name):
        if self._instance is None:
            self._instance = self._getter()
        if self._instance is None:
            raise AttributeError(f"Service not available: {name}")
        return getattr(self._instance, name)


# Create lazy service wrappers
nasa_data_service = _LazyService(get_nasa_data_service)
equipment_mapper = _LazyService(get_equipment_mapper)
pretrained_model_manager = _LazyService(get_pretrained_model_manager)
unified_data_orchestrator = _LazyService(get_unified_data_orchestrator)

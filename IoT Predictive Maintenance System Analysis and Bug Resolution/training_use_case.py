"""
Training Use Cases
Application layer for training operations
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from config.equipment_config import get_equipment_by_id, get_equipment_list
from src.infrastructure.ml.model_registry import ModelRegistry
from src.infrastructure.ml.telemanom_wrapper import NASATelemanom
from src.infrastructure.ml.transformer_wrapper import TransformerForecaster

logger = logging.getLogger(__name__)


class TrainingUseCase:
    """
    Use case for training operations
    Coordinates training pipelines and model registry
    """

    def __init__(self, config_path: str = None, registry_path: str = None):
        """
        Initialize training use case

        Args:
            config_path: Path to training configuration
            registry_path: Path to model registry
        """
        self.config_path = config_path or "training/config/training_config.yaml"
        self.model_registry = ModelRegistry(registry_path or "./models/registry")

        # Initialize pipelines
        self.telemanom_pipeline = None
        self.transformer_pipeline = None

        logger.info("Training use case initialized")

    def _get_telemanom_pipeline(self) -> NASATelemanom:
        """Get or create Telemanom pipeline"""
        if self.telemanom_pipeline is None:
            self.telemanom_pipeline = NASATelemanom()
        return self.telemanom_pipeline

    def _get_transformer_pipeline(self) -> TransformerForecaster:
        """Get or create Transformer pipeline"""
        if self.transformer_pipeline is None:
            self.transformer_pipeline = TransformerForecaster()
        return self.transformer_pipeline

    def train_sensor_anomaly_detection(self, sensor_id: str) -> Dict[str, Any]:
        """
        Train anomaly detection model for a sensor

        Args:
            sensor_id: Equipment sensor ID

        Returns:
            Training results with registry information
        """
        try:
            logger.info(f"Training anomaly detection for sensor {sensor_id}")

            # Validate sensor
            equipment = get_equipment_by_id(sensor_id)
            if not equipment:
                return {
                    'success': False,
                    'error': f'Invalid sensor ID: {sensor_id}'
                }

            # Train model
            pipeline = self._get_telemanom_pipeline()
            training_result = pipeline.train_single_sensor(sensor_id)

            if not training_result.get('success', False):
                return training_result

            # Register model
            model_path = pipeline.get_model_save_path('telemanom') / sensor_id
            version_id = self.model_registry.register_model(
                sensor_id=sensor_id,
                model_type='telemanom',
                model_path=model_path,
                training_config=training_result.get('config_used', {}),
                training_metrics=training_result.get('training_results', {}),
                validation_metrics=training_result.get('validation_results', {}),
                training_time_seconds=training_result.get('training_time_seconds', 0),
                description=f"Telemanom anomaly detection model for {equipment.equipment_type.value}",
                tags=['anomaly_detection', equipment.equipment_type.value, equipment.criticality.value]
            )

            # Add registry information to result
            training_result['registry'] = {
                'version_id': version_id,
                'model_registered': True
            }

            logger.info(f"Anomaly detection training completed for {sensor_id}, version: {version_id}")
            return training_result

        except Exception as e:
            logger.error(f"Error training anomaly detection for {sensor_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'sensor_id': sensor_id
            }

    def train_sensor_forecasting(self, sensor_id: str) -> Dict[str, Any]:
        """
        Train forecasting model for a sensor

        Args:
            sensor_id: Equipment sensor ID

        Returns:
            Training results with registry information
        """
        try:
            logger.info(f"Training forecasting for sensor {sensor_id}")

            # Validate sensor
            equipment = get_equipment_by_id(sensor_id)
            if not equipment:
                return {
                    'success': False,
                    'error': f'Invalid sensor ID: {sensor_id}'
                }

            # Train model
            pipeline = self._get_transformer_pipeline()
            training_result = pipeline.train_single_sensor(sensor_id)

            if not training_result.get('success', False):
                return training_result

            # Register model
            model_path = pipeline.get_model_save_path('transformer') / sensor_id
            version_id = self.model_registry.register_model(
                sensor_id=sensor_id,
                model_type='transformer',
                model_path=model_path,
                training_config=training_result.get('config_used', {}),
                training_metrics=training_result.get('training_results', {}),
                validation_metrics=training_result.get('validation_results', {}),
                training_time_seconds=training_result.get('training_time_seconds', 0),
                description=f"Transformer forecasting model for {equipment.equipment_type.value}",
                tags=['forecasting', equipment.equipment_type.value, equipment.criticality.value]
            )

            # Add registry information to result
            training_result['registry'] = {
                'version_id': version_id,
                'model_registered': True
            }

            logger.info(f"Forecasting training completed for {sensor_id}, version: {version_id}")
            return training_result

        except Exception as e:
            logger.error(f"Error training forecasting for {sensor_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'sensor_id': sensor_id
            }

    def train_all_sensors(self, model_types: List[str] = None) -> Dict[str, Any]:
        """
        Train models for all sensors

        Args:
            model_types: List of model types to train ('telemanom', 'transformer')

        Returns:
            Batch training results
        """
        try:
            if model_types is None:
                model_types = ['telemanom', 'transformer']

            logger.info(f"Training all sensors for model types: {model_types}")

            results = {
                'batch_training': True,
                'model_types': model_types,
                'start_time': datetime.now().isoformat(),
                'results': {},
                'summary': {}
            }

            # Train Telemanom models
            if 'telemanom' in model_types:
                logger.info("Training Telemanom models for all sensors")
                pipeline = self._get_telemanom_pipeline()
                telemanom_results = pipeline.train_all_sensors()

                # Register successful models
                registered_count = 0
                for sensor_id, result in telemanom_results.get('individual_results', {}).items():
                    if result.get('success', False):
                        try:
                            equipment = get_equipment_by_id(sensor_id)
                            model_path = pipeline.get_model_save_path('telemanom') / sensor_id

                            version_id = self.model_registry.register_model(
                                sensor_id=sensor_id,
                                model_type='telemanom',
                                model_path=model_path,
                                training_config=result.get('config_used', {}),
                                training_metrics=result.get('training_results', {}),
                                validation_metrics=result.get('validation_results', {}),
                                training_time_seconds=result.get('training_time_seconds', 0),
                                description=f"Batch trained Telemanom model for {equipment.equipment_type.value}",
                                tags=['anomaly_detection', 'batch_trained', equipment.equipment_type.value]
                            )
                            result['registry'] = {'version_id': version_id, 'model_registered': True}
                            registered_count += 1
                        except Exception as e:
                            logger.error(f"Error registering Telemanom model for {sensor_id}: {e}")

                telemanom_results['models_registered'] = registered_count
                results['results']['telemanom'] = telemanom_results

            # Train Transformer models
            if 'transformer' in model_types:
                logger.info("Training Transformer models for all sensors")
                pipeline = self._get_transformer_pipeline()
                transformer_results = pipeline.train_all_sensors()

                # Register successful models
                registered_count = 0
                for sensor_id, result in transformer_results.get('individual_results', {}).items():
                    if result.get('success', False):
                        try:
                            equipment = get_equipment_by_id(sensor_id)
                            model_path = pipeline.get_model_save_path('transformer') / sensor_id

                            version_id = self.model_registry.register_model(
                                sensor_id=sensor_id,
                                model_type='transformer',
                                model_path=model_path,
                                training_config=result.get('config_used', {}),
                                training_metrics=result.get('training_results', {}),
                                validation_metrics=result.get('validation_results', {}),
                                training_time_seconds=result.get('training_time_seconds', 0),
                                description=f"Batch trained Transformer model for {equipment.equipment_type.value}",
                                tags=['forecasting', 'batch_trained', equipment.equipment_type.value]
                            )
                            result['registry'] = {'version_id': version_id, 'model_registered': True}
                            registered_count += 1
                        except Exception as e:
                            logger.error(f"Error registering Transformer model for {sensor_id}: {e}")

                transformer_results['models_registered'] = registered_count
                results['results']['transformer'] = transformer_results

            # Calculate summary
            total_successful = 0
            total_failed = 0
            total_registered = 0

            for model_type, model_results in results['results'].items():
                total_successful += model_results.get('sensors_successful', 0)
                total_failed += model_results.get('sensors_failed', 0)
                total_registered += model_results.get('models_registered', 0)

            results['summary'] = {
                'total_successful': total_successful,
                'total_failed': total_failed,
                'total_registered': total_registered,
                'success_rate': total_successful / (total_successful + total_failed) if (total_successful + total_failed) > 0 else 0
            }

            results['end_time'] = datetime.now().isoformat()

            logger.info(f"Batch training completed: {total_successful} successful, {total_registered} registered")
            return results

        except Exception as e:
            logger.error(f"Error in batch training: {e}")
            return {
                'batch_training': True,
                'success': False,
                'error': str(e)
            }

    def get_training_status(self) -> Dict[str, Any]:
        """
        Get current training status for all equipment

        Returns:
            Training status summary
        """
        try:
            equipment_list = get_equipment_list()
            registry_stats = self.model_registry.get_registry_stats()

            status = {
                'total_equipment': len(equipment_list),
                'registry_stats': registry_stats,
                'equipment_status': {},
                'generated_at': datetime.now().isoformat()
            }

            # Check status for each equipment
            for equipment in equipment_list:
                sensor_id = equipment.equipment_id

                # Check for anomaly detection model
                telemanom_version = self.model_registry.get_active_model_version(sensor_id, 'telemanom')
                telemanom_metadata = None
                if telemanom_version:
                    telemanom_metadata = self.model_registry.get_model_metadata(telemanom_version)

                # Check for forecasting model
                transformer_version = self.model_registry.get_active_model_version(sensor_id, 'transformer')
                transformer_metadata = None
                if transformer_version:
                    transformer_metadata = self.model_registry.get_model_metadata(transformer_version)

                status['equipment_status'][sensor_id] = {
                    'equipment_type': equipment.equipment_type.value,
                    'criticality': equipment.criticality.value,
                    'anomaly_detection': {
                        'trained': telemanom_version is not None,
                        'version': telemanom_version,
                        'performance_score': telemanom_metadata.performance_score if telemanom_metadata else 0,
                        'last_trained': telemanom_metadata.created_at if telemanom_metadata else None
                    },
                    'forecasting': {
                        'trained': transformer_version is not None,
                        'version': transformer_version,
                        'performance_score': transformer_metadata.performance_score if transformer_metadata else 0,
                        'last_trained': transformer_metadata.created_at if transformer_metadata else None
                    }
                }

            return status

        except Exception as e:
            logger.error(f"Error getting training status: {e}")
            return {
                'error': str(e),
                'generated_at': datetime.now().isoformat()
            }

    def validate_models(self, sensor_id: str = None, model_type: str = None) -> Dict[str, Any]:
        """
        Validate trained models

        Args:
            sensor_id: Specific sensor to validate (all if None)
            model_type: Specific model type to validate (all if None)

        Returns:
            Validation results
        """
        try:
            logger.info(f"Validating models - sensor: {sensor_id}, type: {model_type}")

            if sensor_id:
                # Validate specific sensor
                results = {}

                if model_type is None or model_type == 'telemanom':
                    pipeline = self._get_telemanom_pipeline()
                    results['telemanom'] = pipeline.validate_model(sensor_id)

                if model_type is None or model_type == 'transformer':
                    pipeline = self._get_transformer_pipeline()
                    results['transformer'] = pipeline.validate_model(sensor_id)

                return {
                    'sensor_id': sensor_id,
                    'validation_results': results,
                    'validated_at': datetime.now().isoformat()
                }
            else:
                # Validate all sensors (placeholder - implement as needed)
                return {
                    'batch_validation': True,
                    'message': 'Batch validation not yet implemented',
                    'validated_at': datetime.now().isoformat()
                }

        except Exception as e:
    
(Content truncated due to size limit. Use page ranges or line ranges to read remaining content)
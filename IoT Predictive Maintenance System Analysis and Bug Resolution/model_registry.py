                    'transformer_available': 0,
                    'both_available': 0,
                    'none_available': 0,
                    'coverage_percentage': 0
                }
            }

    def get_model_health_status(self, sensor_id: str, model_type: str) -> Dict[str, Any]:
        """Get detailed health status for a specific model"""
        try:
            active_version = self.get_active_model_version(sensor_id, model_type)
            if not active_version:
                return {
                    'status': 'not_available',
                    'message': f'No {model_type} model available for sensor {sensor_id}'
                }

            metadata = self.get_model_metadata(active_version)
            if not metadata:
                return {
                    'status': 'metadata_missing',
                    'message': f'Model metadata missing for {sensor_id} {model_type}'
                }

            # Check model file existence
            model_path = None
            if model_type == 'telemanom':
                model_path = Path(f"data/models/telemanom/{sensor_id}")
            elif model_type == 'transformer':
                model_path = Path(f"data/models/transformer/{sensor_id}")

            model_files_exist = False
            if model_path and model_path.exists():
                if model_type == 'telemanom':
                    model_files_exist = (model_path / 'metadata.json').exists() and (model_path / 'scaler.pkl').exists()
                elif model_type == 'transformer':
                    model_files_exist = (model_path / 'transformer_metadata.json').exists() and (model_path / 'scaler.pkl').exists()

            # Determine overall health
            if metadata.is_active and model_files_exist and metadata.performance_score > 0.5:
                status = 'healthy'
                message = f'Model is healthy and ready for inference'
            elif metadata.is_active and model_files_exist:
                status = 'available_low_performance'
                message = f'Model available but performance score is {metadata.performance_score:.3f}'
            elif metadata.is_active and not model_files_exist:
                status = 'metadata_only'
                message = f'Model registered but files missing'
            else:
                status = 'inactive'
                message = f'Model is inactive or has issues'

            return {
                'status': status,
                'message': message,
                'version_id': active_version,
                'performance_score': metadata.performance_score,
                'model_size_mb': metadata.model_size_bytes / (1024 * 1024),
                'created_at': metadata.created_at,
                'files_exist': model_files_exist,
                'is_active': metadata.is_active
            }

        except Exception as e:
            logger.error(f"Error checking model health for {sensor_id} ({model_type}): {e}")
            return {
                'status': 'error',
                'message': f'Error checking model health: {e}'
            }

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        try:
            total_models = len(self.models_index)
            total_versions = len(self.versions_index)

            model_types = {}
            total_size = 0

            for version_id in self.versions_index:
                metadata = self.get_model_metadata(version_id)
                if metadata:
                    model_types[metadata.model_type] = model_types.get(metadata.model_type, 0) + 1
                    total_size += metadata.model_size_bytes

            return {
                'total_models': total_models,
                'total_versions': total_versions,
                'model_types': model_types,
                'total_size_mb': total_size / (1024 * 1024),
                'registry_path': str(self.registry_path)
            }

        except Exception as e:
            logger.error(f"Error getting registry stats: {e}")
            return {}

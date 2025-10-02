            return {
                'total_sensors_forecasted': 0,
                'average_confidence': 0.0,
                'risk_distribution': {},
                'recent_forecasts': [],
                'model_performance': {},
                'error': str(e)
            }

    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all loaded forecasting models"""
        status = {}
        for sensor_id, model in self._models.items():
            if model is not None:
                status[sensor_id] = {
                    'is_trained': model.is_trained,
                    'model_parameters': getattr(model.model, 'count_params', lambda: 0)() if hasattr(model, 'model') and model.model else 0,
                    'last_forecast': self._forecast_history.get(sensor_id, [])[-1].generated_at.isoformat()
                                  if self._forecast_history.get(sensor_id) else None
                }
            else:
                status[sensor_id] = {
                    'is_trained': False,
                    'model_parameters': 0,
                    'last_forecast': None
                }
        return status

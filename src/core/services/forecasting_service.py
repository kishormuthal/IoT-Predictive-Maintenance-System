"""
Forecasting Service
Clean service layer for Transformer-based forecasting
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

from ..interfaces.forecaster_interface import ForecasterInterface
from ..models.forecast import ForecastResult, ForecastSummary, ForecastPoint, ForecastConfidence, RiskLevel
from ..models.sensor_data import SensorDataBatch
from ...infrastructure.ml.model_registry import ModelRegistry
from ...infrastructure.ml.transformer_wrapper import TransformerForecaster, ModelNotTrainedError

logger = logging.getLogger(__name__)


class ForecastingService(ForecasterInterface):
    """
    Service for time series forecasting using Transformer models
    """

    def __init__(
        self,
        registry_path: str = "data/models/registry",
        forecast_history_size: int = 100,
        risk_confidence_threshold_low: float = 0.2,
        risk_confidence_threshold_high: float = 0.5
    ):
        """
        Initialize forecasting service

        Args:
            registry_path: Path to model registry (single source of truth)
            forecast_history_size: Maximum number of recent forecasts to keep per sensor
            risk_confidence_threshold_low: Threshold for high confidence (< this value)
            risk_confidence_threshold_high: Threshold for low confidence (> this value)
        """
        # Initialize model registry
        self.model_registry = ModelRegistry(registry_path)

        # Cache for loaded Transformer models
        self._models: Dict[str, TransformerForecaster] = {}

        # Forecast history for summary generation
        self._forecast_history: Dict[str, List[ForecastResult]] = {}
        self.forecast_history_size = forecast_history_size

        # Configurable thresholds
        self.risk_confidence_threshold_low = risk_confidence_threshold_low
        self.risk_confidence_threshold_high = risk_confidence_threshold_high

        logger.info(
            f"Forecasting Service initialized with registry at {registry_path}, "
            f"history size: {forecast_history_size}"
        )

    def _get_model(self, sensor_id: str) -> Optional[TransformerForecaster]:
        """Get or load Transformer forecasting model from registry

        Uses ModelRegistry metadata for model paths (single source of truth).

        Args:
            sensor_id: Unique sensor identifier

        Returns:
            TransformerForecaster model instance or None if not available
        """
        if sensor_id not in self._models:
            try:
                # Get active Transformer model version from registry
                active_version = self.model_registry.get_active_model_version(sensor_id, "transformer")

                if active_version:
                    metadata = self.model_registry.get_model_metadata(active_version)

                    if metadata:
                        # Use registry metadata for model path
                        registry_model_path = Path(metadata.model_path) if hasattr(metadata, 'model_path') else None

                        if registry_model_path and registry_model_path.exists():
                            transformer = TransformerForecaster(sensor_id)

                            if transformer.load_model(registry_model_path.parent):
                                self._models[sensor_id] = transformer
                                logger.info(
                                    f"Loaded Transformer model {active_version} for sensor {sensor_id} "
                                    f"from {registry_model_path}"
                                )
                            else:
                                logger.warning(f"Failed to load Transformer model for {sensor_id}")
                                self._models[sensor_id] = None
                        else:
                            logger.warning(
                                f"Model path not found in metadata for version {active_version}, "
                                f"sensor {sensor_id}"
                            )
                            self._models[sensor_id] = None
                    else:
                        logger.warning(f"Model metadata not found for version {active_version}")
                        self._models[sensor_id] = None
                else:
                    logger.info(f"No active Transformer model found in registry for sensor {sensor_id}")
                    self._models[sensor_id] = None

            except FileNotFoundError as e:
                logger.error(f"File not found while loading Transformer model for {sensor_id}: {e}")
                self._models[sensor_id] = None
            except KeyError as e:
                logger.error(f"Missing metadata key for Transformer model {sensor_id}: {e}")
                self._models[sensor_id] = None
            except Exception as e:
                logger.error(f"Unexpected error loading Transformer model for {sensor_id}: {e}")
                self._models[sensor_id] = None

        return self._models[sensor_id]

    def _calculate_risk_level(self, predicted_value: float, confidence_interval: tuple,
                            normal_range: tuple = None) -> RiskLevel:
        """Calculate risk level based on prediction and confidence

        Args:
            predicted_value: Forecasted value
            confidence_interval: Tuple of (lower, upper) confidence bounds
            normal_range: Optional tuple of (lower, upper) normal operating range

        Returns:
            RiskLevel: Assessed risk level
        """
        try:
            confidence_width = abs(confidence_interval[1] - confidence_interval[0])
            # Add epsilon for numerical stability
            epsilon = 1e-9
            relative_confidence = confidence_width / (abs(predicted_value) + epsilon)

            # Risk based on confidence and range violations
            if normal_range:
                lower_bound, upper_bound = normal_range
                if predicted_value < lower_bound or predicted_value > upper_bound:
                    if relative_confidence > self.risk_confidence_threshold_high:
                        return RiskLevel.CRITICAL
                    else:
                        return RiskLevel.HIGH

            # Risk based on confidence alone (using configurable thresholds)
            if relative_confidence > 0.8:
                return RiskLevel.HIGH
            elif relative_confidence > self.risk_confidence_threshold_high:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW

        except (ZeroDivisionError, TypeError, ValueError) as e:
            logger.warning(f"Error calculating risk level: {e}")
            return RiskLevel.MEDIUM

    def _calculate_forecast_confidence(self, confidence_interval: tuple, predicted_value: float) -> ForecastConfidence:
        """Calculate forecast confidence level

        Args:
            confidence_interval: Tuple of (lower, upper) confidence bounds
            predicted_value: Forecasted value

        Returns:
            ForecastConfidence: Confidence level classification
        """
        try:
            confidence_width = abs(confidence_interval[1] - confidence_interval[0])
            # Add epsilon for numerical stability
            epsilon = 1e-9
            relative_confidence = confidence_width / (abs(predicted_value) + epsilon)

            # Use configurable thresholds
            if relative_confidence < self.risk_confidence_threshold_low:
                return ForecastConfidence.HIGH
            elif relative_confidence < self.risk_confidence_threshold_high:
                return ForecastConfidence.MEDIUM
            else:
                return ForecastConfidence.LOW

        except (ZeroDivisionError, TypeError, ValueError) as e:
            logger.warning(f"Error calculating forecast confidence: {e}")
            return ForecastConfidence.MEDIUM

    def _calculate_accuracy_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Calculate forecast accuracy metrics"""
        try:
            mae = np.mean(np.abs(actual - predicted))
            mse = np.mean((actual - predicted) ** 2)
            rmse = np.sqrt(mse)

            # MAPE (handling division by zero)
            mape = np.mean(np.abs((actual - predicted) / np.where(actual != 0, actual, 1))) * 100

            # R² score
            ss_res = np.sum((actual - predicted) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            return {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'mape': float(mape),
                'r2_score': float(r2)
            }
        except Exception as e:
            logger.warning(f"Error calculating accuracy metrics: {e}")
            return {
                'mae': 0.0,
                'mse': 0.0,
                'rmse': 0.0,
                'mape': 0.0,
                'r2_score': 0.0
            }

    def generate_forecast(
        self,
        sensor_id: str,
        data: np.ndarray,
        timestamps: List[datetime],
        horizon_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Generate forecast for sensor data using Transformer

        **CRITICAL FIX**: Timestamps must be provided with data to ensure
        accurate forecast timeline. Using datetime.now() leads to incorrect
        temporal alignment.

        Args:
            sensor_id: Unique sensor identifier
            data: Historical time series data
            timestamps: Corresponding timestamps for historical data
            horizon_hours: Forecast horizon in hours

        Returns:
            Dictionary containing forecast results

        Raises:
            ValueError: If timestamps length doesn't match data length
        """
        start_time = datetime.now()

        try:
            # Validate timestamp and data alignment
            if len(timestamps) != len(data):
                raise ValueError(
                    f"Timestamps length ({len(timestamps)}) must match data length ({len(data)})"
                )

            # Get model for sensor
            model = self._get_model(sensor_id)

            if not model or not model.is_trained:
                logger.warning(f"Model not available or not trained for sensor {sensor_id}, using fallback forecasting")
                return self._fallback_forecast(sensor_id, data, timestamps, horizon_hours)

            # Generate forecast
            forecast_result = model.predict(data, horizon_hours)

            # CRITICAL FIX: Use last data timestamp, NOT datetime.now()
            last_timestamp = timestamps[-1]

            # Create timestamps for forecast (relative to last data point)
            forecast_timestamps = [
                last_timestamp + timedelta(hours=i+1)
                for i in range(len(forecast_result['forecast_values']))
            ]

            # Use provided historical timestamps
            historical_timestamps = timestamps

            # Calculate confidence intervals
            confidence_intervals = {
                'upper': forecast_result['confidence_upper'],
                'lower': forecast_result['confidence_lower']
            }

            # Create forecast points with risk assessment
            forecast_points = []
            for i, (timestamp, value) in enumerate(zip(forecast_timestamps, forecast_result['forecast_values'])):
                upper = forecast_result['confidence_upper'][i] if i < len(forecast_result['confidence_upper']) else value
                lower = forecast_result['confidence_lower'][i] if i < len(forecast_result['confidence_lower']) else value

                confidence_level = self._calculate_forecast_confidence((lower, upper), value)
                risk_level = self._calculate_risk_level(value, (lower, upper))

                forecast_point = ForecastPoint(
                    timestamp=timestamp,
                    predicted_value=value,
                    confidence_lower=lower,
                    confidence_upper=upper,
                    confidence_level=confidence_level,
                    risk_level=risk_level
                )
                forecast_points.append(forecast_point)

            # CRITICAL: This is an IN-SAMPLE fit metric, NOT true forecast accuracy
            # True accuracy requires a held-out validation set with actual future values
            # This metric only shows how well the model fits recent training data
            in_sample_fit_metrics = self._calculate_accuracy_metrics(
                data[-horizon_hours:] if len(data) >= horizon_hours else data,
                np.array(forecast_result['forecast_values'][:min(len(data), len(forecast_result['forecast_values']))])
            )

            # Rename for clarity
            accuracy_metrics = {
                **in_sample_fit_metrics,
                'note': 'IN_SAMPLE_FIT_ONLY - Not true forecast accuracy. Requires validation set.'
            }

            processing_time = (datetime.now() - start_time).total_seconds()

            # Create result object
            result = ForecastResult(
                sensor_id=sensor_id,
                forecast_horizon_hours=horizon_hours,
                historical_timestamps=historical_timestamps,
                historical_values=data,
                forecast_timestamps=forecast_timestamps,
                forecast_values=np.array(forecast_result['forecast_values']),
                confidence_intervals=confidence_intervals,
                accuracy_metrics=accuracy_metrics,
                model_version="Transformer-1.0",
                generated_at=datetime.now(),
                risk_assessment={
                    'high_risk_points': sum(1 for fp in forecast_points if fp.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]),
                    'average_confidence': np.mean([fp.confidence_upper - fp.confidence_lower for fp in forecast_points]),
                    'quality_score': forecast_result.get('forecast_quality', 'unknown')
                }
            )

            # Store in history
            if sensor_id not in self._forecast_history:
                self._forecast_history[sensor_id] = []
            self._forecast_history[sensor_id].append(result)

            # Keep only recent forecasts (configurable size)
            self._forecast_history[sensor_id] = self._forecast_history[sensor_id][-self.forecast_history_size:]

            logger.info(f"Generated {horizon_hours}h forecast for sensor {sensor_id}")

            return {
                'sensor_id': sensor_id,
                'historical_timestamps': [ts.isoformat() for ts in historical_timestamps],
                'historical_values': data.tolist(),
                'forecast_timestamps': [ts.isoformat() for ts in forecast_timestamps],
                'forecast_values': forecast_result['forecast_values'],
                'confidence_upper': forecast_result['confidence_upper'],
                'confidence_lower': forecast_result['confidence_lower'],
                'accuracy_metrics': accuracy_metrics,
                'risk_assessment': result.risk_assessment,
                'processing_time': processing_time,
                'model_status': 'trained'
            }

        except Exception as e:
            logger.error(f"Error generating forecast for sensor {sensor_id}: {e}")
            return self._fallback_forecast(sensor_id, data, timestamps, horizon_hours)

    def _fallback_forecast(
        self,
        sensor_id: str,
        data: np.ndarray,
        timestamps: List[datetime],
        horizon_hours: int
    ) -> Dict[str, Any]:
        """Fallback forecasting when model is not available

        Uses simple linear trend extrapolation with improved uncertainty estimation.

        Args:
            sensor_id: Sensor identifier
            data: Historical data
            timestamps: Historical timestamps
            horizon_hours: Forecast horizon

        Returns:
            dict: Forecast results in consistent format
        """
        try:
            if len(data) < 2:
                # Not enough data for any forecast
                last_value = data[-1] if len(data) > 0 else 0.0
                forecast_values = [last_value] * horizon_hours
            else:
                # Simple linear trend extrapolation
                x = np.arange(len(data))
                coeffs = np.polyfit(x, data, 1)

                # Extrapolate
                future_x = np.arange(len(data), len(data) + horizon_hours)
                forecast_values = np.polyval(coeffs, future_x).tolist()

            # Improved confidence intervals based on data variance
            # Use standard deviation of data for uncertainty estimation
            data_std = np.std(data) if len(data) > 1 else abs(data[0] * 0.1) if len(data) > 0 else 1.0

            # Confidence grows with forecast horizon (more uncertain further out)
            confidence_upper = [
                v + data_std * (1 + 0.1 * i) for i, v in enumerate(forecast_values)
            ]
            confidence_lower = [
                v - data_std * (1 + 0.1 * i) for i, v in enumerate(forecast_values)
            ]

            # CRITICAL FIX: Use last data timestamp, NOT datetime.now()
            last_timestamp = timestamps[-1]
            historical_timestamps = timestamps
            forecast_timestamps = [
                last_timestamp + timedelta(hours=i+1) for i in range(horizon_hours)
            ]

            return {
                'sensor_id': sensor_id,
                'historical_timestamps': [ts.isoformat() for ts in historical_timestamps],
                'historical_values': data.tolist(),
                'forecast_timestamps': [ts.isoformat() for ts in forecast_timestamps],
                'forecast_values': forecast_values,
                'confidence_upper': confidence_upper,
                'confidence_lower': confidence_lower,
                'accuracy_metrics': {
                    'mae': 0.0, 'mse': 0.0, 'rmse': 0.0, 'mape': 0.0, 'r2_score': 0.0
                },
                'risk_assessment': {
                    'high_risk_points': 0,
                    'average_confidence': np.mean([u - l for u, l in zip(confidence_upper, confidence_lower)]),
                    'quality_score': 'fallback'
                },
                'processing_time': 0.001,
                'model_status': 'fallback'
            }

        except Exception as e:
            logger.error(f"Fallback forecasting failed for sensor {sensor_id}: {e}")
            return {
                'sensor_id': sensor_id,
                'historical_timestamps': [],
                'historical_values': [],
                'forecast_timestamps': [],
                'forecast_values': [],
                'confidence_upper': [],
                'confidence_lower': [],
                'accuracy_metrics': {},
                'risk_assessment': {},
                'processing_time': 0.0,
                'model_status': 'error'
            }

    def is_model_trained(self, sensor_id: str) -> bool:
        """Check if forecasting model is trained for given sensor"""
        try:
            model = self._get_model(sensor_id)
            return model is not None and model.is_trained
        except Exception:
            return False

    def get_forecast_accuracy(self, sensor_id: str) -> Dict[str, float]:
        """Get forecast accuracy metrics for sensor"""
        try:
            if sensor_id in self._forecast_history and self._forecast_history[sensor_id]:
                latest_forecast = self._forecast_history[sensor_id][-1]
                return latest_forecast.accuracy_metrics
            else:
                return {
                    'mae': 0.0,
                    'mse': 0.0,
                    'rmse': 0.0,
                    'mape': 0.0,
                    'r2_score': 0.0
                }
        except Exception as e:
            logger.error(f"Error getting forecast accuracy for sensor {sensor_id}: {e}")
            return {}

    def get_forecast_summary(self, sensor_id: str = None, hours_back: int = 24) -> Dict[str, Any]:
        """Get summary of recent forecasts"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)

            if sensor_id:
                # Summary for specific sensor
                sensor_forecasts = self._forecast_history.get(sensor_id, [])
                recent_forecasts = [f for f in sensor_forecasts if f.generated_at >= cutoff_time]

                if recent_forecasts:
                    latest_forecast = recent_forecasts[-1]
                    avg_accuracy = np.mean([f.accuracy_metrics.get('r2_score', 0) for f in recent_forecasts])

                    return {
                        'sensor_id': sensor_id,
                        'total_forecasts': len(recent_forecasts),
                        'average_accuracy': float(avg_accuracy),
                        'latest_forecast': {
                            'generated_at': latest_forecast.generated_at.isoformat(),
                            'horizon_hours': latest_forecast.forecast_horizon_hours,
                            'accuracy_metrics': latest_forecast.accuracy_metrics
                        },
                        'model_trained': self.is_model_trained(sensor_id)
                    }
                else:
                    return {
                        'sensor_id': sensor_id,
                        'total_forecasts': 0,
                        'average_accuracy': 0.0,
                        'latest_forecast': None,
                        'model_trained': self.is_model_trained(sensor_id)
                    }
            else:
                # Summary across all sensors
                all_forecasts = []
                sensor_performance = {}

                for sid, forecasts in self._forecast_history.items():
                    recent_forecasts = [f for f in forecasts if f.generated_at >= cutoff_time]
                    all_forecasts.extend(recent_forecasts)

                    if recent_forecasts:
                        avg_accuracy = np.mean([f.accuracy_metrics.get('r2_score', 0) for f in recent_forecasts])
                        sensor_performance[sid] = {
                            'forecast_count': len(recent_forecasts),
                            'average_accuracy': float(avg_accuracy),
                            'model_trained': self.is_model_trained(sid)
                        }

                # Overall statistics
                total_sensors_forecasted = len(sensor_performance)
                average_confidence = np.mean([
                    f.risk_assessment.get('average_confidence', 0)
                    for f in all_forecasts
                ]) if all_forecasts else 0

                # Risk distribution
                risk_distribution = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}
                for forecast in all_forecasts:
                    high_risk_points = forecast.risk_assessment.get('high_risk_points', 0)
                    if high_risk_points > 10:
                        risk_distribution['HIGH'] += 1
                    elif high_risk_points > 5:
                        risk_distribution['MEDIUM'] += 1
                    else:
                        risk_distribution['LOW'] += 1

                return {
                    'total_sensors_forecasted': total_sensors_forecasted,
                    'average_confidence': float(average_confidence),
                    'risk_distribution': risk_distribution,
                    'recent_forecasts': [
                        {
                            'sensor_id': f.sensor_id,
                            'generated_at': f.generated_at.isoformat(),
                            'horizon_hours': f.forecast_horizon_hours,
                            'accuracy': f.accuracy_metrics.get('r2_score', 0)
                        }
                        for f in sorted(all_forecasts, key=lambda x: x.generated_at, reverse=True)[:10]
                    ],
                    'model_performance': sensor_performance,
                    'generated_at': datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Error generating forecast summary: {e}")
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
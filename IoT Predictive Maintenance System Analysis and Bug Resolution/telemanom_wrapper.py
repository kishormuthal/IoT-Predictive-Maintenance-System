"""
NASA Telemanom Implementation
Official NASA algorithm for spacecraft telemetry anomaly detection
Based on https://github.com/khundman/telemanom
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from pathlib import Path
import pickle
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

# TensorFlow lazy loading to prevent hanging
TENSORFLOW_AVAILABLE = False
tf = None
keras = None
layers = None

def _load_tensorflow():
    """Lazy load TensorFlow only when needed"""
    global TENSORFLOW_AVAILABLE, tf, keras, layers
    if not TENSORFLOW_AVAILABLE:
        try:
            import tensorflow as tf_module
            from tensorflow import keras as keras_module
            from tensorflow.keras import layers as layers_module
            tf = tf_module
            keras = keras_module
            layers = layers_module
            TENSORFLOW_AVAILABLE = True
            print("[INFO] TensorFlow loaded successfully for Telemanom")
        except ImportError:
            print("[WARNING] TensorFlow not available, using mock implementation")
            _setup_mock_implementations()
    return TENSORFLOW_AVAILABLE

# Mock implementations for when TensorFlow is not available
def _setup_mock_implementations():
    """Setup mock TensorFlow implementations"""
    global keras, tf, layers

    class MockKeras:
        class Sequential:
            def __init__(self, layers=None):
                self.layers = layers or []
            def compile(self, **kwargs):
                pass
            def fit(self, X, y, **kwargs):
                return type('MockHistory', (), {'history': {'loss': [0.1]}})()
            def predict(self, X):
                return np.zeros((len(X), 1))

    keras = MockKeras()
    tf = None
    layers = None

    class callbacks:
        class EarlyStopping:
            def __init__(self, **kwargs):
                pass

        class ReduceLROnPlateau:
            def __init__(self, **kwargs):
                pass

    class optimizers:
        class Adam:
            def __init__(self, **kwargs):
                pass

    class layers:
        class LSTM:
            def __init__(self, **kwargs):
                pass

        class Dense:
            def __init__(self, **kwargs):
                pass

        class Reshape:
            def __init__(self, **kwargs):
                pass

    class MockHistory:
        def __init__(self):
            self.history = {'loss': [0.1, 0.08, 0.06], 'val_loss': [0.12, 0.09, 0.07]}

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class Telemanom_Config:
    """Configuration for NASA Telemanom model"""
    # Model architecture
    sequence_length: int = 250  # NASA default l_s
    lstm_units: List[int] = None  # Will default to [80, 80]
    dropout_rate: float = 0.3
    prediction_length: int = 10  # Prediction window

    # Training parameters
    epochs: int = 35  # NASA default
    batch_size: int = 70  # NASA default
    learning_rate: float = 0.001
    validation_split: float = 0.2

    # Anomaly detection
    error_buffer: int = 100  # Buffer for error distribution
    smoothing_window: int = 30  # Smoothing window for errors
    contamination: float = 0.05  # Expected anomaly rate

    def __post_init__(self):
        if self.lstm_units is None:
            self.lstm_units = [80, 80]


class NASATelemanom:
    """
    NASA Telemanom Anomaly Detection Algorithm

    Implementation of the official NASA spacecraft telemetry anomaly detection
    system using LSTM neural networks with dynamic thresholding.
    """

    def __init__(self, sensor_id: str, config: Optional[Telemanom_Config] = None):
        """Initialize Telemanom model for specific sensor

        Args:
            sensor_id: Unique identifier for the sensor
            config: Model configuration
        """
        self.sensor_id = sensor_id
        self.config = config or Telemanom_Config()

        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

        # Training data and statistics
        self.training_errors = None
        self.error_threshold = None
        self.smoothed_errors = None

        # Model metadata
        self.n_features = None
        self.training_history = None

        logger.info(f"Initialized NASA Telemanom for sensor {sensor_id}")

    def _build_model(self):
        """Build LSTM model following NASA Telemanom architecture"""
        model = keras.Sequential([
            # First LSTM layer
            layers.LSTM(
                units=self.config.lstm_units[0],
                return_sequences=True,
                input_shape=(self.config.sequence_length, self.n_features),
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.dropout_rate
            ),

            # Second LSTM layer (if specified)
            layers.LSTM(
                units=self.config.lstm_units[1] if len(self.config.lstm_units) > 1 else self.config.lstm_units[0],
                return_sequences=False,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.dropout_rate
            ),

            # Dense layers for prediction
            layers.Dense(units=self.n_features * self.config.prediction_length, activation='linear'),
            layers.Reshape((self.config.prediction_length, self.n_features))
        ])

        # Compile with Adam optimizer (NASA default)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )

        return model

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences and targets for training"""
        sequences = []
        targets = []

        seq_len = self.config.sequence_length
        pred_len = self.config.prediction_length

        for i in range(len(data) - seq_len - pred_len + 1):
            sequence = data[i:i + seq_len]
            target = data[i + seq_len:i + seq_len + pred_len]
            sequences.append(sequence)
            targets.append(target)

        return np.array(sequences), np.array(targets)

    def _calculate_prediction_errors(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate prediction errors for anomaly detection"""
        # Mean absolute error for each timestep
        errors = np.mean(np.abs(y_true - y_pred), axis=(1, 2))
        return errors

    def _smooth_errors(self, errors: np.ndarray) -> np.ndarray:
        """Apply smoothing to prediction errors"""
        smoothed = pd.Series(errors).rolling(
            window=self.config.smoothing_window,
            center=True,
            min_periods=1
        ).mean().values
        return smoothed

    def _calculate_dynamic_threshold(self, errors: np.ndarray) -> float:
        """Calculate dynamic threshold using NASA methodology"""
        # Use error buffer for stable threshold calculation
        if len(errors) < self.config.error_buffer:
            # Not enough data, use percentile-based threshold
            threshold = np.percentile(errors, (1 - self.config.contamination) * 100)
        else:
            # Use last error_buffer errors for threshold calculation
            recent_errors = errors[-self.config.error_buffer:]

            # Calculate threshold as mean + 3*std (NASA approach)
            threshold = np.mean(recent_errors) + 3 * np.std(recent_errors)

            # Ensure threshold is not too low
            min_threshold = np.percentile(recent_errors, 95)
            threshold = max(threshold, min_threshold)

        return threshold

    def train(self, training_data: np.ndarray) -> Dict[str, Any]:
        """Train the Telemanom model"""
        logger.info(f"Training Telemanom model for sensor {self.sensor_id}")

        # Prepare data
        if len(training_data.shape) == 1:
            training_data = training_data.reshape(-1, 1)

        self.n_features = training_data.shape[1]

        # Scale data
        scaled_data = self.scaler.fit_transform(training_data)

        # Create sequences
        X, y = self._create_sequences(scaled_data)

        if len(X) == 0:
            raise ValueError(f"Not enough data to create sequences for sensor {self.sensor_id}")

        # Build and train model
        self.model = self._build_model()

        # Training callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

        # Train model
        history = self.model.fit(
            X, y,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=self.config.validation_split,
            callbacks=callbacks,
            verbose=0
        )

        # Calculate training errors for threshold
        y_pred = self.model.predict(X, verbose=0)
        self.training_errors = self._calculate_prediction_errors(y, y_pred)
        self.smoothed_errors = self._smooth_errors(self.training_errors)
        self.error_threshold = self._calculate_dynamic_threshold(self.smoothed_errors)

        self.is_trained = True
        self.training_history = history.history

        logger.info(f"Training completed for sensor {self.sensor_id}")
        logger.info(f"Model parameters: {self.model.count_params()}")
        logger.info(f"Error threshold: {self.error_threshold:.4f}")

        return {
            'sensor_id': self.sensor_id,
            'training_samples': len(X),
            'model_parameters': self.model.count_params(),
            'error_threshold': self.error_threshold,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        }

    def detect_anomalies(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect anomalies in new data"""
        if not self.is_trained:
            raise ValueError(f"Model not trained for sensor {self.sensor_id}")

        # Prepare data
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        # Scale data
        scaled_data = self.scaler.transform(data)

        # Create sequences
        X, y = self._create_sequences(scaled_data)

        if len(X) == 0:
            return {
                'sensor_id': self.sensor_id,
                'anomalies': [],
                'scores': [],
                'threshold': self.error_threshold,
                'total_points': len(data)
            }

        # Predict
        y_pred = self.model.predict(X, verbose=0)

        # Calculate errors
        errors = self._calculate_prediction_errors(y, y_pred)
        smoothed_errors = self._smooth_errors(errors)

        # Update threshold dynamically
        all_errors = np.concatenate([self.smoothed_errors, smoothed_errors])
        self.error_threshold = self._calculate_dynamic_threshold(all_errors)

        # Detect anomalies
        anomaly_indices = np.where(smoothed_errors > self.error_threshold)[0]

        # Calculate anomaly scores (normalized)
        max_error = max(np.max(smoothed_errors), self.error_threshold)
        scores = smoothed_errors / max_error

        # Adjust indices to original data coordinates
        adjusted_indices = anomaly_indices + self.config.sequence_length

        results = {
            'sensor_id': self.sensor_id,
            'anomalies': adjusted_indices.tolist(),
            'scores': scores.tolist(),
            'threshold': self.error_threshold,
            'total_points': len(data),
            'anomaly_count': len(anomaly_indices)
        }

        logger.info(f"Detected {len(anomaly_indices)} anomalies for sensor {self.sensor_id}")

        return results

    def save_model(self, model_path: Path) -> None:
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        model_dir = model_path / self.sensor_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save Keras model
        if TENSORFLOW_AVAILABLE:
            self.model.save(model_dir / 'model.h5')

        # Save scaler and metadata
        with open(model_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

        metadata = {
            'sensor_id': self.sensor_id,
            'config': self.config.__dict__,
            'error_threshold': self.error_threshold,
            'n_features': self.n_features,
            'is_trained': self.is_trained,
            'training_history': self.training_history
        }

        with open(model_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        # Save training errors
        np.save(model_dir / 'training_errors.npy', self.training_errors)
        np.save(model_dir / 'smoothed_errors.npy', self.smoothed_errors)

        logger.info(f"Model saved for sensor {self.sensor_id}")

    def load_model(self, model_path: Path) -> bool:
        """Load trained model from disk"""
        try:
            model_dir = model_path / self.sensor_id

            # Load metadata
            with open(model_dir / 'metadata.json', 'r') as f:
                metadata = json.load(f)

            self.sensor_id = metadata['sensor_id']
            self.config = Telemanom_Config(**metadata['config'])
            self.error_threshold = metadata['error_threshold']
            self.n_features = metadata['n_features']
            self.is_trained = metadata['is_trained']
            self.training_history = metadata.get('training_history')

            # Load Keras model
            if TENSORFLOW_AVAILABLE and (model_dir / 'model.h5').exists():
                self.model = keras.models.load_model(model_dir / 'model.h5')

            # Load scaler
            with open(model_dir / 'scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)

            # Load training errors
            self.training_errors = np.load(model_dir / 'training_errors.npy')
            self.smoothed_errors = np.load(model_dir / 'smoothed_errors.npy')

            logger.info(f"Model loaded for sensor {self.sensor_id}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load model for sensor {self.sensor_id}: {e}")
            return False

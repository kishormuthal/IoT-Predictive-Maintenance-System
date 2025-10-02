"""
NASA Telemanom Implementation
Official NASA algorithm for spacecraft telemetry anomaly detection
Based on https://github.com/khundman/telemanom
"""

import json
import logging
import pickle
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

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


# Custom Exceptions
class ModelNotTrainedError(Exception):
    """Raised when attempting to use an untrained model"""

    pass


class InsufficientDataError(Exception):
    """Raised when there's not enough data for training or inference"""

    pass


# Mock implementations for when TensorFlow is not available
def _setup_mock_implementations():
    """Setup mock TensorFlow implementations with informative warnings"""
    global keras, tf, layers

    class MockKeras:
        class Sequential:
            def __init__(self, layers=None):
                self.layers = layers or []
                logger.warning("[MOCK] Using mock Keras Sequential - TensorFlow not available")

            def compile(self, **kwargs):
                logger.debug("[MOCK] Mock compile called - no actual compilation")
                pass

            def fit(self, X, y, **kwargs):
                logger.warning("[MOCK] Mock training - returning dummy history without actual training")
                return type(
                    "MockHistory",
                    (),
                    {
                        "history": {
                            "loss": [0.1, 0.08, 0.06],
                            "val_loss": [0.12, 0.09, 0.07],
                        }
                    },
                )()

            def predict(self, X):
                logger.warning("[MOCK] Mock prediction - returning zeros (no actual prediction)")
                return np.zeros((len(X), 1, 1))

            def count_params(self):
                return 0

            def save(self, path):
                logger.warning("[MOCK] Cannot save model - TensorFlow not available")
                pass

        class models:
            @staticmethod
            def load_model(path):
                logger.warning("[MOCK] Cannot load model - TensorFlow not available")
                return MockKeras.Sequential()

    keras = MockKeras()
    keras.models = MockKeras.models
    tf = None
    layers = None

    class callbacks:
        class EarlyStopping:
            def __init__(self, **kwargs):
                logger.debug("[MOCK] EarlyStopping callback - mock implementation")
                pass

        class ReduceLROnPlateau:
            def __init__(self, **kwargs):
                logger.debug("[MOCK] ReduceLROnPlateau callback - mock implementation")
                pass

    class optimizers:
        class Adam:
            def __init__(self, **kwargs):
                logger.debug("[MOCK] Adam optimizer - mock implementation")
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

    keras.callbacks = callbacks
    keras.optimizers = optimizers


warnings.filterwarnings("ignore")
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
    verbose: int = 0  # 0=silent, 1=progress bar, 2=one line per epoch

    # Anomaly detection
    error_buffer: int = 100  # Buffer for error distribution
    smoothing_window: int = 30  # Smoothing window for errors
    contamination: float = 0.05  # Expected anomaly rate
    feature_wise_errors: bool = False  # Calculate errors per feature

    def __post_init__(self):
        """Validate configuration parameters"""
        if self.lstm_units is None:
            self.lstm_units = [80, 80]

        # Validate ranges
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError(f"dropout_rate must be between 0 and 1, got {self.dropout_rate}")

        if self.sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {self.sequence_length}")

        if self.prediction_length <= 0:
            raise ValueError(f"prediction_length must be positive, got {self.prediction_length}")

        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if not 0 < self.validation_split < 1:
            raise ValueError(f"validation_split must be between 0 and 1, got {self.validation_split}")

        if not 0 <= self.contamination <= 1:
            raise ValueError(f"contamination must be between 0 and 1, got {self.contamination}")

        if self.error_buffer <= 0:
            raise ValueError(f"error_buffer must be positive, got {self.error_buffer}")

        if self.smoothing_window <= 0:
            raise ValueError(f"smoothing_window must be positive, got {self.smoothing_window}")


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

        logger.debug(f"Initialized NASA Telemanom for sensor {sensor_id}")

    def _build_model(self):
        """Build LSTM model following NASA Telemanom architecture

        Note: Requires self.n_features to be set before calling
        """
        if self.n_features is None:
            raise ValueError("n_features must be set before building model")

        model = keras.Sequential(
            [
                # First LSTM layer
                layers.LSTM(
                    units=self.config.lstm_units[0],
                    return_sequences=True,
                    input_shape=(self.config.sequence_length, self.n_features),
                    dropout=self.config.dropout_rate,
                    recurrent_dropout=self.config.dropout_rate,
                ),
                # Second LSTM layer (if specified)
                layers.LSTM(
                    units=(self.config.lstm_units[1] if len(self.config.lstm_units) > 1 else self.config.lstm_units[0]),
                    return_sequences=False,
                    dropout=self.config.dropout_rate,
                    recurrent_dropout=self.config.dropout_rate,
                ),
                # Dense layers for prediction
                layers.Dense(
                    units=self.n_features * self.config.prediction_length,
                    activation="linear",
                ),
                layers.Reshape((self.config.prediction_length, self.n_features)),
            ]
        )

        # Compile with Adam optimizer (NASA default)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="mse",
            metrics=["mae"],
        )

        return model

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences and targets for training"""
        sequences = []
        targets = []

        seq_len = self.config.sequence_length
        pred_len = self.config.prediction_length

        for i in range(len(data) - seq_len - pred_len + 1):
            sequence = data[i : i + seq_len]
            target = data[i + seq_len : i + seq_len + pred_len]
            sequences.append(sequence)
            targets.append(target)

        return np.array(sequences), np.array(targets)

    def _calculate_prediction_errors(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate prediction errors for anomaly detection

        Args:
            y_true: True values (batch_size, prediction_length, n_features)
            y_pred: Predicted values (batch_size, prediction_length, n_features)

        Returns:
            errors: Error values. Shape depends on config.feature_wise_errors:
                    - If False: (batch_size,) - mean error across all features and timesteps
                    - If True: (batch_size, n_features) - mean error per feature across timesteps
        """
        # Calculate absolute error
        abs_errors = np.abs(y_true - y_pred)

        if self.config.feature_wise_errors:
            # Calculate mean error for each feature separately (across prediction timesteps)
            errors = np.mean(abs_errors, axis=1)  # (batch_size, n_features)
        else:
            # Calculate mean error across all timesteps and features
            errors = np.mean(abs_errors, axis=(1, 2))  # (batch_size,)

        return errors

    def _smooth_errors(self, errors: np.ndarray) -> np.ndarray:
        """Apply smoothing to prediction errors using rolling mean

        Note: min_periods=1 means the rolling window will start smoothing from the
        first data point, using fewer points than the window size at the beginning.
        This ensures no NaN values but may result in less reliable smoothing for
        the initial points.

        Args:
            errors: Array of prediction errors

        Returns:
            smoothed: Smoothed errors with same shape as input
        """
        smoothed = (
            pd.Series(errors).rolling(window=self.config.smoothing_window, center=True, min_periods=1).mean().values
        )
        return smoothed

    def _calculate_dynamic_threshold(self, errors: np.ndarray) -> float:
        """Calculate dynamic threshold using refined NASA methodology

        Uses error_buffer for stable threshold calculation. The threshold is computed
        using a combination of mean + 3*std (NASA approach) and contamination-based
        percentile to avoid both overly aggressive and overly lenient detection.

        Args:
            errors: Array of prediction errors

        Returns:
            threshold: Calculated anomaly detection threshold
        """
        if len(errors) == 0:
            return 0.0

        # Use contamination parameter consistently
        contamination_percentile = (1 - self.config.contamination) * 100

        if len(errors) < self.config.error_buffer:
            # Not enough data, use percentile-based threshold with contamination
            threshold = np.percentile(errors, contamination_percentile)
        else:
            # Use last error_buffer errors for threshold calculation
            recent_errors = errors[-self.config.error_buffer :]

            # Calculate threshold as mean + 3*std (NASA approach)
            mean_error = np.mean(recent_errors)
            std_error = np.std(recent_errors)
            threshold = mean_error + 3 * std_error

            # Use contamination parameter to set minimum threshold
            # This prevents the threshold from being too low
            min_threshold = np.percentile(errors, contamination_percentile)
            threshold = max(threshold, min_threshold)

        return float(threshold)

    def train(self, training_data: np.ndarray) -> Dict[str, Any]:
        """Train the Telemanom model

        Args:
            training_data: Training data as numpy array

        Returns:
            dict: Training results including metrics and model info

        Raises:
            InsufficientDataError: If not enough data for training
            ValueError: If data has invalid shape or values
        """
        logger.info(f"Training Telemanom model for sensor {self.sensor_id}")

        # Prepare data
        if len(training_data.shape) == 1:
            training_data = training_data.reshape(-1, 1)

        self.n_features = training_data.shape[1]

        # Pre-check: Ensure minimum data length
        min_required_length = self.config.sequence_length + self.config.prediction_length
        if len(training_data) < min_required_length:
            raise InsufficientDataError(
                f"Insufficient training data for sensor {self.sensor_id}. "
                f"Need at least {min_required_length} samples, got {len(training_data)}"
            )

        # Scale data
        scaled_data = self.scaler.fit_transform(training_data)

        # Create sequences
        X, y = self._create_sequences(scaled_data)

        if len(X) == 0:
            raise InsufficientDataError(f"Not enough data to create sequences for sensor {self.sensor_id}")

        # Build and train model
        self.model = self._build_model()

        # Training callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        ]

        # Train model with configurable verbosity
        history = self.model.fit(
            X,
            y,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=self.config.validation_split,
            callbacks=callbacks,
            verbose=self.config.verbose,
        )

        # Calculate training errors for threshold
        y_pred = self.model.predict(X, verbose=0)
        self.training_errors = self._calculate_prediction_errors(y, y_pred)
        self.smoothed_errors = self._smooth_errors(self.training_errors)
        self.error_threshold = self._calculate_dynamic_threshold(self.smoothed_errors)

        self.is_trained = True

        # Convert NumPy arrays to lists for JSON serialization
        self.training_history = {
            key: ([float(v) for v in values] if isinstance(values, (list, np.ndarray)) else values)
            for key, values in history.history.items()
        }

        logger.info(f"Training completed for sensor {self.sensor_id}")
        logger.info(f"Model parameters: {self.model.count_params()}")
        logger.info(f"Error threshold: {self.error_threshold:.4f}")

        # Robust history access with fallbacks
        final_loss = history.history.get("loss", [0])[-1] if history.history.get("loss") else 0
        final_val_loss = history.history.get("val_loss", [0])[-1] if history.history.get("val_loss") else 0

        return {
            "sensor_id": self.sensor_id,
            "training_samples": len(X),
            "model_parameters": self.model.count_params(),
            "error_threshold": self.error_threshold,
            "final_loss": float(final_loss),
            "final_val_loss": float(final_val_loss),
        }

    def detect_anomalies(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect anomalies in new data

        Args:
            data: Input data for anomaly detection

        Returns:
            dict: Detection results including anomalies, scores, and threshold

        Raises:
            ModelNotTrainedError: If model hasn't been trained yet
        """
        if not self.is_trained:
            raise ModelNotTrainedError(f"Model not trained for sensor {self.sensor_id}")

        # Prepare data
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        # Pre-check for minimum data length
        min_required_length = self.config.sequence_length + self.config.prediction_length
        if len(data) < min_required_length:
            logger.warning(f"Insufficient data for anomaly detection on sensor {self.sensor_id}")
            return {
                "sensor_id": self.sensor_id,
                "anomalies": [],
                "scores": [],
                "threshold": self.error_threshold,
                "total_points": len(data),
                "anomaly_count": 0,
            }

        # Scale data
        scaled_data = self.scaler.transform(data)

        # Create sequences
        X, y = self._create_sequences(scaled_data)

        if len(X) == 0:
            return {
                "sensor_id": self.sensor_id,
                "anomalies": [],
                "scores": [],
                "threshold": self.error_threshold,
                "total_points": len(data),
                "anomaly_count": 0,
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

        # Calculate anomaly scores (normalized with epsilon for stability)
        epsilon = 1e-10
        max_error = max(np.max(smoothed_errors), self.error_threshold, epsilon)
        scores = smoothed_errors / (max_error + epsilon)

        # Adjust indices to original data coordinates
        # Note: adjusted_indices point to the start of the sequence where anomaly was detected
        adjusted_indices = anomaly_indices + self.config.sequence_length

        results = {
            "sensor_id": self.sensor_id,
            "anomalies": adjusted_indices.tolist(),
            "scores": scores.tolist(),
            "threshold": self.error_threshold,
            "total_points": len(data),
            "anomaly_count": len(anomaly_indices),
        }

        logger.info(f"Detected {len(anomaly_indices)} anomalies for sensor {self.sensor_id}")

        return results

    def save_model(self, model_path: Path) -> None:
        """Save trained model to disk

        Args:
            model_path: Base path for model storage

        Raises:
            ModelNotTrainedError: If model hasn't been trained yet
            IOError: If unable to save model files
        """
        if not self.is_trained:
            raise ModelNotTrainedError("Model not trained yet")

        try:
            model_dir = model_path / self.sensor_id
            model_dir.mkdir(parents=True, exist_ok=True)

            # Save Keras model (only if TensorFlow is available)
            if TENSORFLOW_AVAILABLE and self.model is not None:
                self.model.save(model_dir / "model.h5")
                logger.debug(f"Saved Keras model for sensor {self.sensor_id}")
            else:
                logger.warning(f"TensorFlow not available - Keras model not saved for {self.sensor_id}")

            # Save scaler
            with open(model_dir / "scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)

            # Prepare metadata (with training history already converted to lists in train())
            metadata = {
                "sensor_id": self.sensor_id,
                "config": self.config.__dict__,
                "error_threshold": (float(self.error_threshold) if self.error_threshold is not None else None),
                "n_features": (int(self.n_features) if self.n_features is not None else None),
                "is_trained": self.is_trained,
                "training_history": self.training_history,
                "tensorflow_available": TENSORFLOW_AVAILABLE,
            }

            with open(model_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            # Save training errors
            if self.training_errors is not None:
                np.save(model_dir / "training_errors.npy", self.training_errors)
            if self.smoothed_errors is not None:
                np.save(model_dir / "smoothed_errors.npy", self.smoothed_errors)

            logger.info(f"Model saved successfully for sensor {self.sensor_id} at {model_dir}")

        except IOError as e:
            logger.error(f"Failed to save model for sensor {self.sensor_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving model for sensor {self.sensor_id}: {e}")
            raise

    def load_model(self, model_path: Path) -> bool:
        """Load trained model from disk

        Args:
            model_path: Base path for model storage

        Returns:
            bool: True if loading successful, False otherwise
        """
        try:
            model_dir = model_path / self.sensor_id

            # Load metadata
            metadata_file = model_dir / "metadata.json"
            if not metadata_file.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            self.sensor_id = metadata["sensor_id"]
            self.config = Telemanom_Config(**metadata["config"])
            self.error_threshold = metadata.get("error_threshold")
            self.n_features = metadata.get("n_features")
            self.training_history = metadata.get("training_history")

            # Check TensorFlow consistency
            model_file = model_dir / "model.h5"
            was_saved_with_tf = metadata.get("tensorflow_available", True)

            if model_file.exists():
                if TENSORFLOW_AVAILABLE:
                    # Load Keras model
                    self.model = keras.models.load_model(model_file)
                    self.is_trained = True
                    logger.debug(f"Loaded Keras model for sensor {self.sensor_id}")
                else:
                    # TensorFlow not available but model was saved with it
                    logger.warning(
                        f"Model for {self.sensor_id} was saved with TensorFlow but TensorFlow "
                        f"is not currently available. Model cannot be used for inference."
                    )
                    self.model = None
                    self.is_trained = False
                    return False
            else:
                if was_saved_with_tf:
                    logger.warning(f"Expected model file not found: {model_file}")
                    self.is_trained = False
                    self.model = None
                    return False
                else:
                    # Model was saved without TensorFlow (mock mode)
                    logger.info(f"Model for {self.sensor_id} was saved in mock mode")
                    self.is_trained = False
                    self.model = None
                    return False

            # Load scaler
            scaler_file = model_dir / "scaler.pkl"
            if not scaler_file.exists():
                raise FileNotFoundError(f"Scaler file not found: {scaler_file}")

            with open(scaler_file, "rb") as f:
                self.scaler = pickle.load(f)

            # Load training errors
            errors_file = model_dir / "training_errors.npy"
            smoothed_file = model_dir / "smoothed_errors.npy"

            if errors_file.exists():
                self.training_errors = np.load(errors_file)
            else:
                logger.warning(f"Training errors file not found: {errors_file}")
                self.training_errors = None

            if smoothed_file.exists():
                self.smoothed_errors = np.load(smoothed_file)
            else:
                logger.warning(f"Smoothed errors file not found: {smoothed_file}")
                self.smoothed_errors = None

            logger.info(f"Model loaded successfully for sensor {self.sensor_id}")
            return True

        except FileNotFoundError as e:
            logger.error(f"File not found while loading model for sensor {self.sensor_id}: {e}")
            self.is_trained = False
            return False
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing metadata for sensor {self.sensor_id}: {e}")
            self.is_trained = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error loading model for sensor {self.sensor_id}: {e}")
            self.is_trained = False
            return False

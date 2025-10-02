"""
Transformer-based Time Series Forecaster
Implementation of Transformer architecture for multi-step ahead forecasting
"""

import os
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import pickle
import json
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

# Custom Exceptions
class ModelNotTrainedError(Exception):
    """Raised when attempting to use an untrained model"""
    pass

class InsufficientDataError(Exception):
    """Raised when there's not enough data for training or inference"""
    pass

# TensorFlow imports with fallback
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.layers import (
        Dense, Dropout, LayerNormalization, MultiHeadAttention,
        GlobalAveragePooling1D, Input, Embedding
    )
    TENSORFLOW_AVAILABLE = True
    print("[INFO] TensorFlow available for Transformer Forecaster")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("[WARNING] TensorFlow not available, using mock implementation for Transformer Forecaster")

    # Enhanced Mock implementations with informative logging
    logger = logging.getLogger(__name__)

    class Dense:
        def __init__(self, **kwargs):
            logger.debug("[MOCK] Mock Dense layer initialized")

    class Dropout:
        def __init__(self, **kwargs):
            logger.debug("[MOCK] Mock Dropout layer initialized")

    class LayerNormalization:
        def __init__(self, **kwargs):
            logger.debug("[MOCK] Mock LayerNormalization initialized")

    class MultiHeadAttention:
        def __init__(self, **kwargs):
            logger.debug("[MOCK] Mock MultiHeadAttention initialized")

    class GlobalAveragePooling1D:
        def __init__(self, **kwargs):
            logger.debug("[MOCK] Mock GlobalAveragePooling1D initialized")

    class Input:
        def __init__(self, **kwargs):
            logger.debug("[MOCK] Mock Input layer initialized")

    class Model:
        def __init__(self, **kwargs):
            logger.warning("[MOCK] Using mock Model - TensorFlow not available")

        def compile(self, **kwargs):
            logger.debug("[MOCK] Mock compile called - no actual compilation")

        def fit(self, **kwargs):
            logger.warning("[MOCK] Mock training - returning dummy history without actual training")
            return MockHistory()

        def predict(self, x):
            logger.warning("[MOCK] Mock prediction - returning zeros (no actual prediction)")
            return np.zeros((1 if len(x.shape) == 3 else len(x), 1, 1))

        def count_params(self):
            return 0

        def save(self, path):
            logger.warning("[MOCK] Cannot save model - TensorFlow not available")

    class MockHistory:
        def __init__(self):
            self.history = {'loss': [0.1, 0.08, 0.06], 'val_loss': [0.12, 0.09, 0.07]}

    class MockModels:
        @staticmethod
        def Model(*args, **kwargs):
            return Model()

        @staticmethod
        def load_model(path, **kwargs):
            logger.warning("[MOCK] Cannot load model - TensorFlow not available")
            return Model()

    class layers:
        class Layer:
            def __init__(self, **kwargs):
                pass

            def get_config(self):
                return {}

            def call(self, x, training=None):
                return x

        class Reshape:
            def __init__(self, **kwargs):
                pass

    models = MockModels()
    keras = type('MockKeras', (), {'models': models})()

logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    """Configuration for Transformer model"""
    # Model architecture
    sequence_length: int = 168  # 1 week of hourly data
    forecast_horizon: int = 24  # 24 hours ahead
    d_model: int = 128  # Model dimension
    num_heads: int = 8  # Number of attention heads
    num_layers: int = 4  # Number of transformer layers
    dff: int = 512  # Feed forward dimension
    dropout_rate: float = 0.1

    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    patience: int = 15
    verbose: int = 0  # 0=silent, 1=progress bar, 2=one line per epoch

    def __post_init__(self):
        """Validate configuration parameters"""
        # Validate ranges
        if self.sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {self.sequence_length}")

        if self.forecast_horizon <= 0:
            raise ValueError(f"forecast_horizon must be positive, got {self.forecast_horizon}")

        if not 0 <= self.dropout_rate <= 1:
            raise ValueError(f"dropout_rate must be between 0 and 1, got {self.dropout_rate}")

        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")

        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")

        # Validate architectural constraints
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
            )

        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")

        if self.dff <= 0:
            raise ValueError(f"dff must be positive, got {self.dff}")

        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if not 0 < self.validation_split < 1:
            raise ValueError(f"validation_split must be between 0 and 1, got {self.validation_split}")

        if self.patience <= 0:
            raise ValueError(f"patience must be positive, got {self.patience}")


class PositionalEncoding(layers.Layer):
    """Positional encoding layer for Transformer

    Note: This layer requires TensorFlow. If TensorFlow is not available,
    it will pass through input unchanged.
    """

    def __init__(self, sequence_length, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model

        # Initialize pos_encoding regardless of TensorFlow availability
        if TENSORFLOW_AVAILABLE:
            self.pos_encoding = self.positional_encoding(sequence_length, d_model)
        else:
            self.pos_encoding = None
            logger.debug("[MOCK] PositionalEncoding initialized without TensorFlow")

    def get_config(self):
        config = super().get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'd_model': self.d_model
        })
        return config

    def positional_encoding(self, length, depth):
        """Create positional encoding matrix

        Args:
            length: Sequence length
            depth: Model dimension

        Returns:
            TensorFlow tensor with positional encodings
        """
        depth = depth / 2
        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :] / depth

        angle_rates = 1 / (10000**depths)
        angle_rads = positions * angle_rates

        pos_encoding = np.concatenate([
            np.sin(angle_rads), np.cos(angle_rads)
        ], axis=-1)

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        """Apply positional encoding to input

        Args:
            x: Input tensor

        Returns:
            Input with positional encoding added (if TensorFlow available),
            otherwise returns input unchanged
        """
        if TENSORFLOW_AVAILABLE and self.pos_encoding is not None:
            return x + self.pos_encoding[tf.newaxis, :tf.shape(x)[1], :]
        return x


class TransformerBlock(layers.Layer):
    """Transformer block layer

    Note: This layer requires TensorFlow. If TensorFlow is not available,
    mock layers will be initialized and the layer will pass through input unchanged.
    """

    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        if TENSORFLOW_AVAILABLE:
            self.att = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
            self.ffn = keras.Sequential([
                Dense(dff, activation="relu"),
                Dense(d_model),
            ])

            self.layernorm1 = LayerNormalization(epsilon=1e-6)
            self.layernorm2 = LayerNormalization(epsilon=1e-6)

            self.dropout1 = Dropout(dropout_rate)
            self.dropout2 = Dropout(dropout_rate)
        else:
            # Initialize mock layers to ensure all attributes exist
            self.att = None
            self.ffn = None
            self.layernorm1 = None
            self.layernorm2 = None
            self.dropout1 = None
            self.dropout2 = None
            logger.debug("[MOCK] TransformerBlock initialized without TensorFlow")

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate
        })
        return config

    def call(self, x, training=None):
        if not TENSORFLOW_AVAILABLE:
            return x

        attn_output = self.att(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class TransformerForecaster:
    """
    Transformer-based time series forecaster
    """

    def __init__(self, sensor_id: str, config: Optional[TransformerConfig] = None):
        """
        Initialize Transformer forecaster

        Args:
            sensor_id: Unique identifier for the sensor
            config: Model configuration
        """
        self.sensor_id = sensor_id
        self.config = config or TransformerConfig()

        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

        # Training metadata
        self.training_history = None
        self.n_features = None

        logger.debug(f"Initialized Transformer forecaster for sensor {sensor_id}")

    def _build_model(self):
        """Build Transformer model

        Note: Requires self.n_features to be set before calling.
        Positional encoding is applied once at the beginning,
        then the encoded representation flows through transformer blocks.

        Returns:
            Compiled TensorFlow model or mock model if TensorFlow unavailable

        Raises:
            ValueError: If n_features not set
        """
        if self.n_features is None:
            raise ValueError("n_features must be set before building model")

        if not TENSORFLOW_AVAILABLE:
            return Model()

        # Input layer
        inputs = Input(shape=(self.config.sequence_length, self.n_features))

        # Project to model dimension
        x = Dense(self.config.d_model)(inputs)

        # Add positional encoding (applied once at the beginning)
        x = PositionalEncoding(self.config.sequence_length, self.config.d_model)(x)

        # Transformer blocks (sequentially process the positionally-encoded input)
        for _ in range(self.config.num_layers):
            x = TransformerBlock(
                d_model=self.config.d_model,
                num_heads=self.config.num_heads,
                dff=self.config.dff,
                dropout_rate=self.config.dropout_rate
            )(x)

        # Global average pooling
        x = GlobalAveragePooling1D()(x)

        # Output layers
        x = Dense(self.config.dff, activation="relu")(x)
        x = Dropout(self.config.dropout_rate)(x)
        outputs = Dense(self.config.forecast_horizon * self.n_features)(x)

        # Reshape to forecast format
        outputs = layers.Reshape((self.config.forecast_horizon, self.n_features))(outputs)

        model = models.Model(inputs=inputs, outputs=outputs)

        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )

        return model

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences and targets for training"""
        sequences = []
        targets = []

        seq_len = self.config.sequence_length
        horizon = self.config.forecast_horizon

        for i in range(len(data) - seq_len - horizon + 1):
            sequence = data[i:i + seq_len]
            target = data[i + seq_len:i + seq_len + horizon]

            sequences.append(sequence)
            targets.append(target)

        return np.array(sequences), np.array(targets)

    def train(self, training_data: np.ndarray) -> Dict[str, Any]:
        """Train the Transformer model

        Args:
            training_data: Training data as numpy array

        Returns:
            dict: Training results including metrics and model info

        Raises:
            InsufficientDataError: If not enough data for training
            ValueError: If data has invalid shape or values
        """
        logger.info(f"Training Transformer model for sensor {self.sensor_id}")

        # Prepare data
        if len(training_data.shape) == 1:
            training_data = training_data.reshape(-1, 1)

        self.n_features = training_data.shape[1]

        # Pre-check: Ensure minimum data length
        min_required_length = self.config.sequence_length + self.config.forecast_horizon
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

        # Build model
        self.model = self._build_model()

        if not TENSORFLOW_AVAILABLE:
            # Mock training for when TensorFlow is not available
            logger.warning(
                f"[MOCK] Training completed in mock mode for sensor {self.sensor_id}. "
                f"TensorFlow not available - model cannot be used for actual inference."
            )
            self.is_trained = False  # Mark as not trained since it's mock
            self.training_history = {'loss': [0.1, 0.08, 0.06], 'val_loss': [0.12, 0.09, 0.07]}
            return {
                'sensor_id': self.sensor_id,
                'training_samples': len(X),
                'model_parameters': 0,
                'final_loss': 0.1,
                'final_val_loss': 0.1,
                'note': 'Mock training - TensorFlow not available'
            }

        # Training callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.patience,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

        # Train model with configurable verbosity
        history = self.model.fit(
            X, y,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=self.config.validation_split,
            callbacks=callbacks_list,
            verbose=self.config.verbose
        )

        self.is_trained = True

        # Convert NumPy arrays to lists for JSON serialization
        self.training_history = {
            key: [float(v) for v in values] if isinstance(values, (list, np.ndarray)) else values
            for key, values in history.history.items()
        }

        logger.info(f"Training completed for sensor {self.sensor_id}")
        logger.info(f"Model parameters: {self.model.count_params()}")

        # Robust history access with fallbacks
        final_loss = history.history.get('loss', [0])[-1] if history.history.get('loss') else 0
        final_val_loss = history.history.get('val_loss', [0])[-1] if history.history.get('val_loss') else 0

        return {
            'sensor_id': self.sensor_id,
            'training_samples': len(X),
            'model_parameters': self.model.count_params(),
            'final_loss': float(final_loss),
            'final_val_loss': float(final_val_loss)
        }

    def predict(self, data: np.ndarray, horizon_hours: int = None) -> Dict[str, Any]:
        """Generate forecast for sensor data

        Args:
            data: Historical sensor data for forecasting
            horizon_hours: Number of hours to forecast (defaults to config.forecast_horizon)
                          Note: Model is trained for forecast_horizon, requesting more
                          hours won't extend the forecast beyond trained capacity

        Returns:
            dict: Forecast results with values and confidence intervals

        Raises:
            ModelNotTrainedError: If model hasn't been trained yet

        Note: If input data is shorter than sequence_length, it will be padded
        with the mean of available data. This is a simple strategy that may not
        be optimal for all use cases. Consider using interpolation or more
        sophisticated imputation for production systems.
        """
        if not self.is_trained:
            raise ModelNotTrainedError(f"Model not trained for sensor {self.sensor_id}")

        # Use config horizon if not specified
        if horizon_hours is None:
            horizon_hours = self.config.forecast_horizon

        # Ensure horizon doesn't exceed what model was trained for
        if horizon_hours > self.config.forecast_horizon:
            logger.warning(
                f"Requested horizon ({horizon_hours}) exceeds trained forecast_horizon "
                f"({self.config.forecast_horizon}). Using trained horizon."
            )
            horizon_hours = self.config.forecast_horizon

        # Prepare data
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        # Use only the required sequence length from the end
        if len(data) > self.config.sequence_length:
            data = data[-self.config.sequence_length:]
        elif len(data) < self.config.sequence_length:
            # Pad with mean if not enough data (simple strategy)
            # Note: This is a basic imputation method. For production, consider:
            # - Forward filling
            # - Linear interpolation
            # - Model-based imputation
            padding_length = self.config.sequence_length - len(data)
            mean_values = np.mean(data, axis=0)
            padding = np.tile(mean_values, (padding_length, 1))
            data = np.vstack([padding, data])
            logger.warning(
                f"Input data shorter than sequence_length. Padded with mean values. "
                f"Original: {len(data) - padding_length}, Padded to: {len(data)}"
            )

        # Scale data
        scaled_data = self.scaler.transform(data)

        # Reshape for model input
        input_data = scaled_data.reshape(1, self.config.sequence_length, self.n_features)

        # Generate forecast
        if TENSORFLOW_AVAILABLE and self.model:
            forecast_scaled = self.model.predict(input_data, verbose=0)
        else:
            # Mock prediction
            logger.warning("[MOCK] Generating random forecast - TensorFlow not available")
            forecast_scaled = np.random.randn(1, self.config.forecast_horizon, self.n_features)

        # Inverse transform
        forecast_scaled = forecast_scaled.reshape(-1, self.n_features)
        forecast = self.scaler.inverse_transform(forecast_scaled)

        # Limit to requested horizon
        if horizon_hours < len(forecast):
            forecast = forecast[:horizon_hours]

        # Calculate confidence intervals
        # Note: This is a simplified approach using forecast variance.
        # For more robust uncertainty quantification, consider:
        # - Quantile regression
        # - Monte Carlo dropout
        # - Ensemble forecasting
        # - Conformal prediction
        if len(forecast) > 1:
            forecast_std = np.std(forecast, axis=0)
        else:
            # Single timestep - use a default uncertainty
            forecast_std = np.abs(forecast[0] * 0.1)  # 10% of forecast value

        confidence_upper = forecast + 1.96 * forecast_std
        confidence_lower = forecast - 1.96 * forecast_std

        results = {
            'sensor_id': self.sensor_id,
            'forecast_values': forecast.flatten().tolist(),
            'confidence_upper': confidence_upper.flatten().tolist(),
            'confidence_lower': confidence_lower.flatten().tolist(),
            'horizon_hours': len(forecast),
            'forecast_quality': 'high' if self.is_trained else 'mock'
        }

        logger.info(f"Generated {len(forecast)} hour forecast for sensor {self.sensor_id}")

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
                self.model.save(model_dir / 'transformer_model.h5')
                logger.debug(f"Saved Keras model for sensor {self.sensor_id}")
            else:
                logger.warning(f"TensorFlow not available - Keras model not saved for {self.sensor_id}")

            # Save scaler
            with open(model_dir / 'scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)

            # Save metadata (training_history already converted to lists in train())
            metadata = {
                'sensor_id': self.sensor_id,
                'config': self.config.__dict__,
                'n_features': int(self.n_features) if self.n_features is not None else None,
                'is_trained': self.is_trained,
                'training_history': self.training_history,
                'tensorflow_available': TENSORFLOW_AVAILABLE
            }

            with open(model_dir / 'transformer_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Transformer model saved successfully for sensor {self.sensor_id} at {model_dir}")

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
            metadata_file = model_dir / 'transformer_metadata.json'
            if not metadata_file.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            self.sensor_id = metadata['sensor_id']
            self.config = TransformerConfig(**metadata['config'])
            self.n_features = metadata.get('n_features')
            self.training_history = metadata.get('training_history')

            # Check TensorFlow consistency
            model_file = model_dir / 'transformer_model.h5'
            was_saved_with_tf = metadata.get('tensorflow_available', True)

            if model_file.exists():
                if TENSORFLOW_AVAILABLE:
                    # Register custom layers and load model
                    custom_objects = {
                        'PositionalEncoding': PositionalEncoding,
                        'TransformerBlock': TransformerBlock
                    }
                    self.model = keras.models.load_model(
                        model_file,
                        custom_objects=custom_objects
                    )
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
            scaler_file = model_dir / 'scaler.pkl'
            if not scaler_file.exists():
                raise FileNotFoundError(f"Scaler file not found: {scaler_file}")

            with open(scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)

            logger.info(f"Transformer model loaded successfully for sensor {self.sensor_id}")
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
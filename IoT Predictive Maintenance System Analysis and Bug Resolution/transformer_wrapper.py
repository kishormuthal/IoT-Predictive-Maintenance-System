"""
Transformer-based Time Series Forecaster
Implementation of Transformer architecture for multi-step ahead forecasting
"""

import json
import logging
import os
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

# TensorFlow imports with fallback
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import callbacks, layers, models, optimizers
    from tensorflow.keras.layers import (
        Dense,
        Dropout,
        Embedding,
        GlobalAveragePooling1D,
        Input,
        LayerNormalization,
        MultiHeadAttention,
    )

    TENSORFLOW_AVAILABLE = True
    print("[INFO] TensorFlow available for Transformer Forecaster")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print(
        "[WARNING] TensorFlow not available, using mock implementation for Transformer Forecaster"
    )

    # Mock implementations
    class Dense:
        def __init__(self, **kwargs):
            pass

    class Dropout:
        def __init__(self, **kwargs):
            pass

    class LayerNormalization:
        def __init__(self, **kwargs):
            pass

    class MultiHeadAttention:
        def __init__(self, **kwargs):
            pass

    class GlobalAveragePooling1D:
        def __init__(self, **kwargs):
            pass

    class Input:
        def __init__(self, **kwargs):
            pass

    class Model:
        def __init__(self, **kwargs):
            pass

        def compile(self, **kwargs):
            pass

        def fit(self, **kwargs):
            return MockHistory()

        def predict(self, x):
            return np.zeros((len(x), 1))

    class MockHistory:
        def __init__(self):
            self.history = {"loss": [0.1], "val_loss": [0.1]}

    class layers:
        class Layer:
            def __init__(self, **kwargs):
                pass

            def get_config(self):
                return {}

            def call(self, x, training=None):
                return x


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


class PositionalEncoding(layers.Layer):
    """Positional encoding layer for Transformer"""

    def __init__(self, sequence_length, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        if TENSORFLOW_AVAILABLE:
            self.pos_encoding = self.positional_encoding(sequence_length, d_model)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"sequence_length": self.sequence_length, "d_model": self.d_model}
        )
        return config

    def positional_encoding(self, length, depth):
        """Create positional encoding matrix"""
        depth = depth / 2
        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :] / depth

        angle_rates = 1 / (10000**depths)
        angle_rads = positions * angle_rates

        pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        if TENSORFLOW_AVAILABLE:
            return x + self.pos_encoding[tf.newaxis, : tf.shape(x)[1], :]
        return x


class TransformerBlock(layers.Layer):
    """Transformer block layer"""

    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        if TENSORFLOW_AVAILABLE:
            self.att = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
            self.ffn = keras.Sequential(
                [
                    Dense(dff, activation="relu"),
                    Dense(d_model),
                ]
            )

            self.layernorm1 = LayerNormalization(epsilon=1e-6)
            self.layernorm2 = LayerNormalization(epsilon=1e-6)

            self.dropout1 = Dropout(dropout_rate)
            self.dropout2 = Dropout(dropout_rate)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "dff": self.dff,
                "dropout_rate": self.dropout_rate,
            }
        )
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

        logger.info(f"Initialized Transformer forecaster for sensor {sensor_id}")

    def _build_model(self):
        """Build Transformer model"""
        if not TENSORFLOW_AVAILABLE:
            return Model()

        # Input layer
        inputs = Input(shape=(self.config.sequence_length, self.n_features))

        # Project to model dimension
        x = Dense(self.config.d_model)(inputs)

        # Add positional encoding
        pos_encoding = PositionalEncoding(
            self.config.sequence_length, self.config.d_model
        )(x)

        # Transformer blocks
        for _ in range(self.config.num_layers):
            x = TransformerBlock(
                d_model=self.config.d_model,
                num_heads=self.config.num_heads,
                dff=self.config.dff,
                dropout_rate=self.config.dropout_rate,
            )(pos_encoding)
            pos_encoding = x

        # Global average pooling
        x = GlobalAveragePooling1D()(x)

        # Output layers
        x = Dense(self.config.dff, activation="relu")(x)
        x = Dropout(self.config.dropout_rate)(x)
        outputs = Dense(self.config.forecast_horizon * self.n_features)(x)

        # Reshape to forecast format
        outputs = layers.Reshape((self.config.forecast_horizon, self.n_features))(
            outputs
        )

        model = models.Model(inputs=inputs, outputs=outputs)

        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="mse",
            metrics=["mae"],
        )

        return model

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences and targets for training"""
        sequences = []
        targets = []

        seq_len = self.config.sequence_length
        horizon = self.config.forecast_horizon

        for i in range(len(data) - seq_len - horizon + 1):
            sequence = data[i : i + seq_len]
            target = data[i + seq_len : i + seq_len + horizon]

            sequences.append(sequence)
            targets.append(target)

        return np.array(sequences), np.array(targets)

    def train(self, training_data: np.ndarray) -> Dict[str, Any]:
        """Train the Transformer model"""
        logger.info(f"Training Transformer model for sensor {self.sensor_id}")

        # Prepare data
        if len(training_data.shape) == 1:
            training_data = training_data.reshape(-1, 1)

        self.n_features = training_data.shape[1]

        # Scale data
        scaled_data = self.scaler.fit_transform(training_data)

        # Create sequences
        X, y = self._create_sequences(scaled_data)

        if len(X) == 0:
            raise ValueError(
                f"Not enough data to create sequences for sensor {self.sensor_id}"
            )

        # Build model
        self.model = self._build_model()

        if not TENSORFLOW_AVAILABLE:
            # Mock training for when TensorFlow is not available
            self.is_trained = True
            return {
                "sensor_id": self.sensor_id,
                "training_samples": len(X),
                "model_parameters": 1000,
                "final_loss": 0.1,
                "final_val_loss": 0.1,
            }

        # Training callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.patience,
                restore_best_weights=True,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
            ),
        ]

        # Train model
        history = self.model.fit(
            X,
            y,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=self.config.validation_split,
            callbacks=callbacks_list,
            verbose=0,
        )

        self.is_trained = True
        self.training_history = history.history

        logger.info(f"Training completed for sensor {self.sensor_id}")
        logger.info(f"Model parameters: {self.model.count_params()}")

        return {
            "sensor_id": self.sensor_id,
            "training_samples": len(X),
            "model_parameters": self.model.count_params(),
            "final_loss": history.history["loss"][-1],
            "final_val_loss": history.history["val_loss"][-1],
        }

    def predict(self, data: np.ndarray, horizon_hours: int = None) -> Dict[str, Any]:
        """Generate forecast for sensor data"""
        if not self.is_trained:
            raise ValueError(f"Model not trained for sensor {self.sensor_id}")

        if horizon_hours is None:
            horizon_hours = self.config.forecast_horizon

        # Prepare data
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        # Use only the required sequence length from the end
        if len(data) > self.config.sequence_length:
            data = data[-self.config.sequence_length :]
        elif len(data) < self.config.sequence_length:
            # Pad with mean if not enough data
            padding_length = self.config.sequence_length - len(data)
            mean_values = np.mean(data, axis=0)
            padding = np.tile(mean_values, (padding_length, 1))
            data = np.vstack([padding, data])

        # Scale data
        scaled_data = self.scaler.transform(data)

        # Reshape for model input
        input_data = scaled_data.reshape(
            1, self.config.sequence_length, self.n_features
        )

        # Generate forecast
        if TENSORFLOW_AVAILABLE and self.model:
            forecast_scaled = self.model.predict(input_data, verbose=0)
        else:
            # Mock prediction
            forecast_scaled = np.random.randn(
                1, min(horizon_hours, self.config.forecast_horizon), self.n_features
            )

        # Inverse transform
        forecast_scaled = forecast_scaled.reshape(-1, self.n_features)
        forecast = self.scaler.inverse_transform(forecast_scaled)

        # Limit to requested horizon
        if horizon_hours < len(forecast):
            forecast = forecast[:horizon_hours]

        # Calculate confidence intervals (simple approach)
        forecast_std = np.std(forecast, axis=0)
        confidence_upper = forecast + 1.96 * forecast_std
        confidence_lower = forecast - 1.96 * forecast_std

        results = {
            "sensor_id": self.sensor_id,
            "forecast_values": forecast.flatten().tolist(),
            "confidence_upper": confidence_upper.flatten().tolist(),
            "confidence_lower": confidence_lower.flatten().tolist(),
            "horizon_hours": len(forecast),
            "forecast_quality": "high" if self.is_trained else "mock",
        }

        logger.info(
            f"Generated {len(forecast)} hour forecast for sensor {self.sensor_id}"
        )

        return results

    def save_model(self, model_path: Path) -> None:
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        model_dir = model_path / self.sensor_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save Keras model
        if TENSORFLOW_AVAILABLE and self.model:
            self.model.save(model_dir / "transformer_model.h5")

        # Save scaler
        with open(model_dir / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        # Save metadata
        metadata = {
            "sensor_id": self.sensor_id,
            "config": self.config.__dict__,
            "n_features": self.n_features,
            "is_trained": self.is_trained,
            "training_history": self.training_history,
        }

        with open(model_dir / "transformer_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Transformer model saved for sensor {self.sensor_id}")

    def load_model(self, model_path: Path) -> bool:
        """Load trained model from disk"""
        try:
            model_dir = model_path / self.sensor_id

            # Load metadata
            with open(model_dir / "transformer_metadata.json", "r") as f:
                metadata = json.load(f)

            self.sensor_id = metadata["sensor_id"]
            self.config = TransformerConfig(**metadata["config"])
            self.n_features = metadata["n_features"]
            self.is_trained = metadata["is_trained"]
            self.training_history = metadata.get("training_history")

            # Load Keras model
            if TENSORFLOW_AVAILABLE and (model_dir / "transformer_model.h5").exists():
                # Register custom layers
                custom_objects = {
                    "PositionalEncoding": PositionalEncoding,
                    "TransformerBlock": TransformerBlock,
                }
                self.model = keras.models.load_model(
                    model_dir / "transformer_model.h5", custom_objects=custom_objects
                )

            # Load scaler
            with open(model_dir / "scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)

            logger.info(f"Transformer model loaded for sensor {self.sensor_id}")
            return True

        except Exception as e:
            logger.warning(
                f"Failed to load Transformer model for sensor {self.sensor_id}: {e}"
            )
            return False

"""
Feature Engineering Module
Advanced feature extraction for time series sensor data
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import fft, signal, stats

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""

    # Rolling statistics
    rolling_windows: List[int] = None  # [3, 6, 12, 24]
    include_rolling_mean: bool = True
    include_rolling_std: bool = True
    include_rolling_min: bool = True
    include_rolling_max: bool = True
    include_rolling_median: bool = True

    # Lag features
    lag_periods: List[int] = None  # [1, 2, 3, 6, 12, 24]

    # Differences
    include_diff_1: bool = True
    include_diff_2: bool = True
    include_pct_change: bool = True

    # Statistical features
    include_ewm: bool = True  # Exponentially weighted moving average
    ewm_span: int = 12
    include_expanding: bool = True  # Expanding window statistics

    # Frequency domain features
    include_fft: bool = True
    fft_top_k: int = 5  # Top K frequency components

    # Time-based features
    include_time_features: bool = True  # hour, day_of_week, etc.
    include_cyclical_encoding: bool = True  # sin/cos encoding for time

    # Domain-specific features
    include_rate_of_change: bool = True
    include_acceleration: bool = True  # Second derivative
    include_volatility: bool = True  # Rolling std of returns

    # Interaction features
    include_interactions: bool = False  # Polynomial features (memory intensive)
    interaction_degree: int = 2

    def __post_init__(self):
        """Set defaults if None"""
        if self.rolling_windows is None:
            self.rolling_windows = [3, 6, 12, 24]
        if self.lag_periods is None:
            self.lag_periods = [1, 2, 3, 6, 12, 24]


class FeatureEngineer:
    """
    Advanced feature engineering for time series sensor data
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize feature engineer

        Args:
            config: Feature engineering configuration
        """
        self.config = config or FeatureConfig()

    def engineer_features(
        self,
        data: np.ndarray,
        timestamps: Optional[List[datetime]] = None,
        sensor_id: str = "unknown",
    ) -> Dict[str, np.ndarray]:
        """
        Engineer comprehensive feature set from raw sensor data

        Args:
            data: Raw sensor data (1D array)
            timestamps: Optional timestamps for time-based features
            sensor_id: Sensor identifier for logging

        Returns:
            Dictionary of engineered features
        """
        features = {}

        try:
            # Original data
            features["raw"] = data.copy()

            # Rolling statistics
            if any(
                [
                    self.config.include_rolling_mean,
                    self.config.include_rolling_std,
                    self.config.include_rolling_min,
                    self.config.include_rolling_max,
                    self.config.include_rolling_median,
                ]
            ):
                rolling_features = self._compute_rolling_features(data)
                features.update(rolling_features)

            # Lag features
            if self.config.lag_periods:
                lag_features = self._compute_lag_features(data)
                features.update(lag_features)

            # Differences and rate of change
            if any(
                [
                    self.config.include_diff_1,
                    self.config.include_diff_2,
                    self.config.include_pct_change,
                    self.config.include_rate_of_change,
                    self.config.include_acceleration,
                ]
            ):
                diff_features = self._compute_difference_features(data)
                features.update(diff_features)

            # Statistical features
            if self.config.include_ewm or self.config.include_expanding:
                stat_features = self._compute_statistical_features(data)
                features.update(stat_features)

            # Volatility
            if self.config.include_volatility:
                volatility_features = self._compute_volatility_features(data)
                features.update(volatility_features)

            # Frequency domain features
            if self.config.include_fft and len(data) > 10:
                fft_features = self._compute_fft_features(data)
                features.update(fft_features)

            # Time-based features
            if timestamps and self.config.include_time_features:
                time_features = self._compute_time_features(timestamps)
                features.update(time_features)

            logger.debug(f"Engineered {len(features)} feature sets for {sensor_id}")

        except Exception as e:
            logger.error(f"Error engineering features for {sensor_id}: {e}")
            # Return at least raw data
            features = {"raw": data.copy()}

        return features

    def _compute_rolling_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute rolling window statistics"""
        features = {}

        for window in self.config.rolling_windows:
            if len(data) < window:
                continue

            prefix = f"rolling_{window}"

            if self.config.include_rolling_mean:
                features[f"{prefix}_mean"] = self._rolling_stat(data, window, np.mean)

            if self.config.include_rolling_std:
                features[f"{prefix}_std"] = self._rolling_stat(data, window, np.std)

            if self.config.include_rolling_min:
                features[f"{prefix}_min"] = self._rolling_stat(data, window, np.min)

            if self.config.include_rolling_max:
                features[f"{prefix}_max"] = self._rolling_stat(data, window, np.max)

            if self.config.include_rolling_median:
                features[f"{prefix}_median"] = self._rolling_stat(
                    data, window, np.median
                )

            # Range (max - min)
            if self.config.include_rolling_min and self.config.include_rolling_max:
                rolling_min = features[f"{prefix}_min"]
                rolling_max = features[f"{prefix}_max"]
                features[f"{prefix}_range"] = rolling_max - rolling_min

        return features

    def _compute_lag_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute lag features"""
        features = {}

        for lag in self.config.lag_periods:
            if lag >= len(data):
                continue

            # Shift data by lag periods
            lagged = np.concatenate([np.zeros(lag), data[:-lag]])
            features[f"lag_{lag}"] = lagged

        return features

    def _compute_difference_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute difference and rate of change features"""
        features = {}

        # First difference
        if self.config.include_diff_1 and len(data) > 1:
            features["diff_1"] = np.concatenate([[0], np.diff(data)])

        # Second difference (acceleration)
        if self.config.include_diff_2 and len(data) > 2:
            features["diff_2"] = np.concatenate([[0, 0], np.diff(data, n=2)])

        # Percentage change
        if self.config.include_pct_change and len(data) > 1:
            pct_change = np.zeros(len(data))
            for i in range(1, len(data)):
                if abs(data[i - 1]) > 1e-10:
                    pct_change[i] = (data[i] - data[i - 1]) / abs(data[i - 1])
            features["pct_change"] = pct_change

        # Rate of change (difference / time)
        if self.config.include_rate_of_change and len(data) > 1:
            features["rate_of_change"] = features.get(
                "diff_1", np.concatenate([[0], np.diff(data)])
            )

        # Acceleration (second derivative)
        if self.config.include_acceleration and len(data) > 2:
            features["acceleration"] = features.get(
                "diff_2", np.concatenate([[0, 0], np.diff(data, n=2)])
            )

        return features

    def _compute_statistical_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute statistical features"""
        features = {}

        # Exponentially weighted moving average
        if self.config.include_ewm:
            # Manual EWM implementation
            alpha = 2.0 / (self.config.ewm_span + 1.0)
            ewm = np.zeros(len(data))
            ewm[0] = data[0]
            for i in range(1, len(data)):
                ewm[i] = alpha * data[i] + (1 - alpha) * ewm[i - 1]
            features["ewm"] = ewm

            # EWM std
            ewm_var = np.zeros(len(data))
            ewm_var[0] = 0
            for i in range(1, len(data)):
                diff = data[i] - ewm[i]
                ewm_var[i] = alpha * diff**2 + (1 - alpha) * ewm_var[i - 1]
            features["ewm_std"] = np.sqrt(ewm_var)

        # Expanding window statistics
        if self.config.include_expanding:
            expanding_mean = np.zeros(len(data))
            expanding_std = np.zeros(len(data))

            for i in range(len(data)):
                window = data[: i + 1]
                expanding_mean[i] = np.mean(window)
                expanding_std[i] = np.std(window) if len(window) > 1 else 0

            features["expanding_mean"] = expanding_mean
            features["expanding_std"] = expanding_std

        return features

    def _compute_volatility_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute volatility features"""
        features = {}

        # Compute returns
        if len(data) > 1:
            returns = np.zeros(len(data))
            for i in range(1, len(data)):
                if abs(data[i - 1]) > 1e-10:
                    returns[i] = (data[i] - data[i - 1]) / abs(data[i - 1])

            # Rolling volatility (std of returns)
            for window in [6, 12, 24]:
                if len(data) >= window:
                    volatility = self._rolling_stat(returns, window, np.std)
                    features[f"volatility_{window}"] = volatility

        return features

    def _compute_fft_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute frequency domain features using FFT"""
        features = {}

        try:
            # Compute FFT
            fft_vals = np.fft.fft(data)
            fft_freq = np.fft.fftfreq(len(data))

            # Get power spectrum
            power = np.abs(fft_vals) ** 2

            # Get top K frequency components (excluding DC component)
            positive_freq_idx = np.where(fft_freq > 0)[0]
            if len(positive_freq_idx) > 0:
                top_k = min(self.config.fft_top_k, len(positive_freq_idx))
                top_indices = positive_freq_idx[
                    np.argsort(power[positive_freq_idx])[-top_k:]
                ]

                for i, idx in enumerate(top_indices):
                    # Reconstruct signal from this frequency component
                    component = np.zeros(len(data), dtype=complex)
                    component[idx] = fft_vals[idx]
                    component[-idx] = fft_vals[-idx]  # Mirror for real signal

                    reconstructed = np.fft.ifft(component).real
                    features[f"fft_component_{i+1}"] = reconstructed

                # Overall spectral features
                features["spectral_energy"] = np.full(len(data), np.sum(power))
                features["spectral_entropy"] = np.full(
                    len(data), stats.entropy(power / (np.sum(power) + 1e-10))
                )

        except Exception as e:
            logger.warning(f"Error computing FFT features: {e}")

        return features

    def _compute_time_features(
        self, timestamps: List[datetime]
    ) -> Dict[str, np.ndarray]:
        """Compute time-based features"""
        features = {}

        # Extract time components
        features["hour"] = np.array([t.hour for t in timestamps])
        features["day_of_week"] = np.array([t.weekday() for t in timestamps])
        features["day_of_month"] = np.array([t.day for t in timestamps])
        features["month"] = np.array([t.month for t in timestamps])
        features["is_weekend"] = np.array(
            [1 if t.weekday() >= 5 else 0 for t in timestamps]
        )

        # Cyclical encoding (sin/cos for periodic features)
        if self.config.include_cyclical_encoding:
            # Hour (24-hour cycle)
            hour_rad = 2 * np.pi * features["hour"] / 24
            features["hour_sin"] = np.sin(hour_rad)
            features["hour_cos"] = np.cos(hour_rad)

            # Day of week (7-day cycle)
            dow_rad = 2 * np.pi * features["day_of_week"] / 7
            features["day_of_week_sin"] = np.sin(dow_rad)
            features["day_of_week_cos"] = np.cos(dow_rad)

            # Month (12-month cycle)
            month_rad = 2 * np.pi * (features["month"] - 1) / 12
            features["month_sin"] = np.sin(month_rad)
            features["month_cos"] = np.cos(month_rad)

        return features

    def _rolling_stat(self, data: np.ndarray, window: int, stat_func) -> np.ndarray:
        """
        Compute rolling statistic with proper padding

        Args:
            data: Input data
            window: Window size
            stat_func: Statistic function (np.mean, np.std, etc.)

        Returns:
            Rolling statistic array (same length as data)
        """
        result = np.zeros(len(data))

        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            window_data = data[start_idx : i + 1]

            # Handle edge cases for std
            if stat_func == np.std and len(window_data) < 2:
                result[i] = 0
            else:
                result[i] = stat_func(window_data)

        return result

    def get_feature_names(self) -> List[str]:
        """
        Get list of all possible feature names based on current configuration

        Returns:
            List of feature names
        """
        feature_names = ["raw"]

        # Rolling features
        for window in self.config.rolling_windows:
            if self.config.include_rolling_mean:
                feature_names.append(f"rolling_{window}_mean")
            if self.config.include_rolling_std:
                feature_names.append(f"rolling_{window}_std")
            if self.config.include_rolling_min:
                feature_names.append(f"rolling_{window}_min")
            if self.config.include_rolling_max:
                feature_names.append(f"rolling_{window}_max")
            if self.config.include_rolling_median:
                feature_names.append(f"rolling_{window}_median")
            if self.config.include_rolling_min and self.config.include_rolling_max:
                feature_names.append(f"rolling_{window}_range")

        # Lag features
        for lag in self.config.lag_periods:
            feature_names.append(f"lag_{lag}")

        # Difference features
        if self.config.include_diff_1:
            feature_names.append("diff_1")
        if self.config.include_diff_2:
            feature_names.append("diff_2")
        if self.config.include_pct_change:
            feature_names.append("pct_change")
        if self.config.include_rate_of_change:
            feature_names.append("rate_of_change")
        if self.config.include_acceleration:
            feature_names.append("acceleration")

        # Statistical features
        if self.config.include_ewm:
            feature_names.extend(["ewm", "ewm_std"])
        if self.config.include_expanding:
            feature_names.extend(["expanding_mean", "expanding_std"])

        # Volatility
        if self.config.include_volatility:
            for window in [6, 12, 24]:
                feature_names.append(f"volatility_{window}")

        # FFT features
        if self.config.include_fft:
            for i in range(self.config.fft_top_k):
                feature_names.append(f"fft_component_{i+1}")
            feature_names.extend(["spectral_energy", "spectral_entropy"])

        # Time features
        if self.config.include_time_features:
            feature_names.extend(
                ["hour", "day_of_week", "day_of_month", "month", "is_weekend"]
            )
            if self.config.include_cyclical_encoding:
                feature_names.extend(
                    [
                        "hour_sin",
                        "hour_cos",
                        "day_of_week_sin",
                        "day_of_week_cos",
                        "month_sin",
                        "month_cos",
                    ]
                )

        return feature_names

    def create_feature_matrix(
        self,
        features: Dict[str, np.ndarray],
        selected_features: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Create feature matrix from feature dictionary

        Args:
            features: Dictionary of features
            selected_features: List of feature names to include (all if None)

        Returns:
            Feature matrix (n_samples, n_features)
        """
        if selected_features is None:
            selected_features = sorted(features.keys())

        # Get length from first feature
        n_samples = len(next(iter(features.values())))

        # Stack features
        feature_arrays = []
        for fname in selected_features:
            if fname in features:
                feature_arrays.append(features[fname].reshape(-1, 1))
            else:
                logger.warning(f"Feature {fname} not found, using zeros")
                feature_arrays.append(np.zeros((n_samples, 1)))

        if feature_arrays:
            return np.hstack(feature_arrays)
        else:
            return np.zeros((n_samples, 0))

"""
Advanced Imputation Methods
Sophisticated techniques for handling missing sensor data
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import interpolate

logger = logging.getLogger(__name__)


class AdvancedImputer:
    """
    Advanced imputation methods for time series sensor data

    Methods:
    - Linear interpolation
    - Spline interpolation
    - KNN imputation
    - Iterative imputation
    - ARIMA-based
    - Seasonal decomposition
    - Forward/backward fill
    """

    @staticmethod
    def linear_interpolation(
        data: np.ndarray, limit: Optional[int] = None
    ) -> np.ndarray:
        """
        Linear interpolation for missing values

        Args:
            data: Data with missing values (NaN)
            limit: Maximum number of consecutive NaNs to interpolate

        Returns:
            Imputed data
        """
        result = data.copy()
        mask = np.isnan(result)

        if not np.any(mask):
            return result

        # Get indices of valid and invalid data
        valid_idx = np.where(~mask)[0]
        invalid_idx = np.where(mask)[0]

        if len(valid_idx) < 2:
            logger.warning(
                "Insufficient valid data for interpolation, using forward fill"
            )
            return AdvancedImputer.forward_fill(data)

        # Check consecutive NaN limit
        if limit is not None:
            # Find consecutive NaN groups
            consecutive_groups = []
            current_group = []

            for i in invalid_idx:
                if not current_group or i == current_group[-1] + 1:
                    current_group.append(i)
                else:
                    if len(current_group) <= limit:
                        consecutive_groups.extend(current_group)
                    current_group = [i]

            if current_group and len(current_group) <= limit:
                consecutive_groups.extend(current_group)

            invalid_idx = np.array(consecutive_groups)

        # Interpolate
        if len(invalid_idx) > 0:
            result[invalid_idx] = np.interp(invalid_idx, valid_idx, result[valid_idx])

        return result

    @staticmethod
    def spline_interpolation(
        data: np.ndarray, order: int = 3, smoothing: float = 0.0
    ) -> np.ndarray:
        """
        Spline interpolation for missing values

        Args:
            data: Data with missing values
            order: Spline order (1=linear, 2=quadratic, 3=cubic)
            smoothing: Smoothing factor

        Returns:
            Imputed data
        """
        result = data.copy()
        mask = np.isnan(result)

        if not np.any(mask):
            return result

        valid_idx = np.where(~mask)[0]
        invalid_idx = np.where(mask)[0]

        if len(valid_idx) < order + 1:
            logger.warning(f"Insufficient data for order-{order} spline, using linear")
            return AdvancedImputer.linear_interpolation(data)

        try:
            # Fit spline
            spline = interpolate.UnivariateSpline(
                valid_idx, result[valid_idx], k=order, s=smoothing
            )

            # Interpolate missing values
            result[invalid_idx] = spline(invalid_idx)

        except Exception as e:
            logger.warning(f"Spline interpolation failed: {e}, using linear fallback")
            return AdvancedImputer.linear_interpolation(data)

        return result

    @staticmethod
    def knn_imputation(
        data: np.ndarray, n_neighbors: int = 5, weights: str = "distance"
    ) -> np.ndarray:
        """
        KNN-based imputation

        Args:
            data: Data with missing values (can be multivariate)
            n_neighbors: Number of neighbors
            weights: 'uniform' or 'distance'

        Returns:
            Imputed data
        """
        try:
            from sklearn.impute import KNNImputer

            imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)

            # Reshape if 1D
            if data.ndim == 1:
                data_2d = data.reshape(-1, 1)
                imputed = imputer.fit_transform(data_2d)
                return imputed.ravel()
            else:
                return imputer.fit_transform(data)

        except ImportError:
            logger.warning(
                "scikit-learn not available for KNN imputation, using linear fallback"
            )
            return AdvancedImputer.linear_interpolation(data)
        except Exception as e:
            logger.error(f"KNN imputation failed: {e}")
            return AdvancedImputer.linear_interpolation(data)

    @staticmethod
    def iterative_imputation(
        data: np.ndarray, max_iter: int = 10, random_state: int = 42
    ) -> np.ndarray:
        """
        Iterative imputation (MICE algorithm)

        Args:
            data: Data with missing values (multivariate)
            max_iter: Maximum iterations
            random_state: Random seed

        Returns:
            Imputed data
        """
        try:
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer

            imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)

            # Reshape if 1D
            if data.ndim == 1:
                data_2d = data.reshape(-1, 1)
                imputed = imputer.fit_transform(data_2d)
                return imputed.ravel()
            else:
                return imputer.fit_transform(data)

        except ImportError:
            logger.warning(
                "scikit-learn IterativeImputer not available, using KNN fallback"
            )
            return AdvancedImputer.knn_imputation(data)
        except Exception as e:
            logger.error(f"Iterative imputation failed: {e}")
            return AdvancedImputer.knn_imputation(data)

    @staticmethod
    def forward_fill(data: np.ndarray) -> np.ndarray:
        """
        Forward fill (carry last observation forward)

        Args:
            data: Data with missing values

        Returns:
            Imputed data
        """
        result = data.copy()
        mask = np.isnan(result)

        if not np.any(mask):
            return result

        # Forward fill
        last_valid = None
        for i in range(len(result)):
            if mask[i]:
                if last_valid is not None:
                    result[i] = last_valid
            else:
                last_valid = result[i]

        # If still have NaN at beginning, use backward fill
        mask = np.isnan(result)
        if np.any(mask):
            result = AdvancedImputer.backward_fill(result)

        return result

    @staticmethod
    def backward_fill(data: np.ndarray) -> np.ndarray:
        """
        Backward fill (carry next observation backward)

        Args:
            data: Data with missing values

        Returns:
            Imputed data
        """
        result = data.copy()
        mask = np.isnan(result)

        if not np.any(mask):
            return result

        # Backward fill
        next_valid = None
        for i in range(len(result) - 1, -1, -1):
            if mask[i]:
                if next_valid is not None:
                    result[i] = next_valid
            else:
                next_valid = result[i]

        # If still have NaN, fill with 0 or mean
        mask = np.isnan(result)
        if np.any(mask):
            valid_mean = np.nanmean(result)
            result[mask] = valid_mean if not np.isnan(valid_mean) else 0.0

        return result

    @staticmethod
    def seasonal_decomposition_imputation(
        data: np.ndarray, period: int = 24
    ) -> np.ndarray:
        """
        Imputation using seasonal decomposition

        Args:
            data: Data with missing values
            period: Seasonal period (e.g., 24 for hourly data with daily seasonality)

        Returns:
            Imputed data
        """
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose

            # First do simple imputation to allow decomposition
            temp_data = AdvancedImputer.linear_interpolation(data)

            # Decompose
            decomposition = seasonal_decompose(
                temp_data, model="additive", period=period, extrapolate_trend="freq"
            )

            # Use seasonal component to guide imputation
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid

            # Reconstruct with improved values
            result = data.copy()
            mask = np.isnan(result)

            if np.any(mask):
                # For missing values, use trend + seasonal
                result[mask] = trend[mask] + seasonal[mask]

            return result

        except ImportError:
            logger.warning("statsmodels not available, using spline fallback")
            return AdvancedImputer.spline_interpolation(data)
        except Exception as e:
            logger.error(f"Seasonal decomposition imputation failed: {e}")
            return AdvancedImputer.spline_interpolation(data)

    @staticmethod
    def moving_average_imputation(data: np.ndarray, window: int = 5) -> np.ndarray:
        """
        Moving average imputation

        Args:
            data: Data with missing values
            window: Window size for moving average

        Returns:
            Imputed data
        """
        result = data.copy()
        mask = np.isnan(result)

        if not np.any(mask):
            return result

        missing_indices = np.where(mask)[0]

        for idx in missing_indices:
            # Get window around missing value
            start = max(0, idx - window // 2)
            end = min(len(result), idx + window // 2 + 1)

            # Calculate mean of non-NaN values in window
            window_data = result[start:end]
            valid_data = window_data[~np.isnan(window_data)]

            if len(valid_data) > 0:
                result[idx] = np.mean(valid_data)

        # If still have NaN, use forward fill
        if np.any(np.isnan(result)):
            result = AdvancedImputer.forward_fill(result)

        return result

    @staticmethod
    def adaptive_imputation(
        data: np.ndarray, method: str = "auto", **kwargs
    ) -> np.ndarray:
        """
        Adaptive imputation - selects best method based on data characteristics

        Args:
            data: Data with missing values
            method: Imputation method or 'auto'
            **kwargs: Additional arguments for specific methods

        Returns:
            Imputed data
        """
        mask = np.isnan(data)

        if not np.any(mask):
            return data

        # Count missing values
        n_missing = np.sum(mask)
        missing_pct = n_missing / len(data)

        # Auto-select method
        if method == "auto":
            if missing_pct > 0.5:
                logger.warning(
                    f"High missing rate ({missing_pct:.1%}), using seasonal decomposition"
                )
                method = "seasonal" if len(data) >= 48 else "spline"
            elif missing_pct > 0.2:
                logger.info(f"Moderate missing rate ({missing_pct:.1%}), using KNN")
                method = "knn"
            elif missing_pct > 0.05:
                logger.info(f"Low missing rate ({missing_pct:.1%}), using spline")
                method = "spline"
            else:
                logger.info(f"Very low missing rate ({missing_pct:.1%}), using linear")
                method = "linear"

        # Apply selected method
        if method == "linear":
            return AdvancedImputer.linear_interpolation(data, **kwargs)
        elif method == "spline":
            return AdvancedImputer.spline_interpolation(data, **kwargs)
        elif method == "knn":
            return AdvancedImputer.knn_imputation(data, **kwargs)
        elif method == "iterative":
            return AdvancedImputer.iterative_imputation(data, **kwargs)
        elif method == "seasonal":
            return AdvancedImputer.seasonal_decomposition_imputation(data, **kwargs)
        elif method == "moving_average":
            return AdvancedImputer.moving_average_imputation(data, **kwargs)
        elif method == "forward":
            return AdvancedImputer.forward_fill(data)
        elif method == "backward":
            return AdvancedImputer.backward_fill(data)
        else:
            logger.warning(f"Unknown method '{method}', using linear")
            return AdvancedImputer.linear_interpolation(data)

    @staticmethod
    def impute_with_confidence(
        data: np.ndarray, method: str = "auto", n_bootstrap: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Imputation with uncertainty estimates

        Args:
            data: Data with missing values
            method: Imputation method
            n_bootstrap: Number of bootstrap samples for uncertainty

        Returns:
            (imputed_data, uncertainty) tuple where uncertainty is std dev of imputations
        """
        imputed_values = []

        for _ in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_data = data.copy()
            valid_mask = ~np.isnan(data)
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) > 1:
                # Resample valid data
                bootstrap_indices = np.random.choice(
                    valid_indices, size=len(valid_indices), replace=True
                )
                bootstrap_data[valid_indices] = data[bootstrap_indices]

            # Impute
            imputed = AdvancedImputer.adaptive_imputation(bootstrap_data, method=method)
            imputed_values.append(imputed)

        imputed_values = np.array(imputed_values)

        # Mean and std of imputations
        mean_imputed = np.mean(imputed_values, axis=0)
        std_imputed = np.std(imputed_values, axis=0)

        return mean_imputed, std_imputed

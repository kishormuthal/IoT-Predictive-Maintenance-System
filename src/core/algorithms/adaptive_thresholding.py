"""
Advanced Adaptive Thresholding Algorithms
Statistical methods for dynamic anomaly detection thresholds
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ThresholdResult:
    """Result of threshold calculation"""
    threshold: float
    method: str
    parameters: Dict[str, float]
    confidence_level: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'threshold': float(self.threshold),
            'method': self.method,
            'parameters': self.parameters,
            'confidence_level': float(self.confidence_level)
        }


class AdaptiveThresholdCalculator:
    """
    Advanced adaptive thresholding algorithms

    Methods:
    - GEV (Generalized Extreme Value) distribution
    - POT (Peaks Over Threshold)
    - Isolation Forest-based
    - Local Outlier Factor
    - Statistical (Z-score, IQR, MAD)
    """

    @staticmethod
    def gev_threshold(
        data: np.ndarray,
        confidence_level: float = 0.99,
        block_size: int = 100
    ) -> ThresholdResult:
        """
        Calculate threshold using Generalized Extreme Value (GEV) distribution

        Uses block maxima approach for extreme value theory

        Args:
            data: Historical data
            confidence_level: Confidence level (0-1), e.g., 0.99 for 99%
            block_size: Size of blocks for maxima extraction

        Returns:
            ThresholdResult
        """
        try:
            if len(data) < block_size:
                logger.warning(f"Data length {len(data)} < block_size {block_size}, using statistical fallback")
                return AdaptiveThresholdCalculator.zscore_threshold(data, confidence_level)

            # Extract block maxima
            n_blocks = len(data) // block_size
            block_maxima = []

            for i in range(n_blocks):
                block = data[i * block_size:(i + 1) * block_size]
                block_maxima.append(np.max(block))

            block_maxima = np.array(block_maxima)

            # Fit GEV distribution
            # GEV has 3 parameters: shape (xi), location (mu), scale (sigma)
            params = stats.genextreme.fit(block_maxima)
            shape, loc, scale = params

            # Calculate threshold at confidence level
            # Using the inverse CDF (percent point function)
            threshold = stats.genextreme.ppf(confidence_level, shape, loc=loc, scale=scale)

            logger.info(
                f"GEV threshold calculated: {threshold:.4f} "
                f"(shape={shape:.4f}, loc={loc:.4f}, scale={scale:.4f})"
            )

            return ThresholdResult(
                threshold=threshold,
                method="GEV",
                parameters={
                    'shape': float(shape),
                    'location': float(loc),
                    'scale': float(scale),
                    'block_size': block_size,
                    'n_blocks': n_blocks
                },
                confidence_level=confidence_level
            )

        except Exception as e:
            logger.error(f"Error calculating GEV threshold: {e}")
            # Fallback to z-score method
            return AdaptiveThresholdCalculator.zscore_threshold(data, confidence_level)

    @staticmethod
    def pot_threshold(
        data: np.ndarray,
        initial_threshold: Optional[float] = None,
        confidence_level: float = 0.99
    ) -> ThresholdResult:
        """
        Peaks Over Threshold (POT) method using Generalized Pareto Distribution

        Args:
            data: Historical data
            initial_threshold: Initial threshold for peak selection (uses 90th percentile if None)
            confidence_level: Confidence level

        Returns:
            ThresholdResult
        """
        try:
            # Set initial threshold (typically 90th percentile)
            if initial_threshold is None:
                initial_threshold = np.percentile(data, 90)

            # Extract exceedances (peaks over threshold)
            exceedances = data[data > initial_threshold] - initial_threshold

            if len(exceedances) < 10:
                logger.warning("Too few exceedances for POT, using statistical fallback")
                return AdaptiveThresholdCalculator.zscore_threshold(data, confidence_level)

            # Fit Generalized Pareto Distribution to exceedances
            # GPD has 2 parameters: shape (xi) and scale (sigma)
            shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)

            # Calculate threshold
            # Number of exceedances
            n_exceedances = len(exceedances)
            n_total = len(data)

            # Exceedance probability
            p_exceed = n_exceedances / n_total

            # Calculate return level for given confidence
            return_period = 1 / (1 - confidence_level)

            # GPD-based threshold
            if abs(shape) > 1e-10:
                threshold = initial_threshold + (scale / shape) * (
                    ((n_total / n_exceedances) * (1 - confidence_level)) ** (-shape) - 1
                )
            else:
                # Exponential case (shape ≈ 0)
                threshold = initial_threshold - scale * np.log((n_total / n_exceedances) * (1 - confidence_level))

            logger.info(
                f"POT threshold calculated: {threshold:.4f} "
                f"(shape={shape:.4f}, scale={scale:.4f}, n_exceed={n_exceedances})"
            )

            return ThresholdResult(
                threshold=threshold,
                method="POT",
                parameters={
                    'shape': float(shape),
                    'scale': float(scale),
                    'initial_threshold': float(initial_threshold),
                    'n_exceedances': int(n_exceedances),
                    'exceedance_rate': float(p_exceed)
                },
                confidence_level=confidence_level
            )

        except Exception as e:
            logger.error(f"Error calculating POT threshold: {e}")
            return AdaptiveThresholdCalculator.zscore_threshold(data, confidence_level)

    @staticmethod
    def zscore_threshold(
        data: np.ndarray,
        confidence_level: float = 0.99
    ) -> ThresholdResult:
        """
        Z-score based threshold (Gaussian assumption)

        Args:
            data: Historical data
            confidence_level: Confidence level

        Returns:
            ThresholdResult
        """
        mean = np.mean(data)
        std = np.std(data)

        # Convert confidence level to z-score
        # For upper tail: z = inverse CDF of confidence_level
        z_score = stats.norm.ppf(confidence_level)

        threshold = mean + z_score * std

        return ThresholdResult(
            threshold=threshold,
            method="Z-Score",
            parameters={
                'mean': float(mean),
                'std': float(std),
                'z_score': float(z_score)
            },
            confidence_level=confidence_level
        )

    @staticmethod
    def iqr_threshold(
        data: np.ndarray,
        multiplier: float = 1.5,
        confidence_level: float = 0.99
    ) -> ThresholdResult:
        """
        IQR (Interquartile Range) based threshold

        Args:
            data: Historical data
            multiplier: IQR multiplier (1.5 for outliers, 3.0 for extreme outliers)
            confidence_level: Confidence level (for consistency, not directly used)

        Returns:
            ThresholdResult
        """
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1

        threshold = q3 + multiplier * iqr

        return ThresholdResult(
            threshold=threshold,
            method="IQR",
            parameters={
                'q1': float(q1),
                'q3': float(q3),
                'iqr': float(iqr),
                'multiplier': float(multiplier)
            },
            confidence_level=confidence_level
        )

    @staticmethod
    def mad_threshold(
        data: np.ndarray,
        confidence_level: float = 0.99,
        consistency_constant: float = 1.4826
    ) -> ThresholdResult:
        """
        MAD (Median Absolute Deviation) based threshold
        More robust than standard deviation

        Args:
            data: Historical data
            confidence_level: Confidence level
            consistency_constant: Constant to make MAD consistent with std (1.4826 for normal)

        Returns:
            ThresholdResult
        """
        median = np.median(data)
        mad = np.median(np.abs(data - median))

        # Make MAD consistent with standard deviation
        mad_scaled = consistency_constant * mad

        # Use z-score approach with MAD instead of std
        z_score = stats.norm.ppf(confidence_level)

        threshold = median + z_score * mad_scaled

        return ThresholdResult(
            threshold=threshold,
            method="MAD",
            parameters={
                'median': float(median),
                'mad': float(mad),
                'mad_scaled': float(mad_scaled),
                'z_score': float(z_score)
            },
            confidence_level=confidence_level
        )

    @staticmethod
    def isolation_forest_threshold(
        data: np.ndarray,
        contamination: float = 0.01,
        n_estimators: int = 100,
        random_state: int = 42
    ) -> ThresholdResult:
        """
        Isolation Forest based threshold

        Args:
            data: Historical data
            contamination: Expected proportion of outliers
            n_estimators: Number of trees
            random_state: Random seed

        Returns:
            ThresholdResult
        """
        try:
            from sklearn.ensemble import IsolationForest

            # Reshape for sklearn
            X = data.reshape(-1, 1)

            # Fit Isolation Forest
            iso_forest = IsolationForest(
                contamination=contamination,
                n_estimators=n_estimators,
                random_state=random_state
            )
            iso_forest.fit(X)

            # Get anomaly scores
            scores = -iso_forest.score_samples(X)  # Negative for easier interpretation

            # Calculate threshold based on contamination
            threshold = np.percentile(scores, (1 - contamination) * 100)

            return ThresholdResult(
                threshold=threshold,
                method="IsolationForest",
                parameters={
                    'contamination': float(contamination),
                    'n_estimators': int(n_estimators),
                    'mean_score': float(np.mean(scores)),
                    'std_score': float(np.std(scores))
                },
                confidence_level=1 - contamination
            )

        except ImportError:
            logger.warning("scikit-learn not available for Isolation Forest, using fallback")
            return AdaptiveThresholdCalculator.zscore_threshold(data, 1 - contamination)
        except Exception as e:
            logger.error(f"Error calculating Isolation Forest threshold: {e}")
            return AdaptiveThresholdCalculator.zscore_threshold(data, 1 - contamination)

    @staticmethod
    def local_outlier_factor_threshold(
        data: np.ndarray,
        contamination: float = 0.01,
        n_neighbors: int = 20
    ) -> ThresholdResult:
        """
        Local Outlier Factor based threshold

        Args:
            data: Historical data
            contamination: Expected proportion of outliers
            n_neighbors: Number of neighbors

        Returns:
            ThresholdResult
        """
        try:
            from sklearn.neighbors import LocalOutlierFactor

            # Reshape for sklearn
            X = data.reshape(-1, 1)

            # Fit LOF
            lof = LocalOutlierFactor(
                contamination=contamination,
                n_neighbors=n_neighbors,
                novelty=False
            )

            # Get negative outlier factors (higher = more outlying)
            outlier_factors = -lof.fit_predict(X)
            negative_outlier_factors = lof.negative_outlier_factor_

            # Calculate threshold
            threshold = np.percentile(-negative_outlier_factors, (1 - contamination) * 100)

            return ThresholdResult(
                threshold=threshold,
                method="LOF",
                parameters={
                    'contamination': float(contamination),
                    'n_neighbors': int(n_neighbors),
                    'mean_lof': float(np.mean(-negative_outlier_factors)),
                    'std_lof': float(np.std(-negative_outlier_factors))
                },
                confidence_level=1 - contamination
            )

        except ImportError:
            logger.warning("scikit-learn not available for LOF, using fallback")
            return AdaptiveThresholdCalculator.zscore_threshold(data, 1 - contamination)
        except Exception as e:
            logger.error(f"Error calculating LOF threshold: {e}")
            return AdaptiveThresholdCalculator.zscore_threshold(data, 1 - contamination)

    @staticmethod
    def adaptive_threshold_selection(
        data: np.ndarray,
        confidence_level: float = 0.99,
        methods: Optional[List[str]] = None
    ) -> Dict[str, ThresholdResult]:
        """
        Calculate thresholds using multiple methods and return all

        Args:
            data: Historical data
            confidence_level: Confidence level
            methods: List of method names (all if None)

        Returns:
            Dictionary mapping method name to ThresholdResult
        """
        if methods is None:
            methods = ['GEV', 'POT', 'Z-Score', 'IQR', 'MAD', 'IsolationForest', 'LOF']

        results = {}

        for method in methods:
            try:
                if method == 'GEV':
                    results[method] = AdaptiveThresholdCalculator.gev_threshold(
                        data, confidence_level
                    )
                elif method == 'POT':
                    results[method] = AdaptiveThresholdCalculator.pot_threshold(
                        data, confidence_level=confidence_level
                    )
                elif method == 'Z-Score':
                    results[method] = AdaptiveThresholdCalculator.zscore_threshold(
                        data, confidence_level
                    )
                elif method == 'IQR':
                    results[method] = AdaptiveThresholdCalculator.iqr_threshold(
                        data, confidence_level=confidence_level
                    )
                elif method == 'MAD':
                    results[method] = AdaptiveThresholdCalculator.mad_threshold(
                        data, confidence_level
                    )
                elif method == 'IsolationForest':
                    results[method] = AdaptiveThresholdCalculator.isolation_forest_threshold(
                        data, contamination=1 - confidence_level
                    )
                elif method == 'LOF':
                    results[method] = AdaptiveThresholdCalculator.local_outlier_factor_threshold(
                        data, contamination=1 - confidence_level
                    )
            except Exception as e:
                logger.warning(f"Error with method {method}: {e}")

        return results

    @staticmethod
    def consensus_threshold(
        data: np.ndarray,
        confidence_level: float = 0.99,
        methods: Optional[List[str]] = None,
        aggregation: str = 'median'
    ) -> ThresholdResult:
        """
        Calculate consensus threshold from multiple methods

        Args:
            data: Historical data
            confidence_level: Confidence level
            methods: Methods to use (all if None)
            aggregation: 'mean', 'median', 'min', 'max'

        Returns:
            ThresholdResult with consensus threshold
        """
        # Get thresholds from all methods
        all_results = AdaptiveThresholdCalculator.adaptive_threshold_selection(
            data, confidence_level, methods
        )

        if not all_results:
            # Fallback
            return AdaptiveThresholdCalculator.zscore_threshold(data, confidence_level)

        # Extract threshold values
        thresholds = [r.threshold for r in all_results.values()]

        # Aggregate
        if aggregation == 'mean':
            consensus = np.mean(thresholds)
        elif aggregation == 'median':
            consensus = np.median(thresholds)
        elif aggregation == 'min':
            consensus = np.min(thresholds)
        elif aggregation == 'max':
            consensus = np.max(thresholds)
        else:
            consensus = np.median(thresholds)

        # Collect all parameters
        all_params = {
            method: result.to_dict()
            for method, result in all_results.items()
        }

        return ThresholdResult(
            threshold=consensus,
            method=f"Consensus-{aggregation}",
            parameters={
                'aggregation': aggregation,
                'n_methods': len(all_results),
                'methods_used': list(all_results.keys()),
                'individual_thresholds': thresholds,
                'std_thresholds': float(np.std(thresholds)),
                'all_results': all_params
            },
            confidence_level=confidence_level
        )

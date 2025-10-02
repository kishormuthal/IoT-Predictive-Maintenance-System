"""
Ensemble Methods for Model Combination
Advanced techniques for combining multiple anomaly detection and forecasting models
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    """Result from ensemble model"""

    prediction: Any  # Final ensemble prediction
    individual_predictions: List[Any]  # Predictions from individual models
    weights: List[float]  # Model weights
    confidence: float  # Ensemble confidence
    method: str  # Ensemble method used

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "prediction": self.prediction,
            "individual_predictions": self.individual_predictions,
            "weights": self.weights,
            "confidence": float(self.confidence),
            "method": self.method,
        }


class EnsembleAggregator:
    """
    Ensemble aggregation methods

    Methods:
    - Simple averaging
    - Weighted averaging
    - Voting (for classification)
    - Stacking
    - Median aggregation
    - Trimmed mean
    """

    @staticmethod
    def simple_average(predictions: List[float], **kwargs) -> EnsembleResult:
        """
        Simple average of predictions

        Args:
            predictions: List of model predictions

        Returns:
            EnsembleResult
        """
        predictions = np.array(predictions)
        ensemble_pred = np.mean(predictions)
        weights = [1.0 / len(predictions)] * len(predictions)

        # Confidence based on agreement (inverse of variance)
        if len(predictions) > 1:
            std = np.std(predictions)
            confidence = 1 / (1 + std)
        else:
            confidence = 1.0

        return EnsembleResult(
            prediction=float(ensemble_pred),
            individual_predictions=predictions.tolist(),
            weights=weights,
            confidence=confidence,
            method="SimpleAverage",
        )

    @staticmethod
    def weighted_average(predictions: List[float], weights: List[float], **kwargs) -> EnsembleResult:
        """
        Weighted average of predictions

        Args:
            predictions: List of model predictions
            weights: Weight for each prediction

        Returns:
            EnsembleResult
        """
        predictions = np.array(predictions)
        weights = np.array(weights)

        # Normalize weights
        weights = weights / (np.sum(weights) + 1e-10)

        ensemble_pred = np.sum(predictions * weights)

        # Confidence based on weight concentration
        # Higher when one model dominates (high max weight)
        confidence = float(np.max(weights))

        return EnsembleResult(
            prediction=float(ensemble_pred),
            individual_predictions=predictions.tolist(),
            weights=weights.tolist(),
            confidence=confidence,
            method="WeightedAverage",
        )

    @staticmethod
    def majority_voting(predictions: List[int], **kwargs) -> EnsembleResult:
        """
        Majority voting for classification

        Args:
            predictions: List of class predictions

        Returns:
            EnsembleResult
        """
        predictions = np.array(predictions)

        # Count votes
        unique, counts = np.unique(predictions, return_counts=True)
        majority_class = unique[np.argmax(counts)]
        majority_count = np.max(counts)

        # Confidence = proportion of votes for winner
        confidence = majority_count / len(predictions)

        # Weights (1 if predicted majority, 0 otherwise)
        weights = [1.0 if p == majority_class else 0.0 for p in predictions]

        return EnsembleResult(
            prediction=int(majority_class),
            individual_predictions=predictions.tolist(),
            weights=weights,
            confidence=confidence,
            method="MajorityVoting",
        )

    @staticmethod
    def median_aggregation(predictions: List[float], **kwargs) -> EnsembleResult:
        """
        Median aggregation (robust to outliers)

        Args:
            predictions: List of model predictions

        Returns:
            EnsembleResult
        """
        predictions = np.array(predictions)
        ensemble_pred = np.median(predictions)

        # Confidence based on spread (IQR)
        if len(predictions) > 3:
            q1 = np.percentile(predictions, 25)
            q3 = np.percentile(predictions, 75)
            iqr = q3 - q1
            confidence = 1 / (1 + iqr)
        else:
            confidence = 1.0

        weights = [1.0 / len(predictions)] * len(predictions)

        return EnsembleResult(
            prediction=float(ensemble_pred),
            individual_predictions=predictions.tolist(),
            weights=weights,
            confidence=confidence,
            method="MedianAggregation",
        )

    @staticmethod
    def trimmed_mean(predictions: List[float], trim_percent: float = 0.2, **kwargs) -> EnsembleResult:
        """
        Trimmed mean (remove extreme predictions)

        Args:
            predictions: List of model predictions
            trim_percent: Percentage to trim from each end (0-0.5)

        Returns:
            EnsembleResult
        """
        predictions = np.array(predictions)

        # Sort predictions
        sorted_preds = np.sort(predictions)

        # Calculate trim indices
        n = len(sorted_preds)
        trim_count = int(n * trim_percent)

        if trim_count > 0 and n > 2 * trim_count:
            # Trim from both ends
            trimmed = sorted_preds[trim_count:-trim_count]
        else:
            trimmed = sorted_preds

        ensemble_pred = np.mean(trimmed)

        # Confidence based on how much was trimmed
        confidence = len(trimmed) / len(predictions)

        weights = [1.0 / len(predictions)] * len(predictions)

        return EnsembleResult(
            prediction=float(ensemble_pred),
            individual_predictions=predictions.tolist(),
            weights=weights,
            confidence=confidence,
            method=f"TrimmedMean-{trim_percent}",
        )

    @staticmethod
    def performance_weighted_average(
        predictions: List[float], performance_scores: List[float], **kwargs
    ) -> EnsembleResult:
        """
        Weight predictions by past performance

        Args:
            predictions: List of model predictions
            performance_scores: Performance score for each model (e.g., accuracy, R²)

        Returns:
            EnsembleResult
        """
        predictions = np.array(predictions)
        performance_scores = np.array(performance_scores)

        # Normalize performance scores to weights
        weights = performance_scores / (np.sum(performance_scores) + 1e-10)

        ensemble_pred = np.sum(predictions * weights)

        # Confidence = weighted average of performance
        confidence = float(np.sum(performance_scores * weights))

        return EnsembleResult(
            prediction=float(ensemble_pred),
            individual_predictions=predictions.tolist(),
            weights=weights.tolist(),
            confidence=confidence,
            method="PerformanceWeighted",
        )

    @staticmethod
    def inverse_variance_weighting(predictions: List[float], variances: List[float], **kwargs) -> EnsembleResult:
        """
        Weight by inverse variance (precision weighting)

        Args:
            predictions: List of model predictions
            variances: Prediction variance for each model

        Returns:
            EnsembleResult
        """
        predictions = np.array(predictions)
        variances = np.array(variances)

        # Inverse variance weights
        precisions = 1 / (variances + 1e-10)
        weights = precisions / (np.sum(precisions) + 1e-10)

        ensemble_pred = np.sum(predictions * weights)

        # Ensemble variance (lower = higher confidence)
        ensemble_variance = 1 / (np.sum(precisions) + 1e-10)
        confidence = 1 / (1 + ensemble_variance)

        return EnsembleResult(
            prediction=float(ensemble_pred),
            individual_predictions=predictions.tolist(),
            weights=weights.tolist(),
            confidence=confidence,
            method="InverseVariance",
        )


class DynamicEnsemble:
    """
    Dynamic ensemble that adapts weights based on recent performance
    """

    def __init__(
        self,
        n_models: int,
        learning_rate: float = 0.1,
        initial_weights: Optional[List[float]] = None,
    ):
        """
        Initialize dynamic ensemble

        Args:
            n_models: Number of models in ensemble
            learning_rate: Learning rate for weight updates
            initial_weights: Initial weights (uniform if None)
        """
        self.n_models = n_models
        self.learning_rate = learning_rate

        if initial_weights is None:
            self.weights = np.ones(n_models) / n_models
        else:
            self.weights = np.array(initial_weights)
            self.weights = self.weights / np.sum(self.weights)

        self.performance_history = []

    def predict(self, predictions: List[float]) -> float:
        """
        Make ensemble prediction

        Args:
            predictions: Predictions from individual models

        Returns:
            Ensemble prediction
        """
        predictions = np.array(predictions)
        return np.sum(predictions * self.weights)

    def update_weights(self, predictions: List[float], true_value: float):
        """
        Update weights based on prediction errors

        Args:
            predictions: Predictions from individual models
            true_value: True value
        """
        predictions = np.array(predictions)

        # Calculate errors
        errors = np.abs(predictions - true_value)

        # Update weights (decrease weight for larger errors)
        # Using exponential weighting
        weight_updates = np.exp(-self.learning_rate * errors)
        weight_updates = weight_updates / (np.sum(weight_updates) + 1e-10)

        # Weighted average of old and new weights
        self.weights = 0.9 * self.weights + 0.1 * weight_updates

        # Normalize
        self.weights = self.weights / (np.sum(self.weights) + 1e-10)

        # Store performance
        self.performance_history.append(
            {
                "predictions": predictions.tolist(),
                "true_value": float(true_value),
                "errors": errors.tolist(),
                "weights": self.weights.tolist(),
            }
        )

    def get_weights(self) -> np.ndarray:
        """Get current weights"""
        return self.weights.copy()


class StackingEnsemble:
    """
    Stacking ensemble - meta-model learns to combine base models
    """

    def __init__(self, meta_model: Optional[Any] = None):
        """
        Initialize stacking ensemble

        Args:
            meta_model: Meta-model for combining predictions (linear regression if None)
        """
        self.meta_model = meta_model
        self.is_fitted = False

    def fit(self, base_predictions: np.ndarray, true_values: np.ndarray):
        """
        Fit meta-model

        Args:
            base_predictions: Predictions from base models (shape: n_samples x n_models)
            true_values: True values (shape: n_samples)
        """
        try:
            from sklearn.linear_model import Ridge

            if self.meta_model is None:
                self.meta_model = Ridge(alpha=1.0)

            self.meta_model.fit(base_predictions, true_values)
            self.is_fitted = True

            logger.info("Stacking meta-model fitted")

        except ImportError:
            logger.error("scikit-learn required for stacking")
            raise
        except Exception as e:
            logger.error(f"Error fitting stacking model: {e}")
            raise

    def predict(self, base_predictions: List[float]) -> float:
        """
        Make stacked prediction

        Args:
            base_predictions: Predictions from base models

        Returns:
            Stacked prediction
        """
        if not self.is_fitted:
            raise ValueError("Meta-model not fitted. Call fit() first.")

        base_predictions = np.array(base_predictions).reshape(1, -1)
        return self.meta_model.predict(base_predictions)[0]


class AdaptiveEnsembleSelector:
    """
    Adaptively select ensemble method based on data characteristics
    """

    @staticmethod
    def select_method(
        predictions: List[float],
        variances: Optional[List[float]] = None,
        performance_scores: Optional[List[float]] = None,
        task_type: str = "regression",
    ) -> EnsembleResult:
        """
        Automatically select best ensemble method

        Args:
            predictions: Model predictions
            variances: Prediction variances (optional)
            performance_scores: Model performance scores (optional)
            task_type: 'regression' or 'classification'

        Returns:
            EnsembleResult
        """
        # For classification, use voting
        if task_type == "classification":
            return EnsembleAggregator.majority_voting(predictions)

        # For regression, select based on available information
        predictions_array = np.array(predictions)

        # Check agreement among predictions
        std = np.std(predictions_array)
        mean = np.mean(predictions_array)
        cv = std / (abs(mean) + 1e-10)  # Coefficient of variation

        # If high disagreement, use robust method
        if cv > 0.5:
            logger.info("High prediction disagreement, using trimmed mean")
            return EnsembleAggregator.trimmed_mean(predictions, trim_percent=0.2)

        # If have performance scores, use them
        if performance_scores is not None:
            logger.info("Using performance-weighted average")
            return EnsembleAggregator.performance_weighted_average(predictions, performance_scores)

        # If have variances, use inverse variance weighting
        if variances is not None:
            logger.info("Using inverse variance weighting")
            return EnsembleAggregator.inverse_variance_weighting(predictions, variances)

        # Default to simple average
        logger.info("Using simple average")
        return EnsembleAggregator.simple_average(predictions)

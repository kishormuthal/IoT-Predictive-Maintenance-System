"""
Comprehensive Evaluation Metrics
Complete suite of metrics for classification, regression, anomaly detection, and forecasting
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


@dataclass
class ClassificationMetrics:
    """Complete classification metrics"""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    confusion_matrix: np.ndarray
    classification_report: str

    # ROC/AUC (if probabilities provided)
    roc_auc: Optional[float] = None
    fpr: Optional[np.ndarray] = None
    tpr: Optional[np.ndarray] = None
    roc_thresholds: Optional[np.ndarray] = None

    # Precision-Recall
    pr_auc: Optional[float] = None
    precision_curve: Optional[np.ndarray] = None
    recall_curve: Optional[np.ndarray] = None
    pr_thresholds: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "accuracy": float(self.accuracy),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1_score": float(self.f1_score),
            "true_positives": int(self.true_positives),
            "true_negatives": int(self.true_negatives),
            "false_positives": int(self.false_positives),
            "false_negatives": int(self.false_negatives),
            "confusion_matrix": self.confusion_matrix.tolist(),
            "classification_report": self.classification_report,
        }

        if self.roc_auc is not None:
            result["roc_auc"] = float(self.roc_auc)

        if self.pr_auc is not None:
            result["pr_auc"] = float(self.pr_auc)

        return result


@dataclass
class RegressionMetrics:
    """Complete regression metrics"""

    mae: float  # Mean Absolute Error
    mse: float  # Mean Squared Error
    rmse: float  # Root Mean Squared Error
    mape: float  # Mean Absolute Percentage Error
    r2_score: float  # R-squared

    # Additional metrics
    max_error: float
    median_absolute_error: float
    explained_variance: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "mae": float(self.mae),
            "mse": float(self.mse),
            "rmse": float(self.rmse),
            "mape": float(self.mape),
            "r2_score": float(self.r2_score),
            "max_error": float(self.max_error),
            "median_absolute_error": float(self.median_absolute_error),
            "explained_variance": float(self.explained_variance),
        }


@dataclass
class AnomalyDetectionMetrics:
    """Metrics specific to anomaly detection"""

    # Classification metrics
    classification_metrics: ClassificationMetrics

    # Anomaly-specific
    num_anomalies_detected: int
    anomaly_rate: float
    mean_anomaly_score: float
    max_anomaly_score: float

    # For labeled data
    true_anomalies: Optional[int] = None
    detected_true_anomalies: Optional[int] = None
    detection_rate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "classification_metrics": self.classification_metrics.to_dict(),
            "num_anomalies_detected": int(self.num_anomalies_detected),
            "anomaly_rate": float(self.anomaly_rate),
            "mean_anomaly_score": float(self.mean_anomaly_score),
            "max_anomaly_score": float(self.max_anomaly_score),
        }

        if self.detection_rate is not None:
            result["detection_rate"] = float(self.detection_rate)

        return result


@dataclass
class ForecastingMetrics:
    """Metrics specific to time series forecasting"""

    # Regression metrics
    regression_metrics: RegressionMetrics

    # Forecasting-specific
    forecast_horizon: int
    mae_by_step: List[float]
    rmse_by_step: List[float]
    mape_by_step: List[float]

    # Directional accuracy
    directional_accuracy: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "regression_metrics": self.regression_metrics.to_dict(),
            "forecast_horizon": int(self.forecast_horizon),
            "mae_by_step": [float(x) for x in self.mae_by_step],
            "rmse_by_step": [float(x) for x in self.rmse_by_step],
            "mape_by_step": [float(x) for x in self.mape_by_step],
        }

        if self.directional_accuracy is not None:
            result["directional_accuracy"] = float(self.directional_accuracy)

        return result


class EvaluationMetricsCalculator:
    """
    Comprehensive evaluation metrics calculator

    Supports:
    - Binary/multi-class classification
    - Regression
    - Anomaly detection
    - Time series forecasting
    """

    @staticmethod
    def compute_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        average: str = "binary",
    ) -> ClassificationMetrics:
        """
        Compute comprehensive classification metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional, for ROC/AUC)
            average: Averaging method for multi-class ('binary', 'micro', 'macro', 'weighted')

        Returns:
            ClassificationMetrics object
        """
        try:
            # Basic metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(
                y_true, y_pred, average=average, zero_division=0
            )
            recall = recall_score(y_true, y_pred, average=average, zero_division=0)
            f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            # For binary classification, extract TP, TN, FP, FN
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                # For multi-class, sum diagonals and off-diagonals
                tp = np.diag(cm).sum()
                fp = cm.sum(axis=0).sum() - tp
                fn = cm.sum(axis=1).sum() - tp
                tn = cm.sum() - tp - fp - fn

            # Classification report
            report = classification_report(y_true, y_pred, zero_division=0)

            # ROC/AUC (only for binary with probabilities)
            roc_auc = None
            fpr, tpr, roc_thresholds = None, None, None
            pr_auc = None
            precision_curve, recall_curve, pr_thresholds = None, None, None

            if y_prob is not None and len(np.unique(y_true)) == 2:
                try:
                    # ROC curve
                    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
                    roc_auc = auc(fpr, tpr)

                    # Precision-Recall curve
                    precision_curve, recall_curve, pr_thresholds = (
                        precision_recall_curve(y_true, y_prob)
                    )
                    pr_auc = auc(recall_curve, precision_curve)

                except Exception as e:
                    logger.warning(f"Could not compute ROC/PR curves: {e}")

            return ClassificationMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                true_positives=int(tp),
                true_negatives=int(tn),
                false_positives=int(fp),
                false_negatives=int(fn),
                confusion_matrix=cm,
                classification_report=report,
                roc_auc=roc_auc,
                fpr=fpr,
                tpr=tpr,
                roc_thresholds=roc_thresholds,
                pr_auc=pr_auc,
                precision_curve=precision_curve,
                recall_curve=recall_curve,
                pr_thresholds=pr_thresholds,
            )

        except Exception as e:
            logger.error(f"Error computing classification metrics: {e}")
            raise

    @staticmethod
    def compute_regression_metrics(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> RegressionMetrics:
        """
        Compute comprehensive regression metrics

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            RegressionMetrics object
        """
        try:
            # Basic metrics
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)

            # MAPE (Mean Absolute Percentage Error)
            epsilon = 1e-10
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

            # Additional metrics
            errors = np.abs(y_true - y_pred)
            max_error = float(np.max(errors))
            median_absolute_error = float(np.median(errors))

            # Explained variance
            explained_variance = 1 - (
                np.var(y_true - y_pred) / (np.var(y_true) + epsilon)
            )

            return RegressionMetrics(
                mae=mae,
                mse=mse,
                rmse=rmse,
                mape=mape,
                r2_score=r2,
                max_error=max_error,
                median_absolute_error=median_absolute_error,
                explained_variance=explained_variance,
            )

        except Exception as e:
            logger.error(f"Error computing regression metrics: {e}")
            raise

    @staticmethod
    def compute_anomaly_detection_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        anomaly_scores: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> AnomalyDetectionMetrics:
        """
        Compute anomaly detection metrics

        Args:
            y_true: True labels (0=normal, 1=anomaly)
            y_pred: Predicted labels
            anomaly_scores: Anomaly scores
            y_prob: Predicted probabilities (optional)

        Returns:
            AnomalyDetectionMetrics object
        """
        try:
            # Classification metrics
            classification_metrics = (
                EvaluationMetricsCalculator.compute_classification_metrics(
                    y_true=y_true, y_pred=y_pred, y_prob=y_prob
                )
            )

            # Anomaly-specific metrics
            num_anomalies_detected = int(np.sum(y_pred == 1))
            total_samples = len(y_pred)
            anomaly_rate = (
                num_anomalies_detected / total_samples if total_samples > 0 else 0
            )

            mean_anomaly_score = float(np.mean(anomaly_scores))
            max_anomaly_score = float(np.max(anomaly_scores))

            # For labeled data
            true_anomalies = int(np.sum(y_true == 1))
            detected_true_anomalies = int(np.sum((y_true == 1) & (y_pred == 1)))
            detection_rate = (
                detected_true_anomalies / true_anomalies if true_anomalies > 0 else 0
            )

            return AnomalyDetectionMetrics(
                classification_metrics=classification_metrics,
                num_anomalies_detected=num_anomalies_detected,
                anomaly_rate=anomaly_rate,
                mean_anomaly_score=mean_anomaly_score,
                max_anomaly_score=max_anomaly_score,
                true_anomalies=true_anomalies,
                detected_true_anomalies=detected_true_anomalies,
                detection_rate=detection_rate,
            )

        except Exception as e:
            logger.error(f"Error computing anomaly detection metrics: {e}")
            raise

    @staticmethod
    def compute_forecasting_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, forecast_horizon: int
    ) -> ForecastingMetrics:
        """
        Compute time series forecasting metrics

        Args:
            y_true: True values (shape: [samples, horizon] or [samples])
            y_pred: Predicted values (shape: [samples, horizon] or [samples])
            forecast_horizon: Forecast horizon

        Returns:
            ForecastingMetrics object
        """
        try:
            # Flatten if needed
            y_true_flat = y_true.flatten() if y_true.ndim > 1 else y_true
            y_pred_flat = y_pred.flatten() if y_pred.ndim > 1 else y_pred

            # Overall regression metrics
            regression_metrics = EvaluationMetricsCalculator.compute_regression_metrics(
                y_true=y_true_flat, y_pred=y_pred_flat
            )

            # Per-step metrics (if multi-step forecast)
            mae_by_step = []
            rmse_by_step = []
            mape_by_step = []

            if y_true.ndim == 2:
                # Multi-step forecast
                for step in range(forecast_horizon):
                    if step < y_true.shape[1]:
                        step_mae = mean_absolute_error(y_true[:, step], y_pred[:, step])
                        step_mse = mean_squared_error(y_true[:, step], y_pred[:, step])
                        step_rmse = np.sqrt(step_mse)

                        epsilon = 1e-10
                        step_mape = (
                            np.mean(
                                np.abs(
                                    (y_true[:, step] - y_pred[:, step])
                                    / (y_true[:, step] + epsilon)
                                )
                            )
                            * 100
                        )

                        mae_by_step.append(step_mae)
                        rmse_by_step.append(step_rmse)
                        mape_by_step.append(step_mape)
            else:
                # Single-step, use overall metrics
                mae_by_step = [regression_metrics.mae]
                rmse_by_step = [regression_metrics.rmse]
                mape_by_step = [regression_metrics.mape]

            # Directional accuracy (for time series)
            directional_accuracy = None
            if len(y_true_flat) > 1:
                true_direction = np.sign(np.diff(y_true_flat))
                pred_direction = np.sign(np.diff(y_pred_flat))
                directional_accuracy = np.mean(true_direction == pred_direction)

            return ForecastingMetrics(
                regression_metrics=regression_metrics,
                forecast_horizon=forecast_horizon,
                mae_by_step=mae_by_step,
                rmse_by_step=rmse_by_step,
                mape_by_step=mape_by_step,
                directional_accuracy=directional_accuracy,
            )

        except Exception as e:
            logger.error(f"Error computing forecasting metrics: {e}")
            raise

    @staticmethod
    def plot_confusion_matrix(
        cm: np.ndarray, class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate confusion matrix plot data

        Args:
            cm: Confusion matrix
            class_names: Class names (optional)

        Returns:
            Dictionary with plot data
        """
        if class_names is None:
            class_names = [f"Class {i}" for i in range(cm.shape[0])]

        # Normalize confusion matrix
        cm_normalized = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

        return {
            "matrix": cm.tolist(),
            "matrix_normalized": cm_normalized.tolist(),
            "class_names": class_names,
            "total_samples": int(cm.sum()),
        }

    @staticmethod
    def plot_roc_curve(
        fpr: np.ndarray, tpr: np.ndarray, roc_auc: float
    ) -> Dict[str, Any]:
        """
        Generate ROC curve plot data

        Args:
            fpr: False positive rates
            tpr: True positive rates
            roc_auc: Area under ROC curve

        Returns:
            Dictionary with plot data
        """
        return {
            "fpr": fpr.tolist() if fpr is not None else [],
            "tpr": tpr.tolist() if tpr is not None else [],
            "auc": float(roc_auc) if roc_auc is not None else 0.0,
            "diagonal": [[0, 1], [0, 1]],  # Random classifier line
        }

    @staticmethod
    def plot_precision_recall_curve(
        precision: np.ndarray, recall: np.ndarray, pr_auc: float
    ) -> Dict[str, Any]:
        """
        Generate Precision-Recall curve plot data

        Args:
            precision: Precision values
            recall: Recall values
            pr_auc: Area under PR curve

        Returns:
            Dictionary with plot data
        """
        return {
            "precision": precision.tolist() if precision is not None else [],
            "recall": recall.tolist() if recall is not None else [],
            "auc": float(pr_auc) if pr_auc is not None else 0.0,
        }

"""
Advanced Algorithms Usage Examples
Demonstrates the advanced statistical and ML algorithms for anomaly detection and forecasting
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import sys
sys.path.append('..')

from src.core.algorithms.adaptive_thresholding import AdaptiveThresholdCalculator
from src.core.algorithms.probabilistic_scoring import ProbabilisticAnomalyScorer, AnomalyConfidenceEstimator
from src.core.algorithms.advanced_imputation import AdvancedImputer
from src.core.algorithms.ensemble_methods import (
    EnsembleAggregator,
    DynamicEnsemble,
    StackingEnsemble,
    AdaptiveEnsembleSelector
)


def example_adaptive_thresholding():
    """Example: Advanced Adaptive Thresholding"""
    print("=" * 80)
    print("EXAMPLE 1: ADAPTIVE THRESHOLDING")
    print("=" * 80)

    # Generate sample sensor data with anomalies
    np.random.seed(42)
    normal_data = np.random.normal(100, 10, 1000)

    # Test value (potential anomaly)
    test_value = 140

    print(f"\nTest Value: {test_value}")
    print(f"Normal Data: mean={np.mean(normal_data):.2f}, std={np.std(normal_data):.2f}")

    # Method 1: GEV (Generalized Extreme Value) - Best for extreme values
    print("\n--- GEV Threshold (Extreme Value Theory) ---")
    gev_result = AdaptiveThresholdCalculator.gev_threshold(
        normal_data,
        confidence_level=0.99,
        block_size=100
    )
    print(f"Threshold: {gev_result.threshold:.2f}")
    print(f"Is Anomaly: {test_value > gev_result.threshold}")
    print(f"Parameters: shape={gev_result.parameters['shape']:.4f}, "
          f"location={gev_result.parameters['location']:.2f}, "
          f"scale={gev_result.parameters['scale']:.2f}")

    # Method 2: POT (Peaks Over Threshold) - For rare events
    print("\n--- POT Threshold (Peaks Over Threshold) ---")
    pot_result = AdaptiveThresholdCalculator.pot_threshold(
        normal_data,
        confidence_level=0.99
    )
    print(f"Threshold: {pot_result.threshold:.2f}")
    print(f"Is Anomaly: {test_value > pot_result.threshold}")
    print(f"Exceedances: {pot_result.parameters['n_exceedances']}")

    # Method 3: MAD (Median Absolute Deviation) - Robust to outliers
    print("\n--- MAD Threshold (Robust) ---")
    mad_result = AdaptiveThresholdCalculator.mad_threshold(
        normal_data,
        confidence_level=0.99
    )
    print(f"Threshold: {mad_result.threshold:.2f}")
    print(f"Is Anomaly: {test_value > mad_result.threshold}")
    print(f"MAD: {mad_result.parameters['mad']:.2f}")

    # Method 4: Isolation Forest - ML-based
    print("\n--- Isolation Forest Threshold ---")
    iso_result = AdaptiveThresholdCalculator.isolation_forest_threshold(
        normal_data,
        contamination=0.01
    )
    print(f"Threshold: {iso_result.threshold:.4f}")
    print(f"Contamination: {iso_result.parameters['contamination']}")

    # Method 5: Consensus Threshold - Combine all methods
    print("\n--- Consensus Threshold (Median of All Methods) ---")
    consensus_result = AdaptiveThresholdCalculator.consensus_threshold(
        normal_data,
        confidence_level=0.99,
        aggregation='median'
    )
    print(f"Consensus Threshold: {consensus_result.threshold:.2f}")
    print(f"Is Anomaly: {test_value > consensus_result.threshold}")
    print(f"Methods Used: {consensus_result.parameters['methods_used']}")
    print(f"Individual Thresholds: {[f'{t:.2f}' for t in consensus_result.parameters['individual_thresholds']]}")


def example_probabilistic_scoring():
    """Example: Probabilistic Anomaly Scoring"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: PROBABILISTIC ANOMALY SCORING")
    print("=" * 80)

    # Generate reference data
    np.random.seed(42)
    reference_data = np.random.normal(50, 5, 500)

    # Test values
    test_values = [52, 60, 70, 80]  # Normal to extreme

    for test_value in test_values:
        print(f"\n--- Scoring Value: {test_value} ---")

        # Method 1: Gaussian Likelihood
        gaussian_score = ProbabilisticAnomalyScorer.gaussian_likelihood_score(
            test_value,
            reference_data,
            alpha=0.05
        )
        print(f"Gaussian Score: {gaussian_score.score:.4f} "
              f"(prob={gaussian_score.probability:.4f})")

        # Method 2: Bayesian Probability
        bayesian_score = ProbabilisticAnomalyScorer.bayesian_anomaly_probability(
            test_value,
            reference_data,
            prior_anomaly_rate=0.01
        )
        print(f"Bayesian Score: {bayesian_score.score:.4f} "
              f"(P(anomaly|data)={bayesian_score.probability:.4f})")

        # Method 3: KDE Score
        kde_score = ProbabilisticAnomalyScorer.kernel_density_score(
            test_value,
            reference_data
        )
        print(f"KDE Score: {kde_score.score:.4f} "
              f"(prob={kde_score.probability:.4f})")

        # Method 4: Ensemble Score
        ensemble_score = ProbabilisticAnomalyScorer.ensemble_probabilistic_score(
            test_value,
            reference_data,
            methods=['Gaussian', 'Bayesian', 'KDE'],
            weights=[0.4, 0.3, 0.3]
        )
        print(f"Ensemble Score: {ensemble_score.score:.4f} "
              f"(prob={ensemble_score.probability:.4f})")

    # Bootstrap Confidence Estimation
    print("\n--- Bootstrap Confidence Intervals ---")
    test_value = 65
    confidence = AnomalyConfidenceEstimator.bootstrap_confidence(
        test_value,
        reference_data,
        n_bootstrap=100,
        confidence_level=0.95
    )
    print(f"Test Value: {test_value}")
    print(f"Mean Score: {confidence['mean_score']:.4f} ± {confidence['std_score']:.4f}")
    print(f"95% CI: [{confidence['confidence_interval'][0]:.4f}, "
          f"{confidence['confidence_interval'][1]:.4f}]")


def example_advanced_imputation():
    """Example: Advanced Imputation Methods"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: ADVANCED IMPUTATION")
    print("=" * 80)

    # Generate time series with missing values
    np.random.seed(42)
    true_data = 50 + 10 * np.sin(np.linspace(0, 4 * np.pi, 200)) + np.random.normal(0, 2, 200)

    # Create missing values (20% missing)
    data_with_missing = true_data.copy()
    missing_indices = np.random.choice(200, size=40, replace=False)
    data_with_missing[missing_indices] = np.nan

    print(f"Original Data Length: {len(true_data)}")
    print(f"Missing Values: {np.sum(np.isnan(data_with_missing))} ({np.sum(np.isnan(data_with_missing))/len(data_with_missing)*100:.1f}%)")

    # Method 1: Linear Interpolation
    print("\n--- Linear Interpolation ---")
    linear_imputed = AdvancedImputer.linear_interpolation(data_with_missing)
    linear_error = np.mean(np.abs(linear_imputed[missing_indices] - true_data[missing_indices]))
    print(f"MAE on imputed values: {linear_error:.4f}")

    # Method 2: Spline Interpolation
    print("\n--- Spline Interpolation (Cubic) ---")
    spline_imputed = AdvancedImputer.spline_interpolation(
        data_with_missing,
        order=3,
        smoothing=0.1
    )
    spline_error = np.mean(np.abs(spline_imputed[missing_indices] - true_data[missing_indices]))
    print(f"MAE on imputed values: {spline_error:.4f}")

    # Method 3: KNN Imputation
    print("\n--- KNN Imputation ---")
    knn_imputed = AdvancedImputer.knn_imputation(
        data_with_missing,
        n_neighbors=5,
        weights='distance'
    )
    knn_error = np.mean(np.abs(knn_imputed[missing_indices] - true_data[missing_indices]))
    print(f"MAE on imputed values: {knn_error:.4f}")

    # Method 4: Seasonal Decomposition
    print("\n--- Seasonal Decomposition Imputation ---")
    seasonal_imputed = AdvancedImputer.seasonal_decomposition_imputation(
        data_with_missing,
        period=50  # Period of sine wave
    )
    seasonal_error = np.mean(np.abs(seasonal_imputed[missing_indices] - true_data[missing_indices]))
    print(f"MAE on imputed values: {seasonal_error:.4f}")

    # Method 5: Adaptive Imputation (Auto-select)
    print("\n--- Adaptive Imputation (Auto) ---")
    adaptive_imputed = AdvancedImputer.adaptive_imputation(
        data_with_missing,
        method='auto'
    )
    adaptive_error = np.mean(np.abs(adaptive_imputed[missing_indices] - true_data[missing_indices]))
    print(f"MAE on imputed values: {adaptive_error:.4f}")
    print(f"Auto-selected method based on 20% missing rate")

    # Method 6: Imputation with Confidence
    print("\n--- Imputation with Uncertainty Estimates ---")
    imputed_mean, imputed_std = AdvancedImputer.impute_with_confidence(
        data_with_missing,
        method='auto',
        n_bootstrap=50
    )
    print(f"Mean imputed value at first missing index: {imputed_mean[missing_indices[0]]:.2f}")
    print(f"Uncertainty (std): {imputed_std[missing_indices[0]]:.2f}")


def example_ensemble_methods():
    """Example: Ensemble Methods"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: ENSEMBLE METHODS")
    print("=" * 80)

    # Simulate predictions from 5 different models
    np.random.seed(42)

    # Regression example
    true_value = 75.0
    model_predictions = [73.5, 76.2, 74.8, 77.1, 72.9]
    model_variances = [2.0, 1.5, 1.0, 3.0, 2.5]
    model_performance = [0.85, 0.90, 0.92, 0.80, 0.87]  # R² scores

    print(f"True Value: {true_value}")
    print(f"Model Predictions: {model_predictions}")

    # Method 1: Simple Average
    print("\n--- Simple Average ---")
    simple_result = EnsembleAggregator.simple_average(model_predictions)
    print(f"Ensemble Prediction: {simple_result.prediction:.2f}")
    print(f"Confidence: {simple_result.confidence:.4f}")
    print(f"Error: {abs(simple_result.prediction - true_value):.2f}")

    # Method 2: Performance-Weighted Average
    print("\n--- Performance-Weighted Average ---")
    perf_result = EnsembleAggregator.performance_weighted_average(
        model_predictions,
        model_performance
    )
    print(f"Ensemble Prediction: {perf_result.prediction:.2f}")
    print(f"Weights: {[f'{w:.3f}' for w in perf_result.weights]}")
    print(f"Error: {abs(perf_result.prediction - true_value):.2f}")

    # Method 3: Inverse Variance Weighting
    print("\n--- Inverse Variance Weighting (Precision) ---")
    var_result = EnsembleAggregator.inverse_variance_weighting(
        model_predictions,
        model_variances
    )
    print(f"Ensemble Prediction: {var_result.prediction:.2f}")
    print(f"Weights: {[f'{w:.3f}' for w in var_result.weights]}")
    print(f"Error: {abs(var_result.prediction - true_value):.2f}")

    # Method 4: Median Aggregation (Robust)
    print("\n--- Median Aggregation (Robust) ---")
    median_result = EnsembleAggregator.median_aggregation(model_predictions)
    print(f"Ensemble Prediction: {median_result.prediction:.2f}")
    print(f"Error: {abs(median_result.prediction - true_value):.2f}")

    # Method 5: Trimmed Mean (Remove extremes)
    print("\n--- Trimmed Mean (20% trim) ---")
    trimmed_result = EnsembleAggregator.trimmed_mean(
        model_predictions,
        trim_percent=0.2
    )
    print(f"Ensemble Prediction: {trimmed_result.prediction:.2f}")
    print(f"Error: {abs(trimmed_result.prediction - true_value):.2f}")

    # Method 6: Adaptive Selection
    print("\n--- Adaptive Ensemble Selection ---")
    adaptive_result = AdaptiveEnsembleSelector.select_method(
        model_predictions,
        variances=model_variances,
        performance_scores=model_performance,
        task_type='regression'
    )
    print(f"Selected Method: {adaptive_result.method}")
    print(f"Ensemble Prediction: {adaptive_result.prediction:.2f}")
    print(f"Error: {abs(adaptive_result.prediction - true_value):.2f}")

    # Classification example
    print("\n--- Majority Voting (Classification) ---")
    class_predictions = [1, 1, 0, 1, 1]  # 5 binary classifiers
    voting_result = EnsembleAggregator.majority_voting(class_predictions)
    print(f"Class Predictions: {class_predictions}")
    print(f"Ensemble Prediction: {voting_result.prediction}")
    print(f"Confidence: {voting_result.confidence:.2f} (80% voted for class 1)")


def example_dynamic_ensemble():
    """Example: Dynamic Ensemble with Adaptive Weights"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: DYNAMIC ENSEMBLE (ADAPTIVE WEIGHTS)")
    print("=" * 80)

    # Create dynamic ensemble with 3 models
    ensemble = DynamicEnsemble(n_models=3, learning_rate=0.1)

    print(f"Initial Weights: {[f'{w:.3f}' for w in ensemble.get_weights()]}")

    # Simulate online learning over 10 time steps
    np.random.seed(42)
    for i in range(10):
        # Simulate predictions (Model 2 is best, Model 3 is worst)
        true_value = 100 + i * 2
        predictions = [
            true_value + np.random.normal(0, 3),    # Model 1: medium error
            true_value + np.random.normal(0, 1),    # Model 2: low error (best)
            true_value + np.random.normal(0, 5)     # Model 3: high error (worst)
        ]

        # Make ensemble prediction
        ensemble_pred = ensemble.predict(predictions)

        # Update weights based on errors
        ensemble.update_weights(predictions, true_value)

        if i % 3 == 0 or i == 9:
            print(f"\nStep {i+1}:")
            print(f"  True Value: {true_value:.2f}")
            print(f"  Predictions: {[f'{p:.2f}' for p in predictions]}")
            print(f"  Ensemble: {ensemble_pred:.2f}")
            print(f"  Updated Weights: {[f'{w:.3f}' for w in ensemble.get_weights()]}")

    print("\nObservation: Model 2 (best) receives highest weight over time!")


def example_stacking_ensemble():
    """Example: Stacking Ensemble with Meta-Model"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: STACKING ENSEMBLE (META-LEARNING)")
    print("=" * 80)

    # Generate training data
    np.random.seed(42)
    n_samples = 100

    # Base model predictions (3 models)
    base_predictions_train = np.column_stack([
        np.random.normal(100, 5, n_samples),    # Model 1
        np.random.normal(100, 3, n_samples),    # Model 2
        np.random.normal(100, 4, n_samples)     # Model 3
    ])

    # True values
    true_values_train = 100 + np.random.normal(0, 2, n_samples)

    # Create and fit stacking ensemble
    stacking = StackingEnsemble()
    stacking.fit(base_predictions_train, true_values_train)

    print("Stacking Meta-Model Trained!")
    print(f"Training samples: {n_samples}")

    # Test on new data
    test_predictions = [98.5, 101.2, 99.8]
    stacked_prediction = stacking.predict(test_predictions)

    print(f"\nTest Predictions from Base Models: {test_predictions}")
    print(f"Stacked Ensemble Prediction: {stacked_prediction:.2f}")
    print("\nMeta-model learns optimal combination weights!")


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("ADVANCED ALGORITHMS - COMPREHENSIVE USAGE EXAMPLES")
    print("=" * 80)

    example_adaptive_thresholding()
    example_probabilistic_scoring()
    example_advanced_imputation()
    example_ensemble_methods()
    example_dynamic_ensemble()
    example_stacking_ensemble()

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()

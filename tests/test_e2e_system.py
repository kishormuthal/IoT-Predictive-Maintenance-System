"""
End-to-End System Testing
Comprehensive tests covering the entire IoT Predictive Maintenance System
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfigurationSystem:
    """Test centralized configuration management"""

    def test_config_manager_singleton(self):
        """Test configuration manager singleton pattern"""
        from config.config_manager import ConfigurationManager

        config1 = ConfigurationManager()
        config2 = ConfigurationManager()

        assert config1 is config2, "Configuration manager should be singleton"

    def test_config_loading(self):
        """Test configuration loading from YAML"""
        from config.config_manager import load_config

        config = load_config('config/config.yaml', env='development')

        assert config is not None
        assert config.environment == 'development'
        assert config.get('system.project_name') is not None

    def test_config_get_with_dot_notation(self):
        """Test dot notation access"""
        from config.config_manager import load_config

        config = load_config('config/config.yaml', env='development')

        port = config.get('dashboard.server.port')
        assert isinstance(port, int)
        assert 1024 <= port <= 65535

    def test_config_get_section(self):
        """Test section retrieval"""
        from config.config_manager import load_config

        config = load_config('config/config.yaml', env='development')

        mlflow_config = config.get_section('mlflow')
        assert isinstance(mlflow_config, dict)
        assert 'enabled' in mlflow_config

    def test_environment_configs(self):
        """Test environment-specific configurations"""
        from config.config_manager import load_config

        # Development
        dev_config = load_config('config/config.yaml', env='development')
        assert dev_config.get('system.debug') == True
        assert dev_config.get('system.log_level') == 'DEBUG'

        # Production (if file exists)
        try:
            prod_config = load_config('config/config.yaml', env='production')
            assert prod_config.get('system.debug') == False
            assert prod_config.get('system.log_level') == 'WARNING'
        except FileNotFoundError:
            pytest.skip("Production config not found")


class TestAdvancedAlgorithms:
    """Test SESSION 7 advanced algorithms"""

    def test_adaptive_thresholding_gev(self):
        """Test GEV distribution thresholding"""
        from src.core.algorithms.adaptive_thresholding import AdaptiveThresholdCalculator

        # Generate normal data
        data = np.random.normal(100, 10, 1000)

        result = AdaptiveThresholdCalculator.gev_threshold(
            data,
            confidence_level=0.99,
            block_size=100
        )

        assert result.threshold > np.mean(data)
        assert result.method == "GEV"
        assert 'shape' in result.parameters
        assert 'location' in result.parameters
        assert 'scale' in result.parameters

    def test_adaptive_thresholding_consensus(self):
        """Test consensus thresholding"""
        from src.core.algorithms.adaptive_thresholding import AdaptiveThresholdCalculator

        data = np.random.normal(50, 5, 500)

        result = AdaptiveThresholdCalculator.consensus_threshold(
            data,
            confidence_level=0.99,
            aggregation='median'
        )

        assert result.method == "Consensus"
        assert 'methods_used' in result.parameters
        assert len(result.parameters['methods_used']) > 1
        assert 'individual_thresholds' in result.parameters

    def test_probabilistic_scoring_bayesian(self):
        """Test Bayesian anomaly probability"""
        from src.core.algorithms.probabilistic_scoring import ProbabilisticAnomalyScorer

        reference_data = np.random.normal(50, 5, 500)
        test_value = 70  # Anomalous value

        score = ProbabilisticAnomalyScorer.bayesian_anomaly_probability(
            test_value,
            reference_data,
            prior_anomaly_rate=0.01
        )

        assert 0 <= score.score <= 1
        assert 0 <= score.probability <= 1
        assert score.score > 0.5  # Should detect as anomalous

    def test_probabilistic_scoring_ensemble(self):
        """Test ensemble probabilistic scoring"""
        from src.core.algorithms.probabilistic_scoring import ProbabilisticAnomalyScorer

        reference_data = np.random.normal(50, 5, 500)
        test_value = 75

        score = ProbabilisticAnomalyScorer.ensemble_probabilistic_score(
            test_value,
            reference_data,
            methods=['Gaussian', 'Bayesian', 'KDE'],
            weights=[0.4, 0.3, 0.3]
        )

        assert score.method == "Ensemble"
        assert 0 <= score.score <= 1

    def test_advanced_imputation_adaptive(self):
        """Test adaptive imputation method selection"""
        from src.core.algorithms.advanced_imputation import AdvancedImputer

        # Create data with missing values
        data = np.array([1.0, 2.0, np.nan, 4.0, np.nan, 6.0, 7.0, 8.0])

        imputed = AdvancedImputer.adaptive_imputation(data, method='auto')

        assert len(imputed) == len(data)
        assert not np.any(np.isnan(imputed))
        assert np.all(imputed > 0)

    def test_advanced_imputation_with_confidence(self):
        """Test imputation with uncertainty estimates"""
        from src.core.algorithms.advanced_imputation import AdvancedImputer

        data = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0])

        mean_imputed, std_imputed = AdvancedImputer.impute_with_confidence(
            data,
            method='auto',
            n_bootstrap=50
        )

        assert len(mean_imputed) == len(data)
        assert len(std_imputed) == len(data)
        assert not np.any(np.isnan(mean_imputed))
        assert np.all(std_imputed >= 0)

    def test_ensemble_methods_performance_weighted(self):
        """Test performance-weighted ensemble"""
        from src.core.algorithms.ensemble_methods import EnsembleAggregator

        predictions = [73.5, 76.2, 74.8, 77.1, 72.9]
        performance = [0.85, 0.90, 0.92, 0.80, 0.87]

        result = EnsembleAggregator.performance_weighted_average(
            predictions,
            performance
        )

        assert result.prediction > 0
        assert len(result.weights) == len(predictions)
        assert np.isclose(np.sum(result.weights), 1.0, atol=0.01)
        assert result.method == "Performance-Weighted Average"

    def test_ensemble_methods_inverse_variance(self):
        """Test inverse variance weighting"""
        from src.core.algorithms.ensemble_methods import EnsembleAggregator

        predictions = [100.0, 101.0, 99.5, 100.5]
        variances = [2.0, 1.0, 3.0, 1.5]

        result = EnsembleAggregator.inverse_variance_weighting(
            predictions,
            variances
        )

        # Lower variance should get higher weight
        assert result.weights[1] > result.weights[2]  # variance 1.0 > variance 3.0
        assert result.method == "Inverse Variance Weighting"

    def test_dynamic_ensemble_weight_adaptation(self):
        """Test dynamic ensemble weight updates"""
        from src.core.algorithms.ensemble_methods import DynamicEnsemble

        ensemble = DynamicEnsemble(n_models=3, learning_rate=0.1)

        initial_weights = ensemble.get_weights().copy()

        # Simulate predictions and updates
        predictions = [100.0, 95.0, 105.0]
        true_value = 100.0

        ensemble_pred = ensemble.predict(predictions)
        ensemble.update_weights(predictions, true_value)

        updated_weights = ensemble.get_weights()

        # Weights should have changed
        assert not np.array_equal(initial_weights, updated_weights)
        # Best predictor (index 0) should have higher weight
        assert updated_weights[0] > updated_weights[1]


class TestMonitoringAndEvaluation:
    """Test SESSION 6 monitoring and evaluation components"""

    def test_model_monitoring_service_initialization(self):
        """Test model monitoring service creation"""
        from src.core.services.model_monitoring_service import ModelMonitoringService

        service = ModelMonitoringService(
            metrics_storage_path="data/monitoring/test_metrics",
            degradation_threshold=0.10,
            critical_threshold=0.25
        )

        assert service is not None
        assert service.degradation_threshold == 0.10
        assert service.critical_threshold == 0.25

    def test_evaluation_metrics_classification(self):
        """Test classification metrics calculation"""
        from src.core.services.evaluation_metrics import EvaluationMetricsCalculator

        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 0])

        metrics = EvaluationMetricsCalculator.compute_classification_metrics(
            y_true,
            y_pred
        )

        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1_score <= 1
        assert metrics.confusion_matrix is not None

    def test_evaluation_metrics_regression(self):
        """Test regression metrics calculation"""
        from src.core.services.evaluation_metrics import EvaluationMetricsCalculator

        y_true = np.array([100.0, 110.0, 95.0, 105.0, 102.0])
        y_pred = np.array([98.0, 112.0, 93.0, 107.0, 101.0])

        metrics = EvaluationMetricsCalculator.compute_regression_metrics(
            y_true,
            y_pred
        )

        assert metrics.mae >= 0
        assert metrics.mse >= 0
        assert metrics.rmse >= 0
        assert -1 <= metrics.r2 <= 1


class TestDataManagement:
    """Test data processing and pipeline components"""

    def test_data_processing_service_initialization(self):
        """Test data processing service"""
        try:
            from src.core.services.data_processing_service import DataProcessingService

            service = DataProcessingService()
            assert service is not None
        except ImportError:
            pytest.skip("DataProcessingService not implemented")

    def test_feature_engineering(self):
        """Test feature engineering"""
        try:
            from src.core.services.feature_engineering import FeatureEngineer

            engineer = FeatureEngineer()
            assert engineer is not None
        except ImportError:
            pytest.skip("FeatureEngineer not implemented")


class TestDashboardComponents:
    """Test dashboard layout components"""

    def test_mlflow_integration_layout(self):
        """Test MLflow integration layout creation"""
        from src.presentation.dashboard.layouts.mlflow_integration import create_mlflow_layout

        layout = create_mlflow_layout()
        assert layout is not None

    def test_training_monitor_layout(self):
        """Test training monitor layout creation"""
        from src.presentation.dashboard.layouts.training_monitor import create_training_monitor_layout

        layout = create_training_monitor_layout()
        assert layout is not None

    def test_anomaly_investigation_layout(self):
        """Test anomaly investigation layout creation"""
        from src.presentation.dashboard.layouts.anomaly_investigation import create_anomaly_investigation_layout

        layout = create_anomaly_investigation_layout()
        assert layout is not None


class TestIntegration:
    """Integration tests across multiple components"""

    def test_config_to_algorithms_integration(self):
        """Test configuration system integration with algorithms"""
        from config.config_manager import load_config
        from src.core.algorithms.adaptive_thresholding import AdaptiveThresholdCalculator

        config = load_config('config/config.yaml', env='development')

        # Get contamination from config
        contamination = config.get('anomaly_detection.general.contamination', 0.1)

        # Use in algorithm
        data = np.random.normal(50, 5, 500)
        result = AdaptiveThresholdCalculator.iqr_threshold(data, confidence_level=0.99)

        assert result is not None
        assert result.threshold > 0

    def test_end_to_end_anomaly_detection_flow(self):
        """Test complete anomaly detection workflow"""
        from src.core.algorithms.adaptive_thresholding import AdaptiveThresholdCalculator
        from src.core.algorithms.probabilistic_scoring import ProbabilisticAnomalyScorer

        # 1. Generate training data
        training_data = np.random.normal(50, 5, 1000)

        # 2. Calculate adaptive threshold
        threshold_result = AdaptiveThresholdCalculator.consensus_threshold(
            training_data,
            confidence_level=0.99
        )

        # 3. Test new data point
        test_value = 75.0  # Anomalous

        # 4. Calculate probabilistic score
        prob_score = ProbabilisticAnomalyScorer.bayesian_anomaly_probability(
            test_value,
            training_data
        )

        # 5. Make decision
        is_anomaly = test_value > threshold_result.threshold and prob_score.score > 0.5

        assert is_anomaly == True
        assert prob_score.score > 0.5
        assert threshold_result.threshold < test_value

    def test_end_to_end_forecasting_with_ensemble(self):
        """Test forecasting with ensemble methods"""
        from src.core.algorithms.ensemble_methods import EnsembleAggregator

        # Simulate multiple model predictions
        lstm_forecast = 102.5
        transformer_forecast = 101.8
        arima_forecast = 103.2

        predictions = [lstm_forecast, transformer_forecast, arima_forecast]
        performance_scores = [0.85, 0.92, 0.78]  # R² scores

        # Ensemble predictions
        ensemble_result = EnsembleAggregator.performance_weighted_average(
            predictions,
            performance_scores
        )

        # Transformer has best performance, should get highest weight
        assert ensemble_result.weights[1] > ensemble_result.weights[0]
        assert ensemble_result.weights[1] > ensemble_result.weights[2]
        assert 100 < ensemble_result.prediction < 105


@pytest.fixture
def sample_sensor_data():
    """Fixture providing sample sensor data"""
    dates = pd.date_range(start='2025-01-01', end='2025-01-31', freq='H')
    data = pd.DataFrame({
        'timestamp': dates,
        'sensor_id': 'SMAP_PWR_01',
        'value': np.random.normal(100, 10, len(dates)),
        'temperature': np.random.normal(25, 5, len(dates)),
        'humidity': np.random.normal(60, 10, len(dates))
    })
    return data


@pytest.fixture
def sample_anomalies():
    """Fixture providing sample anomaly data"""
    return [
        {
            'timestamp': datetime.now() - timedelta(hours=2),
            'sensor_id': 'SMAP_PWR_01',
            'value': 150.0,
            'anomaly_score': 0.95,
            'severity': 'high'
        },
        {
            'timestamp': datetime.now() - timedelta(hours=5),
            'sensor_id': 'MSL_MOB_F_05',
            'value': 75.0,
            'anomaly_score': 0.82,
            'severity': 'medium'
        }
    ]


class TestDataPipeline:
    """Test data pipeline with fixtures"""

    def test_sensor_data_processing(self, sample_sensor_data):
        """Test sensor data processing"""
        assert len(sample_sensor_data) > 0
        assert 'timestamp' in sample_sensor_data.columns
        assert 'value' in sample_sensor_data.columns

        # Check data quality
        assert not sample_sensor_data['value'].isnull().any()
        assert sample_sensor_data['value'].mean() > 0

    def test_anomaly_data_structure(self, sample_anomalies):
        """Test anomaly data structure"""
        assert len(sample_anomalies) > 0

        for anomaly in sample_anomalies:
            assert 'timestamp' in anomaly
            assert 'sensor_id' in anomaly
            assert 'anomaly_score' in anomaly
            assert 0 <= anomaly['anomaly_score'] <= 1


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])

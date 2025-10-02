"""
Probabilistic Anomaly Scoring
Bayesian and probabilistic methods for anomaly detection
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ProbabilisticScore:
    """Probabilistic anomaly score result"""

    score: float  # Anomaly score (0-1, higher = more anomalous)
    probability: float  # Probability of being anomalous
    likelihood: float  # Likelihood under normal model
    method: str
    confidence_interval: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "score": float(self.score),
            "probability": float(self.probability),
            "likelihood": float(self.likelihood),
            "method": self.method,
        }
        if self.confidence_interval:
            result["confidence_interval"] = [
                float(self.confidence_interval[0]),
                float(self.confidence_interval[1]),
            ]
        return result


class ProbabilisticAnomalyScorer:
    """
    Probabilistic anomaly scoring methods

    Methods:
    - Gaussian likelihood
    - Mixture model likelihood
    - Bayesian anomaly probability
    - CUSUM (Cumulative Sum)
    - Kernel density estimation
    """

    @staticmethod
    def gaussian_likelihood_score(value: float, reference_data: np.ndarray, alpha: float = 0.05) -> ProbabilisticScore:
        """
        Gaussian likelihood-based anomaly score

        Args:
            value: Value to score
            reference_data: Reference (normal) data
            alpha: Significance level for confidence interval

        Returns:
            ProbabilisticScore
        """
        # Fit Gaussian to reference data
        mean = np.mean(reference_data)
        std = np.std(reference_data)

        if std < 1e-10:
            std = 1e-10  # Avoid division by zero

        # Calculate likelihood (PDF value)
        likelihood = stats.norm.pdf(value, loc=mean, scale=std)

        # Calculate probability of being more extreme
        # P(X > value) or P(X < value)
        if value > mean:
            p_extreme = 1 - stats.norm.cdf(value, loc=mean, scale=std)
        else:
            p_extreme = stats.norm.cdf(value, loc=mean, scale=std)

        # Anomaly probability (two-tailed)
        anomaly_prob = 2 * min(p_extreme, 1 - p_extreme)

        # Anomaly score (0-1, higher = more anomalous)
        # Using negative log-likelihood normalized
        max_likelihood = stats.norm.pdf(mean, loc=mean, scale=std)
        score = 1 - (likelihood / (max_likelihood + 1e-10))

        # Confidence interval
        ci_lower, ci_upper = stats.norm.interval(1 - alpha, loc=mean, scale=std)

        return ProbabilisticScore(
            score=score,
            probability=anomaly_prob,
            likelihood=likelihood,
            method="Gaussian",
            confidence_interval=(ci_lower, ci_upper),
        )

    @staticmethod
    def mixture_model_score(value: float, reference_data: np.ndarray, n_components: int = 2) -> ProbabilisticScore:
        """
        Gaussian Mixture Model likelihood score

        Args:
            value: Value to score
            reference_data: Reference data
            n_components: Number of mixture components

        Returns:
            ProbabilisticScore
        """
        try:
            from sklearn.mixture import GaussianMixture

            # Fit GMM
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(reference_data.reshape(-1, 1))

            # Score sample
            log_likelihood = gmm.score_samples(np.array([[value]]))[0]
            likelihood = np.exp(log_likelihood)

            # Get maximum possible likelihood (at mode)
            max_log_likelihood = np.max(gmm.score_samples(reference_data.reshape(-1, 1)))
            max_likelihood = np.exp(max_log_likelihood)

            # Anomaly score
            score = 1 - (likelihood / (max_likelihood + 1e-10))

            # Estimate anomaly probability using threshold
            threshold = np.percentile(gmm.score_samples(reference_data.reshape(-1, 1)), 5)
            anomaly_prob = 1.0 if log_likelihood < threshold else 0.0

            return ProbabilisticScore(
                score=score,
                probability=anomaly_prob,
                likelihood=likelihood,
                method="GaussianMixture",
            )

        except ImportError:
            logger.warning("scikit-learn not available, using Gaussian fallback")
            return ProbabilisticAnomalyScorer.gaussian_likelihood_score(value, reference_data)
        except Exception as e:
            logger.error(f"Error in mixture model scoring: {e}")
            return ProbabilisticAnomalyScorer.gaussian_likelihood_score(value, reference_data)

    @staticmethod
    def bayesian_anomaly_probability(
        value: float, reference_data: np.ndarray, prior_anomaly_rate: float = 0.01
    ) -> ProbabilisticScore:
        """
        Bayesian anomaly probability using Bayes' theorem

        P(anomaly|data) = P(data|anomaly) * P(anomaly) / P(data)

        Args:
            value: Value to score
            reference_data: Reference (normal) data
            prior_anomaly_rate: Prior probability of anomaly

        Returns:
            ProbabilisticScore
        """
        # Fit normal model
        mean = np.mean(reference_data)
        std = np.std(reference_data)

        if std < 1e-10:
            std = 1e-10

        # P(data|normal) - likelihood under normal model
        p_data_given_normal = stats.norm.pdf(value, loc=mean, scale=std)

        # P(data|anomaly) - assume uniform distribution over wider range
        # Use 5x the range of reference data
        data_range = np.max(reference_data) - np.min(reference_data)
        anomaly_range = data_range * 5
        p_data_given_anomaly = 1 / (anomaly_range + 1e-10)

        # Priors
        p_normal = 1 - prior_anomaly_rate
        p_anomaly = prior_anomaly_rate

        # Total probability: P(data) = P(data|normal)P(normal) + P(data|anomaly)P(anomaly)
        p_data = p_data_given_normal * p_normal + p_data_given_anomaly * p_anomaly

        # Posterior: P(anomaly|data)
        p_anomaly_given_data = (p_data_given_anomaly * p_anomaly) / (p_data + 1e-10)

        # Anomaly score (0-1)
        score = p_anomaly_given_data

        return ProbabilisticScore(
            score=score,
            probability=p_anomaly_given_data,
            likelihood=p_data_given_normal,
            method="Bayesian",
        )

    @staticmethod
    def kernel_density_score(
        value: float, reference_data: np.ndarray, bandwidth: Optional[float] = None
    ) -> ProbabilisticScore:
        """
        Kernel Density Estimation based score

        Args:
            value: Value to score
            reference_data: Reference data
            bandwidth: KDE bandwidth (Scott's rule if None)

        Returns:
            ProbabilisticScore
        """
        try:
            from sklearn.neighbors import KernelDensity

            # Estimate bandwidth if not provided (Scott's rule)
            if bandwidth is None:
                n = len(reference_data)
                std = np.std(reference_data)
                bandwidth = 1.06 * std * (n ** (-1 / 5))

            # Fit KDE
            kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
            kde.fit(reference_data.reshape(-1, 1))

            # Score sample
            log_likelihood = kde.score_samples(np.array([[value]]))[0]
            likelihood = np.exp(log_likelihood)

            # Get maximum likelihood in reference data
            max_log_likelihood = np.max(kde.score_samples(reference_data.reshape(-1, 1)))
            max_likelihood = np.exp(max_log_likelihood)

            # Anomaly score
            score = 1 - (likelihood / (max_likelihood + 1e-10))

            # Estimate probability (using percentile)
            ref_scores = kde.score_samples(reference_data.reshape(-1, 1))
            percentile = stats.percentileofscore(ref_scores, log_likelihood)
            anomaly_prob = 1 - (percentile / 100)

            return ProbabilisticScore(
                score=score,
                probability=anomaly_prob,
                likelihood=likelihood,
                method="KDE",
            )

        except ImportError:
            logger.warning("scikit-learn not available, using Gaussian fallback")
            return ProbabilisticAnomalyScorer.gaussian_likelihood_score(value, reference_data)
        except Exception as e:
            logger.error(f"Error in KDE scoring: {e}")
            return ProbabilisticAnomalyScorer.gaussian_likelihood_score(value, reference_data)

    @staticmethod
    def cusum_score(
        values: np.ndarray,
        reference_mean: float,
        reference_std: float,
        threshold: float = 5.0,
        slack: float = 0.5,
    ) -> List[ProbabilisticScore]:
        """
        CUSUM (Cumulative Sum) anomaly detection

        Detects shifts in mean

        Args:
            values: Time series values
            reference_mean: Reference mean
            reference_std: Reference standard deviation
            threshold: Detection threshold (in std units)
            slack: Slack parameter (allowable deviation)

        Returns:
            List of ProbabilisticScore for each value
        """
        scores = []

        # Initialize CUSUM
        s_high = 0  # Cumulative sum for upward shift
        s_low = 0  # Cumulative sum for downward shift

        for value in values:
            # Standardize
            z = (value - reference_mean) / (reference_std + 1e-10)

            # Update CUSUM
            s_high = max(0, s_high + z - slack)
            s_low = max(0, s_low - z - slack)

            # Anomaly score (based on CUSUM values)
            cusum_score = max(s_high, s_low) / (threshold + 1e-10)
            cusum_score = min(cusum_score, 1.0)  # Clip to [0, 1]

            # Anomaly probability
            anomaly_prob = 1.0 if (s_high > threshold or s_low > threshold) else cusum_score

            # Likelihood (inverse of anomaly score)
            likelihood = 1 - cusum_score

            scores.append(
                ProbabilisticScore(
                    score=cusum_score,
                    probability=anomaly_prob,
                    likelihood=likelihood,
                    method="CUSUM",
                )
            )

        return scores

    @staticmethod
    def ensemble_probabilistic_score(
        value: float,
        reference_data: np.ndarray,
        methods: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
    ) -> ProbabilisticScore:
        """
        Ensemble probabilistic scoring using multiple methods

        Args:
            value: Value to score
            reference_data: Reference data
            methods: Methods to use (all if None)
            weights: Weights for each method (uniform if None)

        Returns:
            Ensemble ProbabilisticScore
        """
        if methods is None:
            methods = ["Gaussian", "Bayesian", "KDE"]

        if weights is None:
            weights = [1.0 / len(methods)] * len(methods)

        scores_list = []

        for method in methods:
            try:
                if method == "Gaussian":
                    score = ProbabilisticAnomalyScorer.gaussian_likelihood_score(value, reference_data)
                elif method == "GaussianMixture":
                    score = ProbabilisticAnomalyScorer.mixture_model_score(value, reference_data)
                elif method == "Bayesian":
                    score = ProbabilisticAnomalyScorer.bayesian_anomaly_probability(value, reference_data)
                elif method == "KDE":
                    score = ProbabilisticAnomalyScorer.kernel_density_score(value, reference_data)
                else:
                    continue

                scores_list.append(score)

            except Exception as e:
                logger.warning(f"Error with method {method}: {e}")

        if not scores_list:
            # Fallback
            return ProbabilisticAnomalyScorer.gaussian_likelihood_score(value, reference_data)

        # Weighted average
        ensemble_score = sum(s.score * w for s, w in zip(scores_list, weights[: len(scores_list)]))
        ensemble_prob = sum(s.probability * w for s, w in zip(scores_list, weights[: len(scores_list)]))
        ensemble_likelihood = sum(s.likelihood * w for s, w in zip(scores_list, weights[: len(scores_list)]))

        return ProbabilisticScore(
            score=ensemble_score,
            probability=ensemble_prob,
            likelihood=ensemble_likelihood,
            method=f"Ensemble-{len(scores_list)}methods",
        )


class AnomalyConfidenceEstimator:
    """
    Estimate confidence/uncertainty in anomaly predictions
    """

    @staticmethod
    def bootstrap_confidence(
        value: float,
        reference_data: np.ndarray,
        n_bootstrap: int = 100,
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Bootstrap confidence interval for anomaly score

        Args:
            value: Value to score
            reference_data: Reference data
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level

        Returns:
            Dictionary with confidence estimates
        """
        scores = []

        for _ in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_sample = np.random.choice(reference_data, size=len(reference_data), replace=True)

            # Compute score
            result = ProbabilisticAnomalyScorer.gaussian_likelihood_score(value, bootstrap_sample)
            scores.append(result.score)

        scores = np.array(scores)

        # Confidence interval
        alpha = 1 - confidence_level
        ci_lower = np.percentile(scores, alpha / 2 * 100)
        ci_upper = np.percentile(scores, (1 - alpha / 2) * 100)

        return {
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "confidence_interval": (float(ci_lower), float(ci_upper)),
            "confidence_level": confidence_level,
        }

    @staticmethod
    def prediction_interval(reference_data: np.ndarray, confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Prediction interval for normal values

        Args:
            reference_data: Reference data
            confidence_level: Confidence level

        Returns:
            (lower_bound, upper_bound) tuple
        """
        mean = np.mean(reference_data)
        std = np.std(reference_data)
        n = len(reference_data)

        # Prediction interval (wider than confidence interval)
        # Accounts for both estimation uncertainty and natural variability
        t_value = stats.t.ppf((1 + confidence_level) / 2, df=n - 1)

        margin = t_value * std * np.sqrt(1 + 1 / n)

        lower = mean - margin
        upper = mean + margin

        return (lower, upper)

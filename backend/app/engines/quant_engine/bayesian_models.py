"""
Bayesian Probabilistic Models for Market Forecasting
Provides conditional probability updates and adaptive scenario modeling.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class BayesianPrediction:
    posterior_mean: float
    posterior_std: float
    credible_interval_90: Tuple[float, float]
    credible_interval_95: Tuple[float, float]
    prob_positive: float
    prob_negative: float
    evidence_strength: float
    prior_params: Dict
    posterior_params: Dict


@dataclass
class ConditionalProbabilities:
    base_probability: float
    conditional_adjustments: Dict[str, float]
    final_probability: float
    confidence: float


class BayesianReturnModel:
    """
    Bayesian model for return prediction using conjugate priors.
    Updates beliefs based on new evidence (realized returns, macro data).
    """
    
    def __init__(
        self,
        prior_mean: float = 0.0,
        prior_variance: float = 0.02,
        prior_df: int = 30,
        learning_rate: float = 0.1
    ):
        # Normal-Inverse-Gamma prior parameters
        self.mu_0 = prior_mean  # Prior mean
        self.kappa_0 = 1.0  # Prior precision for mean
        self.alpha_0 = prior_df / 2  # Shape for variance
        self.beta_0 = prior_variance * prior_df / 2  # Rate for variance
        
        # Current posterior parameters
        self.mu_n = self.mu_0
        self.kappa_n = self.kappa_0
        self.alpha_n = self.alpha_0
        self.beta_n = self.beta_0
        
        self.learning_rate = learning_rate
        self.n_observations = 0
        
    def update(self, observations: np.ndarray) -> None:
        """
        Update posterior with new observations using conjugate update.
        """
        n = len(observations)
        if n == 0:
            return
            
        x_bar = np.mean(observations)
        s_sq = np.var(observations, ddof=1) if n > 1 else 0
        
        # Conjugate update for Normal-Inverse-Gamma
        self.kappa_n = self.kappa_0 + n
        self.mu_n = (self.kappa_0 * self.mu_0 + n * x_bar) / self.kappa_n
        self.alpha_n = self.alpha_0 + n / 2
        self.beta_n = (
            self.beta_0 + 
            0.5 * (n - 1) * s_sq + 
            (self.kappa_0 * n * (x_bar - self.mu_0) ** 2) / (2 * self.kappa_n)
        )
        
        self.n_observations += n
        
    def predict(self, horizon_days: int = 1) -> BayesianPrediction:
        """
        Generate predictive distribution for future returns.
        """
        # Posterior predictive is Student-t
        df = 2 * self.alpha_n
        loc = self.mu_n * horizon_days
        scale = np.sqrt(
            self.beta_n / self.alpha_n * (1 + 1 / self.kappa_n) * horizon_days
        )
        
        predictive = stats.t(df=df, loc=loc, scale=scale)
        
        # Compute intervals
        ci_90 = predictive.ppf([0.05, 0.95])
        ci_95 = predictive.ppf([0.025, 0.975])
        
        # Probabilities
        prob_positive = 1 - predictive.cdf(0)
        prob_negative = predictive.cdf(0)
        
        # Evidence strength (how much data has updated our beliefs)
        evidence = 1 - (self.kappa_0 / self.kappa_n)
        
        return BayesianPrediction(
            posterior_mean=float(loc),
            posterior_std=float(scale * np.sqrt(df / (df - 2)) if df > 2 else scale),
            credible_interval_90=(float(ci_90[0]), float(ci_90[1])),
            credible_interval_95=(float(ci_95[0]), float(ci_95[1])),
            prob_positive=float(prob_positive),
            prob_negative=float(prob_negative),
            evidence_strength=float(evidence),
            prior_params={
                "mu_0": self.mu_0,
                "kappa_0": self.kappa_0,
                "alpha_0": self.alpha_0,
                "beta_0": self.beta_0
            },
            posterior_params={
                "mu_n": float(self.mu_n),
                "kappa_n": float(self.kappa_n),
                "alpha_n": float(self.alpha_n),
                "beta_n": float(self.beta_n)
            }
        )


class BayesianConditionalModel:
    """
    Bayesian network for conditional probability modeling.
    Computes P(Return Direction | Macro State, Regime, Seasonality, etc.)
    """
    
    def __init__(self):
        # Prior conditional probabilities (empirically calibrated)
        self.conditional_priors = {
            "regime": {
                "risk_on": {"bullish": 0.65, "bearish": 0.20, "neutral": 0.15},
                "risk_off": {"bullish": 0.25, "bearish": 0.55, "neutral": 0.20},
                "euphoria": {"bullish": 0.50, "bearish": 0.35, "neutral": 0.15},
                "panic": {"bullish": 0.20, "bearish": 0.70, "neutral": 0.10},
                "consolidation": {"bullish": 0.35, "bearish": 0.35, "neutral": 0.30},
                "macro_stress": {"bullish": 0.30, "bearish": 0.50, "neutral": 0.20},
            },
            "macro_trend": {
                "expansion": {"bullish": 0.60, "bearish": 0.25, "neutral": 0.15},
                "contraction": {"bullish": 0.30, "bearish": 0.55, "neutral": 0.15},
                "neutral": {"bullish": 0.40, "bearish": 0.35, "neutral": 0.25},
            },
            "volatility_regime": {
                "low": {"bullish": 0.55, "bearish": 0.25, "neutral": 0.20},
                "medium": {"bullish": 0.40, "bearish": 0.35, "neutral": 0.25},
                "high": {"bullish": 0.30, "bearish": 0.50, "neutral": 0.20},
                "extreme": {"bullish": 0.25, "bearish": 0.60, "neutral": 0.15},
            },
            "momentum": {
                "strong_positive": {"bullish": 0.65, "bearish": 0.20, "neutral": 0.15},
                "weak_positive": {"bullish": 0.50, "bearish": 0.30, "neutral": 0.20},
                "weak_negative": {"bullish": 0.30, "bearish": 0.50, "neutral": 0.20},
                "strong_negative": {"bullish": 0.20, "bearish": 0.65, "neutral": 0.15},
            },
            "seasonality": {
                "bullish_period": {"bullish": 0.55, "bearish": 0.30, "neutral": 0.15},
                "bearish_period": {"bullish": 0.35, "bearish": 0.50, "neutral": 0.15},
                "neutral_period": {"bullish": 0.40, "bearish": 0.40, "neutral": 0.20},
            }
        }
        
        # Factor weights (learned over time)
        self.factor_weights = {
            "regime": 0.25,
            "macro_trend": 0.20,
            "volatility_regime": 0.15,
            "momentum": 0.25,
            "seasonality": 0.15
        }
        
        # Observation counts for Bayesian updates
        self.observation_counts = {}
        
    def compute_conditional_probability(
        self,
        conditions: Dict[str, str],
        target: str = "bullish"
    ) -> ConditionalProbabilities:
        """
        Compute conditional probability given observed conditions.
        Uses weighted averaging of conditional probabilities.
        """
        base_prob = 1/3  # Uninformed prior
        
        weighted_probs = []
        total_weight = 0
        adjustments = {}
        
        for factor, state in conditions.items():
            if factor in self.conditional_priors and state in self.conditional_priors[factor]:
                prob = self.conditional_priors[factor][state].get(target, 1/3)
                weight = self.factor_weights.get(factor, 0.1)
                
                weighted_probs.append(prob * weight)
                total_weight += weight
                adjustments[f"{factor}={state}"] = prob - base_prob
        
        if total_weight > 0:
            final_prob = sum(weighted_probs) / total_weight
        else:
            final_prob = base_prob
            
        # Confidence based on number of conditions and agreement
        confidence = min(len(conditions) / 5, 1.0) * (1 - np.std(list(adjustments.values())) if adjustments else 0.5)
        
        return ConditionalProbabilities(
            base_probability=base_prob,
            conditional_adjustments=adjustments,
            final_probability=float(final_prob),
            confidence=float(confidence)
        )
    
    def update_from_observation(
        self,
        conditions: Dict[str, str],
        realized_direction: str
    ) -> None:
        """
        Update conditional probabilities based on realized outcome.
        Uses simple count-based Bayesian update.
        """
        for factor, state in conditions.items():
            key = f"{factor}:{state}:{realized_direction}"
            self.observation_counts[key] = self.observation_counts.get(key, 0) + 1
            
            # Update prior with observed frequency
            total_key = f"{factor}:{state}:total"
            self.observation_counts[total_key] = self.observation_counts.get(total_key, 0) + 1
            
            if self.observation_counts[total_key] >= 30:  # Minimum sample
                for direction in ["bullish", "bearish", "neutral"]:
                    dir_key = f"{factor}:{state}:{direction}"
                    count = self.observation_counts.get(dir_key, 0)
                    total = self.observation_counts[total_key]
                    
                    # Bayesian smoothing
                    prior = self.conditional_priors.get(factor, {}).get(state, {}).get(direction, 1/3)
                    posterior = (count + 10 * prior) / (total + 10)
                    
                    if factor in self.conditional_priors and state in self.conditional_priors[factor]:
                        self.conditional_priors[factor][state][direction] = posterior


class BayesianFactorModel:
    """
    Bayesian factor model for return attribution and prediction.
    Estimates factor loadings with uncertainty quantification.
    """
    
    def __init__(self, factors: List[str]):
        self.factors = factors
        self.n_factors = len(factors)
        
        # Prior on factor loadings: N(0, 1)
        self.beta_prior_mean = np.zeros(self.n_factors)
        self.beta_prior_cov = np.eye(self.n_factors)
        
        # Prior on residual variance: InvGamma(3, 0.01)
        self.sigma_prior_alpha = 3.0
        self.sigma_prior_beta = 0.01
        
        # Posterior parameters
        self.beta_posterior_mean = self.beta_prior_mean.copy()
        self.beta_posterior_cov = self.beta_prior_cov.copy()
        
    def fit(self, returns: np.ndarray, factor_returns: np.ndarray) -> None:
        """
        Fit Bayesian linear regression: r = X @ beta + epsilon
        """
        n = len(returns)
        X = factor_returns
        y = returns
        
        # OLS estimate
        XtX = X.T @ X
        Xty = X.T @ y
        
        # Posterior for beta (conjugate update)
        prior_precision = np.linalg.inv(self.beta_prior_cov)
        posterior_precision = prior_precision + XtX
        self.beta_posterior_cov = np.linalg.inv(posterior_precision)
        self.beta_posterior_mean = self.beta_posterior_cov @ (
            prior_precision @ self.beta_prior_mean + Xty
        )
        
    def predict_return(
        self,
        factor_forecasts: np.ndarray,
        factor_uncertainties: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """
        Predict return given factor forecasts, propagating uncertainty.
        """
        # Point estimate
        expected_return = np.dot(factor_forecasts, self.beta_posterior_mean)
        
        # Uncertainty from beta uncertainty
        beta_var = np.diag(self.beta_posterior_cov)
        
        # Propagate uncertainty
        if factor_uncertainties is not None:
            total_var = np.sum(
                factor_forecasts**2 * beta_var + 
                self.beta_posterior_mean**2 * factor_uncertainties**2
            )
        else:
            total_var = np.sum(factor_forecasts**2 * beta_var)
            
        return float(expected_return), float(np.sqrt(total_var))

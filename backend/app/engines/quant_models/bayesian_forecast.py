"""
ZoneSignal - Bayesian Forecast Engine
Modèle de prévision probabiliste avec mise à jour bayésienne dynamique
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from scipy import stats
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')


@dataclass
class BayesianPrior:
    """Prior bayésien pour un paramètre"""
    mean: float
    std: float
    distribution: str = "normal"  # normal, student_t, beta
    df: float = 5.0  # degrés de liberté pour student_t


@dataclass
class BayesianPosterior:
    """Posterior bayésien après mise à jour"""
    mean: float
    std: float
    credible_interval_90: Tuple[float, float]
    credible_interval_95: Tuple[float, float]
    samples: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class BayesianForecastResult:
    """Résultat complet de la prévision bayésienne"""
    # Prévision centrale
    expected_return: float
    expected_volatility: float
    
    # Distributions
    return_posterior: BayesianPosterior
    volatility_posterior: BayesianPosterior
    
    # Probabilités directionnelles
    prob_positive: float
    prob_negative: float
    prob_above_threshold: Dict[float, float]  # {5%: 0.3, 10%: 0.15, ...}
    prob_below_threshold: Dict[float, float]  # {-5%: 0.25, -10%: 0.12, ...}
    
    # Scénarios
    scenarios: Dict[str, Dict[str, float]]
    
    # Métriques de confiance
    model_confidence: float
    information_ratio: float
    prior_weight: float  # Poids du prior vs données


class BayesianForecastEngine:
    """
    Moteur de prévision bayésienne institutionnel
    
    Caractéristiques:
    - Priors informatifs basés sur les régimes de marché
    - Mise à jour conjuguée pour efficacité
    - Modélisation de l'incertitude paramétrique
    - Scénarios conditionnels
    - Shrinkage bayésien pour régularisation
    """
    
    def __init__(self):
        # Priors par défaut (peuvent être ajustés par régime)
        self.default_return_prior = BayesianPrior(
            mean=0.0003,  # ~8% annualisé
            std=0.02,
            distribution="student_t",
            df=5
        )
        
        self.default_vol_prior = BayesianPrior(
            mean=0.16,  # 16% annualisé
            std=0.05,
            distribution="normal"
        )
        
        # Priors conditionnels par régime
        self.regime_priors = self._initialize_regime_priors()
        
        # Hyperparamètres
        self.n_samples = 5000
        self.shrinkage_factor = 0.3  # Shrinkage vers le prior
    
    def _initialize_regime_priors(self) -> Dict[str, Dict[str, BayesianPrior]]:
        """Initialise les priors conditionnels par régime de marché"""
        
        return {
            "risk_on": {
                "return": BayesianPrior(mean=0.0006, std=0.015, distribution="normal"),
                "volatility": BayesianPrior(mean=0.12, std=0.03, distribution="normal")
            },
            "risk_off": {
                "return": BayesianPrior(mean=-0.0003, std=0.025, distribution="student_t", df=4),
                "volatility": BayesianPrior(mean=0.22, std=0.06, distribution="normal")
            },
            "euphoria": {
                "return": BayesianPrior(mean=0.001, std=0.02, distribution="normal"),
                "volatility": BayesianPrior(mean=0.10, std=0.025, distribution="normal")
            },
            "panic": {
                "return": BayesianPrior(mean=-0.002, std=0.04, distribution="student_t", df=3),
                "volatility": BayesianPrior(mean=0.35, std=0.10, distribution="normal")
            },
            "consolidation": {
                "return": BayesianPrior(mean=0.0001, std=0.01, distribution="normal"),
                "volatility": BayesianPrior(mean=0.14, std=0.03, distribution="normal")
            },
            "macro_stress": {
                "return": BayesianPrior(mean=-0.001, std=0.03, distribution="student_t", df=4),
                "volatility": BayesianPrior(mean=0.28, std=0.08, distribution="normal")
            }
        }
    
    # ═══════════════════════════════════════════════════════════════
    # MISE À JOUR BAYÉSIENNE
    # ═══════════════════════════════════════════════════════════════
    
    def _conjugate_normal_update(
        self,
        prior: BayesianPrior,
        observations: np.ndarray,
        known_variance: Optional[float] = None
    ) -> BayesianPosterior:
        """
        Mise à jour conjuguée Normal-Normal
        
        Pour mean avec variance connue ou estimée
        """
        
        n = len(observations)
        if n == 0:
            # Pas de données, retourner le prior
            return BayesianPosterior(
                mean=prior.mean,
                std=prior.std,
                credible_interval_90=(
                    prior.mean - 1.645 * prior.std,
                    prior.mean + 1.645 * prior.std
                ),
                credible_interval_95=(
                    prior.mean - 1.96 * prior.std,
                    prior.mean + 1.96 * prior.std
                ),
                samples=np.random.normal(prior.mean, prior.std, self.n_samples)
            )
        
        # Variance des observations
        if known_variance is None:
            obs_variance = np.var(observations, ddof=1)
        else:
            obs_variance = known_variance
        
        obs_mean = np.mean(observations)
        
        # Précision (inverse de variance)
        prior_precision = 1 / (prior.std ** 2)
        likelihood_precision = n / obs_variance
        
        # Posterior precision et variance
        posterior_precision = prior_precision + likelihood_precision
        posterior_variance = 1 / posterior_precision
        posterior_std = np.sqrt(posterior_variance)
        
        # Posterior mean (moyenne pondérée)
        posterior_mean = (
            prior_precision * prior.mean + likelihood_precision * obs_mean
        ) / posterior_precision
        
        # Shrinkage vers le prior
        posterior_mean = (
            self.shrinkage_factor * prior.mean + 
            (1 - self.shrinkage_factor) * posterior_mean
        )
        
        # Générer des échantillons
        if prior.distribution == "student_t":
            # Utiliser Student-t pour queues épaisses
            samples = stats.t.rvs(
                df=prior.df,
                loc=posterior_mean,
                scale=posterior_std,
                size=self.n_samples
            )
        else:
            samples = np.random.normal(posterior_mean, posterior_std, self.n_samples)
        
        # Intervalles de crédibilité
        ci_90 = (np.percentile(samples, 5), np.percentile(samples, 95))
        ci_95 = (np.percentile(samples, 2.5), np.percentile(samples, 97.5))
        
        return BayesianPosterior(
            mean=posterior_mean,
            std=posterior_std,
            credible_interval_90=ci_90,
            credible_interval_95=ci_95,
            samples=samples
        )
    
    def _inverse_gamma_update(
        self,
        prior: BayesianPrior,
        observations: np.ndarray
    ) -> BayesianPosterior:
        """
        Mise à jour pour la variance (distribution inverse-gamma)
        """
        
        n = len(observations)
        
        if n < 2:
            return BayesianPosterior(
                mean=prior.mean,
                std=prior.std,
                credible_interval_90=(prior.mean * 0.7, prior.mean * 1.3),
                credible_interval_95=(prior.mean * 0.6, prior.mean * 1.4),
                samples=np.abs(np.random.normal(prior.mean, prior.std, self.n_samples))
            )
        
        # Paramètres prior (approximation inverse-gamma)
        alpha_prior = (prior.mean / prior.std) ** 2
        beta_prior = prior.mean * alpha_prior
        
        # Statistiques suffisantes
        sample_var = np.var(observations, ddof=1)
        
        # Posterior parameters
        alpha_post = alpha_prior + n / 2
        beta_post = beta_prior + (n - 1) * sample_var / 2
        
        # Posterior mean pour la variance
        posterior_var_mean = beta_post / (alpha_post - 1) if alpha_post > 1 else sample_var
        
        # Convertir en volatilité (écart-type annualisé)
        daily_vol = np.sqrt(posterior_var_mean)
        annual_vol = daily_vol * np.sqrt(252)
        
        # Générer des échantillons de volatilité
        var_samples = stats.invgamma.rvs(alpha_post, scale=beta_post, size=self.n_samples)
        vol_samples = np.sqrt(var_samples) * np.sqrt(252)
        
        # Intervalles
        ci_90 = (np.percentile(vol_samples, 5), np.percentile(vol_samples, 95))
        ci_95 = (np.percentile(vol_samples, 2.5), np.percentile(vol_samples, 97.5))
        
        return BayesianPosterior(
            mean=annual_vol,
            std=np.std(vol_samples),
            credible_interval_90=ci_90,
            credible_interval_95=ci_95,
            samples=vol_samples
        )
    
    # ═══════════════════════════════════════════════════════════════
    # PRÉVISION PRINCIPALE
    # ═══════════════════════════════════════════════════════════════
    
    def forecast(
        self,
        df: pd.DataFrame,
        horizon_days: int,
        current_regime: Optional[str] = None,
        macro_adjustment: float = 0.0,
        lookback_days: int = 60
    ) -> BayesianForecastResult:
        """
        Génère une prévision bayésienne complète
        
        Args:
            df: Données historiques avec 'daily_return'
            horizon_days: Horizon de prévision en jours
            current_regime: Régime de marché actuel (optionnel)
            macro_adjustment: Ajustement macro (ex: +0.001 pour biais haussier)
            lookback_days: Jours de données à utiliser
            
        Returns:
            BayesianForecastResult complet
        """
        
        # Extraire les returns récents
        returns = df['daily_return'].dropna().tail(lookback_days).values
        
        if len(returns) < 10:
            raise ValueError("Insufficient data for Bayesian forecast")
        
        # Sélectionner les priors selon le régime
        if current_regime and current_regime.lower() in self.regime_priors:
            regime_prior = self.regime_priors[current_regime.lower()]
            return_prior = regime_prior["return"]
            vol_prior = regime_prior["volatility"]
        else:
            return_prior = self.default_return_prior
            vol_prior = self.default_vol_prior
        
        # Mise à jour bayésienne pour le return journalier
        return_posterior = self._conjugate_normal_update(
            return_prior, 
            returns
        )
        
        # Mise à jour bayésienne pour la volatilité
        vol_posterior = self._inverse_gamma_update(
            vol_prior,
            returns
        )
        
        # Appliquer l'ajustement macro
        adjusted_return_mean = return_posterior.mean + macro_adjustment / 252
        
        # Projeter sur l'horizon
        horizon_return_mean = adjusted_return_mean * horizon_days
        horizon_return_std = return_posterior.std * np.sqrt(horizon_days)
        
        # Générer les échantillons de return sur l'horizon
        horizon_samples = np.random.normal(
            horizon_return_mean,
            horizon_return_std,
            self.n_samples
        )
        
        # Créer le posterior pour l'horizon
        horizon_return_posterior = BayesianPosterior(
            mean=horizon_return_mean,
            std=horizon_return_std,
            credible_interval_90=(
                np.percentile(horizon_samples, 5),
                np.percentile(horizon_samples, 95)
            ),
            credible_interval_95=(
                np.percentile(horizon_samples, 2.5),
                np.percentile(horizon_samples, 97.5)
            ),
            samples=horizon_samples
        )
        
        # Calculer les probabilités directionnelles
        prob_positive = float(np.mean(horizon_samples > 0))
        prob_negative = float(np.mean(horizon_samples < 0))
        
        # Probabilités par seuil
        thresholds_up = [0.02, 0.05, 0.10, 0.15, 0.20]
        thresholds_down = [-0.02, -0.05, -0.10, -0.15, -0.20]
        
        prob_above = {
            thresh: float(np.mean(horizon_samples > thresh))
            for thresh in thresholds_up
        }
        
        prob_below = {
            thresh: float(np.mean(horizon_samples < thresh))
            for thresh in thresholds_down
        }
        
        # Générer des scénarios
        scenarios = self._generate_scenarios(
            horizon_samples,
            vol_posterior.samples,
            horizon_days
        )
        
        # Calculer la confiance du modèle
        model_confidence = self._compute_model_confidence(
            return_posterior,
            vol_posterior,
            len(returns)
        )
        
        # Information ratio (signal / bruit)
        information_ratio = abs(horizon_return_mean) / horizon_return_std
        
        # Poids du prior (combien le prior influence le posterior)
        prior_weight = self._compute_prior_weight(returns, return_prior)
        
        return BayesianForecastResult(
            expected_return=horizon_return_mean,
            expected_volatility=vol_posterior.mean,
            return_posterior=horizon_return_posterior,
            volatility_posterior=vol_posterior,
            prob_positive=prob_positive,
            prob_negative=prob_negative,
            prob_above_threshold=prob_above,
            prob_below_threshold=prob_below,
            scenarios=scenarios,
            model_confidence=model_confidence,
            information_ratio=information_ratio,
            prior_weight=prior_weight
        )
    
    def _generate_scenarios(
        self,
        return_samples: np.ndarray,
        vol_samples: np.ndarray,
        horizon_days: int
    ) -> Dict[str, Dict[str, float]]:
        """Génère des scénarios probabilistes"""
        
        return {
            "base_case": {
                "return": float(np.median(return_samples)),
                "volatility": float(np.median(vol_samples)),
                "probability": 0.50,
                "description": "Scénario médian"
            },
            "bullish": {
                "return": float(np.percentile(return_samples, 75)),
                "volatility": float(np.percentile(vol_samples, 25)),
                "probability": 0.25,
                "description": "Scénario haussier modéré"
            },
            "strongly_bullish": {
                "return": float(np.percentile(return_samples, 90)),
                "volatility": float(np.percentile(vol_samples, 10)),
                "probability": 0.10,
                "description": "Scénario fortement haussier"
            },
            "bearish": {
                "return": float(np.percentile(return_samples, 25)),
                "volatility": float(np.percentile(vol_samples, 75)),
                "probability": 0.25,
                "description": "Scénario baissier modéré"
            },
            "strongly_bearish": {
                "return": float(np.percentile(return_samples, 10)),
                "volatility": float(np.percentile(vol_samples, 90)),
                "probability": 0.10,
                "description": "Scénario fortement baissier"
            },
            "tail_risk": {
                "return": float(np.percentile(return_samples, 5)),
                "volatility": float(np.percentile(vol_samples, 95)),
                "probability": 0.05,
                "description": "Scénario de stress extrême"
            }
        }
    
    def _compute_model_confidence(
        self,
        return_posterior: BayesianPosterior,
        vol_posterior: BayesianPosterior,
        n_observations: int
    ) -> float:
        """
        Calcule un score de confiance du modèle (0-1)
        
        Basé sur:
        - Précision du posterior (faible std)
        - Nombre d'observations
        - Cohérence des distributions
        """
        
        # Score basé sur la précision (inverse du coefficient de variation)
        cv_return = abs(return_posterior.std / (return_posterior.mean + 1e-10))
        cv_vol = vol_posterior.std / vol_posterior.mean
        
        precision_score = 1 / (1 + cv_return + cv_vol)
        
        # Score basé sur le nombre d'observations
        sample_score = min(1.0, n_observations / 252)  # Saturé à 1 an
        
        # Score combiné
        confidence = 0.6 * precision_score + 0.4 * sample_score
        
        return float(np.clip(confidence, 0, 1))
    
    def _compute_prior_weight(
        self,
        observations: np.ndarray,
        prior: BayesianPrior
    ) -> float:
        """Calcule le poids effectif du prior dans le posterior"""
        
        n = len(observations)
        obs_var = np.var(observations, ddof=1)
        prior_var = prior.std ** 2
        
        # Poids du prior = precision_prior / precision_totale
        prior_precision = 1 / prior_var
        likelihood_precision = n / obs_var
        
        prior_weight = prior_precision / (prior_precision + likelihood_precision)
        
        return float(prior_weight)
    
    # ═══════════════════════════════════════════════════════════════
    # MISE À JOUR ADAPTATIVE DES PRIORS
    # ═══════════════════════════════════════════════════════════════
    
    def update_regime_priors(
        self,
        regime: str,
        historical_returns: np.ndarray,
        historical_volatility: np.ndarray
    ):
        """
        Met à jour les priors d'un régime basé sur les données historiques
        
        Apprentissage empirique bayésien
        """
        
        if regime.lower() not in self.regime_priors:
            return
        
        # Calculer les statistiques empiriques
        emp_return_mean = np.mean(historical_returns)
        emp_return_std = np.std(historical_returns)
        emp_vol_mean = np.mean(historical_volatility)
        emp_vol_std = np.std(historical_volatility)
        
        # Mise à jour avec shrinkage vers les priors existants
        current = self.regime_priors[regime.lower()]
        alpha = 0.3  # Poids des nouvelles données
        
        self.regime_priors[regime.lower()] = {
            "return": BayesianPrior(
                mean=(1 - alpha) * current["return"].mean + alpha * emp_return_mean,
                std=(1 - alpha) * current["return"].std + alpha * emp_return_std,
                distribution=current["return"].distribution,
                df=current["return"].df
            ),
            "volatility": BayesianPrior(
                mean=(1 - alpha) * current["volatility"].mean + alpha * emp_vol_mean,
                std=(1 - alpha) * current["volatility"].std + alpha * emp_vol_std,
                distribution=current["volatility"].distribution
            )
        }

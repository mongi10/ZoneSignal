"""
ZoneSignal - Monte Carlo Simulation Engine
Simulations de trajectoires futures avec processus stochastiques avancés
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings('ignore')


@dataclass
class MonteCarloPath:
    """Une trajectoire simulée"""
    prices: np.ndarray
    returns: np.ndarray
    max_drawdown: float
    final_return: float
    volatility_realized: float
    sharpe_ratio: float


@dataclass 
class MonteCarloResult:
    """Résultat complet des simulations Monte Carlo"""
    # Statistiques sur les trajectoires
    n_simulations: int
    horizon_days: int
    
    # Distribution des returns finaux
    return_mean: float
    return_median: float
    return_std: float
    return_skewness: float
    return_kurtosis: float
    
    # Percentiles des returns
    return_percentiles: Dict[int, float]  # {5: -0.12, 10: -0.08, ..., 95: 0.15}
    
    # Probabilités
    prob_positive: float
    prob_loss_5pct: float
    prob_loss_10pct: float
    prob_loss_20pct: float
    prob_gain_5pct: float
    prob_gain_10pct: float
    prob_gain_20pct: float
    
    # Drawdown analysis
    avg_max_drawdown: float
    median_max_drawdown: float
    worst_drawdown_5pct: float  # 5th percentile (worst)
    
    # Value at Risk
    var_95: float  # 95% VaR
    var_99: float  # 99% VaR
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    
    # Trajectoires représentatives
    median_path: np.ndarray
    upper_bound_90: np.ndarray
    lower_bound_90: np.ndarray
    upper_bound_95: np.ndarray
    lower_bound_95: np.ndarray
    worst_case_path: np.ndarray
    best_case_path: np.ndarray
    
    # Probabilistic cone data
    cone_percentiles: Dict[int, np.ndarray]
    
    # Métriques de qualité
    convergence_metric: float
    simulation_time_ms: int


class MonteCarloEngine:
    """
    Moteur de simulation Monte Carlo institutionnel
    
    Caractéristiques:
    - Geometric Brownian Motion (GBM) avec volatilité stochastique
    - Jump-diffusion (Merton)
    - Régime switching
    - Corrélations cross-assets
    - Stress scenarios
    - Parallélisation
    """
    
    def __init__(self, n_simulations: int = None):
        from app.core.config import settings
        self.n_simulations = n_simulations or settings.MONTE_CARLO_SIMULATIONS
        self.random_state = np.random.RandomState(42)
    
    # ═══════════════════════════════════════════════════════════════
    # PROCESSUS STOCHASTIQUES
    # ═══════════════════════════════════════════════════════════════
    
    def _geometric_brownian_motion(
        self,
        S0: float,
        mu: float,
        sigma: float,
        T: int,
        n_paths: int
    ) -> np.ndarray:
        """
        Geometric Brownian Motion standard
        
        dS = μS dt + σS dW
        
        Args:
            S0: Prix initial
            mu: Drift (return attendu annualisé)
            sigma: Volatilité annualisée
            T: Horizon en jours
            n_paths: Nombre de trajectoires
            
        Returns:
            Array (n_paths, T+1) des prix simulés
        """
        
        dt = 1 / 252  # Pas journalier
        
        # Génération des chocs
        Z = self.random_state.standard_normal((n_paths, T))
        
        # Log-returns
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        
        # Cumul pour obtenir les prix
        log_prices = np.zeros((n_paths, T + 1))
        log_prices[:, 0] = np.log(S0)
        log_prices[:, 1:] = np.log(S0) + np.cumsum(log_returns, axis=1)
        
        return np.exp(log_prices)
    
    def _jump_diffusion(
        self,
        S0: float,
        mu: float,
        sigma: float,
        T: int,
        n_paths: int,
        lambda_jump: float = 0.1,  # Intensité des sauts (par an)
        mu_jump: float = -0.05,    # Taille moyenne des sauts
        sigma_jump: float = 0.10   # Volatilité des sauts
    ) -> np.ndarray:
        """
        Modèle de Merton Jump-Diffusion
        
        dS = μS dt + σS dW + S dJ
        
        Permet de capturer les événements extrêmes (fat tails)
        """
        
        dt = 1 / 252
        
        # GBM de base
        Z = self.random_state.standard_normal((n_paths, T))
        
        # Processus de saut (Poisson)
        n_jumps = self.random_state.poisson(lambda_jump * dt, (n_paths, T))
        jump_sizes = self.random_state.normal(mu_jump, sigma_jump, (n_paths, T))
        jumps = n_jumps * jump_sizes
        
        # Ajustement du drift pour compenser les sauts
        drift_adjustment = lambda_jump * (np.exp(mu_jump + 0.5 * sigma_jump**2) - 1)
        adjusted_mu = mu - drift_adjustment
        
        # Log-returns avec sauts
        log_returns = (
            (adjusted_mu - 0.5 * sigma**2) * dt + 
            sigma * np.sqrt(dt) * Z + 
            jumps
        )
        
        # Cumul
        log_prices = np.zeros((n_paths, T + 1))
        log_prices[:, 0] = np.log(S0)
        log_prices[:, 1:] = np.log(S0) + np.cumsum(log_returns, axis=1)
        
        return np.exp(log_prices)
    
    def _stochastic_volatility_heston(
        self,
        S0: float,
        mu: float,
        v0: float,          # Variance initiale
        T: int,
        n_paths: int,
        kappa: float = 2.0,  # Vitesse de retour à la moyenne
        theta: float = 0.04, # Variance long terme
        xi: float = 0.3,     # Vol of vol
        rho: float = -0.7    # Corrélation prix-vol
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Modèle de Heston (volatilité stochastique)
        
        dS = μS dt + √v S dW1
        dv = κ(θ - v) dt + ξ√v dW2
        corr(dW1, dW2) = ρ
        """
        
        dt = 1 / 252
        
        # Génération des chocs corrélés
        Z1 = self.random_state.standard_normal((n_paths, T))
        Z2 = self.random_state.standard_normal((n_paths, T))
        W1 = Z1
        W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2
        
        # Initialisation
        prices = np.zeros((n_paths, T + 1))
        variances = np.zeros((n_paths, T + 1))
        prices[:, 0] = S0
        variances[:, 0] = v0
        
        # Simulation
        for t in range(T):
            v_t = np.maximum(variances[:, t], 0)  # Variance positive
            sqrt_v = np.sqrt(v_t)
            
            # Update variance (CIR process)
            variances[:, t + 1] = (
                v_t + 
                kappa * (theta - v_t) * dt + 
                xi * sqrt_v * np.sqrt(dt) * W2[:, t]
            )
            variances[:, t + 1] = np.maximum(variances[:, t + 1], 0)
            
            # Update price
            prices[:, t + 1] = prices[:, t] * np.exp(
                (mu - 0.5 * v_t) * dt + 
                sqrt_v * np.sqrt(dt) * W1[:, t]
            )
        
        return prices, variances
    
    def _regime_switching_gbm(
        self,
        S0: float,
        T: int,
        n_paths: int,
        regime_params: Dict[str, Dict[str, float]],
        transition_matrix: np.ndarray,
        initial_regime: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        GBM avec changement de régime (Markov switching)
        
        Permet de simuler des environnements de marché changeants
        """
        
        dt = 1 / 252
        n_regimes = len(regime_params)
        regime_list = list(regime_params.keys())
        
        # Initialisation
        prices = np.zeros((n_paths, T + 1))
        regimes = np.zeros((n_paths, T + 1), dtype=int)
        prices[:, 0] = S0
        regimes[:, 0] = initial_regime
        
        for t in range(T):
            # Déterminer le régime pour chaque path
            for p in range(n_paths):
                current_regime = regimes[p, t]
                # Transition probabiliste
                new_regime = self.random_state.choice(
                    n_regimes,
                    p=transition_matrix[current_regime]
                )
                regimes[p, t + 1] = new_regime
                
                # Paramètres du régime
                regime_name = regime_list[new_regime]
                mu = regime_params[regime_name]["mu"]
                sigma = regime_params[regime_name]["sigma"]
                
                # Evolution du prix
                Z = self.random_state.standard_normal()
                prices[p, t + 1] = prices[p, t] * np.exp(
                    (mu - 0.5 * sigma**2) * dt + 
                    sigma * np.sqrt(dt) * Z
                )
        
        return prices, regimes
    
    # ═══════════════════════════════════════════════════════════════
    # SIMULATION PRINCIPALE
    # ═══════════════════════════════════════════════════════════════
    
    def simulate(
        self,
        df: pd.DataFrame,
        horizon_days: int,
        model: str = "jump_diffusion",
        regime_info: Optional[Dict] = None,
        stress_multiplier: float = 1.0
    ) -> MonteCarloResult:
        """
        Exécute les simulations Monte Carlo
        
        Args:
            df: Données historiques (avec 'close', 'daily_return')
            horizon_days: Horizon de simulation
            model: Type de modèle ('gbm', 'jump_diffusion', 'heston', 'regime_switching')
            regime_info: Informations sur le régime actuel
            stress_multiplier: Multiplicateur de stress (1.0 = normal, 1.5 = stress)
            
        Returns:
            MonteCarloResult complet
        """
        
        import time
        start_time = time.time()
        
        # Extraire les paramètres des données historiques
        S0 = df['close'].iloc[-1]
        returns = df['daily_return'].dropna().values
        
        # Estimation des paramètres
        mu = np.mean(returns) * 252  # Annualisé
        sigma = np.std(returns) * np.sqrt(252) * stress_multiplier
        
        # Exécuter les simulations selon le modèle choisi
        if model == "gbm":
            paths = self._geometric_brownian_motion(
                S0, mu, sigma, horizon_days, self.n_simulations
            )
        elif model == "jump_diffusion":
            # Calibrer les paramètres de saut
            lambda_jump = self._estimate_jump_intensity(returns)
            paths = self._jump_diffusion(
                S0, mu, sigma, horizon_days, self.n_simulations,
                lambda_jump=lambda_jump * stress_multiplier
            )
        elif model == "heston":
            v0 = sigma**2
            paths, _ = self._stochastic_volatility_heston(
                S0, mu, v0, horizon_days, self.n_simulations
            )
        elif model == "regime_switching" and regime_info:
            regime_params = regime_info.get("params", {
                "bull": {"mu": 0.15, "sigma": 0.12},
                "bear": {"mu": -0.10, "sigma": 0.25},
                "neutral": {"mu": 0.05, "sigma": 0.16}
            })
            transition_matrix = regime_info.get("transition_matrix", 
                np.array([[0.95, 0.03, 0.02],
                          [0.05, 0.90, 0.05],
                          [0.03, 0.02, 0.95]])
            )
            paths, _ = self._regime_switching_gbm(
                S0, horizon_days, self.n_simulations,
                regime_params, transition_matrix
            )
        else:
            # Default: jump diffusion
            paths = self._jump_diffusion(
                S0, mu, sigma, horizon_days, self.n_simulations
            )
        
        # Analyser les résultats
        result = self._analyze_paths(paths, horizon_days, start_time)
        
        return result
    
    def _estimate_jump_intensity(self, returns: np.ndarray) -> float:
        """Estime l'intensité des sauts à partir des données historiques"""
        
        # Détecter les returns extrêmes (> 3 sigma)
        sigma = np.std(returns)
        extreme_returns = np.abs(returns) > 3 * sigma
        
        # Fréquence annualisée
        jump_frequency = np.sum(extreme_returns) / len(returns) * 252
        
        return max(0.05, min(0.5, jump_frequency))  # Borné entre 0.05 et 0.5
    
    def _analyze_paths(
        self,
        paths: np.ndarray,
        horizon_days: int,
        start_time: float
    ) -> MonteCarloResult:
        """Analyse les trajectoires simulées et calcule les statistiques"""
        
        import time
        
        n_paths = paths.shape[0]
        S0 = paths[0, 0]
        
        # Returns finaux
        final_prices = paths[:, -1]
        final_returns = (final_prices - S0) / S0
        
        # Statistiques des returns
        return_mean = np.mean(final_returns)
        return_median = np.median(final_returns)
        return_std = np.std(final_returns)
        return_skew = stats.skew(final_returns)
        return_kurt = stats.kurtosis(final_returns)
        
        # Percentiles
        percentile_levels = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        return_percentiles = {
            p: float(np.percentile(final_returns, p))
            for p in percentile_levels
        }
        
        # Probabilités
        prob_positive = float(np.mean(final_returns > 0))
        prob_loss_5 = float(np.mean(final_returns < -0.05))
        prob_loss_10 = float(np.mean(final_returns < -0.10))
        prob_loss_20 = float(np.mean(final_returns < -0.20))
        prob_gain_5 = float(np.mean(final_returns > 0.05))
        prob_gain_10 = float(np.mean(final_returns > 0.10))
        prob_gain_20 = float(np.mean(final_returns > 0.20))
        
        # Drawdown analysis
        max_drawdowns = []
        for path in paths:
            running_max = np.maximum.accumulate(path)
            drawdowns = (path - running_max) / running_max
            max_drawdowns.append(np.min(drawdowns))
        
        max_drawdowns = np.array(max_drawdowns)
        avg_max_dd = float(np.mean(max_drawdowns))
        median_max_dd = float(np.median(max_drawdowns))
        worst_dd_5pct = float(np.percentile(max_drawdowns, 5))
        
        # Value at Risk
        var_95 = float(-np.percentile(final_returns, 5))
        var_99 = float(-np.percentile(final_returns, 1))
        
        # Conditional VaR (Expected Shortfall)
        cvar_95 = float(-np.mean(final_returns[final_returns < -var_95]))
        
        # Trajectoires représentatives
        median_idx = np.argsort(final_returns)[n_paths // 2]
        worst_idx = np.argmin(final_returns)
        best_idx = np.argmax(final_returns)
        
        # Calculer les bornes du cone probabiliste
        cone_percentiles = {}
        for p in [5, 10, 25, 50, 75, 90, 95]:
            cone_percentiles[p] = np.percentile(paths, p, axis=0)
        
        # Convergence metric
        # Comparer mean des 2 moitiés
        half = n_paths // 2
        mean_1 = np.mean(final_returns[:half])
        mean_2 = np.mean(final_returns[half:])
        convergence = 1 - abs(mean_1 - mean_2) / (abs(return_mean) + 1e-10)
        convergence = float(np.clip(convergence, 0, 1))
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        return MonteCarloResult(
            n_simulations=n_paths,
            horizon_days=horizon_days,
            return_mean=return_mean,
            return_median=return_median,
            return_std=return_std,
            return_skewness=return_skew,
            return_kurtosis=return_kurt,
            return_percentiles=return_percentiles,
            prob_positive=prob_positive,
            prob_loss_5pct=prob_loss_5,
            prob_loss_10pct=prob_loss_10,
            prob_loss_20pct=prob_loss_20,
            prob_gain_5pct=prob_gain_5,
            prob_gain_10pct=prob_gain_10,
            prob_gain_20pct=prob_gain_20,
            avg_max_drawdown=avg_max_dd,
            median_max_drawdown=median_max_dd,
            worst_drawdown_5pct=worst_dd_5pct,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            median_path=paths[median_idx],
            upper_bound_90=cone_percentiles[95],
            lower_bound_90=cone_percentiles[5],
            upper_bound_95=np.percentile(paths, 97.5, axis=0),
            lower_bound_95=np.percentile(paths, 2.5, axis=0),
            worst_case_path=paths[worst_idx],
            best_case_path=paths[best_idx],
            cone_percentiles=cone_percentiles,
            convergence_metric=convergence,
            simulation_time_ms=elapsed_ms
        )
    
    # ═══════════════════════════════════════════════════════════════
    # STRESS TESTING
    # ═══════════════════════════════════════════════════════════════
    
    def run_stress_scenarios(
        self,
        df: pd.DataFrame,
        horizon_days: int,
        scenarios: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, MonteCarloResult]:
        """
        Exécute des simulations sous différents scénarios de stress
        """
        
        if scenarios is None:
            scenarios = {
                "base": {"stress_multiplier": 1.0, "model": "jump_diffusion"},
                "mild_stress": {"stress_multiplier": 1.25, "model": "jump_diffusion"},
                "severe_stress": {"stress_multiplier": 1.75, "model": "jump_diffusion"},
                "2008_crisis": {"stress_multiplier": 2.5, "model": "heston"},
                "covid_crash": {"stress_multiplier": 3.0, "model": "jump_diffusion"}
            }
        
        results = {}
        
        for name, params in scenarios.items():
            results[name] = self.simulate(
                df,
                horizon_days,
                model=params.get("model", "jump_diffusion"),
                stress_multiplier=params.get("stress_multiplier", 1.0)
            )
        
        return results

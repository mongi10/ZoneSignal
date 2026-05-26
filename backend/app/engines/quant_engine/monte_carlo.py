"""
Monte Carlo Simulation Engine for Probabilistic Forecasting
Generates thousands of future scenarios with proper uncertainty quantification.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings('ignore')


@dataclass
class MonteCarloResult:
    # Distribution statistics
    mean_return: float
    median_return: float
    std_return: float
    skewness: float
    kurtosis: float
    
    # Percentiles
    percentiles: Dict[int, float]
    
    # Probabilities
    prob_positive: float
    prob_negative: float
    prob_above_threshold: Dict[float, float]
    prob_below_threshold: Dict[float, float]
    
    # Risk metrics
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown_expected: float
    
    # Scenarios
    scenarios: np.ndarray
    paths: Optional[np.ndarray]


@dataclass
class StressScenario:
    name: str
    probability: float
    expected_return: float
    worst_case_return: float
    path: Optional[np.ndarray]


class MonteCarloEngine:
    """
    Advanced Monte Carlo simulation engine with:
    - Multiple distribution models (Normal, Student-t, Jump-Diffusion)
    - Regime-dependent parameters
    - Correlation structure
    - Path generation for multi-period analysis
    """
    
    def __init__(
        self,
        n_simulations: int = 10000,
        random_state: int = 42
    ):
        self.n_simulations = n_simulations
        self.rng = np.random.RandomState(random_state)
        
    def simulate_returns(
        self,
        mu: float,
        sigma: float,
        horizon_days: int,
        distribution: str = "student_t",
        df: float = 5.0,
        jump_intensity: float = 0.0,
        jump_mean: float = 0.0,
        jump_std: float = 0.02,
        regime_params: Optional[Dict] = None
    ) -> MonteCarloResult:
        """
        Simulate future returns using specified distribution.
        
        Parameters:
        -----------
        mu : float
            Expected daily return
        sigma : float
            Daily volatility
        horizon_days : int
            Number of days to simulate
        distribution : str
            'normal', 'student_t', 'jump_diffusion'
        df : float
            Degrees of freedom for Student-t
        jump_intensity : float
            Poisson intensity for jumps (jumps per day)
        jump_mean : float
            Mean jump size
        jump_std : float
            Jump size standard deviation
        regime_params : dict
            Regime-specific parameter adjustments
        """
        # Apply regime adjustments
        if regime_params:
            mu = mu * regime_params.get('mu_mult', 1.0) + regime_params.get('mu_add', 0.0)
            sigma = sigma * regime_params.get('sigma_mult', 1.0)
            
        # Generate daily returns
        if distribution == "normal":
            daily_returns = self.rng.normal(mu, sigma, (self.n_simulations, horizon_days))
            
        elif distribution == "student_t":
            # Scale t-distribution to have correct variance
            scale = sigma * np.sqrt((df - 2) / df) if df > 2 else sigma
            daily_returns = stats.t.rvs(df, loc=mu, scale=scale, 
                                        size=(self.n_simulations, horizon_days),
                                        random_state=self.rng)
            
        elif distribution == "jump_diffusion":
            # Merton jump-diffusion model
            # Continuous part
            continuous = self.rng.normal(mu, sigma, (self.n_simulations, horizon_days))
            
            # Jump part
            n_jumps = self.rng.poisson(jump_intensity, (self.n_simulations, horizon_days))
            jump_sizes = self.rng.normal(jump_mean, jump_std, (self.n_simulations, horizon_days))
            jumps = n_jumps * jump_sizes
            
            daily_returns = continuous + jumps
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
            
        # Compute cumulative returns
        cumulative_returns = np.sum(daily_returns, axis=1)
        
        # Generate paths for analysis
        paths = np.cumprod(1 + daily_returns, axis=1)
        
        # Compute statistics
        result = self._compute_statistics(cumulative_returns, paths)
        
        return result
    
    def _compute_statistics(
        self,
        cumulative_returns: np.ndarray,
        paths: np.ndarray
    ) -> MonteCarloResult:
        """
        Compute comprehensive statistics from simulation results.
        """
        # Basic statistics
        mean_ret = np.mean(cumulative_returns)
        median_ret = np.median(cumulative_returns)
        std_ret = np.std(cumulative_returns)
        skew = stats.skew(cumulative_returns)
        kurt = stats.kurtosis(cumulative_returns)
        
        # Percentiles
        percentile_points = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentiles = {p: float(np.percentile(cumulative_returns, p)) for p in percentile_points}
        
        # Probabilities
        prob_positive = float(np.mean(cumulative_returns > 0))
        prob_negative = float(np.mean(cumulative_returns < 0))
        
        # Threshold probabilities
        thresholds = [0.01, 0.02, 0.05, 0.10, 0.20]
        prob_above = {t: float(np.mean(cumulative_returns > t)) for t in thresholds}
        prob_below = {-t: float(np.mean(cumulative_returns < -t)) for t in thresholds}
        
        # Risk metrics
        var_95 = float(np.percentile(cumulative_returns, 5))
        var_99 = float(np.percentile(cumulative_returns, 1))
        cvar_95 = float(np.mean(cumulative_returns[cumulative_returns <= var_95]))
        cvar_99 = float(np.mean(cumulative_returns[cumulative_returns <= var_99]))
        
        # Max drawdown from paths
        running_max = np.maximum.accumulate(paths, axis=1)
        drawdowns = (paths - running_max) / running_max
        max_drawdowns = np.min(drawdowns, axis=1)
        max_drawdown_expected = float(np.mean(max_drawdowns))
        
        return MonteCarloResult(
            mean_return=float(mean_ret),
            median_return=float(median_ret),
            std_return=float(std_ret),
            skewness=float(skew),
            kurtosis=float(kurt),
            percentiles=percentiles,
            prob_positive=prob_positive,
            prob_negative=prob_negative,
            prob_above_threshold=prob_above,
            prob_below_threshold=prob_below,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown_expected=max_drawdown_expected,
            scenarios=cumulative_returns,
            paths=paths
        )
    
    def generate_stress_scenarios(
        self,
        base_mu: float,
        base_sigma: float,
        horizon_days: int,
        current_regime: str
    ) -> List[StressScenario]:
        """
        Generate specific stress scenarios based on historical analogs.
        """
        scenarios = []
        
        # 1. Flash Crash Scenario
        flash_crash = self.simulate_returns(
            mu=-0.02,
            sigma=base_sigma * 3,
            horizon_days=min(5, horizon_days),
            distribution="jump_diffusion",
            jump_intensity=0.3,
            jump_mean=-0.03,
            jump_std=0.02
        )
        scenarios.append(StressScenario(
            name="Flash Crash",
            probability=0.02,
            expected_return=flash_crash.mean_return,
            worst_case_return=flash_crash.var_99,
            path=None
        ))
        
        # 2. 2008-style Crisis
        crisis = self.simulate_returns(
            mu=-0.005,
            sigma=base_sigma * 2.5,
            horizon_days=horizon_days,
            distribution="student_t",
            df=3.0
        )
        scenarios.append(StressScenario(
            name="Financial Crisis",
            probability=0.01,
            expected_return=crisis.mean_return,
            worst_case_return=crisis.var_99,
            path=None
        ))
        
        # 3. V-shaped Recovery
        recovery = self.simulate_returns(
            mu=0.003,
            sigma=base_sigma * 1.5,
            horizon_days=horizon_days,
            distribution="normal"
        )
        scenarios.append(StressScenario(
            name="V-Recovery",
            probability=0.15,
            expected_return=recovery.mean_return,
            worst_case_return=recovery.percentiles[25],
            path=None
        ))
        
        # 4. Stagflation
        stagflation = self.simulate_returns(
            mu=-0.001,
            sigma=base_sigma * 1.2,
            horizon_days=horizon_days,
            distribution="normal"
        )
        scenarios.append(StressScenario(
            name="Stagflation",
            probability=0.05,
            expected_return=stagflation.mean_return,
            worst_case_return=stagflation.var_95,
            path=None
        ))
        
        # 5. Liquidity Crisis
        liquidity = self.simulate_returns(
            mu=-0.008,
            sigma=base_sigma * 2,
            horizon_days=min(20, horizon_days),
            distribution="jump_diffusion",
            jump_intensity=0.2,
            jump_mean=-0.02,
            jump_std=0.015
        )
        scenarios.append(StressScenario(
            name="Liquidity Crisis",
            probability=0.03,
            expected_return=liquidity.mean_return,
            worst_case_return=liquidity.var_99,
            path=None
        ))
        
        return scenarios
    
    def simulate_correlated_assets(
        self,
        mus: np.ndarray,
        sigmas: np.ndarray,
        correlation_matrix: np.ndarray,
        horizon_days: int
    ) -> Dict[int, MonteCarloResult]:
        """
        Simulate multiple correlated assets using Cholesky decomposition.
        """
        n_assets = len(mus)
        
        # Cholesky decomposition
        L = np.linalg.cholesky(correlation_matrix)
        
        # Generate correlated standard normals
        Z = self.rng.standard_normal((self.n_simulations, horizon_days, n_assets))
        
        # Apply correlation
        correlated_Z = np.einsum('ijk,lk->ijl', Z, L)
        
        # Scale to returns
        daily_returns = mus + sigmas * correlated_Z
        
        results = {}
        for i in range(n_assets):
            cumulative = np.sum(daily_returns[:, :, i], axis=1)
            paths = np.cumprod(1 + daily_returns[:, :, i], axis=1)
            results[i] = self._compute_statistics(cumulative, paths)
            
        return results

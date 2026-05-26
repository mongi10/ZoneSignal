"""
Stochastic Volatility Models: GARCH, EGARCH, and variants
For volatility forecasting and uncertainty quantification.
"""

import numpy as np
from scipy import optimize, stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from arch import arch_model
import warnings

warnings.filterwarnings('ignore')


@dataclass
class VolatilityForecast:
    current_volatility: float
    forecast_volatility: np.ndarray
    forecast_horizon: int
    confidence_intervals: Dict[str, Tuple[float, float]]
    volatility_regime: str
    persistence: float
    half_life: float


@dataclass
class GARCHParameters:
    omega: float
    alpha: float
    beta: float
    gamma: Optional[float]  # For EGARCH/GJR
    persistence: float
    long_run_variance: float


class VolatilityEngine:
    """
    Comprehensive volatility modeling with multiple model types.
    """
    
    def __init__(self, model_type: str = "GARCH"):
        self.model_type = model_type
        self.fitted_model = None
        self.parameters: Optional[GARCHParameters] = None
        self.returns: Optional[np.ndarray] = None
        
    def fit(
        self,
        returns: np.ndarray,
        model_type: str = "GARCH",
        p: int = 1,
        q: int = 1,
        dist: str = "t"
    ) -> "VolatilityEngine":
        """
        Fit volatility model to return series.
        
        Parameters:
        -----------
        returns : np.ndarray
            Return series (in percentage points for arch library)
        model_type : str
            'GARCH', 'EGARCH', 'GJR', 'FIGARCH'
        p : int
            ARCH order
        q : int
            GARCH order
        dist : str
            Distribution: 'normal', 't', 'skewt', 'ged'
        """
        self.returns = returns * 100  # Convert to percentage for arch
        self.model_type = model_type
        
        vol_type = model_type.lower()
        if vol_type == "garch":
            model = arch_model(self.returns, vol=vol_type, p=p, q=q, dist=dist)
        elif vol_type == "egarch":
            model = arch_model(self.returns, vol='EGARCH', p=p, o=1, q=q, dist=dist)
        elif vol_type == "gjr":
            model = arch_model(self.returns, vol='GARCH', p=p, o=1, q=q, dist=dist)
        elif vol_type == "figarch":
            model = arch_model(self.returns, vol='FIGARCH', p=1, q=1, dist=dist)
        else:
            model = arch_model(self.returns, vol='GARCH', p=p, q=q, dist=dist)
            
        self.fitted_model = model.fit(disp='off')
        self._extract_parameters()
        
        return self
    
    def _extract_parameters(self) -> None:
        """
        Extract model parameters into standardized format.
        """
        params = self.fitted_model.params
        
        if self.model_type.upper() == "GARCH":
            omega = params.get('omega', 0)
            alpha = params.get('alpha[1]', 0)
            beta = params.get('beta[1]', 0)
            gamma = None
            persistence = alpha + beta
        elif self.model_type.upper() == "EGARCH":
            omega = params.get('omega', 0)
            alpha = params.get('alpha[1]', 0)
            beta = params.get('beta[1]', 0)
            gamma = params.get('gamma[1]', 0)
            persistence = beta
        elif self.model_type.upper() == "GJR":
            omega = params.get('omega', 0)
            alpha = params.get('alpha[1]', 0)
            beta = params.get('beta[1]', 0)
            gamma = params.get('gamma[1]', 0)
            persistence = alpha + beta + 0.5 * gamma
        else:
            omega = alpha = beta = gamma = persistence = 0
            
        long_run_var = omega / (1 - persistence) if persistence < 1 else np.inf
        
        self.parameters = GARCHParameters(
            omega=float(omega),
            alpha=float(alpha),
            beta=float(beta),
            gamma=float(gamma) if gamma else None,
            persistence=float(persistence),
            long_run_variance=float(long_run_var)
        )
    
    def forecast(self, horizon: int = 22) -> VolatilityForecast:
        """
        Forecast volatility over specified horizon.
        
        Parameters:
        -----------
        horizon : int
            Forecast horizon in days
        
        Returns:
        --------
        VolatilityForecast
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
            
        # Get forecasts
        forecasts = self.fitted_model.forecast(horizon=horizon)
        variance_forecast = forecasts.variance.iloc[-1].values / 10000  # Convert back
        volatility_forecast = np.sqrt(variance_forecast)
        
        # Current conditional volatility
        current_var = self.fitted_model.conditional_volatility.iloc[-1] / 100
        
        # Confidence intervals using simulation
        sim = self.fitted_model.forecast(horizon=horizon, method='simulation', simulations=1000)
        var_5 = np.sqrt(np.percentile(sim.variance.values[-1], 5, axis=0) / 10000)
        var_95 = np.sqrt(np.percentile(sim.variance.values[-1], 95, axis=0) / 10000)
        
        # Volatility regime classification
        long_run_vol = np.sqrt(self.parameters.long_run_variance) / 100
        vol_ratio = current_var / long_run_vol if long_run_vol > 0 else 1
        
        if vol_ratio < 0.7:
            regime = "low"
        elif vol_ratio < 1.0:
            regime = "medium"
        elif vol_ratio < 1.5:
            regime = "high"
        else:
            regime = "extreme"
            
        # Half-life of volatility shocks
        if self.parameters.persistence < 1:
            half_life = np.log(0.5) / np.log(self.parameters.persistence)
        else:
            half_life = np.inf
            
        return VolatilityForecast(
            current_volatility=float(current_var),
            forecast_volatility=volatility_forecast,
            forecast_horizon=horizon,
            confidence_intervals={
                "5%": (float(var_5.mean()), float(var_95.mean()))
            },
            volatility_regime=regime,
            persistence=self.parameters.persistence,
            half_life=float(half_life)
        )
    
    def compute_realized_volatility(
        self,
        returns: np.ndarray,
        window: int = 20,
        method: str = "standard"
    ) -> np.ndarray:
        """
        Compute realized volatility using various estimators.
        
        Parameters:
        -----------
        returns : np.ndarray
            Return series
        window : int
            Rolling window size
        method : str
            'standard', 'parkinson', 'garman_klass', 'yang_zhang'
        """
        if method == "standard":
            # Standard deviation
            vol = np.array([
                np.std(returns[max(0, i-window):i]) * np.sqrt(252)
                for i in range(1, len(returns) + 1)
            ])
        elif method == "exponential":
            # EWMA
            alpha = 2 / (window + 1)
            vol = np.zeros(len(returns))
            vol[0] = np.abs(returns[0])
            for i in range(1, len(returns)):
                vol[i] = np.sqrt(alpha * returns[i]**2 + (1 - alpha) * vol[i-1]**2)
            vol = vol * np.sqrt(252)
        else:
            vol = np.std(returns) * np.sqrt(252) * np.ones(len(returns))
            
        return vol


class RealizedVolatilityEstimator:
    """
    High-frequency realized volatility estimators.
    """
    
    @staticmethod
    def parkinson(high: np.ndarray, low: np.ndarray) -> float:
        """
        Parkinson volatility estimator using high-low range.
        """
        log_hl = np.log(high / low)
        return np.sqrt(np.mean(log_hl ** 2) / (4 * np.log(2))) * np.sqrt(252)
    
    @staticmethod
    def garman_klass(
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> float:
        """
        Garman-Klass volatility estimator.
        """
        log_hl = np.log(high / low)
        log_co = np.log(close / open_)
        
        term1 = 0.5 * log_hl ** 2
        term2 = (2 * np.log(2) - 1) * log_co ** 2
        
        return np.sqrt(np.mean(term1 - term2)) * np.sqrt(252)
    
    @staticmethod
    def yang_zhang(
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        window: int = 20
    ) -> float:
        """
        Yang-Zhang volatility estimator (handles overnight jumps).
        """
        n = len(close)
        
        # Overnight volatility
        log_oc = np.log(open_[1:] / close[:-1])
        overnight_var = np.var(log_oc, ddof=1)
        
        # Open-to-close volatility
        log_co = np.log(close / open_)
        open_close_var = np.var(log_co, ddof=1)
        
        # Rogers-Satchell component
        log_ho = np.log(high / open_)
        log_lo = np.log(low / open_)
        log_hc = np.log(high / close)
        log_lc = np.log(low / close)
        rs_var = np.mean(log_ho * log_hc + log_lo * log_lc)
        
        # Combine
        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        variance = overnight_var + k * open_close_var + (1 - k) * rs_var
        
        return np.sqrt(variance * 252)

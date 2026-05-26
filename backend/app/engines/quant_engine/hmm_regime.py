"""
Hidden Markov Model for Regime Detection
Detects latent market regimes: Risk-On, Risk-Off, Euphoria, Panic, Consolidation, etc.
"""

import numpy as np
from hmmlearn import hmm
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


class MarketRegime(str, Enum):
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    EUPHORIA = "euphoria"
    PANIC = "panic"
    CONSOLIDATION = "consolidation"
    MACRO_STRESS = "macro_stress"
    REFLATION = "reflation"
    DISINFLATION = "disinflation"
    LIQUIDITY_EXPANSION = "liquidity_expansion"
    LIQUIDITY_CONTRACTION = "liquidity_contraction"


@dataclass
class RegimeDetectionResult:
    current_regime: MarketRegime
    regime_probability: float
    regime_stability: float
    transition_probabilities: Dict[str, float]
    regime_history: List[MarketRegime]
    hidden_states: np.ndarray
    state_probabilities: np.ndarray


class HMMRegimeDetector:
    """
    Multi-state Hidden Markov Model for market regime detection.
    Uses returns, volatility, and cross-asset signals as observations.
    """
    
    def __init__(
        self,
        n_regimes: int = 6,
        n_iter: int = 100,
        covariance_type: str = "full",
        random_state: int = 42
    ):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model: Optional[hmm.GaussianHMM] = None
        self.regime_mapping: Dict[int, MarketRegime] = {}
        self.is_fitted = False
        
    def _prepare_observations(
        self,
        returns: np.ndarray,
        volatility: np.ndarray,
        volume_ratio: Optional[np.ndarray] = None,
        vix_level: Optional[np.ndarray] = None,
        credit_spread: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Prepare multi-dimensional observation matrix.
        Each row is [return, volatility, volume_ratio, vix, credit_spread]
        """
        features = [returns.reshape(-1, 1), volatility.reshape(-1, 1)]
        
        if volume_ratio is not None:
            features.append(volume_ratio.reshape(-1, 1))
        if vix_level is not None:
            features.append(vix_level.reshape(-1, 1))
        if credit_spread is not None:
            features.append(credit_spread.reshape(-1, 1))
            
        observations = np.hstack(features)
        
        # Handle NaN values
        mask = ~np.isnan(observations).any(axis=1)
        return observations[mask], mask
    
    def _classify_regimes(self, means: np.ndarray, covars: np.ndarray) -> Dict[int, MarketRegime]:
        """
        Map HMM states to semantic regime labels based on state characteristics.
        """
        regime_map = {}
        
        # Extract characteristics
        return_means = means[:, 0]
        vol_means = means[:, 1] if means.shape[1] > 1 else np.zeros(self.n_regimes)
        
        # Sort by return mean
        sorted_indices = np.argsort(return_means)
        
        # Assign regimes based on return/volatility characteristics
        for i, state_idx in enumerate(sorted_indices):
            ret = return_means[state_idx]
            vol = vol_means[state_idx]
            
            if ret < -0.002 and vol > np.percentile(vol_means, 75):
                regime_map[state_idx] = MarketRegime.PANIC
            elif ret < -0.001 and vol > np.percentile(vol_means, 50):
                regime_map[state_idx] = MarketRegime.RISK_OFF
            elif ret < 0 and vol < np.percentile(vol_means, 50):
                regime_map[state_idx] = MarketRegime.MACRO_STRESS
            elif abs(ret) < 0.0005 and vol < np.percentile(vol_means, 40):
                regime_map[state_idx] = MarketRegime.CONSOLIDATION
            elif ret > 0.001 and vol > np.percentile(vol_means, 60):
                regime_map[state_idx] = MarketRegime.EUPHORIA
            elif ret > 0.0005:
                regime_map[state_idx] = MarketRegime.RISK_ON
            else:
                regime_map[state_idx] = MarketRegime.CONSOLIDATION
                
        return regime_map
    
    def fit(
        self,
        returns: np.ndarray,
        volatility: np.ndarray,
        volume_ratio: Optional[np.ndarray] = None,
        vix_level: Optional[np.ndarray] = None,
        credit_spread: Optional[np.ndarray] = None
    ) -> "HMMRegimeDetector":
        """
        Fit the HMM model on historical data.
        """
        observations, mask = self._prepare_observations(
            returns, volatility, volume_ratio, vix_level, credit_spread
        )
        
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        
        self.model.fit(observations)
        self.regime_mapping = self._classify_regimes(self.model.means_, self.model.covars_)
        self.is_fitted = True
        
        return self
    
    def detect(
        self,
        returns: np.ndarray,
        volatility: np.ndarray,
        volume_ratio: Optional[np.ndarray] = None,
        vix_level: Optional[np.ndarray] = None,
        credit_spread: Optional[np.ndarray] = None,
        lookback: int = 60
    ) -> RegimeDetectionResult:
        """
        Detect current regime and compute transition probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before detection")
            
        observations, mask = self._prepare_observations(
            returns, volatility, volume_ratio, vix_level, credit_spread
        )
        
        # Use recent data for detection
        recent_obs = observations[-lookback:] if len(observations) > lookback else observations
        
        # Predict hidden states
        hidden_states = self.model.predict(recent_obs)
        state_probs = self.model.predict_proba(recent_obs)
        
        # Current state
        current_state = hidden_states[-1]
        current_regime = self.regime_mapping.get(current_state, MarketRegime.CONSOLIDATION)
        current_prob = state_probs[-1, current_state]
        
        # Regime stability (how long in current state)
        state_run = 1
        for i in range(len(hidden_states) - 2, -1, -1):
            if hidden_states[i] == current_state:
                state_run += 1
            else:
                break
        stability = min(state_run / 20, 1.0)  # Normalize to [0, 1]
        
        # Transition probabilities for next period
        trans_matrix = self.model.transmat_
        next_state_probs = trans_matrix[current_state]
        transition_probs = {
            self.regime_mapping.get(i, MarketRegime.CONSOLIDATION).value: float(next_state_probs[i])
            for i in range(self.n_regimes)
        }
        
        # Regime history
        regime_history = [
            self.regime_mapping.get(s, MarketRegime.CONSOLIDATION)
            for s in hidden_states
        ]
        
        return RegimeDetectionResult(
            current_regime=current_regime,
            regime_probability=float(current_prob),
            regime_stability=stability,
            transition_probabilities=transition_probs,
            regime_history=regime_history,
            hidden_states=hidden_states,
            state_probabilities=state_probs
        )
    
    def get_regime_statistics(self) -> Dict:
        """
        Return statistics about fitted regimes.
        """
        if not self.is_fitted:
            return {}
            
        stats = {
            "n_regimes": self.n_regimes,
            "regime_characteristics": {}
        }
        
        for state_idx, regime in self.regime_mapping.items():
            stats["regime_characteristics"][regime.value] = {
                "mean_return": float(self.model.means_[state_idx, 0]),
                "mean_volatility": float(self.model.means_[state_idx, 1]) if self.model.means_.shape[1] > 1 else None,
                "stationary_probability": float(self._get_stationary_distribution()[state_idx])
            }
            
        return stats
    
    def _get_stationary_distribution(self) -> np.ndarray:
        """
        Compute stationary distribution of the Markov chain.
        """
        trans = self.model.transmat_
        eigenvalues, eigenvectors = np.linalg.eig(trans.T)
        stationary_idx = np.argmin(np.abs(eigenvalues - 1))
        stationary = np.real(eigenvectors[:, stationary_idx])
        stationary = stationary / stationary.sum()
        return stationary

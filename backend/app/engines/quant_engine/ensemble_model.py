"""
Ensemble Model for Combining Multiple Forecasting Approaches
Implements adaptive weighting based on regime and historical performance.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from scipy import optimize
import warnings

warnings.filterwarnings('ignore')


class ModelType(str, Enum):
    HMM_REGIME = "hmm_regime"
    BAYESIAN = "bayesian"
    MONTE_CARLO = "monte_carlo"
    VOLATILITY = "volatility"
    MACRO = "macro"
    SEASONALITY = "seasonality"
    MOMENTUM = "momentum"
    FACTOR = "factor"


@dataclass
class ModelPrediction:
    model_type: ModelType
    direction_probs: Dict[str, float]  # bullish, bearish, neutral
    expected_return: float
    confidence: float
    regime_affinity: Dict[str, float]  # Performance by regime


@dataclass
class EnsemblePrediction:
    # Aggregated probabilities
    prob_bullish: float
    prob_bearish: float
    prob_neutral: float
    
    # Direction
    direction: str
    intensity: str
    
    # Returns
    expected_return: float
    return_std: float
    return_percentiles: Dict[int, float]
    
    # Confidence
    confidence_score: float
    stability_score: float
    uncertainty_score: float
    signal_robustness: float
    
    # Model contributions
    model_weights: Dict[str, float]
    model_contributions: Dict[str, Dict[str, float]]
    
    # Disagreement metrics
    model_disagreement: float
    prediction_entropy: float


class EnsembleForecaster:
    """
    Adaptive ensemble that combines multiple forecasting models.
    Weights are adjusted based on recent performance and current regime.
    """
    
    def __init__(self):
        # Base weights (will be adapted)
        self.base_weights = {
            ModelType.HMM_REGIME: 0.15,
            ModelType.BAYESIAN: 0.15,
            ModelType.MONTE_CARLO: 0.15,
            ModelType.VOLATILITY: 0.10,
            ModelType.MACRO: 0.15,
            ModelType.SEASONALITY: 0.10,
            ModelType.MOMENTUM: 0.10,
            ModelType.FACTOR: 0.10
        }
        
        # Regime-specific weight adjustments
        self.regime_adjustments = {
            "risk_on": {
                ModelType.MOMENTUM: 1.3,
                ModelType.SEASONALITY: 1.2,
                ModelType.VOLATILITY: 0.8
            },
            "risk_off": {
                ModelType.VOLATILITY: 1.4,
                ModelType.MACRO: 1.3,
                ModelType.MOMENTUM: 0.7
            },
            "panic": {
                ModelType.VOLATILITY: 1.5,
                ModelType.HMM_REGIME: 1.3,
                ModelType.SEASONALITY: 0.5
            },
            "euphoria": {
                ModelType.BAYESIAN: 1.2,
                ModelType.MONTE_CARLO: 1.2,
                ModelType.MOMENTUM: 0.8
            },
            "consolidation": {
                ModelType.SEASONALITY: 1.3,
                ModelType.FACTOR: 1.2,
                ModelType.MOMENTUM: 0.8
            },
            "macro_stress": {
                ModelType.MACRO: 1.5,
                ModelType.VOLATILITY: 1.3,
                ModelType.SEASONALITY: 0.6
            }
        }
        
        # Performance tracking
        self.model_performance: Dict[ModelType, List[float]] = {
            m: [] for m in ModelType
        }
        
        # Current adaptive weights
        self.current_weights = self.base_weights.copy()
        
    def combine_predictions(
        self,
        predictions: List[ModelPrediction],
        current_regime: str,
        use_adaptive_weights: bool = True
    ) -> EnsemblePrediction:
        """
        Combine predictions from multiple models into ensemble forecast.
        """
        # Calculate weights
        if use_adaptive_weights:
            weights = self._calculate_adaptive_weights(predictions, current_regime)
        else:
            weights = {p.model_type: self.base_weights.get(p.model_type, 0.1) 
                      for p in predictions}
            
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Aggregate probabilities
        prob_bullish = 0.0
        prob_bearish = 0.0
        prob_neutral = 0.0
        expected_returns = []
        confidences = []
        
        model_contributions = {}
        
        for pred in predictions:
            w = weights.get(pred.model_type, 0)
            
            prob_bullish += w * pred.direction_probs.get("bullish", 0.33)
            prob_bearish += w * pred.direction_probs.get("bearish", 0.33)
            prob_neutral += w * pred.direction_probs.get("neutral", 0.34)
            
            expected_returns.append((pred.expected_return, w))
            confidences.append((pred.confidence, w))
            
            model_contributions[pred.model_type.value] = {
                "weight": float(w),
                "direction_probs": pred.direction_probs,
                "expected_return": float(pred.expected_return),
                "confidence": float(pred.confidence)
            }
        
        # Weighted expected return
        expected_return = sum(r * w for r, w in expected_returns)
        
        # Return uncertainty from model disagreement
        returns_array = np.array([r for r, w in expected_returns])
        weights_array = np.array([w for r, w in expected_returns])
        return_std = np.sqrt(np.sum(weights_array * (returns_array - expected_return)**2))
        
        # Direction and intensity
        direction = self._classify_direction(prob_bullish, prob_bearish, prob_neutral)
        intensity = self._classify_intensity(
            max(prob_bullish, prob_bearish, prob_neutral),
            abs(expected_return)
        )
        
        # Confidence metrics
        confidence_score = sum(c * w for c, w in confidences)
        
        # Stability: how much do models agree?
        model_disagreement = self._calculate_disagreement(predictions, weights)
        stability_score = 1 - model_disagreement
        
        # Uncertainty: entropy of direction probabilities
        probs = np.array([prob_bullish, prob_bearish, prob_neutral])
        probs = probs / probs.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(3)
        prediction_entropy = entropy / max_entropy
        uncertainty_score = prediction_entropy
        
        # Signal robustness
        signal_robustness = confidence_score * stability_score * (1 - uncertainty_score * 0.5)
        
        # Return percentiles (simplified)
        return_percentiles = {
            5: expected_return - 1.65 * return_std,
            25: expected_return - 0.67 * return_std,
            50: expected_return,
            75: expected_return + 0.67 * return_std,
            95: expected_return + 1.65 * return_std
        }
        
        return EnsemblePrediction(
            prob_bullish=float(prob_bullish),
            prob_bearish=float(prob_bearish),
            prob_neutral=float(prob_neutral),
            direction=direction,
            intensity=intensity,
            expected_return=float(expected_return),
            return_std=float(return_std),
            return_percentiles={k: float(v) for k, v in return_percentiles.items()},
            confidence_score=float(confidence_score),
            stability_score=float(stability_score),
            uncertainty_score=float(uncertainty_score),
            signal_robustness=float(signal_robustness),
            model_weights={k.value: float(v) for k, v in weights.items()},
            model_contributions=model_contributions,
            model_disagreement=float(model_disagreement),
            prediction_entropy=float(prediction_entropy)
        )
    
    def _calculate_adaptive_weights(
        self,
        predictions: List[ModelPrediction],
        current_regime: str
    ) -> Dict[ModelType, float]:
        """
        Calculate adaptive weights based on regime and performance.
        """
        weights = {}
        
        for pred in predictions:
            model = pred.model_type
            base_w = self.base_weights.get(model, 0.1)
            
            # Apply regime adjustment
            regime_adj = self.regime_adjustments.get(current_regime, {})
            regime_mult = regime_adj.get(model, 1.0)
            
            # Apply performance adjustment
            perf_history = self.model_performance.get(model, [])
            if len(perf_history) >= 10:
                recent_perf = np.mean(perf_history[-10:])
                perf_mult = 0.5 + recent_perf  # Scale from 0.5 to 1.5
            else:
                perf_mult = 1.0
                
            # Apply model's regime affinity
            regime_affinity = pred.regime_affinity.get(current_regime, 1.0)
            
            weights[model] = base_w * regime_mult * perf_mult * regime_affinity
            
        return weights
    
    def _calculate_disagreement(
        self,
        predictions: List[ModelPrediction],
        weights: Dict[ModelType, float]
    ) -> float:
        """
        Calculate weighted disagreement between models.
        """
        if len(predictions) < 2:
            return 0.0
            
        directions = []
        ws = []
        
        for pred in predictions:
            probs = pred.direction_probs
            direction_score = probs.get("bullish", 0) - probs.get("bearish", 0)
            directions.append(direction_score)
            ws.append(weights.get(pred.model_type, 0.1))
            
        directions = np.array(directions)
        ws = np.array(ws) / np.sum(ws)
        
        weighted_mean = np.sum(directions * ws)
        disagreement = np.sqrt(np.sum(ws * (directions - weighted_mean)**2))
        
        return min(disagreement, 1.0)
    
    def _classify_direction(
        self,
        prob_bull: float,
        prob_bear: float,
        prob_neutral: float
    ) -> str:
        """
        Classify direction based on probabilities.
        """
        if prob_bull > prob_bear and prob_bull > prob_neutral:
            return "bullish"
        elif prob_bear > prob_bull and prob_bear > prob_neutral:
            return "bearish"
        else:
            return "neutral"
    
    def _classify_intensity(self, max_prob: float, abs_return: float) -> str:
        """
        Classify intensity based on probability and expected return.
        """
        # Combine probability strength and return magnitude
        intensity_score = max_prob * 0.6 + min(abs_return * 20, 1.0) * 0.4
        
        if intensity_score > 0.8:
            return "extreme"
        elif intensity_score > 0.6:
            return "strong"
        elif intensity_score > 0.4:
            return "moderate"
        else:
            return "weak"
    
    def update_performance(
        self,
        model_type: ModelType,
        accuracy: float
    ) -> None:
        """
        Update model performance history for adaptive weighting.
        """
        if model_type not in self.model_performance:
            self.model_performance[model_type] = []
            
        self.model_performance[model_type].append(accuracy)
        
        # Keep last 100 observations
        if len(self.model_performance[model_type]) > 100:
            self.model_performance[model_type] = self.model_performance[model_type][-100:]
    
    def optimize_weights(
        self,
        historical_predictions: List[Dict],
        historical_returns: np.ndarray
    ) -> Dict[ModelType, float]:
        """
        Optimize ensemble weights using historical data.
        Minimizes prediction error while maintaining diversification.
        """
        n_models = len(ModelType)
        
        def objective(weights):
            weights = weights / weights.sum()
            
            errors = []
            for i, preds in enumerate(historical_predictions):
                ensemble_return = sum(
                    preds.get(m.value, {}).get("expected_return", 0) * weights[j]
                    for j, m in enumerate(ModelType)
                )
                errors.append((ensemble_return - historical_returns[i])**2)
                
            # Add regularization for diversification
            concentration = np.sum(weights**2)
            
            return np.mean(errors) + 0.1 * concentration
        
        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # Sum to 1
        ]
        
        # Bounds (each weight between 0.05 and 0.4)
        bounds = [(0.05, 0.4) for _ in range(n_models)]
        
        # Initial weights
        x0 = np.array([self.base_weights.get(m, 0.1) for m in ModelType])
        x0 = x0 / x0.sum()
        
        result = optimize.minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints
        )
        
        if result.success:
            optimized = result.x / result.x.sum()
            return {m: float(optimized[i]) for i, m in enumerate(ModelType)}
        else:
            return self.base_weights

"""
Main Probabilistic Forecast Engine
Orchestrates all quantitative models to produce final forecasts.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from app.engines.quant_engine.hmm_regime import HMMRegimeDetector, RegimeDetectionResult
from app.engines.quant_engine.bayesian_models import (
    BayesianReturnModel, BayesianConditionalModel, BayesianPrediction
)
from app.engines.quant_engine.monte_carlo import MonteCarloEngine, MonteCarloResult
from app.engines.quant_engine.volatility_models import VolatilityEngine, VolatilityForecast
from app.engines.quant_engine.ensemble_model import (
    EnsembleForecaster, EnsemblePrediction, ModelPrediction, ModelType
)
from app.models.forecasts import DirectionEnum, IntensityEnum, RegimeEnum


@dataclass
class ForecastOutput:
    symbol: str
    horizon: str
    horizon_days: int
    forecast_date: datetime
    target_date: datetime
    
    # Direction
    direction: DirectionEnum
    intensity: IntensityEnum
    
    # Probabilities
    prob_bullish: float
    prob_bearish: float
    prob_neutral: float
    
    # Expected returns
    expected_return: float
    expected_return_5th: float
    expected_return_25th: float
    expected_return_75th: float
    expected_return_95th: float
    
    # Regime
    detected_regime: RegimeEnum
    regime_probability: float
    regime_stability: float
    
    # Confidence
    confidence_score: float
    stability_score: float
    uncertainty_score: float
    signal_robustness: float
    
    # Model details
    model_weights: Dict[str, float]
    factor_contributions: Dict[str, float]
    
    # Scenarios
    scenarios: Dict[str, Dict]
    stress_scenarios: List[Dict]


class ProbabilisticForecastEngine:
    """
    Main engine that orchestrates all forecasting models.
    """
    
    HORIZON_DAYS = {
        "1D": 1,
        "1W": 5,
        "1M": 22,
        "3M": 66,
        "6M": 132,
        "1Y": 252
    }
    
    def __init__(
        self,
        hmm_states: int = 6,
        monte_carlo_sims: int = 10000
    ):
        # Initialize sub-engines
        self.hmm_detector = HMMRegimeDetector(n_regimes=hmm_states)
        self.bayesian_model = BayesianReturnModel()
        self.conditional_model = BayesianConditionalModel()
        self.monte_carlo = MonteCarloEngine(n_simulations=monte_carlo_sims)
        self.volatility_engine = VolatilityEngine()
        self.ensemble = EnsembleForecaster()
        
        self.is_fitted = False
        
    def fit(
        self,
        returns: np.ndarray,
        volatility: np.ndarray,
        volume_ratio: Optional[np.ndarray] = None,
        vix_level: Optional[np.ndarray] = None,
        macro_data: Optional[Dict[str, np.ndarray]] = None
    ) -> "ProbabilisticForecastEngine":
        """
        Fit all models on historical data.
        """
        # Fit HMM regime detector
        self.hmm_detector.fit(returns, volatility, volume_ratio, vix_level)
        
        # Update Bayesian model with historical returns
        self.bayesian_model.update(returns)
        
        # Fit volatility model
        self.volatility_engine.fit(returns, model_type="GARCH")
        
        self.is_fitted = True
        return self
        
    def forecast(
        self,
        symbol: str,
        horizon: str,
        current_data: Dict,
        macro_conditions: Optional[Dict[str, str]] = None
    ) -> ForecastOutput:
        """
        Generate probabilistic forecast for specified horizon.
        
        Parameters:
        -----------
        symbol : str
            Market symbol (e.g., 'SPX', 'NDX')
        horizon : str
            Forecast horizon ('1D', '1W', '1M', '3M', '6M', '1Y')
        current_data : dict
            Current market data including returns, volatility, etc.
        macro_conditions : dict
            Current macro conditions for conditional model
        """
        if not self.is_fitted:
            raise ValueError("Engine must be fitted before forecasting")
            
        horizon_days = self.HORIZON_DAYS.get(horizon, 22)
        
        # Extract current data
        returns = current_data["returns"]
        volatility = current_data["volatility"]
        current_vol = volatility[-1] if len(volatility) > 0 else 0.01
        
        # 1. Detect current regime
        regime_result = self.hmm_detector.detect(
            returns, volatility,
            volume_ratio=current_data.get("volume_ratio"),
            vix_level=current_data.get("vix_level")
        )
        
        # 2. Get Bayesian prediction
        bayesian_pred = self.bayesian_model.predict(horizon_days)
        
        # 3. Get conditional probabilities
        if macro_conditions:
            conditional = self.conditional_model.compute_conditional_probability(
                macro_conditions, target="bullish"
            )
        else:
            conditional = None
            
        # 4. Get volatility forecast
        vol_forecast = self.volatility_engine.forecast(horizon_days)
        
        # 5. Run Monte Carlo
        mc_result = self.monte_carlo.simulate_returns(
            mu=bayesian_pred.posterior_mean / horizon_days,
            sigma=vol_forecast.forecast_volatility.mean() if len(vol_forecast.forecast_volatility) > 0 else current_vol,
            horizon_days=horizon_days,
            distribution="student_t",
            regime_params=self._get_regime_params(regime_result.current_regime.value)
        )
        
        # 6. Generate stress scenarios
        stress_scenarios = self.monte_carlo.generate_stress_scenarios(
            base_mu=returns[-20:].mean() if len(returns) >= 20 else 0,
            base_sigma=current_vol,
            horizon_days=horizon_days,
            current_regime=regime_result.current_regime.value
        )
        
        # 7. Combine predictions via ensemble
        model_predictions = self._create_model_predictions(
            bayesian_pred, mc_result, vol_forecast, 
            regime_result, conditional
        )
        
        ensemble_pred = self.ensemble.combine_predictions(
            model_predictions,
            current_regime=regime_result.current_regime.value
        )
        
        # 8. Build output
        forecast_date = datetime.utcnow()
        target_date = forecast_date + timedelta(days=horizon_days)
        
        return ForecastOutput(
            symbol=symbol,
            horizon=horizon,
            horizon_days=horizon_days,
            forecast_date=forecast_date,
            target_date=target_date.date(),
            direction=self._map_direction(ensemble_pred.direction),
            intensity=self._map_intensity(ensemble_pred.intensity),
            prob_bullish=ensemble_pred.prob_bullish,
            prob_bearish=ensemble_pred.prob_bearish,
            prob_neutral=ensemble_pred.prob_neutral,
            expected_return=ensemble_pred.expected_return,
            expected_return_5th=ensemble_pred.return_percentiles.get(5, 0),
            expected_return_25th=ensemble_pred.return_percentiles.get(25, 0),
            expected_return_75th=ensemble_pred.return_percentiles.get(75, 0),
            expected_return_95th=ensemble_pred.return_percentiles.get(95, 0),
            detected_regime=self._map_regime(regime_result.current_regime.value),
            regime_probability=regime_result.regime_probability,
            regime_stability=regime_result.regime_stability,
            confidence_score=ensemble_pred.confidence_score,
            stability_score=ensemble_pred.stability_score,
            uncertainty_score=ensemble_pred.uncertainty_score,
            signal_robustness=ensemble_pred.signal_robustness,
            model_weights=ensemble_pred.model_weights,
            factor_contributions=self._compute_factor_contributions(ensemble_pred),
            scenarios={
                "base": {
                    "expected_return": ensemble_pred.expected_return,
                    "probability": 0.6
                },
                "bullish": {
                    "expected_return": ensemble_pred.return_percentiles.get(75, 0),
                    "probability": ensemble_pred.prob_bullish * 0.5
                },
                "bearish": {
                    "expected_return": ensemble_pred.return_percentiles.get(25, 0),
                    "probability": ensemble_pred.prob_bearish * 0.5
                }
            },
            stress_scenarios=[
                {
                    "name": s.name,
                    "probability": s.probability,
                    "expected_return": s.expected_return,
                    "worst_case": s.worst_case_return
                }
                for s in stress_scenarios
            ]
        )
    
    def _create_model_predictions(
        self,
        bayesian: BayesianPrediction,
        monte_carlo: MonteCarloResult,
        volatility: VolatilityForecast,
        regime: RegimeDetectionResult,
        conditional: Optional[object]
    ) -> List[ModelPrediction]:
        """
        Convert individual model outputs to standardized predictions.
        """
        predictions = []
        
        # Bayesian model prediction
        predictions.append(ModelPrediction(
            model_type=ModelType.BAYESIAN,
            direction_probs={
                "bullish": bayesian.prob_positive,
                "bearish": bayesian.prob_negative,
                "neutral": max(0, 1 - bayesian.prob_positive - bayesian.prob_negative)
            },
            expected_return=bayesian.posterior_mean,
            confidence=bayesian.evidence_strength,
            regime_affinity={"risk_on": 1.0, "risk_off": 1.0}
        ))
        
        # Monte Carlo prediction
        predictions.append(ModelPrediction(
            model_type=ModelType.MONTE_CARLO,
            direction_probs={
                "bullish": monte_carlo.prob_positive,
                "bearish": monte_carlo.prob_negative,
                "neutral": max(0, 1 - monte_carlo.prob_positive - monte_carlo.prob_negative)
            },
            expected_return=monte_carlo.mean_return,
            confidence=1 - min(monte_carlo.std_return / 0.1, 1),
            regime_affinity={"panic": 1.2, "euphoria": 1.2}
        ))
        
        # Volatility model prediction
        vol_regime = volatility.volatility_regime
        vol_direction_probs = {
            "low": {"bullish": 0.55, "bearish": 0.25, "neutral": 0.20},
            "medium": {"bullish": 0.40, "bearish": 0.35, "neutral": 0.25},
            "high": {"bullish": 0.30, "bearish": 0.50, "neutral": 0.20},
            "extreme": {"bullish": 0.20, "bearish": 0.65, "neutral": 0.15}
        }.get(vol_regime, {"bullish": 0.33, "bearish": 0.33, "neutral": 0.34})
        
        predictions.append(ModelPrediction(
            model_type=ModelType.VOLATILITY,
            direction_probs=vol_direction_probs,
            expected_return=0,  # Volatility doesn't predict direction
            confidence=0.8 if volatility.persistence < 0.99 else 0.5,
            regime_affinity={"panic": 1.5, "risk_off": 1.3}
        ))
        
        # HMM regime prediction
        regime_direction = {
            "risk_on": {"bullish": 0.60, "bearish": 0.25, "neutral": 0.15},
            "risk_off": {"bullish": 0.25, "bearish": 0.55, "neutral": 0.20},
            "euphoria": {"bullish": 0.55, "bearish": 0.30, "neutral": 0.15},
            "panic": {"bullish": 0.15, "bearish": 0.75, "neutral": 0.10},
            "consolidation": {"bullish": 0.35, "bearish": 0.35, "neutral": 0.30},
            "macro_stress": {"bullish": 0.25, "bearish": 0.55, "neutral": 0.20}
        }.get(regime.current_regime.value, {"bullish": 0.33, "bearish": 0.33, "neutral": 0.34})
        
        predictions.append(ModelPrediction(
            model_type=ModelType.HMM_REGIME,
            direction_probs=regime_direction,
            expected_return=0,
            confidence=regime.regime_probability * regime.regime_stability,
            regime_affinity={regime.current_regime.value: 1.5}
        ))
        
        return predictions
    
    def _get_regime_params(self, regime: str) -> Dict:
        """
        Get Monte Carlo parameter adjustments based on regime.
        """
        params = {
            "risk_on": {"mu_mult": 1.2, "sigma_mult": 0.9},
            "risk_off": {"mu_mult": 0.8, "sigma_mult": 1.2},
            "euphoria": {"mu_mult": 1.3, "sigma_mult": 1.1},
            "panic": {"mu_mult": 0.5, "sigma_mult": 2.0},
            "consolidation": {"mu_mult": 1.0, "sigma_mult": 0.8},
            "macro_stress": {"mu_mult": 0.7, "sigma_mult": 1.5}
        }
        return params.get(regime, {"mu_mult": 1.0, "sigma_mult": 1.0})
    
    def _map_direction(self, direction: str) -> DirectionEnum:
        mapping = {
            "bullish": DirectionEnum.BULLISH,
            "bearish": DirectionEnum.BEARISH,
            "neutral": DirectionEnum.NEUTRAL
        }
        return mapping.get(direction, DirectionEnum.NEUTRAL)
    
    def _map_intensity(self, intensity: str) -> IntensityEnum:
        mapping = {
            "weak": IntensityEnum.WEAK,
            "moderate": IntensityEnum.MODERATE,
            "strong": IntensityEnum.STRONG,
            "extreme": IntensityEnum.EXTREME
        }
        return mapping.get(intensity, IntensityEnum.MODERATE)
    
    def _map_regime(self, regime: str) -> RegimeEnum:
        mapping = {
            "risk_on": RegimeEnum.RISK_ON,
            "risk_off": RegimeEnum.RISK_OFF,
            "euphoria": RegimeEnum.EUPHORIA,
            "panic": RegimeEnum.PANIC,
            "consolidation": RegimeEnum.CONSOLIDATION,
            "macro_stress": RegimeEnum.MACRO_STRESS,
            "reflation": RegimeEnum.REFLATION,
            "disinflation": RegimeEnum.DISINFLATION,
            "liquidity_expansion": RegimeEnum.LIQUIDITY_EXPANSION,
            "liquidity_contraction": RegimeEnum.LIQUIDITY_CONTRACTION
        }
        return mapping.get(regime, RegimeEnum.CONSOLIDATION)
    
    def _compute_factor_contributions(self, ensemble: EnsemblePrediction) -> Dict[str, float]:
        """
        Compute factor contributions to the forecast.
        """
        contributions = {}
        for model_name, details in ensemble.model_contributions.items():
            weight = details.get("weight", 0)
            expected_ret = details.get("expected_return", 0)
            contributions[model_name] = weight * expected_ret
        return contributions

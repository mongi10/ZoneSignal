"""
ZoneSignal - Ensemble Model Engine
Combinaison intelligente de tous les modèles quantitatifs
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import date, datetime

from app.engines.quant_models.hmm_regime import HMMRegimeEngine, HMMResult
from app.engines.quant_models.bayesian_forecast import BayesianForecastEngine, BayesianForecastResult
from app.engines.quant_models.monte_carlo import MonteCarloEngine, MonteCarloResult
from app.engines.seasonality.seasonality_engine import SeasonalityEngine, SeasonalityForecast
from app.models.market import DirectionEnum, IntensityEnum, RegimeEnum


@dataclass
class ModelContribution:
    """Contribution d'un modèle à la prévision finale"""
    model_name: str
    weight: float
    direction_vote: DirectionEnum
    confidence: float
    expected_return: float
    key_insight: str


@dataclass
class EnsembleForecast:
    """Prévision finale de l'ensemble"""
    # Prévision principale
    symbol: str
    horizon: str
    horizon_days: int
    forecast_date: date
    target_date: date
    
    # Direction et intensité
    direction: DirectionEnum
    intensity: IntensityEnum
    
    # Probabilités
    prob_bullish: float
    prob_bearish: float
    prob_neutral: float
    
    # Estimations quantitatives
    expected_return: float
    return_lower_bound: float  # 10th percentile
    return_upper_bound: float  # 90th percentile
    expected_volatility: float
    max_drawdown_estimate: float
    
    # Régime
    regime: RegimeEnum
    regime_confidence: float
    regime_transition_prob: float
    
    # Scores de confiance
    confidence_score: float
    stability_score: float
    uncertainty_score: float
    robustness_score: float
    
    # Contributions des modèles
    model_contributions: Dict[str, float]
    model_details: List[ModelContribution]
    
    # Facteurs clés
    key_factors: Dict[str, float]
    
    # Scénarios alternatifs
    scenarios: Dict[str, Dict[str, Any]]
    
    # Cone probabiliste
    probabilistic_cone: Dict[int, List[float]]
    
    # Métadonnées
    computation_time_ms: int
    model_version: str = "1.0.0"


class EnsembleModelEngine:
    """
    Moteur d'ensemble institutionnel
    
    Combine intelligemment:
    - HMM Regime Detection
    - Bayesian Forecast
    - Monte Carlo Simulations
    - Seasonality Analysis
    - (Future: Macro factors, Technical factors, etc.)
    
    Avec:
    - Pondération adaptative
    - Consensus voting
    - Uncertainty aggregation
    - Confidence calibration
    """
    
    def __init__(self, db_session=None):
        self.db = db_session
        
        # Initialiser les moteurs
        self.hmm_engine = HMMRegimeEngine()
        self.bayesian_engine = BayesianForecastEngine()
        self.monte_carlo_engine = MonteCarloEngine()
        self.seasonality_engine = SeasonalityEngine()
        
        # Poids par défaut des modèles
        self.default_weights = {
            "hmm_regime": 0.20,
            "bayesian": 0.25,
            "monte_carlo": 0.30,
            "seasonality": 0.15,
            "momentum": 0.10
        }
        
        # Horizons supportés
        self.horizons = {
            "1D": 1,
            "1W": 5,
            "1M": 21,
            "3M": 63,
            "6M": 126,
            "1Y": 252
        }
        
        # Seuils de direction
        self.direction_thresholds = {
            "bullish": 0.02,    # > 2% expected
            "bearish": -0.02,   # < -2% expected
        }
        
        # Seuils d'intensité (en terms de return attendu annualisé)
        self.intensity_thresholds = {
            "faible": 0.05,
            "moderee": 0.12,
            "forte": 0.20,
            "extreme": 0.30
        }
    
    # ═══════════════════════════════════════════════════════════════
    # PRÉVISION PRINCIPALE
    # ═══════════════════════════════════════════════════════════════
    
    def forecast(
        self,
        df: pd.DataFrame,
        symbol: str,
        horizon: str,
        weights: Optional[Dict[str, float]] = None,
        macro_bias: float = 0.0
    ) -> EnsembleForecast:
        """
        Génère une prévision d'ensemble complète
        
        Args:
            df: DataFrame avec OHLCV et métriques
            symbol: Symbole de l'indice
            horizon: Horizon ("1D", "1W", "1M", "3M", "6M", "1Y")
            weights: Poids personnalisés des modèles
            macro_bias: Biais macro (-1 à +1)
            
        Returns:
            EnsembleForecast complet
        """
        
        import time
        start_time = time.time()
        
        # Valider l'horizon
        if horizon not in self.horizons:
            raise ValueError(f"Invalid horizon: {horizon}")
        
        horizon_days = self.horizons[horizon]
        weights = weights or self.default_weights
        
        # Normaliser les poids
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # 1. HMM Regime Detection
        hmm_result = self._run_hmm(df)
        current_regime = hmm_result.current_regime.regime if hmm_result else RegimeEnum.CONSOLIDATION
        
        # 2. Bayesian Forecast
        bayesian_result = self._run_bayesian(
            df, horizon_days, current_regime.value, macro_bias
        )
        
        # 3. Monte Carlo Simulations
        monte_carlo_result = self._run_monte_carlo(
            df, horizon_days, hmm_result
        )
        
        # 4. Seasonality Forecast
        seasonality_result = self._run_seasonality(
            df, horizon_days
        )
        
        # 5. Momentum Analysis (simple)
        momentum_result = self._compute_momentum(df, horizon_days)
        
        # ═══════════════════════════════════════════════════════════
        # AGRÉGATION DES RÉSULTATS
        # ═══════════════════════════════════════════════════════════
        
        # Contributions des modèles
        contributions = []
        weighted_returns = []
        direction_votes = {"bullish": 0, "bearish": 0, "neutral": 0}
        
        # HMM contribution
        if hmm_result:
            hmm_return = self._regime_to_expected_return(
                hmm_result.current_regime.regime,
                horizon_days
            )
            hmm_direction = self._return_to_direction(hmm_return)
            weighted_returns.append(hmm_return * weights.get("hmm_regime", 0))
            direction_votes[hmm_direction.value] += weights.get("hmm_regime", 0)
            
            contributions.append(ModelContribution(
                model_name="HMM Regime",
                weight=weights.get("hmm_regime", 0),
                direction_vote=hmm_direction,
                confidence=hmm_result.current_regime.probability,
                expected_return=hmm_return,
                key_insight=f"Régime {hmm_result.current_regime.regime.value}"
            ))
        
        # Bayesian contribution
        if bayesian_result:
            bay_return = bayesian_result.expected_return
            bay_direction = self._return_to_direction(bay_return)
            weighted_returns.append(bay_return * weights.get("bayesian", 0))
            direction_votes[bay_direction.value] += weights.get("bayesian", 0)
            
            contributions.append(ModelContribution(
                model_name="Bayesian",
                weight=weights.get("bayesian", 0),
                direction_vote=bay_direction,
                confidence=bayesian_result.model_confidence,
                expected_return=bay_return,
                key_insight=f"P(+)={bayesian_result.prob_positive:.0%}"
            ))
        
        # Monte Carlo contribution  
        if monte_carlo_result:
            mc_return = monte_carlo_result.return_median
            mc_direction = self._return_to_direction(mc_return)
            weighted_returns.append(mc_return * weights.get("monte_carlo", 0))
            direction_votes[mc_direction.value] += weights.get("monte_carlo", 0)
            
            contributions.append(ModelContribution(
                model_name="Monte Carlo",
                weight=weights.get("monte_carlo", 0),
                direction_vote=mc_direction,
                confidence=monte_carlo_result.convergence_metric,
                expected_return=mc_return,
                key_insight=f"VaR95={monte_carlo_result.var_95:.1%}"
            ))
        
        # Seasonality contribution
        if seasonality_result:
            seas_return = seasonality_result.expected_return
            seas_direction = self._return_to_direction(seas_return)
            weighted_returns.append(seas_return * weights.get("seasonality", 0))
            direction_votes[seas_direction.value] += weights.get("seasonality", 0)
            
            contributions.append(ModelContribution(
                model_name="Seasonality",
                weight=weights.get("seasonality", 0),
                direction_vote=seas_direction,
                confidence=seasonality_result.confidence,
                expected_return=seas_return,
                key_insight=f"Score={seasonality_result.composite_score:.2f}"
            ))
        
        # Momentum contribution
        if momentum_result:
            mom_return = momentum_result["expected_return"]
            mom_direction = self._return_to_direction(mom_return)
            weighted_returns.append(mom_return * weights.get("momentum", 0))
            direction_votes[mom_direction.value] += weights.get("momentum", 0)
            
            contributions.append(ModelContribution(
                model_name="Momentum",
                weight=weights.get("momentum", 0),
                direction_vote=mom_direction,
                confidence=momentum_result["confidence"],
                expected_return=mom_return,
                key_insight=momentum_result["trend"]
            ))
        
        # ═══════════════════════════════════════════════════════════
        # CALCUL DES MÉTRIQUES FINALES
        # ═══════════════════════════════════════════════════════════
        
        # Return attendu (moyenne pondérée)
        expected_return = sum(weighted_returns)
        
        # Direction finale (vote majoritaire pondéré)
        final_direction = max(direction_votes, key=direction_votes.get)
        direction_enum = DirectionEnum(final_direction)
        
        # Intensité
        intensity = self._compute_intensity(expected_return, horizon_days)
        
        # Probabilités directionnelles (from Monte Carlo principalement)
        if monte_carlo_result:
            prob_bullish = monte_carlo_result.prob_positive
            prob_bearish = 1 - monte_carlo_result.prob_positive
        elif bayesian_result:
            prob_bullish = bayesian_result.prob_positive
            prob_bearish = bayesian_result.prob_negative
        else:
            prob_bullish = 0.5 if final_direction == "bullish" else 0.3
            prob_bearish = 0.5 if final_direction == "bearish" else 0.3
        
        prob_neutral = max(0, 1 - prob_bullish - prob_bearish)
        
        # Bounds
        if monte_carlo_result:
            return_lower = monte_carlo_result.return_percentiles[10]
            return_upper = monte_carlo_result.return_percentiles[90]
            expected_vol = monte_carlo_result.return_std
            max_dd = monte_carlo_result.worst_drawdown_5pct
        else:
            return_lower = expected_return - 0.1
            return_upper = expected_return + 0.1
            expected_vol = 0.16
            max_dd = -0.15
        
        # Scores de confiance
        confidence_score = self._compute_ensemble_confidence(contributions)
        stability_score = self._compute_stability_score(contributions)
        uncertainty_score = self._compute_uncertainty_score(
            monte_carlo_result, bayesian_result
        )
        robustness_score = self._compute_robustness_score(direction_votes)
        
        # Key factors
        key_factors = self._extract_key_factors(
            hmm_result, bayesian_result, monte_carlo_result, 
            seasonality_result, momentum_result
        )
        
        # Scénarios
        scenarios = self._build_scenarios(
            bayesian_result, monte_carlo_result
        )
        
        # Probabilistic cone (from Monte Carlo)
        cone = {}
        if monte_carlo_result:
            cone = {
                p: monte_carlo_result.cone_percentiles[p].tolist()
                for p in [5, 25, 50, 75, 95]
            }
        
        # Régime transition
        regime_trans_prob = 0.0
        if hmm_result:
            # Probabilité de quitter le régime actuel
            current_state = list(hmm_result.current_regime.transition_probs.keys())[0]
            stay_prob = hmm_result.current_regime.transition_probs.get(
                hmm_result.current_regime.regime.value, 0.9
            )
            regime_trans_prob = 1 - stay_prob
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        return EnsembleForecast(
            symbol=symbol,
            horizon=horizon,
            horizon_days=horizon_days,
            forecast_date=date.today(),
            target_date=date.today() + pd.Timedelta(days=horizon_days),
            direction=direction_enum,
            intensity=intensity,
            prob_bullish=float(prob_bullish),
            prob_bearish=float(prob_bearish),
            prob_neutral=float(prob_neutral),
            expected_return=float(expected_return),
            return_lower_bound=float(return_lower),
            return_upper_bound=float(return_upper),
            expected_volatility=float(expected_vol),
            max_drawdown_estimate=float(max_dd),
            regime=current_regime,
            regime_confidence=float(hmm_result.current_regime.probability if hmm_result else 0.5),
            regime_transition_prob=float(regime_trans_prob),
            confidence_score=float(confidence_score),
            stability_score=float(stability_score),
            uncertainty_score=float(uncertainty_score),
            robustness_score=float(robustness_score),
            model_contributions={c.model_name: c.weight for c in contributions},
            model_details=contributions,
            key_factors=key_factors,
            scenarios=scenarios,
            probabilistic_cone=cone,
            computation_time_ms=elapsed_ms
        )
    
    # ═══════════════════════════════════════════════════════════════
    # EXÉCUTION DES MODÈLES INDIVIDUELS
    # ═══════════════════════════════════════════════════════════════
    
    def _run_hmm(self, df: pd.DataFrame) -> Optional[HMMResult]:
        """Exécute le modèle HMM"""
        try:
            # Vérifier si le modèle est entraîné
            if not self.hmm_engine.is_fitted:
                self.hmm_engine.fit(df)
            return self.hmm_engine.predict(df)
        except Exception as e:
            print(f"HMM error: {e}")
            return None
    
    def _run_bayesian(
        self, 
        df: pd.DataFrame, 
        horizon_days: int,
        current_regime: str,
        macro_bias: float
    ) -> Optional[BayesianForecastResult]:
        """Exécute le modèle Bayésien"""
        try:
            return self.bayesian_engine.forecast(
                df,
                horizon_days=horizon_days,
                current_regime=current_regime,
                macro_adjustment=macro_bias * 0.05  # Scale le biais
            )
        except Exception as e:
            print(f"Bayesian error: {e}")
            return None
    
    def _run_monte_carlo(
        self,
        df: pd.DataFrame,
        horizon_days: int,
        hmm_result: Optional[HMMResult]
    ) -> Optional[MonteCarloResult]:
        """Exécute les simulations Monte Carlo"""
        try:
            # Adapter le modèle selon le régime
            model = "jump_diffusion"
            stress = 1.0
            
            if hmm_result:
                regime = hmm_result.current_regime.regime
                if regime in [RegimeEnum.PANIC, RegimeEnum.MACRO_STRESS]:
                    model = "heston"
                    stress = 1.5
                elif regime == RegimeEnum.EUPHORIA:
                    stress = 0.8
            
            return self.monte_carlo_engine.simulate(
                df,
                horizon_days=horizon_days,
                model=model,
                stress_multiplier=stress
            )
        except Exception as e:
            print(f"Monte Carlo error: {e}")
            return None
    
    def _run_seasonality(
        self,
        df: pd.DataFrame,
        horizon_days: int
    ) -> Optional[SeasonalityForecast]:
        """Exécute l'analyse de saisonnalité"""
        try:
            return self.seasonality_engine.get_seasonality_forecast(
                df,
                target_date=date.today(),
                horizon_days=horizon_days
            )
        except Exception as e:
            print(f"Seasonality error: {e}")
            return None
    
    def _compute_momentum(
        self,
        df: pd.DataFrame,
        horizon_days: int
    ) -> Dict[str, Any]:
        """Calcul simple du momentum"""
        try:
            returns = df['daily_return'].dropna()
            
            # Momentum sur différentes périodes
            mom_5d = returns.tail(5).mean() * 252
            mom_20d = returns.tail(20).mean() * 252
            mom_60d = returns.tail(60).mean() * 252
            
            # Score composite
            composite = 0.5 * mom_5d + 0.3 * mom_20d + 0.2 * mom_60d
            
            # Tendance
            if mom_5d > mom_20d > 0:
                trend = "Uptrend accélérant"
            elif mom_5d < mom_20d < 0:
                trend = "Downtrend accélérant"
            elif mom_5d > 0 and mom_20d > 0:
                trend = "Uptrend stable"
            elif mom_5d < 0 and mom_20d < 0:
                trend = "Downtrend stable"
            else:
                trend = "Transition/Mixte"
            
            # Expected return ajusté pour l'horizon
            expected_return = composite * (horizon_days / 252)
            
            return {
                "expected_return": float(expected_return),
                "confidence": min(0.7, abs(composite) / 0.2),
                "trend": trend,
                "mom_5d": float(mom_5d),
                "mom_20d": float(mom_20d),
                "mom_60d": float(mom_60d)
            }
        except Exception as e:
            return None
    
    # ═══════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════
    
    def _regime_to_expected_return(
        self, 
        regime: RegimeEnum, 
        horizon_days: int
    ) -> float:
        """Convertit un régime en return attendu"""
        
        regime_returns = {
            RegimeEnum.RISK_ON: 0.12,
            RegimeEnum.RISK_OFF: -0.05,
            RegimeEnum.EUPHORIA: 0.20,
            RegimeEnum.PANIC: -0.25,
            RegimeEnum.CONSOLIDATION: 0.03,
            RegimeEnum.MACRO_STRESS: -0.15,
            RegimeEnum.REFLATION: 0.15,
            RegimeEnum.DISINFLATION: 0.08,
            RegimeEnum.LIQUIDITY_EXPANSION: 0.18,
            RegimeEnum.LIQUIDITY_CONTRACTION: -0.10
        }
        
        annual_return = regime_returns.get(regime, 0.05)
        return annual_return * (horizon_days / 252)
    
    def _return_to_direction(self, expected_return: float) -> DirectionEnum:
        """Convertit un return en direction"""
        if expected_return > self.direction_thresholds["bullish"]:
            return DirectionEnum.BULLISH
        elif expected_return < self.direction_thresholds["bearish"]:
            return DirectionEnum.BEARISH
        return DirectionEnum.NEUTRAL
    
    def _compute_intensity(
        self, 
        expected_return: float, 
        horizon_days: int
    ) -> IntensityEnum:
        """Calcule l'intensité du signal"""
        
        # Annualiser le return
        annual_return = abs(expected_return) * (252 / horizon_days)
        
        if annual_return > self.intensity_thresholds["extreme"]:
            return IntensityEnum.EXTREME
        elif annual_return > self.intensity_thresholds["forte"]:
            return IntensityEnum.FORTE
        elif annual_return > self.intensity_thresholds["moderee"]:
            return IntensityEnum.MODEREE
        return IntensityEnum.FAIBLE
    
    def _compute_ensemble_confidence(
        self, 
        contributions: List[ModelContribution]
    ) -> float:
        """Calcule la confiance de l'ensemble"""
        if not contributions:
            return 0.5
        
        weighted_confidence = sum(
            c.confidence * c.weight for c in contributions
        )
        return float(np.clip(weighted_confidence, 0, 1))
    
    def _compute_stability_score(
        self, 
        contributions: List[ModelContribution]
    ) -> float:
        """Calcule la stabilité (accord entre modèles)"""
        if not contributions:
            return 0.5
        
        directions = [c.direction_vote for c in contributions]
        # Score = % du modèle majoritaire
        from collections import Counter
        counts = Counter(directions)
        most_common_count = counts.most_common(1)[0][1]
        
        return float(most_common_count / len(directions))
    
    def _compute_uncertainty_score(
        self,
        mc_result: Optional[MonteCarloResult],
        bay_result: Optional[BayesianForecastResult]
    ) -> float:
        """Calcule l'incertitude (0 = certain, 1 = très incertain)"""
        
        uncertainties = []
        
        if mc_result:
            # Largeur relative de l'intervalle 80%
            spread = mc_result.return_percentiles[90] - mc_result.return_percentiles[10]
            uncertainties.append(min(1, abs(spread) / 0.3))
        
        if bay_result:
            # Information ratio inverse
            ir = bay_result.information_ratio
            uncertainties.append(max(0, 1 - ir))
        
        if uncertainties:
            return float(np.mean(uncertainties))
        return 0.5
    
    def _compute_robustness_score(
        self, 
        direction_votes: Dict[str, float]
    ) -> float:
        """Calcule la robustesse du signal"""
        
        max_vote = max(direction_votes.values())
        second_vote = sorted(direction_votes.values())[-2]
        
        # Différence entre premier et second
        robustness = max_vote - second_vote
        
        return float(np.clip(robustness * 2, 0, 1))
    
    def _extract_key_factors(
        self,
        hmm_result: Optional[HMMResult],
        bay_result: Optional[BayesianForecastResult],
        mc_result: Optional[MonteCarloResult],
        seas_result: Optional[SeasonalityForecast],
        mom_result: Optional[Dict]
    ) -> Dict[str, float]:
        """Extrait les facteurs clés de la prévision"""
        
        factors = {}
        
        if hmm_result:
            factors["regime_confidence"] = hmm_result.current_regime.probability
            factors["regime_stability"] = hmm_result.model_metrics.get("regime_stability", 0.5)
        
        if bay_result:
            factors["bayesian_info_ratio"] = bay_result.information_ratio
            factors["prior_influence"] = bay_result.prior_weight
        
        if mc_result:
            factors["tail_risk"] = mc_result.var_95
            factors["skewness"] = mc_result.return_skewness
        
        if seas_result:
            factors["seasonality_score"] = seas_result.composite_score
        
        if mom_result:
            factors["momentum_5d"] = mom_result.get("mom_5d", 0)
            factors["momentum_20d"] = mom_result.get("mom_20d", 0)
        
        return {k: float(v) for k, v in factors.items()}
    
    def _build_scenarios(
        self,
        bay_result: Optional[BayesianForecastResult],
        mc_result: Optional[MonteCarloResult]
    ) -> Dict[str, Dict[str, Any]]:
        """Construit les scénarios alternatifs"""
        
        scenarios = {}
        
        if bay_result and bay_result.scenarios:
            for name, data in bay_result.scenarios.items():
                scenarios[name] = {
                    "return": data.get("return", 0),
                    "probability": data.get("probability", 0),
                    "description": data.get("description", ""),
                    "source": "bayesian"
                }
        
        if mc_result:
            scenarios["stress_95"] = {
                "return": mc_result.return_percentiles[5],
                "probability": 0.05,
                "description": "Scénario adverse (5th percentile)",
                "source": "monte_carlo"
            }
            scenarios["best_case_95"] = {
                "return": mc_result.return_percentiles[95],
                "probability": 0.05,
                "description": "Scénario favorable (95th percentile)",
                "source": "monte_carlo"
            }
        
        return scenarios

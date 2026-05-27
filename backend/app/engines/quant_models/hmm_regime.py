"""
ZoneSignal - Hidden Markov Model Regime Detection Engine
Détection de régimes cachés de marché avec HMM Gaussien
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from scipy import stats

from app.core.config import settings
from app.models.market import RegimeEnum


@dataclass
class RegimeState:
    """État d'un régime détecté"""
    regime: RegimeEnum
    probability: float
    duration_estimate: int  # jours estimés dans ce régime
    transition_probs: Dict[str, float]  # probabilités de transition


@dataclass
class HMMResult:
    """Résultat complet de l'analyse HMM"""
    current_regime: RegimeState
    regime_history: List[int]
    regime_probabilities: np.ndarray
    transition_matrix: np.ndarray
    regime_statistics: Dict[str, Dict]
    model_metrics: Dict[str, float]


class HMMRegimeEngine:
    """
    Moteur de détection de régimes basé sur Hidden Markov Models
    
    Caractéristiques:
    - Détection de 4-6 régimes de marché cachés
    - Estimation des probabilités de transition
    - Classification automatique des régimes
    - Mise à jour adaptative du modèle
    """
    
    def __init__(self, n_regimes: int = None):
        self.n_regimes = n_regimes or settings.HMM_N_REGIMES
        self.model: Optional[GaussianHMM] = None
        self.scaler = StandardScaler()
        self.regime_mapping: Dict[int, RegimeEnum] = {}
        self.is_fitted = False
    
    # ═══════════════════════════════════════════════════════════════
    # PRÉPARATION DES FEATURES
    # ═══════════════════════════════════════════════════════════════
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prépare les features pour le HMM
        
        Features utilisées:
        - Returns (momentum)
        - Volatilité réalisée
        - Drawdown
        - Volume relatif
        - Dispersion des returns
        """
        
        features = pd.DataFrame(index=df.index)
        
        # Returns sur différentes périodes
        features['return_1d'] = df['daily_return']
        features['return_5d'] = df['close'].pct_change(5)
        features['return_20d'] = df['close'].pct_change(20)
        
        # Volatilité
        features['volatility'] = df['daily_return'].rolling(20).std() * np.sqrt(252)
        features['vol_change'] = features['volatility'].pct_change(5)
        
        # Drawdown
        features['drawdown'] = df['drawdown_from_high']
        
        # Skewness des returns (asymétrie)
        features['skewness'] = df['daily_return'].rolling(20).apply(
            lambda x: stats.skew(x), raw=True
        )
        
        # Volume relatif
        if 'relative_volume' in df.columns:
            features['rel_volume'] = df['relative_volume']
        
        # Nettoyer les NaN
        features = features.dropna()
        
        return features.values
    
    # ═══════════════════════════════════════════════════════════════
    # ENTRAÎNEMENT DU MODÈLE
    # ═══════════════════════════════════════════════════════════════
    
    def fit(
        self, 
        df: pd.DataFrame,
        min_samples: int = 252
    ) -> Dict[str, Any]:
        """
        Entraîne le modèle HMM sur les données historiques
        
        Args:
            df: DataFrame avec colonnes OHLCV + métriques
            min_samples: Nombre minimum d'observations requises
            
        Returns:
            Métriques d'entraînement
        """
        
        # Préparer les features
        features = self._prepare_features(df)
        
        if len(features) < min_samples:
            raise ValueError(f"Insufficient data: {len(features)} < {min_samples}")
        
        # Normaliser
        features_scaled = self.scaler.fit_transform(features)
        
        # Créer et entraîner le modèle HMM
        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=200,
            random_state=42,
            verbose=False
        )
        
        self.model.fit(features_scaled)
        
        # Décoder les états cachés
        hidden_states = self.model.predict(features_scaled)
        
        # Classifier les régimes
        self._classify_regimes(df, features, hidden_states)
        
        # Calculer les métriques
        log_likelihood = self.model.score(features_scaled)
        aic = -2 * log_likelihood + 2 * self._count_parameters()
        bic = -2 * log_likelihood + np.log(len(features)) * self._count_parameters()
        
        self.is_fitted = True
        
        return {
            "n_samples": len(features),
            "n_regimes": self.n_regimes,
            "log_likelihood": log_likelihood,
            "aic": aic,
            "bic": bic,
            "regime_mapping": {k: v.value for k, v in self.regime_mapping.items()}
        }
    
    def _count_parameters(self) -> int:
        """Compte le nombre de paramètres du modèle"""
        n = self.n_regimes
        k = self.model.n_features
        # Transitions + means + covariances
        return n * (n - 1) + n * k + n * k * (k + 1) // 2
    
    def _classify_regimes(
        self, 
        df: pd.DataFrame, 
        features: np.ndarray,
        hidden_states: np.ndarray
    ):
        """
        Classifie automatiquement les régimes HMM en catégories économiques
        
        Basé sur:
        - Return moyen dans chaque état
        - Volatilité moyenne dans chaque état
        - Drawdown moyen
        """
        
        # Calculer les statistiques par régime
        regime_stats = {}
        
        for state in range(self.n_regimes):
            mask = hidden_states == state
            
            if mask.sum() == 0:
                continue
            
            regime_stats[state] = {
                "avg_return": features[mask, 0].mean(),  # return_1d
                "avg_volatility": features[mask, 3].mean() if features.shape[1] > 3 else 0,
                "avg_drawdown": features[mask, 5].mean() if features.shape[1] > 5 else 0,
                "count": mask.sum()
            }
        
        # Trier par return moyen pour classification
        sorted_states = sorted(
            regime_stats.keys(),
            key=lambda x: regime_stats[x]["avg_return"]
        )
        
        # Mapper aux régimes économiques
        if self.n_regimes == 4:
            # 4 régimes: Panique, Risk-Off, Consolidation, Risk-On
            mapping = {
                sorted_states[0]: RegimeEnum.PANIC,           # Pire returns
                sorted_states[1]: RegimeEnum.RISK_OFF,        # Returns négatifs
                sorted_states[2]: RegimeEnum.CONSOLIDATION,   # Returns neutres
                sorted_states[3]: RegimeEnum.RISK_ON          # Meilleurs returns
            }
        elif self.n_regimes == 6:
            # 6 régimes: Panique, Risk-Off, Disinflation, Consolidation, Reflation, Euphorie
            mapping = {
                sorted_states[0]: RegimeEnum.PANIC,
                sorted_states[1]: RegimeEnum.RISK_OFF,
                sorted_states[2]: RegimeEnum.DISINFLATION,
                sorted_states[3]: RegimeEnum.CONSOLIDATION,
                sorted_states[4]: RegimeEnum.REFLATION,
                sorted_states[5]: RegimeEnum.EUPHORIA
            }
        else:
            # Mapping générique
            mapping = {}
            regime_list = list(RegimeEnum)
            for i, state in enumerate(sorted_states):
                mapping[state] = regime_list[i % len(regime_list)]
        
        self.regime_mapping = mapping
        self.regime_stats = regime_stats
    
    # ═══════════════════════════════════════════════════════════════
    # PRÉDICTION ET ANALYSE
    # ═══════════════════════════════════════════════════════════════
    
    def predict(self, df: pd.DataFrame) -> HMMResult:
        """
        Prédit le régime actuel et les probabilités
        
        Returns:
            HMMResult avec état actuel et historique
        """
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Préparer les features
        features = self._prepare_features(df)
        features_scaled = self.scaler.transform(features)
        
        # Prédire les états cachés
        hidden_states = self.model.predict(features_scaled)
        
        # Probabilités des états pour chaque observation
        state_probs = self.model.predict_proba(features_scaled)
        
        # État actuel (dernier)
        current_state = hidden_states[-1]
        current_probs = state_probs[-1]
        
        # Calculer la durée estimée dans le régime actuel
        duration = self._estimate_regime_duration(hidden_states, current_state)
        
        # Probabilités de transition depuis l'état actuel
        transition_probs = {
            self.regime_mapping[i].value: float(self.model.transmat_[current_state, i])
            for i in range(self.n_regimes)
            if i in self.regime_mapping
        }
        
        # Construire l'état actuel
        current_regime = RegimeState(
            regime=self.regime_mapping[current_state],
            probability=float(current_probs[current_state]),
            duration_estimate=duration,
            transition_probs=transition_probs
        )
        
        # Statistiques par régime
        regime_statistics = {}
        for state, regime_enum in self.regime_mapping.items():
            if state in self.regime_stats:
                regime_statistics[regime_enum.value] = {
                    "avg_return_daily": self.regime_stats[state]["avg_return"],
                    "avg_return_annual": self.regime_stats[state]["avg_return"] * 252,
                    "avg_volatility": self.regime_stats[state]["avg_volatility"],
                    "frequency": self.regime_stats[state]["count"] / len(hidden_states),
                    "current_probability": float(current_probs[state])
                }
        
        # Métriques du modèle
        model_metrics = {
            "confidence": float(current_probs.max()),
            "entropy": float(-np.sum(current_probs * np.log(current_probs + 1e-10))),
            "regime_stability": self._compute_regime_stability(hidden_states)
        }
        
        return HMMResult(
            current_regime=current_regime,
            regime_history=hidden_states.tolist(),
            regime_probabilities=state_probs,
            transition_matrix=self.model.transmat_,
            regime_statistics=regime_statistics,
            model_metrics=model_metrics
        )
    
    def _estimate_regime_duration(
        self, 
        hidden_states: np.ndarray, 
        current_state: int
    ) -> int:
        """Estime la durée restante dans le régime actuel"""
        
        # Compter les durées historiques de ce régime
        durations = []
        count = 0
        
        for state in hidden_states:
            if state == current_state:
                count += 1
            else:
                if count > 0:
                    durations.append(count)
                count = 0
        
        if count > 0:
            durations.append(count)
        
        if not durations:
            return 20  # Valeur par défaut
        
        # Durée médiane historique
        median_duration = int(np.median(durations))
        
        # Durée actuelle
        current_duration = 0
        for state in reversed(hidden_states):
            if state == current_state:
                current_duration += 1
            else:
                break
        
        # Estimation: durée médiane - durée déjà passée
        estimated_remaining = max(1, median_duration - current_duration)
        
        return estimated_remaining
    
    def _compute_regime_stability(self, hidden_states: np.ndarray) -> float:
        """Calcule un score de stabilité du régime (0-1)"""
        
        # Compter les transitions
        transitions = np.sum(hidden_states[1:] != hidden_states[:-1])
        
        # Normaliser par le nombre d'observations
        transition_rate = transitions / len(hidden_states)
        
        # Stabilité = 1 - taux de transition (normalisé)
        stability = 1 - min(1, transition_rate * 10)
        
        return float(stability)
    
    # ═══════════════════════════════════════════════════════════════
    # PRÉDICTION DE TRANSITIONS
    # ═══════════════════════════════════════════════════════════════
    
    def predict_regime_transition(
        self, 
        current_result: HMMResult,
        horizon_days: int = 20
    ) -> Dict[str, float]:
        """
        Prédit les probabilités de régime à un horizon donné
        
        Utilise la matrice de transition élevée à la puissance n
        """
        
        if horizon_days <= 0:
            return {
                regime.value: float(prob)
                for regime, prob in zip(
                    self.regime_mapping.values(),
                    current_result.regime_probabilities[-1]
                )
            }
        
        # Élever la matrice de transition à la puissance n
        transition_n = np.linalg.matrix_power(
            current_result.transition_matrix, 
            horizon_days
        )
        
        # Probabilités actuelles
        current_probs = current_result.regime_probabilities[-1]
        
        # Probabilités futures
        future_probs = current_probs @ transition_n
        
        return {
            self.regime_mapping[i].value: float(future_probs[i])
            for i in range(self.n_regimes)
            if i in self.regime_mapping
        }

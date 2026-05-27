"""
ZoneSignal - Modèles de données marché
Architecture SQL institutionnelle
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Date, 
    JSON, ForeignKey, Index, Boolean, Text, Enum
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
import enum


class DirectionEnum(enum.Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class IntensityEnum(enum.Enum):
    FAIBLE = "faible"
    MODEREE = "moderee"
    FORTE = "forte"
    EXTREME = "extreme"


class RegimeEnum(enum.Enum):
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


# ═══════════════════════════════════════════════════════════════════════
# TABLE: market_history - Historiques complets des indices
# ═══════════════════════════════════════════════════════════════════════

class MarketHistory(Base):
    __tablename__ = "market_history"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    
    # OHLCV
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)
    volume = Column(Float)
    
    # Métriques dérivées
    daily_return = Column(Float)
    log_return = Column(Float)
    volatility_20d = Column(Float)
    volatility_60d = Column(Float)
    atr_14 = Column(Float)
    
    # Breadth & Internals
    advance_decline_ratio = Column(Float)
    new_highs_lows_ratio = Column(Float)
    percent_above_ma50 = Column(Float)
    percent_above_ma200 = Column(Float)
    
    # Liquidité
    dollar_volume = Column(Float)
    relative_volume = Column(Float)
    
    # Drawdown
    drawdown_from_high = Column(Float)
    days_since_high = Column(Integer)
    
    # Métadonnées
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    
    __table_args__ = (
        Index('ix_market_history_symbol_date', 'symbol', 'date', unique=True),
    )


# ═══════════════════════════════════════════════════════════════════════
# TABLE: macro_indicators - Données macroéconomiques
# ═══════════════════════════════════════════════════════════════════════

class MacroIndicator(Base):
    __tablename__ = "macro_indicators"
    
    id = Column(Integer, primary_key=True, index=True)
    indicator_code = Column(String(50), nullable=False, index=True)
    indicator_name = Column(String(200))
    date = Column(Date, nullable=False, index=True)
    value = Column(Float)
    previous_value = Column(Float)
    change_pct = Column(Float)
    
    # Catégorisation
    category = Column(String(50))  # inflation, employment, growth, monetary, etc.
    frequency = Column(String(20))  # daily, weekly, monthly, quarterly
    country = Column(String(10), default="US")
    
    # Impact marché estimé
    market_impact_score = Column(Float)
    
    created_at = Column(DateTime, server_default=func.now())
    
    __table_args__ = (
        Index('ix_macro_indicator_date', 'indicator_code', 'date', unique=True),
    )


# ═══════════════════════════════════════════════════════════════════════
# TABLE: forecasts - Prévisions générées
# ═══════════════════════════════════════════════════════════════════════

class Forecast(Base):
    __tablename__ = "forecasts"
    
    id = Column(Integer, primary_key=True, index=True)
    forecast_id = Column(String(50), unique=True, nullable=False)
    
    # Identifiants
    symbol = Column(String(20), nullable=False, index=True)
    horizon = Column(String(10), nullable=False)  # 1D, 1W, 1M, 3M, 6M, 1Y
    
    # Dates
    forecast_date = Column(Date, nullable=False, index=True)
    target_date = Column(Date, nullable=False)
    
    # Prévision principale
    direction = Column(Enum(DirectionEnum), nullable=False)
    intensity = Column(Enum(IntensityEnum), nullable=False)
    
    # Probabilités
    prob_bullish = Column(Float, nullable=False)
    prob_bearish = Column(Float, nullable=False)
    prob_neutral = Column(Float, nullable=False)
    
    # Estimations quantitatives
    expected_return = Column(Float)
    return_lower_bound = Column(Float)  # 10th percentile
    return_upper_bound = Column(Float)  # 90th percentile
    expected_volatility = Column(Float)
    max_drawdown_estimate = Column(Float)
    
    # Régime détecté
    regime = Column(Enum(RegimeEnum))
    regime_confidence = Column(Float)
    regime_transition_prob = Column(Float)
    
    # Scores de confiance
    confidence_score = Column(Float, nullable=False)
    stability_score = Column(Float)
    uncertainty_score = Column(Float)
    robustness_score = Column(Float)
    
    # Contribution des modèles (JSON)
    model_contributions = Column(JSON)
    # Ex: {"hmm": 0.25, "bayesian": 0.20, "monte_carlo": 0.15, ...}
    
    # Facteurs clés (JSON)
    key_factors = Column(JSON)
    # Ex: {"macro_score": 0.6, "seasonality": 0.3, "momentum": -0.2, ...}
    
    # Scénarios alternatifs (JSON)
    alternative_scenarios = Column(JSON)
    # Ex: [{"scenario": "Fed hawkish", "prob": 0.3, "impact": -0.05}, ...]
    
    # Métadonnées
    model_version = Column(String(20))
    computation_time_ms = Column(Integer)
    created_at = Column(DateTime, server_default=func.now())
    
    # Relations
    results = relationship("ForecastResult", back_populates="forecast")
    
    __table_args__ = (
        Index('ix_forecast_symbol_horizon_date', 'symbol', 'horizon', 'forecast_date'),
    )


# ═══════════════════════════════════════════════════════════════════════
# TABLE: forecast_results - Évaluation des prévisions (réalité vs prévision)
# ═══════════════════════════════════════════════════════════════════════

class ForecastResult(Base):
    __tablename__ = "forecast_results"
    
    id = Column(Integer, primary_key=True, index=True)
    forecast_id = Column(String(50), ForeignKey("forecasts.forecast_id"), nullable=False)
    
    # Résultat réel
    realized_return = Column(Float)
    realized_direction = Column(Enum(DirectionEnum))
    realized_volatility = Column(Float)
    realized_max_drawdown = Column(Float)
    
    # Métriques d'erreur
    return_error = Column(Float)  # predicted - realized
    return_error_abs = Column(Float)
    directional_accuracy = Column(Boolean)  # True si direction correcte
    
    # Calibration probabiliste
    probability_error = Column(Float)  # Brier score component
    calibration_bucket = Column(String(20))  # ex: "70-80%"
    
    # Score global
    forecast_score = Column(Float)  # Score composite de qualité
    
    # Analyse d'erreur (JSON)
    error_attribution = Column(JSON)
    # Ex: {"regime_miss": 0.4, "macro_surprise": 0.3, "vol_underestimate": 0.3}
    
    evaluated_at = Column(DateTime, server_default=func.now())
    
    # Relations
    forecast = relationship("Forecast", back_populates="results")


# ═══════════════════════════════════════════════════════════════════════
# TABLE: adaptive_learning - Poids adaptatifs des modèles
# ═══════════════════════════════════════════════════════════════════════

class AdaptiveLearning(Base):
    __tablename__ = "adaptive_learning"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Contexte
    symbol = Column(String(20), index=True)  # Null = global
    regime = Column(Enum(RegimeEnum))  # Null = all regimes
    horizon = Column(String(10))  # Null = all horizons
    
    # Poids des modèles (JSON)
    model_weights = Column(JSON, nullable=False)
    # Ex: {"hmm": 0.18, "bayesian": 0.22, "monte_carlo": 0.15, ...}
    
    # Poids des facteurs (JSON)
    factor_weights = Column(JSON, nullable=False)
    # Ex: {"macro": 0.25, "momentum": 0.20, "seasonality": 0.15, ...}
    
    # Performance tracking
    model_performance = Column(JSON)
    # Ex: {"hmm": {"accuracy": 0.65, "sharpe": 1.2}, ...}
    
    # Ajustements de confiance
    confidence_adjustments = Column(JSON)
    # Ex: {"base_adjustment": -0.05, "regime_adjustment": 0.02}
    
    # Métadonnées
    samples_count = Column(Integer)
    last_updated = Column(DateTime, server_default=func.now())
    valid_from = Column(Date)
    valid_to = Column(Date)
    
    __table_args__ = (
        Index('ix_adaptive_learning_context', 'symbol', 'regime', 'horizon'),
    )


# ═══════════════════════════════════════════════════════════════════════
# TABLE: seasonality_patterns - Patterns de saisonnalité
# ═══════════════════════════════════════════════════════════════════════

class SeasonalityPattern(Base):
    __tablename__ = "seasonality_patterns"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    pattern_type = Column(String(50), nullable=False)
    # monthly, quarterly, election_cycle, presidential_cycle, etc.
    
    # Identifiant du pattern
    period_key = Column(String(50))  # ex: "january", "Q1", "year_1_of_cycle"
    
    # Statistiques historiques
    avg_return = Column(Float)
    median_return = Column(Float)
    win_rate = Column(Float)
    avg_volatility = Column(Float)
    sample_size = Column(Integer)
    
    # Significativité statistique
    t_statistic = Column(Float)
    p_value = Column(Float)
    is_significant = Column(Boolean)
    
    # Métadonnées
    start_year = Column(Integer)
    end_year = Column(Integer)
    last_computed = Column(DateTime, server_default=func.now())
    
    __table_args__ = (
        Index('ix_seasonality_pattern', 'symbol', 'pattern_type', 'period_key', unique=True),
    )


# ═══════════════════════════════════════════════════════════════════════
# TABLE: backtest_results - Résultats de backtesting
# ═══════════════════════════════════════════════════════════════════════

class BacktestResult(Base):
    __tablename__ = "backtest_results"
    
    id = Column(Integer, primary_key=True, index=True)
    backtest_id = Column(String(50), unique=True, nullable=False)
    
    # Configuration
    symbol = Column(String(20), nullable=False)
    horizon = Column(String(10), nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    model_config = Column(JSON)
    
    # Métriques de performance
    total_forecasts = Column(Integer)
    directional_accuracy = Column(Float)
    probabilistic_accuracy = Column(Float)  # Brier score inverse
    confidence_calibration = Column(Float)
    
    # Métriques style trading
    signal_sharpe_ratio = Column(Float)
    max_consecutive_errors = Column(Integer)
    avg_confidence_when_correct = Column(Float)
    avg_confidence_when_wrong = Column(Float)
    
    # Métriques par régime (JSON)
    regime_performance = Column(JSON)
    
    # Métriques drawdown
    drawdown_prediction_quality = Column(Float)
    regime_detection_accuracy = Column(Float)
    
    # Métadonnées
    computation_time_sec = Column(Float)
    created_at = Column(DateTime, server_default=func.now())

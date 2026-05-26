from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Enum, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from app.database import Base
from datetime import datetime
import enum
import uuid


class DirectionEnum(str, enum.Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class IntensityEnum(str, enum.Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    EXTREME = "extreme"


class RegimeEnum(str, enum.Enum):
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


class Forecast(Base):
    __tablename__ = "forecasts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(20), nullable=False, index=True)
    horizon = Column(String(10), nullable=False, index=True)  # 1D, 1W, 1M, 3M, 6M, 1Y
    
    # Timestamps
    forecast_date = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    target_date = Column(Date, nullable=False, index=True)
    
    # Core Predictions
    direction = Column(Enum(DirectionEnum), nullable=False)
    intensity = Column(Enum(IntensityEnum), nullable=False)
    
    # Probabilities
    prob_bullish = Column(Float, nullable=False)
    prob_bearish = Column(Float, nullable=False)
    prob_neutral = Column(Float, nullable=False)
    
    # Expected Returns
    expected_return = Column(Float, nullable=True)
    expected_return_5th = Column(Float, nullable=True)  # 5th percentile
    expected_return_25th = Column(Float, nullable=True)
    expected_return_75th = Column(Float, nullable=True)
    expected_return_95th = Column(Float, nullable=True)  # 95th percentile
    
    # Regime Detection
    detected_regime = Column(Enum(RegimeEnum), nullable=False)
    regime_probability = Column(Float, nullable=True)
    regime_stability = Column(Float, nullable=True)
    
    # Confidence Metrics
    confidence_score = Column(Float, nullable=False)
    stability_score = Column(Float, nullable=False)
    uncertainty_score = Column(Float, nullable=False)
    signal_robustness = Column(Float, nullable=False)
    
    # Model Contributions
    model_weights = Column(JSONB, nullable=True)
    factor_contributions = Column(JSONB, nullable=True)
    
    # Scenarios
    scenarios = Column(JSONB, nullable=True)
    stress_scenarios = Column(JSONB, nullable=True)
    
    # Metadata
    model_version = Column(String(20), nullable=True)
    computation_time_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to results
    result = relationship("ForecastResult", back_populates="forecast", uselist=False)
    
    __table_args__ = (
        Index('ix_forecast_symbol_horizon_date', 'symbol', 'horizon', 'forecast_date'),
    )


class ForecastResult(Base):
    __tablename__ = "forecast_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    forecast_id = Column(UUID(as_uuid=True), ForeignKey('forecasts.id'), nullable=False, unique=True)
    
    # Realized Outcomes
    realized_return = Column(Float, nullable=True)
    realized_direction = Column(Enum(DirectionEnum), nullable=True)
    realized_volatility = Column(Float, nullable=True)
    
    # Accuracy Metrics
    directional_accuracy = Column(Float, nullable=True)  # 1 if correct, 0 if not
    return_error = Column(Float, nullable=True)  # Expected - Realized
    return_error_abs = Column(Float, nullable=True)
    probabilistic_error = Column(Float, nullable=True)  # Brier score component
    
    # Calibration
    confidence_calibration = Column(Float, nullable=True)
    regime_accuracy = Column(Float, nullable=True)
    
    # Timing
    evaluated_at = Column(DateTime, nullable=True)
    
    forecast = relationship("Forecast", back_populates="result")

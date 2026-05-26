"""
Pydantic Schemas for Forecast API
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime, date
from uuid import UUID
from enum import Enum


class DirectionEnum(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class IntensityEnum(str, Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    EXTREME = "extreme"


class RegimeEnum(str, Enum):
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


class ForecastCreate(BaseModel):
    symbol: str
    horizon: str


class BatchForecastRequest(BaseModel):
    symbols: List[str] = Field(default=["SPX", "NDX", "DJI", "RUT"])
    horizons: List[str] = Field(default=["1D", "1W", "1M", "3M", "6M", "1Y"])


class ScenarioSchema(BaseModel):
    name: str
    probability: float
    expected_return: float
    worst_case: Optional[float] = None


class ForecastResponse(BaseModel):
    id: UUID
    symbol: str
    horizon: str
    forecast_date: datetime
    target_date: date
    
    # Direction
    direction: DirectionEnum
    intensity: IntensityEnum
    
    # Probabilities
    prob_bullish: float
    prob_bearish: float
    prob_neutral: float
    
    # Expected Returns
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
    
    # Model Details
    model_weights: Dict[str, float]
    factor_contributions: Optional[Dict[str, float]] = None
    
    # Scenarios
    scenarios: Optional[Dict[str, Dict]] = None
    stress_scenarios: Optional[List[ScenarioSchema]] = None
    
    class Config:
        from_attributes = True


class ForecastListResponse(BaseModel):
    forecasts: List[ForecastResponse]
    total: int
    limit: int
    offset: int


class ForecastResultResponse(BaseModel):
    id: UUID
    forecast_id: UUID
    
    realized_return: Optional[float]
    realized_direction: Optional[DirectionEnum]
    realized_volatility: Optional[float]
    
    directional_accuracy: Optional[float]
    return_error: Optional[float]
    return_error_abs: Optional[float]
    probabilistic_error: Optional[float]
    
    confidence_calibration: Optional[float]
    regime_accuracy: Optional[float]
    
    evaluated_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class PerformanceSummary(BaseModel):
    total_forecasts: int
    evaluated_forecasts: int
    
    directional_accuracy: float
    average_return_error: float
    average_confidence: float
    confidence_calibration: float
    
    accuracy_by_horizon: Dict[str, float]
    accuracy_by_regime: Dict[str, float]
    
    sharpe_like_score: float
    drawdown_prediction_quality: float

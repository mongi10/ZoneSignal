from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.dialects.postgresql import JSONB, UUID
from app.database import Base
from datetime import datetime
import uuid


class AdaptiveLearningState(Base):
    __tablename__ = "adaptive_learning_state"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(20), nullable=False, index=True)
    horizon = Column(String(10), nullable=False, index=True)
    
    # Current Model Weights
    factor_weights = Column(JSONB, nullable=False)
    model_weights = Column(JSONB, nullable=False)
    
    # Performance by Regime
    regime_performance = Column(JSONB, nullable=True)
    
    # Learning Metrics
    learning_rate = Column(Float, default=0.01)
    momentum = Column(Float, default=0.9)
    weight_decay = Column(Float, default=0.001)
    
    # Historical Performance
    rolling_accuracy_30d = Column(Float, nullable=True)
    rolling_accuracy_90d = Column(Float, nullable=True)
    rolling_sharpe = Column(Float, nullable=True)
    
    # Confidence Adjustments
    confidence_calibration_factor = Column(Float, default=1.0)
    uncertainty_scaling = Column(Float, default=1.0)
    
    # State Management
    version = Column(Integer, default=1)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_recalibration = Column(DateTime, nullable=True)
    
    __table_args__ = (
        Index('ix_adaptive_learning_symbol_horizon', 'symbol', 'horizon', unique=True),
    )


class ModelPerformanceLog(Base):
    __tablename__ = "model_performance_log"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Context
    symbol = Column(String(20), nullable=False, index=True)
    horizon = Column(String(10), nullable=False)
    detected_regime = Column(String(50), nullable=True)
    date = Column(DateTime, nullable=False, index=True)
    
    # Model-level Performance
    model_name = Column(String(50), nullable=False, index=True)
    model_weight = Column(Float, nullable=False)
    model_prediction = Column(Float, nullable=True)
    model_confidence = Column(Float, nullable=True)
    
    # Realized
    realized_outcome = Column(Float, nullable=True)
    model_error = Column(Float, nullable=True)
    directional_hit = Column(Float, nullable=True)
    
    # Attribution
    contribution_to_ensemble = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class LearningEvent(Base):
    __tablename__ = "learning_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    event_type = Column(String(50), nullable=False)  # weight_update, recalibration, regime_shift
    symbol = Column(String(20), nullable=True)
    horizon = Column(String(10), nullable=True)
    
    # Changes
    previous_state = Column(JSONB, nullable=True)
    new_state = Column(JSONB, nullable=True)
    
    # Trigger
    trigger_reason = Column(String(200), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)

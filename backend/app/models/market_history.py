from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Index, JSON, BigInteger
from sqlalchemy.dialects.postgresql import JSONB
from app.database import Base
from datetime import datetime


class MarketHistory(Base):
    __tablename__ = "market_history"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    
    # OHLCV
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(BigInteger, nullable=True)
    adjusted_close = Column(Float, nullable=True)
    
    # Derived Metrics
    daily_return = Column(Float, nullable=True)
    log_return = Column(Float, nullable=True)
    volatility_20d = Column(Float, nullable=True)
    volatility_60d = Column(Float, nullable=True)
    atr_14 = Column(Float, nullable=True)
    
    # Market Internals
    breadth_advance_decline = Column(Float, nullable=True)
    breadth_new_highs_lows = Column(Float, nullable=True)
    volume_ratio = Column(Float, nullable=True)
    
    # Liquidity Metrics
    liquidity_score = Column(Float, nullable=True)
    spread_estimate = Column(Float, nullable=True)
    
    # Technical Structure
    rsi_14 = Column(Float, nullable=True)
    macd_signal = Column(Float, nullable=True)
    bollinger_position = Column(Float, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    data_source = Column(String(50), nullable=True)
    
    __table_args__ = (
        Index('ix_market_history_symbol_date', 'symbol', 'date', unique=True),
        Index('ix_market_history_date_desc', 'date', postgresql_using='btree'),
    )


class MacroIndicator(Base):
    __tablename__ = "macro_indicators"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    indicator_code = Column(String(50), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    value = Column(Float, nullable=False)
    
    # Transformations
    yoy_change = Column(Float, nullable=True)
    mom_change = Column(Float, nullable=True)
    z_score = Column(Float, nullable=True)
    percentile = Column(Float, nullable=True)
    
    # Metadata
    source = Column(String(50), nullable=True)
    release_date = Column(Date, nullable=True)
    revision_number = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_macro_indicator_code_date', 'indicator_code', 'date', unique=True),
    )


class VolatilityStructure(Base):
    __tablename__ = "volatility_structure"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    
    # VIX-style metrics
    implied_vol_30d = Column(Float, nullable=True)
    implied_vol_60d = Column(Float, nullable=True)
    implied_vol_90d = Column(Float, nullable=True)
    realized_vol_20d = Column(Float, nullable=True)
    vol_risk_premium = Column(Float, nullable=True)
    
    # Term Structure
    vix_term_structure = Column(JSONB, nullable=True)
    contango_backwardation = Column(Float, nullable=True)
    
    # Skew
    put_call_skew = Column(Float, nullable=True)
    skew_index = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_vol_structure_symbol_date', 'symbol', 'date', unique=True),
    )

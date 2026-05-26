from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional
import os


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/zoneflow_forecast"
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # API Keys (optional, ZoneFlow priority)
    ALPHA_VANTAGE_KEY: Optional[str] = None
    POLYGON_KEY: Optional[str] = None
    FRED_KEY: Optional[str] = None
    
    # ZoneFlow Integration
    ZONEFLOW_DATA_PATH: str = "./zoneflow_data"
    ZONEFLOW_CACHE_PATH: str = "./zoneflow_cache"
    USE_ZONEFLOW_FEEDS: bool = True
    
    # Engine Configuration
    MONTE_CARLO_SIMULATIONS: int = 10000
    HMM_STATES: int = 4
    FORECAST_HORIZONS: list = ["1D", "1W", "1M", "3M", "6M", "1Y"]
    
    # Scheduler
    DAILY_UPDATE_HOUR: int = 18
    WEEKLY_RECALIBRATION_DAY: int = 6  # Sunday
    
    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()

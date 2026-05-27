"""
ZoneSignal - Configuration centrale
Moteur Quantitatif Institutionnel de Prévision d'Indices Boursiers
"""

from pydantic_settings import BaseSettings
from typing import List
from functools import lru_cache


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "ZoneSignal"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Database
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/zonesignal"
    
    # Redis Cache
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # API Keys (sources de données)
    ALPHA_VANTAGE_API_KEY: str = ""
    POLYGON_API_KEY: str = ""
    FRED_API_KEY: str = ""
    
    # Indices supportés
    SUPPORTED_INDICES: List[str] = [
        "^GSPC",   # S&P 500
        "^DJI",    # Dow Jones
        "^IXIC",   # NASDAQ
        "^RUT",    # Russell 2000
        "^VIX",    # VIX
        "^STOXX50E",  # Euro Stoxx 50
        "^FTSE",   # FTSE 100
        "^N225",   # Nikkei 225
        "^HSI",    # Hang Seng
        "^FCHI",   # CAC 40
        "^GDAXI",  # DAX
    ]
    
    # Horizons de prévision (en jours)
    FORECAST_HORIZONS: dict = {
        "1D": 1,
        "1W": 5,
        "1M": 21,
        "3M": 63,
        "6M": 126,
        "1Y": 252
    }
    
    # Monte Carlo
    MONTE_CARLO_SIMULATIONS: int = 10000
    
    # HMM Régimes
    HMM_N_REGIMES: int = 4
    
    # Adaptive Learning
    LEARNING_LOOKBACK_DAYS: int = 252
    MIN_CONFIDENCE_THRESHOLD: float = 0.3
    
    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

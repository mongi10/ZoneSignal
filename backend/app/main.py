"""
FastAPI Application Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.config import get_settings
from app.database import engine, Base
from app.api.routes import forecasts, market_data, backtesting, learning
from app.scheduler.jobs import start_scheduler, shutdown_scheduler

settings = get_settings()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler for startup/shutdown events.
    """
    # Startup
    logger.info("Starting Forecast Engine...")
    
    # Create database tables
    Base.metadata.create_all(bind=engine)
    
    # Start scheduler for periodic tasks
    start_scheduler()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Forecast Engine...")
    shutdown_scheduler()


app = FastAPI(
    title="ZoneFlow Forecast Engine",
    description="Institutional-grade probabilistic market forecasting system",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(forecasts.router, prefix="/api/v1")
app.include_router(market_data.router, prefix="/api/v1")
app.include_router(backtesting.router, prefix="/api/v1")
app.include_router(learning.router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/")
async def root():
    return {
        "name": "ZoneFlow Forecast Engine",
        "version": "1.0.0",
        "endpoints": {
            "forecasts": "/api/v1/forecasts",
            "market_data": "/api/v1/market-data",
            "backtesting": "/api/v1/backtesting",
            "learning": "/api/v1/learning",
            "docs": "/docs"
        }
    }

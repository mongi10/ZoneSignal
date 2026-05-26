"""
API Routes for Forecast Operations
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, date
from uuid import UUID

from app.database import get_db
from app.schemas.forecast import (
    ForecastCreate, ForecastResponse, ForecastListResponse,
    ForecastResultResponse, BatchForecastRequest
)
from app.services.forecast_service import ForecastService
from app.models.forecasts import Forecast, ForecastResult

router = APIRouter(prefix="/forecasts", tags=["Forecasts"])


@router.post("/generate", response_model=ForecastResponse)
async def generate_forecast(
    symbol: str = Query(..., description="Market symbol (e.g., SPX, NDX, DJI)"),
    horizon: str = Query(..., description="Forecast horizon (1D, 1W, 1M, 3M, 6M, 1Y)"),
    db: Session = Depends(get_db)
):
    """
    Generate a new probabilistic forecast for the specified symbol and horizon.
    """
    service = ForecastService(db)
    
    try:
        forecast = await service.generate_forecast(symbol, horizon)
        return ForecastResponse.from_orm(forecast)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")


@router.post("/generate-batch", response_model=List[ForecastResponse])
async def generate_batch_forecasts(
    request: BatchForecastRequest,
    db: Session = Depends(get_db)
):
    """
    Generate forecasts for multiple symbols and horizons.
    """
    service = ForecastService(db)
    
    forecasts = []
    for symbol in request.symbols:
        for horizon in request.horizons:
            try:
                forecast = await service.generate_forecast(symbol, horizon)
                forecasts.append(ForecastResponse.from_orm(forecast))
            except Exception as e:
                # Log error but continue with other forecasts
                continue
                
    return forecasts


@router.get("/latest/{symbol}", response_model=List[ForecastResponse])
async def get_latest_forecasts(
    symbol: str,
    db: Session = Depends(get_db)
):
    """
    Get the latest forecasts for a symbol across all horizons.
    """
    service = ForecastService(db)
    forecasts = await service.get_latest_forecasts(symbol)
    return [ForecastResponse.from_orm(f) for f in forecasts]


@router.get("/history/{symbol}", response_model=ForecastListResponse)
async def get_forecast_history(
    symbol: str,
    horizon: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit: int = Query(100, le=1000),
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    Get historical forecasts with optional filtering.
    """
    service = ForecastService(db)
    forecasts, total = await service.get_forecast_history(
        symbol, horizon, start_date, end_date, limit, offset
    )
    
    return ForecastListResponse(
        forecasts=[ForecastResponse.from_orm(f) for f in forecasts],
        total=total,
        limit=limit,
        offset=offset
    )


@router.get("/{forecast_id}", response_model=ForecastResponse)
async def get_forecast(
    forecast_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get a specific forecast by ID.
    """
    service = ForecastService(db)
    forecast = await service.get_forecast(forecast_id)
    
    if not forecast:
        raise HTTPException(status_code=404, detail="Forecast not found")
        
    return ForecastResponse.from_orm(forecast)


@router.get("/{forecast_id}/result", response_model=ForecastResultResponse)
async def get_forecast_result(
    forecast_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get the evaluation result for a forecast (if available).
    """
    service = ForecastService(db)
    result = await service.get_forecast_result(forecast_id)
    
    if not result:
        raise HTTPException(
            status_code=404, 
            detail="Result not available (forecast may not have reached target date)"
        )
        
    return ForecastResultResponse.from_orm(result)


@router.get("/performance/summary", response_model=dict)
async def get_performance_summary(
    symbol: Optional[str] = None,
    horizon: Optional[str] = None,
    days: int = Query(90, description="Number of days to analyze"),
    db: Session = Depends(get_db)
):
    """
    Get aggregated performance metrics for forecasts.
    """
    service = ForecastService(db)
    summary = await service.get_performance_summary(symbol, horizon, days)
    return summary


@router.get("/regimes/current")
async def get_current_regimes(
    symbols: List[str] = Query(default=["SPX", "NDX", "DJI"]),
    db: Session = Depends(get_db)
):
    """
    Get current detected market regime for multiple symbols.
    """
    service = ForecastService(db)
    regimes = {}
    
    for symbol in symbols:
        regime_info = await service.get_current_regime(symbol)
        regimes[symbol] = regime_info
        
    return regimes

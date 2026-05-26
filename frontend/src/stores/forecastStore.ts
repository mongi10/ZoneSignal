import { create } from 'zustand';
import { devtools, subscribeWithSelector } from 'zustand/middleware';
import { Forecast, Horizon, PerformanceMetrics, ForecastResult } from '../types/forecast';
import { forecastApi } from '../services/api';

interface ForecastState {
  // Data
  forecasts: Record<string, Forecast[]>; // Keyed by symbol
  latestForecasts: Record<string, Record<Horizon, Forecast | null>>;
  results: Record<string, ForecastResult>;
  performance: PerformanceMetrics | null;
  
  // UI State
  selectedSymbol: string;
  selectedHorizon: Horizon;
  isLoading: boolean;
  error: string | null;
  
  // Actions
  setSelectedSymbol: (symbol: string) => void;
  setSelectedHorizon: (horizon: Horizon) => void;
  fetchLatestForecasts: (symbol: string) => Promise<void>;
  generateForecast: (symbol: string, horizon: Horizon) => Promise<Forecast>;
  fetchForecastHistory: (symbol: string, horizon?: Horizon) => Promise<void>;
  fetchPerformance: (symbol?: string, horizon?: Horizon) => Promise<void>;
}

export const useForecastStore = create<ForecastState>()(
  devtools(
    subscribeWithSelector((set, get) => ({
      // Initial state
      forecasts: {},
      latestForecasts: {},
      results: {},
      performance: null,
      selectedSymbol: 'SPX',
      selectedHorizon: '1M',
      isLoading: false,
      error: null,

      setSelectedSymbol: (symbol) => {
        set({ selectedSymbol: symbol });
        get().fetchLatestForecasts(symbol);
      },

      setSelectedHorizon: (horizon) => {
        set({ selectedHorizon: horizon });
      },

      fetchLatestForecasts: async (symbol) => {
        set({ isLoading: true, error: null });
        try {
          const forecasts = await forecastApi.getLatest(symbol);
          
          const byHorizon: Record<Horizon, Forecast | null> = {
            '1D': null, '1W': null, '1M': null, 
            '3M': null, '6M': null, '1Y': null
          };
          
          forecasts.forEach(f => {
            byHorizon[f.horizon] = f;
          });
          
          set(state => ({
            latestForecasts: {
              ...state.latestForecasts,
              [symbol]: byHorizon
            },
            isLoading: false
          }));
        } catch (error) {
          set({ error: (error as Error).message, isLoading: false });
        }
      },

      generateForecast: async (symbol, horizon) => {
        set({ isLoading: true, error: null });
        try {
          const forecast = await forecastApi.generate(symbol, horizon);
          
          set(state => ({
            latestForecasts: {
              ...state.latestForecasts,
              [symbol]: {
                ...state.latestForecasts[symbol],
                [horizon]: forecast
              }
            },
            isLoading: false
          }));
          
          return forecast;
        } catch (error) {
          set({ error: (error as Error).message, isLoading: false });
          throw error;
        }
      },

      fetchForecastHistory: async (symbol, horizon) => {
        set({ isLoading: true, error: null });
        try {
          const { forecasts } = await forecastApi.getHistory(symbol, horizon);
          
          set(state => ({
            forecasts: {
              ...state.forecasts,
              [symbol]: forecasts
            },
            isLoading: false
          }));
        } catch (error) {
          set({ error: (error as Error).message, isLoading: false });
        }
      },

      fetchPerformance: async (symbol, horizon) => {
        try {
          const performance = await forecastApi.getPerformance(symbol, horizon);
          set({ performance });
        } catch (error) {
          console.error('Failed to fetch performance:', error);
        }
      }
    })),
    { name: 'ForecastStore' }
  )
);

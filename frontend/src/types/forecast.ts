export type Direction = 'bullish' | 'bearish' | 'neutral';
export type Intensity = 'weak' | 'moderate' | 'strong' | 'extreme';
export type Regime = 
  | 'risk_on' 
  | 'risk_off' 
  | 'euphoria' 
  | 'panic' 
  | 'consolidation'
  | 'macro_stress'
  | 'reflation'
  | 'disinflation'
  | 'liquidity_expansion'
  | 'liquidity_contraction';

export type Horizon = '1D' | '1W' | '1M' | '3M' | '6M' | '1Y';

export interface Forecast {
  id: string;
  symbol: string;
  horizon: Horizon;
  forecast_date: string;
  target_date: string;
  
  direction: Direction;
  intensity: Intensity;
  
  prob_bullish: number;
  prob_bearish: number;
  prob_neutral: number;
  
  expected_return: number;
  expected_return_5th: number;
  expected_return_25th: number;
  expected_return_75th: number;
  expected_return_95th: number;
  
  detected_regime: Regime;
  regime_probability: number;
  regime_stability: number;
  
  confidence_score: number;
  stability_score: number;
  uncertainty_score: number;
  signal_robustness: number;
  
  model_weights: Record<string, number>;
  factor_contributions?: Record<string, number>;
  
  scenarios?: Record<string, { expected_return: number; probability: number }>;
  stress_scenarios?: StressScenario[];
}

export interface StressScenario {
  name: string;
  probability: number;
  expected_return: number;
  worst_case: number;
}

export interface ForecastResult {
  id: string;
  forecast_id: string;
  realized_return: number | null;
  realized_direction: Direction | null;
  directional_accuracy: number | null;
  return_error: number | null;
  probabilistic_error: number | null;
  evaluated_at: string | null;
}

export interface PerformanceMetrics {
  total_forecasts: number;
  evaluated_forecasts: number;
  directional_accuracy: number;
  average_return_error: number;
  average_confidence: number;
  confidence_calibration: number;
  accuracy_by_horizon: Record<Horizon, number>;
  accuracy_by_regime: Record<Regime, number>;
  sharpe_like_score: number;
}

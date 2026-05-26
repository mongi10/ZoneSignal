import React, { useEffect } from 'react';
import { motion } from 'framer-motion';
import { useForecastStore } from '../../stores/forecastStore';
import { ForecastCard } from '../widgets/ForecastCard';
import { RegimeIndicator } from '../widgets/RegimeIndicator';
import { ConfidenceMeter } from '../widgets/ConfidenceMeter';
import { ProbabilityChart } from '../charts/ProbabilityChart';
import { ProbabilisticCone } from '../charts/ProbabilisticCone';
import { HorizonSelector } from '../common/HorizonSelector';
import { SymbolSelector } from '../common/SymbolSelector';
import { Horizon } from '../../types/forecast';

const SYMBOLS = ['SPX', 'NDX', 'DJI', 'RUT', 'VIX'];
const HORIZONS: Horizon[] = ['1D', '1W', '1M', '3M', '6M', '1Y'];

export const ForecastDashboard: React.FC = () => {
  const {
    selectedSymbol,
    selectedHorizon,
    latestForecasts,
    isLoading,
    setSelectedSymbol,
    setSelectedHorizon,
    fetchLatestForecasts,
    generateForecast
  } = useForecastStore();

  useEffect(() => {
    fetchLatestForecasts(selectedSymbol);
  }, [selectedSymbol]);

  const currentForecast = latestForecasts[selectedSymbol]?.[selectedHorizon];

  const handleGenerateForecast = async () => {
    await generateForecast(selectedSymbol, selectedHorizon);
  };

  return (
    <div className="h-full bg-[#0a0a0f] text-gray-100 flex flex-col">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-2 border-b border-gray-800 bg-[#0f0f15]">
        <div className="flex items-center gap-4">
          <h1 className="text-lg font-semibold text-blue-400">
            FORECAST ENGINE
          </h1>
          <SymbolSelector
            symbols={SYMBOLS}
            selected={selectedSymbol}
            onSelect={setSelectedSymbol}
          />
        </div>
        <div className="flex items-center gap-4">
          <HorizonSelector
            horizons={HORIZONS}
            selected={selectedHorizon}
            onSelect={setSelectedHorizon}
          />
          <button
            onClick={handleGenerateForecast}
            disabled={isLoading}
            className="px-4 py-1.5 bg-blue-600 hover:bg-blue-500 rounded text-sm font-medium transition-colors disabled:opacity-50"
          >
            {isLoading ? 'Generating...' : 'Generate Forecast'}
          </button>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 grid grid-cols-12 gap-2 p-2 overflow-hidden">
        {/* Left Panel - Horizons Overview */}
        <aside className="col-span-2 bg-[#0f0f15] rounded border border-gray-800 overflow-y-auto">
          <div className="p-3 border-b border-gray-800">
            <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
              All Horizons
            </h2>
          </div>
          <div className="p-2 space-y-2">
            {HORIZONS.map(horizon => {
              const forecast = latestForecasts[selectedSymbol]?.[horizon];
              return (
                <motion.div
                  key={horizon}
                  whileHover={{ scale: 1.02 }}
                  onClick={() => setSelectedHorizon(horizon)}
                  className={`p-3 rounded cursor-pointer transition-colors ${
                    selectedHorizon === horizon
                      ? 'bg-blue-900/30 border border-blue-500/50'
                      : 'bg-gray-900/50 hover:bg-gray-800/50 border border-transparent'
                  }`}
                >
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-mono text-sm">{horizon}</span>
                    {forecast && (
                      <DirectionBadge direction={forecast.direction} />
                    )}
                  </div>
                  {forecast ? (
                    <div className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="text-gray-500">Expected</span>
                        <span className={forecast.expected_return >= 0 ? 'text-green-400' : 'text-red-400'}>
                          {(forecast.expected_return * 100).toFixed(2)}%
                        </span>
                      </div>
                      <div className="flex justify-between text-xs">
                        <span className="text-gray-500">Confidence</span>
                        <span className="text-blue-400">
                          {(forecast.confidence_score * 100).toFixed(0)}%
                        </span>
                      </div>
                      <ConfidenceBar value={forecast.confidence_score} />
                    </div>
                  ) : (
                    <div className="text-xs text-gray-500 text-center py-2">
                      No forecast
                    </div>
                  )}
                </motion.div>
              );
            })}
          </div>
        </aside>

        {/* Center Panel - Main Visualization */}
        <main className="col-span-7 flex flex-col gap-2">
          {/* Top Row - Charts */}
          <div className="flex-1 grid grid-cols-2 gap-2">
            {/* Probabilistic Cone */}
            <div className="bg-[#0f0f15] rounded border border-gray-800 p-3">
              <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
                Probabilistic Forecast Cone
              </h3>
              {currentForecast ? (
                <ProbabilisticCone forecast={currentForecast} />
              ) : (
                <EmptyState message="Generate a forecast to see the probability cone" />
              )}
            </div>

            {/* Probability Distribution */}
            <div className="bg-[#0f0f15] rounded border border-gray-800 p-3">
              <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
                Return Distribution
              </h3>
              {currentForecast ? (
                <ProbabilityChart forecast={currentForecast} />
              ) : (
                <EmptyState message="No distribution data available" />
              )}
            </div>
          </div>

          {/* Bottom Row - Scenarios */}
          <div className="h-48 bg-[#0f0f15] rounded border border-gray-800 p-3">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
              Scenario Analysis
            </h3>
            {currentForecast?.stress_scenarios ? (
              <div className="grid grid-cols-5 gap-2 h-[calc(100%-24px)]">
                {currentForecast.stress_scenarios.map((scenario, i) => (
                  <ScenarioCard key={i} scenario={scenario} />
                ))}
              </div>
            ) : (
              <EmptyState message="No scenario data" />
            )}
          </div>
        </main>

        {/* Right Panel - Details & Metrics */}
        <aside className="col-span-3 flex flex-col gap-2">
          {/* Current Forecast Summary */}
          <div className="bg-[#0f0f15] rounded border border-gray-800 p-3">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
              Forecast Summary
            </h3>
            {currentForecast ? (
              <ForecastCard forecast={currentForecast} />
            ) : (
              <EmptyState message="No forecast selected" />
            )}
          </div>

          {/* Regime Detection */}
          <div className="bg-[#0f0f15] rounded border border-gray-800 p-3">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
              Market Regime
            </h3>
            {currentForecast ? (
              <RegimeIndicator
                regime={currentForecast.detected_regime}
                probability={currentForecast.regime_probability}
                stability={currentForecast.regime_stability}
              />
            ) : (
              <EmptyState message="No regime data" />
            )}
          </div>

          {/* Confidence Metrics */}
          <div className="flex-1 bg-[#0f0f15] rounded border border-gray-800 p-3">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
              Confidence Metrics
            </h3>
            {currentForecast ? (
              <div className="space-y-4">
                <ConfidenceMeter
                  label="Confidence Score"
                  value={currentForecast.confidence_score}
                />
                <ConfidenceMeter
                  label="Signal Stability"
                  value={currentForecast.stability_score}
                />
                <ConfidenceMeter
                  label="Uncertainty"
                  value={currentForecast.uncertainty_score}
                  inverted
                />
                <ConfidenceMeter
                  label="Signal Robustness"
                  value={currentForecast.signal_robustness}
                />
              </div>
            ) : (
              <EmptyState message="No metrics available" />
            )}
          </div>

          {/* Model Weights */}
          <div className="bg-[#0f0f15] rounded border border-gray-800 p-3 h-40">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
              Model Contributions
            </h3>
            {currentForecast?.model_weights ? (
              <div className="space-y-1.5 overflow-y-auto h-[calc(100%-24px)]">
                {Object.entries(currentForecast.model_weights)
                  .sort(([, a], [, b]) => b - a)
                  .map(([model, weight]) => (
                    <div key={model} className="flex items-center gap-2">
                      <div className="w-24 text-xs text-gray-400 truncate">
                        {formatModelName(model)}
                      </div>
                      <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-blue-500 rounded-full"
                          style={{ width: `${weight * 100}%` }}
                        />
                      </div>
                      <div className="w-12 text-xs text-right text-gray-400">
                        {(weight * 100).toFixed(1)}%
                      </div>
                    </div>
                  ))}
              </div>
            ) : (
              <EmptyState message="No model data" />
            )}
          </div>
        </aside>
      </div>
    </div>
  );
};

// Helper Components
const DirectionBadge: React.FC<{ direction: string }> = ({ direction }) => {
  const colors = {
    bullish: 'bg-green-500/20 text-green-400 border-green-500/50',
    bearish: 'bg-red-500/20 text-red-400 border-red-500/50',
    neutral: 'bg-gray-500/20 text-gray-400 border-gray-500/50'
  };
  
  return (
    <span className={`px-2 py-0.5 text-xs rounded border ${colors[direction as keyof typeof colors]}`}>
      {direction.toUpperCase()}
    </span>
  );
};

const ConfidenceBar: React.FC<{ value: number }> = ({ value }) => {
  const color = value > 0.7 ? 'bg-green-500' : value > 0.4 ? 'bg-yellow-500' : 'bg-red-500';
  
  return (
    <div className="h-1 bg-gray-800 rounded-full overflow-hidden mt-1">
      <div className={`h-full ${color} rounded-full`} style={{ width: `${value * 100}%` }} />
    </div>
  );
};

const ScenarioCard: React.FC<{ scenario: any }> = ({ scenario }) => (
  <div className="bg-gray-900/50 rounded p-2 border border-gray-800">
    <div className="text-xs font-medium text-gray-300 mb-2 truncate">
      {scenario.name}
    </div>
    <div className="text-lg font-mono text-red-400">
      {(scenario.expected_return * 100).toFixed(1)}%
    </div>
    <div className="text-xs text-gray-500 mt-1">
      Prob: {(scenario.probability * 100).toFixed(1)}%
    </div>
  </div>
);

const EmptyState: React.FC<{ message: string }> = ({ message }) => (
  <div className="h-full flex items-center justify-center text-gray-500 text-sm">
    {message}
  </div>
);

const formatModelName = (name: string): string => {
  return name
    .replace(/_/g, ' ')
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

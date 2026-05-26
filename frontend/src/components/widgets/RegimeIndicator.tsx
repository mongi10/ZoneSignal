import React from 'react';
import { motion } from 'framer-motion';
import { Regime } from '../../types/forecast';

interface RegimeIndicatorProps {
  regime: Regime;
  probability: number;
  stability: number;
}

const REGIME_CONFIG: Record<Regime, { label: string; color: string; bgColor: string; icon: string }> = {
  risk_on: { label: 'Risk-On', color: 'text-green-400', bgColor: 'bg-green-500/20', icon: '📈' },
  risk_off: { label: 'Risk-Off', color: 'text-red-400', bgColor: 'bg-red-500/20', icon: '📉' },
  euphoria: { label: 'Euphoria', color: 'text-yellow-400', bgColor: 'bg-yellow-500/20', icon: '🚀' },
  panic: { label: 'Panic', color: 'text-red-500', bgColor: 'bg-red-600/20', icon: '🔴' },
  consolidation: { label: 'Consolidation', color: 'text-gray-400', bgColor: 'bg-gray-500/20', icon: '➡️' },
  macro_stress: { label: 'Macro Stress', color: 'text-orange-400', bgColor: 'bg-orange-500/20', icon: '⚠️' },
  reflation: { label: 'Reflation', color: 'text-blue-400', bgColor: 'bg-blue-500/20', icon: '🔄' },
  disinflation: { label: 'Disinflation', color: 'text-cyan-400', bgColor: 'bg-cyan-500/20', icon: '📊' },
  liquidity_expansion: { label: 'Liquidity+', color: 'text-green-300', bgColor: 'bg-green-400/20', icon: '💧' },
  liquidity_contraction: { label: 'Liquidity-', color: 'text-red-300', bgColor: 'bg-red-400/20', icon: '🏜️' }
};

export const RegimeIndicator: React.FC<RegimeIndicatorProps> = ({
  regime,
  probability,
  stability
}) => {
  const config = REGIME_CONFIG[regime];
  
  return (
    <div className="space-y-4">
      {/* Main Regime Display */}
      <motion.div
        className={`p-4 rounded-lg ${config.bgColor} border border-gray-700`}
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.3 }}
      >
        <div className="flex items-center gap-3">
          <span className="text-2xl">{config.icon}</span>
          <div>
            <div className={`text-lg font-semibold ${config.color}`}>
              {config.label}
            </div>
            <div className="text-xs text-gray-500">
              Detected Market Regime
            </div>
          </div>
        </div>
      </motion.div>
      
      {/* Probability & Stability */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-gray-900/50 rounded p-3">
          <div className="text-xs text-gray-500 mb-1">Probability</div>
          <div className="text-lg font-mono text-blue-400">
            {(probability * 100).toFixed(0)}%
          </div>
          <div className="mt-2 h-1.5 bg-gray-800 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-blue-500 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${probability * 100}%` }}
            />
          </div>
        </div>
        
        <div className="bg-gray-900/50 rounded p-3">
          <div className="text-xs text-gray-500 mb-1">Stability</div>
          <div className="text-lg font-mono text-purple-400">
            {(stability * 100).toFixed(0)}%
          </div>
          <div className="mt-2 h-1.5 bg-gray-800 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-purple-500 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${stability * 100}%` }}
            />
          </div>
        </div>
      </div>
      
      {/* Regime Description */}
      <div className="text-xs text-gray-500 leading-relaxed">
        {getRegimeDescription(regime)}
      </div>
    </div>
  );
};

function getRegimeDescription(regime: Regime): string {
  const descriptions: Record<Regime, string> = {
    risk_on: 'Markets favor risk assets. Equities typically outperform bonds. Positive momentum and breadth.',
    risk_off: 'Flight to safety underway. Defensive positioning recommended. Bonds and gold may outperform.',
    euphoria: 'Extreme bullish sentiment. Valuations stretched. Watch for reversal signals.',
    panic: 'Severe market stress. High volatility and correlation. Liquidity concerns.',
    consolidation: 'Range-bound trading. Low conviction. Wait for breakout confirmation.',
    macro_stress: 'Economic indicators deteriorating. Central bank policy uncertainty.',
    reflation: 'Growth accelerating with inflation. Cyclicals and commodities may outperform.',
    disinflation: 'Inflation cooling. Growth stocks and duration may benefit.',
    liquidity_expansion: 'Central banks adding liquidity. Asset prices supported.',
    liquidity_contraction: 'Quantitative tightening. Financial conditions tightening.'
  };
  
  return descriptions[regime];
}

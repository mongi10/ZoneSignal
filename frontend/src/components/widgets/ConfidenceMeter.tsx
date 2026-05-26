import React from 'react';
import { motion } from 'framer-motion';

interface ConfidenceMeterProps {
  label: string;
  value: number;
  inverted?: boolean;
}

export const ConfidenceMeter: React.FC<ConfidenceMeterProps> = ({
  label,
  value,
  inverted = false
}) => {
  const displayValue = inverted ? 1 - value : value;
  const percentage = displayValue * 100;
  
  const getColor = (val: number, inv: boolean) => {
    const effectiveVal = inv ? 1 - val : val;
    if (effectiveVal > 0.7) return 'bg-green-500';
    if (effectiveVal > 0.4) return 'bg-yellow-500';
    return 'bg-red-500';
  };
  
  const color = getColor(value, inverted);
  
  return (
    <div>
      <div className="flex justify-between items-center mb-1">
        <span className="text-xs text-gray-400">{label}</span>
        <span className="text-xs font-mono text-gray-300">
          {percentage.toFixed(0)}%
        </span>
      </div>
      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
        <motion.div
          className={`h-full ${color} rounded-full`}
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
        />
      </div>
    </div>
  );
};

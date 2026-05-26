import React, { useMemo } from 'react';
import Highcharts from 'highcharts';
import HighchartsReact from 'highcharts-react-official';
import { Forecast } from '../../types/forecast';

interface ProbabilisticConeProps {
  forecast: Forecast;
  historicalPrices?: number[];
}

export const ProbabilisticCone: React.FC<ProbabilisticConeProps> = ({
  forecast,
  historicalPrices = []
}) => {
  const options = useMemo<Highcharts.Options>(() => {
    const horizonDays = getHorizonDays(forecast.horizon);
    const basePrice = 100; // Normalized
    
    // Generate cone data points
    const coneData = generateConeData(forecast, horizonDays, basePrice);
    
    return {
      chart: {
        type: 'arearange',
        backgroundColor: 'transparent',
        height: '100%',
        animation: false
      },
      title: { text: undefined },
      xAxis: {
        type: 'datetime',
        labels: {
          style: { color: '#6b7280', fontSize: '10px' }
        },
        gridLineColor: '#1f2937',
        lineColor: '#374151'
      },
      yAxis: {
        title: { text: undefined },
        labels: {
          style: { color: '#6b7280', fontSize: '10px' },
          formatter: function() {
            return ((this.value as number - 100)).toFixed(1) + '%';
          }
        },
        gridLineColor: '#1f2937'
      },
      legend: { enabled: false },
      tooltip: {
        shared: true,
        backgroundColor: '#1f2937',
        borderColor: '#374151',
        style: { color: '#e5e7eb' },
        formatter: function() {
          const points = this.points || [];
          let html = `<b>${Highcharts.dateFormat('%Y-%m-%d', this.x as number)}</b><br/>`;
          points.forEach(point => {
            const low = ((point.point as any).low - 100).toFixed(2);
            const high = ((point.point as any).high - 100).toFixed(2);
            html += `<span style="color:${point.color}">●</span> ${point.series.name}: ${low}% to ${high}%<br/>`;
          });
          return html;
        }
      },
      plotOptions: {
        arearange: {
          fillOpacity: 0.3,
          lineWidth: 1,
          marker: { enabled: false }
        },
        line: {
          marker: { enabled: false }
        }
      },
      series: [
        // 95% confidence band
        {
          name: '95% CI',
          type: 'arearange',
          data: coneData.band95,
          color: '#3b82f6',
          fillOpacity: 0.1
        },
        // 50% confidence band
        {
          name: '50% CI',
          type: 'arearange',
          data: coneData.band50,
          color: '#3b82f6',
          fillOpacity: 0.2
        },
        // Expected path
        {
          name: 'Expected',
          type: 'line',
          data: coneData.expected,
          color: '#10b981',
          lineWidth: 2
        }
      ],
      credits: { enabled: false }
    };
  }, [forecast, historicalPrices]);

  return (
    <div className="h-full">
      <HighchartsReact highcharts={Highcharts} options={options} />
    </div>
  );
};

function getHorizonDays(horizon: string): number {
  const mapping: Record<string, number> = {
    '1D': 1, '1W': 5, '1M': 22, '3M': 66, '6M': 132, '1Y': 252
  };
  return mapping[horizon] || 22;
}

function generateConeData(forecast: Forecast, days: number, basePrice: number) {
  const now = Date.now();
  const dayMs = 24 * 60 * 60 * 1000;
  
  const expectedReturn = forecast.expected_return;
  const dailyReturn = expectedReturn / days;
  
  // Standard deviation from percentiles
  const std = (forecast.expected_return_95th - forecast.expected_return_5th) / 3.29; // ~2 std on each side for 95%
  
  const band95: [number, number, number][] = [];
  const band50: [number, number, number][] = [];
  const expected: [number, number][] = [];
  
  for (let i = 0; i <= days; i++) {
    const timestamp = now + i * dayMs;
    const t = i / days;
    
    // Expected cumulative return
    const expCumReturn = expectedReturn * t;
    const expPrice = basePrice * (1 + expCumReturn);
    
    // Uncertainty grows with sqrt(t)
    const uncertainty = std * Math.sqrt(t * days);
    
    // 95% band (±1.96 std)
    const low95 = basePrice * (1 + expCumReturn - 1.96 * uncertainty);
    const high95 = basePrice * (1 + expCumReturn + 1.96 * uncertainty);
    
    // 50% band (±0.67 std)
    const low50 = basePrice * (1 + expCumReturn - 0.67 * uncertainty);
    const high50 = basePrice * (1 + expCumReturn + 0.67 * uncertainty);
    
    band95.push([timestamp, low95, high95]);
    band50.push([timestamp, low50, high50]);
    expected.push([timestamp, expPrice]);
  }
  
  return { band95, band50, expected };
}

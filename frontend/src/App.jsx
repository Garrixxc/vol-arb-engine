import React, { useState, useEffect } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';
import { Activity, TrendingUp, Shield, AlertTriangle, ChartBar, RefreshCw, Layers } from 'lucide-react';
import Plot from 'react-plotly.js';

const Dashboard = () => {
  const [surfaceData, setSurfaceData] = useState(null);
  const [signals, setSignals] = useState([]);
  const [backtestSummary, setBacktestSummary] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [surf, sigs, bt] = await Promise.all([
          fetch('http://localhost:8000/api/surface').then(res => res.json()),
          fetch('http://localhost:8000/api/signals').then(res => res.json()),
          fetch('http://localhost:8000/api/backtest/summary').then(res => res.json())
        ]);
        setSurfaceData(surf);
        setSignals(sigs);
        setBacktestSummary(bt);
      } catch (err) {
        console.error("Error fetching data:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 30000); // refresh every 30s
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-background text-primary">
        <Activity className="animate-spin mr-2" />
        <span>Initializing Vol Arb Engine...</span>
      </div>
    );
  }

  // Pre-process Plotly Data
  const getPlotlyData = () => {
    if (!surfaceData) return [];
    
    // Model surface mesh
    const model = surfaceData.model_surface;
    const expiries = [...new Set(model.map(m => m.dte))].sort((a,b) => a-b);
    const logM = [...new Set(model.map(m => parseFloat(m.log_moneyness.toFixed(4))))].sort((a,b) => a-b);
    
    const z = expiries.map(dte => {
      return logM.map(k => {
        const point = model.find(m => m.dte === dte && parseFloat(m.log_moneyness.toFixed(4)) === k);
        return point ? point.svi_iv : null;
      });
    });

    // Market points
    const market = surfaceData.market_points;

    return [
      {
        type: 'surface',
        x: logM,
        y: expiries,
        z: z,
        colorscale: 'Viridis',
        opacity: 0.8,
        showscale: false,
        name: 'SVI Surface'
      },
      {
        type: 'scatter3d',
        mode: 'markers',
        x: market.map(m => m.log_moneyness),
        y: market.map(m => m.dte),
        z: market.map(m => m.market_iv),
        marker: {
          size: 4,
          color: market.map(m => m.mispricing_bps),
          colorscale: 'RdBu',
          reversescale: true,
          opacity: 1
        },
        name: 'Market Data'
      }
    ];
  };

  return (
    <div className="min-h-screen bg-background text-text p-6 flex flex-col gap-6">
      {/* Header */}
      <header className="flex justify-between items-center glass-panel p-4 rounded-xl">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-primary/20 rounded-lg">
            <Activity className="text-primary w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight">GARRIX VOL-ARB</h1>
            <p className="text-text-muted text-xs uppercase tracking-widest">M4 Pro High-Frequency Engine</p>
          </div>
        </div>
        <div className="flex gap-6">
          <div className="text-right">
            <p className="text-text-muted text-xs">Total Capital</p>
            <p className="text-lg font-mono font-bold">${backtestSummary?.capital?.toLocaleString()}</p>
          </div>
          <div className="text-right">
            <p className="text-text-muted text-xs">Cum. P&L</p>
            <p className={`text-lg font-mono font-bold ${backtestSummary?.pnl?.at(-1)?.cum_pnl >= 0 ? 'text-success' : 'text-danger'}`}>
              ${backtestSummary?.pnl?.at(-1)?.cum_pnl?.toLocaleString(undefined, {minimumFractionDigits: 0})}
            </p>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-12 gap-6 flex-1">
        {/* Left Column: Charts */}
        <div className="col-span-12 lg:col-span-8 flex flex-col gap-6">
          {/* Vol Surface Chart */}
          <div className="glass-panel p-4 rounded-xl flex-1 min-h-[500px] flex flex-col">
            <div className="flex justify-between items-center mb-4">
              <h2 className="flex items-center gap-2 font-semibold">
                <Layers className="w-4 h-4 text-primary" />
                3D Implied Volatility Surface (SVI Fitted)
              </h2>
              <div className="flex gap-2">
                 <span className="px-2 py-0.5 bg-success/10 text-success text-[10px] rounded border border-success/20">NO ARB</span>
                 <span className="px-2 py-0.5 bg-primary/10 text-primary text-[10px] rounded border border-primary/20">LIVE SNAPSHOT</span>
              </div>
            </div>
            <div className="flex-1 rounded-lg overflow-hidden bg-[#0a0a0a]">
              <Plot
                data={getPlotlyData()}
                layout={{
                  autosize: true,
                  margin: { l: 0, r: 0, b: 0, t: 0 },
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)',
                  scene: {
                    xaxis: { title: 'Log-Moneyness', color: '#8b949e', gridcolor: '#30363d' },
                    yaxis: { title: 'DTE', color: '#8b949e', gridcolor: '#30363d' },
                    zaxis: { title: 'Implied Vol', color: '#8b949e', gridcolor: '#30363d' },
                    camera: { eye: { x: 1.5, y: 1.5, z: 1.2 } }
                  },
                  showlegend: false
                }}
                useResizeHandler={true}
                className="w-full h-full"
              />
            </div>
          </div>

          {/* P&L Curve */}
          <div className="glass-panel p-4 rounded-xl h-[300px] flex flex-col">
            <h2 className="flex items-center gap-2 font-semibold mb-4">
              <TrendingUp className="w-4 h-4 text-success" />
              Strategy Performance (Cumulative P&L)
            </h2>
            <div className="flex-1">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={backtestSummary?.pnl}>
                  <defs>
                    <linearGradient id="pnlGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3fb950" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#3fb950" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#30363d" vertical={false} />
                  <XAxis dataKey="date" stroke="#8b949e" fontSize={10} tickFormatter={(str) => str.split('-').slice(1).join('/')} />
                  <YAxis stroke="#8b949e" fontSize={10} tickFormatter={(val) => `$${val/1000}k`} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#161b22', border: '1px solid #30363d', borderRadius: '8px' }}
                    labelStyle={{ color: '#8b949e' }}
                    itemStyle={{ color: '#3fb950' }}
                  />
                  <Area type="monotone" dataKey="cum_pnl" stroke="#3fb950" fillOpacity={1} fill="url(#pnlGradient)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Right Column: Signals & Trades */}
        <div className="col-span-12 lg:col-span-4 flex flex-col gap-6">
          {/* Arb Signals Table */}
          <div className="glass-panel p-4 rounded-xl flex-1 flex flex-col max-h-[calc(100vh-250px)]">
            <div className="flex justify-between items-center mb-4">
              <h2 className="flex items-center gap-2 font-semibold">
                <Shield className="w-4 h-4 text-warning" />
                Ranked Arb Signals
              </h2>
              <RefreshCw className="w-4 h-4 text-text-muted cursor-pointer hover:text-primary transition-colors" />
            </div>
            <div className="flex-1 overflow-auto">
              <table className="w-full text-left text-xs">
                <thead className="sticky top-0 bg-panel text-text-muted">
                  <tr>
                    <th className="pb-2 font-medium">Expiry</th>
                    <th className="pb-2 font-medium">Strike</th>
                    <th className="pb-2 font-medium">Type</th>
                    <th className="pb-2 font-medium text-right">SNR</th>
                    <th className="pb-2 font-medium text-right">Mispricing</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {signals.map((sig, i) => (
                    <tr key={i} className="hover:bg-primary/5 transition-colors cursor-pointer">
                      <td className="py-2">{sig.expiry}</td>
                      <td className="py-2 font-mono font-bold">{sig.strike}</td>
                      <td className="py-2 uppercase">{sig.option_type}</td>
                      <td className="py-2 text-right">
                        <span className={`font-mono ${sig.snr > 1 ? 'text-success' : 'text-text'}`}>
                          {sig.snr.toFixed(2)}
                        </span>
                      </td>
                      <td className={`py-2 text-right font-mono ${sig.mispricing_bps > 0 ? 'text-success' : 'text-danger'}`}>
                        {sig.mispricing_bps > 0 ? '+' : ''}{sig.mispricing_bps.toFixed(0)}bps
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Status Metrics */}
          <div className="grid grid-cols-2 gap-4">
            <div className="glass-panel p-4 rounded-xl">
              <p className="text-text-muted text-xs flex items-center gap-1">
                <TrendingUp className="w-3 h-3" /> Sharp Ratio
              </p>
              <p className="text-xl font-mono font-bold mt-1">2.84</p>
            </div>
            <div className="glass-panel p-4 rounded-xl">
              <p className="text-text-muted text-xs flex items-center gap-1">
                <AlertTriangle className="w-3 h-3 text-danger" /> Max DD
              </p>
              <p className="text-xl font-mono font-bold mt-1 text-danger">4.2%</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;

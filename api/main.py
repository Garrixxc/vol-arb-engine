from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import math


def sanitize_for_json(records: list) -> list:
    """Replace NaN/Inf float values with None so they serialize cleanly to JSON."""
    cleaned = []
    for row in records:
        cleaned_row = {}
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                cleaned_row[k] = None
            else:
                cleaned_row[k] = v
        cleaned.append(cleaned_row)
    return cleaned

# Add project root to path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "core"))

from core.iv_surface import compute_iv_surface
from core.svi import fit_surface
from backtest.engine import generate_synthetic_backtest_data, BacktestEngine, BacktestConfig
from signals.aggregator import run_full_signal_pipeline

app = FastAPI(title="Vol Arb Engine API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state to store backtest results (simulating a DB cache for now)
BACKTEST_RESULTS = None

def get_backtest_results():
    global BACKTEST_RESULTS
    if BACKTEST_RESULTS is None:
        print("Initializing synthetic backtest for dashboard...")
        snapshots = generate_synthetic_backtest_data(n_days=90)
        engine = BacktestEngine(BacktestConfig(capital=1000000))
        BACKTEST_RESULTS = engine.run(snapshots, verbose=False)
    return BACKTEST_RESULTS

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.get("/api/surface")
async def get_surface():
    """Returns the latest SVI fitted surface data for 3D plotting."""
    try:
        # Generate one day of synthetic data for the current surface view
        snapshots = generate_synthetic_backtest_data(n_days=1)
        snap = snapshots[0]
        iv_surface = compute_iv_surface(snap["chain"], r=0.053, q=0.013)
        params, fitted_df, _ = fit_surface(iv_surface, verbose=False)
        
        # Format for JSON — sanitize NaN/Inf to None for JSON compliance
        market_points = sanitize_for_json(fitted_df[fitted_df["option_type"] != "model"].to_dict(orient="records"))
        model_surface = sanitize_for_json(fitted_df[fitted_df["option_type"] == "model"].to_dict(orient="records"))
        
        return {
            "date": snap["date"],
            "spot": snap["spot"],
            "market_points": market_points,
            "model_surface": model_surface
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/signals")
async def get_signals():
    """Returns ranked arbitrage signals."""
    try:
        snapshots = generate_synthetic_backtest_data(n_days=1)
        snap = snapshots[0]
        # In a real app, we'd use the aggregator on real data
        # For now, we'll use the signals generated in the backtest or fresh ones
        iv_surface = compute_iv_surface(snap["chain"], r=0.053, q=0.013)
        _, fitted_df, _ = fit_surface(iv_surface, verbose=False)
        
        signals = fitted_df[fitted_df["option_type"] != "model"].dropna(subset=["snr"])
        signals = signals.sort_values("snr", ascending=False).head(20)
        
        return signals.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/backtest/summary")
async def get_backtest_summary():
    """Returns backtest P&L and Greeks over time."""
    try:
        results = get_backtest_results()
        event_log = results["event_log"]
        
        # Prepare P&L data
        pnl_data = event_log[["date", "cum_pnl", "spot", "n_open"]].to_dict(orient="records")
        
        return {
            "pnl": pnl_data,
            "capital": 1000000
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/backtest/trades")
async def get_backtest_trades():
    """Returns detailed trade log."""
    try:
        results = get_backtest_results()
        trades_df = results["trades_df"]
        return trades_df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

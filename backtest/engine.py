"""
backtest/engine.py

Event-driven backtester for the vol arb strategy.

Architecture:
  - Each "event" is a new surface snapshot (daily or intraday)
  - On each event: (1) update existing positions, (2) run signal pipeline,
    (3) enter new trades if signals are strong enough
  - No lookahead bias: signals are computed only from data available
    at that point in time

Simulation pipeline per day:
  ┌─────────────────────────────────────────────────────┐
  │  1. Ingest new snapshot (spot, options chain)        │
  │  2. Compute IV surface (C++ IV solver)              │
  │  3. Fit SVI surface (C++ LM calibrator)             │
  │  4. Update all open positions with new IVs + spot   │
  │  5. Check exit conditions (stop-loss, profit-take,  │
  │     DTE threshold)                                   │
  │  6. Run signal pipeline on fresh surface            │
  │  7. Enter top-scored signals (subject to limits)    │
  │  8. Log portfolio state                             │
  └─────────────────────────────────────────────────────┘

To avoid lookahead:
  - Signals computed on day T → trades entered at day T+1 open
  - This simulates realistic execution delay
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Callable
from datetime import datetime, timedelta
import vol_core

from core.iv_surface import compute_iv_surface
from core.svi import fit_surface
from signals.aggregator import run_full_signal_pipeline
from signals.surface_mispricing import MispricingSignal, compute_mispricing_signals
from backtest.position_manager import PortfolioManager, Position


class BacktestConfig:
    def __init__(self,
        capital:             float = 100_000,
        r:                   float = 0.053,
        q:                   float = 0.013,
        max_positions:       int   = 8,
        contracts_per_trade: float = 1.0,
        min_signal_score:    float = 60.0,    # min composite score to enter
        min_snr:             float = 0.3,     # min SNR for surface mispricing
        stop_loss_pct:       float = 2.0,     # stop at 2x initial vega $ loss
        profit_take_pct:     float = 0.7,     # take profit at 70% of max theoretical
        min_dte_entry:       int   = 10,      # don't enter < 10 DTE
        exit_dte:            int   = 5,       # force exit at 5 DTE
        max_trades_per_day:  int   = 2,       # max new trades per snapshot
        rehedge_daily:       bool  = True,
    ):
        self.capital             = capital
        self.r                   = r
        self.q                   = q
        self.max_positions       = max_positions
        self.contracts           = contracts_per_trade
        self.min_signal_score    = min_signal_score
        self.min_snr             = min_snr
        self.stop_loss_pct       = stop_loss_pct
        self.profit_take_pct     = profit_take_pct
        self.min_dte_entry       = min_dte_entry
        self.exit_dte            = exit_dte
        self.max_trades_per_day  = max_trades_per_day
        self.rehedge_daily       = rehedge_daily


class BacktestEngine:
    def __init__(self, config: BacktestConfig = None):
        self.config   = config or BacktestConfig()
        self.portfolio = PortfolioManager(
            max_positions       = self.config.max_positions,
            contracts_per_trade = self.config.contracts,
        )
        self.event_log: List[dict] = []
        self.signal_history: List[dict] = []
        self._pending_signals: List[MispricingSignal] = []   # T+1 execution

    def run(self, snapshots: List[dict], verbose: bool = True) -> dict:
        """
        Run the backtest over a list of surface snapshots.

        Each snapshot is a dict:
          {
            "date":  "YYYY-MM-DD",
            "spot":  float,
            "chain": pd.DataFrame,   # options chain for this date
          }

        Returns dict with portfolio_df, trades_df, metrics.
        """
        n = len(snapshots)
        if verbose:
            print(f"\n{'='*65}")
            print(f"  BACKTEST ENGINE STARTING")
            print(f"  {n} snapshots | Capital: ${self.config.capital:,.0f}")
            print(f"{'='*65}")

        skew_history_rows = []

        for i, snap in enumerate(snapshots):
            date  = snap["date"]
            spot  = snap["spot"]
            chain = snap["chain"]

            if verbose and i % max(1, n//10) == 0:
                n_open = len(self.portfolio.open_positions)
                cum_pnl = sum(p.total_pnl for p in self.portfolio.open_positions) + \
                          sum(p.total_pnl for p in self.portfolio.closed_positions)
                print(f"  [{i+1:>4}/{n}] {date}  spot={spot:.1f}  "
                      f"positions={n_open}  cum_pnl=${cum_pnl:,.0f}")

            # ── Step 1: Compute IV surface ──────────────────────────
            try:
                iv_surface = compute_iv_surface(chain, r=self.config.r,
                                                 q=self.config.q)
            except Exception as e:
                if verbose: print(f"    [WARN] IV surface failed: {e}")
                continue

            # ── Step 2: Fit SVI surface ─────────────────────────────
            try:
                params_by_expiry, fitted_df, _ = fit_surface(
                    iv_surface, verbose=False
                )
            except Exception as e:
                if verbose: print(f"    [WARN] SVI fit failed: {e}")
                continue

            # ── Step 3: Build IV lookup for position updates ────────
            iv_lookup = {}
            for _, row in iv_surface.iterrows():
                key = (row["expiry"], row["strike"], row["option_type"])
                iv_lookup[key] = row["market_iv"]

            # ── Step 4: Update existing positions ──────────────────
            day_pnl = self.portfolio.update_all(date, spot, iv_lookup)

            # ── Step 5: Check exit conditions ──────────────────────
            self._check_exits(date, spot, iv_lookup)

            # ── Step 6: Execute T+1 signals from yesterday ─────────
            if self._pending_signals:
                entered = 0
                for sig in self._pending_signals:
                    if entered >= self.config.max_trades_per_day:
                        break
                    if sig.dte < self.config.min_dte_entry:
                        continue
                    pos = self.portfolio.enter_position(
                        date, sig, spot, self.config.r, self.config.q
                    )
                    if pos:
                        entered += 1
                        if verbose:
                            print(f"    [ENTER] {pos.pos_id}  "
                                  f"{'SHORT' if pos.direction==-1 else 'LONG'} "
                                  f"{pos.option_type.upper()} K={pos.strike:.0f} "
                                  f"{pos.expiry}  entry_iv={pos.entry_iv:.1%}")
                self._pending_signals = []

            # ── Step 7: Generate new signals for T+1 ───────────────
            mispricing_sigs = compute_mispricing_signals(
                fitted_df, chain,
                min_snr  = self.config.min_snr,
                top_n    = self.config.max_trades_per_day * 3,
            )

            # Filter by score threshold and DTE
            new_signals = [
                s for s in mispricing_sigs
                if s.score >= self.config.min_signal_score
                and s.dte   >= self.config.min_dte_entry
            ]

            # Dedup: don't re-enter positions we already hold
            existing_keys = {
                (p.expiry, p.strike, p.option_type)
                for p in self.portfolio.open_positions
            }
            new_signals = [
                s for s in new_signals
                if (s.expiry, s.strike, s.option_type) not in existing_keys
            ]

            self._pending_signals = new_signals[:self.config.max_trades_per_day]

            # ── Step 8: Log event ───────────────────────────────────
            self.event_log.append({
                "date":          date,
                "spot":          spot,
                "day_pnl":       day_pnl,
                "cum_pnl":       sum(p.total_pnl for p in self.portfolio.open_positions)
                                 + sum(p.total_pnl for p in self.portfolio.closed_positions),
                "n_open":        len(self.portfolio.open_positions),
                "n_signals":     len(new_signals),
            })

        # ── Compile results ─────────────────────────────────────────
        portfolio_df = self.portfolio.get_portfolio_df()
        trades_df    = self.portfolio.get_trades_df()

        return {
            "portfolio_df": portfolio_df,
            "trades_df":    trades_df,
            "event_log":    pd.DataFrame(self.event_log),
        }

    def _check_exits(self, date: str, spot: float,
                     iv_lookup: Dict[tuple, float]):
        """Check stop-loss and DTE-based exit conditions."""
        for pos in list(self.portfolio.open_positions):
            # Force exit near expiry
            if pos.current_T * 365 <= self.config.exit_dte:
                iv = iv_lookup.get((pos.expiry, pos.strike, pos.option_type),
                                   pos.current_iv)
                self.portfolio.close_position(pos.pos_id, date, spot, iv,
                                               reason="EXIT_DTE")
                continue

            # Stop loss: if position has lost more than entry vega $ * multiplier
            entry_vega_dollar = abs(pos.vega * pos.contracts * 100)
            stop_threshold    = -entry_vega_dollar * self.config.stop_loss_pct * 100
            if pos.total_pnl < stop_threshold and stop_threshold < 0:
                iv = iv_lookup.get((pos.expiry, pos.strike, pos.option_type),
                                   pos.current_iv)
                self.portfolio.close_position(pos.pos_id, date, spot, iv,
                                               reason="STOP_LOSS")


def generate_synthetic_backtest_data(
    n_days:    int   = 60,
    start_spot: float = 580.0,
    r:         float = 0.053,
    q:         float = 0.013,
    ann_vol:   float = 0.18,
    seed:      int   = 42,
) -> List[dict]:
    """
    Generate realistic synthetic backtest data:
      - GBM spot process
      - SVI-consistent options chain per day
      - With mean-reverting vol regime changes
    """
    np.random.seed(seed)
    dt     = 1 / 252
    sqrtdt = np.sqrt(dt)

    spots = [start_spot]
    vols  = [ann_vol]

    # Vol-of-vol process: mean-reverting vol
    kappa, theta_v, xi = 2.0, ann_vol, 0.3

    for _ in range(n_days - 1):
        dW_s = np.random.randn()
        dW_v = 0.7 * dW_s + 0.3 * np.random.randn()  # correlated shocks

        v  = vols[-1]
        dv = kappa * (theta_v - v) * dt + xi * v * sqrtdt * dW_v
        new_v = max(v + dv, 0.05)

        dS = spots[-1] * ((r - q) * dt + new_v * sqrtdt * dW_s)
        spots.append(spots[-1] + dS)
        vols.append(new_v)

    snapshots = []
    base_date = datetime(2026, 1, 5)

    for i, (spot, vol) in enumerate(zip(spots, vols)):
        date = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
        records = []

        # Build chain: 5 expiries, each with 14 strikes
        for dte_offset in [14, 30, 60, 90, 180]:
            exp_date = (base_date + timedelta(days=i+dte_offset)).strftime("%Y-%m-%d")
            T = dte_offset / 365.0

            for lm in np.linspace(-0.18, 0.14, 14):
                k = spot * np.exp(lm)
                # SVI surface consistent with current vol level
                scale = vol / ann_vol   # scale surface with current vol
                a, b, rho_s, m, sig = 0.04*scale**2, 0.15, -0.65, 0.01, 0.13
                w = a + b * (rho_s*(lm-m) + np.sqrt((lm-m)**2+sig**2))

                # Add small random noise (bid-ask uncertainty)
                noise = np.random.normal(0, 0.002 * scale)
                iv    = max(np.sqrt(w/T) + noise, 0.05)

                for opt, cp in [("call",1), ("put",-1)]:
                    mid  = vol_core.bs_price(spot, k, r, q, iv, T, cp)
                    half = mid * (0.010 + 0.012*abs(lm))
                    records.append({
                        "ticker": "SPY", "expiry": exp_date,
                        "dte": dte_offset, "T": T,
                        "strike": round(k, 2), "option_type": opt,
                        "bid": round(max(mid-half, 0.01), 2),
                        "ask": round(mid+half, 2),
                        "mid": round(mid, 2),
                        "spread_pct": 2*half/mid,
                        "openInterest": int(400*np.exp(-3*lm**2)),
                        "volume":       int(80*np.exp(-3*lm**2)),
                        "spot": spot, "moneyness": k/spot,
                    })

        snapshots.append({
            "date":  date,
            "spot":  spot,
            "chain": pd.DataFrame(records),
        })

    return snapshots

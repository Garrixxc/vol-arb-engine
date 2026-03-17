"""
backtest/position_manager.py

Tracks delta-hedged volatility positions through time.

The core idea of a vol arb trade:
  1. Enter: buy/sell the mispriced option
  2. Delta-hedge: immediately buy/sell the underlying to make
     the position delta-neutral (zero first-order directional exposure)
  3. Daily re-hedge: as spot moves and time passes, delta drifts.
     Rebalance daily (or at some threshold) to stay delta-neutral.
  4. P&L source: the position makes money from realized vol ≠ implied vol.
     If you sold vol at 25% and realized vol is 18%, you profit.
     This P&L is called "theta-gamma P&L" or "vega P&L".

Daily P&L decomposition (Black-Scholes):
    PnL_day ≈ Vega * Δσ + Theta * Δt + 0.5 * Gamma * (ΔS)²
                ↑                ↑              ↑
           vol change      time decay      gamma scalp

For a SELL_VOL position (short option, long delta-hedge):
    - Theta > 0 (collect time decay)
    - Gamma < 0 (pay out when market moves)
    - Net P&L > 0 when realized vol < implied vol at entry
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import vol_core


@dataclass
class Position:
    # Identity
    pos_id:       str
    entry_date:   str
    expiry:       str
    strike:       float
    option_type:  str             # "call" or "put"
    direction:    int             # +1 = long vol, -1 = short vol
    contracts:    float           # number of option contracts (1 = 100 shares)

    # Entry state
    entry_spot:   float
    entry_iv:     float           # IV at entry
    entry_model_iv: float         # SVI model IV at entry
    entry_price:  float           # option mid-price at entry
    entry_delta:  float           # BS delta at entry
    r:            float
    q:            float

    # Current state (updated daily)
    current_spot:  float = 0.0
    current_iv:    float = 0.0
    current_price: float = 0.0
    current_delta: float = 0.0
    current_T:     float = 0.0

    # Hedge: shares of underlying held to offset delta
    # hedge_shares = -direction * contracts * 100 * delta
    hedge_shares:  float = 0.0

    # Cumulative P&L components
    option_pnl:   float = 0.0    # P&L from option position
    hedge_pnl:    float = 0.0    # P&L from delta hedge
    total_pnl:    float = 0.0

    # Greeks at current state
    gamma:        float = 0.0
    vega:         float = 0.0
    theta:        float = 0.0

    # Status
    is_open:      bool  = True
    exit_date:    Optional[str] = None
    exit_reason:  str = ""

    # Daily P&L log
    daily_pnl:    List[dict] = field(default_factory=list)

    def __post_init__(self):
        self.current_spot  = self.entry_spot
        self.current_iv    = self.entry_iv
        self.current_T     = self._dte_from_expiry(self.entry_date) / 365.0
        cp = 1 if self.option_type == "call" else -1
        g  = vol_core.bs_greeks(self.entry_spot, self.strike, self.r, self.q,
                                 self.entry_iv, self.current_T, cp)
        self.current_delta = g.delta
        self.gamma         = g.gamma
        self.vega          = g.vega
        self.theta         = g.theta
        # Initial hedge: delta-neutral
        self.hedge_shares  = -self.direction * self.contracts * 100 * self.current_delta

    def _dte_from_expiry(self, as_of_date: str) -> int:
        from datetime import datetime
        exp = datetime.strptime(self.expiry, "%Y-%m-%d")
        aod = datetime.strptime(as_of_date, "%Y-%m-%d")
        return max((exp - aod).days, 0)

    def update(self, date: str, new_spot: float, new_iv: float,
               rehedge: bool = True) -> float:
        """
        Update position for a new day.
        Returns today's total P&L (option + hedge).
        """
        if not self.is_open:
            return 0.0

        T_new = self._dte_from_expiry(date) / 365.0
        cp    = 1 if self.option_type == "call" else -1

        # Option P&L: (new_price - old_price) * direction * contracts * 100
        if T_new > 0:
            new_price = vol_core.bs_price(new_spot, self.strike, self.r, self.q,
                                          new_iv, T_new, cp)
        else:
            new_price = max(cp * (new_spot - self.strike), 0.0)

        option_pnl_today = (new_price - self.current_price) * self.direction * self.contracts * 100

        # Hedge P&L: spot move * hedge shares
        spot_move        = new_spot - self.current_spot
        hedge_pnl_today  = spot_move * self.hedge_shares

        day_pnl = option_pnl_today + hedge_pnl_today

        # Update Greeks
        if T_new > 1/365:
            g = vol_core.bs_greeks(new_spot, self.strike, self.r, self.q,
                                    new_iv, T_new, cp)
            self.current_delta = g.delta
            self.gamma         = g.gamma
            self.vega          = g.vega
            self.theta         = g.theta
        else:
            self.current_delta = cp * 1.0 if cp * (new_spot - self.strike) > 0 else 0.0
            self.gamma = self.vega = 0.0
            self.theta = 0.0

        # Re-hedge to new delta
        if rehedge:
            self.hedge_shares = -self.direction * self.contracts * 100 * self.current_delta

        # Accumulate
        self.option_pnl   += option_pnl_today
        self.hedge_pnl    += hedge_pnl_today
        self.total_pnl    += day_pnl
        self.current_spot  = new_spot
        self.current_iv    = new_iv
        self.current_price = new_price
        self.current_T     = T_new

        self.daily_pnl.append({
            "date":            date,
            "spot":            new_spot,
            "iv":              new_iv,
            "T":               T_new,
            "option_pnl":      option_pnl_today,
            "hedge_pnl":       hedge_pnl_today,
            "day_pnl":         day_pnl,
            "cum_pnl":         self.total_pnl,
            "delta":           self.current_delta,
            "gamma":           self.gamma,
            "vega":            self.vega,
            "theta":           self.theta,
        })

        # Auto-close at expiry
        if T_new <= 0:
            self.close(date, new_spot, new_iv, "EXPIRY")

        return day_pnl

    def close(self, date: str, spot: float, iv: float, reason: str = "MANUAL"):
        self.is_open    = False
        self.exit_date  = date
        self.exit_reason = reason

    def summary(self) -> dict:
        return {
            "pos_id":       self.pos_id,
            "expiry":       self.expiry,
            "strike":       self.strike,
            "option_type":  self.option_type,
            "direction":    "LONG_VOL" if self.direction == 1 else "SHORT_VOL",
            "entry_date":   self.entry_date,
            "exit_date":    self.exit_date or "OPEN",
            "entry_iv":     self.entry_iv,
            "entry_model_iv": self.entry_model_iv,
            "mispricing_entry_bps": (self.entry_iv - self.entry_model_iv) * 10000,
            "total_pnl":    self.total_pnl,
            "option_pnl":   self.option_pnl,
            "hedge_pnl":    self.hedge_pnl,
            "exit_reason":  self.exit_reason,
        }


class PortfolioManager:
    """
    Manages a book of delta-hedged positions.
    Enforces position limits and tracks portfolio-level Greeks.
    """
    def __init__(
        self,
        max_positions:      int   = 10,
        max_vega_per_expiry: float = 5000,   # $ vega limit per expiry
        contracts_per_trade: float = 1.0,
    ):
        self.max_positions       = max_positions
        self.max_vega_per_expiry = max_vega_per_expiry
        self.contracts           = contracts_per_trade
        self.positions:   List[Position] = []
        self.closed_positions: List[Position] = []
        self.daily_portfolio_pnl: List[dict] = []

    @property
    def open_positions(self) -> List[Position]:
        return [p for p in self.positions if p.is_open]

    def can_enter(self, expiry: str) -> bool:
        if len(self.open_positions) >= self.max_positions:
            return False
        # Check vega concentration
        expiry_vega = sum(
            abs(p.vega * p.contracts * 100 * p.direction)
            for p in self.open_positions if p.expiry == expiry
        )
        return expiry_vega < self.max_vega_per_expiry

    def enter_position(self, date: str, signal, spot: float,
                       r: float, q: float) -> Optional[Position]:
        """Enter a new delta-hedged position from a MispricingSignal."""
        if not self.can_enter(signal.expiry):
            return None

        # Determine direction: SELL_VOL = short option = direction -1
        direction = -1 if signal.direction == "SELL_VOL" else 1

        cp = 1 if signal.option_type == "call" else -1
        T  = signal.dte / 365.0
        if T <= 0:
            return None

        price = vol_core.bs_price(spot, signal.strike, r, q,
                                   signal.market_iv, T, cp)

        pos = Position(
            pos_id         = f"{date}_{signal.expiry}_{signal.strike:.0f}_{signal.option_type[0]}",
            entry_date     = date,
            expiry         = signal.expiry,
            strike         = signal.strike,
            option_type    = signal.option_type,
            direction      = direction,
            contracts      = self.contracts,
            entry_spot     = spot,
            entry_iv       = signal.market_iv,
            entry_model_iv = signal.model_iv,
            entry_price    = price,
            entry_delta    = 0.0,  # will be set in __post_init__
            r              = r,
            q              = q,
        )
        self.positions.append(pos)
        return pos

    def update_all(self, date: str, spot: float,
                   iv_lookup: Dict[tuple, float]) -> float:
        """
        Update all open positions for a new day.
        iv_lookup: {(expiry, strike, option_type): current_iv}
        Returns total portfolio P&L for the day.
        """
        total_day_pnl = 0.0
        for pos in self.open_positions:
            key = (pos.expiry, pos.strike, pos.option_type)
            iv  = iv_lookup.get(key, pos.current_iv)  # fall back to last known IV
            pnl = pos.update(date, spot, iv)
            total_day_pnl += pnl

            if not pos.is_open:
                self.closed_positions.append(pos)

        # Remove closed from active list
        self.positions = [p for p in self.positions if p.is_open]

        # Portfolio Greeks
        port_delta = sum(p.current_delta * p.contracts * 100 * p.direction
                         for p in self.open_positions)
        port_gamma = sum(p.gamma * p.contracts * 100 * p.direction
                         for p in self.open_positions)
        port_vega  = sum(p.vega  * p.contracts * 100 * p.direction
                         for p in self.open_positions)
        port_theta = sum(p.theta * p.contracts * 100 * p.direction
                         for p in self.open_positions)

        self.daily_portfolio_pnl.append({
            "date":        date,
            "day_pnl":     total_day_pnl,
            "n_positions": len(self.open_positions),
            "port_delta":  port_delta,
            "port_gamma":  port_gamma,
            "port_vega":   port_vega,
            "port_theta":  port_theta,
        })

        return total_day_pnl

    def close_position(self, pos_id: str, date: str, spot: float,
                        iv: float, reason: str = "MANUAL"):
        for pos in self.open_positions:
            if pos.pos_id == pos_id:
                pos.close(date, spot, iv, reason)
                self.closed_positions.append(pos)
                self.positions = [p for p in self.positions if p.is_open]
                return

    def get_portfolio_df(self) -> pd.DataFrame:
        if not self.daily_portfolio_pnl:
            return pd.DataFrame()
        df = pd.DataFrame(self.daily_portfolio_pnl)
        df["cum_pnl"] = df["day_pnl"].cumsum()
        return df

    def get_trades_df(self) -> pd.DataFrame:
        all_pos = [p for p in self.positions] + self.closed_positions
        if not all_pos:
            return pd.DataFrame()
        return pd.DataFrame([p.summary() for p in all_pos])

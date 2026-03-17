"""
backtest/metrics.py

Computes institutional-grade performance metrics from backtest results.

Metrics implemented:
  RETURNS
    Total return, annualized return, daily return distribution

  RISK-ADJUSTED
    Sharpe ratio (annualized, assuming 0 risk-free rate for vol arb)
    Sortino ratio (penalizes only downside volatility)
    Calmar ratio  (return / max drawdown)
    Information ratio vs benchmark

  DRAWDOWN
    Maximum drawdown (peak-to-trough)
    Max drawdown duration (days underwater)
    Average drawdown recovery time

  TRADE STATISTICS
    Hit rate (% profitable trades)
    Avg win / avg loss (profit factor)
    Average holding period
    Average entry mispricing vs realized P&L correlation

  VOL-SPECIFIC
    Avg vega at entry vs realized P&L
    Theta collected vs gamma paid
    VRP capture rate
"""

import numpy as np
import pandas as pd
from typing import Optional


def compute_returns_metrics(portfolio_df: pd.DataFrame,
                             capital: float = 100_000) -> dict:
    """Compute return metrics from daily P&L series."""
    if portfolio_df.empty or "day_pnl" not in portfolio_df.columns:
        return {}

    pnl    = portfolio_df["day_pnl"].values
    cum    = portfolio_df["cum_pnl"].values
    n_days = len(pnl)

    total_pnl        = cum[-1]
    total_return_pct = total_pnl / capital * 100
    ann_factor       = 252 / n_days
    ann_return_pct   = total_return_pct * ann_factor

    daily_ret = pnl / capital
    daily_std = daily_ret.std()
    ann_vol   = daily_std * np.sqrt(252) * 100

    return {
        "total_pnl":        total_pnl,
        "total_return_pct": total_return_pct,
        "ann_return_pct":   ann_return_pct,
        "ann_vol_pct":      ann_vol,
        "n_days":           n_days,
    }


def sharpe_ratio(portfolio_df: pd.DataFrame, capital: float = 100_000,
                 risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio."""
    if portfolio_df.empty:
        return 0.0
    daily_ret = portfolio_df["day_pnl"].values / capital
    excess    = daily_ret - risk_free_rate / 252
    std       = excess.std()
    if std < 1e-10:
        return 0.0
    return float(excess.mean() / std * np.sqrt(252))


def sortino_ratio(portfolio_df: pd.DataFrame, capital: float = 100_000,
                  mar: float = 0.0) -> float:
    """
    Sortino ratio — like Sharpe but only penalizes downside vol.
    MAR = minimum acceptable return (daily).
    """
    if portfolio_df.empty:
        return 0.0
    daily_ret  = portfolio_df["day_pnl"].values / capital
    excess     = daily_ret - mar / 252
    downside   = excess[excess < 0]
    if len(downside) == 0 or downside.std() < 1e-10:
        return np.inf if excess.mean() > 0 else 0.0
    downside_std = np.sqrt(np.mean(downside**2))
    return float(excess.mean() / downside_std * np.sqrt(252))


def max_drawdown(portfolio_df: pd.DataFrame) -> dict:
    """
    Maximum drawdown and duration.
    Returns dict with: max_dd_pct, max_dd_pnl, max_dd_duration_days,
                       dd_start, dd_end, dd_recovery
    """
    if portfolio_df.empty:
        return {"max_dd_pct": 0, "max_dd_pnl": 0, "max_dd_duration": 0}

    cum = portfolio_df["cum_pnl"].values

    peak        = np.maximum.accumulate(cum)
    drawdown    = cum - peak
    max_dd_pnl  = float(drawdown.min())

    # Find the drawdown period
    end_idx   = int(np.argmin(drawdown))
    start_idx = int(np.argmax(cum[:end_idx+1])) if end_idx > 0 else 0

    # Recovery: first day after trough where cum >= peak at start
    peak_val  = cum[start_idx]
    recovery_idx = None
    for i in range(end_idx, len(cum)):
        if cum[i] >= peak_val:
            recovery_idx = i
            break

    duration = end_idx - start_idx
    recovery_duration = (recovery_idx - end_idx) if recovery_idx else None

    # As % of peak value (use absolute, since we're working in $)
    # Normalize to starting capital for % interpretation
    max_dd_pct = max_dd_pnl  # in $ — caller normalizes

    dates = portfolio_df["date"].values if "date" in portfolio_df.columns else None

    return {
        "max_dd_pnl":           max_dd_pnl,
        "max_dd_start_idx":     start_idx,
        "max_dd_end_idx":       end_idx,
        "max_dd_duration_days": duration,
        "recovery_duration":    recovery_duration,
        "dd_start_date":        dates[start_idx] if dates is not None else None,
        "dd_end_date":          dates[end_idx]   if dates is not None else None,
    }


def calmar_ratio(portfolio_df: pd.DataFrame, capital: float = 100_000) -> float:
    """Annualized return / |Max drawdown|."""
    if portfolio_df.empty:
        return 0.0
    ann_ret = portfolio_df["day_pnl"].sum() / capital * (252 / len(portfolio_df)) * 100
    dd      = abs(max_drawdown(portfolio_df)["max_dd_pnl"])
    if dd < 1e-6:
        return np.inf
    return float(ann_ret / (dd / capital * 100))


def trade_statistics(trades_df: pd.DataFrame) -> dict:
    """Compute trade-level statistics."""
    if trades_df.empty:
        return {}

    closed = trades_df[trades_df["exit_date"] != "OPEN"].copy()
    if closed.empty:
        return {"n_trades": 0}

    winners = closed[closed["total_pnl"] > 0]
    losers  = closed[closed["total_pnl"] < 0]

    hit_rate    = len(winners) / len(closed) if len(closed) > 0 else 0
    avg_win     = winners["total_pnl"].mean() if len(winners) > 0 else 0
    avg_loss    = abs(losers["total_pnl"].mean()) if len(losers) > 0 else 0
    profit_factor = (avg_win * len(winners)) / (avg_loss * len(losers)) \
                    if (avg_loss > 0 and len(losers) > 0) else np.inf

    # Entry mispricing vs trade P&L correlation
    corr = np.nan
    if "mispricing_entry_bps" in closed.columns and len(closed) > 3:
        # For SELL_VOL: positive mispricing should correlate with positive PnL
        sell_trades = closed[closed["direction"] == "SHORT_VOL"]
        if len(sell_trades) > 3:
            corr = sell_trades[["mispricing_entry_bps", "total_pnl"]].corr().iloc[0, 1]

    return {
        "n_trades":             len(closed),
        "n_winners":            len(winners),
        "n_losers":             len(losers),
        "hit_rate_pct":         hit_rate * 100,
        "avg_win":              avg_win,
        "avg_loss":             avg_loss,
        "profit_factor":        profit_factor,
        "total_pnl":            closed["total_pnl"].sum(),
        "best_trade":           closed["total_pnl"].max(),
        "worst_trade":          closed["total_pnl"].min(),
        "mispricing_pnl_corr":  corr,
    }


def greeks_analysis(portfolio_df: pd.DataFrame) -> dict:
    """Analyze Greeks attribution over backtest period."""
    if portfolio_df.empty or "port_theta" not in portfolio_df.columns:
        return {}

    total_theta_collected = portfolio_df["port_theta"].sum()
    avg_gamma_exposure    = portfolio_df["port_gamma"].mean()
    avg_vega_exposure     = portfolio_df["port_vega"].mean()
    avg_positions         = portfolio_df["n_positions"].mean()

    return {
        "total_theta_collected":  total_theta_collected,
        "avg_gamma_exposure":     avg_gamma_exposure,
        "avg_vega_exposure":      avg_vega_exposure,
        "avg_open_positions":     avg_positions,
    }


def full_metrics_report(
    portfolio_df: pd.DataFrame,
    trades_df:    pd.DataFrame,
    capital:      float = 100_000,
    verbose:      bool  = True,
) -> dict:
    """Compute and optionally print full metrics report."""
    ret_m    = compute_returns_metrics(portfolio_df, capital)
    sharpe   = sharpe_ratio(portfolio_df, capital)
    sortino  = sortino_ratio(portfolio_df, capital)
    calmar   = calmar_ratio(portfolio_df, capital)
    dd_m     = max_drawdown(portfolio_df)
    trade_m  = trade_statistics(trades_df)
    greeks_m = greeks_analysis(portfolio_df)

    all_metrics = {
        **ret_m, "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
        **dd_m, **trade_m, **greeks_m
    }

    if verbose:
        _print_report(all_metrics, capital)

    return all_metrics


def _print_report(m: dict, capital: float):
    print(f"\n{'='*65}")
    print(f"  BACKTEST PERFORMANCE REPORT")
    print(f"{'='*65}")

    print(f"\n  ── RETURNS ────────────────────────────────────────")
    print(f"  Total P&L:          ${m.get('total_pnl',0):>12,.2f}")
    print(f"  Total Return:        {m.get('total_return_pct',0):>11.2f}%")
    print(f"  Annualized Return:   {m.get('ann_return_pct',0):>11.2f}%")
    print(f"  Annualized Vol:      {m.get('ann_vol_pct',0):>11.2f}%")

    print(f"\n  ── RISK-ADJUSTED ──────────────────────────────────")
    print(f"  Sharpe Ratio:        {m.get('sharpe',0):>11.3f}")
    print(f"  Sortino Ratio:       {m.get('sortino',0):>11.3f}")
    print(f"  Calmar Ratio:        {m.get('calmar',0):>11.3f}")

    print(f"\n  ── DRAWDOWN ───────────────────────────────────────")
    print(f"  Max Drawdown:       ${m.get('max_dd_pnl',0):>12,.2f}  "
          f"({m.get('max_dd_pnl',0)/capital*100:.2f}% of capital)")
    print(f"  DD Duration:         {m.get('max_dd_duration_days',0):>10} days")
    recovery = m.get('recovery_duration')
    print(f"  DD Recovery:         {str(recovery)+' days' if recovery else 'Not recovered':>10}")

    print(f"\n  ── TRADE STATISTICS ───────────────────────────────")
    print(f"  Total Trades:        {m.get('n_trades',0):>11}")
    print(f"  Hit Rate:            {m.get('hit_rate_pct',0):>11.1f}%")
    print(f"  Profit Factor:       {m.get('profit_factor',0):>11.2f}")
    print(f"  Avg Win:            ${m.get('avg_win',0):>12,.2f}")
    print(f"  Avg Loss:           ${m.get('avg_loss',0):>12,.2f}")
    print(f"  Best Trade:         ${m.get('best_trade',0):>12,.2f}")
    print(f"  Worst Trade:        ${m.get('worst_trade',0):>12,.2f}")
    corr = m.get('mispricing_pnl_corr', float('nan'))
    print(f"  Mispricing→PnL Corr: {corr:>11.3f}" if not (isinstance(corr, float) and np.isnan(corr))
          else f"  Mispricing→PnL Corr:         N/A")

    print(f"\n  ── GREEKS ATTRIBUTION ─────────────────────────────")
    print(f"  Total Theta Collected: {m.get('total_theta_collected',0):>9,.2f}")
    print(f"  Avg Gamma Exposure:    {m.get('avg_gamma_exposure',0):>9.5f}")
    print(f"  Avg Vega Exposure:     {m.get('avg_vega_exposure',0):>9.4f}")
    print(f"  Avg Open Positions:    {m.get('avg_open_positions',0):>9.1f}")
    print()

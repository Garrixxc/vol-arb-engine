"""
data/fetchers/yfinance_fetcher.py

Fetches live SPX options chain from yfinance.
Computes mid-prices, filters bad quotes, prepares
chain for IV calculation.

Note on SPX vs SPY:
- SPX options are cash-settled, European — cleaner for vol surface work
- yfinance doesn't carry SPX well; we use SPY and scale by ~10x
- Or pass any ticker (AAPL, QQQ etc.) for equity options
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date
import warnings

warnings.filterwarnings("ignore")


def fetch_options_chain(
    ticker: str = "SPY",
    min_dte: int = 7,        # Minimum days to expiry — avoid pin risk near expiry
    max_dte: int = 365,      # Maximum DTE — far-dated options are illiquid
    min_open_interest: int = 50,
    min_volume: int = 0,
    spread_filter: float = 1.5,  # Drop quotes where (ask-bid)/mid > this threshold
) -> pd.DataFrame:
    """
    Returns a cleaned DataFrame with columns:
        ticker, expiry, dte, T, strike, option_type,
        bid, ask, mid, open_interest, volume,
        spot, forward  (spot * e^(r-q)*T approximation)
    """
    tk = yf.Ticker(ticker)
    spot = tk.fast_info.get("lastPrice") or tk.fast_info.get("previousClose")
    if spot is None:
        raise ValueError(f"Could not fetch spot price for {ticker}")

    today = date.today()
    expirations = tk.options  # List of expiry strings 'YYYY-MM-DD'

    records = []
    for expiry_str in expirations:
        expiry = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        dte    = (expiry - today).days

        if dte < min_dte or dte > max_dte:
            continue

        T = dte / 365.0

        try:
            chain = tk.option_chain(expiry_str)
        except Exception:
            continue

        for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
            df = df.copy()
            df["option_type"] = opt_type
            df["expiry"]       = expiry_str
            df["dte"]          = dte
            df["T"]            = T
            df["spot"]         = spot
            records.append(df)

    if not records:
        raise ValueError("No options data fetched — check ticker and date range")

    raw = pd.concat(records, ignore_index=True)

    # ── Compute mid price ──────────────────────────────────────
    raw["mid"] = (raw["bid"] + raw["ask"]) / 2.0

    # ── Filters ───────────────────────────────────────────────
    # Remove zero/negative bids
    raw = raw[raw["bid"] > 0]
    raw = raw[raw["ask"] > raw["bid"]]

    # Spread quality filter: (ask - bid) / mid < threshold
    raw["spread_pct"] = (raw["ask"] - raw["bid"]) / raw["mid"]
    raw = raw[raw["spread_pct"] < spread_filter]

    # Open interest filter
    if "openInterest" in raw.columns:
        raw = raw[raw["openInterest"] >= min_open_interest]

    # Volume filter
    if "volume" in raw.columns and min_volume > 0:
        raw = raw[raw["volume"] >= min_volume]

    # Remove deep ITM/OTM by moneyness (keep 0.7 to 1.4 moneyness range)
    raw["moneyness"] = raw["strike"] / raw["spot"]
    raw = raw[(raw["moneyness"] >= 0.7) & (raw["moneyness"] <= 1.4)]

    # ── Clean up columns ──────────────────────────────────────
    cols = ["ticker", "expiry", "dte", "T", "strike", "option_type",
            "bid", "ask", "mid", "spread_pct", "openInterest", "volume",
            "spot", "moneyness"]

    raw["ticker"] = ticker
    available_cols = [c for c in cols if c in raw.columns]
    result = raw[available_cols].copy()
    result = result.sort_values(["expiry", "option_type", "strike"])
    result = result.reset_index(drop=True)

    print(f"[Fetcher] {ticker} @ ${spot:.2f} | "
          f"{result['expiry'].nunique()} expiries | "
          f"{len(result)} contracts after filtering")

    return result, spot


def get_expiry_slice(chain: pd.DataFrame, expiry: str, option_type: str = "call") -> pd.DataFrame:
    """Get a single expiry + option_type slice, sorted by strike."""
    mask = (chain["expiry"] == expiry) & (chain["option_type"] == option_type)
    return chain[mask].sort_values("strike").reset_index(drop=True)


def list_expiries(chain: pd.DataFrame) -> list:
    """Return sorted list of available expiries with DTE."""
    df = chain[["expiry", "dte"]].drop_duplicates().sort_values("dte")
    return list(zip(df["expiry"], df["dte"]))


if __name__ == "__main__":
    # Quick test
    chain, spot = fetch_options_chain("SPY", min_dte=7, max_dte=180)
    print("\nAvailable expiries:")
    for exp, dte in list_expiries(chain):
        n = len(chain[chain["expiry"] == exp])
        print(f"  {exp}  ({dte:>3}d)  —  {n} contracts")
    print(f"\nSample:\n{chain.head(5).to_string()}")

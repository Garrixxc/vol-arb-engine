"""
core/iv_surface.py

Computes implied volatility surface from a raw options chain.
Calls the C++ IV solver (vol_core) in vectorized form.

Output: DataFrame with one row per (expiry, strike, option_type)
containing market_iv, log_moneyness, and metadata.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import vol_core


def compute_iv_surface(
    chain: pd.DataFrame,
    r: float = 0.05,    # Risk-free rate — use current T-bill rate
    q: float = 0.013,   # Dividend yield — SPY ~1.3%
    option_type: str = "call",  # Use calls by default (more liquid ATM)
) -> pd.DataFrame:
    """
    Compute implied vol for each option in the chain.

    We use calls for strikes above spot, puts below spot (put-call parity
    means both should give same IV, but OTM options have tighter spreads
    and more reliable quotes).

    Returns DataFrame with:
        expiry, dte, T, strike, log_moneyness, market_iv, mid_price, spot
    """
    records = []

    for expiry, group in chain.groupby("expiry"):
        spot = group["spot"].iloc[0]
        T    = group["T"].iloc[0]

        # Use OTM options for cleaner quotes:
        # calls for K > spot, puts for K < spot
        calls = group[group["option_type"] == "call"]
        puts  = group[group["option_type"] == "put"]

        otm_calls = calls[calls["strike"] >= spot]
        otm_puts  = puts[puts["strike"] <  spot]
        slice_df  = pd.concat([otm_puts, otm_calls]).sort_values("strike")

        if len(slice_df) < 3:
            continue

        strikes   = slice_df["strike"].values.astype(float)
        prices    = slice_df["mid"].values.astype(float)
        cp_flags  = np.where(slice_df["option_type"].values == "call", 1, -1).astype(np.int32)
        T_arr     = np.full(len(strikes), T)
        S         = float(spot)

        # Call vectorized Python IV solver
        ivs = vol_core.implied_vol_vec(S, strikes, r, q, prices, T_arr, cp_flags)

        for i, (_, row) in enumerate(slice_df.iterrows()):
            iv = ivs[i]
            if np.isnan(iv) or iv <= 0 or iv > 5.0:
                continue  # Skip bad solves

            log_moneyness = np.log(row["strike"] / (S * np.exp((r - q) * T)))

            records.append({
                "expiry":        expiry,
                "dte":           int(row["dte"]),
                "T":             T,
                "strike":        row["strike"],
                "log_moneyness": log_moneyness,
                "option_type":   row["option_type"],
                "market_iv":     iv,
                "mid_price":     row["mid"],
                "spot":          S,
                "spread_pct":    row.get("spread_pct", np.nan),
            })

    if not records:
        raise ValueError("No valid IVs computed — check chain quality")

    df = pd.DataFrame(records).sort_values(["expiry", "strike"]).reset_index(drop=True)

    print(f"[IV Surface] {df['expiry'].nunique()} expiries | "
          f"{len(df)} valid IV points | "
          f"IV range: [{df['market_iv'].min():.1%}, {df['market_iv'].max():.1%}]")

    return df


def plot_smile(iv_surface: pd.DataFrame, expiry: str = None):
    """Quick ASCII smile plot for terminal inspection."""
    if expiry is None:
        # Pick the nearest expiry
        expiry = iv_surface.sort_values("dte")["expiry"].iloc[0]

    smile = iv_surface[iv_surface["expiry"] == expiry].sort_values("log_moneyness")
    dte   = smile["dte"].iloc[0]

    print(f"\nVol Smile — {expiry} ({dte}d)")
    print(f"{'Strike':>8}  {'Log-M':>7}  {'IV':>7}  Bar")
    print("-" * 50)
    for _, row in smile.iterrows():
        bar_len = int(row["market_iv"] * 100)
        bar = "█" * bar_len
        print(f"{row['strike']:>8.1f}  {row['log_moneyness']:>+7.4f}  "
              f"{row['market_iv']:>6.1%}  {bar}")

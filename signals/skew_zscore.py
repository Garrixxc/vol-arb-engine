"""
signals/skew_zscore.py

Tracks the richness/cheapness of the volatility skew over time.

The 25-delta skew is one of the most widely-used vol signals:

    Skew(T) = IV(25Δ put) - IV(25Δ call)

When skew is historically wide (high z-score):
  → OTM puts are expensive vs history → potential sell put vol signal
  → Or the market is pricing in tail risk premium → directional signal

When skew is historically tight (low z-score):
  → Cheap tail protection → potential buy put vol signal

We also track:
  - Skew term structure: near-term vs long-term skew spread
  - Skew per unit ATM vol (normalized skew): Skew / ATM_IV
    This removes the vol-of-vol effect and isolates pure skew richness

All z-scores are computed using a rolling window of historical
surface snapshots stored in DuckDB.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import vol_math


def interpolate_iv_at_delta(
    params,
    T: float,
    target_delta: float,
    spot: float,
    r: float = 0.05,
    q: float = 0.013,
    option_type: str = "put",
    tol: float = 1e-6,
    max_iter: int = 50,
) -> Tuple[float, float]:
    """
    Find the strike K* where BS delta = target_delta, then return
    (K*, IV(K*)) using the fitted SVI surface.

    Uses Newton-Raphson on delta(K) = target_delta.

    For 25Δ put: target_delta = -0.25, option_type = "put"
    For 25Δ call: target_delta = +0.25, option_type = "call"
    """
    import math

    F = spot * np.exp((r - q) * T)
    cp = 1 if option_type == "call" else -1

    # Initial guess: approximate using ATM vol
    atm_iv  = vol_math.svi_vol(0.0, T, params)
    sqrtT   = math.sqrt(T)

    # Approximate strike for target delta:
    # For call: K ≈ F * exp(-Φ⁻¹(Δ)*σ*√T + 0.5*σ²*T)
    # For put:  K ≈ F * exp(-Φ⁻¹(Δ+1)*σ*√T + 0.5*σ²*T)
    from scipy.stats import norm
    if option_type == "call":
        d1_target = norm.ppf(target_delta * math.exp(q * T))
    else:
        d1_target = norm.ppf(target_delta * math.exp(q * T) + 1.0)

    log_K_init = math.log(F) - d1_target * atm_iv * sqrtT + 0.5 * atm_iv**2 * T
    K = math.exp(log_K_init)
    K = max(K, spot * 0.5)
    K = min(K, spot * 1.5)

    # NR loop: find K such that delta(K) = target_delta
    for _ in range(max_iter):
        k_lm  = math.log(K / F)
        iv    = vol_math.svi_vol(k_lm, T, params)
        if iv <= 0:
            break

        g     = vol_math.bs_greeks(spot, K, r, q, iv, T, cp)
        delta = g.delta
        gamma = g.gamma

        if abs(delta - target_delta) < tol:
            break

        # ∂delta/∂K ≈ gamma * ∂S/∂K ... actually ∂delta/∂K from chain rule
        # delta(K) ≈ Φ(d1), ∂delta/∂K = φ(d1) * ∂d1/∂K
        # ∂d1/∂K = -1/(K*σ*√T)
        d1   = (math.log(spot/K) + (r - q + 0.5*iv**2)*T) / (iv * sqrtT)
        from math import exp, pi
        phi_d1 = exp(-0.5*d1**2) / math.sqrt(2*pi)
        dd_dK  = cp * math.exp(-q*T) * phi_d1 * (-1.0 / (K * iv * sqrtT))

        if abs(dd_dK) < 1e-15:
            break
        K = K - (delta - target_delta) / dd_dK
        K = max(K, spot * 0.3)
        K = min(K, spot * 2.0)

    k_lm = math.log(max(K, 1e-6) / F)
    iv   = vol_math.svi_vol(k_lm, T, params)
    return K, iv


def compute_skew_metrics(
    params_by_expiry: Dict,
    spot: float,
    r: float = 0.05,
    q: float = 0.013,
) -> pd.DataFrame:
    """
    Compute skew metrics for each expiry:
        - ATM vol
        - 25Δ put vol
        - 25Δ call vol
        - Skew (25Δ put - 25Δ call)
        - Normalized skew (skew / ATM vol)
        - Risk reversal = 25Δ call - 25Δ put (sign convention varies)
        - Butterfly = 0.5*(25Δ put + 25Δ call) - ATM
    """
    records = []

    for expiry, (params, T) in sorted(params_by_expiry.items(),
                                       key=lambda x: x[1][1]):
        dte = int(round(T * 365))

        # ATM vol
        atm_iv = vol_math.svi_vol(0.0, T, params)

        # 25Δ put and call vols
        try:
            _, iv_25p = interpolate_iv_at_delta(params, T, -0.25, spot, r, q, "put")
            _, iv_25c = interpolate_iv_at_delta(params, T,  0.25, spot, r, q, "call")
        except Exception:
            iv_25p = iv_25c = atm_iv

        skew       = iv_25p - iv_25c          # > 0 means put wing rich (normal)
        norm_skew  = skew / atm_iv if atm_iv > 0 else 0.0
        rr         = iv_25c - iv_25p          # risk reversal
        fly        = 0.5 * (iv_25p + iv_25c) - atm_iv  # butterfly

        records.append({
            "expiry":     expiry,
            "dte":        dte,
            "T":          T,
            "atm_iv":     atm_iv,
            "iv_25p":     iv_25p,
            "iv_25c":     iv_25c,
            "skew":       skew,
            "norm_skew":  norm_skew,
            "rr":         rr,
            "fly":        fly,
        })

    return pd.DataFrame(records).sort_values("dte").reset_index(drop=True)


def compute_skew_zscore(
    current_skew_df: pd.DataFrame,
    history: pd.DataFrame,
    window: int = 20,  # rolling window in days/snapshots
) -> pd.DataFrame:
    """
    Compute z-scores for skew metrics vs historical distribution.

    history: DataFrame with same columns as current_skew_df but
             multiple snapshots (from DuckDB iv_surface history)
             Must have a 'snapshot_ts' column.

    Returns current_skew_df enriched with z-score columns.
    """
    result = current_skew_df.copy()
    result["skew_zscore"]      = np.nan
    result["norm_skew_zscore"] = np.nan
    result["fly_zscore"]       = np.nan
    result["skew_pctile"]      = np.nan
    result["signal"]           = "NEUTRAL"

    if history is None or len(history) == 0:
        return result

    for i, row in result.iterrows():
        expiry = row["expiry"]
        hist_slice = history[history["expiry"] == expiry].tail(window)

        if len(hist_slice) < 5:
            continue

        for metric, col in [("skew", "skew_zscore"),
                              ("norm_skew", "norm_skew_zscore"),
                              ("fly", "fly_zscore")]:
            if metric not in hist_slice.columns:
                continue
            hist_vals = hist_slice[metric].dropna()
            if len(hist_vals) < 3:
                continue
            mu  = hist_vals.mean()
            std = hist_vals.std()
            if std > 1e-8:
                result.at[i, col] = (row[metric] - mu) / std

        if "skew" in hist_slice.columns:
            hist_skew = hist_slice["skew"].dropna()
            if len(hist_skew) > 0:
                result.at[i, "skew_pctile"] = float(
                    (hist_skew <= row["skew"]).mean() * 100
                )

        # Generate signal from normalized skew z-score
        z = result.at[i, "norm_skew_zscore"]
        if not np.isnan(z):
            if z > 1.5:
                result.at[i, "signal"] = "SELL_PUT_SKEW"   # skew unusually wide → sell puts
            elif z < -1.5:
                result.at[i, "signal"] = "BUY_PUT_SKEW"    # skew unusually tight → buy puts
            elif z > 0.8:
                result.at[i, "signal"] = "MILD_SELL_SKEW"
            elif z < -0.8:
                result.at[i, "signal"] = "MILD_BUY_SKEW"

    return result


def print_skew_report(skew_df: pd.DataFrame, title: str = "Skew Report"):
    print(f"\n{'='*75}")
    print(f"  {title}")
    print(f"{'='*75}")
    has_zscore = "skew_zscore" in skew_df.columns and skew_df["skew_zscore"].notna().any()

    if has_zscore:
        print(f"{'Expiry':>12}  {'DTE':>5}  {'ATM':>7}  {'25Δ Put':>8}  "
              f"{'25Δ Call':>9}  {'Skew':>7}  {'Z-Score':>8}  {'Signal':>18}")
        print("-" * 90)
        for _, row in skew_df.iterrows():
            z_str = f"{row.get('skew_zscore', np.nan):+.2f}" if not np.isnan(row.get('skew_zscore', np.nan)) else "  N/A"
            sig   = row.get("signal", "")
            print(f"{row['expiry']:>12}  {row['dte']:>5}  {row['atm_iv']:>6.1%}  "
                  f"{row['iv_25p']:>8.1%}  {row['iv_25c']:>9.1%}  "
                  f"{row['skew']:>6.1%}  {z_str:>8}  {sig:>18}")
    else:
        print(f"{'Expiry':>12}  {'DTE':>5}  {'ATM':>7}  {'25Δ Put':>8}  "
              f"{'25Δ Call':>9}  {'Skew':>7}  {'NormSkew':>9}  {'Butterfly':>10}")
        print("-" * 80)
        for _, row in skew_df.iterrows():
            print(f"{row['expiry']:>12}  {row['dte']:>5}  {row['atm_iv']:>6.1%}  "
                  f"{row['iv_25p']:>8.1%}  {row['iv_25c']:>9.1%}  "
                  f"{row['skew']:>6.1%}  {row['norm_skew']:>+9.3f}  {row['fly']:>+10.3f}")

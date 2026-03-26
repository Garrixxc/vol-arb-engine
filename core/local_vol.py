"""
core/local_vol.py

Extracts the Dupire local volatility surface σ_local(K, T) from
the fitted SVI surface.

Dupire's formula (in terms of total implied variance w(k, T)):

    σ²_local(k, T) = ∂w/∂T / D(k, w)

where the denominator D is:

    D = 1 - (k/w) * ∂w/∂k
        + (1/4) * (-1/4 - 1/w + k²/w²) * (∂w/∂k)²
        + (1/2) * ∂²w/∂k²

This is Gatheral's (2011) formulation of Dupire in log-moneyness space.

Why local vol matters:
  - Local vol is the unique diffusion σ(S,t) consistent with all
    European option prices simultaneously
  - If local vol goes negative anywhere → your SVI surface has a
    calendar spread arbitrage (free money)
  - Used for pricing path-dependent products (barriers, Asians)
  - Comparing implied vol surface vs local vol surface reveals
    structural information about forward skew and term structure
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import vol_core


def local_vol_surface(
    params_by_expiry: Dict[str, Tuple],
    spot: float,
    r: float = 0.05,
    q: float = 0.013,
    k_grid: np.ndarray = None,
    dT: float = 1e-4,        # finite difference step for ∂w/∂T
) -> pd.DataFrame:
    """
    Compute Dupire local vol surface across all expiries.

    Uses finite differences in T for ∂w/∂T, and analytical
    SVI derivatives for the k-derivatives.

    Returns DataFrame with columns:
        expiry, T, k, K, implied_vol, local_vol, is_valid
    """
    if k_grid is None:
        k_grid = np.linspace(-0.35, 0.25, 120)

    # Sort by T
    sorted_items = sorted(params_by_expiry.items(), key=lambda x: x[1][1])
    expiries = [e for e, _ in sorted_items]
    Ts       = [t for _, (_, t) in sorted_items]
    params   = [p for _, (p, _) in sorted_items]

    records = []

    for i, (expiry, p, T) in enumerate(zip(expiries, params, Ts)):
        # ── ∂w/∂T via central finite differences ───────────────
        # Use neighbouring expiries when available, else one-sided
        if i == 0 and len(Ts) > 1:
            # Forward difference
            T2, p2 = Ts[1], params[1]
            dT_actual = T2 - T
            w1 = np.array([vol_core.svi_w(k, p)  for k in k_grid])
            w2 = np.array([vol_core.svi_w(k, p2) for k in k_grid])
            dw_dT = (w2 - w1) / dT_actual

        elif i == len(Ts) - 1 and len(Ts) > 1:
            # Backward difference
            T0, p0 = Ts[i-1], params[i-1]
            dT_actual = T - T0
            w0 = np.array([vol_core.svi_w(k, p0) for k in k_grid])
            w1 = np.array([vol_core.svi_w(k, p)  for k in k_grid])
            dw_dT = (w1 - w0) / dT_actual

        elif len(Ts) > 2:
            # Central difference
            T0, p0 = Ts[i-1], params[i-1]
            T2, p2 = Ts[i+1], params[i+1]
            w0 = np.array([vol_core.svi_w(k, p0) for k in k_grid])
            w2 = np.array([vol_core.svi_w(k, p2) for k in k_grid])
            dw_dT = (w2 - w0) / (T2 - T0)
        else:
            # Only one expiry — can't compute ∂w/∂T
            continue

        # ── Analytical k-derivatives of SVI ────────────────────
        a, b, rho, m, sigma = p.a, p.b, p.rho, p.m, p.sigma
        dk   = k_grid - m
        disc = np.sqrt(dk**2 + sigma**2)

        w    = a + b * (rho * dk + disc)
        w_1  = b * (rho + dk / disc)           # ∂w/∂k
        w_2  = b * sigma**2 / disc**3          # ∂²w/∂k²

        w = np.maximum(w, 1e-10)

        # ── Dupire denominator D(k, w) ──────────────────────────
        D = (1.0 - (k_grid / w) * w_1
             + 0.25 * (-0.25 - 1.0/w + k_grid**2 / w**2) * w_1**2
             + 0.5 * w_2)

        # ── Local variance σ²_local = (∂w/∂T) / D ──────────────
        dw_dT_clipped = np.maximum(dw_dT, 1e-10)  # must be > 0 (calendar no-arb)
        local_var = dw_dT_clipped / np.maximum(D, 1e-10)

        # Negative local var = calendar arb in surface
        is_valid  = (local_var > 0) & (D > 0) & (dw_dT > 0)
        local_vol_arr = np.where(is_valid, np.sqrt(local_var), np.nan)

        F     = spot * np.exp((r - q) * T)
        K_arr = F * np.exp(k_grid)
        iv_arr = vol_core.svi_vol_vec(k_grid, T, p)

        for j in range(len(k_grid)):
            records.append({
                "expiry":       expiry,
                "T":            T,
                "k":            k_grid[j],
                "K":            K_arr[j],
                "implied_vol":  iv_arr[j],
                "local_vol":    local_vol_arr[j],
                "dw_dT":        dw_dT[j],
                "D":            D[j],
                "is_valid":     bool(is_valid[j]),
            })

    df = pd.DataFrame(records)
    return df


def local_vol_summary(lv_df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Summarize local vol surface quality.
    Key check: what fraction of the surface has valid (positive) local vol?
    """
    valid_pct = lv_df["is_valid"].mean() * 100
    atm_rows  = lv_df[(lv_df["k"].abs() < 0.02) & lv_df["is_valid"]]

    summary = {
        "valid_pct":      valid_pct,
        "n_invalid":      (~lv_df["is_valid"]).sum(),
        "atm_local_vols": atm_rows.groupby("expiry")["local_vol"].mean().to_dict(),
    }

    if verbose:
        print("\n" + "=" * 65)
        print("  Dupire Local Vol Surface Summary")
        print("=" * 65)
        print(f"  Surface validity: {valid_pct:.1f}% of grid points are valid")
        if summary["n_invalid"] > 0:
            print(f"  ⚠ {summary['n_invalid']} invalid points (calendar arb indicator)")
        print(f"\n  {'Expiry':>12}  {'ATM Impl Vol':>13}  {'ATM Local Vol':>14}")
        print("  " + "-" * 45)
        for expiry, lv in sorted(summary["atm_local_vols"].items()):
            iv_row = lv_df[(lv_df["expiry"]==expiry) & (lv_df["k"].abs() < 0.02)]
            iv_mean = iv_row["implied_vol"].mean() if len(iv_row) else np.nan
            print(f"  {expiry:>12}  {iv_mean:>12.1%}  {lv:>13.1%}")

    return summary

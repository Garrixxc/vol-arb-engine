"""
core/breeden_litzenberger.py

Extracts the risk-neutral (risk-neutral) probability density from
the fitted SVI surface using the Breeden-Litzenberger formula:

    q(K) = e^(rT) * ∂²C / ∂K²

In terms of total variance w(k) and log-moneyness k = log(K/F):

    q(K) = g(k) / (F * √(2π * w)) * exp(-d2²/2)

where g(k) is the Gatheral density factor from no_arb_checks.py

Why this matters:
  - The density must be non-negative (butterfly no-arb condition)
  - Fat left tail = expensive OTM puts = fear premium
  - Right skew = call premium = demand for upside exposure
  - You can compare model density vs market-implied density to
    find structural dislocations in the smile
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import vol_core


def risk_neutral_density(
    params,
    T: float,
    spot: float,
    r: float = 0.05,
    q: float = 0.013,
    n_points: int = 500,
    k_range: tuple = (-0.5, 0.5),
) -> pd.DataFrame:
    """
    Compute the risk-neutral density q(K) from SVI params.

    Returns DataFrame with columns:
        K, k (log-moneyness), w, vol, density, cum_prob
    """
    F = spot * np.exp((r - q) * T)  # Forward price

    k_grid = np.linspace(k_range[0], k_range[1], n_points)
    K_grid = F * np.exp(k_grid)

    a, b, rho, m, sigma = params.a, params.b, params.rho, params.m, params.sigma

    dk    = k_grid - m
    disc  = np.sqrt(dk**2 + sigma**2)

    w     = a + b * (rho * dk + disc)
    w_1   = b * (rho + dk / disc)            # ∂w/∂k
    w_2   = b * sigma**2 / disc**3           # ∂²w/∂k²

    w     = np.maximum(w, 1e-10)
    vol   = np.sqrt(w / T)

    # d2 in Black-Scholes
    d2    = -k_grid / np.sqrt(w) - 0.5 * np.sqrt(w)

    # Gatheral g(k) factor
    g     = (1.0 - k_grid * w_1 / (2.0 * w))**2 \
            - w_1**2 / 4.0 * (1.0 / w + 0.25) \
            + w_2 / 2.0

    # Risk-neutral density per unit log-moneyness
    # q(k) = g(k) * phi(d2) / √(2π * w)
    # where phi is the standard normal PDF
    phi_d2   = np.exp(-0.5 * d2**2) / np.sqrt(2 * np.pi)
    density_k = np.maximum(g, 0.0) * phi_d2 / np.sqrt(w)

    # Convert to density per unit K: q(K) = q(k) / (K * √(2πw))
    # (Jacobian from log-moneyness to strike)
    density_K = density_k / K_grid

    # Normalize to integrate to ~1 (discrete approximation)
    dk_step     = k_grid[1] - k_grid[0]
    total_prob  = np.trapezoid(density_k, k_grid)
    if total_prob > 0:
        density_k_norm = density_k / total_prob
    else:
        density_k_norm = density_k

    cum_prob = np.cumsum(density_k_norm) * dk_step

    return pd.DataFrame({
        "K":          K_grid,
        "k":          k_grid,
        "w":          w,
        "vol":        vol,
        "density_k":  density_k_norm,   # density per unit log-moneyness (normalized)
        "density_K":  density_K,        # density per unit K (raw)
        "cum_prob":   np.clip(cum_prob, 0, 1),
        "d2":         d2,
        "g":          g,                # Gatheral factor (must be ≥ 0)
    })


def density_moments(density_df: pd.DataFrame) -> dict:
    """
    Compute moments of the risk-neutral distribution:
        mean, variance, skewness, excess kurtosis
    Uses the density per unit log-moneyness (density_k).
    """
    k      = density_df["k"].values
    q      = density_df["density_k"].values
    dk     = k[1] - k[0]

    mean   = np.trapezoid(k * q, k)
    var    = np.trapezoid((k - mean)**2 * q, k)
    std    = np.sqrt(max(var, 1e-10))
    skew   = np.trapezoid(((k - mean) / std)**3 * q, k)
    kurt   = np.trapezoid(((k - mean) / std)**4 * q, k) - 3.0

    return {
        "mean_logm":  mean,
        "variance":   var,
        "std":        std,
        "skewness":   skew,        # negative = left-skewed = put fear premium
        "ex_kurtosis": kurt,       # positive = fat tails
    }


def print_density_summary(params_by_expiry: dict,
                          spot: float, r: float = 0.05, q: float = 0.013):
    """Print risk-neutral density moments across all expiries."""
    print("\n" + "=" * 70)
    print("  Risk-Neutral Density Moments (Breeden-Litzenberger)")
    print("=" * 70)
    print(f"{'Expiry':>12}  {'DTE':>5}  {'Skewness':>10}  {'Ex.Kurt':>10}  {'Std(k)':>8}")
    print("-" * 55)

    for expiry, (params, T) in sorted(params_by_expiry.items(),
                                       key=lambda x: x[1][1]):
        dte  = int(round(T * 365))
        df   = risk_neutral_density(params, T, spot, r, q)
        moms = density_moments(df)
        print(f"{expiry:>12}  {dte:>5}  {moms['skewness']:>+10.4f}  "
              f"{moms['ex_kurtosis']:>+10.4f}  {moms['std']:>8.4f}")

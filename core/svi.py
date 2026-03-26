"""
core/svi.py

Fits SVI model to the full implied vol surface (all expiries).
Orchestrates:
  1. Per-expiry SVI calibration (C++ LM solver)
  2. No-arb constraint checking
  3. Mispricing signal computation
  4. Fitted surface output for downstream use

The "smart" initialisation:
  Rather than cold-starting each expiry, we use the neighbouring
  expiry's fitted params as warm start. This dramatically improves
  convergence and avoids local minima.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import vol_math
from no_arb_checks import full_arb_report


# ─────────────────────────────────────────────
#  Heuristic initial guess for SVI params
#  based on ATM vol and rough skew estimate
# ─────────────────────────────────────────────
def _initial_guess(iv_slice: pd.DataFrame, T: float) -> vol_math.SVIParams:
    p = vol_math.SVIParams()

    # ATM total variance
    atm_row = iv_slice.iloc[(iv_slice["log_moneyness"].abs()).argsort()].iloc[0]
    atm_iv  = atm_row["market_iv"]
    atm_w   = atm_iv**2 * T

    # Skew estimate: slope of IV vs log_moneyness
    if len(iv_slice) >= 4:
        slope = np.polyfit(iv_slice["log_moneyness"], iv_slice["market_iv"], 1)[0]
    else:
        slope = -0.1

    p.a     = atm_w * 0.8        # Most variance comes from a
    p.b     = max(atm_w * 0.3, 0.01)
    p.rho   = np.clip(slope / (atm_iv + 0.01) * 0.5, -0.95, 0.95)
    p.m     = 0.0
    p.sigma = 0.15
    return p


def fit_surface(
    iv_surface: pd.DataFrame,
    min_points: int = 5,
    verbose: bool = True,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Fit SVI to every expiry in the IV surface DataFrame.

    Returns:
        params_by_expiry: {expiry: (SVIParams, T)}
        fitted_df:        DataFrame with market_iv, svi_iv, mispricing columns
    """
    expiries = sorted(iv_surface["expiry"].unique(),
                      key=lambda e: iv_surface[iv_surface["expiry"]==e]["T"].iloc[0])

    params_by_expiry = {}
    fitted_records   = []
    prev_params      = None  # warm-start from previous expiry

    if verbose:
        print("\n" + "=" * 70)
        print("  SVI Surface Calibration")
        print("=" * 70)
        print(f"{'Expiry':>12}  {'DTE':>5}  {'RMSE':>10}  {'Iter':>6}  {'Status':>10}")
        print("-" * 55)

    for expiry in expiries:
        slice_df = iv_surface[iv_surface["expiry"] == expiry].copy()
        slice_df = slice_df.dropna(subset=["market_iv", "log_moneyness"])
        slice_df = slice_df.sort_values("log_moneyness")

        T   = float(slice_df["T"].iloc[0])
        dte = int(slice_df["dte"].iloc[0])

        if len(slice_df) < min_points:
            if verbose:
                print(f"{expiry:>12}  {dte:>5}  {'SKIPPED':>10}  {'—':>6}  (too few points)")
            continue

        ks   = slice_df["log_moneyness"].values.tolist()
        ivs  = slice_df["market_iv"].values.tolist()

        # Warm start: use previous expiry params or heuristic guess
        init = prev_params if prev_params is not None else _initial_guess(slice_df, T)

        # Fit SVI — Pure Python solver
        result = vol_math.calibrate_svi(ks, ivs, T, init)
        p      = result.params

        status = "✓ OK" if result.converged else "⚠ WARN"
        if verbose:
            print(f"{expiry:>12}  {dte:>5}  {result.rmse:>10.6f}  "
                  f"{result.iterations:>6}  {status:>10}")

        params_by_expiry[expiry] = (p, T)
        prev_params = p  # warm start for next expiry

        # ── Build fitted records ──────────────────────────────
        k_dense = np.linspace(
            slice_df["log_moneyness"].min() - 0.02,
            slice_df["log_moneyness"].max() + 0.02,
            150
        )
        svi_vols_dense = vol_math.svi_vol_vec(k_dense, T, p)

        # Market points vs model
        for _, row in slice_df.iterrows():
            k         = row["log_moneyness"]
            mkt_iv    = row["market_iv"]
            model_iv  = vol_math.svi_vol(k, T, p)
            mispricing = mkt_iv - model_iv  # +ve = market richer than model

            # Normalize by bid-ask spread for signal quality
            spread    = row.get("spread_pct", np.nan)
            snr       = abs(mispricing) / (mkt_iv * spread + 1e-9) if not np.isnan(spread) else np.nan

            fitted_records.append({
                "expiry":        expiry,
                "dte":           dte,
                "T":             T,
                "strike":        row["strike"],
                "log_moneyness": k,
                "option_type":   row.get("option_type", ""),
                "market_iv":     mkt_iv,
                "svi_iv":        model_iv,
                "mispricing":    mispricing,   # in vol points
                "mispricing_bps": mispricing * 10000,
                "snr":           snr,
                "spot":          row.get("spot", np.nan),
            })

        # Add dense model curve rows (no market IV) for plotting
        for k, sv in zip(k_dense, svi_vols_dense):
            fitted_records.append({
                "expiry": expiry, "dte": dte, "T": T,
                "strike": np.nan, "log_moneyness": k,
                "option_type": "model",
                "market_iv": np.nan, "svi_iv": sv,
                "mispricing": np.nan, "mispricing_bps": np.nan,
                "snr": np.nan, "spot": np.nan,
            })

    fitted_df = pd.DataFrame(fitted_records)

    # ── No-arb checks ─────────────────────────────────────────
    if params_by_expiry:
        arb_results = full_arb_report(params_by_expiry, verbose=verbose)
    else:
        arb_results = {}

    if verbose:
        # Mispricing summary
        mkt_rows = fitted_df[fitted_df["option_type"] != "model"].dropna(subset=["mispricing"])
        if len(mkt_rows):
            print(f"\nMispricing summary across surface:")
            print(f"  Mean |mispricing|: {mkt_rows['mispricing'].abs().mean()*10000:.1f} bps")
            print(f"  Max  |mispricing|: {mkt_rows['mispricing'].abs().max()*10000:.1f} bps")
            top = mkt_rows.nlargest(5, "snr")[["expiry","strike","market_iv","svi_iv","mispricing_bps","snr"]]
            print(f"\n  Top 5 mispricings by SNR:")
            print(top.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    return params_by_expiry, fitted_df, arb_results

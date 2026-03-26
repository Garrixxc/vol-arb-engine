"""
signals/term_structure.py

Analyzes the volatility term structure for trading signals.

The vol term structure σ_ATM(T) carries rich information:

1. CONTANGO vs BACKWARDATION
   Normal markets: σ(T_long) > σ(T_short)  (contango)
   Stressed markets: σ(T_short) > σ(T_long) (backwardation — fear spike)

   Signal: When term structure is very steep (unusually high contango),
   calendar spreads are attractive — buy short-dated vol, sell long-dated.

2. KINKS / DISLOCATIONS
   The term structure should be smooth. A kink at a specific expiry
   often means an event is priced in (earnings, FOMC, election).
   If the kink is disproportionate → arb opportunity between
   adjacent expiries.

3. VIX-STYLE TERM PREMIUM
   Fit a parametric curve to ATM vols: σ(T) = a + b*exp(-c*T)
   Residuals from this curve = over/under-priced expiries.

4. VARIANCE RISK PREMIUM PROXY
   Compare ATM implied vol to a rolling realized vol estimate.
   Persistent gap → structural premium to sell.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from typing import Dict, Tuple, Optional
import vol_core


# ─────────────────────────────────────────────
#  Parametric term structure model
#  σ(T) = L + (S - L) * exp(-k * T)
#  L = long-run level, S = short-run level, k = mean-reversion speed
#  This is the Samuelson effect parametrization
# ─────────────────────────────────────────────
def _ts_model(T, L, S, k):
    return L + (S - L) * np.exp(-k * T)


def fit_term_structure_curve(
    atm_vols: pd.DataFrame,   # columns: T, atm_iv, expiry
) -> Tuple[dict, np.ndarray]:
    """
    Fit parametric curve to ATM vol term structure.
    Returns (params_dict, residuals_array).
    """
    T_arr  = atm_vols["T"].values
    iv_arr = atm_vols["atm_iv"].values

    if len(T_arr) < 3:
        return {}, np.zeros(len(T_arr))

    try:
        p0 = [iv_arr[-1], iv_arr[0], 2.0]   # L=long vol, S=short vol, k=speed
        bounds = ([0.01, 0.01, 0.1], [2.0, 2.0, 20.0])
        popt, _ = curve_fit(_ts_model, T_arr, iv_arr, p0=p0,
                            bounds=bounds, maxfev=5000)
        fitted  = _ts_model(T_arr, *popt)
        resids  = iv_arr - fitted

        return {
            "L": popt[0],   # long-run ATM vol
            "S": popt[1],   # short-run ATM vol (extrapolated)
            "k": popt[2],   # mean-reversion speed
            "contango": popt[0] - popt[1],  # L - S: positive = normal contango
            "half_life_days": np.log(2) / popt[2] * 365,
        }, resids

    except Exception:
        return {}, np.zeros(len(T_arr))


def compute_term_structure_signals(
    params_by_expiry: Dict,
    spot: float,
    r: float = 0.05,
    q: float = 0.013,
    realized_vol: Optional[float] = None,   # 30d realized vol for VRP
) -> dict:
    """
    Full term structure analysis.

    Returns a dict with:
        ts_df        — ATM vols + curve fit + residuals per expiry
        curve_params — fitted parametric curve parameters
        signals      — list of signal dicts
        vrp          — variance risk premium (if realized_vol provided)
    """
    # ── Build ATM vol term structure ──────────────────────────
    records = []
    for expiry, (params, T) in sorted(params_by_expiry.items(),
                                       key=lambda x: x[1][1]):
        atm_iv = vol_core.svi_vol(0.0, T, params)
        dte    = int(round(T * 365))

        # Also compute 10Δ put vol for tail richness signal
        try:
            F    = spot * np.exp((r - 0.013) * T)
            # Approximate 10Δ put log-moneyness
            try:
                from scipy.stats import norm
            except ImportError:
                import math
                class norm:
                    @staticmethod
                    def ppf(p): return math.sqrt(2)*math.erfinv(2*p-1)
            d1_10d = norm.ppf(0.10 + 1.0)  # put delta = -0.10 → N(d1) ≈ 0.10
            lm_10p = -d1_10d * atm_iv * np.sqrt(T) + 0.5 * atm_iv**2 * T
            iv_10p = vol_core.svi_vol(lm_10p, T, params)
        except Exception:
            iv_10p = atm_iv

        records.append({
            "expiry": expiry, "dte": dte, "T": T,
            "atm_iv": atm_iv,
            "iv_10p": iv_10p,
            "tail_premium": iv_10p - atm_iv,  # extra cost of tail protection
        })

    ts_df = pd.DataFrame(records).sort_values("T").reset_index(drop=True)

    if len(ts_df) == 0:
        return {"ts_df": ts_df, "curve_params": {}, "signals": [], "vrp": None}

    # ── Fit parametric curve ──────────────────────────────────
    curve_params, residuals = fit_term_structure_curve(ts_df)
    ts_df["curve_iv"]  = _ts_model(ts_df["T"].values, **{
        k: curve_params[k] for k in ["L", "S", "k"]
    }) if curve_params else ts_df["atm_iv"]
    ts_df["residual"]      = residuals  # > 0 = expensive vs curve
    ts_df["residual_bps"]  = residuals * 10000

    # ── Generate signals ──────────────────────────────────────
    signals = []

    # 1. Calendar spread signal: kinks in term structure
    if len(ts_df) >= 2:
        for i in range(len(ts_df) - 1):
            r1, r2 = ts_df.iloc[i], ts_df.iloc[i+1]
            # Expected smoothness: residual shouldn't flip sign sharply
            if abs(r1["residual_bps"]) > 30:
                direction = "SELL" if r1["residual_bps"] > 0 else "BUY"
                signals.append({
                    "type":      "TERM_KINK",
                    "expiry":    r1["expiry"],
                    "dte":       r1["dte"],
                    "direction": f"{direction}_CALENDAR",
                    "magnitude_bps": abs(r1["residual_bps"]),
                    "desc": (
                        f"{direction} {r1['expiry']} vol ({r1['atm_iv']:.1%}) — "
                        f"{abs(r1['residual_bps']):.0f}bps {'rich' if r1['residual_bps']>0 else 'cheap'} "
                        f"vs parametric curve"
                    )
                })

    # 2. Contango steepness signal
    if curve_params:
        contango_ann = curve_params.get("contango", 0)
        if abs(contango_ann) > 0.05:   # > 5 vol points contango/backwardation
            direction = "STEEP_CONTANGO" if contango_ann > 0 else "BACKWARDATION"
            signals.append({
                "type":      "TERM_SHAPE",
                "direction": direction,
                "magnitude_bps": abs(contango_ann) * 10000,
                "desc": (
                    f"Term structure {direction}: "
                    f"short={curve_params['S']:.1%}, long={curve_params['L']:.1%}, "
                    f"half-life={curve_params.get('half_life_days',0):.0f}d"
                )
            })

    # 3. Variance Risk Premium
    vrp = None
    if realized_vol is not None:
        # Use 30d ATM implied vol
        near_30d = ts_df.iloc[(ts_df["dte"] - 30).abs().argsort()].iloc[0]
        vrp = near_30d["atm_iv"] - realized_vol
        vrp_bps = vrp * 10000
        if abs(vrp_bps) > 100:  # > 10 vol points premium
            direction = "SELL_VOL" if vrp > 0 else "BUY_VOL"
            signals.append({
                "type":      "VRP",
                "direction": direction,
                "magnitude_bps": abs(vrp_bps),
                "desc": (
                    f"VRP signal: implied={near_30d['atm_iv']:.1%} vs "
                    f"realized={realized_vol:.1%} → {vrp_bps:+.0f}bps premium → {direction}"
                )
            })

    return {
        "ts_df":        ts_df,
        "curve_params": curve_params,
        "signals":      signals,
        "vrp":          vrp,
    }


def print_term_structure_report(ts_result: dict):
    ts_df  = ts_result["ts_df"]
    params = ts_result["curve_params"]
    sigs   = ts_result["signals"]

    print(f"\n{'='*75}")
    print(f"  Volatility Term Structure Analysis")
    print(f"{'='*75}")

    if params:
        print(f"  Curve fit: L={params['L']:.1%}  S={params['S']:.1%}  "
              f"k={params['k']:.2f}  half-life={params.get('half_life_days',0):.0f}d  "
              f"contango={params['contango']:+.1%}")

    print(f"\n  {'Expiry':>12}  {'DTE':>5}  {'ATM IV':>8}  "
          f"{'Curve':>8}  {'Resid(bps)':>11}  {'10Δ Put':>8}  Tail Prem")
    print("  " + "-" * 70)
    for _, row in ts_df.iterrows():
        resid_flag = " ◄ RICH" if row["residual_bps"] > 30 else \
                     " ◄ CHEAP" if row["residual_bps"] < -30 else ""
        print(f"  {row['expiry']:>12}  {row['dte']:>5}  {row['atm_iv']:>7.1%}  "
              f"{row['curve_iv']:>7.1%}  {row['residual_bps']:>+10.1f}  "
              f"{row['iv_10p']:>7.1%}  {row['tail_premium']:>+7.1%}{resid_flag}")

    if sigs:
        print(f"\n  Signals detected ({len(sigs)}):")
        for s in sigs:
            print(f"    [{s['type']:>12}] {s['desc']}")
    else:
        print(f"\n  No term structure signals above threshold.")

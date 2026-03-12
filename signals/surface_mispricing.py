"""
signals/surface_mispricing.py

Converts raw SVI fit residuals into actionable trade signals.

The core idea:
  After fitting SVI to the market surface, the residuals
      mispricing(k, T) = σ_market(k,T) - σ_SVI(k,T)
  are not all equal. We need to filter for:

  1. MAGNITUDE    — is the mispricing large enough vs bid-ask spread?
  2. LIQUIDITY    — is there enough OI/volume to actually trade?
  3. PERSISTENCE  — has this strike been consistently mispriced
                    (structural) or is it noise?
  4. DIRECTION    — is the market paying too much (sell signal)
                    or too little (buy signal)?

Signal quality score (0-100):
    score = w1 * SNR_score
          + w2 * liquidity_score
          + w3 * persistence_score
          + w4 * moneyness_score   (prefer OTM options, more liquid)

Trade direction:
    mispricing > 0  → market IV > model IV → sell vol (short gamma)
    mispricing < 0  → market IV < model IV → buy vol  (long gamma)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


# Signal score weights
W_SNR         = 0.40   # bid-ask adjusted mispricing
W_LIQUIDITY   = 0.25   # open interest + volume
W_MONEYNESS   = 0.20   # prefer near-the-money (tighter spreads, more liquid)
W_TERM        = 0.15   # prefer medium-term (30-90d, more vol to trade)


@dataclass
class MispricingSignal:
    expiry:          str
    dte:             int
    strike:          float
    option_type:     str
    direction:       str          # "SELL_VOL" or "BUY_VOL"
    market_iv:       float
    model_iv:        float
    mispricing_bps:  float        # market - model, in basis points
    snr:             float        # mispricing / half-spread
    score:           float        # composite 0-100 signal quality
    log_moneyness:   float
    open_interest:   int
    spot:            float
    # Suggested trade
    trade_desc:      str = ""

    def __post_init__(self):
        direction_str = "SELL" if self.mispricing_bps > 0 else "BUY"
        otm = "OTM" if (
            (self.option_type == "call" and self.log_moneyness > 0) or
            (self.option_type == "put"  and self.log_moneyness < 0)
        ) else "ITM"
        self.trade_desc = (
            f"{direction_str} {otm} {self.option_type.upper()} "
            f"K={self.strike:.0f} {self.expiry} | "
            f"{abs(self.mispricing_bps):.0f}bps {'rich' if self.mispricing_bps > 0 else 'cheap'} | "
            f"delta-hedge to isolate vol"
        )


def compute_mispricing_signals(
    fitted_df:   pd.DataFrame,
    chain:       pd.DataFrame,
    min_snr:     float = 0.8,    # minimum signal-to-noise ratio
    min_oi:      int   = 50,     # minimum open interest
    min_dte:     int   = 7,      # avoid pin risk
    max_dte:     int   = 180,    # illiquid beyond this
    top_n:       int   = 20,     # return top N signals
) -> List[MispricingSignal]:
    """
    Score and rank mispricing signals from the fitted surface.

    fitted_df: output of svi.fit_surface() — contains market_iv, svi_iv columns
    chain:     raw options chain — used for OI/volume lookup
    """

    # ── Get market rows only (not the dense model curve) ──────
    mkt = fitted_df[
        (fitted_df["option_type"] != "model") &
        fitted_df["mispricing"].notna() &
        fitted_df["snr"].notna()
    ].copy()

    if len(mkt) == 0:
        return []

    # ── Merge in OI/volume from raw chain ─────────────────────
    chain_lookup = chain[["expiry", "strike", "option_type", "openInterest", "volume"]].copy()
    chain_lookup.columns = ["expiry", "strike", "option_type", "open_interest", "volume"]
    mkt = mkt.merge(chain_lookup, on=["expiry", "strike", "option_type"], how="left")
    mkt["open_interest"] = mkt["open_interest"].fillna(0).astype(int)
    mkt["volume"]        = mkt["volume"].fillna(0).astype(int)

    # ── DTE filter ─────────────────────────────────────────────
    mkt = mkt[(mkt["dte"] >= min_dte) & (mkt["dte"] <= max_dte)]

    # ── Compute sub-scores ────────────────────────────────────

    # 1. SNR score (0-100): scaled by min/max
    snr_vals = mkt["snr"].clip(0, 5)
    mkt["snr_score"] = (snr_vals / snr_vals.max() * 100).fillna(0) if snr_vals.max() > 0 else 0.0

    # 2. Liquidity score: log(OI + 1), normalized
    oi_log = np.log1p(mkt["open_interest"])
    mkt["liq_score"] = (oi_log / oi_log.max() * 100).fillna(0) if oi_log.max() > 0 else 0.0

    # 3. Moneyness score: prefer |log_moneyness| in [0.02, 0.15]
    #    Peak at ~5-15d OTM, falls off for deep OTM and ITM
    lm_abs = mkt["log_moneyness"].abs()
    mkt["mono_score"] = np.where(
        lm_abs < 0.02,  50.0,                        # ATM: decent
        np.where(
            lm_abs <= 0.15,
            100.0 * np.exp(-((lm_abs - 0.07)**2) / (2 * 0.05**2)),  # peak at ~7% OTM
            20.0 * np.exp(-(lm_abs - 0.15) * 10)    # deep OTM: drops fast
        )
    )

    # 4. Term score: prefer 20-90 DTE (sweet spot for vol trading)
    dte_vals = mkt["dte"]
    mkt["term_score"] = np.where(
        (dte_vals >= 20) & (dte_vals <= 90),
        100.0,
        np.where(
            dte_vals < 20,
            dte_vals / 20.0 * 70,
            np.maximum(0, 100 - (dte_vals - 90) * 0.8)
        )
    )

    # ── Composite score ───────────────────────────────────────
    mkt["score"] = (
        W_SNR       * mkt["snr_score"] +
        W_LIQUIDITY * mkt["liq_score"] +
        W_MONEYNESS * mkt["mono_score"] +
        W_TERM      * mkt["term_score"]
    )

    # ── Filter by minimum SNR ──────────────────────────────────
    filtered = mkt[mkt["snr"] >= min_snr].copy()

    if len(filtered) == 0:
        # Relax threshold and return best available
        filtered = mkt.nlargest(min(top_n, len(mkt)), "score")

    # ── Sort and build signal objects ─────────────────────────
    filtered = filtered.nlargest(top_n, "score")

    signals = []
    for _, row in filtered.iterrows():
        direction = "SELL_VOL" if row["mispricing"] > 0 else "BUY_VOL"
        sig = MispricingSignal(
            expiry         = row["expiry"],
            dte            = int(row["dte"]),
            strike         = row["strike"],
            option_type    = row["option_type"],
            direction      = direction,
            market_iv      = row["market_iv"],
            model_iv       = row["svi_iv"],
            mispricing_bps = row["mispricing_bps"],
            snr            = row["snr"],
            score          = row["score"],
            log_moneyness  = row["log_moneyness"],
            open_interest  = int(row["open_interest"]),
            spot           = row.get("spot", 0.0),
        )
        signals.append(sig)

    return signals


def print_signals(signals: List[MispricingSignal], title: str = "Mispricing Signals"):
    print(f"\n{'='*75}")
    print(f"  {title}  ({len(signals)} signals)")
    print(f"{'='*75}")
    print(f"{'#':>3}  {'Expiry':>12}  {'K':>7}  {'Type':>5}  "
          f"{'Misprice':>10}  {'SNR':>6}  {'Score':>7}  Direction")
    print("-" * 75)
    for i, s in enumerate(signals, 1):
        mp_str = f"{s.mispricing_bps:+.0f}bps"
        print(f"{i:>3}  {s.expiry:>12}  {s.strike:>7.1f}  {s.option_type:>5}  "
              f"{mp_str:>10}  {s.snr:>6.2f}  {s.score:>7.1f}  {s.direction}")
    print()
    print("  Trade descriptions:")
    for i, s in enumerate(signals[:5], 1):
        print(f"  {i}. {s.trade_desc}")

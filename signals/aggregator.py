"""
signals/aggregator.py

Combines all signal sources into a single ranked, deduplicated
trade opportunity list.

Signal sources:
  1. Surface mispricing  — per-strike SVI residuals
  2. Skew z-score        — 25Δ skew richness/cheapness vs history
  3. Term structure      — calendar spread kinks, VRP, contango

Aggregation logic:
  - Each signal is tagged with: type, direction, expiry, strike (optional),
    magnitude, score, confidence
  - Signals on the same expiry/strike are merged and scores boosted
    if multiple sources agree (confluence)
  - Opposing signals on same expiry cancel or reduce score
  - Final output sorted by composite score descending

Confluence boost:
  If surface mispricing AND skew z-score both say SELL PUT on same
  expiry → score multiplied by 1.4 (two independent signals agree)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

from signals.surface_mispricing import MispricingSignal
from signals.skew_zscore import compute_skew_metrics, print_skew_report
from signals.term_structure import compute_term_structure_signals, print_term_structure_report


@dataclass
class TradeOpportunity:
    # Identity
    id:           str
    expiry:       str
    dte:          int
    strike:       Optional[float]
    option_type:  Optional[str]

    # Signal
    direction:    str            # SELL_VOL / BUY_VOL / SELL_CALENDAR / etc.
    sources:      List[str]      # which signals fired
    confluence:   int            # number of agreeing signals

    # Magnitudes
    mispricing_bps: Optional[float]
    skew_zscore:    Optional[float]
    ts_residual_bps: Optional[float]

    # Scores
    raw_score:    float
    final_score:  float          # after confluence boost

    # Trade description
    trade_type:   str            # SINGLE_LEG / SPREAD / CALENDAR
    trade_desc:   str

    timestamp:    str = field(default_factory=lambda: datetime.utcnow().isoformat())


CONFLUENCE_BOOST = 1.35   # score multiplier per additional agreeing signal


def aggregate_signals(
    mispricing_signals: List[MispricingSignal],
    skew_df:            pd.DataFrame,
    ts_result:          dict,
    params_by_expiry:   dict,
    spot:               float,
    top_n:              int = 15,
) -> List[TradeOpportunity]:
    """
    Aggregate all signals into ranked trade opportunities.
    """
    opportunities: Dict[str, dict] = {}   # key → opportunity dict

    # ── Source 1: Surface mispricing signals ──────────────────
    for sig in mispricing_signals:
        key = f"{sig.expiry}_{sig.strike}_{sig.option_type}"
        if key not in opportunities:
            opportunities[key] = {
                "expiry":       sig.expiry,
                "dte":          sig.dte,
                "strike":       sig.strike,
                "option_type":  sig.option_type,
                "direction":    sig.direction,
                "sources":      ["SURFACE_MISPRICING"],
                "mispricing_bps": sig.mispricing_bps,
                "skew_zscore":  None,
                "ts_residual":  None,
                "score":        sig.score,
                "trade_type":   "SINGLE_LEG",
            }
        else:
            opportunities[key]["sources"].append("SURFACE_MISPRICING")
            opportunities[key]["mispricing_bps"] = sig.mispricing_bps
            opportunities[key]["score"] = max(opportunities[key]["score"], sig.score)

    # ── Source 2: Skew z-score signals ────────────────────────
    for _, row in skew_df.iterrows():
        signal = row.get("signal", "NEUTRAL")
        if signal == "NEUTRAL" or pd.isna(signal):
            continue

        expiry = row["expiry"]
        direction = (
            "SELL_VOL" if "SELL" in signal else
            "BUY_VOL"  if "BUY"  in signal else "NEUTRAL"
        )
        z     = row.get("skew_zscore", 0.0) or 0.0
        score = min(abs(z) * 20, 80)   # scale z-score to 0-80

        # Skew signals target the put wing specifically
        key = f"{expiry}_skew"
        if key not in opportunities:
            opportunities[key] = {
                "expiry":      expiry,
                "dte":         int(row["dte"]),
                "strike":      None,
                "option_type": "put",
                "direction":   direction,
                "sources":     ["SKEW_ZSCORE"],
                "mispricing_bps": None,
                "skew_zscore": z,
                "ts_residual": None,
                "score":       score,
                "trade_type":  "SINGLE_LEG",
            }
        else:
            opportunities[key]["sources"].append("SKEW_ZSCORE")
            opportunities[key]["skew_zscore"] = z
            # Boost if direction agrees
            if opportunities[key]["direction"] == direction:
                opportunities[key]["score"] *= CONFLUENCE_BOOST

    # ── Source 3: Term structure signals ──────────────────────
    for ts_sig in ts_result.get("signals", []):
        sig_type  = ts_sig["type"]
        direction = ts_sig.get("direction", "")
        mag_bps   = ts_sig.get("magnitude_bps", 0)
        expiry    = ts_sig.get("expiry", "term_structure")
        dte       = ts_sig.get("dte", 0)
        score     = min(mag_bps / 10, 80)

        key = f"{expiry}_{sig_type}"
        if key not in opportunities:
            opportunities[key] = {
                "expiry":      expiry,
                "dte":         dte,
                "strike":      None,
                "option_type": None,
                "direction":   direction,
                "sources":     [sig_type],
                "mispricing_bps": None,
                "skew_zscore": None,
                "ts_residual": mag_bps,
                "score":       score,
                "trade_type":  "CALENDAR" if "CALENDAR" in direction else "SINGLE_LEG",
            }
        else:
            opportunities[key]["sources"].append(sig_type)
            if opportunities[key]["direction"] == direction:
                opportunities[key]["score"] *= CONFLUENCE_BOOST

    # ── Build TradeOpportunity objects ────────────────────────
    result = []
    for key, opp in opportunities.items():
        confluence = len(opp["sources"])
        raw_score  = opp["score"]
        # Apply confluence boost for multi-source agreement
        final_score = raw_score * (CONFLUENCE_BOOST ** (confluence - 1))
        final_score = min(final_score, 100.0)

        # Build human-readable trade description
        strike_str = f"K={opp['strike']:.0f}" if opp["strike"] else "ATM area"
        desc_parts = [
            f"{opp['direction']} {opp.get('option_type','vol') or 'vol'} {strike_str}",
            f"{opp['expiry']} ({opp['dte']}d)",
        ]
        if opp["mispricing_bps"] is not None:
            desc_parts.append(f"mispricing={opp['mispricing_bps']:+.0f}bps")
        if opp["skew_zscore"] is not None:
            desc_parts.append(f"skew_z={opp['skew_zscore']:+.2f}")
        if opp["ts_residual"] is not None:
            desc_parts.append(f"ts_resid={opp['ts_residual']:+.0f}bps")
        desc_parts.append(f"[{', '.join(opp['sources'])}]")

        op = TradeOpportunity(
            id              = key,
            expiry          = opp["expiry"],
            dte             = int(opp["dte"]),
            strike          = opp.get("strike"),
            option_type     = opp.get("option_type"),
            direction       = opp["direction"],
            sources         = opp["sources"],
            confluence      = confluence,
            mispricing_bps  = opp.get("mispricing_bps"),
            skew_zscore     = opp.get("skew_zscore"),
            ts_residual_bps = opp.get("ts_residual"),
            raw_score       = raw_score,
            final_score     = final_score,
            trade_type      = opp["trade_type"],
            trade_desc      = " | ".join(desc_parts),
        )
        result.append(op)

    # Sort by final score descending
    result.sort(key=lambda x: x.final_score, reverse=True)
    return result[:top_n]


def print_trade_opportunities(opportunities: List[TradeOpportunity]):
    print(f"\n{'='*85}")
    print(f"  RANKED TRADE OPPORTUNITIES  ({len(opportunities)} total)")
    print(f"{'='*85}")
    print(f"{'#':>3}  {'Score':>7}  {'Conf':>5}  {'Direction':>22}  "
          f"{'Expiry':>12}  {'Strike':>8}  Sources")
    print("-" * 85)

    for i, op in enumerate(opportunities, 1):
        strike_str = f"{op.strike:>8.1f}" if op.strike else "     ATM"
        conf_str   = "★" * op.confluence + "·" * (3 - min(op.confluence, 3))
        print(f"{i:>3}  {op.final_score:>7.1f}  {conf_str:>5}  "
              f"{op.direction:>22}  {op.expiry:>12}  {strike_str}  "
              f"{', '.join(op.sources)}")

    print(f"\n  Top 5 trade descriptions:")
    print("  " + "-" * 75)
    for i, op in enumerate(opportunities[:5], 1):
        print(f"  {i}. [{op.final_score:.1f}] {op.trade_desc}")


def run_full_signal_pipeline(
    fitted_df:          pd.DataFrame,
    chain:              pd.DataFrame,
    params_by_expiry:   dict,
    spot:               float,
    r:                  float = 0.05,
    q:                  float = 0.013,
    realized_vol:       Optional[float] = None,
    skew_history:       Optional[pd.DataFrame] = None,
    verbose:            bool = True,
) -> List[TradeOpportunity]:
    """
    Master function: run all signal modules and aggregate.
    Call this from main.py or the dashboard.
    """
    from signals.surface_mispricing import compute_mispricing_signals, print_signals
    from signals.skew_zscore import compute_skew_metrics, compute_skew_zscore, print_skew_report
    from signals.term_structure import compute_term_structure_signals, print_term_structure_report

    # 1. Surface mispricing
    mispricing_sigs = compute_mispricing_signals(
        fitted_df, chain, min_snr=0.3, top_n=30
    )
    if verbose:
        print_signals(mispricing_sigs, "Surface Mispricing Signals")

    # 2. Skew z-scores
    skew_df = compute_skew_metrics(params_by_expiry, spot, r, q)
    skew_df = compute_skew_zscore(skew_df, skew_history)
    if verbose:
        print_skew_report(skew_df)

    # 3. Term structure
    ts_result = compute_term_structure_signals(
        params_by_expiry, spot, r, q, realized_vol
    )
    if verbose:
        print_term_structure_report(ts_result)

    # 4. Aggregate
    opportunities = aggregate_signals(
        mispricing_sigs, skew_df, ts_result,
        params_by_expiry, spot
    )

    if verbose:
        print_trade_opportunities(opportunities)

    return opportunities

"""
core/no_arb_checks.py

Enforces the two fundamental no-arbitrage conditions on a vol surface:

1. BUTTERFLY ARBITRAGE (per expiry slice)
   A single expiry slice is butterfly-arb-free iff the risk-neutral
   density is non-negative everywhere:

       g(k) = (1 - k*∂w/∂k / (2w))² - (∂w/∂k)²/4 * (1/w + 1/4)
               + ∂²w/∂k² / 2  ≥ 0   ∀k

   This is the Gatheral (2004) condition derived from Breeden-Litzenberger.
   For SVI, this can be checked analytically.

2. CALENDAR SPREAD ARBITRAGE (across expiries)
   Total variance must be strictly increasing in time for fixed
   log-moneyness:

       w(k, T₁) ≤ w(k, T₂)   for T₁ < T₂, ∀k

   Violation means you could buy a calendar spread for negative cost
   — free money.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import vol_core


@dataclass
class ArbCheckResult:
    expiry:            str
    butterfly_free:    bool
    calendar_free:     bool
    min_density:       float        # min of g(k) — negative means butterfly arb
    calendar_violations: List[str]  # expiry pairs that violate calendar condition
    violation_strikes: np.ndarray = field(default_factory=lambda: np.array([]))


def check_butterfly(params, T: float,
                    k_grid: np.ndarray = None) -> Tuple[bool, float, np.ndarray]:
    """
    Check butterfly no-arb condition for a single SVI slice.

    Returns:
        (is_free, min_g, k_violations)

    The Gatheral g(k) function:
        Let w  = svi_w(k)
            w' = ∂w/∂k
            w'' = ∂²w/∂k²

        g(k) = (1 - k*w'/(2w))² - w'²/4*(1/w + 1/4) + w''/2

    For SVI w(k) = a + b*(ρ*(k-m) + √((k-m)²+σ²)):
        w'  = b * (ρ + (k-m)/√((k-m)²+σ²))
        w'' = b * σ² / ((k-m)²+σ²)^(3/2)
    """
    if k_grid is None:
        k_grid = np.linspace(-0.5, 0.5, 200)

    a, b, rho, m, sigma = params.a, params.b, params.rho, params.m, params.sigma

    dk    = k_grid - m
    disc  = np.sqrt(dk**2 + sigma**2)

    w     = a + b * (rho * dk + disc)
    w_1   = b * (rho + dk / disc)                    # ∂w/∂k
    w_2   = b * sigma**2 / disc**3                   # ∂²w/∂k²

    # Clip w to avoid division by zero in degenerate cases
    w_safe = np.maximum(w, 1e-12)

    g = (1.0 - k_grid * w_1 / (2.0 * w_safe))**2 \
        - w_1**2 / 4.0 * (1.0 / w_safe + 0.25) \
        + w_2 / 2.0

    min_g   = float(g.min())
    is_free = min_g >= -1e-8  # small tolerance for numerical noise
    violations = k_grid[g < -1e-8]

    return is_free, min_g, violations


def check_calendar(params_by_expiry: Dict[str, Tuple],
                   k_grid: np.ndarray = None) -> Tuple[bool, List[str]]:
    """
    Check calendar spread no-arb across all expiries.

    params_by_expiry: dict of {expiry_str: (SVIParams, T)}
    sorted by T (nearest first).

    Returns:
        (is_free, list_of_violating_expiry_pairs)
    """
    if k_grid is None:
        k_grid = np.linspace(-0.4, 0.4, 100)

    # Sort by T
    items = sorted(params_by_expiry.items(), key=lambda x: x[1][1])
    violations = []

    for i in range(len(items) - 1):
        exp1, (p1, T1) = items[i]
        exp2, (p2, T2) = items[i + 1]

        w1 = np.array([vol_core.svi_w(k, p1) for k in k_grid])
        w2 = np.array([vol_core.svi_w(k, p2) for k in k_grid])

        # Calendar arb: w2 must be >= w1 everywhere
        # Allow small tolerance for numerical noise
        if np.any(w2 < w1 - 1e-8):
            n_viols = np.sum(w2 < w1 - 1e-8)
            violations.append(
                f"{exp1} → {exp2}: {n_viols} strikes violate w(T₁) ≤ w(T₂)"
            )

    return len(violations) == 0, violations


def full_arb_report(params_by_expiry: Dict[str, Tuple],
                    verbose: bool = True) -> Dict[str, ArbCheckResult]:
    """
    Run full butterfly + calendar arb check on the fitted surface.

    params_by_expiry: {expiry: (SVIParams, T)}
    """
    _, calendar_violations = check_calendar(params_by_expiry)
    results = {}

    for expiry, (params, T) in params_by_expiry.items():
        bf_free, min_g, viols = check_butterfly(params, T)

        cal_viols_for_this = [v for v in calendar_violations if expiry in v]

        result = ArbCheckResult(
            expiry=expiry,
            butterfly_free=bf_free,
            calendar_free=len(cal_viols_for_this) == 0,
            min_density=min_g,
            calendar_violations=cal_viols_for_this,
            violation_strikes=viols,
        )
        results[expiry] = result

    if verbose:
        print("\n" + "=" * 65)
        print("  No-Arbitrage Surface Check")
        print("=" * 65)
        print(f"{'Expiry':>12}  {'Butterfly':>10}  {'Calendar':>10}  {'Min g(k)':>10}")
        print("-" * 55)
        for expiry, r in sorted(results.items()):
            bf  = "✓ FREE" if r.butterfly_free else "✗ ARB!"
            cal = "✓ FREE" if r.calendar_free  else "✗ ARB!"
            print(f"{expiry:>12}  {bf:>10}  {cal:>10}  {r.min_density:>+10.6f}")

        all_bf  = all(r.butterfly_free for r in results.values())
        all_cal = all(r.calendar_free  for r in results.values())
        print("-" * 55)
        print(f"{'SURFACE':>12}  {'✓ CLEAN' if all_bf else '✗ ARBS':>10}  "
              f"{'✓ CLEAN' if all_cal else '✗ ARBS':>10}")
        if calendar_violations:
            print("\nCalendar violations:")
            for v in calendar_violations:
                print(f"  {v}")

    return results

#include <algorithm>
#include "iv_solver.h"
#include "black_scholes.h"
#include <cmath>
#include <stdexcept>
#include <limits>

// ─────────────────────────────────────────────
//  Implied Volatility Solver
//
//  Strategy:
//  1. Start with Brenner-Subrahmanyam initial guess
//     (closed-form ATM approximation — fast, decent accuracy)
//  2. Refine with Newton-Raphson using vega as derivative
//     (quadratic convergence near the solution)
//  3. Fall back to Brent's method if NR diverges
//     (robust bracket search — always converges)
//
//  Why this matters: you'll call this ~10,000 times per
//  surface calibration, so every microsecond counts
// ─────────────────────────────────────────────

static const double IV_TOL     = 1e-8;   // convergence tolerance
static const int    MAX_ITER_NR = 50;    // Newton-Raphson max iterations
static const int    MAX_ITER_BR = 100;   // Brent max iterations
static const double IV_MIN     = 1e-6;   // vol floor
static const double IV_MAX     = 10.0;   // vol ceiling (1000%)

// Brenner-Subrahmanyam ATM approximation: σ ≈ (C/S) * √(2π/T)
static double initial_guess(double S, double K, double r, double q,
                             double price, double T, int call_put) {
    // Forward price
    double F = S * std::exp((r - q) * T);
    // Use ATM approximation as starting point regardless of moneyness
    double atm_guess = std::sqrt(2.0 * M_PI / T) * price / F;

    // Clamp to sensible range
    if (atm_guess < IV_MIN) atm_guess = 0.1;
    if (atm_guess > IV_MAX) atm_guess = 2.0;
    return atm_guess;
}

// ─────────────────────────────────────────────
//  Newton-Raphson step:
//  σ_new = σ - (BS(σ) - market_price) / vega(σ)
// ─────────────────────────────────────────────
static double newton_raphson(double S, double K, double r, double q,
                              double T, int call_put, double market_price,
                              double sigma_init) {
    double sigma = sigma_init;

    for (int i = 0; i < MAX_ITER_NR; ++i) {
        double price = bs_price(S, K, r, q, sigma, T, call_put);
        double diff  = price - market_price;

        if (std::abs(diff) < IV_TOL) return sigma;

        // Vega in raw form (not per 1%) for the NR step
        double sqrtT = std::sqrt(T);
        double d1    = (std::log(S / K) + (r - q + 0.5*sigma*sigma)*T) / (sigma * sqrtT);
        double vega  = S * std::exp(-q * T) * norm_pdf(d1) * sqrtT;

        if (vega < 1e-12) break;  // Too small vega → fall back to Brent

        double sigma_new = sigma - diff / vega;

        // Keep within bounds
        if (sigma_new < IV_MIN) sigma_new = 0.5 * (sigma + IV_MIN);
        if (sigma_new > IV_MAX) sigma_new = 0.5 * (sigma + IV_MAX);

        sigma = sigma_new;
    }
    return std::numeric_limits<double>::quiet_NaN();  // NR failed, signal fallback
}

// ─────────────────────────────────────────────
//  Brent's method fallback
//  Guaranteed convergence on [a, b] if signs differ
// ─────────────────────────────────────────────
static double brent(double S, double K, double r, double q,
                    double T, int call_put, double market_price) {
    auto f = [&](double v) {
        return bs_price(S, K, r, q, v, T, call_put) - market_price;
    };

    double a = IV_MIN, b = IV_MAX;
    double fa = f(a), fb = f(b);

    if (fa * fb > 0) return std::numeric_limits<double>::quiet_NaN();

    double c = a, fc = fa, s = 0, fs = 0;
    double d = 0;
    bool mflag = true;

    for (int i = 0; i < MAX_ITER_BR; ++i) {
        if (std::abs(b - a) < IV_TOL) return (a + b) * 0.5;

        if (fa != fc && fb != fc) {
            // Inverse quadratic interpolation
            s = a*fb*fc/((fa-fb)*(fa-fc))
              + b*fa*fc/((fb-fa)*(fb-fc))
              + c*fa*fb/((fc-fa)*(fc-fb));
        } else {
            // Secant method
            s = b - fb * (b - a) / (fb - fa);
        }

        bool cond1 = !((3*a+b)/4 < s && s < b);
        bool cond2 = mflag  && std::abs(s-b) >= std::abs(b-c)/2;
        bool cond3 = !mflag && std::abs(s-b) >= std::abs(c-d)/2;

        if (cond1 || cond2 || cond3) {
            s = (a + b) * 0.5;
            mflag = true;
        } else {
            mflag = false;
        }

        fs = f(s);
        d = c; c = b; fc = fb;

        if (fa * fs < 0) { b = s; fb = fs; }
        else             { a = s; fa = fs; }

        if (std::abs(fa) < std::abs(fb)) {
            std::swap(a, b);
            std::swap(fa, fb);
        }
    }
    return (a + b) * 0.5;
}

// ─────────────────────────────────────────────
//  Public interface
// ─────────────────────────────────────────────
double implied_vol(double S, double K, double r, double q,
                   double market_price, double T, int call_put) {
    // Basic sanity checks
    if (T <= 0)            return std::numeric_limits<double>::quiet_NaN();
    if (market_price <= 0) return std::numeric_limits<double>::quiet_NaN();

    // Intrinsic value check — price below intrinsic has no solution
    double intrinsic = std::max(call_put * (S * std::exp(-q*T) - K * std::exp(-r*T)), 0.0);
    if (market_price < intrinsic - IV_TOL)
        return std::numeric_limits<double>::quiet_NaN();

    double guess = initial_guess(S, K, r, q, market_price, T, call_put);
    double iv    = newton_raphson(S, K, r, q, T, call_put, market_price, guess);

    // If NR converged and is in bounds, return it
    if (!std::isnan(iv) && iv > IV_MIN && iv < IV_MAX) {
        double check = bs_price(S, K, r, q, iv, T, call_put);
        if (std::abs(check - market_price) < IV_TOL * 10) return iv;
    }

    // Fall back to Brent
    return brent(S, K, r, q, T, call_put, market_price);
}

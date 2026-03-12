#include <algorithm>
#include "black_scholes.h"
#include <cmath>
#include <stdexcept>

// ─────────────────────────────────────────────
//  Standard normal CDF and PDF
//  Using Abramowitz & Stegun approximation
//  Max error: 7.5e-8 — fast and accurate enough
// ─────────────────────────────────────────────
double norm_cdf(double x) {
    static const double a1 =  0.254829592;
    static const double a2 = -0.284496736;
    static const double a3 =  1.421413741;
    static const double a4 = -1.453152027;
    static const double a5 =  1.061405429;
    static const double p  =  0.3275911;

    int sign = (x < 0) ? -1 : 1;
    x = std::abs(x) / std::sqrt(2.0);
    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t * std::exp(-x*x);
    return 0.5 * (1.0 + sign * y);
}

double norm_pdf(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

// ─────────────────────────────────────────────
//  Core d1, d2 computation
//  S  = spot price
//  K  = strike
//  r  = risk-free rate (continuously compounded)
//  q  = dividend yield
//  v  = volatility (annualized)
//  T  = time to expiry (years)
// ─────────────────────────────────────────────
static void compute_d1_d2(double S, double K, double r, double q,
                           double v, double T, double &d1, double &d2) {
    double vT = v * std::sqrt(T);
    d1 = (std::log(S / K) + (r - q + 0.5 * v * v) * T) / vT;
    d2 = d1 - vT;
}

// ─────────────────────────────────────────────
//  Black-Scholes Price
//  call_put: 1 = call, -1 = put
// ─────────────────────────────────────────────
double bs_price(double S, double K, double r, double q,
                double v, double T, int call_put) {
    if (T <= 0.0) {
        // At expiry: intrinsic value only
        return std::max(call_put * (S - K), 0.0);
    }
    if (v <= 0.0) throw std::invalid_argument("Volatility must be positive");

    double d1, d2;
    compute_d1_d2(S, K, r, q, v, T, d1, d2);

    double cp = static_cast<double>(call_put);
    return cp * (S * std::exp(-q * T) * norm_cdf(cp * d1)
                 - K * std::exp(-r * T) * norm_cdf(cp * d2));
}

// ─────────────────────────────────────────────
//  Greeks
// ─────────────────────────────────────────────
Greeks bs_greeks(double S, double K, double r, double q,
                 double v, double T, int call_put) {
    Greeks g{};
    if (T <= 0.0) return g;

    double d1, d2;
    compute_d1_d2(S, K, r, q, v, T, d1, d2);

    double cp     = static_cast<double>(call_put);
    double sqrtT  = std::sqrt(T);
    double expqT  = std::exp(-q * T);
    double exprT  = std::exp(-r * T);
    double nd1    = norm_pdf(d1);

    // Delta: ∂C/∂S
    g.delta = cp * expqT * norm_cdf(cp * d1);

    // Gamma: ∂²C/∂S²  (same for calls and puts)
    g.gamma = expqT * nd1 / (S * v * sqrtT);

    // Vega: ∂C/∂σ  (per 1% move in vol → divide by 100)
    g.vega = S * expqT * nd1 * sqrtT / 100.0;

    // Theta: ∂C/∂T  (per calendar day → divide by 365)
    g.theta = (- S * expqT * nd1 * v / (2.0 * sqrtT)
               - cp * r * K * exprT * norm_cdf(cp * d2)
               + cp * q * S * expqT * norm_cdf(cp * d1)) / 365.0;

    // Rho: ∂C/∂r  (per 1% move in rates → divide by 100)
    g.rho = cp * K * T * exprT * norm_cdf(cp * d2) / 100.0;

    return g;
}

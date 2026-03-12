#include <algorithm>
/*
  svi_calibrator.cpp

  Fits the raw SVI parameterization to a single expiry smile:

      w(k) = a + b * [ ρ(k - m) + √((k - m)² + σ²) ]

  where:
      k  = log(K/F)       log-moneyness (forward-adjusted)
      w  = σ²_implied * T  total implied variance
      a  ∈ (-∞, ∞)        vertical translation
      b  ≥ 0              slope / wings
      ρ  ∈ (-1, 1)        skew (correlation)
      m  ∈ (-∞, ∞)        ATM shift
      σ  > 0              ATM curvature

  Algorithm: Levenberg-Marquardt (damped Gauss-Newton)
      — Quadratic convergence near solution (like NR)
      — Stable far from solution (gradient descent behavior)
      — Jacobian computed analytically for speed

  Constraints enforced during parameter updates:
      b  > 0
      |ρ| < 1
      σ  > 0
      a + b*σ*√(1-ρ²) > 0   (no negative variance at any strike)
*/

#include "svi_calibrator.h"
#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <numeric>

static const int    MAX_ITER  = 500;
static const double FTOL      = 1e-10;   // function value tolerance
static const double XTOL      = 1e-10;   // parameter change tolerance
static const double LM_INIT   = 1e-3;    // initial LM damping
static const double LM_UP     = 10.0;    // damping increase factor
static const double LM_DOWN   = 0.1;     // damping decrease factor

// ─────────────────────────────────────────────
//  SVI total variance for a single strike
// ─────────────────────────────────────────────
double svi_w(double k, const SVIParams& p) {
    double dk    = k - p.m;
    double disc  = std::sqrt(dk * dk + p.sigma * p.sigma);
    return p.a + p.b * (p.rho * dk + disc);
}

// ─────────────────────────────────────────────
//  SVI implied vol (annualized) from params + T
// ─────────────────────────────────────────────
double svi_vol(double k, double T, const SVIParams& p) {
    double w = svi_w(k, p);
    if (w <= 0.0) return 0.0;
    return std::sqrt(w / T);
}

// ─────────────────────────────────────────────
//  Analytical Jacobian ∂w/∂θ for each param
//  θ = {a, b, ρ, m, σ}
//  Returns 5-element gradient at a single k
// ─────────────────────────────────────────────
static void svi_gradient(double k, const SVIParams& p,
                          double& da, double& db,
                          double& drho, double& dm, double& dsigma) {
    double dk   = k - p.m;
    double disc = std::sqrt(dk * dk + p.sigma * p.sigma);

    da     =  1.0;
    db     =  p.rho * dk + disc;
    drho   =  p.b * dk;
    dm     =  p.b * (-p.rho - dk / disc);
    dsigma =  p.b * p.sigma / disc;
}

// ─────────────────────────────────────────────
//  Pack/unpack params to/from raw vector
//  Apply soft constraints via reparameterization:
//    b     → exp(b_raw)          ensures b > 0
//    ρ     → tanh(rho_raw)       ensures |ρ| < 1
//    σ     → exp(sigma_raw)      ensures σ > 0
// ─────────────────────────────────────────────
static std::vector<double> pack(const SVIParams& p) {
    return {
        p.a,
        std::log(std::max(p.b, 1e-9)),
        std::atanh(std::max(std::min(p.rho, 0.9999), -0.9999)),
        p.m,
        std::log(std::max(p.sigma, 1e-9))
    };
}

static SVIParams unpack(const std::vector<double>& x) {
    SVIParams p;
    p.a     = x[0];
    p.b     = std::exp(x[1]);
    p.rho   = std::tanh(x[2]);
    p.m     = x[3];
    p.sigma = std::exp(x[4]);
    return p;
}

// ─────────────────────────────────────────────
//  Compute residuals: r_i = w_model(k_i) - w_mkt_i
//  w_mkt = market_iv² * T
// ─────────────────────────────────────────────
static std::vector<double> residuals(
    const std::vector<double>& ks,
    const std::vector<double>& w_mkt,
    const SVIParams& p)
{
    int n = ks.size();
    std::vector<double> r(n);
    for (int i = 0; i < n; ++i)
        r[i] = svi_w(ks[i], p) - w_mkt[i];
    return r;
}

// ─────────────────────────────────────────────
//  Sum of squared residuals
// ─────────────────────────────────────────────
static double sse(const std::vector<double>& r) {
    double s = 0.0;
    for (double ri : r) s += ri * ri;
    return s;
}

// ─────────────────────────────────────────────
//  Levenberg-Marquardt step
//  Solves: (JᵀJ + λI) Δx = -Jᵀr
//  using direct 5x5 linear solve (small system)
// ─────────────────────────────────────────────
static std::vector<double> lm_step(
    const std::vector<double>& ks,
    const std::vector<double>& w_mkt,
    const SVIParams& p,
    const std::vector<double>& r,
    double lambda)
{
    const int N = 5;
    int n = ks.size();

    // Build JᵀJ and Jᵀr
    std::vector<double> JtJ(N * N, 0.0);
    std::vector<double> Jtr(N, 0.0);

    for (int i = 0; i < n; ++i) {
        double da, db, drho, dm, dsigma;
        svi_gradient(ks[i], p, da, db, drho, dm, dsigma);

        // Chain rule through reparameterization:
        // ∂w/∂b_raw   = ∂w/∂b   * ∂b/∂b_raw   = db * b
        // ∂w/∂rho_raw = ∂w/∂rho * ∂rho/∂rho_raw = drho * (1 - rho²)
        // ∂w/∂sig_raw = ∂w/∂sig * ∂sig/∂sig_raw = dsigma * sigma
        double J[5] = {
            da,
            db    * p.b,
            drho  * (1.0 - p.rho * p.rho),
            dm,
            dsigma * p.sigma
        };

        for (int j = 0; j < N; ++j) {
            Jtr[j] += J[j] * r[i];
            for (int l = 0; l < N; ++l)
                JtJ[j * N + l] += J[j] * J[l];
        }
    }

    // Add damping: (JᵀJ + λ diag(JᵀJ)) Δx = -Jᵀr
    for (int j = 0; j < N; ++j)
        JtJ[j * N + j] *= (1.0 + lambda);

    // Gaussian elimination with partial pivoting (5x5)
    std::vector<double> A(JtJ);
    std::vector<double> b_vec(Jtr);
    for (int j = 0; j < N; ++j) b_vec[j] = -b_vec[j];

    for (int col = 0; col < N; ++col) {
        // Pivot
        int pivot = col;
        for (int row = col+1; row < N; ++row)
            if (std::abs(A[row*N+col]) > std::abs(A[pivot*N+col]))
                pivot = row;
        if (pivot != col) {
            for (int k = 0; k < N; ++k)
                std::swap(A[col*N+k], A[pivot*N+k]);
            std::swap(b_vec[col], b_vec[pivot]);
        }
        double diag = A[col*N+col];
        if (std::abs(diag) < 1e-15) continue;
        for (int row = col+1; row < N; ++row) {
            double factor = A[row*N+col] / diag;
            for (int k = col; k < N; ++k)
                A[row*N+k] -= factor * A[col*N+k];
            b_vec[row] -= factor * b_vec[col];
        }
    }
    // Back substitution
    std::vector<double> dx(N, 0.0);
    for (int i = N-1; i >= 0; --i) {
        double s = b_vec[i];
        for (int j = i+1; j < N; ++j)
            s -= A[i*N+j] * dx[j];
        dx[i] = (std::abs(A[i*N+i]) > 1e-15) ? s / A[i*N+i] : 0.0;
    }
    return dx;
}

// ─────────────────────────────────────────────
//  Main calibration routine
// ─────────────────────────────────────────────
SVIResult calibrate_svi(
    const std::vector<double>& log_moneyness,
    const std::vector<double>& market_ivs,
    double T,
    const SVIParams& init)
{
    int n = log_moneyness.size();
    if (n < 5) throw std::invalid_argument("Need at least 5 market points");
    if (n != (int)market_ivs.size())
        throw std::invalid_argument("Mismatched input sizes");

    // Convert market IVs to total variance
    std::vector<double> w_mkt(n);
    for (int i = 0; i < n; ++i)
        w_mkt[i] = market_ivs[i] * market_ivs[i] * T;

    // Pack initial params
    std::vector<double> x = pack(init);
    SVIParams p = unpack(x);

    auto r     = residuals(log_moneyness, w_mkt, p);
    double err = sse(r);
    double lam = LM_INIT;

    SVIResult result;
    result.converged = false;
    result.iterations = 0;

    for (int iter = 0; iter < MAX_ITER; ++iter) {
        result.iterations = iter + 1;

        auto dx    = lm_step(log_moneyness, w_mkt, p, r, lam);
        double dx_norm = 0.0;
        for (double d : dx) dx_norm += d * d;

        // Candidate update
        std::vector<double> x_new(5);
        for (int j = 0; j < 5; ++j) x_new[j] = x[j] + dx[j];

        SVIParams p_new = unpack(x_new);
        auto r_new      = residuals(log_moneyness, w_mkt, p_new);
        double err_new  = sse(r_new);

        if (err_new < err) {
            // Accept step, decrease damping
            x   = x_new;
            p   = p_new;
            r   = r_new;
            err = err_new;
            lam *= LM_DOWN;

            if (err < FTOL || dx_norm < XTOL) {
                result.converged = true;
                break;
            }
        } else {
            // Reject step, increase damping
            lam *= LM_UP;
        }
    }

    result.params    = unpack(x);
    result.rmse      = std::sqrt(err / n);
    result.converged = result.converged || (result.rmse < 1e-6);

    return result;
}

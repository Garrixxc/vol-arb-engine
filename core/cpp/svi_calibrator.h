#pragma once
#include <vector>

struct SVIParams {
    double a     = 0.04;   // vertical translation
    double b     = 0.15;   // slope / wings
    double rho   = -0.5;   // skew
    double m     = 0.0;    // ATM shift
    double sigma = 0.15;   // ATM curvature
};

struct SVIResult {
    SVIParams params;
    double    rmse       = 0.0;
    bool      converged  = false;
    int       iterations = 0;
};

// Evaluate SVI total variance w(k) at log-moneyness k
double svi_w(double k, const SVIParams& p);

// Evaluate SVI implied vol at log-moneyness k and expiry T
double svi_vol(double k, double T, const SVIParams& p);

// Calibrate SVI to market data via Levenberg-Marquardt
SVIResult calibrate_svi(
    const std::vector<double>& log_moneyness,
    const std::vector<double>& market_ivs,
    double T,
    const SVIParams& init = SVIParams{}
);

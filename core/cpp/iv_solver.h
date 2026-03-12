#pragma once

// Implied volatility via Newton-Raphson + Brent fallback
// Returns NaN if no solution exists (deep ITM/OTM, bad price etc.)
double implied_vol(double S, double K, double r, double q,
                   double market_price, double T, int call_put);

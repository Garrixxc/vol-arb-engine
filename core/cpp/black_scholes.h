#pragma once

struct Greeks {
    double delta = 0.0;
    double gamma = 0.0;
    double vega  = 0.0;
    double theta = 0.0;
    double rho   = 0.0;
};

double norm_cdf(double x);
double norm_pdf(double x);

double bs_price(double S, double K, double r, double q,
                double v, double T, int call_put);

Greeks bs_greeks(double S, double K, double r, double q,
                 double v, double T, int call_put);

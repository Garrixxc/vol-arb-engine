import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, minimize
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any

@dataclass
class Greeks:
    delta: float = 0.0
    gamma: float = 0.0
    vega:  float = 0.0
    theta: float = 0.0
    rho:   float = 0.0

@dataclass
class SVIParams:
    a: float = 0.0
    b: float = 0.0
    rho: float = 0.0
    m: float = 0.0
    sigma: float = 0.0

@dataclasses.dataclass
class SVIResult:
    params: SVIParams
    rmse: float
    iterations: int
    converged: bool

def bs_price(S, K, r, q, v, T, call_put):
    """Black-Scholes Price (Vectorized)"""
    # Force float for scalar inputs if needed, though NumPy handles it.
    if T <= 0:
        return np.maximum(call_put * (S - K), 0.0)
    
    if v <= 0:
        return np.maximum(call_put * (S - K), 0.0)

    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * v**2) * T) / (v * sqrtT)
    d2 = d1 - v * sqrtT
    
    return call_put * (S * np.exp(-q * T) * norm.cdf(call_put * d1) - 
                      K * np.exp(-r * T) * norm.cdf(call_put * d2))

def bs_greeks(S, K, r, q, v, T, call_put):
    """Black-Scholes Greeks (Scalar)"""
    if T <= 0 or v <= 0:
        return Greeks()
    
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * v**2) * T) / (v * sqrtT)
    d2 = d1 - v * sqrtT
    
    n_d1 = norm.pdf(d1)
    exp_qT = np.exp(-q * T)
    exp_rT = np.exp(-r * T)
    
    g = Greeks()
    g.delta = call_put * exp_qT * norm.cdf(call_put * d1)
    g.gamma = exp_qT * n_d1 / (S * v * sqrtT)
    g.vega  = S * exp_qT * n_d1 * sqrtT / 100.0  # per 1% vol
    
    # Raw vega for theta calculation (per 1 unit vol)
    vega_raw = S * exp_qT * n_d1 * sqrtT
    
    theta_term1 = - (S * exp_qT * n_d1 * v) / (2 * sqrtT)
    theta_term2 = - call_put * r * K * exp_rT * norm.cdf(call_put * d2)
    theta_term3 = call_put * q * S * exp_qT * norm.cdf(call_put * d1)
    
    g.theta = (theta_term1 + theta_term2 + theta_term3) / 365.0 # per day
    g.rho   = call_put * K * T * exp_rT * norm.cdf(call_put * d2) / 100.0 # per 1% rate
    
    return g

def bs_vega(S, K, r, q, v, T):
    """Black-Scholes Vega (Vectorized)"""
    if T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * v**2) * T) / (v * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

def implied_vol(S, K, r, q, market_price, T, call_put):
    """Find implied volatility using Brent's method (Robust)"""
    if T <= 0 or market_price <= 0:
        return np.nan
        
    intrinsic = max(call_put * (S * np.exp(-q * T) - K * np.exp(-r * T)), 0.0)
    if market_price < intrinsic:
        return np.nan

    def objective(v):
        return bs_price(S, K, r, q, v, T, call_put) - market_price

    try:
        # Search range [1e-6, 5.0] (up to 500% vol)
        return brentq(objective, 1e-6, 5.0, xtol=1e-8)
    except ValueError:
        return np.nan

def implied_vol_vec(S, K_vec, r, q, price_vec, T, call_put_vec):
    """Vectorized version of implied_vol. T can be a scalar or an array."""
    vols = []
    # If T is a scalar, repeat it for the length of K_vec
    if np.isscalar(T):
        T_vec = np.full(len(K_vec), T)
    else:
        T_vec = T
        
    for k, p, t, cp in zip(K_vec, price_vec, T_vec, call_put_vec):
        vols.append(implied_vol(S, k, r, q, p, t, cp))
    return np.array(vols)

def svi_w(k, p: SVIParams):
    """SVI Total Variance: w(k) = a + b * [rho*(k-m) + sqrt((k-m)^2 + sigma^2)]"""
    dk = k - p.m
    disc = np.sqrt(dk**2 + p.sigma**2)
    return p.a + p.b * (p.rho * dk + disc)

def svi_vol(k, T, p: SVIParams):
    """SVI Implied Vol (Annualized)"""
    w = svi_w(k, p)
    return np.sqrt(np.maximum(w, 1e-9) / T)

def svi_vol_vec(k_vec, T, p: SVIParams):
    """Vectorized SVI Implied Vol"""
    return np.array([svi_vol(k, T, p) for k in k_vec])

def calibrate_svi(log_moneyness, market_ivs, T, init: SVIParams) -> SVIResult:
    """Calibrate SVI to a single expiry using L-BFGS-B"""
    w_mkt = np.array(market_ivs)**2 * T
    ks = np.array(log_moneyness)
    
    def objective(x):
        # x = [a, b, rho, m, sigma]
        p = SVIParams(a=x[0], b=x[1], rho=x[2], m=x[3], sigma=x[4])
        w_model = svi_w(ks, p)
        return np.sum((w_model - w_mkt)**2)

    # Bonds for params [a, b, rho, m, sigma]
    # a: can be slightly negative if b is large, but usually > 0
    # b: must be > 0
    # rho: must be in (-1, 1)
    # sigma: must be > 0
    bounds = [
        (-1.0, 1.0),    # a
        (1e-6, 2.0),    # b
        (-0.99, 0.99),  # rho
        (-1.0, 1.0),    # m
        (1e-6, 1.0)     # sigma
    ]
    
    x0 = [init.a, init.b, init.rho, init.m, init.sigma]
    
    res = minimize(objective, x0, bounds=bounds, method='L-BFGS-B', tol=1e-8)
    
    p_final = SVIParams(a=res.x[0], b=res.x[1], rho=res.x[2], m=res.x[3], sigma=res.x[4])
    rmse = np.sqrt(res.fun / len(ks))
    
    return SVIResult(
        params=p_final,
        rmse=rmse,
        iterations=res.nit,
        converged=res.success
    )

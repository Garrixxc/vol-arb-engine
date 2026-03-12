# Vol Arb Engine — Layer 1: BS Pricer + IV Solver + Data Pipeline

## What's built
- **C++ Black-Scholes pricer** with full Greeks (delta, gamma, vega, theta, rho)
- **C++ IV solver** — Newton-Raphson with Brent fallback, converges to 1e-8
- **pybind11 bindings** — vectorized numpy interface, process full chains in one call
- **yfinance fetcher** — live SPX/SPY options chain with spread filtering
- **DuckDB storage** — persist and query chains + IV surfaces

## Setup (on your machine)

```bash
# Install Python deps
pip install pybind11 yfinance duckdb pandas numpy scipy

# Install cmake (macOS: brew install cmake, Ubuntu: apt install cmake)

# Build C++ extension
cd core/cpp
mkdir build && cd build
cmake .. -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
make -j$(nproc)
cp vol_core*.so ../../    # copy .so to core/
```

## Quick start

```python
import sys
sys.path.insert(0, 'core')
import vol_core

# Single option
price = vol_core.bs_price(S=580, K=580, r=0.053, q=0.013, v=0.20, T=0.25, call_put=1)
iv    = vol_core.implied_vol(S=580, K=580, r=0.053, q=0.013,
                              market_price=price, T=0.25, call_put=1)

# Full chain (vectorized)
from data.fetchers.yfinance_fetcher import fetch_options_chain
from core.iv_surface import compute_iv_surface

chain, spot = fetch_options_chain("SPY", min_dte=7, max_dte=120)
iv_surface  = compute_iv_surface(chain, r=0.053, q=0.013)
```

## What's next — Layer 2
- SVI parameterization: fit `w(k) = a + b[ρ(k-m) + √((k-m)²+σ²)]` to each expiry
- Constrained optimization enforcing butterfly + calendar no-arb conditions
- Mispricing signal: `market_iv - SVI_iv` normalized by bid-ask spread

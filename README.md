# 🌌 Aether Vol-Arb Engine

A high-performance, real-time volatility arbitrage engine designed for speed and precision. Aether identifies "mispriced" options by comparing live market data against a sophisticated mathematical model in real-time.

---

## 🧐 What does it do? (In simple terms)

In the options market, "Volatility" is a measure of how much a stock's price is expected to swing. Every option has an "Implied Volatility" (IV) based on its market price.

**Aether's Job:**
1.  **Build a Map**: It looks at hundreds of options at once and builds a 3D "Volatility Surface" (the "map" of what volatility *should* look like).
2.  **Spot the Gaps**: It uses a mathematical model (SVI) to find the "true" shape of that map.
3.  **Find the Deals**: If a specific option's price is significantly different from the model (too cheap or too expensive), Aether flags it as a **Signal**. 

**Arbitrage** means buying the cheap one and selling the expensive one to capture the difference.

---

## 🛠 How it's Implemented

Aether is built using a **Pure Python** architecture for maximum portability and simplicity.

### 1. The Muscle (NumPy & SciPy)
The heavy math — calculating option prices, Greeks (Delta, Gamma, Vega, Theta), and solving for "Implied Volatility" — is implemented using highly optimized **NumPy** and **SciPy** routines. This ensures high performance without the complexity of C++ dependencies.

### 2. The Brain (SVI Calibration)
We use a model called **SVI (Stochastic Volatility Inspired)**. 
- It "fits" a smooth curve to the messy market data using a robust L-BFGS-B optimizer.
- It enforces "no-arbitrage" conditions (Butterfly and Calendar spread), ensuring the model is mathematically consistent.

### 3. The Eyes (Dash Dashboard)
The UI is built with **Dash by Plotly**. 
- **Glassmorphism Design**: A premium, translucent dark-mode theme.
- **Real-Time Updates**: Fetches live data from `yfinance` every minute with automatic synthetic fallback.
- **Interactive 3D**: Visualize the volatility surface in 3D to spot mispriced options instantly.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- Requirements: `pip install numpy scipy pandas dash plotly yfinance duckdb`

### 2. Run the Dashboard
```bash
./run.sh
```
Open **`http://localhost:8050`** in your browser.

---

## 🧪 Technical Strategy
- **Layer 1**: BS Pricer + IV Solver (C++ Vectorized)
- **Layer 2**: SVI Surface Calibration (Levenberg-Marquardt)
- **Layer 3**: SNR-based Relative Value Signals

---
*Created by Antigravity for Gaurav Salvi.*

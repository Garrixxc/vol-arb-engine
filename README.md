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

Aether is built using a "Hybrid" architecture to ensure it's both fast and beautiful.

### 1. The Muscle (C++)
The heavy math — calculating option prices and solving for "Implied Volatility" — is written in **C++**. This allows Aether to process thousands of data points in milliseconds, which is critical for high-frequency trading.

### 2. The Brain (SVI & Python)
We use a model called **SVI (Stochastic Volatility Inspired)**. 
- It "fits" a smooth curve to the messy market data.
- It enforces "no-arbitrage" conditions, ensuring the model itself is mathematically sound.

### 3. The Eyes (Dash Dashboard)
The UI is built with **Python Dash**. 
- **Glassmorphism Design**: A premium, translucent dark-mode theme.
- **Real-Time Updates**: Fetches live data from `yfinance` every minute.
- **Interactive 3D**: Let's you rotate the volatility surface to see exactly where the market is mispriced.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- `cmake` (for building the math engine)

### 2. Build the Math Engine (One-time)
```bash
cd core/cpp
mkdir build && cd build
cmake ..
make -j4
cp vol_core*.so ../../
```

### 3. Run the Dashboard
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

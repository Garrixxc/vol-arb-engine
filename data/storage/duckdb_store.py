"""
data/storage/duckdb_store.py

Persists options chains and computed IV surfaces to DuckDB.
DuckDB is columnar and fast — querying 1M rows is instant.
Everything is stored locally as a single .duckdb file.

Schema:
  options_chain  — raw quotes with timestamps
  iv_surface     — computed implied vols per snapshot
  arb_signals    — detected mispricings (populated later)
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


DB_PATH = Path(__file__).parent.parent.parent / "data" / "vol_arb.duckdb"


class VolDataStore:
    def __init__(self, db_path: str = None):
        self.db_path = str(db_path or DB_PATH)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(self.db_path)
        self._init_schema()
        print(f"[Store] Connected to {self.db_path}")

    def _init_schema(self):
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS options_chain (
                snapshot_ts   TIMESTAMP,
                ticker        VARCHAR,
                expiry        VARCHAR,
                dte           INTEGER,
                T             DOUBLE,
                strike        DOUBLE,
                option_type   VARCHAR,
                bid           DOUBLE,
                ask           DOUBLE,
                mid           DOUBLE,
                spread_pct    DOUBLE,
                open_interest INTEGER,
                volume        INTEGER,
                spot          DOUBLE,
                moneyness     DOUBLE
            )
        """)

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS iv_surface (
                snapshot_ts   TIMESTAMP,
                ticker        VARCHAR,
                expiry        VARCHAR,
                dte           INTEGER,
                T             DOUBLE,
                strike        DOUBLE,
                log_moneyness DOUBLE,
                option_type   VARCHAR,
                market_iv     DOUBLE,
                mid_price     DOUBLE,
                spot          DOUBLE
            )
        """)

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS arb_signals (
                detected_ts   TIMESTAMP,
                ticker        VARCHAR,
                expiry        VARCHAR,
                strike        DOUBLE,
                option_type   VARCHAR,
                market_iv     DOUBLE,
                model_iv      DOUBLE,
                mispricing    DOUBLE,
                spread        DOUBLE,
                signal_snr    DOUBLE
            )
        """)

    def save_chain(self, chain: pd.DataFrame, ticker: str):
        """Persist a raw options chain snapshot."""
        ts = datetime.utcnow()
        df = chain.copy()
        df["snapshot_ts"] = ts

        col_map = {"openInterest": "open_interest"}
        df = df.rename(columns=col_map)

        # Ensure all required columns exist
        required = ["snapshot_ts", "ticker", "expiry", "dte", "T", "strike",
                    "option_type", "bid", "ask", "mid", "spread_pct",
                    "open_interest", "volume", "spot", "moneyness"]
        for col in required:
            if col not in df.columns:
                df[col] = None

        insert_cols = ["snapshot_ts", "ticker", "expiry", "dte", "T", "strike",
                       "option_type", "bid", "ask", "mid", "spread_pct",
                       "open_interest", "volume", "spot", "moneyness"]
        df_insert = df[[c for c in insert_cols if c in df.columns]]
        self.con.execute("INSERT INTO options_chain SELECT * FROM df_insert")
        print(f"[Store] Saved {len(df)} contracts for {ticker} @ {ts.strftime('%H:%M:%S')}")

    def save_iv_surface(self, iv_df: pd.DataFrame):
        """Persist a computed IV surface snapshot."""
        ts = datetime.utcnow()
        iv_df = iv_df.copy()
        iv_df["snapshot_ts"] = ts
        iv_cols = ["snapshot_ts","ticker","expiry","dte","T","strike",
                   "log_moneyness","option_type","market_iv","mid_price","spot"]
        iv_insert = iv_df[[c for c in iv_cols if c in iv_df.columns]]
        self.con.execute("INSERT INTO iv_surface SELECT * FROM iv_insert")
        print(f"[Store] Saved IV surface: {len(iv_df)} points")

    def save_arb_signals(self, signals: pd.DataFrame):
        """Persist detected arb signals."""
        ts = datetime.utcnow()
        signals = signals.copy()
        signals["detected_ts"] = ts
        self.con.execute("INSERT INTO arb_signals SELECT * FROM signals")
        print(f"[Store] Saved {len(signals)} arb signals")

    def get_latest_chain(self, ticker: str) -> pd.DataFrame:
        """Retrieve the most recent options chain snapshot."""
        return self.con.execute("""
            SELECT * FROM options_chain
            WHERE ticker = ? 
              AND snapshot_ts = (
                SELECT MAX(snapshot_ts) FROM options_chain WHERE ticker = ?
              )
        """, [ticker, ticker]).df()

    def get_iv_history(self, ticker: str, expiry: str) -> pd.DataFrame:
        """Time series of IV surface for a given expiry — useful for backtesting."""
        return self.con.execute("""
            SELECT * FROM iv_surface
            WHERE ticker = ? AND expiry = ?
            ORDER BY snapshot_ts, strike
        """, [ticker, expiry]).df()

    def get_arb_signals(self, min_snr: float = 1.5) -> pd.DataFrame:
        """Retrieve arb signals above a minimum signal-to-noise threshold."""
        return self.con.execute("""
            SELECT * FROM arb_signals
            WHERE signal_snr >= ?
            ORDER BY detected_ts DESC, signal_snr DESC
        """, [min_snr]).df()

    def stats(self):
        """Print database summary."""
        for table in ["options_chain", "iv_surface", "arb_signals"]:
            count = self.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table:<20}: {count:>8,} rows")

    def close(self):
        self.con.close()

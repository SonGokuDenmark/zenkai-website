#!/usr/bin/env python3
"""
Continuous processor for HMM regime detection and strategy signals.
Runs as a daemon, periodically checking for and processing new candles.

Usage:
    python processor_continuous.py                # Run continuously
    python processor_continuous.py --interval 60  # Check every 60 seconds
"""

import argparse
import os
import sys
import time
import signal
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Load .env file if it exists
from dotenv import load_dotenv
load_dotenv()

import requests
import psycopg2
from psycopg2.extras import execute_batch
import pandas as pd
import numpy as np

# Import HMM detector and strategies
from hmm_detector import HMMRegimeDetector
from strategies import (
    MACDStrategy,
    MACrossoverStrategy,
    StochasticMRStrategy,
    TurtleStrategy,
    SupportResistanceStrategy,
    CandlestickStrategy,
    RSIDivergenceStrategy,
)


# Database configuration
DB_CONFIG = {
    "host": os.getenv("ZENKAI_DB_HOST", "localhost"),
    "database": os.getenv("ZENKAI_DB_NAME", "zenkai_data"),
    "user": os.getenv("ZENKAI_DB_USER", "zenkai"),
    "password": os.getenv("ZENKAI_DB_PASSWORD"),
}

MODEL_DIR = Path("models")

# Home Assistant webhook URL for light pulse
HA_WEBHOOK_URL = os.getenv("HA_WEBHOOK_URL", "http://192.168.0.160:8123/api/webhook/zenkai_pulse")

# Rate limiting for light pulse
_last_pulse_time = 0
PULSE_RATE_LIMIT = 30  # seconds between pulses

# Global flag for graceful shutdown
running = True


def pulse_lights():
    """Pulse lights via Home Assistant to indicate batch completion."""
    global _last_pulse_time
    now = time.time()
    if now - _last_pulse_time < PULSE_RATE_LIMIT:
        return  # Skip if we pulsed recently
    _last_pulse_time = now
    try:
        requests.post(HA_WEBHOOK_URL, timeout=1)
    except Exception:
        pass  # Don't break processing if HA is down


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global running
    print(f"\n[{datetime.now()}] Received shutdown signal, finishing current batch...")
    running = False


def get_db_connection():
    """Get PostgreSQL connection."""
    return psycopg2.connect(**DB_CONFIG)


def compute_hmm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features needed for HMM from raw OHLCV data."""
    df = df.copy()

    if "returns_1" not in df.columns or df["returns_1"].isna().all():
        df["returns_1"] = df["close"].pct_change()

    if "volatility" not in df.columns or df["volatility"].isna().all():
        df["volatility"] = df["close"].pct_change().rolling(20).std()

    if "volume_ratio" not in df.columns or df["volume_ratio"].isna().all():
        vol_sma = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / vol_sma.replace(0, np.nan)

    df["returns_1"] = df["returns_1"].fillna(0.0)
    df["volatility"] = df["volatility"].fillna(
        df["volatility"].mean() if df["volatility"].notna().any() else 0.01
    )
    df["volume_ratio"] = df["volume_ratio"].fillna(1.0)

    return df


def _to_python_type(val):
    """Convert numpy types to Python native types for PostgreSQL."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def load_hmm_models() -> Dict[str, HMMRegimeDetector]:
    """Load all pre-trained HMM models."""
    models = {}
    for model_file in MODEL_DIR.glob("hmm_regime_*.pkl"):
        timeframe = model_file.stem.replace("hmm_regime_", "")
        try:
            models[timeframe] = HMMRegimeDetector.load(str(model_file))
            print(f"  Loaded model for {timeframe}")
        except Exception as e:
            print(f"  Warning: Failed to load {model_file}: {e}")
    return models


def get_all_strategies() -> List:
    """Get all strategy instances."""
    return [
        MACDStrategy(),
        MACrossoverStrategy(),
        StochasticMRStrategy(),
        TurtleStrategy(),
        SupportResistanceStrategy(),
        CandlestickStrategy(),
        RSIDivergenceStrategy(),
    ]


def get_unprocessed_count() -> Dict[str, int]:
    """Get count of unprocessed candles per timeframe."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT timeframe, COUNT(*)
        FROM ohlcv
        WHERE regime IS NULL
        GROUP BY timeframe
    """)

    counts = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return counts


def process_batch(
    df: pd.DataFrame,
    detector: HMMRegimeDetector,
    strategies: List,
    symbol: str,
) -> pd.DataFrame:
    """Process a batch of candles with HMM regime and strategy signals."""
    df = compute_hmm_features(df)

    try:
        df["regime"] = detector.predict(df)
    except Exception as e:
        print(f"    Warning: HMM prediction failed: {e}")
        df["regime"] = None

    for strategy in strategies:
        try:
            df = strategy.compute_indicators(df)
            signals = strategy.generate_signals(df, symbol)

            signal_col = strategy.get_signal_column()
            conf_col = strategy.get_confidence_column()

            df[signal_col] = 0
            df[conf_col] = None

            signal_map = {s.timestamp: s for s in signals}

            for idx, row in df.iterrows():
                ts = row["timestamp"]
                if ts in signal_map:
                    sig = signal_map[ts]
                    df.at[idx, signal_col] = int(sig.direction)
                    df.at[idx, conf_col] = sig.confidence

        except Exception as e:
            df[strategy.get_signal_column()] = 0
            df[strategy.get_confidence_column()] = None

    return df


def update_database(df: pd.DataFrame, conn) -> int:
    """Update database with processed data."""
    cursor = conn.cursor()

    updates = []
    for _, row in df.iterrows():
        updates.append((
            row.get("regime"),
            int(row.get("signal_macd", 0) or 0),
            int(row.get("signal_rsi_div", 0) or 0),
            int(row.get("signal_turtle", 0) or 0),
            int(row.get("signal_ma_cross", 0) or 0),
            int(row.get("signal_stoch_mr", 0) or 0),
            int(row.get("signal_sr_bounce", 0) or 0),
            int(row.get("signal_candlestick", 0) or 0),
            _to_python_type(row.get("conf_macd")),
            _to_python_type(row.get("conf_rsi_div")),
            _to_python_type(row.get("conf_turtle")),
            _to_python_type(row.get("conf_ma_cross")),
            _to_python_type(row.get("conf_stoch_mr")),
            _to_python_type(row.get("conf_sr_bounce")),
            _to_python_type(row.get("conf_candlestick")),
            datetime.now(),
            row["exchange"],
            row["symbol"],
            row["timeframe"],
            int(row["open_time"]),
        ))

    query = """
        UPDATE ohlcv SET
            regime = %s,
            signal_macd = %s,
            signal_rsi_div = %s,
            signal_turtle = %s,
            signal_ma_cross = %s,
            signal_stoch = %s,
            signal_sr = %s,
            signal_candle = %s,
            conf_macd = %s,
            conf_rsi_div = %s,
            conf_turtle = %s,
            conf_ma_cross = %s,
            conf_stoch = %s,
            conf_sr = %s,
            conf_candle = %s,
            processed_at = %s
        WHERE exchange = %s AND symbol = %s AND timeframe = %s AND open_time = %s
    """

    execute_batch(cursor, query, updates, page_size=1000)
    conn.commit()

    return len(updates)


def process_timeframe(
    timeframe: str,
    detector: HMMRegimeDetector,
    strategies: List,
    batch_size: int = 5000,
    max_per_cycle: int = 50000,
) -> int:
    """Process unprocessed candles for a timeframe."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get symbols with unprocessed data for this timeframe
    cursor.execute("""
        SELECT DISTINCT symbol
        FROM ohlcv
        WHERE timeframe = %s AND regime IS NULL
        ORDER BY symbol
    """, (timeframe,))
    symbols = [row[0] for row in cursor.fetchall()]

    if not symbols:
        conn.close()
        return 0

    total_updated = 0

    for symbol in symbols:
        if not running or total_updated >= max_per_cycle:
            break

        # Fetch batch
        query = """
            SELECT exchange, symbol, timeframe, open_time,
                   open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = %s AND timeframe = %s AND regime IS NULL
            ORDER BY open_time
            LIMIT %s
        """
        batch_df = pd.read_sql_query(
            query, conn,
            params=(symbol, timeframe, batch_size)
        )

        if len(batch_df) == 0:
            continue

        batch_df["timestamp"] = pd.to_datetime(batch_df["open_time"], unit="ms")

        # Process
        batch_df = process_batch(batch_df, detector, strategies, symbol)

        # Update
        n_updated = update_database(batch_df, conn)
        total_updated += n_updated

        # Pulse lights to indicate batch completion
        pulse_lights()

    conn.close()
    return total_updated


def run_continuous(interval: int = 30):
    """Run continuous processing loop."""
    global running

    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    print("=" * 60)
    print("AlphaTrader Continuous Processor")
    print("=" * 60)
    print(f"Started: {datetime.now()}")
    print(f"Check interval: {interval} seconds")

    # Load models
    print("\nLoading HMM models...")
    models = load_hmm_models()

    if not models:
        print("ERROR: No HMM models found! Run backfill first.")
        return

    print(f"Loaded {len(models)} models: {', '.join(models.keys())}")

    # Load strategies
    strategies = get_all_strategies()
    print(f"Strategies: {', '.join(s.name for s in strategies)}")

    print("\nStarting continuous processing loop...")
    print("-" * 60)

    cycle = 0
    while running:
        cycle += 1
        cycle_start = datetime.now()

        # Check for unprocessed
        unprocessed = get_unprocessed_count()
        total_unprocessed = sum(unprocessed.values())

        if total_unprocessed > 0:
            print(f"\n[{cycle_start}] Cycle {cycle}: {total_unprocessed:,} unprocessed candles")

            cycle_updated = 0
            for timeframe, count in sorted(unprocessed.items()):
                if not running:
                    break

                if timeframe not in models:
                    continue

                if count > 0:
                    updated = process_timeframe(
                        timeframe,
                        models[timeframe],
                        strategies,
                        batch_size=5000,
                        max_per_cycle=50000,
                    )
                    if updated > 0:
                        print(f"  {timeframe}: processed {updated:,} candles")
                        cycle_updated += updated

            elapsed = (datetime.now() - cycle_start).total_seconds()
            print(f"  Cycle {cycle} complete: {cycle_updated:,} updated in {elapsed:.1f}s")
        else:
            # No unprocessed, just log occasionally
            if cycle % 10 == 0:
                print(f"[{cycle_start}] Cycle {cycle}: No unprocessed candles, waiting...")

        # Wait for next cycle
        if running:
            time.sleep(interval)

    print(f"\n[{datetime.now()}] Processor stopped gracefully")


def main():
    parser = argparse.ArgumentParser(
        description="Continuous processor for regime and signals"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Seconds between processing cycles (default: 30)"
    )

    args = parser.parse_args()
    run_continuous(interval=args.interval)


if __name__ == "__main__":
    main()

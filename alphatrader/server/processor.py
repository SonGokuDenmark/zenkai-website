#!/usr/bin/env python3
"""
Batch processor for HMM regime detection and strategy signals.
Processes unprocessed candles in PostgreSQL and updates with regime + signals.

Usage:
    python processor.py                    # Process all unprocessed
    python processor.py --symbol BTCUSDT   # Process single symbol
    python processor.py --timeframe 15m    # Process single timeframe
    python processor.py --train            # Train HMM models first
    python processor.py --backfill         # Full backfill mode
"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import pickle

# Load .env file if it exists
from dotenv import load_dotenv
load_dotenv()

import requests
import psycopg2
from psycopg2.extras import execute_batch
import pandas as pd
import numpy as np
from tqdm import tqdm

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


# Database configuration - load from environment or use defaults
DB_CONFIG = {
    "host": os.getenv("ZENKAI_DB_HOST", "localhost"),
    "database": os.getenv("ZENKAI_DB_NAME", "zenkai_data"),
    "user": os.getenv("ZENKAI_DB_USER", "zenkai"),
    "password": os.getenv("ZENKAI_DB_PASSWORD"),
}

# Model directory
MODEL_DIR = Path("models")

# Home Assistant webhook URL for light pulse
HA_WEBHOOK_URL = os.getenv("HA_WEBHOOK_URL", "http://192.168.0.160:8123/api/webhook/zenkai_pulse")

# Rate limiting for light pulse
_last_pulse_time = 0
PULSE_RATE_LIMIT = 30  # seconds between pulses


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


def get_db_connection():
    """Get PostgreSQL connection."""
    return psycopg2.connect(**DB_CONFIG)


def get_db_stats() -> Dict:
    """Get database statistics."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Total rows
    cursor.execute("SELECT COUNT(*) FROM ohlcv")
    total = cursor.fetchone()[0]

    # Unprocessed rows
    cursor.execute("SELECT COUNT(*) FROM ohlcv WHERE regime IS NULL")
    unprocessed = cursor.fetchone()[0]

    # Symbols
    cursor.execute("SELECT COUNT(DISTINCT symbol) FROM ohlcv")
    n_symbols = cursor.fetchone()[0]

    # Timeframes
    cursor.execute("SELECT DISTINCT timeframe FROM ohlcv ORDER BY timeframe")
    timeframes = [row[0] for row in cursor.fetchall()]

    conn.close()

    return {
        "total": total,
        "unprocessed": unprocessed,
        "processed": total - unprocessed,
        "n_symbols": n_symbols,
        "timeframes": timeframes,
    }


def compute_hmm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features needed for HMM from raw OHLCV data.
    """
    df = df.copy()

    # Returns (percentage change)
    if "returns_1" not in df.columns or df["returns_1"].isna().all():
        df["returns_1"] = df["close"].pct_change()

    # Volatility (20-bar rolling std of returns)
    if "volatility" not in df.columns or df["volatility"].isna().all():
        df["volatility"] = df["close"].pct_change().rolling(20).std()

    # Volume ratio (current / 20-bar SMA)
    if "volume_ratio" not in df.columns or df["volume_ratio"].isna().all():
        vol_sma = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / vol_sma.replace(0, np.nan)

    # Fill NaN with neutral values
    df["returns_1"] = df["returns_1"].fillna(0.0)
    df["volatility"] = df["volatility"].fillna(
        df["volatility"].mean() if df["volatility"].notna().any() else 0.01
    )
    df["volume_ratio"] = df["volume_ratio"].fillna(1.0)

    return df


def load_training_sample(timeframe: str, max_samples: int = 500000) -> pd.DataFrame:
    """Load a sample of data for HMM training."""
    conn = get_db_connection()

    # Count for this timeframe
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM ohlcv WHERE timeframe = %s",
        (timeframe,)
    )
    count = cursor.fetchone()[0]

    if count == 0:
        conn.close()
        return pd.DataFrame()

    # Fetch sample - use composite key (exchange, symbol, timeframe, open_time)
    if count > max_samples:
        query = """
            SELECT exchange, symbol, timeframe, open_time,
                   open, high, low, close, volume
            FROM ohlcv
            WHERE timeframe = %s
            ORDER BY RANDOM()
            LIMIT %s
        """
        df = pd.read_sql_query(query, conn, params=(timeframe, max_samples))
    else:
        query = """
            SELECT exchange, symbol, timeframe, open_time,
                   open, high, low, close, volume
            FROM ohlcv
            WHERE timeframe = %s
        """
        df = pd.read_sql_query(query, conn, params=(timeframe,))

    # Convert open_time to datetime for timestamp column
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")

    conn.close()

    # Compute features
    df = compute_hmm_features(df)

    return df


def train_hmm_models(timeframes: List[str] = None) -> Dict[str, HMMRegimeDetector]:
    """Train HMM models for each timeframe."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if timeframes is None:
        stats = get_db_stats()
        timeframes = stats["timeframes"]

    models = {}

    for tf in timeframes:
        print(f"\n{'='*50}")
        print(f"Training HMM for timeframe: {tf}")
        print(f"{'='*50}")

        model_path = MODEL_DIR / f"hmm_regime_{tf}.pkl"

        # Load training data
        print("Loading training sample...")
        train_df = load_training_sample(tf, max_samples=500000)

        if len(train_df) == 0:
            print(f"  No data for {tf}, skipping...")
            continue

        print(f"  Training samples: {len(train_df):,}")

        # Train model
        detector = HMMRegimeDetector(
            n_states=4,
            features=["returns_1", "volatility", "volume_ratio"],
        )

        detector.fit_cross_symbol(train_df, verbose=True)

        # Save model
        detector.save(str(model_path))
        print(f"  Saved: {model_path}")

        models[tf] = detector

    return models


def load_hmm_model(timeframe: str) -> Optional[HMMRegimeDetector]:
    """Load pre-trained HMM model for a timeframe."""
    model_path = MODEL_DIR / f"hmm_regime_{timeframe}.pkl"

    if not model_path.exists():
        return None

    return HMMRegimeDetector.load(str(model_path))


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


def process_batch(
    df: pd.DataFrame,
    detector: HMMRegimeDetector,
    strategies: List,
    symbol: str,
) -> pd.DataFrame:
    """
    Process a batch of candles with HMM regime and strategy signals.

    Returns DataFrame with regime and signal columns.
    """
    # Compute HMM features
    df = compute_hmm_features(df)

    # Predict regimes
    try:
        df["regime"] = detector.predict(df)
    except Exception as e:
        print(f"  Warning: HMM prediction failed: {e}")
        df["regime"] = None

    # Generate strategy signals
    for strategy in strategies:
        try:
            # Compute indicators
            df = strategy.compute_indicators(df)

            # Generate signals
            signals = strategy.generate_signals(df, symbol)

            # Map signals to rows
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
            print(f"  Warning: {strategy.name} failed: {e}")
            df[strategy.get_signal_column()] = 0
            df[strategy.get_confidence_column()] = None

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


def update_database(df: pd.DataFrame, conn) -> int:
    """Update database with processed data using composite key."""
    cursor = conn.cursor()

    # Build update query using composite key (exchange, symbol, timeframe, open_time)
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
            int(row["open_time"]),  # Ensure open_time is Python int
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


def process_symbol_timeframe(
    symbol: str,
    timeframe: str,
    detector: HMMRegimeDetector,
    strategies: List,
    batch_size: int = 10000,
    limit: int = None,
) -> int:
    """Process all unprocessed candles for a symbol/timeframe."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Count unprocessed
    cursor.execute(
        """
        SELECT COUNT(*) FROM ohlcv
        WHERE symbol = %s AND timeframe = %s AND regime IS NULL
        """,
        (symbol, timeframe)
    )
    total = cursor.fetchone()[0]

    if total == 0:
        conn.close()
        return 0

    if limit:
        total = min(total, limit)

    updated = 0
    pbar = tqdm(total=total, desc=f"  {symbol}/{timeframe}", unit="rows")

    offset = 0
    while offset < total:
        # Fetch batch - use composite key columns
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

        # Convert open_time to datetime for timestamp column
        if len(batch_df) > 0:
            batch_df["timestamp"] = pd.to_datetime(batch_df["open_time"], unit="ms")

        if len(batch_df) == 0:
            break

        # Process batch
        batch_df = process_batch(batch_df, detector, strategies, symbol)

        # Update database
        n_updated = update_database(batch_df, conn)
        updated += n_updated
        pbar.update(n_updated)

        # Pulse lights to indicate batch completion
        pulse_lights()

        offset += batch_size

        if limit and updated >= limit:
            break

    pbar.close()
    conn.close()

    return updated


def run_processor(
    symbol: str = None,
    timeframe: str = None,
    train: bool = False,
    batch_size: int = 10000,
    limit: int = None,
):
    """
    Main processor entry point.

    Args:
        symbol: Process only this symbol
        timeframe: Process only this timeframe
        train: Train HMM models before processing
        batch_size: Rows per batch
        limit: Maximum rows to process (for testing)
    """
    print("=" * 60)
    print("AlphaTrader Server Processor")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Get database stats
    stats = get_db_stats()
    print(f"\nDatabase stats:")
    print(f"  Total candles: {stats['total']:,}")
    print(f"  Processed: {stats['processed']:,}")
    print(f"  Unprocessed: {stats['unprocessed']:,}")
    print(f"  Symbols: {stats['n_symbols']}")
    print(f"  Timeframes: {', '.join(stats['timeframes'])}")

    # Determine timeframes to process
    timeframes = [timeframe] if timeframe else stats["timeframes"]

    # Train or load HMM models
    models = {}
    if train:
        print("\n--- Training HMM Models ---")
        models = train_hmm_models(timeframes)
    else:
        print("\n--- Loading HMM Models ---")
        for tf in timeframes:
            model = load_hmm_model(tf)
            if model:
                models[tf] = model
                print(f"  Loaded: hmm_regime_{tf}.pkl")
            else:
                print(f"  No model for {tf} - will train...")
                train_df = load_training_sample(tf, max_samples=500000)
                if len(train_df) > 0:
                    detector = HMMRegimeDetector(
                        n_states=4,
                        features=["returns_1", "volatility", "volume_ratio"],
                    )
                    detector.fit_cross_symbol(train_df, verbose=True)
                    detector.save(str(MODEL_DIR / f"hmm_regime_{tf}.pkl"))
                    models[tf] = detector

    # Get strategies
    strategies = get_all_strategies()
    print(f"\nStrategies: {', '.join(s.name for s in strategies)}")

    # Get symbols to process
    conn = get_db_connection()
    cursor = conn.cursor()

    if symbol:
        symbols = [symbol]
    else:
        cursor.execute("SELECT DISTINCT symbol FROM ohlcv ORDER BY symbol")
        symbols = [row[0] for row in cursor.fetchall()]

    conn.close()

    # Process
    total_updated = 0

    for tf in timeframes:
        if tf not in models:
            print(f"\nSkipping {tf} - no model available")
            continue

        print(f"\n{'='*50}")
        print(f"Processing timeframe: {tf}")
        print(f"{'='*50}")

        detector = models[tf]

        for sym in symbols:
            updated = process_symbol_timeframe(
                sym, tf, detector, strategies,
                batch_size=batch_size,
                limit=limit,
            )
            total_updated += updated

            if limit and total_updated >= limit:
                break

        if limit and total_updated >= limit:
            break

    # Final stats
    print("\n" + "=" * 60)
    print("Processing Complete")
    print("=" * 60)
    print(f"Total rows updated: {total_updated:,}")

    # Show regime distribution
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT regime, COUNT(*) as cnt
        FROM ohlcv
        WHERE regime IS NOT NULL
        GROUP BY regime
        ORDER BY cnt DESC
    """)
    print("\nRegime Distribution:")
    for row in cursor.fetchall():
        print(f"  {row[0] or 'NULL':20s}: {row[1]:>12,}")
    conn.close()

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    parser = argparse.ArgumentParser(
        description="Process candles with HMM regime and strategy signals"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        help="Process only this symbol"
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        help="Process only this timeframe"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train HMM models before processing"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Rows per batch"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum rows to process (for testing)"
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Full backfill mode (process all data)"
    )

    args = parser.parse_args()

    run_processor(
        symbol=args.symbol,
        timeframe=args.timeframe,
        train=args.train or args.backfill,
        batch_size=args.batch_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

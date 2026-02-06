#!/usr/bin/env python3
"""
Distributed HMM + 72 Strategy Processor.

Multiple workers can run simultaneously on different machines.
Database-coordinated work queue ensures NO duplicate processing.

Key: FOR UPDATE SKIP LOCKED
- Workers claim batches atomically
- If another worker has rows locked, they're skipped
- Zero coordination needed between machines

Usage:
    # On Server (Goku)
    python distributed_processor.py --worker-id goku

    # On Your PC
    python distributed_processor.py --worker-id pc1

    # Status (from anywhere)
    python distributed_processor.py --status

Options:
    --worker-id NAME    Unique worker name (required for processing)
    --batch-size N      Rows per batch (default: 5000)
    --max-batches N     Stop after N batches (0 = unlimited)
    --status            Show processing progress
    --daemon            Run continuously
"""

import argparse
import os
import sys
import time
import signal
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from dotenv import load_dotenv
load_dotenv()

import psycopg2
from psycopg2.extras import execute_batch
import pandas as pd
import numpy as np

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.regime import HMMRegimeDetector
from src.strategies import get_all_strategies, count_strategies

# Database config - connects to server
DB_CONFIG = {
    "host": os.getenv("ZENKAI_DB_HOST", "192.168.0.160"),
    "database": os.getenv("ZENKAI_DB_NAME", "zenkai_data"),
    "user": os.getenv("ZENKAI_DB_USER", "zenkai"),
    "password": os.getenv("ZENKAI_DB_PASSWORD"),
}

MODEL_DIR = Path(__file__).parent / "models"

# Global shutdown flag
running = True


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global running
    print(f"\n[{datetime.now()}] Shutdown signal received, finishing current batch...")
    running = False


def get_db_connection():
    """Get PostgreSQL connection."""
    return psycopg2.connect(**DB_CONFIG)


def _to_python_type(val):
    """Convert numpy types to Python native types."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def compute_hmm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features needed for HMM."""
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


def get_or_train_hmm_model(timeframe: str) -> Optional[HMMRegimeDetector]:
    """Load or train HMM model for timeframe."""
    model_path = MODEL_DIR / f"hmm_regime_{timeframe}.pkl"

    if model_path.exists():
        try:
            return HMMRegimeDetector.load(str(model_path))
        except Exception as e:
            print(f"  Warning: Failed to load {timeframe} model: {e}")

    # Train new model
    print(f"  Training HMM model for {timeframe}...")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT open_time, open, high, low, close, volume
        FROM ohlcv
        WHERE timeframe = %s
        ORDER BY RANDOM()
        LIMIT 500000
    """, (timeframe,))

    rows = cursor.fetchall()
    conn.close()

    if len(rows) < 1000:
        print(f"  Not enough data for {timeframe}")
        return None

    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    df = compute_hmm_features(df)

    detector = HMMRegimeDetector(
        n_states=4,
        features=["returns_1", "volatility", "volume_ratio"],
    )
    detector.fit_cross_symbol(df, verbose=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    detector.save(str(model_path))
    print(f"  Saved: {model_path}")

    return detector


def get_processing_status() -> dict:
    """Get current processing status."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM ohlcv")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM ohlcv WHERE processed_at IS NOT NULL")
    processed = cursor.fetchone()[0]

    cursor.execute("""
        SELECT exchange,
               COUNT(*) as total,
               COUNT(processed_at) as processed
        FROM ohlcv
        GROUP BY exchange
    """)
    by_exchange = {row[0]: {"total": row[1], "processed": row[2]} for row in cursor.fetchall()}

    cursor.execute("""
        SELECT timeframe,
               COUNT(*) as total,
               COUNT(processed_at) as processed
        FROM ohlcv
        GROUP BY timeframe
        ORDER BY total DESC
    """)
    by_timeframe = {row[0]: {"total": row[1], "processed": row[2]} for row in cursor.fetchall()}

    conn.close()

    return {
        "total": total,
        "processed": processed,
        "remaining": total - processed,
        "percent": (processed / total * 100) if total > 0 else 0,
        "by_exchange": by_exchange,
        "by_timeframe": by_timeframe,
    }


def print_status():
    """Print processing status."""
    status = get_processing_status()
    strategies = count_strategies()

    print("=" * 70)
    print("Distributed HMM + Strategy Processing Status")
    print("=" * 70)
    print(f"\nStrategies loaded: {strategies['total']}")
    print(f"  General: {strategies['general']}, UP: {strategies['trending_up']}, "
          f"DOWN: {strategies['trending_down']}, Ranging: {strategies['ranging']}")
    print(f"  Volume: {strategies['volume']}, Volatility: {strategies['volatility']}, "
          f"Momentum: {strategies['momentum']}, Pattern: {strategies['pattern']}")

    print(f"\nTotal rows:     {status['total']:>15,}")
    print(f"Processed:      {status['processed']:>15,}")
    print(f"Remaining:      {status['remaining']:>15,}")
    print(f"Complete:       {status['percent']:>14.2f}%")

    print("\n--- By Exchange ---")
    for exchange, data in status["by_exchange"].items():
        pct = (data["processed"] / data["total"] * 100) if data["total"] > 0 else 0
        print(f"  {exchange:12s}: {data['processed']:>12,} / {data['total']:>12,} ({pct:5.1f}%)")

    print("\n--- By Timeframe (top 10) ---")
    for tf, data in list(status["by_timeframe"].items())[:10]:
        pct = (data["processed"] / data["total"] * 100) if data["total"] > 0 else 0
        print(f"  {tf:8s}: {data['processed']:>12,} / {data['total']:>12,} ({pct:5.1f}%)")


def process_batch(
    df: pd.DataFrame,
    detector: HMMRegimeDetector,
    strategies: List,
) -> pd.DataFrame:
    """Process batch with HMM and all strategies."""
    df = compute_hmm_features(df)

    # HMM regime
    try:
        df["regime"] = detector.predict(df)
    except Exception:
        df["regime"] = "UNKNOWN"

    # All strategies
    for strategy in strategies:
        try:
            df = strategy.compute_indicators(df)
            symbol = df["symbol"].iloc[0] if "symbol" in df.columns else "UNKNOWN"
            signals = strategy.generate_signals(df, symbol)

            signal_col = strategy.get_signal_column()
            conf_col = strategy.get_confidence_column()

            df[signal_col] = 0
            df[conf_col] = None

            signal_map = {s.timestamp: s for s in signals}
            for idx, row in df.iterrows():
                ts = row.get("timestamp")
                if ts in signal_map:
                    sig = signal_map[ts]
                    df.at[idx, signal_col] = int(sig.direction)
                    df.at[idx, conf_col] = sig.confidence
        except Exception:
            df[strategy.get_signal_column()] = 0
            df[strategy.get_confidence_column()] = None

    return df


def run_worker(
    worker_id: str,
    batch_size: int = 5000,
    max_batches: int = 0,
    daemon: bool = False,
):
    """Run distributed worker."""
    global running

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    print("=" * 70)
    print(f"Distributed Processor - Worker: {worker_id}")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print(f"Batch size: {batch_size:,}")
    print(f"Database: {DB_CONFIG['host']}")

    # Load strategies
    print("\nLoading 72 strategies...")
    strategies = get_all_strategies()
    print(f"Loaded {len(strategies)} strategies")

    # Get timeframes
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT timeframe FROM ohlcv WHERE processed_at IS NULL")
    timeframes = [row[0] for row in cursor.fetchall()]
    conn.close()

    print(f"Timeframes to process: {', '.join(timeframes[:5])}...")

    # Load HMM models
    print("\nLoading HMM models...")
    models = {}
    for tf in timeframes:
        model = get_or_train_hmm_model(tf)
        if model:
            models[tf] = model

    if not models:
        print("ERROR: No HMM models available!")
        return

    print(f"Models ready: {len(models)}")

    # Main processing loop
    print("\n" + "-" * 70)
    print("Starting distributed processing...")
    print("Press Ctrl+C to gracefully stop")
    print("-" * 70)

    total_processed = 0
    batch_count = 0
    start_time = time.time()

    while running:
        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            # KEY: FOR UPDATE SKIP LOCKED
            # This is what makes distributed processing work!
            # If another worker has rows locked, we skip them
            # Randomly pick an exchange to ensure fair processing
            import random
            exchange = random.choice(['binance', 'us_stock'])
            cursor.execute("""
                SELECT exchange, symbol, timeframe, open_time,
                       open, high, low, close, volume
                FROM ohlcv
                WHERE processed_at IS NULL AND exchange = %s
                LIMIT %s
                FOR UPDATE SKIP LOCKED
            """, (exchange, batch_size,))

            rows = cursor.fetchall()

            if not rows:
                conn.close()
                if daemon:
                    print(f"[{worker_id}] No unprocessed rows, waiting 60s...")
                    time.sleep(60)
                    continue
                else:
                    print(f"\n[{worker_id}] All data processed!")
                    break

            # Convert to DataFrame
            batch_df = pd.DataFrame(rows, columns=[
                "exchange", "symbol", "timeframe", "open_time",
                "open", "high", "low", "close", "volume"
            ])
            batch_df["timestamp"] = pd.to_datetime(batch_df["open_time"], unit="ms")

            # Group by timeframe and process
            updates = []
            for timeframe, group_df in batch_df.groupby("timeframe"):
                if timeframe not in models:
                    continue

                processed_df = process_batch(group_df.copy(), models[timeframe], strategies)

                for _, row in processed_df.iterrows():
                    update = {
                        "regime": row.get("regime"),
                        "exchange": row["exchange"],
                        "symbol": row["symbol"],
                        "timeframe": row["timeframe"],
                        "open_time": int(row["open_time"]),
                    }

                    for strategy in strategies:
                        sig_col = strategy.get_signal_column()
                        conf_col = strategy.get_confidence_column()
                        update[sig_col] = _to_python_type(row.get(sig_col, 0))
                        update[conf_col] = _to_python_type(row.get(conf_col))

                    updates.append(update)

            # Build and execute update
            if updates:
                # Dynamic columns based on strategies
                set_parts = ["regime = %(regime)s", "processed_at = NOW()"]
                for strategy in strategies:
                    sig_col = strategy.get_signal_column()
                    conf_col = strategy.get_confidence_column()
                    set_parts.append(f"{sig_col} = %({sig_col})s")
                    set_parts.append(f"{conf_col} = %({conf_col})s")

                query = f"""
                    UPDATE ohlcv SET {', '.join(set_parts)}
                    WHERE exchange = %(exchange)s
                      AND symbol = %(symbol)s
                      AND timeframe = %(timeframe)s
                      AND open_time = %(open_time)s
                """

                execute_batch(cursor, query, updates, page_size=1000)
                conn.commit()

                total_processed += len(updates)
                batch_count += 1

                elapsed = time.time() - start_time
                rate = total_processed / elapsed if elapsed > 0 else 0
                print(f"[{worker_id}] Batch {batch_count}: {total_processed:,} total ({rate:.0f}/sec)")

        except Exception as e:
            print(f"[{worker_id}] Error: {e}")
            conn.rollback()

        finally:
            conn.close()

        # Check max batches
        if max_batches > 0 and batch_count >= max_batches:
            print(f"\n[{worker_id}] Reached max batches ({max_batches})")
            break

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Worker {worker_id} Complete")
    print("=" * 70)
    print(f"Processed: {total_processed:,} rows")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Rate: {total_processed/elapsed:.0f} rows/sec")


def main():
    parser = argparse.ArgumentParser(
        description="Distributed HMM + 72 Strategy Processor"
    )
    parser.add_argument(
        "--worker-id", "-w",
        type=str,
        help="Unique worker identifier (e.g., 'goku', 'pc1')"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=5000,
        help="Rows per batch (default: 5000)"
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Max batches (0 = unlimited)"
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show processing status"
    )
    parser.add_argument(
        "--daemon", "-d",
        action="store_true",
        help="Run continuously (daemon mode)"
    )

    args = parser.parse_args()

    if args.status:
        print_status()
    elif args.worker_id:
        run_worker(
            worker_id=args.worker_id,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            daemon=args.daemon,
        )
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python distributed_processor.py --status")
        print("  python distributed_processor.py --worker-id pc1")
        print("  python distributed_processor.py --worker-id goku --daemon")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Comprehensive HMM + 34 Strategy Processor with Pause/Resume.

Processes ALL data in the database:
- Applies 4-state HMM regime detection
- Runs 34 strategies (7 general + 10 UP + 10 DOWN + 7 ranging)
- Supports pause/resume for processing billions of rows
- Checkpoints progress to survive restarts

Usage:
    python hmm_processor.py                    # Process all unprocessed data
    python hmm_processor.py --resume           # Resume from last checkpoint
    python hmm_processor.py --status           # Show processing status
    python hmm_processor.py --reset            # Reset checkpoint (start over)
    python hmm_processor.py --daemon           # Run continuously
"""

import argparse
import json
import os
import sys
import time
import signal
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any

from dotenv import load_dotenv
load_dotenv()

import psycopg2
from psycopg2.extras import execute_batch
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.regime import HMMRegimeDetector
from src.strategies import get_all_strategies
from ranging_strategies import get_ranging_strategies

# Database config
DB_CONFIG = {
    "host": os.getenv("ZENKAI_DB_HOST", "localhost"),
    "database": os.getenv("ZENKAI_DB_NAME", "zenkai_data"),
    "user": os.getenv("ZENKAI_DB_USER", "zenkai"),
    "password": os.getenv("ZENKAI_DB_PASSWORD"),
}

# Paths
MODEL_DIR = Path(__file__).parent / "models"
CHECKPOINT_FILE = Path(__file__).parent / ".hmm_processor_checkpoint.json"

# Global shutdown flag
running = True


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global running
    print(f"\n[{datetime.now()}] Received shutdown signal, saving checkpoint...")
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


class Checkpoint:
    """Manages processing checkpoint for pause/resume."""

    def __init__(self, filepath: Path = CHECKPOINT_FILE):
        self.filepath = filepath
        self.data = self._load()

    def _load(self) -> dict:
        """Load checkpoint from disk."""
        if self.filepath.exists():
            with open(self.filepath, "r") as f:
                return json.load(f)
        return {
            "last_processed_id": 0,
            "total_processed": 0,
            "started_at": None,
            "last_update": None,
            "current_exchange": None,
            "current_symbol": None,
            "current_timeframe": None,
        }

    def save(self):
        """Save checkpoint to disk."""
        self.data["last_update"] = datetime.now().isoformat()
        with open(self.filepath, "w") as f:
            json.dump(self.data, f, indent=2)

    def update(self, last_id: int, count: int, exchange: str = None,
               symbol: str = None, timeframe: str = None):
        """Update checkpoint state."""
        self.data["last_processed_id"] = last_id
        self.data["total_processed"] += count
        if exchange:
            self.data["current_exchange"] = exchange
        if symbol:
            self.data["current_symbol"] = symbol
        if timeframe:
            self.data["current_timeframe"] = timeframe
        self.save()

    def reset(self):
        """Reset checkpoint."""
        self.data = {
            "last_processed_id": 0,
            "total_processed": 0,
            "started_at": datetime.now().isoformat(),
            "last_update": None,
            "current_exchange": None,
            "current_symbol": None,
            "current_timeframe": None,
        }
        self.save()

    @property
    def last_id(self) -> int:
        return self.data.get("last_processed_id", 0)

    @property
    def total_processed(self) -> int:
        return self.data.get("total_processed", 0)


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


def get_or_train_hmm_model(timeframe: str, conn) -> Optional[HMMRegimeDetector]:
    """Load existing HMM model or train a new one."""
    model_path = MODEL_DIR / f"hmm_regime_{timeframe}.pkl"

    if model_path.exists():
        try:
            return HMMRegimeDetector.load(str(model_path))
        except Exception as e:
            print(f"  Warning: Failed to load model for {timeframe}: {e}")

    # Train new model
    print(f"  Training HMM model for {timeframe}...")

    # Get training data
    cursor = conn.cursor()
    cursor.execute("""
        SELECT open_time, open, high, low, close, volume
        FROM ohlcv
        WHERE timeframe = %s
        ORDER BY RANDOM()
        LIMIT 500000
    """, (timeframe,))

    rows = cursor.fetchall()
    if len(rows) < 1000:
        print(f"  Not enough data for {timeframe} ({len(rows)} rows)")
        return None

    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    df = compute_hmm_features(df)

    # Train
    detector = HMMRegimeDetector(
        n_states=4,
        features=["returns_1", "volatility", "volume_ratio"],
    )
    detector.fit_cross_symbol(df, verbose=True)

    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    detector.save(str(model_path))
    print(f"  Saved model: {model_path}")

    return detector


def get_all_strategy_instances() -> List:
    """Get all 34 strategy instances."""
    strategies = get_all_strategies()  # 27 from strategies module
    strategies.extend(get_ranging_strategies())  # 7 ranging
    return strategies


def process_batch(
    df: pd.DataFrame,
    detector: HMMRegimeDetector,
    strategies: List,
) -> pd.DataFrame:
    """Process a batch with HMM regime detection and all strategy signals."""
    df = compute_hmm_features(df)

    # HMM regime detection
    try:
        df["regime"] = detector.predict(df)
    except Exception as e:
        print(f"    Warning: HMM prediction failed: {e}")
        df["regime"] = "UNKNOWN"

    # Run all strategies
    for strategy in strategies:
        try:
            df = strategy.compute_indicators(df)

            # Get signals
            symbol = df["symbol"].iloc[0] if "symbol" in df.columns else "UNKNOWN"
            signals = strategy.generate_signals(df, symbol)

            # Map signals to dataframe
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

        except Exception as e:
            # Strategy failed, fill with neutral
            df[strategy.get_signal_column()] = 0
            df[strategy.get_confidence_column()] = None

    return df


def get_processing_status() -> dict:
    """Get current processing status."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Total rows
    cursor.execute("SELECT COUNT(*) FROM ohlcv")
    total = cursor.fetchone()[0]

    # Processed rows (have regime)
    cursor.execute("SELECT COUNT(*) FROM ohlcv WHERE regime IS NOT NULL")
    processed = cursor.fetchone()[0]

    # By exchange
    cursor.execute("""
        SELECT exchange,
               COUNT(*) as total,
               COUNT(regime) as processed
        FROM ohlcv
        GROUP BY exchange
    """)
    by_exchange = {row[0]: {"total": row[1], "processed": row[2]} for row in cursor.fetchall()}

    # By timeframe
    cursor.execute("""
        SELECT timeframe,
               COUNT(*) as total,
               COUNT(regime) as processed
        FROM ohlcv
        GROUP BY timeframe
        ORDER BY total DESC
    """)
    by_timeframe = {row[0]: {"total": row[1], "processed": row[2]} for row in cursor.fetchall()}

    conn.close()

    checkpoint = Checkpoint()

    return {
        "total_rows": total,
        "processed_rows": processed,
        "remaining": total - processed,
        "percent_complete": (processed / total * 100) if total > 0 else 0,
        "checkpoint": checkpoint.data,
        "by_exchange": by_exchange,
        "by_timeframe": by_timeframe,
    }


def print_status():
    """Print processing status."""
    status = get_processing_status()

    print("=" * 70)
    print("HMM + Strategy Processing Status")
    print("=" * 70)
    print(f"\nTotal rows:     {status['total_rows']:>15,}")
    print(f"Processed:      {status['processed_rows']:>15,}")
    print(f"Remaining:      {status['remaining']:>15,}")
    print(f"Complete:       {status['percent_complete']:>14.2f}%")

    print("\n--- By Exchange ---")
    for exchange, data in status["by_exchange"].items():
        pct = (data["processed"] / data["total"] * 100) if data["total"] > 0 else 0
        print(f"  {exchange:12s}: {data['processed']:>12,} / {data['total']:>12,} ({pct:5.1f}%)")

    print("\n--- By Timeframe ---")
    for tf, data in list(status["by_timeframe"].items())[:10]:
        pct = (data["processed"] / data["total"] * 100) if data["total"] > 0 else 0
        print(f"  {tf:8s}: {data['processed']:>12,} / {data['total']:>12,} ({pct:5.1f}%)")

    if status["checkpoint"]["last_update"]:
        print(f"\n--- Checkpoint ---")
        print(f"  Last update: {status['checkpoint']['last_update']}")
        print(f"  Last ID: {status['checkpoint']['last_processed_id']:,}")
        print(f"  Session total: {status['checkpoint']['total_processed']:,}")


def run_processing(
    batch_size: int = 5000,
    max_batches: int = 0,
    resume: bool = True,
):
    """Run the main processing loop."""
    global running

    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    print("=" * 70)
    print("HMM + 34 Strategy Processor")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print(f"Batch size: {batch_size:,}")

    # Load or reset checkpoint
    checkpoint = Checkpoint()
    if not resume:
        checkpoint.reset()
        print("Starting fresh (checkpoint reset)")
    else:
        print(f"Resuming from ID: {checkpoint.last_id:,}")
        print(f"Previously processed: {checkpoint.total_processed:,}")

    if checkpoint.data["started_at"] is None:
        checkpoint.data["started_at"] = datetime.now().isoformat()
        checkpoint.save()

    conn = get_db_connection()

    # Load strategies
    print("\nLoading strategies...")
    strategies = get_all_strategy_instances()
    print(f"Loaded {len(strategies)} strategies:")
    for s in strategies:
        print(f"  - {s.name}")

    # Get timeframes to process
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT timeframe
        FROM ohlcv
        WHERE regime IS NULL
        ORDER BY timeframe
    """)
    timeframes = [row[0] for row in cursor.fetchall()]
    print(f"\nTimeframes to process: {', '.join(timeframes)}")

    # Load/train HMM models
    print("\nLoading HMM models...")
    models = {}
    for tf in timeframes:
        model = get_or_train_hmm_model(tf, conn)
        if model:
            models[tf] = model
            print(f"  {tf}: ready")

    if not models:
        print("ERROR: No HMM models available!")
        return

    # Main processing loop
    print("\n" + "-" * 70)
    print("Starting processing...")
    print("-" * 70)

    total_processed = 0
    batch_count = 0
    start_time = time.time()

    while running:
        # Get next batch of unprocessed rows
        cursor.execute("""
            SELECT exchange, symbol, timeframe, open_time,
                   open, high, low, close, volume
            FROM ohlcv
            WHERE regime IS NULL
            ORDER BY open_time
            LIMIT %s
        """, (batch_size,))

        rows = cursor.fetchall()
        if not rows:
            print("\nAll data processed!")
            break

        batch_df = pd.DataFrame(rows, columns=[
            "exchange", "symbol", "timeframe", "open_time",
            "open", "high", "low", "close", "volume"
        ])
        batch_df["timestamp"] = pd.to_datetime(batch_df["open_time"], unit="ms")

        # Group by timeframe and process
        for timeframe, group_df in batch_df.groupby("timeframe"):
            if timeframe not in models:
                continue

            # Process batch
            processed_df = process_batch(group_df.copy(), models[timeframe], strategies)

            # Build updates
            updates = []
            for _, row in processed_df.iterrows():
                update = {
                    "regime": row.get("regime"),
                    "exchange": row["exchange"],
                    "symbol": row["symbol"],
                    "timeframe": row["timeframe"],
                    "open_time": int(row["open_time"]),
                }

                # Add all strategy signals
                for strategy in strategies:
                    sig_col = strategy.get_signal_column()
                    conf_col = strategy.get_confidence_column()
                    update[sig_col] = _to_python_type(row.get(sig_col, 0))
                    update[conf_col] = _to_python_type(row.get(conf_col))

                updates.append(update)

            # Update database
            if updates:
                # Build dynamic UPDATE query for all strategy columns
                set_clause = "regime = %(regime)s"
                for strategy in strategies:
                    sig_col = strategy.get_signal_column()
                    conf_col = strategy.get_confidence_column()
                    # Use safe column names (replace special chars)
                    set_clause += f", {sig_col} = %({sig_col})s"
                    set_clause += f", {conf_col} = %({conf_col})s"

                query = f"""
                    UPDATE ohlcv SET {set_clause}, processed_at = NOW()
                    WHERE exchange = %(exchange)s
                      AND symbol = %(symbol)s
                      AND timeframe = %(timeframe)s
                      AND open_time = %(open_time)s
                """

                execute_batch(cursor, query, updates, page_size=1000)
                conn.commit()

                total_processed += len(updates)

        # Update checkpoint
        last_open_time = int(batch_df["open_time"].max())
        checkpoint.update(
            last_id=last_open_time,
            count=len(batch_df),
            exchange=batch_df["exchange"].iloc[-1],
            symbol=batch_df["symbol"].iloc[-1],
            timeframe=batch_df["timeframe"].iloc[-1],
        )

        batch_count += 1

        # Progress update
        elapsed = time.time() - start_time
        rate = total_processed / elapsed if elapsed > 0 else 0
        print(f"  Batch {batch_count}: {total_processed:,} processed ({rate:.0f} rows/sec)")

        # Check max batches
        if max_batches > 0 and batch_count >= max_batches:
            print(f"\nReached max batches ({max_batches})")
            break

    # Final summary
    conn.close()
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("Processing Complete")
    print("=" * 70)
    print(f"Total processed this session: {total_processed:,}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Average rate: {total_processed/elapsed:.0f} rows/sec")
    print(f"\nCheckpoint saved. Run with --resume to continue later.")


def run_daemon(interval: int = 60, batch_size: int = 5000):
    """Run as a daemon, continuously processing new data."""
    global running

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    print("=" * 70)
    print("HMM + Strategy Processor - DAEMON MODE")
    print("=" * 70)
    print(f"Check interval: {interval} seconds")

    while running:
        status = get_processing_status()
        remaining = status["remaining"]

        if remaining > 0:
            print(f"\n[{datetime.now()}] Found {remaining:,} unprocessed rows")
            run_processing(batch_size=batch_size, max_batches=100, resume=True)
        else:
            print(f"[{datetime.now()}] All data processed, waiting...")

        if running:
            time.sleep(interval)

    print(f"\n[{datetime.now()}] Daemon stopped")


def main():
    parser = argparse.ArgumentParser(
        description="HMM + 34 Strategy Processor with pause/resume"
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show processing status"
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        default=True,
        help="Resume from checkpoint (default)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset checkpoint and start fresh"
    )
    parser.add_argument(
        "--daemon", "-d",
        action="store_true",
        help="Run as daemon (continuous processing)"
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
        help="Max batches to process (0 = unlimited)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Daemon check interval in seconds (default: 60)"
    )

    args = parser.parse_args()

    if args.status:
        print_status()
    elif args.daemon:
        run_daemon(interval=args.interval, batch_size=args.batch_size)
    elif args.reset:
        Checkpoint().reset()
        print("Checkpoint reset. Run without --reset to start processing.")
    else:
        run_processing(
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()

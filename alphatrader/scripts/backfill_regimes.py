#!/usr/bin/env python3
"""
Backfill regime labels for existing market_states data.

Usage:
    python scripts/backfill_regimes.py
    python scripts/backfill_regimes.py --batch-size 50000
    python scripts/backfill_regimes.py --symbol BTCUSDT --timeframe 1m
    python scripts/backfill_regimes.py --test  # Quick test on 10k rows
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.regime import HMMRegimeDetector


def get_db_stats(db_path: str) -> dict:
    """Get database statistics."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Total rows
    cursor.execute("SELECT COUNT(*) FROM market_states")
    total = cursor.fetchone()[0]

    # Symbols
    cursor.execute("SELECT DISTINCT symbol FROM market_states")
    symbols = [row[0] for row in cursor.fetchall()]

    # Timeframes
    cursor.execute("SELECT DISTINCT timeframe FROM market_states")
    timeframes = [row[0] for row in cursor.fetchall()]

    # Rows with regime already set
    cursor.execute("SELECT COUNT(*) FROM market_states WHERE regime IS NOT NULL")
    with_regime = cursor.fetchone()[0]

    conn.close()

    return {
        "total": total,
        "symbols": symbols,
        "timeframes": timeframes,
        "with_regime": with_regime,
        "without_regime": total - with_regime,
    }


def compute_hmm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features needed for HMM from raw OHLCV data.

    Args:
        df: DataFrame with open, high, low, close, volume columns

    Returns:
        DataFrame with returns_1, volatility, volume_ratio added
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
    df["volatility"] = df["volatility"].fillna(df["volatility"].mean() if df["volatility"].notna().any() else 0.01)
    df["volume_ratio"] = df["volume_ratio"].fillna(1.0)

    return df


def load_training_sample(db_path: str, timeframe: str, max_samples: int = 500000) -> pd.DataFrame:
    """Load a sample of data for HMM training."""
    conn = sqlite3.connect(db_path)

    # Get count for this timeframe
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM market_states WHERE timeframe = ?",
        (timeframe,)
    )
    count = cursor.fetchone()[0]

    if count == 0:
        conn.close()
        return pd.DataFrame()

    # Only fetch raw OHLCV - compute features later
    if count > max_samples:
        query = f"""
            SELECT id, timestamp, symbol, timeframe,
                   open, high, low, close, volume
            FROM market_states
            WHERE timeframe = ?
            ORDER BY RANDOM()
            LIMIT {max_samples}
        """
    else:
        query = """
            SELECT id, timestamp, symbol, timeframe,
                   open, high, low, close, volume
            FROM market_states
            WHERE timeframe = ?
        """

    df = pd.read_sql_query(query, conn, params=(timeframe,))
    conn.close()

    # Compute HMM features from OHLCV
    df = compute_hmm_features(df)

    return df


def backfill_timeframe(
    db_path: str,
    detector: HMMRegimeDetector,
    timeframe: str,
    symbol: str = None,
    batch_size: int = 50000,
) -> int:
    """
    Backfill regime labels for a specific timeframe.

    Returns:
        Number of rows updated
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Build query
    if symbol:
        cursor.execute(
            "SELECT COUNT(*) FROM market_states WHERE timeframe = ? AND symbol = ?",
            (timeframe, symbol)
        )
    else:
        cursor.execute(
            "SELECT COUNT(*) FROM market_states WHERE timeframe = ?",
            (timeframe,)
        )
    total = cursor.fetchone()[0]

    if total == 0:
        conn.close()
        return 0

    # Process in batches
    updated = 0
    pbar = tqdm(total=total, desc=f"  {timeframe}", unit="rows")

    offset = 0
    while offset < total:
        # Fetch batch - only raw OHLCV
        if symbol:
            query = """
                SELECT id, timestamp, symbol, timeframe,
                       open, high, low, close, volume
                FROM market_states
                WHERE timeframe = ? AND symbol = ?
                ORDER BY id
                LIMIT ? OFFSET ?
            """
            batch_df = pd.read_sql_query(
                query, conn,
                params=(timeframe, symbol, batch_size, offset)
            )
        else:
            query = """
                SELECT id, timestamp, symbol, timeframe,
                       open, high, low, close, volume
                FROM market_states
                WHERE timeframe = ?
                ORDER BY id
                LIMIT ? OFFSET ?
            """
            batch_df = pd.read_sql_query(
                query, conn,
                params=(timeframe, batch_size, offset)
            )

        if len(batch_df) == 0:
            break

        # Compute HMM features from OHLCV
        batch_df = compute_hmm_features(batch_df)

        # Predict regimes
        try:
            regimes = detector.predict(batch_df)
        except Exception as e:
            print(f"\n    Error predicting batch: {e}")
            offset += batch_size
            pbar.update(len(batch_df))
            continue

        # Update database
        updates = [
            (regime, int(row_id))
            for regime, row_id in zip(regimes, batch_df["id"])
        ]

        cursor.executemany(
            "UPDATE market_states SET regime = ? WHERE id = ?",
            updates
        )
        conn.commit()

        updated += len(batch_df)
        pbar.update(len(batch_df))
        offset += batch_size

    pbar.close()
    conn.close()

    return updated


def run_backfill(
    db_path: str = "data/alphatrader.db",
    batch_size: int = 50000,
    symbol: str = None,
    timeframe: str = None,
    test_mode: bool = False,
    model_dir: str = "models",
):
    """
    Run the full backfill process.

    Args:
        db_path: Path to SQLite database
        batch_size: Rows per batch
        symbol: Filter to specific symbol
        timeframe: Filter to specific timeframe
        test_mode: Only process 10k rows for testing
        model_dir: Directory to save/load HMM models
    """
    print("=" * 60)
    print("AlphaTrader Regime Backfill")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check database
    if not Path(db_path).exists():
        print(f"Error: Database not found at {db_path}")
        return

    stats = get_db_stats(db_path)
    print(f"\nDatabase: {db_path}")
    print(f"  Total rows: {stats['total']:,}")
    print(f"  Symbols: {', '.join(stats['symbols'])}")
    print(f"  Timeframes: {', '.join(stats['timeframes'])}")
    print(f"  Already labeled: {stats['with_regime']:,}")
    print(f"  Need labeling: {stats['without_regime']:,}")

    # Filter timeframes
    timeframes = [timeframe] if timeframe else stats["timeframes"]

    if test_mode:
        print("\n*** TEST MODE: Processing max 10k rows ***")
        batch_size = min(batch_size, 10000)

    # Create model directory
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    total_updated = 0

    for tf in timeframes:
        print(f"\n{'='*40}")
        print(f"Processing timeframe: {tf}")
        print(f"{'='*40}")

        model_path = Path(model_dir) / f"hmm_regime_{tf}.pkl"

        # Check for existing model
        if model_path.exists():
            print(f"Loading existing model: {model_path}")
            detector = HMMRegimeDetector.load(str(model_path))
        else:
            # Train new model
            print(f"Training new HMM model for {tf}...")

            train_samples = 100000 if test_mode else 500000
            train_df = load_training_sample(db_path, tf, max_samples=train_samples)

            if len(train_df) == 0:
                print(f"  No data for timeframe {tf}, skipping...")
                continue

            print(f"  Training samples: {len(train_df):,}")

            detector = HMMRegimeDetector(
                n_states=4,
                features=["returns_1", "volatility", "volume_ratio"],
            )

            detector.fit_cross_symbol(train_df, verbose=True)

            # Save model
            detector.save(str(model_path))
            print(f"  Saved model: {model_path}")

        # Run backfill
        if test_mode:
            # Only update 10k rows in test mode
            updated = backfill_timeframe(
                db_path, detector, tf, symbol,
                batch_size=min(10000, batch_size)
            )
            total_updated += min(updated, 10000)
            if total_updated >= 10000:
                break
        else:
            updated = backfill_timeframe(
                db_path, detector, tf, symbol, batch_size
            )
            total_updated += updated

    # Print final stats
    print("\n" + "=" * 60)
    print("Backfill Complete")
    print("=" * 60)
    print(f"Total rows updated: {total_updated:,}")

    # Show regime distribution
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT regime, COUNT(*) as cnt
        FROM market_states
        WHERE regime IS NOT NULL
        GROUP BY regime
        ORDER BY cnt DESC
    """)
    print("\nRegime Distribution:")
    for row in cursor.fetchall():
        pct = row[1] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {row[0] or 'NULL':15s}: {row[1]:>12,} ({pct:5.1f}%)")
    conn.close()

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    parser = argparse.ArgumentParser(
        description="Backfill regime labels for existing market_states data"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/alphatrader.db",
        help="Path to database"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Rows per batch"
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
        "--test",
        action="store_true",
        help="Test mode: only process 10k rows"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory for HMM models"
    )

    args = parser.parse_args()

    run_backfill(
        db_path=args.db,
        batch_size=args.batch_size,
        symbol=args.symbol,
        timeframe=args.timeframe,
        test_mode=args.test,
        model_dir=args.model_dir,
    )


if __name__ == "__main__":
    main()

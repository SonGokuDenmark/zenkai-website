#!/usr/bin/env python3
"""
Complete Stock Data Setup - Download Kaggle datasets and start continuous collection.

This script:
1. Downloads historical stock data from Kaggle (20+ years)
2. Imports it into PostgreSQL
3. Starts continuous Yahoo Finance collection

Usage:
    python setup_stock_data.py              # Full setup
    python setup_stock_data.py --skip-kaggle  # Skip Kaggle, just start collector
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Check dependencies
try:
    import kagglehub
except ImportError:
    print("Installing kagglehub...")
    os.system(f"{sys.executable} -m pip install kagglehub")
    import kagglehub

try:
    import yfinance
except ImportError:
    print("Installing yfinance...")
    os.system(f"{sys.executable} -m pip install yfinance")

try:
    import pandas as pd
    import psycopg2
    from psycopg2.extras import execute_values
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install pandas psycopg2-binary tqdm")
    sys.exit(1)

from dotenv import load_dotenv
load_dotenv()

# Database config
DB_CONFIG = {
    "host": os.getenv("ZENKAI_DB_HOST", "192.168.0.160"),
    "database": os.getenv("ZENKAI_DB_NAME", "zenkai_data"),
    "user": os.getenv("ZENKAI_DB_USER", "zenkai"),
    "password": os.getenv("ZENKAI_DB_PASSWORD"),
}


def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


def download_kaggle_datasets():
    """Download both Kaggle stock datasets."""
    print("\n" + "=" * 60)
    print("Step 1: Downloading Kaggle Datasets")
    print("=" * 60)

    datasets = {}

    # Boris dataset - Huge Stock Market Dataset
    print("\n[1/2] Downloading Boris dataset (all US stocks + ETFs)...")
    try:
        path = kagglehub.dataset_download("borismarjanovic/price-volume-data-for-all-us-stocks-etfs")
        print(f"  Downloaded to: {path}")
        datasets["boris"] = Path(path)
    except Exception as e:
        print(f"  Error: {e}")
        datasets["boris"] = None

    # Jackson dataset - Stock Market Dataset
    print("\n[2/2] Downloading Jackson dataset (NASDAQ stocks)...")
    try:
        path = kagglehub.dataset_download("jacksoncrow/stock-market-dataset")
        print(f"  Downloaded to: {path}")
        datasets["jackson"] = Path(path)
    except Exception as e:
        print(f"  Error: {e}")
        datasets["jackson"] = None

    return datasets


def import_boris_dataset(path: Path) -> dict:
    """Import Boris dataset (Stocks/ and ETFs/ folders with .txt files)."""
    stats = {"files": 0, "rows": 0, "errors": 0}

    if not path:
        return stats

    conn = get_db_connection()
    cursor = conn.cursor()

    # Find all data files
    files = []

    stocks_dir = path / "Stocks"
    etfs_dir = path / "ETFs"

    # Also check for lowercase
    if not stocks_dir.exists():
        stocks_dir = path / "stocks"
    if not etfs_dir.exists():
        etfs_dir = path / "etfs"

    if stocks_dir.exists():
        for f in stocks_dir.glob("*.txt"):
            files.append((f, "stock"))
        for f in stocks_dir.glob("*.csv"):
            files.append((f, "stock"))

    if etfs_dir.exists():
        for f in etfs_dir.glob("*.txt"):
            files.append((f, "etf"))
        for f in etfs_dir.glob("*.csv"):
            files.append((f, "etf"))

    print(f"  Found {len(files)} files to import")

    pbar = tqdm(files, desc="  Importing Boris", unit="files")
    for filepath, asset_type in pbar:
        symbol = filepath.stem.upper()
        pbar.set_postfix(symbol=symbol)

        try:
            # Read CSV (Boris format: Date,Open,High,Low,Close,Volume,OpenInt)
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.lower()

            if "date" not in df.columns or df.empty:
                continue

            df["date"] = pd.to_datetime(df["date"])

            # Prepare rows
            rows = []
            for _, row in df.iterrows():
                open_time = int(row["date"].timestamp() * 1000)
                close_time = open_time + 86400000 - 1

                rows.append((
                    "us_stock", symbol, "1d", open_time,
                    float(row["open"]), float(row["high"]),
                    float(row["low"]), float(row["close"]),
                    float(row["volume"]), close_time, asset_type
                ))

            if rows:
                execute_values(cursor, """
                    INSERT INTO ohlcv (exchange, symbol, timeframe, open_time,
                        open, high, low, close, volume, close_time, asset_type)
                    VALUES %s
                    ON CONFLICT (exchange, symbol, timeframe, open_time) DO UPDATE SET
                        open = EXCLUDED.open, high = EXCLUDED.high,
                        low = EXCLUDED.low, close = EXCLUDED.close,
                        volume = EXCLUDED.volume
                """, rows)
                conn.commit()
                stats["files"] += 1
                stats["rows"] += len(rows)

        except Exception as e:
            stats["errors"] += 1

    conn.close()
    return stats


def import_jackson_dataset(path: Path) -> dict:
    """Import Jackson dataset (stocks/ and etfs/ folders with CSV files)."""
    stats = {"files": 0, "rows": 0, "errors": 0}

    if not path:
        return stats

    conn = get_db_connection()
    cursor = conn.cursor()

    # Find all data files
    files = []

    stocks_dir = path / "stocks"
    etfs_dir = path / "etfs"

    if stocks_dir.exists():
        for f in stocks_dir.glob("*.csv"):
            files.append((f, "stock"))

    if etfs_dir.exists():
        for f in etfs_dir.glob("*.csv"):
            files.append((f, "etf"))

    print(f"  Found {len(files)} files to import")

    pbar = tqdm(files, desc="  Importing Jackson", unit="files")
    for filepath, asset_type in pbar:
        symbol = filepath.stem.upper()
        pbar.set_postfix(symbol=symbol)

        try:
            # Read CSV (Jackson format: Date,Open,High,Low,Close,Adj Close,Volume)
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.lower().str.replace(" ", "_")

            if "date" not in df.columns or df.empty:
                continue

            df["date"] = pd.to_datetime(df["date"])

            # Use adjusted close if available
            close_col = "adj_close" if "adj_close" in df.columns else "close"

            # Prepare rows
            rows = []
            for _, row in df.iterrows():
                open_time = int(row["date"].timestamp() * 1000)
                close_time = open_time + 86400000 - 1

                rows.append((
                    "us_stock", symbol, "1d", open_time,
                    float(row["open"]), float(row["high"]),
                    float(row["low"]), float(row[close_col]),
                    float(row["volume"]), close_time, asset_type
                ))

            if rows:
                execute_values(cursor, """
                    INSERT INTO ohlcv (exchange, symbol, timeframe, open_time,
                        open, high, low, close, volume, close_time, asset_type)
                    VALUES %s
                    ON CONFLICT (exchange, symbol, timeframe, open_time) DO UPDATE SET
                        open = EXCLUDED.open, high = EXCLUDED.high,
                        low = EXCLUDED.low, close = EXCLUDED.close,
                        volume = EXCLUDED.volume
                """, rows)
                conn.commit()
                stats["files"] += 1
                stats["rows"] += len(rows)

        except Exception as e:
            stats["errors"] += 1

    conn.close()
    return stats


def import_all_datasets(datasets: dict):
    """Import all downloaded datasets."""
    print("\n" + "=" * 60)
    print("Step 2: Importing Data to PostgreSQL")
    print("=" * 60)

    total_stats = {"files": 0, "rows": 0, "errors": 0}

    # Import Boris dataset
    if datasets.get("boris"):
        print("\n[1/2] Importing Boris dataset...")
        stats = import_boris_dataset(datasets["boris"])
        print(f"  Imported {stats['files']:,} files, {stats['rows']:,} rows")
        for k in total_stats:
            total_stats[k] += stats[k]

    # Import Jackson dataset
    if datasets.get("jackson"):
        print("\n[2/2] Importing Jackson dataset...")
        stats = import_jackson_dataset(datasets["jackson"])
        print(f"  Imported {stats['files']:,} files, {stats['rows']:,} rows")
        for k in total_stats:
            total_stats[k] += stats[k]

    return total_stats


def show_database_stats():
    """Show current database statistics."""
    conn = get_db_connection()
    cursor = conn.cursor()

    print("\n" + "=" * 60)
    print("Database Statistics")
    print("=" * 60)

    # Total rows by asset type
    cursor.execute("""
        SELECT asset_type, COUNT(*) as rows, COUNT(DISTINCT symbol) as symbols
        FROM ohlcv
        GROUP BY asset_type
        ORDER BY rows DESC
    """)

    print(f"\n{'Asset Type':<15} {'Rows':<15} {'Symbols':<10}")
    print("-" * 40)
    for row in cursor.fetchall():
        print(f"{row[0] or 'unknown':<15} {row[1]:>12,} {row[2]:>8,}")

    # Date range for stocks
    cursor.execute("""
        SELECT MIN(open_time), MAX(open_time)
        FROM ohlcv WHERE asset_type IN ('stock', 'etf')
    """)
    result = cursor.fetchone()
    if result[0]:
        min_date = datetime.fromtimestamp(result[0] / 1000)
        max_date = datetime.fromtimestamp(result[1] / 1000)
        print(f"\nStock data range: {min_date.date()} to {max_date.date()}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Complete Stock Data Setup")
    parser.add_argument(
        "--skip-kaggle",
        action="store_true",
        help="Skip Kaggle download, just show stats"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show database statistics"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("AlphaTrader Stock Data Setup")
    print("=" * 60)
    print(f"Started: {datetime.now()}")

    if args.stats_only:
        show_database_stats()
        return

    if not args.skip_kaggle:
        # Download Kaggle datasets
        datasets = download_kaggle_datasets()

        # Import to PostgreSQL
        stats = import_all_datasets(datasets)

        print("\n" + "=" * 60)
        print("Import Summary")
        print("=" * 60)
        print(f"Total files: {stats['files']:,}")
        print(f"Total rows: {stats['rows']:,}")
        print(f"Errors: {stats['errors']:,}")

    # Show final stats
    show_database_stats()

    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Start continuous collection:")
    print("     python stock_collector.py --daemon --sp500")
    print("")
    print("  2. Or run as systemd service on server:")
    print("     sudo systemctl start stock-collector")
    print("")
    print(f"Finished: {datetime.now()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Import Kaggle Stock Datasets into PostgreSQL.

Supports:
- Huge Stock Market Dataset (borismarjanovic/price-volume-data-for-all-us-stocks-etfs)
- Stock Market Dataset (jacksoncrow/stock-market-dataset)

Usage:
    python import_kaggle.py --path /path/to/kaggle/data
    python import_kaggle.py --path /path/to/data --limit 100  # Test with 100 stocks
    python import_kaggle.py --path /path/to/data --format boris  # Boris dataset format
    python import_kaggle.py --path /path/to/data --format jackson  # Jackson dataset format
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Database config
DB_CONFIG = {
    "host": os.getenv("ZENKAI_DB_HOST", "192.168.0.160"),
    "database": os.getenv("ZENKAI_DB_NAME", "zenkai_data"),
    "user": os.getenv("ZENKAI_DB_USER", "zenkai"),
    "password": os.getenv("ZENKAI_DB_PASSWORD"),
}


def get_db_connection():
    """Get PostgreSQL connection."""
    return psycopg2.connect(**DB_CONFIG)


def detect_format(path: Path) -> str:
    """Auto-detect Kaggle dataset format."""
    # Boris format has Stocks/ and ETFs/ folders with individual CSV files
    if (path / "Stocks").exists() or (path / "ETFs").exists():
        return "boris"

    # Jackson format has stocks/ and etfs/ folders
    if (path / "stocks").exists() or (path / "etfs").exists():
        return "jackson"

    # Check for direct CSV files
    csv_files = list(path.glob("*.csv"))
    if csv_files:
        # Read first file to detect columns
        df = pd.read_csv(csv_files[0], nrows=1)
        if "Adj Close" in df.columns:
            return "jackson"
        elif "OpenInt" in df.columns:
            return "boris"

    return "unknown"


def get_csv_files(path: Path, format_type: str) -> List[tuple]:
    """Get list of CSV files with their asset type."""
    files = []

    if format_type == "boris":
        # Boris format: Stocks/ and ETFs/ directories
        stocks_dir = path / "Stocks"
        etfs_dir = path / "ETFs"

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

    elif format_type == "jackson":
        # Jackson format: stocks/ and etfs/ directories
        stocks_dir = path / "stocks"
        etfs_dir = path / "etfs"

        if stocks_dir.exists():
            for f in stocks_dir.glob("*.csv"):
                files.append((f, "stock"))

        if etfs_dir.exists():
            for f in etfs_dir.glob("*.csv"):
                files.append((f, "etf"))

    return files


def parse_boris_csv(filepath: Path) -> pd.DataFrame:
    """Parse Boris dataset CSV format."""
    # Boris format: Date,Open,High,Low,Close,Volume,OpenInt
    df = pd.read_csv(filepath)

    # Standardize column names
    df.columns = df.columns.str.lower()

    # Ensure required columns
    required = ["date", "open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Parse date
    df["date"] = pd.to_datetime(df["date"])

    return df[["date", "open", "high", "low", "close", "volume"]]


def parse_jackson_csv(filepath: Path) -> pd.DataFrame:
    """Parse Jackson dataset CSV format."""
    # Jackson format: Date,Open,High,Low,Close,Adj Close,Volume
    df = pd.read_csv(filepath)

    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # Ensure required columns
    required = ["date", "open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Parse date
    df["date"] = pd.to_datetime(df["date"])

    # Use adjusted close if available
    if "adj_close" in df.columns:
        df["close"] = df["adj_close"]

    return df[["date", "open", "high", "low", "close", "volume"]]


def insert_stock_data(
    symbol: str,
    asset_type: str,
    df: pd.DataFrame,
    exchange: str = "us_stock",
) -> int:
    """Insert stock data into database. Returns rows inserted."""
    if df.empty:
        return 0

    conn = get_db_connection()
    cursor = conn.cursor()

    # Prepare data - convert date to timestamp (milliseconds)
    rows = []
    for _, row in df.iterrows():
        open_time = int(row["date"].timestamp() * 1000)
        close_time = open_time + 86400000 - 1  # End of day

        rows.append((
            exchange,           # exchange
            symbol,             # symbol
            "1d",               # timeframe (daily)
            open_time,          # open_time
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            float(row["volume"]),
            close_time,         # close_time
            asset_type,         # asset_type
        ))

    # Upsert
    query = """
        INSERT INTO ohlcv (
            exchange, symbol, timeframe, open_time,
            open, high, low, close, volume,
            close_time, asset_type
        ) VALUES %s
        ON CONFLICT (exchange, symbol, timeframe, open_time)
        DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            close_time = EXCLUDED.close_time,
            asset_type = EXCLUDED.asset_type
    """

    try:
        execute_values(cursor, query, rows)
        conn.commit()
        inserted = len(rows)
    except Exception as e:
        print(f"  Error inserting {symbol}: {e}")
        conn.rollback()
        inserted = 0
    finally:
        conn.close()

    return inserted


def import_dataset(
    path: Path,
    format_type: str,
    limit: Optional[int] = None,
    exchange: str = "us_stock",
) -> dict:
    """Import entire dataset."""
    stats = {
        "files_processed": 0,
        "files_failed": 0,
        "rows_inserted": 0,
        "symbols": [],
    }

    # Get files
    files = get_csv_files(path, format_type)
    if limit:
        files = files[:limit]

    print(f"Found {len(files)} files to import")

    # Parse function based on format
    if format_type == "boris":
        parse_func = parse_boris_csv
    else:
        parse_func = parse_jackson_csv

    # Process files
    pbar = tqdm(files, desc="Importing", unit="files")
    for filepath, asset_type in pbar:
        # Extract symbol from filename
        symbol = filepath.stem.upper()
        pbar.set_postfix(symbol=symbol)

        try:
            # Parse CSV
            df = parse_func(filepath)

            if df.empty:
                continue

            # Insert to database
            inserted = insert_stock_data(symbol, asset_type, df, exchange)

            if inserted > 0:
                stats["files_processed"] += 1
                stats["rows_inserted"] += inserted
                stats["symbols"].append(symbol)

        except Exception as e:
            stats["files_failed"] += 1
            # Only print for unexpected errors, not empty files
            if "Missing column" not in str(e):
                print(f"\n  Error processing {filepath.name}: {e}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Import Kaggle Stock Data")
    parser.add_argument(
        "--path", "-p",
        type=str,
        required=True,
        help="Path to extracted Kaggle dataset"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["boris", "jackson", "auto"],
        default="auto",
        help="Dataset format (default: auto-detect)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        help="Limit number of files to import (for testing)"
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default="us_stock",
        help="Exchange name to use (default: us_stock)"
    )

    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        sys.exit(1)

    print("=" * 60)
    print("Kaggle Stock Data Importer")
    print("=" * 60)
    print(f"Started: {datetime.now()}")
    print(f"Path: {path}")

    # Detect format
    if args.format == "auto":
        format_type = detect_format(path)
        if format_type == "unknown":
            print("Error: Could not auto-detect dataset format")
            print("Please specify --format boris or --format jackson")
            sys.exit(1)
        print(f"Detected format: {format_type}")
    else:
        format_type = args.format
        print(f"Format: {format_type}")

    if args.limit:
        print(f"Limit: {args.limit} files")

    print()

    # Import
    stats = import_dataset(path, format_type, args.limit, args.exchange)

    # Summary
    print("\n" + "=" * 60)
    print("Import Complete")
    print("=" * 60)
    print(f"Files processed: {stats['files_processed']:,}")
    print(f"Files failed: {stats['files_failed']:,}")
    print(f"Rows inserted: {stats['rows_inserted']:,}")
    print(f"Symbols: {len(stats['symbols']):,}")

    # Date range check
    if stats["symbols"]:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                MIN(open_time), MAX(open_time),
                COUNT(DISTINCT symbol)
            FROM ohlcv
            WHERE exchange = %s
        """, (args.exchange,))
        result = cursor.fetchone()
        conn.close()

        if result[0]:
            min_date = datetime.fromtimestamp(result[0] / 1000)
            max_date = datetime.fromtimestamp(result[1] / 1000)
            print(f"\nData range: {min_date.date()} to {max_date.date()}")
            print(f"Total symbols in database: {result[2]:,}")

    print(f"\nFinished: {datetime.now()}")


if __name__ == "__main__":
    main()

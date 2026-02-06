#!/usr/bin/env python3
"""
Parallel Database Import - Speed up bulk import using multiple processes.
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
import psycopg2
import pandas as pd
from io import StringIO

from dotenv import load_dotenv
load_dotenv()

# Config
DATA_DIR = Path("/mnt/data/binance_bulk/spot")
DB_CONFIG = {
    "host": os.getenv("ZENKAI_DB_HOST", "192.168.0.160"),
    "database": os.getenv("ZENKAI_DB_NAME", "zenkai_data"),
    "user": os.getenv("ZENKAI_DB_USER", "zenkai"),
    "password": os.getenv("ZENKAI_DB_PASSWORD"),
}
WORKERS = 8  # Parallel workers


def import_symbol(symbol_dir):
    """Import all CSV files for one symbol using fast execute_values."""
    from psycopg2.extras import execute_values

    symbol = symbol_dir.name
    csv_files = list(symbol_dir.glob("*.csv"))

    if not csv_files:
        return (symbol, 0, 0)

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        total_rows = 0
        files_done = 0

        for csv_file in csv_files:
            try:
                df = pd.read_csv(
                    csv_file,
                    header=None,
                    names=[
                        "open_time", "open", "high", "low", "close", "volume",
                        "close_time", "quote_volume", "trades", "taker_buy_base",
                        "taker_buy_quote", "ignore"
                    ]
                )

                # Fast batch insert with execute_values
                insert_sql = """
                    INSERT INTO ohlcv (exchange, symbol, timeframe, open_time, open, high, low, close, volume, quote_volume, close_time)
                    VALUES %s
                    ON CONFLICT (exchange, symbol, timeframe, open_time) DO NOTHING
                """

                values = [
                    (
                        "binance",
                        symbol,
                        "1m",
                        int(row["open_time"]),
                        float(row["open"]),
                        float(row["high"]),
                        float(row["low"]),
                        float(row["close"]),
                        float(row["volume"]),
                        float(row["quote_volume"]),
                        int(row["close_time"]),
                    )
                    for _, row in df.iterrows()
                ]

                execute_values(cursor, insert_sql, values, page_size=5000)
                conn.commit()

                total_rows += len(values)
                files_done += 1

            except Exception as e:
                conn.rollback()
                continue

        cursor.close()
        conn.close()

        return (symbol, files_done, total_rows)

    except Exception as e:
        return (symbol, 0, 0)


def main():
    print("=" * 60)
    print("PARALLEL DATABASE IMPORT")
    print("=" * 60)
    print(f"Workers: {WORKERS}")
    print(f"Data dir: {DATA_DIR}")

    # Get all symbol directories
    symbol_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
    print(f"Symbols to import: {len(symbol_dirs)}")
    print("=" * 60 + "\n")

    start_time = time.time()
    completed = 0
    total_rows = 0

    with Pool(WORKERS) as pool:
        for result in pool.imap_unordered(import_symbol, symbol_dirs):
            symbol, files, rows = result
            completed += 1
            total_rows += rows

            if rows > 0:
                print(f"[{completed}/{len(symbol_dirs)}] {symbol}: {files} files, {rows:,} rows")
            else:
                print(f"[{completed}/{len(symbol_dirs)}] {symbol}: skipped")

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("IMPORT COMPLETE")
    print("=" * 60)
    print(f"Symbols: {completed}")
    print(f"Total rows: {total_rows:,}")
    print(f"Time: {elapsed/60:.1f} minutes")
    print("=" * 60)

    # Sync symbol status
    print("\nSyncing symbol status...")
    os.system("python3 /home/shared/alphatrader/scripts/bulk_download_binance.py --sync-status")


if __name__ == "__main__":
    main()

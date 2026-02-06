#!/usr/bin/env python3
"""
Smart Import - Skip already imported symbols, only import new ones.
"""

import os
import sys
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
import time

from dotenv import load_dotenv
load_dotenv()

DATA_DIR = Path("/mnt/data/binance_bulk/spot")
DB_CONFIG = {
    "host": os.getenv("ZENKAI_DB_HOST", "192.168.0.160"),
    "database": os.getenv("ZENKAI_DB_NAME", "zenkai_data"),
    "user": os.getenv("ZENKAI_DB_USER", "zenkai"),
    "password": os.getenv("ZENKAI_DB_PASSWORD"),
}


def get_imported_symbols():
    """Get list of symbols already in database from symbol_status table."""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    # Use symbol_status table instead of scanning ohlcv (much faster)
    cursor.execute("SELECT symbol FROM symbol_status")
    symbols = set(row[0] for row in cursor.fetchall())
    cursor.close()
    conn.close()
    return symbols


def import_symbol(symbol_dir, conn):
    """Import all files for a symbol."""
    symbol = symbol_dir.name
    csv_files = sorted(symbol_dir.glob("*.csv"))

    if not csv_files:
        return 0

    cursor = conn.cursor()
    total_rows = 0

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

        except Exception as e:
            conn.rollback()
            print(f"    Error {csv_file.name}: {e}")

    cursor.close()
    return total_rows


def main():
    print("=" * 60)
    print("SMART IMPORT - Skip existing symbols")
    print("=" * 60)

    # Get already imported
    print("Checking database for existing symbols...")
    imported = get_imported_symbols()
    print(f"Already imported: {len(imported)} symbols")

    # Get all symbol dirs
    all_symbols = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
    print(f"Total available: {len(all_symbols)} symbols")

    # Find new symbols
    new_symbols = [d for d in all_symbols if d.name not in imported]
    print(f"New to import: {len(new_symbols)} symbols")
    print("=" * 60 + "\n")

    if not new_symbols:
        print("Nothing to import! All symbols already in database.")
        return

    # Import new symbols
    conn = psycopg2.connect(**DB_CONFIG)
    start_time = time.time()
    total_rows = 0

    for i, symbol_dir in enumerate(new_symbols, 1):
        symbol = symbol_dir.name
        file_count = len(list(symbol_dir.glob("*.csv")))
        print(f"[{i}/{len(new_symbols)}] {symbol} ({file_count} files)...", end=" ", flush=True)

        rows = import_symbol(symbol_dir, conn)
        total_rows += rows
        print(f"{rows:,} rows")

    conn.close()
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("IMPORT COMPLETE")
    print("=" * 60)
    print(f"Symbols imported: {len(new_symbols)}")
    print(f"Total rows: {total_rows:,}")
    print(f"Time: {elapsed/60:.1f} minutes")

    # Sync status
    print("\nSyncing symbol status...")
    os.system("python3 /home/shared/alphatrader/scripts/bulk_download_binance.py --sync-status")


if __name__ == "__main__":
    main()

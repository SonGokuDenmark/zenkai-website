#!/usr/bin/env python3
"""
Fast Import - Use COPY for untouched symbols (no conflict checking = 10x faster)
"""

import os
import sys
from pathlib import Path
from io import StringIO
import psycopg2
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

# Symbols already partially imported (from previous import logs)
# Just skip these - they already have data
SKIP_SYMBOLS = {
    "AEVOUSDT", "ARPAUSDT", "DOTUSDT", "ENSOUSDT", "FFUSDT",
    "FIDAUSDT", "LRCUSDT", "METUSDT", "RADUSDT", "SYRUPUSDT", "XPLUSDT",
    "0GUSDT", "1000CATUSDT", "1000CHEEMSUSDT", "1000SATSUSDT", "1INCHUSDT"
}


def import_symbol_fast(symbol_dir, conn):
    """Import using COPY - no conflict check, super fast."""
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

            # Add required columns
            df["exchange"] = "binance"
            df["symbol"] = symbol
            df["timeframe"] = "1m"

            # Select columns in correct order for COPY
            df_out = df[["exchange", "symbol", "timeframe", "open_time", "open", "high", "low", "close", "volume", "quote_volume", "close_time"]]

            # Use COPY via StringIO - super fast, no conflict check
            output = StringIO()
            df_out.to_csv(output, sep='\t', header=False, index=False)
            output.seek(0)

            cursor.copy_from(
                output,
                'ohlcv',
                sep='\t',
                columns=["exchange", "symbol", "timeframe", "open_time", "open", "high", "low", "close", "volume", "quote_volume", "close_time"]
            )
            conn.commit()
            total_rows += len(df_out)

        except Exception as e:
            conn.rollback()
            # Skip duplicate errors silently
            if "duplicate" in str(e).lower() or "already exists" in str(e).lower():
                continue
            # Print other errors but continue
            print(f"\n    Error {csv_file.name}: {str(e)[:50]}")
            continue

    cursor.close()
    return total_rows


def main():
    print("=" * 60)
    print("FAST IMPORT - COPY method (no conflict check)")
    print("=" * 60)

    # Get all symbol dirs
    all_symbols = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])

    # Filter out already-processed symbols
    new_symbols = [d for d in all_symbols if d.name not in SKIP_SYMBOLS]

    print(f"Total symbols: {len(all_symbols)}")
    print(f"Skipping (known): {len(SKIP_SYMBOLS)}")
    print(f"To import: {len(new_symbols)}")
    print("=" * 60 + "\n")

    conn = psycopg2.connect(**DB_CONFIG)
    start_time = time.time()
    total_rows = 0

    for i, symbol_dir in enumerate(new_symbols, 1):
        symbol = symbol_dir.name
        file_count = len(list(symbol_dir.glob("*.csv")))
        print(f"[{i}/{len(new_symbols)}] {symbol} ({file_count} files)...", end=" ", flush=True)

        symbol_start = time.time()
        rows = import_symbol_fast(symbol_dir, conn)
        elapsed = time.time() - symbol_start

        total_rows += rows
        print(f"{rows:,} rows ({elapsed:.1f}s)")

    conn.close()
    total_elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("IMPORT COMPLETE")
    print("=" * 60)
    print(f"Symbols: {len(new_symbols)}")
    print(f"Total rows: {total_rows:,}")
    print(f"Time: {total_elapsed/60:.1f} minutes")
    print(f"Rate: {total_rows/total_elapsed:,.0f} rows/sec")

    # Sync symbol status
    print("\nSyncing symbol status...")
    os.system("python3 /home/shared/alphatrader/scripts/bulk_download_binance.py --sync-status")

    # Start HMM processing
    print("\n" + "=" * 60)
    print("STARTING HMM REGIME DETECTION")
    print("=" * 60)
    print("This will run in background. Check status with:")
    print("  python3 /home/shared/alphatrader/server/hmm_processor.py --status")
    os.system("cd /home/shared/alphatrader/server && nohup python3 hmm_processor.py --daemon > /home/shared/alphatrader/logs/hmm_processor.log 2>&1 &")


if __name__ == "__main__":
    main()

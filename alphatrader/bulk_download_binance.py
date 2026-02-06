#!/usr/bin/env python3
"""
Binance Bulk Data Downloader + Dead Coin Filter

Downloads ALL historical data from Binance Public Data (data.binance.vision).
No API limits, no rate limiting - just bulk downloads.

Features:
- Download 1-minute data for ALL USDT pairs
- Automatic dead coin detection (delisted symbols)
- Tag delisted coins in DB (keep data, exclude from training)
- Resample to any timeframe
- Direct pipe to TimescaleDB

Usage:
    python bulk_download_binance.py --check              # Check available data
    python bulk_download_binance.py --all                # Download ALL symbols
    python bulk_download_binance.py --import-to-db       # Import to PostgreSQL
    python bulk_download_binance.py --sync-status        # Update coin status (live/delisted)
    python bulk_download_binance.py --list-delisted      # Show delisted coins in DB
"""

import os
import sys
import argparse
import glob
import gzip
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Set, Tuple
import requests
import zipfile
import io
import time
import concurrent.futures
from threading import Lock

from dotenv import load_dotenv
load_dotenv()

# Try to import optional packages
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not installed. Some features disabled.")

try:
    import psycopg2
    from psycopg2.extras import execute_values
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False
    print("Warning: psycopg2 not installed. DB import disabled.")

try:
    from binance_historical_data import BinanceDataDumper
    HAS_DUMPER = True
except ImportError:
    HAS_DUMPER = False


# ============================================================
# CONFIGURATION
# ============================================================

# Download directory - use /mnt/data HDD if available (1.7TB), otherwise local
DATA_DIR = Path(os.getenv("BINANCE_DATA_DIR", "/mnt/data/binance_bulk"))

# Database config
DB_CONFIG = {
    "host": os.getenv("ZENKAI_DB_HOST", "192.168.0.160"),
    "database": os.getenv("ZENKAI_DB_NAME", "zenkai_data"),
    "user": os.getenv("ZENKAI_DB_USER", "zenkai"),
    "password": os.getenv("ZENKAI_DB_PASSWORD"),
}

# Binance public data URLs
BINANCE_DATA_URL = "https://data.binance.vision"
BINANCE_API_URL = "https://api.binance.com"

# Timeframes to generate from 1m data
RESAMPLE_TIMEFRAMES = {
    "1m": "1T",
    "3m": "3T",
    "5m": "5T",
    "15m": "15T",
    "30m": "30T",
    "1h": "1H",
    "2h": "2H",
    "4h": "4H",
    "6h": "6H",
    "8h": "8H",
    "12h": "12H",
    "1d": "1D",
}

# Thread-safe print
print_lock = Lock()
def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


# ============================================================
# SYMBOL STATUS / DEAD COIN FILTER
# ============================================================

def get_binance_exchange_info() -> Dict:
    """Get full exchange info from Binance API."""
    url = f"{BINANCE_API_URL}/api/v3/exchangeInfo"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def get_active_symbols() -> Dict[str, Dict]:
    """
    Get all currently active TRADING symbols from Binance.
    Returns dict of symbol -> info
    """
    print("Fetching active symbols from Binance API...")
    data = get_binance_exchange_info()

    active = {}
    for s in data["symbols"]:
        if s["symbol"].endswith("USDT"):
            active[s["symbol"]] = {
                "status": s.get("status"),  # TRADING, BREAK, HALT, etc.
                "base_asset": s.get("baseAsset"),
                "quote_asset": s.get("quoteAsset"),
                "is_trading": s.get("status") == "TRADING",
            }

    trading_count = sum(1 for v in active.values() if v["is_trading"])
    print(f"Found {len(active)} USDT pairs, {trading_count} actively TRADING")
    return active


def get_all_usdt_symbols() -> List[str]:
    """Get all USDT trading pairs from Binance (actively trading only)."""
    active = get_active_symbols()
    symbols = [s for s, info in active.items() if info["is_trading"]]
    return sorted(symbols)


def create_symbol_status_table(conn):
    """Create symbol_status table if it doesn't exist."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS symbol_status (
            symbol VARCHAR(50) PRIMARY KEY,
            status VARCHAR(20) NOT NULL DEFAULT 'UNKNOWN',
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            base_asset VARCHAR(20),
            quote_asset VARCHAR(20),
            first_seen TIMESTAMP DEFAULT NOW(),
            last_checked TIMESTAMP DEFAULT NOW(),
            last_traded TIMESTAMP,
            notes TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_symbol_status_active
            ON symbol_status(is_active);
        CREATE INDEX IF NOT EXISTS idx_symbol_status_status
            ON symbol_status(status);
    """)
    conn.commit()
    cursor.close()
    print("Symbol status table ready")


def sync_symbol_status():
    """
    Sync symbol status from Binance API to database.
    - Marks delisted coins as inactive
    - Keeps their historical data
    - Training pipeline can exclude inactive symbols
    """
    if not HAS_PSYCOPG2:
        raise ImportError("psycopg2 required for database operations")

    print("\n" + "=" * 60)
    print("SYNCING SYMBOL STATUS")
    print("=" * 60)

    # Get active symbols from Binance
    active_symbols = get_active_symbols()

    # Connect to DB
    conn = psycopg2.connect(**DB_CONFIG)
    create_symbol_status_table(conn)
    cursor = conn.cursor()

    # Get symbols we have data for
    cursor.execute("""
        SELECT DISTINCT symbol FROM ohlcv WHERE symbol LIKE '%USDT'
    """)
    db_symbols = set(row[0] for row in cursor.fetchall())
    print(f"Symbols in database: {len(db_symbols)}")

    # Get current status from our table
    cursor.execute("SELECT symbol, status, is_active FROM symbol_status")
    current_status = {row[0]: {"status": row[1], "is_active": row[2]} for row in cursor.fetchall()}

    # Update status for all symbols
    updated = 0
    newly_delisted = []
    newly_active = []

    all_symbols = db_symbols | set(active_symbols.keys())

    for symbol in all_symbols:
        binance_info = active_symbols.get(symbol, {})
        is_trading = binance_info.get("is_trading", False)
        status = binance_info.get("status", "DELISTED") if binance_info else "DELISTED"

        prev_status = current_status.get(symbol, {})
        was_active = prev_status.get("is_active", True)

        # Detect status changes
        if was_active and not is_trading and symbol in current_status:
            newly_delisted.append(symbol)
        elif not was_active and is_trading:
            newly_active.append(symbol)

        # Upsert status
        cursor.execute("""
            INSERT INTO symbol_status (symbol, status, is_active, base_asset, quote_asset, last_checked)
            VALUES (%s, %s, %s, %s, %s, NOW())
            ON CONFLICT (symbol) DO UPDATE SET
                status = EXCLUDED.status,
                is_active = EXCLUDED.is_active,
                last_checked = NOW()
        """, (
            symbol,
            status,
            is_trading,
            binance_info.get("base_asset"),
            binance_info.get("quote_asset"),
        ))
        updated += 1

    conn.commit()

    # Summary
    cursor.execute("SELECT COUNT(*) FROM symbol_status WHERE is_active = TRUE")
    active_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM symbol_status WHERE is_active = FALSE")
    inactive_count = cursor.fetchone()[0]

    print(f"\nStatus updated for {updated} symbols")
    print(f"Active: {active_count}, Inactive/Delisted: {inactive_count}")

    if newly_delisted:
        print(f"\n⚠️  NEWLY DELISTED ({len(newly_delisted)}):")
        for s in newly_delisted[:20]:
            print(f"   - {s}")
        if len(newly_delisted) > 20:
            print(f"   ... and {len(newly_delisted) - 20} more")

    if newly_active:
        print(f"\n✅ NEWLY ACTIVE ({len(newly_active)}):")
        for s in newly_active[:10]:
            print(f"   - {s}")

    cursor.close()
    conn.close()

    print("\n" + "=" * 60)
    print("Symbol status sync complete!")
    print("Training pipeline should use: WHERE symbol IN (SELECT symbol FROM symbol_status WHERE is_active = TRUE)")
    print("=" * 60)


def list_delisted_symbols():
    """Show all delisted/inactive symbols in the database."""
    if not HAS_PSYCOPG2:
        raise ImportError("psycopg2 required")

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Check if table exists
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'symbol_status'
        )
    """)
    if not cursor.fetchone()[0]:
        print("Symbol status table doesn't exist. Run --sync-status first.")
        return

    cursor.execute("""
        SELECT s.symbol, s.status, s.last_checked,
               (SELECT COUNT(*) FROM ohlcv WHERE symbol = s.symbol) as row_count
        FROM symbol_status s
        WHERE s.is_active = FALSE
        ORDER BY s.symbol
    """)

    results = cursor.fetchall()

    print("\n" + "=" * 60)
    print(f"DELISTED/INACTIVE SYMBOLS ({len(results)})")
    print("=" * 60)
    print(f"{'Symbol':<15} {'Status':<12} {'Last Checked':<20} {'Rows in DB':>12}")
    print("-" * 60)

    total_rows = 0
    for symbol, status, last_checked, row_count in results:
        print(f"{symbol:<15} {status:<12} {str(last_checked)[:19]:<20} {row_count:>12,}")
        total_rows += row_count or 0

    print("-" * 60)
    print(f"Total historical data rows from delisted coins: {total_rows:,}")
    print("\nThis data is preserved for historical analysis but excluded from training.")

    cursor.close()
    conn.close()


def get_trainable_symbols() -> List[str]:
    """Get list of symbols that are active and can be used for training."""
    if not HAS_PSYCOPG2:
        raise ImportError("psycopg2 required")

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Check if status table exists
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'symbol_status'
        )
    """)

    if cursor.fetchone()[0]:
        # Use status table
        cursor.execute("""
            SELECT symbol FROM symbol_status WHERE is_active = TRUE ORDER BY symbol
        """)
    else:
        # Fall back to all USDT symbols
        cursor.execute("""
            SELECT DISTINCT symbol FROM ohlcv WHERE symbol LIKE '%USDT' ORDER BY symbol
        """)

    symbols = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()

    return symbols


# ============================================================
# DOWNLOAD FUNCTIONS
# ============================================================

def download_direct(
    symbol: str,
    year: int,
    month: int,
    asset_class: str = "spot",
) -> Optional[Path]:
    """Download directly from data.binance.vision."""

    # Construct URL
    url = (
        f"{BINANCE_DATA_URL}/data/{asset_class}/monthly/klines/"
        f"{symbol}/1m/{symbol}-1m-{year}-{month:02d}.zip"
    )

    output_dir = DATA_DIR / asset_class / symbol
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{symbol}-1m-{year}-{month:02d}.csv"

    if output_file.exists():
        return output_file  # Already downloaded

    try:
        response = requests.get(url, timeout=120)
        if response.status_code == 404:
            return None
        response.raise_for_status()

        # Extract CSV from ZIP
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as zf:
                with open(output_file, 'wb') as f:
                    f.write(zf.read())

        return output_file

    except Exception as e:
        safe_print(f"  Error downloading {symbol} {year}-{month:02d}: {e}")
        return None


def download_symbol(
    symbol: str,
    asset_class: str = "spot",
    start_year: int = 2017,
) -> Tuple[str, int]:
    """Download all history for a single symbol. Returns (symbol, file_count)."""
    current_year = datetime.now().year
    current_month = datetime.now().month

    files_downloaded = 0

    for year in range(start_year, current_year + 1):
        end_month = current_month if year == current_year else 12

        for month in range(1, end_month + 1):
            file_path = download_direct(symbol, year, month, asset_class)
            if file_path:
                files_downloaded += 1

    return (symbol, files_downloaded)


def download_all_history(
    symbols: List[str],
    asset_class: str = "spot",
    start_year: int = 2017,
    max_workers: int = 5,
) -> Dict[str, int]:
    """Download all historical data for symbols with parallel downloads."""

    print(f"\n{'=' * 60}")
    print(f"BULK DOWNLOAD - ALL HISTORY")
    print(f"{'=' * 60}")
    print(f"Symbols: {len(symbols)}")
    print(f"Asset class: {asset_class}")
    print(f"Start year: {start_year}")
    print(f"Parallel workers: {max_workers}")
    print(f"Started: {datetime.now()}")
    print("=" * 60 + "\n")

    results = {}
    completed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_symbol, s, asset_class, start_year): s
            for s in symbols
        }

        for future in concurrent.futures.as_completed(futures):
            symbol, file_count = future.result()
            results[symbol] = file_count
            completed += 1

            if file_count > 0:
                safe_print(f"[{completed}/{len(symbols)}] {symbol}: {file_count} files")
            else:
                safe_print(f"[{completed}/{len(symbols)}] {symbol}: no data available")

    total_files = sum(results.values())
    print(f"\n{'=' * 60}")
    print(f"Download complete!")
    print(f"Total files: {total_files}")
    print(f"Finished: {datetime.now()}")
    print("=" * 60)

    return results


def check_available_data():
    """Check what data is available on Binance public data."""
    print("\n" + "=" * 60)
    print("BINANCE PUBLIC DATA - AVAILABLE DATA")
    print("=" * 60)

    # Check active spot symbols
    active_symbols = get_active_symbols()
    trading = [s for s, info in active_symbols.items() if info["is_trading"]]
    print(f"\nActive USDT pairs (TRADING): {len(trading)}")
    print(f"Examples: {trading[:10]}...")

    # Check what's downloaded locally
    if DATA_DIR.exists():
        spot_dir = DATA_DIR / "spot"
        if spot_dir.exists():
            downloaded = list(spot_dir.glob("*"))
            print(f"\nAlready downloaded: {len(downloaded)} symbols")

            total_files = sum(1 for _ in spot_dir.glob("*/*.csv"))
            total_size = sum(f.stat().st_size for f in spot_dir.glob("*/*.csv"))
            print(f"Total CSV files: {total_files}")
            print(f"Total size: {total_size / 1e9:.2f} GB")

    # Check database
    if HAS_PSYCOPG2:
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM ohlcv")
            total_rows = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT symbol) FROM ohlcv")
            db_symbols = cursor.fetchone()[0]

            cursor.execute("SELECT pg_size_pretty(pg_total_relation_size('ohlcv'))")
            db_size = cursor.fetchone()[0]

            print(f"\nDatabase status:")
            print(f"  Rows: {total_rows:,}")
            print(f"  Symbols: {db_symbols}")
            print(f"  Size: {db_size}")

            cursor.close()
            conn.close()
        except Exception as e:
            print(f"\nDatabase check failed: {e}")

    # Estimate download size
    print("\n" + "-" * 60)
    print("ESTIMATED DOWNLOAD SIZE FOR ALL DATA")
    print("-" * 60)
    print(f"~{len(trading)} symbols × ~7 years of 1m data")
    print(f"Estimated: {len(trading) * 7 * 0.075:.0f} GB (compressed downloads)")
    print(f"Disk space needed: ~{len(trading) * 7 * 0.3:.0f} GB (uncompressed CSV)")

    return trading


def get_top_symbols_by_volume(n: int = 50) -> List[str]:
    """Get top N symbols by 24h trading volume."""
    url = f"{BINANCE_API_URL}/api/v3/ticker/24hr"

    print(f"Fetching top {n} symbols by volume...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    tickers = response.json()

    usdt_tickers = [
        t for t in tickers
        if t["symbol"].endswith("USDT") and float(t["quoteVolume"]) > 0
    ]

    sorted_tickers = sorted(
        usdt_tickers,
        key=lambda x: float(x["quoteVolume"]),
        reverse=True
    )

    top_symbols = [t["symbol"] for t in sorted_tickers[:n]]
    print(f"Top {n} by volume: {top_symbols[:10]}...")
    return top_symbols


# ============================================================
# DATABASE IMPORT
# ============================================================

def import_csv_file(csv_file: Path, conn, timeframe: str = "1m") -> int:
    """Import a single CSV file. Returns row count."""
    symbol = csv_file.parent.name

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

        cursor = conn.cursor()

        # Use INSERT with ON CONFLICT to handle duplicates
        insert_sql = """
            INSERT INTO ohlcv (exchange, symbol, timeframe, open_time, open, high, low, close, volume, quote_volume, close_time)
            VALUES %s
            ON CONFLICT (exchange, symbol, timeframe, open_time) DO NOTHING
        """

        values = [
            (
                "binance",
                symbol,
                timeframe,
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
        cursor.close()

        return len(values)

    except Exception as e:
        safe_print(f"Error importing {csv_file}: {e}")
        conn.rollback()
        return 0


def import_to_database(data_dir: Path, timeframe: str = "1m", max_workers: int = 4):
    """Import downloaded CSV data to PostgreSQL with parallel processing."""
    if not HAS_PSYCOPG2 or not HAS_PANDAS:
        raise ImportError("psycopg2 and pandas required for DB import")

    print(f"\n{'=' * 60}")
    print("IMPORTING TO DATABASE")
    print(f"{'=' * 60}")

    csv_files = list(data_dir.glob("*/*.csv"))
    print(f"Found {len(csv_files)} CSV files to import")

    if not csv_files:
        print("No files to import!")
        return

    # Single connection for sequential import (safer for large imports)
    conn = psycopg2.connect(**DB_CONFIG)

    total_rows = 0
    for i, csv_file in enumerate(csv_files, 1):
        symbol = csv_file.parent.name
        print(f"[{i}/{len(csv_files)}] {symbol} - {csv_file.name}...", end=" ", flush=True)

        rows = import_csv_file(csv_file, conn, timeframe)
        total_rows += rows
        print(f"{rows:,} rows")

    conn.close()

    print(f"\n{'=' * 60}")
    print(f"Import complete! Total rows: {total_rows:,}")
    print(f"{'=' * 60}")

    # Auto-sync symbol status after import
    print("\nAuto-syncing symbol status...")
    sync_symbol_status()


def resample_data(input_dir: Path, output_dir: Path, target_timeframe: str):
    """Resample 1m data to target timeframe."""
    if not HAS_PANDAS:
        raise ImportError("pandas required for resampling")

    print(f"\n{'=' * 60}")
    print(f"RESAMPLING TO {target_timeframe}")
    print(f"{'=' * 60}")

    resample_rule = RESAMPLE_TIMEFRAMES.get(target_timeframe)
    if not resample_rule:
        raise ValueError(f"Unknown timeframe: {target_timeframe}")

    csv_files = list(input_dir.glob("*/*.csv"))
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, csv_file in enumerate(csv_files, 1):
        symbol = csv_file.parent.name
        print(f"[{i}/{len(csv_files)}] {symbol}...", end=" ", flush=True)

        try:
            df = pd.read_csv(
                csv_file,
                header=None,
                names=["open_time", "open", "high", "low", "close", "volume",
                       "close_time", "quote_volume", "trades", "taker_buy_base",
                       "taker_buy_quote", "ignore"]
            )

            df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
            df.set_index("datetime", inplace=True)

            resampled = df.resample(resample_rule).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "quote_volume": "sum",
                "trades": "sum",
            }).dropna()

            out_dir = output_dir / symbol
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / csv_file.name.replace("1m", target_timeframe)
            resampled.to_csv(out_file)

            print(f"{len(resampled):,} candles")

        except Exception as e:
            print(f"ERROR: {e}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Binance Bulk Data Downloader + Dead Coin Filter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --check                  # Check what data is available
  %(prog)s --all                    # Download ALL USDT pairs
  %(prog)s --top 100                # Download top 100 by volume
  %(prog)s --import-to-db           # Import downloaded CSVs to PostgreSQL
  %(prog)s --sync-status            # Update coin status (mark delisted)
  %(prog)s --list-delisted          # Show delisted coins in DB
        """
    )

    # Mode flags
    parser.add_argument("--check", action="store_true", help="Check available data")
    parser.add_argument("--sync-status", action="store_true", help="Sync symbol status from Binance (detect delisted)")
    parser.add_argument("--list-delisted", action="store_true", help="List all delisted/inactive symbols")
    parser.add_argument("--import-to-db", action="store_true", help="Import downloaded data to database")
    parser.add_argument("--resample", type=str, help="Resample to timeframe (e.g., 1h, 4h)")

    # Download options
    parser.add_argument("--symbols", nargs="+", help="Specific symbols (without USDT)")
    parser.add_argument("--top", type=int, default=0, help="Download top N symbols by volume")
    parser.add_argument("--all", action="store_true", help="Download ALL symbols (EVERYTHING)")
    parser.add_argument("--futures", action="store_true", help="Download futures (um) instead of spot")
    parser.add_argument("--start-year", type=int, default=2017, help="Start year for download (default: 2017)")
    parser.add_argument("--workers", type=int, default=5, help="Parallel download workers (default: 5)")

    args = parser.parse_args()

    asset_class = "um" if args.futures else "spot"

    # Handle mode flags
    if args.check:
        check_available_data()
        return

    if args.sync_status:
        sync_symbol_status()
        return

    if args.list_delisted:
        list_delisted_symbols()
        return

    if args.import_to_db:
        data_dir = DATA_DIR / asset_class
        import_to_database(data_dir)
        return

    if args.resample:
        input_dir = DATA_DIR / asset_class
        output_dir = DATA_DIR / f"{asset_class}_{args.resample}"
        resample_data(input_dir, output_dir, args.resample)
        return

    # Download mode
    if args.symbols:
        symbols = [s.upper() + "USDT" if not s.endswith("USDT") else s.upper() for s in args.symbols]
    elif args.top > 0:
        symbols = get_top_symbols_by_volume(args.top)
    elif args.all:
        symbols = get_all_usdt_symbols()
        print(f"\n🚀 DOWNLOADING EVERYTHING - {len(symbols)} symbols")
        print("This will take a while. Make sure you have enough disk space!")
        print("-" * 60)
    else:
        # Default: show help
        parser.print_help()
        print("\n💡 Tip: Use --all to download everything, or --top N for top symbols")
        return

    print(f"\nWill download {len(symbols)} symbols")
    print(f"Symbols: {symbols[:20]}{'...' if len(symbols) > 20 else ''}")

    # Download
    download_all_history(
        symbols,
        asset_class=asset_class,
        start_year=args.start_year,
        max_workers=args.workers
    )

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("  1. python bulk_download_binance.py --import-to-db    # Import to PostgreSQL")
    print("  2. python bulk_download_binance.py --sync-status     # Mark delisted coins")
    print("  3. Training will auto-exclude delisted coins")


if __name__ == "__main__":
    main()

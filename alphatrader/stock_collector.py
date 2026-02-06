#!/usr/bin/env python3
"""
Stock Data Collector - Yahoo Finance daily data collection.

Usage:
    python stock_collector.py                       # Update all tracked stocks
    python stock_collector.py --daemon              # Run continuously
    python stock_collector.py --add AAPL MSFT NVDA  # Add symbols to track
    python stock_collector.py --sp500               # Add S&P 500 constituents
    python stock_collector.py --list                # List tracked symbols
    python stock_collector.py --backfill --years 10 # Backfill 10 years
"""

import argparse
import os
import sys
import time
import signal
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

try:
    import yfinance as yf
except ImportError:
    print("Error: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

load_dotenv()

# Database config
DB_CONFIG = {
    "host": os.getenv("ZENKAI_DB_HOST", "192.168.0.160"),
    "database": os.getenv("ZENKAI_DB_NAME", "zenkai_data"),
    "user": os.getenv("ZENKAI_DB_USER", "zenkai"),
    "password": os.getenv("ZENKAI_DB_PASSWORD"),
}

# Default stock universe
SP500_SAMPLE = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM",
    # Finance
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP", "V", "MA",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "LLY", "BMY",
    # Consumer
    "WMT", "PG", "KO", "PEP", "COST", "HD", "MCD", "NKE", "SBUX", "TGT",
    # Industrial
    "CAT", "BA", "GE", "MMM", "HON", "UPS", "RTX", "LMT", "DE", "UNP",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "PXD", "MPC", "VLO", "PSX", "OXY",
    # ETFs
    "SPY", "QQQ", "DIA", "IWM", "VTI", "VOO", "XLF", "XLE", "XLK", "XLV",
]

# Global flag for graceful shutdown
running = True


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global running
    print(f"\nReceived signal {signum}, shutting down gracefully...")
    running = False


def get_db_connection():
    """Get PostgreSQL connection."""
    return psycopg2.connect(**DB_CONFIG)


def get_tracked_symbols() -> List[str]:
    """Get list of stock symbols already in database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT DISTINCT symbol FROM ohlcv
        WHERE asset_type IN ('stock', 'etf') AND exchange = 'us_stock'
        ORDER BY symbol
    """)
    symbols = [row[0] for row in cursor.fetchall()]
    conn.close()
    return symbols


def get_latest_date(symbol: str) -> Optional[datetime]:
    """Get the latest date for a symbol in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT MAX(open_time) FROM ohlcv
        WHERE symbol = %s AND exchange = 'us_stock'
    """, (symbol,))
    result = cursor.fetchone()[0]
    conn.close()

    if result:
        return datetime.fromtimestamp(result / 1000)
    return None


def fetch_yahoo_data(
    symbol: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    period: str = "1y",
) -> Optional[dict]:
    """Fetch OHLCV data from Yahoo Finance."""
    try:
        ticker = yf.Ticker(symbol)

        if start_date and end_date:
            df = ticker.history(start=start_date, end=end_date)
        else:
            df = ticker.history(period=period)

        if df.empty:
            return None

        # Get ticker info for asset type
        info = ticker.info
        asset_type = "etf" if info.get("quoteType") == "ETF" else "stock"

        return {
            "data": df,
            "asset_type": asset_type,
        }

    except Exception as e:
        print(f"  Error fetching {symbol}: {e}")
        return None


def insert_yahoo_data(symbol: str, asset_type: str, df) -> int:
    """Insert Yahoo Finance data into database."""
    if df.empty:
        return 0

    conn = get_db_connection()
    cursor = conn.cursor()

    rows = []
    for date, row in df.iterrows():
        # Convert to timestamp (milliseconds)
        open_time = int(date.timestamp() * 1000)
        close_time = open_time + 86400000 - 1

        rows.append((
            "us_stock",         # exchange
            symbol,             # symbol
            "1d",               # timeframe
            open_time,          # open_time
            float(row["Open"]),
            float(row["High"]),
            float(row["Low"]),
            float(row["Close"]),
            float(row["Volume"]),
            close_time,         # close_time
            asset_type,         # asset_type
        ))

    if not rows:
        conn.close()
        return 0

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
            volume = EXCLUDED.volume
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


def collect_symbol(
    symbol: str,
    backfill_years: int = 0,
) -> int:
    """Collect data for a single symbol. Returns rows inserted."""
    # Get latest date in database
    latest = get_latest_date(symbol)

    if backfill_years > 0:
        # Backfill mode
        start_date = datetime.now() - timedelta(days=backfill_years * 365)
        end_date = datetime.now()
        result = fetch_yahoo_data(symbol, start_date, end_date)
    elif latest:
        # Continue from last known date
        start_date = latest + timedelta(days=1)
        end_date = datetime.now()

        if start_date >= end_date:
            return 0  # Already up to date

        result = fetch_yahoo_data(symbol, start_date, end_date)
    else:
        # New symbol - fetch 1 year by default
        result = fetch_yahoo_data(symbol, period="1y")

    if not result:
        return 0

    return insert_yahoo_data(symbol, result["asset_type"], result["data"])


def collect_all(
    symbols: List[str],
    backfill_years: int = 0,
    rate_limit: float = 0.5,
) -> Dict[str, int]:
    """Collect data for all symbols."""
    stats = {"total": 0, "symbols": 0, "errors": 0}

    for symbol in symbols:
        if not running:
            break

        try:
            inserted = collect_symbol(symbol, backfill_years)
            if inserted > 0:
                stats["symbols"] += 1
                stats["total"] += inserted
                print(f"  {symbol}: {inserted} rows")
        except Exception as e:
            stats["errors"] += 1
            print(f"  {symbol}: Error - {e}")

        time.sleep(rate_limit)

    return stats


def fetch_sp500_constituents() -> List[str]:
    """Fetch current S&P 500 constituents from Wikipedia."""
    try:
        import pandas as pd
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        symbols = df["Symbol"].tolist()
        # Clean up symbols (some have dots that need to be dashes for Yahoo)
        symbols = [s.replace(".", "-") for s in symbols]
        return symbols
    except Exception as e:
        print(f"Error fetching S&P 500 list: {e}")
        print("Using default sample list instead...")
        return SP500_SAMPLE


def run_daemon(
    symbols: List[str],
    rate_limit: float = 0.5,
):
    """Run as continuous daemon."""
    global running

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 60)
    print("Stock Data Collector (Daemon Mode)")
    print("=" * 60)
    print(f"Started: {datetime.now()}")
    print(f"Tracking {len(symbols)} symbols")
    print("Press Ctrl+C to stop")
    print()

    cycle = 1

    while running:
        cycle_start = datetime.now()
        print(f"\n[Cycle {cycle}] {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}")

        stats = collect_all(symbols, rate_limit=rate_limit)

        print(f"  Collected {stats['total']:,} rows from {stats['symbols']} symbols")

        # Wait until next day (stocks are daily)
        now = datetime.now()
        # Target 6 AM next day (after market close + buffer)
        next_run = (now + timedelta(days=1)).replace(hour=6, minute=0, second=0)
        sleep_seconds = (next_run - now).total_seconds()

        print(f"  Next run: {next_run.strftime('%Y-%m-%d %H:%M')}")

        # Interruptible sleep
        for _ in range(int(min(sleep_seconds, 3600))):  # Check every hour max
            if not running:
                break
            time.sleep(1)

        cycle += 1

    print("\nDaemon stopped.")


def main():
    parser = argparse.ArgumentParser(description="Stock Data Collector (Yahoo Finance)")
    parser.add_argument(
        "--daemon", "-d",
        action="store_true",
        help="Run as continuous daemon"
    )
    parser.add_argument(
        "--add",
        nargs="+",
        help="Add symbols to collect"
    )
    parser.add_argument(
        "--sp500",
        action="store_true",
        help="Collect S&P 500 constituents"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List tracked symbols"
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Backfill historical data"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=10,
        help="Years to backfill (default: 10)"
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.5,
        help="Seconds between API calls (default: 0.5)"
    )

    args = parser.parse_args()

    if args.list:
        symbols = get_tracked_symbols()
        print(f"Tracked symbols ({len(symbols)}):")
        for i, s in enumerate(symbols):
            print(f"  {s}", end="")
            if (i + 1) % 10 == 0:
                print()
        print()
        return

    # Determine symbols to collect
    if args.add:
        symbols = [s.upper() for s in args.add]
    elif args.sp500:
        print("Fetching S&P 500 constituents...")
        symbols = fetch_sp500_constituents()
        print(f"Found {len(symbols)} symbols")
    else:
        # Use existing tracked symbols or default list
        symbols = get_tracked_symbols()
        if not symbols:
            print("No tracked symbols. Using default sample...")
            symbols = SP500_SAMPLE

    print("=" * 60)
    print("Stock Data Collector")
    print("=" * 60)
    print(f"Started: {datetime.now()}")
    print(f"Symbols: {len(symbols)}")

    if args.backfill:
        print(f"Backfill: {args.years} years")

    print()

    if args.daemon:
        run_daemon(symbols, args.rate_limit)
    else:
        backfill_years = args.years if args.backfill else 0
        stats = collect_all(symbols, backfill_years, args.rate_limit)

        print(f"\nDone! Collected {stats['total']:,} rows from {stats['symbols']} symbols")
        if stats["errors"] > 0:
            print(f"Errors: {stats['errors']}")
        print(f"Finished: {datetime.now()}")


if __name__ == "__main__":
    main()

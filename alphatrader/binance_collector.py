#!/usr/bin/env python3
"""
Binance 24/7 Data Collector - Continuous OHLCV data collection daemon.

Usage:
    python binance_collector.py                      # Run once (fetch latest)
    python binance_collector.py --daemon             # Run continuously (24/7)
    python binance_collector.py --timeframe 1h       # Specific timeframe only
    python binance_collector.py --backfill --days 30 # Backfill last 30 days
"""

import argparse
import os
import time
import signal
import sys
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import requests
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

# Database config
DB_CONFIG = {
    "host": os.getenv("ZENKAI_DB_HOST", "192.168.0.160"),
    "database": os.getenv("ZENKAI_DB_NAME", "zenkai_data"),
    "user": os.getenv("ZENKAI_DB_USER", "zenkai"),
    "password": os.getenv("ZENKAI_DB_PASSWORD"),
}

# Binance API
BINANCE_SPOT_URL = "https://api.binance.com"

# Timeframe configs (interval, seconds between candles)
TIMEFRAMES = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
    "3d": 259200,
    "1w": 604800,
    "1M": 2592000,
}

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


def get_usdt_pairs() -> List[str]:
    """Fetch all USDT trading pairs from Binance."""
    url = f"{BINANCE_SPOT_URL}/api/v3/exchangeInfo"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    symbols = []
    for s in data["symbols"]:
        if s.get("status") == "TRADING" and s["symbol"].endswith("USDT"):
            symbols.append(s["symbol"])

    return sorted(symbols)


def fetch_klines(
    symbol: str,
    timeframe: str,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: int = 1000,
) -> List[List]:
    """Fetch klines from Binance API."""
    url = f"{BINANCE_SPOT_URL}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": timeframe,
        "limit": limit,
    }
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time

    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def get_latest_timestamp(symbol: str, timeframe: str) -> Optional[int]:
    """Get the latest timestamp for a symbol/timeframe in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT MAX(open_time) FROM ohlcv
        WHERE symbol = %s AND timeframe = %s AND exchange = 'binance'
        """,
        (symbol, timeframe),
    )
    result = cursor.fetchone()[0]
    conn.close()
    return result


def insert_klines(symbol: str, timeframe: str, klines: List[List]) -> int:
    """Insert klines into database. Returns count of inserted rows."""
    if not klines:
        return 0

    conn = get_db_connection()
    cursor = conn.cursor()

    # Prepare data
    rows = []
    for k in klines:
        rows.append((
            "binance",              # exchange
            symbol,                 # symbol
            timeframe,              # timeframe
            int(k[0]),              # open_time
            float(k[1]),            # open
            float(k[2]),            # high
            float(k[3]),            # low
            float(k[4]),            # close
            float(k[5]),            # volume
            int(k[6]),              # close_time
            float(k[7]),            # quote_volume
            int(k[8]),              # trades
            float(k[9]),            # taker_buy_volume
            float(k[10]),           # taker_buy_quote_volume
        ))

    # Upsert (insert or update on conflict)
    query = """
        INSERT INTO ohlcv (
            exchange, symbol, timeframe, open_time,
            open, high, low, close, volume,
            close_time, quote_volume, trades,
            taker_buy_volume, taker_buy_quote_volume
        ) VALUES %s
        ON CONFLICT (exchange, symbol, timeframe, open_time)
        DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            close_time = EXCLUDED.close_time,
            quote_volume = EXCLUDED.quote_volume,
            trades = EXCLUDED.trades,
            taker_buy_volume = EXCLUDED.taker_buy_volume,
            taker_buy_quote_volume = EXCLUDED.taker_buy_quote_volume
    """

    try:
        execute_values(cursor, query, rows)
        conn.commit()
        inserted = cursor.rowcount
    except Exception as e:
        print(f"  Error inserting {symbol}/{timeframe}: {e}")
        conn.rollback()
        inserted = 0
    finally:
        conn.close()

    return inserted


def collect_symbol(
    symbol: str,
    timeframe: str,
    backfill_days: int = 0,
) -> int:
    """Collect data for a single symbol/timeframe. Returns rows inserted."""
    try:
        # Get latest timestamp in database
        latest = get_latest_timestamp(symbol, timeframe)

        if backfill_days > 0:
            # Backfill mode - fetch from X days ago
            start_time = int((datetime.now() - timedelta(days=backfill_days)).timestamp() * 1000)
        elif latest:
            # Continue from last known timestamp
            start_time = latest + 1  # +1ms to avoid duplicate
        else:
            # No data yet - fetch last 7 days as initial seed
            start_time = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)

        end_time = int(datetime.now().timestamp() * 1000)

        # Fetch klines
        klines = fetch_klines(symbol, timeframe, start_time, end_time)

        if not klines:
            return 0

        # Insert to database
        inserted = insert_klines(symbol, timeframe, klines)
        return inserted

    except Exception as e:
        print(f"  Error collecting {symbol}/{timeframe}: {e}")
        return 0


def collect_all(
    symbols: List[str],
    timeframes: List[str],
    backfill_days: int = 0,
    rate_limit: float = 0.1,
) -> Dict[str, int]:
    """Collect data for all symbols and timeframes."""
    stats = {"total": 0, "symbols": 0, "errors": 0}

    for symbol in symbols:
        symbol_total = 0
        for tf in timeframes:
            if not running:
                return stats

            inserted = collect_symbol(symbol, tf, backfill_days)
            symbol_total += inserted

            # Rate limiting
            time.sleep(rate_limit)

        if symbol_total > 0:
            stats["symbols"] += 1
            stats["total"] += symbol_total

    return stats


def run_daemon(
    timeframes: Optional[List[str]] = None,
    rate_limit: float = 0.1,
):
    """Run as continuous daemon."""
    global running

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if timeframes is None:
        timeframes = list(TIMEFRAMES.keys())

    print("=" * 60)
    print("Binance 24/7 Data Collector")
    print("=" * 60)
    print(f"Started: {datetime.now()}")
    print(f"Timeframes: {', '.join(timeframes)}")
    print("Press Ctrl+C to stop")
    print()

    # Initial collection - get all symbols
    print("Fetching symbol list...")
    symbols = get_usdt_pairs()
    print(f"Found {len(symbols)} USDT pairs")

    # Initial full collection
    print("\nRunning initial collection...")
    stats = collect_all(symbols, timeframes, rate_limit=rate_limit)
    print(f"Initial collection: {stats['total']:,} rows from {stats['symbols']} symbols")

    # Continuous loop
    last_refresh = datetime.now()
    cycle = 1

    while running:
        # Refresh symbol list every hour
        if (datetime.now() - last_refresh).total_seconds() > 3600:
            print("\nRefreshing symbol list...")
            symbols = get_usdt_pairs()
            print(f"Updated to {len(symbols)} USDT pairs")
            last_refresh = datetime.now()

        # Collect latest data
        cycle_start = datetime.now()
        print(f"\n[Cycle {cycle}] {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}")

        stats = collect_all(symbols, timeframes, rate_limit=rate_limit)

        cycle_time = (datetime.now() - cycle_start).total_seconds()
        print(f"  Collected {stats['total']:,} rows in {cycle_time:.1f}s")

        cycle += 1

        # Wait before next cycle (minimum 60 seconds to avoid hammering API)
        sleep_time = max(60, 300 - cycle_time)  # Target 5 min cycles
        print(f"  Sleeping {sleep_time:.0f}s until next cycle...")

        # Interruptible sleep
        for _ in range(int(sleep_time)):
            if not running:
                break
            time.sleep(1)

    print("\nDaemon stopped.")


def run_once(
    timeframes: Optional[List[str]] = None,
    backfill_days: int = 0,
    rate_limit: float = 0.1,
):
    """Run once and exit."""
    if timeframes is None:
        timeframes = list(TIMEFRAMES.keys())

    print("=" * 60)
    print("Binance Data Collector (Single Run)")
    print("=" * 60)
    print(f"Started: {datetime.now()}")
    print(f"Timeframes: {', '.join(timeframes)}")
    if backfill_days > 0:
        print(f"Backfill: {backfill_days} days")
    print()

    # Get symbols
    print("Fetching symbol list...")
    symbols = get_usdt_pairs()
    print(f"Found {len(symbols)} USDT pairs")

    # Collect
    print("\nCollecting data...")
    stats = collect_all(symbols, timeframes, backfill_days, rate_limit)

    print(f"\nDone! Collected {stats['total']:,} rows from {stats['symbols']} symbols")
    print(f"Finished: {datetime.now()}")


def main():
    parser = argparse.ArgumentParser(description="Binance 24/7 Data Collector")
    parser.add_argument(
        "--daemon", "-d",
        action="store_true",
        help="Run as continuous daemon (24/7)"
    )
    parser.add_argument(
        "--timeframe", "-tf",
        type=str,
        help="Specific timeframe only (e.g., 1h, 4h)"
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Backfill historical data"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days to backfill (default: 30)"
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.1,
        help="Seconds between API calls (default: 0.1)"
    )

    args = parser.parse_args()

    # Parse timeframes
    timeframes = None
    if args.timeframe:
        timeframes = [args.timeframe]

    if args.daemon:
        run_daemon(timeframes, args.rate_limit)
    else:
        backfill_days = args.days if args.backfill else 0
        run_once(timeframes, backfill_days, args.rate_limit)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Yahoo Finance Intraday Data Collector.

Fetches intraday data (1m, 5m, 15m, 30m, 1h) for top US stocks.
Limited history: 1m=7days, 5m/15m/30m=60days, 1h=730days

Usage:
    python yfinance_intraday_collector.py              # Collect all
    python yfinance_intraday_collector.py --top 100    # Top 100 stocks
    python yfinance_intraday_collector.py --status     # Show progress
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import yfinance as yf
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# Database config
DB_CONFIG = {
    "host": os.getenv("ZENKAI_DB_HOST", "192.168.0.160"),
    "database": os.getenv("ZENKAI_DB_NAME", "zenkai_data"),
    "user": os.getenv("ZENKAI_DB_USER", "zenkai"),
    "password": os.getenv("ZENKAI_DB_PASSWORD"),
}

# Top US stocks by market cap + popular ETFs
TOP_STOCKS = [
    # Mega caps
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH",
    "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
    "LLY", "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO", "TMO", "ABT",
    "ACN", "DHR", "ADBE", "CRM", "NKE", "CMCSA", "PFE", "VZ", "INTC", "TXN",
    "PM", "NFLX", "WFC", "BMY", "UPS", "RTX", "QCOM", "NEE", "HON", "UNP",

    # Tech growth
    "AMD", "CRM", "NOW", "INTU", "PYPL", "SQ", "SHOP", "SNOW", "PLTR", "COIN",
    "UBER", "ABNB", "RBLX", "RIVN", "LCID", "NIO", "XPEV", "LI", "DKNG", "PENN",

    # Popular ETFs
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VXX", "ARKK", "XLF", "XLE",
    "XLK", "XLV", "XLI", "XLY", "XLP", "GLD", "SLV", "USO", "TLT", "HYG",

    # Meme stocks / high volume
    "GME", "AMC", "BBBY", "BB", "NOK", "SOFI", "HOOD", "WISH", "CLOV", "SPCE",

    # More large caps
    "DIS", "ORCL", "IBM", "GE", "CAT", "DE", "BA", "LMT", "GS", "MS",
    "AXP", "BLK", "SCHW", "C", "USB", "PNC", "TFC", "AIG", "MET", "PRU",
    "T", "TMUS", "CCI", "AMT", "EQIX", "DLR", "PSA", "O", "WELL", "AVB",

    # Healthcare / Biotech
    "MRNA", "BNTX", "REGN", "VRTX", "GILD", "BIIB", "AMGN", "ISRG", "MDT", "SYK",
    "ZTS", "CI", "HUM", "ELV", "MCK", "CAH", "ABC", "CVS", "WBA", "RAD",

    # Semiconductors
    "TSM", "ASML", "LRCX", "KLAC", "AMAT", "MU", "MRVL", "ON", "SWKS", "QRVO",

    # Energy
    "OXY", "COP", "SLB", "EOG", "PXD", "DVN", "MPC", "VLO", "PSX", "HES",

    # Consumer
    "SBUX", "CMG", "YUM", "DPZ", "MCD", "QSR", "WING", "CAVA", "SHAK", "DRI",
    "TGT", "DLTR", "DG", "ROST", "TJX", "BURL", "GPS", "ANF", "AEO", "LULU",

    # Industrials
    "FDX", "DAL", "UAL", "AAL", "LUV", "JBLU", "ALK", "SAVE", "HA", "SKYW",

    # Financials
    "KKR", "APO", "BX", "CG", "ARES", "OWL", "TROW", "IVZ", "BEN", "NTRS",
]

# Timeframe configs: (yf_interval, yf_period, our_timeframe)
TIMEFRAMES = [
    ("1m", "7d", "1m"),       # 1 minute - max 7 days
    ("5m", "60d", "5m"),      # 5 minute - max 60 days
    ("15m", "60d", "15m"),    # 15 minute - max 60 days
    ("30m", "60d", "30m"),    # 30 minute - max 60 days
    ("1h", "730d", "1h"),     # 1 hour - max 730 days (2 years)
]


def get_db_connection():
    """Get PostgreSQL connection."""
    return psycopg2.connect(**DB_CONFIG)


def get_existing_data_range(conn, symbol: str, timeframe: str):
    """Get the date range of existing data for a symbol/timeframe."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT MIN(open_time), MAX(open_time)
        FROM ohlcv
        WHERE symbol = %s AND timeframe = %s AND exchange = 'us_stock'
    """, (symbol, timeframe))
    row = cursor.fetchone()
    if row and row[0]:
        return row[0], row[1]
    return None, None


def fetch_stock_data(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Fetch data from Yahoo Finance."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]

        # Rename columns to match our schema
        if 'datetime' in df.columns:
            df['timestamp'] = pd.to_datetime(df['datetime'])
        elif 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
        else:
            return pd.DataFrame()

        df['open_time'] = df['timestamp'].astype('int64') // 10**6  # ms timestamp

        return df[['open_time', 'timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()

    except Exception as e:
        print(f"  Error fetching {symbol}: {e}")
        return pd.DataFrame()


def get_interval_ms(timeframe: str) -> int:
    """Get interval duration in milliseconds."""
    intervals = {
        "1m": 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
    }
    return intervals.get(timeframe, 60 * 1000)


def insert_data(conn, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
    """Insert data into database, skipping duplicates."""
    if df.empty:
        return 0

    cursor = conn.cursor()
    interval_ms = get_interval_ms(timeframe)

    # Prepare data
    rows = []
    for _, row in df.iterrows():
        open_time = int(row['open_time'])
        close_time = open_time + interval_ms - 1
        rows.append((
            'us_stock',
            symbol,
            timeframe,
            open_time,
            close_time,
            float(row['open']),
            float(row['high']),
            float(row['low']),
            float(row['close']),
            float(row['volume']) if pd.notna(row['volume']) else 0,
        ))

    # Insert with ON CONFLICT DO NOTHING
    query = """
        INSERT INTO ohlcv (exchange, symbol, timeframe, open_time, close_time, open, high, low, close, volume)
        VALUES %s
        ON CONFLICT (exchange, symbol, timeframe, open_time) DO NOTHING
    """

    execute_values(cursor, query, rows, page_size=1000)
    inserted = cursor.rowcount
    conn.commit()

    return inserted


def collect_symbol(symbol: str, timeframes: list = None) -> dict:
    """Collect all timeframes for a symbol."""
    if timeframes is None:
        timeframes = TIMEFRAMES

    conn = get_db_connection()
    results = {}

    for yf_interval, yf_period, our_tf in timeframes:
        df = fetch_stock_data(symbol, yf_interval, yf_period)

        if not df.empty:
            inserted = insert_data(conn, symbol, our_tf, df)
            results[our_tf] = {"fetched": len(df), "inserted": inserted}
        else:
            results[our_tf] = {"fetched": 0, "inserted": 0}

        time.sleep(0.5)  # Rate limit

    conn.close()
    return results


def collect_all(top_n: int = None, resume_from: str = None):
    """Collect data for all stocks."""
    stocks = TOP_STOCKS[:top_n] if top_n else TOP_STOCKS

    # Resume support
    start_idx = 0
    if resume_from:
        try:
            start_idx = stocks.index(resume_from)
            print(f"Resuming from {resume_from} (index {start_idx})")
        except ValueError:
            pass

    print("=" * 70)
    print("Yahoo Finance Intraday Collector")
    print("=" * 70)
    print(f"Stocks: {len(stocks)}")
    print(f"Timeframes: 1m (7d), 5m/15m/30m (60d), 1h (2yr)")
    print(f"Started: {datetime.now()}")
    print("-" * 70)

    total_inserted = 0

    for i, symbol in enumerate(stocks[start_idx:], start=start_idx + 1):
        print(f"\n[{i}/{len(stocks)}] {symbol}...")

        try:
            results = collect_symbol(symbol)

            for tf, data in results.items():
                if data['inserted'] > 0:
                    print(f"  {tf}: +{data['inserted']} rows")
                    total_inserted += data['inserted']

        except Exception as e:
            print(f"  ERROR: {e}")

        # Progress save point
        if i % 10 == 0:
            print(f"\n--- Progress: {i}/{len(stocks)} symbols, {total_inserted:,} rows inserted ---\n")

    print("\n" + "=" * 70)
    print(f"Complete! Inserted {total_inserted:,} rows")
    print("=" * 70)


def show_status():
    """Show current intraday data status."""
    conn = get_db_connection()
    cursor = conn.cursor()

    print("=" * 70)
    print("Yahoo Finance Intraday Data Status")
    print("=" * 70)

    # Count by timeframe for us_stock
    cursor.execute("""
        SELECT timeframe, COUNT(*) as rows, COUNT(DISTINCT symbol) as symbols
        FROM ohlcv
        WHERE exchange = 'us_stock'
        GROUP BY timeframe
        ORDER BY rows DESC
    """)

    print("\nUS Stock Data by Timeframe:")
    print("-" * 40)
    total = 0
    for row in cursor.fetchall():
        print(f"  {row[0]:8s}: {row[1]:>12,} rows ({row[2]:>5} symbols)")
        total += row[1]
    print("-" * 40)
    print(f"  {'TOTAL':8s}: {total:>12,} rows")

    # Date range for intraday
    cursor.execute("""
        SELECT timeframe,
               MIN(to_timestamp(open_time/1000)) as earliest,
               MAX(to_timestamp(open_time/1000)) as latest
        FROM ohlcv
        WHERE exchange = 'us_stock' AND timeframe != '1d'
        GROUP BY timeframe
        ORDER BY timeframe
    """)

    print("\nIntraday Date Ranges:")
    print("-" * 60)
    for row in cursor.fetchall():
        if row[1]:
            print(f"  {row[0]:8s}: {row[1].strftime('%Y-%m-%d')} to {row[2].strftime('%Y-%m-%d')}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Yahoo Finance Intraday Collector")
    parser.add_argument("--top", type=int, help="Only collect top N stocks")
    parser.add_argument("--resume", type=str, help="Resume from symbol")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--symbol", type=str, help="Collect single symbol")

    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.symbol:
        print(f"Collecting {args.symbol}...")
        results = collect_symbol(args.symbol)
        for tf, data in results.items():
            print(f"  {tf}: fetched={data['fetched']}, inserted={data['inserted']}")
    else:
        collect_all(top_n=args.top, resume_from=args.resume)


if __name__ == "__main__":
    main()

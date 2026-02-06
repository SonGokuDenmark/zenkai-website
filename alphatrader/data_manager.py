#!/usr/bin/env python3
"""
AlphaTrader Data Manager - Unified data collection from all sources.

Handles:
- Binance (crypto) - 24/7 continuous
- Yahoo Finance (stocks) - Daily updates
- Kaggle (historical dumps) - One-time import
- Future: Bybit, OKX, Polygon, etc.

Usage:
    python data_manager.py --setup           # First-time setup (download + import all)
    python data_manager.py --daemon          # Run all collectors 24/7
    python data_manager.py --status          # Show data statistics
    python data_manager.py --add-source xyz  # Add new data source
"""

import argparse
import os
import sys
import time
import signal
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from abc import ABC, abstractmethod

# Install dependencies if needed
def install_if_missing(package, pip_name=None):
    try:
        __import__(package)
    except ImportError:
        os.system(f"{sys.executable} -m pip install {pip_name or package}")

install_if_missing("kagglehub")
install_if_missing("yfinance")
install_if_missing("requests")
install_if_missing("pandas")
install_if_missing("psycopg2", "psycopg2-binary")
install_if_missing("tqdm")

import kagglehub
import yfinance as yf
import requests
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Database config
DB_CONFIG = {
    "host": os.getenv("ZENKAI_DB_HOST", "192.168.0.160"),
    "database": os.getenv("ZENKAI_DB_NAME", "zenkai_data"),
    "user": os.getenv("ZENKAI_DB_USER", "zenkai"),
    "password": os.getenv("ZENKAI_DB_PASSWORD"),
}

# Global shutdown flag
running = True


def signal_handler(signum, frame):
    global running
    print(f"\nShutdown signal received...")
    running = False


def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


# =============================================================================
# Base Data Source Class
# =============================================================================

class DataSource(ABC):
    """Base class for all data sources."""

    name: str = "base"
    exchange: str = "unknown"
    asset_type: str = "unknown"

    @abstractmethod
    def setup(self) -> dict:
        """One-time setup (download historical data, etc). Returns stats."""
        pass

    @abstractmethod
    def collect(self) -> dict:
        """Collect latest data. Returns stats."""
        pass

    @abstractmethod
    def get_symbols(self) -> List[str]:
        """Get list of symbols this source provides."""
        pass

    def insert_ohlcv(self, symbol: str, df: pd.DataFrame, timeframe: str = "1d") -> int:
        """Insert OHLCV data to database. Returns rows inserted."""
        if df.empty:
            return 0

        conn = get_db_connection()
        cursor = conn.cursor()

        rows = []
        for idx, row in df.iterrows():
            # Handle both datetime index and date column
            if isinstance(idx, pd.Timestamp):
                date = idx
            else:
                date = pd.to_datetime(row.get("date", row.get("Date", idx)))

            open_time = int(date.timestamp() * 1000)
            close_time = open_time + (86400000 if timeframe == "1d" else 60000) - 1

            rows.append((
                self.exchange, symbol, timeframe, open_time,
                float(row.get("Open", row.get("open", 0))),
                float(row.get("High", row.get("high", 0))),
                float(row.get("Low", row.get("low", 0))),
                float(row.get("Close", row.get("close", row.get("Adj Close", 0)))),
                float(row.get("Volume", row.get("volume", 0))),
                close_time, self.asset_type
            ))

        if not rows:
            conn.close()
            return 0

        try:
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
            inserted = len(rows)
        except Exception as e:
            print(f"    Error inserting {symbol}: {e}")
            conn.rollback()
            inserted = 0
        finally:
            conn.close()

        return inserted


# =============================================================================
# Binance Data Source
# =============================================================================

class BinanceSource(DataSource):
    """Binance crypto data - 24/7 continuous collection."""

    name = "Binance"
    exchange = "binance"
    asset_type = "crypto"

    TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    BASE_URL = "https://api.binance.com"

    def __init__(self, timeframes: List[str] = None):
        self.timeframes = timeframes or ["15m", "1h", "4h", "1d"]
        self._symbols = None

    def get_symbols(self) -> List[str]:
        if self._symbols is None:
            url = f"{self.BASE_URL}/api/v3/exchangeInfo"
            response = requests.get(url)
            data = response.json()
            self._symbols = [
                s["symbol"] for s in data["symbols"]
                if s.get("status") == "TRADING" and s["symbol"].endswith("USDT")
            ]
        return self._symbols

    def setup(self) -> dict:
        """Backfill last 30 days of data."""
        print(f"\n[{self.name}] Setting up - backfilling 30 days...")
        return self._collect_all(backfill_days=30)

    def collect(self) -> dict:
        """Collect latest candles."""
        return self._collect_all(backfill_days=0)

    def _collect_all(self, backfill_days: int = 0) -> dict:
        stats = {"symbols": 0, "rows": 0, "errors": 0}
        symbols = self.get_symbols()

        for symbol in symbols:
            if not running:
                break

            for tf in self.timeframes:
                try:
                    rows = self._fetch_and_insert(symbol, tf, backfill_days)
                    stats["rows"] += rows
                except Exception as e:
                    stats["errors"] += 1

                time.sleep(0.1)  # Rate limit

            stats["symbols"] += 1

        return stats

    def _fetch_and_insert(self, symbol: str, timeframe: str, backfill_days: int) -> int:
        # Get latest timestamp in DB
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT MAX(open_time) FROM ohlcv
            WHERE symbol = %s AND timeframe = %s AND exchange = 'binance'
        """, (symbol, timeframe))
        latest = cursor.fetchone()[0]
        conn.close()

        # Calculate start time
        if backfill_days > 0:
            start_time = int((datetime.now() - timedelta(days=backfill_days)).timestamp() * 1000)
        elif latest:
            start_time = latest + 1
        else:
            start_time = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)

        # Fetch from Binance
        url = f"{self.BASE_URL}/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": timeframe,
            "startTime": start_time,
            "limit": 1000
        }

        response = requests.get(url, params=params)
        klines = response.json()

        if not klines or isinstance(klines, dict):
            return 0

        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        df["date"] = pd.to_datetime(df["open_time"], unit="ms")
        df = df.set_index("date")

        return self.insert_ohlcv(symbol, df, timeframe)


# =============================================================================
# Yahoo Finance Data Source
# =============================================================================

class YahooSource(DataSource):
    """Yahoo Finance stock data - daily updates."""

    name = "Yahoo Finance"
    exchange = "us_stock"
    asset_type = "stock"

    # Top 100 stocks + ETFs
    DEFAULT_SYMBOLS = [
        # Tech
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM",
        "ORCL", "ADBE", "CSCO", "IBM", "QCOM", "TXN", "AVGO", "NOW", "SHOP", "SNOW",
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
        "SPY", "QQQ", "DIA", "IWM", "VTI", "VOO", "VEA", "VWO", "BND", "GLD",
        "XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU", "XLRE",
    ]

    def __init__(self, symbols: List[str] = None):
        self._symbols = symbols or self.DEFAULT_SYMBOLS

    def get_symbols(self) -> List[str]:
        return self._symbols

    def setup(self) -> dict:
        """Download max history for all symbols."""
        print(f"\n[{self.name}] Setting up - downloading max history...")
        return self._collect_all(period="max")

    def collect(self) -> dict:
        """Update with latest data."""
        return self._collect_all(period="5d")

    def _collect_all(self, period: str = "5d") -> dict:
        stats = {"symbols": 0, "rows": 0, "errors": 0}

        for symbol in tqdm(self._symbols, desc=f"  [{self.name}]", unit="sym"):
            if not running:
                break

            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period)

                if not df.empty:
                    # Determine asset type
                    info = ticker.info
                    self.asset_type = "etf" if info.get("quoteType") == "ETF" else "stock"

                    rows = self.insert_ohlcv(symbol, df)
                    stats["rows"] += rows
                    stats["symbols"] += 1

            except Exception as e:
                stats["errors"] += 1

            time.sleep(0.3)  # Rate limit

        return stats


# =============================================================================
# Kaggle Data Source
# =============================================================================

class KaggleSource(DataSource):
    """Kaggle historical stock datasets - one-time import."""

    name = "Kaggle"
    exchange = "us_stock"
    asset_type = "stock"

    DATASETS = [
        ("borismarjanovic/price-volume-data-for-all-us-stocks-etfs", "boris"),
        ("jacksoncrow/stock-market-dataset", "jackson"),
    ]

    def get_symbols(self) -> List[str]:
        # Kaggle provides thousands of symbols
        return ["*"]

    def setup(self) -> dict:
        """Download and import all Kaggle datasets."""
        total_stats = {"symbols": 0, "rows": 0, "errors": 0}

        for dataset_id, format_type in self.DATASETS:
            print(f"\n[{self.name}] Downloading {dataset_id}...")

            try:
                path = Path(kagglehub.dataset_download(dataset_id))
                print(f"  Downloaded to: {path}")

                stats = self._import_dataset(path, format_type)
                for k in total_stats:
                    total_stats[k] += stats[k]

            except Exception as e:
                print(f"  Error: {e}")
                total_stats["errors"] += 1

        return total_stats

    def collect(self) -> dict:
        """Kaggle is one-time import only."""
        return {"symbols": 0, "rows": 0, "errors": 0, "message": "Kaggle is setup-only"}

    def _import_dataset(self, path: Path, format_type: str) -> dict:
        stats = {"symbols": 0, "rows": 0, "errors": 0}

        # Find files
        files = []
        for subdir in ["Stocks", "stocks", "ETFs", "etfs"]:
            dir_path = path / subdir
            if dir_path.exists():
                asset = "etf" if "etf" in subdir.lower() else "stock"
                for f in dir_path.glob("*.txt"):
                    files.append((f, asset))
                for f in dir_path.glob("*.csv"):
                    files.append((f, asset))

        print(f"  Found {len(files)} files")

        for filepath, asset_type in tqdm(files, desc="  Importing", unit="files"):
            if not running:
                break

            symbol = filepath.stem.upper()
            self.asset_type = asset_type

            try:
                df = pd.read_csv(filepath)
                df.columns = df.columns.str.lower().str.replace(" ", "_")

                if "date" not in df.columns:
                    continue

                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")

                # Rename columns to standard names
                col_map = {"adj_close": "close"} if "adj_close" in df.columns else {}
                df = df.rename(columns=col_map)

                rows = self.insert_ohlcv(symbol, df)
                if rows > 0:
                    stats["symbols"] += 1
                    stats["rows"] += rows

            except Exception as e:
                stats["errors"] += 1

        return stats


# =============================================================================
# Data Manager
# =============================================================================

class DataManager:
    """Unified manager for all data sources."""

    def __init__(self):
        self.sources: Dict[str, DataSource] = {
            "binance": BinanceSource(),
            "yahoo": YahooSource(),
            "kaggle": KaggleSource(),
        }

    def setup_all(self):
        """Run setup for all sources."""
        print("\n" + "=" * 70)
        print("AlphaTrader Data Manager - Full Setup")
        print("=" * 70)
        print(f"Started: {datetime.now()}")

        total_stats = {"sources": 0, "symbols": 0, "rows": 0, "errors": 0}

        for name, source in self.sources.items():
            if not running:
                break

            print(f"\n{'='*60}")
            print(f"Setting up: {source.name}")
            print("=" * 60)

            stats = source.setup()
            print(f"  Result: {stats['symbols']:,} symbols, {stats['rows']:,} rows")

            total_stats["sources"] += 1
            for k in ["symbols", "rows", "errors"]:
                total_stats[k] += stats.get(k, 0)

        print("\n" + "=" * 70)
        print("Setup Complete!")
        print("=" * 70)
        print(f"Sources: {total_stats['sources']}")
        print(f"Total symbols: {total_stats['symbols']:,}")
        print(f"Total rows: {total_stats['rows']:,}")
        print(f"Errors: {total_stats['errors']}")
        print(f"Finished: {datetime.now()}")

    def run_daemon(self):
        """Run continuous collection for all sources."""
        global running

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        print("\n" + "=" * 70)
        print("AlphaTrader Data Manager - Daemon Mode (24/7)")
        print("=" * 70)
        print(f"Started: {datetime.now()}")
        print("Sources:", ", ".join(self.sources.keys()))
        print("Press Ctrl+C to stop")

        cycle = 1

        while running:
            print(f"\n[Cycle {cycle}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # Collect from each source
            for name, source in self.sources.items():
                if not running:
                    break

                if name == "kaggle":
                    continue  # Kaggle is setup-only

                try:
                    stats = source.collect()
                    print(f"  {source.name}: {stats['rows']:,} rows")
                except Exception as e:
                    print(f"  {source.name}: Error - {e}")

            cycle += 1

            # Sleep until next cycle (5 minutes)
            print("  Sleeping 5 minutes...")
            for _ in range(300):
                if not running:
                    break
                time.sleep(1)

        print("\nDaemon stopped.")

    def show_status(self):
        """Show data statistics."""
        conn = get_db_connection()
        cursor = conn.cursor()

        print("\n" + "=" * 70)
        print("AlphaTrader Data Status")
        print("=" * 70)

        # By exchange
        cursor.execute("""
            SELECT exchange, asset_type, COUNT(*) as rows,
                   COUNT(DISTINCT symbol) as symbols,
                   MIN(open_time), MAX(open_time)
            FROM ohlcv
            GROUP BY exchange, asset_type
            ORDER BY rows DESC
        """)

        print(f"\n{'Exchange':<15} {'Type':<10} {'Rows':<15} {'Symbols':<10} {'Date Range'}")
        print("-" * 70)
        for row in cursor.fetchall():
            min_date = datetime.fromtimestamp(row[4]/1000).date() if row[4] else "N/A"
            max_date = datetime.fromtimestamp(row[5]/1000).date() if row[5] else "N/A"
            print(f"{row[0]:<15} {row[1] or 'N/A':<10} {row[2]:>12,} {row[3]:>8,}   {min_date} to {max_date}")

        # Totals
        cursor.execute("SELECT COUNT(*), COUNT(DISTINCT symbol) FROM ohlcv")
        total = cursor.fetchone()
        print(f"\n{'TOTAL':<15} {'':<10} {total[0]:>12,} {total[1]:>8,}")

        conn.close()


def main():
    parser = argparse.ArgumentParser(description="AlphaTrader Data Manager")
    parser.add_argument("--setup", action="store_true", help="First-time setup (download + import all)")
    parser.add_argument("--daemon", "-d", action="store_true", help="Run all collectors 24/7")
    parser.add_argument("--status", "-s", action="store_true", help="Show data statistics")
    parser.add_argument("--collect", "-c", action="store_true", help="Run one collection cycle")

    args = parser.parse_args()

    manager = DataManager()

    if args.setup:
        manager.setup_all()
    elif args.daemon:
        manager.run_daemon()
    elif args.collect:
        for name, source in manager.sources.items():
            if name != "kaggle":
                stats = source.collect()
                print(f"{source.name}: {stats}")
    else:
        manager.show_status()


if __name__ == "__main__":
    main()

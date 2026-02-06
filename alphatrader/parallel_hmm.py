#!/usr/bin/env python3
"""
Parallel HMM Processor - Scalable multi-machine processing.

Supports ANY number of machines joining dynamically:
- Machines register themselves and claim symbol batches
- Work is distributed automatically
- Add more power anytime by starting on another machine

Usage:
    python parallel_hmm.py --workers 8                    # Auto-claim work
    python parallel_hmm.py --workers 16 --name "thomas"   # Named worker
    python parallel_hmm.py --status                       # Check progress
    python parallel_hmm.py --reset                        # Reset all claims (restart)

Machines:
    Server:     python parallel_hmm.py -w 8 --name server
    Thomas PC:  python parallel_hmm.py -w 16 --name thomas
    Laptop:     python parallel_hmm.py -w 8 --name laptop
    Fips PC:    python parallel_hmm.py -w 12 --name fips
"""

import os
import sys
import argparse
import time
import socket
import platform
from pathlib import Path
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Tuple
import signal

from dotenv import load_dotenv
load_dotenv()

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

import psycopg2
import pandas as pd
import numpy as np

# Try to import HMM detector
try:
    from src.regime import HMMRegimeDetector
    HAS_HMM = True
except ImportError:
    HAS_HMM = False
    print("Warning: HMMRegimeDetector not available. Install dependencies.")

# Database config - auto-detect network
def get_db_host():
    """Auto-detect database host."""
    if os.getenv("ZENKAI_DB_HOST"):
        return os.getenv("ZENKAI_DB_HOST")
    local_ip = os.getenv("ZENKAI_DB_HOST", "192.168.0.160")
    tailscale_ip = os.getenv("ZENKAI_DB_HOST_TAILSCALE", "100.110.101.78")

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        if sock.connect_ex((local_ip, 5432)) == 0:
            sock.close()
            return local_ip
    except:
        pass
    return tailscale_ip

DB_CONFIG = {
    "host": os.getenv("ZENKAI_DB_HOST", get_db_host()),
    "database": os.getenv("ZENKAI_DB_NAME", "zenkai_data"),
    "user": os.getenv("ZENKAI_DB_USER", "zenkai"),
    "password": os.getenv("ZENKAI_DB_PASSWORD"),
}

MODEL_DIR = Path(__file__).parent / "models"

# Work claim timeout - if a machine doesn't update for this long, its work is reclaimed
CLAIM_TIMEOUT_MINUTES = 10
BATCH_SIZE = 10  # Symbols per batch claim

# Global flag for graceful shutdown
shutdown_flag = False

def signal_handler(signum, frame):
    global shutdown_flag
    print("\nShutdown requested...")
    shutdown_flag = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def get_machine_id(name: str = None) -> str:
    """Get unique machine identifier."""
    if name:
        return name
    # Auto-generate from hostname
    hostname = socket.gethostname().lower()
    if "server" in hostname or "goku" in hostname:
        return "server"
    elif "thoma" in hostname:
        return "thomas"
    elif "fips" in hostname:
        return "fips"
    else:
        return hostname[:20]


def ensure_coordination_table(conn):
    """Create coordination table if it doesn't exist."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hmm_work_claims (
            symbol VARCHAR(50) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            machine_id VARCHAR(50) NOT NULL,
            claimed_at TIMESTAMP DEFAULT NOW(),
            last_heartbeat TIMESTAMP DEFAULT NOW(),
            status VARCHAR(20) DEFAULT 'claimed',
            rows_processed INTEGER DEFAULT 0,
            PRIMARY KEY (symbol, timeframe)
        )
    """)
    conn.commit()
    cursor.close()


def claim_work_batch(conn, timeframe: str, machine_id: str, batch_size: int = BATCH_SIZE) -> List[str]:
    """Claim a batch of symbols to process."""
    cursor = conn.cursor()

    # First, release stale claims (machines that disappeared)
    stale_threshold = datetime.now() - timedelta(minutes=CLAIM_TIMEOUT_MINUTES)
    cursor.execute("""
        DELETE FROM hmm_work_claims
        WHERE last_heartbeat < %s AND status = 'claimed'
    """, (stale_threshold,))

    # Find unclaimed symbols that need processing
    cursor.execute("""
        SELECT DISTINCT o.symbol
        FROM ohlcv o
        LEFT JOIN hmm_work_claims c ON o.symbol = c.symbol AND o.timeframe = c.timeframe
        WHERE o.timeframe = %s
        AND o.regime IS NULL
        AND c.symbol IS NULL
        ORDER BY o.symbol
        LIMIT %s
    """, (timeframe, batch_size))

    symbols = [row[0] for row in cursor.fetchall()]

    # Claim them
    for symbol in symbols:
        try:
            cursor.execute("""
                INSERT INTO hmm_work_claims (symbol, timeframe, machine_id, claimed_at, last_heartbeat, status)
                VALUES (%s, %s, %s, NOW(), NOW(), 'claimed')
                ON CONFLICT (symbol, timeframe) DO NOTHING
            """, (symbol, timeframe, machine_id))
        except:
            pass  # Another machine claimed it first

    conn.commit()
    cursor.close()

    return symbols


def update_heartbeat(conn, symbols: List[str], timeframe: str, machine_id: str):
    """Update heartbeat for claimed symbols."""
    if not symbols:
        return
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE hmm_work_claims
        SET last_heartbeat = NOW()
        WHERE symbol = ANY(%s) AND timeframe = %s AND machine_id = %s
    """, (symbols, timeframe, machine_id))
    conn.commit()
    cursor.close()


def mark_complete(conn, symbol: str, timeframe: str, rows_processed: int):
    """Mark a symbol as complete."""
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE hmm_work_claims
        SET status = 'complete', rows_processed = %s, last_heartbeat = NOW()
        WHERE symbol = %s AND timeframe = %s
    """, (rows_processed, symbol, timeframe))
    conn.commit()
    cursor.close()


def release_claims(conn, machine_id: str):
    """Release all claims for this machine (on shutdown)."""
    cursor = conn.cursor()
    cursor.execute("""
        DELETE FROM hmm_work_claims
        WHERE machine_id = %s AND status = 'claimed'
    """, (machine_id,))
    conn.commit()
    cursor.close()


def compute_hmm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features needed for HMM."""
    df = df.copy()

    # Returns
    df["returns_1"] = df["close"].pct_change().fillna(0)

    # Volatility (20-bar rolling std)
    vol = df["close"].pct_change().rolling(20).std()
    df["volatility"] = vol.fillna(vol.mean() if vol.notna().any() else 0.01)

    # Volume ratio
    vol_sma = df["volume"].rolling(20).mean().replace(0, np.nan)
    df["volume_ratio"] = (df["volume"] / vol_sma).fillna(1.0)

    return df


def get_or_load_model(timeframe: str) -> Optional[HMMRegimeDetector]:
    """Load HMM model for timeframe."""
    if not HAS_HMM:
        return None

    model_path = MODEL_DIR / f"hmm_regime_{timeframe}.pkl"

    if model_path.exists():
        try:
            return HMMRegimeDetector.load(str(model_path))
        except Exception as e:
            print(f"  Failed to load model for {timeframe}: {e}")

    return None


def process_symbol(args) -> Tuple[str, int, str]:
    """Process all unprocessed rows for a single symbol."""
    symbol, timeframe, model_path = args

    if shutdown_flag:
        return (symbol, 0, "shutdown")

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Get unprocessed rows for this symbol
        cursor.execute("""
            SELECT open_time, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = %s AND timeframe = %s AND regime IS NULL
            ORDER BY open_time
            LIMIT 100000
        """, (symbol, timeframe))

        rows = cursor.fetchall()
        if not rows:
            conn.close()
            return (symbol, 0, "done")

        df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
        df = compute_hmm_features(df)

        # Load model (each process loads its own)
        if not HAS_HMM:
            conn.close()
            return (symbol, 0, "no_hmm_lib")

        try:
            model = HMMRegimeDetector.load(model_path)
        except Exception as e:
            conn.close()
            return (symbol, 0, f"model_error: {e}")

        # Predict regimes
        try:
            regimes = model.predict(df)
        except Exception as e:
            conn.close()
            return (symbol, 0, f"predict_error: {e}")

        # Update database in batches
        batch_size = 5000
        total_updated = 0

        for i in range(0, len(regimes), batch_size):
            batch_regimes = regimes[i:i+batch_size]
            batch_times = df["open_time"].iloc[i:i+batch_size]

            updates = [(regime, symbol, timeframe, int(ot))
                       for regime, ot in zip(batch_regimes, batch_times)]

            cursor.executemany("""
                UPDATE ohlcv SET regime = %s
                WHERE symbol = %s AND timeframe = %s AND open_time = %s
            """, updates)
            conn.commit()
            total_updated += len(updates)

        cursor.close()
        conn.close()

        return (symbol, total_updated, "ok")

    except Exception as e:
        return (symbol, 0, f"error: {e}")


def get_status() -> dict:
    """Get processing status."""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    ensure_coordination_table(conn)

    cursor.execute("SELECT COUNT(*) FROM ohlcv WHERE regime IS NULL")
    unprocessed = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM ohlcv WHERE regime IS NOT NULL")
    processed = cursor.fetchone()[0]

    cursor.execute("""
        SELECT timeframe, COUNT(*) as total,
               SUM(CASE WHEN regime IS NOT NULL THEN 1 ELSE 0 END) as done
        FROM ohlcv
        GROUP BY timeframe
        ORDER BY total DESC
    """)
    by_tf = cursor.fetchall()

    # Get active workers
    stale_threshold = datetime.now() - timedelta(minutes=CLAIM_TIMEOUT_MINUTES)
    cursor.execute("""
        SELECT machine_id, COUNT(*) as symbols, SUM(rows_processed) as rows,
               MAX(last_heartbeat) as last_seen
        FROM hmm_work_claims
        WHERE last_heartbeat > %s OR status = 'complete'
        GROUP BY machine_id
        ORDER BY machine_id
    """, (stale_threshold,))
    workers = cursor.fetchall()

    cursor.close()
    conn.close()

    return {
        "processed": processed,
        "unprocessed": unprocessed,
        "total": processed + unprocessed,
        "by_timeframe": by_tf,
        "workers": workers,
    }


def print_status():
    """Print processing status."""
    status = get_status()

    print("=" * 70)
    print("HMM PROCESSING STATUS")
    print("=" * 70)
    print(f"Total rows:      {status['total']:>15,}")
    print(f"Processed:       {status['processed']:>15,}")
    print(f"Remaining:       {status['unprocessed']:>15,}")

    if status['total'] > 0:
        pct = status['processed'] / status['total'] * 100
        print(f"Progress:        {pct:>14.2f}%")

    print("\n--- By Timeframe ---")
    for tf, total, done in status['by_timeframe'][:10]:
        pct = (done / total * 100) if total > 0 else 0
        bar = "=" * int(pct / 5) + "-" * (20 - int(pct / 5))
        print(f"  {tf:8s}: [{bar}] {pct:5.1f}% ({done:,}/{total:,})")

    if status['workers']:
        print("\n--- Active Workers ---")
        for machine_id, symbols, rows, last_seen in status['workers']:
            rows = rows or 0
            ago = (datetime.now() - last_seen).seconds if last_seen else 0
            status_str = f"{ago}s ago" if ago < 120 else "stale"
            print(f"  {machine_id:15s}: {symbols:4d} symbols, {rows:>12,} rows ({status_str})")
    else:
        print("\n--- No Active Workers ---")
        print("  Start processing with: python parallel_hmm.py -w 8")


def reset_claims():
    """Reset all work claims (for fresh start)."""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    ensure_coordination_table(conn)

    cursor.execute("DELETE FROM hmm_work_claims")
    conn.commit()

    deleted = cursor.rowcount
    cursor.close()
    conn.close()

    print(f"Reset {deleted} work claims. Ready for fresh start.")


def run_parallel(workers: int, timeframe: str = "1m", machine_name: str = None):
    """Run parallel HMM processing with dynamic work claiming."""
    global shutdown_flag

    machine_id = get_machine_id(machine_name)

    print("=" * 70)
    print("PARALLEL HMM PROCESSOR")
    print("=" * 70)
    print(f"Machine:    {machine_id}")
    print(f"Workers:    {workers}")
    print(f"Timeframe:  {timeframe}")
    print(f"DB Host:    {DB_CONFIG['host']}")
    print(f"Started:    {datetime.now()}")
    print("=" * 70)
    print("\nAdd more machines anytime:")
    print("  Server:  python parallel_hmm.py -w 8 --name server")
    print("  Laptop:  python parallel_hmm.py -w 8 --name laptop")
    print("  Fips:    python parallel_hmm.py -w 12 --name fips")
    print("=" * 70)

    # Setup
    conn = psycopg2.connect(**DB_CONFIG)
    ensure_coordination_table(conn)
    conn.close()

    # Check for models
    model_path = MODEL_DIR / f"hmm_regime_{timeframe}.pkl"
    if not model_path.exists():
        print(f"\nERROR: No HMM model found for {timeframe}")
        print(f"Expected: {model_path}")
        print("\nTrain a model first using hmm_processor.py on the server.")
        return

    print(f"\nHMM model loaded for {timeframe}")

    start_time = time.time()
    total_processed = 0
    rounds = 0

    try:
        while not shutdown_flag:
            # Claim a batch of work
            conn = psycopg2.connect(**DB_CONFIG)
            symbols = claim_work_batch(conn, timeframe, machine_id, BATCH_SIZE)
            conn.close()

            if not symbols:
                # Check if there's really nothing left
                status = get_status()
                if status['unprocessed'] == 0:
                    print("\nAll symbols processed!")
                    break
                else:
                    # Other machines might be working, wait and retry
                    print(f"\nNo unclaimed work. {status['unprocessed']:,} rows still processing by others...")
                    print("Waiting 30s for work to become available...")
                    time.sleep(30)
                    continue

            rounds += 1
            print(f"\n--- Round {rounds}: Claimed {len(symbols)} symbols ---")
            print(f"    {', '.join(symbols)}")

            # Create work items with model path
            work_items = [(s, timeframe, str(model_path)) for s in symbols]

            # Process in parallel
            round_processed = 0
            conn = psycopg2.connect(**DB_CONFIG)

            with Pool(workers) as pool:
                for result in pool.imap_unordered(process_symbol, work_items):
                    symbol, count, status = result

                    if shutdown_flag:
                        break

                    if count > 0:
                        round_processed += count
                        total_processed += count
                        mark_complete(conn, symbol, timeframe, count)
                        print(f"  {symbol}: {count:,} rows")
                    elif status not in ["done", "ok"]:
                        print(f"  {symbol}: {status}")
                        mark_complete(conn, symbol, timeframe, 0)

                    # Update heartbeat periodically
                    update_heartbeat(conn, symbols, timeframe, machine_id)

            conn.close()

            elapsed = time.time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            print(f"\nRound {rounds}: {round_processed:,} rows | Total: {total_processed:,} | Rate: {rate:,.0f}/sec")

    finally:
        # Release uncompleted claims on shutdown
        conn = psycopg2.connect(**DB_CONFIG)
        release_claims(conn, machine_id)
        conn.close()
        print(f"\nReleased uncompleted claims for {machine_id}")

    # Final summary
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Machine:         {machine_id}")
    print(f"Total processed: {total_processed:,}")
    print(f"Time:            {elapsed/60:.1f} minutes")
    if elapsed > 0:
        print(f"Rate:            {total_processed/elapsed:,.0f} rows/sec")


def main():
    parser = argparse.ArgumentParser(
        description="Parallel HMM Processor - Scalable multi-machine processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python parallel_hmm.py --status                  # Check progress
    python parallel_hmm.py -w 8 --name server        # Run on server
    python parallel_hmm.py -w 16 --name thomas       # Run on Thomas PC
    python parallel_hmm.py -w 8 --name laptop        # Run on laptop
    python parallel_hmm.py --reset                   # Reset all claims
        """
    )
    parser.add_argument("--workers", "-w", type=int, default=cpu_count(),
                        help=f"Number of workers (default: {cpu_count()})")
    parser.add_argument("--timeframe", "-t", type=str, default="1m",
                        help="Timeframe to process (default: 1m)")
    parser.add_argument("--name", "-n", type=str,
                        help="Machine name (auto-detected if not specified)")
    parser.add_argument("--status", action="store_true",
                        help="Show processing status")
    parser.add_argument("--reset", action="store_true",
                        help="Reset all work claims (fresh start)")

    # Keep old args for compatibility
    parser.add_argument("--start-from", "-s", type=str,
                        help="(Deprecated) Use --name instead")

    args = parser.parse_args()

    if args.status:
        print_status()
    elif args.reset:
        reset_claims()
    else:
        run_parallel(
            workers=args.workers,
            timeframe=args.timeframe,
            machine_name=args.name,
        )


if __name__ == "__main__":
    main()

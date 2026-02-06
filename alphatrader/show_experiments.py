#!/usr/bin/env python3
"""
Show Experiments - View and filter LSTM training experiments.

Usage:
    python show_experiments.py                    # Show all experiments
    python show_experiments.py --worker thomas    # Filter by worker
    python show_experiments.py --timeframe 1h     # Filter by timeframe
    python show_experiments.py --sort accuracy    # Sort by test accuracy
    python show_experiments.py --last 10          # Show last 10 experiments
    python show_experiments.py --id 42            # Show details for experiment #42
    python show_experiments.py --summary          # Show summary statistics
    python show_experiments.py --leaderboard      # Show top experiments
"""

import argparse
import os
from datetime import datetime
from typing import Optional

import psycopg2
from dotenv import load_dotenv

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


def format_duration(seconds: int) -> str:
    """Format seconds as human-readable duration."""
    if seconds is None:
        return "-"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def format_accuracy(acc: float) -> str:
    """Format accuracy as percentage."""
    if acc is None:
        return "-"
    return f"{acc*100:.1f}%"


def list_experiments(
    worker: Optional[str] = None,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    status: Optional[str] = None,
    sort_by: str = "started_at",
    sort_desc: bool = True,
    limit: int = 50,
):
    """List experiments with filters."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Build query
    query = """
        SELECT
            id, run_id, worker, status,
            symbol, timeframe, epochs_completed, epochs_requested,
            hidden_size, num_layers, seq_length, batch_size,
            test_accuracy, acc_down, acc_flat, acc_up,
            training_time_seconds, started_at, hostname
        FROM experiments
        WHERE 1=1
    """
    params = []

    if worker:
        query += " AND worker = %s"
        params.append(worker)
    if symbol:
        query += " AND symbol = %s"
        params.append(symbol)
    if timeframe:
        query += " AND timeframe = %s"
        params.append(timeframe)
    if status:
        query += " AND status = %s"
        params.append(status)

    # Sort mapping
    sort_columns = {
        'started_at': 'started_at',
        'date': 'started_at',
        'accuracy': 'test_accuracy',
        'acc': 'test_accuracy',
        'worker': 'worker',
        'timeframe': 'timeframe',
        'epochs': 'epochs_completed',
        'time': 'training_time_seconds',
    }

    sort_col = sort_columns.get(sort_by.lower(), 'started_at')
    sort_order = "DESC" if sort_desc else "ASC"

    # Handle NULL values in sort
    if sort_col in ['test_accuracy', 'training_time_seconds']:
        query += f" ORDER BY {sort_col} IS NULL, {sort_col} {sort_order}"
    else:
        query += f" ORDER BY {sort_col} {sort_order}"

    query += " LIMIT %s"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    # Print header
    print("\n" + "=" * 120)
    print("AlphaTrader Experiments")
    print("=" * 120)

    if worker or symbol or timeframe or status:
        filters = []
        if worker: filters.append(f"worker={worker}")
        if symbol: filters.append(f"symbol={symbol}")
        if timeframe: filters.append(f"timeframe={timeframe}")
        if status: filters.append(f"status={status}")
        print(f"Filters: {', '.join(filters)}")

    # Print table header
    print()
    print(f"{'ID':>4} {'Worker':<10} {'Status':<8} {'Symbol':<8} {'TF':<4} {'Epochs':<8} "
          f"{'Hidden':>6} {'Layers':>6} {'Seq':>4} "
          f"{'Acc':>7} {'Down':>7} {'Flat':>7} {'Up':>7} {'Time':>8} {'Date':<12}")
    print("-" * 120)

    for row in rows:
        (id, run_id, worker, status, symbol, timeframe,
         epochs_done, epochs_req, hidden, layers, seq, batch,
         test_acc, acc_down, acc_flat, acc_up,
         train_time, started_at, hostname) = row

        status_str = status[:7] if status else "-"
        symbol_str = symbol[:7] if symbol else "ALL"
        epochs_str = f"{epochs_done or 0}/{epochs_req}"
        date_str = started_at.strftime("%m-%d %H:%M") if started_at else "-"

        print(f"{id:>4} {worker:<10} {status_str:<8} {symbol_str:<8} {timeframe:<4} {epochs_str:<8} "
              f"{hidden:>6} {layers:>6} {seq:>4} "
              f"{format_accuracy(test_acc):>7} {format_accuracy(acc_down):>7} "
              f"{format_accuracy(acc_flat):>7} {format_accuracy(acc_up):>7} "
              f"{format_duration(train_time):>8} {date_str:<12}")

    print(f"\nShowing {len(rows)} experiments")


def show_experiment_details(experiment_id: int):
    """Show detailed info for a single experiment."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM experiments WHERE id = %s", (experiment_id,))
    row = cursor.fetchone()

    if not row:
        print(f"Experiment {experiment_id} not found")
        return

    # Get column names
    columns = [desc[0] for desc in cursor.description]
    data = dict(zip(columns, row))
    conn.close()

    print("\n" + "=" * 60)
    print(f"Experiment #{experiment_id} Details")
    print("=" * 60)

    # Basic info
    print(f"\n--- Run Info ---")
    print(f"Run ID:     {data['run_id']}")
    print(f"Worker:     {data['worker']}")
    print(f"Status:     {data['status']}")
    print(f"Hostname:   {data['hostname']}")
    print(f"GPU:        {data['gpu_name'] or 'N/A'}")

    # Timing
    print(f"\n--- Timing ---")
    print(f"Started:    {data['started_at']}")
    print(f"Finished:   {data['finished_at']}")
    print(f"Duration:   {format_duration(data['training_time_seconds'])}")

    # Configuration
    print(f"\n--- Configuration ---")
    print(f"Symbol:     {data['symbol'] or 'ALL'}")
    print(f"Timeframe:  {data['timeframe']}")
    print(f"Epochs:     {data['epochs_completed']}/{data['epochs_requested']}")
    print(f"Batch Size: {data['batch_size']}")
    print(f"Hidden:     {data['hidden_size']}")
    print(f"Layers:     {data['num_layers']}")
    print(f"Seq Length: {data['seq_length']}")
    print(f"Data Limit: {data['data_limit']:,}")

    # Data
    print(f"\n--- Data ---")
    print(f"Train:      {data['train_samples']:,}" if data['train_samples'] else "Train:      -")
    print(f"Validation: {data['val_samples']:,}" if data['val_samples'] else "Validation: -")
    print(f"Test:       {data['test_samples']:,}" if data['test_samples'] else "Test:       -")

    # Results
    print(f"\n--- Results ---")
    print(f"Test Accuracy:    {format_accuracy(data['test_accuracy'])}")
    print(f"Best Val Loss:    {data['best_val_loss']:.4f}" if data['best_val_loss'] else "Best Val Loss:    -")

    # Per-class
    print(f"\n--- Per-Class Accuracy ---")
    if data['samples_down']:
        print(f"DOWN:  {format_accuracy(data['acc_down'])} ({data['samples_down']:,} samples)")
    else:
        print("DOWN:  -")
    if data['samples_flat']:
        print(f"FLAT:  {format_accuracy(data['acc_flat'])} ({data['samples_flat']:,} samples)")
    else:
        print("FLAT:  -")
    if data['samples_up']:
        print(f"UP:    {format_accuracy(data['acc_up'])} ({data['samples_up']:,} samples)")
    else:
        print("UP:    -")

    # Checkpoint
    print(f"\n--- Checkpoint ---")
    print(f"Path: {data['checkpoint_path']}")


def show_summary_stats():
    """Show summary statistics across all experiments."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Overall stats
    cursor.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE status = 'completed') as completed,
            COUNT(*) FILTER (WHERE status = 'failed') as failed,
            COUNT(*) FILTER (WHERE status = 'running') as running,
            AVG(test_accuracy) FILTER (WHERE status = 'completed') as avg_acc,
            MAX(test_accuracy) FILTER (WHERE status = 'completed') as max_acc,
            AVG(training_time_seconds) FILTER (WHERE status = 'completed') as avg_time
        FROM experiments
    """)
    stats = cursor.fetchone()

    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"\nTotal Experiments: {stats[0]}")
    print(f"  Completed: {stats[1]}")
    print(f"  Failed:    {stats[2]}")
    print(f"  Running:   {stats[3]}")
    print(f"\nAverage Test Accuracy: {format_accuracy(stats[4])}")
    print(f"Best Test Accuracy:    {format_accuracy(stats[5])}")
    print(f"Average Training Time: {format_duration(int(stats[6]) if stats[6] else 0)}")

    # Per-worker stats
    cursor.execute("""
        SELECT
            worker,
            COUNT(*) as runs,
            AVG(test_accuracy) FILTER (WHERE status = 'completed') as avg_acc,
            MAX(test_accuracy) FILTER (WHERE status = 'completed') as max_acc
        FROM experiments
        GROUP BY worker
        ORDER BY max_acc DESC NULLS LAST
    """)
    workers = cursor.fetchall()

    if workers:
        print("\n--- Per Worker ---")
        for w in workers:
            print(f"  {w[0]}: {w[1]} runs, avg {format_accuracy(w[2])}, best {format_accuracy(w[3])}")

    # Per-timeframe stats
    cursor.execute("""
        SELECT
            timeframe,
            COUNT(*) as runs,
            AVG(test_accuracy) FILTER (WHERE status = 'completed') as avg_acc,
            MAX(test_accuracy) FILTER (WHERE status = 'completed') as max_acc
        FROM experiments
        GROUP BY timeframe
        ORDER BY max_acc DESC NULLS LAST
    """)
    timeframes = cursor.fetchall()

    if timeframes:
        print("\n--- Per Timeframe ---")
        for tf in timeframes:
            print(f"  {tf[0]}: {tf[1]} runs, avg {format_accuracy(tf[2])}, best {format_accuracy(tf[3])}")

    conn.close()


def show_leaderboard(top_n: int = 10):
    """Show top N experiments by accuracy."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id, worker, symbol, timeframe,
            hidden_size, num_layers, seq_length,
            test_accuracy, acc_down, acc_flat, acc_up,
            started_at
        FROM experiments
        WHERE status = 'completed' AND test_accuracy IS NOT NULL
        ORDER BY test_accuracy DESC
        LIMIT %s
    """, (top_n,))

    rows = cursor.fetchall()
    conn.close()

    print("\n" + "=" * 100)
    print(f"Top {top_n} Experiments (Leaderboard)")
    print("=" * 100)

    print(f"\n{'Rank':<5} {'ID':>4} {'Worker':<10} {'Symbol':<8} {'TF':<4} "
          f"{'Hidden':>6} {'Layers':>6} {'Seq':>4} "
          f"{'Accuracy':>9} {'Down':>7} {'Flat':>7} {'Up':>7} {'Date':<12}")
    print("-" * 100)

    for i, row in enumerate(rows, 1):
        (id, worker, symbol, timeframe, hidden, layers, seq,
         test_acc, acc_down, acc_flat, acc_up, started_at) = row

        symbol_str = symbol[:7] if symbol else "ALL"
        date_str = started_at.strftime("%Y-%m-%d") if started_at else "-"

        print(f"#{i:<4} {id:>4} {worker:<10} {symbol_str:<8} {timeframe:<4} "
              f"{hidden:>6} {layers:>6} {seq:>4} "
              f"{format_accuracy(test_acc):>9} {format_accuracy(acc_down):>7} "
              f"{format_accuracy(acc_flat):>7} {format_accuracy(acc_up):>7} {date_str:<12}")


def main():
    parser = argparse.ArgumentParser(description="View AlphaTrader experiments")

    # Filters
    parser.add_argument("--worker", "-w", type=str, help="Filter by worker name")
    parser.add_argument("--symbol", "-s", type=str, help="Filter by symbol")
    parser.add_argument("--timeframe", "-tf", type=str, help="Filter by timeframe")
    parser.add_argument("--status", type=str, help="Filter by status (running/completed/failed)")

    # Sorting
    parser.add_argument("--sort", type=str, default="started_at",
                       help="Sort by: date, accuracy, worker, timeframe, epochs, time")
    parser.add_argument("--asc", action="store_true", help="Sort ascending (default: descending)")

    # Limits and views
    parser.add_argument("--last", "-n", type=int, default=50, help="Number of experiments to show")
    parser.add_argument("--id", type=int, help="Show details for specific experiment ID")
    parser.add_argument("--summary", action="store_true", help="Show summary statistics")
    parser.add_argument("--leaderboard", "-lb", action="store_true", help="Show top experiments")
    parser.add_argument("--top", type=int, default=10, help="Number of top experiments for leaderboard")

    args = parser.parse_args()

    if args.id:
        show_experiment_details(args.id)
    elif args.summary:
        show_summary_stats()
    elif args.leaderboard:
        show_leaderboard(args.top)
    else:
        list_experiments(
            worker=args.worker,
            symbol=args.symbol,
            timeframe=args.timeframe,
            status=args.status,
            sort_by=args.sort,
            sort_desc=not args.asc,
            limit=args.last,
        )


if __name__ == "__main__":
    main()

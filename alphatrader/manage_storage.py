#!/usr/bin/env python3
"""
Storage Management for AlphaTrader PostgreSQL Database.

Manages data across SSD (fast) and HDD (large capacity) tablespaces.

Strategy:
- Recent data (last 6 months) stays on SSD for fast queries
- Older data moves to HDD tablespace for cost-effective storage
- Training queries can access all data transparently

Usage:
    python manage_storage.py --status          # Show storage usage
    python manage_storage.py --migrate         # Move old data to HDD
    python manage_storage.py --migrate --dry-run  # Preview migration
"""

import argparse
import os
from datetime import datetime, timedelta
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


def show_tablespace_usage():
    """Show current tablespace usage."""
    conn = get_db_connection()
    cursor = conn.cursor()

    print("=" * 60)
    print("PostgreSQL Tablespace Usage")
    print("=" * 60)

    # Get tablespace sizes
    cursor.execute("""
        SELECT
            spcname as name,
            pg_tablespace_location(oid) as location,
            pg_size_pretty(pg_tablespace_size(oid)) as size
        FROM pg_tablespace
        ORDER BY pg_tablespace_size(oid) DESC
    """)

    print(f"\n{'Tablespace':<15} {'Location':<30} {'Size':<15}")
    print("-" * 60)
    for row in cursor.fetchall():
        loc = row[1] or "(default)"
        print(f"{row[0]:<15} {loc:<30} {row[2]:<15}")

    # Get table sizes in zenkai_data
    cursor.execute("""
        SELECT
            relname as table_name,
            pg_size_pretty(pg_total_relation_size(oid)) as total_size,
            pg_size_pretty(pg_relation_size(oid)) as table_size,
            pg_size_pretty(pg_indexes_size(oid)) as index_size
        FROM pg_class
        WHERE relnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
          AND relkind = 'r'
        ORDER BY pg_total_relation_size(oid) DESC
        LIMIT 10
    """)

    print(f"\n{'Table':<25} {'Total':<15} {'Data':<15} {'Indexes':<15}")
    print("-" * 70)
    for row in cursor.fetchall():
        print(f"{row[0]:<25} {row[1]:<15} {row[2]:<15} {row[3]:<15}")

    # Get data distribution by date
    cursor.execute("""
        SELECT
            date_trunc('month', to_timestamp(open_time / 1000)) as month,
            COUNT(*) as rows,
            pg_size_pretty(COUNT(*) * 100) as est_size
        FROM ohlcv
        GROUP BY month
        ORDER BY month DESC
        LIMIT 12
    """)

    print("\n--- Data by Month ---")
    print(f"{'Month':<12} {'Rows':<15} {'Est. Size':<15}")
    print("-" * 42)
    for row in cursor.fetchall():
        if row[0]:
            month_str = row[0].strftime("%Y-%m")
            print(f"{month_str:<12} {row[1]:>12,} {row[2]:<15}")

    conn.close()


def create_archive_table_if_needed():
    """Create archive table on HDD tablespace if it doesn't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Check if archive table exists
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'ohlcv_archive'
        )
    """)

    if cursor.fetchone()[0]:
        print("Archive table already exists.")
        conn.close()
        return

    print("Creating archive table on HDD tablespace...")

    cursor.execute("""
        CREATE TABLE ohlcv_archive (
            LIKE ohlcv INCLUDING ALL
        ) TABLESPACE hdd_storage
    """)

    # Create indexes on archive table
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_archive_lookup
        ON ohlcv_archive(exchange, symbol, timeframe, open_time)
        TABLESPACE hdd_storage
    """)

    conn.commit()
    conn.close()
    print("Archive table created on HDD.")


def migrate_old_data(months_to_keep: int = 6, dry_run: bool = False):
    """Move old data to HDD archive table."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Calculate cutoff date
    cutoff = datetime.now() - timedelta(days=months_to_keep * 30)
    cutoff_ms = int(cutoff.timestamp() * 1000)

    print(f"\nCutoff date: {cutoff.date()}")
    print(f"Data older than this will be moved to HDD archive.")

    # Count rows to migrate
    cursor.execute("""
        SELECT COUNT(*) FROM ohlcv
        WHERE open_time < %s
    """, (cutoff_ms,))
    rows_to_migrate = cursor.fetchone()[0]

    print(f"Rows to migrate: {rows_to_migrate:,}")

    if rows_to_migrate == 0:
        print("No data to migrate.")
        conn.close()
        return

    if dry_run:
        print("\n[DRY RUN] Would migrate data. Use --migrate without --dry-run to execute.")
        conn.close()
        return

    # Create archive table if needed
    create_archive_table_if_needed()

    print(f"\nMigrating {rows_to_migrate:,} rows to HDD archive...")

    # Insert into archive (in batches to avoid memory issues)
    batch_size = 100000
    migrated = 0

    while migrated < rows_to_migrate:
        cursor.execute("""
            WITH moved AS (
                DELETE FROM ohlcv
                WHERE ctid IN (
                    SELECT ctid FROM ohlcv
                    WHERE open_time < %s
                    LIMIT %s
                )
                RETURNING *
            )
            INSERT INTO ohlcv_archive
            SELECT * FROM moved
        """, (cutoff_ms, batch_size))

        batch_migrated = cursor.rowcount
        migrated += batch_migrated
        conn.commit()

        print(f"  Migrated {migrated:,}/{rows_to_migrate:,} rows...")

        if batch_migrated < batch_size:
            break

    print(f"\nMigration complete! Moved {migrated:,} rows to HDD archive.")

    # Vacuum to reclaim space
    print("Running VACUUM ANALYZE...")
    cursor.execute("VACUUM ANALYZE ohlcv")
    conn.commit()
    conn.close()

    print("Done!")


def create_unified_view():
    """Create a view that combines current and archive data."""
    conn = get_db_connection()
    cursor = conn.cursor()

    print("Creating unified view...")

    cursor.execute("""
        CREATE OR REPLACE VIEW ohlcv_all AS
        SELECT * FROM ohlcv
        UNION ALL
        SELECT * FROM ohlcv_archive
    """)

    conn.commit()
    conn.close()
    print("View 'ohlcv_all' created - queries can access all data.")


def main():
    parser = argparse.ArgumentParser(description="AlphaTrader Storage Management")
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show storage usage and data distribution"
    )
    parser.add_argument(
        "--migrate", "-m",
        action="store_true",
        help="Migrate old data to HDD tablespace"
    )
    parser.add_argument(
        "--months",
        type=int,
        default=6,
        help="Months of data to keep on SSD (default: 6)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview migration without executing"
    )
    parser.add_argument(
        "--create-view",
        action="store_true",
        help="Create unified view combining SSD and HDD data"
    )

    args = parser.parse_args()

    if args.status:
        show_tablespace_usage()
    elif args.migrate:
        migrate_old_data(args.months, args.dry_run)
    elif args.create_view:
        create_unified_view()
    else:
        # Default: show status
        show_tablespace_usage()


if __name__ == "__main__":
    main()

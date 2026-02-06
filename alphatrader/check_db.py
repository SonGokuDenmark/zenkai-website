#!/usr/bin/env python3
"""Quick database inspection script."""

import sqlite3
from pathlib import Path

db_path = Path("data/alphatrader.db")

if not db_path.exists():
    print(f"Database not found at {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("=" * 60)
print("AlphaTrader Database Summary")
print("=" * 60)

# Total rows
cursor.execute("SELECT COUNT(*) FROM market_states")
total = cursor.fetchone()[0]
print(f"\nTotal rows: {total:,}")

# By symbol
print("\nRows by symbol:")
cursor.execute("""
    SELECT symbol, COUNT(*) as cnt
    FROM market_states
    GROUP BY symbol
    ORDER BY cnt DESC
""")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]:,}")

# By timeframe
print("\nRows by timeframe:")
cursor.execute("""
    SELECT timeframe, COUNT(*) as cnt
    FROM market_states
    GROUP BY timeframe
    ORDER BY cnt DESC
""")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]:,}")

# Date range
print("\nDate range:")
cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM market_states")
row = cursor.fetchone()
print(f"  From: {row[0]}")
print(f"  To:   {row[1]}")

# Signal counts
print("\nStrategy signals (non-zero):")
signal_cols = [
    "signal_rsi_div", "signal_macd", "signal_turtle",
    "signal_sr_bounce", "signal_candlestick", "signal_ma_cross", "signal_stoch_mr"
]
for col in signal_cols:
    try:
        cursor.execute(f"SELECT COUNT(*) FROM market_states WHERE {col} != 0")
        cnt = cursor.fetchone()[0]
        strategy = col.replace("signal_", "")
        print(f"  {strategy}: {cnt:,} signals")
    except:
        pass

# Direction distribution
print("\nFuture direction distribution:")
cursor.execute("""
    SELECT future_direction, COUNT(*) as cnt
    FROM market_states
    WHERE future_direction IS NOT NULL
    GROUP BY future_direction
    ORDER BY future_direction
""")
labels = {-1: "DOWN", 0: "FLAT", 1: "UP"}
for row in cursor.fetchall():
    direction = labels.get(row[0], str(row[0]))
    print(f"  {direction}: {row[1]:,}")

# Sample row
print("\nSample row (first 10 columns):")
cursor.execute("SELECT * FROM market_states LIMIT 1")
cols = [desc[0] for desc in cursor.description]
row = cursor.fetchone()
for i, (col, val) in enumerate(zip(cols[:10], row[:10])):
    print(f"  {col}: {val}")
print("  ...")

conn.close()
print("\n" + "=" * 60)

"""
═══════════════════════════════════════════════════════════════════════════════
⚡ Zenkai Signal Hub — Database Module
© 2026 Zenkai Corporation
═══════════════════════════════════════════════════════════════════════════════

SQLite database for storing:
- Signals history
- Subscribers list
- Signal updates/results
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

from config import DATABASE_PATH

# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Signal:
    id: Optional[int]
    pair: str
    direction: str  # LONG or SHORT
    entry_low: float
    entry_high: float
    stop_loss: float
    tp1: float
    tp2: Optional[float]
    tp3: Optional[float]
    timeframe: str
    confidence: str
    note: Optional[str]
    created_at: str
    status: str  # ACTIVE, TP1_HIT, TP2_HIT, TP3_HIT, STOPPED, CANCELLED
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    closed_at: Optional[str] = None
    result_pct: Optional[float] = None


@dataclass
class Subscriber:
    user_id: int
    username: Optional[str]
    first_name: Optional[str]
    subscribed_at: str
    is_active: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Database Class
# ─────────────────────────────────────────────────────────────────────────────

class Database:
    """SQLite database handler for Signal Hub."""

    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initialize database tables."""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Signals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_low REAL NOT NULL,
                entry_high REAL NOT NULL,
                stop_loss REAL NOT NULL,
                tp1 REAL NOT NULL,
                tp2 REAL,
                tp3 REAL,
                timeframe TEXT NOT NULL,
                confidence TEXT NOT NULL,
                note TEXT,
                created_at TEXT NOT NULL,
                status TEXT DEFAULT 'ACTIVE',
                tp1_hit INTEGER DEFAULT 0,
                tp2_hit INTEGER DEFAULT 0,
                tp3_hit INTEGER DEFAULT 0,
                closed_at TEXT,
                result_pct REAL
            )
        """)

        # Subscribers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS subscribers (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                subscribed_at TEXT NOT NULL,
                is_active INTEGER DEFAULT 1
            )
        """)

        # Signal updates log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER NOT NULL,
                update_type TEXT NOT NULL,
                message TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (signal_id) REFERENCES signals (id)
            )
        """)

        conn.commit()
        conn.close()

    # ─────────────────────────────────────────────────────────────────────────
    # Signal Methods
    # ─────────────────────────────────────────────────────────────────────────

    def create_signal(self, signal: Signal) -> int:
        """Create a new signal and return its ID."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO signals (
                pair, direction, entry_low, entry_high, stop_loss,
                tp1, tp2, tp3, timeframe, confidence, note, created_at, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal.pair, signal.direction, signal.entry_low, signal.entry_high,
            signal.stop_loss, signal.tp1, signal.tp2, signal.tp3,
            signal.timeframe, signal.confidence, signal.note,
            signal.created_at, signal.status
        ))

        signal_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return signal_id

    def get_signal(self, signal_id: int) -> Optional[Signal]:
        """Get a signal by ID."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM signals WHERE id = ?", (signal_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return Signal(
                id=row["id"],
                pair=row["pair"],
                direction=row["direction"],
                entry_low=row["entry_low"],
                entry_high=row["entry_high"],
                stop_loss=row["stop_loss"],
                tp1=row["tp1"],
                tp2=row["tp2"],
                tp3=row["tp3"],
                timeframe=row["timeframe"],
                confidence=row["confidence"],
                note=row["note"],
                created_at=row["created_at"],
                status=row["status"],
                tp1_hit=bool(row["tp1_hit"]),
                tp2_hit=bool(row["tp2_hit"]),
                tp3_hit=bool(row["tp3_hit"]),
                closed_at=row["closed_at"],
                result_pct=row["result_pct"]
            )
        return None

    def get_active_signals(self) -> list[Signal]:
        """Get all active signals."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM signals WHERE status = 'ACTIVE' ORDER BY created_at DESC"
        )
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_signal(row) for row in rows]

    def get_recent_signals(self, limit: int = 10) -> list[Signal]:
        """Get recent signals."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM signals ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_signal(row) for row in rows]

    def update_signal_status(
        self,
        signal_id: int,
        status: str,
        tp1_hit: bool = None,
        tp2_hit: bool = None,
        tp3_hit: bool = None,
        result_pct: float = None
    ):
        """Update a signal's status."""
        conn = self.get_connection()
        cursor = conn.cursor()

        updates = ["status = ?"]
        values = [status]

        if tp1_hit is not None:
            updates.append("tp1_hit = ?")
            values.append(int(tp1_hit))

        if tp2_hit is not None:
            updates.append("tp2_hit = ?")
            values.append(int(tp2_hit))

        if tp3_hit is not None:
            updates.append("tp3_hit = ?")
            values.append(int(tp3_hit))

        if result_pct is not None:
            updates.append("result_pct = ?")
            values.append(result_pct)

        if status in ["STOPPED", "TP1_HIT", "TP2_HIT", "TP3_HIT", "CANCELLED"]:
            updates.append("closed_at = ?")
            values.append(datetime.utcnow().isoformat())

        values.append(signal_id)

        cursor.execute(
            f"UPDATE signals SET {', '.join(updates)} WHERE id = ?",
            values
        )

        conn.commit()
        conn.close()

    def log_signal_update(self, signal_id: int, update_type: str, message: str = ""):
        """Log a signal update."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO signal_updates (signal_id, update_type, message, created_at)
            VALUES (?, ?, ?, ?)
        """, (signal_id, update_type, message, datetime.utcnow().isoformat()))

        conn.commit()
        conn.close()

    def _row_to_signal(self, row) -> Signal:
        """Convert a database row to a Signal object."""
        return Signal(
            id=row["id"],
            pair=row["pair"],
            direction=row["direction"],
            entry_low=row["entry_low"],
            entry_high=row["entry_high"],
            stop_loss=row["stop_loss"],
            tp1=row["tp1"],
            tp2=row["tp2"],
            tp3=row["tp3"],
            timeframe=row["timeframe"],
            confidence=row["confidence"],
            note=row["note"],
            created_at=row["created_at"],
            status=row["status"],
            tp1_hit=bool(row["tp1_hit"]),
            tp2_hit=bool(row["tp2_hit"]),
            tp3_hit=bool(row["tp3_hit"]),
            closed_at=row["closed_at"],
            result_pct=row["result_pct"]
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Subscriber Methods
    # ─────────────────────────────────────────────────────────────────────────

    def add_subscriber(self, user_id: int, username: str = None, first_name: str = None) -> bool:
        """Add or reactivate a subscriber. Returns True if new, False if existing."""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Check if exists
        cursor.execute("SELECT * FROM subscribers WHERE user_id = ?", (user_id,))
        existing = cursor.fetchone()

        if existing:
            # Reactivate if inactive
            cursor.execute(
                "UPDATE subscribers SET is_active = 1, username = ?, first_name = ? WHERE user_id = ?",
                (username, first_name, user_id)
            )
            is_new = not existing["is_active"]
        else:
            # New subscriber
            cursor.execute("""
                INSERT INTO subscribers (user_id, username, first_name, subscribed_at, is_active)
                VALUES (?, ?, ?, ?, 1)
            """, (user_id, username, first_name, datetime.utcnow().isoformat()))
            is_new = True

        conn.commit()
        conn.close()

        return is_new

    def remove_subscriber(self, user_id: int) -> bool:
        """Deactivate a subscriber. Returns True if was active."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT is_active FROM subscribers WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()

        if row and row["is_active"]:
            cursor.execute(
                "UPDATE subscribers SET is_active = 0 WHERE user_id = ?",
                (user_id,)
            )
            conn.commit()
            conn.close()
            return True

        conn.close()
        return False

    def get_active_subscribers(self) -> list[Subscriber]:
        """Get all active subscribers."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM subscribers WHERE is_active = 1")
        rows = cursor.fetchall()
        conn.close()

        return [
            Subscriber(
                user_id=row["user_id"],
                username=row["username"],
                first_name=row["first_name"],
                subscribed_at=row["subscribed_at"],
                is_active=bool(row["is_active"])
            )
            for row in rows
        ]

    def get_subscriber_count(self) -> int:
        """Get count of active subscribers."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) as count FROM subscribers WHERE is_active = 1")
        row = cursor.fetchone()
        conn.close()

        return row["count"] if row else 0

    def is_subscribed(self, user_id: int) -> bool:
        """Check if a user is subscribed."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT is_active FROM subscribers WHERE user_id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        conn.close()

        return bool(row and row["is_active"])

    # ─────────────────────────────────────────────────────────────────────────
    # Stats Methods
    # ─────────────────────────────────────────────────────────────────────────

    def get_signal_stats(self) -> dict:
        """Get signal performance statistics."""
        conn = self.get_connection()
        cursor = conn.cursor()

        stats = {
            "total_signals": 0,
            "active_signals": 0,
            "closed_signals": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "tp1_hits": 0,
            "tp2_hits": 0,
            "tp3_hits": 0,
            "avg_result_pct": 0.0,
            "current_streak": 0,
            "streak_type": "none"
        }

        # Total and active
        cursor.execute("SELECT COUNT(*) as count FROM signals")
        stats["total_signals"] = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) as count FROM signals WHERE status = 'ACTIVE'")
        stats["active_signals"] = cursor.fetchone()["count"]

        # Closed signals
        cursor.execute("""
            SELECT COUNT(*) as count FROM signals
            WHERE status IN ('TP1_HIT', 'TP2_HIT', 'TP3_HIT', 'STOPPED')
        """)
        stats["closed_signals"] = cursor.fetchone()["count"]

        # Wins (TP1+ hit)
        cursor.execute("SELECT COUNT(*) as count FROM signals WHERE tp1_hit = 1")
        stats["wins"] = cursor.fetchone()["count"]
        stats["tp1_hits"] = stats["wins"]

        cursor.execute("SELECT COUNT(*) as count FROM signals WHERE tp2_hit = 1")
        stats["tp2_hits"] = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) as count FROM signals WHERE tp3_hit = 1")
        stats["tp3_hits"] = cursor.fetchone()["count"]

        # Losses (stopped out)
        cursor.execute("SELECT COUNT(*) as count FROM signals WHERE status = 'STOPPED'")
        stats["losses"] = cursor.fetchone()["count"]

        # Win rate
        if stats["closed_signals"] > 0:
            stats["win_rate"] = (stats["wins"] / stats["closed_signals"]) * 100

        # Average result
        cursor.execute("""
            SELECT AVG(result_pct) as avg FROM signals
            WHERE result_pct IS NOT NULL
        """)
        row = cursor.fetchone()
        if row["avg"]:
            stats["avg_result_pct"] = round(row["avg"], 2)

        # Current streak
        cursor.execute("""
            SELECT status FROM signals
            WHERE status IN ('TP1_HIT', 'TP2_HIT', 'TP3_HIT', 'STOPPED')
            ORDER BY closed_at DESC
            LIMIT 20
        """)
        rows = cursor.fetchall()

        if rows:
            first_result = "win" if rows[0]["status"] != "STOPPED" else "loss"
            streak = 0
            for row in rows:
                is_win = row["status"] != "STOPPED"
                if (first_result == "win" and is_win) or (first_result == "loss" and not is_win):
                    streak += 1
                else:
                    break

            stats["current_streak"] = streak
            stats["streak_type"] = first_result

        conn.close()
        return stats

    def get_monthly_stats(self, year: int, month: int) -> dict:
        """Get stats for a specific month."""
        conn = self.get_connection()
        cursor = conn.cursor()

        start_date = f"{year}-{month:02d}-01"
        if month == 12:
            end_date = f"{year + 1}-01-01"
        else:
            end_date = f"{year}-{month + 1:02d}-01"

        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN tp1_hit = 1 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN status = 'STOPPED' THEN 1 ELSE 0 END) as losses,
                AVG(result_pct) as avg_result
            FROM signals
            WHERE created_at >= ? AND created_at < ?
        """, (start_date, end_date))

        row = cursor.fetchone()
        conn.close()

        return {
            "total": row["total"] or 0,
            "wins": row["wins"] or 0,
            "losses": row["losses"] or 0,
            "avg_result": round(row["avg_result"] or 0, 2)
        }


# Global database instance
db = Database()

"""
═══════════════════════════════════════════════════════════════════════════════
⚡ Zenkai Signal Hub — Statistics Module
© 2026 Zenkai Corporation
═══════════════════════════════════════════════════════════════════════════════

Performance calculations and statistics formatting.
"""

from datetime import datetime, timedelta
from typing import Optional

from db import db


def get_all_time_stats() -> dict:
    """Get all-time signal statistics."""
    return db.get_signal_stats()


def get_monthly_stats(year: int = None, month: int = None) -> dict:
    """Get statistics for a specific month."""
    if year is None or month is None:
        now = datetime.utcnow()
        year = now.year
        month = now.month

    return db.get_monthly_stats(year, month)


def get_performance_summary() -> str:
    """Generate a performance summary string."""
    stats = get_all_time_stats()
    monthly = get_monthly_stats()

    summary = []

    # All-time
    summary.append("📊 ALL-TIME PERFORMANCE")
    summary.append(f"   Signals: {stats['total_signals']}")
    summary.append(f"   Win Rate: {stats['win_rate']:.1f}%")
    summary.append(f"   Avg Result: {stats['avg_result_pct']:+.2f}%")

    # Monthly
    now = datetime.utcnow()
    summary.append(f"\n📅 {now.strftime('%B %Y').upper()}")
    summary.append(f"   Signals: {monthly['total']}")
    if monthly['total'] > 0:
        win_rate = (monthly['wins'] / monthly['total']) * 100
        summary.append(f"   Win Rate: {win_rate:.1f}%")

    return "\n".join(summary)


def calculate_risk_reward(entry: float, stop_loss: float, take_profit: float) -> float:
    """Calculate risk-reward ratio."""
    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    return reward / risk if risk > 0 else 0


def get_streak_info() -> tuple[int, str]:
    """Get current win/loss streak."""
    stats = db.get_signal_stats()
    return stats.get("current_streak", 0), stats.get("streak_type", "none")

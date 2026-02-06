"""
═══════════════════════════════════════════════════════════════════════════════
⚡ Zenkai Signal Hub — Signal Handler
© 2026 Zenkai Corporation
═══════════════════════════════════════════════════════════════════════════════

Handles signal creation, formatting, and updates.
"""

from datetime import datetime
from typing import Optional
from dataclasses import dataclass

from db import db, Signal
from config import DISCLAIMER_SHORT, BOT_USERNAME

# ─────────────────────────────────────────────────────────────────────────────
# Signal Builder State
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SignalDraft:
    """Draft signal being built by admin."""
    pair: Optional[str] = None
    direction: Optional[str] = None
    entry_low: Optional[float] = None
    entry_high: Optional[float] = None
    stop_loss: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    timeframe: Optional[str] = None
    confidence: Optional[str] = None
    note: Optional[str] = None
    step: int = 0  # Current step in the builder

    def is_complete(self) -> bool:
        """Check if all required fields are filled."""
        return all([
            self.pair,
            self.direction,
            self.entry_low,
            self.entry_high,
            self.stop_loss,
            self.tp1,
            self.timeframe,
            self.confidence
        ])


# Store active signal drafts by user_id
active_drafts: dict[int, SignalDraft] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Signal Formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_signal_message(signal: Signal) -> str:
    """Format a signal into a broadcast message."""
    # Direction emoji and text
    if signal.direction == "LONG":
        direction_emoji = "🟢"
        direction_text = "LONG SIGNAL"
    else:
        direction_emoji = "🔴"
        direction_text = "SHORT SIGNAL"

    # Confidence emoji
    confidence_emoji = {
        "LOW": "⚪",
        "MEDIUM": "🟡",
        "HIGH": "🟢"
    }.get(signal.confidence, "⚪")

    # Build message
    message = f"""
{direction_emoji} <b>{direction_text} — {signal.pair}</b>

📊 Timeframe: <b>{signal.timeframe}</b>
🎯 Confidence: <b>{confidence_emoji} {signal.confidence}</b>

━━━━━━━━━━━━━━━━━━━━━━

<b>Entry Zone:</b> ${signal.entry_low:,.2f} — ${signal.entry_high:,.2f}
🛑 <b>Stop Loss:</b> ${signal.stop_loss:,.2f}
✅ <b>TP1:</b> ${signal.tp1:,.2f}"""

    if signal.tp2:
        message += f"\n✅ <b>TP2:</b> ${signal.tp2:,.2f}"

    if signal.tp3:
        message += f"\n✅ <b>TP3:</b> ${signal.tp3:,.2f}"

    # Calculate R:R for TP1
    entry_mid = (signal.entry_low + signal.entry_high) / 2
    risk = abs(entry_mid - signal.stop_loss)
    reward = abs(signal.tp1 - entry_mid)
    rr_ratio = reward / risk if risk > 0 else 0

    message += f"""

📐 <b>R:R (TP1):</b> 1:{rr_ratio:.1f}

━━━━━━━━━━━━━━━━━━━━━━"""

    if signal.note:
        message += f"""

📝 <b>Note:</b>
{signal.note}

━━━━━━━━━━━━━━━━━━━━━━"""

    message += f"""

⚡ <b>Zenkai Signal Hub</b> | {BOT_USERNAME}
{DISCLAIMER_SHORT}

🆔 Signal ID: <code>#{signal.id}</code>
"""

    return message.strip()


def format_signal_update(signal: Signal, update_type: str, extra_info: str = "") -> str:
    """Format a signal update message."""
    # Direction info
    direction_emoji = "🟢" if signal.direction == "LONG" else "🔴"

    message = f"""
📊 <b>SIGNAL UPDATE — {signal.pair} {signal.direction}</b>

━━━━━━━━━━━━━━━━━━━━━━

"""

    if update_type == "TP1_HIT":
        # Calculate profit
        entry_mid = (signal.entry_low + signal.entry_high) / 2
        profit_pct = abs((signal.tp1 - entry_mid) / entry_mid) * 100

        message += f"""✅ <b>TP1 HIT</b> — ${signal.tp1:,.2f}
📈 Profit: <b>+{profit_pct:.1f}%</b>
"""

        # Show remaining targets
        remaining = []
        if signal.tp2 and not signal.tp2_hit:
            remaining.append(f"✅ TP2: ${signal.tp2:,.2f}")
        if signal.tp3 and not signal.tp3_hit:
            remaining.append(f"✅ TP3: ${signal.tp3:,.2f}")

        if remaining:
            message += f"""
<b>Remaining Targets:</b>
{chr(10).join(remaining)}
"""

    elif update_type == "TP2_HIT":
        entry_mid = (signal.entry_low + signal.entry_high) / 2
        profit_pct = abs((signal.tp2 - entry_mid) / entry_mid) * 100

        message += f"""✅ <b>TP2 HIT</b> — ${signal.tp2:,.2f}
📈 Profit: <b>+{profit_pct:.1f}%</b>
"""

        if signal.tp3 and not signal.tp3_hit:
            message += f"""
<b>Remaining Target:</b>
✅ TP3: ${signal.tp3:,.2f}
"""

    elif update_type == "TP3_HIT":
        entry_mid = (signal.entry_low + signal.entry_high) / 2
        profit_pct = abs((signal.tp3 - entry_mid) / entry_mid) * 100

        message += f"""✅ <b>TP3 HIT — FULL TARGET!</b> 🎯
📈 Profit: <b>+{profit_pct:.1f}%</b>

🏆 All targets reached!
"""

    elif update_type == "STOPPED":
        entry_mid = (signal.entry_low + signal.entry_high) / 2
        loss_pct = abs((signal.stop_loss - entry_mid) / entry_mid) * 100

        message += f"""❌ <b>STOPPED OUT</b> — ${signal.stop_loss:,.2f}
📉 Loss: <b>-{loss_pct:.1f}%</b>
"""

    elif update_type == "CANCELLED":
        message += f"""⚪ <b>SIGNAL CANCELLED</b>

{extra_info if extra_info else "Signal invalidated - do not trade."}
"""

    elif update_type == "BREAKEVEN":
        message += f"""⚖️ <b>STOP MOVED TO BREAKEVEN</b>

Risk eliminated. Let profits run!
"""

    if extra_info and update_type not in ["CANCELLED"]:
        message += f"""
📝 {extra_info}
"""

    message += f"""
━━━━━━━━━━━━━━━━━━━━━━

⚡ <b>Zenkai Signal Hub</b>
🆔 Signal ID: <code>#{signal.id}</code>
"""

    return message.strip()


def format_stats_message(stats: dict) -> str:
    """Format statistics into a message."""
    # Win rate bar
    win_rate = stats.get("win_rate", 0)
    bar_filled = int(win_rate / 10)
    bar_empty = 10 - bar_filled
    win_bar = "🟩" * bar_filled + "⬜" * bar_empty

    # Streak
    streak = stats.get("current_streak", 0)
    streak_type = stats.get("streak_type", "none")
    if streak_type == "win":
        streak_text = f"🔥 {streak}W"
    elif streak_type == "loss":
        streak_text = f"❄️ {streak}L"
    else:
        streak_text = "—"

    message = f"""
📊 <b>ZENKAI SIGNAL HUB — PERFORMANCE</b>

━━━━━━━━━━━━━━━━━━━━━━

<b>📈 Overall Statistics</b>

Total Signals: <b>{stats.get('total_signals', 0)}</b>
Active Signals: <b>{stats.get('active_signals', 0)}</b>
Closed Signals: <b>{stats.get('closed_signals', 0)}</b>

━━━━━━━━━━━━━━━━━━━━━━

<b>🎯 Win Rate</b>

{win_bar} <b>{win_rate:.1f}%</b>

Wins: <b>{stats.get('wins', 0)}</b> ✅
Losses: <b>{stats.get('losses', 0)}</b> ❌

━━━━━━━━━━━━━━━━━━━━━━

<b>📐 Target Hits</b>

TP1 Hits: <b>{stats.get('tp1_hits', 0)}</b>
TP2 Hits: <b>{stats.get('tp2_hits', 0)}</b>
TP3 Hits: <b>{stats.get('tp3_hits', 0)}</b>

━━━━━━━━━━━━━━━━━━━━━━

<b>📊 Performance</b>

Avg Result: <b>{stats.get('avg_result_pct', 0):+.1f}%</b>
Current Streak: <b>{streak_text}</b>

━━━━━━━━━━━━━━━━━━━━━━

⚡ <b>Zenkai Signal Hub</b>
<i>Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC</i>
"""

    return message.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Signal Operations
# ─────────────────────────────────────────────────────────────────────────────

def create_signal_from_draft(draft: SignalDraft) -> Signal:
    """Create and save a signal from a draft."""
    signal = Signal(
        id=None,
        pair=draft.pair,
        direction=draft.direction,
        entry_low=draft.entry_low,
        entry_high=draft.entry_high,
        stop_loss=draft.stop_loss,
        tp1=draft.tp1,
        tp2=draft.tp2,
        tp3=draft.tp3,
        timeframe=draft.timeframe,
        confidence=draft.confidence,
        note=draft.note,
        created_at=datetime.utcnow().isoformat(),
        status="ACTIVE"
    )

    signal.id = db.create_signal(signal)
    return signal


def update_signal(signal_id: int, update_type: str, extra_info: str = "") -> Optional[Signal]:
    """Update a signal status and return the updated signal."""
    signal = db.get_signal(signal_id)
    if not signal:
        return None

    # Calculate result percentage based on update type
    entry_mid = (signal.entry_low + signal.entry_high) / 2
    result_pct = None

    if update_type == "TP1_HIT":
        result_pct = abs((signal.tp1 - entry_mid) / entry_mid) * 100
        db.update_signal_status(signal_id, update_type, tp1_hit=True, result_pct=result_pct)

    elif update_type == "TP2_HIT":
        result_pct = abs((signal.tp2 - entry_mid) / entry_mid) * 100
        db.update_signal_status(signal_id, update_type, tp1_hit=True, tp2_hit=True, result_pct=result_pct)

    elif update_type == "TP3_HIT":
        result_pct = abs((signal.tp3 - entry_mid) / entry_mid) * 100
        db.update_signal_status(signal_id, update_type, tp1_hit=True, tp2_hit=True, tp3_hit=True, result_pct=result_pct)

    elif update_type == "STOPPED":
        result_pct = -abs((signal.stop_loss - entry_mid) / entry_mid) * 100
        db.update_signal_status(signal_id, update_type, result_pct=result_pct)

    elif update_type == "CANCELLED":
        db.update_signal_status(signal_id, update_type)

    elif update_type == "BREAKEVEN":
        # Just log it, don't close the signal
        pass

    # Log the update
    db.log_signal_update(signal_id, update_type, extra_info)

    return db.get_signal(signal_id)


def get_active_signals() -> list[Signal]:
    """Get all active signals."""
    return db.get_active_signals()


def get_signal(signal_id: int) -> Optional[Signal]:
    """Get a signal by ID."""
    return db.get_signal(signal_id)


def get_stats() -> dict:
    """Get signal statistics."""
    return db.get_signal_stats()

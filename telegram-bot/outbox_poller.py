"""
⚡ Zenkai Signal Hub — Outbox Poller
Watches C:\Zenkai\alphatrader\signals\outbox\ for new signal JSON files
and auto-broadcasts them to Telegram subscribers.

Same pipeline as the Discord bot — both read from the same outbox.
Discord bot handles file moves. We track what we've seen in seen_signals.json.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode

from config import DISCLAIMER_SHORT, BOT_USERNAME

logger = logging.getLogger("zenkai-signal-hub.outbox")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SIGNALS_OUTBOX = Path(os.getenv(
    "SIGNALS_OUTBOX",
    r"C:\Zenkai\alphatrader\signals\outbox"
))
SIGNALS_PROCESSED = Path(os.getenv(
    "SIGNALS_PROCESSED",
    r"C:\Zenkai\alphatrader\signals\processed"
))

POLL_INTERVAL = 30  # seconds — matches Discord bot
HIGH_CONFIDENCE_THRESHOLD = 0.80


# ─────────────────────────────────────────────────────────────────────────────
# Signal Formatting (for automated pipeline signals)
# ─────────────────────────────────────────────────────────────────────────────

def format_pipeline_signal(signal: dict) -> str:
    """
    Format a signal JSON from AlphaTrader into a Telegram message.
    Handles the automated pipeline format (different from manual /signal).
    """
    direction = signal.get("direction", "LONG")
    pair = signal.get("pair", "UNKNOWN")
    entry = signal.get("entry", 0)
    stop_loss = signal.get("stop_loss", 0)
    take_profits = signal.get("take_profit", [])
    confidence = signal.get("confidence", 0)
    timeframe = signal.get("timeframe", "4H")
    strategy = signal.get("strategy", "AlphaTrader")
    notes = signal.get("notes", "")
    regime = signal.get("regime", "")
    rr_ratios = signal.get("risk_reward", [])
    risk_pct = signal.get("risk_pct", 0)
    signal_id = signal.get("id", "N/A")

    # Direction styling
    if direction == "LONG":
        dir_emoji = "🟢"
        dir_text = "LONG SIGNAL"
    else:
        dir_emoji = "🔴"
        dir_text = "SHORT SIGNAL"

    # Confidence styling
    conf_pct = confidence * 100 if confidence <= 1 else confidence
    if conf_pct >= 80:
        conf_emoji = "🔥"
        conf_label = "HIGH"
    elif conf_pct >= 60:
        conf_emoji = "⚡"
        conf_label = "MEDIUM"
    else:
        conf_emoji = "⚪"
        conf_label = "LOW"

    # Confidence bar
    filled = int(conf_pct / 10)
    bar = "█" * filled + "░" * (10 - filled)

    # Build message
    msg = (
        f"{dir_emoji} <b>{dir_text} — {pair}</b>\n"
        f"\n"
        f"📊 Timeframe: <b>{timeframe}</b>\n"
        f"🎯 Confidence: <b>{conf_emoji} {conf_label} ({conf_pct:.0f}%)</b>\n"
        f"<code>[{bar}]</code>\n"
        f"\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
        f"📍 <b>Entry:</b> <code>${entry:,.2f}</code>\n"
        f"🛑 <b>Stop Loss:</b> <code>${stop_loss:,.2f}</code>\n"
    )

    # Take profit levels
    for i, tp in enumerate(take_profits, 1):
        rr = f" (R:R 1:{rr_ratios[i-1]:.1f})" if i - 1 < len(rr_ratios) else ""
        msg += f"✅ <b>TP{i}:</b> <code>${tp:,.2f}</code>{rr}\n"

    msg += f"\n"

    # Risk info
    if risk_pct > 0:
        msg += f"📐 <b>Risk:</b> {risk_pct:.2f}%\n"

    # Strategy + regime
    if strategy:
        msg += f"🤖 <b>Strategy:</b> {strategy}\n"
    if regime:
        regime_display = regime.replace("_", " ").title()
        msg += f"📈 <b>Regime:</b> {regime_display}\n"

    # Notes
    if notes:
        msg += f"\n📝 <i>{notes}</i>\n"

    msg += (
        f"\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
        f"⚡ <b>Zenkai Signal Hub</b> | {BOT_USERNAME}\n"
        f"{DISCLAIMER_SHORT}\n"
        f"\n"
        f"🆔 <code>{signal_id}</code>"
    )

    return msg


def build_signal_keyboard(signal: dict) -> InlineKeyboardMarkup:
    """Build inline keyboard with TradingView link."""
    pair = signal.get("pair", "BTCUSDT")
    tv_symbol = pair.replace("/", "")

    keyboard = [
        [InlineKeyboardButton(
            "📊 View on TradingView",
            url=f"https://www.tradingview.com/chart/?symbol=BINANCE:{tv_symbol}"
        )],
    ]

    return InlineKeyboardMarkup(keyboard)


# ─────────────────────────────────────────────────────────────────────────────
# Outbox Poller
# ─────────────────────────────────────────────────────────────────────────────

class OutboxPoller:
    """
    Polls the AlphaTrader signal outbox for new JSON files.
    When found, formats and broadcasts to Telegram subscribers.

    IMPORTANT: Does NOT move files. The Discord bot handles file moves.
    Instead tracks seen filenames in seen_signals.json to avoid duplicates.
    Also checks the processed/ dir in case Discord moved a file before we saw it.
    """

    def __init__(self, broadcast_callback: Callable, app=None):
        """
        Args:
            broadcast_callback: async function(message, parse_mode, reply_markup)
                               that sends to all subscribers
            app: The telegram Application instance
        """
        self.broadcast = broadcast_callback
        self.app = app
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.signals_processed = 0

        # Track which files we've already sent
        self._seen_file = Path(os.getenv(
            "SEEN_SIGNALS_FILE",
            r"C:\Zenkai\telegram-bot\data\seen_signals.json"
        ))
        self._seen_file.parent.mkdir(parents=True, exist_ok=True)
        self._seen: set = self._load_seen()

        # Ensure signal dirs exist
        SIGNALS_OUTBOX.mkdir(parents=True, exist_ok=True)
        SIGNALS_PROCESSED.mkdir(parents=True, exist_ok=True)

    def _load_seen(self) -> set:
        """Load set of already-processed signal filenames."""
        try:
            if self._seen_file.exists():
                with open(self._seen_file, "r") as f:
                    return set(json.load(f))
        except Exception:
            pass
        return set()

    def _save_seen(self):
        """Persist seen signals to disk."""
        try:
            # Keep only last 500 to prevent unbounded growth
            trimmed = sorted(self._seen)[-500:]
            with open(self._seen_file, "w") as f:
                json.dump(trimmed, f)
        except Exception as e:
            logger.warning(f"Could not save seen signals: {e}")

    async def start(self):
        """Start the outbox polling loop."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(
            f"Outbox poller started — watching {SIGNALS_OUTBOX} "
            f"every {POLL_INTERVAL}s"
        )

    async def stop(self):
        """Stop the outbox polling loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(
            f"Outbox poller stopped — processed {self.signals_processed} "
            f"signals this session"
        )

    async def _poll_loop(self):
        """Main polling loop."""
        while self._running:
            try:
                await self._check_outbox()
            except Exception as e:
                logger.error(f"Outbox poll error: {e}", exc_info=True)

            await asyncio.sleep(POLL_INTERVAL)

    async def _check_outbox(self):
        """Check outbox AND processed dir for new signal files we haven't sent."""
        # Check both — Discord bot may have already moved the file
        json_files = sorted(SIGNALS_OUTBOX.glob("*.json"))
        json_files += sorted(SIGNALS_PROCESSED.glob("*.json"))

        for filepath in json_files:
            if filepath.name in self._seen:
                continue  # Already sent
            try:
                await self._process_signal_file(filepath)
            except Exception as e:
                logger.error(f"Failed to process {filepath.name}: {e}")

    async def _process_signal_file(self, filepath: Path):
        """Process a single signal JSON file."""
        # Read the signal
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                signal = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {filepath.name}: {e}")
            self._seen.add(filepath.name)
            self._save_seen()
            return

        # Validate minimum fields
        required = ["pair", "direction", "entry", "stop_loss"]
        if not all(k in signal for k in required):
            logger.warning(
                f"Signal {filepath.name} missing required fields, skipping"
            )
            self._seen.add(filepath.name)
            self._save_seen()
            return

        signal_id = signal.get("id", filepath.stem)
        pair = signal.get("pair", "?")
        direction = signal.get("direction", "?")
        confidence = signal.get("confidence", 0)

        logger.info(
            f"New signal: {signal_id} | {pair} {direction} | "
            f"conf={confidence:.2f}"
        )

        # Format the message
        message = format_pipeline_signal(signal)
        keyboard = build_signal_keyboard(signal)

        # Broadcast to all subscribers
        try:
            await self.broadcast(
                message=message,
                parse_mode=ParseMode.HTML,
                reply_markup=keyboard
            )
            logger.info(f"Broadcasted signal {signal_id} to subscribers")
        except Exception as e:
            logger.error(f"Broadcast failed for {signal_id}: {e}")

        # Mark as seen (Discord bot handles file moves)
        self._seen.add(filepath.name)
        self._save_seen()
        self.signals_processed += 1

    def get_stats(self) -> dict:
        """Get poller stats."""
        pending = len(list(SIGNALS_OUTBOX.glob("*.json")))
        processed = len(list(SIGNALS_PROCESSED.glob("*.json")))
        return {
            "pending": pending,
            "processed_total": processed,
            "processed_session": self.signals_processed,
            "running": self._running,
            "seen_count": len(self._seen),
        }

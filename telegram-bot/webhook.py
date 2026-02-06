"""
═══════════════════════════════════════════════════════════════════════════════
⚡ Zenkai Signal Hub — AlphaTrader Integration (Webhook)
© 2026 Zenkai Corporation
═══════════════════════════════════════════════════════════════════════════════

Placeholder module for AlphaTrader integration.
When the server comes back online, AlphaTrader will push signals here.

Integration Options:
1. REST Webhook - AlphaTrader POSTs signals to this endpoint
2. File Watcher - Watch for new signal files
3. Direct Function Call - Import and call from AlphaTrader
4. Message Queue - Redis/RabbitMQ for async signals
"""

import json
import logging
from datetime import datetime
from typing import Optional, Callable
from dataclasses import dataclass

from db import Signal
from signal_handler import create_signal_from_draft, SignalDraft

logger = logging.getLogger("zenkai-signal-hub.webhook")

# ─────────────────────────────────────────────────────────────────────────────
# Callback Registration
# ─────────────────────────────────────────────────────────────────────────────

# This will be set by the bot to broadcast signals
_signal_broadcast_callback: Optional[Callable] = None


def register_broadcast_callback(callback: Callable):
    """
    Register the callback function for broadcasting signals.
    Called by the bot during initialization.

    Args:
        callback: Async function that takes a Signal and broadcasts it
    """
    global _signal_broadcast_callback
    _signal_broadcast_callback = callback
    logger.info("Signal broadcast callback registered")


# ─────────────────────────────────────────────────────────────────────────────
# AlphaTrader Signal Format
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AlphaTraderSignal:
    """Signal format from AlphaTrader."""
    pair: str
    direction: str  # "LONG" or "SHORT"
    entry_zone: tuple[float, float]  # (low, high)
    stop_loss: float
    take_profits: list[float]  # [TP1, TP2, TP3] - can have 1-3
    timeframe: str
    confidence: float  # 0.0 - 1.0
    analysis: str  # AI-generated analysis text
    model_version: str
    timestamp: str


# ─────────────────────────────────────────────────────────────────────────────
# Signal Processing
# ─────────────────────────────────────────────────────────────────────────────

def confidence_to_level(confidence: float) -> str:
    """Convert numeric confidence to level string."""
    if confidence >= 0.75:
        return "HIGH"
    elif confidence >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"


async def process_alphatrader_signal(data: dict) -> Optional[Signal]:
    """
    Process a signal received from AlphaTrader.

    Expected data format:
    {
        "pair": "BTC/USDT",
        "direction": "LONG",
        "entry_zone": [94500, 95200],
        "stop_loss": 93100,
        "take_profits": [96800, 98500, 101000],
        "timeframe": "4H",
        "confidence": 0.85,
        "analysis": "CHoCH on 4H with unfilled bullish FVG...",
        "model_version": "alphatrader-v2.1",
        "timestamp": "2026-02-06T12:00:00Z"
    }
    """
    try:
        # Validate required fields
        required = ["pair", "direction", "entry_zone", "stop_loss", "take_profits", "timeframe", "confidence"]
        for field in required:
            if field not in data:
                logger.error(f"Missing required field: {field}")
                return None

        # Create signal draft
        draft = SignalDraft(
            pair=data["pair"],
            direction=data["direction"].upper(),
            entry_low=float(data["entry_zone"][0]),
            entry_high=float(data["entry_zone"][1]),
            stop_loss=float(data["stop_loss"]),
            tp1=float(data["take_profits"][0]),
            tp2=float(data["take_profits"][1]) if len(data["take_profits"]) > 1 else None,
            tp3=float(data["take_profits"][2]) if len(data["take_profits"]) > 2 else None,
            timeframe=data["timeframe"],
            confidence=confidence_to_level(data["confidence"]),
            note=data.get("analysis", "")
        )

        # Create signal in database
        signal = create_signal_from_draft(draft)
        logger.info(f"Created signal #{signal.id} from AlphaTrader: {signal.pair} {signal.direction}")

        # Broadcast if callback is registered
        if _signal_broadcast_callback:
            await _signal_broadcast_callback(signal)
            logger.info(f"Signal #{signal.id} broadcasted")

        return signal

    except Exception as e:
        logger.error(f"Error processing AlphaTrader signal: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Integration Methods (Placeholders)
# ─────────────────────────────────────────────────────────────────────────────

# Option 1: REST Webhook (use with Flask/FastAPI)
"""
To use with FastAPI:

from fastapi import FastAPI, HTTPException
from webhook import process_alphatrader_signal

app = FastAPI()

@app.post("/webhook/signal")
async def receive_signal(data: dict):
    signal = await process_alphatrader_signal(data)
    if signal:
        return {"status": "ok", "signal_id": signal.id}
    raise HTTPException(status_code=400, detail="Invalid signal data")
"""


# Option 2: File Watcher
"""
To use file watching:

import asyncio
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class SignalFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.endswith('.json'):
            with open(event.src_path) as f:
                data = json.load(f)
            asyncio.create_task(process_alphatrader_signal(data))

# In bot startup:
observer = Observer()
observer.schedule(SignalFileHandler(), '/path/to/signals', recursive=False)
observer.start()
"""


# Option 3: Direct Function Call (simplest)
async def receive_signal_from_alphatrader(
    pair: str,
    direction: str,
    entry_low: float,
    entry_high: float,
    stop_loss: float,
    tp1: float,
    tp2: float = None,
    tp3: float = None,
    timeframe: str = "4H",
    confidence: float = 0.5,
    analysis: str = ""
) -> Optional[Signal]:
    """
    Direct function call for AlphaTrader integration.

    Usage from AlphaTrader:
        from webhook import receive_signal_from_alphatrader

        await receive_signal_from_alphatrader(
            pair="BTC/USDT",
            direction="LONG",
            entry_low=94500,
            entry_high=95200,
            stop_loss=93100,
            tp1=96800,
            tp2=98500,
            tp3=101000,
            timeframe="4H",
            confidence=0.85,
            analysis="CHoCH on 4H..."
        )
    """
    data = {
        "pair": pair,
        "direction": direction,
        "entry_zone": [entry_low, entry_high],
        "stop_loss": stop_loss,
        "take_profits": [tp for tp in [tp1, tp2, tp3] if tp is not None],
        "timeframe": timeframe,
        "confidence": confidence,
        "analysis": analysis
    }

    return await process_alphatrader_signal(data)


# ─────────────────────────────────────────────────────────────────────────────
# Testing / Manual Signal
# ─────────────────────────────────────────────────────────────────────────────

async def create_test_signal() -> Optional[Signal]:
    """Create a test signal for development/testing."""
    test_data = {
        "pair": "BTC/USDT",
        "direction": "LONG",
        "entry_zone": [94500, 95200],
        "stop_loss": 93100,
        "take_profits": [96800, 98500, 101000],
        "timeframe": "4H",
        "confidence": 0.85,
        "analysis": "TEST SIGNAL - CHoCH on 4H with unfilled bullish FVG at entry zone. Strong institutional interest detected.",
        "model_version": "test",
        "timestamp": datetime.utcnow().isoformat()
    }

    return await process_alphatrader_signal(test_data)

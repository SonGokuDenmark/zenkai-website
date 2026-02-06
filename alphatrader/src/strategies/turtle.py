"""
Donchian Channel / Turtle Trading Strategy.

The classic trend-following system used by the Turtle traders.
"""

from typing import List, Dict, Any
import pandas as pd
import numpy as np

from .base import (
    SignalGenerator, Signal, SignalDirection,
    donchian_channel, atr
)


class TurtleStrategy(SignalGenerator):
    """
    Donchian Channel / Turtle Breakout Strategy.

    Original Turtle Trading rules:
    - System 1: 20-day breakout for entry, 10-day for exit
    - System 2: 55-day breakout for entry, 20-day for exit

    This implementation uses System 1 by default.

    Parameters:
        entry_period: Period for entry channel (default: 20)
        exit_period: Period for exit channel (default: 10)
        atr_period: ATR period for position sizing (default: 20)
        atr_stop_mult: ATR multiplier for stop loss (default: 2)
    """

    def __init__(
        self,
        entry_period: int = 20,
        exit_period: int = 10,
        atr_period: int = 20,
        atr_stop_mult: float = 2.0,
        **kwargs
    ):
        super().__init__(
            name="turtle",
            entry_period=entry_period,
            exit_period=exit_period,
            atr_period=atr_period,
            atr_stop_mult=atr_stop_mult,
            **kwargs
        )
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Donchian channel indicators."""
        df = df.copy()

        # Entry channel
        entry_upper, entry_lower = donchian_channel(
            df["high"], df["low"], self.entry_period
        )
        df["donchian_upper"] = entry_upper
        df["donchian_lower"] = entry_lower

        # Exit channel
        exit_upper, exit_lower = donchian_channel(
            df["high"], df["low"], self.exit_period
        )
        df["exit_upper"] = exit_upper
        df["exit_lower"] = exit_lower

        # Previous channel values
        df["donchian_upper_prev"] = df["donchian_upper"].shift(1)
        df["donchian_lower_prev"] = df["donchian_lower"].shift(1)

        # ATR for stop loss
        df["atr"] = atr(df["high"], df["low"], df["close"], self.atr_period)

        return df

    def generate_signals(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> List[Signal]:
        """Generate Turtle breakout signals."""
        df = self.compute_indicators(df)
        signals = []

        for idx in range(1, len(df)):
            row = df.iloc[idx]
            prev_row = df.iloc[idx - 1]

            # Skip if indicators not ready
            if pd.isna(row["donchian_upper"]) or pd.isna(row["donchian_lower"]):
                continue

            signal_dir = SignalDirection.NEUTRAL
            confidence = 0.5

            # Breakout above upper channel - LONG
            if (row["high"] > prev_row["donchian_upper"] and
                prev_row["high"] <= prev_row["donchian_upper_prev"]):

                signal_dir = SignalDirection.LONG
                # Higher confidence if breaking multi-day high
                channel_width = prev_row["donchian_upper"] - prev_row["donchian_lower"]
                if channel_width > 0:
                    breakout_strength = (row["close"] - prev_row["donchian_upper"]) / channel_width
                    confidence = min(0.5 + breakout_strength * 0.3, 0.8)
                else:
                    confidence = 0.55

            # Breakout below lower channel - SHORT
            elif (row["low"] < prev_row["donchian_lower"] and
                  prev_row["low"] >= prev_row["donchian_lower_prev"]):

                signal_dir = SignalDirection.SHORT
                channel_width = prev_row["donchian_upper"] - prev_row["donchian_lower"]
                if channel_width > 0:
                    breakout_strength = (prev_row["donchian_lower"] - row["close"]) / channel_width
                    confidence = min(0.5 + breakout_strength * 0.3, 0.8)
                else:
                    confidence = 0.55

            if signal_dir != SignalDirection.NEUTRAL:
                entry_price = row["close"]
                atr_val = row["atr"] if not pd.isna(row["atr"]) else entry_price * 0.02

                if signal_dir == SignalDirection.LONG:
                    stop_loss = entry_price - self.atr_stop_mult * atr_val
                    # Use exit channel as take profit reference
                    take_profit = entry_price + 3 * atr_val
                else:
                    stop_loss = entry_price + self.atr_stop_mult * atr_val
                    take_profit = entry_price - 3 * atr_val

                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=signal_dir,
                    confidence=confidence,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        "donchian_upper": prev_row["donchian_upper"],
                        "donchian_lower": prev_row["donchian_lower"],
                        "atr": row["atr"],
                        "breakout_type": "upper" if signal_dir == SignalDirection.LONG else "lower"
                    }
                ))

        return signals

"""
MACD Crossover Strategy.

Generates signals when MACD line crosses the signal line.
"""

from typing import List, Dict, Any
import pandas as pd
import numpy as np

from .base import (
    SignalGenerator, Signal, SignalDirection,
    macd, ema, atr
)


class MACDStrategy(SignalGenerator):
    """
    MACD Crossover Strategy.

    Generates LONG signal when MACD crosses above signal line.
    Generates SHORT signal when MACD crosses below signal line.

    Parameters:
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)
        histogram_threshold: Minimum histogram value for confirmation (default: 0)
    """

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        histogram_threshold: float = 0,
        **kwargs
    ):
        super().__init__(
            name="macd",
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
            histogram_threshold=histogram_threshold,
            **kwargs
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.histogram_threshold = histogram_threshold

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute MACD indicators."""
        df = df.copy()

        macd_line, signal_line, histogram = macd(
            df["close"],
            fast=self.fast_period,
            slow=self.slow_period,
            signal=self.signal_period
        )

        df["macd"] = macd_line
        df["macd_signal"] = signal_line
        df["macd_hist"] = histogram

        # Previous values for crossover detection
        df["macd_prev"] = df["macd"].shift(1)
        df["macd_signal_prev"] = df["macd_signal"].shift(1)

        # ATR for stop loss calculation
        df["atr"] = atr(df["high"], df["low"], df["close"], 14)

        return df

    def generate_signals(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> List[Signal]:
        """Generate MACD crossover signals."""
        df = self.compute_indicators(df)
        signals = []

        for idx in range(1, len(df)):
            row = df.iloc[idx]
            prev_row = df.iloc[idx - 1]

            # Skip if indicators not ready
            if pd.isna(row["macd"]) or pd.isna(row["macd_signal"]):
                continue

            signal_dir = SignalDirection.NEUTRAL
            confidence = 0.5

            # Bullish crossover: MACD crosses above signal
            if (prev_row["macd"] <= prev_row["macd_signal"] and
                row["macd"] > row["macd_signal"]):

                # Confirm with histogram threshold
                if abs(row["macd_hist"]) >= self.histogram_threshold:
                    signal_dir = SignalDirection.LONG

                    # Higher confidence if crossing from below zero
                    if row["macd"] < 0:
                        confidence = 0.7  # Early signal
                    else:
                        confidence = 0.6

            # Bearish crossover: MACD crosses below signal
            elif (prev_row["macd"] >= prev_row["macd_signal"] and
                  row["macd"] < row["macd_signal"]):

                if abs(row["macd_hist"]) >= self.histogram_threshold:
                    signal_dir = SignalDirection.SHORT

                    # Higher confidence if crossing from above zero
                    if row["macd"] > 0:
                        confidence = 0.7
                    else:
                        confidence = 0.6

            if signal_dir != SignalDirection.NEUTRAL:
                entry_price = row["close"]
                atr_val = row["atr"] if not pd.isna(row["atr"]) else entry_price * 0.02

                if signal_dir == SignalDirection.LONG:
                    stop_loss = entry_price - 2 * atr_val
                    take_profit = entry_price + 3 * atr_val
                else:
                    stop_loss = entry_price + 2 * atr_val
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
                        "macd": row["macd"],
                        "signal": row["macd_signal"],
                        "histogram": row["macd_hist"]
                    }
                ))

        return signals

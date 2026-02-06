"""
Moving Average Crossover Strategy.

Simple baseline strategy using EMA crossovers.
"""

from typing import List, Dict, Any
import pandas as pd
import numpy as np

from .base import (
    SignalGenerator, Signal, SignalDirection,
    ema, sma, atr
)


class MACrossoverStrategy(SignalGenerator):
    """
    Moving Average Crossover Strategy.

    Generates LONG signal when fast MA crosses above slow MA.
    Generates SHORT signal when fast MA crosses below slow MA.

    This is a baseline strategy - simple but useful for
    comparison against more complex strategies.

    Parameters:
        fast_period: Fast MA period (default: 9)
        slow_period: Slow MA period (default: 21)
        ma_type: 'ema' or 'sma' (default: 'ema')
        trend_filter: Use 200 MA as trend filter (default: True)
    """

    def __init__(
        self,
        fast_period: int = 9,
        slow_period: int = 21,
        ma_type: str = "ema",
        trend_filter: bool = True,
        **kwargs
    ):
        super().__init__(
            name="ma_cross",
            fast_period=fast_period,
            slow_period=slow_period,
            ma_type=ma_type,
            trend_filter=trend_filter,
            **kwargs
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.ma_type = ma_type
        self.trend_filter = trend_filter

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute moving average indicators."""
        df = df.copy()

        ma_func = ema if self.ma_type == "ema" else sma

        df["ma_fast"] = ma_func(df["close"], self.fast_period)
        df["ma_slow"] = ma_func(df["close"], self.slow_period)

        # Previous values for crossover detection
        df["ma_fast_prev"] = df["ma_fast"].shift(1)
        df["ma_slow_prev"] = df["ma_slow"].shift(1)

        # Trend filter
        if self.trend_filter:
            df["ma_200"] = ema(df["close"], 200)

        # ATR for stop loss
        df["atr"] = atr(df["high"], df["low"], df["close"], 14)

        return df

    def generate_signals(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> List[Signal]:
        """Generate MA crossover signals."""
        df = self.compute_indicators(df)
        signals = []

        for idx in range(1, len(df)):
            row = df.iloc[idx]
            prev_row = df.iloc[idx - 1]

            # Skip if indicators not ready
            if pd.isna(row["ma_fast"]) or pd.isna(row["ma_slow"]):
                continue

            signal_dir = SignalDirection.NEUTRAL
            confidence = 0.5

            # Golden cross: fast crosses above slow
            if (prev_row["ma_fast"] <= prev_row["ma_slow"] and
                row["ma_fast"] > row["ma_slow"]):

                # Apply trend filter if enabled
                if self.trend_filter:
                    if pd.isna(row["ma_200"]) or row["close"] < row["ma_200"]:
                        continue  # Skip if below 200 MA (bearish trend)
                    confidence = 0.65
                else:
                    confidence = 0.55

                signal_dir = SignalDirection.LONG

            # Death cross: fast crosses below slow
            elif (prev_row["ma_fast"] >= prev_row["ma_slow"] and
                  row["ma_fast"] < row["ma_slow"]):

                if self.trend_filter:
                    if pd.isna(row["ma_200"]) or row["close"] > row["ma_200"]:
                        continue  # Skip if above 200 MA (bullish trend)
                    confidence = 0.65
                else:
                    confidence = 0.55

                signal_dir = SignalDirection.SHORT

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
                        "ma_fast": row["ma_fast"],
                        "ma_slow": row["ma_slow"],
                        "ma_200": row.get("ma_200", None)
                    }
                ))

        return signals

"""
RSI Divergence Strategy.

Detects bullish and bearish divergences between price and RSI.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from .base import (
    SignalGenerator, Signal, SignalDirection,
    rsi, pivot_high, pivot_low, atr, ema
)


class RSIDivergenceStrategy(SignalGenerator):
    """
    RSI Divergence Strategy.

    Detects:
    - Regular Bullish Divergence: Price makes lower low, RSI makes higher low
    - Regular Bearish Divergence: Price makes higher high, RSI makes lower high
    - Hidden Bullish Divergence: Price makes higher low, RSI makes lower low (trend continuation)
    - Hidden Bearish Divergence: Price makes lower high, RSI makes higher high (trend continuation)

    Parameters:
        rsi_period: RSI calculation period (default: 14)
        pivot_lookback: Bars on each side for pivot detection (default: 5)
        min_div_bars: Minimum bars between pivots for divergence (default: 3)
        max_div_bars: Maximum bars between pivots for divergence (default: 50)
        signal_type: 'regular', 'hidden', or 'both' (default: 'regular')
    """

    def __init__(
        self,
        rsi_period: int = 14,
        pivot_lookback: int = 5,
        min_div_bars: int = 3,
        max_div_bars: int = 50,
        signal_type: str = "regular",
        **kwargs
    ):
        super().__init__(
            name="rsi_div",
            rsi_period=rsi_period,
            pivot_lookback=pivot_lookback,
            min_div_bars=min_div_bars,
            max_div_bars=max_div_bars,
            signal_type=signal_type,
            **kwargs
        )
        self.rsi_period = rsi_period
        self.pivot_lookback = pivot_lookback
        self.min_div_bars = min_div_bars
        self.max_div_bars = max_div_bars
        self.signal_type = signal_type

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute RSI divergence indicators."""
        df = df.copy()

        # RSI
        df["rsi"] = rsi(df["close"], self.rsi_period)

        # Price pivots
        df["pivot_high"] = pivot_high(df["high"], self.pivot_lookback)
        df["pivot_low"] = pivot_low(df["low"], self.pivot_lookback)

        # RSI at pivot points
        df["rsi_at_pivot_high"] = np.where(
            df["pivot_high"].notna(),
            df["rsi"],
            np.nan
        )
        df["rsi_at_pivot_low"] = np.where(
            df["pivot_low"].notna(),
            df["rsi"],
            np.nan
        )

        # ATR for stop loss
        df["atr"] = atr(df["high"], df["low"], df["close"], 14)

        # Trend filter
        df["ema_50"] = ema(df["close"], 50)
        df["ema_200"] = ema(df["close"], 200)

        return df

    def _find_divergence(
        self,
        df: pd.DataFrame,
        idx: int,
        lookback: int = 100
    ) -> Optional[Dict[str, Any]]:
        """
        Look for divergence at current index.

        Returns:
            Dictionary with divergence info or None
        """
        row = df.iloc[idx]

        # Need a confirmed pivot at or near current bar
        # Check recent pivot lows for bullish divergence
        # Check recent pivot highs for bearish divergence

        start_idx = max(0, idx - lookback)
        subset = df.iloc[start_idx:idx + 1]

        # Find pivot lows for bullish divergence
        pivot_lows = subset[subset["pivot_low"].notna()].copy()
        if len(pivot_lows) >= 2:
            # Get the two most recent pivot lows
            recent_pivots = pivot_lows.tail(2)
            pivot1 = recent_pivots.iloc[0]
            pivot2 = recent_pivots.iloc[1]

            bars_between = pivot_lows.index[-1] - pivot_lows.index[-2]

            if self.min_div_bars <= bars_between <= self.max_div_bars:
                # Check for Regular Bullish Divergence
                # Price: lower low, RSI: higher low
                if (self.signal_type in ["regular", "both"] and
                    pivot2["pivot_low"] < pivot1["pivot_low"] and
                    pivot2["rsi_at_pivot_low"] > pivot1["rsi_at_pivot_low"]):
                    return {
                        "type": "regular_bullish",
                        "direction": SignalDirection.LONG,
                        "pivot1_price": pivot1["pivot_low"],
                        "pivot2_price": pivot2["pivot_low"],
                        "pivot1_rsi": pivot1["rsi_at_pivot_low"],
                        "pivot2_rsi": pivot2["rsi_at_pivot_low"],
                        "bars_between": bars_between
                    }

                # Check for Hidden Bullish Divergence
                # Price: higher low, RSI: lower low (trend continuation)
                if (self.signal_type in ["hidden", "both"] and
                    pivot2["pivot_low"] > pivot1["pivot_low"] and
                    pivot2["rsi_at_pivot_low"] < pivot1["rsi_at_pivot_low"]):
                    return {
                        "type": "hidden_bullish",
                        "direction": SignalDirection.LONG,
                        "pivot1_price": pivot1["pivot_low"],
                        "pivot2_price": pivot2["pivot_low"],
                        "pivot1_rsi": pivot1["rsi_at_pivot_low"],
                        "pivot2_rsi": pivot2["rsi_at_pivot_low"],
                        "bars_between": bars_between
                    }

        # Find pivot highs for bearish divergence
        pivot_highs = subset[subset["pivot_high"].notna()].copy()
        if len(pivot_highs) >= 2:
            recent_pivots = pivot_highs.tail(2)
            pivot1 = recent_pivots.iloc[0]
            pivot2 = recent_pivots.iloc[1]

            bars_between = pivot_highs.index[-1] - pivot_highs.index[-2]

            if self.min_div_bars <= bars_between <= self.max_div_bars:
                # Check for Regular Bearish Divergence
                # Price: higher high, RSI: lower high
                if (self.signal_type in ["regular", "both"] and
                    pivot2["pivot_high"] > pivot1["pivot_high"] and
                    pivot2["rsi_at_pivot_high"] < pivot1["rsi_at_pivot_high"]):
                    return {
                        "type": "regular_bearish",
                        "direction": SignalDirection.SHORT,
                        "pivot1_price": pivot1["pivot_high"],
                        "pivot2_price": pivot2["pivot_high"],
                        "pivot1_rsi": pivot1["rsi_at_pivot_high"],
                        "pivot2_rsi": pivot2["rsi_at_pivot_high"],
                        "bars_between": bars_between
                    }

                # Check for Hidden Bearish Divergence
                # Price: lower high, RSI: higher high (trend continuation)
                if (self.signal_type in ["hidden", "both"] and
                    pivot2["pivot_high"] < pivot1["pivot_high"] and
                    pivot2["rsi_at_pivot_high"] > pivot1["rsi_at_pivot_high"]):
                    return {
                        "type": "hidden_bearish",
                        "direction": SignalDirection.SHORT,
                        "pivot1_price": pivot1["pivot_high"],
                        "pivot2_price": pivot2["pivot_high"],
                        "pivot1_rsi": pivot1["rsi_at_pivot_high"],
                        "pivot2_rsi": pivot2["rsi_at_pivot_high"],
                        "bars_between": bars_between
                    }

        return None

    def generate_signals(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> List[Signal]:
        """Generate RSI divergence signals."""
        df = self.compute_indicators(df)
        signals = []

        # Track last signal to avoid duplicates
        last_signal_idx = -self.min_div_bars

        # Need enough history for pivot detection
        min_bars = max(self.pivot_lookback * 2, 50)

        for idx in range(min_bars, len(df)):
            # Skip if too close to last signal
            if idx - last_signal_idx < self.min_div_bars:
                continue

            row = df.iloc[idx]

            if pd.isna(row["atr"]) or pd.isna(row["rsi"]):
                continue

            # Look for divergence
            divergence = self._find_divergence(df, idx)

            if divergence is None:
                continue

            signal_dir = divergence["direction"]

            # Calculate confidence based on divergence strength
            rsi_diff = abs(divergence["pivot2_rsi"] - divergence["pivot1_rsi"])
            confidence = min(0.5 + rsi_diff / 50, 0.8)  # Scale 0.5-0.8

            # Regular divergences get slightly higher confidence
            if "regular" in divergence["type"]:
                confidence = min(confidence + 0.05, 0.85)

            entry_price = row["close"]
            atr_val = row["atr"]

            if signal_dir == SignalDirection.LONG:
                stop_loss = divergence["pivot2_price"] - atr_val
                take_profit = entry_price + 2 * atr_val
            else:
                stop_loss = divergence["pivot2_price"] + atr_val
                take_profit = entry_price - 2 * atr_val

            signals.append(Signal(
                timestamp=row["timestamp"],
                symbol=symbol,
                direction=signal_dir,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    "divergence_type": divergence["type"],
                    "pivot1_price": divergence["pivot1_price"],
                    "pivot2_price": divergence["pivot2_price"],
                    "pivot1_rsi": divergence["pivot1_rsi"],
                    "pivot2_rsi": divergence["pivot2_rsi"],
                    "current_rsi": row["rsi"],
                    "bars_between": divergence["bars_between"]
                }
            ))

            last_signal_idx = idx

        return signals

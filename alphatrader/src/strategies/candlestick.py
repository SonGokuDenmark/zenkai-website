"""
Candlestick Pattern Recognition Strategy.

Identifies key candlestick patterns with high historical reliability
based on Thomas Bulkowski's research.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from .base import (
    SignalGenerator, Signal, SignalDirection,
    ema, atr
)


class CandlestickStrategy(SignalGenerator):
    """
    Candlestick Pattern Strategy.

    Implements high-reliability patterns from Bulkowski's Encyclopedia:
    - Bullish Engulfing (63% bullish)
    - Bearish Engulfing (79% bearish)
    - Morning Star (78% bullish)
    - Evening Star (72% bearish)
    - Hammer (60% bullish)
    - Shooting Star (59% bearish)

    Parameters:
        min_body_ratio: Minimum body/range ratio for valid candle (default: 0.3)
        trend_lookback: Bars to determine prior trend (default: 5)
        use_trend_filter: Only take patterns in correct trend context (default: True)
    """

    def __init__(
        self,
        min_body_ratio: float = 0.3,
        trend_lookback: int = 5,
        use_trend_filter: bool = True,
        **kwargs
    ):
        super().__init__(
            name="candlestick",
            min_body_ratio=min_body_ratio,
            trend_lookback=trend_lookback,
            use_trend_filter=use_trend_filter,
            **kwargs
        )
        self.min_body_ratio = min_body_ratio
        self.trend_lookback = trend_lookback
        self.use_trend_filter = use_trend_filter

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute candlestick analysis indicators."""
        df = df.copy()

        # Body calculations
        df["body"] = df["close"] - df["open"]
        df["body_abs"] = abs(df["body"])
        df["range"] = df["high"] - df["low"]
        df["body_ratio"] = df["body_abs"] / df["range"].replace(0, np.nan)

        # Wick calculations
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

        # Candle characteristics
        df["is_bullish"] = df["close"] > df["open"]
        df["is_bearish"] = df["close"] < df["open"]
        df["is_doji"] = df["body_ratio"] < 0.1

        # Trend determination
        df["trend"] = np.where(
            df["close"].rolling(self.trend_lookback).mean() >
            df["close"].shift(self.trend_lookback),
            1, -1
        )

        # Average body size for comparison
        df["avg_body"] = df["body_abs"].rolling(20).mean()

        # ATR for stop loss
        df["atr"] = atr(df["high"], df["low"], df["close"], 14)

        return df

    def _is_bullish_engulfing(
        self,
        current: pd.Series,
        prev: pd.Series
    ) -> bool:
        """Check for bullish engulfing pattern."""
        return (
            prev["is_bearish"] and
            current["is_bullish"] and
            current["open"] < prev["close"] and
            current["close"] > prev["open"] and
            current["body_abs"] > prev["body_abs"]
        )

    def _is_bearish_engulfing(
        self,
        current: pd.Series,
        prev: pd.Series
    ) -> bool:
        """Check for bearish engulfing pattern."""
        return (
            prev["is_bullish"] and
            current["is_bearish"] and
            current["open"] > prev["close"] and
            current["close"] < prev["open"] and
            current["body_abs"] > prev["body_abs"]
        )

    def _is_morning_star(
        self,
        candle1: pd.Series,
        candle2: pd.Series,
        candle3: pd.Series
    ) -> bool:
        """
        Check for morning star pattern (3-candle bullish reversal).

        Pattern: Large bearish -> Small body/doji -> Large bullish
        """
        return (
            # First candle: large bearish
            candle1["is_bearish"] and
            candle1["body_ratio"] > self.min_body_ratio and
            # Second candle: small body or doji, gaps down
            candle2["body_ratio"] < 0.3 and
            candle2["close"] < candle1["close"] and
            # Third candle: large bullish, closes above midpoint of first
            candle3["is_bullish"] and
            candle3["body_ratio"] > self.min_body_ratio and
            candle3["close"] > (candle1["open"] + candle1["close"]) / 2
        )

    def _is_evening_star(
        self,
        candle1: pd.Series,
        candle2: pd.Series,
        candle3: pd.Series
    ) -> bool:
        """
        Check for evening star pattern (3-candle bearish reversal).

        Pattern: Large bullish -> Small body/doji -> Large bearish
        """
        return (
            # First candle: large bullish
            candle1["is_bullish"] and
            candle1["body_ratio"] > self.min_body_ratio and
            # Second candle: small body or doji, gaps up
            candle2["body_ratio"] < 0.3 and
            candle2["close"] > candle1["close"] and
            # Third candle: large bearish, closes below midpoint of first
            candle3["is_bearish"] and
            candle3["body_ratio"] > self.min_body_ratio and
            candle3["close"] < (candle1["open"] + candle1["close"]) / 2
        )

    def _is_hammer(self, row: pd.Series) -> bool:
        """
        Check for hammer pattern (bullish reversal).

        Small body at top, long lower wick (2x+ body), little/no upper wick.
        """
        if row["range"] == 0:
            return False

        lower_wick_ratio = row["lower_wick"] / row["range"]
        upper_wick_ratio = row["upper_wick"] / row["range"]
        body_ratio = row["body_ratio"] if not pd.isna(row["body_ratio"]) else 0

        return (
            body_ratio < 0.35 and  # Small body
            lower_wick_ratio > 0.5 and  # Long lower wick (50%+ of range)
            upper_wick_ratio < 0.15 and  # Small upper wick
            row["lower_wick"] > 2 * row["body_abs"]  # Wick at least 2x body
        )

    def _is_shooting_star(self, row: pd.Series) -> bool:
        """
        Check for shooting star pattern (bearish reversal).

        Small body at bottom, long upper wick (2x+ body), little/no lower wick.
        """
        if row["range"] == 0:
            return False

        lower_wick_ratio = row["lower_wick"] / row["range"]
        upper_wick_ratio = row["upper_wick"] / row["range"]
        body_ratio = row["body_ratio"] if not pd.isna(row["body_ratio"]) else 0

        return (
            body_ratio < 0.35 and  # Small body
            upper_wick_ratio > 0.5 and  # Long upper wick (50%+ of range)
            lower_wick_ratio < 0.15 and  # Small lower wick
            row["upper_wick"] > 2 * row["body_abs"]  # Wick at least 2x body
        )

    def generate_signals(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> List[Signal]:
        """Generate candlestick pattern signals."""
        df = self.compute_indicators(df)
        signals = []

        for idx in range(3, len(df)):
            row = df.iloc[idx]
            prev = df.iloc[idx - 1]
            prev2 = df.iloc[idx - 2]
            prev3 = df.iloc[idx - 3]

            if pd.isna(row["atr"]):
                continue

            signal_dir = SignalDirection.NEUTRAL
            confidence = 0.5
            pattern_name = None

            # Check patterns
            # Bullish patterns (require prior downtrend if filter enabled)
            prior_trend = prev["trend"]

            # Bullish Engulfing
            if self._is_bullish_engulfing(row, prev):
                if not self.use_trend_filter or prior_trend == -1:
                    signal_dir = SignalDirection.LONG
                    confidence = 0.63
                    pattern_name = "bullish_engulfing"

            # Bearish Engulfing
            elif self._is_bearish_engulfing(row, prev):
                if not self.use_trend_filter or prior_trend == 1:
                    signal_dir = SignalDirection.SHORT
                    confidence = 0.79
                    pattern_name = "bearish_engulfing"

            # Morning Star (check 3-candle pattern on completion)
            elif self._is_morning_star(prev2, prev, row):
                if not self.use_trend_filter or prev2["trend"] == -1:
                    signal_dir = SignalDirection.LONG
                    confidence = 0.78
                    pattern_name = "morning_star"

            # Evening Star
            elif self._is_evening_star(prev2, prev, row):
                if not self.use_trend_filter or prev2["trend"] == 1:
                    signal_dir = SignalDirection.SHORT
                    confidence = 0.72
                    pattern_name = "evening_star"

            # Hammer (single candle, needs downtrend)
            elif self._is_hammer(row):
                if not self.use_trend_filter or prior_trend == -1:
                    signal_dir = SignalDirection.LONG
                    confidence = 0.60
                    pattern_name = "hammer"

            # Shooting Star (single candle, needs uptrend)
            elif self._is_shooting_star(row):
                if not self.use_trend_filter or prior_trend == 1:
                    signal_dir = SignalDirection.SHORT
                    confidence = 0.59
                    pattern_name = "shooting_star"

            if signal_dir != SignalDirection.NEUTRAL:
                entry_price = row["close"]
                atr_val = row["atr"]

                if signal_dir == SignalDirection.LONG:
                    stop_loss = row["low"] - atr_val
                    take_profit = entry_price + 2 * atr_val
                else:
                    stop_loss = row["high"] + atr_val
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
                        "pattern": pattern_name,
                        "prior_trend": prior_trend,
                        "body_ratio": row["body_ratio"]
                    }
                ))

        return signals

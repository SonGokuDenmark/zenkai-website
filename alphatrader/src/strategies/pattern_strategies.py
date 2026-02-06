"""
Pattern-Based Trading Strategies.

Chart patterns, candlestick patterns, and structural patterns.

Strategies:
1. DoubleBottom - Double bottom reversal
2. DoubleTop - Double top reversal
3. ThreeWhiteSoldiers - Bullish candlestick pattern
4. ThreeBlackCrows - Bearish candlestick pattern
5. MorningStar - Bullish reversal pattern
6. EveningStar - Bearish reversal pattern
7. Engulfing - Engulfing patterns
8. Hammer - Hammer/Hanging Man patterns
9. Doji - Doji indecision patterns
10. InsideBar - Inside bar breakout
"""

from typing import List
import pandas as pd
import numpy as np

from .base import (
    SignalGenerator, Signal, SignalDirection,
    sma
)


class DoubleBottomStrategy(SignalGenerator):
    """
    Double Bottom reversal pattern.

    Two similar lows with a peak in between = bullish reversal.
    """

    def __init__(self, lookback: int = 50, tolerance: float = 0.02):
        super().__init__(name="double_bottom")
        self.lookback = lookback
        self.tolerance = tolerance

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["swing_low"] = df["low"].rolling(window=5, center=True).min()
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(self.lookback, len(df)):
            row = df.iloc[i]
            window = df.iloc[i-self.lookback:i]

            # Find two lows within tolerance
            lows = window[window["low"] == window["swing_low"]]["low"]

            if len(lows) >= 2:
                low1 = lows.iloc[-2]
                low2 = lows.iloc[-1]

                # Check if similar level
                if abs(low1 - low2) / low1 < self.tolerance:
                    # Check if there's a peak between them
                    between = window.loc[lows.index[-2]:lows.index[-1]]
                    peak = between["high"].max()

                    if peak > low1 * 1.02:  # At least 2% higher
                        # Price breaking above peak confirms pattern
                        if row["close"] > peak:
                            signals.append(Signal(
                                timestamp=row["timestamp"],
                                symbol=symbol,
                                direction=SignalDirection.LONG,
                                confidence=0.8,
                                metadata={"reason": "double_bottom"}
                            ))

        return signals


class DoubleTopStrategy(SignalGenerator):
    """
    Double Top reversal pattern.

    Two similar highs with a trough in between = bearish reversal.
    """

    def __init__(self, lookback: int = 50, tolerance: float = 0.02):
        super().__init__(name="double_top")
        self.lookback = lookback
        self.tolerance = tolerance

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["swing_high"] = df["high"].rolling(window=5, center=True).max()
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(self.lookback, len(df)):
            row = df.iloc[i]
            window = df.iloc[i-self.lookback:i]

            # Find two highs within tolerance
            highs = window[window["high"] == window["swing_high"]]["high"]

            if len(highs) >= 2:
                high1 = highs.iloc[-2]
                high2 = highs.iloc[-1]

                if abs(high1 - high2) / high1 < self.tolerance:
                    between = window.loc[highs.index[-2]:highs.index[-1]]
                    trough = between["low"].min()

                    if trough < high1 * 0.98:
                        if row["close"] < trough:
                            signals.append(Signal(
                                timestamp=row["timestamp"],
                                symbol=symbol,
                                direction=SignalDirection.SHORT,
                                confidence=0.8,
                                metadata={"reason": "double_top"}
                            ))

        return signals


class ThreeWhiteSoldiersStrategy(SignalGenerator):
    """
    Three White Soldiers - Strong bullish reversal.

    Three consecutive bullish candles with higher closes.
    """

    def __init__(self, min_body_pct: float = 0.6):
        super().__init__(name="three_white_soldiers")
        self.min_body_pct = min_body_pct

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["body"] = df["close"] - df["open"]
        df["range"] = df["high"] - df["low"]
        df["body_pct"] = abs(df["body"]) / df["range"]
        df["is_bullish"] = df["close"] > df["open"]
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(2, len(df)):
            c1, c2, c3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]

            # All three bullish with good bodies
            if (c1["is_bullish"] and c2["is_bullish"] and c3["is_bullish"] and
                c1["body_pct"] > self.min_body_pct and
                c2["body_pct"] > self.min_body_pct and
                c3["body_pct"] > self.min_body_pct):

                # Each closes higher
                if c2["close"] > c1["close"] and c3["close"] > c2["close"]:
                    # Each opens within previous body
                    if c1["open"] < c2["open"] < c1["close"] and c2["open"] < c3["open"] < c2["close"]:
                        signals.append(Signal(
                            timestamp=c3["timestamp"],
                            symbol=symbol,
                            direction=SignalDirection.LONG,
                            confidence=0.85,
                            metadata={"reason": "three_white_soldiers"}
                        ))

        return signals


class ThreeBlackCrowsStrategy(SignalGenerator):
    """
    Three Black Crows - Strong bearish reversal.

    Three consecutive bearish candles with lower closes.
    """

    def __init__(self, min_body_pct: float = 0.6):
        super().__init__(name="three_black_crows")
        self.min_body_pct = min_body_pct

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["body"] = df["close"] - df["open"]
        df["range"] = df["high"] - df["low"]
        df["body_pct"] = abs(df["body"]) / df["range"]
        df["is_bearish"] = df["close"] < df["open"]
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(2, len(df)):
            c1, c2, c3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]

            if (c1["is_bearish"] and c2["is_bearish"] and c3["is_bearish"] and
                c1["body_pct"] > self.min_body_pct and
                c2["body_pct"] > self.min_body_pct and
                c3["body_pct"] > self.min_body_pct):

                if c2["close"] < c1["close"] and c3["close"] < c2["close"]:
                    if c1["close"] < c2["open"] < c1["open"] and c2["close"] < c3["open"] < c2["open"]:
                        signals.append(Signal(
                            timestamp=c3["timestamp"],
                            symbol=symbol,
                            direction=SignalDirection.SHORT,
                            confidence=0.85,
                            metadata={"reason": "three_black_crows"}
                        ))

        return signals


class MorningStarStrategy(SignalGenerator):
    """
    Morning Star - Bullish reversal pattern.

    Down candle, small body (star), up candle.
    """

    def __init__(self):
        super().__init__(name="morning_star")

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["body"] = abs(df["close"] - df["open"])
        df["range"] = df["high"] - df["low"]
        df["body_pct"] = df["body"] / df["range"]
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(2, len(df)):
            c1, c2, c3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]

            # First candle: large bearish
            is_bearish1 = c1["close"] < c1["open"] and c1["body_pct"] > 0.5

            # Second candle: small body (star)
            is_star = c2["body_pct"] < 0.3

            # Third candle: large bullish closing above midpoint of first
            is_bullish3 = c3["close"] > c3["open"] and c3["body_pct"] > 0.5
            closes_above_mid = c3["close"] > (c1["open"] + c1["close"]) / 2

            if is_bearish1 and is_star and is_bullish3 and closes_above_mid:
                signals.append(Signal(
                    timestamp=c3["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.8,
                    metadata={"reason": "morning_star"}
                ))

        return signals


class EveningStarStrategy(SignalGenerator):
    """
    Evening Star - Bearish reversal pattern.

    Up candle, small body (star), down candle.
    """

    def __init__(self):
        super().__init__(name="evening_star")

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["body"] = abs(df["close"] - df["open"])
        df["range"] = df["high"] - df["low"]
        df["body_pct"] = df["body"] / df["range"]
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(2, len(df)):
            c1, c2, c3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]

            is_bullish1 = c1["close"] > c1["open"] and c1["body_pct"] > 0.5
            is_star = c2["body_pct"] < 0.3
            is_bearish3 = c3["close"] < c3["open"] and c3["body_pct"] > 0.5
            closes_below_mid = c3["close"] < (c1["open"] + c1["close"]) / 2

            if is_bullish1 and is_star and is_bearish3 and closes_below_mid:
                signals.append(Signal(
                    timestamp=c3["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.8,
                    metadata={"reason": "evening_star"}
                ))

        return signals


class EngulfingStrategy(SignalGenerator):
    """
    Engulfing patterns - Strong reversal signal.

    Bullish: small bearish followed by large bullish that engulfs
    Bearish: small bullish followed by large bearish that engulfs
    """

    def __init__(self):
        super().__init__(name="engulfing")

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["body"] = df["close"] - df["open"]
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(1, len(df)):
            prev, curr = df.iloc[i-1], df.iloc[i]

            # Bullish engulfing
            if (prev["body"] < 0 and curr["body"] > 0 and
                curr["open"] < prev["close"] and curr["close"] > prev["open"]):
                signals.append(Signal(
                    timestamp=curr["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.8,
                    metadata={"reason": "bullish_engulfing"}
                ))

            # Bearish engulfing
            if (prev["body"] > 0 and curr["body"] < 0 and
                curr["open"] > prev["close"] and curr["close"] < prev["open"]):
                signals.append(Signal(
                    timestamp=curr["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.8,
                    metadata={"reason": "bearish_engulfing"}
                ))

        return signals


class HammerStrategy(SignalGenerator):
    """
    Hammer and Hanging Man patterns.

    Small body at top with long lower shadow.
    Hammer at bottom = bullish, Hanging Man at top = bearish.
    """

    def __init__(self, shadow_ratio: float = 2.0):
        super().__init__(name="hammer")
        self.shadow_ratio = shadow_ratio

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["body"] = abs(df["close"] - df["open"])
        df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]
        df["range"] = df["high"] - df["low"]

        # Price trend (for context)
        df["ma20"] = sma(df["close"], 20)

        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for idx, row in df.iterrows():
            if row["body"] == 0 or pd.isna(row["ma20"]):
                continue

            # Hammer characteristics
            is_hammer = (row["lower_shadow"] > self.shadow_ratio * row["body"] and
                        row["upper_shadow"] < row["body"])

            if is_hammer:
                # Hammer at bottom (below MA) = bullish
                if row["close"] < row["ma20"]:
                    signals.append(Signal(
                        timestamp=row["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.LONG,
                        confidence=0.75,
                        metadata={"reason": "hammer_bullish"}
                    ))
                # Hanging man at top (above MA) = bearish
                else:
                    signals.append(Signal(
                        timestamp=row["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.SHORT,
                        confidence=0.75,
                        metadata={"reason": "hanging_man_bearish"}
                    ))

        return signals


class DojiStrategy(SignalGenerator):
    """
    Doji patterns - Indecision, potential reversal.

    Very small body relative to range.
    """

    def __init__(self, body_threshold: float = 0.1):
        super().__init__(name="doji")
        self.body_threshold = body_threshold

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["body"] = abs(df["close"] - df["open"])
        df["range"] = df["high"] - df["low"]
        df["body_pct"] = df["body"] / df["range"].replace(0, np.nan)
        df["ma20"] = sma(df["close"], 20)
        df["trend"] = np.where(df["close"] > df["ma20"], 1, -1)
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]

            if pd.isna(row["body_pct"]) or pd.isna(row["trend"]):
                continue

            is_doji = row["body_pct"] < self.body_threshold

            if is_doji:
                # Doji in uptrend = potential reversal down
                if row["trend"] == 1:
                    signals.append(Signal(
                        timestamp=row["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.SHORT,
                        confidence=0.65,
                        metadata={"reason": "doji_bearish_reversal"}
                    ))
                # Doji in downtrend = potential reversal up
                else:
                    signals.append(Signal(
                        timestamp=row["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.LONG,
                        confidence=0.65,
                        metadata={"reason": "doji_bullish_reversal"}
                    ))

        return signals


class InsideBarStrategy(SignalGenerator):
    """
    Inside Bar Breakout.

    Current bar entirely within previous bar's range.
    Breakout direction = trade direction.
    """

    def __init__(self):
        super().__init__(name="inside_bar")

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["prev_high"] = df["high"].shift(1)
        df["prev_low"] = df["low"].shift(1)
        df["is_inside"] = (df["high"] < df["prev_high"]) & (df["low"] > df["prev_low"])
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(2, len(df)):
            prev = df.iloc[i-1]
            curr = df.iloc[i]

            # Previous bar was inside bar
            if prev["is_inside"]:
                # Current bar breaks out
                mother_high = df.iloc[i-2]["high"]
                mother_low = df.iloc[i-2]["low"]

                if curr["close"] > mother_high:
                    signals.append(Signal(
                        timestamp=curr["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.LONG,
                        confidence=0.75,
                        metadata={"reason": "inside_bar_breakout_up"}
                    ))
                elif curr["close"] < mother_low:
                    signals.append(Signal(
                        timestamp=curr["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.SHORT,
                        confidence=0.75,
                        metadata={"reason": "inside_bar_breakout_down"}
                    ))

        return signals


def get_pattern_strategies() -> List[SignalGenerator]:
    """Get all pattern-based strategy instances."""
    return [
        DoubleBottomStrategy(),
        DoubleTopStrategy(),
        ThreeWhiteSoldiersStrategy(),
        ThreeBlackCrowsStrategy(),
        MorningStarStrategy(),
        EveningStarStrategy(),
        EngulfingStrategy(),
        HammerStrategy(),
        DojiStrategy(),
        InsideBarStrategy(),
    ]


__all__ = [
    "DoubleBottomStrategy",
    "DoubleTopStrategy",
    "ThreeWhiteSoldiersStrategy",
    "ThreeBlackCrowsStrategy",
    "MorningStarStrategy",
    "EveningStarStrategy",
    "EngulfingStrategy",
    "HammerStrategy",
    "DojiStrategy",
    "InsideBarStrategy",
    "get_pattern_strategies",
]

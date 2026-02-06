"""
Ranging Market TRADING Strategies.

These strategies generate LONG/SHORT signals specifically for ranging/sideways markets.
They profit from mean reversion, range-bound oscillations, and channel trading.

Unlike ranging_strategies.py which DETECTS ranging markets (signals FLAT),
these strategies TRADE ranging markets (signals LONG at support, SHORT at resistance).

Strategies:
1. BBReversion - Buy lower BB, sell upper BB
2. RSIReversion - Buy oversold, sell overbought
3. StochReversion - Stochastic mean reversion
4. RangeBreakFade - Fade false breakouts
5. KeltnerBounce - Trade Keltner channel bounces
6. PivotBounce - Trade pivot point bounces
7. MeanReversion - Statistical mean reversion
8. ChannelTrade - Trade within price channels
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

from .base import (
    SignalGenerator, Signal, SignalDirection,
    sma, ema, atr, rsi, stochastic, bollinger_bands
)


def keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    atr_mult: float = 1.5
) -> tuple:
    """Calculate Keltner Channels."""
    middle = ema(close, period)
    atr_val = atr(high, low, close, period)
    upper = middle + atr_mult * atr_val
    lower = middle - atr_mult * atr_val
    return upper, middle, lower


class BBReversionStrategy(SignalGenerator):
    """
    Bollinger Band Mean Reversion.

    Buy when price touches/crosses lower BB (oversold in range)
    Sell when price touches/crosses upper BB (overbought in range)

    Best in ranging/low volatility environments.
    """

    def __init__(self, period: int = 20, std_mult: float = 2.0):
        super().__init__(name="bb_reversion")
        self.period = period
        self.std_mult = std_mult

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        upper, middle, lower = bollinger_bands(df["close"], self.period, self.std_mult)
        df["bb_upper"] = upper
        df["bb_middle"] = middle
        df["bb_lower"] = lower
        df["bb_pct"] = (df["close"] - lower) / (upper - lower)  # 0-1 position in bands
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]

            if pd.isna(row["bb_pct"]):
                continue

            # Buy signal: Price at/below lower band
            if row["low"] <= row["bb_lower"] or row["bb_pct"] < 0.05:
                # Stronger signal if price was above and dropped
                confidence = 0.8 if prev["bb_pct"] > 0.1 else 0.65
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=confidence,
                    entry_price=row["close"],
                    take_profit=row["bb_middle"],
                    stop_loss=row["bb_lower"] - (row["bb_upper"] - row["bb_lower"]) * 0.1,
                    metadata={"reason": "bb_lower_touch", "bb_pct": row["bb_pct"]}
                ))

            # Sell signal: Price at/above upper band
            elif row["high"] >= row["bb_upper"] or row["bb_pct"] > 0.95:
                confidence = 0.8 if prev["bb_pct"] < 0.9 else 0.65
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=confidence,
                    entry_price=row["close"],
                    take_profit=row["bb_middle"],
                    stop_loss=row["bb_upper"] + (row["bb_upper"] - row["bb_lower"]) * 0.1,
                    metadata={"reason": "bb_upper_touch", "bb_pct": row["bb_pct"]}
                ))

        return signals


class RSIReversionStrategy(SignalGenerator):
    """
    RSI Mean Reversion for ranging markets.

    Buy when RSI < 30 (oversold)
    Sell when RSI > 70 (overbought)
    """

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__(name="rsi_reversion")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["rsi"] = rsi(df["close"], self.period)
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]

            if pd.isna(row["rsi"]):
                continue

            # Buy on oversold
            if row["rsi"] < self.oversold:
                # Stronger if crossing into oversold
                crossing = prev["rsi"] >= self.oversold
                confidence = 0.85 if crossing else 0.7
                # Extra confidence for extreme readings
                if row["rsi"] < 20:
                    confidence = min(0.95, confidence + 0.1)

                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=confidence,
                    metadata={"rsi": row["rsi"], "reason": "rsi_oversold"}
                ))

            # Sell on overbought
            elif row["rsi"] > self.overbought:
                crossing = prev["rsi"] <= self.overbought
                confidence = 0.85 if crossing else 0.7
                if row["rsi"] > 80:
                    confidence = min(0.95, confidence + 0.1)

                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=confidence,
                    metadata={"rsi": row["rsi"], "reason": "rsi_overbought"}
                ))

        return signals


class StochReversionStrategy(SignalGenerator):
    """
    Stochastic Oscillator Mean Reversion.

    Buy when %K < 20 and %K crosses above %D
    Sell when %K > 80 and %K crosses below %D
    """

    def __init__(self, k_period: int = 14, d_period: int = 3,
                 oversold: float = 20, overbought: float = 80):
        super().__init__(name="stoch_reversion")
        self.k_period = k_period
        self.d_period = d_period
        self.oversold = oversold
        self.overbought = overbought

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        k, d = stochastic(df["high"], df["low"], df["close"],
                         self.k_period, self.d_period)
        df["stoch_k"] = k
        df["stoch_d"] = d
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]

            if pd.isna(row["stoch_k"]) or pd.isna(row["stoch_d"]):
                continue

            # Buy: %K crosses above %D in oversold zone
            if (row["stoch_k"] < self.oversold and
                prev["stoch_k"] <= prev["stoch_d"] and
                row["stoch_k"] > row["stoch_d"]):

                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.8,
                    metadata={"stoch_k": row["stoch_k"], "reason": "stoch_oversold_cross"}
                ))

            # Or just deeply oversold
            elif row["stoch_k"] < 10:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.7,
                    metadata={"stoch_k": row["stoch_k"], "reason": "stoch_extreme_oversold"}
                ))

            # Sell: %K crosses below %D in overbought zone
            if (row["stoch_k"] > self.overbought and
                prev["stoch_k"] >= prev["stoch_d"] and
                row["stoch_k"] < row["stoch_d"]):

                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.8,
                    metadata={"stoch_k": row["stoch_k"], "reason": "stoch_overbought_cross"}
                ))

            elif row["stoch_k"] > 90:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.7,
                    metadata={"stoch_k": row["stoch_k"], "reason": "stoch_extreme_overbought"}
                ))

        return signals


class RangeBreakFadeStrategy(SignalGenerator):
    """
    Fade False Breakouts.

    When price breaks out of a range but fails to hold, trade the reversal.
    This works best in ranging/choppy markets where breakouts often fail.
    """

    def __init__(self, range_period: int = 20, confirmation_bars: int = 2):
        super().__init__(name="fade_breakout")
        self.range_period = range_period
        self.confirmation_bars = confirmation_bars

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["range_high"] = df["high"].rolling(window=self.range_period).max().shift(1)
        df["range_low"] = df["low"].rolling(window=self.range_period).min().shift(1)
        df["range_mid"] = (df["range_high"] + df["range_low"]) / 2
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(self.confirmation_bars + 1, len(df)):
            row = df.iloc[i]

            if pd.isna(row["range_high"]):
                continue

            # Check for failed upside breakout
            # Previous bars broke above range_high but current bar closed back inside
            prev_bars = df.iloc[i-self.confirmation_bars:i]
            any_above = (prev_bars["high"] > row["range_high"]).any()
            closed_back = row["close"] < row["range_high"]

            if any_above and closed_back:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.75,
                    take_profit=row["range_mid"],
                    stop_loss=row["range_high"] * 1.01,
                    metadata={"reason": "failed_breakout_up"}
                ))

            # Check for failed downside breakout
            any_below = (prev_bars["low"] < row["range_low"]).any()
            closed_back = row["close"] > row["range_low"]

            if any_below and closed_back:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.75,
                    take_profit=row["range_mid"],
                    stop_loss=row["range_low"] * 0.99,
                    metadata={"reason": "failed_breakout_down"}
                ))

        return signals


class KeltnerBounceStrategy(SignalGenerator):
    """
    Keltner Channel Bounce trading.

    Buy at lower Keltner, sell at upper Keltner.
    Similar to BB but uses ATR instead of standard deviation.
    """

    def __init__(self, period: int = 20, atr_mult: float = 1.5):
        super().__init__(name="keltner_bounce")
        self.period = period
        self.atr_mult = atr_mult

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        upper, middle, lower = keltner_channels(
            df["high"], df["low"], df["close"], self.period, self.atr_mult
        )
        df["kc_upper"] = upper
        df["kc_middle"] = middle
        df["kc_lower"] = lower
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for idx, row in df.iterrows():
            if pd.isna(row["kc_lower"]):
                continue

            # Buy at lower Keltner
            if row["low"] <= row["kc_lower"]:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.75,
                    entry_price=row["close"],
                    take_profit=row["kc_middle"],
                    stop_loss=row["kc_lower"] * 0.99,
                    metadata={"reason": "keltner_lower_touch"}
                ))

            # Sell at upper Keltner
            elif row["high"] >= row["kc_upper"]:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.75,
                    entry_price=row["close"],
                    take_profit=row["kc_middle"],
                    stop_loss=row["kc_upper"] * 1.01,
                    metadata={"reason": "keltner_upper_touch"}
                ))

        return signals


class PivotBounceStrategy(SignalGenerator):
    """
    Pivot Point Bounce Strategy.

    Trade bounces off daily pivot levels (support/resistance).
    """

    def __init__(self):
        super().__init__(name="pivot_bounce")

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Calculate pivot points (using previous bar's HLC)
        df["pivot"] = (df["high"].shift(1) + df["low"].shift(1) + df["close"].shift(1)) / 3
        df["r1"] = 2 * df["pivot"] - df["low"].shift(1)
        df["s1"] = 2 * df["pivot"] - df["high"].shift(1)
        df["r2"] = df["pivot"] + (df["high"].shift(1) - df["low"].shift(1))
        df["s2"] = df["pivot"] - (df["high"].shift(1) - df["low"].shift(1))

        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for idx, row in df.iterrows():
            if pd.isna(row["pivot"]):
                continue

            price = row["close"]
            low = row["low"]
            high = row["high"]

            # Buy at support levels
            for level, name in [(row["s1"], "s1"), (row["s2"], "s2")]:
                if low <= level <= price:  # Touched and bounced
                    signals.append(Signal(
                        timestamp=row["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.LONG,
                        confidence=0.7 if name == "s1" else 0.8,
                        take_profit=row["pivot"],
                        metadata={"reason": f"pivot_bounce_{name}"}
                    ))
                    break

            # Sell at resistance levels
            for level, name in [(row["r1"], "r1"), (row["r2"], "r2")]:
                if price <= level <= high:  # Touched and rejected
                    signals.append(Signal(
                        timestamp=row["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.SHORT,
                        confidence=0.7 if name == "r1" else 0.8,
                        take_profit=row["pivot"],
                        metadata={"reason": f"pivot_reject_{name}"}
                    ))
                    break

        return signals


class MeanReversionStrategy(SignalGenerator):
    """
    Statistical Mean Reversion.

    Trade when price deviates significantly from its moving average.
    Uses z-score to identify extreme deviations.
    """

    def __init__(self, period: int = 20, z_threshold: float = 2.0):
        super().__init__(name="mean_reversion")
        self.period = period
        self.z_threshold = z_threshold

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ma"] = sma(df["close"], self.period)
        df["std"] = df["close"].rolling(window=self.period).std()
        df["z_score"] = (df["close"] - df["ma"]) / df["std"]
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for idx, row in df.iterrows():
            if pd.isna(row["z_score"]):
                continue

            z = row["z_score"]

            # Buy when significantly below mean
            if z < -self.z_threshold:
                confidence = min(0.95, 0.6 + abs(z) / 10)
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=confidence,
                    take_profit=row["ma"],
                    metadata={"z_score": z, "reason": "mean_reversion_long"}
                ))

            # Sell when significantly above mean
            elif z > self.z_threshold:
                confidence = min(0.95, 0.6 + abs(z) / 10)
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=confidence,
                    take_profit=row["ma"],
                    metadata={"z_score": z, "reason": "mean_reversion_short"}
                ))

        return signals


class ChannelTradeStrategy(SignalGenerator):
    """
    Trade within established price channels.

    Buy at channel bottom, sell at channel top.
    Uses Donchian-style channels.
    """

    def __init__(self, period: int = 20, entry_pct: float = 0.1):
        super().__init__(name="channel_trade")
        self.period = period
        self.entry_pct = entry_pct  # How close to boundary to trigger

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["channel_high"] = df["high"].rolling(window=self.period).max()
        df["channel_low"] = df["low"].rolling(window=self.period).min()
        df["channel_mid"] = (df["channel_high"] + df["channel_low"]) / 2
        df["channel_range"] = df["channel_high"] - df["channel_low"]
        df["channel_pos"] = (df["close"] - df["channel_low"]) / df["channel_range"]
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for idx, row in df.iterrows():
            if pd.isna(row["channel_pos"]):
                continue

            pos = row["channel_pos"]

            # Buy near channel bottom
            if pos < self.entry_pct:
                confidence = 0.6 + (self.entry_pct - pos) * 3
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=min(0.9, confidence),
                    entry_price=row["close"],
                    take_profit=row["channel_mid"],
                    stop_loss=row["channel_low"] * 0.99,
                    metadata={"channel_pos": pos, "reason": "channel_bottom"}
                ))

            # Sell near channel top
            elif pos > (1 - self.entry_pct):
                confidence = 0.6 + (pos - (1 - self.entry_pct)) * 3
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=min(0.9, confidence),
                    entry_price=row["close"],
                    take_profit=row["channel_mid"],
                    stop_loss=row["channel_high"] * 1.01,
                    metadata={"channel_pos": pos, "reason": "channel_top"}
                ))

        return signals


def get_ranging_trade_strategies() -> List[SignalGenerator]:
    """Get all ranging market TRADING strategy instances."""
    return [
        BBReversionStrategy(),
        RSIReversionStrategy(),
        StochReversionStrategy(),
        RangeBreakFadeStrategy(),
        KeltnerBounceStrategy(),
        PivotBounceStrategy(),
        MeanReversionStrategy(),
        ChannelTradeStrategy(),
    ]


__all__ = [
    "BBReversionStrategy",
    "RSIReversionStrategy",
    "StochReversionStrategy",
    "RangeBreakFadeStrategy",
    "KeltnerBounceStrategy",
    "PivotBounceStrategy",
    "MeanReversionStrategy",
    "ChannelTradeStrategy",
    "get_ranging_trade_strategies",
]

"""
Trending UP Market Strategies.

These strategies generate LONG signals when the market is trending upward.
Each uses different indicators/approaches to capture bullish momentum.

Strategies:
1. EMA Stack - Multiple EMAs stacked bullishly
2. Golden Cross - 50/200 SMA crossover
3. Parabolic SAR - SAR below price
4. Ichimoku Bullish - Price above cloud, bullish signals
5. ADX Trend Up - ADX > 25 with +DI > -DI
6. VWAP Breakout - Price breaks above VWAP
7. Higher Highs - Price making higher highs/lows
8. Momentum Breakout - RSI > 50 with price breakout
9. Volume Surge Up - Price up on high volume
10. Supertrend Long - Supertrend indicator bullish
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

from .base import (
    SignalGenerator, Signal, SignalDirection,
    sma, ema, atr, rsi, adx, macd, bollinger_bands
)


def parabolic_sar(
    high: pd.Series,
    low: pd.Series,
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.2
) -> pd.Series:
    """Calculate Parabolic SAR."""
    length = len(high)
    sar = pd.Series(index=high.index, dtype=float)
    trend = pd.Series(index=high.index, dtype=int)

    sar.iloc[0] = low.iloc[0]
    trend.iloc[0] = 1
    ep = high.iloc[0]
    af = af_start

    for i in range(1, length):
        if trend.iloc[i-1] == 1:  # Uptrend
            sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
            sar.iloc[i] = min(sar.iloc[i], low.iloc[i-1], low.iloc[i-2] if i > 1 else low.iloc[i-1])

            if low.iloc[i] < sar.iloc[i]:
                trend.iloc[i] = -1
                sar.iloc[i] = ep
                ep = low.iloc[i]
                af = af_start
            else:
                trend.iloc[i] = 1
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + af_step, af_max)
        else:  # Downtrend
            sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
            sar.iloc[i] = max(sar.iloc[i], high.iloc[i-1], high.iloc[i-2] if i > 1 else high.iloc[i-1])

            if high.iloc[i] > sar.iloc[i]:
                trend.iloc[i] = 1
                sar.iloc[i] = ep
                ep = high.iloc[i]
                af = af_start
            else:
                trend.iloc[i] = -1
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + af_step, af_max)

    return sar, trend


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0
) -> tuple:
    """Calculate Supertrend indicator."""
    hl2 = (high + low) / 2
    atr_val = atr(high, low, close, period)

    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val

    supertrend_line = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=int)

    supertrend_line.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = -1

    for i in range(1, len(close)):
        if close.iloc[i] > supertrend_line.iloc[i-1]:
            supertrend_line.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        elif close.iloc[i] < supertrend_line.iloc[i-1]:
            supertrend_line.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1
        else:
            supertrend_line.iloc[i] = supertrend_line.iloc[i-1]
            direction.iloc[i] = direction.iloc[i-1]

        if direction.iloc[i] == 1 and lower_band.iloc[i] < supertrend_line.iloc[i]:
            supertrend_line.iloc[i] = supertrend_line.iloc[i]
        if direction.iloc[i] == -1 and upper_band.iloc[i] > supertrend_line.iloc[i]:
            supertrend_line.iloc[i] = supertrend_line.iloc[i]

    return supertrend_line, direction


class EMAStackStrategy(SignalGenerator):
    """
    EMA Stack strategy - Multiple EMAs aligned bullishly.

    When EMA8 > EMA21 > EMA50 > EMA200, market is strongly bullish.
    """

    def __init__(self, periods: tuple = (8, 21, 50, 200)):
        super().__init__(name="ema_stack_up")
        self.periods = periods

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for p in self.periods:
            df[f"ema_{p}"] = ema(df["close"], p)
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        cols = [f"ema_{p}" for p in self.periods]

        for idx, row in df.iterrows():
            if any(pd.isna(row[c]) for c in cols):
                continue

            values = [row[c] for c in cols]

            # Check if perfectly stacked (each EMA > next)
            perfectly_stacked = all(values[i] > values[i+1] for i in range(len(values)-1))

            if perfectly_stacked:
                # Strong uptrend
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.9,
                    metadata={"reason": "ema_stack_bullish"}
                ))
            elif values[0] > values[1] > values[2]:
                # Partial stack (short-term bullish)
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.7,
                    metadata={"reason": "ema_partial_stack"}
                ))

        return signals


class GoldenCrossStrategy(SignalGenerator):
    """
    Golden Cross - 50 SMA crosses above 200 SMA.

    Classic long-term bullish signal.
    """

    def __init__(self, fast_period: int = 50, slow_period: int = 200):
        super().__init__(name="golden_cross")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["sma_fast"] = sma(df["close"], self.fast_period)
        df["sma_slow"] = sma(df["close"], self.slow_period)
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]

            if pd.isna(row["sma_fast"]) or pd.isna(row["sma_slow"]):
                continue

            # Golden cross: fast crosses above slow
            if prev["sma_fast"] <= prev["sma_slow"] and row["sma_fast"] > row["sma_slow"]:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.85,
                    metadata={"reason": "golden_cross"}
                ))
            # Already above (continuation)
            elif row["sma_fast"] > row["sma_slow"]:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.6,
                    metadata={"reason": "above_200sma"}
                ))

        return signals


class ParabolicSARUpStrategy(SignalGenerator):
    """
    Parabolic SAR bullish - SAR below price indicates uptrend.
    """

    def __init__(self, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2):
        super().__init__(name="psar_up")
        self.af_start = af_start
        self.af_step = af_step
        self.af_max = af_max

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        sar, trend = parabolic_sar(df["high"], df["low"], self.af_start, self.af_step, self.af_max)
        df["psar"] = sar
        df["psar_trend"] = trend
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]

            if pd.isna(row["psar_trend"]):
                continue

            # Trend flip to bullish
            if prev["psar_trend"] == -1 and row["psar_trend"] == 1:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.85,
                    metadata={"reason": "psar_flip_bullish"}
                ))
            # Continuation
            elif row["psar_trend"] == 1:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.6,
                    metadata={"reason": "psar_bullish"}
                ))

        return signals


class ADXTrendUpStrategy(SignalGenerator):
    """
    ADX Trend Up - Strong trend with +DI > -DI.

    ADX > 25 indicates trend, +DI > -DI indicates bullish.
    """

    def __init__(self, period: int = 14, adx_threshold: float = 25):
        super().__init__(name="adx_up")
        self.period = period
        self.adx_threshold = adx_threshold

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        adx_val, plus_di, minus_di = adx(df["high"], df["low"], df["close"], self.period)
        df["adx"] = adx_val
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for idx, row in df.iterrows():
            if pd.isna(row["adx"]):
                continue

            if row["adx"] > self.adx_threshold and row["plus_di"] > row["minus_di"]:
                # Strong bullish trend
                di_diff = row["plus_di"] - row["minus_di"]
                confidence = min(0.9, 0.6 + di_diff / 50)

                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=confidence,
                    metadata={"adx": row["adx"], "+di": row["plus_di"], "-di": row["minus_di"]}
                ))

        return signals


class HigherHighsStrategy(SignalGenerator):
    """
    Higher Highs/Higher Lows pattern - Classic uptrend structure.
    """

    def __init__(self, lookback: int = 10):
        super().__init__(name="higher_highs")
        self.lookback = lookback

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["swing_high"] = df["high"].rolling(window=self.lookback, center=True).max()
        df["swing_low"] = df["low"].rolling(window=self.lookback, center=True).min()
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        lookback = self.lookback * 2

        for i in range(lookback, len(df)):
            row = df.iloc[i]

            # Get recent swing highs and lows
            recent = df.iloc[i-lookback:i]

            highs = recent[recent["high"] == recent["swing_high"]]["high"].dropna()
            lows = recent[recent["low"] == recent["swing_low"]]["low"].dropna()

            if len(highs) >= 2 and len(lows) >= 2:
                # Check for higher highs and higher lows
                hh = all(highs.iloc[j] > highs.iloc[j-1] for j in range(1, len(highs)))
                hl = all(lows.iloc[j] > lows.iloc[j-1] for j in range(1, len(lows)))

                if hh and hl:
                    signals.append(Signal(
                        timestamp=row["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.LONG,
                        confidence=0.8,
                        metadata={"reason": "higher_highs_higher_lows"}
                    ))

        return signals


class MomentumBreakoutStrategy(SignalGenerator):
    """
    Momentum Breakout - RSI > 50 with price breaking recent high.
    """

    def __init__(self, rsi_period: int = 14, breakout_period: int = 20):
        super().__init__(name="momentum_breakout_up")
        self.rsi_period = rsi_period
        self.breakout_period = breakout_period

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["rsi"] = rsi(df["close"], self.rsi_period)
        df["recent_high"] = df["high"].rolling(window=self.breakout_period).max().shift(1)
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for idx, row in df.iterrows():
            if pd.isna(row["rsi"]) or pd.isna(row["recent_high"]):
                continue

            # Price breaks above recent high with RSI > 50
            if row["high"] > row["recent_high"] and row["rsi"] > 50:
                confidence = min(0.9, 0.5 + (row["rsi"] - 50) / 100)

                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=confidence,
                    metadata={"rsi": row["rsi"], "reason": "momentum_breakout"}
                ))

        return signals


class VolumeSurgeUpStrategy(SignalGenerator):
    """
    Volume Surge Up - Price increase on significantly higher volume.
    """

    def __init__(self, volume_mult: float = 2.0, period: int = 20):
        super().__init__(name="volume_surge_up")
        self.volume_mult = volume_mult
        self.period = period

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["volume_sma"] = sma(df["volume"], self.period)
        df["volume_ratio"] = df["volume"] / df["volume_sma"]
        df["price_change"] = df["close"].pct_change()
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for idx, row in df.iterrows():
            if pd.isna(row["volume_ratio"]) or pd.isna(row["price_change"]):
                continue

            # High volume with price increase
            if row["volume_ratio"] > self.volume_mult and row["price_change"] > 0:
                confidence = min(0.9, 0.5 + row["volume_ratio"] / 10)

                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=confidence,
                    metadata={"volume_ratio": row["volume_ratio"], "reason": "volume_surge"}
                ))

        return signals


class SupertrendLongStrategy(SignalGenerator):
    """
    Supertrend Long - Supertrend indicator shows bullish.
    """

    def __init__(self, period: int = 10, multiplier: float = 3.0):
        super().__init__(name="supertrend_up")
        self.period = period
        self.multiplier = multiplier

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        st_line, st_dir = supertrend(df["high"], df["low"], df["close"], self.period, self.multiplier)
        df["supertrend"] = st_line
        df["st_direction"] = st_dir
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]

            if pd.isna(row["st_direction"]):
                continue

            # Flip to bullish
            if prev.get("st_direction", 0) == -1 and row["st_direction"] == 1:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.85,
                    metadata={"reason": "supertrend_flip_bullish"}
                ))
            # Continuation
            elif row["st_direction"] == 1:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.6,
                    metadata={"reason": "supertrend_bullish"}
                ))

        return signals


class VWAPBreakoutStrategy(SignalGenerator):
    """
    VWAP Breakout - Price breaks above VWAP.

    Note: VWAP typically resets daily, simplified here as rolling.
    """

    def __init__(self, period: int = 20):
        super().__init__(name="vwap_breakout_up")
        self.period = period

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        df["vwap"] = (typical_price * df["volume"]).rolling(window=self.period).sum() / \
                     df["volume"].rolling(window=self.period).sum()
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]

            if pd.isna(row["vwap"]):
                continue

            # Cross above VWAP
            if prev["close"] <= prev["vwap"] and row["close"] > row["vwap"]:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.8,
                    metadata={"reason": "vwap_breakout"}
                ))
            # Staying above VWAP
            elif row["close"] > row["vwap"]:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.55,
                    metadata={"reason": "above_vwap"}
                ))

        return signals


class IchimokuBullishStrategy(SignalGenerator):
    """
    Ichimoku Cloud Bullish signals.

    - Price above cloud
    - Tenkan-sen > Kijun-sen
    - Future cloud bullish (Senkou Span A > B)
    """

    def __init__(self, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52):
        super().__init__(name="ichimoku_up")
        self.tenkan = tenkan
        self.kijun = kijun
        self.senkou_b = senkou_b

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Tenkan-sen (Conversion Line)
        high_tenkan = df["high"].rolling(window=self.tenkan).max()
        low_tenkan = df["low"].rolling(window=self.tenkan).min()
        df["tenkan"] = (high_tenkan + low_tenkan) / 2

        # Kijun-sen (Base Line)
        high_kijun = df["high"].rolling(window=self.kijun).max()
        low_kijun = df["low"].rolling(window=self.kijun).min()
        df["kijun"] = (high_kijun + low_kijun) / 2

        # Senkou Span A (Leading Span A)
        df["senkou_a"] = ((df["tenkan"] + df["kijun"]) / 2).shift(self.kijun)

        # Senkou Span B (Leading Span B)
        high_senkou = df["high"].rolling(window=self.senkou_b).max()
        low_senkou = df["low"].rolling(window=self.senkou_b).min()
        df["senkou_b"] = ((high_senkou + low_senkou) / 2).shift(self.kijun)

        # Cloud top and bottom
        df["cloud_top"] = df[["senkou_a", "senkou_b"]].max(axis=1)
        df["cloud_bottom"] = df[["senkou_a", "senkou_b"]].min(axis=1)

        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for idx, row in df.iterrows():
            if pd.isna(row["cloud_top"]):
                continue

            above_cloud = row["close"] > row["cloud_top"]
            tk_cross = row["tenkan"] > row["kijun"]
            bullish_cloud = row["senkou_a"] > row["senkou_b"]

            score = sum([above_cloud, tk_cross, bullish_cloud])

            if score >= 3:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.9,
                    metadata={"reason": "ichimoku_full_bullish"}
                ))
            elif score >= 2 and above_cloud:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.7,
                    metadata={"reason": "ichimoku_bullish"}
                ))

        return signals


def get_trending_up_strategies() -> List[SignalGenerator]:
    """Get all trending UP strategy instances."""
    return [
        EMAStackStrategy(),
        GoldenCrossStrategy(),
        ParabolicSARUpStrategy(),
        ADXTrendUpStrategy(),
        HigherHighsStrategy(),
        MomentumBreakoutStrategy(),
        VolumeSurgeUpStrategy(),
        SupertrendLongStrategy(),
        VWAPBreakoutStrategy(),
        IchimokuBullishStrategy(),
    ]


__all__ = [
    "EMAStackStrategy",
    "GoldenCrossStrategy",
    "ParabolicSARUpStrategy",
    "ADXTrendUpStrategy",
    "HigherHighsStrategy",
    "MomentumBreakoutStrategy",
    "VolumeSurgeUpStrategy",
    "SupertrendLongStrategy",
    "VWAPBreakoutStrategy",
    "IchimokuBullishStrategy",
    "get_trending_up_strategies",
]

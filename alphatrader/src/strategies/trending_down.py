"""
Trending DOWN Market Strategies.

These strategies generate SHORT signals when the market is trending downward.
Mirror of trending_up strategies but for bearish conditions.

Strategies:
1. EMA Stack Down - Multiple EMAs stacked bearishly
2. Death Cross - 50/200 SMA crossover (bearish)
3. Parabolic SAR Down - SAR above price
4. Ichimoku Bearish - Price below cloud, bearish signals
5. ADX Trend Down - ADX > 25 with -DI > +DI
6. VWAP Breakdown - Price breaks below VWAP
7. Lower Lows - Price making lower highs/lows
8. Momentum Breakdown - RSI < 50 with price breakdown
9. Volume Surge Down - Price down on high volume
10. Supertrend Short - Supertrend indicator bearish
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

from .trending_up import parabolic_sar, supertrend


class EMAStackDownStrategy(SignalGenerator):
    """
    EMA Stack Down - Multiple EMAs aligned bearishly.

    When EMA8 < EMA21 < EMA50 < EMA200, market is strongly bearish.
    """

    def __init__(self, periods: tuple = (8, 21, 50, 200)):
        super().__init__(name="ema_stack_down")
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

            # Check if perfectly stacked bearish (each EMA < next)
            perfectly_stacked = all(values[i] < values[i+1] for i in range(len(values)-1))

            if perfectly_stacked:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.9,
                    metadata={"reason": "ema_stack_bearish"}
                ))
            elif values[0] < values[1] < values[2]:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.7,
                    metadata={"reason": "ema_partial_stack_bearish"}
                ))

        return signals


class DeathCrossStrategy(SignalGenerator):
    """
    Death Cross - 50 SMA crosses below 200 SMA.

    Classic long-term bearish signal.
    """

    def __init__(self, fast_period: int = 50, slow_period: int = 200):
        super().__init__(name="death_cross")
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

            # Death cross: fast crosses below slow
            if prev["sma_fast"] >= prev["sma_slow"] and row["sma_fast"] < row["sma_slow"]:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.85,
                    metadata={"reason": "death_cross"}
                ))
            # Already below (continuation)
            elif row["sma_fast"] < row["sma_slow"]:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.6,
                    metadata={"reason": "below_200sma"}
                ))

        return signals


class ParabolicSARDownStrategy(SignalGenerator):
    """
    Parabolic SAR bearish - SAR above price indicates downtrend.
    """

    def __init__(self, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2):
        super().__init__(name="psar_down")
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

            # Trend flip to bearish
            if prev["psar_trend"] == 1 and row["psar_trend"] == -1:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.85,
                    metadata={"reason": "psar_flip_bearish"}
                ))
            # Continuation
            elif row["psar_trend"] == -1:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.6,
                    metadata={"reason": "psar_bearish"}
                ))

        return signals


class ADXTrendDownStrategy(SignalGenerator):
    """
    ADX Trend Down - Strong trend with -DI > +DI.

    ADX > 25 indicates trend, -DI > +DI indicates bearish.
    """

    def __init__(self, period: int = 14, adx_threshold: float = 25):
        super().__init__(name="adx_down")
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

            if row["adx"] > self.adx_threshold and row["minus_di"] > row["plus_di"]:
                di_diff = row["minus_di"] - row["plus_di"]
                confidence = min(0.9, 0.6 + di_diff / 50)

                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=confidence,
                    metadata={"adx": row["adx"], "+di": row["plus_di"], "-di": row["minus_di"]}
                ))

        return signals


class LowerLowsStrategy(SignalGenerator):
    """
    Lower Lows/Lower Highs pattern - Classic downtrend structure.
    """

    def __init__(self, lookback: int = 10):
        super().__init__(name="lower_lows")
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

            recent = df.iloc[i-lookback:i]

            highs = recent[recent["high"] == recent["swing_high"]]["high"].dropna()
            lows = recent[recent["low"] == recent["swing_low"]]["low"].dropna()

            if len(highs) >= 2 and len(lows) >= 2:
                # Check for lower highs and lower lows
                lh = all(highs.iloc[j] < highs.iloc[j-1] for j in range(1, len(highs)))
                ll = all(lows.iloc[j] < lows.iloc[j-1] for j in range(1, len(lows)))

                if lh and ll:
                    signals.append(Signal(
                        timestamp=row["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.SHORT,
                        confidence=0.8,
                        metadata={"reason": "lower_highs_lower_lows"}
                    ))

        return signals


class MomentumBreakdownStrategy(SignalGenerator):
    """
    Momentum Breakdown - RSI < 50 with price breaking recent low.
    """

    def __init__(self, rsi_period: int = 14, breakdown_period: int = 20):
        super().__init__(name="momentum_breakdown")
        self.rsi_period = rsi_period
        self.breakdown_period = breakdown_period

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["rsi"] = rsi(df["close"], self.rsi_period)
        df["recent_low"] = df["low"].rolling(window=self.breakdown_period).min().shift(1)
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for idx, row in df.iterrows():
            if pd.isna(row["rsi"]) or pd.isna(row["recent_low"]):
                continue

            # Price breaks below recent low with RSI < 50
            if row["low"] < row["recent_low"] and row["rsi"] < 50:
                confidence = min(0.9, 0.5 + (50 - row["rsi"]) / 100)

                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=confidence,
                    metadata={"rsi": row["rsi"], "reason": "momentum_breakdown"}
                ))

        return signals


class VolumeSurgeDownStrategy(SignalGenerator):
    """
    Volume Surge Down - Price decrease on significantly higher volume.
    """

    def __init__(self, volume_mult: float = 2.0, period: int = 20):
        super().__init__(name="volume_surge_down")
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

            # High volume with price decrease
            if row["volume_ratio"] > self.volume_mult and row["price_change"] < 0:
                confidence = min(0.9, 0.5 + row["volume_ratio"] / 10)

                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=confidence,
                    metadata={"volume_ratio": row["volume_ratio"], "reason": "volume_surge_down"}
                ))

        return signals


class SupertrendShortStrategy(SignalGenerator):
    """
    Supertrend Short - Supertrend indicator shows bearish.
    """

    def __init__(self, period: int = 10, multiplier: float = 3.0):
        super().__init__(name="supertrend_down")
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

            # Flip to bearish
            if prev.get("st_direction", 0) == 1 and row["st_direction"] == -1:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.85,
                    metadata={"reason": "supertrend_flip_bearish"}
                ))
            # Continuation
            elif row["st_direction"] == -1:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.6,
                    metadata={"reason": "supertrend_bearish"}
                ))

        return signals


class VWAPBreakdownStrategy(SignalGenerator):
    """
    VWAP Breakdown - Price breaks below VWAP.
    """

    def __init__(self, period: int = 20):
        super().__init__(name="vwap_breakdown")
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

            # Cross below VWAP
            if prev["close"] >= prev["vwap"] and row["close"] < row["vwap"]:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.8,
                    metadata={"reason": "vwap_breakdown"}
                ))
            # Staying below VWAP
            elif row["close"] < row["vwap"]:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.55,
                    metadata={"reason": "below_vwap"}
                ))

        return signals


class IchimokuBearishStrategy(SignalGenerator):
    """
    Ichimoku Cloud Bearish signals.

    - Price below cloud
    - Tenkan-sen < Kijun-sen
    - Future cloud bearish (Senkou Span A < B)
    """

    def __init__(self, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52):
        super().__init__(name="ichimoku_down")
        self.tenkan = tenkan
        self.kijun = kijun
        self.senkou_b = senkou_b

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Tenkan-sen
        high_tenkan = df["high"].rolling(window=self.tenkan).max()
        low_tenkan = df["low"].rolling(window=self.tenkan).min()
        df["tenkan"] = (high_tenkan + low_tenkan) / 2

        # Kijun-sen
        high_kijun = df["high"].rolling(window=self.kijun).max()
        low_kijun = df["low"].rolling(window=self.kijun).min()
        df["kijun"] = (high_kijun + low_kijun) / 2

        # Senkou Span A
        df["senkou_a"] = ((df["tenkan"] + df["kijun"]) / 2).shift(self.kijun)

        # Senkou Span B
        high_senkou = df["high"].rolling(window=self.senkou_b).max()
        low_senkou = df["low"].rolling(window=self.senkou_b).min()
        df["senkou_b"] = ((high_senkou + low_senkou) / 2).shift(self.kijun)

        df["cloud_top"] = df[["senkou_a", "senkou_b"]].max(axis=1)
        df["cloud_bottom"] = df[["senkou_a", "senkou_b"]].min(axis=1)

        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for idx, row in df.iterrows():
            if pd.isna(row["cloud_bottom"]):
                continue

            below_cloud = row["close"] < row["cloud_bottom"]
            tk_cross = row["tenkan"] < row["kijun"]
            bearish_cloud = row["senkou_a"] < row["senkou_b"]

            score = sum([below_cloud, tk_cross, bearish_cloud])

            if score >= 3:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.9,
                    metadata={"reason": "ichimoku_full_bearish"}
                ))
            elif score >= 2 and below_cloud:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.7,
                    metadata={"reason": "ichimoku_bearish"}
                ))

        return signals


def get_trending_down_strategies() -> List[SignalGenerator]:
    """Get all trending DOWN strategy instances."""
    return [
        EMAStackDownStrategy(),
        DeathCrossStrategy(),
        ParabolicSARDownStrategy(),
        ADXTrendDownStrategy(),
        LowerLowsStrategy(),
        MomentumBreakdownStrategy(),
        VolumeSurgeDownStrategy(),
        SupertrendShortStrategy(),
        VWAPBreakdownStrategy(),
        IchimokuBearishStrategy(),
    ]


__all__ = [
    "EMAStackDownStrategy",
    "DeathCrossStrategy",
    "ParabolicSARDownStrategy",
    "ADXTrendDownStrategy",
    "LowerLowsStrategy",
    "MomentumBreakdownStrategy",
    "VolumeSurgeDownStrategy",
    "SupertrendShortStrategy",
    "VWAPBreakdownStrategy",
    "IchimokuBearishStrategy",
    "get_trending_down_strategies",
]

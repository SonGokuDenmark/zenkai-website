"""
Ranging/Choppy Market Detection Strategies.

These strategies detect when the market is NOT trending - i.e., ranging,
consolidating, or choppy. They help the LSTM model learn to predict FLAT.

Strategies included:
1. ADXRangingStrategy - ADX < 25 indicates weak trend
2. BollingerSqueezeStrategy - BB width contraction indicates compression
3. ChopIndexStrategy - Chop Index > 50 indicates choppy market
4. KeltnerSqueezeStrategy - BB inside Keltner = squeeze
5. RSIMidZoneStrategy - RSI 40-60 indicates no momentum
6. ATRContractionStrategy - Decreasing ATR indicates ranging
7. PriceChannelStrategy - Price near middle of range
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

from strategies.base import (
    SignalGenerator, Signal, SignalDirection,
    sma, ema, atr, bollinger_bands, rsi
)


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> tuple:
    """Calculate ADX, +DI, -DI."""
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Smoothed values
    tr_smooth = pd.Series(tr).rolling(window=period).sum()
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=period).sum()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=period).sum()

    # DI values
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth

    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_val = dx.rolling(window=period).mean()

    return adx_val, plus_di, minus_di


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


def chop_index(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate Choppiness Index.

    Values:
    - > 61.8: Very choppy/ranging
    - 38.2-61.8: Transitional
    - < 38.2: Trending
    """
    atr_val = atr(high, low, close, 1)  # 1-period ATR = True Range
    atr_sum = atr_val.rolling(window=period).sum()

    high_max = high.rolling(window=period).max()
    low_min = low.rolling(window=period).min()
    range_val = high_max - low_min

    # Avoid division by zero
    range_val = range_val.replace(0, np.nan)

    chop = 100 * np.log10(atr_sum / range_val) / np.log10(period)
    return chop


class ADXRangingStrategy(SignalGenerator):
    """
    Detects ranging market using ADX.

    ADX < 25: Weak trend = ranging
    ADX < 20: Very weak trend = strong ranging signal
    """

    def __init__(self, period: int = 14, threshold_weak: float = 25, threshold_strong: float = 20):
        super().__init__(name="adx_range")
        self.period = period
        self.threshold_weak = threshold_weak
        self.threshold_strong = threshold_strong

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

            adx_val = row["adx"]

            if adx_val < self.threshold_strong:
                # Very weak trend - strong ranging signal
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.NEUTRAL,  # FLAT
                    confidence=0.9,
                    metadata={"adx": adx_val, "reason": "very_weak_trend"}
                ))
            elif adx_val < self.threshold_weak:
                # Weak trend - ranging signal
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.NEUTRAL,
                    confidence=0.7,
                    metadata={"adx": adx_val, "reason": "weak_trend"}
                ))

        return signals


class BollingerSqueezeStrategy(SignalGenerator):
    """
    Detects Bollinger Band squeeze indicating consolidation.

    When BB width contracts to historical lows, market is ranging.
    """

    def __init__(self, bb_period: int = 20, bb_std: float = 2.0, squeeze_lookback: int = 50):
        super().__init__(name="bb_squeeze")
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.squeeze_lookback = squeeze_lookback

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        upper, middle, lower = bollinger_bands(df["close"], self.bb_period, self.bb_std)
        df["bb_upper"] = upper
        df["bb_middle"] = middle
        df["bb_lower"] = lower
        df["bb_width"] = (upper - lower) / middle * 100  # Percentage width
        df["bb_width_percentile"] = df["bb_width"].rolling(window=self.squeeze_lookback).apply(
            lambda x: (x.iloc[-1] <= x).mean() * 100 if len(x) > 0 else 50
        )
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for idx, row in df.iterrows():
            if pd.isna(row.get("bb_width_percentile")):
                continue

            percentile = row["bb_width_percentile"]

            if percentile < 10:
                # Very tight squeeze - strong ranging
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.NEUTRAL,
                    confidence=0.9,
                    metadata={"bb_width_percentile": percentile, "reason": "extreme_squeeze"}
                ))
            elif percentile < 25:
                # Moderate squeeze - ranging
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.NEUTRAL,
                    confidence=0.7,
                    metadata={"bb_width_percentile": percentile, "reason": "moderate_squeeze"}
                ))

        return signals


class ChopIndexStrategy(SignalGenerator):
    """
    Uses Choppiness Index to detect ranging markets.

    Chop Index > 61.8: Very choppy
    Chop Index 50-61.8: Choppy
    Chop Index < 38.2: Trending
    """

    def __init__(self, period: int = 14, chop_threshold: float = 50, strong_threshold: float = 61.8):
        super().__init__(name="chop")
        self.period = period
        self.chop_threshold = chop_threshold
        self.strong_threshold = strong_threshold

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["chop_index"] = chop_index(df["high"], df["low"], df["close"], self.period)
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for idx, row in df.iterrows():
            if pd.isna(row.get("chop_index")):
                continue

            chop = row["chop_index"]

            if chop > self.strong_threshold:
                # Very choppy
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.NEUTRAL,
                    confidence=0.9,
                    metadata={"chop_index": chop, "reason": "very_choppy"}
                ))
            elif chop > self.chop_threshold:
                # Choppy
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.NEUTRAL,
                    confidence=0.7,
                    metadata={"chop_index": chop, "reason": "choppy"}
                ))

        return signals


class KeltnerSqueezeStrategy(SignalGenerator):
    """
    Detects squeeze when Bollinger Bands are inside Keltner Channels.

    This is the classic TTM Squeeze indicator.
    """

    def __init__(self, bb_period: int = 20, bb_std: float = 2.0,
                 kc_period: int = 20, kc_mult: float = 1.5):
        super().__init__(name="kc_squeeze")
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.kc_period = kc_period
        self.kc_mult = kc_mult

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = bollinger_bands(
            df["close"], self.bb_period, self.bb_std
        )
        df["bb_upper"] = bb_upper
        df["bb_lower"] = bb_lower

        # Keltner Channels
        kc_upper, kc_middle, kc_lower = keltner_channels(
            df["high"], df["low"], df["close"], self.kc_period, self.kc_mult
        )
        df["kc_upper"] = kc_upper
        df["kc_lower"] = kc_lower

        # Squeeze: BB inside KC
        df["squeeze_on"] = (bb_lower > kc_lower) & (bb_upper < kc_upper)

        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for idx, row in df.iterrows():
            if pd.isna(row.get("squeeze_on")):
                continue

            if row["squeeze_on"]:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.NEUTRAL,
                    confidence=0.85,
                    metadata={"reason": "keltner_squeeze"}
                ))

        return signals


class RSIMidZoneStrategy(SignalGenerator):
    """
    RSI in mid-zone (40-60) indicates no clear momentum = ranging.
    """

    def __init__(self, period: int = 14, low_bound: float = 40, high_bound: float = 60):
        super().__init__(name="rsi_mid")
        self.period = period
        self.low_bound = low_bound
        self.high_bound = high_bound

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["rsi"] = rsi(df["close"], self.period)
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for idx, row in df.iterrows():
            if pd.isna(row.get("rsi")):
                continue

            rsi_val = row["rsi"]

            if self.low_bound <= rsi_val <= self.high_bound:
                # RSI in neutral zone
                # Confidence based on how close to 50
                distance_from_50 = abs(rsi_val - 50)
                confidence = 0.5 + (10 - distance_from_50) * 0.04  # 0.5-0.9 range
                confidence = min(0.9, max(0.5, confidence))

                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.NEUTRAL,
                    confidence=confidence,
                    metadata={"rsi": rsi_val, "reason": "rsi_neutral_zone"}
                ))

        return signals


class ATRContractionStrategy(SignalGenerator):
    """
    Detects ATR contraction indicating reduced volatility = ranging.
    """

    def __init__(self, atr_period: int = 14, lookback: int = 50):
        super().__init__(name="atr_contract")
        self.atr_period = atr_period
        self.lookback = lookback

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["atr"] = atr(df["high"], df["low"], df["close"], self.atr_period)
        df["atr_percentile"] = df["atr"].rolling(window=self.lookback).apply(
            lambda x: (x.iloc[-1] <= x).mean() * 100 if len(x) > 0 else 50
        )
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for idx, row in df.iterrows():
            if pd.isna(row.get("atr_percentile")):
                continue

            percentile = row["atr_percentile"]

            if percentile < 15:
                # Very low volatility
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.NEUTRAL,
                    confidence=0.85,
                    metadata={"atr_percentile": percentile, "reason": "very_low_volatility"}
                ))
            elif percentile < 30:
                # Low volatility
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.NEUTRAL,
                    confidence=0.65,
                    metadata={"atr_percentile": percentile, "reason": "low_volatility"}
                ))

        return signals


class PriceChannelStrategy(SignalGenerator):
    """
    Detects when price is near the middle of its range = ranging.

    If price is far from recent highs AND far from recent lows,
    it suggests no clear direction.
    """

    def __init__(self, period: int = 20, middle_zone: float = 0.3):
        super().__init__(name="price_channel")
        self.period = period
        self.middle_zone = middle_zone  # 30% around the middle

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["channel_high"] = df["high"].rolling(window=self.period).max()
        df["channel_low"] = df["low"].rolling(window=self.period).min()
        df["channel_range"] = df["channel_high"] - df["channel_low"]

        # Position in channel (0 = at low, 1 = at high)
        df["channel_position"] = (df["close"] - df["channel_low"]) / df["channel_range"].replace(0, np.nan)

        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        lower_bound = 0.5 - self.middle_zone / 2  # 0.35
        upper_bound = 0.5 + self.middle_zone / 2  # 0.65

        for idx, row in df.iterrows():
            if pd.isna(row.get("channel_position")):
                continue

            pos = row["channel_position"]

            if lower_bound <= pos <= upper_bound:
                # Price in middle of channel
                distance_from_center = abs(pos - 0.5)
                confidence = 0.5 + (0.15 - distance_from_center) * 3  # Higher when closer to 0.5
                confidence = min(0.9, max(0.5, confidence))

                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.NEUTRAL,
                    confidence=confidence,
                    metadata={"channel_position": pos, "reason": "price_mid_channel"}
                ))

        return signals


# Export all strategies
def get_ranging_strategies() -> List[SignalGenerator]:
    """Get all ranging strategy instances."""
    return [
        ADXRangingStrategy(),
        BollingerSqueezeStrategy(),
        ChopIndexStrategy(),
        KeltnerSqueezeStrategy(),
        RSIMidZoneStrategy(),
        ATRContractionStrategy(),
        PriceChannelStrategy(),
    ]


__all__ = [
    "ADXRangingStrategy",
    "BollingerSqueezeStrategy",
    "ChopIndexStrategy",
    "KeltnerSqueezeStrategy",
    "RSIMidZoneStrategy",
    "ATRContractionStrategy",
    "PriceChannelStrategy",
    "get_ranging_strategies",
]

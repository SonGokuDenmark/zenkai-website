"""
Volume-Based Trading Strategies.

These strategies use volume analysis to generate signals.
Volume often leads price - smart money moves first.

Strategies:
1. OBVTrend - On-Balance Volume trend following
2. VolumeBreakout - Price breakout on high volume
3. VolumeClimaxReversal - Extreme volume = reversal
4. AccumulationDistribution - A/D line divergence
5. VolumeWeightedMomentum - VWMA crossovers
6. ChaikinMoneyFlow - CMF overbought/oversold
7. ForceIndex - Elder's Force Index
8. VolumeProfile - High volume nodes as S/R
9. RelativeVolume - RVOL spikes
10. NegativeVolumeIndex - NVI divergence
"""

from typing import List
import pandas as pd
import numpy as np

from .base import (
    SignalGenerator, Signal, SignalDirection,
    sma, ema, obv
)


class OBVTrendStrategy(SignalGenerator):
    """
    On-Balance Volume Trend.

    OBV rising = accumulation = bullish
    OBV falling = distribution = bearish
    OBV divergence from price = reversal signal
    """

    def __init__(self, period: int = 20):
        super().__init__(name="obv_trend")
        self.period = period

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["obv"] = obv(df["close"], df["volume"])
        df["obv_sma"] = sma(df["obv"], self.period)
        df["obv_slope"] = df["obv"].diff(5) / 5
        df["price_slope"] = df["close"].diff(5) / 5
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(self.period, len(df)):
            row = df.iloc[i]

            if pd.isna(row["obv_sma"]):
                continue

            # OBV above SMA and rising = bullish
            if row["obv"] > row["obv_sma"] and row["obv_slope"] > 0:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.7,
                    metadata={"reason": "obv_bullish"}
                ))
            # OBV below SMA and falling = bearish
            elif row["obv"] < row["obv_sma"] and row["obv_slope"] < 0:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.7,
                    metadata={"reason": "obv_bearish"}
                ))

            # Bullish divergence: price falling but OBV rising
            if row["price_slope"] < 0 and row["obv_slope"] > 0:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.8,
                    metadata={"reason": "obv_bullish_divergence"}
                ))
            # Bearish divergence
            elif row["price_slope"] > 0 and row["obv_slope"] < 0:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.8,
                    metadata={"reason": "obv_bearish_divergence"}
                ))

        return signals


class VolumeBreakoutStrategy(SignalGenerator):
    """
    Volume Breakout - Price breakout confirmed by volume.

    Breakout on 2x+ average volume = more likely to follow through.
    """

    def __init__(self, price_period: int = 20, vol_period: int = 20, vol_mult: float = 2.0):
        super().__init__(name="volume_breakout")
        self.price_period = price_period
        self.vol_period = vol_period
        self.vol_mult = vol_mult

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["resistance"] = df["high"].rolling(window=self.price_period).max().shift(1)
        df["support"] = df["low"].rolling(window=self.price_period).min().shift(1)
        df["vol_avg"] = sma(df["volume"], self.vol_period)
        df["vol_ratio"] = df["volume"] / df["vol_avg"]
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for idx, row in df.iterrows():
            if pd.isna(row["resistance"]) or pd.isna(row["vol_ratio"]):
                continue

            high_volume = row["vol_ratio"] >= self.vol_mult

            # Bullish breakout on volume
            if row["close"] > row["resistance"] and high_volume:
                confidence = min(0.95, 0.6 + row["vol_ratio"] / 10)
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=confidence,
                    metadata={"vol_ratio": row["vol_ratio"], "reason": "volume_breakout_up"}
                ))

            # Bearish breakdown on volume
            elif row["close"] < row["support"] and high_volume:
                confidence = min(0.95, 0.6 + row["vol_ratio"] / 10)
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=confidence,
                    metadata={"vol_ratio": row["vol_ratio"], "reason": "volume_breakdown"}
                ))

        return signals


class VolumeClimaxStrategy(SignalGenerator):
    """
    Volume Climax Reversal.

    Extreme volume spikes often mark exhaustion points.
    Climax at top = sell, climax at bottom = buy.
    """

    def __init__(self, period: int = 20, climax_mult: float = 3.0):
        super().__init__(name="volume_climax")
        self.period = period
        self.climax_mult = climax_mult

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["vol_avg"] = sma(df["volume"], self.period)
        df["vol_ratio"] = df["volume"] / df["vol_avg"]
        df["price_change"] = df["close"].pct_change()

        # Detect if we're at swing high/low
        df["swing_high"] = df["high"] == df["high"].rolling(window=10, center=True).max()
        df["swing_low"] = df["low"] == df["low"].rolling(window=10, center=True).min()
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for idx, row in df.iterrows():
            if pd.isna(row["vol_ratio"]):
                continue

            is_climax = row["vol_ratio"] >= self.climax_mult

            if is_climax:
                # Climax on up move near high = reversal down
                if row["price_change"] > 0.01 and row.get("swing_high", False):
                    signals.append(Signal(
                        timestamp=row["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.SHORT,
                        confidence=0.8,
                        metadata={"vol_ratio": row["vol_ratio"], "reason": "climax_top"}
                    ))
                # Climax on down move near low = reversal up
                elif row["price_change"] < -0.01 and row.get("swing_low", False):
                    signals.append(Signal(
                        timestamp=row["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.LONG,
                        confidence=0.8,
                        metadata={"vol_ratio": row["vol_ratio"], "reason": "climax_bottom"}
                    ))

        return signals


class AccumulationDistributionStrategy(SignalGenerator):
    """
    Accumulation/Distribution Line.

    A/D rising while price flat = accumulation = bullish
    A/D falling while price flat = distribution = bearish
    """

    def __init__(self, period: int = 20):
        super().__init__(name="accum_dist")
        self.period = period

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Money Flow Multiplier
        mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"])
        mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)

        # Money Flow Volume
        mfv = mfm * df["volume"]

        # A/D Line
        df["ad_line"] = mfv.cumsum()
        df["ad_sma"] = sma(df["ad_line"], self.period)
        df["ad_slope"] = df["ad_line"].diff(5)
        df["price_slope"] = df["close"].diff(5)

        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for idx, row in df.iterrows():
            if pd.isna(row["ad_sma"]):
                continue

            # A/D line trend
            if row["ad_line"] > row["ad_sma"] and row["ad_slope"] > 0:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.65,
                    metadata={"reason": "ad_accumulation"}
                ))
            elif row["ad_line"] < row["ad_sma"] and row["ad_slope"] < 0:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.65,
                    metadata={"reason": "ad_distribution"}
                ))

            # Divergence
            if row["price_slope"] < 0 and row["ad_slope"] > 0:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.8,
                    metadata={"reason": "ad_bullish_divergence"}
                ))
            elif row["price_slope"] > 0 and row["ad_slope"] < 0:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.8,
                    metadata={"reason": "ad_bearish_divergence"}
                ))

        return signals


class ChaikinMoneyFlowStrategy(SignalGenerator):
    """
    Chaikin Money Flow (CMF).

    CMF > 0.1 = buying pressure = bullish
    CMF < -0.1 = selling pressure = bearish
    """

    def __init__(self, period: int = 20):
        super().__init__(name="cmf")
        self.period = period

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Money Flow Multiplier
        mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"])
        mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)

        # Money Flow Volume
        mfv = mfm * df["volume"]

        # CMF
        df["cmf"] = mfv.rolling(window=self.period).sum() / df["volume"].rolling(window=self.period).sum()

        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]

            if pd.isna(row["cmf"]):
                continue

            # Strong buying pressure
            if row["cmf"] > 0.1:
                confidence = min(0.9, 0.5 + row["cmf"])
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=confidence,
                    metadata={"cmf": row["cmf"], "reason": "cmf_bullish"}
                ))
            # Strong selling pressure
            elif row["cmf"] < -0.1:
                confidence = min(0.9, 0.5 + abs(row["cmf"]))
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=confidence,
                    metadata={"cmf": row["cmf"], "reason": "cmf_bearish"}
                ))

            # Zero line crossover
            if prev["cmf"] <= 0 and row["cmf"] > 0:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.75,
                    metadata={"reason": "cmf_cross_bullish"}
                ))
            elif prev["cmf"] >= 0 and row["cmf"] < 0:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.75,
                    metadata={"reason": "cmf_cross_bearish"}
                ))

        return signals


class ForceIndexStrategy(SignalGenerator):
    """
    Elder's Force Index.

    Force = Price Change × Volume
    Measures the power behind price moves.
    """

    def __init__(self, short_period: int = 2, long_period: int = 13):
        super().__init__(name="force_index")
        self.short_period = short_period
        self.long_period = long_period

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Raw Force Index
        df["force_raw"] = df["close"].diff() * df["volume"]

        # Smoothed versions
        df["force_short"] = ema(df["force_raw"], self.short_period)
        df["force_long"] = ema(df["force_raw"], self.long_period)

        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]

            if pd.isna(row["force_long"]):
                continue

            # Short-term Force crossover
            if prev["force_short"] <= 0 and row["force_short"] > 0:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.7,
                    metadata={"reason": "force_short_bullish"}
                ))
            elif prev["force_short"] >= 0 and row["force_short"] < 0:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.7,
                    metadata={"reason": "force_short_bearish"}
                ))

            # Long-term Force trend
            if row["force_long"] > 0:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.6,
                    metadata={"reason": "force_long_bullish"}
                ))
            elif row["force_long"] < 0:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.6,
                    metadata={"reason": "force_long_bearish"}
                ))

        return signals


class VWMAStrategy(SignalGenerator):
    """
    Volume Weighted Moving Average crossovers.

    VWMA gives more weight to high-volume periods.
    """

    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        super().__init__(name="vwma")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        def vwma(close, volume, period):
            return (close * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()

        df["vwma_fast"] = vwma(df["close"], df["volume"], self.fast_period)
        df["vwma_slow"] = vwma(df["close"], df["volume"], self.slow_period)

        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]

            if pd.isna(row["vwma_slow"]):
                continue

            # Golden cross
            if prev["vwma_fast"] <= prev["vwma_slow"] and row["vwma_fast"] > row["vwma_slow"]:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.8,
                    metadata={"reason": "vwma_golden_cross"}
                ))
            # Death cross
            elif prev["vwma_fast"] >= prev["vwma_slow"] and row["vwma_fast"] < row["vwma_slow"]:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.8,
                    metadata={"reason": "vwma_death_cross"}
                ))

            # Trend following
            if row["vwma_fast"] > row["vwma_slow"]:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.55,
                    metadata={"reason": "vwma_bullish"}
                ))
            else:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.55,
                    metadata={"reason": "vwma_bearish"}
                ))

        return signals


class RelativeVolumeStrategy(SignalGenerator):
    """
    Relative Volume (RVOL) Strategy.

    RVOL > 2 = unusual activity = potential move
    """

    def __init__(self, period: int = 20, threshold: float = 2.0):
        super().__init__(name="rvol")
        self.period = period
        self.threshold = threshold

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["vol_avg"] = sma(df["volume"], self.period)
        df["rvol"] = df["volume"] / df["vol_avg"]
        df["price_change"] = df["close"].pct_change()
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for idx, row in df.iterrows():
            if pd.isna(row["rvol"]):
                continue

            if row["rvol"] >= self.threshold:
                # High RVOL with up move = bullish
                if row["price_change"] > 0:
                    confidence = min(0.9, 0.5 + row["rvol"] / 10)
                    signals.append(Signal(
                        timestamp=row["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.LONG,
                        confidence=confidence,
                        metadata={"rvol": row["rvol"], "reason": "rvol_bullish"}
                    ))
                # High RVOL with down move = bearish
                elif row["price_change"] < 0:
                    confidence = min(0.9, 0.5 + row["rvol"] / 10)
                    signals.append(Signal(
                        timestamp=row["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.SHORT,
                        confidence=confidence,
                        metadata={"rvol": row["rvol"], "reason": "rvol_bearish"}
                    ))

        return signals


class MFIStrategy(SignalGenerator):
    """
    Money Flow Index - Volume-weighted RSI.

    MFI > 80 = overbought
    MFI < 20 = oversold
    """

    def __init__(self, period: int = 14):
        super().__init__(name="mfi")
        self.period = period

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        raw_money_flow = typical_price * df["volume"]

        positive_flow = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)

        positive_mf = pd.Series(positive_flow).rolling(window=self.period).sum()
        negative_mf = pd.Series(negative_flow).rolling(window=self.period).sum()

        mfi_ratio = positive_mf / negative_mf
        df["mfi"] = 100 - (100 / (1 + mfi_ratio))

        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]

            if pd.isna(row["mfi"]):
                continue

            # Oversold
            if row["mfi"] < 20:
                confidence = 0.7 + (20 - row["mfi"]) / 50
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=min(0.9, confidence),
                    metadata={"mfi": row["mfi"], "reason": "mfi_oversold"}
                ))
            # Overbought
            elif row["mfi"] > 80:
                confidence = 0.7 + (row["mfi"] - 80) / 50
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=min(0.9, confidence),
                    metadata={"mfi": row["mfi"], "reason": "mfi_overbought"}
                ))

            # 50 line crossover
            if prev["mfi"] <= 50 and row["mfi"] > 50:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.65,
                    metadata={"reason": "mfi_cross_bullish"}
                ))
            elif prev["mfi"] >= 50 and row["mfi"] < 50:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.65,
                    metadata={"reason": "mfi_cross_bearish"}
                ))

        return signals


def get_volume_strategies() -> List[SignalGenerator]:
    """Get all volume-based strategy instances."""
    return [
        OBVTrendStrategy(),
        VolumeBreakoutStrategy(),
        VolumeClimaxStrategy(),
        AccumulationDistributionStrategy(),
        ChaikinMoneyFlowStrategy(),
        ForceIndexStrategy(),
        VWMAStrategy(),
        RelativeVolumeStrategy(),
        MFIStrategy(),
    ]


__all__ = [
    "OBVTrendStrategy",
    "VolumeBreakoutStrategy",
    "VolumeClimaxStrategy",
    "AccumulationDistributionStrategy",
    "ChaikinMoneyFlowStrategy",
    "ForceIndexStrategy",
    "VWMAStrategy",
    "RelativeVolumeStrategy",
    "MFIStrategy",
    "get_volume_strategies",
]

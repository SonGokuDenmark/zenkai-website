"""
Volatility-Based Trading Strategies.

These strategies use volatility measures to time entries and exits.
Volatility expansion/contraction cycles are predictable.

Strategies:
1. ATRBreakout - Breakout based on ATR expansion
2. BollingerExpansion - Trade BB expansion after squeeze
3. KeltnerExpansion - Keltner channel expansion
4. VolatilityBreakout - Price moves > N*ATR
5. ATRTrailing - ATR-based trailing stops
6. HistoricalVolatility - HV rank trading
7. VIXStyle - Volatility mean reversion
8. RangeExpansion - Daily range expansion
9. ATRChannelBreak - ATR channel breakouts
10. VolatilityRegime - Trade based on vol regime
"""

from typing import List
import pandas as pd
import numpy as np

from .base import (
    SignalGenerator, Signal, SignalDirection,
    sma, ema, atr, bollinger_bands
)


class ATRBreakoutStrategy(SignalGenerator):
    """
    ATR Breakout - Enter when price moves more than N*ATR.

    Large moves relative to ATR indicate momentum.
    """

    def __init__(self, period: int = 14, multiplier: float = 1.5):
        super().__init__(name="atr_breakout")
        self.period = period
        self.multiplier = multiplier

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["atr"] = atr(df["high"], df["low"], df["close"], self.period)
        df["price_change"] = df["close"] - df["open"]
        df["atr_move"] = abs(df["price_change"]) / df["atr"]
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for idx, row in df.iterrows():
            if pd.isna(row["atr"]) or row["atr"] == 0:
                continue

            if row["atr_move"] >= self.multiplier:
                # Strong up move
                if row["price_change"] > 0:
                    confidence = min(0.9, 0.5 + row["atr_move"] / 5)
                    signals.append(Signal(
                        timestamp=row["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.LONG,
                        confidence=confidence,
                        metadata={"atr_move": row["atr_move"], "reason": "atr_breakout_up"}
                    ))
                # Strong down move
                else:
                    confidence = min(0.9, 0.5 + row["atr_move"] / 5)
                    signals.append(Signal(
                        timestamp=row["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.SHORT,
                        confidence=confidence,
                        metadata={"atr_move": row["atr_move"], "reason": "atr_breakout_down"}
                    ))

        return signals


class BollingerExpansionStrategy(SignalGenerator):
    """
    Bollinger Band Expansion after Squeeze.

    When BB width expands after contraction, trade the direction.
    """

    def __init__(self, period: int = 20, std: float = 2.0, squeeze_pct: float = 20):
        super().__init__(name="bb_expansion")
        self.period = period
        self.std = std
        self.squeeze_pct = squeeze_pct

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        upper, middle, lower = bollinger_bands(df["close"], self.period, self.std)
        df["bb_upper"] = upper
        df["bb_middle"] = middle
        df["bb_lower"] = lower
        df["bb_width"] = (upper - lower) / middle * 100

        # Width percentile
        df["width_percentile"] = df["bb_width"].rolling(window=50).apply(
            lambda x: (x.iloc[-1] > x).mean() * 100 if len(x) > 0 else 50
        )

        # Was in squeeze (low percentile) recently
        df["was_squeeze"] = df["width_percentile"].shift(1) < self.squeeze_pct

        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for idx, row in df.iterrows():
            if pd.isna(row["width_percentile"]):
                continue

            # Expansion out of squeeze
            if row["was_squeeze"] and row["width_percentile"] > self.squeeze_pct:
                # Direction based on price vs middle band
                if row["close"] > row["bb_middle"]:
                    signals.append(Signal(
                        timestamp=row["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.LONG,
                        confidence=0.8,
                        metadata={"reason": "bb_expansion_up"}
                    ))
                else:
                    signals.append(Signal(
                        timestamp=row["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.SHORT,
                        confidence=0.8,
                        metadata={"reason": "bb_expansion_down"}
                    ))

        return signals


class VolatilityBreakoutStrategy(SignalGenerator):
    """
    Price breakout measured in ATR units.

    Price moves > 2*ATR from previous close = significant.
    """

    def __init__(self, period: int = 14, threshold: float = 2.0):
        super().__init__(name="vol_breakout")
        self.period = period
        self.threshold = threshold

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["atr"] = atr(df["high"], df["low"], df["close"], self.period)
        df["move"] = df["close"] - df["close"].shift(1)
        df["move_atr"] = df["move"] / df["atr"]
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for idx, row in df.iterrows():
            if pd.isna(row["move_atr"]):
                continue

            if abs(row["move_atr"]) >= self.threshold:
                direction = SignalDirection.LONG if row["move_atr"] > 0 else SignalDirection.SHORT
                confidence = min(0.9, 0.5 + abs(row["move_atr"]) / 5)
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=direction,
                    confidence=confidence,
                    metadata={"move_atr": row["move_atr"], "reason": "volatility_breakout"}
                ))

        return signals


class HistoricalVolatilityStrategy(SignalGenerator):
    """
    Historical Volatility Rank trading.

    Low HV rank = expect expansion = trade breakout
    High HV rank = expect contraction = mean reversion
    """

    def __init__(self, period: int = 20, rank_period: int = 252):
        super().__init__(name="hv_rank")
        self.period = period
        self.rank_period = rank_period

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Historical volatility (annualized)
        returns = df["close"].pct_change()
        df["hv"] = returns.rolling(window=self.period).std() * np.sqrt(252) * 100

        # HV Rank (percentile over past year)
        df["hv_rank"] = df["hv"].rolling(window=self.rank_period).apply(
            lambda x: (x.iloc[-1] > x).mean() * 100 if len(x) > 0 else 50
        )

        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for idx, row in df.iterrows():
            if pd.isna(row["hv_rank"]):
                continue

            # Low HV rank = expect expansion = trade with momentum
            if row["hv_rank"] < 20:
                # Trade breakout direction
                price_change = row["close"] - df["close"].shift(1).loc[idx] if idx > 0 else 0
                if price_change > 0:
                    signals.append(Signal(
                        timestamp=row["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.LONG,
                        confidence=0.7,
                        metadata={"hv_rank": row["hv_rank"], "reason": "low_vol_breakout_up"}
                    ))
                elif price_change < 0:
                    signals.append(Signal(
                        timestamp=row["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.SHORT,
                        confidence=0.7,
                        metadata={"hv_rank": row["hv_rank"], "reason": "low_vol_breakout_down"}
                    ))

            # High HV rank = expect contraction = mean reversion
            elif row["hv_rank"] > 80:
                # Fade the move
                price_ma = sma(df["close"], 20).loc[idx]
                if row["close"] > price_ma:
                    signals.append(Signal(
                        timestamp=row["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.SHORT,
                        confidence=0.65,
                        metadata={"hv_rank": row["hv_rank"], "reason": "high_vol_reversion_short"}
                    ))
                else:
                    signals.append(Signal(
                        timestamp=row["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.LONG,
                        confidence=0.65,
                        metadata={"hv_rank": row["hv_rank"], "reason": "high_vol_reversion_long"}
                    ))

        return signals


class RangeExpansionStrategy(SignalGenerator):
    """
    Daily Range Expansion.

    When today's range exceeds recent average range significantly.
    """

    def __init__(self, period: int = 14, expansion_mult: float = 1.5):
        super().__init__(name="range_expansion")
        self.period = period
        self.expansion_mult = expansion_mult

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["range"] = df["high"] - df["low"]
        df["avg_range"] = sma(df["range"], self.period)
        df["range_ratio"] = df["range"] / df["avg_range"]
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for idx, row in df.iterrows():
            if pd.isna(row["range_ratio"]):
                continue

            if row["range_ratio"] >= self.expansion_mult:
                # Range expansion - trade in direction of close
                if row["close"] > row["open"]:
                    confidence = min(0.9, 0.5 + row["range_ratio"] / 5)
                    signals.append(Signal(
                        timestamp=row["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.LONG,
                        confidence=confidence,
                        metadata={"range_ratio": row["range_ratio"], "reason": "range_expand_up"}
                    ))
                else:
                    confidence = min(0.9, 0.5 + row["range_ratio"] / 5)
                    signals.append(Signal(
                        timestamp=row["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.SHORT,
                        confidence=confidence,
                        metadata={"range_ratio": row["range_ratio"], "reason": "range_expand_down"}
                    ))

        return signals


class ATRChannelStrategy(SignalGenerator):
    """
    ATR Channel Breakout.

    Price breaking above/below ATR channel from EMA.
    """

    def __init__(self, ema_period: int = 20, atr_period: int = 14, multiplier: float = 2.0):
        super().__init__(name="atr_channel")
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.multiplier = multiplier

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ema"] = ema(df["close"], self.ema_period)
        df["atr"] = atr(df["high"], df["low"], df["close"], self.atr_period)
        df["upper_channel"] = df["ema"] + self.multiplier * df["atr"]
        df["lower_channel"] = df["ema"] - self.multiplier * df["atr"]
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]

            if pd.isna(row["upper_channel"]):
                continue

            # Break above upper channel
            if prev["close"] <= prev["upper_channel"] and row["close"] > row["upper_channel"]:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.8,
                    metadata={"reason": "atr_channel_break_up"}
                ))
            # Break below lower channel
            elif prev["close"] >= prev["lower_channel"] and row["close"] < row["lower_channel"]:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.8,
                    metadata={"reason": "atr_channel_break_down"}
                ))

            # Trend following within channel
            if row["close"] > row["ema"]:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.55,
                    metadata={"reason": "above_ema"}
                ))
            else:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.55,
                    metadata={"reason": "below_ema"}
                ))

        return signals


class VolatilityContractStrategy(SignalGenerator):
    """
    Volatility Contraction Pattern (VCP).

    Decreasing volatility = coiling = expect breakout.
    """

    def __init__(self, period: int = 14, lookback: int = 5):
        super().__init__(name="vol_contract")
        self.period = period
        self.lookback = lookback

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["atr"] = atr(df["high"], df["low"], df["close"], self.period)
        df["atr_slope"] = df["atr"].diff(self.lookback) / self.lookback
        df["contracting"] = df["atr_slope"] < 0
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(self.lookback, len(df)):
            row = df.iloc[i]

            if pd.isna(row["atr_slope"]):
                continue

            # Check for consistent contraction
            recent = df.iloc[i-self.lookback:i]
            if recent["contracting"].all():
                # Breakout direction
                if row["close"] > row["open"]:
                    signals.append(Signal(
                        timestamp=row["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.LONG,
                        confidence=0.75,
                        metadata={"reason": "vcp_breakout_up"}
                    ))
                elif row["close"] < row["open"]:
                    signals.append(Signal(
                        timestamp=row["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.SHORT,
                        confidence=0.75,
                        metadata={"reason": "vcp_breakout_down"}
                    ))

        return signals


class StandardDeviationStrategy(SignalGenerator):
    """
    Standard Deviation based entries.

    Enter when price moves N standard deviations.
    """

    def __init__(self, period: int = 20, threshold: float = 2.0):
        super().__init__(name="std_dev")
        self.period = period
        self.threshold = threshold

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

            # Mean reversion when extended
            if row["z_score"] > self.threshold:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=min(0.9, 0.5 + abs(row["z_score"]) / 5),
                    metadata={"z_score": row["z_score"], "reason": "std_overbought"}
                ))
            elif row["z_score"] < -self.threshold:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=min(0.9, 0.5 + abs(row["z_score"]) / 5),
                    metadata={"z_score": row["z_score"], "reason": "std_oversold"}
                ))

        return signals


class TrueRangeStrategy(SignalGenerator):
    """
    True Range breakout strategy.

    Large true range = significant move.
    """

    def __init__(self, period: int = 14, multiplier: float = 1.5):
        super().__init__(name="true_range")
        self.period = period
        self.multiplier = multiplier

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # True Range
        tr1 = df["high"] - df["low"]
        tr2 = abs(df["high"] - df["close"].shift(1))
        tr3 = abs(df["low"] - df["close"].shift(1))
        df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        df["atr"] = df["tr"].rolling(window=self.period).mean()
        df["tr_ratio"] = df["tr"] / df["atr"]

        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for idx, row in df.iterrows():
            if pd.isna(row["tr_ratio"]):
                continue

            if row["tr_ratio"] >= self.multiplier:
                # Direction based on close position
                if row["close"] > row["open"]:
                    signals.append(Signal(
                        timestamp=row["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.LONG,
                        confidence=min(0.9, 0.5 + row["tr_ratio"] / 5),
                        metadata={"tr_ratio": row["tr_ratio"], "reason": "tr_breakout_up"}
                    ))
                else:
                    signals.append(Signal(
                        timestamp=row["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.SHORT,
                        confidence=min(0.9, 0.5 + row["tr_ratio"] / 5),
                        metadata={"tr_ratio": row["tr_ratio"], "reason": "tr_breakout_down"}
                    ))

        return signals


def get_volatility_strategies() -> List[SignalGenerator]:
    """Get all volatility-based strategy instances."""
    return [
        ATRBreakoutStrategy(),
        BollingerExpansionStrategy(),
        VolatilityBreakoutStrategy(),
        HistoricalVolatilityStrategy(),
        RangeExpansionStrategy(),
        ATRChannelStrategy(),
        VolatilityContractStrategy(),
        StandardDeviationStrategy(),
        TrueRangeStrategy(),
    ]


__all__ = [
    "ATRBreakoutStrategy",
    "BollingerExpansionStrategy",
    "VolatilityBreakoutStrategy",
    "HistoricalVolatilityStrategy",
    "RangeExpansionStrategy",
    "ATRChannelStrategy",
    "VolatilityContractStrategy",
    "StandardDeviationStrategy",
    "TrueRangeStrategy",
    "get_volatility_strategies",
]

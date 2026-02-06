"""
Momentum-Based Trading Strategies.

Pure momentum and trend-following strategies.
"The trend is your friend" - trade with momentum.

Strategies:
1. ROCStrategy - Rate of Change momentum
2. MomentumOscillator - Classic momentum
3. TSIStrategy - True Strength Index
4. WilliamsR - Williams %R
5. UltimateOscillator - Ultimate Oscillator
6. AOMStrategy - Awesome Oscillator
7. PPOStrategy - Percentage Price Oscillator
8. KSTStrategy - Know Sure Thing
9. DMIStrategy - Directional Movement Index
10. AroonStrategy - Aroon indicator
"""

from typing import List
import pandas as pd
import numpy as np

from .base import (
    SignalGenerator, Signal, SignalDirection,
    sma, ema, rsi
)


class ROCStrategy(SignalGenerator):
    """
    Rate of Change momentum.

    ROC = (Price - Price_n) / Price_n * 100
    """

    def __init__(self, period: int = 12, signal_period: int = 9):
        super().__init__(name="roc")
        self.period = period
        self.signal_period = signal_period

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["roc"] = (df["close"] - df["close"].shift(self.period)) / df["close"].shift(self.period) * 100
        df["roc_signal"] = sma(df["roc"], self.signal_period)
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]

            if pd.isna(row["roc_signal"]):
                continue

            # ROC crosses above signal line
            if prev["roc"] <= prev["roc_signal"] and row["roc"] > row["roc_signal"]:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.75,
                    metadata={"roc": row["roc"], "reason": "roc_cross_up"}
                ))
            # ROC crosses below signal line
            elif prev["roc"] >= prev["roc_signal"] and row["roc"] < row["roc_signal"]:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.75,
                    metadata={"roc": row["roc"], "reason": "roc_cross_down"}
                ))

            # Zero line cross
            if prev["roc"] <= 0 and row["roc"] > 0:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.7,
                    metadata={"reason": "roc_zero_cross_up"}
                ))
            elif prev["roc"] >= 0 and row["roc"] < 0:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.7,
                    metadata={"reason": "roc_zero_cross_down"}
                ))

        return signals


class MomentumOscillatorStrategy(SignalGenerator):
    """
    Classic Momentum Oscillator.

    Momentum = Close - Close_n
    """

    def __init__(self, period: int = 14):
        super().__init__(name="momentum")
        self.period = period

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["momentum"] = df["close"] - df["close"].shift(self.period)
        df["momentum_ma"] = sma(df["momentum"], 5)
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]

            if pd.isna(row["momentum"]):
                continue

            # Zero line cross
            if prev["momentum"] <= 0 and row["momentum"] > 0:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.75,
                    metadata={"momentum": row["momentum"], "reason": "momentum_bullish"}
                ))
            elif prev["momentum"] >= 0 and row["momentum"] < 0:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.75,
                    metadata={"momentum": row["momentum"], "reason": "momentum_bearish"}
                ))

            # Strong momentum
            if row["momentum"] > 0:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.55,
                    metadata={"reason": "positive_momentum"}
                ))
            else:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.55,
                    metadata={"reason": "negative_momentum"}
                ))

        return signals


class TSIStrategy(SignalGenerator):
    """
    True Strength Index.

    Double-smoothed momentum oscillator.
    """

    def __init__(self, long_period: int = 25, short_period: int = 13, signal_period: int = 7):
        super().__init__(name="tsi")
        self.long_period = long_period
        self.short_period = short_period
        self.signal_period = signal_period

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Price change
        pc = df["close"].diff()

        # Double smooth price change
        pcs = ema(ema(pc, self.long_period), self.short_period)

        # Double smooth absolute price change
        apcs = ema(ema(abs(pc), self.long_period), self.short_period)

        # TSI
        df["tsi"] = 100 * pcs / apcs
        df["tsi_signal"] = ema(df["tsi"], self.signal_period)

        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]

            if pd.isna(row["tsi_signal"]):
                continue

            # TSI crosses signal line
            if prev["tsi"] <= prev["tsi_signal"] and row["tsi"] > row["tsi_signal"]:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.8,
                    metadata={"tsi": row["tsi"], "reason": "tsi_cross_up"}
                ))
            elif prev["tsi"] >= prev["tsi_signal"] and row["tsi"] < row["tsi_signal"]:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.8,
                    metadata={"tsi": row["tsi"], "reason": "tsi_cross_down"}
                ))

            # Zero line cross
            if prev["tsi"] <= 0 and row["tsi"] > 0:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.7,
                    metadata={"reason": "tsi_zero_cross_up"}
                ))
            elif prev["tsi"] >= 0 and row["tsi"] < 0:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.7,
                    metadata={"reason": "tsi_zero_cross_down"}
                ))

        return signals


class WilliamsRStrategy(SignalGenerator):
    """
    Williams %R.

    Momentum oscillator showing close relative to high-low range.
    """

    def __init__(self, period: int = 14):
        super().__init__(name="williams_r")
        self.period = period

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        highest_high = df["high"].rolling(window=self.period).max()
        lowest_low = df["low"].rolling(window=self.period).min()
        df["williams_r"] = -100 * (highest_high - df["close"]) / (highest_high - lowest_low)
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]

            if pd.isna(row["williams_r"]):
                continue

            # Oversold (< -80) -> buy
            if row["williams_r"] < -80:
                if prev["williams_r"] >= -80:  # Just crossed into oversold
                    signals.append(Signal(
                        timestamp=row["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.LONG,
                        confidence=0.8,
                        metadata={"williams_r": row["williams_r"], "reason": "williams_oversold"}
                    ))
                else:
                    signals.append(Signal(
                        timestamp=row["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.LONG,
                        confidence=0.65,
                        metadata={"reason": "williams_oversold_cont"}
                    ))

            # Overbought (> -20) -> sell
            elif row["williams_r"] > -20:
                if prev["williams_r"] <= -20:  # Just crossed into overbought
                    signals.append(Signal(
                        timestamp=row["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.SHORT,
                        confidence=0.8,
                        metadata={"williams_r": row["williams_r"], "reason": "williams_overbought"}
                    ))
                else:
                    signals.append(Signal(
                        timestamp=row["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.SHORT,
                        confidence=0.65,
                        metadata={"reason": "williams_overbought_cont"}
                    ))

        return signals


class UltimateOscillatorStrategy(SignalGenerator):
    """
    Ultimate Oscillator.

    Combines three timeframes to reduce false signals.
    """

    def __init__(self, period1: int = 7, period2: int = 14, period3: int = 28):
        super().__init__(name="ultimate_osc")
        self.period1 = period1
        self.period2 = period2
        self.period3 = period3

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Buying Pressure
        bp = df["close"] - pd.concat([df["low"], df["close"].shift(1)], axis=1).min(axis=1)

        # True Range
        tr1 = df["high"] - df["low"]
        tr2 = abs(df["high"] - df["close"].shift(1))
        tr3 = abs(df["low"] - df["close"].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Averages
        avg1 = bp.rolling(window=self.period1).sum() / tr.rolling(window=self.period1).sum()
        avg2 = bp.rolling(window=self.period2).sum() / tr.rolling(window=self.period2).sum()
        avg3 = bp.rolling(window=self.period3).sum() / tr.rolling(window=self.period3).sum()

        # Ultimate Oscillator
        df["uo"] = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7

        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]

            if pd.isna(row["uo"]):
                continue

            # Oversold < 30
            if row["uo"] < 30:
                confidence = 0.7 + (30 - row["uo"]) / 100
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=min(0.9, confidence),
                    metadata={"uo": row["uo"], "reason": "uo_oversold"}
                ))

            # Overbought > 70
            elif row["uo"] > 70:
                confidence = 0.7 + (row["uo"] - 70) / 100
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=min(0.9, confidence),
                    metadata={"uo": row["uo"], "reason": "uo_overbought"}
                ))

            # 50 line cross
            if prev["uo"] <= 50 and row["uo"] > 50:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.65,
                    metadata={"reason": "uo_cross_bullish"}
                ))
            elif prev["uo"] >= 50 and row["uo"] < 50:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.65,
                    metadata={"reason": "uo_cross_bearish"}
                ))

        return signals


class AwesomeOscillatorStrategy(SignalGenerator):
    """
    Awesome Oscillator (AO).

    Difference between 5-period and 34-period SMA of median price.
    """

    def __init__(self, fast_period: int = 5, slow_period: int = 34):
        super().__init__(name="awesome_osc")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        median_price = (df["high"] + df["low"]) / 2
        df["ao"] = sma(median_price, self.fast_period) - sma(median_price, self.slow_period)
        df["ao_color"] = np.where(df["ao"] > df["ao"].shift(1), 1, -1)
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(2, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            prev2 = df.iloc[i-2]

            if pd.isna(row["ao"]):
                continue

            # Zero line cross
            if prev["ao"] <= 0 and row["ao"] > 0:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.8,
                    metadata={"ao": row["ao"], "reason": "ao_zero_cross_up"}
                ))
            elif prev["ao"] >= 0 and row["ao"] < 0:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.8,
                    metadata={"ao": row["ao"], "reason": "ao_zero_cross_down"}
                ))

            # Saucer (color change above/below zero)
            if row["ao"] > 0 and prev["ao_color"] == -1 and row["ao_color"] == 1:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.75,
                    metadata={"reason": "ao_saucer_bullish"}
                ))
            elif row["ao"] < 0 and prev["ao_color"] == 1 and row["ao_color"] == -1:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.75,
                    metadata={"reason": "ao_saucer_bearish"}
                ))

        return signals


class PPOStrategy(SignalGenerator):
    """
    Percentage Price Oscillator (PPO).

    Like MACD but as percentage, better for comparing across assets.
    """

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__(name="ppo")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        fast_ema = ema(df["close"], self.fast_period)
        slow_ema = ema(df["close"], self.slow_period)
        df["ppo"] = (fast_ema - slow_ema) / slow_ema * 100
        df["ppo_signal"] = ema(df["ppo"], self.signal_period)
        df["ppo_hist"] = df["ppo"] - df["ppo_signal"]
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]

            if pd.isna(row["ppo_signal"]):
                continue

            # Signal line cross
            if prev["ppo"] <= prev["ppo_signal"] and row["ppo"] > row["ppo_signal"]:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.8,
                    metadata={"ppo": row["ppo"], "reason": "ppo_cross_up"}
                ))
            elif prev["ppo"] >= prev["ppo_signal"] and row["ppo"] < row["ppo_signal"]:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.8,
                    metadata={"ppo": row["ppo"], "reason": "ppo_cross_down"}
                ))

            # Zero line cross
            if prev["ppo"] <= 0 and row["ppo"] > 0:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.7,
                    metadata={"reason": "ppo_zero_cross_up"}
                ))
            elif prev["ppo"] >= 0 and row["ppo"] < 0:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.7,
                    metadata={"reason": "ppo_zero_cross_down"}
                ))

        return signals


class AroonStrategy(SignalGenerator):
    """
    Aroon Indicator.

    Measures time since highest high and lowest low.
    """

    def __init__(self, period: int = 25):
        super().__init__(name="aroon")
        self.period = period

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        def aroon_up(high, period):
            return high.rolling(window=period + 1).apply(
                lambda x: (period - (period - x.argmax())) / period * 100
            )

        def aroon_down(low, period):
            return low.rolling(window=period + 1).apply(
                lambda x: (period - (period - x.argmin())) / period * 100
            )

        df["aroon_up"] = aroon_up(df["high"], self.period)
        df["aroon_down"] = aroon_down(df["low"], self.period)
        df["aroon_osc"] = df["aroon_up"] - df["aroon_down"]

        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]

            if pd.isna(row["aroon_osc"]):
                continue

            # Aroon Up crosses above Aroon Down
            if prev["aroon_up"] <= prev["aroon_down"] and row["aroon_up"] > row["aroon_down"]:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.8,
                    metadata={"aroon_osc": row["aroon_osc"], "reason": "aroon_cross_up"}
                ))
            elif prev["aroon_up"] >= prev["aroon_down"] and row["aroon_up"] < row["aroon_down"]:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.8,
                    metadata={"aroon_osc": row["aroon_osc"], "reason": "aroon_cross_down"}
                ))

            # Strong trend (Aroon Up > 70 and Down < 30)
            if row["aroon_up"] > 70 and row["aroon_down"] < 30:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.75,
                    metadata={"reason": "aroon_strong_up"}
                ))
            elif row["aroon_down"] > 70 and row["aroon_up"] < 30:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.75,
                    metadata={"reason": "aroon_strong_down"}
                ))

        return signals


class CCIStrategy(SignalGenerator):
    """
    Commodity Channel Index.

    Measures price deviation from statistical mean.
    """

    def __init__(self, period: int = 20):
        super().__init__(name="cci")
        self.period = period

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        sma_tp = typical_price.rolling(window=self.period).mean()
        mean_dev = typical_price.rolling(window=self.period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        df["cci"] = (typical_price - sma_tp) / (0.015 * mean_dev)
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        df = self.compute_indicators(df)
        signals = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]

            if pd.isna(row["cci"]):
                continue

            # Oversold < -100
            if row["cci"] < -100:
                if prev["cci"] >= -100:  # Just crossed
                    signals.append(Signal(
                        timestamp=row["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.LONG,
                        confidence=0.8,
                        metadata={"cci": row["cci"], "reason": "cci_oversold"}
                    ))

            # Overbought > 100
            elif row["cci"] > 100:
                if prev["cci"] <= 100:  # Just crossed
                    signals.append(Signal(
                        timestamp=row["timestamp"],
                        symbol=symbol,
                        direction=SignalDirection.SHORT,
                        confidence=0.8,
                        metadata={"cci": row["cci"], "reason": "cci_overbought"}
                    ))

            # Zero line cross for trend
            if prev["cci"] <= 0 and row["cci"] > 0:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence=0.65,
                    metadata={"reason": "cci_zero_cross_up"}
                ))
            elif prev["cci"] >= 0 and row["cci"] < 0:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence=0.65,
                    metadata={"reason": "cci_zero_cross_down"}
                ))

        return signals


def get_momentum_strategies() -> List[SignalGenerator]:
    """Get all momentum-based strategy instances."""
    return [
        ROCStrategy(),
        MomentumOscillatorStrategy(),
        TSIStrategy(),
        WilliamsRStrategy(),
        UltimateOscillatorStrategy(),
        AwesomeOscillatorStrategy(),
        PPOStrategy(),
        AroonStrategy(),
        CCIStrategy(),
    ]


__all__ = [
    "ROCStrategy",
    "MomentumOscillatorStrategy",
    "TSIStrategy",
    "WilliamsRStrategy",
    "UltimateOscillatorStrategy",
    "AwesomeOscillatorStrategy",
    "PPOStrategy",
    "AroonStrategy",
    "CCIStrategy",
    "get_momentum_strategies",
]

"""
Stochastic Mean Reversion Strategy.

Buys oversold conditions, sells overbought conditions.
"""

from typing import List, Dict, Any
import pandas as pd
import numpy as np

from .base import (
    SignalGenerator, Signal, SignalDirection,
    stochastic, ema, atr
)


class StochasticMRStrategy(SignalGenerator):
    """
    Stochastic Mean Reversion Strategy.

    Generates LONG signal when Stochastic is oversold and turning up.
    Generates SHORT signal when Stochastic is overbought and turning down.

    Optionally uses trend filter to only take signals in trend direction.

    Parameters:
        k_period: %K period (default: 14)
        d_period: %D smoothing period (default: 3)
        overbought: Overbought threshold (default: 80)
        oversold: Oversold threshold (default: 20)
        use_trend_filter: Only take signals in trend direction (default: True)
    """

    def __init__(
        self,
        k_period: int = 14,
        d_period: int = 3,
        overbought: int = 80,
        oversold: int = 20,
        use_trend_filter: bool = True,
        **kwargs
    ):
        super().__init__(
            name="stoch_mr",
            k_period=k_period,
            d_period=d_period,
            overbought=overbought,
            oversold=oversold,
            use_trend_filter=use_trend_filter,
            **kwargs
        )
        self.k_period = k_period
        self.d_period = d_period
        self.overbought = overbought
        self.oversold = oversold
        self.use_trend_filter = use_trend_filter

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute stochastic indicators."""
        df = df.copy()

        k, d = stochastic(
            df["high"], df["low"], df["close"],
            k_period=self.k_period,
            d_period=self.d_period
        )

        df["stoch_k"] = k
        df["stoch_d"] = d

        # Previous values for crossover detection
        df["stoch_k_prev"] = df["stoch_k"].shift(1)
        df["stoch_d_prev"] = df["stoch_d"].shift(1)

        # Trend filter
        if self.use_trend_filter:
            df["ema_50"] = ema(df["close"], 50)
            df["ema_200"] = ema(df["close"], 200)

        # ATR for stop loss
        df["atr"] = atr(df["high"], df["low"], df["close"], 14)

        return df

    def generate_signals(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> List[Signal]:
        """Generate stochastic mean reversion signals."""
        df = self.compute_indicators(df)
        signals = []

        for idx in range(1, len(df)):
            row = df.iloc[idx]
            prev_row = df.iloc[idx - 1]

            # Skip if indicators not ready
            if pd.isna(row["stoch_k"]) or pd.isna(row["stoch_d"]):
                continue

            signal_dir = SignalDirection.NEUTRAL
            confidence = 0.5

            # Oversold condition: %K crosses above %D from oversold zone
            if (prev_row["stoch_k"] < self.oversold and
                row["stoch_k"] >= row["stoch_d"] and
                prev_row["stoch_k"] < prev_row["stoch_d"]):

                # Apply trend filter
                if self.use_trend_filter:
                    if pd.isna(row["ema_50"]) or pd.isna(row["ema_200"]):
                        continue
                    # Only long if in uptrend (50 > 200)
                    if row["ema_50"] <= row["ema_200"]:
                        continue
                    confidence = 0.7  # Higher confidence with trend
                else:
                    confidence = 0.55

                signal_dir = SignalDirection.LONG

            # Overbought condition: %K crosses below %D from overbought zone
            elif (prev_row["stoch_k"] > self.overbought and
                  row["stoch_k"] <= row["stoch_d"] and
                  prev_row["stoch_k"] > prev_row["stoch_d"]):

                if self.use_trend_filter:
                    if pd.isna(row["ema_50"]) or pd.isna(row["ema_200"]):
                        continue
                    # Only short if in downtrend (50 < 200)
                    if row["ema_50"] >= row["ema_200"]:
                        continue
                    confidence = 0.7
                else:
                    confidence = 0.55

                signal_dir = SignalDirection.SHORT

            if signal_dir != SignalDirection.NEUTRAL:
                entry_price = row["close"]
                atr_val = row["atr"] if not pd.isna(row["atr"]) else entry_price * 0.02

                if signal_dir == SignalDirection.LONG:
                    stop_loss = entry_price - 1.5 * atr_val
                    take_profit = entry_price + 2 * atr_val
                else:
                    stop_loss = entry_price + 1.5 * atr_val
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
                        "stoch_k": row["stoch_k"],
                        "stoch_d": row["stoch_d"],
                        "zone": "oversold" if signal_dir == SignalDirection.LONG else "overbought"
                    }
                ))

        return signals

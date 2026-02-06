"""
Support and Resistance Bounce Strategy.

Detects key S/R levels and generates signals on bounces.
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

from .base import (
    SignalGenerator, Signal, SignalDirection,
    pivot_high, pivot_low, atr, ema
)


class SupportResistanceStrategy(SignalGenerator):
    """
    Support and Resistance Bounce Strategy.

    Identifies S/R zones from pivot points and generates signals
    when price tests and rejects these zones.

    Parameters:
        pivot_lookback: Bars on each side to confirm pivot (default: 5)
        zone_atr_mult: ATR multiplier for zone width (default: 0.5)
        min_touches: Minimum touches to validate zone (default: 2)
        rejection_candle: Require rejection candle pattern (default: True)
    """

    def __init__(
        self,
        pivot_lookback: int = 5,
        zone_atr_mult: float = 0.5,
        min_touches: int = 2,
        rejection_candle: bool = True,
        **kwargs
    ):
        super().__init__(
            name="sr_bounce",
            pivot_lookback=pivot_lookback,
            zone_atr_mult=zone_atr_mult,
            min_touches=min_touches,
            rejection_candle=rejection_candle,
            **kwargs
        )
        self.pivot_lookback = pivot_lookback
        self.zone_atr_mult = zone_atr_mult
        self.min_touches = min_touches
        self.rejection_candle = rejection_candle

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute S/R indicators."""
        df = df.copy()

        # Pivot points
        df["pivot_high"] = pivot_high(df["high"], self.pivot_lookback)
        df["pivot_low"] = pivot_low(df["low"], self.pivot_lookback)

        # ATR for zone calculation
        df["atr"] = atr(df["high"], df["low"], df["close"], 14)

        # EMA for trend context
        df["ema_50"] = ema(df["close"], 50)

        # Candle body and wick analysis
        df["body"] = abs(df["close"] - df["open"])
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
        df["is_bullish"] = df["close"] > df["open"]

        return df

    def _find_sr_zones(
        self,
        df: pd.DataFrame,
        current_idx: int,
        lookback: int = 100
    ) -> Tuple[List[float], List[float]]:
        """
        Find support and resistance zones from historical pivots.

        Returns:
            Tuple of (resistance_levels, support_levels)
        """
        start_idx = max(0, current_idx - lookback)
        subset = df.iloc[start_idx:current_idx]

        resistance_levels = []
        support_levels = []

        # Collect pivot highs (resistance)
        pivots_high = subset[subset["pivot_high"].notna()]["pivot_high"].values
        for level in pivots_high:
            resistance_levels.append(level)

        # Collect pivot lows (support)
        pivots_low = subset[subset["pivot_low"].notna()]["pivot_low"].values
        for level in pivots_low:
            support_levels.append(level)

        return resistance_levels, support_levels

    def _is_rejection_candle(
        self,
        row: pd.Series,
        direction: str
    ) -> bool:
        """
        Check if candle shows rejection pattern.

        Args:
            row: Current candle row
            direction: 'support' or 'resistance'

        Returns:
            True if rejection pattern detected
        """
        body = row["body"]
        upper_wick = row["upper_wick"]
        lower_wick = row["lower_wick"]

        # Minimum wick to body ratio for rejection
        min_wick_ratio = 1.5

        if direction == "support":
            # Lower wick should be longer than body for support rejection
            if body > 0 and lower_wick / body >= min_wick_ratio:
                return row["is_bullish"]  # Bullish close preferred
            elif body == 0 and lower_wick > 0:  # Doji with lower wick
                return True
        else:  # resistance
            # Upper wick should be longer than body for resistance rejection
            if body > 0 and upper_wick / body >= min_wick_ratio:
                return not row["is_bullish"]  # Bearish close preferred
            elif body == 0 and upper_wick > 0:  # Doji with upper wick
                return True

        return False

    def _price_in_zone(
        self,
        price: float,
        level: float,
        zone_width: float
    ) -> bool:
        """Check if price is within zone around level."""
        return abs(price - level) <= zone_width

    def generate_signals(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> List[Signal]:
        """Generate S/R bounce signals."""
        df = self.compute_indicators(df)
        signals = []

        # Need enough history for pivot detection
        min_bars = max(self.pivot_lookback * 2, 50)

        for idx in range(min_bars, len(df)):
            row = df.iloc[idx]

            if pd.isna(row["atr"]):
                continue

            zone_width = row["atr"] * self.zone_atr_mult

            # Find S/R zones from history
            resistance_levels, support_levels = self._find_sr_zones(df, idx)

            if not resistance_levels and not support_levels:
                continue

            signal_dir = SignalDirection.NEUTRAL
            confidence = 0.5
            level_touched = None
            zone_type = None

            # Check for support bounce
            for support in support_levels:
                if self._price_in_zone(row["low"], support, zone_width):
                    # Price touched support zone
                    if self.rejection_candle:
                        if self._is_rejection_candle(row, "support"):
                            signal_dir = SignalDirection.LONG
                            level_touched = support
                            zone_type = "support"
                            confidence = 0.65
                            break
                    else:
                        # Just touching support is enough
                        if row["close"] > support:  # Bouncing up
                            signal_dir = SignalDirection.LONG
                            level_touched = support
                            zone_type = "support"
                            confidence = 0.55
                            break

            # Check for resistance rejection (only if no support signal)
            if signal_dir == SignalDirection.NEUTRAL:
                for resistance in resistance_levels:
                    if self._price_in_zone(row["high"], resistance, zone_width):
                        if self.rejection_candle:
                            if self._is_rejection_candle(row, "resistance"):
                                signal_dir = SignalDirection.SHORT
                                level_touched = resistance
                                zone_type = "resistance"
                                confidence = 0.65
                                break
                        else:
                            if row["close"] < resistance:  # Bouncing down
                                signal_dir = SignalDirection.SHORT
                                level_touched = resistance
                                zone_type = "resistance"
                                confidence = 0.55
                                break

            if signal_dir != SignalDirection.NEUTRAL:
                entry_price = row["close"]
                atr_val = row["atr"]

                if signal_dir == SignalDirection.LONG:
                    stop_loss = level_touched - zone_width - atr_val
                    take_profit = entry_price + 2 * atr_val
                else:
                    stop_loss = level_touched + zone_width + atr_val
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
                        "level": level_touched,
                        "zone_type": zone_type,
                        "zone_width": zone_width,
                        "rejection_candle": self._is_rejection_candle(row, zone_type)
                    }
                ))

        return signals

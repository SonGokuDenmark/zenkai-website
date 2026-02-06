"""
Pairs Trading Strategy.

Statistical arbitrage based on cointegrated pairs.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from .base import SignalGenerator, Signal, SignalDirection


class PairsTradingStrategy(SignalGenerator):
    """
    Pairs Trading / Statistical Arbitrage Strategy.

    Trades the spread between two cointegrated assets.
    When spread deviates significantly from mean, trade the reversion.

    Default pair: BTC/ETH ratio

    Parameters:
        lookback: Period for calculating spread statistics (default: 100)
        entry_zscore: Z-score threshold for entry (default: 2.0)
        exit_zscore: Z-score threshold for exit (default: 0.5)
        stop_zscore: Z-score threshold for stop loss (default: 3.0)
    """

    def __init__(
        self,
        lookback: int = 100,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        stop_zscore: float = 3.0,
        **kwargs
    ):
        super().__init__(
            name="pairs",
            lookback=lookback,
            entry_zscore=entry_zscore,
            exit_zscore=exit_zscore,
            stop_zscore=stop_zscore,
            **kwargs
        )
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.stop_zscore = stop_zscore

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute pairs trading indicators.

        Note: This requires a DataFrame with both assets.
        The DataFrame should have columns like:
        - close_BTCUSDT
        - close_ETHUSDT
        """
        df = df.copy()

        # Placeholder - actual implementation would use both assets
        # For now, this is a stub that would need the paired data

        return df

    def compute_spread(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        hedge_ratio: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Compute the spread between two assets.

        Args:
            df1: DataFrame for first asset (e.g., BTC)
            df2: DataFrame for second asset (e.g., ETH)
            hedge_ratio: Fixed hedge ratio (if None, calculated dynamically)

        Returns:
            DataFrame with spread calculations
        """
        # Merge on timestamp
        merged = pd.merge(
            df1[["timestamp", "close"]].rename(columns={"close": "close_1"}),
            df2[["timestamp", "close"]].rename(columns={"close": "close_2"}),
            on="timestamp",
            how="inner"
        )

        if hedge_ratio is None:
            # Calculate rolling hedge ratio using OLS
            # log(price1) = alpha + beta * log(price2)
            log_price1 = np.log(merged["close_1"])
            log_price2 = np.log(merged["close_2"])

            # Rolling regression for hedge ratio
            def rolling_beta(x, y, window):
                betas = []
                for i in range(len(x)):
                    if i < window:
                        betas.append(np.nan)
                    else:
                        x_window = x.iloc[i-window:i]
                        y_window = y.iloc[i-window:i]
                        cov = np.cov(x_window, y_window)[0, 1]
                        var = np.var(y_window)
                        betas.append(cov / var if var > 0 else np.nan)
                return pd.Series(betas, index=x.index)

            merged["hedge_ratio"] = rolling_beta(log_price1, log_price2, self.lookback)
        else:
            merged["hedge_ratio"] = hedge_ratio

        # Calculate spread
        merged["spread"] = (
            np.log(merged["close_1"]) -
            merged["hedge_ratio"] * np.log(merged["close_2"])
        )

        # Z-score of spread
        merged["spread_mean"] = merged["spread"].rolling(self.lookback).mean()
        merged["spread_std"] = merged["spread"].rolling(self.lookback).std()
        merged["zscore"] = (
            (merged["spread"] - merged["spread_mean"]) /
            merged["spread_std"]
        )

        return merged

    def generate_signals_from_spread(
        self,
        spread_df: pd.DataFrame,
        symbol: str = "BTCUSDT/ETHUSDT"
    ) -> List[Signal]:
        """
        Generate signals from computed spread.

        Args:
            spread_df: DataFrame with zscore column
            symbol: Symbol identifier for the pair

        Returns:
            List of signals
        """
        signals = []

        for idx in range(1, len(spread_df)):
            row = spread_df.iloc[idx]
            prev_row = spread_df.iloc[idx - 1]

            if pd.isna(row["zscore"]):
                continue

            signal_dir = SignalDirection.NEUTRAL
            confidence = 0.5

            zscore = row["zscore"]
            prev_zscore = prev_row["zscore"]

            # Entry signals
            # Short spread when zscore > entry_threshold (spread too high, expect reversion)
            # This means: short asset1, long asset2
            if prev_zscore <= self.entry_zscore and zscore > self.entry_zscore:
                signal_dir = SignalDirection.SHORT  # Short the spread
                confidence = min(0.5 + abs(zscore) / 10, 0.8)

            # Long spread when zscore < -entry_threshold (spread too low, expect reversion)
            # This means: long asset1, short asset2
            elif prev_zscore >= -self.entry_zscore and zscore < -self.entry_zscore:
                signal_dir = SignalDirection.LONG  # Long the spread
                confidence = min(0.5 + abs(zscore) / 10, 0.8)

            if signal_dir != SignalDirection.NEUTRAL:
                signals.append(Signal(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    direction=signal_dir,
                    confidence=confidence,
                    entry_price=row["close_1"],  # Use first asset price
                    stop_loss=None,  # Stop based on zscore, not price
                    take_profit=None,  # Exit based on zscore
                    metadata={
                        "zscore": zscore,
                        "spread": row["spread"],
                        "spread_mean": row["spread_mean"],
                        "hedge_ratio": row["hedge_ratio"],
                        "close_1": row["close_1"],
                        "close_2": row["close_2"]
                    }
                ))

        return signals

    def generate_signals(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> List[Signal]:
        """
        Generate pairs trading signals.

        Note: For actual pairs trading, you need two DataFrames.
        This method is a placeholder that returns empty list.
        Use generate_signals_from_spread() with properly computed spread.
        """
        # This would need modification to accept two DataFrames
        # For now, return empty list as placeholder
        return []

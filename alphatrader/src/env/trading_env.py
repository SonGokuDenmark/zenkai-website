"""
Gymnasium-compatible trading environment for AlphaTrader.

Uses preprocessed features (LSTM predictions, HMM regime, strategy signals)
instead of raw OHLCV. The agent learns when to trust which signal combination.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
from enum import IntEnum


class Action(IntEnum):
    """Discrete action space for trading."""
    HOLD = 0   # No position change
    BUY = 1    # Open long or close short
    SELL = 2   # Open short or close long


@dataclass
class Position:
    """Track current position state."""
    direction: int  # 1=long, 0=flat, -1=short
    entry_price: float
    entry_step: int
    size: float = 1.0


@dataclass
class EpisodeStats:
    """Track statistics for current episode."""
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_balance: float = 0.0
    pnl_history: List[float] = field(default_factory=list)
    trade_timestamps: List[int] = field(default_factory=list)


class TradingEnv(gym.Env):
    """
    Gymnasium-compatible trading environment for AlphaTrader.

    State consists of preprocessed features:
    - LSTM predictions (direction + confidence + probabilities)
    - HMM regime (one-hot encoded)
    - Strategy signals and confidences (72 strategies)
    - Portfolio state (position, P&L, holding time)

    The agent does NOT see raw OHLCV - that's already processed by LSTM/HMM.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001,  # 0.1% per trade
        max_position_holding: int = 100,   # Max bars before forced exit
        reward_scaling: float = 1.0,
        episode_length: Optional[int] = 500,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize trading environment.

        Args:
            df: DataFrame with preprocessed features (must have regime, signal_* columns)
            initial_balance: Starting capital
            transaction_cost: Cost per trade as fraction (0.001 = 0.1%)
            max_position_holding: Max bars to hold position before penalty
            reward_scaling: Multiplier for rewards
            episode_length: Max steps per episode (None = full data)
            render_mode: Rendering mode ("human" or "ansi")
        """
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position_holding = max_position_holding
        self.reward_scaling = reward_scaling
        self.episode_length = episode_length
        self.render_mode = render_mode

        # Identify feature columns
        self._setup_feature_columns()

        # Define spaces
        self.action_space = spaces.Discrete(3)  # HOLD, BUY, SELL
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._get_state_dim(),),
            dtype=np.float32
        )

        # Episode state (initialized in reset)
        self.current_step = 0
        self.start_step = 0
        self.balance = initial_balance
        self.position: Optional[Position] = None
        self.stats = EpisodeStats()

    def _setup_feature_columns(self):
        """Identify which columns to use for state."""
        # Strategy signal columns (72 strategies)
        self.signal_cols = sorted([c for c in self.df.columns if c.startswith("signal_")])
        self.conf_cols = sorted([c for c in self.df.columns if c.startswith("conf_")])

        # Regime columns (check what format we have)
        if "regime" in self.df.columns:
            self.has_regime = True
            # Get unique regimes for one-hot encoding
            self.regimes = ["TRENDING_UP", "TRENDING_DOWN", "RANGING", "HIGH_VOL"]
        else:
            self.has_regime = False
            self.regimes = []

        # LSTM columns (will be added when LSTM predictions are available)
        # Supports both single (lstm_*) and multi-timeframe (lstm_4h_*) columns
        self.lstm_cols = []
        for prefix in ["lstm", "lstm_4h"]:
            for suffix in ["pred", "conf", "prob_down", "prob_flat", "prob_up"]:
                col = f"{prefix}_{suffix}"
                if col in self.df.columns:
                    self.lstm_cols.append(col)

        print(f"TradingEnv initialized:")
        print(f"  Signal columns: {len(self.signal_cols)}")
        print(f"  Confidence columns: {len(self.conf_cols)}")
        print(f"  Has regime: {self.has_regime}")
        print(f"  LSTM columns: {len(self.lstm_cols)} ({self.lstm_cols})")

    def _get_state_dim(self) -> int:
        """Calculate total state dimension."""
        dim = 0

        # LSTM features (5 if available)
        dim += len(self.lstm_cols) if self.lstm_cols else 0

        # Regime one-hot (4 states)
        dim += len(self.regimes) if self.has_regime else 0

        # Strategy signals and confidences
        dim += len(self.signal_cols)
        dim += len(self.conf_cols)

        # Portfolio state: position_direction, holding_time_norm, unrealized_pnl_norm, realized_pnl_norm
        dim += 4

        return dim

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Reset episode state
        self.balance = self.initial_balance
        self.position = None
        self.stats = EpisodeStats(peak_balance=self.initial_balance)

        # Set starting point
        if options and "start_idx" in options:
            self.current_step = options["start_idx"]
        else:
            # Random start, leaving room for episode
            max_start = len(self.df) - (self.episode_length or 1000) - 1
            max_start = max(0, max_start)
            self.current_step = self.np_random.integers(0, max(1, max_start))

        self.start_step = self.current_step

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: 0=HOLD, 1=BUY, 2=SELL

        Returns:
            observation, reward, terminated, truncated, info
        """
        action = Action(action)

        # Get current price
        current_price = self.df.loc[self.current_step, "close"]

        # Calculate unrealized P&L before action (for reward)
        unrealized_before = self._get_unrealized_pnl(current_price)

        # Execute action
        trade_pnl = self._execute_action(action, current_price)

        # Advance time
        self.current_step += 1

        # Get new price and unrealized P&L
        if self.current_step < len(self.df):
            new_price = self.df.loc[self.current_step, "close"]
            unrealized_after = self._get_unrealized_pnl(new_price)
        else:
            unrealized_after = 0.0

        # Calculate reward
        reward = self._calculate_reward(action, trade_pnl, unrealized_before, unrealized_after)

        # Update stats
        self.stats.pnl_history.append(trade_pnl + (unrealized_after - unrealized_before))
        self._update_drawdown()

        # Check termination
        terminated = self._check_terminated()
        truncated = self._check_truncated()

        # Get observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _execute_action(self, action: Action, current_price: float) -> float:
        """
        Execute trading action.

        Returns:
            Realized P&L from any closed positions
        """
        realized_pnl = 0.0

        if action == Action.HOLD:
            pass  # No action

        elif action == Action.BUY:
            if self.position is None:
                # Open long
                self._open_position(1, current_price)
            elif self.position.direction == -1:
                # Close short, open long
                realized_pnl = self._close_position(current_price)
                self._open_position(1, current_price)
            # If already long, do nothing

        elif action == Action.SELL:
            if self.position is None:
                # Open short
                self._open_position(-1, current_price)
            elif self.position.direction == 1:
                # Close long, open short
                realized_pnl = self._close_position(current_price)
                self._open_position(-1, current_price)
            # If already short, do nothing

        return realized_pnl

    def _open_position(self, direction: int, price: float):
        """Open a new position."""
        # Apply transaction cost
        cost = self.balance * self.transaction_cost
        self.balance -= cost

        self.position = Position(
            direction=direction,
            entry_price=price,
            entry_step=self.current_step,
            size=1.0
        )

        self.stats.total_trades += 1
        self.stats.trade_timestamps.append(self.current_step)

    def _close_position(self, price: float) -> float:
        """Close current position and return realized P&L."""
        if self.position is None:
            return 0.0

        # Calculate P&L
        if self.position.direction == 1:  # Long
            pnl_pct = (price - self.position.entry_price) / self.position.entry_price
        else:  # Short
            pnl_pct = (self.position.entry_price - price) / self.position.entry_price

        realized_pnl = self.balance * pnl_pct * self.position.size

        # Apply transaction cost
        cost = self.balance * self.transaction_cost
        self.balance += realized_pnl - cost

        # Update stats
        self.stats.total_pnl += realized_pnl
        if realized_pnl > 0:
            self.stats.winning_trades += 1

        self.position = None
        return realized_pnl

    def _get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L of current position."""
        if self.position is None:
            return 0.0

        if self.position.direction == 1:  # Long
            pnl_pct = (current_price - self.position.entry_price) / self.position.entry_price
        else:  # Short
            pnl_pct = (self.position.entry_price - current_price) / self.position.entry_price

        return self.balance * pnl_pct * self.position.size

    def _calculate_reward(
        self,
        action: Action,
        trade_pnl: float,
        unrealized_before: float,
        unrealized_after: float
    ) -> float:
        """
        Calculate reward for this step.

        Components:
        1. Realized P&L from trades
        2. Change in unrealized P&L
        3. Overtrading penalty
        4. Holding time penalty
        """
        reward = 0.0

        # 1. Realized P&L (scaled to reasonable range)
        reward += trade_pnl / self.initial_balance * 100

        # 2. Unrealized P&L change
        unrealized_change = unrealized_after - unrealized_before
        reward += unrealized_change / self.initial_balance * 100

        # 3. Overtrading penalty (if traded within last 3 bars)
        if action != Action.HOLD and len(self.stats.trade_timestamps) >= 2:
            bars_since_last = self.current_step - self.stats.trade_timestamps[-2]
            if bars_since_last < 3:
                reward -= 0.1 * (3 - bars_since_last)

        # 4. Holding time penalty (for stuck positions)
        if self.position is not None:
            holding_time = self.current_step - self.position.entry_step
            if holding_time > self.max_position_holding:
                excess = (holding_time - self.max_position_holding) / self.max_position_holding
                reward -= 0.01 * excess

        return float(np.clip(reward * self.reward_scaling, -10.0, 10.0))

    def _update_drawdown(self):
        """Update max drawdown tracking."""
        current_value = self.balance + self._get_unrealized_pnl(
            self.df.loc[min(self.current_step, len(self.df)-1), "close"]
        )

        if current_value > self.stats.peak_balance:
            self.stats.peak_balance = current_value

        drawdown = (self.stats.peak_balance - current_value) / self.stats.peak_balance
        self.stats.max_drawdown = max(self.stats.max_drawdown, drawdown)

    def _check_terminated(self) -> bool:
        """Check if episode should terminate (natural end)."""
        # End of data
        if self.current_step >= len(self.df) - 1:
            return True

        # Account blowup (lost 50% of capital)
        current_value = self.balance + self._get_unrealized_pnl(
            self.df.loc[self.current_step, "close"]
        )
        if current_value < self.initial_balance * 0.5:
            return True

        return False

    def _check_truncated(self) -> bool:
        """Check if episode should truncate (time limit)."""
        if self.episode_length is not None:
            steps_taken = self.current_step - self.start_step
            if steps_taken >= self.episode_length:
                return True
        return False

    def _get_observation(self) -> np.ndarray:
        """Construct observation from current state."""
        if self.current_step >= len(self.df):
            # Return zeros if past end
            return np.zeros(self._get_state_dim(), dtype=np.float32)

        row = self.df.iloc[self.current_step]
        features = []

        # 1. LSTM predictions (if available)
        for col in self.lstm_cols:
            features.append(float(row.get(col, 0)))

        # 2. Regime one-hot
        if self.has_regime:
            regime = row.get("regime", "UNKNOWN")
            for r in self.regimes:
                features.append(1.0 if regime == r else 0.0)

        # 3. Strategy signals (normalized to [-1, 1])
        for col in self.signal_cols:
            val = row.get(col, 0)
            features.append(float(val) if pd.notna(val) else 0.0)

        # 4. Strategy confidences (normalized to [0, 1])
        for col in self.conf_cols:
            val = row.get(col, 0)
            features.append(float(val) if pd.notna(val) else 0.0)

        # 5. Portfolio state
        features.extend(self._get_portfolio_features())

        return np.array(features, dtype=np.float32)

    def _get_portfolio_features(self) -> List[float]:
        """Get normalized portfolio state features."""
        if self.position is None:
            return [
                0.0,  # No position
                0.0,  # No holding time
                0.0,  # No unrealized P&L
                self.stats.total_pnl / self.initial_balance  # Realized P&L
            ]

        current_price = self.df.loc[min(self.current_step, len(self.df)-1), "close"]
        holding_time = (self.current_step - self.position.entry_step) / self.max_position_holding
        unrealized = self._get_unrealized_pnl(current_price)

        return [
            float(self.position.direction),
            min(holding_time, 2.0),  # Cap at 2x max
            unrealized / self.initial_balance,
            self.stats.total_pnl / self.initial_balance
        ]

    def _get_info(self) -> Dict[str, Any]:
        """Get info dict for debugging/logging."""
        return {
            "step": self.current_step,
            "balance": self.balance,
            "position": self.position.direction if self.position else 0,
            "total_trades": self.stats.total_trades,
            "winning_trades": self.stats.winning_trades,
            "total_pnl": self.stats.total_pnl,
            "max_drawdown": self.stats.max_drawdown,
            "win_rate": self.stats.winning_trades / max(1, self.stats.total_trades),
        }

    def render(self):
        """Render current state."""
        if self.render_mode == "human":
            info = self._get_info()
            print(f"Step {info['step']} | Balance: ${info['balance']:.2f} | "
                  f"Position: {info['position']} | Trades: {info['total_trades']} | "
                  f"Win Rate: {info['win_rate']:.1%} | Max DD: {info['max_drawdown']:.1%}")

    def close(self):
        """Clean up resources."""
        pass


def create_train_test_envs(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    **env_kwargs
) -> Tuple[TradingEnv, TradingEnv, TradingEnv]:
    """
    Create train/val/test environments with temporal split.

    Args:
        df: Full dataframe with features
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        **env_kwargs: Passed to TradingEnv

    Returns:
        (train_env, val_env, test_env)
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    return (
        TradingEnv(train_df, **env_kwargs),
        TradingEnv(val_df, **env_kwargs),
        TradingEnv(test_df, **env_kwargs),
    )

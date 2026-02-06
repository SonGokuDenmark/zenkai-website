#!/usr/bin/env python3
"""
AlphaTrader - Neural Network Imitation Learning System

Main training entry point.
"""

import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.database import Database
from src.data.binance_loader import BinanceLoader
from src.data.feature_engineer import FeatureEngineer
from src.data.labeler import Labeler
from src.strategies import (
    MACDStrategy, MACrossoverStrategy, StochasticMRStrategy,
    TurtleStrategy, SupportResistanceStrategy, CandlestickStrategy,
    RSIDivergenceStrategy
)
from src.utils.discord import DiscordNotifier
from src.utils.gpu_monitor import GPUMonitor
from src.utils.logging import TrainingLogger, setup_logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_strategies(config: Dict[str, Any]) -> list:
    """Initialize enabled strategies from config."""
    enabled = config.get("strategies", {}).get("enabled", [])
    strategies = []

    strategy_classes = {
        "rsi_divergence": RSIDivergenceStrategy,
        "macd": MACDStrategy,
        "turtle": TurtleStrategy,
        "sr_bounce": SupportResistanceStrategy,
        "candlestick": CandlestickStrategy,
        "ma_crossover": MACrossoverStrategy,
        "stochastic": StochasticMRStrategy,
    }

    for name in enabled:
        if name in strategy_classes:
            # Get strategy-specific params
            params = config.get("strategies", {}).get(name, {})
            strategies.append(strategy_classes[name](**params))

    return strategies


class AlphaTrader:
    """
    Main AlphaTrader class.

    Orchestrates data loading, feature engineering, signal generation,
    and neural network training.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize AlphaTrader.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.start_time = datetime.now()

        # Initialize components
        self.db = Database(config["data"]["database_path"])
        self.loader = BinanceLoader(cache_dir="data/raw")
        self.feature_engineer = FeatureEngineer(**config.get("features", {}))
        self.labeler = Labeler(**config.get("labels", {}))
        self.strategies = get_strategies(config)

        # Monitoring
        self.logger = TrainingLogger(
            run_name=config.get("project", {}).get("name", "alphatrader")
        )

        webhook_url = config.get("monitoring", {}).get("discord_webhook", "")
        self.discord = DiscordNotifier(
            webhook_url=webhook_url,
            enabled=bool(webhook_url)
        )

        gpu_config = config.get("gpu", {})
        self.gpu_monitor = GPUMonitor(
            limit_percent=gpu_config.get("limit_percent", 75),
            max_temp=gpu_config.get("max_temp", 80),
            throttle_temp=gpu_config.get("throttle_temp", 75),
            on_warning=self._on_gpu_warning
        )

        # State
        self._shutdown_requested = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Set up graceful shutdown handlers."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print("\nShutdown requested, saving state...")
        self._shutdown_requested = True

    def _on_gpu_warning(self, stats):
        """Handle GPU warning callback."""
        self.discord.notify_gpu_warning(
            temp=stats.temperature,
            usage=stats.usage,
            action="throttled"
        )

    def prepare_data(self):
        """
        Load and prepare training data.

        1. Fetch OHLCV data from Binance
        2. Compute features
        3. Generate labels
        4. Generate strategy signals
        5. Store in database
        """
        self.logger.info("Preparing training data...")

        symbols = self.config["data"]["symbols"]
        days = self.config["data"]["days"]

        # Support multiple timeframes
        timeframes = self.config["data"].get("timeframes", [self.config["data"]["timeframe"]])
        if isinstance(timeframes, str):
            timeframes = [timeframes]

        total_rows = 0
        total_combinations = len(symbols) * len(timeframes)
        current_combo = 0

        for timeframe in timeframes:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Processing timeframe: {timeframe}")
            self.logger.info(f"{'='*60}")

            # Fetch data for all symbols at this timeframe
            data = self.loader.fetch_multiple(
                symbols=symbols,
                timeframe=timeframe,
                days=days,
                market="spot",
                show_progress=True
            )

            for symbol, df in data.items():
                current_combo += 1
                self.logger.info(f"\n[{current_combo}/{total_combinations}] Processing {symbol} @ {timeframe}...")

                # Add symbol and timeframe columns
                df["symbol"] = symbol
                df["timeframe"] = timeframe

                # Compute features
                self.logger.info(f"  Computing features...")
                df = self.feature_engineer.compute_all_features(df)

                # Compute labels
                self.logger.info(f"  Computing labels...")
                df = self.labeler.compute_all_labels(df)

                # Generate signals for each strategy
                for strategy in self.strategies:
                    self.logger.info(f"  Generating {strategy.name} signals...")
                    df = strategy.add_signals_to_df(df, symbol)

                # Calculate expert consensus
                signal_cols = [
                    col for col in df.columns
                    if col.startswith("signal_") and not col.endswith("_prev")
                ]
                if signal_cols:
                    df["expert_consensus"] = df[signal_cols].mean(axis=1)
                    df["expert_agreement"] = (df[signal_cols] != 0).sum(axis=1)

                # Filter to only columns that exist in the database schema
                # Drop columns not in schema to avoid errors
                columns_to_drop = [
                    "quote_volume", "num_trades",  # From Binance loader
                    "macd_prev", "macd_signal_prev",  # Strategy internals
                    "ma_fast", "ma_slow", "ma_fast_prev", "ma_slow_prev", "ma_200",
                    "stoch_k_prev", "stoch_d_prev", "ema_50", "ema_200_dup",
                    "donchian_upper_prev", "donchian_lower_prev", "exit_upper", "exit_lower",
                    "body", "body_abs", "range", "is_bullish", "is_bearish", "is_doji",
                    "trend", "avg_body", "rsi_at_pivot_high", "rsi_at_pivot_low",
                    "upper_wick", "lower_wick",
                    "future_log_return_10bar", "future_log_return_25bar", "future_log_return_50bar",
                ]
                for col in columns_to_drop:
                    if col in df.columns:
                        df = df.drop(columns=[col])

                # Store in database
                rows = self.db.insert_market_states_df(df)
                total_rows += rows
                self.logger.info(f"  Stored {rows} rows for {symbol} @ {timeframe}")

                if self._shutdown_requested:
                    break

            if self._shutdown_requested:
                break

        self.logger.info(f"\nData preparation complete. Total rows: {total_rows}")

        # Log stats
        stats = self.db.get_stats()
        self.logger.info(f"Database stats: {stats}")

        return total_rows

    def run(self):
        """
        Main training loop.

        This is the entry point for training.
        """
        self.logger.info("=" * 60)
        self.logger.info("AlphaTrader Training Started")
        self.logger.info("=" * 60)
        self.logger.info(f"Config: {self.config['project']['name']} v{self.config['project']['version']}")

        # Show run configuration
        symbols = self.config["data"]["symbols"]
        timeframes = self.config["data"].get("timeframes", [self.config["data"]["timeframe"]])
        days = self.config["data"]["days"]
        self.logger.info(f"Symbols: {', '.join(symbols)}")
        self.logger.info(f"Timeframes: {', '.join(timeframes) if isinstance(timeframes, list) else timeframes}")
        self.logger.info(f"Days: {days}")

        # Estimate total candles
        candles_per_day = {
            "1m": 1440, "3m": 480, "5m": 288, "15m": 96, "30m": 48,
            "1h": 24, "2h": 12, "4h": 6, "6h": 4, "8h": 3, "12h": 2, "1d": 1
        }
        total_estimated = 0
        for tf in (timeframes if isinstance(timeframes, list) else [timeframes]):
            total_estimated += len(symbols) * days * candles_per_day.get(tf, 96)
        self.logger.info(f"Estimated rows: ~{total_estimated:,}")

        # Start GPU monitoring
        self.gpu_monitor.start()

        # Print GPU stats
        self.gpu_monitor.print_stats()

        try:
            # Notify start
            if self.config.get("monitoring", {}).get("notify_on_start", True):
                self.discord.notify_training_start(
                    run_name=self.config["project"]["name"],
                    config=self.config,
                    total_epochs=self.config["training"]["epochs"],
                    total_samples=0  # Will update after data prep
                )

            # Prepare data
            total_rows = self.prepare_data()

            if self._shutdown_requested:
                self.logger.info("Shutdown during data preparation")
                return

            # TODO: Implement actual neural network training
            # This is the placeholder for Phase 0 Week 5
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("DATA PREPARATION COMPLETE")
            self.logger.info("=" * 60)
            self.logger.info("")
            self.logger.info("Next steps (Week 5):")
            self.logger.info("  1. Implement LSTM classifier")
            self.logger.info("  2. Implement training loop")
            self.logger.info("  3. Implement validation with WFO")
            self.logger.info("")

            # Show data statistics
            stats = self.db.get_stats()
            self.logger.info("Database Statistics:")
            self.logger.info(f"  Market states: {stats['market_states']:,}")
            self.logger.info(f"  Symbols: {', '.join(stats['symbols'])}")
            self.logger.info(f"  Timeframes: {', '.join(stats['timeframes'])}")
            self.logger.info(f"  Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")

            # Show strategy signal counts
            for strategy in self.strategies:
                perf = self.db.get_strategy_performance(strategy.name)
                self.logger.info(f"  {strategy.name}: {perf.get('total_trades', 0)} signals")

            # Notify completion
            if self.config.get("monitoring", {}).get("notify_on_complete", True):
                self.discord.notify_training_complete(
                    run_name=self.config["project"]["name"],
                    final_loss=0.0,
                    final_accuracy=0.0,
                    training_time_seconds=int((datetime.now() - self.start_time).total_seconds())
                )

        except Exception as e:
            self.logger.error(f"Training error: {e}")
            if self.config.get("monitoring", {}).get("notify_on_error", True):
                self.discord.notify_error(e, context="Training loop")
            raise

        finally:
            self.gpu_monitor.stop()
            self.logger.info("Training session ended")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AlphaTrader - Neural Network Imitation Learning"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare data, don't train"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Override symbols (comma-separated)"
    )
    parser.add_argument(
        "--days",
        type=int,
        help="Override number of days"
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        help="Override timeframe (comma-separated for multiple, e.g., '15m,1h,4h')"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override from command line
    if args.symbols:
        config["data"]["symbols"] = [s.strip() for s in args.symbols.split(",")]
    if args.days:
        config["data"]["days"] = args.days
    if args.timeframe:
        # Support multiple timeframes
        timeframes = [t.strip() for t in args.timeframe.split(",")]
        config["data"]["timeframes"] = timeframes
        config["data"]["timeframe"] = timeframes[0]  # Primary for backwards compat

    # Run
    trainer = AlphaTrader(config)
    trainer.run()


if __name__ == "__main__":
    main()

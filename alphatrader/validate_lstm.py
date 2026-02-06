#!/usr/bin/env python3
"""
LSTM Model Validation Script

Runs Walk-Forward Optimization and Monte Carlo simulation to validate
that the LSTM model generalizes across time periods.

Usage:
    python validate_lstm.py                        # Default: 1h timeframe
    python validate_lstm.py --timeframe 15m        # Different timeframe
    python validate_lstm.py --train-months 4       # Shorter training window
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Optional
import json

import numpy as np
import pandas as pd
import psycopg2
import requests
from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models import LSTMClassifier
from validation import WalkForwardOptimizer, MonteCarloSimulator

# Discord webhook
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# Database config
DB_CONFIG = {
    "host": os.getenv("ZENKAI_DB_HOST", "192.168.0.160"),
    "database": os.getenv("ZENKAI_DB_NAME", "zenkai_data"),
    "user": os.getenv("ZENKAI_DB_USER", "zenkai"),
    "password": os.getenv("ZENKAI_DB_PASSWORD"),
}

# Regime mapping
REGIME_MAP = {
    "TRENDING_UP": 0,
    "TRENDING_DOWN": 1,
    "RANGING": 2,
    "HIGH_VOL": 3,
}


def get_db_connection():
    """Get PostgreSQL connection."""
    return psycopg2.connect(**DB_CONFIG)


def load_data_with_timestamps(
    timeframe: str = "1h",
    limit: int = 500000,
) -> pd.DataFrame:
    """Load data with timestamps for WFO."""
    conn = get_db_connection()

    query = """
        SELECT
            open_time, symbol,
            open, high, low, close, volume,
            regime,
            signal_macd, signal_rsi_div, signal_turtle,
            signal_ma_cross, signal_stoch, signal_sr, signal_candle,
            conf_macd, conf_rsi_div, conf_turtle,
            conf_ma_cross, conf_stoch, conf_sr, conf_candle
        FROM ohlcv
        WHERE timeframe = %s
          AND regime IS NOT NULL
        ORDER BY open_time
        LIMIT %s
    """

    df = pd.read_sql_query(query, conn, params=[timeframe, limit])
    conn.close()

    print(f"Loaded {len(df):,} rows from database")
    return df


def compute_labels(df: pd.DataFrame, forward_bars: int = 10, flat_threshold: float = 0.005) -> pd.DataFrame:
    """Compute forward-looking labels."""
    df = df.copy()

    df["future_return"] = df.groupby("symbol")["close"].transform(
        lambda x: x.shift(-forward_bars) / x - 1
    )

    df["label"] = 1  # Default FLAT
    df.loc[df["future_return"] > flat_threshold, "label"] = 2  # UP
    df.loc[df["future_return"] < -flat_threshold, "label"] = 0  # DOWN

    df = df.dropna(subset=["future_return"])
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for training."""
    df = df.copy()

    # Normalize OHLCV per symbol
    for col in ["open", "high", "low", "close"]:
        df[col] = df.groupby("symbol")[col].transform(
            lambda x: (x - x.rolling(50, min_periods=1).mean()) / x.rolling(50, min_periods=1).std().clip(lower=1e-8)
        )

    df["volume"] = df.groupby("symbol")["volume"].transform(
        lambda x: (x - x.rolling(50, min_periods=1).mean()) / x.rolling(50, min_periods=1).std().clip(lower=1e-8)
    )

    # One-hot encode regime
    for regime, idx in REGIME_MAP.items():
        df[f"regime_{regime.lower()}"] = (df["regime"] == regime).astype(float)

    # Fill NaN
    signal_cols = [c for c in df.columns if c.startswith("signal_") or c.startswith("conf_")]
    df[signal_cols] = df[signal_cols].fillna(0)
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)

    return df


def create_sequences_with_timestamps(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_length: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sequences and return timestamps for each sequence."""
    X_list = []
    y_list = []
    ts_list = []

    for symbol in df["symbol"].unique():
        symbol_df = df[df["symbol"] == symbol].sort_values("open_time")

        if len(symbol_df) < seq_length + 1:
            continue

        features = symbol_df[feature_cols].values
        labels = symbol_df["label"].values
        timestamps = symbol_df["open_time"].values

        for i in range(len(symbol_df) - seq_length):
            X_list.append(features[i:i + seq_length])
            y_list.append(labels[i + seq_length - 1])
            ts_list.append(timestamps[i + seq_length - 1])  # Timestamp at end of sequence

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    timestamps = np.array(ts_list)

    print(f"Created {len(X):,} sequences of length {seq_length}")
    return X, y, timestamps


def send_discord_notification(results: dict):
    """Send validation results to Discord."""
    wfo = results["wfo"]
    mc = results["mc"]

    embed = {
        "title": "📊 LSTM Validation Complete",
        "color": 3447003,  # Blue
        "fields": [
            {"name": "Timeframe", "value": results["timeframe"], "inline": True},
            {"name": "WFO Folds", "value": str(len(wfo.folds)), "inline": True},
            {"name": "Test Samples", "value": f"{wfo.total_test_samples:,}", "inline": True},
            {
                "name": "Walk-Forward Results",
                "value": f"**Mean Accuracy: {wfo.mean_accuracy:.1%}**\n"
                        f"Std: {wfo.std_accuracy:.1%}\n"
                        f"WFE: {wfo.wfe:.3f}",
                "inline": False
            },
            {
                "name": "Per-Class Accuracy",
                "value": "\n".join([f"{k}: {v:.1%}" for k, v in wfo.class_accuracies.items()]),
                "inline": True
            },
            {
                "name": "Monte Carlo Risk",
                "value": f"P(Loss): {mc.probability_of_loss:.1%}\n"
                        f"VaR 95%: {mc.var_95:.1%}\n"
                        f"Worst DD: {mc.worst_drawdown:.1%}",
                "inline": True
            },
        ],
        "timestamp": datetime.now().isoformat() + "Z",
    }

    payload = {"embeds": [embed]}

    try:
        print("\nSending Discord notification...")
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        if response.status_code == 204:
            print("Discord notification sent successfully!")
        else:
            print(f"Discord notification failed: {response.status_code}")
    except Exception as e:
        print(f"Discord notification error: {e}")


def validate(
    timeframe: str = "1h",
    train_months: int = 6,
    test_months: int = 2,
    hidden_size: int = 128,
    num_layers: int = 2,
    seq_length: int = 50,
    limit: int = 500000,
    epochs: int = 50,
    batch_size: int = 64,
):
    """Run full validation pipeline."""
    print("=" * 60)
    print("AlphaTrader LSTM Validation")
    print("=" * 60)
    print(f"Started: {datetime.now()}")
    print(f"Timeframe: {timeframe}")
    print(f"WFO: {train_months}d train / {test_months}d test")
    print()

    # Load data
    print("Loading data...")
    df = load_data_with_timestamps(timeframe=timeframe, limit=limit)

    if len(df) < 10000:
        print("ERROR: Not enough data for validation")
        return

    # Compute labels
    print("Computing labels...")
    df = compute_labels(df)

    # Prepare features
    print("Preparing features...")
    df = prepare_features(df)

    # Feature columns
    feature_cols = [
        "open", "high", "low", "close", "volume",
        "signal_macd", "signal_rsi_div", "signal_turtle",
        "signal_ma_cross", "signal_stoch", "signal_sr", "signal_candle",
        "conf_macd", "conf_rsi_div", "conf_turtle",
        "conf_ma_cross", "conf_stoch", "conf_sr", "conf_candle",
        "regime_trending_up", "regime_trending_down", "regime_ranging", "regime_high_vol",
    ]

    # Create sequences with timestamps
    print("Creating sequences...")
    X, y, timestamps = create_sequences_with_timestamps(df, feature_cols, seq_length=seq_length)

    if len(X) < 5000:
        print("ERROR: Not enough sequences for validation")
        return

    # Define model factory
    input_size = X.shape[2]

    def model_factory():
        return LSTMClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.3,
            learning_rate=0.001,
        )

    # Run Walk-Forward Optimization
    print("\n" + "=" * 60)
    print("Running Walk-Forward Optimization...")
    print("=" * 60)

    wfo = WalkForwardOptimizer(
        train_periods=train_months,  # Reusing param name but now in days
        test_periods=test_months,
        step_periods=test_months,  # Non-overlapping test periods
        period_type="days",  # Use days for shorter data spans
        min_train_samples=1000,
    )

    wfo_result = wfo.validate(
        X=X,
        y=y,
        timestamps=timestamps,
        model_factory=model_factory,
        fit_kwargs={"epochs": epochs, "batch_size": batch_size, "early_stopping_patience": 10},
        verbose=True,
    )

    # Collect all predictions for Monte Carlo
    all_predictions = np.concatenate([f.predictions for f in wfo_result.folds])
    all_actuals = np.concatenate([f.actuals for f in wfo_result.folds])
    all_confidences = np.concatenate([f.confidences for f in wfo_result.folds])

    # Run Monte Carlo simulation
    print("\n" + "=" * 60)
    print("Running Monte Carlo Simulation...")
    print("=" * 60)

    mc = MonteCarloSimulator(n_simulations=10000)
    mc_result = mc.simulate_from_predictions(
        predictions=all_predictions,
        actuals=all_actuals,
        confidences=all_confidences,
        return_per_correct=0.01,  # 1% per correct
        return_per_wrong=-0.012,  # -1.2% per wrong (slightly asymmetric)
        verbose=True,
    )

    # Send Discord notification
    send_discord_notification({
        "timeframe": timeframe,
        "wfo": wfo_result,
        "mc": mc_result,
    })

    # Save results
    results_path = f"checkpoints/validation_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path("checkpoints").mkdir(exist_ok=True)

    results_dict = {
        "timeframe": timeframe,
        "train_months": train_months,
        "test_months": test_months,
        "wfo": {
            "n_folds": len(wfo_result.folds),
            "mean_accuracy": wfo_result.mean_accuracy,
            "std_accuracy": wfo_result.std_accuracy,
            "wfe": wfo_result.wfe,
            "class_accuracies": wfo_result.class_accuracies,
            "total_test_samples": wfo_result.total_test_samples,
            "fold_accuracies": [f.accuracy for f in wfo_result.folds],
        },
        "monte_carlo": {
            "mean_return": mc_result.mean_return,
            "median_return": mc_result.median_return,
            "return_5th_pct": mc_result.return_5th_percentile,
            "return_95th_pct": mc_result.return_95th_percentile,
            "probability_of_loss": mc_result.probability_of_loss,
            "var_95": mc_result.var_95,
            "cvar_95": mc_result.cvar_95,
            "worst_drawdown": mc_result.worst_drawdown,
            "mean_sharpe": mc_result.mean_sharpe,
        },
    }

    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print(f"\nValidation complete!")
    print(f"Finished: {datetime.now()}")


def main():
    parser = argparse.ArgumentParser(description="Validate LSTM model with WFO and Monte Carlo")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe (default: 1h)")
    parser.add_argument("--train-days", type=int, default=21, help="Training days per fold (default: 21)")
    parser.add_argument("--test-days", type=int, default=7, help="Test days per fold (default: 7)")
    parser.add_argument("--hidden-size", type=int, default=128, help="LSTM hidden size (default: 128)")
    parser.add_argument("--num-layers", type=int, default=2, help="LSTM layers (default: 2)")
    parser.add_argument("--seq-length", type=int, default=50, help="Sequence length (default: 50)")
    parser.add_argument("--limit", type=int, default=500000, help="Max rows to load (default: 500000)")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per fold (default: 50)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")

    args = parser.parse_args()

    validate(
        timeframe=args.timeframe,
        train_months=args.train_days,
        test_months=args.test_days,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        seq_length=args.seq_length,
        limit=args.limit,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

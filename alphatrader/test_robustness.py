#!/usr/bin/env python3
"""
Robustness Test - Parameter Sensitivity Analysis

Tests if the best config is robust to parameter variations.
A good strategy should not collapse when parameters change by 10-20%.

Red flag if results collapse with small parameter changes!
"""

import os
import sys
import numpy as np
import pandas as pd
import psycopg2
import json
from datetime import datetime
from typing import Dict, Optional, List
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.env import TradingEnv, create_train_test_envs
from src.models.lstm_classifier import LSTMClassifier
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

# Base config (best from previous tests)
BASE_CONFIG = {
    "timeframe": "4h",
    "timesteps": 200000,
    "learning_rate": 1e-4,
    "episode_length": 500,
    "train_ratio": 0.7,
    "transaction_cost": 0.001,
}

# Parameter variations to test (multiply base value)
VARIATIONS = {
    "timesteps": [0.5, 0.75, 1.0, 1.25, 1.5],      # 100k to 300k
    "learning_rate": [0.5, 0.75, 1.0, 1.5, 2.0],   # 5e-5 to 2e-4
    "episode_length": [0.6, 0.8, 1.0, 1.2, 1.5],   # 300 to 750
    "transaction_cost": [0.5, 1.0, 1.5, 2.0, 3.0], # 0.05% to 0.3%
}

# Test on main symbols
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']


def get_data(symbol: str, timeframe: str, limit: int = 10000) -> Optional[pd.DataFrame]:
    """Load data."""
    conn = psycopg2.connect(
        host=os.getenv("ZENKAI_DB_HOST", "192.168.0.160"),
        database=os.getenv("ZENKAI_DB_NAME", "zenkai_data"),
        user=os.getenv("ZENKAI_DB_USER", "zenkai"),
        password=os.getenv("ZENKAI_DB_PASSWORD"),
    )

    cursor = conn.cursor()
    cursor.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'ohlcv'
        AND (column_name LIKE 'signal_%' OR column_name LIKE 'conf_%')
    """)
    signal_cols = [row[0] for row in cursor.fetchall()]
    cursor.close()

    base_cols = ["open_time", "symbol", "open", "high", "low", "close", "volume", "regime"]
    all_cols = base_cols + signal_cols

    query = f"""
        SELECT {', '.join(all_cols)}
        FROM ohlcv
        WHERE timeframe = %s AND regime IS NOT NULL AND symbol = %s
        ORDER BY open_time DESC
        LIMIT %s
    """
    df = pd.read_sql_query(query, conn, params=[timeframe, symbol, limit])
    conn.close()

    if len(df) < 300:
        return None

    df = df.sort_values('open_time').reset_index(drop=True)
    for col in signal_cols:
        df[col] = df[col].fillna(0)

    # Filter HIGH_VOL
    df = df[df['regime'] != 'HIGH_VOL'].reset_index(drop=True)

    return df


def add_lstm_predictions(df: pd.DataFrame, lstm_model: LSTMClassifier, seq_length: int = 50) -> pd.DataFrame:
    """Add LSTM predictions."""
    feature_cols = ["open", "high", "low", "close", "volume"]
    signal_cols = sorted([c for c in df.columns if c.startswith("signal_") or c.startswith("conf_")])
    feature_cols.extend(signal_cols)

    regimes = ["TRENDING_UP", "TRENDING_DOWN", "RANGING", "HIGH_VOL"]
    for regime in regimes:
        col_name = f"regime_{regime.lower()}"
        if col_name not in df.columns:
            df[col_name] = (df["regime"] == regime).astype(float)
        feature_cols.append(col_name)

    df_norm = df.copy()
    for col in ["open", "high", "low", "close"]:
        df_norm[col] = (df_norm[col] - df_norm[col].rolling(50, min_periods=1).mean()) / \
                       df_norm[col].rolling(50, min_periods=1).std().clip(lower=1e-8)
    df_norm["volume"] = (df_norm["volume"] - df_norm["volume"].rolling(50, min_periods=1).mean()) / \
                        df_norm["volume"].rolling(50, min_periods=1).std().clip(lower=1e-8)
    df_norm = df_norm.fillna(0).replace([np.inf, -np.inf], 0)

    n_rows = len(df)
    lstm_pred = np.full(n_rows, np.nan)
    features = df_norm[feature_cols].values

    batch_size = 1024
    sequences = []
    indices = []

    for i in range(seq_length, n_rows):
        sequences.append(features[i - seq_length:i])
        indices.append(i)

        if len(sequences) >= batch_size or i == n_rows - 1:
            X = np.array(sequences, dtype=np.float32)
            proba = lstm_model.predict_proba(X)
            preds = np.argmax(proba, axis=1) - 1

            for j, idx in enumerate(indices):
                lstm_pred[idx] = preds[j]

            sequences = []
            indices = []

    df["lstm_pred"] = lstm_pred
    return df


def train_and_test(
    df: pd.DataFrame,
    timesteps: int,
    learning_rate: float,
    episode_length: int,
    train_ratio: float,
    transaction_cost: float,
) -> Dict:
    """Train DQN with specific parameters."""
    train_env, val_env, test_env = create_train_test_envs(
        df,
        train_ratio=train_ratio,
        val_ratio=0.15,
        episode_length=episode_length,
        initial_balance=10000.0,
        transaction_cost=transaction_cost,
    )

    train_env = Monitor(train_env)
    model = DQN("MlpPolicy", train_env, learning_rate=learning_rate, verbose=0)
    model.learn(total_timesteps=timesteps, progress_bar=False)

    # Evaluate
    test_df = df.iloc[int(len(df) * 0.85):].copy()
    test_env_eval = TradingEnv(
        test_df,
        episode_length=None,
        initial_balance=10000.0,
        transaction_cost=transaction_cost,
    )

    obs, info = test_env_eval.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env_eval.step(action)
        done = terminated or truncated

    return {
        "pnl": info["total_pnl"],
        "win_rate": info["win_rate"],
        "trades": info["total_trades"],
    }


def test_parameter(
    param_name: str,
    variations: List[float],
    df: pd.DataFrame,
    base_config: Dict,
) -> Dict:
    """Test sensitivity to a single parameter."""
    results = {}

    for mult in variations:
        # Create config with variation
        config = base_config.copy()
        base_value = config[param_name]

        if param_name == "timesteps":
            config[param_name] = int(base_value * mult)
        elif param_name == "episode_length":
            config[param_name] = int(base_value * mult)
        else:
            config[param_name] = base_value * mult

        actual_value = config[param_name]

        print(f"    {param_name}={actual_value}...", end=" ", flush=True)

        result = train_and_test(
            df,
            timesteps=config["timesteps"],
            learning_rate=config["learning_rate"],
            episode_length=config["episode_length"],
            train_ratio=config["train_ratio"],
            transaction_cost=config["transaction_cost"],
        )

        results[str(mult)] = {
            "multiplier": mult,
            "actual_value": actual_value,
            "pnl": result["pnl"],
            "win_rate": result["win_rate"],
            "trades": result["trades"],
        }

        status = "[WIN]" if result["pnl"] > 0 else "[LOSS]"
        print(f"PnL: ${result['pnl']:>8.2f} {status}")

    return results


def main():
    print("=" * 70)
    print("ROBUSTNESS TEST - Parameter Sensitivity Analysis")
    print("=" * 70)
    print(f"Base config: {BASE_CONFIG}")
    print(f"Symbols: {SYMBOLS}")
    print(f"Parameters to test: {list(VARIATIONS.keys())}")
    print()

    # Load LSTM
    lstm_path = f"checkpoints/lstm_{BASE_CONFIG['timeframe']}_mtf"
    try:
        lstm = LSTMClassifier.load(lstm_path)
        print(f"LSTM loaded: {lstm.input_size} features\n")
    except:
        print(f"Warning: Could not load LSTM from {lstm_path}")
        lstm = None

    all_results = {}

    for symbol in SYMBOLS:
        print(f"\n{'=' * 70}")
        print(f"Testing: {symbol}")
        print("=" * 70)

        # Load data
        df = get_data(symbol, BASE_CONFIG["timeframe"])
        if df is None:
            print(f"  Skipped: insufficient data")
            continue

        print(f"  Loaded {len(df)} rows")

        # Add LSTM
        if lstm:
            df = add_lstm_predictions(df, lstm)
            df = df.dropna(subset=["lstm_pred"]).reset_index(drop=True)
            print(f"  After LSTM: {len(df)} rows")

        if len(df) < 300:
            print(f"  Skipped: insufficient data")
            continue

        all_results[symbol] = {}

        for param_name, variations in VARIATIONS.items():
            print(f"\n  Testing {param_name}:")
            results = test_parameter(param_name, variations, df, BASE_CONFIG)
            all_results[symbol][param_name] = results

    # Analyze robustness
    print("\n" + "=" * 70)
    print("ROBUSTNESS ANALYSIS")
    print("=" * 70)

    red_flags = []

    for param_name in VARIATIONS.keys():
        print(f"\n{param_name}:")

        for symbol in SYMBOLS:
            if symbol not in all_results:
                continue

            param_results = all_results[symbol].get(param_name, {})
            if not param_results:
                continue

            pnls = [r["pnl"] for r in param_results.values()]
            base_pnl = param_results.get("1.0", {}).get("pnl", 0)

            # Check if results collapse
            profitable = sum(1 for p in pnls if p > 0)
            pnl_range = max(pnls) - min(pnls) if pnls else 0
            pnl_std = np.std(pnls) if pnls else 0

            stability = "STABLE" if profitable >= len(pnls) * 0.6 else "UNSTABLE"
            if stability == "UNSTABLE":
                red_flags.append(f"{symbol}/{param_name}: Only {profitable}/{len(pnls)} profitable")

            print(f"  {symbol:12} Profitable: {profitable}/{len(pnls)}  "
                  f"Range: ${pnl_range:,.0f}  Std: ${pnl_std:,.0f}  [{stability}]")

    # Red flags summary
    print("\n" + "=" * 70)
    print("RED FLAGS")
    print("=" * 70)

    if red_flags:
        for flag in red_flags:
            print(f"  [!] {flag}")
        print("\n  WARNING: Strategy may be overfit to specific parameters!")
    else:
        print("  No major red flags detected.")
        print("  Strategy appears robust to parameter variations.")

    # Save
    output_file = "robustness_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "base_config": BASE_CONFIG,
            "variations": VARIATIONS,
            "symbols": SYMBOLS,
            "results": all_results,
            "red_flags": red_flags,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()

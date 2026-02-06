#!/usr/bin/env python3
"""
Rigorous test: Does the LSTM add value?

Compare RL performance WITH vs WITHOUT LSTM predictions.
- More timesteps (100k)
- More runs (3)
- Statistical comparison
"""

import os
import sys
import numpy as np
import pandas as pd
import psycopg2
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.env import TradingEnv, create_train_test_envs
from src.models.lstm_classifier import LSTMClassifier
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor


# LSTM model path (173 input features - uses signal columns)
LSTM_MODEL_PATH = "checkpoints/lstm_1h_20260204_160901"


def get_data(symbol='BTCUSDT', timeframe='1h', limit=2000):
    """Load data from database."""
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

    df = df.sort_values('open_time').reset_index(drop=True)

    for col in signal_cols:
        df[col] = df[col].fillna(0)

    print(f"  Loaded {len(df)} rows, {len(signal_cols)} signal cols")
    return df


def add_lstm_predictions(
    df: pd.DataFrame,
    lstm_model: LSTMClassifier,
    seq_length: int = 50,
    prefix: str = "lstm",
) -> pd.DataFrame:
    """Add LSTM predictions as columns (same as validate_rl.py)."""
    feature_cols = ["open", "high", "low", "close", "volume"]
    signal_cols = sorted([c for c in df.columns if c.startswith("signal_") or c.startswith("conf_")])
    feature_cols.extend(signal_cols)

    regimes = ["TRENDING_UP", "TRENDING_DOWN", "RANGING", "HIGH_VOL"]
    for regime in regimes:
        col_name = f"regime_{regime.lower()}"
        if col_name not in df.columns:
            df[col_name] = (df["regime"] == regime).astype(float)
        feature_cols.append(col_name)

    # Normalize
    df_norm = df.copy()
    for col in ["open", "high", "low", "close"]:
        df_norm[col] = (df_norm[col] - df_norm[col].rolling(50, min_periods=1).mean()) / \
                       df_norm[col].rolling(50, min_periods=1).std().clip(lower=1e-8)
    df_norm["volume"] = (df_norm["volume"] - df_norm["volume"].rolling(50, min_periods=1).mean()) / \
                        df_norm["volume"].rolling(50, min_periods=1).std().clip(lower=1e-8)

    df_norm = df_norm.fillna(0).replace([np.inf, -np.inf], 0)

    n_rows = len(df)
    lstm_pred = np.full(n_rows, np.nan)
    lstm_conf = np.full(n_rows, np.nan)
    lstm_prob_down = np.full(n_rows, np.nan)
    lstm_prob_flat = np.full(n_rows, np.nan)
    lstm_prob_up = np.full(n_rows, np.nan)

    batch_size = 1024
    sequences = []
    indices = []
    features = df_norm[feature_cols].values

    for i in range(seq_length, n_rows):
        sequences.append(features[i - seq_length:i])
        indices.append(i)

        if len(sequences) >= batch_size or i == n_rows - 1:
            X = np.array(sequences, dtype=np.float32)
            proba = lstm_model.predict_proba(X)
            preds = np.argmax(proba, axis=1) - 1
            confs = np.max(proba, axis=1)

            for j, idx in enumerate(indices):
                lstm_pred[idx] = preds[j]
                lstm_conf[idx] = confs[j]
                lstm_prob_down[idx] = proba[j, 0]
                lstm_prob_flat[idx] = proba[j, 1]
                lstm_prob_up[idx] = proba[j, 2]

            sequences = []
            indices = []

    df[f"{prefix}_pred"] = lstm_pred
    df[f"{prefix}_conf"] = lstm_conf
    df[f"{prefix}_prob_down"] = lstm_prob_down
    df[f"{prefix}_prob_flat"] = lstm_prob_flat
    df[f"{prefix}_prob_up"] = lstm_prob_up

    return df


def train_and_test(df, algorithm='DQN', timesteps=100000):
    """Train and evaluate."""
    train_env, val_env, test_env = create_train_test_envs(
        df,
        train_ratio=0.7,
        val_ratio=0.15,
        episode_length=500,
        initial_balance=10000.0,
        transaction_cost=0.001,
    )

    train_env = Monitor(train_env)

    if algorithm == 'PPO':
        model = PPO("MlpPolicy", train_env, learning_rate=3e-4, verbose=0, device='cpu')
    else:
        model = DQN("MlpPolicy", train_env, learning_rate=1e-4, verbose=0, device='cpu')

    model.learn(total_timesteps=timesteps, progress_bar=False)

    # Evaluate on test
    test_df = df.iloc[int(len(df) * 0.85):].copy()
    test_env_eval = TradingEnv(
        test_df,
        episode_length=None,
        initial_balance=10000.0,
        transaction_cost=0.001,
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


def main():
    print("=" * 70)
    print("LSTM VALUE TEST - Rigorous Comparison")
    print("=" * 70)
    print("Does the 32.9% accuracy LSTM actually help the RL agent?")
    print()

    # Load LSTM model
    lstm_path = os.path.join(os.path.dirname(__file__), LSTM_MODEL_PATH)
    print(f"Loading LSTM model: {lstm_path}")
    lstm_model = LSTMClassifier.load(lstm_path)
    print(f"  LSTM loaded: {lstm_model.summary()}")
    print()

    symbols = ['BTCUSDT', 'ETHUSDT']
    algo = 'DQN'  # DQN showed best results with LSTM
    runs = 3
    timesteps = 100000

    results_with = []
    results_without = []

    for symbol in symbols:
        print(f"\n{'=' * 70}")
        print(f"Testing: {symbol}")
        print("=" * 70)

        # Get base data
        df_base = get_data(symbol=symbol, timeframe='1h', limit=1500)

        # Create WITH and WITHOUT versions
        df_with = df_base.copy()
        df_with = add_lstm_predictions(df_with, lstm_model, prefix="lstm")
        df_with = df_with.dropna(subset=["lstm_pred"]).reset_index(drop=True)
        print(f"  WITH LSTM: {len(df_with)} rows, has lstm_pred: True")

        df_without = df_base.copy()
        print(f"  WITHOUT LSTM: {len(df_without)} rows")

        pnls_with = []
        pnls_without = []

        for run in range(runs):
            print(f"\n  Run {run + 1}/{runs}")

            # WITH LSTM
            print(f"    Training WITH LSTM...", end=" ", flush=True)
            result_with = train_and_test(df_with, algorithm=algo, timesteps=timesteps)
            pnls_with.append(result_with['pnl'])
            print(f"PnL: ${result_with['pnl']:.2f}, WR: {result_with['win_rate']:.1%}")

            # WITHOUT LSTM
            print(f"    Training WITHOUT LSTM...", end=" ", flush=True)
            result_without = train_and_test(df_without, algorithm=algo, timesteps=timesteps)
            pnls_without.append(result_without['pnl'])
            print(f"PnL: ${result_without['pnl']:.2f}, WR: {result_without['win_rate']:.1%}")

        # Stats for this symbol
        avg_with = np.mean(pnls_with)
        avg_without = np.mean(pnls_without)
        std_with = np.std(pnls_with)
        std_without = np.std(pnls_without)

        results_with.append({'symbol': symbol, 'avg': avg_with, 'std': std_with, 'runs': pnls_with})
        results_without.append({'symbol': symbol, 'avg': avg_without, 'std': std_without, 'runs': pnls_without})

        print(f"\n  {symbol} Summary:")
        print(f"    WITH LSTM:    Avg=${avg_with:>8.2f} (std=${std_with:.2f})")
        print(f"    WITHOUT LSTM: Avg=${avg_without:>8.2f} (std=${std_without:.2f})")
        print(f"    Difference:   ${avg_with - avg_without:>+8.2f}")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Algorithm: {algo}, Timesteps: {timesteps:,}, Runs: {runs}")
    print()

    total_with = 0
    total_without = 0

    for rw, rwo in zip(results_with, results_without):
        diff = rw['avg'] - rwo['avg']
        status = "[LSTM HELPS]" if diff > 0 else "[LSTM HURTS]"
        print(f"  {rw['symbol']:10} WITH: ${rw['avg']:>8.2f}  WITHOUT: ${rwo['avg']:>8.2f}  DIFF: ${diff:>+8.2f} {status}")
        total_with += rw['avg']
        total_without += rwo['avg']

    print()
    total_diff = total_with - total_without
    if total_diff > 0:
        print(f"  CONCLUSION: LSTM adds ${total_diff:.2f} value on average")
        print(f"  -> Keep LSTM in pipeline despite low accuracy")
        print(f"  -> The 'noise' may be helping exploration")
    else:
        print(f"  CONCLUSION: LSTM hurts performance by ${-total_diff:.2f}")
        print(f"  -> Remove LSTM from pipeline")
        print(f"  -> Focus on strategy signals + regime only")


if __name__ == "__main__":
    main()

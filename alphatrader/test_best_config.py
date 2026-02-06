#!/usr/bin/env python3
"""
Test Best Configuration with Extended Runs

Best config found:
- Timeframe: 4H
- Filter: Avoid HIGH_VOL regime
- LSTM: Yes (trained on 4H)
- More timesteps: 200k
- More runs: 5
"""

import os
import sys
import numpy as np
import pandas as pd
import psycopg2
import json
from typing import Dict, Optional
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.env import TradingEnv, create_train_test_envs
from src.models.lstm_classifier import LSTMClassifier
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

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

    df = df.sort_values('open_time').reset_index(drop=True)

    for col in signal_cols:
        df[col] = df[col].fillna(0)

    return df


def add_lstm_predictions(
    df: pd.DataFrame,
    lstm_model: LSTMClassifier,
    seq_length: int = 50,
    prefix: str = "lstm",
) -> pd.DataFrame:
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
    lstm_conf = np.full(n_rows, np.nan)
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
                lstm_prob_up[idx] = proba[j, 2]

            sequences = []
            indices = []

    df[f"{prefix}_pred"] = lstm_pred
    df[f"{prefix}_conf"] = lstm_conf
    df[f"{prefix}_prob_up"] = lstm_prob_up

    return df


def train_and_test_rl(df: pd.DataFrame, timesteps: int = 200000) -> Dict:
    """Train DQN and evaluate."""
    train_env, val_env, test_env = create_train_test_envs(
        df,
        train_ratio=0.7,
        val_ratio=0.15,
        episode_length=500,
        initial_balance=10000.0,
        transaction_cost=0.001,
    )

    train_env = Monitor(train_env)
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
    print("BEST CONFIGURATION TEST - Extended Validation")
    print("=" * 70)
    print("Config: 4H timeframe + LSTM + no HIGH_VOL filter")
    print("Timesteps: 200,000")
    print("Runs: 5")
    print()

    timeframe = "4h"
    timesteps = 200000
    runs = 5

    # Load LSTM
    lstm = LSTMClassifier.load(f"checkpoints/lstm_{timeframe}_mtf")
    print(f"LSTM loaded: {lstm.input_size} features")

    results = {}

    for symbol in SYMBOLS:
        print(f"\n{'=' * 70}")
        print(f"Testing: {symbol}")
        print("=" * 70)

        # Load data
        df = get_data(symbol, timeframe, limit=10000)
        print(f"  Loaded {len(df)} rows")

        # Add LSTM
        df = add_lstm_predictions(df, lstm, prefix="lstm")
        df = df.dropna(subset=["lstm_pred"]).reset_index(drop=True)
        print(f"  After LSTM: {len(df)} rows")

        # Filter out HIGH_VOL
        df = df[df['regime'] != 'HIGH_VOL'].reset_index(drop=True)
        print(f"  After no_high_vol filter: {len(df)} rows")

        # Run multiple times
        pnls = []
        win_rates = []
        trades = []

        for run in range(runs):
            print(f"  Run {run+1}/{runs}...", end=" ", flush=True)
            result = train_and_test_rl(df, timesteps=timesteps)
            pnls.append(result['pnl'])
            win_rates.append(result['win_rate'])
            trades.append(result['trades'])
            status = "[WIN]" if result['pnl'] > 0 else "[LOSS]"
            print(f"PnL: ${result['pnl']:>8.2f}, WR: {result['win_rate']:.1%}, Trades: {result['trades']} {status}")

        # Stats
        avg_pnl = np.mean(pnls)
        std_pnl = np.std(pnls)
        avg_wr = np.mean(win_rates)
        avg_trades = np.mean(trades)
        win_pct = sum(1 for p in pnls if p > 0) / len(pnls)

        results[symbol] = {
            "avg_pnl": avg_pnl,
            "std_pnl": std_pnl,
            "avg_win_rate": avg_wr,
            "avg_trades": avg_trades,
            "runs": pnls,
            "profitable_runs": f"{sum(1 for p in pnls if p > 0)}/{len(pnls)}",
        }

        print(f"\n  {symbol} Summary:")
        print(f"    Avg PnL: ${avg_pnl:>8.2f} (std: ${std_pnl:.2f})")
        print(f"    Avg Win Rate: {avg_wr:.1%}")
        print(f"    Avg Trades: {avg_trades:.0f}")
        print(f"    Profitable Runs: {sum(1 for p in pnls if p > 0)}/{len(pnls)} ({win_pct:.0%})")

    # Overall Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    total_avg = sum(r['avg_pnl'] for r in results.values())
    print(f"\nTotal Avg PnL across all symbols: ${total_avg:,.2f}")
    print()

    for symbol, r in results.items():
        status = "[PROFITABLE]" if r['avg_pnl'] > 0 else "[LOSING]"
        print(f"  {symbol:12} ${r['avg_pnl']:>8.2f} +/- ${r['std_pnl']:.2f}  WR: {r['avg_win_rate']:.1%}  Runs: {r['profitable_runs']} {status}")

    # Save
    with open("best_config_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to best_config_results.json")


if __name__ == "__main__":
    main()

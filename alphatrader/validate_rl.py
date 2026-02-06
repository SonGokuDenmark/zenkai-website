#!/usr/bin/env python3
"""
Multi-symbol, multi-timeframe RL validation.

Tests trained RL agents across multiple symbols and timeframes to ensure
results generalize beyond BTC/1H.

Usage:
    # Quick validation (BTC, ETH, SOL on 1h)
    python validate_rl.py --timesteps 100000

    # Full validation (more symbols, more timeframes)
    python validate_rl.py --timesteps 200000 --symbols BTCUSDT ETHUSDT SOLUSDT DOTUSDT PEPEUSDT --timeframes 1h 4h

    # Use existing LSTM model
    python validate_rl.py --lstm-model checkpoints/lstm_1h
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import socket

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import psycopg2

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.env import TradingEnv, create_train_test_envs
from src.models.lstm_classifier import LSTMClassifier


def _get_db_host():
    """Auto-detect which IP to use for database connection."""
    if os.getenv("ZENKAI_DB_HOST"):
        return os.getenv("ZENKAI_DB_HOST")
    local_ip = os.getenv("ZENKAI_DB_HOST", "192.168.0.160")
    tailscale_ip = os.getenv("ZENKAI_DB_HOST_TAILSCALE", "100.110.101.78")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((local_ip, 5432))
        sock.close()
        if result == 0:
            return local_ip
    except:
        pass
    return tailscale_ip


DB_CONFIG = {
    "host": _get_db_host(),
    "database": os.getenv("ZENKAI_DB_NAME", "zenkai_data"),
    "user": os.getenv("ZENKAI_DB_USER", "zenkai"),
    "password": os.getenv("ZENKAI_DB_PASSWORD"),
}


def load_trading_data(
    symbol: str,
    timeframe: str,
    limit: int = 50000,
) -> pd.DataFrame:
    """Load preprocessed data from database."""
    conn = psycopg2.connect(**DB_CONFIG)

    # Get signal columns
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
        WHERE timeframe = %s
        AND regime IS NOT NULL
        AND symbol = %s
        ORDER BY open_time
        LIMIT %s
    """

    df = pd.read_sql_query(query, conn, params=[timeframe, symbol, limit])
    conn.close()

    return df


def add_lstm_predictions(
    df: pd.DataFrame,
    lstm_model: LSTMClassifier,
    seq_length: int = 50,
    prefix: str = "lstm",
) -> pd.DataFrame:
    """Add LSTM predictions as columns."""
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


def train_and_evaluate(
    df: pd.DataFrame,
    algorithm: str,
    timesteps: int,
    episode_length: int = 500,
) -> Dict:
    """Train a single agent and evaluate on test set."""
    from stable_baselines3 import PPO, A2C, DQN
    from stable_baselines3.common.monitor import Monitor

    algorithm_map = {"PPO": PPO, "A2C": A2C, "DQN": DQN}
    AlgorithmClass = algorithm_map[algorithm]

    # Create environments
    train_env, val_env, test_env = create_train_test_envs(
        df,
        train_ratio=0.7,
        val_ratio=0.15,
        episode_length=episode_length,
        initial_balance=10000.0,
        transaction_cost=0.001,
    )

    train_env = Monitor(train_env)

    # Create and train model
    if algorithm == "PPO":
        model = AlgorithmClass("MlpPolicy", train_env, learning_rate=3e-4, verbose=0)
    elif algorithm == "A2C":
        model = AlgorithmClass("MlpPolicy", train_env, learning_rate=7e-4, verbose=0)
    elif algorithm == "DQN":
        model = AlgorithmClass("MlpPolicy", train_env, learning_rate=1e-4, verbose=0)

    model.learn(total_timesteps=timesteps, progress_bar=False)

    # Evaluate on test set (full episode)
    test_env_eval = TradingEnv(
        df.iloc[int(len(df) * 0.85):].copy(),
        episode_length=None,
        initial_balance=10000.0,
        transaction_cost=0.001,
    )

    obs, info = test_env_eval.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env_eval.step(action)
        total_reward += reward
        done = terminated or truncated

    return {
        "total_pnl": info["total_pnl"],
        "win_rate": info["win_rate"],
        "max_drawdown": info["max_drawdown"],
        "total_trades": info["total_trades"],
        "final_balance": info["balance"],
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-symbol RL validation")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
                        help="Symbols to test")
    parser.add_argument("--timeframes", nargs="+", default=["1h"],
                        help="Timeframes to test")
    parser.add_argument("--algorithms", nargs="+", default=["PPO"],
                        choices=["PPO", "DQN", "A2C"], help="Algorithms to test")
    parser.add_argument("--timesteps", type=int, default=100000, help="Training timesteps")
    parser.add_argument("--limit", type=int, default=50000, help="Max rows per symbol")
    parser.add_argument("--lstm-model", default=None, help="Path to LSTM model")
    parser.add_argument("--runs", type=int, default=1, help="Runs per configuration (for variance)")
    parser.add_argument("--output", default="validation_results.json", help="Output file")

    args = parser.parse_args()

    print("=" * 70)
    print("Multi-Symbol RL Validation")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print(f"Symbols: {args.symbols}")
    print(f"Timeframes: {args.timeframes}")
    print(f"Algorithms: {args.algorithms}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Runs per config: {args.runs}")
    print()

    # Load LSTM model if specified
    lstm_model = None
    if args.lstm_model:
        print(f"Loading LSTM model: {args.lstm_model}")
        lstm_model = LSTMClassifier.load(args.lstm_model)

    all_results = []

    for symbol in args.symbols:
        for timeframe in args.timeframes:
            print(f"\n{'=' * 70}")
            print(f"Testing: {symbol} / {timeframe}")
            print("=" * 70)

            # Load data
            df = load_trading_data(symbol, timeframe, args.limit)
            print(f"  Loaded {len(df)} rows")

            if len(df) < 500:
                print(f"  SKIP: Not enough data")
                continue

            # Add LSTM predictions if available
            if lstm_model:
                df = add_lstm_predictions(df, lstm_model, prefix="lstm")
                df = df.dropna(subset=["lstm_pred"]).reset_index(drop=True)
                print(f"  After LSTM: {len(df)} rows")

            if len(df) < 300:
                print(f"  SKIP: Not enough data after LSTM")
                continue

            for algo in args.algorithms:
                run_results = []

                for run in range(args.runs):
                    print(f"  {algo} run {run+1}/{args.runs}...", end=" ", flush=True)

                    try:
                        result = train_and_evaluate(df, algo, args.timesteps)
                        run_results.append(result)
                        print(f"PnL: ${result['total_pnl']:,.2f}, "
                              f"WR: {result['win_rate']:.1%}, "
                              f"Trades: {result['total_trades']}")
                    except Exception as e:
                        print(f"ERROR: {e}")
                        continue

                if run_results:
                    # Aggregate results
                    avg_result = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "algorithm": algo,
                        "runs": len(run_results),
                        "avg_pnl": np.mean([r["total_pnl"] for r in run_results]),
                        "std_pnl": np.std([r["total_pnl"] for r in run_results]) if len(run_results) > 1 else 0,
                        "avg_win_rate": np.mean([r["win_rate"] for r in run_results]),
                        "avg_max_drawdown": np.mean([r["max_drawdown"] for r in run_results]),
                        "avg_trades": np.mean([r["total_trades"] for r in run_results]),
                        "with_lstm": lstm_model is not None,
                    }
                    all_results.append(avg_result)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print(f"\n{'Symbol':<10} {'TF':<5} {'Algo':<6} {'Avg PnL':>12} {'Win Rate':>10} {'Trades':>8} {'Status'}")
    print("-" * 70)

    profitable = 0
    total = 0

    for r in all_results:
        total += 1
        status = "[OK]" if r["avg_pnl"] > 0 else "[LOSS]"
        if r["avg_pnl"] > 0:
            profitable += 1

        # Flag suspicious results
        if r["avg_win_rate"] > 0.80:
            status += " [!WR]"
        if r["avg_trades"] < 30:
            status += " [!LOW]"

        print(f"{r['symbol']:<10} {r['timeframe']:<5} {r['algorithm']:<6} "
              f"${r['avg_pnl']:>10,.2f} {r['avg_win_rate']:>9.1%} {r['avg_trades']:>7.0f}  {status}")

    print("-" * 70)
    print(f"Profitable: {profitable}/{total} ({profitable/total*100:.0f}%)" if total > 0 else "No results")

    # Red flags check
    print("\n" + "=" * 70)
    print("RED FLAGS CHECK")
    print("=" * 70)

    flags = []
    for r in all_results:
        if r["avg_win_rate"] > 0.80:
            flags.append(f"  [!] {r['symbol']}/{r['timeframe']}/{r['algorithm']}: Win rate {r['avg_win_rate']:.1%} > 80%")
        if r["avg_trades"] < 30:
            flags.append(f"  [!] {r['symbol']}/{r['timeframe']}/{r['algorithm']}: Only {r['avg_trades']:.0f} trades (need 50+)")

    if flags:
        print("\n".join(flags))
    else:
        print("  No red flags detected")

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {
                "symbols": args.symbols,
                "timeframes": args.timeframes,
                "algorithms": args.algorithms,
                "timesteps": args.timesteps,
                "runs": args.runs,
                "lstm_model": args.lstm_model,
            },
            "results": all_results,
            "summary": {
                "total_tests": total,
                "profitable": profitable,
                "profit_rate": profitable / total if total > 0 else 0,
            }
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Finished: {datetime.now()}")


if __name__ == "__main__":
    main()

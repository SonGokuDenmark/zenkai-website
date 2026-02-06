#!/usr/bin/env python3
"""
Extended Symbol Test - Best Config on Top 10 Symbols

Tests the winning config (4H + LSTM + no_high_vol) on more symbols
to validate generalization.
"""

import os
import sys
import numpy as np
import pandas as pd
import psycopg2
import json
from datetime import datetime
from typing import Dict, Optional
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.env import TradingEnv, create_train_test_envs
from src.models.lstm_classifier import LSTMClassifier
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

# Top 10 symbols by volume (diverse: majors, alts, meme)
SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
    'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'PEPEUSDT'
]

# Best config from previous tests
CONFIG = {
    "timeframe": "4h",
    "timesteps": 200000,
    "runs": 3,
    "filter_regime": "HIGH_VOL",  # Filter out this regime
}


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
    lstm_conf = np.full(n_rows, np.nan)

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

            sequences = []
            indices = []

    df["lstm_pred"] = lstm_pred
    df["lstm_conf"] = lstm_conf

    return df


def train_and_test(df: pd.DataFrame, timesteps: int = 200000) -> Dict:
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
    model = DQN("MlpPolicy", train_env, learning_rate=1e-4, verbose=0)
    model.learn(total_timesteps=timesteps, progress_bar=False)

    # Evaluate
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
        "max_dd": info["max_drawdown"],
    }


def main():
    print("=" * 70)
    print("EXTENDED SYMBOL TEST")
    print("=" * 70)
    print(f"Config: {CONFIG['timeframe']} + LSTM + no_{CONFIG['filter_regime'].lower()}")
    print(f"Symbols: {len(SYMBOLS)}")
    print(f"Timesteps: {CONFIG['timesteps']:,}")
    print(f"Runs per symbol: {CONFIG['runs']}")
    print()

    # Load LSTM
    lstm_path = f"checkpoints/lstm_{CONFIG['timeframe']}_mtf"
    try:
        lstm = LSTMClassifier.load(lstm_path)
        print(f"LSTM loaded: {lstm.input_size} features")
    except:
        print(f"Warning: Could not load LSTM from {lstm_path}")
        print("Running without LSTM features")
        lstm = None

    results = {}

    for symbol in SYMBOLS:
        print(f"\n{'=' * 70}")
        print(f"Testing: {symbol}")
        print("=" * 70)

        # Load data
        df = get_data(symbol, CONFIG["timeframe"])
        if df is None:
            print(f"  Skipped: insufficient data")
            results[symbol] = {"error": "insufficient_data"}
            continue

        print(f"  Loaded {len(df)} rows")

        # Add LSTM
        if lstm:
            df = add_lstm_predictions(df, lstm)
            df = df.dropna(subset=["lstm_pred"]).reset_index(drop=True)
            print(f"  After LSTM: {len(df)} rows")

        # Filter regime
        if CONFIG["filter_regime"]:
            df = df[df['regime'] != CONFIG["filter_regime"]].reset_index(drop=True)
            print(f"  After no_{CONFIG['filter_regime'].lower()}: {len(df)} rows")

        if len(df) < 300:
            print(f"  Skipped: insufficient data after filtering")
            results[symbol] = {"error": "insufficient_data_after_filter"}
            continue

        # Run tests
        pnls = []
        win_rates = []
        trades = []
        max_dds = []

        for run in range(CONFIG["runs"]):
            print(f"  Run {run+1}/{CONFIG['runs']}...", end=" ", flush=True)
            result = train_and_test(df, timesteps=CONFIG["timesteps"])
            pnls.append(result['pnl'])
            win_rates.append(result['win_rate'])
            trades.append(result['trades'])
            max_dds.append(result['max_dd'])
            status = "[WIN]" if result['pnl'] > 0 else "[LOSS]"
            print(f"PnL: ${result['pnl']:>8.2f}, WR: {result['win_rate']:.1%} {status}")

        # Stats
        results[symbol] = {
            "avg_pnl": np.mean(pnls),
            "std_pnl": np.std(pnls),
            "avg_win_rate": np.mean(win_rates),
            "avg_trades": np.mean(trades),
            "avg_max_dd": np.mean(max_dds),
            "runs": pnls,
            "profitable_runs": sum(1 for p in pnls if p > 0),
            "total_runs": len(pnls),
        }

        print(f"\n  {symbol} Summary: Avg PnL: ${np.mean(pnls):,.2f}, "
              f"Profitable: {sum(1 for p in pnls if p > 0)}/{len(pnls)}")

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    total_pnl = 0
    profitable = 0
    total = 0

    for symbol, r in results.items():
        if "error" in r:
            print(f"  {symbol:12} ERROR: {r['error']}")
            continue

        total += 1
        if r['avg_pnl'] > 0:
            profitable += 1
        total_pnl += r['avg_pnl']

        status = "[PROFITABLE]" if r['avg_pnl'] > 0 else "[LOSING]"
        print(f"  {symbol:12} ${r['avg_pnl']:>8.2f} +/- ${r['std_pnl']:.2f}  "
              f"WR: {r['avg_win_rate']:.1%}  DD: {r['avg_max_dd']:.1%}  "
              f"Runs: {r['profitable_runs']}/{r['total_runs']} {status}")

    print("-" * 70)
    print(f"  Total symbols tested: {total}")
    print(f"  Profitable symbols: {profitable} ({profitable/total*100:.0f}%)" if total > 0 else "")
    print(f"  Total avg PnL: ${total_pnl:,.2f}")

    # Save
    output_file = "extended_symbols_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": CONFIG,
            "symbols": SYMBOLS,
            "results": results,
            "summary": {
                "total_tested": total,
                "profitable": profitable,
                "total_avg_pnl": total_pnl,
            }
        }, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Regime-Based Filtering Tests

Since regime prediction has 14% edge, test using regime as a trade filter:
1. Only trade in favorable regimes
2. Use higher TF regime to filter lower TF entries
3. Regime-specific position sizing
"""

import os
import sys
import numpy as np
import pandas as pd
import psycopg2
import json
from typing import Dict, Optional, List
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.env import TradingEnv, create_train_test_envs
from src.models.lstm_classifier import LSTMClassifier
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']


def get_data(symbol: str, timeframe: str, limit: int = 5000) -> Optional[pd.DataFrame]:
    """Load data for a specific symbol/timeframe."""
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

    if len(df) < 200:
        return None

    df = df.sort_values('open_time').reset_index(drop=True)

    for col in signal_cols:
        df[col] = df[col].fillna(0)

    return df


def load_lstm_model(timeframe: str) -> Optional[LSTMClassifier]:
    """Load LSTM model for a timeframe."""
    model_path = f"checkpoints/lstm_{timeframe}_mtf"
    try:
        return LSTMClassifier.load(model_path)
    except:
        return None


def add_lstm_predictions(
    df: pd.DataFrame,
    lstm_model: LSTMClassifier,
    seq_length: int = 50,
    prefix: str = "lstm",
) -> pd.DataFrame:
    """Add LSTM predictions to dataframe."""
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


def filter_by_regime(df: pd.DataFrame, allowed_regimes: List[str]) -> pd.DataFrame:
    """Filter data to only include allowed regimes."""
    return df[df['regime'].isin(allowed_regimes)].copy().reset_index(drop=True)


def add_higher_tf_regime(
    entry_df: pd.DataFrame,
    higher_tf: str,
    symbol: str,
    limit: int = 5000,
) -> pd.DataFrame:
    """Add regime from higher timeframe."""
    higher_df = get_data(symbol, higher_tf, limit)
    if higher_df is None:
        return entry_df

    # Prepare higher TF data
    higher_subset = higher_df[["open_time", "regime"]].copy()
    higher_subset = higher_subset.rename(columns={"regime": f"regime_{higher_tf}"})

    # Merge
    entry_df = entry_df.sort_values("open_time")
    higher_subset = higher_subset.sort_values("open_time")

    merged = pd.merge_asof(
        entry_df,
        higher_subset,
        on="open_time",
        direction="backward"
    )

    return merged


def train_and_test_rl(df: pd.DataFrame, timesteps: int = 100000) -> Dict:
    """Train DQN and evaluate on test set."""
    if len(df) < 300:
        return {"pnl": None, "error": "insufficient_data"}

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
    print("REGIME-BASED FILTERING TESTS")
    print("=" * 70)
    print("Testing if regime filtering improves performance")
    print()

    timesteps = 100000
    runs = 2
    timeframe = "4h"  # Best performing TF

    # Test configurations
    tests = [
        ("baseline", None, None),  # No filtering
        ("trending_only", ["TRENDING_UP", "TRENDING_DOWN"], None),
        ("trending_up_only", ["TRENDING_UP"], None),
        ("no_high_vol", ["TRENDING_UP", "TRENDING_DOWN", "RANGING"], None),
        ("4h_trend_filter", None, "1d"),  # Use 1d regime to filter 4h
    ]

    results = {}

    for test_name, allowed_regimes, filter_tf in tests:
        print(f"\n{'=' * 70}")
        print(f"Test: {test_name}")
        if allowed_regimes:
            print(f"  Allowed regimes: {allowed_regimes}")
        if filter_tf:
            print(f"  Filter TF: {filter_tf}")
        print("=" * 70)

        results[test_name] = {}

        for symbol in SYMBOLS:
            print(f"\n  {symbol}:", end=" ", flush=True)

            # Load data
            df = get_data(symbol, timeframe, limit=5000)
            if df is None:
                print("no data")
                continue

            # Add LSTM
            lstm = load_lstm_model(timeframe)
            if lstm:
                df = add_lstm_predictions(df, lstm, prefix="lstm")
                df = df.dropna(subset=["lstm_pred"]).reset_index(drop=True)

            # Add higher TF regime if needed
            if filter_tf:
                df = add_higher_tf_regime(df, filter_tf, symbol)
                # Filter by higher TF trending
                if f"regime_{filter_tf}" in df.columns:
                    df = df[df[f"regime_{filter_tf}"].isin(["TRENDING_UP", "TRENDING_DOWN"])]
                    df = df.reset_index(drop=True)

            # Filter by regime if specified
            if allowed_regimes:
                df = filter_by_regime(df, allowed_regimes)

            if len(df) < 300:
                print(f"insufficient data after filtering ({len(df)} rows)")
                results[test_name][symbol] = {"pnl": None, "error": "insufficient_data"}
                continue

            print(f"{len(df)} rows...", end=" ", flush=True)

            # Run tests
            pnls = []
            for run in range(runs):
                result = train_and_test_rl(df, timesteps=timesteps)
                if result.get('pnl') is not None:
                    pnls.append(result['pnl'])

            if pnls:
                avg_pnl = np.mean(pnls)
                std_pnl = np.std(pnls)
                status = "[WIN]" if avg_pnl > 0 else "[LOSS]"
                print(f"PnL: ${avg_pnl:>8.2f} (std: ${std_pnl:.2f}) {status}")
                results[test_name][symbol] = {
                    "pnl": avg_pnl,
                    "std": std_pnl,
                    "runs": pnls,
                    "rows": len(df),
                }
            else:
                print("failed")
                results[test_name][symbol] = {"pnl": None, "error": "training_failed"}

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nTotal PnL by Test:")
    for test_name, test_results in results.items():
        total = sum(r.get('pnl', 0) or 0 for r in test_results.values())
        count = sum(1 for r in test_results.values() if r.get('pnl') is not None)
        print(f"  {test_name:25} Total: ${total:>10.2f} ({count} symbols)")

    print("\nPer-Symbol Comparison:")
    for symbol in SYMBOLS:
        print(f"\n  {symbol}:")
        for test_name, test_results in results.items():
            if symbol in test_results and test_results[symbol].get('pnl') is not None:
                pnl = test_results[symbol]['pnl']
                status = "[WIN]" if pnl > 0 else "[LOSS]"
                print(f"    {test_name:25} ${pnl:>8.2f} {status}")

    # Save results
    output_file = "regime_filter_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()

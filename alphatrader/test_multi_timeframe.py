#!/usr/bin/env python3
"""
Multi-Timeframe Testing Framework

Tests:
1. Individual timeframes per symbol (find best TF per symbol)
2. Multi-timeframe signals (use higher TF for trend, lower for entry)
3. Cross-timeframe validation
"""

import os
import sys
import numpy as np
import pandas as pd
import psycopg2
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.env import TradingEnv, create_train_test_envs
from src.models.lstm_classifier import LSTMClassifier
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor


# All available timeframes (ordered by duration)
TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w']

# Main symbols to test
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

# Minimum rows needed for training
MIN_ROWS = 500


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

    if len(df) < MIN_ROWS:
        return None

    df = df.sort_values('open_time').reset_index(drop=True)

    for col in signal_cols:
        df[col] = df[col].fillna(0)

    return df


def train_lstm_for_timeframe(
    timeframe: str,
    symbols: List[str] = SYMBOLS,
    limit: int = 10000,
    epochs: int = 30,
    seq_length: int = 50,
) -> Optional[LSTMClassifier]:
    """Train an LSTM model for a specific timeframe using multiple symbols."""
    print(f"\n  Training LSTM for {timeframe}...")

    all_dfs = []
    for symbol in symbols:
        df = get_data(symbol, timeframe, limit)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        print(f"    No data for {timeframe}")
        return None

    # Combine data
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"    Combined data: {len(combined_df):,} rows from {len(all_dfs)} symbols")

    # Prepare features (same as validate_rl.py)
    feature_cols = ["open", "high", "low", "close", "volume"]
    signal_cols = sorted([c for c in combined_df.columns if c.startswith("signal_") or c.startswith("conf_")])
    feature_cols.extend(signal_cols)

    regimes = ["TRENDING_UP", "TRENDING_DOWN", "RANGING", "HIGH_VOL"]
    for regime in regimes:
        col_name = f"regime_{regime.lower()}"
        combined_df[col_name] = (combined_df["regime"] == regime).astype(float)
        feature_cols.append(col_name)

    # Normalize
    df_norm = combined_df.copy()
    for col in ["open", "high", "low", "close"]:
        df_norm[col] = (df_norm[col] - df_norm[col].rolling(50, min_periods=1).mean()) / \
                       df_norm[col].rolling(50, min_periods=1).std().clip(lower=1e-8)
    df_norm["volume"] = (df_norm["volume"] - df_norm["volume"].rolling(50, min_periods=1).mean()) / \
                        df_norm["volume"].rolling(50, min_periods=1).std().clip(lower=1e-8)
    df_norm = df_norm.fillna(0).replace([np.inf, -np.inf], 0)

    # Compute labels (direction prediction)
    combined_df['future_return'] = combined_df['close'].shift(-10) / combined_df['close'] - 1
    combined_df['label'] = 1  # FLAT
    combined_df.loc[combined_df['future_return'] > 0.015, 'label'] = 2  # UP
    combined_df.loc[combined_df['future_return'] < -0.015, 'label'] = 0  # DOWN

    # Create sequences per symbol (to avoid crossing symbol boundaries)
    X_list = []
    y_list = []

    for symbol in combined_df['symbol'].unique():
        sym_mask = combined_df['symbol'] == symbol
        sym_features = df_norm.loc[sym_mask, feature_cols].values
        sym_labels = combined_df.loc[sym_mask, 'label'].values

        for i in range(len(sym_features) - seq_length - 10):
            X_list.append(sym_features[i:i + seq_length])
            y_list.append(sym_labels[i + seq_length])

    if len(X_list) < 1000:
        print(f"    Not enough sequences for {timeframe}")
        return None

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"    Train: {len(X_train):,}, Val: {len(X_val):,}, Features: {X.shape[2]}")

    # Train LSTM
    lstm = LSTMClassifier(
        input_size=X.shape[2],
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        learning_rate=0.001,
    )

    lstm.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=64,
        early_stopping_patience=10,
        verbose=False,
    )

    # Evaluate
    val_acc = lstm.evaluate(X_val, y_val)
    print(f"    Val accuracy: {val_acc:.1%}")

    return lstm


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


def train_and_test_rl(df: pd.DataFrame, timesteps: int = 100000) -> Dict:
    """Train DQN and evaluate on test set."""
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


def test_single_timeframe(
    symbol: str,
    timeframe: str,
    lstm_model: Optional[LSTMClassifier],
    timesteps: int = 100000,
    runs: int = 2,
) -> Dict:
    """Test a single timeframe for a symbol."""
    df = get_data(symbol, timeframe, limit=5000)
    if df is None:
        return {"pnl": None, "error": "insufficient_data"}

    # Add LSTM predictions if model available
    if lstm_model:
        df = add_lstm_predictions(df, lstm_model, prefix="lstm")
        df = df.dropna(subset=["lstm_pred"]).reset_index(drop=True)

    if len(df) < MIN_ROWS:
        return {"pnl": None, "error": "insufficient_data_after_lstm"}

    pnls = []
    for run in range(runs):
        result = train_and_test_rl(df, timesteps=timesteps)
        pnls.append(result['pnl'])

    return {
        "pnl": np.mean(pnls),
        "std": np.std(pnls),
        "runs": pnls,
        "rows": len(df),
    }


def main():
    print("=" * 70)
    print("MULTI-TIMEFRAME TESTING FRAMEWORK")
    print("=" * 70)
    print(f"Symbols: {SYMBOLS}")
    print(f"Timeframes: {TIMEFRAMES}")
    print()

    # Test parameters
    timesteps = 100000
    runs_per_config = 2
    lstm_epochs = 20

    # Timeframes to test (start with common ones)
    test_timeframes = ['15m', '30m', '1h', '4h', '1d']

    results = {}
    lstm_models = {}

    # Phase 1: Train LSTM for each timeframe
    print("=" * 70)
    print("PHASE 1: Training LSTM models per timeframe")
    print("=" * 70)

    for tf in test_timeframes:
        lstm = train_lstm_for_timeframe(tf, symbols=SYMBOLS, limit=10000, epochs=lstm_epochs)
        if lstm:
            lstm_models[tf] = lstm
            # Save model
            save_path = f"checkpoints/lstm_{tf}_mtf"
            lstm.save(save_path)
            print(f"    Saved to {save_path}")

    # Phase 2: Test each symbol on each timeframe
    print("\n" + "=" * 70)
    print("PHASE 2: Testing symbol/timeframe combinations")
    print("=" * 70)

    for symbol in SYMBOLS:
        results[symbol] = {}
        print(f"\n{symbol}:")
        print("-" * 50)

        for tf in test_timeframes:
            lstm = lstm_models.get(tf)
            print(f"  {tf:6}...", end=" ", flush=True)

            result = test_single_timeframe(
                symbol, tf, lstm,
                timesteps=timesteps,
                runs=runs_per_config,
            )

            results[symbol][tf] = result

            if result.get('pnl') is not None:
                status = "[WIN]" if result['pnl'] > 0 else "[LOSS]"
                print(f"PnL: ${result['pnl']:>8.2f} (std: ${result['std']:.2f}) {status}")
            else:
                print(f"Error: {result.get('error', 'unknown')}")

    # Phase 3: Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Best Timeframe per Symbol")
    print("=" * 70)

    for symbol in SYMBOLS:
        best_tf = None
        best_pnl = float('-inf')

        for tf, result in results[symbol].items():
            if result.get('pnl') is not None and result['pnl'] > best_pnl:
                best_pnl = result['pnl']
                best_tf = tf

        if best_tf:
            print(f"  {symbol:12} Best: {best_tf:6} PnL: ${best_pnl:>8.2f}")
        else:
            print(f"  {symbol:12} No profitable timeframe found")

    # Save results
    output_file = "mtf_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Identify best TF per symbol")
    print("2. Test multi-TF combinations (e.g., 4h trend + 1h entry)")
    print("3. Build regime-aware filters using higher TF regime")


if __name__ == "__main__":
    main()

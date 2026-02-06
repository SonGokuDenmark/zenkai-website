#!/usr/bin/env python3
"""
Multi-Timeframe Combination Tests

Test different MTF strategies:
1. Higher TF trend + Lower TF entry
2. Multiple TF confirmation
3. TF-specific regime filtering
"""

import os
import sys
import numpy as np
import pandas as pd
import psycopg2
import json
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.env import TradingEnv, create_train_test_envs
from src.models.lstm_classifier import LSTMClassifier
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

# Symbols to test
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

# MTF Combinations to test
MTF_COMBOS = [
    # (entry_tf, trend_tf, validation_tf) - None means not used
    ('1h', '4h', None),           # 4h trend, 1h entry
    ('15m', '1h', '4h'),          # 4h validation, 1h trend, 15m entry
    ('30m', '4h', None),          # 4h trend, 30m entry
    ('1h', '4h', '1d'),           # 1d validation, 4h trend, 1h entry
    ('15m', '4h', None),          # 4h trend, 15m entry (scalping)
]


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


def merge_higher_tf_signals(
    entry_df: pd.DataFrame,
    higher_df: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """
    Merge higher timeframe signals into entry timeframe data.
    Uses forward-fill to propagate higher TF signals.
    """
    # Get the columns we want from higher TF
    cols_to_merge = [
        f"{prefix}_pred", f"{prefix}_conf", f"{prefix}_prob_up",
        "regime"
    ]

    # Prepare higher TF data with timestamp
    higher_subset = higher_df[["open_time"] + [c for c in cols_to_merge if c in higher_df.columns]].copy()

    # Rename columns with prefix
    rename_map = {c: f"{prefix}_{c}" if c != "open_time" else c for c in higher_subset.columns}
    rename_map["regime"] = f"{prefix}_regime"
    higher_subset = higher_subset.rename(columns=rename_map)

    # Sort both by time
    entry_df = entry_df.sort_values("open_time")
    higher_subset = higher_subset.sort_values("open_time")

    # Merge using asof (forward fill from higher TF)
    merged = pd.merge_asof(
        entry_df,
        higher_subset,
        on="open_time",
        direction="backward"
    )

    return merged


def create_mtf_features(
    symbol: str,
    entry_tf: str,
    trend_tf: str,
    validation_tf: Optional[str] = None,
    limit: int = 5000,
) -> Optional[pd.DataFrame]:
    """
    Create multi-timeframe feature set.

    Args:
        symbol: Trading symbol
        entry_tf: Timeframe for entry signals
        trend_tf: Timeframe for trend confirmation
        validation_tf: Optional higher TF for validation
        limit: Max rows to fetch
    """
    # Load entry TF data
    entry_df = get_data(symbol, entry_tf, limit)
    if entry_df is None:
        return None

    # Load and add LSTM for entry TF
    entry_lstm = load_lstm_model(entry_tf)
    if entry_lstm:
        entry_df = add_lstm_predictions(entry_df, entry_lstm, prefix="entry_lstm")

    # Load trend TF data
    trend_df = get_data(symbol, trend_tf, limit)
    if trend_df is None:
        return None

    # Load and add LSTM for trend TF
    trend_lstm = load_lstm_model(trend_tf)
    if trend_lstm:
        trend_df = add_lstm_predictions(trend_df, trend_lstm, prefix="trend_lstm")

    # Merge trend signals into entry data
    entry_df = merge_higher_tf_signals(entry_df, trend_df, "trend")

    # Optionally add validation TF
    if validation_tf:
        val_df = get_data(symbol, validation_tf, limit)
        if val_df is not None:
            val_lstm = load_lstm_model(validation_tf)
            if val_lstm:
                val_df = add_lstm_predictions(val_df, val_lstm, prefix="val_lstm")
            entry_df = merge_higher_tf_signals(entry_df, val_df, "val")

    # Drop rows with NaN from merging
    entry_df = entry_df.dropna().reset_index(drop=True)

    return entry_df


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
    print("MULTI-TIMEFRAME COMBINATION TESTS")
    print("=" * 70)
    print(f"Symbols: {SYMBOLS}")
    print(f"Combos: {len(MTF_COMBOS)}")
    print()

    timesteps = 100000
    runs = 2

    results = {}

    for entry_tf, trend_tf, val_tf in MTF_COMBOS:
        combo_name = f"{entry_tf}+{trend_tf}" + (f"+{val_tf}" if val_tf else "")
        print(f"\n{'=' * 70}")
        print(f"Testing Combo: {combo_name}")
        print(f"  Entry: {entry_tf}, Trend: {trend_tf}, Validation: {val_tf or 'None'}")
        print("=" * 70)

        results[combo_name] = {}

        for symbol in SYMBOLS:
            print(f"\n  {symbol}:", end=" ", flush=True)

            # Create MTF features
            df = create_mtf_features(
                symbol, entry_tf, trend_tf, val_tf,
                limit=5000
            )

            if df is None or len(df) < 300:
                print("insufficient data")
                results[combo_name][symbol] = {"pnl": None, "error": "insufficient_data"}
                continue

            print(f"{len(df)} rows...", end=" ", flush=True)

            # Run multiple times
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
                results[combo_name][symbol] = {
                    "pnl": avg_pnl,
                    "std": std_pnl,
                    "runs": pnls,
                    "rows": len(df),
                }
            else:
                print("failed")
                results[combo_name][symbol] = {"pnl": None, "error": "training_failed"}

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Best Combo per Symbol")
    print("=" * 70)

    for symbol in SYMBOLS:
        best_combo = None
        best_pnl = float('-inf')

        for combo_name, combo_results in results.items():
            if symbol in combo_results:
                pnl = combo_results[symbol].get('pnl')
                if pnl is not None and pnl > best_pnl:
                    best_pnl = pnl
                    best_combo = combo_name

        if best_combo:
            print(f"  {symbol:12} Best: {best_combo:20} PnL: ${best_pnl:>8.2f}")
        else:
            print(f"  {symbol:12} No profitable combo found")

    # Overall best combo
    print("\n" + "=" * 70)
    print("OVERALL: Best Combo Across All Symbols")
    print("=" * 70)

    combo_totals = {}
    for combo_name, combo_results in results.items():
        total = 0
        count = 0
        for symbol, result in combo_results.items():
            if result.get('pnl') is not None:
                total += result['pnl']
                count += 1
        if count == len(SYMBOLS):
            combo_totals[combo_name] = total

    if combo_totals:
        best_overall = max(combo_totals.items(), key=lambda x: x[1])
        print(f"  Best Combo: {best_overall[0]}")
        print(f"  Total PnL:  ${best_overall[1]:.2f}")
        print()
        print("  All combos:")
        for combo, total in sorted(combo_totals.items(), key=lambda x: -x[1]):
            status = "[BEST]" if combo == best_overall[0] else ""
            print(f"    {combo:25} Total: ${total:>10.2f} {status}")

    # Save results
    output_file = "mtf_combo_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()

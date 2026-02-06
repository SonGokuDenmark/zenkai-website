"""
Train RL agent using TradingEnv and preprocessed features.

Uses Stable-Baselines3 PPO with LSTM predictions + strategy signals.
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import socket

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import psycopg2
import mlflow
from tqdm import tqdm

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


def _get_mlflow_uri():
    """Auto-detect MLflow tracking server URI."""
    local_ip = os.getenv("ZENKAI_DB_HOST", "192.168.0.160")
    tailscale_ip = os.getenv("ZENKAI_DB_HOST_TAILSCALE", "100.110.101.78")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        if sock.connect_ex((local_ip, 5000)) == 0:
            sock.close()
            return f"http://{local_ip}:5000"
    except:
        pass
    return f"http://{tailscale_ip}:5000"


DB_CONFIG = {
    "host": _get_db_host(),
    "database": os.getenv("ZENKAI_DB_NAME", "zenkai_data"),
    "user": os.getenv("ZENKAI_DB_USER", "zenkai"),
    "password": os.getenv("ZENKAI_DB_PASSWORD"),
}


def load_trading_data(
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    limit: int = 50000,
) -> pd.DataFrame:
    """Load preprocessed data from database."""
    print(f"Loading {symbol} {timeframe} data (limit={limit})...")

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

    # Build query
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

    print(f"  Loaded {len(df)} rows")
    print(f"  Date range: {df['open_time'].min()} to {df['open_time'].max()}")

    return df


def add_lstm_predictions(
    df: pd.DataFrame,
    lstm_model: LSTMClassifier,
    seq_length: int = 50,
    prefix: str = "lstm",
) -> pd.DataFrame:
    """
    Add LSTM predictions as columns to the DataFrame.

    For each row, uses the preceding seq_length rows to make a prediction.
    Rows without enough history get NaN.

    Adds columns (with prefix):
        - {prefix}_pred: prediction (-1, 0, 1)
        - {prefix}_conf: confidence (0-1)
        - {prefix}_prob_down: P(down)
        - {prefix}_prob_flat: P(flat)
        - {prefix}_prob_up: P(up)
    """
    print(f"\nAdding {prefix} predictions (seq_length={seq_length})...")

    # Get feature columns used by LSTM (OHLCV + signals + regime one-hot)
    # Must match what train_lstm.py uses
    feature_cols = ["open", "high", "low", "close", "volume"]
    signal_cols = sorted([c for c in df.columns if c.startswith("signal_") or c.startswith("conf_")])
    feature_cols.extend(signal_cols)

    # Add regime one-hot columns (create them if needed)
    regimes = ["TRENDING_UP", "TRENDING_DOWN", "RANGING", "HIGH_VOL"]
    for regime in regimes:
        col_name = f"regime_{regime.lower()}"
        if col_name not in df.columns:
            df[col_name] = (df["regime"] == regime).astype(float)
        feature_cols.append(col_name)

    # Normalize OHLCV (same as train_lstm.py)
    df_norm = df.copy()
    for col in ["open", "high", "low", "close"]:
        df_norm[col] = (df_norm[col] - df_norm[col].rolling(50, min_periods=1).mean()) / \
                       df_norm[col].rolling(50, min_periods=1).std().clip(lower=1e-8)
    df_norm["volume"] = (df_norm["volume"] - df_norm["volume"].rolling(50, min_periods=1).mean()) / \
                        df_norm["volume"].rolling(50, min_periods=1).std().clip(lower=1e-8)

    # Fill NaN and inf
    df_norm = df_norm.fillna(0).replace([np.inf, -np.inf], 0)

    # Initialize output columns
    n_rows = len(df)
    lstm_pred = np.full(n_rows, np.nan)
    lstm_conf = np.full(n_rows, np.nan)
    lstm_prob_down = np.full(n_rows, np.nan)
    lstm_prob_flat = np.full(n_rows, np.nan)
    lstm_prob_up = np.full(n_rows, np.nan)

    # Create sequences and predict in batches
    batch_size = 1024
    sequences = []
    indices = []

    features = df_norm[feature_cols].values

    for i in range(seq_length, n_rows):
        sequences.append(features[i - seq_length:i])
        indices.append(i)

        # Process batch
        if len(sequences) >= batch_size or i == n_rows - 1:
            X = np.array(sequences, dtype=np.float32)
            proba = lstm_model.predict_proba(X)
            preds = np.argmax(proba, axis=1) - 1  # 0,1,2 -> -1,0,1
            confs = np.max(proba, axis=1)

            for j, idx in enumerate(indices):
                lstm_pred[idx] = preds[j]
                lstm_conf[idx] = confs[j]
                lstm_prob_down[idx] = proba[j, 0]
                lstm_prob_flat[idx] = proba[j, 1]
                lstm_prob_up[idx] = proba[j, 2]

            sequences = []
            indices = []

    # Add columns to original DataFrame with prefix
    df[f"{prefix}_pred"] = lstm_pred
    df[f"{prefix}_conf"] = lstm_conf
    df[f"{prefix}_prob_down"] = lstm_prob_down
    df[f"{prefix}_prob_flat"] = lstm_prob_flat
    df[f"{prefix}_prob_up"] = lstm_prob_up

    valid_preds = np.sum(~np.isnan(lstm_pred))
    print(f"  Added {valid_preds:,} {prefix} predictions ({valid_preds/n_rows*100:.1f}% of rows)")

    return df


def main():
    parser = argparse.ArgumentParser(description="Train RL trading agent")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--timeframe", default="1h", help="Data timeframe")
    parser.add_argument("--limit", type=int, default=50000, help="Max rows to load")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--algorithm", default="PPO", choices=["PPO", "A2C", "DQN"], help="RL algorithm")
    parser.add_argument("--episode-length", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--lstm-model", default=None, help="Path to LSTM model (without extension)")
    parser.add_argument("--lstm-4h", default=None, help="Path to 4h LSTM model for multi-timeframe")
    parser.add_argument("--seq-length", type=int, default=50, help="LSTM sequence length (default: 50)")
    args = parser.parse_args()

    # Setup MLflow tracking
    mlflow_run = None
    try:
        mlflow.set_tracking_uri(_get_mlflow_uri())
        mlflow.set_experiment("alphatrader-rl")
        mlflow_run = mlflow.start_run(run_name=f"rl-{args.symbol}-{args.algorithm}")
        mlflow.log_params({
            "symbol": args.symbol,
            "timeframe": args.timeframe,
            "algorithm": args.algorithm,
            "timesteps": args.timesteps,
            "episode_length": args.episode_length,
            "limit": args.limit,
            "lstm_model": args.lstm_model or "none",
            "lstm_4h": args.lstm_4h or "none",
        })
        print(f"MLflow tracking: {_get_mlflow_uri()}")
    except Exception as e:
        print(f"Warning: MLflow setup failed: {e}")

    print("=" * 60)
    print("AlphaTrader RL Training")
    print("=" * 60)
    print(f"Started: {datetime.now()}")
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Timesteps: {args.timesteps}")
    print()

    # Load data
    df = load_trading_data(
        symbol=args.symbol,
        timeframe=args.timeframe,
        limit=args.limit,
    )

    if len(df) < 1000:
        print(f"ERROR: Not enough data ({len(df)} rows). Need at least 1000.")
        return

    # Load LSTM models and add predictions if specified
    lstm_cols_to_check = []

    if args.lstm_model:
        print(f"\nLoading 1h LSTM model: {args.lstm_model}")
        lstm_model = LSTMClassifier.load(args.lstm_model)
        print(f"  {lstm_model.name}")
        print(f"  Input size: {lstm_model.input_size}")

        # Add LSTM predictions to DataFrame
        df = add_lstm_predictions(df, lstm_model, seq_length=args.seq_length, prefix="lstm")
        lstm_cols_to_check.append("lstm_pred")

    if args.lstm_4h:
        print(f"\nLoading 4h LSTM model: {args.lstm_4h}")
        lstm_4h_model = LSTMClassifier.load(args.lstm_4h)
        print(f"  {lstm_4h_model.name}")
        print(f"  Input size: {lstm_4h_model.input_size}")

        # Add 4h LSTM predictions with different prefix
        df = add_lstm_predictions(df, lstm_4h_model, seq_length=args.seq_length, prefix="lstm_4h")
        lstm_cols_to_check.append("lstm_4h_pred")

    # Drop rows without any LSTM predictions
    if lstm_cols_to_check:
        df = df.dropna(subset=lstm_cols_to_check).reset_index(drop=True)
        print(f"  After dropping rows without LSTM: {len(df)} rows")

    # Create train/test split
    print("\nCreating environments...")
    train_env, val_env, test_env = create_train_test_envs(
        df,
        train_ratio=0.7,
        val_ratio=0.15,
        episode_length=args.episode_length,
        initial_balance=10000.0,
        transaction_cost=0.001,
    )

    print(f"  Train env: {len(train_env.df)} rows")
    print(f"  Val env: {len(val_env.df)} rows")
    print(f"  Test env: {len(test_env.df)} rows")
    print(f"  Observation space: {train_env.observation_space.shape}")
    print(f"  Action space: {train_env.action_space}")

    # Import Stable-Baselines3
    try:
        from stable_baselines3 import PPO, A2C, DQN
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        print("\nERROR: stable-baselines3 not installed.")
        print("Install with: pip install stable-baselines3")
        return

    # Wrap environments
    train_env = Monitor(train_env)
    val_env = Monitor(val_env)

    # Create algorithm
    algorithm_map = {"PPO": PPO, "A2C": A2C, "DQN": DQN}
    AlgorithmClass = algorithm_map[args.algorithm]

    print(f"\nCreating {args.algorithm} agent...")

    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints/rl")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"{args.algorithm}_{args.symbol}_{args.timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create model
    model = AlgorithmClass(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048 if args.algorithm == "PPO" else 5,
        batch_size=64,
        gamma=0.99,
        tensorboard_log=f"logs/{run_name}",
    )

    # Callbacks
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=str(checkpoint_dir / run_name),
        log_path=str(checkpoint_dir / run_name),
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path=str(checkpoint_dir / run_name),
        name_prefix="rl_model",
    )

    # Train
    print(f"\nTraining for {args.timesteps} timesteps...")
    print("-" * 60)

    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = checkpoint_dir / f"{run_name}_final"
    model.save(str(final_path))
    print(f"\nModel saved to: {final_path}")

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)

    obs, info = test_env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    print(f"  Steps: {steps}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Total Trades: {info['total_trades']}")
    print(f"  Win Rate: {info['win_rate']:.1%}")
    print(f"  Total PnL: ${info['total_pnl']:.2f}")
    print(f"  Max Drawdown: {info['max_drawdown']:.1%}")
    print(f"  Final Balance: ${info['balance']:.2f}")

    # Log metrics to MLflow
    try:
        mlflow.log_metrics({
            "test_pnl": float(info['total_pnl']),
            "win_rate": float(info['win_rate']),
            "max_drawdown": float(info['max_drawdown']),
            "total_trades": int(info['total_trades']),
            "final_balance": float(info['balance']),
            "total_reward": float(total_reward),
            "test_steps": steps,
        })
        print(f"MLflow metrics logged successfully")
    except Exception as e:
        print(f"Warning: MLflow metric logging failed: {e}")
    finally:
        if mlflow_run:
            mlflow.end_run()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()

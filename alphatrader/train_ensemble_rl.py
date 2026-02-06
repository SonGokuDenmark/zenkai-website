#!/usr/bin/env python3
"""
Train and evaluate ensemble of RL agents for trading.

Trains multiple algorithms (PPO, DQN, A2C) and combines their predictions
via voting ensemble for more robust trading decisions.

Usage:
    # Train all agents
    python train_ensemble_rl.py --symbol BTCUSDT --timesteps 200000

    # Train specific agents only
    python train_ensemble_rl.py --symbol BTCUSDT --agents PPO DQN

    # Evaluate existing ensemble
    python train_ensemble_rl.py --symbol BTCUSDT --eval-only --ensemble-dir checkpoints/ensemble_20260204
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import socket

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import psycopg2
import mlflow

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
    return df


def add_lstm_predictions(
    df: pd.DataFrame,
    lstm_model: LSTMClassifier,
    seq_length: int = 50,
    prefix: str = "lstm",
) -> pd.DataFrame:
    """Add LSTM predictions as columns to the DataFrame."""
    print(f"Adding {prefix} predictions...")

    # Get feature columns
    feature_cols = ["open", "high", "low", "close", "volume"]
    signal_cols = sorted([c for c in df.columns if c.startswith("signal_") or c.startswith("conf_")])
    feature_cols.extend(signal_cols)

    # Add regime one-hot
    regimes = ["TRENDING_UP", "TRENDING_DOWN", "RANGING", "HIGH_VOL"]
    for regime in regimes:
        col_name = f"regime_{regime.lower()}"
        if col_name not in df.columns:
            df[col_name] = (df["regime"] == regime).astype(float)
        feature_cols.append(col_name)

    # Normalize OHLCV
    df_norm = df.copy()
    for col in ["open", "high", "low", "close"]:
        df_norm[col] = (df_norm[col] - df_norm[col].rolling(50, min_periods=1).mean()) / \
                       df_norm[col].rolling(50, min_periods=1).std().clip(lower=1e-8)
    df_norm["volume"] = (df_norm["volume"] - df_norm["volume"].rolling(50, min_periods=1).mean()) / \
                        df_norm["volume"].rolling(50, min_periods=1).std().clip(lower=1e-8)

    df_norm = df_norm.fillna(0).replace([np.inf, -np.inf], 0)

    # Initialize output columns
    n_rows = len(df)
    lstm_pred = np.full(n_rows, np.nan)
    lstm_conf = np.full(n_rows, np.nan)
    lstm_prob_down = np.full(n_rows, np.nan)
    lstm_prob_flat = np.full(n_rows, np.nan)
    lstm_prob_up = np.full(n_rows, np.nan)

    # Batch prediction
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


class EnsembleAgent:
    """
    Voting ensemble of multiple RL agents.

    Combines predictions from multiple trained agents using:
    - Majority voting: Each agent gets 1 vote, majority wins
    - Agreement threshold: Only act if N agents agree (else HOLD)
    """

    def __init__(
        self,
        models: Dict[str, any],
        voting_mode: str = "majority",  # "majority" or "unanimous"
        agreement_threshold: int = 2,    # Min agents that must agree to act
    ):
        """
        Initialize ensemble.

        Args:
            models: Dict of {name: model} pairs
            voting_mode: "majority" (simple vote) or "unanimous" (all must agree)
            agreement_threshold: Minimum agents that must agree for non-HOLD action
        """
        self.models = models
        self.voting_mode = voting_mode
        self.agreement_threshold = agreement_threshold

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[int, Dict]:
        """
        Get ensemble prediction via majority voting.

        Each agent gets exactly 1 vote (no weighting).
        If agreement_threshold not met, defaults to HOLD (0).

        Returns:
            action: Voted action (0=HOLD, 1=BUY, 2=SELL)
            info: Dict with individual predictions and vote counts
        """
        # Each agent gets exactly 1 vote
        votes = {0: 0, 1: 0, 2: 0}  # HOLD, BUY, SELL
        individual_actions = {}

        for name, model in self.models.items():
            action, _ = model.predict(obs, deterministic=deterministic)
            action = int(action)
            individual_actions[name] = action
            votes[action] += 1

        # Find action with most votes
        max_votes = max(votes.values())
        winning_action = max(votes.keys(), key=lambda k: votes[k])

        # Check agreement threshold for non-HOLD actions
        if winning_action != 0 and max_votes < self.agreement_threshold:
            # Not enough agreement for BUY/SELL, default to HOLD
            ensemble_action = 0
            agreement_met = False
        else:
            ensemble_action = winning_action
            agreement_met = True

        # For unanimous mode, all agents must agree
        if self.voting_mode == "unanimous" and max_votes < len(self.models):
            ensemble_action = 0
            agreement_met = False

        info = {
            "individual_actions": individual_actions,
            "vote_counts": votes,
            "winning_votes": max_votes,
            "agreement_met": agreement_met,
            "total_agents": len(self.models),
        }

        return ensemble_action, info

    def save(self, path: Path):
        """Save ensemble models and config."""
        path.mkdir(parents=True, exist_ok=True)

        # Save each model
        for name, model in self.models.items():
            model.save(str(path / f"{name}_model"))

        # Save config
        with open(path / "ensemble_config.json", "w") as f:
            json.dump({
                "model_names": list(self.models.keys()),
                "voting_mode": self.voting_mode,
                "agreement_threshold": self.agreement_threshold,
            }, f, indent=2)

        print(f"Ensemble saved to {path}")

    @classmethod
    def load(cls, path: Path, algorithm_map: Dict[str, type]) -> "EnsembleAgent":
        """Load ensemble from saved directory."""
        with open(path / "ensemble_config.json") as f:
            config = json.load(f)

        models = {}
        for name in config["model_names"]:
            # Determine algorithm class from name
            algo = name.upper()
            if algo in algorithm_map:
                models[name] = algorithm_map[algo].load(str(path / f"{name}_model"))

        return cls(
            models,
            voting_mode=config.get("voting_mode", "majority"),
            agreement_threshold=config.get("agreement_threshold", 2),
        )


def train_single_agent(
    algorithm: str,
    train_env,
    val_env,
    timesteps: int,
    checkpoint_dir: Path,
    run_name: str,
) -> Tuple[any, float]:
    """
    Train a single RL agent.

    Returns:
        (model, validation_score)
    """
    from stable_baselines3 import PPO, A2C, DQN
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor

    algorithm_map = {"PPO": PPO, "A2C": A2C, "DQN": DQN}
    AlgorithmClass = algorithm_map[algorithm]

    # Wrap environments
    train_env_wrapped = Monitor(train_env)
    val_env_wrapped = Monitor(val_env)

    print(f"\n{'=' * 50}")
    print(f"Training {algorithm}")
    print("=" * 50)

    # Algorithm-specific hyperparameters
    if algorithm == "PPO":
        model = AlgorithmClass(
            "MlpPolicy",
            train_env_wrapped,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=0,
        )
    elif algorithm == "A2C":
        model = AlgorithmClass(
            "MlpPolicy",
            train_env_wrapped,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=0.95,
            verbose=0,
        )
    elif algorithm == "DQN":
        model = AlgorithmClass(
            "MlpPolicy",
            train_env_wrapped,
            learning_rate=1e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_final_eps=0.05,
            verbose=0,
        )

    # Callbacks
    model_save_path = checkpoint_dir / f"{algorithm.lower()}_{run_name}"
    eval_callback = EvalCallback(
        val_env_wrapped,
        best_model_save_path=str(model_save_path),
        log_path=str(model_save_path),
        eval_freq=max(timesteps // 20, 1000),
        deterministic=True,
        render=False,
        verbose=0,
    )

    # Train
    model.learn(
        total_timesteps=timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    # Get validation score from eval callback
    val_score = 0.0
    eval_results_path = model_save_path / "evaluations.npz"
    if eval_results_path.exists():
        data = np.load(eval_results_path)
        val_score = float(data["results"].mean())

    print(f"{algorithm} training complete. Validation score: {val_score:.2f}")

    return model, val_score


def evaluate_agent(agent, test_env, name: str = "Agent") -> Dict:
    """Evaluate a single agent or ensemble on test environment."""
    obs, info = test_env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        if hasattr(agent, "models"):
            # Ensemble
            action, _ = agent.predict(obs)
        else:
            # Single agent
            action, _ = agent.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = test_env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    results = {
        "name": name,
        "steps": steps,
        "total_reward": total_reward,
        "total_trades": info["total_trades"],
        "win_rate": info["win_rate"],
        "total_pnl": info["total_pnl"],
        "max_drawdown": info["max_drawdown"],
        "final_balance": info["balance"],
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Train ensemble of RL trading agents")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--timeframe", default="1h", help="Data timeframe")
    parser.add_argument("--limit", type=int, default=50000, help="Max rows to load")
    parser.add_argument("--timesteps", type=int, default=200000, help="Timesteps per agent")
    parser.add_argument("--agents", nargs="+", default=["PPO", "DQN", "A2C"],
                        choices=["PPO", "DQN", "A2C"], help="Agents to train")
    parser.add_argument("--episode-length", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--lstm-model", default=None, help="Path to LSTM model")
    parser.add_argument("--lstm-4h", default=None, help="Path to 4h LSTM model")
    parser.add_argument("--seq-length", type=int, default=50, help="LSTM sequence length")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate existing ensemble")
    parser.add_argument("--ensemble-dir", default=None, help="Directory with saved ensemble")
    parser.add_argument("--worker", default="thomas", help="Worker name for tracking")

    args = parser.parse_args()

    print("=" * 60)
    print("AlphaTrader Ensemble RL Training")
    print("=" * 60)
    print(f"Started: {datetime.now()}")
    print(f"Symbol: {args.symbol}")
    print(f"Agents: {args.agents}")
    print(f"Timesteps per agent: {args.timesteps}")
    print()

    # Setup MLflow
    mlflow_run = None
    try:
        mlflow.set_tracking_uri(_get_mlflow_uri())
        mlflow.set_experiment("alphatrader-ensemble")
        mlflow_run = mlflow.start_run(run_name=f"ensemble-{args.symbol}-{args.worker}")
        mlflow.log_params({
            "symbol": args.symbol,
            "timeframe": args.timeframe,
            "agents": ",".join(args.agents),
            "timesteps": args.timesteps,
            "worker": args.worker,
        })
    except Exception as e:
        print(f"Warning: MLflow setup failed: {e}")

    # Load data
    df = load_trading_data(
        symbol=args.symbol,
        timeframe=args.timeframe,
        limit=args.limit,
    )

    if len(df) < 1000:
        print(f"ERROR: Not enough data ({len(df)} rows)")
        return

    # Add LSTM predictions if specified
    lstm_cols_to_check = []

    if args.lstm_model:
        lstm_model = LSTMClassifier.load(args.lstm_model)
        df = add_lstm_predictions(df, lstm_model, seq_length=args.seq_length, prefix="lstm")
        lstm_cols_to_check.append("lstm_pred")

    if args.lstm_4h:
        lstm_4h_model = LSTMClassifier.load(args.lstm_4h)
        df = add_lstm_predictions(df, lstm_4h_model, seq_length=args.seq_length, prefix="lstm_4h")
        lstm_cols_to_check.append("lstm_4h_pred")

    if lstm_cols_to_check:
        df = df.dropna(subset=lstm_cols_to_check).reset_index(drop=True)

    # Create environments
    print("\nCreating environments...")
    train_env, val_env, test_env = create_train_test_envs(
        df,
        train_ratio=0.7,
        val_ratio=0.15,
        episode_length=args.episode_length,
        initial_balance=10000.0,
        transaction_cost=0.001,
    )

    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path("checkpoints/ensemble") / f"ensemble_{args.symbol}_{timestamp}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only and args.ensemble_dir:
        # Load existing ensemble
        from stable_baselines3 import PPO, A2C, DQN
        algorithm_map = {"PPO": PPO, "A2C": A2C, "DQN": DQN}
        ensemble = EnsembleAgent.load(Path(args.ensemble_dir), algorithm_map)
        trained_models = ensemble.models
        val_scores = {name: 1.0 for name in trained_models}
    else:
        # Train each agent
        trained_models = {}
        val_scores = {}

        for algo in args.agents:
            # Create fresh environments for each agent
            train_env_i, val_env_i, _ = create_train_test_envs(
                df,
                train_ratio=0.7,
                val_ratio=0.15,
                episode_length=args.episode_length,
                initial_balance=10000.0,
                transaction_cost=0.001,
            )

            model, val_score = train_single_agent(
                algo,
                train_env_i,
                val_env_i,
                args.timesteps,
                checkpoint_dir,
                run_name=f"{args.symbol}_{timestamp}",
            )

            trained_models[algo.lower()] = model
            val_scores[algo.lower()] = val_score

    # Create ensemble with MAJORITY VOTING (each agent = 1 vote)
    print("\n" + "=" * 60)
    print("Creating Ensemble (Majority Voting)")
    print("=" * 60)

    # Majority voting: need 2/3 agents to agree for BUY/SELL
    ensemble = EnsembleAgent(
        trained_models,
        voting_mode="majority",
        agreement_threshold=2,  # At least 2 agents must agree
    )
    print(f"Voting mode: majority, agreement threshold: 2/{len(trained_models)}")

    # Save ensemble
    ensemble.save(checkpoint_dir)

    # Evaluate all agents on test set
    print("\n" + "=" * 60)
    print("Evaluation on Test Set")
    print("=" * 60)

    all_results = []

    # Evaluate individual agents
    for name, model in trained_models.items():
        # Reset test env
        test_env_eval = TradingEnv(
            df.iloc[int(len(df) * 0.85):].copy(),
            episode_length=None,  # Full test set
            initial_balance=10000.0,
            transaction_cost=0.001,
        )
        results = evaluate_agent(model, test_env_eval, name.upper())
        all_results.append(results)

        print(f"\n{results['name']}:")
        print(f"  PnL: ${results['total_pnl']:.2f}")
        print(f"  Win Rate: {results['win_rate']:.1%}")
        print(f"  Max Drawdown: {results['max_drawdown']:.1%}")
        print(f"  Trades: {results['total_trades']}")

    # Evaluate ensemble
    test_env_eval = TradingEnv(
        df.iloc[int(len(df) * 0.85):].copy(),
        episode_length=None,
        initial_balance=10000.0,
        transaction_cost=0.001,
    )
    ensemble_results = evaluate_agent(ensemble, test_env_eval, "ENSEMBLE")
    all_results.append(ensemble_results)

    print(f"\n{ensemble_results['name']}:")
    print(f"  PnL: ${ensemble_results['total_pnl']:.2f}")
    print(f"  Win Rate: {ensemble_results['win_rate']:.1%}")
    print(f"  Max Drawdown: {ensemble_results['max_drawdown']:.1%}")
    print(f"  Trades: {ensemble_results['total_trades']}")

    # Summary table
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    print(f"\n{'Agent':<12} {'PnL':>10} {'Win Rate':>10} {'Max DD':>10} {'Trades':>8}")
    print("-" * 55)

    for r in all_results:
        print(f"{r['name']:<12} ${r['total_pnl']:>8.2f} {r['win_rate']:>9.1%} "
              f"{r['max_drawdown']:>9.1%} {r['total_trades']:>8}")

    # Log to MLflow
    try:
        for r in all_results:
            mlflow.log_metrics({
                f"{r['name'].lower()}_pnl": r["total_pnl"],
                f"{r['name'].lower()}_win_rate": r["win_rate"],
                f"{r['name'].lower()}_max_dd": r["max_drawdown"],
                f"{r['name'].lower()}_trades": r["total_trades"],
            })
    except Exception as e:
        print(f"Warning: MLflow logging failed: {e}")
    finally:
        if mlflow_run:
            mlflow.end_run()

    # Save results
    results_file = checkpoint_dir / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "symbol": args.symbol,
            "timeframe": args.timeframe,
            "timesteps": args.timesteps,
            "agents": args.agents,
            "weights": ensemble.weights,
            "results": all_results,
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"Ensemble saved to: {checkpoint_dir}")
    print(f"\nFinished: {datetime.now()}")


if __name__ == "__main__":
    main()

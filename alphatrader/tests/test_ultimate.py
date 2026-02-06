#!/usr/bin/env python3
"""
ULTIMATE COMPREHENSIVE TEST - Leave No Stone Unturned

This is the FULL test suite that covers EVERYTHING:

TIMEFRAMES: 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d
SYMBOLS: Top 20 by volume
ALGORITHMS: DQN, PPO, A2C
TRAINING: 100k, 300k, 500k, 1M, 2M timesteps
LSTM: With/Without, different seq lengths (25, 50, 100)
FILTERS: All regime combinations
EPISODE LENGTHS: 250, 500, 1000
LEARNING RATES: 1e-5, 5e-5, 1e-4, 5e-4, 1e-3
TRANSACTION COSTS: 0.05%, 0.1%, 0.15%, 0.2%
VALIDATION: Standard split + Walk-forward

SAFETY FEATURES:
- GPU memory monitoring & auto-cleanup
- CPU/RAM monitoring
- Cooling breaks between phases
- Auto-throttle if resources critical
- Checkpoint/resume support

Estimated time: 24-48 hours (run over weekend)

Usage:
    python test_ultimate.py                    # Full suite
    python test_ultimate.py --phase 1          # Phase 1 only (quick scan)
    python test_ultimate.py --phase 2          # Phase 2 (validation)
    python test_ultimate.py --resume           # Resume from checkpoint
    python test_ultimate.py --safe             # Safe mode (extra cooling breaks)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import psycopg2
import json
import pickle
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from itertools import product
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.env import TradingEnv, create_train_test_envs
from src.models.lstm_classifier import LSTMClassifier
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.monitor import Monitor

# ============================================================
# SAFETY & RESOURCE MANAGEMENT
# ============================================================

# Safety thresholds
SAFETY_CONFIG = {
    "gpu_memory_threshold": 0.85,      # Pause if GPU memory > 85%
    "ram_threshold": 0.90,             # Pause if RAM > 90%
    "gpu_temp_threshold": 80,          # Pause if GPU temp > 80C
    "cooling_break_seconds": 30,       # Break between phases
    "mini_break_seconds": 5,           # Break between tests
    "critical_break_seconds": 120,     # Break if resources critical
    "checkpoint_frequency": 10,        # Save checkpoint every N tests
}


def get_gpu_stats() -> Dict:
    """Get GPU memory and temperature stats."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            mem_used = int(parts[0])
            mem_total = int(parts[1])
            temp = int(parts[2])
            return {
                "memory_used_mb": mem_used,
                "memory_total_mb": mem_total,
                "memory_percent": mem_used / mem_total,
                "temperature": temp,
                "available": True,
            }
    except:
        pass
    return {"available": False}


def get_ram_stats() -> Dict:
    """Get RAM usage stats."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "used_gb": mem.used / (1024**3),
            "total_gb": mem.total / (1024**3),
            "percent": mem.percent / 100,
            "available": True,
        }
    except:
        pass
    return {"available": False}


def clear_gpu_memory():
    """Clear GPU memory cache."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import gc
            gc.collect()
            print("    [CLEANUP] GPU memory cleared")
    except:
        pass


def check_resources_and_wait(safe_mode: bool = False) -> bool:
    """
    Check system resources and wait if necessary.
    Returns True if OK to continue, False if should abort.
    """
    gpu = get_gpu_stats()
    ram = get_ram_stats()

    warnings = []

    if gpu.get("available"):
        if gpu["memory_percent"] > SAFETY_CONFIG["gpu_memory_threshold"]:
            warnings.append(f"GPU memory high: {gpu['memory_percent']*100:.1f}%")
        if gpu["temperature"] > SAFETY_CONFIG["gpu_temp_threshold"]:
            warnings.append(f"GPU temp high: {gpu['temperature']}C")

    if ram.get("available"):
        if ram["percent"] > SAFETY_CONFIG["ram_threshold"]:
            warnings.append(f"RAM high: {ram['percent']*100:.1f}%")

    if warnings:
        print(f"\n    [WARNING] Resource alert: {', '.join(warnings)}")
        clear_gpu_memory()

        # Wait for resources to recover
        wait_time = SAFETY_CONFIG["critical_break_seconds"] if safe_mode else SAFETY_CONFIG["cooling_break_seconds"]
        print(f"    [COOLING] Waiting {wait_time}s for resources to recover...")
        time.sleep(wait_time)

        # Check again
        gpu = get_gpu_stats()
        if gpu.get("available") and gpu["temperature"] > 85:
            print("    [CRITICAL] GPU still too hot. Taking extended break...")
            time.sleep(SAFETY_CONFIG["critical_break_seconds"])

    return True


def print_system_status():
    """Print current system status."""
    gpu = get_gpu_stats()
    ram = get_ram_stats()

    status = "    [STATUS] "
    if gpu.get("available"):
        status += f"GPU: {gpu['memory_percent']*100:.0f}% mem, {gpu['temperature']}C | "
    if ram.get("available"):
        status += f"RAM: {ram['percent']*100:.0f}%"

    print(status)


def cooling_break(phase_name: str, safe_mode: bool = False):
    """Take a cooling break between phases."""
    wait_time = SAFETY_CONFIG["cooling_break_seconds"]
    if safe_mode:
        wait_time *= 2

    print(f"\n{'=' * 70}")
    print(f"[COOLING BREAK] Phase {phase_name} complete")
    print_system_status()
    clear_gpu_memory()
    print(f"Cooling for {wait_time}s before next phase...")
    print("=" * 70)

    time.sleep(wait_time)


def mini_break(test_num: int, safe_mode: bool = False):
    """Small break between tests."""
    # Only take mini breaks every few tests
    if test_num % 5 != 0:
        return

    clear_gpu_memory()

    if safe_mode:
        time.sleep(SAFETY_CONFIG["mini_break_seconds"])

# ============================================================
# EXHAUSTIVE CONFIGURATIONS
# ============================================================

# ALL timeframes available
ALL_TIMEFRAMES = ["5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]

# Top 20 symbols by volume (majors, L1s, L2s, meme, defi)
ALL_SYMBOLS = [
    # Majors
    "BTCUSDT", "ETHUSDT",
    # L1 Chains
    "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "TRXUSDT", "LINKUSDT",
    # L2 / DeFi
    "MATICUSDT", "UNIUSDT", "AAVEUSDT",
    # Meme coins (high volatility)
    "DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "BONKUSDT", "FLOKIUSDT", "WIFUSDT",
    # Stablecoin pair (sanity check - should NOT be profitable)
    # "USDCUSDT",  # Uncomment for sanity check
]

# All algorithms
ALL_ALGORITHMS = {
    "DQN": DQN,
    "PPO": PPO,
    "A2C": A2C,
}

# Training levels (timesteps)
ALL_TRAINING_LEVELS = {
    "tiny": 50000,       # 50k - ultra quick
    "short": 100000,     # 100k - quick scan
    "medium": 300000,    # 300k - balanced
    "long": 500000,      # 500k - thorough
    "extended": 1000000, # 1M - deep training
    "maximum": 2000000,  # 2M - maximum (overnight per test)
}

# Regime filters
ALL_REGIME_FILTERS = {
    "none": None,
    "no_high_vol": ["HIGH_VOL"],
    "no_ranging": ["RANGING"],
    "trending_only": ["RANGING", "HIGH_VOL"],
    "trending_up_only": ["TRENDING_DOWN", "RANGING", "HIGH_VOL"],
    "trending_down_only": ["TRENDING_UP", "RANGING", "HIGH_VOL"],
}

# Episode lengths
ALL_EPISODE_LENGTHS = [250, 500, 750, 1000]

# Learning rates
ALL_LEARNING_RATES = {
    "DQN": [1e-5, 5e-5, 1e-4, 5e-4],
    "PPO": [1e-5, 5e-5, 1e-4, 3e-4, 5e-4],
    "A2C": [1e-4, 3e-4, 5e-4, 7e-4, 1e-3],
}

# Transaction costs
ALL_TRANSACTION_COSTS = [0.0005, 0.001, 0.0015, 0.002]  # 0.05% to 0.2%

# LSTM sequence lengths
ALL_SEQ_LENGTHS = [25, 50, 75, 100]

# ============================================================
# TEST PHASES (divide and conquer)
# ============================================================

TEST_PHASES = {
    # Phase 1: Quick scan to find promising configs
    "phase1_scan": {
        "description": "Quick scan - find promising timeframes and symbols",
        "training": ["short"],
        "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "timeframes": ALL_TIMEFRAMES,
        "algorithms": ["DQN"],
        "filters": ["none", "no_high_vol"],
        "episode_lengths": [500],
        "learning_rates": {"DQN": [1e-4]},
        "transaction_costs": [0.001],
        "use_lstm": [False, True],
        "seq_lengths": [50],
        "runs": 1,
    },

    # Phase 2: Validate winners from phase 1
    "phase2_validate": {
        "description": "Validate top performers with more runs",
        "training": ["medium"],
        "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "timeframes": ["30m", "1h", "4h"],  # Best from typical results
        "algorithms": ["DQN", "PPO"],
        "filters": ["none", "no_high_vol", "trending_only"],
        "episode_lengths": [500],
        "learning_rates": {"DQN": [1e-4], "PPO": [3e-4]},
        "transaction_costs": [0.001],
        "use_lstm": [True],
        "seq_lengths": [50],
        "runs": 3,
    },

    # Phase 3: Extended symbols
    "phase3_symbols": {
        "description": "Test best config on all 20 symbols",
        "training": ["medium"],
        "symbols": ALL_SYMBOLS,
        "timeframes": ["4h"],
        "algorithms": ["DQN"],
        "filters": ["no_high_vol"],
        "episode_lengths": [500],
        "learning_rates": {"DQN": [1e-4]},
        "transaction_costs": [0.001],
        "use_lstm": [True],
        "seq_lengths": [50],
        "runs": 2,
    },

    # Phase 4: Algorithm comparison
    "phase4_algorithms": {
        "description": "Deep algorithm comparison",
        "training": ["medium", "long"],
        "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "timeframes": ["4h"],
        "algorithms": ["DQN", "PPO", "A2C"],
        "filters": ["no_high_vol"],
        "episode_lengths": [500],
        "learning_rates": ALL_LEARNING_RATES,
        "transaction_costs": [0.001],
        "use_lstm": [True],
        "seq_lengths": [50],
        "runs": 3,
    },

    # Phase 5: Hyperparameter sensitivity
    "phase5_hyperparams": {
        "description": "Test sensitivity to hyperparameters",
        "training": ["medium"],
        "symbols": ["BTCUSDT"],
        "timeframes": ["4h"],
        "algorithms": ["DQN"],
        "filters": ["no_high_vol"],
        "episode_lengths": ALL_EPISODE_LENGTHS,
        "learning_rates": {"DQN": ALL_LEARNING_RATES["DQN"]},
        "transaction_costs": ALL_TRANSACTION_COSTS,
        "use_lstm": [True],
        "seq_lengths": ALL_SEQ_LENGTHS,
        "runs": 2,
    },

    # Phase 6: Long training validation
    "phase6_long_training": {
        "description": "Extended training to check for overfit",
        "training": ["short", "medium", "long", "extended"],
        "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "timeframes": ["4h"],
        "algorithms": ["DQN", "PPO"],
        "filters": ["no_high_vol"],
        "episode_lengths": [500],
        "learning_rates": {"DQN": [1e-4], "PPO": [3e-4]},
        "transaction_costs": [0.001],
        "use_lstm": [True],
        "seq_lengths": [50],
        "runs": 3,
    },

    # Phase 7: Maximum training (overnight)
    "phase7_maximum": {
        "description": "Maximum training for final validation",
        "training": ["maximum"],
        "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "timeframes": ["4h"],
        "algorithms": ["DQN"],
        "filters": ["no_high_vol"],
        "episode_lengths": [500],
        "learning_rates": {"DQN": [1e-4]},
        "transaction_costs": [0.001],
        "use_lstm": [True],
        "seq_lengths": [50],
        "runs": 5,
    },

    # Phase 8: Walk-forward validation
    "phase8_walkforward": {
        "description": "Walk-forward out-of-sample validation",
        "training": ["medium"],
        "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "timeframes": ["4h"],
        "algorithms": ["DQN"],
        "filters": ["no_high_vol"],
        "episode_lengths": [500],
        "learning_rates": {"DQN": [1e-4]},
        "transaction_costs": [0.001],
        "use_lstm": [True],
        "seq_lengths": [50],
        "runs": 1,
        "walk_forward": True,
        "wf_train_months": 6,
        "wf_test_months": 1,
        "wf_folds": 6,
    },
}

# Checkpoint file for resume
CHECKPOINT_FILE = "ultimate_test_checkpoint.pkl"


def get_data(symbol: str, timeframe: str, limit: int = 50000) -> Optional[pd.DataFrame]:
    """Load data from database."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("ZENKAI_DB_HOST", "192.168.0.160"),
            database=os.getenv("ZENKAI_DB_NAME", "zenkai_data"),
            user=os.getenv("ZENKAI_DB_USER", "zenkai"),
            password=os.getenv("ZENKAI_DB_PASSWORD"),
            connect_timeout=10,
        )
    except:
        # Try Tailscale
        conn = psycopg2.connect(
            host=os.getenv("ZENKAI_DB_HOST_TAILSCALE", "100.110.101.78"),
            database=os.getenv("ZENKAI_DB_NAME", "zenkai_data"),
            user=os.getenv("ZENKAI_DB_USER", "zenkai"),
            password=os.getenv("ZENKAI_DB_PASSWORD"),
            connect_timeout=10,
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

    if len(df) < 500:
        return None

    df = df.sort_values('open_time').reset_index(drop=True)
    for col in signal_cols:
        df[col] = df[col].fillna(0)

    return df


def add_lstm_predictions(
    df: pd.DataFrame,
    lstm_model: LSTMClassifier,
    seq_length: int = 50
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


def apply_regime_filter(df: pd.DataFrame, filter_config) -> pd.DataFrame:
    """Apply regime filter."""
    if filter_config is None:
        return df
    if isinstance(filter_config, list):
        return df[~df['regime'].isin(filter_config)].reset_index(drop=True)
    return df


def train_and_test(
    df: pd.DataFrame,
    algorithm: str,
    timesteps: int,
    learning_rate: float,
    episode_length: int,
    transaction_cost: float,
) -> Dict:
    """Train and evaluate model."""
    AlgorithmClass = ALL_ALGORITHMS[algorithm]

    train_env, val_env, test_env = create_train_test_envs(
        df,
        train_ratio=0.7,
        val_ratio=0.15,
        episode_length=episode_length,
        initial_balance=10000.0,
        transaction_cost=transaction_cost,
    )

    train_env = Monitor(train_env)

    model = AlgorithmClass("MlpPolicy", train_env, learning_rate=learning_rate, verbose=0)
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
        "max_dd": info["max_drawdown"],
        "final_balance": info["balance"],
    }


def walk_forward_test(
    df: pd.DataFrame,
    algorithm: str,
    timesteps: int,
    learning_rate: float,
    episode_length: int,
    transaction_cost: float,
    train_months: int,
    test_months: int,
    n_folds: int,
) -> List[Dict]:
    """Walk-forward validation."""
    results = []

    # Calculate rows per month (approximate)
    total_rows = len(df)
    rows_per_month = total_rows // 12  # Assume ~12 months of data

    train_rows = train_months * rows_per_month
    test_rows = test_months * rows_per_month
    step_rows = test_rows

    for fold in range(n_folds):
        start_idx = fold * step_rows
        train_end = start_idx + train_rows
        test_end = train_end + test_rows

        if test_end > total_rows:
            break

        train_df = df.iloc[start_idx:train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()

        if len(train_df) < 500 or len(test_df) < 100:
            continue

        # Train
        AlgorithmClass = ALL_ALGORITHMS[algorithm]
        train_env = TradingEnv(
            train_df,
            episode_length=episode_length,
            initial_balance=10000.0,
            transaction_cost=transaction_cost,
        )
        train_env = Monitor(train_env)

        model = AlgorithmClass("MlpPolicy", train_env, learning_rate=learning_rate, verbose=0)
        model.learn(total_timesteps=timesteps, progress_bar=False)

        # Test
        test_env = TradingEnv(
            test_df,
            episode_length=None,
            initial_balance=10000.0,
            transaction_cost=transaction_cost,
        )

        obs, info = test_env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated

        results.append({
            "fold": fold + 1,
            "pnl": info["total_pnl"],
            "win_rate": info["win_rate"],
            "trades": info["total_trades"],
        })

    return results


def save_checkpoint(state: Dict):
    """Save checkpoint for resume."""
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(state, f)


def load_checkpoint() -> Optional[Dict]:
    """Load checkpoint if exists."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'rb') as f:
            return pickle.load(f)
    return None


def run_phase(
    phase_name: str,
    config: Dict,
    lstm_models: Dict,
    completed_tests: set,
    safe_mode: bool = False,
) -> Tuple[List[Dict], set]:
    """Run a single test phase with safety monitoring."""
    results = []

    print(f"\n{'=' * 70}")
    print(f"PHASE: {phase_name}")
    print(f"Description: {config['description']}")
    if safe_mode:
        print("[SAFE MODE ENABLED]")
    print("=" * 70)
    print_system_status()

    # Check for walk-forward
    is_walk_forward = config.get("walk_forward", False)

    # Generate combinations
    combos = []
    for training in config["training"]:
        for symbol in config["symbols"]:
            for tf in config["timeframes"]:
                for algo in config["algorithms"]:
                    for filt_name, filt_val in [(k, ALL_REGIME_FILTERS[k]) for k in config["filters"]]:
                        for ep_len in config["episode_lengths"]:
                            for lr in config["learning_rates"].get(algo, [1e-4]):
                                for tc in config["transaction_costs"]:
                                    for use_lstm in config["use_lstm"]:
                                        for seq_len in config["seq_lengths"]:
                                            combos.append({
                                                "training": training,
                                                "symbol": symbol,
                                                "timeframe": tf,
                                                "algorithm": algo,
                                                "filter_name": filt_name,
                                                "filter_val": filt_val,
                                                "episode_length": ep_len,
                                                "learning_rate": lr,
                                                "transaction_cost": tc,
                                                "use_lstm": use_lstm,
                                                "seq_length": seq_len,
                                            })

    total = len(combos) * config["runs"]
    current = 0

    print(f"Total test combinations: {len(combos)} x {config['runs']} runs = {total} tests")

    for combo in combos:
        test_id = (
            f"{combo['symbol']}_{combo['timeframe']}_{combo['algorithm']}_"
            f"{combo['filter_name']}_lstm{combo['use_lstm']}_seq{combo['seq_length']}_"
            f"ep{combo['episode_length']}_lr{combo['learning_rate']}_tc{combo['transaction_cost']}_"
            f"{combo['training']}"
        )

        # Skip if already completed
        if test_id in completed_tests:
            print(f"  [SKIP] {test_id} (already completed)")
            current += config["runs"]
            continue

        print(f"\n  [{current+1}/{total}] {test_id}")

        # Load data
        df = get_data(combo["symbol"], combo["timeframe"])
        if df is None:
            print(f"    Skipped: insufficient data")
            results.append({"test_id": test_id, "phase": phase_name, "error": "insufficient_data"})
            current += config["runs"]
            completed_tests.add(test_id)
            continue

        # Add LSTM
        if combo["use_lstm"]:
            lstm = lstm_models.get(combo["timeframe"])
            if lstm:
                df = add_lstm_predictions(df, lstm, combo["seq_length"])
                df = df.dropna(subset=["lstm_pred"]).reset_index(drop=True)
            else:
                print(f"    Warning: No LSTM for {combo['timeframe']}")

        # Apply filter
        df = apply_regime_filter(df, combo["filter_val"])

        if len(df) < 500:
            print(f"    Skipped: insufficient data after filter ({len(df)} rows)")
            results.append({"test_id": test_id, "phase": phase_name, "error": "insufficient_data_filtered"})
            current += config["runs"]
            completed_tests.add(test_id)
            continue

        print(f"    Data: {len(df)} rows")

        timesteps = ALL_TRAINING_LEVELS[combo["training"]]

        if is_walk_forward:
            # Walk-forward validation
            print(f"    Walk-forward: {config['wf_folds']} folds")
            wf_results = walk_forward_test(
                df, combo["algorithm"], timesteps, combo["learning_rate"],
                combo["episode_length"], combo["transaction_cost"],
                config["wf_train_months"], config["wf_test_months"], config["wf_folds"]
            )

            if wf_results:
                avg_pnl = np.mean([r["pnl"] for r in wf_results])
                print(f"    Avg PnL across folds: ${avg_pnl:.2f}")

                results.append({
                    "test_id": test_id,
                    "phase": phase_name,
                    **combo,
                    "timesteps": timesteps,
                    "walk_forward": True,
                    "fold_results": wf_results,
                    "avg_pnl": avg_pnl,
                    "profitable_folds": sum(1 for r in wf_results if r["pnl"] > 0),
                    "total_folds": len(wf_results),
                })
            current += 1
        else:
            # Standard runs
            pnls = []
            for run in range(config["runs"]):
                current += 1
                print(f"    Run {run+1}/{config['runs']}...", end=" ", flush=True)

                result = train_and_test(
                    df, combo["algorithm"], timesteps, combo["learning_rate"],
                    combo["episode_length"], combo["transaction_cost"]
                )
                pnls.append(result["pnl"])

                status = "[WIN]" if result["pnl"] > 0 else "[LOSS]"
                print(f"PnL: ${result['pnl']:>8.2f} {status}")

            results.append({
                "test_id": test_id,
                "phase": phase_name,
                **combo,
                "timesteps": timesteps,
                "runs": config["runs"],
                "avg_pnl": np.mean(pnls),
                "std_pnl": np.std(pnls) if len(pnls) > 1 else 0,
                "min_pnl": min(pnls),
                "max_pnl": max(pnls),
                "profitable_runs": sum(1 for p in pnls if p > 0),
                "all_pnls": pnls,
            })

        completed_tests.add(test_id)

        # Safety: mini break and resource check
        mini_break(current, safe_mode)
        check_resources_and_wait(safe_mode)

        # Save checkpoint periodically
        if len(completed_tests) % SAFETY_CONFIG["checkpoint_frequency"] == 0:
            save_checkpoint({
                "phase": phase_name,
                "completed_tests": completed_tests,
                "results": results,
            })
            print(f"    [CHECKPOINT] Progress saved ({len(completed_tests)} tests)")

    # End of phase cleanup
    clear_gpu_memory()

    return results, completed_tests


def analyze_all_results(all_results: List[Dict]) -> Dict:
    """Comprehensive analysis of all results."""
    analysis = {
        "total_tests": len(all_results),
        "successful_tests": sum(1 for r in all_results if "error" not in r),
        "profitable_tests": sum(1 for r in all_results if r.get("avg_pnl", 0) > 0),

        "by_phase": {},
        "by_timeframe": {},
        "by_symbol": {},
        "by_algorithm": {},
        "by_filter": {},
        "by_training_level": {},
        "by_lstm": {},

        "best_configs": [],
        "worst_configs": [],
        "overfit_warnings": [],
        "insights": [],
    }

    # Aggregate by category
    for r in all_results:
        if "error" in r:
            continue

        pnl = r.get("avg_pnl", 0)

        # By phase
        phase = r.get("phase", "unknown")
        if phase not in analysis["by_phase"]:
            analysis["by_phase"][phase] = {"pnls": [], "count": 0, "profitable": 0}
        analysis["by_phase"][phase]["pnls"].append(pnl)
        analysis["by_phase"][phase]["count"] += 1
        if pnl > 0:
            analysis["by_phase"][phase]["profitable"] += 1

        # Similar for other categories...
        for key, field in [
            ("by_timeframe", "timeframe"),
            ("by_symbol", "symbol"),
            ("by_algorithm", "algorithm"),
            ("by_filter", "filter_name"),
            ("by_training_level", "training"),
        ]:
            val = r.get(field, "unknown")
            if val not in analysis[key]:
                analysis[key][val] = {"pnls": [], "count": 0, "profitable": 0}
            analysis[key][val]["pnls"].append(pnl)
            analysis[key][val]["count"] += 1
            if pnl > 0:
                analysis[key][val]["profitable"] += 1

        # By LSTM
        lstm_key = "with_lstm" if r.get("use_lstm", False) else "no_lstm"
        if lstm_key not in analysis["by_lstm"]:
            analysis["by_lstm"][lstm_key] = {"pnls": [], "count": 0, "profitable": 0}
        analysis["by_lstm"][lstm_key]["pnls"].append(pnl)
        analysis["by_lstm"][lstm_key]["count"] += 1
        if pnl > 0:
            analysis["by_lstm"][lstm_key]["profitable"] += 1

    # Calculate averages
    for category in ["by_phase", "by_timeframe", "by_symbol", "by_algorithm", "by_filter", "by_training_level", "by_lstm"]:
        for key, data in analysis[category].items():
            if data["pnls"]:
                data["avg_pnl"] = np.mean(data["pnls"])
                data["std_pnl"] = np.std(data["pnls"])
                data["profit_rate"] = data["profitable"] / data["count"] * 100
            del data["pnls"]

    # Find best/worst configs
    valid_results = [r for r in all_results if "error" not in r and "avg_pnl" in r]
    sorted_results = sorted(valid_results, key=lambda x: x["avg_pnl"], reverse=True)

    analysis["best_configs"] = sorted_results[:10]
    analysis["worst_configs"] = sorted_results[-10:]

    # Overfit detection
    training_levels = analysis["by_training_level"]
    if "short" in training_levels and "long" in training_levels:
        short_pnl = training_levels["short"].get("avg_pnl", 0)
        long_pnl = training_levels["long"].get("avg_pnl", 0)
        if short_pnl > long_pnl * 1.5 and short_pnl > 500:
            analysis["overfit_warnings"].append(
                f"Short training ({short_pnl:.0f}) >> Long training ({long_pnl:.0f}) - possible noise fitting"
            )

    # Generate insights
    if analysis["by_timeframe"]:
        best_tf = max(analysis["by_timeframe"].items(), key=lambda x: x[1].get("avg_pnl", 0))
        analysis["insights"].append(f"Best timeframe: {best_tf[0]} (avg PnL: ${best_tf[1]['avg_pnl']:.2f})")

    if analysis["by_filter"]:
        best_filter = max(analysis["by_filter"].items(), key=lambda x: x[1].get("avg_pnl", 0))
        analysis["insights"].append(f"Best filter: {best_filter[0]} (avg PnL: ${best_filter[1]['avg_pnl']:.2f})")

    if analysis["by_lstm"]:
        lstm_data = analysis["by_lstm"]
        if "with_lstm" in lstm_data and "no_lstm" in lstm_data:
            lstm_diff = lstm_data["with_lstm"].get("avg_pnl", 0) - lstm_data["no_lstm"].get("avg_pnl", 0)
            if abs(lstm_diff) > 100:
                better = "LSTM" if lstm_diff > 0 else "No LSTM"
                analysis["insights"].append(f"{better} performs better by ${abs(lstm_diff):.2f}")

    return analysis


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ultimate Comprehensive Test")
    parser.add_argument("--phase", type=int, help="Run specific phase (1-8)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--list", action="store_true", help="List all phases")
    parser.add_argument("--safe", action="store_true", help="Safe mode (extra cooling breaks, slower)")
    parser.add_argument("--status", action="store_true", help="Show system status and exit")
    args = parser.parse_args()

    if args.status:
        print("\n" + "=" * 70)
        print("SYSTEM STATUS")
        print("=" * 70)
        gpu = get_gpu_stats()
        ram = get_ram_stats()
        if gpu.get("available"):
            print(f"GPU Memory: {gpu['memory_used_mb']}MB / {gpu['memory_total_mb']}MB ({gpu['memory_percent']*100:.1f}%)")
            print(f"GPU Temp:   {gpu['temperature']}C")
        if ram.get("available"):
            print(f"RAM:        {ram['used_gb']:.1f}GB / {ram['total_gb']:.1f}GB ({ram['percent']*100:.1f}%)")
        print("=" * 70)
        return

    if args.list:
        print("\n" + "=" * 70)
        print("AVAILABLE PHASES")
        print("=" * 70)
        for name, config in TEST_PHASES.items():
            print(f"\n  {name}")
            print(f"    {config['description']}")
        return

    print("=" * 70)
    print("ULTIMATE COMPREHENSIVE TEST")
    print("Leave No Stone Unturned")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print()

    # Load LSTM models
    lstm_models = {}
    for tf in ALL_TIMEFRAMES:
        try:
            lstm_models[tf] = LSTMClassifier.load(f"checkpoints/lstm_{tf}_mtf")
            print(f"Loaded LSTM for {tf}")
        except:
            pass

    print()

    # Resume or start fresh
    completed_tests = set()
    all_results = []

    if args.resume:
        checkpoint = load_checkpoint()
        if checkpoint:
            completed_tests = checkpoint.get("completed_tests", set())
            all_results = checkpoint.get("results", [])
            print(f"Resumed from checkpoint: {len(completed_tests)} tests completed")

    # Determine phases to run
    if args.phase:
        phase_names = [f"phase{args.phase}_" + list(TEST_PHASES.keys())[args.phase - 1].split("_", 1)[1]]
    else:
        phase_names = list(TEST_PHASES.keys())

    # Initial system check
    print("\n" + "=" * 70)
    print("SYSTEM CHECK")
    print("=" * 70)
    print_system_status()

    if args.safe:
        print("\n[SAFE MODE] Extra cooling breaks enabled")
        print("Tests will run slower but safer for your PC")

    # Run phases
    for i, phase_name in enumerate(phase_names):
        if phase_name not in TEST_PHASES:
            print(f"Unknown phase: {phase_name}")
            continue

        # Resource check before starting phase
        check_resources_and_wait(args.safe)

        config = TEST_PHASES[phase_name]
        results, completed_tests = run_phase(phase_name, config, lstm_models, completed_tests, safe_mode=args.safe)
        all_results.extend(results)

        # Cooling break between phases (not after last one)
        if i < len(phase_names) - 1:
            cooling_break(phase_name, safe_mode=args.safe)

    # Final analysis
    print("\n" + "=" * 70)
    print("FINAL ANALYSIS")
    print("=" * 70)

    analysis = analyze_all_results(all_results)

    print(f"\nTotal tests: {analysis['total_tests']}")
    print(f"Successful: {analysis['successful_tests']}")
    print(f"Profitable: {analysis['profitable_tests']} ({analysis['profitable_tests']/max(analysis['successful_tests'],1)*100:.0f}%)")

    print("\nBy Timeframe:")
    for tf, data in sorted(analysis["by_timeframe"].items(), key=lambda x: x[1].get("avg_pnl", 0), reverse=True):
        print(f"  {tf:6} Avg: ${data.get('avg_pnl', 0):>8.2f}  Win: {data.get('profit_rate', 0):.0f}%  ({data['count']} tests)")

    print("\nBy Algorithm:")
    for algo, data in sorted(analysis["by_algorithm"].items(), key=lambda x: x[1].get("avg_pnl", 0), reverse=True):
        print(f"  {algo:6} Avg: ${data.get('avg_pnl', 0):>8.2f}  Win: {data.get('profit_rate', 0):.0f}%  ({data['count']} tests)")

    print("\nBy Training Level:")
    for train, data in sorted(analysis["by_training_level"].items(), key=lambda x: ALL_TRAINING_LEVELS.get(x[0], 0)):
        print(f"  {train:10} Avg: ${data.get('avg_pnl', 0):>8.2f}  Win: {data.get('profit_rate', 0):.0f}%  ({data['count']} tests)")

    print("\nInsights:")
    for insight in analysis["insights"]:
        print(f"  - {insight}")

    print("\nOverfit Warnings:")
    if analysis["overfit_warnings"]:
        for warning in analysis["overfit_warnings"]:
            print(f"  [!] {warning}")
    else:
        print("  No major warnings")

    print("\nTop 5 Configs:")
    for i, config in enumerate(analysis["best_configs"][:5], 1):
        print(f"  {i}. {config['test_id'][:50]}... Avg: ${config['avg_pnl']:.2f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"ultimate_results_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": all_results,
            "analysis": analysis,
        }, f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")
    print(f"Finished: {datetime.now()}")

    # Cleanup checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)


if __name__ == "__main__":
    main()

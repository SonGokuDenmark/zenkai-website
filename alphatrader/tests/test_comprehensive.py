#!/usr/bin/env python3
"""
COMPREHENSIVE TEST SUITE - Every Corner Stone Turned

Tests across:
- SHORT (100k), MEDIUM (300k), LONG (500k) training
- Multiple timeframes (15m, 30m, 1h, 4h)
- Multiple algorithms (DQN, PPO, A2C)
- Multiple symbols (BTC, ETH, SOL + alts)
- With/without LSTM
- With/without regime filters
- Parameter variations

Goal: Find overfit vs robust strategies
"""

import os
import sys
import numpy as np
import pandas as pd
import psycopg2
import json
from datetime import datetime
from typing import Dict, Optional, List
from dotenv import load_dotenv
load_dotenv()
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.env import TradingEnv, create_train_test_envs
from src.models.lstm_classifier import LSTMClassifier
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.monitor import Monitor

# ============================================================
# TEST CONFIGURATIONS
# ============================================================

# Training durations
TRAINING_LEVELS = {
    "short": 100000,    # 100k - quick tests
    "medium": 300000,   # 300k - balanced
    "long": 500000,     # 500k - thorough
}

# Timeframes to test
TIMEFRAMES = ["15m", "30m", "1h", "4h"]

# Algorithms
ALGORITHMS = {
    "DQN": DQN,
    "PPO": PPO,
    "A2C": A2C,
}

# Symbols (core + alts)
SYMBOLS_CORE = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
SYMBOLS_ALTS = ["DOGEUSDT", "PEPEUSDT", "AVAXUSDT", "DOTUSDT"]

# Regime filters
REGIME_FILTERS = {
    "none": None,
    "no_high_vol": "HIGH_VOL",
    "trending_only": ["RANGING", "HIGH_VOL"],  # Remove these
    "trending_up_only": ["TRENDING_DOWN", "RANGING", "HIGH_VOL"],
}

# Test matrix (what combinations to run)
TEST_MATRIX = {
    # Phase 1: Quick scan (short training, core symbols, all TFs)
    "phase1_scan": {
        "training": ["short"],
        "symbols": SYMBOLS_CORE,
        "timeframes": TIMEFRAMES,
        "algorithms": ["DQN"],
        "filters": ["none", "no_high_vol"],
        "use_lstm": [True, False],
        "runs": 1,
    },
    # Phase 2: Deep dive on winners (medium training)
    "phase2_validate": {
        "training": ["medium"],
        "symbols": SYMBOLS_CORE,
        "timeframes": ["4h"],  # Best from phase 1
        "algorithms": ["DQN", "PPO"],
        "filters": ["no_high_vol"],
        "use_lstm": [True],
        "runs": 3,
    },
    # Phase 3: Extended symbols (medium training)
    "phase3_extend": {
        "training": ["medium"],
        "symbols": SYMBOLS_CORE + SYMBOLS_ALTS,
        "timeframes": ["4h"],
        "algorithms": ["DQN"],
        "filters": ["no_high_vol"],
        "use_lstm": [True],
        "runs": 2,
    },
    # Phase 4: Long training validation (best config only)
    "phase4_long": {
        "training": ["long"],
        "symbols": SYMBOLS_CORE,
        "timeframes": ["4h"],
        "algorithms": ["DQN", "PPO"],
        "filters": ["no_high_vol"],
        "use_lstm": [True],
        "runs": 5,
    },
}


def get_data(symbol: str, timeframe: str, limit: int = 20000) -> Optional[pd.DataFrame]:
    """Load data from DB."""
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


def apply_regime_filter(df: pd.DataFrame, filter_name: str) -> pd.DataFrame:
    """Apply regime filter."""
    filter_config = REGIME_FILTERS.get(filter_name)

    if filter_config is None:
        return df

    if isinstance(filter_config, str):
        return df[df['regime'] != filter_config].reset_index(drop=True)
    elif isinstance(filter_config, list):
        return df[~df['regime'].isin(filter_config)].reset_index(drop=True)

    return df


def train_and_test(
    df: pd.DataFrame,
    algorithm: str,
    timesteps: int,
) -> Dict:
    """Train and evaluate a model."""
    AlgorithmClass = ALGORITHMS[algorithm]

    train_env, val_env, test_env = create_train_test_envs(
        df,
        train_ratio=0.7,
        val_ratio=0.15,
        episode_length=500,
        initial_balance=10000.0,
        transaction_cost=0.001,
    )

    train_env = Monitor(train_env)

    # Algorithm-specific configs
    if algorithm == "PPO":
        model = AlgorithmClass("MlpPolicy", train_env, learning_rate=3e-4, verbose=0)
    elif algorithm == "A2C":
        model = AlgorithmClass("MlpPolicy", train_env, learning_rate=7e-4, verbose=0)
    else:  # DQN
        model = AlgorithmClass("MlpPolicy", train_env, learning_rate=1e-4, verbose=0)

    model.learn(total_timesteps=timesteps, progress_bar=False)

    # Evaluate on test set
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
        "final_balance": info["balance"],
    }


def run_test_phase(phase_name: str, config: Dict, lstm_models: Dict) -> List[Dict]:
    """Run a single test phase."""
    results = []

    # Generate all combinations
    combinations = list(product(
        config["training"],
        config["symbols"],
        config["timeframes"],
        config["algorithms"],
        config["filters"],
        config["use_lstm"],
    ))

    total_tests = len(combinations) * config["runs"]
    current_test = 0

    print(f"\n{'=' * 70}")
    print(f"PHASE: {phase_name}")
    print(f"Total test combinations: {len(combinations)} x {config['runs']} runs = {total_tests} tests")
    print("=" * 70)

    for training, symbol, timeframe, algorithm, filter_name, use_lstm in combinations:
        test_id = f"{symbol}_{timeframe}_{algorithm}_{filter_name}_lstm{use_lstm}_{training}"

        print(f"\n  [{current_test+1}/{total_tests}] {test_id}")

        # Load data
        df = get_data(symbol, timeframe)
        if df is None:
            print(f"    Skipped: insufficient data")
            results.append({
                "test_id": test_id,
                "phase": phase_name,
                "error": "insufficient_data",
            })
            current_test += config["runs"]
            continue

        # Add LSTM if requested
        if use_lstm:
            lstm_model = lstm_models.get(timeframe)
            if lstm_model:
                df = add_lstm_predictions(df, lstm_model)
                df = df.dropna(subset=["lstm_pred"]).reset_index(drop=True)
            else:
                print(f"    Warning: No LSTM for {timeframe}, skipping LSTM")

        # Apply filter
        df = apply_regime_filter(df, filter_name)

        if len(df) < 300:
            print(f"    Skipped: insufficient data after filtering ({len(df)} rows)")
            results.append({
                "test_id": test_id,
                "phase": phase_name,
                "error": "insufficient_data_after_filter",
            })
            current_test += config["runs"]
            continue

        print(f"    Data: {len(df)} rows")

        # Run tests
        timesteps = TRAINING_LEVELS[training]
        pnls = []

        for run in range(config["runs"]):
            current_test += 1
            print(f"    Run {run+1}/{config['runs']}...", end=" ", flush=True)

            result = train_and_test(df, algorithm, timesteps)
            pnls.append(result["pnl"])

            status = "[WIN]" if result["pnl"] > 0 else "[LOSS]"
            print(f"PnL: ${result['pnl']:>8.2f} {status}")

        # Aggregate
        results.append({
            "test_id": test_id,
            "phase": phase_name,
            "symbol": symbol,
            "timeframe": timeframe,
            "algorithm": algorithm,
            "filter": filter_name,
            "use_lstm": use_lstm,
            "training": training,
            "timesteps": timesteps,
            "runs": config["runs"],
            "avg_pnl": np.mean(pnls),
            "std_pnl": np.std(pnls) if len(pnls) > 1 else 0,
            "min_pnl": min(pnls),
            "max_pnl": max(pnls),
            "profitable_runs": sum(1 for p in pnls if p > 0),
            "all_pnls": pnls,
        })

    return results


def analyze_results(all_results: List[Dict]) -> Dict:
    """Analyze results for overfit detection."""
    analysis = {
        "total_tests": len(all_results),
        "profitable_tests": sum(1 for r in all_results if r.get("avg_pnl", 0) > 0),
        "by_timeframe": {},
        "by_algorithm": {},
        "by_filter": {},
        "by_training_level": {},
        "overfit_warnings": [],
    }

    # Group by categories
    for r in all_results:
        if "error" in r:
            continue

        # By timeframe
        tf = r.get("timeframe", "unknown")
        if tf not in analysis["by_timeframe"]:
            analysis["by_timeframe"][tf] = {"pnls": [], "profitable": 0, "total": 0}
        analysis["by_timeframe"][tf]["pnls"].append(r["avg_pnl"])
        analysis["by_timeframe"][tf]["total"] += 1
        if r["avg_pnl"] > 0:
            analysis["by_timeframe"][tf]["profitable"] += 1

        # By algorithm
        algo = r.get("algorithm", "unknown")
        if algo not in analysis["by_algorithm"]:
            analysis["by_algorithm"][algo] = {"pnls": [], "profitable": 0, "total": 0}
        analysis["by_algorithm"][algo]["pnls"].append(r["avg_pnl"])
        analysis["by_algorithm"][algo]["total"] += 1
        if r["avg_pnl"] > 0:
            analysis["by_algorithm"][algo]["profitable"] += 1

        # By filter
        filt = r.get("filter", "unknown")
        if filt not in analysis["by_filter"]:
            analysis["by_filter"][filt] = {"pnls": [], "profitable": 0, "total": 0}
        analysis["by_filter"][filt]["pnls"].append(r["avg_pnl"])
        analysis["by_filter"][filt]["total"] += 1
        if r["avg_pnl"] > 0:
            analysis["by_filter"][filt]["profitable"] += 1

        # By training level
        train = r.get("training", "unknown")
        if train not in analysis["by_training_level"]:
            analysis["by_training_level"][train] = {"pnls": [], "profitable": 0, "total": 0}
        analysis["by_training_level"][train]["pnls"].append(r["avg_pnl"])
        analysis["by_training_level"][train]["total"] += 1
        if r["avg_pnl"] > 0:
            analysis["by_training_level"][train]["profitable"] += 1

    # Calculate averages
    for category in ["by_timeframe", "by_algorithm", "by_filter", "by_training_level"]:
        for key, data in analysis[category].items():
            if data["pnls"]:
                data["avg_pnl"] = np.mean(data["pnls"])
                data["std_pnl"] = np.std(data["pnls"])
            del data["pnls"]

    # Overfit detection
    # Check if short training >> long training (overfit to noise)
    short_pnl = analysis["by_training_level"].get("short", {}).get("avg_pnl", 0)
    long_pnl = analysis["by_training_level"].get("long", {}).get("avg_pnl", 0)
    if short_pnl > long_pnl * 1.5 and short_pnl > 500:
        analysis["overfit_warnings"].append(
            f"Short training ({short_pnl:.0f}) >> Long training ({long_pnl:.0f}) - possible overfit to noise"
        )

    # Check if win rate too high
    for r in all_results:
        if r.get("avg_pnl", 0) > 1000:
            pnl_std = r.get("std_pnl", 0)
            if pnl_std > abs(r["avg_pnl"]) * 0.8:
                analysis["overfit_warnings"].append(
                    f"{r.get('test_id', 'unknown')}: High variance (std={pnl_std:.0f} vs avg={r['avg_pnl']:.0f})"
                )

    return analysis


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive Test Suite")
    parser.add_argument("--phase", type=str, help="Run specific phase (1-4)")
    parser.add_argument("--quick", action="store_true", help="Quick mode (phase 1 only)")
    args = parser.parse_args()

    print("=" * 70)
    print("COMPREHENSIVE TEST SUITE")
    print("Every Corner Stone Turned")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print()

    # Load LSTM models for each timeframe
    lstm_models = {}
    for tf in TIMEFRAMES:
        try:
            lstm_models[tf] = LSTMClassifier.load(f"checkpoints/lstm_{tf}_mtf")
            print(f"Loaded LSTM for {tf}")
        except:
            print(f"No LSTM found for {tf}")

    print()

    # Determine which phases to run
    if args.quick:
        phases_to_run = ["phase1_scan"]
    elif args.phase:
        phases_to_run = [f"phase{args.phase}_{'scan' if args.phase == '1' else 'validate' if args.phase == '2' else 'extend' if args.phase == '3' else 'long'}"]
    else:
        phases_to_run = list(TEST_MATRIX.keys())

    all_results = []

    for phase_name in phases_to_run:
        if phase_name not in TEST_MATRIX:
            print(f"Unknown phase: {phase_name}")
            continue

        config = TEST_MATRIX[phase_name]
        results = run_test_phase(phase_name, config, lstm_models)
        all_results.extend(results)

    # Analyze
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    analysis = analyze_results(all_results)

    print(f"\nTotal tests: {analysis['total_tests']}")
    print(f"Profitable: {analysis['profitable_tests']} ({analysis['profitable_tests']/max(analysis['total_tests'],1)*100:.0f}%)")

    print("\nBy Timeframe:")
    for tf, data in sorted(analysis["by_timeframe"].items()):
        print(f"  {tf:6} Avg: ${data.get('avg_pnl', 0):>8.2f}  "
              f"Profitable: {data['profitable']}/{data['total']}")

    print("\nBy Algorithm:")
    for algo, data in sorted(analysis["by_algorithm"].items()):
        print(f"  {algo:6} Avg: ${data.get('avg_pnl', 0):>8.2f}  "
              f"Profitable: {data['profitable']}/{data['total']}")

    print("\nBy Filter:")
    for filt, data in sorted(analysis["by_filter"].items()):
        print(f"  {filt:15} Avg: ${data.get('avg_pnl', 0):>8.2f}  "
              f"Profitable: {data['profitable']}/{data['total']}")

    print("\nBy Training Level:")
    for train, data in sorted(analysis["by_training_level"].items()):
        print(f"  {train:8} Avg: ${data.get('avg_pnl', 0):>8.2f}  "
              f"Profitable: {data['profitable']}/{data['total']}")

    # Overfit warnings
    print("\n" + "=" * 70)
    print("OVERFIT DETECTION")
    print("=" * 70)

    if analysis["overfit_warnings"]:
        for warning in analysis["overfit_warnings"]:
            print(f"  [!] {warning}")
    else:
        print("  No major overfit warnings detected.")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"comprehensive_results_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "phases_run": phases_to_run,
            "results": all_results,
            "analysis": analysis,
        }, f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")
    print(f"Finished: {datetime.now()}")


if __name__ == "__main__":
    main()

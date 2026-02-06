#!/usr/bin/env python3
"""
Optuna Hyperparameter Tuning for LSTM

Uses Optuna to find optimal hyperparameters for the LSTM classifier.
Integrates with MLflow for experiment tracking.

Usage:
    python tune_lstm.py --worker thomas --n-trials 50
    python tune_lstm.py --worker thomas --n-trials 20 --timeframe 1h --limit 100000
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import optuna
from optuna.integration import MLflowCallback
import mlflow

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import from train_lstm
from train_lstm import (
    load_training_data,
    compute_labels,
    prepare_features,
    create_sequences,
    _get_mlflow_uri,
)
from models import LSTMClassifier


def create_objective(
    timeframe: str,
    limit: int,
    top_n_symbols: int,
    worker: str,
    seq_length_fixed: int = None,
):
    """
    Create an Optuna objective function with fixed data parameters.

    Args:
        timeframe: Data timeframe
        limit: Max rows to load
        top_n_symbols: Number of symbols to train on
        worker: Worker name
        seq_length_fixed: If set, don't tune seq_length (for faster trials)

    Returns:
        objective function for Optuna
    """
    # Load data once (expensive operation)
    print(f"\nLoading data for tuning (timeframe={timeframe}, limit={limit})...")
    df = load_training_data(
        symbol=None,
        timeframe=timeframe,
        limit=limit,
        exchange=None,
        top_n_symbols=top_n_symbols,
    )

    if len(df) < 1000:
        raise ValueError(f"Not enough data: {len(df)} rows (need 1000+)")

    print(f"Loaded {len(df):,} rows")

    # Compute labels
    df = compute_labels(df, forward_bars=10, flat_threshold=0.005)

    # Define feature columns
    signal_cols = sorted([c for c in df.columns if c.startswith('signal_') or c.startswith('conf_')])
    feature_cols = ['open', 'high', 'low', 'close', 'volume'] + signal_cols

    # Add regime one-hot
    regimes = ['TRENDING_UP', 'TRENDING_DOWN', 'RANGING', 'HIGH_VOL']
    for regime in regimes:
        col_name = f'regime_{regime.lower()}'
        df[col_name] = (df['regime'] == regime).astype(float)
        feature_cols.append(col_name)

    print(f"Using {len(feature_cols)} features")

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function - returns validation accuracy to maximize."""

        # Hyperparameters to tune
        hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
        num_layers = trial.suggest_int("num_layers", 1, 4)
        dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

        if seq_length_fixed:
            seq_length = seq_length_fixed
        else:
            seq_length = trial.suggest_categorical("seq_length", [25, 50, 75, 100])

        print(f"\n--- Trial {trial.number} ---")
        print(f"hidden_size={hidden_size}, layers={num_layers}, dropout={dropout}")
        print(f"lr={learning_rate:.6f}, batch={batch_size}, seq={seq_length}")

        try:
            # Prepare features (normalize)
            df_prep, _ = prepare_features(df.copy(), feature_cols)

            # TEMPORAL SPLIT (70/15/15) - critical for avoiding data leakage
            n = len(df_prep)
            train_end = int(n * 0.70)
            val_end = int(n * 0.85)

            df_train = df_prep.iloc[:train_end]
            df_val = df_prep.iloc[train_end:val_end]

            # Create sequences per split (no boundary crossing)
            X_train, y_train = create_sequences(df_train, feature_cols, seq_length)
            X_val, y_val = create_sequences(df_val, feature_cols, seq_length)

            if len(X_train) < 100 or len(X_val) < 100:
                print("  Not enough sequences, pruning trial")
                raise optuna.TrialPruned()

            # Create model
            model = LSTMClassifier(
                input_size=len(feature_cols),
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_classes=3,
                dropout=dropout,
            )

            # Train with reduced epochs for tuning
            model.fit(
                X_train, y_train,
                X_val, y_val,
                epochs=30,  # Reduced for faster tuning
                batch_size=batch_size,
                learning_rate=learning_rate,
                patience=5,  # Quick early stopping
            )

            # Evaluate on validation set
            val_accuracy = model.evaluate(X_val, y_val)

            print(f"  Val accuracy: {val_accuracy:.1%}")

            # Report intermediate value for pruning
            trial.report(val_accuracy, model.epochs_trained)

            if trial.should_prune():
                raise optuna.TrialPruned()

            return val_accuracy

        except Exception as e:
            print(f"  Trial failed: {e}")
            raise optuna.TrialPruned()

    return objective


def main():
    parser = argparse.ArgumentParser(description="Tune LSTM hyperparameters with Optuna")
    parser.add_argument("--worker", type=str, required=True, help="Worker name")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--timeframe", type=str, default="1h", help="Data timeframe")
    parser.add_argument("--limit", type=int, default=200000, help="Max rows (smaller for faster tuning)")
    parser.add_argument("--top-n-symbols", type=int, default=100, help="Top N symbols by volume")
    parser.add_argument("--seq-length", type=int, default=None, help="Fixed seq length (skip tuning)")
    parser.add_argument("--study-name", type=str, default=None, help="Optuna study name")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (for distributed)")

    args = parser.parse_args()

    print("=" * 60)
    print("AlphaTrader LSTM Hyperparameter Tuning")
    print("=" * 60)
    print(f"Started: {datetime.now()}")
    print(f"Worker: {args.worker}")
    print(f"Trials: {args.n_trials}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Limit: {args.limit:,}")
    print(f"Top N symbols: {args.top_n_symbols}")
    print()

    # Setup MLflow
    mlflow_uri = _get_mlflow_uri()
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("alphatrader-lstm-tuning")
    print(f"MLflow tracking: {mlflow_uri}")

    # Create objective function (loads data once)
    objective = create_objective(
        timeframe=args.timeframe,
        limit=args.limit,
        top_n_symbols=args.top_n_symbols,
        worker=args.worker,
        seq_length_fixed=args.seq_length,
    )

    # Study name
    study_name = args.study_name or f"lstm-{args.timeframe}-{args.worker}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create Optuna study
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",  # Maximize validation accuracy
        storage=args.storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    # MLflow callback for logging each trial
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow_uri,
        metric_name="val_accuracy",
        create_experiment=False,
    )

    print(f"\nStarting optimization: {args.n_trials} trials")
    print("-" * 60)

    # Run optimization
    study.optimize(
        objective,
        n_trials=args.n_trials,
        callbacks=[mlflow_callback],
        show_progress_bar=True,
    )

    # Results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.1%}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save results
    results_dir = Path("optuna_results")
    results_dir.mkdir(exist_ok=True)

    # Save best params as JSON
    import json
    results_file = results_dir / f"{study_name}_best.json"
    with open(results_file, "w") as f:
        json.dump({
            "study_name": study_name,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "timestamp": datetime.now().isoformat(),
            "worker": args.worker,
            "timeframe": args.timeframe,
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Print command to train with best params
    print("\n" + "-" * 60)
    print("To train with best hyperparameters:")
    print("-" * 60)

    bp = study.best_params
    cmd = f"python train_lstm.py --worker {args.worker} --timeframe {args.timeframe}"
    cmd += f" --hidden-size {bp.get('hidden_size', 128)}"
    cmd += f" --num-layers {bp.get('num_layers', 2)}"
    cmd += f" --batch-size {bp.get('batch_size', 64)}"
    if 'seq_length' in bp:
        cmd += f" --seq-length {bp['seq_length']}"
    elif args.seq_length:
        cmd += f" --seq-length {args.seq_length}"
    cmd += " --epochs 100 --limit 500000 --top-n-symbols 300"

    print(cmd)

    print(f"\nFinished: {datetime.now()}")


if __name__ == "__main__":
    main()

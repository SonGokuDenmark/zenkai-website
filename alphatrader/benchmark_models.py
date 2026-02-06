#!/usr/bin/env python3
"""
Benchmark LSTM vs xLSTM models.

Compares training speed, memory usage, and accuracy between:
- Standard LSTM (nn.LSTM)
- xLSTM with sLSTM blocks (exponential gating)
- xLSTM with mLSTM blocks (matrix memory)
- xLSTM mixed (alternating sLSTM/mLSTM)

Usage:
    python benchmark_models.py --worker thomas --limit 50000
    python benchmark_models.py --worker thomas --limit 100000 --epochs 30
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import time
import json

import numpy as np
import torch
import mlflow

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models import LSTMClassifier, xLSTMClassifier

# Import data loading from train_lstm
from train_lstm import (
    load_training_data,
    compute_labels,
    prepare_features,
    create_sequences,
    _get_mlflow_uri,
)


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def benchmark_model(
    model_class,
    model_kwargs: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 30,
    batch_size: int = 64,
) -> dict:
    """
    Benchmark a single model.

    Returns dict with timing, memory, and accuracy metrics.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Create model
    start_mem = get_gpu_memory_mb()
    model = model_class(**model_kwargs)
    model_mem = get_gpu_memory_mb() - start_mem

    print(f"\n{model.name}")
    print("-" * 40)
    print(model.summary())

    # Training
    start_time = time.time()
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=10,
        verbose=True,
    )
    train_time = time.time() - start_time

    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

    # Evaluation
    test_acc = model.evaluate(X_test, y_test)

    # Per-class accuracy
    preds, _ = model.predict(X_test)
    preds_mapped = preds + 1  # -1,0,1 -> 0,1,2

    class_acc = {}
    for cls_idx, cls_name in enumerate(['DOWN', 'FLAT', 'UP']):
        mask = y_test == cls_idx
        if mask.sum() > 0:
            cls_correct = (preds_mapped[mask] == y_test[mask]).sum()
            class_acc[cls_name] = cls_correct / mask.sum()

    results = {
        'model_name': model.name,
        'model_class': model_class.__name__,
        'parameters': sum(p.numel() for p in model.model.parameters()),
        'train_time_seconds': train_time,
        'epochs_completed': model.epochs_trained,
        'seconds_per_epoch': train_time / model.epochs_trained if model.epochs_trained > 0 else 0,
        'model_memory_mb': model_mem,
        'peak_memory_mb': peak_mem,
        'test_accuracy': test_acc,
        'acc_down': class_acc.get('DOWN', 0),
        'acc_flat': class_acc.get('FLAT', 0),
        'acc_up': class_acc.get('UP', 0),
        'best_val_loss': model.best_val_loss,
    }

    print(f"\nResults:")
    print(f"  Test Accuracy: {test_acc:.1%}")
    print(f"  DOWN: {class_acc.get('DOWN', 0):.1%}, FLAT: {class_acc.get('FLAT', 0):.1%}, UP: {class_acc.get('UP', 0):.1%}")
    print(f"  Training Time: {train_time:.1f}s ({results['seconds_per_epoch']:.1f}s/epoch)")
    print(f"  Peak Memory: {peak_mem:.1f} MB")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark LSTM vs xLSTM models")
    parser.add_argument("--worker", type=str, required=True, help="Worker name")
    parser.add_argument("--timeframe", type=str, default="1h", help="Data timeframe")
    parser.add_argument("--limit", type=int, default=100000, help="Max rows to load")
    parser.add_argument("--top-n-symbols", type=int, default=100, help="Top N symbols")
    parser.add_argument("--epochs", type=int, default=30, help="Max epochs per model")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden/embedding size")
    parser.add_argument("--seq-length", type=int, default=50, help="Sequence length")
    parser.add_argument("--skip-lstm", action="store_true", help="Skip standard LSTM")
    parser.add_argument("--skip-slstm", action="store_true", help="Skip sLSTM")
    parser.add_argument("--skip-mlstm", action="store_true", help="Skip mLSTM")
    parser.add_argument("--skip-mixed", action="store_true", help="Skip mixed xLSTM")

    args = parser.parse_args()

    print("=" * 60)
    print("LSTM vs xLSTM Benchmark")
    print("=" * 60)
    print(f"Started: {datetime.now()}")
    print(f"Worker: {args.worker}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Limit: {args.limit:,}")
    print(f"Epochs: {args.epochs}")
    print(f"Hidden/Embedding size: {args.hidden_size}")
    print(f"Sequence length: {args.seq_length}")
    print()

    # Setup MLflow
    try:
        mlflow.set_tracking_uri(_get_mlflow_uri())
        mlflow.set_experiment("alphatrader-benchmark")
        print(f"MLflow tracking: {_get_mlflow_uri()}")
    except Exception as e:
        print(f"Warning: MLflow setup failed: {e}")

    # Load data
    print("\nLoading data...")
    df = load_training_data(
        symbol=None,
        timeframe=args.timeframe,
        limit=args.limit,
        exchange=None,
        top_n_symbols=args.top_n_symbols,
    )

    if len(df) < 1000:
        print(f"ERROR: Not enough data ({len(df)} rows)")
        return

    print(f"Loaded {len(df):,} rows")

    # Compute labels
    df = compute_labels(df, forward_bars=10, flat_threshold=0.005)

    # Define features
    signal_cols = sorted([c for c in df.columns if c.startswith('signal_') or c.startswith('conf_')])
    feature_cols = ['open', 'high', 'low', 'close', 'volume'] + signal_cols

    # Add regime one-hot
    regimes = ['TRENDING_UP', 'TRENDING_DOWN', 'RANGING', 'HIGH_VOL']
    for regime in regimes:
        col_name = f'regime_{regime.lower()}'
        df[col_name] = (df['regime'] == regime).astype(float)
        feature_cols.append(col_name)

    n_features = len(feature_cols)
    print(f"Using {n_features} features")

    # Prepare and split data (temporal split)
    df_prep, _ = prepare_features(df, feature_cols)

    n = len(df_prep)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    df_train = df_prep.iloc[:train_end]
    df_val = df_prep.iloc[train_end:val_end]
    df_test = df_prep.iloc[val_end:]

    # Create sequences
    print("\nCreating sequences...")
    X_train, y_train = create_sequences(df_train, feature_cols, args.seq_length)
    X_val, y_val = create_sequences(df_val, feature_cols, args.seq_length)
    X_test, y_test = create_sequences(df_test, feature_cols, args.seq_length)

    print(f"  Train: {len(X_train):,}")
    print(f"  Val: {len(X_val):,}")
    print(f"  Test: {len(X_test):,}")

    # Models to benchmark
    models_to_test = []

    if not args.skip_lstm:
        models_to_test.append({
            'class': LSTMClassifier,
            'kwargs': {
                'input_size': n_features,
                'hidden_size': args.hidden_size,
                'num_layers': 2,
                'dropout': 0.3,
            },
            'name': 'Standard LSTM',
        })

    if not args.skip_slstm:
        models_to_test.append({
            'class': xLSTMClassifier,
            'kwargs': {
                'input_size': n_features,
                'embedding_dim': args.hidden_size,
                'num_blocks': 2,
                'context_length': args.seq_length,
                'dropout': 0.3,
                'block_type': 'slstm',
            },
            'name': 'xLSTM (sLSTM only)',
        })

    if not args.skip_mlstm:
        models_to_test.append({
            'class': xLSTMClassifier,
            'kwargs': {
                'input_size': n_features,
                'embedding_dim': args.hidden_size,
                'num_blocks': 2,
                'context_length': args.seq_length,
                'dropout': 0.3,
                'block_type': 'mlstm',
            },
            'name': 'xLSTM (mLSTM only)',
        })

    if not args.skip_mixed:
        models_to_test.append({
            'class': xLSTMClassifier,
            'kwargs': {
                'input_size': n_features,
                'embedding_dim': args.hidden_size,
                'num_blocks': 2,
                'context_length': args.seq_length,
                'dropout': 0.3,
                'block_type': 'mixed',
            },
            'name': 'xLSTM (mixed)',
        })

    # Run benchmarks
    all_results = []

    for model_info in models_to_test:
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {model_info['name']}")
        print("=" * 60)

        try:
            # Run benchmark (don't fail if MLflow is down)
            results = benchmark_model(
                model_info['class'],
                model_info['kwargs'],
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )

            # Try to log to MLflow (optional)
            try:
                with mlflow.start_run(run_name=f"bench-{model_info['name'].replace(' ', '-').lower()}"):
                    mlflow.log_params({
                        'model_type': results['model_class'],
                        'hidden_size': args.hidden_size,
                        'seq_length': args.seq_length,
                        'limit': args.limit,
                        'worker': args.worker,
                    })
                    mlflow.log_metrics({
                        'test_accuracy': results['test_accuracy'],
                        'acc_down': results['acc_down'],
                        'acc_flat': results['acc_flat'],
                        'acc_up': results['acc_up'],
                        'train_time_seconds': results['train_time_seconds'],
                        'peak_memory_mb': results['peak_memory_mb'],
                        'parameters': results['parameters'],
                    })
            except Exception as mlflow_err:
                print(f"  (MLflow logging skipped: {mlflow_err})")

            all_results.append(results)

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    print(f"\n{'Model':<25} {'Accuracy':>10} {'Time':>10} {'Memory':>10} {'Params':>12}")
    print("-" * 70)

    for r in all_results:
        print(f"{r['model_name']:<25} {r['test_accuracy']:>9.1%} {r['train_time_seconds']:>9.1f}s {r['peak_memory_mb']:>9.1f}MB {r['parameters']:>12,}")

    # Save results
    results_dir = Path("benchmark_results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f"benchmark_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'worker': args.worker,
            'timeframe': args.timeframe,
            'limit': args.limit,
            'epochs': args.epochs,
            'hidden_size': args.hidden_size,
            'seq_length': args.seq_length,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'n_features': n_features,
            'results': all_results,
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"\nFinished: {datetime.now()}")


if __name__ == "__main__":
    main()

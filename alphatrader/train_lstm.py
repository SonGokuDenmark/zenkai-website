#!/usr/bin/env python3
"""
Quick LSTM Training Script

Connects to PostgreSQL on zenkaiserver, loads processed candles,
and trains an LSTM to predict market direction.

This is the simple "prove it works" version.

Usage:
    python train_lstm.py                          # Train with defaults
    python train_lstm.py --epochs 50              # Custom epochs
    python train_lstm.py --symbol BTCUSDT         # Single symbol
    python train_lstm.py --timeframe 15m          # Single timeframe
"""

import argparse
import os
import sys
import socket
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Optional
import json
import time

import numpy as np
import pandas as pd
import psycopg2
import requests
import mlflow
# train_test_split replaced with index-based splitting to save RAM
from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models import LSTMClassifier, ModelRegistry


# Discord webhook for training notifications
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")


def send_discord_notification(
    model_name: str,
    test_accuracy: float,
    class_accuracies: dict,
    training_time_seconds: int,
    epochs_completed: int,
    checkpoint_path: str,
    settings: dict,
    worker: str = None,
    error: str = None,
):
    """Send training results to Discord."""
    worker_str = worker or "unknown"

    if error:
        # Error notification
        embed = {
            "title": f"❌ LSTM Training Failed ({worker_str})",
            "color": 15158332,  # Red
            "fields": [
                {"name": "Worker", "value": worker_str, "inline": True},
                {"name": "Error", "value": f"```{error[:500]}```", "inline": False},
            ],
            "timestamp": datetime.now().isoformat() + "Z",
        }
    else:
        # Format training time
        hours = training_time_seconds // 3600
        minutes = (training_time_seconds % 3600) // 60
        seconds = training_time_seconds % 60
        if hours > 0:
            time_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            time_str = f"{minutes}m {seconds}s"
        else:
            time_str = f"{seconds}s"

        # Build accuracy breakdown
        acc_lines = [f"**Overall: {test_accuracy:.1%}**"]
        for cls_name, (acc, count) in class_accuracies.items():
            acc_lines.append(f"{cls_name}: {acc:.1%} ({count:,} samples)")

        # Settings summary
        settings_str = " | ".join([f"{k}: {v}" for k, v in settings.items()])

        embed = {
            "title": f"🧠 LSTM Training Complete ({worker_str})",
            "color": 5763719,  # Green
            "fields": [
                {"name": "Worker", "value": worker_str, "inline": True},
                {"name": "Model", "value": model_name, "inline": True},
                {"name": "Epochs", "value": str(epochs_completed), "inline": True},
                {"name": "Time", "value": time_str, "inline": True},
                {"name": "Test Accuracy", "value": "\n".join(acc_lines), "inline": False},
                {"name": "Settings", "value": f"```{settings_str}```", "inline": False},
                {"name": "Checkpoint", "value": f"`{checkpoint_path}`", "inline": False},
            ],
            "timestamp": datetime.now().isoformat() + "Z",
        }

    payload = {"embeds": [embed]}

    try:
        print("Sending Discord notification...")
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        if response.status_code == 204:
            print("Discord notification sent successfully!")
        else:
            print(f"Discord notification failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Discord notification error: {e}")


def log_experiment_start(run_id: str, worker: str, config: dict) -> Optional[int]:
    """Log experiment start to database. Returns experiment ID."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Detect GPU
        gpu_name = None
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
        except:
            pass

        cursor.execute("""
            INSERT INTO experiments (
                run_id, worker, started_at, status,
                symbol, timeframe, epochs_requested, batch_size,
                hidden_size, num_layers, seq_length, data_limit,
                hostname, gpu_name, config_json
            ) VALUES (
                %s, %s, NOW(), 'running',
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s
            )
            RETURNING id
        """, (
            run_id, worker,
            config.get('symbol'), config['timeframe'], config['epochs'], config['batch_size'],
            config['hidden_size'], config['num_layers'], config['seq_length'], config['limit'],
            socket.gethostname(), gpu_name, json.dumps(config)
        ))

        experiment_id = cursor.fetchone()[0]
        conn.commit()
        conn.close()
        return experiment_id
    except Exception as e:
        print(f"Warning: Could not log experiment start: {e}")
        return None


def log_experiment_end(experiment_id: Optional[int], status: str, results: dict):
    """Update experiment with final results."""
    if experiment_id is None:
        return

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE experiments SET
                status = %s,
                finished_at = NOW(),
                epochs_completed = %s,
                train_samples = %s,
                val_samples = %s,
                test_samples = %s,
                test_accuracy = %s,
                best_val_loss = %s,
                acc_down = %s,
                acc_flat = %s,
                acc_up = %s,
                samples_down = %s,
                samples_flat = %s,
                samples_up = %s,
                training_time_seconds = %s,
                checkpoint_path = %s,
                history_json = %s
            WHERE id = %s
        """, (
            status,
            results.get('epochs_completed'),
            results.get('train_samples'),
            results.get('val_samples'),
            results.get('test_samples'),
            results.get('test_accuracy'),
            results.get('best_val_loss'),
            results.get('acc_down'),
            results.get('acc_flat'),
            results.get('acc_up'),
            results.get('samples_down'),
            results.get('samples_flat'),
            results.get('samples_up'),
            results.get('training_time_seconds'),
            results.get('checkpoint_path'),
            json.dumps(results.get('history', {})),
            experiment_id
        ))

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Warning: Could not log experiment end: {e}")


# Database config - auto-detect network
def _get_db_host():
    """Auto-detect which IP to use for database connection."""
    if os.getenv("ZENKAI_DB_HOST"):
        return os.getenv("ZENKAI_DB_HOST")

    # Try local network first (faster)
    local_ip = os.getenv("ZENKAI_DB_HOST", "192.168.0.160")
    tailscale_ip = os.getenv("ZENKAI_DB_HOST_TAILSCALE", "100.110.101.78")

    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((local_ip, 5432))
        sock.close()
        if result == 0:
            return local_ip
    except:
        pass

    # Fall back to Tailscale
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

# Regime mapping
REGIME_MAP = {
    "TRENDING_UP": 0,
    "TRENDING_DOWN": 1,
    "RANGING": 2,
    "HIGH_VOL": 3,
}


def get_db_connection():
    """Get PostgreSQL connection."""
    return psycopg2.connect(**DB_CONFIG)


def load_training_data(
    symbol: Optional[str] = None,
    timeframe: str = "15m",
    limit: int = 500000,
    exchange: str = None,
    top_n_symbols: int = None,
    min_rows_per_symbol: int = 500,
) -> pd.DataFrame:
    """
    Load processed candles from database.

    Args:
        symbol: Specific symbol or None for all
        timeframe: Timeframe to load
        limit: Max rows to load
        exchange: Filter by exchange (binance, us_stock)
        top_n_symbols: If set, only load top N symbols by volume
        min_rows_per_symbol: Minimum rows required per symbol (default: 500)

    Returns:
        DataFrame with OHLCV + signals + regime
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get all signal/conf columns dynamically
    cursor.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'ohlcv'
        AND (column_name LIKE 'signal_%' OR column_name LIKE 'conf_%')
        ORDER BY column_name
    """)
    signal_cols = [row[0] for row in cursor.fetchall()]
    print(f"Found {len(signal_cols)} signal/conf columns")

    # If top_n_symbols specified, get list of symbols first
    symbols_filter = None
    if top_n_symbols and not symbol:
        print(f"Finding top {top_n_symbols} symbols by volume...")
        cursor.execute("""
            SELECT symbol, COUNT(*) as rows, AVG(volume) as avg_vol
            FROM ohlcv
            WHERE timeframe = %s AND regime IS NOT NULL
            GROUP BY symbol
            HAVING COUNT(*) >= %s
            ORDER BY AVG(volume) DESC
            LIMIT %s
        """, (timeframe, min_rows_per_symbol, top_n_symbols))

        symbols_filter = [row[0] for row in cursor.fetchall()]
        print(f"  Selected {len(symbols_filter)} symbols with {min_rows_per_symbol}+ rows each")
        if symbols_filter:
            print(f"  Top 5: {', '.join(symbols_filter[:5])}")
            print(f"  Bottom 5: {', '.join(symbols_filter[-5:])}")

    # Build query
    base_cols = ["open_time", "symbol", "open", "high", "low", "close", "volume", "regime"]
    all_cols = base_cols + signal_cols

    where_parts = ["timeframe = %s", "regime IS NOT NULL", "processed_at IS NOT NULL"]
    params = [timeframe]

    if symbol:
        where_parts.append("symbol = %s")
        params.append(symbol)
    elif symbols_filter:
        # Use IN clause for top N symbols
        placeholders = ','.join(['%s'] * len(symbols_filter))
        where_parts.append(f"symbol IN ({placeholders})")
        params.extend(symbols_filter)

    if exchange:
        where_parts.append("exchange = %s")
        params.append(exchange)

    query = f"""
        SELECT {', '.join(all_cols)}
        FROM ohlcv
        WHERE {' AND '.join(where_parts)}
        ORDER BY symbol, open_time
        LIMIT %s
    """
    params.append(limit)

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    print(f"Loaded {len(df):,} rows from database ({df['symbol'].nunique()} symbols)")
    return df


def compute_labels(df: pd.DataFrame, forward_bars: int = 10, flat_threshold: float = 0.005) -> pd.DataFrame:
    """
    Compute forward-looking labels.

    Args:
        df: DataFrame with close prices
        forward_bars: Bars to look ahead
        flat_threshold: +/- threshold for FLAT classification

    Returns:
        DataFrame with 'label' column (0=DOWN, 1=FLAT, 2=UP)
    """
    df = df.copy()

    # Future return
    df["future_return"] = df.groupby("symbol")["close"].transform(
        lambda x: x.shift(-forward_bars) / x - 1
    )

    # Classify
    # 0=DOWN, 1=FLAT, 2=UP for PyTorch CrossEntropyLoss
    df["label"] = 1  # Default FLAT
    df.loc[df["future_return"] > flat_threshold, "label"] = 2  # UP
    df.loc[df["future_return"] < -flat_threshold, "label"] = 0  # DOWN

    # Drop rows without labels (last forward_bars rows per symbol)
    df = df.dropna(subset=["future_return"])

    return df


def prepare_features(df: pd.DataFrame, norm_stats: dict = None, fit_norm: bool = True) -> Tuple[pd.DataFrame, dict]:
    """
    Prepare features for training.

    Normalizes OHLCV, encodes regime, fills NaN.

    IMPORTANT: For proper train/test split, call with:
    - fit_norm=True on TRAIN data to compute normalization stats
    - fit_norm=False on VAL/TEST data, passing the train stats

    Args:
        df: DataFrame to prepare
        norm_stats: Pre-computed normalization stats (for val/test)
        fit_norm: Whether to compute new stats (True for train, False for val/test)

    Returns:
        Tuple of (prepared_df, norm_stats)
    """
    df = df.copy()

    if norm_stats is None:
        norm_stats = {}

    # Normalize OHLCV per symbol using rolling stats
    # Note: Rolling normalization is applied per-row using only PAST data (no leakage)
    # The rolling window only looks backwards, so this is safe
    for col in ["open", "high", "low", "close"]:
        df[col] = df.groupby("symbol")[col].transform(
            lambda x: (x - x.rolling(50, min_periods=1).mean()) / x.rolling(50, min_periods=1).std().clip(lower=1e-8)
        )

    # Normalize volume
    df["volume"] = df.groupby("symbol")["volume"].transform(
        lambda x: (x - x.rolling(50, min_periods=1).mean()) / x.rolling(50, min_periods=1).std().clip(lower=1e-8)
    )

    # One-hot encode regime
    for regime, idx in REGIME_MAP.items():
        df[f"regime_{regime.lower()}"] = (df["regime"] == regime).astype(float)

    # Fill NaN in technical indicators and signals
    # Technical indicators may have NaN in early rows (warmup period)
    # Using forward fill first, then 0 for remaining NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill().fillna(0)

    # Replace inf
    df = df.replace([np.inf, -np.inf], 0)

    return df, norm_stats


def create_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_length: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training.

    Args:
        df: Prepared DataFrame
        feature_cols: Columns to use as features
        seq_length: Sequence length

    Returns:
        X: shape (n_samples, seq_length, n_features)
        y: shape (n_samples,)
    """
    X_list = []
    y_list = []

    # Process per symbol to avoid mixing data
    for symbol in df["symbol"].unique():
        symbol_df = df[df["symbol"] == symbol].sort_values("open_time")

        if len(symbol_df) < seq_length + 1:
            continue

        features = symbol_df[feature_cols].values
        labels = symbol_df["label"].values

        # Create sequences
        # IMPORTANT: Label is at i + seq_length (the bar AFTER the sequence ends)
        # This ensures we're predicting the future, not the present
        for i in range(len(symbol_df) - seq_length - 1):
            X_list.append(features[i:i + seq_length])
            y_list.append(labels[i + seq_length])  # Label AFTER sequence ends

    # Use float16 to halve memory usage (from ~37GB to ~18GB for 1M sequences)
    X = np.array(X_list, dtype=np.float16)
    y = np.array(y_list, dtype=np.int64)
    del X_list, y_list  # Free memory immediately

    print(f"Created {len(X):,} sequences of length {seq_length}")
    print(f"  Features: {X.shape[2]}")
    print(f"  Label distribution: DOWN={np.sum(y==0):,} FLAT={np.sum(y==1):,} UP={np.sum(y==2):,}")

    return X, y


def train(
    symbol: Optional[str] = None,
    exchange: Optional[str] = None,
    timeframe: str = "15m",
    epochs: int = 100,
    batch_size: int = 64,
    hidden_size: int = 128,
    num_layers: int = 2,
    seq_length: int = 50,
    limit: int = 500000,
    checkpoint_dir: str = "checkpoints",
    worker: str = "unknown",
    top_n_symbols: int = None,
):
    """
    Main training function.
    """
    start_time = datetime.now()

    # Generate run ID and config
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{worker}"
    config = {
        'symbol': symbol,
        'timeframe': timeframe,
        'epochs': epochs,
        'batch_size': batch_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'seq_length': seq_length,
        'limit': limit,
        'checkpoint_dir': checkpoint_dir,
        'worker': worker,
    }

    # Setup MLflow tracking
    try:
        mlflow.set_tracking_uri(_get_mlflow_uri())
        mlflow.set_experiment("alphatrader-lstm")
        print(f"MLflow tracking: {_get_mlflow_uri()}")
    except Exception as e:
        print(f"Warning: MLflow setup failed: {e}")

    # Log experiment start
    experiment_id = log_experiment_start(run_id, worker, config)
    if experiment_id:
        print(f"Experiment logged: {run_id} (ID: {experiment_id})")

    # Start MLflow run
    mlflow_run = None
    try:
        mlflow_run = mlflow.start_run(run_name=f"lstm-{timeframe}-{worker}")
        mlflow.log_params({
            "symbol": symbol or "ALL",
            "timeframe": timeframe,
            "epochs": epochs,
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "seq_length": seq_length,
            "limit": limit,
            "worker": worker,
            "top_n_symbols": top_n_symbols or "ALL",
        })
    except Exception as e:
        print(f"Warning: MLflow run start failed: {e}")

    print("=" * 60)
    print("AlphaTrader LSTM Training")
    print("=" * 60)
    print(f"Started: {datetime.now()}")
    print(f"Worker: {worker}")
    print(f"Symbol: {symbol or 'ALL'}")
    print(f"Exchange: {exchange or 'ALL'}")
    print(f"Timeframe: {timeframe}")
    print(f"Epochs: {epochs}")
    print()

    # Load data
    print("Loading data...")
    df = load_training_data(
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
        exchange=exchange,
        top_n_symbols=top_n_symbols,
    )

    if len(df) < 1000:
        print("ERROR: Not enough data. Need at least 1000 rows.")
        return

    # =========================================================================
    # COMPUTE TECHNICAL INDICATORS
    # =========================================================================
    # The database has sparse signal_* columns (mostly 0). We need continuous
    # technical indicators (RSI, MACD, ATR, etc.) for the LSTM to learn from.
    # =========================================================================
    print("Computing technical indicators per symbol...")
    from src.data.feature_engineer import FeatureEngineer
    fe = FeatureEngineer()

    # Compute features per symbol (need to sort by time first)
    df = df.rename(columns={"open_time": "timestamp"})  # FeatureEngineer expects 'timestamp'
    processed_dfs = []
    for sym in df["symbol"].unique():
        sym_df = df[df["symbol"] == sym].sort_values("timestamp").copy()
        if len(sym_df) >= 200:  # Need enough data for indicator warmup
            try:
                sym_df = fe.compute_all_features(sym_df)
                processed_dfs.append(sym_df)
            except Exception as e:
                print(f"  Warning: Failed to compute features for {sym}: {e}")

    df = pd.concat(processed_dfs, ignore_index=True)
    df = df.rename(columns={"timestamp": "open_time"})  # Rename back
    del processed_dfs
    print(f"  Computed indicators for {df['symbol'].nunique()} symbols ({len(df):,} rows)")

    # Compute labels
    # NOTE: 0.5% threshold was too small (noise). 1.5% captures real directional moves.
    print("Computing labels...")
    df = compute_labels(df, forward_bars=10, flat_threshold=0.015)

    # =========================================================================
    # FEATURE SELECTION - Use technical indicators instead of sparse signals
    # =========================================================================
    # Technical indicators are continuous and "always on" - the model can learn
    # gradients from them. Sparse binary signals are much harder to learn from.
    # =========================================================================
    feature_cols = [
        # Price features (will be normalized)
        "open", "high", "low", "close", "volume",

        # Momentum indicators (always-on, continuous)
        "rsi_2", "rsi_7", "rsi_14",
        "macd", "macd_signal", "macd_hist", "macd_norm",
        "stoch_k", "stoch_d",
        "cci", "williams_r", "roc_10", "momentum_10",
        "mfi",

        # Trend indicators
        "adx", "plus_di", "minus_di", "di_diff",
        "ema_9_21_diff",
        "close_above_ema50", "close_above_ema200",
        "trend_strength",

        # Volatility indicators
        "atr_pct", "bb_width", "bb_pct",
        "volatility", "volatility_percentile",

        # Volume indicators
        "volume_ratio", "cmf",

        # Price action
        "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
        "is_bullish",

        # Returns (lagging, no leakage)
        "returns_1", "returns_5", "returns_10",
    ]

    # Filter to only columns that exist
    available_cols = [c for c in feature_cols if c in df.columns]
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        print(f"  Warning: Missing {len(missing_cols)} feature columns: {missing_cols[:5]}...")
    feature_cols = available_cols

    # Add regime one-hot (will be added by prepare_features)
    regime_cols = ["regime_trending_up", "regime_trending_down", "regime_ranging", "regime_high_vol"]

    print(f"Using {len(feature_cols) + 4} features: {len(feature_cols)} technical + 4 regime")

    # =========================================================================
    # TEMPORAL SPLIT - CRITICAL FOR AVOIDING DATA LEAKAGE
    # =========================================================================
    # We MUST split by time BEFORE creating sequences to prevent:
    # 1. Future data leaking into training (via rolling normalization)
    # 2. Sequences crossing the train/test boundary
    # 3. Test data influencing feature preparation
    #
    # The split is done PER SYMBOL to maintain temporal order within each symbol
    # =========================================================================
    print("\nPerforming TEMPORAL split (70/15/15)...")
    print("  (Splitting by time to prevent data leakage)")

    train_dfs = []
    val_dfs = []
    test_dfs = []

    for sym in df["symbol"].unique():
        sym_df = df[df["symbol"] == sym].sort_values("open_time").reset_index(drop=True)
        n = len(sym_df)

        if n < seq_length + 10:  # Need enough data for at least a few sequences
            continue

        train_end = int(n * 0.7)
        val_end = int(n * 0.85)

        train_dfs.append(sym_df.iloc[:train_end])
        val_dfs.append(sym_df.iloc[train_end:val_end])
        test_dfs.append(sym_df.iloc[val_end:])

    df_train = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
    df_val = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame()
    df_test = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()

    del train_dfs, val_dfs, test_dfs, df  # Free memory
    import gc
    gc.collect()

    print(f"  Train rows: {len(df_train):,}")
    print(f"  Val rows:   {len(df_val):,}")
    print(f"  Test rows:  {len(df_test):,}")

    # Prepare features - fit normalization on TRAIN only
    print("\nPreparing features (normalization fitted on train only)...")
    df_train, norm_stats = prepare_features(df_train, fit_norm=True)
    df_val, _ = prepare_features(df_val, norm_stats=norm_stats, fit_norm=False)
    df_test, _ = prepare_features(df_test, norm_stats=norm_stats, fit_norm=False)

    # Update feature cols with regime
    feature_cols.extend(regime_cols)

    # Create sequences PER SPLIT (no boundary crossing)
    print("\nCreating sequences per split (no boundary crossing)...")
    X_train, y_train = create_sequences(df_train, feature_cols, seq_length=seq_length)
    X_val, y_val = create_sequences(df_val, feature_cols, seq_length=seq_length)
    X_test, y_test = create_sequences(df_test, feature_cols, seq_length=seq_length)

    # Free DataFrames
    del df_train, df_val, df_test
    gc.collect()

    if len(X_train) < 100:
        print("ERROR: Not enough training sequences. Try loading more data.")
        return

    print(f"\nData splits (after sequence creation):")
    print(f"  Train: {len(X_train):,}")
    print(f"  Val:   {len(X_val):,}")
    print(f"  Test:  {len(X_test):,}")

    # Create model
    input_size = X_train.shape[2]
    print(f"\nCreating LSTM model...")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Layers: {num_layers}")

    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.3,
        learning_rate=0.001,
    )
    print(model.summary())

    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Train
    print(f"\nTraining...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=15,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=10,
    )

    # Final evaluation on test set
    print(f"\nEvaluating on test set...")
    preds, confs = model.predict(X_test)

    # Convert predictions back to 0,1,2 for comparison
    preds_mapped = preds + 1  # -1,0,1 -> 0,1,2

    accuracy = np.mean(preds_mapped == y_test)
    print(f"  Test Accuracy: {accuracy:.1%}")

    # Per-class accuracy
    class_accuracies = {}
    for cls, name in [(0, "DOWN"), (1, "FLAT"), (2, "UP")]:
        mask = y_test == cls
        if mask.sum() > 0:
            cls_acc = np.mean(preds_mapped[mask] == cls)
            class_accuracies[name] = (cls_acc, int(mask.sum()))
            print(f"  {name}: {cls_acc:.1%} ({mask.sum():,} samples)")

    # Save final model (include timeframe in name for clarity)
    final_path = f"{checkpoint_dir}/lstm_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model.save(final_path)
    print(f"\nModel saved to: {final_path}")

    # Save training summary
    summary = {
        "symbol": symbol,
        "timeframe": timeframe,
        "epochs_trained": model.epochs_trained,
        "test_accuracy": float(accuracy),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "history": model.history,
    }
    with open(f"{final_path}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Calculate training time
    training_time = int((datetime.now() - start_time).total_seconds())

    # Log experiment completion
    results = {
        'epochs_completed': model.epochs_trained,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'test_accuracy': float(accuracy),
        'best_val_loss': float(model.best_val_loss) if model.best_val_loss else None,
        'acc_down': class_accuracies.get('DOWN', (None, 0))[0],
        'acc_flat': class_accuracies.get('FLAT', (None, 0))[0],
        'acc_up': class_accuracies.get('UP', (None, 0))[0],
        'samples_down': class_accuracies.get('DOWN', (None, 0))[1],
        'samples_flat': class_accuracies.get('FLAT', (None, 0))[1],
        'samples_up': class_accuracies.get('UP', (None, 0))[1],
        'training_time_seconds': training_time,
        'checkpoint_path': final_path,
        'history': model.history,
    }
    log_experiment_end(experiment_id, 'completed', results)

    # Log metrics to MLflow
    try:
        mlflow.log_metrics({
            "test_accuracy": float(accuracy),
            "acc_down": float(class_accuracies.get('DOWN', (0, 0))[0] or 0),
            "acc_flat": float(class_accuracies.get('FLAT', (0, 0))[0] or 0),
            "acc_up": float(class_accuracies.get('UP', (0, 0))[0] or 0),
            "epochs_completed": model.epochs_trained,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "training_time_seconds": training_time,
        })
        mlflow.log_artifact(f"{final_path}_summary.json")
        print(f"MLflow metrics logged successfully")
    except Exception as e:
        print(f"Warning: MLflow metric logging failed: {e}")
    finally:
        if mlflow_run:
            mlflow.end_run()

    # Send Discord notification
    send_discord_notification(
        model_name=model.name,
        test_accuracy=accuracy,
        class_accuracies=class_accuracies,
        training_time_seconds=training_time,
        epochs_completed=model.epochs_trained,
        checkpoint_path=final_path,
        settings={
            "symbol": symbol or "ALL",
            "tf": timeframe,
            "hidden": hidden_size,
            "layers": num_layers,
            "seq": seq_length,
            "batch": batch_size,
        },
        worker=worker,
    )

    print(f"\nTraining complete!")
    print(f"Finished: {datetime.now()}")


def main():
    parser = argparse.ArgumentParser(description="Train AlphaTrader LSTM")
    parser.add_argument("--worker", type=str, required=True,
                        help="Worker name (required) - who is running this (e.g., fips, thomas, goku)")
    parser.add_argument("--symbol", type=str, default=None, help="Symbol to train on (default: all)")
    parser.add_argument("--exchange", type=str, default=None, help="Exchange filter (binance, us_stock)")
    parser.add_argument("--timeframe", type=str, default="15m", help="Timeframe (default: 15m)")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--hidden-size", type=int, default=128, help="LSTM hidden size (default: 128)")
    parser.add_argument("--num-layers", type=int, default=2, help="LSTM layers (default: 2)")
    parser.add_argument("--seq-length", type=int, default=50, help="Sequence length (default: 50)")
    parser.add_argument("--limit", type=int, default=500000, help="Max rows to load (default: 500000)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--top-n-symbols", type=int, default=None,
                        help="Train on top N symbols by volume (default: all symbols)")

    args = parser.parse_args()

    train(
        symbol=args.symbol,
        exchange=args.exchange,
        timeframe=args.timeframe,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        seq_length=args.seq_length,
        limit=args.limit,
        checkpoint_dir=args.checkpoint_dir,
        worker=args.worker,
        top_n_symbols=args.top_n_symbols,
    )


if __name__ == "__main__":
    main()

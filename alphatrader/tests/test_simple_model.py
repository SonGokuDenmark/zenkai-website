#!/usr/bin/env python3
"""
Simple baseline test - can we predict ANYTHING?

Tests:
1. Binary classification (UP vs DOWN, ignoring FLAT)
2. Longer horizon (20 bars)
3. Simple logistic regression as baseline
4. LSTM with same setup

If logistic regression beats random, there's signal.
If LSTM can't beat logistic regression, architecture is wrong.
"""

import os
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
load_dotenv()
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Connect to DB
def get_data(timeframe='1h', limit=50000):
    conn = psycopg2.connect(
        host=os.getenv("ZENKAI_DB_HOST", "192.168.0.160"),
        database=os.getenv("ZENKAI_DB_NAME", "zenkai_data"),
        user=os.getenv("ZENKAI_DB_USER", "zenkai"),
        password=os.getenv("ZENKAI_DB_PASSWORD"),
    )

    query = """
        SELECT symbol, open_time, open, high, low, close, volume, regime
        FROM ohlcv
        WHERE timeframe = %s AND regime IS NOT NULL
        ORDER BY symbol, open_time
        LIMIT %s
    """
    df = pd.read_sql_query(query, conn, params=[timeframe, limit])
    conn.close()
    return df


def compute_features(df):
    """Compute simple technical features per symbol."""
    features = []

    for sym in df['symbol'].unique():
        sym_df = df[df['symbol'] == sym].sort_values('open_time').copy()

        if len(sym_df) < 100:
            continue

        # Simple features (no lookahead)
        sym_df['return_1'] = sym_df['close'].pct_change(1)
        sym_df['return_5'] = sym_df['close'].pct_change(5)
        sym_df['return_10'] = sym_df['close'].pct_change(10)
        sym_df['return_20'] = sym_df['close'].pct_change(20)

        sym_df['volatility'] = sym_df['return_1'].rolling(20).std()
        sym_df['volume_ratio'] = sym_df['volume'] / sym_df['volume'].rolling(20).mean()

        # RSI
        delta = sym_df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        sym_df['rsi'] = 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))

        # Price position
        sym_df['close_vs_high20'] = sym_df['close'] / sym_df['high'].rolling(20).max()
        sym_df['close_vs_low20'] = sym_df['close'] / sym_df['low'].rolling(20).min()

        features.append(sym_df)

    return pd.concat(features, ignore_index=True)


def compute_labels(df, forward_bars=20, threshold=0.02):
    """Binary labels: 1 if price goes up > threshold, 0 otherwise."""
    df = df.copy()

    df['future_return'] = df.groupby('symbol')['close'].transform(
        lambda x: x.shift(-forward_bars) / x - 1
    )

    # Binary: UP (1) vs NOT-UP (0)
    df['label'] = (df['future_return'] > threshold).astype(int)

    df = df.dropna()
    return df


def main():
    print("="*60)
    print("SIMPLE BASELINE TEST")
    print("="*60)
    print("Testing if there's ANY predictable signal in the data")
    print()

    # Load data
    print("Loading data...")
    df = get_data(timeframe='1h', limit=100000)
    print(f"  Loaded {len(df):,} rows, {df['symbol'].nunique()} symbols")

    # Compute features
    print("Computing features...")
    df = compute_features(df)
    print(f"  After features: {len(df):,} rows")

    # Test different prediction horizons and thresholds
    configs = [
        (10, 0.01, "10 bars, 1% threshold"),
        (10, 0.02, "10 bars, 2% threshold"),
        (20, 0.02, "20 bars, 2% threshold"),
        (20, 0.03, "20 bars, 3% threshold"),
        (50, 0.05, "50 bars, 5% threshold"),
    ]

    feature_cols = [
        'return_1', 'return_5', 'return_10', 'return_20',
        'volatility', 'volume_ratio', 'rsi',
        'close_vs_high20', 'close_vs_low20'
    ]

    for forward_bars, threshold, desc in configs:
        print(f"\n{'='*60}")
        print(f"Config: {desc}")
        print(f"{'='*60}")

        # Compute labels
        df_labeled = compute_labels(df.copy(), forward_bars, threshold)

        # Get features
        X = df_labeled[feature_cols].values
        y = df_labeled['label'].values

        # Remove any remaining NaN
        mask = ~np.isnan(X).any(axis=1)
        X, y = X[mask], y[mask]

        print(f"  Samples: {len(X):,}")
        print(f"  Class balance: UP={y.sum():,} ({y.mean():.1%}), NOT-UP={len(y)-y.sum():,}")

        if y.mean() < 0.1 or y.mean() > 0.9:
            print("  SKIPPED - too imbalanced")
            continue

        # Temporal split (70/30)
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Random baseline
        random_acc = max(y_test.mean(), 1 - y_test.mean())
        print(f"\n  Random baseline: {random_acc:.1%}")

        # Test models
        models = [
            ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight='balanced')),
            ("Random Forest", RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', n_jobs=-1)),
            ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, max_depth=3)),
        ]

        for name, model in models:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)

            # Edge over random
            edge = acc - random_acc
            status = "[SIGNAL]" if edge > 0.02 else "[no edge]" if edge > 0 else "[WORSE]"

            print(f"  {name:25} Acc: {acc:.1%} (edge: {edge:+.1%}) {status}")

    print("\n" + "="*60)
    print("INTERPRETATION:")
    print("="*60)
    print("- If ALL models show 'no edge': The task is too hard or data has no signal")
    print("- If simple models show edge: Signal exists, LSTM should work better")
    print("- If only complex models show edge: Signal is non-linear, need more data")
    print()


if __name__ == "__main__":
    main()

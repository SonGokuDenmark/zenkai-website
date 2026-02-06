#!/usr/bin/env python3
"""
Test if signal exists WITHIN specific regimes.

Maybe the signal is there but only in certain market conditions.
Also test predicting volatility instead of direction.
"""

import os
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
load_dotenv()
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')


def get_data(timeframe='1h', limit=100000):
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
    """Compute features per symbol."""
    features = []
    for sym in df['symbol'].unique():
        sym_df = df[df['symbol'] == sym].sort_values('open_time').copy()
        if len(sym_df) < 100:
            continue

        sym_df['return_1'] = sym_df['close'].pct_change(1)
        sym_df['return_5'] = sym_df['close'].pct_change(5)
        sym_df['return_10'] = sym_df['close'].pct_change(10)
        sym_df['volatility'] = sym_df['return_1'].rolling(20).std()
        sym_df['volume_ratio'] = sym_df['volume'] / sym_df['volume'].rolling(20).mean()

        delta = sym_df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        sym_df['rsi'] = 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))

        sym_df['close_vs_high20'] = sym_df['close'] / sym_df['high'].rolling(20).max()
        sym_df['close_vs_low20'] = sym_df['close'] / sym_df['low'].rolling(20).min()

        # Trend strength
        sym_df['ema_9'] = sym_df['close'].ewm(span=9).mean()
        sym_df['ema_21'] = sym_df['close'].ewm(span=21).mean()
        sym_df['trend'] = (sym_df['ema_9'] - sym_df['ema_21']) / sym_df['close']

        features.append(sym_df)

    return pd.concat(features, ignore_index=True)


def main():
    print("="*60)
    print("REGIME-SPECIFIC SIGNAL TEST")
    print("="*60)

    df = get_data(timeframe='1h', limit=100000)
    print(f"Loaded {len(df):,} rows")

    df = compute_features(df)
    print(f"After features: {len(df):,} rows")

    # Compute labels
    df['future_return'] = df.groupby('symbol')['close'].transform(
        lambda x: x.shift(-10) / x - 1
    )
    df['future_volatility'] = df.groupby('symbol')['return_1'].transform(
        lambda x: x.shift(-10).rolling(10).std()
    )
    df = df.dropna()

    feature_cols = ['return_1', 'return_5', 'return_10', 'volatility',
                    'volume_ratio', 'rsi', 'close_vs_high20', 'close_vs_low20', 'trend']

    print(f"\n{'='*60}")
    print("TEST 1: Direction prediction BY REGIME")
    print("="*60)

    for regime in ['TRENDING_UP', 'TRENDING_DOWN', 'RANGING', 'HIGH_VOL']:
        regime_df = df[df['regime'] == regime].copy()
        if len(regime_df) < 1000:
            continue

        # Binary: UP if future_return > 1%
        regime_df['label'] = (regime_df['future_return'] > 0.01).astype(int)

        X = regime_df[feature_cols].values
        y = regime_df['label'].values
        mask = ~np.isnan(X).any(axis=1)
        X, y = X[mask], y[mask]

        split_idx = int(len(X) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        random_baseline = max(y_test.mean(), 1 - y_test.mean())

        model = GradientBoostingClassifier(n_estimators=100, max_depth=3)
        model.fit(X_train_s, y_train)
        acc = accuracy_score(y_test, model.predict(X_test_s))
        edge = acc - random_baseline

        status = "[SIGNAL]" if edge > 0.02 else "[no edge]"
        print(f"  {regime:15} n={len(X):,}  baseline={random_baseline:.1%}  acc={acc:.1%}  edge={edge:+.1%} {status}")

    print(f"\n{'='*60}")
    print("TEST 2: VOLATILITY prediction (regression)")
    print("="*60)
    print("(Volatility is often more predictable than direction)")

    X = df[feature_cols].values
    y = df['future_volatility'].values
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y = X[mask], y[mask]

    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Baseline: predict mean
    baseline_r2 = 0  # by definition

    for name, model in [
        ("Ridge Regression", Ridge()),
        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, max_depth=3))
    ]:
        model.fit(X_train_s, y_train)
        r2 = r2_score(y_test, model.predict(X_test_s))
        status = "[SIGNAL]" if r2 > 0.05 else "[weak]" if r2 > 0 else "[none]"
        print(f"  {name:25} R2={r2:.3f} {status}")

    print(f"\n{'='*60}")
    print("TEST 3: REGIME prediction")
    print("="*60)
    print("(Can we at least predict what regime we're in?)")

    # Encode regime as numeric
    regime_map = {'TRENDING_UP': 0, 'TRENDING_DOWN': 1, 'RANGING': 2, 'HIGH_VOL': 3}
    df['regime_num'] = df['regime'].map(regime_map)

    X = df[feature_cols].values
    y = df['regime_num'].values
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y = X[mask], y[mask].astype(int)

    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Random baseline for 4-class
    from collections import Counter
    most_common = Counter(y_test).most_common(1)[0][1] / len(y_test)
    print(f"  Random baseline: {most_common:.1%}")

    model = GradientBoostingClassifier(n_estimators=100, max_depth=3)
    model.fit(X_train_s, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_s))
    edge = acc - most_common
    status = "[SIGNAL]" if edge > 0.05 else "[weak]"
    print(f"  Gradient Boosting         Acc={acc:.1%} (edge={edge:+.1%}) {status}")

    print(f"\n{'='*60}")
    print("CONCLUSIONS")
    print("="*60)
    print("If direction has no signal but volatility/regime does:")
    print("  -> Focus on volatility-adjusted position sizing")
    print("  -> Use regime to filter trades (don't trade in bad regimes)")
    print("  -> The RL agent might learn this implicitly")


if __name__ == "__main__":
    main()

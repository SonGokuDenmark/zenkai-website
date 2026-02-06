#!/usr/bin/env python3
"""
LSTM Model Analysis Pipeline - COMPREHENSIVE

Breaks down model predictions step by step before any profit simulation.
Answers ALL the fundamental questions: WHAT, WHEN, WHY, HOW, WHERE, WHO.

Time is not a factor - thoroughness is.

Usage:
    python analyze_lstm.py --timeframe 1h
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Any
import json

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from collections import defaultdict
from scipy import stats

load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models import LSTMClassifier

# Database config
DB_CONFIG = {
    "host": os.getenv("ZENKAI_DB_HOST", "192.168.0.160"),
    "database": os.getenv("ZENKAI_DB_NAME", "zenkai_data"),
    "user": os.getenv("ZENKAI_DB_USER", "zenkai"),
    "password": os.getenv("ZENKAI_DB_PASSWORD"),
}

REGIME_MAP = {
    "TRENDING_UP": 0,
    "TRENDING_DOWN": 1,
    "RANGING": 2,
    "HIGH_VOL": 3,
}


def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


def load_data(timeframe: str = "1h", limit: int = 500000) -> pd.DataFrame:
    """Load data with all metadata for analysis."""
    conn = get_db_connection()

    query = """
        SELECT
            open_time, symbol,
            open, high, low, close, volume,
            regime,
            -- Trending signals (7)
            signal_macd, signal_rsi_div, signal_turtle,
            signal_ma_cross, signal_stoch, signal_sr, signal_candle,
            conf_macd, conf_rsi_div, conf_turtle,
            conf_ma_cross, conf_stoch, conf_sr, conf_candle,
            -- Ranging signals (7)
            signal_adx_range, signal_bb_squeeze, signal_chop,
            signal_kc_squeeze, signal_rsi_mid, signal_atr_contract, signal_price_channel,
            conf_adx_range, conf_bb_squeeze, conf_chop,
            conf_kc_squeeze, conf_rsi_mid, conf_atr_contract, conf_price_channel
        FROM ohlcv
        WHERE timeframe = %s
          AND regime IS NOT NULL
          AND conf_adx_range IS NOT NULL
        ORDER BY open_time
        LIMIT %s
    """

    df = pd.read_sql_query(query, conn, params=[timeframe, limit])
    conn.close()

    print(f"Loaded {len(df):,} rows")
    return df


def compute_labels(df: pd.DataFrame, forward_bars: int = 10, flat_threshold: float = 0.005) -> pd.DataFrame:
    """Compute forward-looking labels."""
    df = df.copy()

    df["future_return"] = df.groupby("symbol")["close"].transform(
        lambda x: x.shift(-forward_bars) / x - 1
    )

    df["label"] = 1  # FLAT
    df.loc[df["future_return"] > flat_threshold, "label"] = 2  # UP
    df.loc[df["future_return"] < -flat_threshold, "label"] = 0  # DOWN

    df = df.dropna(subset=["future_return"])
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for model."""
    df = df.copy()

    # Store raw close for return calculations later
    df["close_raw"] = df["close"].copy()

    for col in ["open", "high", "low", "close"]:
        df[col] = df.groupby("symbol")[col].transform(
            lambda x: (x - x.rolling(50, min_periods=1).mean()) / x.rolling(50, min_periods=1).std().clip(lower=1e-8)
        )

    df["volume"] = df.groupby("symbol")["volume"].transform(
        lambda x: (x - x.rolling(50, min_periods=1).mean()) / x.rolling(50, min_periods=1).std().clip(lower=1e-8)
    )

    for regime, idx in REGIME_MAP.items():
        df[f"regime_{regime.lower()}"] = (df["regime"] == regime).astype(float)

    signal_cols = [c for c in df.columns if c.startswith("signal_") or c.startswith("conf_")]
    df[signal_cols] = df[signal_cols].fillna(0)
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)

    return df


def create_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_length: int = 50,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Create sequences and return metadata DataFrame aligned with sequences."""
    X_list = []
    y_list = []
    meta_list = []

    for symbol in df["symbol"].unique():
        symbol_df = df[df["symbol"] == symbol].sort_values("open_time").reset_index(drop=True)

        if len(symbol_df) < seq_length + 1:
            continue

        features = symbol_df[feature_cols].values
        labels = symbol_df["label"].values

        for i in range(len(symbol_df) - seq_length):
            X_list.append(features[i:i + seq_length])
            y_list.append(labels[i + seq_length - 1])

            # Keep extensive metadata for analysis
            row = symbol_df.iloc[i + seq_length - 1]
            meta_list.append({
                "symbol": row["symbol"],
                "open_time": row["open_time"],
                "regime": row["regime"],
                "future_return": row["future_return"],
                "close_raw": row.get("close_raw", 0),
                # Trending signals
                "signal_macd": row.get("signal_macd", 0),
                "signal_rsi_div": row.get("signal_rsi_div", 0),
                "signal_turtle": row.get("signal_turtle", 0),
                "signal_ma_cross": row.get("signal_ma_cross", 0),
                "signal_stoch": row.get("signal_stoch", 0),
                "signal_sr": row.get("signal_sr", 0),
                "signal_candle": row.get("signal_candle", 0),
                "conf_macd": row.get("conf_macd", 0),
                "conf_rsi_div": row.get("conf_rsi_div", 0),
                "conf_turtle": row.get("conf_turtle", 0),
                "conf_ma_cross": row.get("conf_ma_cross", 0),
                "conf_stoch": row.get("conf_stoch", 0),
                "conf_sr": row.get("conf_sr", 0),
                "conf_candle": row.get("conf_candle", 0),
                # Ranging signals
                "signal_adx_range": row.get("signal_adx_range", 0),
                "signal_bb_squeeze": row.get("signal_bb_squeeze", 0),
                "signal_chop": row.get("signal_chop", 0),
                "signal_kc_squeeze": row.get("signal_kc_squeeze", 0),
                "signal_rsi_mid": row.get("signal_rsi_mid", 0),
                "signal_atr_contract": row.get("signal_atr_contract", 0),
                "signal_price_channel": row.get("signal_price_channel", 0),
                "conf_adx_range": row.get("conf_adx_range", 0),
                "conf_bb_squeeze": row.get("conf_bb_squeeze", 0),
                "conf_chop": row.get("conf_chop", 0),
                "conf_kc_squeeze": row.get("conf_kc_squeeze", 0),
                "conf_rsi_mid": row.get("conf_rsi_mid", 0),
                "conf_atr_contract": row.get("conf_atr_contract", 0),
                "conf_price_channel": row.get("conf_price_channel", 0),
            })

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    meta_df = pd.DataFrame(meta_list)

    print(f"Created {len(X):,} sequences")
    return X, y, meta_df


def print_section(title: str, level: int = 1):
    """Print section header."""
    if level == 1:
        print("\n" + "=" * 80)
        print(f" {title}")
        print("=" * 80)
    else:
        print(f"\n--- {title} ---")


def analyze_data_quality(meta_df: pd.DataFrame, y: np.ndarray):
    """Analyze the data itself before looking at predictions."""
    print_section("0. DATA QUALITY CHECK")

    # Label distribution
    print("\n[Label Distribution]")
    for cls, name in [(0, "DOWN"), (1, "FLAT"), (2, "UP")]:
        count = (y == cls).sum()
        pct = count / len(y) * 100
        print(f"  {name}: {count:,} ({pct:.1f}%)")

    # Check for class imbalance
    counts = [(y == i).sum() for i in range(3)]
    imbalance_ratio = max(counts) / min(counts)
    print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}x")
    if imbalance_ratio > 2:
        print("  ⚠️ WARNING: Significant class imbalance detected")

    # Symbol distribution
    print("\n[Symbol Distribution]")
    symbol_counts = meta_df["symbol"].value_counts()
    print(f"  Total symbols: {len(symbol_counts)}")
    print(f"  Samples per symbol: {symbol_counts.min():,} - {symbol_counts.max():,}")
    print(f"  Mean: {symbol_counts.mean():,.0f}, Std: {symbol_counts.std():,.0f}")

    # Regime distribution
    print("\n[Regime Distribution]")
    regime_counts = meta_df["regime"].value_counts()
    for regime, count in regime_counts.items():
        pct = count / len(meta_df) * 100
        print(f"  {regime}: {count:,} ({pct:.1f}%)")

    # Time range
    print("\n[Time Range]")
    min_time = pd.to_datetime(meta_df["open_time"], unit='ms').min()
    max_time = pd.to_datetime(meta_df["open_time"], unit='ms').max()
    print(f"  From: {min_time}")
    print(f"  To: {max_time}")
    print(f"  Duration: {(max_time - min_time).days} days")

    # Future return distribution
    print("\n[Future Return Distribution]")
    returns = meta_df["future_return"]
    print(f"  Mean: {returns.mean():.4%}")
    print(f"  Std: {returns.std():.4%}")
    print(f"  Min: {returns.min():.4%}")
    print(f"  Max: {returns.max():.4%}")
    print(f"  Skewness: {stats.skew(returns):.3f}")
    print(f"  Kurtosis: {stats.kurtosis(returns):.3f}")


def analyze_what(predictions: np.ndarray, actuals: np.ndarray, confidences: np.ndarray) -> dict:
    """WHAT is the model predicting? Comprehensive confusion analysis."""
    print_section("1. WHAT is the model predicting?")

    results = {}

    # Overall accuracy
    accuracy = np.mean(predictions == actuals)
    results["accuracy"] = accuracy
    print(f"\nOverall Accuracy: {accuracy:.1%} (random baseline: 33.3%)")
    print(f"Improvement over random: +{(accuracy - 0.333) * 100:.1f} percentage points")

    # Statistical significance test
    n = len(predictions)
    expected = n / 3  # random would get 1/3 correct
    observed_correct = (predictions == actuals).sum()
    chi2 = (observed_correct - expected) ** 2 / expected
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    print(f"Statistical significance: chi2 = {chi2:.2f}, p-value = {p_value:.6f}")
    if p_value < 0.001:
        print("  [YES] Results are statistically significant (p < 0.001)")

    # Class distribution in predictions vs actuals
    print("\n[Prediction Distribution]")
    print(f"{'Class':<10} {'Predicted':<18} {'Actual':<18} {'Ratio':<10}")
    print("-" * 60)
    for cls, name in [(0, "DOWN"), (1, "FLAT"), (2, "UP")]:
        pred_count = (predictions == cls).sum()
        actual_count = (actuals == cls).sum()
        pred_pct = pred_count / len(predictions) * 100
        actual_pct = actual_count / len(actuals) * 100
        ratio = pred_pct / actual_pct if actual_pct > 0 else 0
        print(f"{name:<10} {pred_pct:>6.1f}% ({pred_count:>7,})  {actual_pct:>6.1f}% ({actual_count:>7,})  {ratio:.2f}x")
        results[f"pred_dist_{name}"] = pred_pct
        results[f"actual_dist_{name}"] = actual_pct

    # Confusion Matrix
    print("\n[Confusion Matrix]")
    print("(rows = actual, columns = predicted)")
    print(f"{'':>10} {'DOWN':>12} {'FLAT':>12} {'UP':>12} {'Total':>12} {'Recall':>10}")
    print("-" * 70)

    confusion = {}
    for actual_cls, actual_name in [(0, "DOWN"), (1, "FLAT"), (2, "UP")]:
        row = []
        actual_mask = actuals == actual_cls
        total = actual_mask.sum()
        correct_count = 0
        for pred_cls, pred_name in [(0, "DOWN"), (1, "FLAT"), (2, "UP")]:
            count = ((actuals == actual_cls) & (predictions == pred_cls)).sum()
            pct = count / total * 100 if total > 0 else 0
            row.append(f"{pct:>5.1f}%")
            confusion[(actual_name, pred_name)] = pct
            if actual_cls == pred_cls:
                correct_count = count
        recall = correct_count / total * 100 if total > 0 else 0
        print(f"{actual_name:>10} {row[0]:>12} {row[1]:>12} {row[2]:>12} {total:>12,} {recall:>9.1f}%")

    results["confusion"] = confusion

    # Precision (what % of predictions are correct)
    print("\n[Precision per Class]")
    for pred_cls, pred_name in [(0, "DOWN"), (1, "FLAT"), (2, "UP")]:
        pred_mask = predictions == pred_cls
        if pred_mask.sum() > 0:
            precision = (actuals[pred_mask] == pred_cls).mean() * 100
            print(f"  When model predicts {pred_name}: {precision:.1f}% are actually {pred_name}")
            results[f"precision_{pred_name}"] = precision

    # F1 scores
    print("\n[F1 Scores]")
    for cls, name in [(0, "DOWN"), (1, "FLAT"), (2, "UP")]:
        tp = ((predictions == cls) & (actuals == cls)).sum()
        fp = ((predictions == cls) & (actuals != cls)).sum()
        fn = ((predictions != cls) & (actuals == cls)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"  {name}: F1 = {f1:.3f} (Precision: {precision:.3f}, Recall: {recall:.3f})")
        results[f"f1_{name}"] = f1

    # Key insights
    print("\n[Key Insights]")

    # Directional confusion (UP predicted as DOWN or vice versa) - THIS IS BAD
    up_as_down = confusion.get(("UP", "DOWN"), 0)
    down_as_up = confusion.get(("DOWN", "UP"), 0)
    directional_error = (up_as_down + down_as_up) / 2
    status = "GOOD" if directional_error < 15 else "ACCEPTABLE" if directional_error < 25 else "CONCERNING" if directional_error < 35 else "BAD"
    print(f"  Directional confusion (UP↔DOWN): {directional_error:.1f}% - {status}")
    results["directional_error"] = directional_error

    # FLAT confusion
    up_as_flat = confusion.get(("UP", "FLAT"), 0)
    down_as_flat = confusion.get(("DOWN", "FLAT"), 0)
    flat_as_up = confusion.get(("FLAT", "UP"), 0)
    flat_as_down = confusion.get(("FLAT", "DOWN"), 0)
    trend_to_flat = (up_as_flat + down_as_flat) / 2
    flat_to_trend = (flat_as_up + flat_as_down) / 2
    print(f"  Trend→FLAT confusion: {trend_to_flat:.1f}% (expected with trend-focused training)")
    print(f"  FLAT→Trend confusion: {flat_to_trend:.1f}%")

    return results


def analyze_when_regime(predictions: np.ndarray, actuals: np.ndarray, meta_df: pd.DataFrame, confidences: np.ndarray) -> dict:
    """WHEN does it work? By regime - with deep dive."""
    print_section("2. WHEN does it work? (by Regime)")

    results = {}

    print(f"\n{'Regime':<15} {'Accuracy':>10} {'Samples':>12} {'vs Random':>12} {'Avg Conf':>10}")
    print("-" * 65)

    regime_stats = {}
    for regime in ["TRENDING_UP", "TRENDING_DOWN", "RANGING", "HIGH_VOL"]:
        mask = meta_df["regime"] == regime
        if mask.sum() == 0:
            continue

        regime_preds = predictions[mask]
        regime_actuals = actuals[mask]
        regime_conf = confidences[mask]
        acc = np.mean(regime_preds == regime_actuals)
        improvement = (acc - 0.333) * 100
        avg_conf = regime_conf.mean()

        regime_stats[regime] = {
            "accuracy": acc,
            "samples": int(mask.sum()),
            "improvement": improvement,
            "avg_confidence": float(avg_conf)
        }

        status = "+++" if improvement > 12 else "++" if improvement > 6 else "+" if improvement > 0 else "-"
        print(f"{regime:<15} {acc:>9.1%} {mask.sum():>12,} {improvement:>+10.1f}pp {avg_conf:>9.1%} {status}")

    results["regime_stats"] = regime_stats

    # Per-class accuracy within each regime
    print("\n[Per-Class Accuracy by Regime]")
    print(f"{'Regime':<15} {'DOWN':>10} {'FLAT':>10} {'UP':>10}")
    print("-" * 50)

    for regime in ["TRENDING_UP", "TRENDING_DOWN", "RANGING", "HIGH_VOL"]:
        mask = meta_df["regime"] == regime
        if mask.sum() == 0:
            continue

        regime_preds = predictions[mask]
        regime_actuals = actuals[mask]

        accs = []
        for cls in [0, 1, 2]:
            cls_mask = regime_actuals == cls
            if cls_mask.sum() > 0:
                cls_acc = np.mean(regime_preds[cls_mask] == cls) * 100
            else:
                cls_acc = 0
            accs.append(f"{cls_acc:>5.1f}%")

        print(f"{regime:<15} {accs[0]:>10} {accs[1]:>10} {accs[2]:>10}")

    # Key insight
    if regime_stats:
        best_regime = max(regime_stats, key=lambda x: regime_stats[x]["accuracy"])
        worst_regime = min(regime_stats, key=lambda x: regime_stats[x]["accuracy"])
        results["best_regime"] = best_regime
        results["worst_regime"] = worst_regime
        print(f"\n[Insight]")
        print(f"  Best regime: {best_regime} ({regime_stats[best_regime]['accuracy']:.1%})")
        print(f"  Worst regime: {worst_regime} ({regime_stats[worst_regime]['accuracy']:.1%})")
        spread = regime_stats[best_regime]['accuracy'] - regime_stats[worst_regime]['accuracy']
        print(f"  Spread: {spread*100:.1f} percentage points")

    return results


def analyze_when_symbol(predictions: np.ndarray, actuals: np.ndarray, meta_df: pd.DataFrame, confidences: np.ndarray) -> dict:
    """WHEN does it work? By symbol - comprehensive."""
    print_section("3. WHEN does it work? (by Symbol)")

    results = {}
    symbol_stats = {}

    for symbol in meta_df["symbol"].unique():
        mask = meta_df["symbol"] == symbol
        if mask.sum() < 100:  # Skip symbols with too few samples
            continue

        symbol_preds = predictions[mask]
        symbol_actuals = actuals[mask]
        symbol_conf = confidences[mask]
        acc = np.mean(symbol_preds == symbol_actuals)

        # Per-class breakdown
        class_accs = {}
        for cls, name in [(0, "DOWN"), (1, "FLAT"), (2, "UP")]:
            cls_mask = symbol_actuals == cls
            if cls_mask.sum() > 10:
                class_accs[name] = float(np.mean(symbol_preds[cls_mask] == cls))

        symbol_stats[symbol] = {
            "accuracy": float(acc),
            "samples": int(mask.sum()),
            "avg_confidence": float(symbol_conf.mean()),
            "class_accuracies": class_accs
        }

    results["symbol_stats"] = symbol_stats

    # Sort by accuracy
    sorted_symbols = sorted(symbol_stats.keys(), key=lambda x: symbol_stats[x]["accuracy"], reverse=True)

    print(f"\n{'Symbol':<14} {'Accuracy':>10} {'Samples':>10} {'Avg Conf':>10} {'DOWN':>8} {'FLAT':>8} {'UP':>8}")
    print("-" * 80)

    # Top 10
    print("Top 10:")
    for symbol in sorted_symbols[:10]:
        sym_info = symbol_stats[symbol]
        ca = sym_info["class_accuracies"]
        down_acc = f"{ca.get('DOWN', 0)*100:.0f}%" if "DOWN" in ca else "N/A"
        flat_acc = f"{ca.get('FLAT', 0)*100:.0f}%" if "FLAT" in ca else "N/A"
        up_acc = f"{ca.get('UP', 0)*100:.0f}%" if "UP" in ca else "N/A"
        print(f"  {symbol:<14} {sym_info['accuracy']:>9.1%} {sym_info['samples']:>10,} {sym_info['avg_confidence']:>9.1%} {down_acc:>8} {flat_acc:>8} {up_acc:>8}")

    # Bottom 10
    print("\nBottom 10:")
    for symbol in sorted_symbols[-10:]:
        sym_info = symbol_stats[symbol]
        ca = sym_info["class_accuracies"]
        down_acc = f"{ca.get('DOWN', 0)*100:.0f}%" if "DOWN" in ca else "N/A"
        flat_acc = f"{ca.get('FLAT', 0)*100:.0f}%" if "FLAT" in ca else "N/A"
        up_acc = f"{ca.get('UP', 0)*100:.0f}%" if "UP" in ca else "N/A"
        print(f"  {symbol:<14} {sym_info['accuracy']:>9.1%} {sym_info['samples']:>10,} {sym_info['avg_confidence']:>9.1%} {down_acc:>8} {flat_acc:>8} {up_acc:>8}")

    # Statistical summary
    accs = [symbol_stats[s]["accuracy"] for s in symbol_stats]
    print(f"\n[Statistical Summary]")
    print(f"  Symbols analyzed: {len(accs)}")
    print(f"  Accuracy range: {min(accs):.1%} - {max(accs):.1%}")
    print(f"  Mean: {np.mean(accs):.1%}")
    print(f"  Median: {np.median(accs):.1%}")
    print(f"  Std deviation: {np.std(accs):.1%}")

    # Correlation between samples and accuracy
    samples = [symbol_stats[s]["samples"] for s in symbol_stats]
    corr, p_val = stats.pearsonr(samples, accs)
    print(f"  Correlation (samples vs accuracy): {corr:.3f} (p={p_val:.4f})")

    return results


def analyze_when_time(predictions: np.ndarray, actuals: np.ndarray, meta_df: pd.DataFrame) -> dict:
    """WHEN does it work? By time patterns."""
    print_section("4. WHEN does it work? (by Time)")

    results = {}

    # Convert timestamps
    meta_df = meta_df.copy()
    meta_df["datetime"] = pd.to_datetime(meta_df["open_time"], unit='ms')
    meta_df["hour"] = meta_df["datetime"].dt.hour
    meta_df["day_of_week"] = meta_df["datetime"].dt.dayofweek  # 0=Monday
    meta_df["day_name"] = meta_df["datetime"].dt.day_name()

    correct = predictions == actuals

    # By hour of day
    print("\n[Accuracy by Hour of Day (UTC)]")
    hour_stats = {}
    hours = sorted(meta_df["hour"].unique())

    # Print in rows of 6
    print(f"{'Hour':<6}" + "".join([f"{h:>8}" for h in hours[:12]]))
    accs_row1 = []
    for h in hours[:12]:
        mask = meta_df["hour"] == h
        if mask.sum() > 0:
            acc = correct[mask].mean()
            hour_stats[int(h)] = {"accuracy": float(acc), "samples": int(mask.sum())}
            accs_row1.append(f"{acc*100:>7.1f}%")
        else:
            accs_row1.append("    N/A")
    print(f"{'Acc':<6}" + "".join(accs_row1))

    if len(hours) > 12:
        print(f"{'Hour':<6}" + "".join([f"{h:>8}" for h in hours[12:]]))
        accs_row2 = []
        for h in hours[12:]:
            mask = meta_df["hour"] == h
            if mask.sum() > 0:
                acc = correct[mask].mean()
                hour_stats[int(h)] = {"accuracy": float(acc), "samples": int(mask.sum())}
                accs_row2.append(f"{acc*100:>7.1f}%")
            else:
                accs_row2.append("    N/A")
        print(f"{'Acc':<6}" + "".join(accs_row2))

    results["hour_stats"] = hour_stats

    # Best/worst hours
    if hour_stats:
        best_hour = max(hour_stats, key=lambda x: hour_stats[x]["accuracy"])
        worst_hour = min(hour_stats, key=lambda x: hour_stats[x]["accuracy"])
        print(f"\n  Best hour: {best_hour}:00 UTC ({hour_stats[best_hour]['accuracy']:.1%})")
        print(f"  Worst hour: {worst_hour}:00 UTC ({hour_stats[worst_hour]['accuracy']:.1%})")

    # By day of week
    print("\n[Accuracy by Day of Week]")
    print(f"{'Day':<12} {'Accuracy':>10} {'Samples':>12}")
    print("-" * 40)

    day_stats = {}
    for dow in range(7):
        mask = meta_df["day_of_week"] == dow
        if mask.sum() > 0:
            acc = correct[mask].mean()
            day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][dow]
            day_stats[day_name] = {"accuracy": float(acc), "samples": int(mask.sum())}
            print(f"  {day_name:<12} {acc:>9.1%} {mask.sum():>12,}")

    results["day_stats"] = day_stats

    # Time progression (is model degrading over time?)
    print("\n[Accuracy Over Time (by Date)]")
    meta_df["date"] = meta_df["datetime"].dt.date
    dates = sorted(meta_df["date"].unique())

    if len(dates) > 7:
        # Group into weekly chunks
        chunk_size = len(dates) // 5
        for i in range(5):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < 4 else len(dates)
            chunk_dates = dates[start_idx:end_idx]
            mask = meta_df["date"].isin(chunk_dates)
            if mask.sum() > 0:
                acc = correct[mask].mean()
                print(f"  {chunk_dates[0]} to {chunk_dates[-1]}: {acc:.1%} ({mask.sum():,} samples)")

    return results


def analyze_how_confidence(predictions: np.ndarray, actuals: np.ndarray, confidences: np.ndarray, meta_df: pd.DataFrame) -> dict:
    """HOW confident is it? Deep confidence calibration analysis."""
    print_section("5. HOW confident is it? (Confidence Analysis)")

    results = {}

    print("\n[Confidence Distribution]")
    print(f"  Min: {confidences.min():.3f}")
    print(f"  Max: {confidences.max():.3f}")
    print(f"  Mean: {confidences.mean():.3f}")
    print(f"  Median: {np.median(confidences):.3f}")
    print(f"  Std: {confidences.std():.3f}")

    # Percentiles
    print("\n[Confidence Percentiles]")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(confidences, p)
        print(f"  {p}th percentile: {val:.3f}")

    # Accuracy by confidence bucket
    print(f"\n[Accuracy by Confidence Bucket]")
    print(f"{'Confidence':<15} {'Accuracy':>10} {'Samples':>12} {'% of Total':>12} {'Lift':>10}")
    print("-" * 65)

    confidence_stats = {}
    thresholds = [0.33, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    baseline_acc = np.mean(predictions == actuals)

    for i, thresh in enumerate(thresholds):
        upper = thresholds[i + 1] if i + 1 < len(thresholds) else 1.0
        mask = (confidences >= thresh) & (confidences < upper)

        if mask.sum() < 100:
            continue

        bucket_preds = predictions[mask]
        bucket_actuals = actuals[mask]
        acc = np.mean(bucket_preds == bucket_actuals)
        pct_total = mask.sum() / len(predictions) * 100
        lift = (acc / baseline_acc - 1) * 100

        label = f"{thresh:.0%}-{upper:.0%}"
        confidence_stats[label] = {
            "accuracy": float(acc),
            "samples": int(mask.sum()),
            "pct_total": float(pct_total),
            "lift": float(lift)
        }

        lift_str = f"+{lift:.0f}%" if lift > 0 else f"{lift:.0f}%"
        print(f"{label:<15} {acc:>9.1%} {mask.sum():>12,} {pct_total:>11.1f}% {lift_str:>10}")

    results["confidence_stats"] = confidence_stats

    # Cumulative threshold analysis
    print(f"\n[Cumulative Threshold Analysis - If we only trade above X confidence]")
    print(f"{'Threshold':<12} {'Accuracy':>10} {'Trades':>12} {'% Kept':>12} {'Exp. Value':>12}")
    print("-" * 65)

    cumulative_stats = {}
    for thresh in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        mask = confidences >= thresh
        if mask.sum() < 100:
            continue

        acc = np.mean(predictions[mask] == actuals[mask])
        pct_kept = mask.sum() / len(predictions) * 100

        # Simple expected value: if correct +1%, if wrong -1.2%
        exp_val = acc * 0.01 + (1 - acc) * (-0.012)

        cumulative_stats[str(thresh)] = {
            "accuracy": float(acc),
            "trades": int(mask.sum()),
            "pct_kept": float(pct_kept),
            "expected_value": float(exp_val)
        }

        ev_str = f"{exp_val*100:+.3f}%"
        print(f">= {thresh:.0%}       {acc:>9.1%} {mask.sum():>12,} {pct_kept:>11.1f}% {ev_str:>12}")

    results["cumulative_stats"] = cumulative_stats

    # Confidence calibration (is 70% confidence actually 70% accurate?)
    print(f"\n[Confidence Calibration - Is confidence well-calibrated?]")
    print(f"{'Expected':>12} {'Observed':>12} {'Gap':>10} {'Status':<15}")
    print("-" * 55)

    calibration = {}
    for expected in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        mask = (confidences >= expected - 0.05) & (confidences < expected + 0.05)
        if mask.sum() < 100:
            continue
        observed = np.mean(predictions[mask] == actuals[mask])
        gap = observed - expected
        calibration[str(expected)] = {"expected": expected, "observed": float(observed), "gap": float(gap)}

        status = "Well calibrated" if abs(gap) < 0.05 else "Over-confident" if gap < 0 else "Under-confident"
        print(f"{expected:>11.0%} {observed:>11.1%} {gap:>+9.1%} {status:<15}")

    results["calibration"] = calibration

    # Confidence by class prediction
    print(f"\n[Average Confidence by Predicted Class]")
    for cls, name in [(0, "DOWN"), (1, "FLAT"), (2, "UP")]:
        mask = predictions == cls
        if mask.sum() > 0:
            avg_conf = confidences[mask].mean()
            correct_mask = mask & (actuals == cls)
            wrong_mask = mask & (actuals != cls)
            conf_when_correct = confidences[correct_mask].mean() if correct_mask.sum() > 0 else 0
            conf_when_wrong = confidences[wrong_mask].mean() if wrong_mask.sum() > 0 else 0
            print(f"  {name}: Avg={avg_conf:.1%}, When Correct={conf_when_correct:.1%}, When Wrong={conf_when_wrong:.1%}")

    return results


def analyze_why_signals(predictions: np.ndarray, actuals: np.ndarray, meta_df: pd.DataFrame, confidences: np.ndarray) -> dict:
    """WHY is it predicting what it predicts? Comprehensive signal analysis."""
    print_section("6. WHY is it predicting? (Signal Analysis)")

    results = {}
    correct = predictions == actuals

    signal_cols = [c for c in meta_df.columns if c.startswith("signal_")]
    conf_cols = [c for c in meta_df.columns if c.startswith("conf_")]

    # Basic signal correlation
    print(f"\n[Accuracy when Expert Signal is Active (non-zero)]")
    print(f"{'Signal':<15} {'Active Acc':>12} {'Inactive Acc':>13} {'Diff':>10} {'Active %':>10} {'Insight':<20}")
    print("-" * 85)

    signal_analysis = {}
    for col in signal_cols:
        signal_name = col.replace("signal_", "").upper()

        active_mask = meta_df[col] != 0
        inactive_mask = meta_df[col] == 0

        if active_mask.sum() < 100 or inactive_mask.sum() < 100:
            continue

        active_acc = np.mean(correct[active_mask])
        inactive_acc = np.mean(correct[inactive_mask])
        diff = active_acc - inactive_acc
        active_pct = active_mask.sum() / len(meta_df) * 100

        # Additional insights
        # Accuracy when signal agrees with prediction
        bullish_mask = meta_df[col] > 0
        bearish_mask = meta_df[col] < 0

        signal_analysis[signal_name] = {
            "active_acc": float(active_acc),
            "inactive_acc": float(inactive_acc),
            "diff": float(diff),
            "active_pct": float(active_pct),
            "bullish_count": int(bullish_mask.sum()),
            "bearish_count": int(bearish_mask.sum())
        }

        # Determine insight
        if diff > 0.05:
            insight = "STRONG positive"
        elif diff > 0.02:
            insight = "Moderate positive"
        elif diff < -0.05:
            insight = "STRONG negative!"
        elif diff < -0.02:
            insight = "Moderate negative"
        else:
            insight = "Neutral"

        indicator = "↑↑" if diff > 0.05 else "↑" if diff > 0.02 else "↓↓" if diff < -0.05 else "↓" if diff < -0.02 else "="
        print(f"{signal_name:<15} {active_acc:>11.1%} {inactive_acc:>12.1%} {diff:>+9.1%} {active_pct:>9.1f}% {insight:<20} {indicator}")

    results["signal_analysis"] = signal_analysis

    # Signal direction analysis
    print(f"\n[Signal Direction Analysis]")
    print(f"{'Signal':<15} {'Bullish Acc':>12} {'Bearish Acc':>12} {'Neutral Acc':>12}")
    print("-" * 55)

    for col in signal_cols:
        signal_name = col.replace("signal_", "").upper()

        bullish_mask = meta_df[col] > 0
        bearish_mask = meta_df[col] < 0
        neutral_mask = meta_df[col] == 0

        accs = []
        for mask, name in [(bullish_mask, "Bull"), (bearish_mask, "Bear"), (neutral_mask, "Neutral")]:
            if mask.sum() > 100:
                acc = np.mean(correct[mask])
                accs.append(f"{acc*100:>5.1f}%")
            else:
                accs.append("  N/A")

        if bullish_mask.sum() > 100 or bearish_mask.sum() > 100:
            print(f"{signal_name:<15} {accs[0]:>12} {accs[1]:>12} {accs[2]:>12}")

    # Signal agreement analysis
    print(f"\n[Accuracy by Number of Active Signals]")
    signal_counts = meta_df[signal_cols].apply(lambda row: (row != 0).sum(), axis=1)

    print(f"{'# Signals':>12} {'Accuracy':>10} {'Samples':>12} {'% Total':>10} {'Insight':<20}")
    print("-" * 70)

    agreement_stats = {}
    for n_signals in sorted(signal_counts.unique()):
        mask = signal_counts == n_signals
        if mask.sum() < 100:
            continue
        acc = np.mean(correct[mask])
        pct = mask.sum() / len(predictions) * 100
        agreement_stats[int(n_signals)] = {"accuracy": float(acc), "samples": int(mask.sum()), "pct": float(pct)}

        if n_signals == 0:
            insight = "No expert input"
        elif n_signals >= 4:
            insight = "Strong consensus"
        elif n_signals >= 2:
            insight = "Some agreement"
        else:
            insight = "Single signal"

        print(f"{int(n_signals):>12} {acc:>9.1%} {mask.sum():>12,} {pct:>9.1f}% {insight:<20}")

    results["agreement_stats"] = agreement_stats

    # Signal consensus (when signals agree on direction)
    print(f"\n[Signal Consensus Analysis]")

    # Count bullish and bearish signals
    bullish_count = meta_df[signal_cols].apply(lambda row: (row > 0).sum(), axis=1)
    bearish_count = meta_df[signal_cols].apply(lambda row: (row < 0).sum(), axis=1)

    # Strong bullish consensus (3+ bullish, 0 bearish)
    strong_bull_mask = (bullish_count >= 3) & (bearish_count == 0)
    strong_bear_mask = (bearish_count >= 3) & (bullish_count == 0)
    mixed_mask = (bullish_count >= 1) & (bearish_count >= 1)

    for mask, name in [(strong_bull_mask, "Strong Bullish (3+ bull, 0 bear)"),
                       (strong_bear_mask, "Strong Bearish (3+ bear, 0 bull)"),
                       (mixed_mask, "Mixed signals")]:
        if mask.sum() > 100:
            acc = np.mean(correct[mask])
            print(f"  {name}: {acc:.1%} accuracy ({mask.sum():,} samples)")

    return results


def analyze_where_fails(predictions: np.ndarray, actuals: np.ndarray, meta_df: pd.DataFrame, confidences: np.ndarray) -> dict:
    """WHERE does it fail? Comprehensive error analysis."""
    print_section("7. WHERE does it fail? (Error Analysis)")

    results = {}
    incorrect = predictions != actuals
    correct = ~incorrect

    # Overall error breakdown
    print(f"\n[Error Type Breakdown]")
    print(f"{'Error Type':<30} {'Count':>10} {'% of All':>12} {'% of Errors':>12}")
    print("-" * 70)

    error_types = {
        "UP → DOWN (worst)": ((actuals == 2) & (predictions == 0)),
        "DOWN → UP (worst)": ((actuals == 0) & (predictions == 2)),
        "UP → FLAT": ((actuals == 2) & (predictions == 1)),
        "DOWN → FLAT": ((actuals == 0) & (predictions == 1)),
        "FLAT → UP": ((actuals == 1) & (predictions == 2)),
        "FLAT → DOWN": ((actuals == 1) & (predictions == 0)),
    }

    total_errors = incorrect.sum()
    error_breakdown = {}
    for error_name, error_mask in error_types.items():
        count = error_mask.sum()
        pct_all = count / len(predictions) * 100
        pct_errors = count / total_errors * 100 if total_errors > 0 else 0
        error_breakdown[error_name] = {"count": int(count), "pct_all": float(pct_all), "pct_errors": float(pct_errors)}
        print(f"{error_name:<30} {count:>10,} {pct_all:>11.2f}% {pct_errors:>11.1f}%")

    results["error_breakdown"] = error_breakdown

    # Directional errors deep dive
    print(f"\n[Directional Errors Deep Dive]")
    up_as_down = error_types["UP → DOWN (worst)"]
    down_as_up = error_types["DOWN → UP (worst)"]
    directional_error_mask = up_as_down | down_as_up

    total_directional = directional_error_mask.sum()
    print(f"Total directional errors: {total_directional:,} ({total_directional/len(predictions)*100:.2f}%)")
    results["total_directional_errors"] = int(total_directional)

    # When do directional errors happen?
    print(f"\n[Directional Errors by Regime]")
    print(f"{'Regime':<15} {'Dir Error Rate':>15} {'Count':>10} {'% of Dir Errors':>15}")
    print("-" * 60)

    regime_dir_errors = {}
    for regime in ["TRENDING_UP", "TRENDING_DOWN", "RANGING", "HIGH_VOL"]:
        regime_mask = meta_df["regime"] == regime
        if regime_mask.sum() == 0:
            continue
        dir_error_in_regime = (directional_error_mask & regime_mask).sum()
        total_in_regime = regime_mask.sum()
        rate = dir_error_in_regime / total_in_regime * 100 if total_in_regime > 0 else 0
        pct_of_total = dir_error_in_regime / total_directional * 100 if total_directional > 0 else 0

        regime_dir_errors[regime] = {"rate": float(rate), "count": int(dir_error_in_regime), "pct_of_total": float(pct_of_total)}
        print(f"{regime:<15} {rate:>14.2f}% {dir_error_in_regime:>10,} {pct_of_total:>14.1f}%")

    results["regime_dir_errors"] = regime_dir_errors

    # Confidence of errors
    print(f"\n[Confidence Distribution of Errors]")
    correct_conf = confidences[correct]
    incorrect_conf = confidences[incorrect]
    dir_error_conf = confidences[directional_error_mask]

    print(f"  Correct predictions: Mean conf = {correct_conf.mean():.3f}, Median = {np.median(correct_conf):.3f}")
    print(f"  All errors: Mean conf = {incorrect_conf.mean():.3f}, Median = {np.median(incorrect_conf):.3f}")
    print(f"  Directional errors: Mean conf = {dir_error_conf.mean():.3f}, Median = {np.median(dir_error_conf):.3f}")

    # High-confidence errors (most dangerous)
    print(f"\n[High-Confidence Errors (Confidence >= 60%)]")
    high_conf_mask = confidences >= 0.6
    high_conf_errors = incorrect & high_conf_mask
    high_conf_dir_errors = directional_error_mask & high_conf_mask

    print(f"  All high-conf errors: {high_conf_errors.sum():,} ({high_conf_errors.sum()/high_conf_mask.sum()*100:.1f}% of high-conf predictions)")
    print(f"  High-conf directional errors: {high_conf_dir_errors.sum():,}")

    results["high_conf_errors"] = int(high_conf_errors.sum())
    results["high_conf_dir_errors"] = int(high_conf_dir_errors.sum())

    # Error clustering by symbol
    print(f"\n[Symbols with Highest Directional Error Rates]")
    print(f"{'Symbol':<14} {'Dir Error Rate':>15} {'Count':>10} {'Total Samples':>15}")
    print("-" * 60)

    symbol_errors = {}
    for symbol in meta_df["symbol"].unique():
        symbol_mask = meta_df["symbol"] == symbol
        if symbol_mask.sum() < 100:
            continue
        dir_errors = (directional_error_mask & symbol_mask).sum()
        rate = dir_errors / symbol_mask.sum() * 100
        symbol_errors[symbol] = {"rate": rate, "count": dir_errors, "total": symbol_mask.sum()}

    # Sort by rate and show worst 10
    worst_symbols = sorted(symbol_errors.keys(), key=lambda x: symbol_errors[x]["rate"], reverse=True)[:10]
    for symbol in worst_symbols:
        err_info = symbol_errors[symbol]
        print(f"  {symbol:<14} {err_info['rate']:>14.2f}% {err_info['count']:>10,} {err_info['total']:>15,}")

    results["symbol_errors"] = {k: {kk: float(vv) if isinstance(vv, float) else vv for kk, vv in v.items()}
                                for k, v in symbol_errors.items()}

    return results


def analyze_profitability_potential(predictions: np.ndarray, actuals: np.ndarray, confidences: np.ndarray, meta_df: pd.DataFrame) -> dict:
    """Analyze potential profitability with different filters applied."""
    print_section("8. PROFITABILITY POTENTIAL (Pre-Monte Carlo)")

    results = {}

    # Basic expected value calculation
    print(f"\n[Basic Expected Value Calculation]")
    print("Assumptions: +1% per correct, -1.2% per wrong (asymmetric)")

    acc = np.mean(predictions == actuals)
    ev_basic = acc * 0.01 + (1 - acc) * (-0.012)
    print(f"\n  Overall accuracy: {acc:.1%}")
    print(f"  Expected value per trade: {ev_basic*100:+.4f}%")
    print(f"  Status: {'PROFITABLE' if ev_basic > 0 else 'UNPROFITABLE'}")

    results["basic_ev"] = float(ev_basic)

    # Break-even analysis
    print(f"\n[Break-Even Analysis]")
    # With 1:1.2 risk/reward ratio, break-even accuracy is:
    # acc * 1 + (1-acc) * (-1.2) = 0
    # acc - 1.2 + 1.2*acc = 0
    # 2.2*acc = 1.2
    # acc = 1.2/2.2 = 54.5%
    breakeven = 1.2 / 2.2
    print(f"  With 1:1.2 R:R, break-even accuracy = {breakeven:.1%}")
    print(f"  Current accuracy: {acc:.1%}")
    print(f"  Gap to break-even: {(acc - breakeven)*100:+.1f} percentage points")

    # What R:R do we need to be profitable?
    # acc * win + (1-acc) * (-loss) = 0
    # win * acc = loss * (1-acc)
    # win/loss = (1-acc)/acc
    if acc > 0:
        required_rr = (1 - acc) / acc
        print(f"  Required R:R for profitability at {acc:.1%} accuracy: 1:{required_rr:.2f}")

    # Expected value with different filters
    print(f"\n[Expected Value with Filters Applied]")
    print(f"{'Filter':<40} {'Accuracy':>10} {'Trades':>12} {'EV/Trade':>12} {'Status':<12}")
    print("-" * 90)

    filter_results = {}

    # Various filters
    filters = [
        ("No filter (baseline)", np.ones(len(predictions), dtype=bool)),
        ("Confidence >= 50%", confidences >= 0.5),
        ("Confidence >= 60%", confidences >= 0.6),
        ("Confidence >= 70%", confidences >= 0.7),
        ("TRENDING regimes only", meta_df["regime"].isin(["TRENDING_UP", "TRENDING_DOWN"])),
        ("Not RANGING", meta_df["regime"] != "RANGING"),
        ("Conf >= 60% + Trending", (confidences >= 0.6) & meta_df["regime"].isin(["TRENDING_UP", "TRENDING_DOWN"])),
        ("Conf >= 50% + Not Ranging", (confidences >= 0.5) & (meta_df["regime"] != "RANGING")),
    ]

    for filter_name, filter_mask in filters:
        if filter_mask.sum() < 100:
            continue

        filtered_acc = np.mean(predictions[filter_mask] == actuals[filter_mask])
        filtered_ev = filtered_acc * 0.01 + (1 - filtered_acc) * (-0.012)
        n_trades = filter_mask.sum()
        pct_kept = n_trades / len(predictions) * 100

        status = "PROFITABLE" if filtered_ev > 0 else "Break-even" if abs(filtered_ev) < 0.0001 else "Unprofitable"
        ev_str = f"{filtered_ev*100:+.4f}%"

        filter_results[filter_name] = {
            "accuracy": float(filtered_acc),
            "trades": int(n_trades),
            "pct_kept": float(pct_kept),
            "ev_per_trade": float(filtered_ev),
            "profitable": filtered_ev > 0
        }

        print(f"{filter_name:<40} {filtered_acc:>9.1%} {n_trades:>12,} {ev_str:>12} {status:<12}")

    results["filter_results"] = filter_results

    # Directional-only analysis (ignore FLAT)
    print(f"\n[Directional Trades Only (UP or DOWN predictions)]")
    directional_mask = predictions != 1  # Not FLAT predictions

    if directional_mask.sum() > 100:
        dir_acc = np.mean(predictions[directional_mask] == actuals[directional_mask])
        dir_ev = dir_acc * 0.01 + (1 - dir_acc) * (-0.012)

        print(f"  Directional predictions only: {directional_mask.sum():,} trades")
        print(f"  Accuracy: {dir_acc:.1%}")
        print(f"  Expected value: {dir_ev*100:+.4f}%")

        results["directional_only"] = {
            "accuracy": float(dir_acc),
            "trades": int(directional_mask.sum()),
            "ev": float(dir_ev)
        }

    # Actual return simulation (using real future returns)
    print(f"\n[Simulated Returns Using Actual Future Returns]")

    # When we predict UP and it goes UP, we capture the positive return
    # When we predict DOWN and it goes DOWN, we capture the positive return (from shorting)
    # When wrong, we lose

    returns = meta_df["future_return"].values

    # Simplified: if prediction matches actual, we "capture" the absolute return
    # if prediction doesn't match, we "lose" the absolute return
    correct_mask = predictions == actuals
    simulated_returns = np.where(correct_mask, np.abs(returns), -np.abs(returns))

    print(f"  Mean simulated return per trade: {simulated_returns.mean()*100:+.4f}%")
    print(f"  Median simulated return: {np.median(simulated_returns)*100:+.4f}%")
    print(f"  Total simulated return: {simulated_returns.sum()*100:+.2f}%")

    results["simulated_returns"] = {
        "mean": float(simulated_returns.mean()),
        "median": float(np.median(simulated_returns)),
        "total": float(simulated_returns.sum())
    }

    return results


def final_summary(all_results: dict):
    """Print comprehensive final summary and actionable recommendations."""
    print_section("9. COMPREHENSIVE SUMMARY & RECOMMENDATIONS")

    what_results = all_results.get("what", {})
    regime_results = all_results.get("regime", {})
    confidence_results = all_results.get("confidence", {})
    error_results = all_results.get("errors", {})
    profit_results = all_results.get("profitability", {})

    print(f"\n{'='*80}")
    print(f" KEY FINDINGS")
    print(f"{'='*80}")

    # 1. Model Performance Summary
    accuracy = what_results.get("accuracy", 0)
    print(f"\n[Model Performance]")
    print(f"  Overall Accuracy: {accuracy:.1%} (vs 33.3% random)")
    print(f"  Improvement: +{(accuracy - 0.333)*100:.1f} percentage points")

    f1_down = what_results.get("f1_DOWN", 0)
    f1_flat = what_results.get("f1_FLAT", 0)
    f1_up = what_results.get("f1_UP", 0)
    print(f"  F1 Scores: DOWN={f1_down:.3f}, FLAT={f1_flat:.3f}, UP={f1_up:.3f}")

    dir_error = what_results.get("directional_error", 0)
    print(f"  Directional Error Rate: {dir_error:.1f}%")

    # 2. Best Conditions
    print(f"\n[Best Conditions for Trading]")
    regime_stats = regime_results.get("regime_stats", {})
    if regime_stats:
        best_regime = max(regime_stats, key=lambda x: regime_stats[x]["accuracy"])
        print(f"  Best regime: {best_regime} ({regime_stats[best_regime]['accuracy']:.1%})")

    cum_stats = confidence_results.get("cumulative_stats", {})
    best_thresh = None
    best_ev = -999
    for thresh, stats in cum_stats.items():
        if stats["expected_value"] > best_ev:
            best_ev = stats["expected_value"]
            best_thresh = thresh
    if best_thresh:
        print(f"  Best confidence threshold: >= {float(best_thresh):.0%} ({cum_stats[best_thresh]['accuracy']:.1%} acc)")

    # 3. Problem Areas
    print(f"\n[Problem Areas]")
    worst_regime = regime_results.get("worst_regime", "")
    if worst_regime and regime_stats:
        print(f"  Worst regime: {worst_regime} ({regime_stats[worst_regime]['accuracy']:.1%})")

    total_dir_errors = error_results.get("total_directional_errors", 0)
    print(f"  Total directional errors: {total_dir_errors:,}")

    high_conf_dir = error_results.get("high_conf_dir_errors", 0)
    if high_conf_dir > 0:
        print(f"  ⚠️ High-confidence directional errors: {high_conf_dir:,}")

    # 4. Profitability Assessment
    print(f"\n[Profitability Assessment]")
    basic_ev = profit_results.get("basic_ev", 0)
    print(f"  Base expected value (no filters): {basic_ev*100:+.4f}% per trade")

    filter_results = profit_results.get("filter_results", {})
    profitable_filters = [k for k, v in filter_results.items() if v.get("profitable", False)]
    if profitable_filters:
        print(f"  [YES] Profitable filter combinations found: {len(profitable_filters)}")
        for f in profitable_filters[:3]:
            print(f"    - {f}: EV = {filter_results[f]['ev_per_trade']*100:+.4f}%")
    else:
        print(f"  [NO] No profitable filter combination found with current R:R assumptions")

    # 5. Actionable Recommendations
    print(f"\n{'='*80}")
    print(f" ACTIONABLE RECOMMENDATIONS")
    print(f"{'='*80}")

    recommendations = []

    # Based on regime performance
    if regime_stats:
        ranging_acc = regime_stats.get("RANGING", {}).get("accuracy", 0.5)
        if ranging_acc < 0.38:
            recommendations.append({
                "priority": "HIGH",
                "action": f"Filter out RANGING regime (only {ranging_acc:.1%} accuracy)",
                "impact": "Could significantly improve overall performance"
            })

        trending_accs = [regime_stats.get(r, {}).get("accuracy", 0)
                        for r in ["TRENDING_UP", "TRENDING_DOWN"]]
        if all(a > 0.42 for a in trending_accs if a > 0):
            recommendations.append({
                "priority": "HIGH",
                "action": "Focus exclusively on TRENDING regimes",
                "impact": "Model excels in trending markets"
            })

    # Based on confidence calibration
    calibration = confidence_results.get("calibration", {})
    if calibration:
        gaps = [abs(v.get("gap", 0)) for v in calibration.values()]
        if max(gaps) > 0.1:
            recommendations.append({
                "priority": "MEDIUM",
                "action": "Recalibrate confidence scores",
                "impact": "Confidence is not well-calibrated"
            })

    # Based on FLAT performance
    if f1_flat < 0.3:
        recommendations.append({
            "priority": "HIGH",
            "action": "Consider binary classification (UP/DOWN only)",
            "impact": f"FLAT F1 score is very low ({f1_flat:.3f})"
        })

    # Based on error analysis
    if dir_error > 25:
        recommendations.append({
            "priority": "CRITICAL",
            "action": "Reduce directional errors - they're costing the most",
            "impact": f"Directional error rate ({dir_error:.1f}%) is too high"
        })

    # Add ranging strategies recommendation
    recommendations.append({
        "priority": "MEDIUM",
        "action": "Add ranging expert strategies (Bollinger squeeze, ADX, Chop Index)",
        "impact": "Model was trained only on trending strategies"
    })

    # Print recommendations
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. [{rec['priority']}] {rec['action']}")
        print(f"   Impact: {rec['impact']}")

    # 6. Next Steps
    print(f"\n{'='*80}")
    print(f" NEXT STEPS")
    print(f"{'='*80}")
    print("""
1. IMMEDIATE: Apply best filter combination and re-run Monte Carlo
2. SHORT-TERM: Add ranging expert strategies to training data
3. MEDIUM-TERM: Implement confidence recalibration
4. EXPLORATION: Test binary classification (UP/DOWN only)
5. ANALYSIS: Run SHAP values for feature importance
6. PRODUCTION: Only trade with best filter combination
""")


def analyze(
    timeframe: str = "1h",
    seq_length: int = 50,
    hidden_size: int = 128,
    num_layers: int = 2,
    limit: int = 500000,
    epochs: int = 30,
):
    """Main comprehensive analysis pipeline."""
    print("=" * 80)
    print(" AlphaTrader LSTM COMPREHENSIVE Analysis Pipeline")
    print(" Time is not a factor - thoroughness is.")
    print("=" * 80)
    print(f"Started: {datetime.now()}")
    print(f"Timeframe: {timeframe}")

    # Load and prepare data
    print("\n[1/5] Loading data...")
    df = load_data(timeframe=timeframe, limit=limit)

    print("[2/5] Computing labels...")
    df = compute_labels(df)

    print("[3/5] Preparing features...")
    df = prepare_features(df)

    # 5 OHLCV + 14 trending signals + 14 ranging signals + 4 regime = 37 features
    feature_cols = [
        # OHLCV (5)
        "open", "high", "low", "close", "volume",
        # Trending signals (7 signal + 7 conf = 14)
        "signal_macd", "signal_rsi_div", "signal_turtle",
        "signal_ma_cross", "signal_stoch", "signal_sr", "signal_candle",
        "conf_macd", "conf_rsi_div", "conf_turtle",
        "conf_ma_cross", "conf_stoch", "conf_sr", "conf_candle",
        # Ranging signals (7 signal + 7 conf = 14)
        "signal_adx_range", "signal_bb_squeeze", "signal_chop",
        "signal_kc_squeeze", "signal_rsi_mid", "signal_atr_contract", "signal_price_channel",
        "conf_adx_range", "conf_bb_squeeze", "conf_chop",
        "conf_kc_squeeze", "conf_rsi_mid", "conf_atr_contract", "conf_price_channel",
        # Regime one-hot (4)
        "regime_trending_up", "regime_trending_down", "regime_ranging", "regime_high_vol",
    ]

    print("[4/5] Creating sequences...")
    X, y, meta_df = create_sequences(df, feature_cols, seq_length=seq_length)

    if len(X) < 1000:
        print("ERROR: Not enough data for analysis")
        return

    # Train model
    print(f"[5/5] Training model for analysis ({epochs} epochs)...")
    input_size = X.shape[2]

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    meta_test = meta_df.iloc[split_idx:].reset_index(drop=True)

    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.3,
        learning_rate=0.001,
    )

    val_split = int(len(X_train) * 0.9)
    model.fit(
        X_train[:val_split], y_train[:val_split],
        validation_data=(X_train[val_split:], y_train[val_split:]),
        epochs=epochs,
        batch_size=64,
        early_stopping_patience=10,
        verbose=True,
    )

    # Get predictions
    print("\n[Getting predictions...]")
    predictions, confidences = model.predict(X_test)
    predictions = predictions + 1  # Map -1,0,1 to 0,1,2

    # ===== RUN ALL ANALYSES =====
    all_results = {}

    analyze_data_quality(meta_test, y_test)
    all_results["what"] = analyze_what(predictions, y_test, confidences)
    all_results["regime"] = analyze_when_regime(predictions, y_test, meta_test, confidences)
    all_results["symbol"] = analyze_when_symbol(predictions, y_test, meta_test, confidences)
    all_results["time"] = analyze_when_time(predictions, y_test, meta_test)
    all_results["confidence"] = analyze_how_confidence(predictions, y_test, confidences, meta_test)
    all_results["signals"] = analyze_why_signals(predictions, y_test, meta_test, confidences)
    all_results["errors"] = analyze_where_fails(predictions, y_test, meta_test, confidences)
    all_results["profitability"] = analyze_profitability_potential(predictions, y_test, confidences, meta_test)

    final_summary(all_results)

    # Save results
    results_path = f"checkpoints/analysis_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path("checkpoints").mkdir(exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            # Convert tuple keys to strings
            return {str(k) if isinstance(k, tuple) else k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    with open(results_path, "w") as f:
        json.dump(convert_numpy(all_results), f, indent=2, default=str)

    print(f"\n\nResults saved to: {results_path}")
    print(f"Finished: {datetime.now()}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive LSTM model analysis")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe (default: 1h)")
    parser.add_argument("--seq-length", type=int, default=50, help="Sequence length (default: 50)")
    parser.add_argument("--hidden-size", type=int, default=128, help="LSTM hidden size (default: 128)")
    parser.add_argument("--num-layers", type=int, default=2, help="LSTM layers (default: 2)")
    parser.add_argument("--limit", type=int, default=500000, help="Max rows to load (default: 500000)")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30)")

    args = parser.parse_args()

    analyze(
        timeframe=args.timeframe,
        seq_length=args.seq_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        limit=args.limit,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()

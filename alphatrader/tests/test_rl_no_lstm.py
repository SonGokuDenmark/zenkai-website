#!/usr/bin/env python3
"""
Test RL without LSTM - just strategy signals + regime.

If this works as well as with LSTM, then LSTM is useless.
If this fails, LSTM might be providing SOME value (even if random).
"""

import os
import sys
import numpy as np
import pandas as pd
import psycopg2
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.env import TradingEnv, create_train_test_envs
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor


def get_data(symbol='BTCUSDT', timeframe='1h', limit=2000):
    """Load data WITHOUT adding LSTM predictions."""
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

    df = df.sort_values('open_time').reset_index(drop=True)

    # Fill NaN
    for col in signal_cols:
        df[col] = df[col].fillna(0)

    # NO LSTM columns added - just signals and regime
    print(f"  Loaded {len(df)} rows, {len(signal_cols)} signal columns")
    return df


def train_and_test(df, algorithm='DQN', timesteps=50000):
    """Train and evaluate."""
    train_env, val_env, test_env = create_train_test_envs(
        df,
        train_ratio=0.7,
        val_ratio=0.15,
        episode_length=500,
        initial_balance=10000.0,
        transaction_cost=0.001,
    )

    train_env = Monitor(train_env)

    if algorithm == 'PPO':
        model = PPO("MlpPolicy", train_env, learning_rate=3e-4, verbose=0)
    else:
        model = DQN("MlpPolicy", train_env, learning_rate=1e-4, verbose=0)

    model.learn(total_timesteps=timesteps, progress_bar=False)

    # Evaluate on test
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
    }


def main():
    print("="*60)
    print("RL WITHOUT LSTM TEST")
    print("="*60)
    print("Testing if RL agents can be profitable using only:")
    print("  - 82 strategy signals")
    print("  - 82 confidence scores")
    print("  - Regime encoding")
    print("  - NO LSTM predictions")
    print()

    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    algos = ['PPO', 'DQN']
    runs = 2
    timesteps = 50000

    results = []

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Testing: {symbol}")
        print("="*60)

        df = get_data(symbol=symbol, timeframe='1h', limit=1500)

        for algo in algos:
            pnls = []
            for run in range(runs):
                print(f"  {algo} run {run+1}/{runs}...", end=" ", flush=True)
                result = train_and_test(df, algorithm=algo, timesteps=timesteps)
                pnls.append(result['pnl'])
                print(f"PnL: ${result['pnl']:.2f}, WR: {result['win_rate']:.1%}, Trades: {result['trades']}")

            avg_pnl = np.mean(pnls)
            status = "[OK]" if avg_pnl > 0 else "[LOSS]"
            results.append({
                'symbol': symbol,
                'algo': algo,
                'avg_pnl': avg_pnl,
                'status': status
            })

    print("\n" + "="*60)
    print("SUMMARY (NO LSTM)")
    print("="*60)
    for r in results:
        print(f"  {r['symbol']:10} {r['algo']:5} Avg PnL: ${r['avg_pnl']:>8.2f} {r['status']}")

    profitable = sum(1 for r in results if r['avg_pnl'] > 0)
    print(f"\nProfitable: {profitable}/{len(results)} ({profitable/len(results)*100:.0f}%)")

    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print("With LSTM (34.8% accuracy, honest):")
    print("  BTC PPO: ~-$900, BTC DQN: ~+$50")
    print("  ETH PPO: ~+$280, ETH DQN: ~+$1346")
    print()
    print("If no-LSTM results are SIMILAR:")
    print("  -> LSTM is useless, remove it")
    print("If no-LSTM results are WORSE:")
    print("  -> LSTM might be providing noise that helps exploration")
    print("If no-LSTM results are BETTER:")
    print("  -> LSTM is actively hurting performance")


if __name__ == "__main__":
    main()

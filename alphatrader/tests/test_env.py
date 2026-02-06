"""Quick test of TradingEnv."""

import sys
import os
import numpy as np
import psycopg2
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.env import TradingEnv, Action


def _get_db_host():
    """Auto-detect which IP to use for database connection."""
    if os.getenv("ZENKAI_DB_HOST"):
        return os.getenv("ZENKAI_DB_HOST")
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
    return tailscale_ip


def load_test_data(limit: int = 10000) -> pd.DataFrame:
    """Load sample data from database."""
    conn = psycopg2.connect(
        host=_get_db_host(),
        database="zenkai_data",
        user="zenkai",
        password=os.getenv("ZENKAI_DB_PASSWORD"),
    )

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
        WHERE timeframe = '4h'
        AND regime IS NOT NULL
        AND symbol = 'BTCUSDT'
        ORDER BY open_time
        LIMIT {limit}
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    print(f"Loaded {len(df)} rows with {len(signal_cols)} signal/conf columns")
    return df


def test_random_agent(env: TradingEnv, n_episodes: int = 3):
    """Test with random actions."""
    print("\n=== Testing Random Agent ===")

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        print(f"Episode {ep+1}: Steps={steps}, Reward={total_reward:.2f}, "
              f"Trades={info['total_trades']}, WinRate={info['win_rate']:.1%}, "
              f"MaxDD={info['max_drawdown']:.1%}")


def test_hold_agent(env: TradingEnv, n_episodes: int = 3):
    """Test with always HOLD."""
    print("\n=== Testing Hold Agent ===")

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            obs, reward, terminated, truncated, info = env.step(Action.HOLD)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        print(f"Episode {ep+1}: Steps={steps}, Reward={total_reward:.2f}, "
              f"Trades={info['total_trades']}")


def test_observation_shape(env: TradingEnv):
    """Verify observation shape matches observation_space."""
    print("\n=== Testing Observation Shape ===")
    obs, _ = env.reset()

    expected = env.observation_space.shape[0]
    actual = len(obs)

    print(f"Expected shape: {expected}")
    print(f"Actual shape: {actual}")
    print(f"Match: {expected == actual}")

    if expected != actual:
        print("ERROR: Shape mismatch!")
        return False
    return True


def main():
    print("Loading test data...")
    df = load_test_data(limit=5000)

    print("\nCreating TradingEnv...")
    env = TradingEnv(
        df=df,
        initial_balance=10000.0,
        transaction_cost=0.001,
        episode_length=200,
    )

    print(f"\nObservation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Run tests
    if not test_observation_shape(env):
        return

    test_hold_agent(env, n_episodes=2)
    test_random_agent(env, n_episodes=3)

    print("\n=== All tests passed! ===")


if __name__ == "__main__":
    main()

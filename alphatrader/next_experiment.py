#!/usr/bin/env python3
"""
Next Experiment - Suggest untested LSTM configurations.

Usage:
    python next_experiment.py                    # Suggest variations of best configs
    python next_experiment.py --strategy random  # Suggest random untested configs
    python next_experiment.py --coverage         # Show search space coverage
    python next_experiment.py --gaps             # Show unexplored parameter combinations
"""

import argparse
import os
import random
from collections import defaultdict
from itertools import product
from typing import Dict, List, Optional, Set, Tuple

import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Database config
DB_CONFIG = {
    "host": os.getenv("ZENKAI_DB_HOST", "192.168.0.160"),
    "database": os.getenv("ZENKAI_DB_NAME", "zenkai_data"),
    "user": os.getenv("ZENKAI_DB_USER", "zenkai"),
    "password": os.getenv("ZENKAI_DB_PASSWORD"),
}

# Search space definition
SEARCH_SPACE = {
    "timeframe": ["15m", "1h", "4h"],
    "hidden_size": [64, 128, 256, 512],
    "num_layers": [1, 2, 3, 4],
    "seq_length": [25, 50, 100, 150],
}

# Symbols to optionally filter by (None = all symbols)
SYMBOLS = [None, "BTCUSDT", "ETHUSDT", "SOLUSDT"]


def get_db_connection():
    """Get PostgreSQL connection."""
    return psycopg2.connect(**DB_CONFIG)


def get_tested_configs() -> List[Dict]:
    """Get all tested configurations from database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            timeframe, hidden_size, num_layers, seq_length, symbol,
            test_accuracy, status
        FROM experiments
        WHERE status IN ('completed', 'running')
    """)

    configs = []
    for row in cursor.fetchall():
        configs.append({
            "timeframe": row[0],
            "hidden_size": row[1],
            "num_layers": row[2],
            "seq_length": row[3],
            "symbol": row[4],
            "test_accuracy": row[5],
            "status": row[6],
        })

    conn.close()
    return configs


def get_best_configs(top_n: int = 5) -> List[Dict]:
    """Get top N performing configurations."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            timeframe, hidden_size, num_layers, seq_length, symbol,
            test_accuracy, worker
        FROM experiments
        WHERE status = 'completed' AND test_accuracy IS NOT NULL
        ORDER BY test_accuracy DESC
        LIMIT %s
    """, (top_n,))

    configs = []
    for row in cursor.fetchall():
        configs.append({
            "timeframe": row[0],
            "hidden_size": row[1],
            "num_layers": row[2],
            "seq_length": row[3],
            "symbol": row[4],
            "test_accuracy": row[5],
            "worker": row[6],
        })

    conn.close()
    return configs


def config_to_key(config: Dict) -> Tuple:
    """Convert config dict to hashable tuple."""
    return (
        config.get("timeframe"),
        config.get("hidden_size"),
        config.get("num_layers"),
        config.get("seq_length"),
        config.get("symbol"),
    )


def get_tested_keys(configs: List[Dict]) -> Set[Tuple]:
    """Get set of tested config keys."""
    return {config_to_key(c) for c in configs}


def generate_variations(base_config: Dict) -> List[Dict]:
    """Generate variations of a base config by tweaking one parameter at a time."""
    variations = []

    for param, values in SEARCH_SPACE.items():
        current_val = base_config.get(param)
        for val in values:
            if val != current_val:
                new_config = base_config.copy()
                new_config[param] = val
                variations.append(new_config)

    return variations


def generate_all_configs() -> List[Dict]:
    """Generate all possible configurations in the search space."""
    all_configs = []

    for tf, hidden, layers, seq in product(
        SEARCH_SPACE["timeframe"],
        SEARCH_SPACE["hidden_size"],
        SEARCH_SPACE["num_layers"],
        SEARCH_SPACE["seq_length"],
    ):
        # For now, just use symbol=None (all symbols)
        all_configs.append({
            "timeframe": tf,
            "hidden_size": hidden,
            "num_layers": layers,
            "seq_length": seq,
            "symbol": None,
        })

    return all_configs


def format_command(config: Dict) -> str:
    """Format config as train_lstm.py command."""
    cmd = "python train_lstm.py --worker YOUR_NAME"
    cmd += f" --timeframe {config['timeframe']}"
    cmd += f" --hidden-size {config['hidden_size']}"
    cmd += f" --num-layers {config['num_layers']}"
    cmd += f" --seq-length {config['seq_length']}"

    if config.get("symbol"):
        cmd += f" --symbol {config['symbol']}"

    return cmd


def suggest_best_variations(num_suggestions: int = 5):
    """Suggest variations of the best performing configs."""
    tested = get_tested_configs()
    tested_keys = get_tested_keys(tested)
    best_configs = get_best_configs(top_n=3)

    if not best_configs:
        print("\nNo completed experiments found. Run some experiments first!")
        print("\nTry a random configuration:")
        suggest_random(1)
        return

    print("\n" + "=" * 80)
    print("Suggested Experiments (Variations of Best Configs)")
    print("=" * 80)

    suggestions = []
    for base in best_configs:
        variations = generate_variations(base)
        for var in variations:
            key = config_to_key(var)
            if key not in tested_keys and key not in {config_to_key(s) for s in suggestions}:
                var["base_accuracy"] = base["test_accuracy"]
                suggestions.append(var)

    if not suggestions:
        print("\nAll variations of top configs have been tested!")
        print("Try --strategy random for unexplored configurations.")
        return

    # Sort by base accuracy (variations of better configs first)
    suggestions.sort(key=lambda x: x.get("base_accuracy", 0), reverse=True)
    suggestions = suggestions[:num_suggestions]

    print(f"\nBased on top {len(best_configs)} performing experiments:\n")

    for i, config in enumerate(suggestions, 1):
        print(f"--- Suggestion #{i} ---")
        print(f"  Timeframe:   {config['timeframe']}")
        print(f"  Hidden Size: {config['hidden_size']}")
        print(f"  Num Layers:  {config['num_layers']}")
        print(f"  Seq Length:  {config['seq_length']}")
        print(f"  Symbol:      {config.get('symbol') or 'ALL'}")
        print(f"\n  Command:")
        print(f"  {format_command(config)}")
        print()


def suggest_random(num_suggestions: int = 5):
    """Suggest random untested configurations."""
    tested = get_tested_configs()
    tested_keys = get_tested_keys(tested)
    all_configs = generate_all_configs()

    # Filter to untested
    untested = [c for c in all_configs if config_to_key(c) not in tested_keys]

    if not untested:
        print("\nAll configurations in the search space have been tested!")
        print("Consider expanding the search space or analyzing results.")
        return

    print("\n" + "=" * 80)
    print("Suggested Experiments (Random Untested Configs)")
    print("=" * 80)

    # Random sample
    suggestions = random.sample(untested, min(num_suggestions, len(untested)))

    print(f"\n{len(untested)} untested configurations remaining.\n")

    for i, config in enumerate(suggestions, 1):
        print(f"--- Suggestion #{i} ---")
        print(f"  Timeframe:   {config['timeframe']}")
        print(f"  Hidden Size: {config['hidden_size']}")
        print(f"  Num Layers:  {config['num_layers']}")
        print(f"  Seq Length:  {config['seq_length']}")
        print(f"  Symbol:      {config.get('symbol') or 'ALL'}")
        print(f"\n  Command:")
        print(f"  {format_command(config)}")
        print()


def show_coverage():
    """Show search space coverage statistics."""
    tested = get_tested_configs()
    tested_keys = get_tested_keys(tested)
    all_configs = generate_all_configs()

    total = len(all_configs)
    tested_count = len(tested_keys)
    coverage = (tested_count / total) * 100 if total > 0 else 0

    print("\n" + "=" * 60)
    print("Search Space Coverage")
    print("=" * 60)

    print(f"\nTotal configurations: {total}")
    print(f"Tested configurations: {tested_count}")
    print(f"Coverage: {coverage:.1f}%")

    # Per-parameter coverage
    print("\n--- Per Parameter ---")

    for param, values in SEARCH_SPACE.items():
        counts = defaultdict(int)
        for config in tested:
            val = config.get(param)
            if val in values:
                counts[val] += 1

        print(f"\n{param}:")
        for val in values:
            count = counts.get(val, 0)
            bar = "█" * min(count, 20)
            print(f"  {val:>6}: {count:>3} runs {bar}")

    # Status breakdown
    print("\n--- By Status ---")
    status_counts = defaultdict(int)
    for config in tested:
        status_counts[config.get("status", "unknown")] += 1

    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")


def show_gaps():
    """Show unexplored parameter combinations."""
    tested = get_tested_configs()
    all_configs = generate_all_configs()
    tested_keys = get_tested_keys(tested)

    untested = [c for c in all_configs if config_to_key(c) not in tested_keys]

    print("\n" + "=" * 60)
    print("Unexplored Parameter Gaps")
    print("=" * 60)

    # Group by timeframe
    print("\n--- By Timeframe ---")
    for tf in SEARCH_SPACE["timeframe"]:
        tf_untested = [c for c in untested if c["timeframe"] == tf]
        tf_total = len([c for c in all_configs if c["timeframe"] == tf])
        print(f"  {tf}: {len(tf_untested)}/{tf_total} untested")

    # Group by hidden_size
    print("\n--- By Hidden Size ---")
    for hs in SEARCH_SPACE["hidden_size"]:
        hs_untested = [c for c in untested if c["hidden_size"] == hs]
        hs_total = len([c for c in all_configs if c["hidden_size"] == hs])
        print(f"  {hs}: {len(hs_untested)}/{hs_total} untested")

    # Group by num_layers
    print("\n--- By Num Layers ---")
    for nl in SEARCH_SPACE["num_layers"]:
        nl_untested = [c for c in untested if c["num_layers"] == nl]
        nl_total = len([c for c in all_configs if c["num_layers"] == nl])
        print(f"  {nl}: {len(nl_untested)}/{nl_total} untested")

    # Completely untested combinations
    print("\n--- Completely Untested Combinations ---")

    # Check each timeframe x hidden_size combination
    for tf in SEARCH_SPACE["timeframe"]:
        for hs in SEARCH_SPACE["hidden_size"]:
            combo_tested = any(
                c["timeframe"] == tf and c["hidden_size"] == hs
                for c in tested
            )
            if not combo_tested:
                print(f"  {tf} + hidden={hs}: NO experiments yet")


def main():
    parser = argparse.ArgumentParser(description="Suggest next LSTM experiments")

    parser.add_argument(
        "--strategy", "-s",
        type=str,
        default="best",
        choices=["best", "random"],
        help="Suggestion strategy: 'best' (variations of top configs) or 'random'"
    )
    parser.add_argument(
        "--num", "-n",
        type=int,
        default=5,
        help="Number of suggestions to show"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Show search space coverage statistics"
    )
    parser.add_argument(
        "--gaps",
        action="store_true",
        help="Show unexplored parameter combinations"
    )

    args = parser.parse_args()

    if args.coverage:
        show_coverage()
    elif args.gaps:
        show_gaps()
    elif args.strategy == "random":
        suggest_random(args.num)
    else:
        suggest_best_variations(args.num)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
AlphaTrader Batch Test Runner

Run multiple test configurations overnight and auto-sort results.

Usage:
    python run_tests.py                    # Run all pending tests
    python run_tests.py --quick            # Quick tests (fewer timesteps)
    python run_tests.py --list             # List available tests
    python run_tests.py --test mtf_combo   # Run specific test
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict

# Test configurations
TESTS = {
    # === QUICK TESTS (1-2 hours) ===
    "best_4h_no_highvol": {
        "description": "Best single config: 4H + LSTM + no_high_vol filter",
        "script": "test_best_config.py",
        "priority": 1,
        "estimated_time": "2-3 hours",
    },

    # === MEDIUM TESTS (2-4 hours) ===
    "mtf_combo": {
        "description": "Multi-timeframe combos (30m+4h, 1h+4h, etc.)",
        "script": "test_mtf_combo.py",
        "priority": 2,
        "estimated_time": "3-4 hours",
    },
    "regime_filter": {
        "description": "Regime-based filtering strategies",
        "script": "test_regime_filter.py",
        "priority": 2,
        "estimated_time": "2-3 hours",
    },
    "robustness": {
        "description": "Parameter sensitivity test (overfit detection)",
        "script": "test_robustness.py",
        "priority": 3,
        "estimated_time": "3-4 hours",
    },

    # === LONGER TESTS (4-6 hours) ===
    "multi_timeframe": {
        "description": "Single timeframe tests per symbol (15m, 30m, 1h, 4h)",
        "script": "test_multi_timeframe.py",
        "priority": 4,
        "estimated_time": "4-5 hours",
    },
    "extended_symbols": {
        "description": "Best config on top 10 symbols",
        "script": "test_extended_symbols.py",
        "priority": 4,
        "estimated_time": "4-5 hours",
    },

    # === COMPREHENSIVE (overnight, 8+ hours) ===
    "comprehensive": {
        "description": "FULL TEST: Short/Medium/Long training, all TFs, all algos, overfit detection",
        "script": "test_comprehensive.py",
        "priority": 10,
        "estimated_time": "8-12 hours (overnight)",
    },

    # === ULTIMATE (weekend, 24-48 hours) ===
    "ultimate": {
        "description": "ULTIMATE: 10 TFs, 20 symbols, 6 training levels (50k-2M), walk-forward, EVERYTHING",
        "script": "test_ultimate.py",
        "priority": 99,
        "estimated_time": "24-48 hours (weekend run)",
    },
}

# Results thresholds
THRESHOLDS = {"winning": 500, "losing": -500}


def list_tests():
    """List all available tests."""
    print("\n" + "=" * 70)
    print("AVAILABLE TESTS")
    print("=" * 70)

    for name, config in sorted(TESTS.items(), key=lambda x: x[1]["priority"]):
        script_path = Path(__file__).parent / config["script"]
        exists = "[OK]" if script_path.exists() else "[MISSING]"
        print(f"\n  {name:20} Priority: {config['priority']}")
        print(f"    {config['description']}")
        print(f"    Script: {config['script']} {exists}")
        print(f"    Est. time: {config['estimated_time']}")


def run_test(name: str, quick: bool = False) -> Dict:
    """Run a single test and return results."""
    if name not in TESTS:
        print(f"Error: Unknown test '{name}'")
        return None

    config = TESTS[name]
    script_path = Path(__file__).parent / config["script"]

    if not script_path.exists():
        print(f"Error: Script not found: {script_path}")
        return None

    print(f"\n{'=' * 70}")
    print(f"RUNNING: {name}")
    print(f"  {config['description']}")
    print(f"  Script: {config['script']}")
    print(f"  Started: {datetime.now()}")
    print("=" * 70 + "\n")

    # Build command
    cmd = [sys.executable, str(script_path)]
    if quick:
        cmd.extend(["--quick"])  # Assumes scripts support --quick flag

    # Run test
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent),
            capture_output=False,  # Show output in real-time
        )
        elapsed = time.time() - start_time

        return {
            "name": name,
            "success": result.returncode == 0,
            "elapsed_seconds": elapsed,
            "elapsed_formatted": f"{elapsed/60:.1f} minutes",
        }
    except Exception as e:
        return {
            "name": name,
            "success": False,
            "error": str(e),
        }


def sort_results():
    """Auto-sort any result files in the main directory."""
    root = Path(__file__).parent
    results_dir = root / "results"

    # Find result files
    result_files = []
    for pattern in ["*results*.json", "*_results.json", "mtf_*.json", "regime_*.json", "best_*.json"]:
        result_files.extend(root.glob(pattern))

    # Filter out non-result files
    result_files = [f for f in result_files if f.name not in ["package.json", "sessions-index.json"]]

    if not result_files:
        print("No result files to sort.")
        return

    print(f"\nSorting {len(result_files)} result file(s)...")

    for filepath in result_files:
        try:
            # Analyze result
            with open(filepath) as f:
                data = json.load(f)

            total_pnl = 0
            count = 0

            # Try various formats
            if "results" in data:
                for r in data["results"]:
                    if isinstance(r, dict):
                        pnl = r.get("avg_pnl") or r.get("pnl") or r.get("total_pnl")
                        if pnl is not None:
                            total_pnl += pnl
                            count += 1

            for key, val in data.items():
                if isinstance(val, dict):
                    if "avg_pnl" in val:
                        total_pnl += val["avg_pnl"]
                        count += 1
                    elif "pnl" in val:
                        total_pnl += val["pnl"]
                        count += 1
                    else:
                        for subkey, subval in val.items():
                            if isinstance(subval, dict) and "pnl" in subval:
                                total_pnl += subval["pnl"]
                                count += 1

            avg_pnl = total_pnl / count if count > 0 else 0

            # Categorize
            if avg_pnl > THRESHOLDS["winning"]:
                category = "winning"
            elif avg_pnl < THRESHOLDS["losing"]:
                category = "losing"
            else:
                category = "neutral"

            # Move file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest_dir = results_dir / category
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{filepath.stem}_{timestamp}{filepath.suffix}"

            filepath.rename(dest)
            print(f"  {filepath.name} -> {category}/ (avg PnL: ${avg_pnl:,.2f})")

        except Exception as e:
            print(f"  {filepath.name} -> ERROR: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AlphaTrader Batch Test Runner")
    parser.add_argument("--list", action="store_true", help="List available tests")
    parser.add_argument("--test", type=str, help="Run specific test")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer timesteps)")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--sort-only", action="store_true", help="Only sort existing results")
    args = parser.parse_args()

    if args.list:
        list_tests()
        return

    if args.sort_only:
        sort_results()
        return

    print("\n" + "=" * 70)
    print("ALPHATRADER BATCH TEST RUNNER")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")

    # Determine which tests to run
    if args.test:
        tests_to_run = [args.test]
    elif args.all:
        # Sort by priority
        tests_to_run = sorted(TESTS.keys(), key=lambda x: TESTS[x]["priority"])
    else:
        # Run only existing scripts, sorted by priority
        tests_to_run = []
        for name, config in sorted(TESTS.items(), key=lambda x: x[1]["priority"]):
            script_path = Path(__file__).parent / config["script"]
            if script_path.exists():
                tests_to_run.append(name)

    print(f"Tests to run: {tests_to_run}")

    # Run tests
    results = []
    for test_name in tests_to_run:
        result = run_test(test_name, quick=args.quick)
        if result:
            results.append(result)

    # Sort results
    print("\n" + "=" * 70)
    print("SORTING RESULTS")
    print("=" * 70)
    sort_results()

    # Summary
    print("\n" + "=" * 70)
    print("TEST RUN COMPLETE")
    print("=" * 70)

    successful = sum(1 for r in results if r.get("success"))
    print(f"\nTests run: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")

    for r in results:
        status = "[OK]" if r.get("success") else "[FAIL]"
        elapsed = r.get("elapsed_formatted", "N/A")
        print(f"  {r['name']:20} {status:8} ({elapsed})")

    print(f"\nFinished: {datetime.now()}")
    print("\nResults sorted to results/winning, results/losing, results/neutral")


if __name__ == "__main__":
    main()

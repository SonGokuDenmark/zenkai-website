#!/usr/bin/env python3
"""
Quick Result Sorter for AlphaTrader Tests

After running a test, use this to categorize the results:
    python sort_results.py <result_file.json> winning|losing|neutral

Or run interactively:
    python sort_results.py

Results are moved to:
    results/winning/   - Profitable strategies (avg PnL > $500)
    results/losing/    - Losing strategies (avg PnL < -$500)
    results/neutral/   - Inconclusive (-$500 to $500)
"""

import os
import sys
import json
import shutil
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
THRESHOLDS = {"winning": 500, "losing": -500}


def get_result_files():
    """Find all JSON result files in the main directory."""
    root = Path(__file__).parent
    files = []
    for f in root.glob("*.json"):
        if f.name not in ["package.json", "package-lock.json"]:
            # Check if it looks like a results file
            try:
                with open(f) as fp:
                    data = json.load(fp)
                if any(k in data for k in ["results", "pnl", "avg_pnl", "summary"]):
                    files.append(f)
            except:
                pass
    return files


def analyze_result(filepath: Path) -> dict:
    """Analyze a result file and return summary."""
    with open(filepath) as f:
        data = json.load(f)

    # Try to extract PnL from various formats
    total_pnl = 0
    count = 0

    if "results" in data:
        for r in data["results"]:
            if isinstance(r, dict):
                pnl = r.get("avg_pnl") or r.get("pnl") or r.get("total_pnl")
                if pnl is not None:
                    total_pnl += pnl
                    count += 1

    # Direct format (like best_config_results.json)
    for key, val in data.items():
        if isinstance(val, dict) and "avg_pnl" in val:
            total_pnl += val["avg_pnl"]
            count += 1

    # Combo format
    for key, val in data.items():
        if isinstance(val, dict):
            for symbol, result in val.items():
                if isinstance(result, dict) and "pnl" in result:
                    total_pnl += result["pnl"]
                    count += 1

    return {
        "file": filepath.name,
        "total_pnl": total_pnl,
        "count": count,
        "avg_pnl": total_pnl / count if count > 0 else 0,
    }


def categorize(avg_pnl: float) -> str:
    """Determine category based on PnL."""
    if avg_pnl > THRESHOLDS["winning"]:
        return "winning"
    elif avg_pnl < THRESHOLDS["losing"]:
        return "losing"
    return "neutral"


def move_result(filepath: Path, category: str):
    """Move result file to appropriate folder."""
    dest_dir = RESULTS_DIR / category
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Add timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_name = f"{filepath.stem}_{timestamp}{filepath.suffix}"
    dest = dest_dir / new_name

    shutil.move(str(filepath), str(dest))
    return dest


def interactive_sort():
    """Interactive mode for sorting results."""
    files = get_result_files()

    if not files:
        print("No result files found in the main directory.")
        print("Looking for: *results*.json, *_results.json, etc.")
        return

    print("\n" + "=" * 60)
    print("ALPHATRADER RESULT SORTER")
    print("=" * 60)
    print(f"Found {len(files)} result file(s)\n")

    for filepath in files:
        analysis = analyze_result(filepath)
        suggested = categorize(analysis["avg_pnl"])

        print(f"\nFile: {filepath.name}")
        print(f"  Total PnL: ${analysis['total_pnl']:,.2f}")
        print(f"  Avg PnL:   ${analysis['avg_pnl']:,.2f}")
        print(f"  Tests:     {analysis['count']}")
        print(f"  Suggested: {suggested.upper()}")

        while True:
            choice = input(f"\nMove to [w]inning, [l]osing, [n]eutral, [s]kip? [{suggested[0]}]: ").lower().strip()

            if choice == "" or choice == suggested[0]:
                choice = suggested
                break
            elif choice == "w":
                choice = "winning"
                break
            elif choice == "l":
                choice = "losing"
                break
            elif choice == "n":
                choice = "neutral"
                break
            elif choice == "s":
                choice = None
                break
            else:
                print("Invalid choice. Use w/l/n/s")

        if choice:
            dest = move_result(filepath, choice)
            print(f"  -> Moved to {dest.relative_to(filepath.parent)}")
        else:
            print("  -> Skipped")

    print("\n" + "=" * 60)
    print("Done! Results sorted.")
    print("=" * 60)


def main():
    if len(sys.argv) == 3:
        # Direct mode: python sort_results.py file.json winning
        filepath = Path(sys.argv[1])
        category = sys.argv[2].lower()

        if not filepath.exists():
            print(f"Error: File not found: {filepath}")
            sys.exit(1)

        if category not in ["winning", "losing", "neutral"]:
            print(f"Error: Category must be winning, losing, or neutral")
            sys.exit(1)

        dest = move_result(filepath, category)
        print(f"Moved {filepath.name} -> {dest}")

    elif len(sys.argv) == 1:
        # Interactive mode
        interactive_sort()

    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()

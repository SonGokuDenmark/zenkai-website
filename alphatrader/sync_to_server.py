#!/usr/bin/env python3
"""
Sync alphatrader to server.

Syncs ALL Python files, directories, and models to the shared server.
Any machine in the cluster can then pull the latest version.

Usage:
    python sync_to_server.py          # Sync everything
    python sync_to_server.py --quick  # Skip large directories (checkpoints)
    python sync_to_server.py --clean  # Mirror exactly (deletes removed files)
"""

import subprocess
import socket
import sys
import argparse
from pathlib import Path


def get_server():
    """Auto-detect server IP (local network or Tailscale)."""
    import os
    local_ip = os.getenv("ZENKAI_DB_HOST", "192.168.0.160")
    tailscale_ip = os.getenv("ZENKAI_DB_HOST_TAILSCALE", "100.110.101.78")

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        if sock.connect_ex((local_ip, 22)) == 0:
            sock.close()
            return local_ip
    except:
        pass

    return tailscale_ip


def main():
    parser = argparse.ArgumentParser(description="Sync alphatrader to server")
    parser.add_argument("--quick", action="store_true",
                        help="Skip large directories (checkpoints)")
    parser.add_argument("--clean", action="store_true",
                        help="Mirror exactly using rsync (deletes removed files on server)")
    args = parser.parse_args()

    server_ip = get_server()
    server = f"goku@{server_ip}"
    remote_path = "/home/shared/alphatrader"

    print(f"Syncing to {server}:{remote_path}")
    print("-" * 60)

    # Get the directory where this script is located
    script_dir = Path(__file__).parent

    # Clean mode uses rsync for exact mirror
    if args.clean:
        print("  CLEAN MODE: Using rsync --delete (mirrors exactly)")
        print("  WARNING: This will delete files on server that don't exist locally!")
        print()

        ssh_key = Path.home() / ".ssh" / "id_ed25519"
        ssh_opts = f"-e 'ssh -i {ssh_key}'" if ssh_key.exists() else ""

        # Exclude patterns
        excludes = [
            "--exclude='.git'",
            "--exclude='__pycache__'",
            "--exclude='*.pyc'",
            "--exclude='.venv'",
            "--exclude='venv'",
            "--exclude='logs/*.log'",
        ]

        if args.quick:
            excludes.extend([
                "--exclude='checkpoints/'",
                "--exclude='results/'",
                "--exclude='experiments/'",
            ])

        exclude_str = " ".join(excludes)

        # Use rsync for clean sync
        cmd = f"rsync -avz --delete {exclude_str} {ssh_opts} {script_dir}/ {server}:{remote_path}/"
        print(f"  Running: rsync -avz --delete ...")

        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(result.stdout)
                print("-" * 60)
                print("Clean sync complete!")
            else:
                print(f"FAILED: {result.stderr}")
                sys.exit(1)
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        return

    # Auto-discover all Python files in root
    py_files = sorted([f.name for f in script_dir.glob("*.py")])

    # Auto-discover all markdown files in root
    md_files = sorted([f.name for f in script_dir.glob("*.md")])

    # Auto-discover all JSON files in root (configs, results)
    json_files = sorted([f.name for f in script_dir.glob("*.json")])

    # Core directories (always sync)
    core_dirs = [
        "src/",
        "scripts/",
        "server/",
        "models/",
        "docs/",
    ]

    # Large directories (skip with --quick)
    large_dirs = [
        "checkpoints/",
        "results/",
        "experiments/",
    ]

    # Other config files
    config_files = [
        "requirements.txt",
        ".gitignore",
    ]

    # Build final list
    items = []
    items.extend(py_files)
    items.extend(md_files)
    items.extend(json_files)
    items.extend(config_files)
    items.extend(core_dirs)

    if not args.quick:
        items.extend(large_dirs)
    else:
        print("  (--quick mode: skipping checkpoints, results, experiments)")

    # Remove duplicates while preserving order
    seen = set()
    unique_items = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique_items.append(item)
    items = unique_items

    success = True
    synced = 0
    skipped = 0

    for item in items:
        local_path = script_dir / item
        if not local_path.exists():
            skipped += 1
            continue

        print(f"  {item}...", end=" ", flush=True)

        # Use scp for files, scp -r for directories
        ssh_key = Path.home() / ".ssh" / "id_ed25519"
        cmd = ["scp", "-q"]  # -q for quiet
        if ssh_key.exists():
            cmd.extend(["-i", str(ssh_key)])
        if local_path.is_dir():
            cmd.append("-r")
        cmd.extend([str(local_path), f"{server}:{remote_path}/"])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print("OK")
                synced += 1
            else:
                print(f"FAILED: {result.stderr.strip()}")
                success = False
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            success = False
        except Exception as e:
            print(f"ERROR: {e}")
            success = False

    print("-" * 60)
    print(f"Synced: {synced} | Skipped: {skipped}")

    if success:
        print("Sync complete! All machines can now pull latest from server.")
        print(f"\nOther machines can pull with:")
        print(f"  scp -r {server}:{remote_path} ~/alphatrader")
    else:
        print("Some files failed to sync. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

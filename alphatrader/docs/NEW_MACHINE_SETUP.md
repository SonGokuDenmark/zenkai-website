# New Machine Setup Guide

**For Claude or any new machine joining the AlphaTrader cluster.**

Read this file, then follow the steps for your OS.

---

## Quick Info

| Item | Value |
|------|-------|
| Server IP (local) | `192.168.0.160` |
| Server IP (Tailscale) | `100.110.101.78` |
| Server user | `goku` |
| DB Host | Same as server IP |
| DB Name | `zenkai_data` |
| DB User | `zenkai` |
| DB Password | `ZenkaiHQ2000` |
| Repo path (server) | `/home/shared/alphatrader/` |

---

## Step 1: Install Tailscale (if not on local network)

### Linux
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

### Windows
Download from: https://tailscale.com/download/windows

### macOS
```bash
brew install tailscale
```

After install, authenticate with the same account as other machines.

---

## Step 2: Set Up SSH Key (one-time)

This allows passwordless SSH to the server.

### Linux / macOS
```bash
# Generate key if you don't have one
ssh-keygen -t ed25519 -C "$(hostname)"

# Copy to server (enter password once: ZenkaiHQ2000)
ssh-copy-id goku@192.168.0.160
# OR if not on local network:
ssh-copy-id goku@100.110.101.78

# Test - should connect without password
ssh goku@192.168.0.160 "echo 'SSH works!'"
```

### Windows (PowerShell)
```powershell
# Generate key
ssh-keygen -t ed25519 -C "$(hostname)"

# Copy key manually
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh goku@192.168.0.160 "cat >> ~/.ssh/authorized_keys"
```

---

## Step 3: Clone the Repository

### Linux / macOS
```bash
cd ~
git clone goku@192.168.0.160:/home/shared/alphatrader.git alphatrader
cd alphatrader
```

### Windows
```powershell
cd $env:USERPROFILE
git clone goku@192.168.0.160:/home/shared/alphatrader.git alphatrader
cd alphatrader
```

**If git clone fails**, use scp:
```bash
scp -r goku@192.168.0.160:/home/shared/alphatrader ~/alphatrader
```

---

## Step 4: Install Python Dependencies

### Linux (Ubuntu/Debian)
```bash
# Install Python if needed
sudo apt update && sudo apt install -y python3 python3-pip python3-venv

# Create venv (recommended)
cd ~/alphatrader
python3 -m venv venv
source venv/bin/activate

# Install deps
pip install -r requirements.txt
```

### Linux (Arch)
```bash
sudo pacman -S python python-pip
cd ~/alphatrader
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Windows
```powershell
cd $env:USERPROFILE\alphatrader
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS
```bash
brew install python
cd ~/alphatrader
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Step 5: Test Connection

```bash
cd ~/alphatrader
python parallel_hmm.py --status
```

Should show HMM processing status with no errors.

---

## Step 6: Start Contributing

### For HMM Processing (CPU work)
```bash
# Check your CPU cores
nproc  # Linux/macOS
# Or: echo %NUMBER_OF_PROCESSORS%  # Windows

# Start processing (use 75% of cores to leave headroom)
python parallel_hmm.py -w 6 --name MACHINE_NAME
```

Replace `MACHINE_NAME` with something identifiable (e.g., `laptop`, `fips`, `workpc`).

### For LSTM/RL Training (GPU work)
Only on machines with NVIDIA GPU:
```bash
# Check GPU
nvidia-smi

# Run training
python train_lstm.py --worker MACHINE_NAME --timeframe 4h
```

---

## Machine Roles

| Machine | CPU | GPU | Best For |
|---------|-----|-----|----------|
| Server | 8 cores | AMD (no CUDA) | Database, HMM daemon |
| Thomas PC | 16 threads | RTX 4070 Super | LSTM & RL training |
| Fips PC | 12 threads | RTX 5060 Ti | LSTM & RL training |
| Laptop | 8 threads | None | HMM processing |

---

## Syncing Changes

After making changes locally, sync to server:

```bash
# From the machine with changes
cd ~/alphatrader
scp -r src scripts *.py goku@192.168.0.160:/home/shared/alphatrader/
```

Or use the sync script (if on Windows/Thomas PC):
```bash
python sync_to_server.py
```

---

## Troubleshooting

### "Connection refused" on SSH
- Check if server is up: `ping 192.168.0.160`
- Try Tailscale IP: `ssh goku@100.110.101.78`

### "Permission denied" on SSH
- SSH key not set up. Run `ssh-copy-id` again.

### Python module not found
- Activate venv: `source venv/bin/activate`
- Reinstall: `pip install -r requirements.txt`

### Database connection failed
- Check you can reach server: `ping 192.168.0.160`
- Check PostgreSQL is running: `ssh goku@192.168.0.160 "systemctl status postgresql"`

### HMM model not found
- Models are in `models/` folder
- If missing, copy from server: `scp goku@192.168.0.160:/home/shared/alphatrader/models/*.pkl ~/alphatrader/models/`

---

## Environment Variables (Optional)

Add to `~/.bashrc` or `~/.zshrc`:

```bash
export ZENKAI_DB_HOST="192.168.0.160"
export ZENKAI_DB_NAME="zenkai_data"
export ZENKAI_DB_USER="zenkai"
export ZENKAI_DB_PASSWORD="ZenkaiHQ2000"
```

The scripts auto-detect these, so this is optional.

---

## For Claude (AI Assistant)

When helping set up a new machine:

1. Ask what OS they're running
2. Check if Tailscale is needed (are they on local network?)
3. Follow this guide step by step
4. Verify with `python parallel_hmm.py --status`
5. Add the machine to SYNC_STATUS.md machine list

**Commands to verify setup:**
```bash
ssh goku@192.168.0.160 "echo 'SSH OK'"
python parallel_hmm.py --status
python -c "import psycopg2; print('psycopg2 OK')"
python -c "from src.regime import HMMRegimeDetector; print('HMM OK')"
```

# AlphaTrader Sync Status

**Last Updated:** 2026-02-05

---

## RULES FOR CLAUDE (read after compaction)

1. **Server scripts must be UNIVERSAL** - auto-detect network, no hardcoded IPs
2. **After ANY script edit** → sync to server immediately
3. **DB credentials**: user=`zenkai`, password=`ZenkaiHQ2000`
4. **Results go to `results/` folder** - sort with `python sort_results.py`
5. **Completed tasks** → move to `ALREADY_DONE.md`
6. **ALWAYS filter dead coins in training** - use `symbol_status` table (see Binance section below)
7. **Bulk data on server HDD** - `/mnt/data/binance_bulk/` (1.7TB drive)

---

## Folder Structure

```
C:\Users\thoma\alphatrader\
├── checkpoints/           # Model checkpoints
│   ├── rl/               # RL agent models (PPO, DQN, A2C)
│   ├── ensemble/         # Ensemble models
│   └── lstm_*.pt         # LSTM checkpoints
├── models/               # HMM models (hmm_regime_*.pkl)
├── results/              # TEST RESULTS (sorted)
│   ├── winning/          # Profitable (avg PnL > $500)
│   ├── losing/           # Losing (avg PnL < -$500)
│   └── neutral/          # Inconclusive (-$500 to $500)
├── logs/                 # Training logs & TensorBoard
├── src/                  # Source code
│   ├── env/             # TradingEnv
│   ├── models/          # LSTM, xLSTM classifiers
│   ├── risk/            # RiskManager
│   ├── regime/          # HMMRegimeDetector
│   └── rl/              # RL utilities
├── scripts/              # Server-side scripts
├── server/               # Server daemons (hmm_processor.py)
├── test_*.py            # Test scripts (run overnight)
├── train_*.py           # Training scripts
├── bulk_download_binance.py  # Binance data downloader
├── fast_import.py       # Fast COPY-based DB import
├── smart_import.py      # Incremental DB import
├── parallel_hmm.py      # Multi-machine HMM processor
├── sort_results.py      # Result sorter (run after tests)
├── sync_to_server.py    # Sync scripts to server
├── SYNC_STATUS.md       # This file
└── ALREADY_DONE.md      # Completed tasks archive
```

---

## Result Sorting System

After running any test, sort the results:

```bash
# Interactive mode (recommended)
python sort_results.py

# Direct mode
python sort_results.py my_results.json winning
python sort_results.py my_results.json losing
python sort_results.py my_results.json neutral
```

**Thresholds:**
- **Winning:** avg PnL > $500
- **Losing:** avg PnL < -$500
- **Neutral:** -$500 to $500

---

## Current Best Configuration

From overnight tests (2026-02-05):

| Setting | Value |
|---------|-------|
| Timeframe | **4H** |
| LSTM | Yes (4H trained) |
| Regime Filter | **no_high_vol** |
| Entry TF | 30m (for MTF combo) |
| Algorithm | DQN or PPO |
| Timesteps | 200,000 |

**Best MTF Combo:** 30m+4h → +$2,102.85 total
**Best Single Config:** 4H + LSTM + no_high_vol → +$3,103.58 total

---

## Server & Database

### Locations
| Location | Path |
|----------|------|
| Thomas PC | `C:\Users\thoma\alphatrader\` |
| Server | `/home/shared/alphatrader/` |
| Server Scripts | `/home/shared/alphatrader/scripts/` |
| **Server HDD** | `/mnt/data/` (1.7TB free) |
| Bulk Data | `/mnt/data/binance_bulk/spot/` (97GB) |

### Database
```
Host: 192.168.0.160 (local) / 100.110.101.78 (Tailscale)
Database: zenkai_data
User: zenkai
Password: ZenkaiHQ2000
```

### SSH Access
```bash
ssh goku@192.168.0.160          # Local network
ssh goku@100.110.101.78         # Tailscale

# From goku PC (uses Thoma's key)
ssh -i /c/Users/Thoma/.ssh/id_ed25519 goku@192.168.0.160
```

### Sync Commands
```bash
# Easy sync (recommended)
python sync_to_server.py

# Manual sync
scp train_*.py goku@192.168.0.160:/home/shared/alphatrader/
scp -r src goku@192.168.0.160:/home/shared/alphatrader/
```

---

## Hardware Roles

| Machine | GPU | Role |
|---------|-----|------|
| Thomas PC | RTX 4070 Super 12GB | LSTM & RL training |
| Fips PC | RTX 5060 Ti 16GB | LSTM & RL training |
| Server | AMD RX 580 (no CUDA) | Database, HMM, daemons only |

**Fips PyTorch:** `pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128`

---

## Experiment Tracking

**MLflow UI:** `http://192.168.0.160:5000`

All experiments auto-logged:
- Parameters: symbol, timeframe, epochs, etc.
- Metrics: accuracy, PnL, win rate, drawdown
- Artifacts: summary JSON files

---

## Auto-Detect Pattern (use in all scripts)

```python
def _get_db_host():
    if os.getenv("ZENKAI_DB_HOST"): return os.getenv("ZENKAI_DB_HOST")
    local_ip, tailscale_ip = "192.168.0.160", "100.110.101.78"
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        if sock.connect_ex((local_ip, 5432)) == 0: return local_ip
    except: pass
    return tailscale_ip
```

---

## Red Flags (Be Suspicious If)

- Sharpe > 3.0
- Win rate > 80%
- Profit factor > 2.0
- Out-of-sample vs in-sample gap > 20%
- Results collapse when parameters change by 10%
- Low trade count (< 50 trades)

---

## Binance Bulk Data System

### Overview
Downloaded ALL historical 1-minute data from Binance (data.binance.vision) - no API limits.

### Data Location
```
Server HDD: /mnt/data/binance_bulk/spot/    # 97 GB, 446 symbols
Script:     /home/shared/alphatrader/scripts/bulk_download_binance.py
```

### Dead Coin Filter
The `symbol_status` table tracks which coins are still actively trading on Binance.
- **Active coins:** Can be used for training (will generate real trades)
- **Delisted coins:** Data preserved for historical analysis, excluded from training

**Training query (ALWAYS USE THIS):**
```sql
SELECT * FROM ohlcv
WHERE symbol IN (SELECT symbol FROM symbol_status WHERE is_active = TRUE)
```

### Commands
```bash
# Check available data
python bulk_download_binance.py --check

# Download ALL symbols (run on server)
python bulk_download_binance.py --all --workers 10

# Import to database
python bulk_download_binance.py --import-to-db

# Sync symbol status (mark delisted coins)
python bulk_download_binance.py --sync-status

# List delisted coins
python bulk_download_binance.py --list-delisted
```

### Check Import Progress
```bash
ssh goku@192.168.0.160 "grep -c 'rows$' /home/shared/alphatrader/logs/db_import.log"
ssh goku@192.168.0.160 "tail -5 /home/shared/alphatrader/logs/db_import.log"
```

### Full Pipeline (after bulk import)
```bash
# 1. Import completes → auto-syncs symbol status
# 2. Run HMM regime detection (REQUIRED before LSTM training!)

# Option A: Single-machine (server daemon)
cd /home/shared/alphatrader/server
python3 hmm_processor.py --status    # Check progress
python3 hmm_processor.py --daemon    # Run in background

# Option B: Multi-machine parallel (RECOMMENDED - 3x faster)
# See "Parallel HMM Processing" section below

# 3. Check HMM progress
tail -f /home/shared/alphatrader/logs/hmm_processor.log
```

**IMPORTANT:** LSTM cannot train without HMM regimes. The `regime` column must be populated first!

---

## Parallel HMM Processing

HMM regime detection is **CPU-bound** (matrix operations), so it benefits from multi-machine parallelization.

**NEW: Dynamic Work Distribution** - Machines automatically claim and process symbols. Add more power anytime!

### Quick Start
```bash
# Check status (any machine)
python parallel_hmm.py --status

# Start processing (auto-claims work)
python parallel_hmm.py -w 8 --name server      # Server
python parallel_hmm.py -w 16 --name thomas     # Thomas PC
python parallel_hmm.py -w 8 --name laptop      # Laptop
python parallel_hmm.py -w 12 --name fips       # Fips PC

# Reset all claims (fresh start)
python parallel_hmm.py --reset
```

### How It Works
1. Each machine registers with a name and claims batches of symbols
2. Work is distributed automatically - no manual splitting needed
3. If a machine goes offline, its unclaimed work is released after 10 minutes
4. Add more machines anytime - they auto-join and claim available work

### Machine Roles (Recommended)
| Machine | CPU Threads | Command |
|---------|-------------|---------|
| Server | 8 cores | `python parallel_hmm.py -w 8 --name server` |
| Thomas PC | 16 threads | `python parallel_hmm.py -w 16 --name thomas` |
| Laptop | 8-12 threads | `python parallel_hmm.py -w 8 --name laptop` |
| Fips PC | 12 threads | `python parallel_hmm.py -w 12 --name fips` |

### Why Multi-Machine Works for HMM (but not Import)
- **Import = I/O bound** → PostgreSQL writes bottleneck, multi-machine doesn't help
- **HMM = CPU bound** → Matrix math, scales linearly with cores across machines
- Each machine processes different symbols → no database conflicts
- Coordination via `hmm_work_claims` table in database

### Import Scripts
| Script | Use Case |
|--------|----------|
| `fast_import.py` | Fresh import - uses COPY (10x faster, no conflict check) |
| `smart_import.py` | Incremental import - skips already-imported symbols |
| `parallel_import.py` | Multi-worker import (still I/O bound though) |

```bash
# Recommended: Fast import for fresh DB
python fast_import.py

# For incremental updates
python smart_import.py
```

### Data Stats (2026-02-05)
- **446 active** USDT pairs (TRADING status)
- **203 delisted** symbols tagged in symbol_status
- **18,090 CSV files** (1 per symbol per month, 2017-2026)
- **97 GB** raw data on /mnt/data HDD (1.7TB free)
- **Import rate:** ~50,000-100,000 rows/sec with COPY method

---

## Current Status

### Active Development
- **Bulk data pipeline running** (download ✓ → import → HMM → LSTM)
- Multi-timeframe RL optimization
- Regime-based filtering
- Production readiness testing

### Pipeline Progress (2026-02-05)
1. ✅ **Download complete**: 446 symbols, 97GB, 18,090 CSV files
2. 🔄 **Import running**: `fast_import.py` on server
3. ⏳ **HMM**: Will auto-start after import (or use `parallel_hmm.py`)
4. ⏳ **LSTM training**: After HMM completes

### On Hold
- **Zenkai Signal Hub** (Telegram Mini-App) - waiting for consistent profits
  - Architecture doc: `C:\Users\thoma\Desktop\ZENKAI_SIGNAL_HUB_ARCHITECTURE.md`

### See Also
- `ALREADY_DONE.md` - Completed tasks and results archive
- `results/winning/` - Profitable test results
- MLflow UI for full experiment history

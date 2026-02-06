# ⚡ Zenkai Signal Hub — Telegram Bot

AI-powered trading signal bot for Telegram.

---

## 📋 Features

### Public Commands
- `/start` — Welcome message & subscribe
- `/help` — Show all commands
- `/stats` — View signal performance
- `/about` — About Zenkai
- `/disclaimer` — MiFID II disclaimer
- `/subscribe` — Subscribe to signals
- `/unsubscribe` — Unsubscribe

### Admin Commands
- `/signal` — Interactive signal builder
- `/update [id] [status]` — Update signal status
- `/broadcast <message>` — Send to all subscribers
- `/subscribers` — View subscriber count
- `/active` — View active signals

### Signal Updates
- TP1 Hit, TP2 Hit, TP3 Hit
- Stopped Out
- Breakeven
- Cancelled

---

## 🚀 Quick Start

### 1. Create Telegram Bot

1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. Send `/newbot` and follow instructions
3. Copy the bot token

### 2. Get Your User ID

1. Message [@userinfobot](https://t.me/userinfobot)
2. Copy your user ID

### 3. Configure

```bash
cp .env.example .env
```

Edit `.env`:
```env
BOT_TOKEN=your_bot_token_here
ADMIN_IDS=your_user_id_here
```

### 4. Install & Run

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run
python bot.py
```

---

## 📁 File Structure

```
zenkai-signal-hub/
├── bot.py              # Main Telegram bot
├── config.py           # Configuration & messages
├── db.py               # SQLite database
├── signal_handler.py   # Signal creation & formatting
├── stats.py            # Performance statistics
├── webhook.py          # AlphaTrader integration (placeholder)
├── requirements.txt    # Dependencies
├── .env.example        # Environment template
├── .env               # Your config (create this)
└── zenkai_signals.db  # SQLite database (auto-created)
```

---

## 📡 Signal Format

```
🟢 LONG SIGNAL — BTC/USDT

📊 Timeframe: 4H
🎯 Confidence: HIGH

━━━━━━━━━━━━━━━━━━━━━━

Entry Zone: $94,500 — $95,200
🛑 Stop Loss: $93,100
✅ TP1: $96,800
✅ TP2: $98,500
✅ TP3: $101,000

📐 R:R (TP1): 1:2.4

━━━━━━━━━━━━━━━━━━━━━━

📝 Note:
CHoCH on 4H with unfilled bullish FVG at entry zone.

━━━━━━━━━━━━━━━━━━━━━━

⚡ Zenkai Signal Hub | @ZenkaiSignals
⚠️ Not financial advice. DYOR.

🆔 Signal ID: #1
```

---

## 🔌 AlphaTrader Integration

The `webhook.py` module provides multiple integration options:

### Option 1: Direct Function Call
```python
from webhook import receive_signal_from_alphatrader

await receive_signal_from_alphatrader(
    pair="BTC/USDT",
    direction="LONG",
    entry_low=94500,
    entry_high=95200,
    stop_loss=93100,
    tp1=96800,
    tp2=98500,
    tp3=101000,
    timeframe="4H",
    confidence=0.85,
    analysis="CHoCH on 4H..."
)
```

### Option 2: JSON Data
```python
from webhook import process_alphatrader_signal

data = {
    "pair": "BTC/USDT",
    "direction": "LONG",
    "entry_zone": [94500, 95200],
    "stop_loss": 93100,
    "take_profits": [96800, 98500, 101000],
    "timeframe": "4H",
    "confidence": 0.85,
    "analysis": "CHoCH on 4H..."
}

await process_alphatrader_signal(data)
```

### Option 3: REST Webhook (requires FastAPI)
See `webhook.py` for FastAPI endpoint example.

---

## ⚠️ MiFID II Compliance

Every signal includes a disclaimer. The full disclaimer is available via `/disclaimer`.

**NEVER use:**
- "Guaranteed"
- "Risk-free"
- "Sure thing"
- "100% accurate"

**ALWAYS frame as:**
- Educational/informational
- Not financial advice
- Past performance ≠ future results

---

## 🏃 Running in Production

### PM2
```bash
pm2 start bot.py --interpreter python --name zenkai-signal-hub
```

### Systemd
```ini
[Unit]
Description=Zenkai Signal Hub
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/zenkai-signal-hub
ExecStart=/path/to/venv/bin/python bot.py
Restart=always

[Install]
WantedBy=multi-user.target
```

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "bot.py"]
```

---

## 📊 Database Schema

### signals
- id, pair, direction
- entry_low, entry_high, stop_loss
- tp1, tp2, tp3
- timeframe, confidence, note
- status (ACTIVE, TP1_HIT, TP2_HIT, TP3_HIT, STOPPED, CANCELLED)
- tp1_hit, tp2_hit, tp3_hit (boolean)
- created_at, closed_at
- result_pct

### subscribers
- user_id, username, first_name
- subscribed_at, is_active

### signal_updates
- id, signal_id, update_type
- message, created_at

---

## ⚡ Zenkai Corporation

Built with 💚 by Son Goku

**Evolve or Die**

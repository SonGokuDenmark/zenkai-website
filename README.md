# Zenkai Corporation

Master repository for all Zenkai Corporation projects.

## Projects

| Project | Description | Status |
|---------|-------------|--------|
| **alphatrader** | ML-powered crypto trading engine (HMM + LSTM + RL) | Active |
| **telegram-bot** | Zenkai Signal Hub - trading signals via Telegram | Active |
| **discord-bot** | Community Discord bot | Active |
| **website** | zenkaicorp.com landing page | Active |
| **indicators** | Zenkai Edge - TradingView indicators (free + premium) | v1 Live |
| **ogame-tools** | OGame automation tools | Planned |
| **youtube** | YouTube channel content & branding | Planned |
| **products** | Digital products (LemonSqueezy) | Planned |

## Structure

```
C:\Zenkai\
├── alphatrader\       # Core ML trading engine
├── telegram-bot\      # Signal distribution
├── discord-bot\       # Community management
├── website\           # Landing page
├── indicators\        # TradingView scripts
├── specs\             # Build specifications
├── docs\              # Internal documentation
└── assets\            # Shared branding
```

## Infrastructure

- **Server**: zenkaiserver (Ubuntu, PostgreSQL, 500M+ OHLCV rows)
- **Network**: Tailscale mesh for remote access
- **ML Pipeline**: HMM regime detection -> LSTM classification -> RL trading
- **Data**: Binance historical + live feeds

## Security

See [SECURITY.md](SECURITY.md) for credential policies.
All secrets live in `.env` files (never committed).

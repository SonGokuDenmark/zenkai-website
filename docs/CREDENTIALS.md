# Credential Reference

Where credentials are stored (NOT the actual secrets).

## Database (PostgreSQL)
- Host: zenkaiserver (192.168.0.160 / Tailscale: 100.110.101.78)
- Database: zenkai_data
- User: zenkai
- Password: stored in each project's .env file
- Port: 5432

## Discord Bot
- Token: Discord Developer Portal -> Applications -> Zenkai Bot
- Stored in: C:\Zenkai\discord-bot\.env

## Telegram Bot
- Token: @BotFather -> @ZenkaiSignalBot
- Stored in: C:\Zenkai\telegram-bot\.env

## AlphaTrader
- Discord webhook: stored in C:\Zenkai\alphatrader\.env
- DB credentials: stored in C:\Zenkai\alphatrader\.env

## Domain
- Provider: Namecheap
- Domain: zenkaicorp.com

## Hosting
- Website: Vercel (connected to git repo)
- Server: Local Ubuntu machine (zenkaiserver)

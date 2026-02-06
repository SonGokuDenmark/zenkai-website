# Zenkai Corporation — Project Intelligence

## Team Communication
Before starting work, always read:
- `C:\Zenkai\.team\BULMA_TO_LARS.md` — Tasks and fixes from Bulma (priority list)
- `C:\Zenkai\.team\LARS_TO_BULMA.md` — Drop your requests/questions for Bulma here

When you complete a task from BULMA_TO_LARS.md, move it to the ✅ COMPLETED section with the date.
When ALL tasks are done, ALWAYS update LARS_TO_BULMA.md with a status report so Bulma knows what was done, what changed, and any issues found.

## Project Structure
- `C:\Zenkai\` — Root workspace
- `alphatrader/` — AI trading system (LSTM, RL, HMM)
- `website/` — zenkaicorp.com (deployed on Vercel, auto-deploys on git push)
- `indicators/` — TradingView Pine Script (Zenkai Edge)
- `discord-bot/` — Zenkai Discord bot
- `telegram-bot/` — Signal Hub bot
- `specs/` — Bulma writes specs here for Lars to implement
- `.team/` — Bulma ↔ Lars task relay

## Key Rules
- NEVER hardcode secrets — use .env files
- All ICT references have been rebranded to "smart money concepts" or "Zenkai Edge"
- Website auto-deploys via Vercel on `git push`
- Pine Script must use `barstate.isconfirmed` — no repainting
- Check `C:\Zenkai\SECURITY.md` before every commit

# Zenkai Corporation — Security Policy

## Rules
1. NEVER hardcode tokens, keys, or credentials in source code
2. ALL secrets go in .env files (one per project)
3. .env files are NEVER committed to git
4. Every project MUST have a .env.example showing required vars (without real values)
5. Admin IDs are stored in .env, not in code
6. Database files are gitignored
7. Webhook endpoints use authentication tokens
8. API rate limiting on all public endpoints

## Credential Locations (reference only)
- Telegram Bot Token: @BotFather -> @ZenkaiSignalBot
- Discord Bot Token: Discord Developer Portal
- Domain: Namecheap (zenkaicorp.com)
- LemonSqueezy: (to be set up)
- TradingView: (to be verified)
- Database: PostgreSQL on zenkaiserver (192.168.0.160)

## Before Every Commit
- [ ] Run: grep -r "TOKEN\|KEY\|SECRET\|PASSWORD" --include="*.py" --include="*.js"
- [ ] Verify no .env files in staged changes
- [ ] Check no hardcoded IDs

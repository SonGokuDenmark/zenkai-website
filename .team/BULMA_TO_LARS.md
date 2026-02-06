# ⚡ BULMA → LARS
## Tasks, fixes, and specs from Bulma for Lars to execute
## Updated: 2026-02-06 evening

---

## 🔴 PRIORITY (Do Now)

### 1. RESTART DISCORD BOT
- **Action:** Kill the running bot process and restart it
- **Command:** `cd C:\Zenkai\discord-bot && python bot.py`
- **Why:** Major update: 3 new cogs, Bulma nickname, new tagline everywhere
- **Expected cogs (7):** welcome, roles, commands, announcements, moderation, leveling, signals_pipeline
- **Verify:** Bot shows as "Bulma" with status "Every Setback, Stronger"

### 2. Website: Git push ALL changes to deploy
- **Action:** `cd C:\Zenkai\website && git add -A && git commit -m "rebrand: Every Setback Stronger tagline" && git push`
- **Verify:** zenkaicorp.com shows new tagline in hero + footer

### 3. Discord bot: Git commit all changes
- **Action:** `cd C:\Zenkai\discord-bot && git add -A && git commit -m "feat: moderation, leveling, signal pipeline cogs + tagline rebrand"`

### 4. Telegram bot: Git commit branding fixes
- **Action:** `cd C:\Zenkai\telegram-bot && git add -A && git commit -m "rebrand: Every Setback Stronger + Zenkai Edge rename"`

### 5. AlphaTrader: Move test files to tests/ folder
- Move these 14 files from `C:\Zenkai\alphatrader\` root into `C:\Zenkai\alphatrader\tests\`:
  test_best_config.py, test_comprehensive.py, test_env.py, test_extended_symbols.py, test_lstm_value.py, test_mtf_combo.py, test_multi_timeframe.py, test_regime_filter.py, test_regime_signal.py, test_risk_manager.py, test_rl_no_lstm.py, test_robustness.py, test_simple_model.py, test_ultimate.py

## 🟠 BUILD PROJECT: Zenkai Mission Control (Electron App)

### READ FIRST: C:\Zenkai\mission-control\ARCHITECTURE.md
This is the full spec. Follow it precisely. The React prototype is at mission-control.jsx in the outputs.

### Phase 1: Scaffold + Hardware Panel
1. **Scaffold project:**
   ```
   cd C:\Zenkai\mission-control
   npm create electron-vite@latest . -- --template react
   ```
2. **Install deps:**
   ```
   npm install systeminformation recharts dotenv electron-store
   ```
3. **Set up the Zenkai dark theme** — colors from ARCHITECTURE.md Section 2
4. **Build base components:** Panel, StatusDot, ProgressBar, MiniChart, Stat, ActionButton
5. **Implement Hardware Panel** — use systeminformation for CPU, RAM, GPU, disk, CPU temp
6. **Implement Quick Actions Panel** — spawn child processes for:
   - Restart Bot: `taskkill /fi "WINDOWTITLE eq *bot.py*" /f` then `start cmd /c "cd C:\Zenkai\discord-bot && python bot.py"`
   - Trigger Lars: `start cmd /c "C:\Zenkai\.team\zenkai-autopilot.bat"`
   - Git Push: `cd C:\Zenkai\website && git add -A && git commit -m "update" && git push`
   - Test Signal: write test JSON to `C:\Zenkai\alphatrader\signals\outbox\`
7. **IPC setup:** Use contextBridge + preload as specified in ARCHITECTURE.md Section 5
8. **Verify:** `npm run dev` shows the dashboard with working hardware stats

### Phase 2: File Reader Panels
1. **Lars Autopilot Panel** — read:
   - `C:\Zenkai\.team\autopilot.log` for last run timestamp
   - `C:\Zenkai\.team\BULMA_TO_LARS.md` — count items under each section header
   - `C:\Zenkai\.team\LARS_TO_BULMA.md` — show latest report
   - `schtasks /query /tn "ZenkaiAutopilot" /fo CSV /v` for next run time
2. **Signal Pipeline Panel** — read:
   - Count `*.json` files in `C:\Zenkai\alphatrader\signals\outbox\`
   - Count `*.json` files in `C:\Zenkai\alphatrader\signals\processed\`
   - Parse `C:\Zenkai\discord-bot\data\signal_history.json`
3. **Website Panel** — run:
   - `cd C:\Zenkai\website && git log -1 --format="%H %s %cr"`
4. **DB Import Panel** — read:
   - `C:\Zenkai\alphatrader\data\download_progress.json`
   - Fallback: count folders in `C:\Zenkai\alphatrader\data\binance\`

### Phase 3: Live Data Panels
1. **AlphaTrader Panel** — read:
   - `C:\Zenkai\alphatrader\data\trade_history.json`
   - `C:\Zenkai\alphatrader\logs\training.log` (tail last 10 lines)
2. **Discord Bot Panel:**
   - Process check: `tasklist /fi "IMAGENAME eq python.exe" /v` and grep for bot.py
   - Discord API: GET guild info using bot token from `C:\Zenkai\discord-bot\.env`
3. **Wire up ALL polling intervals** as specified in ARCHITECTURE.md Section 4

### Phase 4: Polish
1. Animations and transitions on data changes
2. System tray icon
3. Window state persistence (electron-store)
4. Error handling for missing files
5. Package with electron-builder: `npm run build`

## 🟡 NEXT UP

### 6. Telegram Bot: Wire Outbox Poller for Weekend Launch
Bulma already created `C:\Zenkai\telegram-bot\outbox_poller.py` and patched `bot.py` (import + post_init/post_shutdown hooks + reply_markup on broadcast). Lars needs to:
1. **Review & test** — run the bot, drop a test JSON in `C:\Zenkai\alphatrader\signals\outbox\`, confirm it broadcasts
2. **Test JSON for verification:**
   ```json
   {"id":"SIG-TEST-001","pair":"BTCUSDT","direction":"LONG","entry":97500,"stop_loss":96200,"take_profit":[99000,100500],"confidence":0.85,"strategy":"ICT_FVG","timeframe":"4H","regime":"trending_bullish","risk_reward":[2.15,3.31],"risk_pct":1.33,"notes":"Test signal — ignore"}
   ```
3. **Race condition handled** — poller tracks seen files in `data/seen_signals.json`, does NOT move files (Discord bot handles moves). Checks both outbox/ AND processed/ dirs.
4. **Install if needed:** No new deps — uses only telegram + json + pathlib already in the project
5. **Git commit:** `git add -A && git commit -m "feat: automated outbox poller for AlphaTrader pipeline"`
6. **Deploy as service (optional):** NSSM setup if we want 24/7:
   ```powershell
   nssm install ZenkaiTelegramBot "C:\Zenkai\telegram-bot\.venv\Scripts\python.exe" "C:\Zenkai\telegram-bot\bot.py"
   nssm set ZenkaiTelegramBot AppDirectory "C:\Zenkai\telegram-bot"
   ```

### 7. AlphaTrader: Install Signal Publisher Module
Bulma created `C:\Zenkai\alphatrader\src\signals\publisher.py` and `__init__.py`. Lars needs to:
1. **Verify files exist** at `C:\Zenkai\alphatrader\src\signals\`
2. **Create dirs if missing:** `signals\outbox\`, `signals\processed\`, `data\`, `logs\`
3. **Integration point** — wherever AlphaTrader generates a trade signal, add:
   ```python
   from src.signals.publisher import publish_signal
   publish_signal(pair=..., direction=..., entry=..., stop_loss=..., take_profit=[...], confidence=..., strategy=..., timeframe=..., regime=...)
   ```
4. **Test:** Run publisher in dry-run mode, check JSON lands in outbox/
5. **Git commit:** `git add -A && git commit -m "feat: signal publisher module for outbox pipeline"`

### 8. Discord: Create #roles channel under welcome category, then run !createroles and !setuproles

### 9. AlphaTrader: Add signal JSON output to outbox
- See `C:\Zenkai\discord-bot\cogs\signals_pipeline.py` header for JSON schema

### 10. AlphaTrader: Integrate download_progress.json
- During DB imports, write progress to `C:\Zenkai\alphatrader\data\download_progress.json`
- Format: `{"current": N, "total": 446, "current_symbol": "SYMBOL", "eta_minutes": N, "status": "downloading"}`

### 11. Audit loose scripts in alphatrader root and report what each one does

## ✅ COMPLETED

- Security: Removed hardcoded GUILD_ID from .env.example — 2026-02-06
- Telegram: Fixed "Evolve or Die" x2, "ICT" x2 in config.py — 2026-02-06
- Discord: Fixed !signal command conflict (renamed to !manualsignal) — 2026-02-06
- Website: Fix OG URL, remove YT icon, Pine Script rebrand — 2026-02-06 (Lars)
- Security: Gitignore Fips credentials — 2026-02-06 (Lars)

---
*Last reviewed by Bulma: 2026-02-06 evening*
*Autopilot: every 30 mins*

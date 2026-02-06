"""
═══════════════════════════════════════════════════════════════════════════════
⚡ Zenkai Signal Hub — Configuration
© 2026 Zenkai Corporation
═══════════════════════════════════════════════════════════════════════════════
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Bot Configuration
# ─────────────────────────────────────────────────────────────────────────────

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN not found in environment variables!")

# Admin Telegram User IDs (comma-separated in .env)
ADMIN_IDS_STR = os.getenv("ADMIN_IDS", "")
ADMIN_IDS = [int(id.strip()) for id in ADMIN_IDS_STR.split(",") if id.strip()]

# Channel for signal broadcasts (optional, can broadcast to subscribers directly)
SIGNAL_CHANNEL_ID = os.getenv("SIGNAL_CHANNEL_ID", "")

# ─────────────────────────────────────────────────────────────────────────────
# Branding
# ─────────────────────────────────────────────────────────────────────────────

BOT_NAME = "Zenkai Signal Hub"
BOT_USERNAME = "@ZenkaiSignals"
BOT_VERSION = "1.0.0"

# Links
WEBSITE_URL = "https://zenkai.corp"
DISCORD_URL = "https://discord.gg/zenkai"
TRADINGVIEW_URL = "https://tradingview.com/u/ZenkaiTrading"
TELEGRAM_CHANNEL = "https://t.me/ZenkaiSignals"

# ─────────────────────────────────────────────────────────────────────────────
# Messages
# ─────────────────────────────────────────────────────────────────────────────

DISCLAIMER_SHORT = "⚠️ Not financial advice. DYOR."

DISCLAIMER_FULL = """
⚠️ <b>IMPORTANT DISCLAIMER</b>

Signals provided by Zenkai Signal Hub are for <b>educational and informational purposes only</b>.

They do <b>NOT</b> constitute:
• Financial advice
• Investment recommendations
• Solicitation to trade

<b>Past performance is not indicative of future results.</b>

Trading cryptocurrencies and other financial instruments carries significant risk. You may lose some or all of your invested capital.

Always:
• Do your own research (DYOR)
• Never trade with money you cannot afford to lose
• Consult a licensed financial advisor if needed
• Understand the risks involved in trading

By using this service, you acknowledge that you understand and accept these risks.

⚡ Zenkai Signal Hub
"""

WELCOME_MESSAGE = """
⚡ <b>Welcome to Zenkai Signal Hub!</b>

Your AI-powered trading signal service.

<b>What we offer:</b>
• 📊 High-quality trading signals
• 🤖 Powered by AlphaTrader AI
• 📈 ICT-based market analysis
• ⏱️ Real-time alerts

<b>Commands:</b>
/help — Show all commands
/stats — View signal performance
/about — About Zenkai
/disclaimer — Important legal info

<b>Links:</b>
🌐 <a href="{website}">Website</a>
💬 <a href="{discord}">Discord</a>
📊 <a href="{tradingview}">TradingView</a>

━━━━━━━━━━━━━━━━━━━━━━

{disclaimer}

━━━━━━━━━━━━━━━━━━━━━━

⚡ <b>Evolve or Die</b>
"""

ABOUT_MESSAGE = """
⚡ <b>About Zenkai Signal Hub</b>

Zenkai Signal Hub is part of <b>Zenkai Corporation</b> — building the future of trading, gaming, and automation powered by AI.

<b>Our Projects:</b>
• 📊 Zenkai ICT Toolkit (TradingView)
• 🤖 AlphaTrader (AI Trading System)
• 📡 Signal Hub (You're here!)
• 🎮 OGame Strategy Tools
• 🏠 Bulma AI (Smart Home)
• 📺 Chaotic Respawn Bros (YouTube)

<b>Links:</b>
🌐 <a href="{website}">Website</a>
💬 <a href="{discord}">Discord</a>
📊 <a href="{tradingview}">TradingView</a>

<b>Founded by:</b> Son Goku 🇩🇰

⚡ <b>Evolve or Die</b>
"""

# ─────────────────────────────────────────────────────────────────────────────
# Signal Configuration
# ─────────────────────────────────────────────────────────────────────────────

TIMEFRAMES = ["1m", "5m", "15m", "30m", "1H", "4H", "1D", "1W"]
CONFIDENCE_LEVELS = ["LOW", "MEDIUM", "HIGH"]
DIRECTIONS = ["LONG", "SHORT"]

# Database
DATABASE_PATH = "zenkai_signals.db"

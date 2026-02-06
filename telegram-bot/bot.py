"""
═══════════════════════════════════════════════════════════════════════════════
⚡ ZENKAI SIGNAL HUB — Telegram Bot v1.0
© 2026 Zenkai Corporation
═══════════════════════════════════════════════════════════════════════════════

AI-powered trading signal bot for Telegram.
"""

import logging
from datetime import datetime

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ConversationHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ParseMode

from config import (
    BOT_TOKEN,
    ADMIN_IDS,
    SIGNAL_CHANNEL_ID,
    BOT_NAME,
    BOT_USERNAME,
    WEBSITE_URL,
    DISCORD_URL,
    TRADINGVIEW_URL,
    DISCLAIMER_SHORT,
    DISCLAIMER_FULL,
    WELCOME_MESSAGE,
    ABOUT_MESSAGE,
    TIMEFRAMES,
    CONFIDENCE_LEVELS,
    DIRECTIONS,
)
from db import db, Signal
from signal_handler import (
    SignalDraft,
    active_drafts,
    format_signal_message,
    format_signal_update,
    format_stats_message,
    create_signal_from_draft,
    update_signal,
    get_stats,
    get_signal,
    get_active_signals,
)
import webhook
from outbox_poller import OutboxPoller

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("zenkai-signal-hub")

# Reduce noise from telegram library
logging.getLogger("httpx").setLevel(logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_admin(user_id: int) -> bool:
    """Check if user is an admin."""
    return user_id in ADMIN_IDS


async def broadcast_to_subscribers(app, message: str, parse_mode: str = ParseMode.HTML, reply_markup=None):
    """Broadcast a message to all active subscribers."""
    subscribers = db.get_active_subscribers()
    success = 0
    failed = 0

    for sub in subscribers:
        try:
            await app.bot.send_message(
                chat_id=sub.user_id,
                text=message,
                parse_mode=parse_mode,
                reply_markup=reply_markup
            )
            success += 1
        except Exception as e:
            logger.warning(f"Failed to send to {sub.user_id}: {e}")
            failed += 1

    return success, failed


async def broadcast_signal(signal: Signal):
    """Broadcast a signal to subscribers (callback for webhook)."""
    # This will be called from webhook.py
    # We need access to the application, which we'll set during startup
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Command Handlers — Public
# ─────────────────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command - welcome and subscribe."""
    user = update.effective_user

    # Add to subscribers
    is_new = db.add_subscriber(user.id, user.username, user.first_name)

    # Format welcome message
    message = WELCOME_MESSAGE.format(
        website=WEBSITE_URL,
        discord=DISCORD_URL,
        tradingview=TRADINGVIEW_URL,
        disclaimer=DISCLAIMER_SHORT
    )

    # Keyboard
    keyboard = [
        [InlineKeyboardButton("📊 View Stats", callback_data="stats")],
        [
            InlineKeyboardButton("🌐 Website", url=WEBSITE_URL),
            InlineKeyboardButton("💬 Discord", url=DISCORD_URL)
        ],
        [InlineKeyboardButton("⚠️ Disclaimer", callback_data="disclaimer")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        message,
        parse_mode=ParseMode.HTML,
        reply_markup=reply_markup,
        disable_web_page_preview=True
    )

    if is_new:
        logger.info(f"New subscriber: {user.id} (@{user.username})")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command."""
    user = update.effective_user

    message = """
⚡ <b>Zenkai Signal Hub — Commands</b>

<b>📋 General Commands</b>
/start — Welcome & subscribe
/help — Show this help
/stats — View signal performance
/about — About Zenkai
/disclaimer — Important disclaimer
/subscribe — Subscribe to signals
/unsubscribe — Unsubscribe from signals

<b>🔗 Quick Links</b>
/website — Zenkai website
/discord — Discord server
"""

    # Add admin commands if admin
    if is_admin(user.id):
        message += """
<b>🔧 Admin Commands</b>
/signal — Create new signal
/update — Update signal status
/broadcast — Send message to all
/subscribers — View subscriber count
/active — View active signals
"""

    await update.message.reply_text(message, parse_mode=ParseMode.HTML)


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /stats command."""
    stats = get_stats()
    message = format_stats_message(stats)

    await update.message.reply_text(message, parse_mode=ParseMode.HTML)


async def cmd_about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /about command."""
    message = ABOUT_MESSAGE.format(
        website=WEBSITE_URL,
        discord=DISCORD_URL,
        tradingview=TRADINGVIEW_URL
    )

    keyboard = [
        [
            InlineKeyboardButton("🌐 Website", url=WEBSITE_URL),
            InlineKeyboardButton("💬 Discord", url=DISCORD_URL)
        ],
        [InlineKeyboardButton("📊 TradingView", url=TRADINGVIEW_URL)]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        message,
        parse_mode=ParseMode.HTML,
        reply_markup=reply_markup,
        disable_web_page_preview=True
    )


async def cmd_disclaimer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /disclaimer command."""
    await update.message.reply_text(DISCLAIMER_FULL, parse_mode=ParseMode.HTML)


async def cmd_subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /subscribe command."""
    user = update.effective_user
    is_new = db.add_subscriber(user.id, user.username, user.first_name)

    if is_new:
        await update.message.reply_text(
            "✅ <b>You're now subscribed!</b>\n\n"
            "You'll receive trading signals directly in this chat.\n\n"
            f"{DISCLAIMER_SHORT}",
            parse_mode=ParseMode.HTML
        )
    else:
        await update.message.reply_text(
            "✅ You're already subscribed!\n\n"
            "Use /unsubscribe to stop receiving signals.",
            parse_mode=ParseMode.HTML
        )


async def cmd_unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /unsubscribe command."""
    user = update.effective_user
    was_subscribed = db.remove_subscriber(user.id)

    if was_subscribed:
        await update.message.reply_text(
            "👋 <b>You've been unsubscribed.</b>\n\n"
            "You won't receive any more signals.\n"
            "Use /subscribe to re-subscribe anytime.",
            parse_mode=ParseMode.HTML
        )
    else:
        await update.message.reply_text(
            "You weren't subscribed.\n"
            "Use /subscribe to start receiving signals.",
            parse_mode=ParseMode.HTML
        )


async def cmd_website(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /website command."""
    await update.message.reply_text(
        f"🌐 <b>Zenkai Website</b>\n\n{WEBSITE_URL}",
        parse_mode=ParseMode.HTML
    )


async def cmd_discord(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /discord command."""
    await update.message.reply_text(
        f"💬 <b>Zenkai Discord</b>\n\n{DISCORD_URL}",
        parse_mode=ParseMode.HTML
    )


# ─────────────────────────────────────────────────────────────────────────────
# Callback Query Handlers
# ─────────────────────────────────────────────────────────────────────────────

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline keyboard callbacks."""
    query = update.callback_query
    await query.answer()

    if query.data == "stats":
        stats = get_stats()
        message = format_stats_message(stats)
        await query.message.reply_text(message, parse_mode=ParseMode.HTML)

    elif query.data == "disclaimer":
        await query.message.reply_text(DISCLAIMER_FULL, parse_mode=ParseMode.HTML)


# ─────────────────────────────────────────────────────────────────────────────
# Admin Commands
# ─────────────────────────────────────────────────────────────────────────────

async def cmd_subscribers(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /subscribers command (admin only)."""
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("❌ Admin only command.")
        return

    count = db.get_subscriber_count()
    await update.message.reply_text(
        f"📊 <b>Subscriber Count</b>\n\n"
        f"Active subscribers: <b>{count}</b>",
        parse_mode=ParseMode.HTML
    )


async def cmd_active(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /active command (admin only) - show active signals."""
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("❌ Admin only command.")
        return

    signals = get_active_signals()

    if not signals:
        await update.message.reply_text("No active signals.")
        return

    message = "📊 <b>Active Signals</b>\n\n"
    for sig in signals:
        message += f"<b>#{sig.id}</b> — {sig.pair} {sig.direction}\n"
        message += f"   Entry: ${sig.entry_low:,.0f}-${sig.entry_high:,.0f}\n"
        message += f"   Status: {sig.status}\n\n"

    await update.message.reply_text(message, parse_mode=ParseMode.HTML)


async def cmd_broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /broadcast command (admin only)."""
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("❌ Admin only command.")
        return

    if not context.args:
        await update.message.reply_text(
            "Usage: /broadcast <message>\n\n"
            "Example: /broadcast Hello everyone! New update coming soon."
        )
        return

    message_text = " ".join(context.args)
    broadcast_message = f"📢 <b>Announcement</b>\n\n{message_text}\n\n⚡ Zenkai Signal Hub"

    await update.message.reply_text("📤 Broadcasting...")

    success, failed = await broadcast_to_subscribers(
        context.application,
        broadcast_message
    )

    await update.message.reply_text(
        f"✅ Broadcast complete!\n\n"
        f"Sent: {success}\n"
        f"Failed: {failed}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Signal Creation (Conversation Handler)
# ─────────────────────────────────────────────────────────────────────────────

# Conversation states
(
    PAIR,
    DIRECTION,
    ENTRY_LOW,
    ENTRY_HIGH,
    STOP_LOSS,
    TP1,
    TP2,
    TP3,
    TIMEFRAME,
    CONFIDENCE,
    NOTE,
    CONFIRM
) = range(12)


async def signal_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start signal creation."""
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("❌ Admin only command.")
        return ConversationHandler.END

    user_id = update.effective_user.id
    active_drafts[user_id] = SignalDraft()

    await update.message.reply_text(
        "🚀 <b>New Signal Creation</b>\n\n"
        "Let's create a new trading signal.\n"
        "Use /cancel at any time to abort.\n\n"
        "<b>Step 1/11:</b> Enter the trading pair\n"
        "Example: BTC/USDT, ETH/USDT, EUR/USD",
        parse_mode=ParseMode.HTML
    )

    return PAIR


async def signal_pair(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get trading pair."""
    user_id = update.effective_user.id
    pair = update.message.text.upper().strip()

    active_drafts[user_id].pair = pair

    # Direction buttons
    keyboard = [
        [
            InlineKeyboardButton("🟢 LONG", callback_data="dir_LONG"),
            InlineKeyboardButton("🔴 SHORT", callback_data="dir_SHORT")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        f"✅ Pair: <b>{pair}</b>\n\n"
        "<b>Step 2/11:</b> Select direction",
        parse_mode=ParseMode.HTML,
        reply_markup=reply_markup
    )

    return DIRECTION


async def signal_direction(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get direction from callback."""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    direction = query.data.replace("dir_", "")

    active_drafts[user_id].direction = direction

    emoji = "🟢" if direction == "LONG" else "🔴"

    await query.message.reply_text(
        f"✅ Direction: <b>{emoji} {direction}</b>\n\n"
        "<b>Step 3/11:</b> Enter entry zone LOW price\n"
        "Example: 94500",
        parse_mode=ParseMode.HTML
    )

    return ENTRY_LOW


async def signal_entry_low(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get entry low price."""
    user_id = update.effective_user.id

    try:
        price = float(update.message.text.replace(",", "").replace("$", ""))
        active_drafts[user_id].entry_low = price

        await update.message.reply_text(
            f"✅ Entry Low: <b>${price:,.2f}</b>\n\n"
            "<b>Step 4/11:</b> Enter entry zone HIGH price\n"
            "Example: 95200",
            parse_mode=ParseMode.HTML
        )
        return ENTRY_HIGH

    except ValueError:
        await update.message.reply_text("❌ Invalid price. Please enter a number.")
        return ENTRY_LOW


async def signal_entry_high(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get entry high price."""
    user_id = update.effective_user.id

    try:
        price = float(update.message.text.replace(",", "").replace("$", ""))
        active_drafts[user_id].entry_high = price

        await update.message.reply_text(
            f"✅ Entry High: <b>${price:,.2f}</b>\n\n"
            "<b>Step 5/11:</b> Enter stop loss price\n"
            "Example: 93100",
            parse_mode=ParseMode.HTML
        )
        return STOP_LOSS

    except ValueError:
        await update.message.reply_text("❌ Invalid price. Please enter a number.")
        return ENTRY_HIGH


async def signal_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get stop loss price."""
    user_id = update.effective_user.id

    try:
        price = float(update.message.text.replace(",", "").replace("$", ""))
        active_drafts[user_id].stop_loss = price

        await update.message.reply_text(
            f"✅ Stop Loss: <b>${price:,.2f}</b>\n\n"
            "<b>Step 6/11:</b> Enter Take Profit 1 price\n"
            "Example: 96800",
            parse_mode=ParseMode.HTML
        )
        return TP1

    except ValueError:
        await update.message.reply_text("❌ Invalid price. Please enter a number.")
        return STOP_LOSS


async def signal_tp1(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get TP1 price."""
    user_id = update.effective_user.id

    try:
        price = float(update.message.text.replace(",", "").replace("$", ""))
        active_drafts[user_id].tp1 = price

        keyboard = [[InlineKeyboardButton("⏭️ Skip TP2", callback_data="skip_tp2")]]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            f"✅ TP1: <b>${price:,.2f}</b>\n\n"
            "<b>Step 7/11:</b> Enter Take Profit 2 price (optional)\n"
            "Example: 98500\n\n"
            "Or skip to continue.",
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup
        )
        return TP2

    except ValueError:
        await update.message.reply_text("❌ Invalid price. Please enter a number.")
        return TP1


async def signal_tp2(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get TP2 price."""
    user_id = update.effective_user.id

    try:
        price = float(update.message.text.replace(",", "").replace("$", ""))
        active_drafts[user_id].tp2 = price

        keyboard = [[InlineKeyboardButton("⏭️ Skip TP3", callback_data="skip_tp3")]]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            f"✅ TP2: <b>${price:,.2f}</b>\n\n"
            "<b>Step 8/11:</b> Enter Take Profit 3 price (optional)\n"
            "Example: 101000\n\n"
            "Or skip to continue.",
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup
        )
        return TP3

    except ValueError:
        await update.message.reply_text("❌ Invalid price. Please enter a number.")
        return TP2


async def signal_skip_tp2(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Skip TP2."""
    query = update.callback_query
    await query.answer()

    return await show_timeframe_selection(query.message, query.from_user.id)


async def signal_tp3(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get TP3 price."""
    user_id = update.effective_user.id

    try:
        price = float(update.message.text.replace(",", "").replace("$", ""))
        active_drafts[user_id].tp3 = price

        return await show_timeframe_selection(update.message, user_id)

    except ValueError:
        await update.message.reply_text("❌ Invalid price. Please enter a number.")
        return TP3


async def signal_skip_tp3(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Skip TP3."""
    query = update.callback_query
    await query.answer()

    return await show_timeframe_selection(query.message, query.from_user.id)


async def show_timeframe_selection(message, user_id):
    """Show timeframe selection."""
    keyboard = []
    row = []
    for tf in TIMEFRAMES:
        row.append(InlineKeyboardButton(tf, callback_data=f"tf_{tf}"))
        if len(row) == 4:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)

    reply_markup = InlineKeyboardMarkup(keyboard)

    await message.reply_text(
        "<b>Step 9/11:</b> Select timeframe",
        parse_mode=ParseMode.HTML,
        reply_markup=reply_markup
    )

    return TIMEFRAME


async def signal_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get timeframe from callback."""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    timeframe = query.data.replace("tf_", "")

    active_drafts[user_id].timeframe = timeframe

    keyboard = [
        [
            InlineKeyboardButton("⚪ LOW", callback_data="conf_LOW"),
            InlineKeyboardButton("🟡 MEDIUM", callback_data="conf_MEDIUM"),
            InlineKeyboardButton("🟢 HIGH", callback_data="conf_HIGH")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.message.reply_text(
        f"✅ Timeframe: <b>{timeframe}</b>\n\n"
        "<b>Step 10/11:</b> Select confidence level",
        parse_mode=ParseMode.HTML,
        reply_markup=reply_markup
    )

    return CONFIDENCE


async def signal_confidence(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get confidence from callback."""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    confidence = query.data.replace("conf_", "")

    active_drafts[user_id].confidence = confidence

    keyboard = [[InlineKeyboardButton("⏭️ Skip Note", callback_data="skip_note")]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.message.reply_text(
        f"✅ Confidence: <b>{confidence}</b>\n\n"
        "<b>Step 11/11:</b> Enter analysis note (optional)\n"
        "Example: CHoCH on 4H with unfilled bullish FVG at entry zone.",
        parse_mode=ParseMode.HTML,
        reply_markup=reply_markup
    )

    return NOTE


async def signal_note(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get note."""
    user_id = update.effective_user.id
    active_drafts[user_id].note = update.message.text

    return await show_confirmation(update.message, user_id)


async def signal_skip_note(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Skip note."""
    query = update.callback_query
    await query.answer()

    return await show_confirmation(query.message, query.from_user.id)


async def show_confirmation(message, user_id):
    """Show signal confirmation."""
    draft = active_drafts[user_id]

    # Build preview
    preview = f"""
📋 <b>SIGNAL PREVIEW</b>

━━━━━━━━━━━━━━━━━━━━━━

<b>Pair:</b> {draft.pair}
<b>Direction:</b> {"🟢 LONG" if draft.direction == "LONG" else "🔴 SHORT"}
<b>Timeframe:</b> {draft.timeframe}
<b>Confidence:</b> {draft.confidence}

<b>Entry:</b> ${draft.entry_low:,.2f} — ${draft.entry_high:,.2f}
<b>Stop Loss:</b> ${draft.stop_loss:,.2f}
<b>TP1:</b> ${draft.tp1:,.2f}"""

    if draft.tp2:
        preview += f"\n<b>TP2:</b> ${draft.tp2:,.2f}"
    if draft.tp3:
        preview += f"\n<b>TP3:</b> ${draft.tp3:,.2f}"

    if draft.note:
        preview += f"\n\n<b>Note:</b> {draft.note}"

    preview += "\n\n━━━━━━━━━━━━━━━━━━━━━━"

    keyboard = [
        [
            InlineKeyboardButton("✅ Confirm & Send", callback_data="confirm_signal"),
            InlineKeyboardButton("❌ Cancel", callback_data="cancel_signal")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await message.reply_text(
        preview,
        parse_mode=ParseMode.HTML,
        reply_markup=reply_markup
    )

    return CONFIRM


async def signal_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Confirm and send signal."""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    draft = active_drafts.get(user_id)

    if not draft or not draft.is_complete():
        await query.message.reply_text("❌ Signal data incomplete. Please start over with /signal")
        return ConversationHandler.END

    # Create signal
    signal = create_signal_from_draft(draft)

    # Format message
    signal_message = format_signal_message(signal)

    # Broadcast
    await query.message.reply_text("📤 Broadcasting signal...")

    success, failed = await broadcast_to_subscribers(
        context.application,
        signal_message
    )

    # Clean up draft
    del active_drafts[user_id]

    await query.message.reply_text(
        f"✅ <b>Signal #{signal.id} created and broadcast!</b>\n\n"
        f"Sent to: {success} subscribers\n"
        f"Failed: {failed}",
        parse_mode=ParseMode.HTML
    )

    logger.info(f"Signal #{signal.id} created by {user_id}: {signal.pair} {signal.direction}")

    return ConversationHandler.END


async def signal_cancel_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cancel signal creation."""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    if user_id in active_drafts:
        del active_drafts[user_id]

    await query.message.reply_text("❌ Signal creation cancelled.")

    return ConversationHandler.END


async def signal_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cancel signal creation via /cancel."""
    user_id = update.effective_user.id
    if user_id in active_drafts:
        del active_drafts[user_id]

    await update.message.reply_text("❌ Signal creation cancelled.")

    return ConversationHandler.END


# ─────────────────────────────────────────────────────────────────────────────
# Signal Update Command
# ─────────────────────────────────────────────────────────────────────────────

async def cmd_update(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /update command (admin only)."""
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("❌ Admin only command.")
        return

    if not context.args:
        # Show active signals
        signals = get_active_signals()
        if not signals:
            await update.message.reply_text("No active signals to update.")
            return

        message = "📊 <b>Select signal to update:</b>\n\n"
        keyboard = []

        for sig in signals[:10]:  # Limit to 10
            message += f"<b>#{sig.id}</b> — {sig.pair} {sig.direction}\n"
            keyboard.append([
                InlineKeyboardButton(f"#{sig.id} {sig.pair}", callback_data=f"upd_{sig.id}")
            ])

        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(message, parse_mode=ParseMode.HTML, reply_markup=reply_markup)
        return

    # Direct update: /update <id> <status>
    try:
        signal_id = int(context.args[0].replace("#", ""))
        update_type = context.args[1].upper() if len(context.args) > 1 else None

        if not update_type:
            await show_update_options(update.message, signal_id)
            return

        signal = update_signal(signal_id, update_type)
        if not signal:
            await update.message.reply_text(f"❌ Signal #{signal_id} not found.")
            return

        # Broadcast update
        update_message = format_signal_update(signal, update_type)
        await update.message.reply_text("📤 Broadcasting update...")

        success, failed = await broadcast_to_subscribers(context.application, update_message)

        await update.message.reply_text(
            f"✅ Signal #{signal_id} updated to {update_type}\n"
            f"Broadcast: {success} sent, {failed} failed"
        )

    except (ValueError, IndexError):
        await update.message.reply_text(
            "Usage: /update <signal_id> [status]\n\n"
            "Statuses: TP1_HIT, TP2_HIT, TP3_HIT, STOPPED, CANCELLED, BREAKEVEN"
        )


async def callback_update_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle signal update callback."""
    query = update.callback_query
    await query.answer()

    if not is_admin(query.from_user.id):
        return

    data = query.data

    if data.startswith("upd_"):
        signal_id = int(data.replace("upd_", ""))
        await show_update_options(query.message, signal_id)

    elif data.startswith("status_"):
        parts = data.split("_")
        signal_id = int(parts[1])
        update_type = parts[2]

        signal = update_signal(signal_id, update_type)
        if not signal:
            await query.message.reply_text(f"❌ Signal #{signal_id} not found.")
            return

        # Broadcast update
        update_message = format_signal_update(signal, update_type)
        await query.message.reply_text("📤 Broadcasting update...")

        success, failed = await broadcast_to_subscribers(context.application, update_message)

        await query.message.reply_text(
            f"✅ Signal #{signal_id} updated to {update_type}\n"
            f"Broadcast: {success} sent, {failed} failed"
        )


async def show_update_options(message, signal_id: int):
    """Show update options for a signal."""
    signal = get_signal(signal_id)
    if not signal:
        await message.reply_text(f"❌ Signal #{signal_id} not found.")
        return

    keyboard = [
        [InlineKeyboardButton("✅ TP1 Hit", callback_data=f"status_{signal_id}_TP1_HIT")],
    ]

    if signal.tp2:
        keyboard.append([InlineKeyboardButton("✅ TP2 Hit", callback_data=f"status_{signal_id}_TP2_HIT")])
    if signal.tp3:
        keyboard.append([InlineKeyboardButton("✅ TP3 Hit", callback_data=f"status_{signal_id}_TP3_HIT")])

    keyboard.extend([
        [InlineKeyboardButton("❌ Stopped Out", callback_data=f"status_{signal_id}_STOPPED")],
        [InlineKeyboardButton("⚖️ Breakeven", callback_data=f"status_{signal_id}_BREAKEVEN")],
        [InlineKeyboardButton("🚫 Cancel Signal", callback_data=f"status_{signal_id}_CANCELLED")]
    ])

    reply_markup = InlineKeyboardMarkup(keyboard)

    await message.reply_text(
        f"📊 <b>Update Signal #{signal_id}</b>\n\n"
        f"Pair: {signal.pair} {signal.direction}\n"
        f"Current Status: {signal.status}\n\n"
        "Select new status:",
        parse_mode=ParseMode.HTML,
        reply_markup=reply_markup
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Start the bot."""
    logger.info("=" * 50)
    logger.info("⚡ ZENKAI SIGNAL HUB STARTING")
    logger.info("=" * 50)

    # Create application
    app = Application.builder().token(BOT_TOKEN).build()

    # Signal creation conversation handler
    signal_conv = ConversationHandler(
        entry_points=[CommandHandler("signal", signal_start)],
        states={
            PAIR: [MessageHandler(filters.TEXT & ~filters.COMMAND, signal_pair)],
            DIRECTION: [CallbackQueryHandler(signal_direction, pattern="^dir_")],
            ENTRY_LOW: [MessageHandler(filters.TEXT & ~filters.COMMAND, signal_entry_low)],
            ENTRY_HIGH: [MessageHandler(filters.TEXT & ~filters.COMMAND, signal_entry_high)],
            STOP_LOSS: [MessageHandler(filters.TEXT & ~filters.COMMAND, signal_stop_loss)],
            TP1: [MessageHandler(filters.TEXT & ~filters.COMMAND, signal_tp1)],
            TP2: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, signal_tp2),
                CallbackQueryHandler(signal_skip_tp2, pattern="^skip_tp2$")
            ],
            TP3: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, signal_tp3),
                CallbackQueryHandler(signal_skip_tp3, pattern="^skip_tp3$")
            ],
            TIMEFRAME: [CallbackQueryHandler(signal_timeframe, pattern="^tf_")],
            CONFIDENCE: [CallbackQueryHandler(signal_confidence, pattern="^conf_")],
            NOTE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, signal_note),
                CallbackQueryHandler(signal_skip_note, pattern="^skip_note$")
            ],
            CONFIRM: [
                CallbackQueryHandler(signal_confirm, pattern="^confirm_signal$"),
                CallbackQueryHandler(signal_cancel_confirm, pattern="^cancel_signal$")
            ]
        },
        fallbacks=[CommandHandler("cancel", signal_cancel)]
    )

    # Add handlers
    app.add_handler(signal_conv)

    # Public commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("about", cmd_about))
    app.add_handler(CommandHandler("disclaimer", cmd_disclaimer))
    app.add_handler(CommandHandler("subscribe", cmd_subscribe))
    app.add_handler(CommandHandler("unsubscribe", cmd_unsubscribe))
    app.add_handler(CommandHandler("website", cmd_website))
    app.add_handler(CommandHandler("discord", cmd_discord))

    # Admin commands
    app.add_handler(CommandHandler("subscribers", cmd_subscribers))
    app.add_handler(CommandHandler("active", cmd_active))
    app.add_handler(CommandHandler("broadcast", cmd_broadcast))
    app.add_handler(CommandHandler("update", cmd_update))

    # Callbacks
    app.add_handler(CallbackQueryHandler(callback_update_signal, pattern="^(upd_|status_)"))
    app.add_handler(CallbackQueryHandler(callback_handler))

    # Register webhook callback
    async def webhook_broadcast(signal: Signal):
        message = format_signal_message(signal)
        await broadcast_to_subscribers(app, message)

    webhook.register_broadcast_callback(webhook_broadcast)

    # ── Outbox Poller (automated pipeline) ──────────────────────────────
    # Watches AlphaTrader signal outbox and auto-broadcasts to subscribers.
    # Same pipeline as the Discord bot — both read from the same outbox.

    async def outbox_broadcast(message: str, parse_mode: str = ParseMode.HTML, reply_markup=None):
        """Bridge function: OutboxPoller -> broadcast_to_subscribers."""
        success, failed = await broadcast_to_subscribers(app, message, parse_mode, reply_markup)
        logger.info(f"Outbox broadcast: {success} sent, {failed} failed")

    poller = OutboxPoller(broadcast_callback=outbox_broadcast, app=app)

    async def post_init(application):
        """Start outbox poller after bot is initialized."""
        await poller.start()
        logger.info("📡 Outbox poller started — automated signals active")

    async def post_shutdown(application):
        """Stop outbox poller on shutdown."""
        await poller.stop()
        logger.info("📡 Outbox poller stopped")

    app.post_init = post_init
    app.post_shutdown = post_shutdown

    logger.info(f"Admin IDs: {ADMIN_IDS}")
    logger.info(f"Subscribers: {db.get_subscriber_count()}")
    logger.info("Bot is running...")
    logger.info("📡 Outbox poller will start with bot")

    # Run
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

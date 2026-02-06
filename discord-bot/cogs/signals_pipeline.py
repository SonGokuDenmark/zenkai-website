"""
═══════════════════════════════════════════════════════════════════════════════
⚡ Bulma — AlphaTrader Signals Pipeline
═══════════════════════════════════════════════════════════════════════════════

Automatically posts trading signals from AlphaTrader to Discord.
No manual !signal command needed — it watches for new signals.

How it works:
1. AlphaTrader writes signals to: C:/Zenkai/alphatrader/signals/pending/
2. This cog watches that folder every 30 seconds
3. New signal JSON files get posted to #active-signals
4. Posted signals move to: C:/Zenkai/alphatrader/signals/posted/
5. Closed signals auto-archive to #signal-history with P&L

Signal JSON format:
{
    "id": "SIG-20260206-001",
    "pair": "BTCUSD",
    "direction": "LONG",
    "entry": 95000,
    "stop_loss": 94000,
    "take_profit": [97000, 98000, 99000],
    "timeframe": "4H",
    "confidence": "HIGH",
    "strategy": "ICT_FVG",
    "notes": "FVG filled at premium zone, bullish BOS confirmed",
    "timestamp": "2026-02-06T18:30:00Z"
}

Also supports an HTTP webhook endpoint for direct AlphaTrader integration.
"""

import logging
import json
import os
from datetime import datetime
from pathlib import Path

import discord
from discord.ext import commands, tasks

logger = logging.getLogger("bulma.signals")

# Signal directories
SIGNALS_DIR = Path("C:/Zenkai/alphatrader/signals")
PENDING_DIR = SIGNALS_DIR / "pending"
POSTED_DIR = SIGNALS_DIR / "posted"
CLOSED_DIR = SIGNALS_DIR / "closed"

# Signal tracking
TRACKER_FILE = Path(__file__).parent.parent / "data" / "signal_tracker.json"


class SignalsPipelineCog(commands.Cog, name="Signals Pipeline"):
    """Auto-posts AlphaTrader signals to Discord channels."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.data_dir = Path(__file__).parent.parent / "data"
        self.data_dir.mkdir(exist_ok=True)

        # Ensure signal directories exist
        for d in [PENDING_DIR, POSTED_DIR, CLOSED_DIR]:
            d.mkdir(parents=True, exist_ok=True)

        # Load signal tracker (posted signals with message IDs for editing)
        self.tracker = self._load_tracker()

        # Start watching
        self.watch_signals.start()
        self.check_closed_signals.start()

    def cog_unload(self):
        self.watch_signals.cancel()
        self.check_closed_signals.cancel()

    def _load_tracker(self) -> dict:
        if TRACKER_FILE.exists():
            try:
                with open(TRACKER_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"active": {}, "stats": {"total": 0, "wins": 0, "losses": 0}}

    def _save_tracker(self):
        try:
            with open(TRACKER_FILE, "w") as f:
                json.dump(self.tracker, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save signal tracker: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Signal Watcher — checks pending/ every 30 seconds
    # ─────────────────────────────────────────────────────────────────────────

    @tasks.loop(seconds=30)
    async def watch_signals(self):
        """Watch the pending signals directory for new signals."""
        if not PENDING_DIR.exists():
            return

        for signal_file in sorted(PENDING_DIR.glob("*.json")):
            try:
                with open(signal_file, "r") as f:
                    signal = json.load(f)

                # Post to Discord
                posted = await self._post_signal(signal)

                if posted:
                    # Move to posted/
                    dest = POSTED_DIR / signal_file.name
                    signal_file.rename(dest)
                    logger.info(f"Posted signal: {signal.get('id', signal_file.stem)}")
                else:
                    logger.warning(f"Failed to post signal: {signal_file.name}")

            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in signal file: {signal_file}")
            except Exception as e:
                logger.error(f"Error processing signal {signal_file}: {e}")

    @watch_signals.before_loop
    async def before_watch(self):
        await self.bot.wait_until_ready()

    # ─────────────────────────────────────────────────────────────────────────
    # Post Signal to Discord
    # ─────────────────────────────────────────────────────────────────────────

    async def _post_signal(self, signal: dict) -> bool:
        """Post a signal embed to #active-signals. Returns True if successful."""
        # Find the guild
        guild = None
        guild_id = os.getenv("GUILD_ID")
        if guild_id:
            guild = self.bot.get_guild(int(guild_id))
        if not guild and self.bot.guilds:
            guild = self.bot.guilds[0]
        if not guild:
            return False

        # Find the signals channel
        channel_name = self.bot.config["channels"].get("active_signals", "active-signals")
        channel = discord.utils.get(guild.text_channels, name=channel_name)
        if not channel:
            logger.error(f"Channel #{channel_name} not found")
            return False

        # Parse signal data
        pair = signal.get("pair", "UNKNOWN").upper()
        direction = signal.get("direction", "UNKNOWN").upper()
        entry = signal.get("entry", "N/A")
        sl = signal.get("stop_loss", "N/A")
        tp_list = signal.get("take_profit", [])
        timeframe = signal.get("timeframe", "N/A")
        confidence = signal.get("confidence", "MEDIUM").upper()
        strategy = signal.get("strategy", "AlphaTrader").replace("_", " ")
        notes = signal.get("notes", "")
        signal_id = signal.get("id", f"SIG-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}")

        # Direction styling
        if direction in ["LONG", "BUY"]:
            color = self.bot.colors["success"]
            direction_emoji = "🟢"
        elif direction in ["SHORT", "SELL"]:
            color = self.bot.colors["error"]
            direction_emoji = "🔴"
        else:
            color = self.bot.colors["secondary"]
            direction_emoji = "⚪"

        # Confidence badge
        confidence_badges = {
            "HIGH": "🔥 HIGH",
            "MEDIUM": "⚡ MEDIUM",
            "LOW": "💤 LOW",
        }
        conf_badge = confidence_badges.get(confidence, confidence)

        # Build embed
        embed = discord.Embed(
            title=f"{direction_emoji} {pair} — {direction}",
            color=color,
            timestamp=datetime.utcnow()
        )

        embed.add_field(name="📍 Entry", value=f"`{entry}`", inline=True)
        embed.add_field(name="🛑 Stop Loss", value=f"`{sl}`", inline=True)

        # Multiple take profits
        if isinstance(tp_list, list) and tp_list:
            tp_text = "\n".join([f"TP{i+1}: `{tp}`" for i, tp in enumerate(tp_list)])
        else:
            tp_text = f"`{tp_list}`"
        embed.add_field(name="🎯 Take Profit", value=tp_text, inline=True)

        embed.add_field(name="⏱ Timeframe", value=f"`{timeframe}`", inline=True)
        embed.add_field(name="🔥 Confidence", value=conf_badge, inline=True)
        embed.add_field(name="📐 Strategy", value=strategy, inline=True)

        if notes:
            embed.add_field(name="📝 Analysis", value=notes, inline=False)

        embed.set_author(
            name="AlphaTrader Signal",
            icon_url=guild.icon.url if guild.icon else None
        )

        embed.set_footer(
            text=f"Signal ID: {signal_id} • ⚠️ NOT financial advice — trade at your own risk"
        )

        # Determine which channel based on confidence
        # HIGH confidence → also post to selective-signals
        selective_channel = None
        if confidence == "HIGH":
            sel_name = self.bot.config["channels"].get("selective_signals", "selective-signals")
            selective_channel = discord.utils.get(guild.text_channels, name=sel_name)

        # Get Trader role for ping
        trader_role = discord.utils.get(guild.roles, name="Trader")
        ping_text = trader_role.mention if trader_role else ""

        try:
            # Post to active-signals
            if ping_text:
                msg = await channel.send(ping_text, embed=embed)
            else:
                msg = await channel.send(embed=embed)

            # Track the signal
            self.tracker["active"][signal_id] = {
                "message_id": msg.id,
                "channel_id": channel.id,
                "pair": pair,
                "direction": direction,
                "entry": entry,
                "stop_loss": sl,
                "take_profit": tp_list,
                "posted_at": datetime.utcnow().isoformat(),
            }
            self.tracker["stats"]["total"] += 1
            self._save_tracker()

            # Also post to selective-signals if HIGH confidence
            if selective_channel:
                selective_embed = embed.copy()
                selective_embed.title = f"⭐ {direction_emoji} {pair} — {direction} (HIGH CONFIDENCE)"
                try:
                    if ping_text:
                        await selective_channel.send(ping_text, embed=selective_embed)
                    else:
                        await selective_channel.send(embed=selective_embed)
                except discord.Forbidden:
                    pass

            return True

        except discord.Forbidden:
            logger.error(f"Cannot post to #{channel_name}")
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Closed Signal Watcher — archives with P&L
    # ─────────────────────────────────────────────────────────────────────────

    @tasks.loop(minutes=2)
    async def check_closed_signals(self):
        """Watch for closed signals and archive them with results."""
        if not CLOSED_DIR.exists():
            return

        for signal_file in sorted(CLOSED_DIR.glob("*.json")):
            try:
                with open(signal_file, "r") as f:
                    signal = json.load(f)

                await self._archive_signal(signal)

                # Remove the file after processing
                signal_file.unlink()
                logger.info(f"Archived closed signal: {signal.get('id', signal_file.stem)}")

            except Exception as e:
                logger.error(f"Error processing closed signal {signal_file}: {e}")

    @check_closed_signals.before_loop
    async def before_check_closed(self):
        await self.bot.wait_until_ready()

    async def _archive_signal(self, signal: dict):
        """Post closed signal results to #signal-history."""
        guild = None
        guild_id = os.getenv("GUILD_ID")
        if guild_id:
            guild = self.bot.get_guild(int(guild_id))
        if not guild and self.bot.guilds:
            guild = self.bot.guilds[0]
        if not guild:
            return

        history_name = self.bot.config["channels"].get("signal_history", "signal-history")
        history_channel = discord.utils.get(guild.text_channels, name=history_name)
        if not history_channel:
            return

        signal_id = signal.get("id", "Unknown")
        result = signal.get("result", "unknown").upper()  # WIN, LOSS, BREAKEVEN
        pnl = signal.get("pnl", 0)
        pnl_pct = signal.get("pnl_pct", 0)
        close_price = signal.get("close_price", "N/A")
        close_reason = signal.get("close_reason", "Manual close")
        pair = signal.get("pair", "UNKNOWN")
        direction = signal.get("direction", "UNKNOWN")

        # Result styling
        if result == "WIN":
            color = self.bot.colors["success"]
            result_emoji = "✅"
            self.tracker["stats"]["wins"] += 1
        elif result == "LOSS":
            color = self.bot.colors["error"]
            result_emoji = "❌"
            self.tracker["stats"]["losses"] += 1
        else:
            color = self.bot.colors["secondary"]
            result_emoji = "➖"

        # Remove from active
        if signal_id in self.tracker["active"]:
            del self.tracker["active"][signal_id]
        self._save_tracker()

        # Stats
        total = self.tracker["stats"]["total"]
        wins = self.tracker["stats"]["wins"]
        losses = self.tracker["stats"]["losses"]
        win_rate = (wins / max(total, 1)) * 100

        embed = discord.Embed(
            title=f"{result_emoji} {pair} — {direction} — {result}",
            color=color,
            timestamp=datetime.utcnow()
        )

        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        pnl_pct_str = f"+{pnl_pct:.1f}%" if pnl_pct >= 0 else f"{pnl_pct:.1f}%"

        embed.add_field(name="💰 P&L", value=f"`{pnl_str}` ({pnl_pct_str})", inline=True)
        embed.add_field(name="📍 Close Price", value=f"`{close_price}`", inline=True)
        embed.add_field(name="📋 Reason", value=close_reason, inline=True)

        embed.add_field(
            name="📊 Running Stats",
            value=f"Total: `{total}` • Wins: `{wins}` • Losses: `{losses}` • Win Rate: `{win_rate:.0f}%`",
            inline=False
        )

        embed.set_footer(text=f"Signal ID: {signal_id} • ⚡ Zenkai Corporation")

        try:
            await history_channel.send(embed=embed)
        except discord.Forbidden:
            logger.error(f"Cannot post to #{history_name}")

    # ─────────────────────────────────────────────────────────────────────────
    # Manual Commands (for testing and admin override)
    # ─────────────────────────────────────────────────────────────────────────

    @commands.command(name="signal")
    @commands.has_permissions(administrator=True)
    async def manual_signal(
        self,
        ctx: commands.Context,
        pair: str,
        direction: str,
        entry: str,
        sl: str,
        tp: str,
        *,
        notes: str = ""
    ):
        """
        Manually post a signal (admin override).
        Usage: !signal BTCUSD LONG 95000 94000 98000 FVG at premium zone
        """
        try:
            await ctx.message.delete()
        except discord.Forbidden:
            pass

        signal = {
            "id": f"SIG-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            "pair": pair.upper(),
            "direction": direction.upper(),
            "entry": entry,
            "stop_loss": sl,
            "take_profit": [tp],
            "timeframe": "Manual",
            "confidence": "MEDIUM",
            "strategy": "Manual Entry",
            "notes": notes,
        }

        success = await self._post_signal(signal)
        if success:
            await ctx.send("✅ Signal posted!", delete_after=5)
        else:
            await ctx.send("❌ Failed to post signal.", delete_after=5)

    @commands.hybrid_command(name="signalstats", description="View signal performance stats")
    async def signal_stats(self, ctx: commands.Context):
        """Show overall signal performance."""
        stats = self.tracker.get("stats", {})
        active = self.tracker.get("active", {})

        total = stats.get("total", 0)
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        pending = total - wins - losses
        win_rate = (wins / max(wins + losses, 1)) * 100

        embed = discord.Embed(
            title="📊 AlphaTrader Signal Stats",
            description="Performance overview of all posted signals.\n━━━━━━━━━━━━━━━━━━━━━━",
            color=self.bot.colors["primary"],
            timestamp=datetime.utcnow()
        )

        embed.add_field(name="📡 Total Signals", value=f"`{total}`", inline=True)
        embed.add_field(name="✅ Wins", value=f"`{wins}`", inline=True)
        embed.add_field(name="❌ Losses", value=f"`{losses}`", inline=True)
        embed.add_field(name="🔄 Active", value=f"`{len(active)}`", inline=True)
        embed.add_field(name="📈 Win Rate", value=f"`{win_rate:.0f}%`", inline=True)

        # List active signals
        if active:
            active_text = ""
            for sig_id, sig_data in list(active.items())[:5]:
                pair = sig_data.get("pair", "?")
                direction = sig_data.get("direction", "?")
                entry = sig_data.get("entry", "?")
                emoji = "🟢" if direction in ["LONG", "BUY"] else "🔴"
                active_text += f"{emoji} **{pair}** {direction} @ `{entry}`\n"

            embed.add_field(name="🔥 Active Signals", value=active_text, inline=False)

        embed.set_footer(text="⚡ Zenkai Corporation — Every Setback, Stronger")
        await ctx.send(embed=embed)

    @commands.command(name="testsignal")
    @commands.has_permissions(administrator=True)
    async def test_signal(self, ctx: commands.Context):
        """Post a test signal to verify the pipeline works."""
        test_signal = {
            "id": f"TEST-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            "pair": "BTCUSD",
            "direction": "LONG",
            "entry": 95000,
            "stop_loss": 94000,
            "take_profit": [96500, 97500, 99000],
            "timeframe": "4H",
            "confidence": "HIGH",
            "strategy": "ICT FVG + BOS",
            "notes": "⚠️ TEST SIGNAL — This is a pipeline test, not a real trade.\n\nFVG filled at premium zone with bullish BOS confirmed on 4H.",
        }

        success = await self._post_signal(test_signal)
        if success:
            await ctx.send("✅ Test signal posted! Check #active-signals and #selective-signals.", delete_after=10)
        else:
            await ctx.send("❌ Test signal failed. Check channel setup.", delete_after=10)


async def setup(bot: commands.Bot):
    await bot.add_cog(SignalsPipelineCog(bot))

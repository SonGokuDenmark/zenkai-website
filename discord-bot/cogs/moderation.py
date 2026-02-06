"""
═══════════════════════════════════════════════════════════════════════════════
⚡ Bulma — Moderation System
═══════════════════════════════════════════════════════════════════════════════

Auto-delete spam, warning system, temp bans.
- Anti-spam: rate limiting, duplicate detection, invite link filtering
- Warn system: !warn, !warnings, !clearwarnings
- Temp ban: !tempban with auto-unban
- Mod log: all actions logged to bot-status-admin-only
"""

import logging
import json
import re
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

import discord
from discord.ext import commands, tasks

logger = logging.getLogger("bulma.moderation")

# Persistent storage
DATA_DIR = Path(__file__).parent.parent / "data"
WARNINGS_FILE = DATA_DIR / "warnings.json"
TEMPBANS_FILE = DATA_DIR / "tempbans.json"


class ModerationCog(commands.Cog, name="Moderation"):
    """Auto-moderation and manual moderation tools."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

        # Anti-spam tracking: user_id -> list of message timestamps
        self.message_history = defaultdict(list)
        # Duplicate detection: user_id -> last message content
        self.last_messages = defaultdict(lambda: {"content": "", "count": 0})

        # Spam thresholds
        self.RATE_LIMIT = 5          # max messages per window
        self.RATE_WINDOW = 5         # seconds
        self.DUPLICATE_THRESHOLD = 3  # same message repeated X times = spam
        self.MAX_MENTIONS = 5        # max mentions per message
        self.MAX_LINES = 40          # max lines per message (wall of text)

        # Invite link pattern
        self.INVITE_PATTERN = re.compile(
            r"(discord\.gg|discordapp\.com/invite|discord\.com/invite)/[a-zA-Z0-9]+",
            re.IGNORECASE
        )

        # Load persistent data
        DATA_DIR.mkdir(exist_ok=True)
        self.warnings = self._load_json(WARNINGS_FILE, {})
        self.tempbans = self._load_json(TEMPBANS_FILE, [])

        # Start temp ban checker
        self.check_tempbans.start()

    def cog_unload(self):
        self.check_tempbans.cancel()

    # ─────────────────────────────────────────────────────────────────────────
    # Data Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _load_json(self, path: Path, default):
        if path.exists():
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
        return default

    def _save_json(self, path: Path, data):
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save {path}: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Mod Log
    # ─────────────────────────────────────────────────────────────────────────

    async def mod_log(self, guild: discord.Guild, embed: discord.Embed):
        """Send a mod action to the mod log channel."""
        channel_name = self.bot.config["channels"].get("mod_log", "bot-status-admin-only")
        channel = discord.utils.get(guild.text_channels, name=channel_name)
        if channel:
            try:
                await channel.send(embed=embed)
            except discord.Forbidden:
                logger.warning("Cannot post to mod log channel")

    # ─────────────────────────────────────────────────────────────────────────
    # Auto-Mod: Message Filter
    # ─────────────────────────────────────────────────────────────────────────

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Auto-moderation on every message."""
        # Ignore bots and DMs
        if message.author.bot or not message.guild:
            return

        # Ignore admins
        if message.author.guild_permissions.administrator:
            return

        # Run checks
        violation = await self._check_violations(message)
        if violation:
            await self._handle_violation(message, violation)

    async def _check_violations(self, message: discord.Message) -> str | None:
        """Check message against all auto-mod rules. Returns violation type or None."""
        user_id = message.author.id
        now = datetime.utcnow()

        # 1. Rate limiting (message flood)
        self.message_history[user_id].append(now)
        # Clean old entries
        self.message_history[user_id] = [
            t for t in self.message_history[user_id]
            if (now - t).total_seconds() < self.RATE_WINDOW
        ]
        if len(self.message_history[user_id]) > self.RATE_LIMIT:
            return "spam_flood"

        # 2. Duplicate message detection
        content = message.content.strip().lower()
        if content and len(content) > 3:
            last = self.last_messages[user_id]
            if content == last["content"]:
                last["count"] += 1
                if last["count"] >= self.DUPLICATE_THRESHOLD:
                    return "spam_duplicate"
            else:
                self.last_messages[user_id] = {"content": content, "count": 1}

        # 3. Mass mention spam
        total_mentions = len(message.mentions) + len(message.role_mentions)
        if total_mentions > self.MAX_MENTIONS:
            return "spam_mentions"

        # 4. Discord invite links (non-admin)
        if self.INVITE_PATTERN.search(message.content):
            return "invite_link"

        # 5. Wall of text
        if message.content.count("\n") > self.MAX_LINES:
            return "wall_of_text"

        return None

    async def _handle_violation(self, message: discord.Message, violation: str):
        """Handle a detected violation."""
        # Delete the message
        try:
            await message.delete()
        except discord.Forbidden:
            logger.warning(f"Cannot delete message from {message.author}")
            return

        # Violation descriptions
        descriptions = {
            "spam_flood": "🚫 Slow down! You're sending messages too fast.",
            "spam_duplicate": "🚫 Please don't repeat the same message.",
            "spam_mentions": "🚫 Too many mentions in one message.",
            "invite_link": "🚫 Discord invite links are not allowed.",
            "wall_of_text": "🚫 Message too long — please use a paste service.",
        }

        # Notify user
        try:
            warning_msg = await message.channel.send(
                f"{message.author.mention} {descriptions.get(violation, '🚫 Message removed.')}",
            )
            # Auto-delete the warning after 8 seconds
            await warning_msg.delete(delay=8)
        except discord.Forbidden:
            pass

        # Log to mod log
        embed = discord.Embed(
            title="🛡️ Auto-Mod Action",
            color=self.bot.colors["warning"],
            timestamp=datetime.utcnow()
        )
        embed.add_field(name="User", value=f"{message.author.mention} (`{message.author}`)", inline=True)
        embed.add_field(name="Violation", value=violation.replace("_", " ").title(), inline=True)
        embed.add_field(name="Channel", value=message.channel.mention, inline=True)

        # Truncate content for log
        content_preview = message.content[:200] + "..." if len(message.content) > 200 else message.content
        if content_preview:
            embed.add_field(name="Content", value=f"```{content_preview}```", inline=False)

        await self.mod_log(message.guild, embed)
        logger.info(f"Auto-mod: {violation} by {message.author} in #{message.channel}")

    # ─────────────────────────────────────────────────────────────────────────
    # Warning System
    # ─────────────────────────────────────────────────────────────────────────

    @commands.command(name="warn")
    @commands.has_permissions(administrator=True)
    async def warn(self, ctx: commands.Context, member: discord.Member, *, reason: str = "No reason provided"):
        """Warn a member. Auto-actions at thresholds."""
        try:
            await ctx.message.delete()
        except discord.Forbidden:
            pass

        user_id = str(member.id)
        guild_id = str(ctx.guild.id)

        # Initialize if needed
        if guild_id not in self.warnings:
            self.warnings[guild_id] = {}
        if user_id not in self.warnings[guild_id]:
            self.warnings[guild_id][user_id] = []

        # Add warning
        warning = {
            "reason": reason,
            "moderator": str(ctx.author),
            "moderator_id": ctx.author.id,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.warnings[guild_id][user_id].append(warning)
        self._save_json(WARNINGS_FILE, self.warnings)

        warn_count = len(self.warnings[guild_id][user_id])

        # DM the user
        try:
            dm_embed = discord.Embed(
                title="⚠️ Warning Received",
                description=f"You received a warning in **{ctx.guild.name}**.",
                color=self.bot.colors["warning"]
            )
            dm_embed.add_field(name="Reason", value=reason, inline=False)
            dm_embed.add_field(name="Total Warnings", value=f"`{warn_count}`", inline=True)
            dm_embed.set_footer(text="Please follow the server rules to avoid further action.")
            await member.send(embed=dm_embed)
        except discord.Forbidden:
            pass

        # Auto-actions based on warning count
        action_taken = ""
        if warn_count >= 5:
            # Ban at 5 warnings
            try:
                await member.ban(reason=f"Auto-ban: {warn_count} warnings")
                action_taken = "🔨 **BANNED** (5 warnings reached)"
            except discord.Forbidden:
                action_taken = "⚠️ Could not ban (missing permissions)"
        elif warn_count >= 3:
            # 1-hour timeout at 3 warnings
            try:
                until = discord.utils.utcnow() + timedelta(hours=1)
                await member.timeout(until, reason=f"Auto-timeout: {warn_count} warnings")
                action_taken = "🔇 **TIMED OUT 1 HOUR** (3 warnings reached)"
            except discord.Forbidden:
                action_taken = "⚠️ Could not timeout (missing permissions)"

        # Confirm embed
        embed = discord.Embed(
            title="⚠️ Member Warned",
            color=self.bot.colors["warning"],
            timestamp=datetime.utcnow()
        )
        embed.add_field(name="Member", value=f"{member.mention}", inline=True)
        embed.add_field(name="Warnings", value=f"`{warn_count}`", inline=True)
        embed.add_field(name="Reason", value=reason, inline=False)
        if action_taken:
            embed.add_field(name="Auto-Action", value=action_taken, inline=False)
        embed.set_footer(text=f"Warned by {ctx.author}")

        await ctx.send(embed=embed, delete_after=15)
        await self.mod_log(ctx.guild, embed)

    @commands.command(name="warnings")
    @commands.has_permissions(administrator=True)
    async def warnings(self, ctx: commands.Context, member: discord.Member):
        """View warnings for a member."""
        guild_id = str(ctx.guild.id)
        user_id = str(member.id)

        warns = self.warnings.get(guild_id, {}).get(user_id, [])

        if not warns:
            await ctx.send(f"✅ {member.mention} has no warnings.", delete_after=10)
            return

        embed = discord.Embed(
            title=f"⚠️ Warnings for {member}",
            description=f"Total: **{len(warns)}** warnings",
            color=self.bot.colors["warning"],
            timestamp=datetime.utcnow()
        )

        for i, w in enumerate(warns[-10:], 1):  # Show last 10
            ts = w.get("timestamp", "Unknown")
            embed.add_field(
                name=f"#{i} — {w['reason']}",
                value=f"By: {w['moderator']} • {ts[:10]}",
                inline=False
            )

        await ctx.send(embed=embed)

    @commands.command(name="clearwarnings")
    @commands.has_permissions(administrator=True)
    async def clearwarnings(self, ctx: commands.Context, member: discord.Member):
        """Clear all warnings for a member."""
        guild_id = str(ctx.guild.id)
        user_id = str(member.id)

        if guild_id in self.warnings and user_id in self.warnings[guild_id]:
            count = len(self.warnings[guild_id][user_id])
            del self.warnings[guild_id][user_id]
            self._save_json(WARNINGS_FILE, self.warnings)
            await ctx.send(f"✅ Cleared **{count}** warnings for {member.mention}.", delete_after=10)

            # Log
            embed = discord.Embed(
                title="🧹 Warnings Cleared",
                description=f"{member.mention}'s warnings cleared by {ctx.author.mention}",
                color=self.bot.colors["success"],
                timestamp=datetime.utcnow()
            )
            await self.mod_log(ctx.guild, embed)
        else:
            await ctx.send(f"{member.mention} has no warnings.", delete_after=10)

    # ─────────────────────────────────────────────────────────────────────────
    # Temp Ban
    # ─────────────────────────────────────────────────────────────────────────

    @commands.command(name="tempban")
    @commands.has_permissions(ban_members=True)
    async def tempban(self, ctx: commands.Context, member: discord.Member, duration: str, *, reason: str = "No reason"):
        """
        Temporarily ban a member.
        Usage: !tempban @user 1d Reason here
        Duration: 1h, 6h, 1d, 3d, 7d, 30d
        """
        try:
            await ctx.message.delete()
        except discord.Forbidden:
            pass

        # Parse duration
        duration_map = {
            "1h": timedelta(hours=1),
            "6h": timedelta(hours=6),
            "12h": timedelta(hours=12),
            "1d": timedelta(days=1),
            "3d": timedelta(days=3),
            "7d": timedelta(days=7),
            "14d": timedelta(days=14),
            "30d": timedelta(days=30),
        }

        delta = duration_map.get(duration.lower())
        if not delta:
            await ctx.send(
                f"❌ Invalid duration `{duration}`. Use: {', '.join(duration_map.keys())}",
                delete_after=10
            )
            return

        unban_at = datetime.utcnow() + delta

        # DM before ban
        try:
            dm_embed = discord.Embed(
                title="🔨 Temporarily Banned",
                description=f"You have been temporarily banned from **{ctx.guild.name}**.",
                color=self.bot.colors["error"]
            )
            dm_embed.add_field(name="Duration", value=duration, inline=True)
            dm_embed.add_field(name="Reason", value=reason, inline=True)
            dm_embed.add_field(name="Unban", value=f"<t:{int(unban_at.timestamp())}:F>", inline=False)
            await member.send(embed=dm_embed)
        except discord.Forbidden:
            pass

        # Ban
        try:
            await member.ban(reason=f"Temp ban ({duration}): {reason}")
        except discord.Forbidden:
            await ctx.send("❌ Cannot ban this member — missing permissions.", delete_after=10)
            return

        # Store for auto-unban
        self.tempbans.append({
            "guild_id": ctx.guild.id,
            "user_id": member.id,
            "user_name": str(member),
            "unban_at": unban_at.isoformat(),
            "reason": reason,
            "moderator": str(ctx.author),
        })
        self._save_json(TEMPBANS_FILE, self.tempbans)

        # Confirm
        embed = discord.Embed(
            title="🔨 Member Temp-Banned",
            color=self.bot.colors["error"],
            timestamp=datetime.utcnow()
        )
        embed.add_field(name="Member", value=f"{member} (`{member.id}`)", inline=True)
        embed.add_field(name="Duration", value=duration, inline=True)
        embed.add_field(name="Unban At", value=f"<t:{int(unban_at.timestamp())}:F>", inline=True)
        embed.add_field(name="Reason", value=reason, inline=False)
        embed.set_footer(text=f"Banned by {ctx.author}")

        await ctx.send(embed=embed, delete_after=15)
        await self.mod_log(ctx.guild, embed)
        logger.info(f"Temp-banned {member} for {duration} by {ctx.author}")

    @tasks.loop(minutes=5)
    async def check_tempbans(self):
        """Check for expired temp bans and unban users."""
        now = datetime.utcnow()
        remaining = []

        for ban in self.tempbans:
            unban_at = datetime.fromisoformat(ban["unban_at"])
            if now >= unban_at:
                # Time to unban
                guild = self.bot.get_guild(ban["guild_id"])
                if guild:
                    try:
                        user = await self.bot.fetch_user(ban["user_id"])
                        await guild.unban(user, reason="Temp ban expired")
                        logger.info(f"Auto-unbanned {ban['user_name']} in {guild}")

                        # Log it
                        embed = discord.Embed(
                            title="🔓 Auto-Unban",
                            description=f"**{ban['user_name']}** temp ban expired.",
                            color=self.bot.colors["success"],
                            timestamp=datetime.utcnow()
                        )
                        await self.mod_log(guild, embed)
                    except Exception as e:
                        logger.error(f"Failed to unban {ban['user_name']}: {e}")
            else:
                remaining.append(ban)

        if len(remaining) != len(self.tempbans):
            self.tempbans = remaining
            self._save_json(TEMPBANS_FILE, self.tempbans)

    @check_tempbans.before_loop
    async def before_check_tempbans(self):
        await self.bot.wait_until_ready()


async def setup(bot: commands.Bot):
    await bot.add_cog(ModerationCog(bot))

"""
═══════════════════════════════════════════════════════════════════════════════
⚡ Bulma — Leveling System
═══════════════════════════════════════════════════════════════════════════════

XP-based leveling that rewards active community members.
- Earn XP per message (with cooldown to prevent spam farming)
- Level-up announcements with Zenkai-themed embeds
- /rank — check your rank
- /leaderboard — top 10 members
- XP formula: level = floor(sqrt(xp / 100))
"""

import logging
import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

import discord
from discord.ext import commands

logger = logging.getLogger("bulma.leveling")

DATA_DIR = Path(__file__).parent.parent / "data"
LEVELS_FILE = DATA_DIR / "levels.json"

# XP settings
XP_PER_MESSAGE_MIN = 15
XP_PER_MESSAGE_MAX = 25
XP_COOLDOWN_SECONDS = 60  # 1 message per 60s earns XP (anti-spam)

# Level milestones with role rewards
LEVEL_ROLES = {
    5: "Rookie Trader",
    10: "Apprentice",
    20: "Veteran",
    30: "Elite",
    50: "Zenkai Legend",
}

# Zenkai-themed level-up messages
LEVELUP_MESSAGES = [
    "**{member}** just powered up to **Level {level}**! The Zenkai boost is real! 🔥",
    "**{member}** hit **Level {level}**! Every setback made them stronger! ⚡",
    "**{member}** ascended to **Level {level}**! Their power level is over... well, {xp}! 💪",
    "**{member}** broke through to **Level {level}**! The grind pays off! 🚀",
    "**{member}** leveled up to **Level {level}**! Zenkai spirit in action! 🔥",
]


class LevelingCog(commands.Cog, name="Leveling"):
    """XP and leveling system for community engagement."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        DATA_DIR.mkdir(exist_ok=True)
        self.data = self._load()
        # Cooldown tracking: user_id -> last XP timestamp
        self.cooldowns = {}

    def _load(self) -> dict:
        if LEVELS_FILE.exists():
            try:
                with open(LEVELS_FILE, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load levels: {e}")
        return {}

    def _save(self):
        try:
            with open(LEVELS_FILE, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save levels: {e}")

    @staticmethod
    def xp_for_level(level: int) -> int:
        """XP required to reach a given level."""
        return level * level * 100

    @staticmethod
    def level_from_xp(xp: int) -> int:
        """Calculate level from total XP."""
        return int(math.sqrt(xp / 100))

    def _get_user(self, guild_id: str, user_id: str) -> dict:
        """Get or create user data."""
        if guild_id not in self.data:
            self.data[guild_id] = {}
        if user_id not in self.data[guild_id]:
            self.data[guild_id][user_id] = {"xp": 0, "level": 0, "messages": 0}
        return self.data[guild_id][user_id]

    # ─────────────────────────────────────────────────────────────────────────
    # XP Gain on Message
    # ─────────────────────────────────────────────────────────────────────────

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Award XP for messages."""
        # Ignore bots, DMs, and very short messages
        if message.author.bot or not message.guild:
            return
        if len(message.content) < 5:
            return

        user_id = str(message.author.id)
        guild_id = str(message.guild.id)
        now = datetime.utcnow()

        # Cooldown check
        last_xp = self.cooldowns.get(user_id)
        if last_xp and (now - last_xp).total_seconds() < XP_COOLDOWN_SECONDS:
            return

        self.cooldowns[user_id] = now

        # Award random XP
        xp_gained = random.randint(XP_PER_MESSAGE_MIN, XP_PER_MESSAGE_MAX)
        user_data = self._get_user(guild_id, user_id)
        old_level = user_data["level"]

        user_data["xp"] += xp_gained
        user_data["messages"] += 1
        new_level = self.level_from_xp(user_data["xp"])
        user_data["level"] = new_level

        self._save()

        # Level up check
        if new_level > old_level:
            await self._handle_levelup(message, new_level, user_data["xp"])

    async def _handle_levelup(self, message: discord.Message, level: int, xp: int):
        """Handle level-up announcement and role rewards."""
        member = message.author

        # Pick a random level-up message
        msg = random.choice(LEVELUP_MESSAGES).format(
            member=member.display_name,
            level=level,
            xp=xp
        )

        # Build embed
        embed = discord.Embed(
            title="⬆️ LEVEL UP!",
            description=msg,
            color=self.bot.colors["primary"],
            timestamp=datetime.utcnow()
        )
        embed.set_thumbnail(url=member.display_avatar.url)

        # Check for milestone role
        if level in LEVEL_ROLES:
            role_name = LEVEL_ROLES[level]
            embed.add_field(
                name="🏆 New Role Unlocked!",
                value=f"**{role_name}**",
                inline=False
            )
            # Try to assign the role
            role = discord.utils.get(message.guild.roles, name=role_name)
            if role:
                try:
                    await member.add_roles(role, reason=f"Level {level} milestone")
                except discord.Forbidden:
                    logger.warning(f"Cannot assign role {role_name} to {member}")
            else:
                # Create the role if it doesn't exist
                try:
                    role = await message.guild.create_role(
                        name=role_name,
                        mentionable=False,
                        reason=f"Bulma leveling: Level {level} milestone role"
                    )
                    await member.add_roles(role, reason=f"Level {level} milestone")
                except discord.Forbidden:
                    logger.warning(f"Cannot create role {role_name}")

        # Next level info
        next_xp = self.xp_for_level(level + 1)
        embed.add_field(
            name="Next Level",
            value=f"`{xp:,}` / `{next_xp:,}` XP",
            inline=True
        )

        embed.set_footer(text="⚡ Zenkai Corporation — Every Setback, Stronger")

        # Post in the same channel
        try:
            await message.channel.send(embed=embed)
        except discord.Forbidden:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # Commands
    # ─────────────────────────────────────────────────────────────────────────

    @commands.hybrid_command(name="rank", description="Check your level and XP")
    async def rank(self, ctx: commands.Context, member: discord.Member = None):
        """Check your rank or another member's rank."""
        target = member or ctx.author
        guild_id = str(ctx.guild.id)
        user_id = str(target.id)

        user_data = self._get_user(guild_id, user_id)
        level = user_data["level"]
        xp = user_data["xp"]
        messages = user_data["messages"]
        next_xp = self.xp_for_level(level + 1)

        # Calculate rank position
        all_users = self.data.get(guild_id, {})
        sorted_users = sorted(all_users.items(), key=lambda x: x[1].get("xp", 0), reverse=True)
        rank_pos = next((i + 1 for i, (uid, _) in enumerate(sorted_users) if uid == user_id), len(sorted_users))

        # Progress bar
        if level == 0:
            progress_pct = xp / max(self.xp_for_level(1), 1)
        else:
            current_level_xp = self.xp_for_level(level)
            progress_pct = (xp - current_level_xp) / max(next_xp - current_level_xp, 1)

        bar_length = 16
        filled = int(progress_pct * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)

        embed = discord.Embed(
            title=f"📊 {target.display_name}'s Rank",
            color=self.bot.colors["primary"],
            timestamp=datetime.utcnow()
        )
        embed.set_thumbnail(url=target.display_avatar.url)

        embed.add_field(name="Rank", value=f"`#{rank_pos}`", inline=True)
        embed.add_field(name="Level", value=f"`{level}`", inline=True)
        embed.add_field(name="Messages", value=f"`{messages:,}`", inline=True)

        embed.add_field(
            name=f"XP Progress — {xp:,} / {next_xp:,}",
            value=f"`{bar}` {int(progress_pct * 100)}%",
            inline=False
        )

        # Show current milestone role
        current_role = None
        for lvl in sorted(LEVEL_ROLES.keys(), reverse=True):
            if level >= lvl:
                current_role = LEVEL_ROLES[lvl]
                break
        if current_role:
            embed.add_field(name="Title", value=f"🏆 {current_role}", inline=True)

        # Next milestone
        next_milestone = None
        for lvl in sorted(LEVEL_ROLES.keys()):
            if level < lvl:
                next_milestone = (lvl, LEVEL_ROLES[lvl])
                break
        if next_milestone:
            embed.add_field(
                name="Next Title",
                value=f"Level {next_milestone[0]}: {next_milestone[1]}",
                inline=True
            )

        embed.set_footer(text="⚡ Zenkai Corporation — Every Setback, Stronger")
        await ctx.send(embed=embed)

    @commands.hybrid_command(name="leaderboard", description="View the top members by XP")
    async def leaderboard(self, ctx: commands.Context):
        """Show the server XP leaderboard."""
        guild_id = str(ctx.guild.id)
        all_users = self.data.get(guild_id, {})

        if not all_users:
            await ctx.send("No leveling data yet! Start chatting to earn XP.", delete_after=10)
            return

        sorted_users = sorted(all_users.items(), key=lambda x: x[1].get("xp", 0), reverse=True)[:10]

        embed = discord.Embed(
            title="🏆 Zenkai Leaderboard",
            description="Top 10 most active members\n━━━━━━━━━━━━━━━━━━━━━━",
            color=self.bot.colors["primary"],
            timestamp=datetime.utcnow()
        )

        medals = ["🥇", "🥈", "🥉"]
        lines = []

        for i, (user_id, data) in enumerate(sorted_users):
            medal = medals[i] if i < 3 else f"`#{i+1}`"
            member = ctx.guild.get_member(int(user_id))
            name = member.display_name if member else f"User {user_id[:6]}..."
            level = data.get("level", 0)
            xp = data.get("xp", 0)
            lines.append(f"{medal} **{name}** — Level `{level}` • `{xp:,}` XP")

        embed.description += "\n\n" + "\n".join(lines)

        if ctx.guild.icon:
            embed.set_thumbnail(url=ctx.guild.icon.url)

        embed.set_footer(text="⚡ Zenkai Corporation — Every Setback, Stronger")
        await ctx.send(embed=embed)

    @commands.command(name="createlevels")
    @commands.has_permissions(administrator=True)
    async def create_level_roles(self, ctx: commands.Context):
        """Create all milestone level roles. Admin only."""
        created = []
        for level, role_name in LEVEL_ROLES.items():
            if not discord.utils.get(ctx.guild.roles, name=role_name):
                try:
                    await ctx.guild.create_role(
                        name=role_name,
                        mentionable=False,
                        reason=f"Bulma leveling: Level {level} milestone"
                    )
                    created.append(f"Level {level}: {role_name}")
                except discord.Forbidden:
                    pass

        if created:
            await ctx.send(f"✅ Created level roles:\n" + "\n".join(created))
        else:
            await ctx.send("✅ All level roles already exist!")


async def setup(bot: commands.Bot):
    await bot.add_cog(LevelingCog(bot))

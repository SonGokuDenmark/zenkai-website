"""
═══════════════════════════════════════════════════════════════════════════════
⚡ Zenkai Bot — Announcements System
═══════════════════════════════════════════════════════════════════════════════

Handles announcements:
- !announce command for admins
- Auto-ping relevant roles based on channel
- Styled Zenkai-branded embeds
"""

import logging
from datetime import datetime

import discord
from discord.ext import commands

logger = logging.getLogger("zenkai-bot.announcements")


class AnnouncementsCog(commands.Cog, name="Announcements"):
    """Announcement system for admins."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

        # Channel to role mapping for auto-pings
        self.channel_role_map = {
            "signals-preview": "Trader",
            "trading-discussion": "Trader",
            "ict-concepts": "Trader",
            "gaming-chat": "Gamer",
            "ogame": "Gamer",
            "minecraft-hytale": "Gamer",
            "smart-home": "Smart Home",
            "dev-log": "Developer",
            "code-help": "Developer",
            "youtube-uploads": "Content",
            "content-ideas": "Content",
        }

    def get_role_for_channel(self, channel: discord.TextChannel) -> discord.Role | None:
        """Get the role to ping based on channel name."""
        role_name = self.channel_role_map.get(channel.name)
        if role_name:
            return discord.utils.get(channel.guild.roles, name=role_name)
        return None

    @commands.command(name="announce")
    @commands.has_permissions(administrator=True)
    async def announce(
        self,
        ctx: commands.Context,
        channel: discord.TextChannel,
        *,
        message: str
    ):
        """
        Post a styled announcement to a channel.

        Usage: !announce #channel Your announcement message here

        The bot will automatically ping the relevant role based on the channel.
        """
        # Delete the command message
        try:
            await ctx.message.delete()
        except discord.Forbidden:
            pass

        # Build the announcement embed
        embed = discord.Embed(
            title="📢 Announcement",
            description=message,
            color=self.bot.colors["primary"],
            timestamp=datetime.utcnow()
        )

        embed.set_author(
            name="Zenkai Corporation",
            icon_url=ctx.guild.icon.url if ctx.guild.icon else None
        )

        embed.set_footer(
            text=f"Posted by {ctx.author.display_name}",
            icon_url=ctx.author.display_avatar.url
        )

        # Get role to ping
        role = self.get_role_for_channel(channel)
        ping_text = ""
        if role:
            ping_text = role.mention

        # Send the announcement
        try:
            if ping_text:
                await channel.send(ping_text, embed=embed)
            else:
                await channel.send(embed=embed)

            # Confirm to admin
            confirm = discord.Embed(
                title="✅ Announcement Posted",
                description=f"Your announcement was posted to {channel.mention}",
                color=self.bot.colors["success"]
            )
            if role:
                confirm.add_field(name="Pinged Role", value=role.mention, inline=True)

            await ctx.send(embed=confirm, delete_after=10)

            logger.info(f"Announcement posted to {channel} by {ctx.author}")

        except discord.Forbidden:
            await ctx.send(
                f"❌ Cannot post in {channel.mention} - missing permissions",
                delete_after=10
            )

    @commands.command(name="devlog")
    @commands.has_permissions(administrator=True)
    async def devlog(self, ctx: commands.Context, *, message: str):
        """
        Post a development update to #dev-log.

        Usage: !devlog Your dev update here
        """
        # Find dev-log channel
        channel_name = self.bot.config["channels"].get("dev_log", "dev-log")
        channel = discord.utils.get(ctx.guild.text_channels, name=channel_name)

        if not channel:
            await ctx.send(f"❌ Channel `#{channel_name}` not found!", delete_after=10)
            return

        # Delete command
        try:
            await ctx.message.delete()
        except discord.Forbidden:
            pass

        # Build dev log embed
        embed = discord.Embed(
            title="🛠️ Development Update",
            description=message,
            color=self.bot.colors["secondary"],
            timestamp=datetime.utcnow()
        )

        embed.set_author(
            name="Zenkai Dev Log",
            icon_url=ctx.guild.icon.url if ctx.guild.icon else None
        )

        embed.set_footer(
            text=f"Update by {ctx.author.display_name}",
            icon_url=ctx.author.display_avatar.url
        )

        # Get Developer role
        dev_role = discord.utils.get(ctx.guild.roles, name="Developer")

        try:
            if dev_role:
                await channel.send(dev_role.mention, embed=embed)
            else:
                await channel.send(embed=embed)

            await ctx.send(f"✅ Dev log posted to {channel.mention}", delete_after=10)
            logger.info(f"Dev log posted by {ctx.author}")

        except discord.Forbidden:
            await ctx.send(f"❌ Cannot post in {channel.mention}", delete_after=10)

    @commands.command(name="signal")
    @commands.has_permissions(administrator=True)
    async def signal(
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
        Post a trading signal to #signals-preview.

        Usage: !signal BTCUSD LONG 42000 41500 44000 Optional notes here
        """
        # Find signals channel
        channel_name = self.bot.config["channels"].get("signals_preview", "signals-preview")
        channel = discord.utils.get(ctx.guild.text_channels, name=channel_name)

        if not channel:
            await ctx.send(f"❌ Channel `#{channel_name}` not found!", delete_after=10)
            return

        # Delete command
        try:
            await ctx.message.delete()
        except discord.Forbidden:
            pass

        # Determine direction color and emoji
        direction_upper = direction.upper()
        if direction_upper in ["LONG", "BUY"]:
            color = self.bot.colors["success"]
            direction_emoji = "🟢"
        else:
            color = self.bot.colors["error"]
            direction_emoji = "🔴"

        # Build signal embed
        embed = discord.Embed(
            title=f"{direction_emoji} {pair.upper()} — {direction_upper}",
            color=color,
            timestamp=datetime.utcnow()
        )

        embed.add_field(name="📍 Entry", value=f"`{entry}`", inline=True)
        embed.add_field(name="🛑 Stop Loss", value=f"`{sl}`", inline=True)
        embed.add_field(name="🎯 Take Profit", value=f"`{tp}`", inline=True)

        if notes:
            embed.add_field(name="📝 Notes", value=notes, inline=False)

        embed.set_author(
            name="Zenkai Signal Hub",
            icon_url=ctx.guild.icon.url if ctx.guild.icon else None
        )

        embed.set_footer(
            text="⚡ This is NOT financial advice. Trade at your own risk.",
            icon_url=ctx.author.display_avatar.url
        )

        # Get Trader role
        trader_role = discord.utils.get(ctx.guild.roles, name="Trader")

        try:
            if trader_role:
                await channel.send(trader_role.mention, embed=embed)
            else:
                await channel.send(embed=embed)

            await ctx.send(f"✅ Signal posted to {channel.mention}", delete_after=10)
            logger.info(f"Signal posted: {pair} {direction} by {ctx.author}")

        except discord.Forbidden:
            await ctx.send(f"❌ Cannot post in {channel.mention}", delete_after=10)


async def setup(bot: commands.Bot):
    await bot.add_cog(AnnouncementsCog(bot))

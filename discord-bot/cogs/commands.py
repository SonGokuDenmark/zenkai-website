"""
═══════════════════════════════════════════════════════════════════════════════
⚡ Zenkai Bot — General Commands
═══════════════════════════════════════════════════════════════════════════════

General commands:
- /projects - Show all Zenkai projects
- /info - Server information
- /lastsignal - Trading signal placeholder
- /help - Bot help
"""

import logging
from datetime import datetime

import discord
from discord.ext import commands
from discord import app_commands

logger = logging.getLogger("zenkai-bot.commands")


class CommandsCog(commands.Cog, name="Commands"):
    """General bot commands."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    # ─────────────────────────────────────────────────────────────────────────
    # Projects Command
    # ─────────────────────────────────────────────────────────────────────────

    @commands.hybrid_command(name="projects", description="View all Zenkai projects and their status")
    async def projects(self, ctx: commands.Context):
        """Show all Zenkai projects with their current status."""
        projects = self.bot.config.get("projects", [])

        embed = discord.Embed(
            title="🚀 Zenkai Projects",
            description="Current status of all Zenkai Corporation projects.\n━━━━━━━━━━━━━━━━━━━━━━",
            color=self.bot.colors["primary"],
            timestamp=datetime.utcnow()
        )

        for project in projects:
            name = project.get("name", "Unknown")
            status = project.get("status", "Unknown")
            description = project.get("description", "")
            link = project.get("link", "")

            value = f"{description}"
            if link:
                value += f"\n[View Project]({link})"

            embed.add_field(
                name=f"{status} {name}",
                value=value or "No description",
                inline=False
            )

        embed.set_footer(
            text="⚡ Zenkai Corporation — Every Setback, Stronger",
            icon_url=ctx.guild.icon.url if ctx.guild and ctx.guild.icon else None
        )

        await ctx.send(embed=embed)

    # ─────────────────────────────────────────────────────────────────────────
    # Server Info Command
    # ─────────────────────────────────────────────────────────────────────────

    @commands.hybrid_command(name="info", description="View server information")
    async def info(self, ctx: commands.Context):
        """Show server information and Zenkai links."""
        guild = ctx.guild
        if not guild:
            await ctx.send("This command must be used in a server!")
            return

        # Count members
        total_members = guild.member_count
        online_members = sum(1 for m in guild.members if m.status != discord.Status.offline)
        bot_count = sum(1 for m in guild.members if m.bot)

        embed = discord.Embed(
            title=f"⚡ {guild.name}",
            description="Zenkai Corporation Community Server\n━━━━━━━━━━━━━━━━━━━━━━",
            color=self.bot.colors["primary"],
            timestamp=datetime.utcnow()
        )

        # Server stats
        embed.add_field(
            name="👥 Members",
            value=f"Total: `{total_members}`\nOnline: `{online_members}`\nBots: `{bot_count}`",
            inline=True
        )

        embed.add_field(
            name="📅 Created",
            value=f"<t:{int(guild.created_at.timestamp())}:D>\n<t:{int(guild.created_at.timestamp())}:R>",
            inline=True
        )

        embed.add_field(
            name="📊 Channels",
            value=f"Text: `{len(guild.text_channels)}`\nVoice: `{len(guild.voice_channels)}`",
            inline=True
        )

        # Links
        socials = self.bot.config.get("socials", {})
        links_text = ""
        if self.bot.config.get("website_url"):
            links_text += f"🌐 [Website]({self.bot.config['website_url']})\n"
        if socials.get("youtube"):
            links_text += f"📺 [YouTube]({socials['youtube']})\n"
        if socials.get("tradingview"):
            links_text += f"📊 [TradingView]({socials['tradingview']})\n"
        if socials.get("github"):
            links_text += f"💻 [GitHub]({socials['github']})\n"
        if socials.get("telegram"):
            links_text += f"📱 [Telegram]({socials['telegram']})\n"

        if links_text:
            embed.add_field(
                name="🔗 Links",
                value=links_text,
                inline=False
            )

        if guild.icon:
            embed.set_thumbnail(url=guild.icon.url)

        embed.set_footer(
            text="⚡ Zenkai Corporation — Every Setback, Stronger",
            icon_url=guild.icon.url if guild.icon else None
        )

        await ctx.send(embed=embed)

    # ─────────────────────────────────────────────────────────────────────────
    # Last Signal Command (Placeholder)
    # ─────────────────────────────────────────────────────────────────────────

    @commands.hybrid_command(name="lastsignal", description="View the latest trading signal")
    async def lastsignal(self, ctx: commands.Context):
        """Show the latest trading signal (placeholder for now)."""
        embed = discord.Embed(
            title="📡 Zenkai Signal Hub",
            description=(
                "**Signal Hub is coming soon!**\n\n"
                "We're building an AI-powered trading signal system using AlphaTrader.\n"
                "Get real-time alerts with entry, stop loss, and take profit levels.\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "🔔 **Join the waitlist:**\n"
                "Check #how-to-use for guides on reading our signals!\n\n"
                "💬 **Stay updated:**\n"
                "Follow #updates for the latest AlphaTrader progress."
            ),
            color=self.bot.colors["secondary"],
            timestamp=datetime.utcnow()
        )

        embed.add_field(
            name="📊 Current Status",
            value=(
                "```\n"
                "AlphaTrader:   🔄 Training\n"
                "Signal Hub:    🔜 Coming Soon\n"
                "Telegram Bot:  🔜 Coming Soon\n"
                "```"
            ),
            inline=False
        )

        embed.set_footer(
            text="⚡ Zenkai Corporation — Every Setback, Stronger",
            icon_url=ctx.guild.icon.url if ctx.guild and ctx.guild.icon else None
        )

        await ctx.send(embed=embed)

    # ─────────────────────────────────────────────────────────────────────────
    # Help Command
    # ─────────────────────────────────────────────────────────────────────────

    @commands.hybrid_command(name="help", description="View all available commands")
    async def help(self, ctx: commands.Context):
        """Show all available bot commands."""
        embed = discord.Embed(
            title="⚡ Zenkai Bot Commands",
            description="All available commands for the Zenkai Bot.\n━━━━━━━━━━━━━━━━━━━━━━",
            color=self.bot.colors["primary"],
            timestamp=datetime.utcnow()
        )

        # General Commands
        embed.add_field(
            name="📋 General",
            value=(
                "`/projects` — View all Zenkai projects\n"
                "`/info` — Server information & links\n"
                "`/lastsignal` — Latest trading signal\n"
                "`/help` — Show this help message"
            ),
            inline=False
        )

        # Admin Commands
        if ctx.author.guild_permissions.administrator:
            embed.add_field(
                name="🔧 Admin",
                value=(
                    "`!announce #channel message` — Post announcement\n"
                    "`!setuproles` — Setup reaction roles\n"
                    "`!createroles` — Create all roles\n"
                    "`!testwelcome [@user]` — Test welcome message"
                ),
                inline=False
            )

        embed.add_field(
            name="💡 Tips",
            value=(
                "• Commands work with both `!` prefix and `/` slash\n"
                "• Check #how-to-use for signal guides\n"
                "• Follow #updates for announcements"
            ),
            inline=False
        )

        embed.set_footer(
            text="⚡ Zenkai Corporation — Every Setback, Stronger",
            icon_url=ctx.guild.icon.url if ctx.guild and ctx.guild.icon else None
        )

        await ctx.send(embed=embed)

    # ─────────────────────────────────────────────────────────────────────────
    # Ping Command
    # ─────────────────────────────────────────────────────────────────────────

    @commands.hybrid_command(name="ping", description="Check bot latency")
    async def ping(self, ctx: commands.Context):
        """Check bot latency."""
        latency = round(self.bot.latency * 1000)

        embed = discord.Embed(
            title="🏓 Pong!",
            description=f"Bot latency: `{latency}ms`",
            color=self.bot.colors["success"] if latency < 200 else self.bot.colors["warning"],
            timestamp=datetime.utcnow()
        )

        await ctx.send(embed=embed)


async def setup(bot: commands.Bot):
    await bot.add_cog(CommandsCog(bot))

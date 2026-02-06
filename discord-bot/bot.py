"""
═══════════════════════════════════════════════════════════════════════════════
⚡ BULMA — Zenkai Corporation Discord Bot v2.0
© 2026 Zenkai Corporation
═══════════════════════════════════════════════════════════════════════════════

The AI-powered heart of the Zenkai Discord server.
Features: Welcome, roles, announcements, moderation, leveling, signal pipeline.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path

import discord
from discord.ext import commands
from discord import app_commands
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = os.getenv("GUILD_ID")

if not TOKEN:
    raise ValueError("DISCORD_TOKEN not found in environment variables!")

# Load config
CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("bulma")

# ─────────────────────────────────────────────────────────────────────────────
# Bot Setup
# ─────────────────────────────────────────────────────────────────────────────

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.reactions = True
intents.guilds = True

class BulmaBot(commands.Bot):
    """Bulma — Zenkai Corporation's AI-powered Discord bot."""

    def __init__(self):
        super().__init__(
            command_prefix=commands.when_mentioned_or("!"),
            intents=intents,
            help_command=None,
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="🔥 Every Setback, Stronger"
            )
        )
        self.config = CONFIG
        self.colors = {
            "primary": int(CONFIG["colors"]["primary"].replace("#", ""), 16),
            "secondary": int(CONFIG["colors"]["secondary"].replace("#", ""), 16),
            "success": 0x00ff88,
            "warning": 0xffaa00,
            "error": 0xff4466,
        }

    async def setup_hook(self):
        """Called when the bot is starting up."""
        # Load cogs
        cogs = [
            "cogs.welcome",
            "cogs.roles",
            "cogs.commands",
            "cogs.announcements",
            "cogs.moderation",
            "cogs.leveling",
            "cogs.signals_pipeline",
        ]
        for cog in cogs:
            try:
                await self.load_extension(cog)
                logger.info(f"Loaded cog: {cog}")
            except Exception as e:
                logger.error(f"Failed to load cog {cog}: {e}")

        # Sync slash commands
        if GUILD_ID:
            guild = discord.Object(id=int(GUILD_ID))
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            logger.info(f"Synced commands to guild {GUILD_ID}")
        else:
            await self.tree.sync()
            logger.info("Synced global commands")

    async def on_ready(self):
        """Called when the bot is fully ready."""
        logger.info("=" * 50)
        logger.info("⚡ BULMA ONLINE")
        logger.info(f"Logged in as: {self.user} (ID: {self.user.id})")
        logger.info(f"Guilds: {len(self.guilds)}")
        logger.info(f"discord.py version: {discord.__version__}")
        logger.info("=" * 50)

        # Auto-set nickname to "Bulma" in all guilds
        for guild in self.guilds:
            try:
                if guild.me.nick != "Bulma":
                    await guild.me.edit(nick="Bulma")
                    logger.info(f"Set nickname to 'Bulma' in {guild.name}")
            except discord.Forbidden:
                logger.warning(f"Cannot set nickname in {guild.name} - missing permissions")

    async def on_command_error(self, ctx: commands.Context, error: commands.CommandError):
        """Global error handler for commands."""
        if isinstance(error, commands.CommandNotFound):
            return

        if isinstance(error, commands.MissingPermissions):
            embed = discord.Embed(
                title="❌ Permission Denied",
                description="You don't have permission to use this command.",
                color=self.colors["error"]
            )
            await ctx.send(embed=embed, delete_after=10)
            return

        if isinstance(error, commands.MissingRequiredArgument):
            embed = discord.Embed(
                title="❌ Missing Argument",
                description=f"Missing required argument: `{error.param.name}`",
                color=self.colors["error"]
            )
            await ctx.send(embed=embed, delete_after=10)
            return

        # Log unexpected errors
        logger.error(f"Command error in {ctx.command}: {error}")

    def create_embed(
        self,
        title: str = None,
        description: str = None,
        color: str = "primary",
        thumbnail: bool = True
    ) -> discord.Embed:
        """Create a styled Zenkai embed."""
        embed = discord.Embed(
            title=title,
            description=description,
            color=self.colors.get(color, self.colors["primary"]),
            timestamp=datetime.utcnow()
        )
        if thumbnail and self.config.get("bot_avatar_url"):
            embed.set_thumbnail(url=self.config["bot_avatar_url"])
        embed.set_footer(
            text="⚡ Zenkai Corporation — Every Setback, Stronger",
            icon_url=self.config.get("bot_avatar_url")
        )
        return embed


# ─────────────────────────────────────────────────────────────────────────────
# Run Bot
# ─────────────────────────────────────────────────────────────────────────────

def main():
    bot = BulmaBot()
    bot.run(TOKEN, log_handler=None)


if __name__ == "__main__":
    main()

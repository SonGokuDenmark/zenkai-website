"""
═══════════════════════════════════════════════════════════════════════════════
⚡ Zenkai Bot — Server Setup
═══════════════════════════════════════════════════════════════════════════════

One-time command to create all channels and categories from config.json.
Run !setupserver once, then unload this cog.
"""

import logging
import json
import discord
from discord.ext import commands
from pathlib import Path

logger = logging.getLogger("zenkai-bot.setup")

# Channel descriptions for a clean server
CHANNEL_DESCRIPTIONS = {
    "welcome": "Welcome to Zenkai Corporation! New members land here.",
    "roles": "React to pick your roles and interests.",
    "announcements": "Official updates from the Zenkai team.",
    "general": "Main hangout — chat about anything.",
    "memes": "Shitposts, memes, and good vibes only.",
    "tech-talk": "AI, coding, projects, automation — nerd out here.",
    "arena-lobby": "Hatch your Saiyan, check stats, view leaderboards. Use !hatch to start!",
    "battles": "Challenge and fight other players here. Use !fight @user to throw hands!",
    "training-grounds": "Train your Saiyan and boost its power. Use !train and !feed here.",
    "tournament-hall": "Tournament brackets, schedules, and results.",
    "signals": "AlphaTrader trading signals land here.",
    "trading-chat": "Discuss trades, strategies, and market analysis.",
    "bot-status": "Bot logs, errors, and system health. Admin only.",
    "team-chat": "Core team coordination. Admin only.",
}

# Categories that should be hidden (admin only)
ADMIN_CATEGORIES = ["🔧 ADMIN"]

# Categories to hide until ready
HIDDEN_CATEGORIES = ["📈 TRADING"]


class ServerSetup(commands.Cog, name="ServerSetup"):
    """One-time server channel setup from config.json."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @commands.command(name="setupserver")
    @commands.has_permissions(administrator=True)
    async def setup_server(self, ctx: commands.Context):
        """Create all categories and channels from config.json."""
        guild = ctx.guild
        config = self.bot.config
        categories_config = config.get("channel_categories", {})

        if not categories_config:
            await ctx.send("❌ No `channel_categories` found in config.json!")
            return

        status_msg = await ctx.send("⚡ Setting up Zenkai server...\n")
        log_lines = []

        for category_name, channel_names in categories_config.items():
            # Check if category exists
            existing_cat = discord.utils.get(guild.categories, name=category_name)

            if existing_cat:
                category = existing_cat
                log_lines.append(f"📁 Category `{category_name}` already exists")
            else:
                # Set permissions for admin/hidden categories
                overwrites = None
                if category_name in ADMIN_CATEGORIES:
                    overwrites = {
                        guild.default_role: discord.PermissionOverwrite(view_channel=False),
                        guild.me: discord.PermissionOverwrite(view_channel=True),
                    }
                    # Add admin roles
                    for role in guild.roles:
                        if role.permissions.administrator:
                            overwrites[role] = discord.PermissionOverwrite(view_channel=True)

                elif category_name in HIDDEN_CATEGORIES:
                    overwrites = {
                        guild.default_role: discord.PermissionOverwrite(view_channel=False),
                        guild.me: discord.PermissionOverwrite(view_channel=True),
                    }

                category = await guild.create_category(
                    name=category_name,
                    overwrites=overwrites
                )
                log_lines.append(f"✅ Created category `{category_name}`")

            # Create channels in this category
            for ch_name in channel_names:
                existing_ch = discord.utils.get(guild.text_channels, name=ch_name)
                if existing_ch:
                    # Move to correct category if needed
                    if existing_ch.category != category:
                        await existing_ch.edit(category=category)
                        log_lines.append(f"  ↪ Moved `#{ch_name}` to `{category_name}`")
                    else:
                        log_lines.append(f"  ⏭️ `#{ch_name}` already exists")
                else:
                    description = CHANNEL_DESCRIPTIONS.get(ch_name, "")

                    # Announcements channel = news channel (read-only for non-admins)
                    if ch_name == "announcements":
                        overwrites = {
                            guild.default_role: discord.PermissionOverwrite(
                                send_messages=False,
                                view_channel=True
                            ),
                            guild.me: discord.PermissionOverwrite(
                                send_messages=True,
                                view_channel=True
                            ),
                        }
                        await guild.create_text_channel(
                            name=ch_name,
                            category=category,
                            topic=description,
                            overwrites=overwrites,
                        )
                    else:
                        await guild.create_text_channel(
                            name=ch_name,
                            category=category,
                            topic=description,
                        )

                    log_lines.append(f"  ✅ Created `#{ch_name}`")

        # Summary
        log_text = "\n".join(log_lines)
        summary = f"⚡ **Server setup complete!**\n\n{log_text}\n\n🐉 Zenkai Corporation is ready."
        await status_msg.edit(content=summary)
        logger.info("Server setup completed by %s", ctx.author)

    @commands.command(name="nukechannels")
    @commands.has_permissions(administrator=True)
    async def nuke_channels(self, ctx: commands.Context):
        """Delete ALL categories and channels created by setupserver. Use with caution!"""
        await ctx.send("⚠️ This will delete all categories from config. Type `CONFIRM` within 15s to proceed.")

        def check(m):
            return m.author == ctx.author and m.content == "CONFIRM" and m.channel == ctx.channel

        try:
            await self.bot.wait_for("message", check=check, timeout=15.0)
        except Exception:
            await ctx.send("Cancelled.")
            return

        guild = ctx.guild
        config = self.bot.config
        categories_config = config.get("channel_categories", {})

        for category_name in categories_config:
            cat = discord.utils.get(guild.categories, name=category_name)
            if cat:
                for ch in cat.channels:
                    await ch.delete()
                await cat.delete()

        await ctx.send("💥 All Zenkai channels and categories deleted. Run `!setupserver` to rebuild.")


async def setup(bot: commands.Bot):
    await bot.add_cog(ServerSetup(bot))

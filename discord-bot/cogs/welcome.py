"""
═══════════════════════════════════════════════════════════════════════════════
⚡ Zenkai Bot — Welcome System
═══════════════════════════════════════════════════════════════════════════════

Handles new member joins:
- Sends welcome DM with server info
- Posts welcome message in channel
- Auto-assigns Member role
"""

import logging
import discord
from discord.ext import commands

logger = logging.getLogger("zenkai-bot.welcome")


class WelcomeCog(commands.Cog, name="Welcome"):
    """Welcome system for new members."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    def get_welcome_channel(self, guild: discord.Guild) -> discord.TextChannel | None:
        """Get the welcome channel for a guild."""
        channel_name = self.bot.config["channels"].get("welcome", "welcome")
        return discord.utils.get(guild.text_channels, name=channel_name)

    def get_roles_channel(self, guild: discord.Guild) -> discord.TextChannel | None:
        """Get the roles channel for a guild."""
        channel_name = self.bot.config["channels"].get("roles", "roles")
        return discord.utils.get(guild.text_channels, name=channel_name)

    def get_member_role(self, guild: discord.Guild) -> discord.Role | None:
        """Get the Member role for a guild."""
        role_name = self.bot.config["roles"].get("member", "Member")
        return discord.utils.get(guild.roles, name=role_name)

    @commands.Cog.listener()
    async def on_member_join(self, member: discord.Member):
        """Handle new member joins."""
        logger.info(f"New member joined: {member} in {member.guild}")

        # Auto-assign Member role
        member_role = self.get_member_role(member.guild)
        if member_role:
            try:
                await member.add_roles(member_role)
                logger.info(f"Assigned Member role to {member}")
            except discord.Forbidden:
                logger.warning(f"Cannot assign role to {member} - missing permissions")

        # Send welcome DM
        await self.send_welcome_dm(member)

        # Post in welcome channel
        await self.post_welcome_message(member)

    async def send_welcome_dm(self, member: discord.Member):
        """Send a welcome DM to the new member."""
        config = self.bot.config["welcome_message"]
        roles_channel = self.get_roles_channel(member.guild)

        # Build DM description
        dm_text = config["dm_description"].format(
            website=self.bot.config.get("website_url", "https://zenkaicorp.com")
        )

        embed = discord.Embed(
            title=config["title"],
            description=dm_text,
            color=self.bot.colors["primary"]
        )

        # Add projects field
        projects = self.bot.config.get("projects", [])[:3]
        if projects:
            project_text = "\n".join([
                f"• **{p['name']}** — {p['status']}"
                for p in projects
            ])
            embed.add_field(
                name="🚀 Our Projects",
                value=project_text,
                inline=False
            )

        embed.set_footer(text=config["footer"])

        # Add server icon as thumbnail
        if member.guild.icon:
            embed.set_thumbnail(url=member.guild.icon.url)

        try:
            await member.send(embed=embed)
            logger.info(f"Sent welcome DM to {member}")
        except discord.Forbidden:
            logger.warning(f"Cannot send DM to {member} - DMs disabled")

    async def post_welcome_message(self, member: discord.Member):
        """Post a welcome message in the welcome channel."""
        channel = self.get_welcome_channel(member.guild)
        if not channel:
            logger.warning(f"Welcome channel not found in {member.guild}")
            return

        config = self.bot.config["welcome_message"]

        embed = discord.Embed(
            title="👋 New Member!",
            description=config["description"].format(member=member.mention),
            color=self.bot.colors["primary"]
        )

        # Member info
        embed.add_field(
            name="Member",
            value=f"{member.mention}\n`{member.name}`",
            inline=True
        )
        embed.add_field(
            name="Member #",
            value=f"`{member.guild.member_count}`",
            inline=True
        )
        embed.add_field(
            name="Account Created",
            value=f"<t:{int(member.created_at.timestamp())}:R>",
            inline=True
        )

        # Set member avatar
        embed.set_thumbnail(url=member.display_avatar.url)

        embed.set_footer(
            text="⚡ Zenkai Corporation — Every Setback, Stronger",
            icon_url=member.guild.icon.url if member.guild.icon else None
        )

        try:
            await channel.send(embed=embed)
            logger.info(f"Posted welcome message for {member}")
        except discord.Forbidden:
            logger.error(f"Cannot post in welcome channel - missing permissions")

    @commands.command(name="testwelcome")
    @commands.has_permissions(administrator=True)
    async def test_welcome(self, ctx: commands.Context, member: discord.Member = None):
        """Test the welcome system. Admin only."""
        target = member or ctx.author
        await self.post_welcome_message(target)
        await ctx.send(f"✅ Sent test welcome for {target.mention}", delete_after=5)


async def setup(bot: commands.Bot):
    await bot.add_cog(WelcomeCog(bot))

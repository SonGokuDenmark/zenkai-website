"""
═══════════════════════════════════════════════════════════════════════════════
⚡ Zenkai Bot — Reaction Roles System
═══════════════════════════════════════════════════════════════════════════════

Handles reaction-based role assignment:
- Posts role selection embed
- Adds/removes roles based on reactions
"""

import logging
import json
from pathlib import Path

import discord
from discord.ext import commands

logger = logging.getLogger("zenkai-bot.roles")

# Store the role message ID
ROLE_MESSAGE_FILE = Path(__file__).parent.parent / "role_message.json"


class RolesCog(commands.Cog, name="Roles"):
    """Reaction role system."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.role_message_id = self.load_role_message_id()

    def load_role_message_id(self) -> int | None:
        """Load the role message ID from file."""
        if ROLE_MESSAGE_FILE.exists():
            try:
                with open(ROLE_MESSAGE_FILE, "r") as f:
                    data = json.load(f)
                    return data.get("message_id")
            except Exception as e:
                logger.error(f"Failed to load role message ID: {e}")
        return None

    def save_role_message_id(self, message_id: int):
        """Save the role message ID to file."""
        try:
            with open(ROLE_MESSAGE_FILE, "w") as f:
                json.dump({"message_id": message_id}, f)
            self.role_message_id = message_id
        except Exception as e:
            logger.error(f"Failed to save role message ID: {e}")

    def get_roles_channel(self, guild: discord.Guild) -> discord.TextChannel | None:
        """Get the roles channel for a guild."""
        channel_name = self.bot.config["channels"].get("roles", "roles")
        return discord.utils.get(guild.text_channels, name=channel_name)

    def get_role_config(self) -> dict:
        """Get reaction roles configuration."""
        return self.bot.config["roles"].get("reaction_roles", {})

    @commands.command(name="setuproles")
    @commands.has_permissions(administrator=True)
    async def setup_roles(self, ctx: commands.Context):
        """
        Post the reaction roles embed in the roles channel.
        Admin only. Creates roles if they don't exist.
        """
        channel = self.get_roles_channel(ctx.guild)
        if not channel:
            await ctx.send("❌ Roles channel not found! Create a `#roles` channel first.")
            return

        role_config = self.get_role_config()

        # Create roles if they don't exist
        created_roles = []
        for emoji, role_data in role_config.items():
            role_name = role_data["name"]
            existing_role = discord.utils.get(ctx.guild.roles, name=role_name)
            if not existing_role:
                try:
                    await ctx.guild.create_role(
                        name=role_name,
                        mentionable=True,
                        reason="Zenkai Bot - Reaction role setup"
                    )
                    created_roles.append(role_name)
                    logger.info(f"Created role: {role_name}")
                except discord.Forbidden:
                    await ctx.send(f"❌ Cannot create role `{role_name}` - missing permissions")
                    return

        if created_roles:
            await ctx.send(f"✅ Created roles: {', '.join(created_roles)}")

        # Build the embed
        embed = discord.Embed(
            title="🎭 Role Selection",
            description=(
                "React to this message to get roles based on your interests!\n"
                "Remove your reaction to remove the role.\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━"
            ),
            color=self.bot.colors["secondary"]
        )

        # Add role descriptions
        role_text = ""
        for emoji, role_data in role_config.items():
            role_text += f"{emoji} **{role_data['name']}**\n"
            role_text += f"└ {role_data['description']}\n\n"

        embed.add_field(
            name="Available Roles",
            value=role_text,
            inline=False
        )

        embed.set_footer(
            text="⚡ Zenkai Corporation — Pick your interests!",
            icon_url=ctx.guild.icon.url if ctx.guild.icon else None
        )

        # Send the embed
        try:
            message = await channel.send(embed=embed)

            # Add reactions
            for emoji in role_config.keys():
                await message.add_reaction(emoji)

            # Save message ID
            self.save_role_message_id(message.id)

            await ctx.send(f"✅ Role selection posted in {channel.mention}!")
            logger.info(f"Posted role selection embed (ID: {message.id})")

        except discord.Forbidden:
            await ctx.send("❌ Cannot post in roles channel - missing permissions")

    @commands.Cog.listener()
    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        """Handle reaction adds for role assignment."""
        # Ignore bot reactions
        if payload.user_id == self.bot.user.id:
            return

        # Check if this is the role message
        if payload.message_id != self.role_message_id:
            return

        role_config = self.get_role_config()
        emoji_str = str(payload.emoji)

        if emoji_str not in role_config:
            return

        guild = self.bot.get_guild(payload.guild_id)
        if not guild:
            return

        member = guild.get_member(payload.user_id)
        if not member:
            return

        role_name = role_config[emoji_str]["name"]
        role = discord.utils.get(guild.roles, name=role_name)

        if role:
            try:
                await member.add_roles(role, reason="Reaction role")
                logger.info(f"Added role {role_name} to {member}")
            except discord.Forbidden:
                logger.warning(f"Cannot add role {role_name} to {member}")

    @commands.Cog.listener()
    async def on_raw_reaction_remove(self, payload: discord.RawReactionActionEvent):
        """Handle reaction removes for role removal."""
        # Check if this is the role message
        if payload.message_id != self.role_message_id:
            return

        role_config = self.get_role_config()
        emoji_str = str(payload.emoji)

        if emoji_str not in role_config:
            return

        guild = self.bot.get_guild(payload.guild_id)
        if not guild:
            return

        member = guild.get_member(payload.user_id)
        if not member:
            return

        role_name = role_config[emoji_str]["name"]
        role = discord.utils.get(guild.roles, name=role_name)

        if role and role in member.roles:
            try:
                await member.remove_roles(role, reason="Reaction role removed")
                logger.info(f"Removed role {role_name} from {member}")
            except discord.Forbidden:
                logger.warning(f"Cannot remove role {role_name} from {member}")

    @commands.command(name="createroles")
    @commands.has_permissions(administrator=True)
    async def create_roles(self, ctx: commands.Context):
        """Create the Member role and reaction roles. Admin only."""
        created = []

        # Create Member role
        member_role_name = self.bot.config["roles"].get("member", "Member")
        if not discord.utils.get(ctx.guild.roles, name=member_role_name):
            try:
                await ctx.guild.create_role(
                    name=member_role_name,
                    color=discord.Color.from_str(self.bot.config["colors"]["primary"]),
                    mentionable=False,
                    reason="Zenkai Bot - Member role"
                )
                created.append(member_role_name)
            except discord.Forbidden:
                await ctx.send(f"❌ Cannot create Member role")
                return

        # Create reaction roles
        role_config = self.get_role_config()
        for emoji, role_data in role_config.items():
            role_name = role_data["name"]
            if not discord.utils.get(ctx.guild.roles, name=role_name):
                try:
                    await ctx.guild.create_role(
                        name=role_name,
                        mentionable=True,
                        reason="Zenkai Bot - Reaction role"
                    )
                    created.append(role_name)
                except discord.Forbidden:
                    await ctx.send(f"❌ Cannot create role: {role_name}")

        if created:
            await ctx.send(f"✅ Created roles: `{', '.join(created)}`")
        else:
            await ctx.send("✅ All roles already exist!")


async def setup(bot: commands.Bot):
    await bot.add_cog(RolesCog(bot))

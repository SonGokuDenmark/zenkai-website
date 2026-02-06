"""
Zenkai Arena — AI Battle Pets Discord Cog (Phase 1: Text Battles)

Commands:
  !hatch [name]    — Hatch a Saiyan with random stats
  !status          — View your Saiyan's stats
  !fight @user     — Challenge someone to battle
  !accept          — Accept a battle challenge
  !train [type]    — Training session (gravity/ki_control/sparring)
  !feed            — Restore energy, improve mood
  !leaderboard     — Top 10 by power level
  !testbattle      — Test fight between two random-policy Saiyans
"""

import sys
import os
import json
import asyncio
import random
import functools
from datetime import datetime

import discord
from discord.ext import commands

# Add zenkai-arena to path for imports
ARENA_ROOT = r'C:\Zenkai\zenkai-arena'
if ARENA_ROOT not in sys.path:
    sys.path.insert(0, ARENA_ROOT)

from src.database.db import ArenaDB
from src.creatures.saiyan import Saiyan
from src.creatures.stats import roll_stats, calculate_power_level, xp_to_level
from src.creatures.forms import check_form_unlock, FORM_MULTIPLIERS
from src.creatures.training import TRAINING_TYPES, can_train, apply_training
from src.battle.engine import BattleEngine
from src.commentary.commentator import Commentator
from src.models.policy_network import SaiyanPolicy, save_model, load_model, get_model_path
from src.models.trainer import train_from_battle


class ZenkaiArena(commands.Cog):
    """AI Battle Pets — Dragon Ball Edition"""

    def __init__(self, bot):
        self.bot = bot
        self.db = ArenaDB(os.path.join(ARENA_ROOT, 'data', 'arena.db'))
        self.engine = BattleEngine()
        self.commentator = Commentator()
        self.pending_challenges = {}  # {defender_id: {"challenger_id": ..., "timestamp": ...}}

    async def cog_load(self):
        await self.db.init()

    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────

    async def _get_saiyan(self, user_id):
        row = await self.db.get_saiyan_by_user(str(user_id))
        if row:
            return Saiyan.from_row(row)
        return None

    def _make_embed(self, title, description=None, color=0x00ff88):
        embed = discord.Embed(title=title, description=description, color=color)
        embed.set_footer(text="Zenkai Arena — Every Setback, Stronger")
        return embed

    async def _save_saiyan(self, saiyan):
        await self.db.update_saiyan(saiyan.id, **saiyan.to_update_dict())

    # ─────────────────────────────────────────────────────────────
    # !hatch [name]
    # ─────────────────────────────────────────────────────────────

    @commands.command(name="hatch")
    async def hatch(self, ctx, *, name: str = None):
        """Hatch your Saiyan! One per user."""
        user_id = str(ctx.author.id)

        existing = await self._get_saiyan(ctx.author.id)
        if existing:
            await ctx.send(embed=self._make_embed(
                "You Already Have a Saiyan!",
                f"Your Saiyan **{existing.name}** is waiting for you! Use `!status` to check on them.",
                color=0xff4466,
            ))
            return

        if not name:
            name = f"{ctx.author.display_name}'s Saiyan"

        stats = roll_stats()
        model_path = get_model_path(user_id, "pending")

        # Create and save initial policy network
        policy = SaiyanPolicy()
        saiyan_id = await self.db.create_saiyan(user_id, name, stats, "")

        # Save model with correct path
        final_model_path = get_model_path(user_id, saiyan_id)
        save_model(policy, final_model_path)
        await self.db.update_saiyan(saiyan_id, model_path=final_model_path)

        power = int(
            stats['strength'] * 10 + stats['defense'] * 8 +
            stats['speed'] * 9 + stats['ki_power'] * 12 + 50
        )

        embed = self._make_embed(
            f"A Saiyan is Born!",
            f"**{name}** emerges from the incubation pod!",
        )
        embed.add_field(name="STR", value=f"{stats['strength']:.0f}", inline=True)
        embed.add_field(name="DEF", value=f"{stats['defense']:.0f}", inline=True)
        embed.add_field(name="SPD", value=f"{stats['speed']:.0f}", inline=True)
        embed.add_field(name="KI", value=f"{stats['ki_power']:.0f}", inline=True)
        embed.add_field(name="HP", value=f"{stats['max_hp']:.0f}", inline=True)
        embed.add_field(name="Max KI", value=f"{stats['max_ki']:.0f}", inline=True)
        embed.add_field(name="Power Level", value=f"{power:,}", inline=False)
        embed.add_field(name="Form", value="Base", inline=True)
        embed.add_field(name="Neural Net", value="Initialized (untrained)", inline=True)

        await ctx.send(embed=embed)

    # ─────────────────────────────────────────────────────────────
    # !status / !status @user
    # ─────────────────────────────────────────────────────────────

    @commands.command(name="status")
    async def status(self, ctx, member: discord.Member = None):
        """View Saiyan stats, power level, form, mood, W/L."""
        target = member or ctx.author
        saiyan = await self._get_saiyan(target.id)

        if not saiyan:
            if target == ctx.author:
                await ctx.send(embed=self._make_embed(
                    "No Saiyan Found",
                    "You don't have a Saiyan yet! Use `!hatch [name]` to create one.",
                    color=0xff4466,
                ))
            else:
                await ctx.send(embed=self._make_embed(
                    "No Saiyan Found",
                    f"{target.display_name} doesn't have a Saiyan yet.",
                    color=0xff4466,
                ))
            return

        power = calculate_power_level(saiyan)
        level = xp_to_level(saiyan.experience)

        mood_emoji = {"happy": "😄", "neutral": "😐", "angry": "😤", "exhausted": "😩"}

        embed = self._make_embed(
            f"{saiyan.name} — Lv.{level} {saiyan.current_form.upper()}",
            f"Owner: {target.display_name}",
        )
        embed.add_field(name="Power Level", value=f"**{power:,}**", inline=False)
        embed.add_field(name="STR", value=f"{saiyan.strength:.1f}", inline=True)
        embed.add_field(name="DEF", value=f"{saiyan.defense:.1f}", inline=True)
        embed.add_field(name="SPD", value=f"{saiyan.speed:.1f}", inline=True)
        embed.add_field(name="KI Power", value=f"{saiyan.ki_power:.1f}", inline=True)
        embed.add_field(name="Max HP", value=f"{saiyan.max_hp:.1f}", inline=True)
        embed.add_field(name="Max KI", value=f"{saiyan.max_ki:.1f}", inline=True)
        embed.add_field(name="Record", value=f"{saiyan.wins}W / {saiyan.losses}L", inline=True)
        embed.add_field(name="Streak", value=f"{saiyan.win_streak} (best: {saiyan.best_streak})", inline=True)
        embed.add_field(name="Mood", value=f"{mood_emoji.get(saiyan.mood, '😐')} {saiyan.mood}", inline=True)
        embed.add_field(name="Energy", value=f"{saiyan.energy:.0f}/100", inline=True)
        embed.add_field(name="XP", value=f"{saiyan.experience:,}", inline=True)
        embed.add_field(name="Zenkai Boosts", value=f"{saiyan.total_zenkais} total ({saiyan.zenkai_stacks} active)", inline=True)
        embed.add_field(name="Forms", value=", ".join(f.upper() for f in saiyan.unlocked_forms), inline=False)
        embed.add_field(name="Tokens", value=f"{saiyan.battle_tokens:,}", inline=True)
        embed.add_field(name="Training Steps", value=f"{saiyan.training_steps:,}", inline=True)

        await ctx.send(embed=embed)

    # ─────────────────────────────────────────────────────────────
    # !fight @user
    # ─────────────────────────────────────────────────────────────

    @commands.command(name="fight")
    async def fight(self, ctx, member: discord.Member = None):
        """Challenge another player to battle!"""
        if not member:
            await ctx.send(embed=self._make_embed(
                "Who Do You Want to Fight?",
                "Usage: `!fight @user`",
                color=0xff4466,
            ))
            return

        if member.id == ctx.author.id:
            await ctx.send("You can't fight yourself! Use `!testbattle` for a test fight.")
            return

        if member.bot:
            await ctx.send("You can't fight a bot!")
            return

        challenger = await self._get_saiyan(ctx.author.id)
        defender = await self._get_saiyan(member.id)

        if not challenger:
            await ctx.send("You need to `!hatch` a Saiyan first!")
            return
        if not defender:
            await ctx.send(f"{member.display_name} doesn't have a Saiyan yet!")
            return

        if challenger.energy < 20:
            await ctx.send("Your Saiyan is too tired! Use `!feed` to restore energy.")
            return

        # Store challenge
        self.pending_challenges[member.id] = {
            "challenger_id": ctx.author.id,
            "timestamp": datetime.utcnow(),
            "channel_id": ctx.channel.id,
        }

        pl_c = calculate_power_level(challenger)
        pl_d = calculate_power_level(defender)

        embed = self._make_embed(
            "BATTLE CHALLENGE!",
            f"**{challenger.name}** (PL: {pl_c:,}) challenges **{defender.name}** (PL: {pl_d:,})!\n\n"
            f"{member.mention}, type `!accept` within 60 seconds to fight!",
            color=0xffaa00,
        )
        await ctx.send(embed=embed)

        # Auto-expire after 60 seconds
        await asyncio.sleep(60)
        if member.id in self.pending_challenges and self.pending_challenges[member.id]["challenger_id"] == ctx.author.id:
            del self.pending_challenges[member.id]
            await ctx.send(f"The challenge from {ctx.author.display_name} has expired.")

    # ─────────────────────────────────────────────────────────────
    # !accept
    # ─────────────────────────────────────────────────────────────

    @commands.command(name="accept")
    async def accept(self, ctx):
        """Accept a battle challenge!"""
        challenge = self.pending_challenges.pop(ctx.author.id, None)
        if not challenge:
            await ctx.send("You don't have any pending challenges!")
            return

        challenger_id = challenge["challenger_id"]
        challenger_saiyan = await self._get_saiyan(challenger_id)
        defender_saiyan = await self._get_saiyan(ctx.author.id)

        if not challenger_saiyan or not defender_saiyan:
            await ctx.send("Error: One of the Saiyans could not be loaded!")
            return

        await self._run_battle(ctx, challenger_saiyan, defender_saiyan, challenger_id, ctx.author.id)

    # ─────────────────────────────────────────────────────────────
    # !train [type]
    # ─────────────────────────────────────────────────────────────

    @commands.command(name="train")
    async def train(self, ctx, training_type: str = None):
        """Train your Saiyan! Types: gravity, ki_control, sparring"""
        if not training_type or training_type not in TRAINING_TYPES:
            types_list = "\n".join(
                f"**{k}** — {v['description']} (energy: {v['energy_cost']}, cooldown: {v['cooldown_hours']}h)"
                for k, v in TRAINING_TYPES.items()
            )
            await ctx.send(embed=self._make_embed(
                "Training Types",
                f"Usage: `!train <type>`\n\n{types_list}",
            ))
            return

        saiyan = await self._get_saiyan(ctx.author.id)
        if not saiyan:
            await ctx.send("You need to `!hatch` a Saiyan first!")
            return

        ok, msg = can_train(saiyan, training_type)
        if not ok:
            await ctx.send(embed=self._make_embed("Can't Train Right Now", msg, color=0xff4466))
            return

        result_text, injured = apply_training(saiyan, training_type)

        # Update level from XP
        saiyan.level = xp_to_level(saiyan.experience)
        saiyan.last_trained = datetime.utcnow().isoformat()

        await self._save_saiyan(saiyan)

        color = 0xff4466 if injured else 0x00ff88
        embed = self._make_embed("Training Results", result_text, color=color)
        if injured:
            embed.add_field(name="Status", value="INJURED! Rest up before training again.", inline=False)

        await ctx.send(embed=embed)

    # ─────────────────────────────────────────────────────────────
    # !feed
    # ─────────────────────────────────────────────────────────────

    @commands.command(name="feed")
    async def feed(self, ctx):
        """Feed your Saiyan to restore energy and improve mood."""
        saiyan = await self._get_saiyan(ctx.author.id)
        if not saiyan:
            await ctx.send("You need to `!hatch` a Saiyan first!")
            return

        # Restore energy
        old_energy = saiyan.energy
        saiyan.energy = min(100, saiyan.energy + 30)

        # Improve mood
        mood_improve = {"exhausted": "angry", "angry": "neutral", "neutral": "happy", "happy": "happy"}
        old_mood = saiyan.mood
        saiyan.mood = mood_improve.get(saiyan.mood, "neutral")
        saiyan.last_fed = datetime.utcnow().isoformat()

        # XP for feeding
        saiyan.experience += 10
        saiyan.level = xp_to_level(saiyan.experience)

        await self._save_saiyan(saiyan)

        embed = self._make_embed(
            f"{saiyan.name} Enjoyed the Meal!",
            f"Energy: {old_energy:.0f} -> {saiyan.energy:.0f}\n"
            f"Mood: {old_mood} -> {saiyan.mood}\n"
            f"XP +10",
        )
        await ctx.send(embed=embed)

    # ─────────────────────────────────────────────────────────────
    # !leaderboard
    # ─────────────────────────────────────────────────────────────

    @commands.command(name="leaderboard")
    async def leaderboard(self, ctx):
        """Top 10 Saiyans by power level."""
        rows = await self.db.get_leaderboard(10)
        if not rows:
            await ctx.send(embed=self._make_embed(
                "Leaderboard",
                "No Saiyans exist yet! Be the first to `!hatch`!",
            ))
            return

        lines = []
        for i, row in enumerate(rows, 1):
            s = Saiyan.from_row(row)
            pl = calculate_power_level(s)
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"**{i}.**")
            lines.append(f"{medal} **{s.name}** — PL: {pl:,} | Lv.{s.level} {s.current_form.upper()} | {s.wins}W/{s.losses}L")

        embed = self._make_embed(
            "POWER LEVEL RANKINGS",
            "\n".join(lines),
        )
        await ctx.send(embed=embed)

    # ─────────────────────────────────────────────────────────────
    # !testbattle
    # ─────────────────────────────────────────────────────────────

    @commands.command(name="testbattle")
    async def testbattle(self, ctx):
        """Fight two random-policy Saiyans for testing."""
        stats1 = roll_stats()
        stats2 = roll_stats()

        s1 = Saiyan(
            id=0, user_id="test1", name="Test Warrior Alpha",
            **stats1, current_hp=stats1['max_hp'], current_ki=stats1['max_ki'],
        )
        s2 = Saiyan(
            id=0, user_id="test2", name="Test Warrior Beta",
            **stats2, current_hp=stats2['max_hp'], current_ki=stats2['max_ki'],
        )

        await ctx.send(embed=self._make_embed(
            "TEST BATTLE INITIATED",
            f"**{s1.name}** vs **{s2.name}** — both using random AI policies!",
            color=0xffaa00,
        ))

        # Run battle in executor to not block event loop
        loop = asyncio.get_event_loop()
        winner_num, battle_log, state, _, _ = await loop.run_in_executor(
            None, functools.partial(self.engine.run_battle, s1, s2, None, None)
        )

        winner = s1 if winner_num == 1 else s2
        loser = s2 if winner_num == 1 else s1

        # Generate commentary
        commentary = self.commentator.narrate_battle(s1, s2, battle_log, state)

        # Split commentary into chunks for Discord (max 4096 chars per embed)
        chunks = [commentary[i:i+4000] for i in range(0, len(commentary), 4000)]

        for i, chunk in enumerate(chunks):
            if i == 0:
                embed = self._make_embed(f"BATTLE: {s1.name} vs {s2.name}", chunk)
            else:
                embed = self._make_embed("Battle (continued)", chunk)
            await ctx.send(embed=embed)

        # Result summary
        winner_hp_pct = round(state.hp[winner_num] / state.max_hp[winner_num] * 100)
        embed = self._make_embed(
            f"WINNER: {winner.name}!",
            f"Turns: {state.turn}\n"
            f"Winner HP: {winner_hp_pct}% remaining\n"
            f"This was a test battle — no XP or stats were affected.",
            color=0xffd700,
        )
        await ctx.send(embed=embed)

    # ─────────────────────────────────────────────────────────────
    # Battle execution (shared by !accept and future commands)
    # ─────────────────────────────────────────────────────────────

    async def _run_battle(self, ctx, saiyan1, saiyan2, user1_id, user2_id):
        """Execute a full battle between two player Saiyans."""
        # Load neural nets
        policy1 = load_model(saiyan1.model_path) if saiyan1.model_path else SaiyanPolicy()
        policy2 = load_model(saiyan2.model_path) if saiyan2.model_path else SaiyanPolicy()

        await ctx.send(embed=self._make_embed(
            "BATTLE BEGINS!",
            f"**{saiyan1.name}** vs **{saiyan2.name}** — neural networks engaged!",
            color=0xffaa00,
        ))

        # Run battle in executor
        loop = asyncio.get_event_loop()
        winner_num, battle_log, state, exp1, exp2 = await loop.run_in_executor(
            None, functools.partial(self.engine.run_battle, saiyan1, saiyan2, policy1, policy2)
        )

        winner = saiyan1 if winner_num == 1 else saiyan2
        loser = saiyan2 if winner_num == 1 else saiyan1
        winner_user_id = user1_id if winner_num == 1 else user2_id
        loser_user_id = user2_id if winner_num == 1 else user1_id

        # Generate commentary
        commentary = self.commentator.narrate_battle(saiyan1, saiyan2, battle_log, state)

        # Post commentary
        chunks = [commentary[i:i+4000] for i in range(0, len(commentary), 4000)]
        for i, chunk in enumerate(chunks):
            if i == 0:
                embed = self._make_embed(f"BATTLE: {saiyan1.name} vs {saiyan2.name}", chunk)
            else:
                embed = self._make_embed("Battle (continued)", chunk)
            await ctx.send(embed=embed)

        # --- Post-battle logic ---

        # Determine if close fight (winner HP <= 20%)
        winner_hp_pct = state.hp[winner_num] / state.max_hp[winner_num]
        close_fight = winner_hp_pct <= 0.20

        # XP
        winner_xp = 150 if close_fight else 100
        loser_xp = 40
        if close_fight:
            loser_xp += 50

        winner.experience += winner_xp
        loser.experience += loser_xp
        winner.level = xp_to_level(winner.experience)
        loser.level = xp_to_level(loser.experience)

        # Battle tokens
        winner_tokens = 75 if close_fight else 50
        loser_tokens = 40 if close_fight else 15
        winner.battle_tokens += winner_tokens
        loser.battle_tokens += loser_tokens

        # W/L records
        winner.wins += 1
        winner.win_streak += 1
        winner.best_streak = max(winner.best_streak, winner.win_streak)
        loser.losses += 1
        loser.win_streak = 0

        # Zenkai boost for loser
        loser.zenkai_stacks = min(loser.zenkai_stacks + 1, 3)
        loser.total_zenkais += 1
        stat_boost = 0.5 * loser.zenkai_stacks
        loser.strength += stat_boost
        loser.defense += stat_boost

        # Reset zenkai for winner
        winner.zenkai_stacks = 0

        # Energy cost
        winner.energy = max(0, winner.energy - 15)
        loser.energy = max(0, loser.energy - 15)

        # Update last_battle
        now = datetime.utcnow().isoformat()
        winner.last_battle = now
        loser.last_battle = now

        # Check form unlocks
        battle_events_winner = {
            "min_hp_pct": state.min_hp_pct[winner_num],
            "total_turns": state.turn,
            "dodges": state.dodges[winner_num],
        }
        battle_events_loser = {
            "min_hp_pct": state.min_hp_pct[2 if winner_num == 1 else 1],
            "total_turns": state.turn,
            "dodges": state.dodges[2 if winner_num == 1 else 1],
        }

        unlock_w = check_form_unlock(winner, battle_events_winner)
        unlock_l = check_form_unlock(loser, battle_events_loser)

        if unlock_w:
            form, flavor = unlock_w
            winner.unlocked_forms.append(form)
            await ctx.send(embed=self._make_embed("TRANSFORMATION UNLOCKED!", flavor, color=0xffd700))

        if unlock_l:
            form, flavor = unlock_l
            loser.unlocked_forms.append(form)
            await ctx.send(embed=self._make_embed("TRANSFORMATION UNLOCKED!", flavor, color=0xffd700))

        # Train neural nets from battle experience
        try:
            if exp1:
                train_from_battle(policy1, exp1, saiyan1)
                save_model(policy1, saiyan1.model_path)
                saiyan1.training_steps += len(exp1)
            if exp2:
                train_from_battle(policy2, exp2, saiyan2)
                save_model(policy2, saiyan2.model_path)
                saiyan2.training_steps += len(exp2)
        except Exception as e:
            print(f"Warning: NN training failed: {e}")

        # Save to DB
        await self._save_saiyan(winner)
        await self._save_saiyan(loser)

        # Save battle record
        log_json = json.dumps(battle_log)
        xp_c = winner_xp if winner_num == 1 else loser_xp
        xp_d = loser_xp if winner_num == 1 else winner_xp
        await self.db.save_battle(
            saiyan1.id, saiyan2.id,
            winner.id, state.turn,
            log_json, commentary,
            xp_c, xp_d,
        )

        # Result embed
        winner_hp_display = round(winner_hp_pct * 100)
        result_text = (
            f"**Winner:** {winner.name} ({winner_hp_display}% HP remaining)\n"
            f"**Turns:** {state.turn}\n\n"
            f"**{winner.name}:** +{winner_xp} XP, +{winner_tokens} tokens\n"
            f"**{loser.name}:** +{loser_xp} XP, +{loser_tokens} tokens, ZENKAI BOOST!"
        )
        if close_fight:
            result_text += "\n\n**CLOSE FIGHT BONUS!** Extra XP and tokens awarded!"

        embed = self._make_embed(f"WINNER: {winner.name}!", result_text, color=0xffd700)
        await ctx.send(embed=embed)

        # Zenkai flavor text
        zenkai_text = random.choice([
            f"{loser.name} remembers this defeat... and POWER SURGES through them!",
            f"A Saiyan's pride! {loser.name}'s zenkai boost kicks in! They're STRONGER than before!",
        ])
        embed = self._make_embed("ZENKAI BOOST!", zenkai_text, color=0xff6600)
        embed.add_field(name="Zenkai Stacks", value=f"{loser.zenkai_stacks}/3", inline=True)
        embed.add_field(name="Stat Boost", value=f"STR +{stat_boost:.1f}, DEF +{stat_boost:.1f}", inline=True)
        await ctx.send(embed=embed)


async def setup(bot):
    await bot.add_cog(ZenkaiArena(bot))

# ⚡ Zenkai Discord Bot — Setup Guide

Complete setup instructions for the Zenkai Corporation Discord bot.

---

## 📋 Prerequisites

- Python 3.10 or higher
- A Discord account
- Admin access to a Discord server

---

## 🤖 1. Create Discord Bot

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click **"New Application"** → Name it "Zenkai Bot"
3. Go to **Bot** tab → Click **"Add Bot"**
4. Copy the **Bot Token** (keep this secret!)
5. Enable these **Privileged Gateway Intents**:
   - ✅ Presence Intent
   - ✅ Server Members Intent
   - ✅ Message Content Intent

---

## 🔗 2. Invite Bot to Server

1. Go to **OAuth2** → **URL Generator**
2. Select scopes:
   - ✅ `bot`
   - ✅ `applications.commands`
3. Select bot permissions:
   - ✅ Administrator (or select individual permissions below)

   Individual permissions needed:
   - Manage Roles
   - Manage Channels
   - Send Messages
   - Manage Messages
   - Embed Links
   - Attach Files
   - Read Message History
   - Add Reactions
   - Use Slash Commands

4. Copy the generated URL and open it in browser
5. Select your server and authorize

---

## ⚙️ 3. Configure Bot

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your values:
   ```env
   DISCORD_TOKEN=your_bot_token_here
   GUILD_ID=your_server_id_here
   ```

3. (Optional) Update `config.json` with your:
   - Website URL
   - Social links
   - Project info

---

## 📦 4. Install & Run

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the bot
python bot.py
```

---

## 📁 5. Server Channel Setup

Create these channels in your Discord server:

### 📢 INFORMATION (Category)
```
#announcements  (read-only for members)
#rules
#roles          (for reaction roles)
#links-and-resources
```

### 💬 GENERAL (Category)
```
#general-chat
#introductions
#feedback
```

### 📈 TRADING (Category)
```
#trading-discussion
#signals-preview  (read-only for members)
#ict-concepts
#show-your-trades
```

### 🎮 GAMING (Category)
```
#gaming-chat
#ogame
#minecraft-hytale
```

### 🛠️ DEVELOPMENT (Category)
```
#dev-log  (read-only for members)
#code-help
#smart-home
```

### 📺 CONTENT (Category)
```
#youtube-uploads  (read-only for members)
#content-ideas
```

---

## 🎭 6. Setup Roles & Reaction Roles

1. Create the Member role:
   ```
   !createroles
   ```

2. Post reaction roles in #roles:
   ```
   !setuproles
   ```

This will:
- Create all roles if they don't exist
- Post the role selection embed
- Add reactions to the embed

---

## 📋 7. Bot Commands Reference

### General Commands (Everyone)
| Command | Description |
|---------|-------------|
| `/projects` | View all Zenkai projects |
| `/info` | Server info and links |
| `/lastsignal` | Latest trading signal (placeholder) |
| `/help` | Show all commands |
| `/ping` | Check bot latency |

### Admin Commands
| Command | Description |
|---------|-------------|
| `!announce #channel message` | Post announcement |
| `!devlog message` | Post dev update |
| `!signal PAIR DIR ENTRY SL TP notes` | Post trading signal |
| `!setuproles` | Setup reaction roles |
| `!createroles` | Create all roles |
| `!testwelcome [@user]` | Test welcome system |

---

## 🔧 Troubleshooting

### Bot not responding?
1. Check if bot is online in Discord
2. Verify bot has correct permissions
3. Check console for errors
4. Make sure intents are enabled in Developer Portal

### Slash commands not showing?
1. Wait a few minutes (can take up to an hour globally)
2. Try kicking and re-inviting the bot
3. Check GUILD_ID is set correctly for instant sync

### Reaction roles not working?
1. Make sure bot has "Manage Roles" permission
2. Bot's role must be HIGHER than roles it assigns
3. Run `!setuproles` to create the role message

### Can't send DMs to new members?
This is normal — some users have DMs disabled. The welcome channel message will still work.

---

## 📂 File Structure

```
zenkai-discord-bot/
├── bot.py              # Main bot file
├── config.json         # Bot configuration
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (create from .env.example)
├── .env.example       # Template for .env
├── role_message.json  # Stores role message ID (auto-created)
├── SETUP.md           # This file
└── cogs/
    ├── __init__.py
    ├── welcome.py      # Welcome system
    ├── roles.py        # Reaction roles
    ├── commands.py     # General commands
    └── announcements.py # Announcement system
```

---

## 🚀 Running in Production

For 24/7 uptime, consider:

1. **PM2 (Node process manager)**:
   ```bash
   pm2 start bot.py --interpreter python
   ```

2. **Systemd service** (Linux):
   ```ini
   [Unit]
   Description=Zenkai Discord Bot
   After=network.target

   [Service]
   Type=simple
   User=your_user
   WorkingDirectory=/path/to/zenkai-discord-bot
   ExecStart=/path/to/venv/bin/python bot.py
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

3. **Docker** (create Dockerfile):
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["python", "bot.py"]
   ```

---

## ⚡ Zenkai Corporation

Built with 💚 by Son Goku

**Evolve or Die**

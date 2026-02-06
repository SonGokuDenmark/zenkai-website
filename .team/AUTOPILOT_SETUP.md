# ⚡ Zenkai Autopilot — Setup Guide
## Automated Lars execution on a schedule

---

## How the loop works

```
┌─────────────────────────────────────────────────────┐
│                  THE ZENKAI LOOP                     │
│                                                      │
│   BULMA (during chats with Goku)                    │
│     │                                                │
│     ▼                                                │
│   Writes tasks to BULMA_TO_LARS.md                  │
│     │                                                │
│     ▼                                                │
│   ⏰ Windows Task Scheduler (every 2-4 hours)       │
│     │                                                │
│     ▼                                                │
│   LARS wakes up, reads BULMA_TO_LARS.md             │
│     │                                                │
│     ▼                                                │
│   Executes all 🔴 PRIORITY tasks                    │
│     │                                                │
│     ▼                                                │
│   Updates LARS_TO_BULMA.md with status report       │
│     │                                                │
│     ▼                                                │
│   BULMA reads report next conversation              │
│     │                                                │
│     ▼                                                │
│   Reviews work, writes NEW tasks → loop continues   │
│                                                      │
└─────────────────────────────────────────────────────┘
```

## Setup: Windows Task Scheduler

### Option A: Quick setup via command line

Open PowerShell as Administrator and run:

```powershell
# Create a scheduled task that runs every 4 hours
$action = New-ScheduledTaskAction -Execute "C:\Zenkai\.team\zenkai-autopilot.bat"
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 30)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName "ZenkaiAutopilot" -Action $action -Trigger $trigger -Settings $settings -Description "Runs Lars every 30 mins to check Bulma's task board (LAUNCH MODE)"
```

### Option B: Manual setup via GUI

1. Press **Win+R** → type `taskschd.msc` → Enter
2. Click **Create Basic Task**
3. Name: `Zenkai Autopilot`
4. Trigger: **Daily** → then set to repeat every **30 minutes**
5. Action: **Start a program**
6. Program: `C:\Zenkai\.team\zenkai-autopilot.bat`
7. Check: **Open the Properties dialog** → under Conditions, uncheck "Start only if AC power"
8. Finish

### Adjusting frequency

- Every 1 hour: change `-Minutes 30` to `-Hours 1`
- Every 2 hours: change `-Minutes 30` to `-Hours 2`
- Every 4 hours (cruise): change `-Minutes 30` to `-Hours 4`
- Only during work hours: add additional triggers for specific times

## Important notes

- PC must be ON (not sleeping) for the scheduler to trigger
- Lars needs Claude Code CLI installed and authenticated
- The `--print` flag makes Lars run non-interactively
- If no tasks are pending, Lars writes "no pending tasks" and exits
- Check Task Scheduler history to see if runs completed

## Stopping the autopilot

```powershell
Unregister-ScheduledTask -TaskName "ZenkaiAutopilot" -Confirm:$false
```

## The missing 10%

Bulma cannot self-start conversations. She writes tasks during YOUR chats.
To maximize the loop:
- At the end of each Bulma session, ask her to front-load the task board
- She'll fill BULMA_TO_LARS.md with enough work for Lars's next few cycles
- When you come back, ask Bulma to check LARS_TO_BULMA.md for the report

---

*The factory never sleeps. ⚡*

@echo off
REM Zenkai Autopilot - Runs Lars on a schedule

echo [%date% %time%] Autopilot triggered >> "C:\Zenkai\.team\autopilot.log"
cd /d C:\Zenkai

REM Ensure node and npm are in PATH (Task Scheduler may not load user PATH)
set "PATH=C:\Program Files\nodejs;C:\Users\thoma\AppData\Roaming\npm;%PATH%"

echo [%date% %time%] Starting Lars... >> "C:\Zenkai\.team\autopilot.log"

claude --dangerously-skip-permissions -p "Read C:\Zenkai\CLAUDE.md first. Then read C:\Zenkai\.team\BULMA_TO_LARS.md and execute ALL pending PRIORITY tasks. For each task: read the relevant files, make the fix, verify it works. When done, move completed items to the COMPLETED section with todays date. Then write a status report in C:\Zenkai\.team\LARS_TO_BULMA.md describing what you did, any issues found, and any questions. If there are no pending tasks, write No pending tasks in LARS_TO_BULMA.md with the timestamp." >> "C:\Zenkai\.team\autopilot.log" 2>&1

echo [%date% %time%] Lars finished >> "C:\Zenkai\.team\autopilot.log"

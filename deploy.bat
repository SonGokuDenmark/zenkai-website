@echo off
REM Zenkai Website — Git deploy script
REM Commits and pushes all changes to trigger auto-deploy.
REM Usage: deploy.bat or deploy.bat "commit message"

cd /d "C:\Zenkai\website"

set MSG=%~1
if "%MSG%"=="" set MSG=Deploy update

git add -A
git commit -m "%MSG%"
git push origin main

echo.
echo Deploy complete.
pause

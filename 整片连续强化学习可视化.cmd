@echo off
setlocal
cd /d "%~dp0"
set "PLAYER=experiments\2026-09-22_wall_cycle_rl\raw\interactive_replay_v6\index.html"
if not exist "%PLAYER%" (
  echo Replay not found: %PLAYER%
  pause
  exit /b 1
)
start "" "%PLAYER%"

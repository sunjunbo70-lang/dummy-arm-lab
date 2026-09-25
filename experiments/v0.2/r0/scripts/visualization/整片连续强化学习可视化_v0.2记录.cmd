@echo off
setlocal
for %%I in ("%~dp0..\..\..\..\..") do set "ROOT=%%~fI\"
set "PLAYER=%ROOT%experiments\v0.2\r0\records\2026-09-22_wall_cycle_rl\raw\interactive_replay_v6\index.html"
if not exist "%PLAYER%" (
  echo Replay not found: %PLAYER%
  pause
  exit /b 1
)
start "" "%PLAYER%"


